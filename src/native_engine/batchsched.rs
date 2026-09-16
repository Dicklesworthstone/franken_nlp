//! Shared-weight, layer-major eager bf16 execution with independent ragged K/V.
//!
//! This is a synchronous reference kernel, to be called inside the embedding
//! host's already-admitted compute closure. It creates no runtime or threads,
//! loads no model, and grants no activation/admission authority. One immutable
//! weight set is borrowed, never cloned for each row. Each projection visits
//! an output-weight row across all participating sequences before advancing.
//! This is not a SIMD/tiled-GEMM throughput or oracle-parity claim.
//!
//! A step accepts at most one next token per sequence, in any row order.
//! Sequences may have different context lengths. Every row owns all 44 K/V
//! slots and its own causal/RoPE position. Successful steps preserve prefixes;
//! failed or panicking steps retire ALL participating sequences, not untouched
//! siblings. Retiring a sequence clears logical lengths, not memory contents.

use std::{error::Error, fmt};
use super::{
    decode::DecodeCancellationKind,
    hf_bf16_eager::{HfBf16EagerError, HfBf16EagerWeights, HF_BF16_EAGER_PROFILE},
    kv::{KvCache, KV_BYTES_PER_TOKEN, KV_SLOT_COUNT},
    layer::{HfBf16EagerLayerWeights, NANBEIGE_HIDDEN_SIZE, NANBEIGE_INTERMEDIATE_SIZE,
        NANBEIGE_KV_PROJECTION_SIZE, NANBEIGE_Q_PROJECTION_SIZE},
    lmhead::{greedy_argmax, NANBEIGE_VOCAB_SIZE},
    looprun::LoopRunner,
    rope::{RopeTablesF32, NANBEIGE_HEAD_DIM},
    tensor::Bf16,
};
mod kernels;
mod linear;
mod model;
mod pool;
#[cfg(test)] mod tests;

pub const BATCH_EXECUTION_VERSION: &str = "layer-major-bf16-reference-v1";
/// A finite storage/API bound, NOT a measured throughput target.
pub const MAX_BATCH_ROWS: usize = 128;
/// This reference route does not silently award unqualified long context.
pub const MAX_BATCH_CONTEXT: usize = 8192;

/// Host-owned cooperative cancellation authority. Calls occur between stages,
/// before each output-weight row, and before each row's bounded attention work.
/// Attention itself retains the existing eager primitive's cancellation granularity.
pub trait BatchControl {
    fn checkpoint(&mut self) -> Option<DecodeCancellationKind>;
}

#[derive(Debug)]
pub enum BatchError {
    Contract(&'static str), Limit(&'static str), Allocation, InvalidNumerics,
    StaleSequence, SequenceBusy, DuplicateSequence, ContextFull,
    Native(HfBf16EagerError), Cancelled(DecodeCancellationKind),
}
impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(axis) => write!(f, "native batch contract refused: {axis}"),
            Self::Limit(axis) => write!(f, "native batch payload limit: {axis}"),
            Self::Cancelled(cause) => write!(f, "native batch cancelled: {cause:?}"),
            Self::Allocation => f.write_str("native batch allocation refused"),
            Self::InvalidNumerics => f.write_str("native batch nonfinite activation or logits"),
            Self::StaleSequence => f.write_str("native batch sequence handle is stale or foreign"),
            Self::SequenceBusy => f.write_str("native batch sequence slot is already occupied"),
            Self::DuplicateSequence => f.write_str("native batch step repeats a sequence"),
            Self::ContextFull => f.write_str("native batch sequence context is full"),
            Self::Native(_) => f.write_str("native batch model operation failed"),
        }
    }
}
impl Error for BatchError {}
impl From<HfBf16EagerError> for BatchError {
    fn from(error: HfBf16EagerError) -> Self { Self::Native(error) }
}

/// Non-allocating payload estimate. K/V and RoPE are exact requested payloads;
/// scratch deliberately sums overlapping stage maxima as a conservative upper
/// bound. This is NOT RSS, allocator capacity, a process certificate, or weights.
/// The host separately charges borrowed weights, metadata, allocator overhead,
/// safety reserve, and any outputs it retains after a subsequent step begins.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchPayload {
    pub kv_bytes: u64,
    pub rope_bytes: u64,
    pub scratch_upper_bytes: u64,
    pub full_logit_bytes: u64,
    pub total_bytes: u64,
}

/// Frozen per-slot capacities, checked before the first K/V/table allocation.
#[derive(Clone, Debug)]
pub struct BatchEnvelope { capacities: Vec<usize>, payload: BatchPayload }
impl BatchEnvelope {
    pub fn compile(capacities: &[usize], payload_ceiling: u64) -> Result<Self, BatchError> {
        let payload = Self::estimate(capacities)?;
        if payload.total_bytes > payload_ceiling { return Err(BatchError::Limit("complete batch payload")); }
        let mut owned = reserve(capacities.len())?; owned.extend_from_slice(capacities);
        Ok(Self { capacities: owned, payload })
    }
    pub fn estimate(capacities: &[usize]) -> Result<BatchPayload, BatchError> {
        if capacities.is_empty() || capacities.len() > MAX_BATCH_ROWS
            || capacities.iter().any(|&cap| cap == 0 || cap > MAX_BATCH_CONTEXT) {
            return Err(BatchError::Contract("row count or context capacity"));
        }
        let rows = capacities.len() as u64;
        let total_positions = capacities.iter().try_fold(0_u64, |n, &v| add(n, v as u64))?;
        let max_context = *capacities.iter().max().ok_or(BatchError::Contract("empty batch"))? as u64;
        let kv_bytes = mul(total_positions, KV_BYTES_PER_TOKEN as u64)?;
        // Two half-head f32 tables plus one half-head inverse-frequency vector.
        let rope_bytes = mul(add(mul(max_context, NANBEIGE_HEAD_DIM as u64)?, (NANBEIGE_HEAD_DIM / 2) as u64)?, 4)?;
        // All layer intermediates, even those explicitly dropped between stages.
        let bf16_widths = 8 * NANBEIGE_HIDDEN_SIZE + 2 * NANBEIGE_Q_PROJECTION_SIZE
            + 2 * NANBEIGE_KV_PROJECTION_SIZE + 3 * NANBEIGE_INTERMEDIATE_SIZE;
        let group_scratch = mul(mul(rows, bf16_widths as u64)?, 2)?;
        // Only one row's norm/SwiGLU/attention primitive is active at a time.
        // Includes f32 reduction/cast temporaries, context-sized softmax vectors,
        // query/head outputs and K/V bit conversion storage, without KV repeats.
        let primitive_scratch = mul(add(mul(max_context, 6)?,
            (8 * NANBEIGE_INTERMEDIATE_SIZE + 4 * NANBEIGE_Q_PROJECTION_SIZE) as u64)?, 4)?;
        let scratch_upper_bytes = add(group_scratch, primitive_scratch)?;
        let full_logit_bytes = mul(mul(rows, NANBEIGE_VOCAB_SIZE as u64)?, 4)?;
        let total_bytes = add(add(kv_bytes, rope_bytes)?, add(scratch_upper_bytes, full_logit_bytes)?)?;
        Ok(BatchPayload { kv_bytes, rope_bytes, scratch_upper_bytes, full_logit_bytes, total_bytes })
    }
    pub fn capacities(&self) -> &[usize] { &self.capacities }
    pub fn payload(&self) -> BatchPayload { self.payload }
}

/// Arena-scoped, generation-checked handle. Not a caller ID or sampling key.
/// Pool domains are never reused; closing/reopening a slot changes generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchSequence { domain: u64, slot: usize, generation: u64 }
impl BatchSequence { pub fn slot(self) -> usize { self.slot } }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchProjection {
    /// Intermediate prefill tokens do not need a full-vocabulary denominator.
    None,
    /// Final prompt and feedback rows export the complete bf16->f32 lm head.
    FullVocabulary,
}
/// Exact next input token, private data intentionally excluded from Debug.
pub struct BatchToken { pub sequence: BatchSequence, pub token_id: u32, pub projection: BatchProjection }
/// Logits are absent only when explicitly omitted in the input step.
pub struct BatchRowOutput {
    pub sequence: BatchSequence,
    pub position: usize,
    pub logits: Option<Vec<f32>>,
    pub greedy_token: Option<usize>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchStepWork {
    pub forward_positions: usize,
    /// 44 shared logical layer invocations, not 44 separate calls per document.
    pub layer_groups: usize,
    /// Q/K/V/O/gate/up/down per logical layer, plus an optional lm-head group.
    /// These count API groups, NOT DRAM reads or measured FLOPs/throughput.
    pub linear_groups: usize,
    pub full_vocabulary_rows: usize,
}
pub struct BatchStepOutput { pub rows: Vec<BatchRowOutput>, pub work: BatchStepWork }

/// One immutable model and a finite pool of independently resumable sequences.
/// No Clone implementation: neither model weights nor populated caches should
/// be duplicated accidentally. The host retains the actual admission guard.
pub struct EagerBatchEngine<'weights> {
    weights: &'weights HfBf16EagerWeights,
    runner: LoopRunner<'weights, HfBf16EagerLayerWeights>,
    rope: RopeTablesF32,
    pool: pool::SequencePool,
    envelope: BatchEnvelope,
}
impl<'weights> EagerBatchEngine<'weights> {
    pub fn new(weights: &'weights HfBf16EagerWeights, envelope: BatchEnvelope) -> Result<Self, BatchError> {
        model::validate(weights)?;
        let cap = *envelope.capacities.iter().max().ok_or(BatchError::Contract("empty envelope"))?;
        let rope = RopeTablesF32::nanbeige(cap).map_err(HfBf16EagerError::from)?;
        let pool = pool::SequencePool::new(&envelope.capacities)?;
        Ok(Self { weights, runner: LoopRunner::from_layer_weights(&weights.layers), rope, pool, envelope })
    }
    pub fn profile(&self) -> &'static str { HF_BF16_EAGER_PROFILE }
    pub fn envelope(&self) -> &BatchEnvelope { &self.envelope }
    /// Check all planned rows before opening any of them. A busy slot is not
    /// implicitly reset, and the complete requested context must fit its cap.
    pub fn preflight_sequence_slot(&self, slot: usize, required_positions: usize) -> Result<(), BatchError> {
        self.pool.preflight_slot(slot, required_positions)
    }
    pub fn open_sequence(&mut self, slot: usize) -> Result<BatchSequence, BatchError> { self.pool.open(slot) }
    pub fn close_sequence(&mut self, sequence: BatchSequence) -> Result<(), BatchError> { self.pool.close(sequence) }
    pub fn sequence_len(&self, sequence: BatchSequence) -> Result<usize, BatchError> { self.pool.len(sequence) }
    pub fn cache(&self, sequence: BatchSequence) -> Result<&KvCache, BatchError> {
        let slot = self.pool.validate(sequence)?; Ok(&self.pool.rows[slot].cache)
    }
    /// Advance each supplied sequence once through the shared 44-layer schedule.
    /// Validation failures do not touch any cache. Once computation starts,
    /// cancellation/error/panic invalidates only this group's handles and clears
    /// their partial cache lengths. Never continue from a partial 44-slot step.
    pub fn step<C: BatchControl>(&mut self, input: &[BatchToken], control: &mut C) -> Result<BatchStepOutput, BatchError> {
        let (slots, positions) = self.pool.preflight(input)?;
        checkpoint(control)?;
        let mut transaction = pool::StepTransaction::new(&mut self.pool, &slots);
        let mut hidden = reserve(input.len())?;
        for token in input {
            checkpoint(control)?;
            let source = self.weights.embeddings.row(token.token_id as usize).map_err(HfBf16EagerError::from)?;
            let mut row = reserve(source.len())?; row.extend_from_slice(source); hidden.push(row);
        }
        {
            let mut executor = kernels::GroupExecutor {
                rows: &mut transaction.pool.rows, slots: &slots, positions: &positions,
                rope: &self.rope, final_norm: &self.weights.final_norm, control,
            };
            self.runner.run_group(&mut executor, &mut hidden)?;
        }
        let mut projected_indices = reserve(input.len())?;
        let mut selected_hidden = reserve(input.len())?;
        for (index, token) in input.iter().enumerate() {
            if token.projection == BatchProjection::FullVocabulary {
                projected_indices.push(index); selected_hidden.push(hidden[index].as_slice());
            }
        }
        let projected_count = projected_indices.len();
        let projected = if projected_count == 0 { Vec::new() } else {
            linear::project(&self.weights.lm_head, &selected_hidden, control, |v| Bf16::from_f32(v).to_f32())?
        };
        let mut results = reserve(input.len())?;
        for (token, &position) in input.iter().zip(&positions) {
            results.push(BatchRowOutput { sequence: token.sequence, position, logits: None, greedy_token: None });
        }
        for (index, logits) in projected_indices.into_iter().zip(projected) {
            if logits.len() != NANBEIGE_VOCAB_SIZE || logits.iter().any(|x| !x.is_finite()) { return Err(BatchError::InvalidNumerics); }
            results[index].greedy_token = greedy_argmax(&logits);
            results[index].logits = Some(logits);
        }
        checkpoint(control)?;
        transaction.commit(&positions)?;
        Ok(BatchStepOutput { rows: results, work: BatchStepWork {
            forward_positions: input.len(), layer_groups: KV_SLOT_COUNT,
            linear_groups: KV_SLOT_COUNT * 7 + usize::from(projected_count != 0), full_vocabulary_rows: projected_count,
        } })
    }
}
fn checkpoint<C: BatchControl>(control: &mut C) -> Result<(), BatchError> {
    control.checkpoint().map_or(Ok(()), |cause| Err(BatchError::Cancelled(cause)))
}
fn reserve<T>(length: usize) -> Result<Vec<T>, BatchError> {
    let mut result = Vec::new(); result.try_reserve_exact(length).map_err(|_| BatchError::Allocation)?; Ok(result)
}
fn add(a: u64, b: u64) -> Result<u64, BatchError> { a.checked_add(b).ok_or(BatchError::Limit("arithmetic")) }
fn mul(a: u64, b: u64) -> Result<u64, BatchError> { a.checked_mul(b).ok_or(BatchError::Limit("arithmetic")) }
