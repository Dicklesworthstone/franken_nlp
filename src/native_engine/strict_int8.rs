//! Code-first executable portable-int8 model with explicit BF16 activation rails.
//!
//! Profile candidate: strict-quantized-v1, implementation STRICT_INT8_EXECUTION.
//! Integer/scale stages use portable_int8; embedding, norm, residual, SwiGLU,
//! RoPE and eager attention retain the existing BF16 cast program. The final
//! int8 lm-head exports its fixed-order dequantization directly as f32.
//! This candidate is NOT an OQ-30 ratification or a model-fidelity award.
//!
//! There is no model-file opener or product activation route. Existing bridge
//! weights are borrowed, never cloned or expanded. Reference NN/attention
//! primitives still allocate bounded scratch; this is not an allocator-free,
//! SIMD, batching, performance, or process-RSS claim.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use super::{
    artifact_bridge::ArtifactIdentity,
    attention::{QUERY_HEAD_COUNT, eager_gqa_attention_from_cache},
    decode::{DecodeCancellationKind, DecodeStepControl},
    kv::{KV_BYTES_PER_TOKEN, KV_SLOT_COUNT, PHYSICAL_LAYER_COUNT, KvCache, slot_for},
    layer::{NANBEIGE_HIDDEN_SIZE as H, NANBEIGE_INTERMEDIATE_SIZE as I,
        NANBEIGE_Q_PROJECTION_SIZE as Q, NANBEIGE_KV_PROJECTION_SIZE as K},
    lmhead::NANBEIGE_VOCAB_SIZE as V,
    looprun::{LayerBinding, LayerExecutor, LoopRunner, PositionContext},
    nn::{RMS_NORM_EPSILON, residual_add_f32_cast_back, rms_norm_f32_reduce_cast_back, swiglu_f32_cast_back},
    portable_int8::{ActivationBuffer, LinearError, LinearRows, ProjectionLedger, ProjectionWork, QuantizedLinear},
    rope::{DEFAULT_ADMITTED_CONTEXT_CAP, NANBEIGE_HEAD_DIM, RopeTablesF32},
    tensor::Bf16,
};
mod weights;
pub use weights::Int8WeightView;
use weights::Int8Layer;

pub const STRICT_INT8_PROFILE: &str = "strict-quantized-v1";
pub const STRICT_INT8_EXECUTION: &str = "portable-int8-bf16-rails-eager-gqa-v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StrictInt8Error {
    Weights, Input, Context, Memory, Work, Allocation, Cache, Rope, Attention,
    Primitive, Boundary, EmptyHidden, EngineUnavailable, Linear(LinearError),
    Cancelled(DecodeCancellationKind),
}
impl fmt::Display for StrictInt8Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Weights => "strict int8 complete model tensor contract refused",
            Self::Input => "strict int8 requires valid nonempty exact input tokens",
            Self::Context => "strict int8 admitted context exceeded",
            Self::Memory => "strict int8 engine payload budget exceeded",
            Self::Work => "strict int8 aggregate execution work exceeded",
            Self::Allocation => "strict int8 bounded allocation refused",
            Self::Cache => "strict int8 logical KV invariant failed",
            Self::Rope => "strict int8 rotary position failed",
            Self::Attention => "strict int8 eager attention failed",
            Self::Primitive => "strict int8 activation or primitive failed",
            Self::Boundary => "strict int8 shared loop boundary diverged",
            Self::EmptyHidden => "strict int8 projection requires a completed forward",
            Self::EngineUnavailable => "strict int8 engine is active or poisoned",
            Self::Linear(_) => "strict int8 projection failed",
            Self::Cancelled(_) => "strict int8 execution cancelled",
        })
    }
}
impl Error for StrictInt8Error {}
impl From<LinearError> for StrictInt8Error {
    fn from(error: LinearError) -> Self {
        match error { LinearError::Cancelled(cause) => Self::Cancelled(cause), other => Self::Linear(other) }
    }
}

/// Payload ceilings checked BEFORE KV/RoPE/workspace allocation. The embedding
/// host separately admits source-owned weights, allocator overhead and output.
/// These are not a PermitBroker replacement or a measured peak-RSS assertion.
#[derive(Clone, Copy, Debug)]
pub struct Int8MemoryBudget {
    pub max_kv_bytes: u64,
    pub max_rope_bytes: u64,
    pub max_scratch_payload_bytes: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8MemoryRequirement {
    pub kv_bytes: u64, pub rope_bytes: u64, pub scratch_payload_bound: u64,
}
impl Int8MemoryRequirement {
    pub fn for_context(positions: usize) -> Result<Self, StrictInt8Error> {
        // Long-context support remains separately gated, not silently enabled
        // merely because a source config advertises a much larger maximum.
        if positions == 0 || positions > DEFAULT_ADMITTED_CONTEXT_CAP { return Err(StrictInt8Error::Context); }
        let n = positions as u64;
        Ok(Self {
            kv_bytes: n * KV_BYTES_PER_TOKEN as u64,
            rope_bytes: (n * NANBEIGE_HEAD_DIM as u64 + (NANBEIGE_HEAD_DIM / 2) as u64) * 4,
            // Conservative simultaneous reference-vector payload allowance:
            // 32 maximum-intermediate-width f32 rails, one full head, and
            // 16 bytes/context for scores/exp/division/BF16 probabilities.
            // Includes activation bytes, hidden copies and one-head scratch.
            // Vec metadata, allocator slack and source weights are excluded.
            scratch_payload_bound: (32 * I + V + NANBEIGE_HEAD_DIM) as u64 * 4 + 16 * n,
        })
    }
    fn check(self, budget: Int8MemoryBudget) -> Result<(), StrictInt8Error> {
        if self.kv_bytes > budget.max_kv_bytes || self.rope_bytes > budget.max_rope_bytes
            || self.scratch_payload_bound > budget.max_scratch_payload_bytes { return Err(StrictInt8Error::Memory); }
        Ok(())
    }
}

/// All decoder projections AND requested head rows count. Attention pairs
/// count one QK score plus its value-vector accumulation, not machine MACs.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int8Work {
    pub forward_positions: u64,
    pub projected_logits: u64,
    pub attention_pairs: u64,
    pub projections: ProjectionWork,
}
impl Int8Work {
    pub fn for_sequence(start_position: usize, positions: usize, projected_logits: usize) -> Result<Self, StrictInt8Error> {
        let start = u64::try_from(start_position).map_err(|_| StrictInt8Error::Work)?;
        let n = u64::try_from(positions).map_err(|_| StrictInt8Error::Work)?;
        let end = start.checked_add(n).ok_or(StrictInt8Error::Work)?;
        let triangle = |v: u64| v.checked_add(1).and_then(|next| v.checked_mul(next)).map(|x| x / 2).ok_or(StrictInt8Error::Work);
        let pairs = triangle(end)?.checked_sub(triangle(start)?).and_then(|sum| sum.checked_mul((KV_SLOT_COUNT * QUERY_HEAD_COUNT) as u64))
            .ok_or(StrictInt8Error::Work)?;
        let decoder = decoder_projection_work()?;
        let projections = ProjectionWork {
            dot_products: decoder.dot_products.checked_mul(n).ok_or(StrictInt8Error::Work)?,
            multiply_accumulates: decoder.multiply_accumulates.checked_mul(n).ok_or(StrictInt8Error::Work)?,
        }.checked_add(ProjectionWork::for_shape(projected_logits, H)?)?;
        Ok(Self { forward_positions: n, projected_logits: projected_logits as u64, attention_pairs: pairs, projections })
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Int8RunBudget {
    pub max_forward_positions: u64,
    pub max_attention_pairs: u64,
    pub max_projection_work: ProjectionWork,
}
impl Int8RunBudget {
    pub fn exact(work: Int8Work) -> Self {
        Self { max_forward_positions: work.forward_positions, max_attention_pairs: work.attention_pairs, max_projection_work: work.projections }
    }
}
fn decoder_projection_work() -> Result<ProjectionWork, StrictInt8Error> {
    let mut work = ProjectionWork::default();
    for (rows, columns) in [(Q, H), (K, H), (K, H), (H, Q), (I, H), (I, H), (H, I)] {
        work = work.checked_add(ProjectionWork::for_shape(rows, columns)?)?;
    }
    Ok(ProjectionWork { dot_products: work.dot_products * KV_SLOT_COUNT as u64,
        multiply_accumulates: work.multiply_accumulates * KV_SLOT_COUNT as u64 })
}

#[derive(Default)]
struct RunState { active: bool, poisoned: bool }
impl RunState {
    fn open(&mut self) -> Result<(), StrictInt8Error> {
        if self.active || self.poisoned { return Err(StrictInt8Error::EngineUnavailable); }
        self.active = true; Ok(())
    }
    fn check(&self) -> Result<(), StrictInt8Error> {
        if !self.active || self.poisoned { Err(StrictInt8Error::EngineUnavailable) } else { Ok(()) }
    }
}

/// Caller-admitted, single-sequence semantic engine. No clone, runtime, file
/// access or worker spawn. One session exclusively owns logical KV lifetime.
pub struct StrictInt8Engine<'weights> {
    weights: Int8WeightView<'weights>, cache: KvCache, rope: RopeTablesF32,
    activation: ActivationBuffer, state: RunState,
}
impl<'weights> StrictInt8Engine<'weights> {
    pub fn new(weights: Int8WeightView<'weights>, context: usize, memory: Int8MemoryBudget) -> Result<Self, StrictInt8Error> {
        Int8MemoryRequirement::for_context(context)?.check(memory)?;
        Ok(Self { weights, cache: KvCache::try_with_capacity(context).map_err(|_| StrictInt8Error::Allocation)?,
            rope: RopeTablesF32::nanbeige(context).map_err(|_| StrictInt8Error::Rope)?,
            activation: ActivationBuffer::try_new(I)?, state: RunState::default() })
    }
    pub fn profile(&self) -> &'static str { STRICT_INT8_PROFILE }
    pub fn artifact_identity(&self) -> &ArtifactIdentity { self.weights.identity }
    pub fn kv_cache(&self) -> &KvCache { &self.cache }
    pub fn is_poisoned(&self) -> bool { self.state.poisoned }
    pub fn session<'run, C: DecodeStepControl>(&'run mut self, budget: Int8RunBudget, control: &'run mut C)
        -> Result<Int8Session<'run, 'weights, C>, StrictInt8Error> {
        if !self.cache.all_slots_have_len(0) { return Err(StrictInt8Error::Cache); }
        self.state.open()?;
        Ok(Int8Session { engine: self, control, remaining_positions: budget.max_forward_positions,
            remaining_attention: budget.max_attention_pairs, ledger: ProjectionLedger::new(budget.max_projection_work),
            hidden: None, work: Int8Work::default() })
    }
}

/// RAII request/sequence. Drop clears all 44 logical slots while retaining
/// storage. Errors during native work, cancellation and unwinding poison the
/// engine. Forgetting a session leaves active=true and blocks future admission.
/// Budget preflight refusals before native work do not destroy a valid prefix.
pub struct Int8Session<'run, 'weights, C: DecodeStepControl> {
    engine: &'run mut StrictInt8Engine<'weights>, control: &'run mut C,
    remaining_positions: u64, remaining_attention: u64, ledger: ProjectionLedger,
    hidden: Option<Vec<Bf16>>, work: Int8Work,
}
impl<C: DecodeStepControl> Int8Session<'_, '_, C> {
    pub fn work(&self) -> Int8Work {
        Int8Work { projections: self.ledger.reserved(), ..self.work }
    }
    pub fn position(&self) -> Result<usize, StrictInt8Error> {
        self.engine.state.check()?;
        let position = self.engine.cache.len_for_slot(0).map_err(|_| StrictInt8Error::Cache)?;
        if !self.engine.cache.all_slots_have_len(position) { return Err(StrictInt8Error::Cache); }
        Ok(position)
    }
    /// Preflight an entire prompt/continuation and aggregate head rows before
    /// the first forward. Output rows may account for several full projections.
    pub fn preflight(&self, positions: usize, projected_logits: usize) -> Result<Int8Work, StrictInt8Error> {
        let start = self.position()?;
        if start.checked_add(positions).is_none_or(|end| end > self.engine.cache.capacity_positions()) { return Err(StrictInt8Error::Context); }
        let bound = Int8Work::for_sequence(start, positions, projected_logits)?;
        if bound.forward_positions > self.remaining_positions || bound.attention_pairs > self.remaining_attention { return Err(StrictInt8Error::Work); }
        self.ledger.preflight(bound.projections)?;
        Ok(bound)
    }
    /// One exact token through the shared 22 -> norm -> 22 -> norm schedule.
    /// Does not project lm_head or retain all intermediate layer taps.
    pub fn append(&mut self, token: u32) -> Result<(), StrictInt8Error> {
        if token as usize >= V { return Err(StrictInt8Error::Input); }
        let bound = self.preflight(1, 0)?;
        let position = self.position()?;
        self.remaining_positions -= 1; self.remaining_attention -= bound.attention_pairs;
        self.engine.state.poisoned = true; // Cleared ONLY after every native invariant succeeds.
        self.hidden = None;
        poll(self.control)?;
        let row = self.engine.weights.embeddings.row(token as usize).map_err(|_| StrictInt8Error::Input)?;
        let mut hidden = filled(H, Bf16::from_bits(0))?;
        hidden.copy_from_slice(row); finite(&hidden)?;
        let runner = LoopRunner::from_layer_weights(&self.engine.weights.layers);
        let mut executor = Executor { final_norm: self.engine.weights.final_norm, rope: &self.engine.rope,
            activation: &mut self.engine.activation, ledger: &mut self.ledger, control: &mut *self.control, completed: 0, norms: 0 };
        runner.run_token(&mut executor, &mut hidden, PositionContext::at(position), &mut self.engine.cache)?;
        if executor.completed != KV_SLOT_COUNT || executor.norms != 2 { return Err(StrictInt8Error::Boundary); }
        if !self.engine.cache.all_slots_have_len(position + 1) { return Err(StrictInt8Error::Cache); }
        finite(&hidden)?; poll(self.control)?;
        self.hidden = Some(hidden);
        self.work.forward_positions += 1; self.work.attention_pairs += bound.attention_pairs;
        self.engine.state.poisoned = false; Ok(())
    }
    /// True full or selected-row head projection from the last completed token.
    /// The output is f32 dequantization, not BF16-reference logits in disguise.
    pub fn logits(&mut self, rows: LinearRows<'_>) -> Result<Vec<f32>, StrictInt8Error> {
        let count = rows.checked_count(V)?;
        self.preflight(0, count)?;
        if self.hidden.is_none() { return Err(StrictInt8Error::EmptyHidden); }
        let next_count = self.work.projected_logits.checked_add(count as u64).ok_or(StrictInt8Error::Work)?;
        let mut output = filled(count, 0.0_f32)?;
        self.engine.state.poisoned = true;
        self.engine.activation.encode_bf16(self.hidden.as_deref().ok_or(StrictInt8Error::EmptyHidden)?)?;
        self.engine.weights.head.project_f32_into(&self.engine.activation, rows, &mut output, &mut self.ledger, &mut *self.control)?;
        self.work.projected_logits = next_count; self.engine.state.poisoned = false; Ok(output)
    }
    /// A bounded token-ordered prefill; only its final position projects head
    /// rows. The whole input and work closure are checked before any mutation.
    pub fn prefill(&mut self, tokens: &[u32], rows: LinearRows<'_>) -> Result<Vec<f32>, StrictInt8Error> {
        if tokens.is_empty() || tokens.iter().any(|&id| id as usize >= V) { return Err(StrictInt8Error::Input); }
        let count = rows.checked_count(V)?;
        self.preflight(tokens.len(), count)?;
        for &token in tokens { self.append(token)?; }
        self.logits(rows)
    }
}
impl<C: DecodeStepControl> Drop for Int8Session<'_, '_, C> {
    fn drop(&mut self) { end_session(&mut self.engine.cache, &mut self.engine.state); }
}
fn end_session(cache: &mut KvCache, state: &mut RunState) { cache.clear(); state.active = false; }

struct Executor<'a, C> {
    final_norm: &'a [Bf16], rope: &'a RopeTablesF32, activation: &'a mut ActivationBuffer,
    ledger: &'a mut ProjectionLedger, control: &'a mut C, completed: usize, norms: usize,
}
impl<'w, C: DecodeStepControl> LayerExecutor<Int8Layer<'w>> for Executor<'_, C> {
    type Hidden = Vec<Bf16>; type Error = StrictInt8Error;
    fn layer_forward(&mut self, binding: &LayerBinding<'_, Int8Layer<'w>>, hidden: &mut Self::Hidden,
        positions: PositionContext, cache: &mut KvCache) -> Result<(), Self::Error> {
        if slot_for(binding.loop_index(), binding.layer_index()) != Some(binding.kv_slot())
            || positions != PositionContext::at(positions.position) { return Err(StrictInt8Error::Boundary); }
        poll(self.control)?;
        let layer = binding.weights();
        let normalized = norm(hidden, layer.norm1)?;
        self.activation.encode_bf16(&normalized)?;
        // Same quantized activation and scale feed Q/K/V; no repeated quantization.
        let mut query = project(layer.q, self.activation, self.ledger, self.control)?;
        let mut key = project(layer.k, self.activation, self.ledger, self.control)?;
        let value = project(layer.v, self.activation, self.ledger, self.control)?;
        for head in query.chunks_exact_mut(NANBEIGE_HEAD_DIM) { self.rope.apply_split_half(positions.rope_position, head).map_err(|_| StrictInt8Error::Rope)?; }
        for head in key.chunks_exact_mut(NANBEIGE_HEAD_DIM) { self.rope.apply_split_half(positions.rope_position, head).map_err(|_| StrictInt8Error::Rope)?; }
        finite(&query)?; finite(&key)?;
        let mut key_bits = filled(K, 0_u16)?; let mut value_bits = filled(K, 0_u16)?;
        for (out, value) in key_bits.iter_mut().zip(&key) { *out = value.to_bits(); }
        for (out, value) in value_bits.iter_mut().zip(&value) { *out = value.to_bits(); }
        cache.append(binding.kv_slot(), positions.cache_position, &key_bits, &value_bits).map_err(|_| StrictInt8Error::Cache)?;
        poll(self.control)?;
        let attention = eager_gqa_attention_from_cache(&query, cache, binding.kv_slot()).map_err(|_| StrictInt8Error::Attention)?;
        finite(&attention)?; poll(self.control)?;
        self.activation.encode_bf16(&attention)?;
        let update = project(layer.o, self.activation, self.ledger, self.control)?;
        let residual = residual_add_f32_cast_back(hidden, &update).map_err(|_| StrictInt8Error::Primitive)?;
        let normalized = norm(&residual, layer.norm2)?;
        self.activation.encode_bf16(&normalized)?;
        let gate = project(layer.gate, self.activation, self.ledger, self.control)?;
        let up = project(layer.up, self.activation, self.ledger, self.control)?;
        let product = swiglu_f32_cast_back(&gate, &up).map_err(|_| StrictInt8Error::Primitive)?;
        self.activation.encode_bf16(&product)?;
        let update = project(layer.down, self.activation, self.ledger, self.control)?;
        let output = residual_add_f32_cast_back(&residual, &update).map_err(|_| StrictInt8Error::Primitive)?;
        finite(&output)?; *hidden = output; self.completed += 1; Ok(())
    }
    fn final_rms_norm(&mut self, hidden: &mut Self::Hidden, _: PositionContext) -> Result<(), Self::Error> {
        if self.norms >= 2 || self.completed != (self.norms + 1) * PHYSICAL_LAYER_COUNT { return Err(StrictInt8Error::Boundary); }
        poll(self.control)?; *hidden = norm(hidden, self.final_norm)?; self.norms += 1; Ok(())
    }
}
fn norm(input: &[Bf16], scale: &[Bf16]) -> Result<Vec<Bf16>, StrictInt8Error> {
    finite(input)?;
    // The reference primitive does not itself reject variance overflow. Do not
    // silently convert an infinite variance into an apparently valid zero row.
    let sum = input.iter().map(|v| v.to_f32() * v.to_f32()).sum::<f32>();
    if !sum.is_finite() { return Err(StrictInt8Error::Primitive); }
    let result = rms_norm_f32_reduce_cast_back(input, scale, RMS_NORM_EPSILON).map_err(|_| StrictInt8Error::Primitive)?;
    finite(&result)?; Ok(result)
}
fn finite(values: &[Bf16]) -> Result<(), StrictInt8Error> {
    if values.iter().all(|v| v.to_f32().is_finite()) { Ok(()) } else { Err(StrictInt8Error::Primitive) }
}
fn project<C: DecodeStepControl>(matrix: QuantizedLinear<'_>, activation: &ActivationBuffer,
    ledger: &mut ProjectionLedger, control: &mut C) -> Result<Vec<Bf16>, StrictInt8Error> {
    let mut output = filled(matrix.rows(), Bf16::from_bits(0))?;
    matrix.project_bf16_into(activation, LinearRows::All, &mut output, ledger, control)?; Ok(output)
}
fn filled<T: Clone>(length: usize, value: T) -> Result<Vec<T>, StrictInt8Error> {
    let mut output = Vec::new(); output.try_reserve_exact(length).map_err(|_| StrictInt8Error::Allocation)?;
    output.resize(length, value); Ok(output)
}
fn poll<C: DecodeStepControl>(control: &mut C) -> Result<(), StrictInt8Error> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(StrictInt8Error::Cancelled(cause)), None => Ok(()) }
}

#[cfg(test)] mod tests;
