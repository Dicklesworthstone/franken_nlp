//! Explicit full-model grouped-INT4 candidate. This is not an INT8 profile,
//! canonical fnlpq format, activation authority, or BF16 fidelity claim.
//! The common loop runner owns 22 -> norm -> 22 -> norm and all 44 KV slots.

use std::{error::Error, fmt};
use super::{
    attention::{QUERY_HEAD_COUNT, eager_gqa_attention_from_cache},
    decode::{DecodeCancellationKind, DecodeStepControl},
    int4::PackedInt4Kernel,
    kv::{KV_BYTES_PER_TOKEN, KV_SLOT_COUNT, PHYSICAL_LAYER_COUNT, KvCache, slot_for},
    layer::{NANBEIGE_HIDDEN_SIZE as H, NANBEIGE_INTERMEDIATE_SIZE as I, NANBEIGE_KV_PROJECTION_SIZE as K},
    lmhead::NANBEIGE_VOCAB_SIZE as V,
    looprun::{LayerBinding, LayerExecutor, LoopRunner, PositionContext},
    nn::{RMS_NORM_EPSILON, residual_add_f32_cast_back, rms_norm_f32_reduce_cast_back, swiglu_f32_cast_back},
    portable_int8::{ActivationBuffer, LinearError, LinearRows, ProjectionLedger, ProjectionWork,
        int4::{GroupedInt4Error, convert::GroupedInt4Matrix}},
    quant_algebra::MAX_MODEL_K,
    rope::{DEFAULT_ADMITTED_CONTEXT_CAP, NANBEIGE_HEAD_DIM, RopeTablesF32},
    simd::Int8KernelKind,
    tensor::Bf16,
};

pub mod weights;
pub use weights::{Int4ModelError, Int4ModelIdentity, Int4ModelLoadBudget,
    Int4ModelLoadRequirement, Int4ModelWeights};
use weights::{Int4Layer, PROJECTION_SHAPES};

pub const STRICT_INT4_PROFILE: &str = "strict-grouped-int4-bf16-rails-v1";
pub const STRICT_INT4_EXECUTION: &str = "grouped-int4-bf16-rails-eager-gqa-v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StrictInt4Error {
    Input, Context, Memory, Work, Allocation, Cache, Rope, Attention,
    Primitive, Boundary, EmptyHidden, EngineUnavailable,
    Linear(GroupedInt4Error), Cancelled(DecodeCancellationKind),
}
impl fmt::Display for StrictInt4Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Input => "strict INT4 requires valid nonempty exact input tokens",
            Self::Context => "strict INT4 admitted context exceeded",
            Self::Memory => "strict INT4 engine payload budget exceeded",
            Self::Work => "strict INT4 aggregate execution work exceeded",
            Self::Allocation => "strict INT4 bounded allocation refused",
            Self::Cache => "strict INT4 logical KV invariant failed",
            Self::Rope => "strict INT4 rotary position failed",
            Self::Attention => "strict INT4 eager attention failed",
            Self::Primitive => "strict INT4 BF16 rail or primitive failed",
            Self::Boundary => "strict INT4 shared loop boundary diverged",
            Self::EmptyHidden => "strict INT4 projection requires a completed forward",
            Self::EngineUnavailable => "strict INT4 engine is active or poisoned",
            Self::Linear(_) => "strict INT4 group-scaled projection failed",
            Self::Cancelled(_) => "strict INT4 execution cancelled",
        })
    }
}
impl Error for StrictInt4Error {}
impl From<GroupedInt4Error> for StrictInt4Error {
    fn from(error: GroupedInt4Error) -> Self {
        match error { GroupedInt4Error::Linear(LinearError::Cancelled(cause)) => Self::Cancelled(cause),
            other => Self::Linear(other) }
    }
}
impl From<LinearError> for StrictInt4Error {
    fn from(error: LinearError) -> Self { GroupedInt4Error::Linear(error).into() }
}

/// Includes all group dots in both loop passes, all unpadded projection MACs,
/// and each QK/value-vector pair. Not a measured instruction/traffic count.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Int4Work {
    pub forward_positions: u64, pub projected_logits: u64, pub attention_pairs: u64,
    pub projections: ProjectionWork,
}
impl Int4Work {
    pub fn for_sequence(group_width: usize, start: usize, positions: usize, logits: usize) -> Result<Self, StrictInt4Error> {
        if group_width == 0 || group_width > MAX_MODEL_K { return Err(StrictInt4Error::Work); }
        let start = u64::try_from(start).map_err(|_| StrictInt4Error::Work)?;
        let n = u64::try_from(positions).map_err(|_| StrictInt4Error::Work)?;
        start.checked_add(n).ok_or(StrictInt4Error::Work)?;
        // n*start + n*(n+1)/2 avoids subtracting two overflowing triangles.
        let next = n.checked_add(1).ok_or(StrictInt4Error::Work)?;
        let (a, b) = if n % 2 == 0 { (n / 2, next) } else { (n, next / 2) };
        let pairs = multiply(n, start)?.checked_add(multiply(a, b)?).ok_or(StrictInt4Error::Work)?;
        let attention_pairs = multiply(pairs, (KV_SLOT_COUNT * QUERY_HEAD_COUNT) as u64)?;
        let mut decoder = ProjectionWork::default();
        for (rows, columns) in PROJECTION_SHAPES { decoder = decoder.checked_add(group_work(rows, columns, group_width)?)?; }
        let executions = multiply(n, KV_SLOT_COUNT as u64)?;
        let projections = scale_work(decoder, executions)?.checked_add(group_work(logits, H, group_width)?)?;
        Ok(Self { forward_positions: n, projected_logits: u64::try_from(logits).map_err(|_| StrictInt4Error::Work)?,
            attention_pairs, projections })
    }
}
fn group_work(rows: usize, columns: usize, width: usize) -> Result<ProjectionWork, StrictInt4Error> {
    let mut work = ProjectionWork::for_shape(rows, columns)?;
    work.dot_products = multiply(work.dot_products, columns.div_ceil(width) as u64)?;
    Ok(work)
}
fn scale_work(work: ProjectionWork, copies: u64) -> Result<ProjectionWork, StrictInt4Error> {
    Ok(ProjectionWork { dot_products: multiply(work.dot_products, copies)?,
        multiply_accumulates: multiply(work.multiply_accumulates, copies)? })
}
fn multiply(a: u64, b: u64) -> Result<u64, StrictInt4Error> { a.checked_mul(b).ok_or(StrictInt4Error::Work) }

#[derive(Clone, Copy, Debug)]
pub struct Int4RunBudget {
    pub max_forward_positions: u64, pub max_attention_pairs: u64, pub max_projection_work: ProjectionWork,
}
impl Int4RunBudget {
    pub fn exact(work: Int4Work) -> Self {
        Self { max_forward_positions: work.forward_positions, max_attention_pairs: work.attention_pairs,
            max_projection_work: work.projections }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Int4MemoryBudget {
    pub max_kv_bytes: u64, pub max_rope_bytes: u64, pub max_scratch_payload_bytes: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4MemoryRequirement {
    pub kv_bytes: u64, pub rope_bytes: u64, pub scratch_payload_bound: u64,
}
impl Int4MemoryRequirement {
    pub fn for_context(positions: usize, group_width: usize) -> Result<Self, StrictInt4Error> {
        if positions == 0 || positions > DEFAULT_ADMITTED_CONTEXT_CAP { return Err(StrictInt4Error::Context); }
        if group_width == 0 || group_width > MAX_MODEL_K { return Err(StrictInt4Error::Memory); }
        let n = positions as u64;
        Ok(Self { kv_bytes: n * KV_BYTES_PER_TOKEN as u64,
            rope_bytes: (n * NANBEIGE_HEAD_DIM as u64 + (NANBEIGE_HEAD_DIM / 2) as u64) * 4,
            // Same conservative simultaneous reference-vector allowance as
            // strict INT8, plus the reusable group unpacking payload. Weights,
            // allocator slack, caller output and host overhead are separate.
            scratch_payload_bound: (32 * I + V + NANBEIGE_HEAD_DIM) as u64 * 4 + 16 * n + group_width as u64 })
    }
    fn check(self, budget: Int4MemoryBudget) -> Result<(), StrictInt4Error> {
        if self.kv_bytes > budget.max_kv_bytes || self.rope_bytes > budget.max_rope_bytes
            || self.scratch_payload_bound > budget.max_scratch_payload_bytes { return Err(StrictInt4Error::Memory); }
        Ok(())
    }
}

#[derive(Default)]
struct RunState { active: bool, poisoned: bool }
impl RunState {
    fn open(&mut self) -> Result<(), StrictInt4Error> {
        if self.active || self.poisoned { return Err(StrictInt4Error::EngineUnavailable); }
        self.active = true; Ok(())
    }
    fn check(&self) -> Result<(), StrictInt4Error> {
        if !self.active || self.poisoned { Err(StrictInt4Error::EngineUnavailable) } else { Ok(()) }
    }
}

/// Single-sequence, caller-admitted semantic engine. Immutable weights are
/// borrowed once; session exclusivity owns the logical KV lifetime. No IO,
/// network, implicit dispatch, runtime or worker spawn occurs here.
pub struct StrictInt4Engine<'weights> {
    weights: &'weights Int4ModelWeights, cache: KvCache, rope: RopeTablesF32,
    activation: ActivationBuffer, unpack: Vec<i8>, kernel: PackedInt4Kernel, state: RunState,
}
impl<'weights> StrictInt4Engine<'weights> {
    pub fn new(weights: &'weights Int4ModelWeights, context: usize, budget: Int4MemoryBudget) -> Result<Self, StrictInt4Error> {
        Int4MemoryRequirement::for_context(context, weights.group_width())?.check(budget)?;
        Ok(Self { weights, cache: KvCache::try_with_capacity(context).map_err(|_| StrictInt4Error::Allocation)?,
            rope: RopeTablesF32::nanbeige(context).map_err(|_| StrictInt4Error::Rope)?,
            activation: ActivationBuffer::try_new(I)?, unpack: filled(weights.group_width(), 0_i8)?,
            kernel: PackedInt4Kernel::scalar(), state: RunState::default() })
    }
    /// The caller supplies a real-host-checked handle before opening a session.
    pub fn with_kernel(mut self, kernel: PackedInt4Kernel) -> Self { self.kernel = kernel; self }
    pub fn kernel_kind(&self) -> Int8KernelKind { self.kernel.kind() }
    pub fn profile(&self) -> &'static str { STRICT_INT4_PROFILE }
    pub fn model_identity(&self) -> &Int4ModelIdentity { self.weights.identity() }
    pub fn kv_cache(&self) -> &KvCache { &self.cache }
    pub fn is_poisoned(&self) -> bool { self.state.poisoned }
    pub fn session<'run, C: DecodeStepControl>(&'run mut self, budget: Int4RunBudget, control: &'run mut C)
        -> Result<Int4Session<'run, 'weights, C>, StrictInt4Error> {
        if !self.cache.all_slots_have_len(0) { return Err(StrictInt4Error::Cache); }
        self.state.open()?;
        Ok(Int4Session { engine: self, control, remaining_positions: budget.max_forward_positions,
            remaining_attention: budget.max_attention_pairs, ledger: ProjectionLedger::new(budget.max_projection_work),
            hidden: None, work: Int4Work::default() })
    }
}

/// Drop clears all 44 slots without discarding their allocated capacity.
/// Native failures and unwinding poison the engine permanently. Preflight
/// refusal does not discard a valid prefix; forgetting a session blocks reuse.
pub struct Int4Session<'run, 'weights, C: DecodeStepControl> {
    engine: &'run mut StrictInt4Engine<'weights>, control: &'run mut C,
    remaining_positions: u64, remaining_attention: u64, ledger: ProjectionLedger,
    hidden: Option<Vec<Bf16>>, work: Int4Work,
}
impl<C: DecodeStepControl> Int4Session<'_, '_, C> {
    pub fn work(&self) -> Int4Work { Int4Work { projections: self.ledger.reserved(), ..self.work } }
    pub fn position(&self) -> Result<usize, StrictInt4Error> {
        self.engine.state.check()?;
        let position = self.engine.cache.len_for_slot(0).map_err(|_| StrictInt4Error::Cache)?;
        if !self.engine.cache.all_slots_have_len(position) { return Err(StrictInt4Error::Cache); }
        Ok(position)
    }
    pub fn preflight(&self, positions: usize, projected_logits: usize) -> Result<Int4Work, StrictInt4Error> {
        let start = self.position()?;
        if start.checked_add(positions).is_none_or(|end| end > self.engine.cache.capacity_positions()) { return Err(StrictInt4Error::Context); }
        let bound = Int4Work::for_sequence(self.engine.weights.group_width(), start, positions, projected_logits)?;
        if bound.forward_positions > self.remaining_positions || bound.attention_pairs > self.remaining_attention { return Err(StrictInt4Error::Work); }
        self.ledger.preflight(bound.projections)?;
        Ok(bound)
    }
    pub fn append(&mut self, token: u32) -> Result<(), StrictInt4Error> {
        if token as usize >= V { return Err(StrictInt4Error::Input); }
        let bound = self.preflight(1, 0)?;
        let position = self.position()?;
        self.remaining_positions -= 1; self.remaining_attention -= bound.attention_pairs;
        self.engine.state.poisoned = true;
        self.hidden = None;
        poll(self.control)?;
        let row = self.engine.weights.embeddings.row(token as usize).map_err(|_| StrictInt4Error::Input)?;
        let mut hidden = filled(H, Bf16::from_bits(0))?;
        hidden.copy_from_slice(row); finite(&hidden)?;
        let runner = LoopRunner::from_layer_weights(&self.engine.weights.layers);
        let mut executor = Executor { final_norm: &self.engine.weights.final_norm, rope: &self.engine.rope,
            activation: &mut self.engine.activation, unpack: &mut self.engine.unpack, kernel: self.engine.kernel,
            ledger: &mut self.ledger, control: &mut *self.control, completed: 0, norms: 0 };
        runner.run_token(&mut executor, &mut hidden, PositionContext::at(position), &mut self.engine.cache)?;
        if executor.completed != KV_SLOT_COUNT || executor.norms != 2 { return Err(StrictInt4Error::Boundary); }
        if !self.engine.cache.all_slots_have_len(position + 1) { return Err(StrictInt4Error::Cache); }
        finite(&hidden)?; poll(self.control)?;
        self.hidden = Some(hidden);
        self.work.forward_positions += 1; self.work.attention_pairs += bound.attention_pairs;
        self.engine.state.poisoned = false; Ok(())
    }
    /// Selected rows are evaluated directly; the remaining vocabulary is not
    /// read or dequantized. Only a completely successful result is returned.
    pub fn logits(&mut self, rows: LinearRows<'_>) -> Result<Vec<f32>, StrictInt4Error> {
        let count = rows.checked_count(V)?;
        self.preflight(0, count)?;
        if self.hidden.is_none() { return Err(StrictInt4Error::EmptyHidden); }
        let next = self.work.projected_logits.checked_add(count as u64).ok_or(StrictInt4Error::Work)?;
        let mut output = filled(count, 0.0_f32)?;
        self.engine.state.poisoned = true;
        self.engine.activation.encode_bf16(self.hidden.as_deref().ok_or(StrictInt4Error::EmptyHidden)?)?;
        self.engine.weights.head.as_linear().with_kernel(self.engine.kernel).project_f32_into(
            &self.engine.activation, rows, &mut output, &mut self.engine.unpack, &mut self.ledger, &mut *self.control)?;
        self.work.projected_logits = next; self.engine.state.poisoned = false; Ok(output)
    }
    pub fn prefill(&mut self, tokens: &[u32], rows: LinearRows<'_>) -> Result<Vec<f32>, StrictInt4Error> {
        if tokens.is_empty() || tokens.iter().any(|&token| token as usize >= V) { return Err(StrictInt4Error::Input); }
        self.preflight(tokens.len(), rows.checked_count(V)?)?;
        for &token in tokens { self.append(token)?; }
        self.logits(rows)
    }
}
impl<C: DecodeStepControl> Drop for Int4Session<'_, '_, C> {
    fn drop(&mut self) { end_session(&mut self.engine.cache, &mut self.engine.state); }
}
fn end_session(cache: &mut KvCache, state: &mut RunState) {
    state.poisoned |= std::thread::panicking(); cache.clear(); state.active = false;
}

struct Executor<'a, C> {
    final_norm: &'a [Bf16], rope: &'a RopeTablesF32, activation: &'a mut ActivationBuffer,
    unpack: &'a mut [i8], kernel: PackedInt4Kernel, ledger: &'a mut ProjectionLedger,
    control: &'a mut C, completed: usize, norms: usize,
}
impl<C: DecodeStepControl> LayerExecutor<Int4Layer> for Executor<'_, C> {
    type Hidden = Vec<Bf16>;
    type Error = StrictInt4Error;
    fn layer_forward(&mut self, binding: &LayerBinding<'_, Int4Layer>, hidden: &mut Self::Hidden,
        positions: PositionContext, cache: &mut KvCache) -> Result<(), Self::Error> {
        if slot_for(binding.loop_index(), binding.layer_index()) != Some(binding.kv_slot())
            || positions != PositionContext::at(positions.position) { return Err(StrictInt4Error::Boundary); }
        poll(self.control)?;
        let layer = binding.weights();
        let normalized = norm(hidden, &layer.norm1)?;
        self.activation.encode_bf16(&normalized)?;
        let mut query = self.project(&layer.q)?;
        let mut key = self.project(&layer.k)?;
        let value = self.project(&layer.v)?;
        for head in query.chunks_exact_mut(NANBEIGE_HEAD_DIM) { self.rope.apply_split_half(positions.rope_position, head).map_err(|_| StrictInt4Error::Rope)?; }
        for head in key.chunks_exact_mut(NANBEIGE_HEAD_DIM) { self.rope.apply_split_half(positions.rope_position, head).map_err(|_| StrictInt4Error::Rope)?; }
        finite(&query)?; finite(&key)?;
        let mut key_bits = filled(K, 0_u16)?; let mut value_bits = filled(K, 0_u16)?;
        for (out, value) in key_bits.iter_mut().zip(&key) { *out = value.to_bits(); }
        for (out, value) in value_bits.iter_mut().zip(&value) { *out = value.to_bits(); }
        cache.append(binding.kv_slot(), positions.cache_position, &key_bits, &value_bits).map_err(|_| StrictInt4Error::Cache)?;
        poll(self.control)?;
        let attention = eager_gqa_attention_from_cache(&query, cache, binding.kv_slot()).map_err(|_| StrictInt4Error::Attention)?;
        finite(&attention)?; poll(self.control)?;
        self.activation.encode_bf16(&attention)?;
        let update = self.project(&layer.o)?;
        let residual = residual_add_f32_cast_back(hidden, &update).map_err(|_| StrictInt4Error::Primitive)?;
        let normalized = norm(&residual, &layer.norm2)?;
        self.activation.encode_bf16(&normalized)?;
        let gate = self.project(&layer.gate)?;
        let up = self.project(&layer.up)?;
        let product = swiglu_f32_cast_back(&gate, &up).map_err(|_| StrictInt4Error::Primitive)?;
        self.activation.encode_bf16(&product)?;
        let update = self.project(&layer.down)?;
        let output = residual_add_f32_cast_back(&residual, &update).map_err(|_| StrictInt4Error::Primitive)?;
        finite(&output)?; *hidden = output; self.completed += 1; Ok(())
    }
    fn final_rms_norm(&mut self, hidden: &mut Self::Hidden, _: PositionContext) -> Result<(), Self::Error> {
        if self.norms >= 2 || self.completed != (self.norms + 1) * PHYSICAL_LAYER_COUNT { return Err(StrictInt4Error::Boundary); }
        poll(self.control)?; *hidden = norm(hidden, self.final_norm)?; self.norms += 1; Ok(())
    }
}
impl<C: DecodeStepControl> Executor<'_, C> {
    fn project(&mut self, matrix: &GroupedInt4Matrix) -> Result<Vec<Bf16>, StrictInt4Error> {
        let mut output = filled(matrix.rows(), Bf16::from_bits(0))?;
        matrix.as_linear().with_kernel(self.kernel).project_bf16_into(self.activation, LinearRows::All,
            &mut output, self.unpack, self.ledger, self.control)?;
        Ok(output)
    }
}
fn norm(input: &[Bf16], scale: &[Bf16]) -> Result<Vec<Bf16>, StrictInt4Error> {
    finite(input)?;
    let sum = input.iter().map(|v| v.to_f32() * v.to_f32()).sum::<f32>();
    if !sum.is_finite() { return Err(StrictInt4Error::Primitive); }
    let result = rms_norm_f32_reduce_cast_back(input, scale, RMS_NORM_EPSILON).map_err(|_| StrictInt4Error::Primitive)?;
    finite(&result)?; Ok(result)
}
fn finite(values: &[Bf16]) -> Result<(), StrictInt4Error> {
    if values.iter().all(|value| value.to_f32().is_finite()) { Ok(()) } else { Err(StrictInt4Error::Primitive) }
}
fn filled<T: Clone>(length: usize, value: T) -> Result<Vec<T>, StrictInt4Error> {
    let mut output = Vec::new(); output.try_reserve_exact(length).map_err(|_| StrictInt4Error::Allocation)?;
    output.resize(length, value); Ok(output)
}
fn poll<C: DecodeStepControl>(control: &mut C) -> Result<(), StrictInt4Error> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(StrictInt4Error::Cancelled(cause)), None => Ok(()) }
}

#[cfg(test)] mod tests;
