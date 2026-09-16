//! Bounded multi-sequence generation on the real layer-major eager engine.
//!
//! Every scheduling tick advances one token per unfinished row. Short prompts
//! can already decode while long prompts are still prefilling. Only final
//! prompt/feedback positions project logits. Rows stop independently; no EOS,
//! byte-refused proposal, or final budget token is unnecessarily fed back.
//!
//! The scalar driver's Cursor owns sampling, byte decoding, stop semantics and
//! two-phase token delivery here too. This is not a second sampler. Admission
//! must cover the entire bounded cohort before execution; no dynamic daemon,
//! retry, model loader or additional runtime is hidden in this API.

use super::*;
use super::cursor::Cursor;
use crate::native_engine::batchsched::{
    BatchControl, BatchError, BatchProjection, BatchSequence, BatchToken, EagerBatchEngine, MAX_BATCH_ROWS,
};

pub const BATCH_GENERATION_VERSION: &str = "eager-addressed-batch-generation-v1";

/// The host's exact admitted request plus a selected empty physical cache slot.
/// Slot/request_seq never modify the plan's private semantic sampling address.
pub struct BatchGenerationRequest<'a> {
    pub plan: &'a GenerationPlan,
    pub admitted_identity: &'a ExecutionIdentity,
    pub slot: usize,
    pub request_seq: u64,
}
#[derive(Clone, Copy, Debug)]
pub struct BatchGenerationBudget {
    pub max_forward_positions: u64,
    pub max_projected_logits: u64,
    /// Entire arena K/V, shared tables and native transient payload, not just
    /// the requested subset. Weights/RSS/allocator margin remain host-owned.
    pub max_native_payload_bytes: u64,
    pub max_sampler_payload_bytes: u64,
    /// Returned bytes/token arrays plus bounded pending event/prefix buffers.
    /// Decoder-internal work and serialized output envelopes are host-owned.
    pub max_output_payload_bytes: u64,
}
#[derive(Clone, Copy, Debug)]
pub struct BatchGenerationRequirements {
    pub planned_work: GenerationWork,
    pub native_payload_bytes: u64,
    pub sampler_payload_bytes: u64,
    pub output_payload_upper_bytes: u64,
}
pub struct BatchGenerationOutput {
    /// Original input order, regardless of each row's finish tick.
    pub sequences: Vec<GeneratedSequence>,
    pub group_steps: u64,
    pub planned_work: GenerationWork,
    pub actual_work: GenerationWork,
}

/// Reject the entire cohort before opening any cache slot or allocating sampler
/// state. loaded_model is the embedding host's actual loaded-model identity,
/// NOT a certificate fabricated by this module. Its common model/tokenizer/
/// backend projection must match every independently admitted row exactly.
pub fn preflight(engine: &EagerBatchEngine<'_>, loaded_model: &ExecutionIdentity,
    requests: &[BatchGenerationRequest<'_>], budget: BatchGenerationBudget)
    -> Result<BatchGenerationRequirements, GenerationError> {
    if requests.is_empty() || requests.len() > MAX_BATCH_ROWS { return Err(GenerationError::Contract("batch width")); }
    loaded_model.validate().map_err(|_| GenerationError::Identity)?;
    let expected_model = model_binding(loaded_model)?;
    let mut required = requirements(requests, engine.envelope().payload().total_bytes)?;
    for (index, request) in requests.iter().enumerate() {
        request.plan.verify_identity(request.admitted_identity)?;
        if model_binding(request.admitted_identity)? != expected_model { return Err(GenerationError::Identity); }
        if request.request_seq == 0 || requests[..index].iter().any(|prior| prior.request_seq == request.request_seq || prior.slot == request.slot) {
            return Err(GenerationError::Contract("duplicate slot or delivery sequence"));
        }
        let positions = usize::try_from(request.plan.bound.forward_positions).map_err(|_| GenerationError::Limit("batch positions"))?;
        engine.preflight_sequence_slot(request.slot, positions).map_err(native_error)?;
    }
    // Keep this explicit even when an embedding passes an oversized envelope.
    required.native_payload_bytes = engine.envelope().payload().total_bytes;
    check_budget(required, budget)?;
    Ok(required)
}

pub fn execute<D: DecodeByteDecoder, C: DecodeStepControl>(engine: &mut EagerBatchEngine<'_>,
    loaded_model: &ExecutionIdentity, requests: &[BatchGenerationRequest<'_>], budget: BatchGenerationBudget,
    decoder: &D, control: &mut C) -> Result<BatchGenerationOutput, GenerationError> {
    execute_with_sink(engine, loaded_model, requests, budget, decoder, &mut Discard, control)
}

/// Streaming is token-committed, not atomic across the entire cohort. A later
/// failure/cancellation returns an error, never an apparently successful partial
/// batch; already delivered token events are not retracted or retried. The
/// request_seq on every event demultiplexes independent row-local token indexes.
/// All opened cache handles are closed on success/error/unwind, including rows
/// that were not in the final failed native step. Unrelated busy slots survive.
pub fn execute_with_sink<D: DecodeByteDecoder, S: DecodeEventSink, C: DecodeStepControl>(
    engine: &mut EagerBatchEngine<'_>, loaded_model: &ExecutionIdentity,
    requests: &[BatchGenerationRequest<'_>], budget: BatchGenerationBudget, decoder: &D,
    sink: &mut S, control: &mut C) -> Result<BatchGenerationOutput, GenerationError> {
    let required = preflight(engine, loaded_model, requests, budget)?;
    checkpoint(control, 0)?;
    let mut cursors = reserved(requests.len())?;
    for request in requests { cursors.push(Cursor::new(request.plan, request.request_seq, BATCH_GENERATION_VERSION)?); }
    let mut native = NativeGroup { engine, handles: reserved(requests.len())? };
    for request in requests {
        let handle = native.engine.open_sequence(request.slot).map_err(native_error)?;
        native.handles.push(handle);
    }
    let group_steps = drive(&mut cursors, &mut native, decoder, sink, control)?;
    let mut sequences = reserved(cursors.len())?;
    let mut actual_work = GenerationWork::default();
    for row in cursors {
        let result = row.finish()?; actual_work = sum_work(actual_work, result.native_work)?; sequences.push(result);
    }
    if actual_work.forward_positions > required.planned_work.forward_positions
        || actual_work.projected_logits > required.planned_work.projected_logits
        || actual_work.sampled_steps > required.planned_work.sampled_steps {
        return Err(GenerationError::Contract("batch exceeded preflight work"));
    }
    Ok(BatchGenerationOutput { sequences, group_steps, planned_work: required.planned_work, actual_work })
}

struct ScheduledToken { row: usize, token: u32, project: bool }
struct ForwardedRow { row: usize, position: usize, logits: Option<Vec<f32>> }
/// Static internal test seam. Public execution always uses NativeGroup.
trait ForwardGroup {
    fn step<C: DecodeStepControl>(&mut self, tokens: &[ScheduledToken], control: &mut C) -> Result<Vec<ForwardedRow>, GenerationError>;
    fn close(&mut self, row: usize) -> Result<(), GenerationError>;
}
fn drive<D: DecodeByteDecoder, S: DecodeEventSink, C: DecodeStepControl, F: ForwardGroup>(
    cursors: &mut [Cursor<'_>], forward: &mut F, decoder: &D, sink: &mut S, control: &mut C) -> Result<u64, GenerationError> {
    let mut steps = 0_u64;
    loop {
        let mut tokens = reserved(cursors.len())?;
        for (index, cursor) in cursors.iter().enumerate() {
            if cursor.done { continue; }
            cursor.before_forward(control)?;
            let (token, project) = cursor.next_token()?;
            tokens.push(ScheduledToken { row: index, token, project });
        }
        if tokens.is_empty() { return Ok(steps); }
        let results = forward.step(&tokens, control)?;
        steps = plus(steps, 1)?;
        if results.len() != tokens.len() { return Err(GenerationError::Contract("batch forward row count")); }
        for (scheduled, result) in tokens.into_iter().zip(results) {
            let cursor = &mut cursors[scheduled.row];
            if result.row != scheduled.row || result.position as u64 != cursor.output.native_work.forward_positions
                || result.logits.is_some() != scheduled.project {
                return Err(GenerationError::Contract("batch forward routing or projection"));
            }
            cursor.record_forward(scheduled.project)?;
            if let Some(logits) = result.logits {
                check_logits(&logits)?; cursor.emit_next(&logits, decoder, sink, control)?;
            }
            if cursor.done { forward.close(scheduled.row)?; }
        }
    }
}

struct NativeGroup<'engine, 'weights> {
    engine: &'engine mut EagerBatchEngine<'weights>, handles: Vec<BatchSequence>,
}
struct NativeControl<'a, C>(&'a mut C);
impl<C: DecodeStepControl> BatchControl for NativeControl<'_, C> {
    // Stage checkpoints are not progress notifications for a particular row.
    fn checkpoint(&mut self) -> Option<DecodeCancellationKind> { self.0.checkpoint(0) }
}
impl ForwardGroup for NativeGroup<'_, '_> {
    fn step<C: DecodeStepControl>(&mut self, tokens: &[ScheduledToken], control: &mut C) -> Result<Vec<ForwardedRow>, GenerationError> {
        let mut input = reserved(tokens.len())?;
        for token in tokens {
            let sequence = *self.handles.get(token.row).ok_or(GenerationError::Contract("batch handle routing"))?;
            input.push(BatchToken { sequence, token_id: token.token, projection: if token.project {
                BatchProjection::FullVocabulary
            } else { BatchProjection::None } });
        }
        let result = self.engine.step(&input, &mut NativeControl(control)).map_err(native_error)?;
        if result.rows.len() != input.len() { return Err(GenerationError::Contract("native batch row count")); }
        let mut output = reserved(tokens.len())?;
        for ((row, input), scheduled) in result.rows.into_iter().zip(&input).zip(tokens) {
            if row.sequence != input.sequence { return Err(GenerationError::Contract("native batch handle mismatch")); }
            output.push(ForwardedRow { row: scheduled.row, position: row.position, logits: row.logits });
        }
        Ok(output)
    }
    fn close(&mut self, row: usize) -> Result<(), GenerationError> {
        let sequence = *self.handles.get(row).ok_or(GenerationError::Contract("batch close routing"))?;
        self.engine.close_sequence(sequence).map_err(native_error)
    }
}
impl Drop for NativeGroup<'_, '_> {
    fn drop(&mut self) {
        for &sequence in &self.handles {
            // Completed rows and rows retired by native failure are already
            // stale. Generation-scoped handles cannot target a recycled slot.
            let _ = self.engine.close_sequence(sequence);
        }
    }
}
fn requirements(requests: &[BatchGenerationRequest<'_>], native_payload_bytes: u64)
    -> Result<BatchGenerationRequirements, GenerationError> {
    let mut required = BatchGenerationRequirements { planned_work: GenerationWork::default(), native_payload_bytes,
        sampler_payload_bytes: 0, output_payload_upper_bytes: 0 };
    for request in requests {
        let p = &request.plan.options;
        required.planned_work = sum_work(required.planned_work, GenerationWork {
            forward_positions: request.plan.bound.forward_positions,
            projected_logits: times(p.max_new_tokens as u64, NANBEIGE_VOCAB_SIZE as u64)?,
            sampled_steps: request.plan.bound.sampled_steps,
        })?;
        required.sampler_payload_bytes = plus(required.sampler_payload_bytes, request.plan.sampler_bytes)?;
        let tokens_and_scores = times(p.max_new_tokens as u64, 8 + if p.capture_logprobs { 4 } else { 0 })?;
        let buffers = plus(times(p.max_output_bytes as u64, 3)?, tokens_and_scores)?;
        required.output_payload_upper_bytes = plus(required.output_payload_upper_bytes, buffers)?;
    }
    Ok(required)
}
fn check_budget(r: BatchGenerationRequirements, b: BatchGenerationBudget) -> Result<(), GenerationError> {
    for (actual, maximum, axis) in [
        (r.planned_work.forward_positions, b.max_forward_positions, "batch forward work"),
        (r.planned_work.projected_logits, b.max_projected_logits, "batch projection work"),
        (r.native_payload_bytes, b.max_native_payload_bytes, "batch native payload"),
        (r.sampler_payload_bytes, b.max_sampler_payload_bytes, "batch sampler payload"),
        (r.output_payload_upper_bytes, b.max_output_payload_bytes, "batch output payload"),
    ] { if actual > maximum { return Err(GenerationError::Limit(axis)); } }
    Ok(())
}
fn model_binding(identity: &ExecutionIdentity) -> Result<Vec<u8>, GenerationError> {
    // Prompt/task/seed/policy may differ by row. Loaded weights, tokenizer assets
    // and selected numeric/backend semantics may not. Template is retained here
    // because pinned chat's template digest also closes its tokenizer assets.
    canonjson::canonical_bytes(&(&identity.source_revision, identity.logical_model_digest,
        &identity.artifact_format, &identity.quant_recipe, identity.packing_set_digest,
        identity.tokenizer_digest, identity.template_digest, &identity.numerics_profile,
        &identity.kv_dtype, &identity.backend_semantic_version, &identity.host_class, &identity.compiler_identity))
        .map_err(|_| GenerationError::Identity)
}
fn sum_work(a: GenerationWork, b: GenerationWork) -> Result<GenerationWork, GenerationError> {
    Ok(GenerationWork { forward_positions: plus(a.forward_positions, b.forward_positions)?,
        projected_logits: plus(a.projected_logits, b.projected_logits)?, sampled_steps: plus(a.sampled_steps, b.sampled_steps)? })
}
fn plus(a: u64, b: u64) -> Result<u64, GenerationError> { a.checked_add(b).ok_or(GenerationError::Limit("batch arithmetic")) }
fn times(a: u64, b: u64) -> Result<u64, GenerationError> { a.checked_mul(b).ok_or(GenerationError::Limit("batch arithmetic")) }
fn native_error(error: BatchError) -> GenerationError {
    match error {
        BatchError::Native(error) => GenerationError::Engine(error),
        BatchError::Cancelled(cause) => GenerationError::Cancelled(cause),
        BatchError::Allocation => GenerationError::Allocation,
        BatchError::InvalidNumerics => GenerationError::InvalidLogits,
        BatchError::Limit(axis) => GenerationError::Limit(axis),
        BatchError::ContextFull => GenerationError::Limit("batch context"),
        BatchError::SequenceBusy => GenerationError::EngineAlreadyPrimed,
        _ => GenerationError::Contract("native batch sequence state"),
    }
}
#[cfg(test)] mod tests;
