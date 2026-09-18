//! Actual portable-int8 generation through the existing sampler/stream cursor.
//!
//! No alternate sampler, tokenizer, template, model loader or runtime. This
//! plan is a distinct type; its private inner plan cannot enter eager drivers.
//! The caller admits weights and this exact execution identity, retains genuine
//! resource guards through output delivery, and supervises panics. Artifact
//! authenticity, OQ-30 profile ratification and quantized quality remain open.

use super::*;
use crate::native_engine::{
    artifact_bridge::ArtifactIdentity,
    portable_int8::LinearRows,
    rope::DEFAULT_ADMITTED_CONTEXT_CAP,
    strict_int8::{Int8RunBudget, Int8Session, Int8Work, StrictInt8Engine, StrictInt8Error,
        STRICT_INT8_EXECUTION, STRICT_INT8_PROFILE},
};

pub const INT8_GENERATION_VERSION: &str = "portable-int8-addressed-generation-final-prefill-head-v1";

#[derive(Clone, Copy, Debug)]
pub struct Int8GenerationBudget {
    pub native: Int8RunBudget,
    /// Complete resident KV capacity, not only this request's live positions.
    pub max_kv_bytes: u64,
    /// Existing sampler's independently admitted payload workspace.
    pub max_sampler_bytes: u64,
}

/// Completed content plus the complete integer decoder/head and attention work.
/// Sequence logprobs use the full raw quantized-model vocabulary, not processed
/// sampling probabilities. Content-byte limits do not include JSON metadata.
#[derive(Clone, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int8GenerationRun {
    pub schema_version: u32,
    pub sequence: GeneratedSequence,
    pub model_work: Int8Work,
}

#[derive(Debug)]
pub enum Int8GenerationError {
    Generation(GenerationError), Native(StrictInt8Error), ModelIdentity, WorkMismatch,
}
impl fmt::Display for Int8GenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Generation(error) => write!(f, "int8 generation refused: {error}"),
            Self::Native(error) => write!(f, "int8 generation native execution refused: {error}"),
            Self::ModelIdentity => f.write_str("int8 generation model/profile identity mismatch"),
            Self::WorkMismatch => f.write_str("int8 generation completed work disagrees with the native session"),
        }
    }
}
impl Error for Int8GenerationError {}
impl From<GenerationError> for Int8GenerationError {
    fn from(error: GenerationError) -> Self { Self::Generation(error) }
}
impl From<StrictInt8Error> for Int8GenerationError {
    fn from(error: StrictInt8Error) -> Self { Self::Native(error) }
}
impl Int8GenerationError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Generation(GenerationError::Cancelled(cause))
                | Self::Native(StrictInt8Error::Cancelled(cause)) => Some(*cause),
            _ => None,
        }
    }
}

/// Immutable private request; no Deref, conversion to GenerationPlan, or
/// deserialization lets caller data substitute a different native profile.
pub struct Int8GenerationPlan { plan: GenerationPlan, work: Int8Work }
impl Int8GenerationPlan {
    /// Compile exact already-tokenized input under an explicitly supplied
    /// strict-quantized-v1 identity and STRICT_INT8_EXECUTION backend. Never
    /// rewrites a BF16 identity or claims the materialized source is authentic.
    pub fn compile(prompt: Vec<u32>, options: GenerationOptions, identity: ExecutionIdentity,
        item_id: &str, sample_index: u64, limits: GenerationLimits) -> Result<Self, Int8GenerationError> {
        let plan = GenerationPlan::compile_for_backend(prompt, options, identity, item_id, sample_index,
            limits, GenerationBackend::Int8)?;
        let positions = usize::try_from(plan.bound.forward_positions).map_err(|_| StrictInt8Error::Work)?;
        let head_rows = usize::try_from(plan.bound.projected_logits).map_err(|_| StrictInt8Error::Work)?;
        if positions > DEFAULT_ADMITTED_CONTEXT_CAP { return Err(StrictInt8Error::Context.into()); }
        let work = Int8Work::for_sequence(0, positions, head_rows)?;
        Ok(Self { plan, work })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.plan.execution_identity() }
    pub fn options(&self) -> &GenerationOptions { self.plan.options() }
    pub fn prompt_tokens(&self) -> usize { self.plan.prompt_tokens() }
    pub fn sampler_bytes(&self) -> u64 { self.plan.sampler_bytes() }
    pub fn planned_work(&self) -> Int8Work { self.work }
    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), Int8GenerationError> {
        self.plan.verify_identity(admitted).map_err(Int8GenerationError::from)
    }
    pub fn preflight(&self, admitted: &ExecutionIdentity, engine: &StrictInt8Engine<'_>, budget: Int8GenerationBudget)
        -> Result<(), Int8GenerationError> {
        self.verify_identity(admitted)?;
        check_model(admitted, engine.artifact_identity())?;
        if engine.profile() != STRICT_INT8_PROFILE || engine.is_poisoned() {
            return Err(StrictInt8Error::EngineUnavailable.into());
        }
        if !engine.kv_cache().all_slots_have_len(0) { return Err(GenerationError::EngineAlreadyPrimed.into()); }
        check_bounds(self.work, self.sampler_bytes(), engine.kv_cache().capacity_positions(), budget)
    }
    pub fn execute<D: DecodeByteDecoder, C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt8Engine<'_>, decoder: &D, request_seq: u64, budget: Int8GenerationBudget, control: &mut C)
        -> Result<Int8GenerationRun, Int8GenerationError> {
        self.execute_with_sink(admitted, engine, decoder, request_seq, budget, &mut Discard, control)
    }
    /// Preflight the complete request, then run one exclusively borrowed native
    /// session. Every fatal result aborts the session, including decoder/sink
    /// failures after model work. No retry or successful cancellation envelope.
    pub fn execute_with_sink<D: DecodeByteDecoder, S: DecodeEventSink, C: DecodeStepControl>(&self,
        admitted: &ExecutionIdentity, engine: &mut StrictInt8Engine<'_>, decoder: &D, request_seq: u64,
        budget: Int8GenerationBudget, sink: &mut S, control: &mut C) -> Result<Int8GenerationRun, Int8GenerationError> {
        self.preflight(admitted, engine, budget)?;
        let mut session = engine.session(budget.native, control)?;
        self.drive(decoder, request_seq, sink, &mut session)
    }
    fn drive<D: DecodeByteDecoder, S: DecodeEventSink, B: Driver>(&self,
        decoder: &D, request_seq: u64, sink: &mut S, driver: &mut B) -> Result<Int8GenerationRun, Int8GenerationError> {
        let result = (|| {
            let mut row = cursor::Cursor::new(&self.plan, request_seq, INT8_GENERATION_VERSION)?;
            while !row.done {
                row.before_forward(driver.control())?;
                let (token, needs_selection) = row.next_token()?;
                driver.append(token)?;
                if needs_selection {
                    let logits = driver.logits()?;
                    check_logits(&logits)?;
                    row.record_forward(true)?;
                    row.emit_next(&logits, decoder, sink, driver.control())?;
                } else { row.record_forward(false)?; }
            }
            let sequence = row.finish()?;
            let model_work = driver.work();
            check_completed(sequence.native_work, model_work)?;
            Ok(Int8GenerationRun { schema_version: 1, sequence, model_work })
        })();
        if result.is_err() { driver.abort(); }
        result
    }
}

fn check_model(identity: &ExecutionIdentity, source: &ArtifactIdentity) -> Result<(), Int8GenerationError> {
    // Compare the actual materialized view, not just caller-declared plan IDs.
    // The bridge has no packing/authenticity certificate; those host-owned
    // fields remain exactly bound in ExecutionIdentity, never fabricated here.
    if identity.numerics_profile != (NumericsProfile::StrictQuantized { version: 1 })
        || identity.backend_semantic_version != STRICT_INT8_EXECUTION || identity.kv_dtype != "bf16"
        || source.model_id != "Nanbeige4.2-3B" || source.revision != identity.source_revision
        || source.recipe_id != identity.quant_recipe
        || Sha256Digest::from_hex(&source.logical_model_sha256).ok() != Some(identity.logical_model_digest) {
        return Err(Int8GenerationError::ModelIdentity);
    }
    Ok(())
}
fn check_bounds(work: Int8Work, sampler_bytes: u64, capacity: usize, budget: Int8GenerationBudget)
    -> Result<(), Int8GenerationError> {
    let capacity = u64::try_from(capacity).map_err(|_| StrictInt8Error::Context)?;
    if work.forward_positions > capacity { return Err(StrictInt8Error::Context.into()); }
    let kv = capacity.checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or(StrictInt8Error::Memory)?;
    if kv > budget.max_kv_bytes || sampler_bytes > budget.max_sampler_bytes { return Err(StrictInt8Error::Memory.into()); }
    if work.forward_positions > budget.native.max_forward_positions
        || work.attention_pairs > budget.native.max_attention_pairs
        || !work.projections.fits(budget.native.max_projection_work) { return Err(StrictInt8Error::Work.into()); }
    Ok(())
}
fn check_completed(cursor: GenerationWork, native: Int8Work) -> Result<(), Int8GenerationError> {
    let positions = usize::try_from(cursor.forward_positions).map_err(|_| Int8GenerationError::WorkMismatch)?;
    let rows = usize::try_from(cursor.projected_logits).map_err(|_| Int8GenerationError::WorkMismatch)?;
    if native != Int8Work::for_sequence(0, positions, rows)? { return Err(Int8GenerationError::WorkMismatch); }
    Ok(())
}

/// Private static seam permits model-free driver regression tests, not public
/// fake-native execution receipts. The public method accepts ONLY the real
/// StrictInt8Engine. All checkpoints use the one caller-owned control object.
trait Driver {
    type Control: DecodeStepControl;
    fn control(&mut self) -> &mut Self::Control;
    fn append(&mut self, token: u32) -> Result<(), Int8GenerationError>;
    fn logits(&mut self) -> Result<Vec<f32>, Int8GenerationError>;
    fn work(&self) -> Int8Work;
    fn abort(&mut self);
}
impl<C: DecodeStepControl> Driver for Int8Session<'_, '_, C> {
    type Control = C;
    fn control(&mut self) -> &mut C { Int8Session::control(self) }
    fn append(&mut self, token: u32) -> Result<(), Int8GenerationError> { Int8Session::append(self, token).map_err(Int8GenerationError::from) }
    fn logits(&mut self) -> Result<Vec<f32>, Int8GenerationError> { Int8Session::logits(self, LinearRows::All).map_err(Int8GenerationError::from) }
    fn work(&self) -> Int8Work { Int8Session::work(self) }
    fn abort(&mut self) { Int8Session::abort(self); }
}

#[cfg(test)] mod tests;
