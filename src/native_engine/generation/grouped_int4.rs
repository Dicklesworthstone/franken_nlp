//! Source-candidate INT4 generation using the existing compiler and cursor.
//! Public execution accepts only the real full-model engine. Nothing here
//! activates a canonical artifact, adds a runtime, or qualifies model quality.

use super::*;
use crate::{artifact::converter::PINNED_REVISION, native_engine::{
    portable_int8::{LinearRows, int4::convert::GROUPED_INT4_RECIPE},
    quant_algebra::MAX_MODEL_K,
    rope::DEFAULT_ADMITTED_CONTEXT_CAP,
    strict_int4::{Int4ModelIdentity, Int4RunBudget, Int4Session, Int4Work,
        StrictInt4Engine, StrictInt4Error, STRICT_INT4_EXECUTION, STRICT_INT4_PROFILE},
}};

pub const INT4_GENERATION_VERSION: &str = "grouped-int4-addressed-generation-final-prefill-head-v1";
/// An in-memory candidate namespace, NOT fnlpq or release activation authority.
pub const INT4_SOURCE_FORMAT: &str = "fnlp-int4-source-candidate-v1";

#[derive(Clone, Copy, Debug)]
pub struct Int4GenerationBudget {
    pub native: Int4RunBudget,
    /// Price complete allocated KV capacity, not just the live prefix.
    pub max_kv_bytes: u64,
    pub max_sampler_bytes: u64,
}
#[derive(Clone, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int4GenerationRun {
    pub schema_version: u32,
    pub sequence: GeneratedSequence,
    pub model_work: Int4Work,
}
#[derive(Debug)]
pub enum Int4GenerationError {
    Generation(GenerationError), Native(StrictInt4Error), ModelIdentity, WorkMismatch,
}
impl fmt::Display for Int4GenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Generation(error) => write!(f, "INT4 generation refused: {error}"),
            Self::Native(error) => write!(f, "INT4 generation native execution refused: {error}"),
            Self::ModelIdentity => f.write_str("INT4 generation model, partition or candidate identity mismatch"),
            Self::WorkMismatch => f.write_str("INT4 generation completed work disagrees with the native session"),
        }
    }
}
impl Error for Int4GenerationError {}
impl From<GenerationError> for Int4GenerationError {
    fn from(error: GenerationError) -> Self { Self::Generation(error) }
}
impl From<StrictInt4Error> for Int4GenerationError {
    fn from(error: StrictInt4Error) -> Self { Self::Native(error) }
}
impl Int4GenerationError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Generation(GenerationError::Cancelled(cause))
                | Self::Native(StrictInt4Error::Cancelled(cause)) => Some(*cause),
            _ => None,
        }
    }
}

/// Explicitly bind model-owned fields of a host's candidate identity before
/// compilation/admission. The host MUST already choose the INT4 profile and
/// backend; an INT8/BF16 identity is refused, not silently repaired. Tokenizer,
/// template, task, policy and other host facts are preserved and later bound
/// in full. The host still owns actual tokenizer/decoder authenticity.
///
/// `logical_model_digest` carries this in-memory model's content fingerprint
/// under INT4_SOURCE_FORMAT, never a canonical fnlpq digest. The packing key
/// binds its exact nibble/scale layout and group width, not a measured ISA.
pub fn bind_candidate_model(mut identity: ExecutionIdentity, model: &Int4ModelIdentity)
    -> Result<ExecutionIdentity, Int4GenerationError> {
    check_opt_in(&identity)?;
    identity.source_revision = model.revision().to_owned();
    identity.artifact_format = INT4_SOURCE_FORMAT.to_owned();
    identity.quant_recipe = recipe(model.group_width());
    identity.logical_model_digest = content_digest(model)?;
    identity.packing_set_digest = packing_digest(identity.logical_model_digest, model.group_width())?;
    identity.validate().map_err(|_| GenerationError::Identity)?;
    Ok(identity)
}

/// Private immutable compiled request. It cannot be coerced into an eager or
/// INT8 plan. The group partition enters both the identity and work bound.
pub struct Int4GenerationPlan { plan: GenerationPlan, group_width: usize, work: Int4Work }
impl Int4GenerationPlan {
    pub fn compile(prompt: Vec<u32>, options: GenerationOptions, identity: ExecutionIdentity,
        group_width: usize, item_id: &str, sample_index: u64, limits: GenerationLimits)
        -> Result<Self, Int4GenerationError> {
        check_contract(&identity, group_width)?;
        let plan = GenerationPlan::compile_for_backend(prompt, options, identity, item_id, sample_index,
            limits, GenerationBackend::GroupedInt4)?;
        let positions = usize::try_from(plan.bound.forward_positions).map_err(|_| StrictInt4Error::Work)?;
        let rows = usize::try_from(plan.bound.projected_logits).map_err(|_| StrictInt4Error::Work)?;
        if positions > DEFAULT_ADMITTED_CONTEXT_CAP { return Err(StrictInt4Error::Context.into()); }
        let work = Int4Work::for_sequence(group_width, 0, positions, rows)?;
        Ok(Self { plan, group_width, work })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.plan.execution_identity() }
    pub fn options(&self) -> &GenerationOptions { self.plan.options() }
    pub fn prompt_tokens(&self) -> usize { self.plan.prompt_tokens() }
    pub fn sampler_bytes(&self) -> u64 { self.plan.sampler_bytes() }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn planned_work(&self) -> Int4Work { self.work }
    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), Int4GenerationError> {
        self.plan.verify_identity(admitted).map_err(Int4GenerationError::from)
    }
    pub fn preflight(&self, admitted: &ExecutionIdentity, engine: &StrictInt4Engine<'_>, budget: Int4GenerationBudget)
        -> Result<(), Int4GenerationError> {
        self.verify_identity(admitted)?;
        check_model(admitted, self.group_width, engine.model_identity())?;
        if engine.profile() != STRICT_INT4_PROFILE || engine.is_poisoned() {
            return Err(StrictInt4Error::EngineUnavailable.into());
        }
        if !engine.kv_cache().all_slots_have_len(0) { return Err(GenerationError::EngineAlreadyPrimed.into()); }
        check_bounds(self.work, self.sampler_bytes(), engine.kv_cache().capacity_positions(), budget)
    }
    pub fn execute<D: DecodeByteDecoder, C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt4Engine<'_>, decoder: &D, request_seq: u64, budget: Int4GenerationBudget, control: &mut C)
        -> Result<Int4GenerationRun, Int4GenerationError> {
        self.execute_with_sink(admitted, engine, decoder, request_seq, budget, &mut Discard, control)
    }
    /// One exclusively borrowed native session. Native/decoder/sink failure
    /// aborts the session; unwind is covered by its existing RAII cleanup.
    /// Earlier stream events cannot be retracted and are not a success receipt.
    pub fn execute_with_sink<D: DecodeByteDecoder, S: DecodeEventSink, C: DecodeStepControl>(&self,
        admitted: &ExecutionIdentity, engine: &mut StrictInt4Engine<'_>, decoder: &D, request_seq: u64,
        budget: Int4GenerationBudget, sink: &mut S, control: &mut C) -> Result<Int4GenerationRun, Int4GenerationError> {
        self.preflight(admitted, engine, budget)?;
        let mut session = engine.session(budget.native, control)?;
        self.drive(decoder, request_seq, sink, &mut session)
    }
    fn drive<D: DecodeByteDecoder, S: DecodeEventSink, B: Driver>(&self,
        decoder: &D, request_seq: u64, sink: &mut S, driver: &mut B) -> Result<Int4GenerationRun, Int4GenerationError> {
        let result = (|| {
            let mut row = cursor::Cursor::new(&self.plan, request_seq, INT4_GENERATION_VERSION)?;
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
            check_completed(self.group_width, sequence.native_work, model_work)?;
            Ok(Int4GenerationRun { schema_version: 1, sequence, model_work })
        })();
        if result.is_err() { driver.abort(); }
        result
    }
}
fn check_opt_in(identity: &ExecutionIdentity) -> Result<(), Int4GenerationError> {
    if identity.numerics_profile != (NumericsProfile::StrictQuantized { version: 2 })
        || identity.backend_semantic_version != STRICT_INT4_EXECUTION || identity.kv_dtype != "bf16" {
        return Err(Int4GenerationError::ModelIdentity);
    }
    Ok(())
}
fn recipe(width: usize) -> String { format!("{GROUPED_INT4_RECIPE}:group={width}") }
fn packing_digest(content: Sha256Digest, width: usize) -> Result<Sha256Digest, GenerationError> {
    digest(&(INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", content, width))
}
fn content_digest(model: &Int4ModelIdentity) -> Result<Sha256Digest, Int4GenerationError> {
    let mut hex = String::with_capacity(64);
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    for byte in model.content_sha256() {
        hex.push(char::from(DIGITS[usize::from(byte >> 4)]));
        hex.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    Sha256Digest::from_hex(&hex).map_err(|_| Int4GenerationError::ModelIdentity)
}
fn check_contract(identity: &ExecutionIdentity, width: usize) -> Result<(), Int4GenerationError> {
    check_opt_in(identity)?;
    if width == 0 || width > MAX_MODEL_K || identity.source_revision != PINNED_REVISION
        || identity.artifact_format != INT4_SOURCE_FORMAT || identity.quant_recipe != recipe(width)
        || identity.packing_set_digest != packing_digest(identity.logical_model_digest, width)? {
        return Err(Int4GenerationError::ModelIdentity);
    }
    Ok(())
}
fn check_model(identity: &ExecutionIdentity, width: usize, model: &Int4ModelIdentity) -> Result<(), Int4GenerationError> {
    check_model_facts(identity, width, model.group_width(), model.profile(), content_digest(model)?)
}
fn check_model_facts(identity: &ExecutionIdentity, width: usize, actual_width: usize,
    actual_profile: &str, actual_content: Sha256Digest) -> Result<(), Int4GenerationError> {
    check_contract(identity, width)?;
    if width != actual_width || actual_profile != STRICT_INT4_PROFILE
        || identity.logical_model_digest != actual_content {
        return Err(Int4GenerationError::ModelIdentity);
    }
    Ok(())
}
fn check_bounds(work: Int4Work, sampler_bytes: u64, capacity: usize, budget: Int4GenerationBudget)
    -> Result<(), Int4GenerationError> {
    let capacity = u64::try_from(capacity).map_err(|_| StrictInt4Error::Context)?;
    if work.forward_positions > capacity { return Err(StrictInt4Error::Context.into()); }
    let kv = capacity.checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or(StrictInt4Error::Memory)?;
    if kv > budget.max_kv_bytes || sampler_bytes > budget.max_sampler_bytes { return Err(StrictInt4Error::Memory.into()); }
    if work.forward_positions > budget.native.max_forward_positions || work.attention_pairs > budget.native.max_attention_pairs
        || !work.projections.fits(budget.native.max_projection_work) { return Err(StrictInt4Error::Work.into()); }
    Ok(())
}
fn check_completed(width: usize, cursor: GenerationWork, native: Int4Work) -> Result<(), Int4GenerationError> {
    let positions = usize::try_from(cursor.forward_positions).map_err(|_| Int4GenerationError::WorkMismatch)?;
    let rows = usize::try_from(cursor.projected_logits).map_err(|_| Int4GenerationError::WorkMismatch)?;
    if native != Int4Work::for_sequence(width, 0, positions, rows)? { return Err(Int4GenerationError::WorkMismatch); }
    Ok(())
}

// Static test seam only. A caller cannot inject a fake model into public
// execution or mint a native result without borrowing a StrictInt4Engine.
trait Driver {
    type Control: DecodeStepControl;
    fn control(&mut self) -> &mut Self::Control;
    fn append(&mut self, token: u32) -> Result<(), Int4GenerationError>;
    fn logits(&mut self) -> Result<Vec<f32>, Int4GenerationError>;
    fn work(&self) -> Int4Work;
    fn abort(&mut self);
}
impl<C: DecodeStepControl> Driver for Int4Session<'_, '_, C> {
    type Control = C;
    fn control(&mut self) -> &mut C { Int4Session::control(self) }
    fn append(&mut self, token: u32) -> Result<(), Int4GenerationError> { Int4Session::append(self, token).map_err(Int4GenerationError::from) }
    fn logits(&mut self) -> Result<Vec<f32>, Int4GenerationError> { Int4Session::logits(self, LinearRows::All).map_err(Int4GenerationError::from) }
    fn work(&self) -> Int4Work { Int4Session::work(self) }
    fn abort(&mut self) { Int4Session::abort(self); }
}

#[cfg(test)] mod tests;
