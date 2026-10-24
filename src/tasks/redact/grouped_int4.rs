//! Native grouped-INT4 NER plus rules and transactional redaction.
//! Optional verification freshly plans from transformed text with the same
//! detector scope and key. One nonrenewable ledger spans both native passes.
use std::{collections::BTreeSet, error::Error, fmt};
use serde::Serialize;
use crate::{
    canonjson, execution_identity::{ExecutionIdentity, Sha256Digest},
    grammar::mask::MaskWorkLimits,
    native_engine::{constrained::JsonWorkBudget, constrained_int4::{self, Int4JsonBudget},
        decode::{DecodeCancellationKind, DecodeStepControl}, kv::KV_BYTES_PER_TOKEN,
        strict_int4::{Int4ModelIdentity, Int4RunBudget, Int4Work, StrictInt4Engine,
            StrictInt4Error, STRICT_INT4_PROFILE}},
    tasks::{extract::{ExtractionVocabulary, quantized::check_size}, ir::{PlanContext, TaskBudget},
        ner::{EntityType, NerOptions, NerResult, NER_TASK_VERSION},
        source_planning::{SourcePlanningLimits, SourceTaskPlanner, SourceTaskRequest, SourceTaskResult,
            grouped_int4::{Int4SourceError, Int4SourceTaskRun, PreparedInt4SourceTask, INT4_SOURCE_EXECUTION}}},
};
use super::{RedactError, actions::RedactionResult,
    pipeline::{self, LeakReport, NerPass, PipelineError, RedactionRequest},
    pseudonym::Pseudonyms, union::NerProfile};

pub const INT4_REDACTION_EXECUTION: &str = "grouped-int4-ner-rules-redetect-v1";

/// Complete-document rules plus native NER chunks and fresh verification.
pub mod long;

/// Per-pass preparation limits and nonrenewable complete-operation ceilings.
/// Rules, source verification and edits keep the request's independent bounds.
#[derive(Clone, Debug)]
pub struct Int4RedactionConfig {
    pub ner: NerOptions,
    pub per_pass: TaskBudget,
    pub planning: SourcePlanningLimits,
    pub max_model_work: Int4Work,
    pub mask_limits: MaskWorkLimits,
    pub mask_visits_per_pass: u64,
    pub max_mask_visits: u64,
    pub max_result_bytes: u64,
}
impl Int4RedactionConfig {
    fn validate(&self) -> Result<(), Int4RedactionError> {
        self.ner.validate().map_err(|_| RedactError::InvalidOptions)?;
        self.per_pass.validate().map_err(|_| RedactError::InvalidOptions)?;
        if self.mask_visits_per_pass == 0 || self.max_mask_visits < self.mask_visits_per_pass
            || self.mask_limits.max_trie_node_visits == 0 || self.mask_limits.checkpoint_interval_nodes == 0
            || !(1..=64 * 1024 * 1024).contains(&self.max_result_bytes) {
            return Err(RedactError::InvalidOptions.into());
        }
        Ok(())
    }
}

pub enum Int4RedactionError {
    Redaction(RedactError), Source(Int4SourceError), Native(StrictInt4Error),
    Residual(LeakReport), Identity, WorkBudget, InvalidResult, Cancelled(DecodeCancellationKind),
}
impl fmt::Display for Int4RedactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Redaction(_) => "int4 redaction detector or editing failed",
            Self::Source(_) => "int4 redaction source task failed",
            Self::Native(_) => "int4 redaction native engine failed",
            Self::Residual(_) => "int4 redaction found residual declared detections",
            Self::Identity => "int4 redaction actual model, group or pinned planner identity differs",
            Self::WorkBudget => "int4 redaction whole-operation work exceeded",
            Self::InvalidResult => "int4 redaction native result contract diverged",
            Self::Cancelled(_) => "int4 redaction cancelled",
        })
    }
}
impl fmt::Debug for Int4RedactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int4RedactionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Redaction(e) => Some(e), Self::Source(e) => Some(e), Self::Native(e) => Some(e), _ => None }
    }
}
impl From<RedactError> for Int4RedactionError { fn from(e: RedactError) -> Self { Self::Redaction(e) } }
impl From<Int4SourceError> for Int4RedactionError { fn from(e: Int4SourceError) -> Self { Self::Source(e) } }
impl From<StrictInt4Error> for Int4RedactionError { fn from(e: StrictInt4Error) -> Self { Self::Native(e) } }
impl From<PipelineError<Int4RedactionError>> for Int4RedactionError {
    fn from(e: PipelineError<Self>) -> Self {
        match e { PipelineError::Redaction(e) => Self::Redaction(e), PipelineError::Model(e) => e,
            PipelineError::Residual(report) => Self::Residual(report) }
    }
}
impl Int4RedactionError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Cancelled(c) | Self::Native(StrictInt4Error::Cancelled(c)) => Some(*c),
            Self::Source(e) => e.cancellation(), _ => None }
    }
}

/// Only final edited text and explicit public policy/coordinate data escape.
/// Original text, native transcripts, private identities and secrets stay owned.
#[derive(Serialize)]
pub struct Int4RedactionRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub group_width: usize,
    pub result: RedactionResult,
    pub ner_passes: usize,
    pub reserved_model_work: Int4Work,
    pub model_work: Int4Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
}

/// Immutable recipe bound to real source/snapshot model facts before use.
/// Each pass separately seals its exact current source; this is no task factory.
pub struct Int4Redactor<'p> {
    planner: &'p SourceTaskPlanner,
    identity: ExecutionIdentity,
    config: Int4RedactionConfig,
    group_width: usize,
    types: BTreeSet<EntityType>,
}
impl<'p> Int4Redactor<'p> {
    pub fn new(planner: &'p SourceTaskPlanner, ner_identity: ExecutionIdentity,
        config: Int4RedactionConfig, model: &Int4ModelIdentity) -> Result<Self, Int4RedactionError> {
        check_model(&ner_identity, model)?;
        Self::from_bound_parts(planner, ner_identity, config, model.group_width())
    }
    // Private fixtures may construct geometry without resident model weights.
    // Public construction above always checks the sealed actual model identity.
    fn from_bound_parts(planner: &'p SourceTaskPlanner, ner_identity: ExecutionIdentity,
        config: Int4RedactionConfig, group_width: usize) -> Result<Self, Int4RedactionError> {
        config.validate()?;
        constrained_int4::check_profile(&ner_identity, group_width).map_err(|_| Int4RedactionError::Identity)?;
        if ner_identity.task_spec != NER_TASK_VERSION || ner_identity.tokenizer_digest != planner.tokenizer_digest()
            || ner_identity.template_digest != *planner.template_digest() {
            return Err(Int4RedactionError::Identity);
        }
        let types = config.ner.types.iter().copied().collect();
        Ok(Self { planner, identity: ner_identity, config, group_width, types })
    }
    pub fn group_width(&self) -> usize { self.group_width }

    /// One actual model, engine, vocabulary and detector recipe across original
    /// and transformed sources. The caller retains preparation/output admission.
    #[allow(clippy::too_many_arguments)]
    pub fn redact<C: DecodeStepControl>(&self, source: &str, request: &RedactionRequest,
        pseudonyms: Option<&Pseudonyms<'_>>, engine: &mut StrictInt4Engine<'_>,
        vocabulary: &ExtractionVocabulary, control: &mut C) -> Result<Int4RedactionRun, Int4RedactionError> {
        checkpoint(control)?;
        check_engine(&self.identity, self.group_width, engine, self.config.per_pass.max_kv_bytes)?;
        let mut pass = Pass { recipe: self, engine, vocabulary, control, ledger: Ledger::new(&self.config, request.verify)? };
        self.run_pipeline(source, request, pseudonyms, &mut pass)
    }
    fn prepare<C: DecodeStepControl>(&self, source: &str, control: &mut C)
        -> Result<PreparedInt4SourceTask, Int4RedactionError> {
        checkpoint(control)?;
        if source.len() > self.config.planning.max_input_bytes { return Err(RedactError::InputBudget.into()); }
        let mut document = String::new();
        document.try_reserve_exact(source.len()).map_err(|_| RedactError::AllocationRefused)?;
        document.push_str(source);
        let request = SourceTaskRequest::Ner { document, options: self.config.ner.clone(), budget: self.config.per_pass };
        let context = PlanContext::new(&self.identity, self.config.per_pass).map_err(|_| Int4RedactionError::Identity)?;
        Ok(self.planner.plan_int4_with_control(&request, &context, self.config.planning, self.group_width, control)?)
    }
    // Fixtures replace only private native detection. The shared source union,
    // edits, key checks and fresh verification still execute unchanged.
    fn run_pipeline<P: AccountedPass>(&self, source: &str, request: &RedactionRequest,
        pseudonyms: Option<&Pseudonyms<'_>>, pass: &mut P) -> Result<Int4RedactionRun, Int4RedactionError> {
        if request.edit_budget.max_output_bytes as u64 > self.config.max_result_bytes {
            return Err(RedactError::OutputBudget.into());
        }
        pass.checkpoint()?;
        let result = pipeline::redact_with_profile(source, request, pseudonyms, pass, NerProfile::Int4);
        pass.checkpoint()?;
        let mut result = result.map_err(Int4RedactionError::from)?;
        let ledger = pass.ledger();
        if ledger.failed || ledger.completed != 1 + usize::from(request.verify) {
            return Err(Int4RedactionError::InvalidResult);
        }
        result.policy_digest = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT4_REDACTION_EXECUTION, result.policy_digest, &self.config.ner, self.config.per_pass, self.group_width,
        )).map_err(|_| RedactError::Serialization)?);
        let output = Int4RedactionRun { schema_version: 1, execution: INT4_REDACTION_EXECUTION,
            numerics_profile: STRICT_INT4_PROFILE, group_width: self.group_width, result, ner_passes: ledger.completed,
            reserved_model_work: ledger.reserved, model_work: ledger.actual,
            reserved_mask_node_visits: ledger.masks_reserved, mask_node_visit_charge: ledger.masks_actual };
        check_size(&output, self.config.max_result_bytes).map_err(Int4SourceError::from)?;
        pass.checkpoint()?;
        Ok(output)
    }
}

struct Pass<'a, 'p, 'w, C> {
    recipe: &'a Int4Redactor<'p>, engine: &'a mut StrictInt4Engine<'w>,
    vocabulary: &'a ExtractionVocabulary, control: &'a mut C, ledger: Ledger,
}
impl<C: DecodeStepControl> NerPass for Pass<'_, '_, '_, C> {
    type Error = Int4RedactionError;
    fn types(&self) -> &BTreeSet<EntityType> { &self.recipe.types }
    fn run(&mut self, source: &str) -> Result<NerResult, Self::Error> {
        if self.ledger.failed { return Err(Int4RedactionError::InvalidResult); }
        // Planning, native failure or unwind is terminal; neither pass retries.
        self.ledger.failed = true;
        let plan = self.recipe.prepare(source, self.control)?;
        let work = plan.planned_work();
        self.ledger.reserve(work)?;
        let config = &self.recipe.config;
        let run = plan.execute_with_control(plan.execution_identity(), self.engine, self.vocabulary,
            Int4JsonBudget { native: Int4RunBudget::exact(work), json: JsonWorkBudget {
                max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
                max_kv_bytes: config.per_pass.max_kv_bytes, max_total_mask_node_visits: config.mask_visits_per_pass,
                mask_limits: config.mask_limits,
            } }, self.control)?;
        if self.engine.is_poisoned() || !self.engine.kv_cache().all_slots_have_len(0) {
            return Err(Int4RedactionError::InvalidResult);
        }
        let (result, actual) = check_run(&plan, run, config.mask_visits_per_pass)?;
        self.ledger.finish(actual, result.mask_node_visit_charge)?;
        checkpoint(self.control)?;
        Ok(result)
    }
}
trait AccountedPass: NerPass<Error = Int4RedactionError> {
    fn checkpoint(&mut self) -> Result<(), Int4RedactionError>;
    fn ledger(&self) -> &Ledger;
}
impl<C: DecodeStepControl> AccountedPass for Pass<'_, '_, '_, C> {
    fn checkpoint(&mut self) -> Result<(), Int4RedactionError> { checkpoint(self.control) }
    fn ledger(&self) -> &Ledger { &self.ledger }
}

struct Ledger {
    limit: Int4Work, mask_cap: u64, mask_per_pass: u64, passes: usize,
    reserved: Int4Work, actual: Int4Work, current: Int4Work, masks_reserved: u64, masks_actual: u64,
    started: usize, completed: usize, failed: bool,
}
impl Ledger {
    fn new(config: &Int4RedactionConfig, verify: bool) -> Result<Self, Int4RedactionError> {
        let passes = 1 + usize::from(verify);
        if config.mask_visits_per_pass.checked_mul(passes as u64).is_none_or(|n| n > config.max_mask_visits) {
            return Err(Int4RedactionError::WorkBudget);
        }
        Ok(Self { limit: config.max_model_work, mask_cap: config.max_mask_visits,
            mask_per_pass: config.mask_visits_per_pass, passes, reserved: Int4Work::default(), actual: Int4Work::default(),
            current: Int4Work::default(), masks_reserved: 0, masks_actual: 0, started: 0, completed: 0, failed: false })
    }
    fn reserve(&mut self, work: Int4Work) -> Result<(), Int4RedactionError> {
        self.failed = true;
        if self.started >= self.passes || self.started != self.completed { return Err(Int4RedactionError::InvalidResult); }
        let sum = self.reserved.checked_add(work).map_err(|_| Int4RedactionError::WorkBudget)?;
        let masks = self.masks_reserved.checked_add(self.mask_per_pass)
            .filter(|&n| n <= self.mask_cap).ok_or(Int4RedactionError::WorkBudget)?;
        if !within(sum, self.limit) { return Err(Int4RedactionError::WorkBudget); }
        self.reserved = sum; self.current = work; self.masks_reserved = masks; self.started += 1;
        Ok(())
    }
    fn finish(&mut self, actual: Int4Work, masks: u64) -> Result<(), Int4RedactionError> {
        if self.started != self.completed + 1 || masks > self.mask_per_pass || !within(actual, self.current) {
            return Err(Int4RedactionError::InvalidResult);
        }
        let sum = self.actual.checked_add(actual)?;
        let masks = self.masks_actual.checked_add(masks).ok_or(Int4RedactionError::InvalidResult)?;
        if !within(sum, self.reserved) || masks > self.masks_reserved { return Err(Int4RedactionError::InvalidResult); }
        self.actual = sum; self.masks_actual = masks; self.completed += 1; self.failed = false;
        Ok(())
    }
}
fn check_run(plan: &PreparedInt4SourceTask, run: Int4SourceTaskRun, masks: u64)
    -> Result<(NerResult, Int4Work), Int4RedactionError> {
    if run.schema_version != 1 || run.execution != INT4_SOURCE_EXECUTION || run.group_width != plan.group_width() {
        return Err(Int4RedactionError::InvalidResult);
    }
    let SourceTaskResult::Ner(result) = run.result else { return Err(Int4RedactionError::InvalidResult); };
    let expected = constrained_int4::planned_work(plan.group_width(), plan.prompt_tokens(), result.generated_token_ids.len())
        .map_err(|_| Int4RedactionError::InvalidResult)?;
    if result.schema_version != 1 || result.numerics_profile != STRICT_INT4_PROFILE
        || result.task_spec_version != NER_TASK_VERSION || result.generated_token_ids.is_empty()
        || result.generated_token_ids.len() > plan.task_budget().max_output_tokens as usize
        || run.model_work != expected || result.forward_positions != expected.forward_positions
        || result.projected_logits != expected.projected_logits || !within(expected, plan.planned_work())
        || result.mask_node_visit_charge > masks { return Err(Int4RedactionError::InvalidResult); }
    Ok((result, run.model_work))
}
pub(crate) fn check_model(identity: &ExecutionIdentity, model: &Int4ModelIdentity) -> Result<(), Int4RedactionError> {
    if identity.task_spec != NER_TASK_VERSION { return Err(Int4RedactionError::Identity); }
    constrained_int4::check_model(identity, model).map_err(|_| Int4RedactionError::Identity)
}
fn check_engine(identity: &ExecutionIdentity, group_width: usize, engine: &StrictInt4Engine<'_>,
    kv_cap: u64) -> Result<(), Int4RedactionError> {
    check_model(identity, engine.model_identity())?;
    if engine.model_identity().group_width() != group_width { return Err(Int4RedactionError::Identity); }
    if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) { return Err(Int4RedactionError::InvalidResult); }
    check_capacity(engine.kv_cache().capacity_positions(), kv_cap)
}
fn check_capacity(context_tokens: usize, kv_cap: u64) -> Result<(), Int4RedactionError> {
    if (context_tokens as u64).checked_mul(KV_BYTES_PER_TOKEN as u64).is_none_or(|n| n > kv_cap) {
        return Err(Int4RedactionError::WorkBudget);
    }
    Ok(())
}
fn within(a: Int4Work, b: Int4Work) -> bool {
    a.forward_positions <= b.forward_positions && a.projected_logits <= b.projected_logits
        && a.attention_pairs <= b.attention_pairs && a.projections.fits(b.projections)
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), Int4RedactionError> {
    match control.prefill_checkpoint(0) { Some(c) => Err(Int4RedactionError::Cancelled(c)), None => Ok(()) }
}

#[cfg(test)] mod tests;
