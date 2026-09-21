//! Native INT8 NER plus rules, transactional editing, and optional fresh
//! residual verification. The same planner, model and detector scope serve
//! both passes; the second pass never receives the original source/offsets.

use std::{collections::BTreeSet, error::Error, fmt};
use serde::Serialize;
use crate::{
    canonjson, execution_identity::{ExecutionIdentity, Sha256Digest},
    grammar::mask::MaskWorkLimits,
    native_engine::{constrained::JsonWorkBudget, constrained_int8::{self, Int8JsonBudget},
        decode::{DecodeCancellationKind, DecodeStepControl}, kv::KV_BYTES_PER_TOKEN,
        strict_int8::{Int8RunBudget, Int8Work, StrictInt8Engine, StrictInt8Error, STRICT_INT8_PROFILE}},
    tasks::{extract::{ExtractionVocabulary, quantized::check_size}, ir::{PlanContext, TaskBudget},
        ner::{EntityType, NerOptions, NerResult, NER_TASK_VERSION},
        source_planning::{SourcePlanningLimits, SourceTaskPlanner, SourceTaskRequest, SourceTaskResult,
            quantized::{Int8SourceError, Int8SourceTaskRun, PreparedInt8SourceTask, INT8_SOURCE_EXECUTION}}},
};
use super::{RedactError, actions::RedactionResult, pipeline::{self, LeakReport, NerPass, PipelineError, RedactionRequest},
    pseudonym::Pseudonyms, union::NerProfile};

pub const INT8_REDACTION_EXECUTION: &str = "portable-int8-ner-rules-redetect-v1";

/// Per-pass task limits and nonrenewable whole-operation native/mask ceilings.
/// Rule scans/occurrence recovery/edits retain RedactionRequest's own bounds.
/// No default silently grants a corpus-sized neural allowance.
#[derive(Clone, Debug)]
pub struct Int8RedactionConfig {
    pub ner: NerOptions,
    pub per_pass: TaskBudget,
    pub planning: SourcePlanningLimits,
    pub max_model_work: Int8Work,
    pub mask_limits: MaskWorkLimits,
    pub mask_visits_per_pass: u64,
    pub max_mask_visits: u64,
    pub max_result_bytes: u64,
}
impl Int8RedactionConfig {
    fn validate(&self) -> Result<(), Int8RedactionError> {
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

/// Private source-bearing errors stay typed; default diagnostics never print
/// original/redacted text, native JSON, prompt digests or pseudonym secrets.
pub enum Int8RedactionError {
    Redaction(RedactError), Source(Int8SourceError), Native(StrictInt8Error),
    Residual(LeakReport), Identity, WorkBudget, InvalidResult, Cancelled(DecodeCancellationKind),
}
impl fmt::Display for Int8RedactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Redaction(_) => "int8 redaction detector or editing failed",
            Self::Source(_) => "int8 redaction source task failed",
            Self::Native(_) => "int8 redaction native engine failed",
            Self::Residual(_) => "int8 redaction found residual declared detections",
            Self::Identity => "int8 redaction model or pinned planner identity differs",
            Self::WorkBudget => "int8 redaction whole-operation work exceeded",
            Self::InvalidResult => "int8 redaction native result contract diverged",
            Self::Cancelled(_) => "int8 redaction cancelled",
        })
    }
}
impl fmt::Debug for Int8RedactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int8RedactionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Redaction(e) => Some(e), Self::Source(e) => Some(e), Self::Native(e) => Some(e), _ => None }
    }
}
impl From<RedactError> for Int8RedactionError { fn from(e: RedactError) -> Self { Self::Redaction(e) } }
impl From<Int8SourceError> for Int8RedactionError { fn from(e: Int8SourceError) -> Self { Self::Source(e) } }
impl From<StrictInt8Error> for Int8RedactionError { fn from(e: StrictInt8Error) -> Self { Self::Native(e) } }
impl From<PipelineError<Int8RedactionError>> for Int8RedactionError {
    fn from(e: PipelineError<Self>) -> Self {
        match e { PipelineError::Redaction(e) => Self::Redaction(e), PipelineError::Model(e) => e,
            PipelineError::Residual(report) => Self::Residual(report) }
    }
}
impl Int8RedactionError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Cancelled(c) | Self::Native(StrictInt8Error::Cancelled(c)) => Some(*c),
            Self::Source(e) => e.cancellation(), _ => None }
    }
}

/// The only exported text is the completed redaction result. NER transcripts,
/// private identities and rejected intermediate documents are not serialized.
#[derive(Serialize)]
pub struct Int8RedactionRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub result: RedactionResult,
    pub ner_passes: usize,
    pub reserved_model_work: Int8Work,
    pub model_work: Int8Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
}

/// Immutable detector recipe. The identity explicitly describes NER planning,
/// not a precompiled redaction prompt. Each pass derives its exact prompt from
/// the supplied text under this fixed context. No mutable task factory exists.
pub struct Int8Redactor<'p> {
    planner: &'p SourceTaskPlanner,
    identity: ExecutionIdentity,
    config: Int8RedactionConfig,
    types: BTreeSet<EntityType>,
}
impl<'p> Int8Redactor<'p> {
    pub fn new(planner: &'p SourceTaskPlanner, ner_identity: ExecutionIdentity, config: Int8RedactionConfig)
        -> Result<Self, Int8RedactionError> {
        config.validate()?;
        ner_identity.validate().map_err(|_| Int8RedactionError::Identity)?;
        constrained_int8::check_profile(&ner_identity).map_err(|_| Int8RedactionError::Identity)?;
        if ner_identity.task_spec != NER_TASK_VERSION || ner_identity.tokenizer_digest != planner.tokenizer_digest()
            || ner_identity.template_digest != *planner.template_digest() {
            return Err(Int8RedactionError::Identity);
        }
        let types = config.ner.types.iter().copied().collect();
        Ok(Self { planner, identity: ner_identity, config, types })
    }

    /// One real engine and vocabulary across the original and verification
    /// pass. Caller owns native/preparation/output admission through delivery.
    /// Synchronous bounded rules/edits are checkpointed around, not preempted.
    #[allow(clippy::too_many_arguments)]
    pub fn redact<C: DecodeStepControl>(&self, source: &str, request: &RedactionRequest,
        pseudonyms: Option<&Pseudonyms<'_>>, engine: &mut StrictInt8Engine<'_>,
        vocabulary: &ExtractionVocabulary, control: &mut C) -> Result<Int8RedactionRun, Int8RedactionError> {
        checkpoint(control)?;
        check_engine(&self.identity, engine, self.config.per_pass.max_kv_bytes)?;
        let mut pass = Pass { recipe: self, engine, vocabulary, control, ledger: Ledger::new(&self.config, request.verify)? };
        self.run_pipeline(source, request, pseudonyms, &mut pass)
    }

    fn prepare<C: DecodeStepControl>(&self, source: &str, control: &mut C)
        -> Result<PreparedInt8SourceTask, Int8RedactionError> {
        checkpoint(control)?;
        if source.len() > self.config.planning.max_input_bytes { return Err(RedactError::InputBudget.into()); }
        let mut document = String::new();
        document.try_reserve_exact(source.len()).map_err(|_| RedactError::AllocationRefused)?;
        document.push_str(source);
        let request = SourceTaskRequest::Ner { document, options: self.config.ner.clone(), budget: self.config.per_pass };
        let context = PlanContext::new(&self.identity, self.config.per_pass).map_err(|_| Int8RedactionError::Identity)?;
        Ok(self.planner.plan_int8_with_control(&request, &context, self.config.planning, control)?)
    }

    // Private seam lets tests exercise the real union/edit/verification path.
    // Public construction/execution cannot substitute a synthetic NER driver.
    fn run_pipeline<P: AccountedPass>(&self, source: &str, request: &RedactionRequest,
        pseudonyms: Option<&Pseudonyms<'_>>, pass: &mut P) -> Result<Int8RedactionRun, Int8RedactionError> {
        if request.edit_budget.max_output_bytes as u64 > self.config.max_result_bytes {
            return Err(RedactError::OutputBudget.into());
        }
        pass.checkpoint()?;
        let result = pipeline::redact_with_profile(source, request, pseudonyms, pass, NerProfile::Int8);
        pass.checkpoint()?;
        let mut result = result.map_err(Int8RedactionError::from)?;
        let ledger = pass.ledger();
        if ledger.failed || ledger.completed != 1 + usize::from(request.verify) {
            return Err(Int8RedactionError::InvalidResult);
        }
        // Only public detector policy is bound for export, never prompt bytes
        // or private prompt/source digests. Old eager policy identity is distinct.
        result.policy_digest = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT8_REDACTION_EXECUTION, result.policy_digest, &self.config.ner, self.config.per_pass,
        )).map_err(|_| RedactError::Serialization)?);
        let output = Int8RedactionRun { schema_version: 1, execution: INT8_REDACTION_EXECUTION,
            numerics_profile: STRICT_INT8_PROFILE, result, ner_passes: ledger.completed,
            reserved_model_work: ledger.reserved, model_work: ledger.actual,
            reserved_mask_node_visits: ledger.masks_reserved, mask_node_visit_charge: ledger.masks_actual };
        check_size(&output, self.config.max_result_bytes).map_err(Int8SourceError::from)?;
        pass.checkpoint()?;
        Ok(output)
    }
}

struct Pass<'a, 'p, 'w, C> {
    recipe: &'a Int8Redactor<'p>, engine: &'a mut StrictInt8Engine<'w>,
    vocabulary: &'a ExtractionVocabulary, control: &'a mut C, ledger: Ledger,
}
impl<C: DecodeStepControl> NerPass for Pass<'_, '_, '_, C> {
    type Error = Int8RedactionError;
    fn types(&self) -> &BTreeSet<EntityType> { &self.recipe.types }
    fn run(&mut self, source: &str) -> Result<NerResult, Self::Error> {
        if self.ledger.failed { return Err(Int8RedactionError::InvalidResult); }
        // Compilation failure/unwind is terminal too; neither pass is retried.
        self.ledger.failed = true;
        let plan = self.recipe.prepare(source, self.control)?;
        let work = plan.planned_work();
        self.ledger.reserve(work)?;
        let config = &self.recipe.config;
        let run = plan.execute_with_control(plan.execution_identity(), self.engine, self.vocabulary,
            Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
                max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
                max_kv_bytes: config.per_pass.max_kv_bytes, max_total_mask_node_visits: config.mask_visits_per_pass,
                mask_limits: config.mask_limits,
            } }, self.control)?;
        if self.engine.is_poisoned() || !self.engine.kv_cache().all_slots_have_len(0) {
            return Err(Int8RedactionError::InvalidResult);
        }
        let result = check_run(&plan, run, config.mask_visits_per_pass)?;
        self.ledger.finish(result.1, result.0.mask_node_visit_charge)?;
        checkpoint(self.control)?;
        Ok(result.0)
    }
}
trait AccountedPass: NerPass<Error = Int8RedactionError> {
    fn checkpoint(&mut self) -> Result<(), Int8RedactionError>;
    fn ledger(&self) -> &Ledger;
}
impl<C: DecodeStepControl> AccountedPass for Pass<'_, '_, '_, C> {
    fn checkpoint(&mut self) -> Result<(), Int8RedactionError> { checkpoint(self.control) }
    fn ledger(&self) -> &Ledger { &self.ledger }
}

struct Ledger {
    limit: Int8Work, mask_cap: u64, mask_per_pass: u64, passes: usize,
    reserved: Int8Work, actual: Int8Work, masks_reserved: u64, masks_actual: u64,
    started: usize, completed: usize, failed: bool,
}
impl Ledger {
    fn new(config: &Int8RedactionConfig, verify: bool) -> Result<Self, Int8RedactionError> {
        let passes = 1 + usize::from(verify);
        if config.mask_visits_per_pass.checked_mul(passes as u64).is_none_or(|n| n > config.max_mask_visits) {
            return Err(Int8RedactionError::WorkBudget);
        }
        Ok(Self { limit: config.max_model_work, mask_cap: config.max_mask_visits,
            mask_per_pass: config.mask_visits_per_pass, passes, reserved: Int8Work::default(), actual: Int8Work::default(),
            masks_reserved: 0, masks_actual: 0, started: 0, completed: 0, failed: false })
    }
    fn reserve(&mut self, work: Int8Work) -> Result<(), Int8RedactionError> {
        self.failed = true;
        if self.started >= self.passes || self.started != self.completed { return Err(Int8RedactionError::InvalidResult); }
        let sum = self.reserved.checked_add(work).map_err(|_| Int8RedactionError::WorkBudget)?;
        let masks = self.masks_reserved.checked_add(self.mask_per_pass)
            .filter(|&n| n <= self.mask_cap).ok_or(Int8RedactionError::WorkBudget)?;
        if !within(sum, self.limit) { return Err(Int8RedactionError::WorkBudget); }
        self.reserved = sum; self.masks_reserved = masks; self.started += 1;
        Ok(())
    }
    fn finish(&mut self, actual: Int8Work, masks: u64) -> Result<(), Int8RedactionError> {
        if self.started != self.completed + 1 || masks > self.mask_per_pass { return Err(Int8RedactionError::InvalidResult); }
        let sum = self.actual.checked_add(actual)?;
        let masks = self.masks_actual.checked_add(masks).ok_or(Int8RedactionError::InvalidResult)?;
        if !within(sum, self.reserved) || masks > self.masks_reserved { return Err(Int8RedactionError::InvalidResult); }
        self.actual = sum; self.masks_actual = masks; self.completed += 1; self.failed = false;
        Ok(())
    }
}
fn check_run(plan: &PreparedInt8SourceTask, run: Int8SourceTaskRun, masks: u64)
    -> Result<(NerResult, Int8Work), Int8RedactionError> {
    if run.schema_version != 1 || run.execution != INT8_SOURCE_EXECUTION { return Err(Int8RedactionError::InvalidResult); }
    let SourceTaskResult::Ner(result) = run.result else { return Err(Int8RedactionError::InvalidResult); };
    let expected = constrained_int8::planned_work(plan.prompt_tokens(), result.generated_token_ids.len())
        .map_err(|_| Int8RedactionError::InvalidResult)?;
    if result.schema_version != 1 || result.numerics_profile != STRICT_INT8_PROFILE
        || result.task_spec_version != NER_TASK_VERSION || result.generated_token_ids.is_empty()
        || result.generated_token_ids.len() > plan.task_budget().max_output_tokens as usize
        || run.model_work != expected || result.forward_positions != expected.forward_positions
        || result.projected_logits != expected.projected_logits || !within(expected, plan.planned_work())
        || result.mask_node_visit_charge > masks { return Err(Int8RedactionError::InvalidResult); }
    Ok((result, run.model_work))
}
fn check_engine(identity: &ExecutionIdentity, engine: &StrictInt8Engine<'_>, kv_cap: u64) -> Result<(), Int8RedactionError> {
    let model = engine.artifact_identity();
    if model.model_id != "Nanbeige4.2-3B" || model.revision != identity.source_revision
        || model.recipe_id != identity.quant_recipe
        || Sha256Digest::from_hex(&model.logical_model_sha256).ok() != Some(identity.logical_model_digest) {
        return Err(Int8RedactionError::Identity);
    }
    if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) { return Err(Int8RedactionError::InvalidResult); }
    if (engine.kv_cache().capacity_positions() as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
        .is_none_or(|n| n > kv_cap) { return Err(Int8RedactionError::WorkBudget); }
    Ok(())
}
fn within(a: Int8Work, b: Int8Work) -> bool {
    a.forward_positions <= b.forward_positions && a.projected_logits <= b.projected_logits
        && a.attention_pairs <= b.attention_pairs && a.projections.fits(b.projections)
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), Int8RedactionError> {
    match control.prefill_checkpoint(0) { Some(c) => Err(Int8RedactionError::Cancelled(c)), None => Ok(()) }
}

#[cfg(test)] mod tests;
