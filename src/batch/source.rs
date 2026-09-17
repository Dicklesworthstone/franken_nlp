//! Native source tasks on the existing bounded, ordered NDJSON runner.
//! One admitted engine/vocabulary, no alternate model loader or scheduler.
//! Successful QA abstention remains a successful document with a typed inner
//! result; native failure is never rewritten to answerable=false.

use crate::{
    execution_identity::{ExecutionIdentity, NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{constrained::{JsonDecodeError, JsonWorkBudget}, decode::DecodeStepControl,
        hf_bf16_eager::HfBf16EagerEngine, kv::KV_BYTES_PER_TOKEN, lmhead::NANBEIGE_VOCAB_SIZE},
    tasks::{BuiltInTask, answer::{AnswerError, AnswerOptions, AnswerPassage},
        extract::{ExtractError, ExtractionVocabulary}, ir::{PlanContext, TaskBudget},
        keyphrases::{KeyphraseError, KeyphraseOptions}, ner::{NerError, NerOptions},
        summarize::{SummaryError, SummaryOptions}, source_planning::{PreparedSourceTask,
            SourceTaskPlanner, SourceTaskRequest, SourceTaskResult, SourcePlanningLimits, SourcePlanningError}},
};
use super::*;
pub use super::judge::{JudgeBatchAdmission as SourceBatchAdmission, GuardedOutput};
pub use super::extract::ExtractionMaskBudget as SourceMaskBudget;

/// BatchDocument.text is the source for NER/keyphrases/summarize, or the
/// QUESTION for answer. QA passages stay separately typed in task_args.
/// The planner has one fixed task identity; a record cannot switch tasks or
/// inject an execution identity, token sequence, tool mode or higher ceiling.
#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "task", rename_all = "snake_case", deny_unknown_fields)]
pub enum SourceBatchArgs {
    Ner { options: NerOptions, budget: TaskBudget },
    Keyphrases { options: KeyphraseOptions, budget: TaskBudget },
    Summarize { options: SummaryOptions, budget: TaskBudget },
    Answer { passages: Vec<AnswerPassage>, options: AnswerOptions, budget: TaskBudget },
}
impl SourceBatchArgs {
    fn task(&self) -> BuiltInTask {
        match self { Self::Ner { .. } => BuiltInTask::Ner, Self::Keyphrases { .. } => BuiltInTask::Keyphrases,
            Self::Summarize { .. } => BuiltInTask::Summarize, Self::Answer { .. } => BuiltInTask::Answer }
    }
    fn into_request(self, text: String) -> SourceTaskRequest {
        match self {
            Self::Ner { options, budget } => SourceTaskRequest::Ner { document: text, options, budget },
            Self::Keyphrases { options, budget } => SourceTaskRequest::Keyphrases { document: text, options, budget },
            Self::Summarize { options, budget } => SourceTaskRequest::Summarize { document: text, options, budget },
            Self::Answer { passages, options, budget } => SourceTaskRequest::Answer { question: text, passages, options, budget },
        }
    }
}

pub struct SourceBatchPlanner<'p> {
    planner: &'p SourceTaskPlanner,
    identity: ExecutionIdentity,
    ceiling: TaskBudget,
    limits: SourcePlanningLimits,
    defaults: Option<SourceBatchArgs>,
}
pub struct PreparedBatchSource {
    plan: PreparedSourceTask,
    budget: TaskBudget,
    work: BatchWork,
}
impl PreparedBatchSource {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.plan.execution_identity() }
    pub fn planned_work(&self) -> BatchWork { self.work }
    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), BatchFault> {
        self.plan.verify_identity(admitted).map_err(|_| BatchCode::Admission.into())
    }
    fn preflight(&self, engine: &HfBf16EagerEngine) -> Result<(), BatchItemFailure> {
        check_capacity(engine.kv_cache().all_slots_have_len(0), engine.kv_cache().capacity_positions() as u64,
            self.work.forward_positions, self.budget.max_kv_bytes)
    }
}
impl<'p> SourceBatchPlanner<'p> {
    pub fn new(planner: &'p SourceTaskPlanner, identity: ExecutionIdentity, ceiling: TaskBudget,
        limits: SourcePlanningLimits, defaults: Option<SourceBatchArgs>) -> Result<Self, BatchFault> {
        identity.validate().map_err(|_| BatchCode::Admission)?;
        ceiling.validate().map_err(|_| BatchCode::InvalidLimits)?;
        if !matches!(identity.task_spec.as_str(), "ner-v1" | "keyphrases-v1" | "summarize-v1" | "answer-v1")
            || identity.template_digest != *planner.template_digest() || identity.tokenizer_digest != planner.tokenizer_digest()
            || identity.numerics_profile != NumericsProfile::HfBf16Eager || identity.kv_dtype != "bf16"
            || identity.thinking_mode != ThinkingMode::Disabled || identity.tool_mode != ToolMode::None
            || defaults.as_ref().is_some_and(|a| a.task().spec().identity() != identity.task_spec)
        { return Err(BatchCode::Admission.into()); }
        Ok(Self { planner, identity, ceiling, limits, defaults })
    }
    pub fn prepare(&self, document: BatchDocument<SourceBatchArgs>) -> Result<PreparedBatchSource, BatchItemFailure> {
        let args = document.task_args.or_else(|| self.defaults.clone())
            .ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        if args.task().spec().identity() != self.identity.task_spec { return Err(BatchItemFailure::reject(BatchCode::Planning)); }
        let request = args.into_request(document.text); let budget = request.budget();
        let context = PlanContext::new(&self.identity, self.ceiling).map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        let plan = self.planner.plan(&request, &context, self.limits).map_err(preparation_failure)?;
        let forward = (plan.prompt_tokens() as u64).checked_add(u64::from(budget.max_output_tokens))
            .and_then(|n| n.checked_sub(1)).ok_or_else(|| BatchItemFailure::reject(BatchCode::WorkLimit))?;
        let projected = forward.checked_mul(NANBEIGE_VOCAB_SIZE as u64)
            .ok_or_else(|| BatchItemFailure::reject(BatchCode::WorkLimit))?;
        Ok(PreparedBatchSource { plan, budget, work: BatchWork { forward_positions: forward, projected_logits: projected } })
    }
}

/// The host supplies actual admission authority. Its guard travels with the
/// result through serialization/write/flush, including successful abstentions.
/// Mask allowance is nonrenewable across items AND flush epochs, like the
/// runner's conservative forward/projection work reservations.
pub struct NativeSourceBatch<'p, 'e, 'v, A: SourceBatchAdmission> {
    compiler: SourceBatchPlanner<'p>,
    engine: &'e mut HfBf16EagerEngine,
    vocabulary: &'v ExtractionVocabulary,
    admission: A,
    masks: SourceMaskBudget,
    remaining_mask_visits: u64,
}
impl<'p, 'e, 'v, A: SourceBatchAdmission> NativeSourceBatch<'p, 'e, 'v, A> {
    pub fn new(compiler: SourceBatchPlanner<'p>, engine: &'e mut HfBf16EagerEngine,
        vocabulary: &'v ExtractionVocabulary, admission: A, masks: SourceMaskBudget) -> Result<Self, BatchFault> {
        if !engine.kv_cache().all_slots_have_len(0) { return Err(BatchCode::InvalidExecution.into()); }
        if masks.per_mask.max_trie_node_visits == 0 || masks.per_mask.checkpoint_interval_nodes == 0
            || masks.max_visits_per_item == 0 { return Err(BatchCode::InvalidLimits.into()); }
        Ok(Self { compiler, engine, vocabulary, admission, masks, remaining_mask_visits: masks.max_visits_per_run })
    }
    pub fn reserved_mask_visits(&self) -> u64 { self.masks.max_visits_per_run - self.remaining_mask_visits }
}
impl<A: SourceBatchAdmission> BatchProcessor for NativeSourceBatch<'_, '_, '_, A> {
    type Args = SourceBatchArgs;
    type Prepared = PreparedBatchSource;
    type Output = GuardedOutput<SourceTaskResult, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        if self.remaining_mask_visits < self.masks.max_visits_per_item { return Err(BatchItemFailure::reject(BatchCode::WorkLimit)); }
        let prepared = self.compiler.prepare(document)?; prepared.preflight(self.engine)?; Ok(prepared)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { prepared.work }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        reserve_masks(&mut self.remaining_mask_visits, self.masks.max_visits_per_item)?;
        let (identity, guard) = self.admission.admit(prepared.execution_identity(), prepared.work)?;
        prepared.verify_identity(&identity).map_err(BatchItemFailure::fatal)?;
        prepared.preflight(self.engine)?;
        let result = prepared.plan.execute_eager(&identity, self.engine, self.vocabulary, JsonWorkBudget {
            max_forward_positions: prepared.work.forward_positions, max_projected_logits: prepared.work.projected_logits,
            max_kv_bytes: prepared.budget.max_kv_bytes, max_total_mask_node_visits: self.masks.max_visits_per_item,
            mask_limits: self.masks.per_mask,
        }, control);
        if !self.engine.kv_cache().all_slots_have_len(0) { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        let result = result.map_err(execution_failure)?;
        check_accounting(prepared.plan.prompt_tokens() as u64, prepared.budget.max_output_tokens, prepared.work,
            observed(&result), self.masks.max_visits_per_item)?;
        Ok(GuardedOutput::new(result, guard))
    }
}
fn reserve_masks(remaining: &mut u64, charge: u64) -> Result<(), BatchItemFailure> {
    *remaining = remaining.checked_sub(charge).ok_or_else(|| BatchItemFailure::reject(BatchCode::WorkLimit))?;
    Ok(())
}
fn check_capacity(empty: bool, capacity: u64, positions: u64, kv_bytes: u64) -> Result<(), BatchItemFailure> {
    if !empty { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
    let resident = capacity.checked_mul(KV_BYTES_PER_TOKEN as u64)
        .ok_or_else(|| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    if positions > capacity || resident > kv_bytes { return Err(BatchItemFailure::reject(BatchCode::Admission)); }
    Ok(())
}
#[derive(Clone, Copy)]
struct ObservedWork { tokens: usize, positions: u64, logits: u64, mask_visits: u64 }
fn observed(result: &SourceTaskResult) -> ObservedWork {
    let (ids, positions, logits, mask_visits) = match result {
        SourceTaskResult::Ner(r) => (&r.generated_token_ids, r.forward_positions, r.projected_logits, r.mask_node_visit_charge),
        SourceTaskResult::Keyphrases(r) => (&r.generated_token_ids, r.forward_positions, r.projected_logits, r.mask_node_visit_charge),
        SourceTaskResult::Summarize(r) => (&r.generated_token_ids, r.forward_positions, r.projected_logits, r.mask_node_visit_charge),
        SourceTaskResult::Answer(r) => (&r.generated_token_ids, r.forward_positions, r.projected_logits, r.mask_node_visit_charge),
    };
    ObservedWork { tokens: ids.len(), positions, logits, mask_visits }
}
fn check_accounting(prompt: u64, max_tokens: u32, ceiling: BatchWork, actual: ObservedWork, mask_cap: u64)
    -> Result<(), BatchItemFailure> {
    let positions = prompt.checked_add(actual.tokens as u64).and_then(|n| n.checked_sub(1));
    if actual.tokens == 0 || actual.tokens > max_tokens as usize || positions != Some(actual.positions)
        || actual.positions > ceiling.forward_positions || actual.positions.checked_mul(NANBEIGE_VOCAB_SIZE as u64) != Some(actual.logits)
        || actual.logits > ceiling.projected_logits || actual.mask_visits > mask_cap
    { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
    Ok(())
}
fn preparation_failure(error: SourcePlanningError) -> BatchItemFailure {
    if matches!(error, SourcePlanningError::AllocationRefused | SourcePlanningError::Extraction(ExtractError::AllocationRefused)
        | SourcePlanningError::Ner(NerError::AllocationRefused | NerError::Extraction(ExtractError::AllocationRefused))
        | SourcePlanningError::Keyphrases(KeyphraseError::AllocationRefused | KeyphraseError::Extraction(ExtractError::AllocationRefused))
        | SourcePlanningError::Summary(SummaryError::AllocationRefused | SummaryError::Extraction(ExtractError::AllocationRefused))
        | SourcePlanningError::Answer(AnswerError::AllocationRefused | AnswerError::Extraction(ExtractError::AllocationRefused)))
    { BatchItemFailure::fatal(BatchCode::Allocation) } else { BatchItemFailure::reject(BatchCode::Planning) }
}
fn execution_failure(error: SourcePlanningError) -> BatchItemFailure {
    match error {
        SourcePlanningError::Extraction(e) | SourcePlanningError::Ner(NerError::Extraction(e))
        | SourcePlanningError::Keyphrases(KeyphraseError::Extraction(e)) | SourcePlanningError::Summary(SummaryError::Extraction(e))
        | SourcePlanningError::Answer(AnswerError::Extraction(e)) => extraction_failure(e),
        SourcePlanningError::OutputBudget | SourcePlanningError::Ner(NerError::OutputBudgetExceeded)
        | SourcePlanningError::Keyphrases(KeyphraseError::OutputBudgetExceeded) | SourcePlanningError::Summary(SummaryError::OutputBudgetExceeded)
        | SourcePlanningError::Answer(AnswerError::OutputBudgetExceeded) => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        SourcePlanningError::AllocationRefused | SourcePlanningError::Ner(NerError::AllocationRefused)
        | SourcePlanningError::Keyphrases(KeyphraseError::AllocationRefused) | SourcePlanningError::Summary(SummaryError::AllocationRefused)
        | SourcePlanningError::Answer(AnswerError::AllocationRefused) => BatchItemFailure::fatal(BatchCode::Allocation),
        SourcePlanningError::Ner(NerError::InvalidResult) | SourcePlanningError::Keyphrases(KeyphraseError::InvalidResult)
        | SourcePlanningError::Summary(SummaryError::InvalidResult) | SourcePlanningError::Answer(AnswerError::InvalidResult)
            => BatchItemFailure::reject(BatchCode::Execution),
        SourcePlanningError::Serialization | SourcePlanningError::Ner(NerError::Serialization)
        | SourcePlanningError::Keyphrases(KeyphraseError::Serialization) | SourcePlanningError::Summary(SummaryError::Serialization)
        | SourcePlanningError::Answer(AnswerError::Serialization) => BatchItemFailure::fatal(BatchCode::Serialization),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    }
}
fn extraction_failure(error: ExtractError) -> BatchItemFailure {
    match error {
        ExtractError::Decode(JsonDecodeError::Cancelled(cause)) => BatchItemFailure::fatal(BatchFault::cancelled(cause)),
        ExtractError::Decode(JsonDecodeError::BudgetExceeded(_)) => BatchItemFailure::reject(BatchCode::WorkLimit),
        ExtractError::OutputBudgetExceeded => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        ExtractError::Decode(JsonDecodeError::NoLegalToken | JsonDecodeError::Mask(_)) => BatchItemFailure::reject(BatchCode::Execution),
        ExtractError::AllocationRefused | ExtractError::Decode(JsonDecodeError::AllocationRefused) => BatchItemFailure::fatal(BatchCode::Allocation),
        ExtractError::Decode(JsonDecodeError::Engine(_)) => BatchItemFailure::fatal(BatchCode::Execution),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    }
}

#[cfg(test)]
mod tests;
