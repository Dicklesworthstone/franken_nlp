//! Closed native execution policies shared by live streams and durable jobs.
use super::*;
use crate::batch::extract::grouped_int4::memory::MAX_COHORT_RESULT_BYTES;

/// All discovery identities and all future synthesis work are admitted as one
/// document. Preparation and reduction memory remain held by the host.
pub struct Int4SourceDocumentAdmission<'a> {
    pub identities: &'a [ExecutionIdentity],
    pub model_work: Int4Work,
    pub mask_node_visits: u64,
    pub mask_limits: MaskWorkLimits,
    pub kv_reservation_bytes: u64,
    pub max_result_bytes: u64,
    pub max_output_tokens: u64,
    pub chunk_count: usize,
}
pub trait Int4SourceDocumentBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int4SourceDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure>;
}

/// Preserve every native semantic wire object and its exact execution tag.
#[derive(Serialize)]
#[serde(untagged)]
pub enum Int4SourceDocumentRun {
    Map(Int4SourceMapRun),
    Keyphrases(Int4CorpusKeyphraseRun),
    Summary(Int4CorpusSummaryRun),
    Question(Int4QuestionRun),
    SummarySynthesis(Int4SummarySynthesisRun),
    SummaryHierarchy(Int4SummaryHierarchyRun),
    QuestionSynthesis(Int4QuestionSynthesisRun),
    QuestionHierarchy(Int4QuestionHierarchyRun),
}

pub(crate) enum DocumentEngine<'e, 'w> {
    Serial(&'e mut StrictInt4Engine<'w>),
    Cohorts { weights: &'w Int4ModelWeights, budget: Int4SourceMapCohortBudget,
        kernel: PackedInt4Kernel, required: Int4SourceMapCohortRequirement,
        hierarchy: Option<Int4HierarchyCohortSchedule> },
}
impl DocumentEngine<'_, '_> {
    pub(crate) fn check_capacity(&self, config: &Configuration<'_>) -> Result<u64, BatchItemFailure> {
        match self {
            Self::Serial(engine) => {
                if config.uses_native_cohorts() { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
                constrained_int4::check_model(config.identity, engine.model_identity())
                    .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
                if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0)
                    || engine.model_identity().group_width() != config.group_width
                    || engine.kv_cache().capacity_positions() != config.planning.max_context_tokens {
                    return Err(BatchItemFailure::fatal(BatchCode::Admission));
                }
                let required = Int4MemoryRequirement::for_context(config.planning.max_context_tokens, config.group_width)
                    .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
                if required.kv_bytes == 0 || required.kv_bytes != config.budget.max_kv_bytes {
                    return Err(BatchItemFailure::fatal(BatchCode::Admission));
                }
                Ok(required.kv_bytes)
            }
            Self::Cohorts { weights, budget, required, hierarchy, .. } => {
                if !config.uses_native_cohorts() || *hierarchy != config.hierarchy
                    || budget.context_limit != config.planning.max_context_tokens
                    || weights.group_width() != config.group_width {
                    return Err(BatchItemFailure::fatal(BatchCode::Admission));
                }
                constrained_int4::check_model(config.identity, weights.identity())
                    .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
                check_cohort_budget(*required, *budget)?;
                Ok(required.native.kv_bytes)
            }
        }
    }
    pub(crate) fn preflight<C: DecodeStepControl>(&self, plan: &planning::NativePlan<'_>,
        identities: &[ExecutionIdentity], control: &mut C) -> Result<(), BatchItemFailure> {
        match self {
            Self::Serial(engine) => plan.preflight(identities, engine, control),
            Self::Cohorts { weights, budget, required, hierarchy, .. } => {
                let actual = plan.cohort_requirements(budget.context_limit, *hierarchy)?;
                check_cohort_peak(actual, *required)?;
                check_cohort_budget(actual, *budget)?;
                plan.preflight_cohorts(identities, weights, control)
            }
        }
    }
    pub(crate) fn execute<C: DecodeStepControl>(&mut self, plan: planning::NativePlan<'_>, admitted: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int4SourceDocumentRun, BatchItemFailure> {
        match self {
            Self::Serial(engine) => plan.execute(admitted, planner, engine, vocabulary, control),
            Self::Cohorts { weights, budget, kernel, hierarchy, .. } =>
                plan.execute_cohorts(admitted, planner, weights, vocabulary, *budget, *kernel, *hierarchy, control),
        }
    }
}
pub(crate) fn check_cohort_peak(actual: Int4SourceMapCohortRequirement, declared: Int4SourceMapCohortRequirement)
    -> Result<(), BatchItemFailure> {
    if actual.native.kv_bytes > declared.native.kv_bytes || actual.native.rope_bytes > declared.native.rope_bytes
        || actual.native.scratch_payload_bound > declared.native.scratch_payload_bound
        || actual.workspace_bytes > declared.workspace_bytes || actual.max_result_bytes > declared.max_result_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    Ok(())
}
pub(crate) fn check_cohort_budget(required: Int4SourceMapCohortRequirement, budget: Int4SourceMapCohortBudget)
    -> Result<(), BatchItemFailure> {
    if budget.max_result_bytes > MAX_COHORT_RESULT_BYTES { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
    check_cohort_peak(required, Int4SourceMapCohortRequirement {
        native: Int4MemoryRequirement { kv_bytes: budget.native_memory.max_kv_bytes,
            rope_bytes: budget.native_memory.max_rope_bytes,
            scratch_payload_bound: budget.native_memory.max_scratch_payload_bytes },
        workspace_bytes: budget.max_workspace_bytes, max_result_bytes: budget.max_result_bytes,
    })
}
