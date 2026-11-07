//! Only real resident scalar execution or owned native discovery cohorts can
//! execute the sealed question plans. Synthesis keeps its scalar continuation.
use super::*;
use crate::{batch::extract::grouped_int4::memory::MAX_COHORT_RESULT_BYTES,
    native_engine::strict_int4::STRICT_INT4_PROFILE,
    tasks::source_planning::grouped_int4::long::{
        cohort::synthesis::INT4_SYNTHESIS_COHORT_EXECUTION,
        question::cohort::INT4_QUESTION_COHORT_EXECUTION}};

pub(super) enum Backend<'e, 'w> {
    Serial(&'e mut StrictInt4Engine<'w>),
    Cohorts { weights: &'w Int4ModelWeights, budget: Int4SourceMapCohortBudget, kernel: PackedInt4Kernel },
}
impl Backend<'_, '_> {
    fn model(&self) -> &Int4ModelIdentity {
        match self { Self::Serial(engine) => engine.model_identity(), Self::Cohorts { weights, .. } => weights.identity() }
    }
    pub(super) fn clean(&self) -> bool {
        match self { Self::Serial(engine) => !engine.is_poisoned() && engine.kv_cache().all_slots_have_len(0),
            Self::Cohorts { .. } => true }
    }
    pub(super) fn capacity(&self, factory: &Int4QuestionDocumentBatchPlanner) -> Result<u64, BatchItemFailure> {
        constrained_int4::check_model(factory.base_execution_identity(), self.model())
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        if !self.clean() || self.model().group_width() != factory.group_width {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        match self {
            Self::Serial(engine) if !factory.uses_chunk_cohorts() => {
                let c = &factory.config; let context = c.planning.max_context_tokens;
                let required = Int4MemoryRequirement::for_context(context, factory.group_width)
                    .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
                if engine.profile() != STRICT_INT4_PROFILE || engine.kv_cache().capacity_positions() != context
                    || required.kv_bytes != c.budget.max_kv_bytes {
                    return Err(BatchItemFailure::fatal(BatchCode::Admission));
                }
                Ok(required.kv_bytes)
            }
            Self::Cohorts { budget, .. } if factory.uses_chunk_cohorts() => Ok(cohort_budget(factory, *budget)?.native.kv_bytes),
            _ => Err(BatchItemFailure::fatal(BatchCode::Admission)),
        }
    }
    pub(super) fn preflight<C: DecodeStepControl>(&self, factory: &Int4QuestionDocumentBatchPlanner,
        plan: &NativePlan<'_>, identities: &[ExecutionIdentity], control: &mut C) -> Result<(), BatchItemFailure> {
        self.capacity(factory)?;
        plan.preflight(identities, self.model(), control)?;
        match self {
            Self::Serial(engine) => {
                if let NativePlan::Independent(plan) = plan { plan.preflight(identities, engine).map_err(|e| map_failure(e, true))?; }
                // The sealed synthesis method preflights its complete discovery
                // before any forward and its evidence continuation before use.
            }
            Self::Cohorts { budget, .. } => {
                check_peak(plan.cohort_requirements(budget.context_limit)?, cohort_budget(factory, *budget)?)?;
            }
        }
        poll(control)
    }
    pub(super) fn execute<C: DecodeStepControl>(&mut self, plan: NativePlan<'_>, identities: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int4QuestionDocumentRun, BatchItemFailure> {
        match (self, plan) {
            (Self::Serial(engine), NativePlan::Independent(plan)) => plan.execute_with_control(identities, engine, vocabulary, control)
                .map(Int4QuestionDocumentRun::Independent).map_err(|e| map_failure(e, true)),
            (Self::Serial(engine), NativePlan::Synthesis(plan)) => plan.execute_with_control(identities, planner, engine, vocabulary, control)
                .map(Int4QuestionDocumentRun::Synthesis).map_err(|e| map_failure(e, true)),
            (Self::Cohorts { weights, budget, kernel }, NativePlan::Independent(plan)) => {
                let expected = plan.preflight_metadata();
                let run = plan.execute_cohorts_with_control(identities, weights, vocabulary, *budget, *kernel, control)
                    .map_err(|e| map_failure(e, true))?;
                if run.schema_version != 1 || run.execution != INT4_QUESTION_COHORT_EXECUTION {
                    return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
                }
                expected.verify_completed(&run.document).map_err(|e| map_failure(e, true))?;
                check_stats(run.cohorts, &run.document)?;
                Ok(Int4QuestionDocumentRun::Independent(run.document))
            }
            (Self::Cohorts { weights, budget, kernel }, NativePlan::Synthesis(plan)) => {
                let expected = plan.preflight_metadata();
                let run = plan.execute_cohorts_with_control(identities, planner, weights, vocabulary, *budget, *kernel, control)
                    .map_err(|e| map_failure(e, true))?;
                if run.schema_version != 1 || run.execution != INT4_SYNTHESIS_COHORT_EXECUTION {
                    return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
                }
                expected.verify_completed(&run.document).map_err(|e| map_failure(e, true))?;
                check_stats(run.discovery_cohorts, &run.document.discovery)?;
                Ok(Int4QuestionDocumentRun::Synthesis(run.document))
            }
        }
    }
}
pub(super) fn cohort_budget(factory: &Int4QuestionDocumentBatchPlanner, budget: Int4SourceMapCohortBudget)
    -> Result<Int4SourceMapCohortRequirement, BatchItemFailure> {
    if !factory.uses_chunk_cohorts() || budget.context_limit != factory.config.planning.max_context_tokens
        || budget.max_result_bytes > MAX_COHORT_RESULT_BYTES {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let required = factory.cohort_requirements().map_err(BatchItemFailure::fatal)?;
    check_peak(required, Int4SourceMapCohortRequirement {
        native: Int4MemoryRequirement { kv_bytes: budget.native_memory.max_kv_bytes,
            rope_bytes: budget.native_memory.max_rope_bytes, scratch_payload_bound: budget.native_memory.max_scratch_payload_bytes },
        workspace_bytes: budget.max_workspace_bytes, max_result_bytes: budget.max_result_bytes,
    })?;
    Ok(required)
}
fn check_peak(need: Int4SourceMapCohortRequirement, cap: Int4SourceMapCohortRequirement) -> Result<(), BatchItemFailure> {
    if need.native.kv_bytes > cap.native.kv_bytes || need.native.rope_bytes > cap.native.rope_bytes
        || need.native.scratch_payload_bound > cap.native.scratch_payload_bound
        || need.workspace_bytes > cap.workspace_bytes || need.max_result_bytes > cap.max_result_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    Ok(())
}
fn check_stats(stats: crate::tasks::source_planning::grouped_int4::long::cohort::Int4SourceMapCohortStats,
    run: &Int4QuestionRun) -> Result<(), BatchItemFailure> {
    // Concrete native execution already reconciles exact nonblank windows;
    // retain that invariant across the population's semantic-only boundary.
    if stats.chunks != run.native_chunks as u64 || stats.cohorts == 0
        || stats.cohorts > run.mapped.map_batches() as u64 || stats.cohorts > stats.chunks
        || stats.planned_work != run.planned_model_work || stats.model_work != run.model_work
        || stats.group_steps == 0 || stats.group_steps > run.model_work.forward_positions
        || stats.projection_groups == 0 || stats.projection_groups > stats.group_steps
        || stats.projection_groups > run.model_work.projected_logits / crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE as u64 {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    Ok(())
}
