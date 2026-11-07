//! Closed native backends. All real schema/source rows are preflighted before
//! accepting a complete document and again before its first forward.
use super::*;
use crate::{
    native_engine::strict_int4::Int4MemoryBudget,
    tasks::extract::grouped_int4::cohort::{self as extraction, Int4ExtractCohortRequest},
};

pub(super) enum Backend<'e, 'w> {
    Serial(&'e mut StrictInt4Engine<'w>),
    Cohorts { weights: &'w Int4ModelWeights, budget: Int4ExtractionMapCohortBudget, kernel: PackedInt4Kernel },
}
impl Backend<'_, '_> {
    fn model_identity(&self) -> &Int4ModelIdentity {
        match self { Self::Serial(engine) => engine.model_identity(), Self::Cohorts { weights, .. } => weights.identity() }
    }
    pub(super) fn clean(&self) -> bool {
        match self {
            Self::Serial(engine) => !engine.is_poisoned() && engine.kv_cache().all_slots_have_len(0),
            Self::Cohorts { .. } => true,
        }
    }
    pub(super) fn check_capacity(&self, planner: &Int4ExtractionDocumentBatchPlanner) -> Result<u64, BatchItemFailure> {
        constrained_int4::check_model(planner.base_execution_identity(), self.model_identity())
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        if !self.clean() || self.model_identity().group_width() != planner.group_width() {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        match self {
            Self::Serial(engine) if !planner.uses_chunk_cohorts() => {
                let context = planner.document_limits().mapping.chunks.context_tokens;
                let required = Int4MemoryRequirement::for_context(context, planner.group_width())
                    .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
                if engine.kv_cache().capacity_positions() != context || required.kv_bytes == 0
                    || required.kv_bytes != planner.task_ceiling().max_kv_bytes
                    || required.kv_bytes != planner.document_budget().max_kv_bytes {
                    return Err(BatchItemFailure::fatal(BatchCode::Admission));
                }
                Ok(required.kv_bytes)
            }
            Self::Cohorts { budget, .. } if planner.uses_chunk_cohorts() => Ok(cohort_budget(planner, *budget)?.native.kv_bytes),
            _ => Err(BatchItemFailure::fatal(BatchCode::Admission)),
        }
    }
    pub(super) fn preflight<C: DecodeStepControl>(&mut self, planner: &Int4ExtractionDocumentBatchPlanner,
        plan: &PreparedInt4ExtractionMap<'_>, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<(), BatchItemFailure> {
        self.check_capacity(planner)?;
        if plan.plans.len() != plan.chunks.chunks().len() || plan.plans.len() != plan.expected.chunk_count() {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        let masks = ExtractionMaskBudget { per_mask: planner.document_limits().mapping.mask_limits,
            max_visits_per_item: planner.document_limits().mapping.mask_visits_per_chunk,
            max_visits_per_run: planner.document_limits().mapping.max_mask_visits };
        for (row, chunk) in plan.plans.iter().zip(plan.chunks.chunks()) {
            poll(control)?;
            constrained_int4::check_model(row.execution_identity(), self.model_identity())
                .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
            if row.source().text() != chunk.text() { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
            let mut budget = native_budget(row, masks);
            match self {
                Self::Serial(engine) => {
                    preflight(row, &NativeDriver { engine, vocabulary })?;
                }
                Self::Cohorts { .. } => {
                    // Each independent row receives exactly its own actual
                    // prompt+output capacity, never the whole-window KV cap.
                    budget.json.max_kv_bytes = row.planned_work().forward_positions
                        .checked_mul(KV_BYTES_PER_TOKEN as u64)
                        .ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))?;
                }
            }
            extraction::preflight(&Int4ExtractCohortRequest { prepared: row.extraction_plan(),
                admitted_identity: row.execution_identity(), budget }, vocabulary)
                .map_err(|e| map_failure(e.into(), true))?;
        }
        if let Self::Cohorts { budget, .. } = self {
            let actual = plan.cohort_requirements(budget.context_limit).map_err(|e| map_failure(e, true))?;
            check_peak(actual, cohort_budget(planner, *budget)?)?;
        }
        poll(control)
    }
    pub(super) fn execute<C: DecodeStepControl>(&mut self, plan: PreparedInt4ExtractionMap<'_>,
        admitted: &[ExecutionIdentity], vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int4ExtractionMapRun, BatchItemFailure> {
        match self {
            Self::Serial(engine) => plan.execute_with_control(admitted, engine, vocabulary, control).map_err(|e| map_failure(e, true)),
            Self::Cohorts { weights, budget, kernel } => plan.execute_cohorts_with_control(
                admitted, weights, vocabulary, *budget, *kernel, control).map(|run| run.document).map_err(|e| map_failure(e, true)),
        }
    }
}
fn cohort_budget(planner: &Int4ExtractionDocumentBatchPlanner, budget: Int4ExtractionMapCohortBudget)
    -> Result<Int4ExtractionMapCohortRequirement, BatchItemFailure> {
    if !planner.uses_chunk_cohorts() || budget.context_limit != planner.document_limits().mapping.chunks.context_tokens
        || budget.max_result_bytes > crate::batch::extract::grouped_int4::memory::MAX_COHORT_RESULT_BYTES {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let required = planner.cohort_requirements().map_err(BatchItemFailure::fatal)?;
    let Int4MemoryBudget { max_kv_bytes, max_rope_bytes, max_scratch_payload_bytes } = budget.native_memory;
    let supplied = Int4ExtractionMapCohortRequirement {
        native: Int4MemoryRequirement { kv_bytes: max_kv_bytes, rope_bytes: max_rope_bytes, scratch_payload_bound: max_scratch_payload_bytes },
        workspace_bytes: budget.max_workspace_bytes, max_result_bytes: budget.max_result_bytes,
    };
    check_peak(required, supplied)?;
    Ok(required)
}
fn check_peak(required: Int4ExtractionMapCohortRequirement, ceiling: Int4ExtractionMapCohortRequirement)
    -> Result<(), BatchItemFailure> {
    if required.native.kv_bytes > ceiling.native.kv_bytes || required.native.rope_bytes > ceiling.native.rope_bytes
        || required.native.scratch_payload_bound > ceiling.native.scratch_payload_bound
        || required.workspace_bytes > ceiling.workspace_bytes || required.max_result_bytes > ceiling.max_result_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    Ok(())
}
