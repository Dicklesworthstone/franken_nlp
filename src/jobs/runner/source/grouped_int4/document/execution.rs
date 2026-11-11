//! Durable accounting delegates native admission/execution to the live core.
use super::*;
use native_document::backend::DocumentEngine;

pub struct Int4SourceDocumentJobProcessor<'p, 'e, 'w, 'v, A: Int4SourceDocumentBatchAdmission> {
    planner: &'p Int4SourceDocumentJobPlanner,
    engine: DocumentEngine<'e, 'w>,
    vocabulary: &'v ExtractionVocabulary,
    admission: A,
    ledger: Ledger,
}
impl<'p, 'e, 'w, 'v, A: Int4SourceDocumentBatchAdmission> Int4SourceDocumentJobProcessor<'p, 'e, 'w, 'v, A> {
    pub fn new(planner: &'p Int4SourceDocumentJobPlanner, engine: &'e mut StrictInt4Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        Self::configured(planner, DocumentEngine::Serial(engine), vocabulary, admission)
    }
    /// Keep the complete immutable physical quote and the durable result guard
    /// while the shared native plans allocate and drain their chunk cohorts.
    pub fn new_cohorts(planner: &'p Int4SourceDocumentJobPlanner, weights: &'w Int4ModelWeights,
        vocabulary: &'v ExtractionVocabulary, budget: Int4SourceMapCohortBudget,
        kernel: PackedInt4Kernel, admission: A) -> Result<Self, BatchFault> {
        let required = planner.cohort_requirements()?;
        let hierarchy = planner.hierarchy_cohort_schedule();
        Self::configured(planner, DocumentEngine::Cohorts { weights, budget, kernel, required, hierarchy }, vocabulary, admission)
    }
    fn configured(planner: &'p Int4SourceDocumentJobPlanner, engine: DocumentEngine<'e, 'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        engine.check_capacity(&planner.configuration()).map_err(|e| e.fault)?;
        planner.check_job_limits(planner.limits)?;
        Ok(Self { planner, engine, vocabulary, admission, ledger: Ledger::default() })
    }
}
impl<A: Int4SourceDocumentBatchAdmission> BatchProcessor for Int4SourceDocumentJobProcessor<'_, '_, '_, '_, A> {
    type Args = SourceDocumentArgs;
    type Prepared = PreparedInt4SourceDocument;
    type Output = GuardedOutput<Int4SourceDocumentRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.failed = true;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.ready()?;
        self.ledger.failed = true;
        self.engine.check_capacity(&self.planner.configuration())?;
        let prepared = self.planner.prepare_with_control(document, control)?;
        self.ledger.failed = false;
        Ok(prepared)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        BatchWork { forward_positions: prepared.work.model.forward_positions,
            projected_logits: prepared.work.model.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.ledger.failed = true;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        // JobRunner already persisted this whole attempt. The invocation
        // ledger adds no retry authority and never replaces that durable debit.
        self.ledger.begin(prepared.work, context, self.planner.limits)?;
        self.planner.owns(&prepared)?;
        poll(control)?;
        let kv = self.engine.check_capacity(&self.planner.configuration())?;
        let plan = self.planner.plan_with_operation(&prepared.source, &prepared.operation, control)?;
        plan.compare(&prepared, control)?;
        self.engine.preflight(&plan, &prepared.identities, control)?;
        let (admitted, guard) = self.admission.admit(Int4SourceDocumentAdmission {
            identities: &prepared.identities, model_work: prepared.work.int4_model_work(),
            mask_node_visits: prepared.work.mask_node_visits, mask_limits: self.planner.mapping.mask_limits,
            kv_reservation_bytes: kv, max_result_bytes: self.planner.max_result_bytes(),
            max_output_tokens: self.planner.output_tokens(prepared.chunk_count()).map_err(BatchItemFailure::fatal)?,
            chunk_count: prepared.chunk_count(),
        })?;
        poll(control)?;
        if admitted != prepared.identities { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        self.engine.preflight(&plan, &admitted, control)?;
        let output = self.engine.execute(plan, &admitted, &prepared.operation, self.vocabulary, control)?;
        prepared.expected.verify(&output, self.planner.task())?;
        self.engine.check_capacity(&self.planner.configuration())?;
        poll(control)?;
        self.ledger.failed = false;
        // This guard remains held through spool sync and journal acknowledgement.
        Ok(GuardedOutput::new(output, guard))
    }
}
impl<A: Int4SourceDocumentBatchAdmission> DurableBatchProcessor for Int4SourceDocumentJobProcessor<'_, '_, '_, '_, A> {
    type Recipe = Int4SourceDocumentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { self.planner.job_recipe() }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.planner.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        self.ledger.ready()?;
        self.planner.owns(prepared)?;
        self.engine.check_capacity(&self.planner.configuration())?;
        if planning::expected_job_work(prepared.expected)? != prepared.work { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        Ok(prepared.work)
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.planner.max_result_bytes() }
}

#[derive(Default)]
struct Ledger { work: JobWork, attempts: u64, sequence: u64, failed: bool }
impl Ledger {
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn begin(&mut self, work: JobWork, context: BatchRequestContext, limits: JobLimits) -> Result<(), BatchItemFailure> {
        self.ready()?;
        self.failed = true;
        if context.request_seq <= self.sequence || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let attempts = self.attempts.checked_add(1).filter(|&n| n <= limits.max_attempts)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let next = self.work.checked_add(work).ok().filter(|w| w.fits(limits.max_work))
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        self.work = next; self.attempts = attempts; self.sequence = context.request_seq;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use native_document::backend::{check_cohort_budget, check_cohort_peak};
    use crate::{batch::extract::grouped_int4::memory::MAX_COHORT_RESULT_BYTES,
        native_engine::strict_int4::Int4MemoryBudget};

    fn declared() -> Int4SourceMapCohortRequirement {
        Int4SourceMapCohortMemoryLimits { max_chunks: 4, context_tokens: 8192,
            max_output_tokens_per_chunk: 64, max_result_bytes_per_chunk: 65536 }
            .requirements(32).unwrap()
    }
    fn budget(required: Int4SourceMapCohortRequirement) -> Int4SourceMapCohortBudget {
        Int4SourceMapCohortBudget { context_limit: 8192,
            native_memory: Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                max_rope_bytes: required.native.rope_bytes,
                max_scratch_payload_bytes: required.native.scratch_payload_bound },
            max_workspace_bytes: required.workspace_bytes, max_result_bytes: required.max_result_bytes }
    }
    #[test]
    fn every_static_reservation_axis_must_fit_before_admission() {
        let required = declared();
        check_cohort_budget(required, budget(required)).unwrap();
        for axis in 0..5 {
            let mut short = required;
            match axis {
                0 => short.native.kv_bytes -= 1,
                1 => short.native.rope_bytes -= 1,
                2 => short.native.scratch_payload_bound -= 1,
                3 => short.workspace_bytes -= 1,
                _ => short.max_result_bytes -= 1,
            }
            assert!(check_cohort_budget(required, budget(short)).is_err(), "axis {axis}");
            assert!(check_cohort_peak(required, short).is_err(), "axis {axis}");
            check_cohort_peak(short, required).unwrap();
        }
        let mut oversized = budget(required); oversized.max_result_bytes = MAX_COHORT_RESULT_BYTES + 1;
        assert!(check_cohort_budget(required, oversized).is_err());
    }
}
