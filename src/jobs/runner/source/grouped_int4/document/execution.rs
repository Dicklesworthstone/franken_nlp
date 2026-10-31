//! One native execution policy and one result guard through durable acknowledgement.
use super::*;
use crate::batch::extract::grouped_int4::memory::MAX_COHORT_RESULT_BYTES;

/// Complete discovery admission. Future synthesis work is already included in
/// model_work and masks; the native hierarchy authenticates each full dynamic
/// level before its first forward. The host must retain complete preparation
/// and reduction storage in addition to this final-result guard.
pub struct Int4SourceDocumentAdmission<'a> {
    pub identities: &'a [ExecutionIdentity],
    pub model_work: Int4Work,
    pub mask_node_visits: u64,
    pub mask_limits: MaskWorkLimits,
    /// Complete declared native KV peak. With chunk cohorts this is the
    /// invocation's static physical reservation, never per-row task authority.
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

/// Preserve the native wire object and its execution tag exactly. An extra
/// wrapper would consume bytes outside its already-validated result ceiling.
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
    /// Reuse the same whole-attempt ledger and durable result guard while the
    /// sealed document plans allocate and drain their own native chunk cohorts.
    /// The host retains the complete static quote through processor teardown.
    #[allow(clippy::too_many_arguments)]
    pub fn new_cohorts(planner: &'p Int4SourceDocumentJobPlanner, weights: &'w Int4ModelWeights,
        vocabulary: &'v ExtractionVocabulary, budget: Int4SourceMapCohortBudget,
        kernel: PackedInt4Kernel, admission: A) -> Result<Self, BatchFault> {
        let required = planner.cohort_requirements()?;
        Self::configured(planner, DocumentEngine::Cohorts { weights, budget, kernel, required }, vocabulary, admission)
    }
    fn configured(planner: &'p Int4SourceDocumentJobPlanner, engine: DocumentEngine<'e, 'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        engine.check_capacity(planner).map_err(|e| e.fault)?;
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
        self.engine.check_capacity(self.planner)?;
        let prepared = self.planner.prepare_with_control(document, control)?;
        // Preparation snapshots every actual discovery identity and reserves
        // every permitted future synthesis pass before the journal debit.
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
        // JobRunner has durably charged this complete reserved attempt. This
        // invocation ledger adds no retry allowance and never replaces it.
        self.ledger.begin(prepared.work, context, self.planner.limits)?;
        self.planner.owns(&prepared)?;
        poll(control)?;
        let kv = self.engine.check_capacity(self.planner)?;
        let plan = self.planner.plan(&prepared.source, control)?;
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
        // ALL admitted identities are checked before any native forward. The
        // existing drivers also bind the same compiler and actual model for
        // dynamic synthesis; callers cannot supply native results or plans.
        self.engine.preflight(&plan, &admitted, control)?;
        let output = self.engine.execute(plan, &admitted, &self.planner.planner, self.vocabulary, control)?;
        prepared.expected.verify(&output, self.planner.task())?;
        self.engine.check_capacity(self.planner)?;
        poll(control)?;
        self.ledger.failed = false;
        // JobRunner retains this guard through serialization, spool sync and
        // authenticated journal acknowledgement of the whole document.
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
        self.engine.check_capacity(self.planner)?;
        if prepared.expected.work()? != prepared.work { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        Ok(prepared.work)
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.planner.max_result_bytes() }
}

// Closed production policies share one prepare/debit/execute/verify lifecycle.
// There is no task callback, second scheduler or caller-supplied native result.
enum DocumentEngine<'e, 'w> {
    Serial(&'e mut StrictInt4Engine<'w>),
    Cohorts { weights: &'w Int4ModelWeights, budget: Int4SourceMapCohortBudget,
        kernel: PackedInt4Kernel, required: Int4SourceMapCohortRequirement },
}
impl DocumentEngine<'_, '_> {
    fn check_capacity(&self, planner: &Int4SourceDocumentJobPlanner) -> Result<u64, BatchItemFailure> {
        match self {
            Self::Serial(engine) => {
                if planner.uses_chunk_cohorts() { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
                check_capacity(planner, engine)
            }
            Self::Cohorts { weights, budget, required, .. } => {
                if !planner.uses_chunk_cohorts() || budget.context_limit != planner.planning.max_context_tokens
                    || weights.group_width() != planner.group_width {
                    return Err(BatchItemFailure::fatal(BatchCode::Admission));
                }
                constrained_int4::check_model(planner.execution_identity(), weights.identity())
                    .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
                check_cohort_budget(*required, *budget)?;
                Ok(required.native.kv_bytes)
            }
        }
    }
    fn preflight<C: DecodeStepControl>(&self, plan: &planning::NativePlan<'_>,
        identities: &[ExecutionIdentity], control: &mut C) -> Result<(), BatchItemFailure> {
        match self {
            Self::Serial(engine) => plan.preflight(identities, engine, control),
            Self::Cohorts { weights, budget, required, .. } => {
                let actual = plan.cohort_requirements(budget.context_limit)?;
                check_cohort_peak(actual, *required)?;
                check_cohort_budget(actual, *budget)?;
                plan.preflight_cohorts(identities, weights, control)
            }
        }
    }
    fn execute<C: DecodeStepControl>(&mut self, plan: planning::NativePlan<'_>, admitted: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int4SourceDocumentRun, BatchItemFailure> {
        match self {
            Self::Serial(engine) => plan.execute(admitted, planner, engine, vocabulary, control),
            Self::Cohorts { weights, budget, kernel, .. } =>
                plan.execute_cohorts(admitted, planner, weights, vocabulary, *budget, *kernel, control),
        }
    }
}
fn check_cohort_peak(actual: Int4SourceMapCohortRequirement, declared: Int4SourceMapCohortRequirement)
    -> Result<(), BatchItemFailure> {
    if actual.native.kv_bytes > declared.native.kv_bytes || actual.native.rope_bytes > declared.native.rope_bytes
        || actual.native.scratch_payload_bound > declared.native.scratch_payload_bound
        || actual.workspace_bytes > declared.workspace_bytes || actual.max_result_bytes > declared.max_result_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    Ok(())
}
fn check_cohort_budget(required: Int4SourceMapCohortRequirement, budget: Int4SourceMapCohortBudget)
    -> Result<(), BatchItemFailure> {
    if budget.max_result_bytes > MAX_COHORT_RESULT_BYTES { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
    check_cohort_peak(required, Int4SourceMapCohortRequirement {
        native: Int4MemoryRequirement { kv_bytes: budget.native_memory.max_kv_bytes,
            rope_bytes: budget.native_memory.max_rope_bytes,
            scratch_payload_bound: budget.native_memory.max_scratch_payload_bytes },
        workspace_bytes: budget.max_workspace_bytes, max_result_bytes: budget.max_result_bytes,
    })
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
fn check_capacity(planner: &Int4SourceDocumentJobPlanner, engine: &StrictInt4Engine<'_>) -> Result<u64, BatchItemFailure> {
    constrained_int4::check_model(planner.execution_identity(), engine.model_identity())
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0)
        || engine.model_identity().group_width() != planner.group_width
        || engine.kv_cache().capacity_positions() != planner.planning.max_context_tokens {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let required = Int4MemoryRequirement::for_context(planner.planning.max_context_tokens, planner.group_width)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    if required.kv_bytes == 0 || required.kv_bytes != planner.budget.max_kv_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    Ok(required.kv_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_engine::strict_int4::Int4MemoryBudget;

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
