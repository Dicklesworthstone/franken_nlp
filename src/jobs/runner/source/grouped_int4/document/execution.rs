//! One exclusive engine and one result guard through durable acknowledgement.
use super::*;

/// Complete discovery admission. Future synthesis work is already included in
/// model_work and masks; the native hierarchy authenticates each full dynamic
/// level before its first forward. The host must retain complete preparation
/// and reduction storage in addition to this final-result guard.
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

/// Preserve the native wire object and its execution tag exactly. An extra
/// wrapper would consume bytes outside its already-validated result ceiling.
#[derive(Serialize)]
#[serde(untagged)]
pub enum Int4SourceDocumentRun {
    Map(Int4SourceMapRun),
    Summary(Int4CorpusSummaryRun),
    Question(Int4QuestionRun),
    SummarySynthesis(Int4SummarySynthesisRun),
    SummaryHierarchy(Int4SummaryHierarchyRun),
    QuestionSynthesis(Int4QuestionSynthesisRun),
    QuestionHierarchy(Int4QuestionHierarchyRun),
}

pub struct Int4SourceDocumentJobProcessor<'p, 'e, 'w, 'v, A: Int4SourceDocumentBatchAdmission> {
    planner: &'p Int4SourceDocumentJobPlanner,
    engine: &'e mut StrictInt4Engine<'w>,
    vocabulary: &'v ExtractionVocabulary,
    admission: A,
    ledger: Ledger,
}
impl<'p, 'e, 'w, 'v, A: Int4SourceDocumentBatchAdmission> Int4SourceDocumentJobProcessor<'p, 'e, 'w, 'v, A> {
    pub fn new(planner: &'p Int4SourceDocumentJobPlanner, engine: &'e mut StrictInt4Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        check_capacity(planner, engine).map_err(|e| e.fault)?;
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
        check_capacity(self.planner, self.engine)?;
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
        let kv = check_capacity(self.planner, self.engine)?;
        let plan = self.planner.plan(&prepared.source, control)?;
        plan.compare(&prepared, control)?;
        plan.preflight(&prepared.identities, self.engine, control)?;
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
        plan.preflight(&admitted, self.engine, control)?;
        let output = plan.execute(&admitted, &self.planner.planner, self.engine, self.vocabulary, control)?;
        prepared.expected.verify(&output, self.planner.task())?;
        check_capacity(self.planner, self.engine)?;
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
        check_capacity(self.planner, self.engine)?;
        if prepared.expected.work()? != prepared.work { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
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
