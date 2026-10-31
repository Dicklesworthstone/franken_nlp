//! One closed native schedule; the complete result guard survives durable sync.
use super::*;

/// Admission receives every original ordered chunk identity and the actual
/// group-aware planned work. The host separately retains preparation, source,
/// native output staging and reduction storage for the complete operation.
pub struct Int4ExtractionDocumentAdmission<'a> {
    pub identities: &'a [ExecutionIdentity],
    pub model_work: Int4Work,
    pub mask_node_visits: u64,
    pub mask_limits: MaskWorkLimits,
    pub kv_reservation_bytes: u64,
    pub max_result_bytes: u64,
    pub max_output_tokens: u64,
    pub chunk_count: usize,
}
pub trait Int4ExtractionDocumentBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int4ExtractionDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure>;
}

pub struct Int4DocumentExtractionJobProcessor<'p, 'e, 'w, 'v, A: Int4ExtractionDocumentBatchAdmission> {
    planner: &'p Int4DocumentExtractionJobPlanner,
    backend: Backend<'e, 'w>,
    vocabulary: &'v ExtractionVocabulary,
    admission: A,
    ledger: Ledger,
}
impl<'p, 'e, 'w, 'v, A: Int4ExtractionDocumentBatchAdmission> Int4DocumentExtractionJobProcessor<'p, 'e, 'w, 'v, A> {
    pub fn new(planner: &'p Int4DocumentExtractionJobPlanner, engine: &'e mut StrictInt4Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        let backend = Backend::Serial(engine);
        backend.check_capacity(planner).map_err(|e| e.fault)?;
        planner.check_job_limits(planner.limits)?;
        Ok(Self { planner, backend, vocabulary, admission, ledger: Ledger::default() })
    }
    /// The caller retains the declared complete native/decoder peak through
    /// every document, cohort drain and durable acknowledgement. Each row keeps
    /// its original task budget, and the spool retains the original map wire.
    pub fn new_cohorts(planner: &'p Int4DocumentExtractionJobPlanner, weights: &'w Int4ModelWeights,
        vocabulary: &'v ExtractionVocabulary, budget: Int4ExtractionMapCohortBudget,
        kernel: PackedInt4Kernel, admission: A) -> Result<Self, BatchFault> {
        let backend = Backend::Cohorts { weights, budget, kernel };
        backend.check_capacity(planner).map_err(|e| e.fault)?;
        planner.check_job_limits(planner.limits)?;
        Ok(Self { planner, backend, vocabulary, admission, ledger: Ledger::default() })
    }
}
impl<A: Int4ExtractionDocumentBatchAdmission> BatchProcessor for Int4DocumentExtractionJobProcessor<'_, '_, '_, '_, A> {
    type Args = ExtractionDocumentArgs;
    type Prepared = PreparedInt4ExtractionDocument;
    // Preserve the actual native wire object, exact JSON strings and original
    // coordinate evidence. No wrapper consumes unreserved final-result bytes.
    type Output = GuardedOutput<Int4ExtractionMapRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.failed = true;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.ready()?;
        self.ledger.failed = true;
        self.backend.check_capacity(self.planner)?;
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
        // The authenticated job journal already charged the WHOLE attempt.
        // Poison before any fallible callback; recovery requires explicit reopen.
        self.ledger.begin(prepared.work, context, self.planner.limits)?;
        self.planner.owns(&prepared)?;
        poll(control)?;
        let kv = self.backend.check_capacity(self.planner)?;
        let plan = self.planner.plan(&prepared.source, control)?;
        self.planner.compare(&plan, &prepared, control)?;
        self.backend.preflight(self.planner, &plan, &prepared.identities, control)?;
        let (admitted, guard) = self.admission.admit(Int4ExtractionDocumentAdmission {
            identities: &prepared.identities, model_work: prepared.work.int4_model_work(),
            mask_node_visits: prepared.work.mask_node_visits,
            mask_limits: self.planner.document.mapping.mask_limits, kv_reservation_bytes: kv,
            max_result_bytes: self.planner.max_result_bytes(),
            max_output_tokens: self.planner.output_tokens(prepared.chunk_count()).map_err(BatchItemFailure::fatal)?,
            chunk_count: prepared.chunk_count(),
        })?;
        poll(control)?;
        if admitted != prepared.identities { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        // The actual native map authenticates EVERY full schema/source plan,
        // including the final chunk's KV, before the first model forward.
        let output = self.backend.execute(plan, &admitted, self.vocabulary, control)?;
        prepared.expected.verify_completed(&output).map_err(map_failure)?;
        self.backend.check_capacity(self.planner)?;
        poll(control)?;
        self.ledger.failed = false;
        // JobRunner retains this guard through serialization, spool sync and
        // authenticated acknowledgement of the single original document item.
        Ok(GuardedOutput::new(output, guard))
    }
}
impl<A: Int4ExtractionDocumentBatchAdmission> DurableBatchProcessor for Int4DocumentExtractionJobProcessor<'_, '_, '_, '_, A> {
    type Recipe = Int4DocumentExtractionJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { self.planner.job_recipe() }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.planner.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        self.ledger.ready()?;
        self.planner.owns(prepared)?;
        self.backend.check_capacity(self.planner)?;
        if receipt_work(prepared.expected) != prepared.work { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        Ok(prepared.work)
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.planner.max_result_bytes() }
}

/// A physical schedule is fixed by the private job recipe. No caller-supplied
/// driver, result receipt or alternate model can enter the durable lifecycle.
enum Backend<'e, 'w> {
    Serial(&'e mut StrictInt4Engine<'w>),
    Cohorts { weights: &'w Int4ModelWeights, budget: Int4ExtractionMapCohortBudget, kernel: PackedInt4Kernel },
}
impl Backend<'_, '_> {
    fn model_identity(&self) -> &Int4ModelIdentity {
        match self { Self::Serial(engine) => engine.model_identity(), Self::Cohorts { weights, .. } => weights.identity() }
    }
    fn check_capacity(&self, planner: &Int4DocumentExtractionJobPlanner) -> Result<u64, BatchItemFailure> {
        match self {
            Self::Serial(engine) if !planner.uses_chunk_cohorts() => check_capacity(planner, engine),
            Self::Cohorts { weights, budget, .. } if planner.uses_chunk_cohorts() => {
                constrained_int4::check_model(planner.execution_identity(), weights.identity())
                    .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
                if weights.group_width() != planner.group_width() {
                    return Err(BatchItemFailure::fatal(BatchCode::Admission));
                }
                Ok(cohort_budget(planner, *budget)?.native.kv_bytes)
            }
            _ => Err(BatchItemFailure::fatal(BatchCode::Admission)),
        }
    }
    fn preflight<C: DecodeStepControl>(&self, planner: &Int4DocumentExtractionJobPlanner,
        plan: &PreparedInt4ExtractionMap<'_>, identities: &[ExecutionIdentity], control: &mut C)
        -> Result<(), BatchItemFailure> {
        for identity in identities {
            poll(control)?;
            constrained_int4::check_model(identity, self.model_identity())
                .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        }
        if let Self::Cohorts { budget, .. } = self {
            let ceiling = cohort_budget(planner, *budget)?;
            let required = plan.cohort_requirements(budget.context_limit).map_err(map_failure)?;
            check_peak(required, ceiling)?;
        }
        poll(control)
    }
    fn execute<C: DecodeStepControl>(&mut self, plan: PreparedInt4ExtractionMap<'_>,
        admitted: &[ExecutionIdentity], vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int4ExtractionMapRun, BatchItemFailure> {
        match self {
            Self::Serial(engine) => plan.execute_with_control(admitted, engine, vocabulary, control).map_err(map_failure),
            Self::Cohorts { weights, budget, kernel } => plan.execute_cohorts_with_control(
                admitted, weights, vocabulary, *budget, *kernel, control).map(|run| run.document).map_err(map_failure),
        }
    }
}

pub(super) fn cohort_budget(planner: &Int4DocumentExtractionJobPlanner, budget: Int4ExtractionMapCohortBudget)
    -> Result<Int4ExtractionMapCohortRequirement, BatchItemFailure> {
    if !planner.uses_chunk_cohorts() || budget.context_limit != planner.document.mapping.chunks.context_tokens
        || budget.max_result_bytes > crate::batch::extract::grouped_int4::memory::MAX_COHORT_RESULT_BYTES {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let required = planner.cohort_requirements().map_err(BatchItemFailure::fatal)?;
    let supplied = Int4ExtractionMapCohortRequirement { native: Int4MemoryRequirement {
        kv_bytes: budget.native_memory.max_kv_bytes, rope_bytes: budget.native_memory.max_rope_bytes,
        scratch_payload_bound: budget.native_memory.max_scratch_payload_bytes },
        workspace_bytes: budget.max_workspace_bytes, max_result_bytes: budget.max_result_bytes };
    check_peak(required, supplied)?;
    Ok(required)
}
pub(super) fn check_peak(required: Int4ExtractionMapCohortRequirement, ceiling: Int4ExtractionMapCohortRequirement)
    -> Result<(), BatchItemFailure> {
    if required.native.kv_bytes > ceiling.native.kv_bytes || required.native.rope_bytes > ceiling.native.rope_bytes
        || required.native.scratch_payload_bound > ceiling.native.scratch_payload_bound
        || required.workspace_bytes > ceiling.workspace_bytes || required.max_result_bytes > ceiling.max_result_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    Ok(())
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
fn check_capacity(planner: &Int4DocumentExtractionJobPlanner, engine: &StrictInt4Engine<'_>) -> Result<u64, BatchItemFailure> {
    constrained_int4::check_model(planner.execution_identity(), engine.model_identity())
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    let context = planner.document.mapping.chunks.context_tokens;
    if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0)
        || engine.model_identity().group_width() != planner.group_width()
        || engine.kv_cache().capacity_positions() != context {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let required = Int4MemoryRequirement::for_context(context, planner.group_width())
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    if required.kv_bytes == 0 || required.kv_bytes != planner.task_ceiling().max_kv_bytes
        || required.kv_bytes != planner.document_budget().max_kv_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    Ok(required.kv_bytes)
}

#[cfg(test)]
mod tests {
    // UNRUN invocation-ledger fixtures. Persisted nonrefundable debits and
    // authenticated recovery remain owned by the unchanged JobRunner.
    use super::*;
    use crate::native_engine::portable_int8::ProjectionWork;
    fn work(n: u64) -> JobWork {
        JobWork::from_int4(Int4Work { forward_positions: n, projected_logits: n, attention_pairs: n,
            projections: ProjectionWork { dot_products: n, multiply_accumulates: n } }, n)
    }
    fn limits() -> JobLimits {
        JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 4096,
            max_snapshot_bytes: 65_536, max_result_bytes: 65_536, max_spool_bytes: 1 << 20,
            max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 2, max_work: work(2) }
    }
    fn context(sequence: u64) -> BatchRequestContext {
        BatchRequestContext { request_seq: sequence, epoch: 1, input_line: sequence, byte_offset: 0 }
    }
    #[test]
    fn begun_failed_attempt_stays_charged_and_poisoned() {
        let mut ledger = Ledger::default();
        ledger.begin(work(1), context(7), limits()).unwrap();
        assert_eq!(ledger.work, work(1)); assert_eq!(ledger.attempts, 1);
        assert!(ledger.ready().is_err());
        assert!(ledger.begin(work(1), context(8), limits()).is_err());
        assert_eq!(ledger.work, work(1)); assert_eq!(ledger.attempts, 1);
    }
    #[test]
    fn success_does_not_replenish_attempts_or_six_work_axes() {
        let mut ledger = Ledger::default();
        for sequence in [7, 8] {
            ledger.begin(work(1), context(sequence), limits()).unwrap();
            ledger.failed = false; // Only complete native success clears this.
        }
        assert_eq!(ledger.work, work(2)); assert_eq!(ledger.attempts, 2);
        assert_eq!(ledger.begin(work(1), context(9), limits()).err().unwrap().fault.code, BatchCode::WorkLimit);
        assert!(ledger.ready().is_err());
    }
    #[test]
    fn repeated_or_invalid_delivery_context_poison_before_admission() {
        for invalid in [context(0), BatchRequestContext { epoch: 0, ..context(1) },
            BatchRequestContext { input_line: 0, ..context(1) }] {
            let mut ledger = Ledger::default();
            assert!(ledger.begin(work(1), invalid, limits()).is_err()); assert!(ledger.ready().is_err());
        }
        let mut ledger = Ledger::default(); ledger.begin(work(1), context(7), limits()).unwrap();
        ledger.failed = false;
        assert!(ledger.begin(work(1), context(7), limits()).is_err()); assert!(ledger.ready().is_err());
    }
    #[test]
    fn over_budget_and_overflow_are_refused_on_every_work_axis() {
        for field in 0..6 {
            let mut excess = work(1);
            match field {
                0 => excess.model.forward_positions = 3, 1 => excess.model.projected_logits = 3,
                2 => excess.model.attention_pairs = 3, 3 => excess.model.projections.dot_products = 3,
                4 => excess.model.projections.multiply_accumulates = 3, _ => excess.mask_node_visits = 3,
            }
            let mut ledger = Ledger::default();
            assert_eq!(ledger.begin(excess, context(1), limits()).err().unwrap().fault.code, BatchCode::WorkLimit);
            assert!(ledger.ready().is_err());
        }
        let mut ledger = Ledger { work: work(u64::MAX), ..Ledger::default() };
        let mut ceiling = limits(); ceiling.max_work = work(u64::MAX);
        assert_eq!(ledger.begin(work(1), context(1), ceiling).err().unwrap().fault.code, BatchCode::WorkLimit);
        assert!(ledger.ready().is_err());
    }
}
