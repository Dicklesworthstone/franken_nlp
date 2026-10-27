//! One real resident engine; the complete result guard survives durable sync.
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
    engine: &'e mut StrictInt4Engine<'w>,
    vocabulary: &'v ExtractionVocabulary,
    admission: A,
    ledger: Ledger,
}
impl<'p, 'e, 'w, 'v, A: Int4ExtractionDocumentBatchAdmission> Int4DocumentExtractionJobProcessor<'p, 'e, 'w, 'v, A> {
    pub fn new(planner: &'p Int4DocumentExtractionJobPlanner, engine: &'e mut StrictInt4Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        check_capacity(planner, engine).map_err(|e| e.fault)?;
        planner.check_job_limits(planner.limits)?;
        Ok(Self { planner, engine, vocabulary, admission, ledger: Ledger::default() })
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
        check_capacity(self.planner, self.engine)?;
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
        let kv = check_capacity(self.planner, self.engine)?;
        let plan = self.planner.plan(&prepared.source, control)?;
        self.planner.compare(&plan, &prepared, control)?;
        for identity in &prepared.identities {
            poll(control)?;
            constrained_int4::check_model(identity, self.engine.model_identity())
                .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        }
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
        let output = plan.execute_with_control(&admitted, self.engine, self.vocabulary, control).map_err(map_failure)?;
        prepared.expected.verify_completed(&output).map_err(map_failure)?;
        check_capacity(self.planner, self.engine)?;
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
        check_capacity(self.planner, self.engine)?;
        if receipt_work(prepared.expected) != prepared.work { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
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
