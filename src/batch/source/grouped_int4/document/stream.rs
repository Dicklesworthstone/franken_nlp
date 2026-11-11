//! Live prepare/debit/execute/verify lifecycle with no storage dependency.
use super::*;
use super::backend::DocumentEngine;

/// Owned complete original source and sealed reconstruction receipt. The
/// owning factory alone can execute it, including its fixed scheduling policy.
pub struct PreparedInt4SourceDocumentBatch {
    source: String,
    identities: Vec<ExecutionIdentity>,
    expected: planning::Expected,
    work: DocumentWork,
    owner: Arc<()>,
    operation: Arc<SourceTaskPlanner>,
}
impl PreparedInt4SourceDocumentBatch {
    pub fn chunk_count(&self) -> usize { self.expected.chunk_count() }
    pub fn planned_work(&self) -> Int4Work { self.work.model }
    pub fn reserved_mask_visits(&self) -> u64 { self.work.mask_node_visits }
    pub fn execution_identities(&self) -> &[ExecutionIdentity] { &self.identities }
    pub fn map_document_encoding_work(&self) -> Result<Option<crate::tokenizer::untrusted::SafeBpeWork>, BatchFault> {
        self.operation.map_document_encoding_work().map_err(|_| BatchCode::Admission.into())
    }
}
impl Int4SourceDocumentBatchPlanner {
    pub fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<SourceDocumentArgs>,
        control: &mut C) -> Result<PreparedInt4SourceDocumentBatch, BatchItemFailure> {
        self.prepare_inner(document, None, control)
    }
    fn prepare_inner<C: DecodeStepControl>(&self, document: BatchDocument<SourceDocumentArgs>,
        engine: Option<&DocumentEngine<'_, '_>>, control: &mut C)
        -> Result<PreparedInt4SourceDocumentBatch, BatchItemFailure> {
        poll(control)?;
        if document.text.is_empty() || document.text.len() > self.mapping.chunks.max_input_bytes {
            return Err(BatchItemFailure::reject(BatchCode::DocumentLimit));
        }
        let operation = encoding::operation(&self.planner)?;
        let mut config = self.configuration(); config.planner = &operation;
        let plan = config.plan(&document.text, control)?;
        let expected = plan.expected();
        let identities = plan.identities(control)?;
        if identities.len() != expected.native_chunks() { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        let work = expected.work()?;
        if let Some(engine) = engine { engine.preflight(&plan, &identities, control)?; }
        drop(plan);
        poll(control)?;
        Ok(PreparedInt4SourceDocumentBatch { source: document.text, identities, expected, work,
            owner: Arc::clone(&self.owner), operation })
    }
    fn owns(&self, prepared: &PreparedInt4SourceDocumentBatch) -> Result<(), BatchItemFailure> {
        if !Arc::ptr_eq(&self.owner, &prepared.owner) { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        encoding::verify_operation(&self.planner, &prepared.operation)
    }
}

pub struct NativeInt4SourceDocumentBatch<'p, 'e, 'w, 'v, A: Int4SourceDocumentBatchAdmission> {
    planner: &'p Int4SourceDocumentBatchPlanner,
    engine: DocumentEngine<'e, 'w>,
    vocabulary: &'v ExtractionVocabulary,
    admission: A,
    ledger: Ledger,
}
impl<'p, 'e, 'w, 'v, A: Int4SourceDocumentBatchAdmission> NativeInt4SourceDocumentBatch<'p, 'e, 'w, 'v, A> {
    pub fn new(planner: &'p Int4SourceDocumentBatchPlanner, engine: &'e mut StrictInt4Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        Self::configured(planner, DocumentEngine::Serial(engine), vocabulary, admission)
    }
    pub fn new_cohorts(planner: &'p Int4SourceDocumentBatchPlanner, weights: &'w Int4ModelWeights,
        vocabulary: &'v ExtractionVocabulary, budget: Int4SourceMapCohortBudget,
        kernel: PackedInt4Kernel, admission: A) -> Result<Self, BatchFault> {
        let required = planner.cohort_requirements()?;
        let hierarchy = planner.hierarchy_cohort_schedule();
        Self::configured(planner, DocumentEngine::Cohorts { weights, budget, kernel, required, hierarchy }, vocabulary, admission)
    }
    fn configured(planner: &'p Int4SourceDocumentBatchPlanner, engine: DocumentEngine<'e, 'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        engine.check_capacity(&planner.configuration()).map_err(|e| e.fault)?;
        Ok(Self { planner, engine, vocabulary, admission, ledger: Ledger::default() })
    }
    pub fn reserved_model_work(&self) -> Int4Work { self.ledger.work }
    pub fn reserved_mask_visits(&self) -> u64 { self.ledger.masks }
    pub fn reserved_verification(&self) -> GroundingBudget { self.ledger.verification }
    pub fn reserved_summary_scan_steps(&self) -> u64 { self.ledger.summary_scans }
    pub fn reserved_keyphrase_scan_work(&self) -> u64 { self.ledger.keyphrase_scans }
    pub fn is_poisoned(&self) -> bool { self.ledger.failed || self.engine.check_capacity(&self.planner.configuration()).is_err() }
}
impl<A: Int4SourceDocumentBatchAdmission> BatchProcessor for NativeInt4SourceDocumentBatch<'_, '_, '_, '_, A> {
    type Args = SourceDocumentArgs;
    type Prepared = PreparedInt4SourceDocumentBatch;
    type Output = GuardedOutput<Int4SourceDocumentRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.ready()?;
        self.ledger.failed = true;
        let result = (|| {
            self.engine.check_capacity(&self.planner.configuration())?;
            self.planner.prepare_inner(document, Some(&self.engine), control)
        })();
        if !result.as_ref().is_err_and(|e: &BatchItemFailure| e.stop) { self.ledger.failed = false; }
        result
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        BatchWork { forward_positions: prepared.work.model.forward_positions,
            projected_logits: prepared.work.model.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.ledger.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        if let Err(failure) = self.planner.owns(&prepared) {
            self.ledger.failed = true;
            return Err(failure);
        }
        self.ledger.begin(prepared.work, self.planner.task.verification(self.planner.mapping),
            self.planner.task.summary_scan_steps(), self.planner.task.keyphrase_scan_work(),
            self.planner.run_limits, context)?;
        let result = (|| {
            poll(control)?;
            let mut config = self.planner.configuration(); config.planner = &prepared.operation;
            let kv = self.engine.check_capacity(&config)?;
            let plan = config.plan(&prepared.source, control)?;
            plan.compare_receipt(prepared.expected, prepared.work, &prepared.identities, control)?;
            self.engine.preflight(&plan, &prepared.identities, control)?;
            let (admitted, guard) = self.admission.admit(Int4SourceDocumentAdmission {
                identities: &prepared.identities, model_work: prepared.work.model,
                mask_node_visits: prepared.work.mask_node_visits, mask_limits: config.mapping.mask_limits,
                kv_reservation_bytes: kv, max_result_bytes: self.planner.max_result_bytes(),
                max_output_tokens: self.planner.task.output_tokens(prepared.chunk_count(), config.budget).map_err(BatchItemFailure::fatal)?,
                chunk_count: prepared.chunk_count(),
            })?;
            poll(control)?;
            if admitted != prepared.identities { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
            self.engine.preflight(&plan, &admitted, control)?;
            let output = self.engine.execute(plan, &admitted, config.planner, self.vocabulary, control)?;
            prepared.expected.verify(&output, config.task)?;
            self.engine.check_capacity(&config)?;
            poll(control)?;
            Ok(GuardedOutput::new(output, guard))
        })();
        match result {
            Ok(output) => { self.ledger.failed = false; Ok(output) }
            Err(mut failure) => { failure.stop = true; Err(failure) }
        }
    }
}

struct Ledger {
    work: Int4Work, masks: u64, verification: GroundingBudget,
    summary_scans: u64, keyphrase_scans: u64, sequence: u64, failed: bool,
}
impl Default for Ledger {
    fn default() -> Self {
        Self { work: Int4Work::default(), masks: 0,
            verification: GroundingBudget { max_fields: 0, max_matches: 0, max_scan_steps: 0 },
            summary_scans: 0, keyphrase_scans: 0, sequence: 0, failed: false }
    }
}
impl Ledger {
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn begin(&mut self, next: DocumentWork, per: GroundingBudget, summary: u64, keyphrases: u64,
        limits: Int4SourceDocumentBatchLimits, context: BatchRequestContext) -> Result<(), BatchItemFailure> {
        self.ready()?;
        self.failed = true;
        if context.request_seq <= self.sequence || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let exceeded = || BatchItemFailure::fatal(BatchCode::WorkLimit);
        let work = self.work.checked_add(next.model).ok().filter(|&w| fits(w, limits.max_model_work)).ok_or_else(exceeded)?;
        let masks = self.masks.checked_add(next.mask_node_visits).filter(|&n| n <= limits.max_mask_visits).ok_or_else(exceeded)?;
        let verification = GroundingBudget {
            max_fields: self.verification.max_fields.checked_add(per.max_fields).ok_or_else(exceeded)?,
            max_matches: self.verification.max_matches.checked_add(per.max_matches).ok_or_else(exceeded)?,
            max_scan_steps: self.verification.max_scan_steps.checked_add(per.max_scan_steps).ok_or_else(exceeded)?,
        };
        if !verification_fits(verification, limits.max_verification) { return Err(exceeded()); }
        let summary_scans = self.summary_scans.checked_add(summary).filter(|&n| n <= limits.max_summary_scan_steps).ok_or_else(exceeded)?;
        let keyphrase_scans = self.keyphrase_scans.checked_add(keyphrases).filter(|&n| n <= limits.max_keyphrase_scan_work).ok_or_else(exceeded)?;
        // Every quantity commits together. Failure, cancellation, a writer
        // refusal and a new flush epoch never refund any attempted document.
        self.work = work; self.masks = masks; self.verification = verification;
        self.summary_scans = summary_scans; self.keyphrase_scans = keyphrase_scans; self.sequence = context.request_seq;
        Ok(())
    }
}

#[cfg(test)] mod tests;
