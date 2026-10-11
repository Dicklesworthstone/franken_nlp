//! Complete-document extraction on the existing ordered NDJSON runner.
//! One resident native driver; no context-sized truncation or per-row runtime.
use super::*;
use crate::batch::{BatchRequestContext, BatchProcessor, BatchWork};

/// The schema, grounding policy and partition limits are fixed for the stream.
/// Each document's exact native work and mask reservation, plus its complete
/// verification allowance, are charged BEFORE any forward and never refunded.
#[derive(Clone)]
pub struct ExtractionDocumentBatchConfig {
    pub request: ExtractionBatchArgs,
    pub limits: Int8ExtractionMapLimits,
    pub max_model_work: Int8Work,
    pub max_mask_visits: u64,
    pub max_verification: GroundingBudget,
}

/// Records cannot replace the admitted schema, profile, chunking or budgets.
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractionDocumentArgs {}

/// Admission sees ALL exact chunk identities, not a representative first chunk
/// or a compiler identity that omits the actual source/schema binding.
pub struct ExtractionDocumentAdmission<'a> {
    pub identities: &'a [ExecutionIdentity],
    pub model_work: Int8Work,
    pub mask_node_visits: u64,
    pub mask_limits: MaskWorkLimits,
    pub kv_reservation_bytes: u64,
    pub max_result_bytes: u64,
}
pub trait ExtractionDocumentBatchAdmission {
    type Guard;
    fn admit(&mut self, request: ExtractionDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure>;
}

/// Owns the original document and a private snapshot of every compiled identity.
/// Rebuilding avoids a self-referential owner; equality is checked before native
/// execution. Neither the source nor this snapshot has a wire constructor.
pub struct PreparedExtractionDocument {
    source: String,
    expected: Int8ExtractionMapPreflight,
    identities: Vec<ExecutionIdentity>,
}

pub struct NativeInt8DocumentExtractionBatch<'p, 'e, 'w, 'v, A: ExtractionDocumentBatchAdmission> {
    planner: &'p Int8ExtractionBatchPlanner,
    config: ExtractionDocumentBatchConfig,
    driver: NativeDriver<'e, 'w, 'v>,
    admission: A,
    ledger: Ledger,
}
impl<'p, 'e, 'w, 'v, A: ExtractionDocumentBatchAdmission>
    NativeInt8DocumentExtractionBatch<'p, 'e, 'w, 'v, A> {
    pub fn new(planner: &'p Int8ExtractionBatchPlanner, config: ExtractionDocumentBatchConfig,
        engine: &'e mut StrictInt8Engine<'w>, vocabulary: &'v ExtractionVocabulary, admission: A)
        -> Result<Self, BatchFault> {
        config.validate()?;
        let driver = NativeDriver { engine, vocabulary };
        capacity(&driver, &config).map_err(|e| e.fault)?;
        Ok(Self { planner, config, driver, admission, ledger: Ledger::default() })
    }
    pub fn reserved_model_work(&self) -> Int8Work { self.ledger.work }
    pub fn reserved_mask_visits(&self) -> u64 { self.ledger.masks }
    pub fn reserved_verification(&self) -> GroundingBudget { self.ledger.verification }
    pub fn is_poisoned(&self) -> bool { self.ledger.failed || !self.driver.clean() }
}
impl<A: ExtractionDocumentBatchAdmission> BatchProcessor
    for NativeInt8DocumentExtractionBatch<'_, '_, '_, '_, A> {
    type Args = ExtractionDocumentArgs;
    type Prepared = PreparedExtractionDocument;
    type Output = GuardedOutput<Int8ExtractionMapRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.failed = true;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        prepare_document(self.planner, &self.config, &self.driver, &mut self.ledger, document, control)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.expected.planned_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.ledger.failed = true;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        execute_document(self.planner, &self.config, &mut self.driver, &mut self.admission,
            &mut self.ledger, prepared, context, control)
    }
}

impl ExtractionDocumentBatchConfig {
    pub(crate) fn validate(&self) -> Result<(), BatchFault> {
        let m = self.limits.mapping;
        let v = self.limits.verification;
        self.request.budget.validate().map_err(|_| BatchCode::InvalidLimits)?;
        m.chunks.effective_token_limit().map_err(|_| BatchCode::InvalidLimits)?;
        if self.request.schema.is_empty() || m.chunks.max_input_bytes == 0 || m.chunks.max_chunks > 256
            || !nonzero(m.max_model_work) || !fits(m.max_model_work, self.max_model_work)
            || m.mask_limits.max_trie_node_visits == 0 || m.mask_limits.checkpoint_interval_nodes == 0
            || m.mask_visits_per_chunk == 0 || m.max_mask_visits < m.mask_visits_per_chunk
            || m.max_mask_visits > self.max_mask_visits
            || m.reduction.max_result_bytes == 0 || m.reduction.max_live_value_bytes == 0
            || !(1..=1_000_000).contains(&v.max_fields) || !(1..=1_000_000).contains(&v.max_matches)
            || v.max_scan_steps == 0 || !verification_fits(v, self.max_verification) {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}
fn nonzero(w: Int8Work) -> bool {
    w.forward_positions != 0 && w.projected_logits != 0 && w.attention_pairs != 0
        && w.projections.dot_products != 0 && w.projections.multiply_accumulates != 0
}
fn verification_fits(a: GroundingBudget, b: GroundingBudget) -> bool {
    a.max_fields <= b.max_fields && a.max_matches <= b.max_matches && a.max_scan_steps <= b.max_scan_steps
}
fn add_verification(a: GroundingBudget, b: GroundingBudget) -> Option<GroundingBudget> {
    Some(GroundingBudget { max_fields: a.max_fields.checked_add(b.max_fields)?,
        max_matches: a.max_matches.checked_add(b.max_matches)?, max_scan_steps: a.max_scan_steps.checked_add(b.max_scan_steps)? })
}
struct Ledger { work: Int8Work, masks: u64, verification: GroundingBudget, sequence: u64, failed: bool }
impl Default for Ledger {
    fn default() -> Self {
        Self { work: Int8Work::default(), masks: 0,
            verification: GroundingBudget { max_fields: 0, max_matches: 0, max_scan_steps: 0 }, sequence: 0, failed: false }
    }
}
impl Ledger {
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn begin(&mut self, config: &ExtractionDocumentBatchConfig, expected: Int8ExtractionMapPreflight,
        context: BatchRequestContext) -> Result<(), BatchItemFailure> {
        self.ready()?;
        self.failed = true; // Unwinding or ANY execution failure closes this adapter.
        if context.request_seq <= self.sequence || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let work = add_work(self.work, expected.planned_work()).filter(|&w| fits(w, config.max_model_work))
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let masks = self.masks.checked_add(expected.reserved_mask_visits()).filter(|&n| n <= config.max_mask_visits)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let verification = add_verification(self.verification, config.limits.verification)
            .filter(|&v| verification_fits(v, config.max_verification))
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        // Atomic across all five native axes, masks and all verification axes.
        self.work = work; self.masks = masks; self.verification = verification; self.sequence = context.request_seq;
        Ok(())
    }
}
fn capacity(driver: &impl Driver, config: &ExtractionDocumentBatchConfig) -> Result<u64, BatchItemFailure> {
    if !driver.clean() || driver.capacity() < config.limits.mapping.chunks.context_tokens {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    (driver.capacity() as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
        .filter(|&n| n != 0 && n <= config.request.budget.max_kv_bytes)
        .ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    step(control).map_err(|e| map_failure(Int8SourceMapError::Source(e), true))
}
fn map_failure(error: Int8SourceMapError, fatal: bool) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    let (code, terminal) = match error {
        Int8SourceMapError::Allocation => (BatchCode::Allocation, true),
        Int8SourceMapError::WorkLimit => (BatchCode::WorkLimit, fatal),
        Int8SourceMapError::EmptySource => (BatchCode::DocumentLimit, fatal),
        Int8SourceMapError::InvalidLimits => (BatchCode::InvalidLimits, true),
        Int8SourceMapError::Admission => (BatchCode::Admission, fatal),
        Int8SourceMapError::Chunk(_) => (BatchCode::Planning, fatal),
        Int8SourceMapError::Source(_) | Int8SourceMapError::Reduction(_) => (BatchCode::InvalidExecution, true),
    };
    if terminal { BatchItemFailure::fatal(code) } else { BatchItemFailure::reject(code) }
}
fn prepare_document<D: Driver, C: DecodeStepControl>(planner: &Int8ExtractionBatchPlanner,
    config: &ExtractionDocumentBatchConfig, driver: &D, ledger: &mut Ledger,
    document: BatchDocument<ExtractionDocumentArgs>, control: &mut C)
    -> Result<PreparedExtractionDocument, BatchItemFailure> {
    ledger.ready()?; ledger.failed = true;
    let result = (|| {
        checkpoint(control)?;
        capacity(driver, config)?;
        if document.text.is_empty() || document.text.len() > config.limits.mapping.chunks.max_input_bytes {
            return Err(BatchItemFailure::reject(BatchCode::DocumentLimit));
        }
        let plan = planner.plan_document_with_control(&document.text, &config.request, config.limits, control)
            .map_err(|e| map_failure(e, false))?;
        let expected = plan.preflight_metadata();
        let mut identities = Vec::new();
        identities.try_reserve_exact(expected.chunk_count()).map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
        for chunk in &plan.plans {
            checkpoint(control)?;
            preflight(chunk, driver)?;
            identities.push(chunk.execution_identity().clone());
        }
        drop(plan);
        checkpoint(control)?;
        Ok(PreparedExtractionDocument { source: document.text, expected, identities })
    })();
    if !result.as_ref().is_err_and(|e: &BatchItemFailure| e.stop) { ledger.failed = false; }
    result
}

// Only this closed helper can compose a test driver with a completed map. The
// public adapter always supplies NativeDriver. No public result-injection hook.
#[allow(clippy::too_many_arguments)]
fn execute_document<D: Driver, A: ExtractionDocumentBatchAdmission, C: DecodeStepControl>(
    planner: &Int8ExtractionBatchPlanner, config: &ExtractionDocumentBatchConfig,
    driver: &mut D, admission: &mut A, ledger: &mut Ledger,
    prepared: PreparedExtractionDocument, context: BatchRequestContext, control: &mut C)
    -> Result<GuardedOutput<Int8ExtractionMapRun, A::Guard>, BatchItemFailure> {
    ledger.begin(config, prepared.expected, context)?;
    let result = (|| {
        checkpoint(control)?;
        let kv = capacity(driver, config)?;
        let plan = planner.plan_document_with_control(&prepared.source, &config.request, config.limits, control)
            .map_err(|e| map_failure(e, true))?;
        if plan.preflight_metadata() != prepared.expected || plan.plans.len() != prepared.identities.len() {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        for (chunk, expected) in plan.plans.iter().zip(&prepared.identities) {
            checkpoint(control)?;
            chunk.verify_identity(expected).map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
            preflight(chunk, driver)?;
        }
        let (admitted, guard) = admission.admit(ExtractionDocumentAdmission {
            identities: &prepared.identities, model_work: prepared.expected.planned_work(),
            mask_node_visits: prepared.expected.reserved_mask_visits(), mask_limits: config.limits.mapping.mask_limits,
            kv_reservation_bytes: kv, max_result_bytes: plan.max_result_bytes(),
        })?;
        checkpoint(control)?;
        // The map preflights EVERY returned identity before its first forward.
        // Borrowing the driver preserves the SAME resident engine across items.
        let output = plan.execute_with_driver(&admitted, BorrowedDriver(driver), control)
            .map_err(|e| map_failure(e, true))?;
        prepared.expected.verify_completed(&output).map_err(|e| map_failure(e, true))?;
        if !driver.clean() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        checkpoint(control)?;
        Ok(GuardedOutput::new(output, guard))
    })();
    // Even a nominally recoverable admission refusal has consumed this item;
    // never keep going after a half-executed document or renew its reservation.
    match result {
        Ok(output) => { ledger.failed = false; Ok(output) }
        Err(mut failure) => { failure.stop = true; Err(failure) }
    }
}
struct BorrowedDriver<'a, D>(&'a mut D);
impl<D: Driver> Driver for BorrowedDriver<'_, D> {
    fn capacity(&self) -> usize { self.0.capacity() }
    fn clean(&self) -> bool { self.0.clean() }
    fn execute<C: DecodeStepControl>(&mut self, prepared: &PreparedInt8BatchExtraction,
        identity: &ExecutionIdentity, budget: Int8JsonBudget, control: &mut C) -> Result<Int8ExtractRun, Int8ExtractError> {
        self.0.execute(prepared, identity, budget, control)
    }
}

#[cfg(test)] mod tests;
