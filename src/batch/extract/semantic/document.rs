//! Resident complete-document object extraction and correlated field reviews.
//!
//! Each record owns its original source and opaque complete-request proof. One
//! immutable schema, extraction compiler and judge serve the population. Review
//! claims depend on genuine extracted values, so every admitted attempt spends
//! its full declared native, mask and grounding ceilings before either stage.
//! Flushes, early EOS and failed attempts never refund that authority.

use std::sync::Arc;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork,
        extract::{ExtractionBatchArgs, ExtractionBatchGrounding, GuardedOutput,
            grouped_int4::long::{Int4ExtractionMapError,
                object::{Int4ObjectExtractionMapError, Int4ObjectExtractionMapLimits,
                    Int4ObjectExtractionPlanner},
                stream::{ExtractionDocumentArgs, Int4ExtractionDocumentBatchConfig}}}},
    execution_identity::ExecutionIdentity,
    grammar::{CompileLimits, mask::MaskWorkLimits},
    native_engine::{constrained_int4,
        decode::{DecodeCancellationKind, DecodeStepControl},
        strict_int4::{Int4MemoryRequirement, Int4Work, StrictInt4Engine, STRICT_INT4_PROFILE}},
    tasks::{ir::TaskBudget,
        extract::{ExtractionVocabulary, object_reconcile::{RootObjectError, RootObjectSchema},
            semantic::{SemanticLimits, grouped_int4::{Int4SemanticEncoding, Int4SemanticOptions,
                document::{Int4SemanticObjectDocumentError, Int4SemanticObjectDocumentPreflight,
                    Int4SemanticObjectDocumentRun, PreparedInt4SemanticObjectDocument}}}},
        judge::{JudgeLimits, grouped_int4::{Int4JudgePlanner,
            faithfulness_map::Int4FaithfulnessMapLimits}}},
    tokenizer::untrusted::{SafeBpeWork, UntrustedEncoding},
    validation::grounded_fields::GroundingBudget,
};

/// Invocation-wide ceilings. The immutable semantic options and extraction
/// map supply the corresponding complete per-document attempt ceilings.
#[derive(Clone, Copy, Debug)]
pub struct Int4SemanticObjectDocumentBatchLimits {
    pub max_model_work: Int4Work,
    pub max_mask_node_visits: u64,
    pub max_verification: GroundingBudget,
}
impl Int4SemanticObjectDocumentBatchLimits {
    pub fn validate(self) -> Result<(), BatchFault> {
        if !nonzero(self.max_model_work) || self.max_mask_node_visits == 0
            || self.max_verification.max_fields == 0 || self.max_verification.max_matches == 0
            || self.max_verification.max_scan_steps == 0 {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
    pub fn validate_item(self, work: Int4Work, masks: u64, verification: GroundingBudget)
        -> Result<(), BatchFault> {
        self.validate()?;
        if !nonzero(work) || !fits(work, self.max_model_work)
            || masks == 0 || masks > self.max_mask_node_visits
            || verification.max_fields == 0 || verification.max_matches == 0
            || verification.max_scan_steps == 0 || !verification_fits(verification, self.max_verification) {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
    pub fn validate_document(self, work: Int4Work, extraction: Int4ObjectExtractionMapLimits)
        -> Result<(), BatchFault> {
        self.validate_item(work, extraction.mapping.mapping.max_mask_visits, extraction.mapping.verification)
    }
}

/// Fixed original-schema reconciliation and complete-original-source review.
/// No record can replace either task's schema, policy, limits or tokenization.
pub struct Int4SemanticObjectDocumentBatchPlanner {
    extraction: Int4ObjectExtractionPlanner,
    request: ExtractionBatchArgs,
    judge: Arc<Int4JudgePlanner>,
    options: Int4SemanticOptions,
    extraction_limits: Int4ObjectExtractionMapLimits,
    review_limits: Int4FaithfulnessMapLimits,
    limits: Int4SemanticObjectDocumentBatchLimits,
    seal: Arc<()>,
}
impl Int4SemanticObjectDocumentBatchPlanner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(extraction: Int4ObjectExtractionPlanner, request: ExtractionBatchArgs,
        judge: Arc<Int4JudgePlanner>, options: Int4SemanticOptions,
        extraction_limits: Int4ObjectExtractionMapLimits, review_limits: Int4FaithfulnessMapLimits,
        limits: Int4SemanticObjectDocumentBatchLimits) -> Result<Self, BatchFault> {
        options.validate().map_err(|_| BatchCode::InvalidLimits)?;
        limits.validate_document(options.max_model_work, extraction_limits)?;
        extraction_limits.reconciliation_workspace_bytes().map_err(|_| BatchCode::InvalidLimits)?;
        review_limits.validate(options.specification.judge_budget).map_err(|_| BatchCode::InvalidLimits)?;
        // Reuse the map's full fixed-configuration validation without making a
        // second retained copy of the caller's original schema allocation.
        let config = Int4ExtractionDocumentBatchConfig {
            request, limits: extraction_limits.mapping,
            max_model_work: extraction_limits.mapping.mapping.max_model_work,
            max_mask_visits: limits.max_mask_node_visits, max_verification: limits.max_verification,
        };
        config.validate()?;
        let request = config.request;
        let m = extraction_limits.mapping.mapping;
        if extraction.selected_rows().is_some() || m.reduction.map_batch_chunks != 1
            || extraction.group_width() != judge.group_width()
            || extraction.document_encoding() != judge.document_encoding()
            || extraction.base_execution_identity().task_spec != "extract-v1"
            || judge.execution_identity().task_spec != "judge-v1"
            || m.reduction.max_result_bytes as u64 > options.limits.max_output_bytes
            || extraction_limits.reconciliation.max_json_bytes > options.limits.max_input_json_bytes
            || review_limits.max_result_bytes > options.limits.max_output_bytes
            || review_limits.chunks.context_tokens > m.chunks.context_tokens
            || m.chunks.max_input_bytes.checked_add(options.limits.max_claim_bytes)
                .is_none_or(|bytes| bytes > review_limits.chunks.max_input_bytes) {
            return Err(BatchCode::InvalidLimits.into());
        }
        let schema = RootObjectSchema::derive(&request.schema, extraction.compiler_limits(),
            extraction_limits.reconciliation, &mut Continue).map_err(|_| BatchCode::Planning)?;
        if schema.requires_source() != (request.grounding == ExtractionBatchGrounding::SourceMembership) {
            return Err(BatchCode::Planning.into());
        }
        options.validate_schema(schema.original_schema()).map_err(|_| BatchCode::Planning)?;
        Ok(Self { extraction, request, judge, options, extraction_limits, review_limits,
            limits, seal: Arc::new(()) })
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { self.extraction.base_execution_identity() }
    pub fn judge_identity(&self) -> &ExecutionIdentity { self.judge.execution_identity() }
    pub fn group_width(&self) -> usize { self.extraction.group_width() }
    pub fn request(&self) -> &ExtractionBatchArgs { &self.request }
    pub fn extraction_budget(&self) -> TaskBudget { self.request.budget }
    pub fn judge_budget(&self) -> TaskBudget { self.options.specification.judge_budget }
    pub fn judge_planning_limits(&self) -> JudgeLimits { self.judge.semantic_configuration().1 }
    pub fn compiler_limits(&self) -> CompileLimits { self.extraction.compiler_limits() }
    pub fn source_bytes_limit(&self) -> usize { self.extraction_limits.mapping.mapping.chunks.max_input_bytes }
    pub fn extraction_limits(&self) -> Int4ObjectExtractionMapLimits { self.extraction_limits }
    pub fn review_limits(&self) -> Int4FaithfulnessMapLimits { self.review_limits }
    pub fn semantic_limits(&self) -> SemanticLimits { self.options.limits }
    pub fn options(&self) -> &Int4SemanticOptions { &self.options }
    pub fn limits(&self) -> Int4SemanticObjectDocumentBatchLimits { self.limits }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.extraction.document_encoding() }
    pub fn max_result_bytes(&self) -> u64 { self.options.limits.max_output_bytes }
    pub fn retained_defaults_bytes(&self) -> Result<u64, BatchFault> {
        u64::try_from(self.request.schema.capacity()).ok()
            .and_then(|n| n.checked_add(std::mem::size_of::<ExtractionBatchArgs>() as u64))
            .ok_or_else(|| BatchCode::InvalidLimits.into())
    }

    fn plan<'s, C: DecodeStepControl>(&self, source: &'s str, control: &mut C, fatal: bool)
        -> Result<PreparedInt4SemanticObjectDocument<'s>, BatchItemFailure> {
        checkpoint(control)?;
        if source.is_empty() || source.len() > self.source_bytes_limit() {
            return Err(refusal(BatchCode::DocumentLimit, fatal));
        }
        let extraction = self.extraction.plan_document_with_control(source, &self.request,
            self.extraction_limits, control).map_err(|e| document_failure(e.into(), fatal))?;
        PreparedInt4SemanticObjectDocument::new(extraction, Arc::clone(&self.judge),
            self.options.clone(), self.review_limits, control).map_err(|e| document_failure(e, fatal))
    }
    fn prepare<C: DecodeStepControl>(&self, document: BatchDocument<ExtractionDocumentArgs>, control: &mut C)
        -> Result<PreparedInt4SemanticObjectBatchDocument, BatchItemFailure> {
        checkpoint(control)?;
        if document.task_args.is_some() { return Err(BatchItemFailure::reject(BatchCode::InvalidEnvelope)); }
        let plan = self.plan(&document.text, control, false)?;
        let expected = plan.preflight_metadata();
        let mut identities = Vec::new();
        identities.try_reserve_exact(plan.execution_identities().len())
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
        for identity in plan.execution_identities() { checkpoint(control)?; identities.push(identity.clone()); }
        let extraction = plan.extraction_preflight();
        let encoding_work = plan.input_encoding_work().unwrap_or_default();
        if identities.is_empty() || identities.len() > self.extraction_limits.mapping.mapping.chunks.max_chunks
            || plan.group_width() != self.group_width() || plan.source_bytes() != document.text.len()
            || plan.extraction_budget() != self.extraction_budget() || plan.judge_budget() != self.judge_budget()
            || plan.input_encoding() != self.document_encoding()
            || plan.max_model_work() != self.options.max_model_work
            || plan.max_result_bytes() != self.max_result_bytes() {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        drop(plan);
        checkpoint(control)?;
        Ok(PreparedInt4SemanticObjectBatchDocument { source: document.text, expected, identities, encoding_work,
            extraction_work: extraction.planned_work(),
            model_work: self.options.max_model_work, max_result_bytes: self.max_result_bytes(),
            mask_node_visits: self.extraction_limits.mapping.mapping.max_mask_visits,
            verification: self.extraction_limits.mapping.verification, seal: Arc::clone(&self.seal) })
    }
    fn verify_prepared(&self, prepared: &PreparedInt4SemanticObjectBatchDocument)
        -> Result<(), BatchItemFailure> {
        if !Arc::ptr_eq(&self.seal, &prepared.seal) {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        Ok(())
    }
    fn compare<C: DecodeStepControl>(&self, prepared: &PreparedInt4SemanticObjectBatchDocument,
        plan: &PreparedInt4SemanticObjectDocument<'_>, control: &mut C) -> Result<(), BatchItemFailure> {
        self.verify_prepared(prepared)?;
        checkpoint(control)?;
        if plan.preflight_metadata() != prepared.expected
            || plan.execution_identities().len() != prepared.identities.len()
            || plan.max_model_work() != prepared.model_work()
            || plan.max_result_bytes() != prepared.max_result_bytes()
            || plan.input_encoding_work().unwrap_or_default() != prepared.encoding_work {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        for (actual, frozen) in plan.execution_identities().zip(&prepared.identities) {
            checkpoint(control)?;
            if actual != frozen { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        }
        Ok(())
    }
    fn admit_reconstruction(&self, prepared: &PreparedInt4SemanticObjectBatchDocument)
        -> Result<(), BatchItemFailure> {
        // Exact reconstruction repeats the deterministic probes and schemas.
        // Refuse before repeating them if that known cost alone cannot fit.
        let combined = prepared.encoding_work.checked_add(prepared.encoding_work)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        Int4SemanticEncoding::from_work(self.document_encoding(), combined)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        Ok(())
    }
}

/// Private source and exact request proof. Counts alone do not authorize a
/// different source, schema, review policy or immutable resident factory.
pub struct PreparedInt4SemanticObjectBatchDocument {
    source: String,
    expected: Int4SemanticObjectDocumentPreflight,
    identities: Vec<ExecutionIdentity>,
    encoding_work: SafeBpeWork,
    extraction_work: Int4Work,
    model_work: Int4Work,
    max_result_bytes: u64,
    mask_node_visits: u64,
    verification: GroundingBudget,
    seal: Arc<()>,
}
impl PreparedInt4SemanticObjectBatchDocument {
    pub fn model_work(&self) -> Int4Work { self.model_work }
    pub fn max_result_bytes(&self) -> u64 { self.max_result_bytes }
    pub fn mask_node_visits(&self) -> u64 { self.mask_node_visits }
    pub fn verification_budget(&self) -> GroundingBudget { self.verification }
    pub fn chunk_count(&self) -> usize { self.identities.len() }
    pub fn execution_identities(&self) -> &[ExecutionIdentity] { &self.identities }
}

/// Whole attempted document, including all dynamically determined field maps.
pub struct Int4SemanticObjectDocumentAdmission<'a> {
    pub identities: &'a [ExecutionIdentity],
    pub judge_identity: &'a ExecutionIdentity,
    pub group_width: usize,
    pub model_work: Int4Work,
    pub extraction_reserved_work: Int4Work,
    pub mask_node_visits: u64,
    pub mask_limits: MaskWorkLimits,
    pub verification_budget: GroundingBudget,
    pub kv_reservation_bytes: u64,
    pub max_result_bytes: u64,
    pub max_output_tokens: u64,
    pub chunk_count: usize,
    pub source_bytes: usize,
    pub extraction_budget: TaskBudget,
    pub judge_budget: TaskBudget,
    pub extraction_limits: Int4ObjectExtractionMapLimits,
    pub review_limits: Int4FaithfulnessMapLimits,
    pub semantic_limits: SemanticLimits,
    pub judge_planning_limits: JudgeLimits,
    pub input_encoding: &'a UntrustedEncoding,
}
pub trait Int4SemanticObjectDocumentBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int4SemanticObjectDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure>;
}

/// The native engine may carry the host's one persistent projection executor.
/// This processor never creates a worker or changes an admitted task schedule.
pub struct NativeInt4SemanticObjectDocumentBatch<'e, 'w, 'v, A: Int4SemanticObjectDocumentBatchAdmission> {
    compiler: Arc<Int4SemanticObjectDocumentBatchPlanner>,
    engine: &'e mut StrictInt4Engine<'w>,
    vocabulary: &'v ExtractionVocabulary,
    admission: A,
    ledger: Ledger,
}
impl<'e, 'w, 'v, A: Int4SemanticObjectDocumentBatchAdmission>
    NativeInt4SemanticObjectDocumentBatch<'e, 'w, 'v, A> {
    pub fn new(compiler: Int4SemanticObjectDocumentBatchPlanner, engine: &'e mut StrictInt4Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        Self::new_shared(Arc::new(compiler), engine, vocabulary, admission)
    }
    pub fn new_shared(compiler: Arc<Int4SemanticObjectDocumentBatchPlanner>, engine: &'e mut StrictInt4Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        compiler.limits.validate_document(compiler.options.max_model_work, compiler.extraction_limits)?;
        capacity(&compiler, engine).map_err(|failure| failure.fault)?;
        Ok(Self { compiler, engine, vocabulary, admission, ledger: Ledger::default() })
    }
    pub fn reserved_model_work(&self) -> Int4Work { self.ledger.work }
    pub fn reserved_mask_visits(&self) -> u64 { self.ledger.masks }
    pub fn reserved_verification(&self) -> GroundingBudget { self.ledger.verification }
    pub fn is_poisoned(&self) -> bool { self.ledger.failed || !clean(self.engine) }
    /// Durable processing checks the actual factory before charging an
    /// authenticated attempt or accepting a prepared source from another run.
    pub(crate) fn verify_prepared(&self, prepared: &PreparedInt4SemanticObjectBatchDocument)
        -> Result<(), BatchItemFailure> {
        self.ledger.ready()?;
        capacity(&self.compiler, self.engine)?;
        self.compiler.verify_prepared(prepared)
    }
}
impl<A: Int4SemanticObjectDocumentBatchAdmission> BatchProcessor
    for NativeInt4SemanticObjectDocumentBatch<'_, '_, '_, A> {
    type Args = ExtractionDocumentArgs;
    type Prepared = PreparedInt4SemanticObjectBatchDocument;
    type Output = GuardedOutput<Int4SemanticObjectDocumentRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.ready()?; self.ledger.failed = true;
        let result = (|| {
            checkpoint(control)?;
            capacity(&self.compiler, self.engine)?;
            let prepared = self.compiler.prepare(document, control)?;
            self.ledger.totals(&self.compiler)?;
            checkpoint(control)?;
            Ok(prepared)
        })();
        if clean(self.engine) && !result.as_ref().is_err_and(|e: &BatchItemFailure| e.stop) {
            self.ledger.failed = false;
        }
        result
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.model_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        self.ledger.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        if let Err(failure) = self.verify_prepared(&prepared) {
            self.ledger.failed = true;
            return Err(failure);
        }
        // Reserve every complete-attempt axis atomically before reconstructing
        // source programs, calling admission or performing native inference.
        self.ledger.begin(&self.compiler, context)?;
        let result = (|| {
            checkpoint(control)?;
            let kv = capacity(&self.compiler, self.engine)?;
            self.compiler.admit_reconstruction(&prepared)?;
            let plan = self.compiler.plan(&prepared.source, control, true)?;
            self.compiler.compare(&prepared, &plan, control)?;
            let plan = plan.with_prior_preparation_encoding_work(prepared.encoding_work)
                .map_err(|e| document_failure(e, true))?;
            let expected = plan.preflight_metadata();
            plan.preflight_with_control(&prepared.identities, self.engine, control)
                .map_err(|e| document_failure(e, true))?;
            let tokens = (prepared.chunk_count() as u64)
                .checked_mul(u64::from(self.compiler.extraction_budget().max_output_tokens))
                .ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))?;
            let (admitted, guard) = self.admission.admit(Int4SemanticObjectDocumentAdmission {
                identities: &prepared.identities, judge_identity: self.compiler.judge_identity(),
                group_width: self.compiler.group_width(), model_work: prepared.model_work(),
                extraction_reserved_work: prepared.extraction_work,
                mask_node_visits: prepared.mask_node_visits(),
                mask_limits: self.compiler.extraction_limits.mapping.mapping.mask_limits,
                verification_budget: prepared.verification_budget(), kv_reservation_bytes: kv,
                max_result_bytes: prepared.max_result_bytes(), max_output_tokens: tokens,
                chunk_count: prepared.chunk_count(), source_bytes: prepared.source.len(),
                extraction_budget: self.compiler.extraction_budget(), judge_budget: self.compiler.judge_budget(),
                extraction_limits: self.compiler.extraction_limits, review_limits: self.compiler.review_limits,
                semantic_limits: self.compiler.semantic_limits(), judge_planning_limits: self.compiler.judge_planning_limits(),
                input_encoding: self.compiler.document_encoding(),
            })?;
            checkpoint(control)?;
            if admitted != prepared.identities { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
            let output = plan.execute_with_control(&admitted, self.engine, self.vocabulary, control)
                .map_err(|e| document_failure(e, true))?;
            expected.verify_completed_with_control(&output, control).map_err(|e| document_failure(e, true))?;
            capacity(&self.compiler, self.engine)?;
            checkpoint(control)?;
            Ok(GuardedOutput::new(output, guard))
        })();
        match result {
            Ok(output) => { self.ledger.failed = false; Ok(output) }
            Err(mut failure) => { failure.stop = true; Err(failure) }
        }
    }
}

struct Ledger { work: Int4Work, masks: u64, verification: GroundingBudget, sequence: u64, failed: bool }
impl Default for Ledger {
    fn default() -> Self {
        Self { work: Int4Work::default(), masks: 0, sequence: 0, failed: false,
            verification: GroundingBudget { max_fields: 0, max_matches: 0, max_scan_steps: 0 } }
    }
}
impl Ledger {
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn totals(&self, compiler: &Int4SemanticObjectDocumentBatchPlanner)
        -> Result<(Int4Work, u64, GroundingBudget), BatchItemFailure> {
        let limit = || BatchItemFailure::fatal(BatchCode::WorkLimit);
        let work = self.work.checked_add(compiler.options.max_model_work).ok()
            .filter(|&work| fits(work, compiler.limits.max_model_work)).ok_or_else(limit)?;
        let masks = self.masks.checked_add(compiler.extraction_limits.mapping.mapping.max_mask_visits)
            .filter(|&masks| masks <= compiler.limits.max_mask_node_visits).ok_or_else(limit)?;
        let verification = add_verification(self.verification, compiler.extraction_limits.mapping.verification)
            .filter(|&used| verification_fits(used, compiler.limits.max_verification)).ok_or_else(limit)?;
        Ok((work, masks, verification))
    }
    fn begin(&mut self, compiler: &Int4SemanticObjectDocumentBatchPlanner, context: BatchRequestContext)
        -> Result<(), BatchItemFailure> {
        self.ready()?; self.failed = true;
        if context.request_seq <= self.sequence || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let (work, masks, verification) = self.totals(compiler)?;
        self.work = work; self.masks = masks; self.verification = verification; self.sequence = context.request_seq;
        Ok(())
    }
}
fn capacity(compiler: &Int4SemanticObjectDocumentBatchPlanner, engine: &StrictInt4Engine<'_>)
    -> Result<u64, BatchItemFailure> {
    let failure = || BatchItemFailure::fatal(BatchCode::Admission);
    if !clean(engine) || engine.profile() != STRICT_INT4_PROFILE
        || engine.model_identity().group_width() != compiler.group_width()
        || engine.kv_cache().capacity_positions() < compiler.extraction_limits.mapping.mapping.chunks.context_tokens
        || engine.kv_cache().capacity_positions() < compiler.review_limits.chunks.context_tokens {
        return Err(failure());
    }
    for identity in [compiler.base_execution_identity(), compiler.judge_identity()] {
        constrained_int4::check_profile(identity, compiler.group_width()).map_err(|_| failure())?;
        constrained_int4::check_model(identity, engine.model_identity()).map_err(|_| failure())?;
    }
    let memory = Int4MemoryRequirement::for_context(engine.kv_cache().capacity_positions(), compiler.group_width())
        .map_err(|_| failure())?;
    if memory.kv_bytes > compiler.extraction_budget().max_kv_bytes
        || memory.kv_bytes > compiler.judge_budget().max_kv_bytes { return Err(failure()); }
    Ok(memory.kv_bytes)
}
fn clean(engine: &StrictInt4Engine<'_>) -> bool {
    !engine.is_poisoned() && engine.kv_cache().all_slots_have_len(0)
}
fn fits(work: Int4Work, ceiling: Int4Work) -> bool { super::fits(work, ceiling) }
fn nonzero(work: Int4Work) -> bool { super::nonzero(work) }
fn verification_fits(a: GroundingBudget, b: GroundingBudget) -> bool {
    a.max_fields <= b.max_fields && a.max_matches <= b.max_matches && a.max_scan_steps <= b.max_scan_steps
}
fn add_verification(a: GroundingBudget, b: GroundingBudget) -> Option<GroundingBudget> {
    Some(GroundingBudget { max_fields: a.max_fields.checked_add(b.max_fields)?,
        max_matches: a.max_matches.checked_add(b.max_matches)?, max_scan_steps: a.max_scan_steps.checked_add(b.max_scan_steps)? })
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn refusal(code: BatchCode, fatal: bool) -> BatchItemFailure {
    if fatal { BatchItemFailure::fatal(code) } else { BatchItemFailure::reject(code) }
}
fn document_failure(error: Int4SemanticObjectDocumentError, fatal: bool) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    use Int4SemanticObjectDocumentError as E;
    let (code, fatal) = match error {
        E::Contract => (BatchCode::Planning, fatal),
        E::WorkBudget | E::OutputBudget => (BatchCode::WorkLimit, fatal),
        E::Extraction(Int4ObjectExtractionMapError::Map(Int4ExtractionMapError::EmptySource)) =>
            (BatchCode::DocumentLimit, fatal),
        E::Extraction(Int4ObjectExtractionMapError::Map(Int4ExtractionMapError::WorkLimit)) =>
            (BatchCode::WorkLimit, fatal),
        E::Extraction(Int4ObjectExtractionMapError::Map(Int4ExtractionMapError::Allocation))
            | E::Extraction(Int4ObjectExtractionMapError::Reconciliation(RootObjectError::Allocation)) =>
            (BatchCode::Allocation, true),
        E::Accounting | E::Serialization => (BatchCode::InvalidExecution, true),
        E::Extraction(_) | E::Semantic(_) | E::JudgeMap(_) => (BatchCode::Planning, fatal),
        E::Cancelled(_) => (BatchCode::InvalidExecution, true),
    };
    refusal(code, fatal)
}

#[cfg(test)] mod tests;
