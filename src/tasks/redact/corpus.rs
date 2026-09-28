//! Long-document redaction on the existing ordered, bounded NDJSON runner.
//! One native engine, fixed detector/key scope and nonrenewable corpus ledger.
//! No per-row host invocation, partial redacted text, or native transcript.
use serde::Serialize;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, generation::GuardedOutput},
    canonjson, execution_identity::ExecutionIdentity,
    native_engine::{decode::DecodeStepControl, kv::KV_BYTES_PER_TOKEN,
        lmhead::NANBEIGE_VOCAB_SIZE, strict_int8::{Int8Work, StrictInt8Engine, STRICT_INT8_PROFILE}},
    tasks::{extract::ExtractionVocabulary, source_planning::{SourceTaskPlanner,
        quantized::long::Int8SourceMapError}},
};
use super::{RedactError, RedactionRequest, actions::VerificationStatus,
    batch::{RedactionBatchArgs, Int8RedactionAdmission, Int8RedactionBatchAdmission},
    long::{Int8DocumentRedactor, LongRedactionConfig, LongRedactionError,
        LongRedactionPreflight, LongRedactionRun, LongRedactionStage, LONG_REDACTION_EXECUTION},
    pseudonym::Pseudonyms, quantized::Int8RedactionError};

/// The mapping ceilings cover BOTH stages of one document. These additional
/// ceilings cover every admitted document, including failed documents/epochs.
/// Input task_args cannot replace this host-owned policy or its secret scope.
pub struct LongRedactionBatchConfig {
    pub ner_identity: ExecutionIdentity,
    pub detector: LongRedactionConfig,
    pub request: RedactionRequest,
    pub max_model_work: Int8Work,
    pub max_mask_visits: u64,
}
impl LongRedactionBatchConfig {
    /// Native logits are projected in complete vocabulary rows. Rounding down
    /// an arbitrary numeric ceiling excludes NO executable native work; it also
    /// lets the existing corpus admission price an integral token reservation.
    pub fn item_model_work(&self) -> Int8Work {
        let mut work = self.detector.mapping.max_model_work;
        work.projected_logits -= work.projected_logits % NANBEIGE_VOCAB_SIZE as u64;
        work
    }
}

/// A source and its exact original-stage plan geometry, never execution proof.
pub struct PreparedDocumentRedaction { source: String, preflight: LongRedactionPreflight }

pub struct NativeInt8DocumentRedactionBatch<'p, 'e, 'w, 'v, 's, 'k, A: Int8RedactionBatchAdmission> {
    redactor: Int8DocumentRedactor<'p>, config: LongRedactionBatchConfig,
    engine: &'e mut StrictInt8Engine<'w>, vocabulary: &'v ExtractionVocabulary,
    pseudonyms: Option<&'s Pseudonyms<'k>>, admission: A, ledger: Ledger,
}
impl<'p, 'e, 'w, 'v, 's, 'k, A: Int8RedactionBatchAdmission>
    NativeInt8DocumentRedactionBatch<'p, 'e, 'w, 'v, 's, 'k, A> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(planner: &'p SourceTaskPlanner, config: LongRedactionBatchConfig,
        engine: &'e mut StrictInt8Engine<'w>, vocabulary: &'v ExtractionVocabulary,
        pseudonyms: Option<&'s Pseudonyms<'k>>, admission: A) -> Result<Self, BatchFault> {
        check_configuration(planner, &config)?;
        config.request.actions.check_key(pseudonyms).map_err(|_| BatchCode::Admission)?;
        let redactor = Int8DocumentRedactor::new(planner, config.ner_identity.clone(), config.detector.clone())
            .map_err(|_| BatchCode::Admission)?;
        capacity(engine, &config).map_err(|e| e.fault)?;
        Ok(Self { redactor, config, engine, vocabulary, pseudonyms, admission, ledger: Ledger::default() })
    }
    pub fn reserved_model_work(&self) -> Int8Work { self.ledger.reserved }
    pub fn reserved_mask_visits(&self) -> u64 { self.ledger.masks }
    pub fn is_poisoned(&self) -> bool { self.ledger.failed || !clean(self.engine) }
}
impl<A: Int8RedactionBatchAdmission> BatchProcessor
    for NativeInt8DocumentRedactionBatch<'_, '_, '_, '_, '_, '_, A> {
    type Args = RedactionBatchArgs;
    type Prepared = PreparedDocumentRedaction;
    type Output = GuardedOutput<LongRedactionRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.ready()?; self.ledger.failed = true;
        let result = (|| {
            checkpoint(control)?;
            check_source(&document.text, &self.config)?;
            // All exact original-stage plans are checked before its first
            // forward. Verification is planned only from the actual edited text.
            let preflight = self.redactor.preflight(&document.text, control).map_err(failure)?;
            checkpoint(control)?;
            Ok(PreparedDocumentRedaction { source: document.text, preflight })
        })();
        self.ledger.finish(&result); result
    }
    fn planned_work(&self, _: &Self::Prepared) -> BatchWork {
        let work = self.config.item_model_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.ledger.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        let kv = match capacity(self.engine, &self.config) {
            Ok(kv) => kv, Err(error) => { self.ledger.failed = true; return Err(error); }
        };
        let engine = &mut *self.engine;
        let config = &self.config;
        let redactor = &self.redactor;
        let vocabulary = self.vocabulary;
        let pseudonyms = self.pseudonyms;
        execute_reserved(&mut self.ledger, config, context, kv, &mut self.admission, control, |control| {
            let result = redactor.redact(&prepared.source, &config.request, pseudonyms, engine, vocabulary, control)
                .and_then(|out| { validate_result(&out, prepared.preflight, config)?; Ok(out) });
            (result, clean(engine))
        })
    }
}

pub(crate) fn check_configuration(planner: &SourceTaskPlanner, config: &LongRedactionBatchConfig) -> Result<(), BatchFault> {
    Int8DocumentRedactor::new(planner, config.ner_identity.clone(), config.detector.clone())
        .map_err(|_| BatchCode::Admission)?;
    check_limits(config)?;
    super::detectors::detect("", &config.request.rules, config.request.rule_budget)
        .map_err(|_| BatchCode::InvalidLimits)?;
    Ok(())
}
fn check_limits(config: &LongRedactionBatchConfig) -> Result<(), BatchFault> {
    let detector = &config.detector;
    let item = config.item_model_work();
    let stages = 1 + u64::from(config.request.verify);
    if item.forward_positions == 0 || item.projected_logits == 0 || item.attention_pairs == 0
        || item.projections.dot_products == 0 || item.projections.multiply_accumulates == 0
        || !fits(item, config.max_model_work)
        || !(1..=256).contains(&detector.mapping.chunks.max_chunks)
        || !(1..=64 * 1024 * 1024).contains(&detector.mapping.chunks.max_input_bytes)
        || detector.mapping.chunks.max_chunk_bytes > detector.planning.max_input_bytes
        || detector.mapping.chunks.context_tokens > detector.planning.max_context_tokens
        || detector.per_chunk.max_output_bytes > detector.mapping.reduction.max_value_bytes as u64
        || detector.mapping.reduction.max_live_value_bytes == 0
        || detector.mapping.reduction.max_result_bytes == 0
        || config.request.edit_budget.max_regions > 16_384
        || config.request.edit_budget.max_output_bytes as u64 > detector.max_result_bytes
        || detector.mapping.mask_visits_per_chunk == 0
        || detector.mapping.mask_visits_per_chunk.checked_mul(stages)
            .is_none_or(|n| n > detector.mapping.max_mask_visits)
        || detector.mapping.max_mask_visits > config.max_mask_visits {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
fn check_source(source: &str, config: &LongRedactionBatchConfig) -> Result<(), BatchItemFailure> {
    if source.is_empty() || source.len() > config.detector.mapping.chunks.max_input_bytes
        || source.len() > config.request.rule_budget.max_input_bytes {
        return Err(BatchItemFailure::reject(BatchCode::DocumentLimit));
    }
    Ok(())
}
fn clean(engine: &StrictInt8Engine<'_>) -> bool {
    !engine.is_poisoned() && engine.kv_cache().all_slots_have_len(0)
}
fn capacity(engine: &StrictInt8Engine<'_>, config: &LongRedactionBatchConfig) -> Result<u64, BatchItemFailure> {
    let positions = engine.kv_cache().capacity_positions();
    if !clean(engine) || config.detector.planning.max_context_tokens > positions
        || config.detector.mapping.chunks.context_tokens > positions {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    (positions as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
        .filter(|&n| n != 0 && n <= config.detector.per_chunk.max_kv_bytes)
        .ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))
}
fn fits(a: Int8Work, b: Int8Work) -> bool {
    a.forward_positions <= b.forward_positions && a.projected_logits <= b.projected_logits
        && a.attention_pairs <= b.attention_pairs && a.projections.fits(b.projections)
}
#[derive(Default)]
struct Ledger { reserved: Int8Work, masks: u64, sequence: u64, failed: bool }
impl Ledger {
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn begin(&mut self, config: &LongRedactionBatchConfig, context: BatchRequestContext) -> Result<(), BatchItemFailure> {
        self.ready()?; self.failed = true;
        if context.request_seq <= self.sequence || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let next = self.reserved.checked_add(config.item_model_work()).ok()
            .filter(|&n| fits(n, config.max_model_work)).ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let masks = self.masks.checked_add(config.detector.mapping.max_mask_visits)
            .filter(|&n| n <= config.max_mask_visits).ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        self.reserved = next; self.masks = masks; self.sequence = context.request_seq;
        Ok(())
    }
    fn finish<T>(&mut self, result: &Result<T, BatchItemFailure>) {
        if !result.as_ref().is_err_and(|e| e.stop) { self.failed = false; }
    }
}

// Private transaction seam: production calls only the concrete native redactor.
// Error vectors are dropped under admission; transport sees fixed codes only.
#[allow(clippy::too_many_arguments)]
fn execute_reserved<A, C, T, F>(ledger: &mut Ledger, config: &LongRedactionBatchConfig,
    context: BatchRequestContext, kv: u64, admission: &mut A, control: &mut C, execute: F)
    -> Result<GuardedOutput<T, A::Guard>, BatchItemFailure>
where A: Int8RedactionBatchAdmission, C: DecodeStepControl, T: Serialize,
    F: FnOnce(&mut C) -> (Result<T, LongRedactionError>, bool) {
    ledger.begin(config, context)?;
    let result = (|| {
        checkpoint(control)?;
        let (identity, guard) = admission.admit(Int8RedactionAdmission {
            identity: &config.ner_identity, model_work: config.item_model_work(),
            mask_node_visits: config.detector.mapping.max_mask_visits,
            mask_limits: config.detector.mapping.mask_limits, kv_reservation_bytes: kv,
            max_result_bytes: config.detector.max_result_bytes,
        })?;
        if canonjson::canonical_bytes(&identity).map_err(|_| BatchItemFailure::fatal(BatchCode::Serialization))?
            != canonjson::canonical_bytes(&config.ner_identity).map_err(|_| BatchItemFailure::fatal(BatchCode::Serialization))? {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        checkpoint(control)?;
        let (output, clean) = execute(control);
        let output = match output {
            Ok(output) if clean => output,
            Ok(_) => return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)),
            Err(error) => {
                let mut error = failure(error);
                if !clean { error.stop = true; }
                return Err(error);
            }
        };
        checkpoint(control)?;
        Ok(GuardedOutput::new(output, guard))
    })();
    ledger.finish(&result); result
}
fn validate_stage(stage: &LongRedactionStage, config: &LongRedactionBatchConfig) -> Result<(), LongRedactionError> {
    let p = stage.preflight;
    if p.source_bytes == 0 || p.source_bytes > config.detector.mapping.chunks.max_input_bytes
        || p.source_scalars > p.source_bytes || p.chunks == 0 || p.chunks > config.detector.mapping.chunks.max_chunks
        || !fits(stage.model_work, p.planned_model_work)
        || p.planned_model_work.projected_logits % NANBEIGE_VOCAB_SIZE as u64 != 0
        || stage.model_work.projected_logits % NANBEIGE_VOCAB_SIZE as u64 != 0
        || (p.chunks as u64).checked_mul(config.detector.mapping.mask_visits_per_chunk) != Some(p.reserved_mask_node_visits)
        || stage.mask_node_visit_charge > p.reserved_mask_node_visits {
        return Err(Int8RedactionError::InvalidResult.into());
    }
    Ok(())
}
fn validate_result(output: &LongRedactionRun, original: LongRedactionPreflight, config: &LongRedactionBatchConfig)
    -> Result<(), LongRedactionError> {
    let invalid = || LongRedactionError::Redaction(Int8RedactionError::InvalidResult);
    let expected = if config.request.verify { VerificationStatus::CleanDeclaredUnion } else { VerificationStatus::NotRequested };
    if output.schema_version != 1 || output.execution != LONG_REDACTION_EXECUTION || output.numerics_profile != STRICT_INT8_PROFILE
        || output.detector_scope != "whole-source-rules-independent-ner-chunks-v1"
        || output.original.preflight != original || output.verification.is_some() != config.request.verify
        || output.result.verification() != expected
        || (!config.request.actions.include_map && !output.result.edits().is_empty())
        || output.verification_scan_steps > config.request.grounding_budget.max_scan_steps {
        return Err(invalid());
    }
    validate_stage(&output.original, config)?;
    let mut reserved = original.planned_model_work;
    let mut actual = output.original.model_work;
    let mut masks = original.reserved_mask_node_visits;
    let mut charged = output.original.mask_node_visit_charge;
    if let Some(second) = &output.verification {
        validate_stage(second, config)?;
        if second.preflight.source_bytes != output.result.text().len()
            || second.preflight.source_scalars != output.result.text().chars().count() { return Err(invalid()); }
        reserved = reserved.checked_add(second.preflight.planned_model_work).map_err(|_| invalid())?;
        actual = actual.checked_add(second.model_work).map_err(|_| invalid())?;
        masks = masks.checked_add(second.preflight.reserved_mask_node_visits).ok_or_else(invalid)?;
        charged = charged.checked_add(second.mask_node_visit_charge).ok_or_else(invalid)?;
    }
    if reserved != output.reserved_model_work || actual != output.model_work || masks != output.reserved_mask_node_visits
        || charged != output.mask_node_visit_charge || !fits(reserved, config.item_model_work())
        || !fits(actual, reserved) || masks > config.detector.mapping.max_mask_visits || charged > masks {
        return Err(invalid());
    }
    Ok(())
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) { Some(c) => Err(BatchItemFailure::fatal(BatchFault::cancelled(c))), None => Ok(()) }
}
fn failure(error: LongRedactionError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        LongRedactionError::Map(Int8SourceMapError::EmptySource) | LongRedactionError::Redaction(Int8RedactionError::Redaction(RedactError::InputBudget))
            => BatchItemFailure::reject(BatchCode::DocumentLimit),
        LongRedactionError::Map(Int8SourceMapError::WorkLimit) | LongRedactionError::Redaction(Int8RedactionError::WorkBudget)
            | LongRedactionError::Redaction(Int8RedactionError::Redaction(RedactError::WorkBudget | RedactError::DetectionBudget | RedactError::CandidateBudget))
            => BatchItemFailure::reject(BatchCode::WorkLimit),
        LongRedactionError::Redaction(Int8RedactionError::Residual(_))
            | LongRedactionError::Redaction(Int8RedactionError::Redaction(RedactError::VerificationResidual { .. }))
            | LongRedactionError::Map(Int8SourceMapError::Chunk(_)) => BatchItemFailure::reject(BatchCode::Execution),
        LongRedactionError::Redaction(Int8RedactionError::Redaction(RedactError::OutputBudget))
            => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        LongRedactionError::Map(Int8SourceMapError::Admission | Int8SourceMapError::InvalidLimits)
            | LongRedactionError::Redaction(Int8RedactionError::Identity)
            | LongRedactionError::Redaction(Int8RedactionError::Redaction(RedactError::MissingKey | RedactError::KeyMismatch))
            => BatchItemFailure::fatal(BatchCode::Admission),
        LongRedactionError::Map(Int8SourceMapError::Allocation)
            | LongRedactionError::Redaction(Int8RedactionError::Redaction(RedactError::AllocationRefused))
            => BatchItemFailure::fatal(BatchCode::Allocation),
        LongRedactionError::Redaction(Int8RedactionError::InvalidResult)
            | LongRedactionError::Redaction(Int8RedactionError::Redaction(RedactError::InvalidSpan | RedactError::InvalidNerEvidence))
            => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        _ => BatchItemFailure::fatal(BatchCode::Execution),
    }
}

#[cfg(test)] mod tests;
