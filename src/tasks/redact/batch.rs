//! Item-local redaction on the existing bounded NDJSON runner. One immutable
//! detector/action/key scope serves the whole stream; input cannot weaken it.
//! This is serial streaming, not a durable job, retry engine or new scheduler.

use serde::{Deserialize, Serialize};
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, generation::GuardedOutput},
    canonjson, execution_identity::ExecutionIdentity,
    native_engine::{decode::DecodeStepControl, kv::KV_BYTES_PER_TOKEN,
        lmhead::NANBEIGE_VOCAB_SIZE, strict_int8::{Int8Work, StrictInt8Engine, STRICT_INT8_PROFILE}},
    tasks::{extract::ExtractionVocabulary, source_planning::SourceTaskPlanner},
};
use super::{RedactError, RedactionRequest, actions::VerificationStatus,
    pseudonym::Pseudonyms, quantized::{Int8Redactor, Int8RedactionConfig,
        Int8RedactionError, Int8RedactionRun, INT8_REDACTION_EXECUTION}};
// Reuse existing admission authority and output-lifetime obligations.
pub use crate::batch::extract::quantized::{Int8ExtractionAdmission as Int8RedactionAdmission,
    Int8ExtractionBatchAdmission as Int8RedactionBatchAdmission};

/// Fixed host configuration, never deserialized from an input record. The
/// detector's model ceiling covers BOTH passes of one document; the additional
/// ceilings below cover the entire stream, including failed documents.
pub struct Int8RedactionBatchConfig {
    pub ner_identity: ExecutionIdentity,
    pub detector: Int8RedactionConfig,
    pub request: RedactionRequest,
    pub max_model_work: Int8Work,
    pub max_mask_visits: u64,
}

/// No per-record switches, keys, types, actions or verification opt-outs.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RedactionBatchArgs {}

/// Owned untrusted bytes only, NOT a native plan or an execution receipt.
/// Both native prompts are compiled inside the charged operation: the second
/// prompt cannot exist until the first pass and transactional edit complete.
pub struct PreparedRedaction { source: String }

pub struct NativeInt8RedactionBatch<'p, 'e, 'w, 'v, 's, 'k, A: Int8RedactionBatchAdmission> {
    redactor: Int8Redactor<'p>, config: Int8RedactionBatchConfig,
    engine: &'e mut StrictInt8Engine<'w>, vocabulary: &'v ExtractionVocabulary,
    pseudonyms: Option<&'s Pseudonyms<'k>>, admission: A, run: RunState,
}
impl<'p, 'e, 'w, 'v, 's, 'k, A: Int8RedactionBatchAdmission>
    NativeInt8RedactionBatch<'p, 'e, 'w, 'v, 's, 'k, A> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(planner: &'p SourceTaskPlanner, config: Int8RedactionBatchConfig,
        engine: &'e mut StrictInt8Engine<'w>, vocabulary: &'v ExtractionVocabulary,
        pseudonyms: Option<&'s Pseudonyms<'k>>, admission: A) -> Result<Self, BatchFault> {
        check_configuration(planner, &config)?;
        config.request.actions.check_key(pseudonyms).map_err(|_| BatchCode::Admission)?;
        let redactor = Int8Redactor::new(planner, config.ner_identity.clone(), config.detector.clone())
            .map_err(|_| BatchCode::Admission)?;
        capacity_bytes(engine, &config).map_err(|e| e.fault)?;
        Ok(Self { redactor, config, engine, vocabulary, pseudonyms, admission, run: RunState::default() })
    }
    pub fn reserved_model_work(&self) -> Int8Work { self.run.reserved }
    pub fn reserved_mask_visits(&self) -> u64 { self.run.masks }
    pub fn is_poisoned(&self) -> bool { self.run.failed || !clean(self.engine) }
}
impl<A: Int8RedactionBatchAdmission> BatchProcessor for NativeInt8RedactionBatch<'_, '_, '_, '_, '_, '_, A> {
    type Args = RedactionBatchArgs;
    type Prepared = PreparedRedaction;
    type Output = GuardedOutput<Int8RedactionRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        prepare(&mut self.run, &self.config, document, control)
    }
    fn planned_work(&self, _: &Self::Prepared) -> BatchWork {
        let cap = self.config.detector.max_model_work;
        BatchWork { forward_positions: cap.forward_positions, projected_logits: cap.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.run.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        let kv = match capacity_bytes(self.engine, &self.config) {
            Ok(kv) => kv, Err(error) => { self.run.failed = true; return Err(error); }
        };
        let config = &self.config;
        let redactor = &self.redactor;
        let engine = &mut *self.engine;
        let vocabulary = self.vocabulary;
        let pseudonyms = self.pseudonyms;
        execute_reserved(&mut self.run, config, context, kv, &mut self.admission, control, |control| {
            let result = redactor.redact(&prepared.source, &config.request, pseudonyms, engine, vocabulary, control)
                .and_then(|out| { validate_result(&out, config)?; Ok(out) });
            (result, clean(engine))
        })
    }
}

pub(crate) fn check_configuration(planner: &SourceTaskPlanner, config: &Int8RedactionBatchConfig) -> Result<(), BatchFault> {
    Int8Redactor::new(planner, config.ner_identity.clone(), config.detector.clone())
        .map_err(|_| BatchCode::Admission)?;
    check_limits(config)?;
    // Validate the same rule automata/budgets before any document or model work.
    super::detectors::detect("", &config.request.rules, config.request.rule_budget)
        .map_err(|_| BatchCode::InvalidLimits)?;
    Ok(())
}
fn check_limits(config: &Int8RedactionBatchConfig) -> Result<(), BatchFault> {
    let per = config.detector.max_model_work;
    let planning = config.detector.planning;
    if per.forward_positions == 0 || per.projected_logits == 0 || per.attention_pairs == 0
        || per.projections.dot_products == 0 || per.projections.multiply_accumulates == 0
        || per.projected_logits % NANBEIGE_VOCAB_SIZE as u64 != 0
        || !fits(per, config.max_model_work)
        || !(1..=64 * 1024 * 1024).contains(&planning.max_input_bytes)
        || !(1..=262_144).contains(&planning.max_context_tokens) || !(1..=1024).contains(&planning.max_passages)
        || config.request.edit_budget.max_output_bytes as u64 > config.detector.max_result_bytes
        || config.request.edit_budget.max_regions > 16_384 {
        return Err(BatchCode::InvalidLimits.into());
    }
    let masks = masks_per_item(config).ok_or(BatchCode::InvalidLimits)?;
    if masks == 0 || masks > config.detector.max_mask_visits || masks > config.max_mask_visits {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
fn masks_per_item(config: &Int8RedactionBatchConfig) -> Option<u64> {
    config.detector.mask_visits_per_pass.checked_mul(1 + u64::from(config.request.verify))
}
fn clean(engine: &StrictInt8Engine<'_>) -> bool {
    !engine.is_poisoned() && engine.kv_cache().all_slots_have_len(0)
}
fn capacity_bytes(engine: &StrictInt8Engine<'_>, config: &Int8RedactionBatchConfig) -> Result<u64, BatchItemFailure> {
    if !clean(engine) || config.detector.planning.max_context_tokens > engine.kv_cache().capacity_positions() {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let bytes = (engine.kv_cache().capacity_positions() as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
        .filter(|&n| n != 0 && n <= config.detector.per_pass.max_kv_bytes)
        .ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))?;
    Ok(bytes)
}
fn fits(a: Int8Work, b: Int8Work) -> bool {
    a.forward_positions <= b.forward_positions && a.projected_logits <= b.projected_logits
        && a.attention_pairs <= b.attention_pairs && a.projections.fits(b.projections)
}
#[derive(Default)]
struct RunState { reserved: Int8Work, masks: u64, last_sequence: u64, failed: bool }
impl RunState {
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn begin(&mut self, config: &Int8RedactionBatchConfig, context: BatchRequestContext) -> Result<(), BatchItemFailure> {
        self.ready()?; self.failed = true;
        if context.request_seq <= self.last_sequence || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let next = self.reserved.checked_add(config.detector.max_model_work).ok()
            .filter(|&w| fits(w, config.max_model_work)).ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let masks = masks_per_item(config).and_then(|n| self.masks.checked_add(n))
            .filter(|&n| n <= config.max_mask_visits).ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        // Commit both counters together. Success, failure and flush NEVER refund.
        self.reserved = next; self.masks = masks; self.last_sequence = context.request_seq;
        Ok(())
    }
    fn finish<T>(&mut self, result: &Result<T, BatchItemFailure>) {
        if !result.as_ref().is_err_and(|e| e.stop) { self.failed = false; }
    }
}
fn prepare<C: DecodeStepControl>(state: &mut RunState, config: &Int8RedactionBatchConfig,
    document: BatchDocument<RedactionBatchArgs>, control: &mut C) -> Result<PreparedRedaction, BatchItemFailure> {
    state.ready()?; state.failed = true;
    let result = (|| {
        checkpoint(control)?;
        if document.text.len() > config.detector.planning.max_input_bytes
            || document.text.len() > config.request.rule_budget.max_input_bytes {
            return Err(BatchItemFailure::reject(BatchCode::DocumentLimit));
        }
        Ok(PreparedRedaction { source: document.text })
    })();
    state.finish(&result); result
}

// Private transaction seam for failure/ownership tests. Only the public native
// adapter above supplies production execution; no caller-provided driver exists.
#[allow(clippy::too_many_arguments)]
fn execute_reserved<A, C, T, F>(state: &mut RunState, config: &Int8RedactionBatchConfig,
    context: BatchRequestContext, kv: u64, admission: &mut A, control: &mut C, execute: F)
    -> Result<GuardedOutput<T, A::Guard>, BatchItemFailure>
where A: Int8RedactionBatchAdmission, C: DecodeStepControl, T: Serialize,
    F: FnOnce(&mut C) -> (Result<T, Int8RedactionError>, bool) {
    state.begin(config, context)?;
    let result = (|| {
        checkpoint(control)?;
        let (admitted, guard) = admission.admit(Int8RedactionAdmission {
            identity: &config.ner_identity, model_work: config.detector.max_model_work,
            mask_node_visits: masks_per_item(config).ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?,
            mask_limits: config.detector.mask_limits, kv_reservation_bytes: kv,
            max_result_bytes: config.detector.max_result_bytes,
        })?;
        // The admitted seed fixes model, tokenizer, template and detector task;
        // the native redactor derives and checks each exact prompt identity.
        if canonjson::canonical_bytes(&admitted).map_err(|_| BatchItemFailure::fatal(BatchCode::Serialization))?
            != canonjson::canonical_bytes(&config.ner_identity).map_err(|_| BatchItemFailure::fatal(BatchCode::Serialization))? {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        checkpoint(control)?;
        let (output, clean) = execute(control);
        let output = match output {
            Ok(output) if clean => output,
            Ok(_) => return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)),
            Err(error) => {
                // Residual vectors/private native errors are drained while the
                // output reservation is still live. Export only fixed codes.
                let mut failure = execution_failure(error);
                if !clean { failure.stop = true; }
                return Err(failure);
            }
        };
        checkpoint(control)?;
        Ok(GuardedOutput::new(output, guard))
    })();
    state.finish(&result); result
}
fn validate_result(output: &Int8RedactionRun, config: &Int8RedactionBatchConfig) -> Result<(), Int8RedactionError> {
    let expected = if config.request.verify { VerificationStatus::CleanDeclaredUnion } else { VerificationStatus::NotRequested };
    if output.schema_version != 1 || output.execution != INT8_REDACTION_EXECUTION || output.numerics_profile != STRICT_INT8_PROFILE
        || output.ner_passes != 1 + usize::from(config.request.verify) || output.result.verification() != expected
        || !fits(output.reserved_model_work, config.detector.max_model_work) || !fits(output.model_work, output.reserved_model_work)
        || Some(output.reserved_mask_node_visits) != masks_per_item(config)
        || output.mask_node_visit_charge > output.reserved_mask_node_visits {
        return Err(Int8RedactionError::InvalidResult);
    }
    Ok(())
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) { Some(c) => Err(BatchItemFailure::fatal(BatchFault::cancelled(c))), None => Ok(()) }
}
fn execution_failure(error: Int8RedactionError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int8RedactionError::Residual(_) | Int8RedactionError::Redaction(RedactError::VerificationResidual { .. })
            => BatchItemFailure::reject(BatchCode::Execution),
        Int8RedactionError::WorkBudget | Int8RedactionError::Redaction(RedactError::WorkBudget | RedactError::DetectionBudget | RedactError::CandidateBudget)
            => BatchItemFailure::reject(BatchCode::WorkLimit),
        Int8RedactionError::Redaction(RedactError::InputBudget) => BatchItemFailure::reject(BatchCode::DocumentLimit),
        Int8RedactionError::Redaction(RedactError::OutputBudget) => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        Int8RedactionError::Identity | Int8RedactionError::Redaction(RedactError::MissingKey | RedactError::KeyMismatch)
            => BatchItemFailure::fatal(BatchCode::Admission),
        Int8RedactionError::Redaction(RedactError::AllocationRefused) => BatchItemFailure::fatal(BatchCode::Allocation),
        Int8RedactionError::InvalidResult | Int8RedactionError::Redaction(RedactError::InvalidSpan | RedactError::InvalidNerEvidence)
            => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        _ => BatchItemFailure::fatal(BatchCode::Execution),
    }
}

#[cfg(test)] mod tests;
