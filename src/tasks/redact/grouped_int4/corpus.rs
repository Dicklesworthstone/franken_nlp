//! Resident native grouped-INT4 complete-document redaction on ordered NDJSON.
//! One model, detector and key scope span the stream. Every admitted document,
//! including failed documents and later epochs, consumes its whole allowance.
use serde::Serialize;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, generation::GuardedOutput},
    canonjson, execution_identity::ExecutionIdentity, grammar::mask::MaskOracleError,
    native_engine::{constrained::JsonDecodeError, constrained_int4::Int4JsonError,
        decode::DecodeStepControl, kv::KV_BYTES_PER_TOKEN, lmhead::NANBEIGE_VOCAB_SIZE,
        strict_int4::{Int4ModelIdentity, Int4Work, StrictInt4Engine, StrictInt4Error, STRICT_INT4_PROFILE}},
    tasks::{extract::{ExtractError, ExtractionVocabulary, grouped_int4::Int4ExtractError},
        mapreduce::{MapReduceError, execution::ExecutionError}, ner::NerError,
        source_planning::{SourcePlanningError, SourceTaskPlanner, grouped_int4::{Int4SourceError,
            long::{Int4SourceMapError, SourceMapVerification}}}},
    validation::grounded_fields::GroundingBudget,
};
use super::{Int4RedactionError, check_model,
    long::{Int4DocumentRedactor, Int4LongRedactionConfig, Int4LongRedactionError,
        Int4LongRedactionPreflight, Int4LongRedactionRun, Int4LongRedactionStage, INT4_LONG_REDACTION_EXECUTION}};
use super::super::{RedactError, RedactionRequest,
    actions::{VerificationStatus, ACTION_POLICY_VERSION}, detectors::RULE_PROFILE,
    pseudonym::Pseudonyms, union::OVERLAP_POLICY};
pub use crate::batch::extract::grouped_int4::{
    Int4ExtractionAdmission as Int4RedactionAdmission,
    Int4ExtractionBatchAdmission as Int4RedactionBatchAdmission,
};
pub use super::super::batch::RedactionBatchArgs;

/// The detector's allowances cover both stages of one complete document.
/// These additional ceilings cover every admitted document across all epochs.
/// Input task_args cannot change policy, verification, model types or key scope.
pub struct Int4DocumentRedactionBatchConfig {
    pub ner_identity: ExecutionIdentity,
    pub detector: Int4LongRedactionConfig,
    pub request: RedactionRequest,
    pub max_model_work: Int4Work,
    pub max_mask_visits: u64,
}
impl Int4DocumentRedactionBatchConfig {
    /// Native logits are whole vocabulary rows. Removing an unusable fractional
    /// row excludes no executable work and keeps admission's token count exact.
    pub fn item_model_work(&self) -> Int4Work {
        let mut work = self.detector.mapping.max_model_work;
        work.projected_logits -= work.projected_logits % NANBEIGE_VOCAB_SIZE as u64;
        work
    }
}

/// Owns source bytes and original-stage geometry, never an execution receipt.
pub struct PreparedInt4DocumentRedaction { source: String, preflight: Int4LongRedactionPreflight }

pub struct NativeInt4DocumentRedactionBatch<'p, 'e, 'w, 'v, 's, 'k, A: Int4RedactionBatchAdmission> {
    redactor: Int4DocumentRedactor<'p>, config: Int4DocumentRedactionBatchConfig,
    engine: &'e mut StrictInt4Engine<'w>, vocabulary: &'v ExtractionVocabulary,
    pseudonyms: Option<&'s Pseudonyms<'k>>, admission: A, ledger: Ledger,
}
impl<'p, 'e, 'w, 'v, 's, 'k, A: Int4RedactionBatchAdmission>
    NativeInt4DocumentRedactionBatch<'p, 'e, 'w, 'v, 's, 'k, A> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(planner: &'p SourceTaskPlanner, config: Int4DocumentRedactionBatchConfig,
        engine: &'e mut StrictInt4Engine<'w>, vocabulary: &'v ExtractionVocabulary,
        pseudonyms: Option<&'s Pseudonyms<'k>>, admission: A) -> Result<Self, BatchFault> {
        check_configuration(planner, &config, engine.model_identity())?;
        config.request.actions.check_key(pseudonyms).map_err(|_| BatchCode::Admission)?;
        let redactor = Int4DocumentRedactor::new(planner, config.ner_identity.clone(),
            config.detector.clone(), engine.model_identity()).map_err(|_| BatchCode::Admission)?;
        capacity(engine, &config, redactor.group_width()).map_err(|e| e.fault)?;
        Ok(Self { redactor, config, engine, vocabulary, pseudonyms, admission, ledger: Ledger::default() })
    }
    pub fn reserved_model_work(&self) -> Int4Work { self.ledger.reserved }
    pub fn reserved_mask_visits(&self) -> u64 { self.ledger.masks }
    pub fn is_poisoned(&self) -> bool { self.ledger.failed || !clean(self.engine) }
}
impl<A: Int4RedactionBatchAdmission> BatchProcessor
    for NativeInt4DocumentRedactionBatch<'_, '_, '_, '_, '_, '_, A> {
    type Args = RedactionBatchArgs;
    type Prepared = PreparedInt4DocumentRedaction;
    type Output = GuardedOutput<Int4LongRedactionRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.ready()?; self.ledger.failed = true;
        let result = (|| {
            checkpoint(control)?;
            capacity(self.engine, &self.config, self.redactor.group_width())?;
            check_source(&document.text, &self.config)?;
            // All original chunk plans are checked under the outer preparation
            // reservation; only the actual edited source can plan verification.
            let preflight = self.redactor.preflight(&document.text, control).map_err(failure)?;
            checkpoint(control)?;
            Ok(PreparedInt4DocumentRedaction { source: document.text, preflight })
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
        let group_width = self.redactor.group_width();
        let kv = match capacity(self.engine, &self.config, group_width) {
            Ok(kv) => kv, Err(error) => { self.ledger.failed = true; return Err(error); }
        };
        let engine = &mut *self.engine;
        let config = &self.config;
        let redactor = &self.redactor;
        let vocabulary = self.vocabulary;
        let pseudonyms = self.pseudonyms;
        execute_reserved(&mut self.ledger, config, context, kv, &mut self.admission, control, |control| {
            let result = redactor.redact(&prepared.source, &config.request, pseudonyms, engine, vocabulary, control)
                .and_then(|out| { validate_result(&out, prepared.preflight, config, group_width, pseudonyms)?; Ok(out) });
            (result, clean(engine))
        })
    }
}

/// Host/CLI admission before opening a snapshot, stream or native session.
pub(crate) fn check_configuration(planner: &SourceTaskPlanner, config: &Int4DocumentRedactionBatchConfig,
    model: &Int4ModelIdentity) -> Result<(), BatchFault> {
    check_limits(config)?;
    Int4DocumentRedactor::new(planner, config.ner_identity.clone(), config.detector.clone(), model)
        .map_err(|_| BatchCode::Admission)?;
    super::super::detectors::detect("", &config.request.rules, config.request.rule_budget)
        .map_err(|_| BatchCode::InvalidLimits)?;
    Ok(())
}
fn check_limits(config: &Int4DocumentRedactionBatchConfig) -> Result<(), BatchFault> {
    let detector = &config.detector;
    let item = config.item_model_work();
    let stages = 1 + u64::from(config.request.verify);
    if !positive(item) || !fits(item, config.max_model_work)
        || !(1..=256).contains(&detector.mapping.chunks.max_chunks)
        || !(1..=64 * 1024 * 1024).contains(&detector.mapping.chunks.max_input_bytes)
        || detector.mapping.chunks.max_chunk_bytes > detector.planning.max_input_bytes
        || detector.mapping.chunks.context_tokens > detector.planning.max_context_tokens
        || detector.per_chunk.max_output_bytes > detector.mapping.reduction.max_value_bytes as u64
        || detector.mapping.reduction.max_live_value_bytes == 0 || detector.mapping.reduction.max_result_bytes == 0
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
fn check_source(source: &str, config: &Int4DocumentRedactionBatchConfig) -> Result<(), BatchItemFailure> {
    if source.is_empty() || source.len() > config.detector.mapping.chunks.max_input_bytes
        || source.len() > config.request.rule_budget.max_input_bytes {
        return Err(BatchItemFailure::reject(BatchCode::DocumentLimit));
    }
    Ok(())
}
fn clean(engine: &StrictInt4Engine<'_>) -> bool {
    !engine.is_poisoned() && engine.kv_cache().all_slots_have_len(0)
}
fn capacity(engine: &StrictInt4Engine<'_>, config: &Int4DocumentRedactionBatchConfig,
    group_width: usize) -> Result<u64, BatchItemFailure> {
    if !clean(engine) { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
    check_model(&config.ner_identity, engine.model_identity())
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    let positions = engine.kv_cache().capacity_positions();
    if engine.profile() != STRICT_INT4_PROFILE || engine.model_identity().group_width() != group_width
        || config.detector.planning.max_context_tokens > positions
        || config.detector.mapping.chunks.context_tokens > positions {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    (positions as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
        .filter(|&n| n != 0 && n <= config.detector.per_chunk.max_kv_bytes)
        .ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))
}
fn fits(a: Int4Work, b: Int4Work) -> bool {
    a.forward_positions <= b.forward_positions && a.projected_logits <= b.projected_logits
        && a.attention_pairs <= b.attention_pairs && a.projections.fits(b.projections)
}
fn positive(work: Int4Work) -> bool {
    work.forward_positions > 0 && work.projected_logits > 0 && work.attention_pairs > 0
        && work.projections.dot_products > 0 && work.projections.multiply_accumulates > 0
}
#[derive(Default)]
struct Ledger { reserved: Int4Work, masks: u64, sequence: u64, epoch: u64, line: u64, failed: bool }
impl Ledger {
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn begin(&mut self, config: &Int4DocumentRedactionBatchConfig, context: BatchRequestContext) -> Result<(), BatchItemFailure> {
        self.ready()?; self.failed = true;
        // Flushes and preparation refusals can consume request sequence numbers.
        // Gaps are valid; replay, backward epochs and reused lines are not.
        if context.request_seq <= self.sequence || context.epoch == 0 || context.epoch < self.epoch
            || context.input_line <= self.line {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let next = self.reserved.checked_add(config.item_model_work()).ok()
            .filter(|&n| fits(n, config.max_model_work)).ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let masks = self.masks.checked_add(config.detector.mapping.max_mask_visits)
            .filter(|&n| n <= config.max_mask_visits).ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        self.reserved = next; self.masks = masks;
        self.sequence = context.request_seq; self.epoch = context.epoch; self.line = context.input_line;
        Ok(())
    }
    fn finish<T>(&mut self, result: &Result<T, BatchItemFailure>) {
        if !result.as_ref().is_err_and(|e| e.stop) { self.failed = false; }
    }
}

// Private transaction seam; production invokes only the concrete native driver.
// The retained output guard covers validation, serialization and writer flush.
#[allow(clippy::too_many_arguments)]
fn execute_reserved<A, C, T, F>(ledger: &mut Ledger, config: &Int4DocumentRedactionBatchConfig,
    context: BatchRequestContext, kv: u64, admission: &mut A, control: &mut C, execute: F)
    -> Result<GuardedOutput<T, A::Guard>, BatchItemFailure>
where A: Int4RedactionBatchAdmission, C: DecodeStepControl, T: Serialize,
    F: FnOnce(&mut C) -> (Result<T, Int4LongRedactionError>, bool) {
    ledger.begin(config, context)?;
    let result = (|| {
        checkpoint(control)?;
        let (identity, guard) = admission.admit(Int4RedactionAdmission {
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
fn validate_stage(stage: &Int4LongRedactionStage, config: &Int4DocumentRedactionBatchConfig,
    group_width: usize) -> Result<(), Int4LongRedactionError> {
    let p = stage.preflight;
    if p.source_bytes == 0 || p.source_bytes > config.detector.mapping.chunks.max_input_bytes
        || p.source_scalars == 0 || p.source_scalars > p.source_bytes
        || p.chunks == 0 || p.chunks > config.detector.mapping.chunks.max_chunks || p.group_width != group_width
        || !positive(p.planned_model_work) || !positive(stage.model_work) || !fits(stage.model_work, p.planned_model_work)
        || p.planned_model_work.projected_logits % NANBEIGE_VOCAB_SIZE as u64 != 0
        || stage.model_work.projected_logits % NANBEIGE_VOCAB_SIZE as u64 != 0
        || (p.chunks as u64).checked_mul(config.detector.mapping.mask_visits_per_chunk) != Some(p.reserved_mask_node_visits)
        || stage.mask_node_visit_charge > p.reserved_mask_node_visits
        || !verification_fits(stage.map_verification_used, config.detector.mapping.verification)
        || !verification_fits(stage.detection_verification_used, config.request.grounding_budget) {
        return Err(Int4RedactionError::InvalidResult.into());
    }
    Ok(())
}
fn validate_result(output: &Int4LongRedactionRun, original: Int4LongRedactionPreflight,
    config: &Int4DocumentRedactionBatchConfig, group_width: usize, pseudonyms: Option<&Pseudonyms<'_>>)
    -> Result<(), Int4LongRedactionError> {
    let invalid = || Int4LongRedactionError::Redaction(Int4RedactionError::InvalidResult);
    let expected = if config.request.verify { VerificationStatus::CleanDeclaredUnion } else { VerificationStatus::NotRequested };
    if output.schema_version != 1 || output.execution != INT4_LONG_REDACTION_EXECUTION
        || output.numerics_profile != STRICT_INT4_PROFILE || output.group_width != group_width
        || output.detector_scope != "whole-source-rules-independent-ner-chunks-v1"
        || output.warnings != ["ner_chunk_boundaries_may_split_entities", "detector_recall_not_established",
            "clean_scan_and_pseudonyms_are_not_anonymization"]
        || output.original.preflight != original || output.verification.is_some() != config.request.verify
        || output.result.schema_version != 1 || output.result.task_spec_version != "redact-v1"
        || output.result.rule_profile != RULE_PROFILE || output.result.rules != config.request.rules
        || output.result.overlap_policy != OVERLAP_POLICY || output.result.action_policy != ACTION_POLICY_VERSION
        || output.result.verification() != expected || output.result.region_count() > config.request.edit_budget.max_regions
        || output.result.text().len() > config.request.edit_budget.max_output_bytes
        || output.result.model_types().len() != config.detector.ner.types.len()
        || config.detector.ner.types.iter().any(|t| !output.result.model_types().contains(t))
        || output.result.pseudonym_identity.as_ref() != pseudonyms.map(Pseudonyms::identity)
        || (!config.request.actions.include_map && !output.result.edits().is_empty())
        || output.verification_scan_steps != output.detection_verification_used.scan_steps {
        return Err(invalid());
    }
    validate_stage(&output.original, config, group_width)?;
    let mut reserved = original.planned_model_work;
    let mut actual = output.original.model_work;
    let mut masks = original.reserved_mask_node_visits;
    let mut charged = output.original.mask_node_visit_charge;
    let mut map_used = output.original.map_verification_used;
    let mut detection_used = output.original.detection_verification_used;
    if let Some(second) = &output.verification {
        validate_stage(second, config, group_width)?;
        if second.preflight.source_bytes != output.result.text().len()
            || second.preflight.source_scalars != output.result.text().chars().count() { return Err(invalid()); }
        reserved = reserved.checked_add(second.preflight.planned_model_work).map_err(|_| invalid())?;
        actual = actual.checked_add(second.model_work).map_err(|_| invalid())?;
        masks = masks.checked_add(second.preflight.reserved_mask_node_visits).ok_or_else(invalid)?;
        charged = charged.checked_add(second.mask_node_visit_charge).ok_or_else(invalid)?;
        map_used = add_verification(map_used, second.map_verification_used).ok_or_else(invalid)?;
        detection_used = add_verification(detection_used, second.detection_verification_used).ok_or_else(invalid)?;
    }
    if reserved != output.reserved_model_work || actual != output.model_work || masks != output.reserved_mask_node_visits
        || charged != output.mask_node_visit_charge || !fits(reserved, config.item_model_work()) || !fits(actual, reserved)
        || masks > config.detector.mapping.max_mask_visits || charged > masks
        || map_used != output.map_verification_used || detection_used != output.detection_verification_used
        || !verification_fits(map_used, config.detector.mapping.verification)
        || !verification_fits(detection_used, config.request.grounding_budget) {
        return Err(invalid());
    }
    Ok(())
}
fn add_verification(a: SourceMapVerification, b: SourceMapVerification) -> Option<SourceMapVerification> {
    Some(SourceMapVerification { fields: a.fields.checked_add(b.fields)?, matches: a.matches.checked_add(b.matches)?,
        scan_steps: a.scan_steps.checked_add(b.scan_steps)? })
}
fn verification_fits(used: SourceMapVerification, budget: GroundingBudget) -> bool {
    used.fields <= budget.max_fields && used.matches <= budget.max_matches && used.scan_steps <= budget.max_scan_steps
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) { Some(c) => Err(BatchItemFailure::fatal(BatchFault::cancelled(c))), None => Ok(()) }
}

// Only fixed protocol codes escape. Follow nested semantic causes so a clean
// resident engine can continue after one document exhausts its own allowance.
fn failure(error: Int4LongRedactionError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int4LongRedactionError::Map(error) => map_failure(error),
        Int4LongRedactionError::Redaction(error) => match error {
            Int4RedactionError::Source(error) => source_failure(error),
            Int4RedactionError::Native(error) => native_failure(error),
            Int4RedactionError::Residual(_) => BatchItemFailure::reject(BatchCode::Execution),
            Int4RedactionError::WorkBudget => BatchItemFailure::reject(BatchCode::WorkLimit),
            Int4RedactionError::Identity => BatchItemFailure::fatal(BatchCode::Admission),
            Int4RedactionError::InvalidResult => BatchItemFailure::fatal(BatchCode::InvalidExecution),
            Int4RedactionError::Cancelled(cause) => BatchItemFailure::fatal(BatchFault::cancelled(cause)),
            Int4RedactionError::Redaction(error) => match error {
                RedactError::InputBudget => BatchItemFailure::reject(BatchCode::DocumentLimit),
                RedactError::WorkBudget | RedactError::DetectionBudget | RedactError::CandidateBudget
                    => BatchItemFailure::reject(BatchCode::WorkLimit),
                RedactError::OutputBudget => BatchItemFailure::reject(BatchCode::OutputLineLimit),
                RedactError::VerificationResidual { .. } => BatchItemFailure::reject(BatchCode::Execution),
                RedactError::MissingKey | RedactError::KeyMismatch | RedactError::InvalidOptions
                    => BatchItemFailure::fatal(BatchCode::Admission),
                RedactError::AllocationRefused => BatchItemFailure::fatal(BatchCode::Allocation),
                RedactError::Serialization => BatchItemFailure::fatal(BatchCode::Serialization),
                _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
            },
        },
    }
}
fn map_failure(error: Int4SourceMapError) -> BatchItemFailure {
    match error {
        Int4SourceMapError::EmptySource => BatchItemFailure::reject(BatchCode::DocumentLimit),
        Int4SourceMapError::WorkLimit => BatchItemFailure::reject(BatchCode::WorkLimit),
        Int4SourceMapError::Admission | Int4SourceMapError::InvalidLimits => BatchItemFailure::fatal(BatchCode::Admission),
        Int4SourceMapError::Allocation => BatchItemFailure::fatal(BatchCode::Allocation),
        Int4SourceMapError::Chunk(error) => chunk_failure(error),
        Int4SourceMapError::Source(error) => source_failure(error),
        Int4SourceMapError::Reduction(error) => match error {
            ExecutionError::Task { source, .. } => source_failure(source),
            ExecutionError::Checkpoint(error) => chunk_failure(error),
            ExecutionError::EmptySource => BatchItemFailure::reject(BatchCode::DocumentLimit),
            ExecutionError::WorkBudget => BatchItemFailure::reject(BatchCode::WorkLimit),
            ExecutionError::ValueBudget | ExecutionError::LiveValueBudget | ExecutionError::TotalValueBudget
                | ExecutionError::ResultBudget => BatchItemFailure::reject(BatchCode::OutputLineLimit),
            ExecutionError::InvalidLimits | ExecutionError::InvalidPolicy => BatchItemFailure::fatal(BatchCode::Admission),
            ExecutionError::AllocationRefused => BatchItemFailure::fatal(BatchCode::Allocation),
            ExecutionError::Serialization => BatchItemFailure::fatal(BatchCode::Serialization),
            _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        },
    }
}
fn chunk_failure(error: MapReduceError) -> BatchItemFailure {
    match error {
        MapReduceError::InputBudget => BatchItemFailure::reject(BatchCode::DocumentLimit),
        MapReduceError::ChunkBudget | MapReduceError::TokenBudget | MapReduceError::TokenizerBudget
            => BatchItemFailure::reject(BatchCode::WorkLimit),
        MapReduceError::AllocationRefused => BatchItemFailure::fatal(BatchCode::Allocation),
        MapReduceError::InvalidLimits => BatchItemFailure::fatal(BatchCode::Admission),
        // Real map checkpoints restore their typed cause before this boundary.
        MapReduceError::Cancelled => BatchItemFailure::fatal(BatchCode::Cancelled),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    }
}
fn source_failure(error: Int4SourceError) -> BatchItemFailure {
    match error {
        Int4SourceError::Cancelled(cause) => BatchItemFailure::fatal(BatchFault::cancelled(cause)),
        Int4SourceError::InvalidResult => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        Int4SourceError::Extraction(error) => match error {
            Int4ExtractError::Identity => BatchItemFailure::fatal(BatchCode::Admission),
            Int4ExtractError::Extraction(error) => extract_failure(error),
            Int4ExtractError::Native(error) => match error {
                Int4JsonError::Decode(error) => json_failure(error),
                Int4JsonError::Native(error) => native_failure(error),
                Int4JsonError::ModelIdentity => BatchItemFailure::fatal(BatchCode::Admission),
                Int4JsonError::WorkMismatch => BatchItemFailure::fatal(BatchCode::InvalidExecution),
            },
        },
        Int4SourceError::Planning(error) => match error {
            SourcePlanningError::InputBudget => BatchItemFailure::reject(BatchCode::DocumentLimit),
            SourcePlanningError::ContextBudget => BatchItemFailure::reject(BatchCode::WorkLimit),
            SourcePlanningError::OutputBudget => BatchItemFailure::reject(BatchCode::OutputLineLimit),
            SourcePlanningError::Extraction(error) | SourcePlanningError::Ner(NerError::Extraction(error)) => extract_failure(error),
            SourcePlanningError::Ner(NerError::OutputBudgetExceeded) => BatchItemFailure::reject(BatchCode::OutputLineLimit),
            SourcePlanningError::AllocationRefused | SourcePlanningError::Ner(NerError::AllocationRefused)
                => BatchItemFailure::fatal(BatchCode::Allocation),
            SourcePlanningError::Serialization | SourcePlanningError::Ner(NerError::Serialization)
                => BatchItemFailure::fatal(BatchCode::Serialization),
            SourcePlanningError::Contract(_) | SourcePlanningError::Ner(NerError::InvalidOptions | NerError::UnknownType)
                => BatchItemFailure::fatal(BatchCode::Admission),
            _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        },
    }
}
fn extract_failure(error: ExtractError) -> BatchItemFailure {
    match error {
        ExtractError::Decode(error) => json_failure(error),
        ExtractError::OutputBudgetExceeded => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        ExtractError::AllocationRefused => BatchItemFailure::fatal(BatchCode::Allocation),
        ExtractError::Serialization => BatchItemFailure::fatal(BatchCode::Serialization),
        ExtractError::Contract(_) | ExtractError::Schema(_) => BatchItemFailure::fatal(BatchCode::Admission),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    }
}
fn json_failure(error: JsonDecodeError) -> BatchItemFailure {
    match error {
        JsonDecodeError::Cancelled(cause) => BatchItemFailure::fatal(BatchFault::cancelled(cause)),
        JsonDecodeError::BudgetExceeded(_) | JsonDecodeError::Mask(MaskOracleError::WorkCounterOverflow
            | MaskOracleError::WorkBudgetExceeded { .. }) => BatchItemFailure::reject(BatchCode::WorkLimit),
        JsonDecodeError::NoLegalToken => BatchItemFailure::reject(BatchCode::Execution),
        JsonDecodeError::AllocationRefused => BatchItemFailure::fatal(BatchCode::Allocation),
        JsonDecodeError::InvalidRequest(_) | JsonDecodeError::Mask(MaskOracleError::InvalidWorkLimits)
            => BatchItemFailure::fatal(BatchCode::Admission),
        JsonDecodeError::Mask(MaskOracleError::CancelledAtCheckpoint { .. }) => BatchItemFailure::fatal(BatchCode::Cancelled),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    }
}
fn native_failure(error: StrictInt4Error) -> BatchItemFailure {
    match error {
        StrictInt4Error::Cancelled(cause) => BatchItemFailure::fatal(BatchFault::cancelled(cause)),
        StrictInt4Error::Work => BatchItemFailure::reject(BatchCode::WorkLimit),
        StrictInt4Error::Context | StrictInt4Error::Memory => BatchItemFailure::reject(BatchCode::Admission),
        StrictInt4Error::Allocation => BatchItemFailure::fatal(BatchCode::Allocation),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    }
}

#[cfg(test)] mod tests;
