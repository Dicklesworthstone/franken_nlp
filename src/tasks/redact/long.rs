//! Complete-document redaction: whole-source rules plus bounded native NER maps.
//! All spans are independently checked before one transactional edit. Verification
//! partitions the transformed source afresh; no original NER result is reused.
use std::{error::Error, fmt};
use serde::Serialize;
use crate::{
    canonjson, execution_identity::{ExecutionIdentity, Sha256Digest},
    native_engine::{constrained_int8, decode::{DecodeCancellationKind, DecodeStepControl},
        strict_int8::{Int8Work, StrictInt8Engine, STRICT_INT8_PROFILE}},
    tasks::{extract::ExtractionVocabulary, ir::{PlanContext, TaskBudget}, ner::{NerOptions, NER_TASK_VERSION},
        mapreduce::CHUNK_PROFILE, source_planning::{SourcePlanningLimits, SourceTaskPlanner, SourceTaskResult,
            quantized::{Int8SourceError, long::{Int8SourceMapError, Int8SourceMapLimits, PreparedInt8SourceMap, SourceMapTask}}}},
    validation::grounded_fields::GroundingBudget,
};
use super::{RedactError, RedactionRequest, actions::{self, RedactionResult, VerificationStatus},
    pipeline::LeakReport, pseudonym::Pseudonyms, quantized::Int8RedactionError, union::DetectedDocument};
mod detection;

pub const LONG_REDACTION_EXECUTION: &str = "portable-int8-whole-rules-chunk-ner-redetect-v1";

/// mapping's native/mask ceilings cover BOTH complete stages together. Chunk,
/// serialized-map and rule limits apply per stage. Independent grounding work
/// comes from RedactionRequest and is shared across both stages without renewal.
#[derive(Clone, Debug)]
pub struct LongRedactionConfig {
    pub ner: NerOptions,
    pub per_chunk: TaskBudget,
    pub planning: SourcePlanningLimits,
    pub mapping: Int8SourceMapLimits,
    pub max_result_bytes: u64,
}

pub enum LongRedactionError { Map(Int8SourceMapError), Redaction(Int8RedactionError) }
impl fmt::Display for LongRedactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self { Self::Map(_) => "document redaction NER mapping failed",
            Self::Redaction(_) => "document redaction validation or editing failed" })
    }
}
impl fmt::Debug for LongRedactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for LongRedactionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Map(e) => Some(e), Self::Redaction(e) => Some(e) }
    }
}
impl From<Int8SourceMapError> for LongRedactionError { fn from(e: Int8SourceMapError) -> Self { Self::Map(e) } }
impl From<Int8SourceError> for LongRedactionError { fn from(e: Int8SourceError) -> Self { Self::Map(e.into()) } }
impl From<RedactError> for LongRedactionError { fn from(e: RedactError) -> Self { Self::Redaction(e.into()) } }
impl LongRedactionError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Map(e) => e.cancellation(), Self::Redaction(e) => e.cancellation() }
    }
}

/// Exact original-stage geometry/work, not a model-success receipt. The edited
/// source is not yet known, so this does NOT pre-admit the verification stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct LongRedactionPreflight {
    pub source_bytes: usize,
    pub source_scalars: usize,
    pub chunks: usize,
    pub planned_model_work: Int8Work,
    pub reserved_mask_node_visits: u64,
}
#[derive(Serialize)]
pub struct LongRedactionStage {
    pub preflight: LongRedactionPreflight,
    pub model_work: Int8Work,
    pub mask_node_visit_charge: u64,
}
#[derive(Serialize)]
pub struct LongRedactionRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub detector_scope: &'static str,
    pub result: RedactionResult,
    pub original: LongRedactionStage,
    pub verification: Option<LongRedactionStage>,
    pub reserved_model_work: Int8Work,
    pub model_work: Int8Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub verification_scan_steps: u64,
    pub warnings: [&'static str; 3],
}

/// Reuses the existing real source-map planner/driver. No public fake model,
/// alternate tokenizer, runtime, loader or cross-document identity is introduced.
pub struct Int8DocumentRedactor<'p> {
    planner: &'p SourceTaskPlanner,
    identity: ExecutionIdentity,
    config: LongRedactionConfig,
}
impl<'p> Int8DocumentRedactor<'p> {
    pub fn new(planner: &'p SourceTaskPlanner, identity: ExecutionIdentity, config: LongRedactionConfig)
        -> Result<Self, LongRedactionError> {
        config.ner.validate().map_err(|_| RedactError::InvalidOptions)?;
        config.per_chunk.validate().map_err(|_| RedactError::InvalidOptions)?;
        config.mapping.chunks.effective_token_limit().map_err(Int8SourceMapError::Chunk)?;
        identity.validate().map_err(|_| RedactError::InvalidOptions)?;
        constrained_int8::check_profile(&identity).map_err(|_| RedactError::InvalidOptions)?;
        if identity.task_spec != NER_TASK_VERSION || identity.tokenizer_digest != planner.tokenizer_digest()
            || identity.template_digest != *planner.template_digest()
            || config.mapping.chunks.max_chunks > 256 || config.mapping.mask_visits_per_chunk == 0
            || config.mapping.max_mask_visits < config.mapping.mask_visits_per_chunk
            || config.mapping.mask_limits.max_trie_node_visits == 0
            || config.mapping.mask_limits.checkpoint_interval_nodes == 0
            || !(1..=64 * 1024 * 1024).contains(&config.max_result_bytes) {
            return Err(RedactError::InvalidOptions.into());
        }
        Ok(Self { planner, identity, config })
    }
    pub fn preflight<C: DecodeStepControl>(&self, source: &str, control: &mut C)
        -> Result<LongRedactionPreflight, LongRedactionError> {
        let prepared = self.prepare(source, self.config.mapping, control)?;
        Ok(metadata(source, &prepared))
    }
    fn prepare<'s, C: DecodeStepControl>(&self, source: &'s str, mut mapping: Int8SourceMapLimits, control: &mut C)
        -> Result<PreparedInt8SourceMap<'s>, LongRedactionError> {
        poll(control)?;
        let task = SourceMapTask::Ner(self.config.ner.clone());
        let context = PlanContext::new(&self.identity, self.config.per_chunk).map_err(|_| RedactError::InvalidOptions)?;
        let capacity = self.planner.int8_map_capacity_with_control(&task, self.config.per_chunk,
            &context, self.config.planning, control)?;
        mapping.chunks = capacity.constrain_chunks(mapping.chunks)?;
        Ok(self.planner.plan_int8_map_with_control(source, &task, self.config.per_chunk, &context,
            self.config.planning, mapping, control)?)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn redact<C: DecodeStepControl>(&self, source: &str, request: &RedactionRequest,
        pseudonyms: Option<&Pseudonyms<'_>>, engine: &mut StrictInt8Engine<'_>,
        vocabulary: &ExtractionVocabulary, control: &mut C) -> Result<LongRedactionRun, LongRedactionError> {
        poll(control)?;
        request.actions.check_key(pseudonyms)?;
        request.rules.validate()?;
        if source.is_empty() || source.len() > request.rule_budget.max_input_bytes
            || request.edit_budget.max_output_bytes as u64 > self.config.max_result_bytes {
            return Err(RedactError::InputBudget.into());
        }
        let mut accounting = Accounting::default();
        let mut grounding = request.grounding_budget;
        let (document, original) = self.detect(source, request, engine, vocabulary, &mut accounting, &mut grounding, control)?;
        poll(control)?;
        let mut result = actions::apply(&document, &request.actions, pseudonyms, request.edit_budget)?;
        drop(document);
        poll(control)?;
        let verification = if request.verify {
            // Re-tokenize and re-partition ACTUAL edited bytes. Replacements may
            // expand or shrink source and create new cross-edit rule matches.
            let (residual, receipt) = self.detect(&result.text, request, engine, vocabulary,
                &mut accounting, &mut grounding, control)?;
            require_clean(residual, self.config.max_result_bytes)?;
            result.verification = VerificationStatus::CleanDeclaredUnion;
            Some(receipt)
        } else { None };
        poll(control)?;
        // Public detector/chunk policy only, never original/prompt/token hashes.
        result.policy_digest = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            LONG_REDACTION_EXECUTION, CHUNK_PROFILE, result.policy_digest, request,
            &self.config.ner, self.config.per_chunk, self.config.mapping.chunks,
        )).map_err(|_| RedactError::Serialization)?);
        result.check_size(request.edit_budget.max_output_bytes)?;
        let output = LongRedactionRun { schema_version: 1, execution: LONG_REDACTION_EXECUTION,
            numerics_profile: STRICT_INT8_PROFILE, detector_scope: "whole-source-rules-independent-ner-chunks-v1",
            result, original, verification, reserved_model_work: accounting.reserved, model_work: accounting.actual,
            reserved_mask_node_visits: accounting.masks_reserved, mask_node_visit_charge: accounting.masks_actual,
            verification_scan_steps: request.grounding_budget.max_scan_steps - grounding.max_scan_steps,
            warnings: ["ner_chunk_boundaries_may_split_entities", "detector_recall_not_established",
                "clean_scan_and_pseudonyms_are_not_anonymization"] };
        crate::tasks::extract::quantized::check_size(&output, self.config.max_result_bytes)
            .map_err(Int8SourceError::from)?;
        poll(control)?;
        Ok(output)
    }
    #[allow(clippy::too_many_arguments)]
    fn detect<'s, C: DecodeStepControl>(&self, source: &'s str, request: &RedactionRequest,
        engine: &mut StrictInt8Engine<'_>, vocabulary: &ExtractionVocabulary, accounting: &mut Accounting,
        grounding: &mut GroundingBudget, control: &mut C)
        -> Result<(DetectedDocument<'s>, LongRedactionStage), LongRedactionError> {
        poll(control)?;
        let mut mapping = self.config.mapping;
        mapping.max_model_work = subtract(mapping.max_model_work, accounting.reserved)?;
        mapping.max_mask_visits = mapping.max_mask_visits.checked_sub(accounting.masks_reserved)
            .ok_or(Int8SourceMapError::WorkLimit)?;
        // Rules run across the WHOLE text, not once per chunk. A URL/email/card
        // crossing a neural chunk boundary is still visible to its rule scanner.
        let mut detections = detection::Detections::new(source, request, &self.config.ner)?;
        poll(control)?;
        let prepared = self.prepare(source, mapping, control)?;
        let preflight = metadata(source, &prepared);
        accounting.reserve(preflight, self.config.mapping)?;
        let mut admitted = Vec::new();
        admitted.try_reserve_exact(prepared.chunk_count()).map_err(|_| RedactError::AllocationRefused)?;
        for identity in prepared.execution_identities() {
            poll(control)?;
            admitted.push(identity.clone());
        }
        // Existing source-map admission checks every identity and full resident
        // KV capacity before its first forward, with one exclusive native engine.
        let mapped = prepared.execute_with_control(&admitted, engine, vocabulary, control)?;
        if mapped.planned_model_work != preflight.planned_model_work
            || mapped.reserved_mask_node_visits != preflight.reserved_mask_node_visits
            || mapped.mapped.root().value().len() != preflight.chunks {
            return Err(RedactError::InvalidNerEvidence.into());
        }
        for chunk in mapped.mapped.root().value().chunks() {
            poll(control)?;
            let SourceTaskResult::Ner(result) = &chunk.native.result else { return Err(RedactError::InvalidNerEvidence.into()); };
            detections.push(chunk.chunk_id, chunk.source_span, result, grounding)?;
        }
        let document = detections.finish(preflight.chunks)?;
        let receipt = LongRedactionStage { preflight, model_work: mapped.model_work,
            mask_node_visit_charge: mapped.mask_node_visit_charge };
        accounting.finish(&receipt)?;
        // All source-map NER transcripts die here, before editing or residual
        // detection. Only independently checked coordinates survive the stage.
        drop(mapped);
        poll(control)?;
        Ok((document, receipt))
    }
}
fn metadata(source: &str, prepared: &PreparedInt8SourceMap<'_>) -> LongRedactionPreflight {
    LongRedactionPreflight { source_bytes: source.len(), source_scalars: source.chars().count(), chunks: prepared.chunk_count(),
        planned_model_work: prepared.planned_work(), reserved_mask_node_visits: prepared.reserved_mask_visits() }
}
fn require_clean(document: DetectedDocument<'_>, cap: u64) -> Result<(), LongRedactionError> {
    if document.regions.is_empty() { return Ok(()); }
    let report = LeakReport { schema_version: 1, residuals: document.regions,
        rules: document.rule_set, model_types: document.model_types };
    crate::tasks::extract::quantized::check_size(&report, cap).map_err(Int8SourceError::from)?;
    Err(LongRedactionError::Redaction(Int8RedactionError::Residual(report)))
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), LongRedactionError> {
    if let Some(cause) = control.prefill_checkpoint(0) {
        return Err(LongRedactionError::Redaction(Int8RedactionError::Cancelled(cause)));
    }
    Ok(())
}
#[derive(Default)]
struct Accounting { reserved: Int8Work, actual: Int8Work, masks_reserved: u64, masks_actual: u64 }
impl Accounting {
    fn reserve(&mut self, stage: LongRedactionPreflight, limits: Int8SourceMapLimits) -> Result<(), LongRedactionError> {
        let reserved = self.reserved.checked_add(stage.planned_model_work).map_err(|_| Int8SourceMapError::WorkLimit)?;
        subtract(limits.max_model_work, reserved)?;
        let masks = self.masks_reserved.checked_add(stage.reserved_mask_node_visits)
            .filter(|&n| n <= limits.max_mask_visits).ok_or(Int8SourceMapError::WorkLimit)?;
        self.reserved = reserved; self.masks_reserved = masks; Ok(())
    }
    fn finish(&mut self, stage: &LongRedactionStage) -> Result<(), LongRedactionError> {
        subtract(stage.preflight.planned_model_work, stage.model_work)?;
        if stage.mask_node_visit_charge > stage.preflight.reserved_mask_node_visits { return Err(RedactError::InvalidNerEvidence.into()); }
        self.actual = self.actual.checked_add(stage.model_work).map_err(|_| Int8SourceMapError::WorkLimit)?;
        self.masks_actual = self.masks_actual.checked_add(stage.mask_node_visit_charge).ok_or(Int8SourceMapError::WorkLimit)?;
        Ok(())
    }
}
fn subtract(mut cap: Int8Work, used: Int8Work) -> Result<Int8Work, LongRedactionError> {
    cap.forward_positions = cap.forward_positions.checked_sub(used.forward_positions).ok_or(Int8SourceMapError::WorkLimit)?;
    cap.projected_logits = cap.projected_logits.checked_sub(used.projected_logits).ok_or(Int8SourceMapError::WorkLimit)?;
    cap.attention_pairs = cap.attention_pairs.checked_sub(used.attention_pairs).ok_or(Int8SourceMapError::WorkLimit)?;
    cap.projections.dot_products = cap.projections.dot_products.checked_sub(used.projections.dot_products).ok_or(Int8SourceMapError::WorkLimit)?;
    cap.projections.multiply_accumulates = cap.projections.multiply_accumulates.checked_sub(used.projections.multiply_accumulates).ok_or(Int8SourceMapError::WorkLimit)?;
    Ok(cap)
}

#[cfg(test)] mod tests;
