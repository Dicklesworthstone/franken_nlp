//! Complete-document redaction using native grouped-INT4 NER maps.
//! Rules inspect each whole source. Verification freshly partitions the edited
//! bytes; native work, masks and both independent grounding pools never renew.
use std::{error::Error, fmt};
use serde::Serialize;
use crate::{
    canonjson, execution_identity::{ExecutionIdentity, Sha256Digest},
    native_engine::{constrained_int4, decode::{DecodeCancellationKind, DecodeStepControl},
        strict_int4::{Int4ModelIdentity, Int4Work, StrictInt4Engine, STRICT_INT4_PROFILE}},
    tasks::{extract::ExtractionVocabulary, ir::{PlanContext, TaskBudget}, ner::{NerOptions, NER_TASK_VERSION},
        mapreduce::CHUNK_PROFILE, source_planning::{SourcePlanningLimits, SourceTaskPlanner, SourceTaskResult,
            grouped_int4::{Int4SourceError, long::{Int4SourceMapError, Int4SourceMapLimits,
                Int4SourceMapPreflight, PreparedInt4SourceMap, SourceMapTask, SourceMapVerification}}}},
    validation::grounded_fields::GroundingBudget,
};
use super::{Int4RedactionError, check_engine, check_model};
use super::super::{RedactError, RedactionRequest, actions::{self, RedactionResult, VerificationStatus},
    long::detection, pipeline::LeakReport, pseudonym::Pseudonyms, union::{DetectedDocument, NerProfile}};

pub const INT4_LONG_REDACTION_EXECUTION: &str = "grouped-int4-whole-rules-chunk-ner-redetect-v1";
const DETECTOR_SCOPE: &str = "whole-source-rules-independent-ner-chunks-v1";
const WARNINGS: [&str; 3] = ["ner_chunk_boundaries_may_split_entities", "detector_recall_not_established",
    "clean_scan_and_pseudonyms_are_not_anonymization"];
const ZERO_VERIFICATION: SourceMapVerification = SourceMapVerification { fields: 0, matches: 0, scan_steps: 0 };

/// mapping's native, mask and coordinate-verification ceilings span BOTH stages.
/// Chunk/reduction/rule limits apply per stage. The request's grounding budget
/// separately bounds destructive-offset rescans across every chunk and stage.
#[derive(Clone, Debug)]
pub struct Int4LongRedactionConfig {
    pub ner: NerOptions,
    pub per_chunk: TaskBudget,
    pub planning: SourcePlanningLimits,
    pub mapping: Int4SourceMapLimits,
    pub max_result_bytes: u64,
}
pub enum Int4LongRedactionError { Map(Int4SourceMapError), Redaction(Int4RedactionError) }
impl fmt::Display for Int4LongRedactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self { Self::Map(_) => "int4 document redaction NER mapping failed",
            Self::Redaction(_) => "int4 document redaction validation or editing failed" })
    }
}
impl fmt::Debug for Int4LongRedactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int4LongRedactionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Map(e) => Some(e), Self::Redaction(e) => Some(e) }
    }
}
impl From<Int4SourceMapError> for Int4LongRedactionError { fn from(e: Int4SourceMapError) -> Self { Self::Map(e) } }
impl From<Int4SourceError> for Int4LongRedactionError { fn from(e: Int4SourceError) -> Self { Self::Map(e.into()) } }
impl From<Int4RedactionError> for Int4LongRedactionError { fn from(e: Int4RedactionError) -> Self { Self::Redaction(e) } }
impl From<RedactError> for Int4LongRedactionError { fn from(e: RedactError) -> Self { Self::Redaction(e.into()) } }
impl Int4LongRedactionError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Map(e) => e.cancellation(), Self::Redaction(e) => e.cancellation() }
    }
}

/// Exact original-stage plans, not execution proof or a promise that the
/// unknown transformed source will fit the remaining complete-operation cap.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct Int4LongRedactionPreflight {
    pub source_bytes: usize,
    pub source_scalars: usize,
    pub chunks: usize,
    pub group_width: usize,
    pub planned_model_work: Int4Work,
    pub reserved_mask_node_visits: u64,
}
#[derive(Serialize)]
pub struct Int4LongRedactionStage {
    pub preflight: Int4LongRedactionPreflight,
    pub model_work: Int4Work,
    pub mask_node_visit_charge: u64,
    pub map_verification_used: SourceMapVerification,
    pub detection_verification_used: SourceMapVerification,
}
#[derive(Serialize)]
pub struct Int4LongRedactionRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub group_width: usize,
    pub detector_scope: &'static str,
    pub result: RedactionResult,
    pub original: Int4LongRedactionStage,
    pub verification: Option<Int4LongRedactionStage>,
    pub reserved_model_work: Int4Work,
    pub model_work: Int4Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub map_verification_used: SourceMapVerification,
    pub detection_verification_used: SourceMapVerification,
    /// Destructive-offset rescans only; map verification is charged separately.
    pub verification_scan_steps: u64,
    pub warnings: [&'static str; 3],
}

/// Public construction seals the actual resident model and group partition.
/// No caller-defined driver or unverified NER receipt can reach native edits.
pub struct Int4DocumentRedactor<'p> {
    planner: &'p SourceTaskPlanner,
    identity: ExecutionIdentity,
    config: Int4LongRedactionConfig,
    group_width: usize,
}
impl<'p> Int4DocumentRedactor<'p> {
    pub fn new(planner: &'p SourceTaskPlanner, identity: ExecutionIdentity,
        config: Int4LongRedactionConfig, model: &Int4ModelIdentity) -> Result<Self, Int4LongRedactionError> {
        check_model(&identity, model)?;
        Self::from_bound_parts(planner, identity, config, model.group_width())
    }
    // Private model-free fixtures cannot bypass public actual-model binding.
    fn from_bound_parts(planner: &'p SourceTaskPlanner, identity: ExecutionIdentity,
        config: Int4LongRedactionConfig, group_width: usize) -> Result<Self, Int4LongRedactionError> {
        config.ner.validate().map_err(|_| RedactError::InvalidOptions)?;
        config.per_chunk.validate().map_err(|_| RedactError::InvalidOptions)?;
        config.mapping.chunks.effective_token_limit().map_err(Int4SourceMapError::Chunk)?;
        constrained_int4::check_profile(&identity, group_width).map_err(|_| Int4RedactionError::Identity)?;
        if identity.task_spec != NER_TASK_VERSION || identity.tokenizer_digest != planner.tokenizer_digest()
            || identity.template_digest != *planner.template_digest() { return Err(Int4RedactionError::Identity.into()); }
        let mapping = config.mapping;
        if !(1..=256).contains(&mapping.chunks.max_chunks) || mapping.mask_visits_per_chunk == 0
            || mapping.max_mask_visits < mapping.mask_visits_per_chunk
            || mapping.mask_limits.max_trie_node_visits == 0 || mapping.mask_limits.checkpoint_interval_nodes == 0
            || !(1..=1_000_000).contains(&mapping.verification.max_fields)
            || !(1..=1_000_000).contains(&mapping.verification.max_matches) || mapping.verification.max_scan_steps == 0
            || !(1..=64 * 1024 * 1024).contains(&config.max_result_bytes) {
            return Err(RedactError::InvalidOptions.into());
        }
        Ok(Self { planner, identity, config, group_width })
    }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn preflight<C: DecodeStepControl>(&self, source: &str, control: &mut C)
        -> Result<Int4LongRedactionPreflight, Int4LongRedactionError> {
        Ok(metadata(self.prepare(source, self.config.mapping, control)?.preflight_metadata()))
    }
    fn prepare<'s, C: DecodeStepControl>(&self, source: &'s str, mapping: Int4SourceMapLimits, control: &mut C)
        -> Result<PreparedInt4SourceMap<'s>, Int4LongRedactionError> {
        poll(control)?;
        let context = PlanContext::new(&self.identity, self.config.per_chunk).map_err(|_| Int4RedactionError::Identity)?;
        // The actual INT4 planner already constrains chunks using its pinned
        // task scaffold; it precompiles every chunk before the first forward.
        Ok(self.planner.plan_int4_map_with_control(source, &SourceMapTask::Ner(self.config.ner.clone()),
            self.config.per_chunk, &context, self.config.planning, mapping, self.group_width, control)?)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn redact<C: DecodeStepControl>(&self, source: &str, request: &RedactionRequest,
        pseudonyms: Option<&Pseudonyms<'_>>, engine: &mut StrictInt4Engine<'_>,
        vocabulary: &ExtractionVocabulary, control: &mut C) -> Result<Int4LongRedactionRun, Int4LongRedactionError> {
        poll(control)?;
        check_engine(&self.identity, self.group_width, engine, self.config.per_chunk.max_kv_bytes)?;
        self.run_pipeline(source, request, pseudonyms, &mut NativeDetection { engine, vocabulary, control })
    }
    fn run_pipeline<D: DetectionDriver>(&self, source: &str, request: &RedactionRequest,
        pseudonyms: Option<&Pseudonyms<'_>>, driver: &mut D) -> Result<Int4LongRedactionRun, Int4LongRedactionError> {
        driver.checkpoint()?;
        request.actions.check_key(pseudonyms)?;
        request.rules.validate()?;
        if source.is_empty() || source.len() > request.rule_budget.max_input_bytes
            || request.edit_budget.max_output_bytes as u64 > self.config.max_result_bytes {
            return Err(RedactError::InputBudget.into());
        }
        let mut accounting = Accounting::default();
        let mut grounding = request.grounding_budget;
        let (document, original) = driver.detect(self, source, request, &mut accounting, &mut grounding)?;
        driver.checkpoint()?;
        let mut result = actions::apply(&document, &request.actions, pseudonyms, request.edit_budget)?;
        drop(document);
        driver.checkpoint()?;
        let verification = if request.verify {
            // Actual edited bytes are freshly tokenized and partitioned with
            // the same model, type scope, rules, key and remaining allowances.
            let (residual, stage) = driver.detect(self, &result.text, request, &mut accounting, &mut grounding)?;
            require_clean(residual, self.config.max_result_bytes)?;
            result.verification = VerificationStatus::CleanDeclaredUnion;
            Some(stage)
        } else { None };
        driver.checkpoint()?;
        if accounting.failed || accounting.current.is_some() || accounting.completed != 1 + usize::from(request.verify) {
            return Err(Int4RedactionError::InvalidResult.into());
        }
        let detection_used = verification_used(request.grounding_budget, grounding)?;
        if detection_used != accounting.detection_used { return Err(Int4RedactionError::InvalidResult.into()); }
        // Only public policy and partition choices are exported, never private
        // source, chunk/prompt identities, native transcripts or key bytes.
        result.policy_digest = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT4_LONG_REDACTION_EXECUTION, CHUNK_PROFILE, result.policy_digest, request,
            &self.config.ner, self.config.per_chunk, self.config.mapping.chunks, self.group_width,
        )).map_err(|_| RedactError::Serialization)?);
        result.check_size(request.edit_budget.max_output_bytes)?;
        let output = Int4LongRedactionRun {
            schema_version: 1, execution: INT4_LONG_REDACTION_EXECUTION, numerics_profile: STRICT_INT4_PROFILE,
            group_width: self.group_width, detector_scope: DETECTOR_SCOPE, result, original, verification,
            reserved_model_work: accounting.reserved, model_work: accounting.actual,
            reserved_mask_node_visits: accounting.masks_reserved, mask_node_visit_charge: accounting.masks_actual,
            map_verification_used: accounting.map_used, detection_verification_used: detection_used,
            verification_scan_steps: detection_used.scan_steps, warnings: WARNINGS,
        };
        crate::tasks::extract::quantized::check_size(&output, self.config.max_result_bytes).map_err(Int4SourceError::from)?;
        driver.checkpoint()?;
        Ok(output)
    }
    #[allow(clippy::too_many_arguments)]
    fn detect<'s, C: DecodeStepControl>(&self, source: &'s str, request: &RedactionRequest,
        engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary, accounting: &mut Accounting,
        grounding: &mut GroundingBudget, control: &mut C)
        -> Result<(DetectedDocument<'s>, Int4LongRedactionStage), Int4LongRedactionError> {
        poll(control)?;
        let mapping = accounting.begin(self.config.mapping)?;
        // Rules inspect the WHOLE current source before neural chunking, so
        // rule matches crossing model chunk boundaries remain visible.
        let mut detections = detection::Detections::with_profile(source, request, &self.config.ner, NerProfile::Int4)?;
        let before = *grounding;
        let prepared = self.prepare(source, mapping, control)?;
        let proof = prepared.preflight_metadata();
        let preflight = metadata(proof);
        accounting.reserve(preflight, self.config.mapping)?;
        let mut admitted = Vec::new();
        admitted.try_reserve_exact(prepared.chunk_count()).map_err(|_| RedactError::AllocationRefused)?;
        for identity in prepared.execution_identities() { poll(control)?; admitted.push(identity.clone()); }
        let mapped = prepared.execute_with_control(&admitted, engine, vocabulary, control)?;
        // The sealed map proof checks complete source coverage, every actual
        // group/work receipt, native result counts and independent map evidence.
        proof.verify_completed(&mapped)?;
        check_engine(&self.identity, self.group_width, engine, self.config.per_chunk.max_kv_bytes)?;
        for chunk in mapped.mapped.root().value().chunks() {
            poll(control)?;
            let SourceTaskResult::Ner(result) = &chunk.native.result else { return Err(RedactError::InvalidNerEvidence.into()); };
            // Recheck chunk-local occurrences independently before lifting the
            // offsets that authorize edits; do not trust serialized map lifts.
            detections.push(chunk.chunk_id, chunk.source_span, result, grounding)?;
        }
        let document = detections.finish(preflight.chunks)?;
        let stage = Int4LongRedactionStage {
            preflight, model_work: mapped.model_work, mask_node_visit_charge: mapped.mask_node_visit_charge,
            map_verification_used: mapped.verification_used, detection_verification_used: verification_used(before, *grounding)?,
        };
        accounting.finish(&stage, self.config.mapping)?;
        drop(mapped);
        poll(control)?;
        Ok((document, stage))
    }
}

// Only this module's private fixtures replace native detection. Production
// always uses the real sealed map planner, native driver and complete proof.
trait DetectionDriver {
    fn checkpoint(&mut self) -> Result<(), Int4LongRedactionError>;
    fn detect<'s>(&mut self, recipe: &Int4DocumentRedactor<'_>, source: &'s str,
        request: &RedactionRequest, accounting: &mut Accounting, grounding: &mut GroundingBudget)
        -> Result<(DetectedDocument<'s>, Int4LongRedactionStage), Int4LongRedactionError>;
}
struct NativeDetection<'a, 'w, C> {
    engine: &'a mut StrictInt4Engine<'w>, vocabulary: &'a ExtractionVocabulary, control: &'a mut C,
}
impl<C: DecodeStepControl> DetectionDriver for NativeDetection<'_, '_, C> {
    fn checkpoint(&mut self) -> Result<(), Int4LongRedactionError> { poll(self.control) }
    fn detect<'s>(&mut self, recipe: &Int4DocumentRedactor<'_>, source: &'s str,
        request: &RedactionRequest, accounting: &mut Accounting, grounding: &mut GroundingBudget)
        -> Result<(DetectedDocument<'s>, Int4LongRedactionStage), Int4LongRedactionError> {
        recipe.detect(source, request, self.engine, self.vocabulary, accounting, grounding, self.control)
    }
}
fn metadata(proof: Int4SourceMapPreflight) -> Int4LongRedactionPreflight {
    let source = proof.source_span();
    Int4LongRedactionPreflight { source_bytes: source.byte_end, source_scalars: source.scalar_end,
        chunks: proof.chunk_count(), group_width: proof.group_width(), planned_model_work: proof.planned_work(),
        reserved_mask_node_visits: proof.reserved_mask_visits() }
}
fn require_clean(document: DetectedDocument<'_>, cap: u64) -> Result<(), Int4LongRedactionError> {
    if document.regions.is_empty() { return Ok(()); }
    let report = LeakReport { schema_version: 1, residuals: document.regions,
        rules: document.rule_set, model_types: document.model_types };
    crate::tasks::extract::quantized::check_size(&report, cap).map_err(Int4SourceError::from)?;
    Err(Int4RedactionError::Residual(report).into())
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), Int4LongRedactionError> {
    if let Some(cause) = control.prefill_checkpoint(0) { return Err(Int4RedactionError::Cancelled(cause).into()); }
    Ok(())
}
struct Accounting {
    reserved: Int4Work, actual: Int4Work, masks_reserved: u64, masks_actual: u64,
    map_used: SourceMapVerification, detection_used: SourceMapVerification,
    current: Option<Int4LongRedactionPreflight>, completed: usize, failed: bool,
}
impl Default for Accounting {
    fn default() -> Self {
        Self { reserved: Int4Work::default(), actual: Int4Work::default(), masks_reserved: 0, masks_actual: 0,
            map_used: ZERO_VERIFICATION, detection_used: ZERO_VERIFICATION, current: None, completed: 0, failed: false }
    }
}
impl Accounting {
    fn begin(&mut self, mut limits: Int4SourceMapLimits) -> Result<Int4SourceMapLimits, Int4LongRedactionError> {
        if self.failed || self.current.is_some() || self.completed >= 2 { return Err(Int4RedactionError::InvalidResult.into()); }
        // Any failed planning/execution/validation leaves the transaction closed
        // to retries. A new verification stage gets only unspent allowances.
        self.failed = true;
        limits.max_model_work = subtract(limits.max_model_work, self.reserved)?;
        limits.max_mask_visits = limits.max_mask_visits.checked_sub(self.masks_reserved).ok_or(Int4SourceMapError::WorkLimit)?;
        limits.verification = remaining_verification(limits.verification, self.map_used)?;
        // An exhausted original-stage allowance is a document work refusal,
        // not a newly invalid configuration that would poison a corpus host.
        if self.completed != 0 && (limits.max_model_work.forward_positions == 0 || limits.max_model_work.projected_logits == 0
            || limits.max_model_work.attention_pairs == 0 || limits.max_model_work.projections.dot_products == 0
            || limits.max_model_work.projections.multiply_accumulates == 0
            || limits.max_mask_visits < limits.mask_visits_per_chunk
            || limits.verification.max_fields == 0 || limits.verification.max_matches == 0
            || limits.verification.max_scan_steps == 0) { return Err(Int4SourceMapError::WorkLimit.into()); }
        Ok(limits)
    }
    fn reserve(&mut self, stage: Int4LongRedactionPreflight, limits: Int4SourceMapLimits) -> Result<(), Int4LongRedactionError> {
        if !self.failed || self.current.is_some() { return Err(Int4RedactionError::InvalidResult.into()); }
        let work = self.reserved.checked_add(stage.planned_model_work).map_err(|_| Int4SourceMapError::WorkLimit)?;
        subtract(limits.max_model_work, work)?;
        let masks = self.masks_reserved.checked_add(stage.reserved_mask_node_visits)
            .filter(|&n| n <= limits.max_mask_visits).ok_or(Int4SourceMapError::WorkLimit)?;
        self.reserved = work; self.masks_reserved = masks; self.current = Some(stage);
        Ok(())
    }
    fn finish(&mut self, stage: &Int4LongRedactionStage, limits: Int4SourceMapLimits) -> Result<(), Int4LongRedactionError> {
        if !self.failed || self.current != Some(stage.preflight) { return Err(Int4RedactionError::InvalidResult.into()); }
        subtract(stage.preflight.planned_model_work, stage.model_work)?;
        if stage.mask_node_visit_charge > stage.preflight.reserved_mask_node_visits { return Err(Int4RedactionError::InvalidResult.into()); }
        let actual = self.actual.checked_add(stage.model_work).map_err(|_| Int4SourceMapError::WorkLimit)?;
        subtract(self.reserved, actual)?;
        let masks = self.masks_actual.checked_add(stage.mask_node_visit_charge)
            .filter(|&n| n <= self.masks_reserved).ok_or(Int4SourceMapError::WorkLimit)?;
        let map_used = add_verification(self.map_used, stage.map_verification_used)?;
        remaining_verification(limits.verification, map_used)?;
        let detection_used = add_verification(self.detection_used, stage.detection_verification_used)?;
        self.actual = actual; self.masks_actual = masks; self.map_used = map_used; self.detection_used = detection_used;
        self.current = None; self.completed += 1; self.failed = false;
        Ok(())
    }
}
fn subtract(mut cap: Int4Work, used: Int4Work) -> Result<Int4Work, Int4LongRedactionError> {
    cap.forward_positions = cap.forward_positions.checked_sub(used.forward_positions).ok_or(Int4SourceMapError::WorkLimit)?;
    cap.projected_logits = cap.projected_logits.checked_sub(used.projected_logits).ok_or(Int4SourceMapError::WorkLimit)?;
    cap.attention_pairs = cap.attention_pairs.checked_sub(used.attention_pairs).ok_or(Int4SourceMapError::WorkLimit)?;
    cap.projections.dot_products = cap.projections.dot_products.checked_sub(used.projections.dot_products).ok_or(Int4SourceMapError::WorkLimit)?;
    cap.projections.multiply_accumulates = cap.projections.multiply_accumulates.checked_sub(used.projections.multiply_accumulates)
        .ok_or(Int4SourceMapError::WorkLimit)?;
    Ok(cap)
}
fn verification_used(total: GroundingBudget, left: GroundingBudget) -> Result<SourceMapVerification, Int4LongRedactionError> {
    Ok(SourceMapVerification {
        fields: total.max_fields.checked_sub(left.max_fields).ok_or(Int4RedactionError::InvalidResult)?,
        matches: total.max_matches.checked_sub(left.max_matches).ok_or(Int4RedactionError::InvalidResult)?,
        scan_steps: total.max_scan_steps.checked_sub(left.max_scan_steps).ok_or(Int4RedactionError::InvalidResult)?,
    })
}
fn remaining_verification(total: GroundingBudget, used: SourceMapVerification) -> Result<GroundingBudget, Int4LongRedactionError> {
    Ok(GroundingBudget {
        max_fields: total.max_fields.checked_sub(used.fields).ok_or(Int4SourceMapError::WorkLimit)?,
        max_matches: total.max_matches.checked_sub(used.matches).ok_or(Int4SourceMapError::WorkLimit)?,
        max_scan_steps: total.max_scan_steps.checked_sub(used.scan_steps).ok_or(Int4SourceMapError::WorkLimit)?,
    })
}
fn add_verification(a: SourceMapVerification, b: SourceMapVerification) -> Result<SourceMapVerification, Int4LongRedactionError> {
    Ok(SourceMapVerification {
        fields: a.fields.checked_add(b.fields).ok_or(Int4SourceMapError::WorkLimit)?,
        matches: a.matches.checked_add(b.matches).ok_or(Int4SourceMapError::WorkLimit)?,
        scan_steps: a.scan_steps.checked_add(b.scan_steps).ok_or(Int4SourceMapError::WorkLimit)?,
    })
}

#[cfg(test)] mod tests;
