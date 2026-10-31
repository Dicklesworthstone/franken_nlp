//! Closed native ownership, independent source verification and span summaries.
use serde::Serialize;
use sha2::{Digest, Sha256};
use crate::{assay::span_metrics::{Prediction, Span, SpanCounts},
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{constrained_int4, lmhead::NANBEIGE_VOCAB_SIZE,
        strict_int4::{Int4ModelIdentity, Int4Work, STRICT_INT4_PROFILE}},
    tasks::{extract::ExtractionGrounding, ir::{ScoreSpace, TaskBudget},
        ner::{NerResult, NER_TASK_VERSION},
        source_planning::{SourceTaskResult, grouped_int4::{Int4SourceTaskRun,
            PreparedInt4SourceTask, INT4_SOURCE_EXECUTION}}},
    validation::grounded_fields::{GroundingBudget, SourceOccurrence, scan_occurrences}};
use super::*;

/// No public constructor, deserializer or callback can turn caller-supplied
/// entity output into native evidence. Only the resident execution host owns
/// Collector, and each result is consumed immediately after its sealed plan.
pub(crate) struct NativeReplay {
    pub(super) manifest: SuiteManifest,
    pub(super) manifest_digest: Sha256Digest,
    pub(super) input_digest: Sha256Digest,
    pub(super) sizes: dataset::InputSizes,
    pub(super) ner_limits: NerLimits,
    pub(super) rows: Vec<ScoredRow>,
    pub(super) first_identity: ExecutionIdentity,
    pub(super) executions: [u8; 32],
    pub(super) compatible_executions: [u8; 32],
    pub(super) observations: [u8; 32],
    pub(super) model_work: Int4Work,
    pub(super) group_width: usize,
    pub(super) mask_node_visits: u64,
    pub(super) verification_steps: u64,
    pub(super) total_source_verification_steps: u64,
}
pub(super) struct ScoredRow {
    pub id: String,
    pub language: String,
    pub unit: Sha256Digest,
    pub counts: SpanCounts,
}
pub(crate) struct Collector<'a> { model: &'a Int4ModelIdentity, builder: ReplayBuilder<'a> }
impl<'a> Collector<'a> {
    pub(crate) fn new(dataset: &'a Dataset, model: &'a Int4ModelIdentity, limits: Limits, ner_limits: NerLimits)
        -> Result<Self, AssayError> {
        Ok(Self { model, builder: ReplayBuilder::new(dataset, limits, ner_limits)? })
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn accept_with_checkpoint<E>(&mut self, index: usize, prepared: &PreparedInt4SourceTask,
        run: Int4SourceTaskRun, native_verification_steps: u64, verification: &mut GroundingBudget,
        checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(), E>
    where E: From<AssayError> {
        self.builder.enter(index)?; checkpoint()?;
        let identity = prepared.execution_identity();
        prepared.verify_identity(identity).map_err(|_| AssayError::Identity)?;
        constrained_int4::check_model(identity, self.model).map_err(|_| AssayError::Identity)?;
        if prepared.group_width() != self.model.group_width()
            || prepared.request_digest() != self.builder.dataset.request_digest(index, prepared.task_budget())? {
            return Err(AssayError::Identity.into());
        }
        self.builder.append(index, identity, prepared.comparison_policy_digest(), prepared.task_budget(), prepared.prompt_tokens(),
            prepared.planned_work(), prepared.group_width(), run, native_verification_steps,
            verification, checkpoint)?;
        self.builder.failed = false;
        Ok(())
    }
    pub(crate) fn finish(self) -> Result<NativeReplay, AssayError> { self.builder.finish() }
}

/// A private cfg(test) entry point supports arithmetic and tamper fixtures;
/// production callers cannot bypass the sealed model/plan boundary above.
pub(super) struct ReplayBuilder<'a> {
    dataset: &'a Dataset,
    ner_limits: NerLimits,
    rows: Vec<ScoredRow>,
    first_identity: Option<ExecutionIdentity>,
    executions: Sha256,
    compatible_executions: Sha256,
    observations: Sha256,
    model_work: Int4Work,
    group_width: Option<usize>,
    mask_node_visits: u64,
    verification_steps: u64,
    total_source_verification_steps: u64,
    eos: u32,
    failed: bool,
}
impl<'a> ReplayBuilder<'a> {
    pub(super) fn new(dataset: &'a Dataset, limits: Limits, ner_limits: NerLimits) -> Result<Self, AssayError> {
        dataset.validate_limits(limits, ner_limits)?;
        let controls = crate::tokenizer::pinned_controls::pinned().map_err(|_| AssayError::Identity)?;
        let mut ends = controls.template_controls().entries().iter()
            .filter(|entry| entry.special && entry.surface == crate::template::IM_END);
        let eos = ends.next().ok_or(AssayError::Identity)?.id;
        if ends.next().is_some() { return Err(AssayError::Identity); }
        let mut executions = Sha256::new();
        executions.update(b"fnlp.native_ner.execution_collection.v1\0");
        executions.update((dataset.replay_len() as u64).to_be_bytes());
        let mut compatible_executions = Sha256::new();
        compatible_executions.update(b"fnlp.native_ner.pair_compatibility.v1\0");
        compatible_executions.update((dataset.replay_len() as u64).to_be_bytes());
        let mut observations = Sha256::new();
        observations.update(b"fnlp.native_ner.observation_collection.v1\0");
        observations.update((dataset.replay_len() as u64).to_be_bytes());
        Ok(Self { dataset, ner_limits, rows: reserved(dataset.replay_len())?, first_identity: None,
            executions, compatible_executions, observations, model_work: Int4Work::default(), group_width: None,
            mask_node_visits: 0, verification_steps: 0, total_source_verification_steps: 0, eos, failed: false })
    }
    fn enter(&mut self, index: usize) -> Result<(), AssayError> {
        if self.failed { return Err(AssayError::Observation); }
        self.failed = true;
        if index != self.rows.len() || index >= self.dataset.replay_len() { return Err(AssayError::Scope); }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn append<E>(&mut self, index: usize, identity: &ExecutionIdentity, comparison_policy: Sha256Digest, budget: TaskBudget,
        prompt_tokens: usize, planned: Int4Work, width: usize, run: Int4SourceTaskRun,
        native_verification_steps: u64, verification: &mut GroundingBudget,
        checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(), E>
    where E: From<AssayError> {
        let row = self.dataset.replay_rows().get(index).ok_or(AssayError::Scope)?;
        identity.validate().map_err(|_| AssayError::Identity)?;
        budget.validate().map_err(|_| AssayError::Limits)?;
        if identity.task_spec != NER_TASK_VERSION || identity.numerics_profile != (NumericsProfile::StrictQuantized { version: 2 })
            || identity.thinking_mode != ThinkingMode::Disabled || identity.tool_mode != ToolMode::None
            || width == 0 || self.group_width.is_some_and(|prior| prior != width)
            || run.schema_version != ASSAY_SCHEMA_VERSION || run.execution != INT4_SOURCE_EXECUTION
            || run.group_width != width || prompt_tokens > budget.max_input_tokens as usize {
            return Err(AssayError::Identity.into());
        }
        let SourceTaskResult::Ner(result) = &run.result else { return Err(AssayError::Observation.into()); };
        if result.schema_version != 1 || result.task_spec_version != NER_TASK_VERSION
            || result.numerics_profile != STRICT_INT4_PROFILE || result.score_space != ScoreSpace::NotComputed
            || result.grounding != ExtractionGrounding::SourceMembership || result.generated_token_ids.len() < 2
            || result.generated_token_ids.len() > budget.max_output_tokens as usize
            || result.generated_token_ids.last() != Some(&self.eos) {
            return Err(AssayError::Observation.into());
        }
        for chunk in result.generated_token_ids[..result.generated_token_ids.len() - 1].chunks(256) {
            checkpoint()?;
            if chunk.iter().any(|&id| id == self.eos || id as usize >= NANBEIGE_VOCAB_SIZE) {
                return Err(AssayError::Observation.into());
            }
        }
        let expected_maximum = constrained_int4::planned_work(width, prompt_tokens, budget.max_output_tokens as usize)
            .map_err(|_| AssayError::Identity)?;
        let actual = constrained_int4::planned_work(width, prompt_tokens, result.generated_token_ids.len())
            .map_err(|_| AssayError::Observation)?;
        if planned != expected_maximum || run.model_work != actual
            || result.forward_positions != actual.forward_positions || result.projected_logits != actual.projected_logits {
            return Err(AssayError::Observation.into());
        }
        // The whole native envelope, including every candidate interval and
        // generated token, receives its own private commitment before release.
        let observed = crate::canonjson::canonical_bytes(&run).map_err(|_| AssayError::Commitment)?;
        if observed.len() as u64 > budget.max_output_bytes { return Err(AssayError::Output.into()); }
        checkpoint()?;
        let before_steps = verification.max_scan_steps;
        let counts = verify_and_count(self.dataset, row, result, self.ner_limits, verification, checkpoint)?;
        let independent_steps = before_steps.checked_sub(verification.max_scan_steps).ok_or(AssayError::InputLimit)?;
        // The constrained result and native extraction finalizer each execute
        // the same independently bounded source scan before this third pass.
        if independent_steps.checked_mul(2) != Some(native_verification_steps) {
            return Err(AssayError::Observation.into());
        }
        let total_steps = independent_steps.checked_add(native_verification_steps).ok_or(AssayError::InputLimit)?;
        let unit = Sha256Digest::of_bytes(row.text.as_bytes());
        #[derive(Serialize)]
        struct RowBinding<'a> { index: usize, id: &'a str, split: Split, language: &'a str,
            spans: &'a [GoldSpan], source_unit: Sha256Digest, task_budget: TaskBudget }
        let record = crate::canonjson::canonical_bytes(&RowBinding { index, id: &row.id, split: row.split,
            language: &row.language, spans: &row.spans, source_unit: unit, task_budget: budget })
            .map_err(|_| AssayError::Commitment)?;
        let full_identity = identity.canonical_json_bytes().map_err(|_| AssayError::Identity)?;
        let compatible = compatible_identity(identity, comparison_policy).canonical_json_bytes().map_err(|_| AssayError::Identity)?;
        if full_identity.len() > 16_384 || compatible.len() > 16_384 { return Err(AssayError::Identity.into()); }
        frame(&mut self.executions, &record); frame(&mut self.executions, &full_identity);
        frame(&mut self.compatible_executions, &record); frame(&mut self.compatible_executions, &compatible);
        frame(&mut self.observations, &record); frame(&mut self.observations, &observed);
        self.model_work = self.model_work.checked_add(actual).map_err(|_| AssayError::InputLimit)?;
        self.mask_node_visits = self.mask_node_visits.checked_add(result.mask_node_visit_charge).ok_or(AssayError::InputLimit)?;
        self.verification_steps = self.verification_steps.checked_add(independent_steps).ok_or(AssayError::InputLimit)?;
        self.total_source_verification_steps = self.total_source_verification_steps.checked_add(total_steps)
            .ok_or(AssayError::InputLimit)?;
        if self.total_source_verification_steps > self.ner_limits.max_verification_steps {
            return Err(AssayError::InputLimit.into());
        }
        self.rows.push(ScoredRow { id: row.id.clone(), language: row.language.clone(), unit, counts });
        if self.first_identity.is_none() { self.first_identity = Some(identity.clone()); }
        self.group_width = Some(width);
        checkpoint()?;
        Ok(())
    }
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn accept_fixture<E>(&mut self, index: usize, identity: &ExecutionIdentity, budget: TaskBudget,
        prompt_tokens: usize, planned: Int4Work, width: usize, run: Int4SourceTaskRun,
        native_verification_steps: u64, verification: &mut GroundingBudget,
        checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(), E>
    where E: From<AssayError> {
        self.enter(index)?;
        self.append(index, identity, identity.decision_policy_digest, budget, prompt_tokens, planned, width, run,
            native_verification_steps, verification, checkpoint)?;
        self.failed = false; Ok(())
    }
    pub(super) fn finish(self) -> Result<NativeReplay, AssayError> {
        if self.failed || self.rows.len() != self.dataset.replay_len() { return Err(AssayError::Observation); }
        Ok(NativeReplay { manifest: self.dataset.manifest.clone(), manifest_digest: self.dataset.manifest_digest,
            input_digest: self.dataset.input_digest, sizes: self.dataset.sizes, ner_limits: self.ner_limits,
            rows: self.rows, first_identity: self.first_identity.ok_or(AssayError::Identity)?,
            executions: self.executions.finalize().into(), compatible_executions: self.compatible_executions.finalize().into(),
            observations: self.observations.finalize().into(), model_work: self.model_work,
            group_width: self.group_width.ok_or(AssayError::Identity)?, mask_node_visits: self.mask_node_visits,
            verification_steps: self.verification_steps, total_source_verification_steps: self.total_source_verification_steps })
    }
}

fn verify_and_count<E>(dataset: &Dataset, row: &Row, result: &NerResult, limits: NerLimits,
    verification: &mut GroundingBudget, checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<SpanCounts, E>
where E: From<AssayError> {
    if result.entities.len() > dataset.manifest.options.max_entities || result.entities.len() > limits.max_spans_per_row {
        return Err(AssayError::InputLimit.into());
    }
    let mut predictions = reserved(result.entities.len())?;
    let mut occurrences = 0_usize;
    for entity in &result.entities {
        checkpoint()?;
        if entity.text.is_empty() || entity.text.chars().count() > dataset.manifest.options.max_mention_scalars {
            return Err(AssayError::Observation.into());
        }
        let kind = dataset.kind_index(entity.entity_type)?;
        occurrences = occurrences.checked_add(entity.spans.len()).ok_or(AssayError::InputLimit)?;
        if occurrences > limits.max_occurrences_per_row { return Err(AssayError::InputLimit.into()); }
        verification.max_fields = verification.max_fields.checked_sub(1).ok_or(AssayError::InputLimit)?;
        let checked = scan_occurrences(&row.text, &entity.text, verification).map_err(|error| match error {
            crate::validation::grounded_fields::FieldGroundingError::AllocationRefused => AssayError::Allocation,
            crate::validation::grounded_fields::FieldGroundingError::FieldBudget
            | crate::validation::grounded_fields::FieldGroundingError::MatchBudget
            | crate::validation::grounded_fields::FieldGroundingError::WorkBudget => AssayError::InputLimit,
            _ => AssayError::Observation,
        })?;
        let expected_occurrence = if checked.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous };
        if entity.spans != checked || entity.occurrence != expected_occurrence {
            return Err(AssayError::Observation.into());
        }
        predictions.push(Prediction { kind, candidate_spans: checked.len(),
            span: if expected_occurrence == SourceOccurrence::Anchored {
                Some((checked[0].byte_start, checked[0].byte_end))
            } else { None } });
    }
    let mut gold = reserved(row.spans.len())?;
    for (index, span) in row.spans.iter().enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        gold.push(Span { kind: dataset.kind_index(span.entity_type)?, byte_start: span.byte_start, byte_end: span.byte_end });
    }
    SpanCounts::observe_with_checkpoint(&gold, &predictions, dataset.manifest.options.types.len(), checkpoint)
}
fn frame(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_be_bytes()); hash.update(bytes);
}
fn compatible_identity(identity: &ExecutionIdentity, comparison_policy: Sha256Digest) -> ExecutionIdentity {
    let mut value = identity.clone();
    let absent = Sha256Digest::of_bytes(b"native-assay-comparison-subject-v1");
    value.logical_model_digest = absent; value.packing_set_digest = absent;
    value.artifact_format = "comparison-subject".to_owned(); value.quant_recipe = "comparison-subject".to_owned();
    value.numerics_profile = NumericsProfile::StrictQuantized { version: 2 };
    value.backend_semantic_version = "comparison-subject".to_owned();
    value.host_class = None; value.compiler_identity = None;
    // The sealed extraction policy preserves all source, schema, task and
    // finalizer bindings while excluding only the compared group partition.
    value.decision_policy_digest = comparison_policy;
    value
}
