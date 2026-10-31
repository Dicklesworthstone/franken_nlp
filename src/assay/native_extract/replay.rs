//! Closed native ownership, complete-result verification and field summaries.
use serde::Serialize;
use sha2::{Digest, Sha256};
use crate::{assay::field_metrics::FieldCounts,
    batch::{extract::{ExtractionBatchGrounding, grouped_int4::PreparedInt4BatchExtraction}},
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{constrained_int4, lmhead::NANBEIGE_VOCAB_SIZE,
        strict_int4::{Int4ModelIdentity, Int4Work, STRICT_INT4_PROFILE}},
    tasks::{extract::{ExtractionGrounding, grouped_int4::{Int4ExtractRun, INT4_EXTRACT_VERSION}},
        ir::{ScoreSpace, TaskBudget}},
    validation::grounded_fields::GroundingBudget};
use super::*;

/// Only the native host can construct this evidence. No public deserializer,
/// supplied-output callback or replay constructor can authenticate predictions.
pub(crate) struct NativeReplay {
    pub(super) manifest: SuiteManifest,
    pub(super) manifest_digest: Sha256Digest,
    pub(super) input_digest: Sha256Digest,
    pub(super) sizes: dataset::InputSizes,
    pub(super) extract_limits: ExtractLimits,
    pub(super) rows: Vec<ScoredRow>,
    pub(super) first_identity: ExecutionIdentity,
    pub(super) executions: [u8; 32],
    pub(super) compatible_executions: [u8; 32],
    pub(super) observations: [u8; 32],
    pub(super) model_work: Int4Work,
    pub(super) group_width: usize,
    pub(super) mask_node_visits: u64,
    pub(super) verification_steps: u64,
    /// Native twice plus independent collector once, excluding gold.
    pub(super) total_source_verification_steps: u64,
    pub(super) gold_verification_steps: u64,
    pub(super) gold_verification_fields: usize,
    pub(super) gold_verification_matches: usize,
}
pub(super) struct ScoredRow {
    pub id: String,
    pub schema_id: String,
    pub language: String,
    pub unit: Sha256Digest,
    pub counts: FieldCounts,
}
pub(crate) struct Collector<'a> { model: &'a Int4ModelIdentity, builder: ReplayBuilder<'a> }
impl<'a> Collector<'a> {
    pub(crate) fn new(dataset: &'a Dataset, model: &'a Int4ModelIdentity, limits: Limits, extract: ExtractLimits)
        -> Result<Self, AssayError> {
        Ok(Self { model, builder: ReplayBuilder::new(dataset, limits, extract)? })
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn accept_with_checkpoint<E>(&mut self, index: usize, prepared: &PreparedInt4BatchExtraction,
        run: Int4ExtractRun, native_verification_steps: u64, verification: &mut GroundingBudget,
        checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(), E>
    where E: From<AssayError> {
        self.builder.enter(index)?; checkpoint()?;
        let identity = prepared.execution_identity();
        prepared.verify_identity(identity).map_err(|_| AssayError::Identity)?;
        constrained_int4::check_model(identity, self.model).map_err(|_| AssayError::Identity)?;
        let plan = prepared.extraction_plan(); let budget = *prepared.task_plan().ir().budget();
        if plan.group_width() != self.model.group_width()
            || prepared.request_digest() != self.builder.dataset.request_digest(index, budget)? {
            return Err(AssayError::Identity.into());
        }
        self.builder.append(index, identity, prepared.comparison_policy_digest(), budget, plan.prompt_tokens(),
            plan.planned_work(), plan.group_width(), run, native_verification_steps, verification,
            &mut |run| plan.verify_completed(run).map_err(|_| E::from(AssayError::Observation)), checkpoint)?;
        self.builder.failed = false; Ok(())
    }
    pub(crate) fn finish(self) -> Result<NativeReplay, AssayError> { self.builder.finish() }
}

/// Private arithmetic/tamper seam. The fixture entry point exists only under
/// cfg(test); production always supplies the sealed verifier above.
pub(super) struct ReplayBuilder<'a> {
    dataset: &'a Dataset,
    extract_limits: ExtractLimits,
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
    pub(super) fn new(dataset: &'a Dataset, limits: Limits, extract: ExtractLimits) -> Result<Self, AssayError> {
        dataset.validate_replay(limits, extract)?;
        let controls = crate::tokenizer::pinned_controls::pinned().map_err(|_| AssayError::Identity)?;
        let mut ends = controls.template_controls().entries().iter()
            .filter(|entry| entry.special && entry.surface == crate::template::IM_END);
        let eos = ends.next().ok_or(AssayError::Identity)?.id;
        if ends.next().is_some() { return Err(AssayError::Identity); }
        let mut executions = Sha256::new();
        executions.update(b"fnlp.native_extract.execution_collection.v1\0");
        executions.update((dataset.replay_len() as u64).to_be_bytes());
        let mut compatible_executions = Sha256::new();
        compatible_executions.update(b"fnlp.native_extract.pair_compatibility.v1\0");
        compatible_executions.update((dataset.replay_len() as u64).to_be_bytes());
        let mut observations = Sha256::new();
        observations.update(b"fnlp.native_extract.observation_collection.v1\0");
        observations.update((dataset.replay_len() as u64).to_be_bytes());
        Ok(Self { dataset, extract_limits: extract, rows: reserved(dataset.replay_len())?, first_identity: None,
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
    fn append<E>(&mut self, index: usize, identity: &ExecutionIdentity, comparison_policy: Sha256Digest,
        budget: TaskBudget, prompt_tokens: usize, planned: Int4Work, width: usize, run: Int4ExtractRun,
        native_verification_steps: u64, verification: &mut GroundingBudget,
        verify: &mut impl FnMut(&Int4ExtractRun) -> Result<(), E>,
        checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(), E>
    where E: From<AssayError> {
        let row = self.dataset.replay_rows().get(index).ok_or(AssayError::Scope)?;
        let schema = self.dataset.schema_for_row(index)?;
        identity.validate().map_err(|_| AssayError::Identity)?; budget.validate().map_err(|_| AssayError::Limits)?;
        if identity.task_spec != "extract-v1" || identity.numerics_profile != (NumericsProfile::StrictQuantized { version: 2 })
            || identity.thinking_mode != ThinkingMode::Disabled || identity.tool_mode != ToolMode::None
            || width == 0 || self.group_width.is_some_and(|prior| prior != width)
            || run.schema_version != ASSAY_SCHEMA_VERSION || run.execution != INT4_EXTRACT_VERSION
            || run.group_width != width || prompt_tokens > budget.max_input_tokens as usize {
            return Err(AssayError::Identity.into());
        }
        let result = &run.result; let output = &result.output;
        let grounded = schema.grounding == ExtractionBatchGrounding::SourceMembership;
        let expected_grounding = if grounded { ExtractionGrounding::SourceMembership } else { ExtractionGrounding::NotRequested };
        if result.schema_version != if grounded { 2 } else { 1 } || result.task_spec_version != "extract-v1"
            || result.score_space != ScoreSpace::NotComputed || result.grounding != expected_grounding
            || output.schema_version != 1 || output.numerics_profile != STRICT_INT4_PROFILE
            || output.token_ids.is_empty() || output.token_ids.len() > budget.max_output_tokens as usize
            || output.token_ids.last() != Some(&self.eos) || (!grounded && !result.source_fields.is_empty()) {
            return Err(AssayError::Observation.into());
        }
        for chunk in output.token_ids[..output.token_ids.len() - 1].chunks(256) {
            checkpoint()?;
            if chunk.iter().any(|&id| id == self.eos || id as usize >= NANBEIGE_VOCAB_SIZE) {
                return Err(AssayError::Observation.into());
            }
        }
        let expected_maximum = constrained_int4::planned_work(width, prompt_tokens, budget.max_output_tokens as usize)
            .map_err(|_| AssayError::Identity)?;
        let actual = constrained_int4::planned_work(width, prompt_tokens, output.token_ids.len())
            .map_err(|_| AssayError::Observation)?;
        if planned != expected_maximum || run.model_work != actual || output.forward_positions != actual.forward_positions
            || output.projected_logits != actual.projected_logits { return Err(AssayError::Observation.into()); }
        let max_bytes = usize::try_from(budget.max_output_bytes).map_err(|_| AssayError::Limits)?;
        let predicted = dataset::parse_value(&output.json, max_bytes, self.extract_limits)?;
        let observed = crate::canonjson::canonical_bytes(&run).map_err(|_| AssayError::Commitment)?;
        if observed.len() > max_bytes { return Err(AssayError::Output.into()); }
        let charge = dataset::verification_charge(&row.text, &result.source_fields, self.extract_limits)?;
        if charge.steps.checked_mul(2) != Some(native_verification_steps) { return Err(AssayError::Observation.into()); }
        // Host preadmits all three fixed-cap passes. Debit this independent
        // third pass before invoking the sealed complete-result finalizer.
        charge.debit(verification)?; checkpoint()?; verify(&run)?; checkpoint()?;
        let counts = FieldCounts::observe_with_checkpoint(self.dataset.gold(index)?, &predicted,
            self.extract_limits.max_fields_per_row, self.extract_limits.max_json_nodes_per_row,
            self.extract_limits.max_json_depth, checkpoint)?;
        let total_steps = charge.steps.checked_add(native_verification_steps).ok_or(AssayError::InputLimit)?;
        let unit = Sha256Digest::of_bytes(row.text.as_bytes());
        #[derive(Serialize)]
        struct RowBinding<'b> { index: usize, id: &'b str, split: Split, schema_id: &'b str, language: &'b str,
            expected_json: &'b str, source_unit: Sha256Digest, task_budget: TaskBudget }
        let record = crate::canonjson::canonical_bytes(&RowBinding { index, id: &row.id, split: row.split,
            schema_id: &row.schema_id, language: &row.language, expected_json: &row.expected_json,
            source_unit: unit, task_budget: budget }).map_err(|_| AssayError::Commitment)?;
        let full_identity = identity.canonical_json_bytes().map_err(|_| AssayError::Identity)?;
        let compatible = compatible_identity(identity, comparison_policy).canonical_json_bytes().map_err(|_| AssayError::Identity)?;
        if full_identity.len() > 16_384 || compatible.len() > 16_384 { return Err(AssayError::Identity.into()); }
        frame(&mut self.executions, &record); frame(&mut self.executions, &full_identity);
        frame(&mut self.compatible_executions, &record); frame(&mut self.compatible_executions, &compatible);
        frame(&mut self.observations, &record); frame(&mut self.observations, &observed);
        self.model_work = self.model_work.checked_add(actual).map_err(|_| AssayError::InputLimit)?;
        self.mask_node_visits = self.mask_node_visits.checked_add(output.mask_node_visit_charge).ok_or(AssayError::InputLimit)?;
        self.verification_steps = self.verification_steps.checked_add(charge.steps).ok_or(AssayError::InputLimit)?;
        self.total_source_verification_steps = self.total_source_verification_steps.checked_add(total_steps).ok_or(AssayError::InputLimit)?;
        if self.total_source_verification_steps.checked_add(self.dataset.gold_verification_steps())
            .is_none_or(|steps| steps > self.extract_limits.max_verification_steps) { return Err(AssayError::InputLimit.into()); }
        self.rows.push(ScoredRow { id: row.id.clone(), schema_id: row.schema_id.clone(), language: row.language.clone(), unit, counts });
        if self.first_identity.is_none() { self.first_identity = Some(identity.clone()); }
        self.group_width = Some(width); checkpoint()?; Ok(())
    }
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn accept_fixture<E>(&mut self, index: usize, identity: &ExecutionIdentity, budget: TaskBudget,
        prompt_tokens: usize, planned: Int4Work, width: usize, run: Int4ExtractRun,
        native_verification_steps: u64, verification: &mut GroundingBudget,
        checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(), E>
    where E: From<AssayError> {
        self.enter(index)?;
        let dataset = self.dataset;
        let schema = dataset.schema_for_row(index)?; let row = &dataset.replay_rows()[index];
        self.append(index, identity, identity.decision_policy_digest, budget, prompt_tokens, planned, width,
            run, native_verification_steps, verification, &mut |run| {
                let compiled = crate::grammar::runtime::JsonValidationSchema::compile(&schema.schema_json, crate::grammar::CompileLimits::default())
                    .map_err(|_| AssayError::Manifest)?;
                let value = crate::validation::parse_json(&run.result.output.json).map_err(|_| AssayError::Json)?;
                compiled.validate_value(&value).map_err(|_| AssayError::Observation)?;
                if schema.grounding == ExtractionBatchGrounding::SourceMembership {
                    let fields = crate::validation::grounded_fields::verify_source_fields(compiled.root(), &value,
                        &row.text, GroundingBudget::default()).map_err(|_| AssayError::Observation)?;
                    if fields != run.result.source_fields { return Err(AssayError::Observation.into()); }
                }
                Ok(())
            }, checkpoint)?;
        self.failed = false; Ok(())
    }
    pub(super) fn finish(self) -> Result<NativeReplay, AssayError> {
        if self.failed || self.rows.len() != self.dataset.replay_len() { return Err(AssayError::Observation); }
        Ok(NativeReplay { manifest: self.dataset.manifest.clone(), manifest_digest: self.dataset.manifest_digest,
            input_digest: self.dataset.input_digest, sizes: self.dataset.sizes, extract_limits: self.extract_limits,
            rows: self.rows, first_identity: self.first_identity.ok_or(AssayError::Identity)?,
            executions: self.executions.finalize().into(), compatible_executions: self.compatible_executions.finalize().into(),
            observations: self.observations.finalize().into(), model_work: self.model_work,
            group_width: self.group_width.ok_or(AssayError::Identity)?, mask_node_visits: self.mask_node_visits,
            verification_steps: self.verification_steps, total_source_verification_steps: self.total_source_verification_steps,
            gold_verification_steps: self.dataset.gold_verification_steps(), gold_verification_fields: self.dataset.gold_verification_fields(),
            gold_verification_matches: self.dataset.gold_verification_matches() })
    }
}
fn frame(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_be_bytes()); hash.update(bytes);
}
fn compatible_identity(identity: &ExecutionIdentity, comparison_policy: Sha256Digest) -> ExecutionIdentity {
    let mut value = identity.clone(); let absent = Sha256Digest::of_bytes(b"native-assay-comparison-subject-v1");
    value.logical_model_digest = absent; value.packing_set_digest = absent;
    value.artifact_format = "comparison-subject".to_owned(); value.quant_recipe = "comparison-subject".to_owned();
    value.numerics_profile = NumericsProfile::StrictQuantized { version: 2 };
    value.backend_semantic_version = "comparison-subject".to_owned(); value.host_class = None; value.compiler_identity = None;
    value.decision_policy_digest = comparison_policy; value
}
