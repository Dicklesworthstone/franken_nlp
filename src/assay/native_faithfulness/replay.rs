//! Only the resident execution owner can produce native observations.
use serde::Serialize;
use sha2::{Digest, Sha256};
use crate::{execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        strict_int4::{Int4ModelIdentity, Int4Work, STRICT_INT4_PROFILE}},
    tasks::{ir::TaskBudget, judge::{FaithfulnessAggregationError, FaithfulnessAbstention,
        FaithfulnessRelation, JudgeResult, validate_faithfulness_result,
        grouped_int4::{self, Int4JudgeRun, PreparedInt4Judge, INT4_JUDGE_EXECUTION}}}};
use super::*;

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum ReplayError { Assay(AssayError), Cancelled(DecodeCancellationKind) }
impl From<AssayError> for ReplayError { fn from(value: AssayError) -> Self { Self::Assay(value) } }

pub(crate) struct NativeReplay {
    pub(super) manifest: SuiteManifest,
    pub(super) manifest_digest: Sha256Digest,
    pub(super) input_digest: Sha256Digest,
    pub(super) sizes: dataset::InputSizes,
    pub(super) rows: Vec<ScoredRow>,
    pub(super) first_identity: ExecutionIdentity,
    pub(super) executions: [u8; 32],
    pub(super) compatible_executions: [u8; 32],
    pub(super) observations: [u8; 32],
    pub(super) model_work: Int4Work,
    pub(super) group_width: usize,
    pub(super) heads: usize,
}
pub(super) struct ScoredRow {
    pub id: String,
    pub source: Sha256Digest,
    pub claim: Sha256Digest,
    pub gold: usize,
    pub probability: [f64; 3],
    pub top: usize,
    pub predicted: Option<usize>,
    pub abstention: Option<FaithfulnessAbstention>,
}

pub(crate) struct Collector<'a> { model: &'a Int4ModelIdentity, builder: ReplayBuilder<'a> }
impl<'a> Collector<'a> {
    pub(crate) fn new(dataset: &'a Dataset, model: &'a Int4ModelIdentity, limits: Limits) -> Result<Self, AssayError> {
        Ok(Self { model, builder: ReplayBuilder::new(dataset, limits)? })
    }
    pub(crate) fn accept(&mut self, index: usize, prepared: &PreparedInt4Judge,
        run: Int4JudgeRun, control: &mut impl DecodeStepControl) -> Result<(), ReplayError> {
        self.builder.enter(index)?; checkpoint(control)?;
        let identity = prepared.execution_identity();
        prepared.verify_identity(identity).map_err(|_| AssayError::Identity)?;
        grouped_int4::check_model(identity, self.model).map_err(|_| AssayError::Identity)?;
        if prepared.group_width() != self.model.group_width()
            || prepared.request_digest() != self.builder.dataset.request_digest(index, prepared.task_budget())? {
            return Err(AssayError::Identity.into());
        }
        self.builder.append(index, identity, prepared.task_budget(), prepared.planned_work(),
            prepared.group_width(), prepared.head_count(), prepared.max_result_bytes(), run, control)?;
        self.builder.failed = false; Ok(())
    }
    pub(crate) fn finish(self) -> Result<NativeReplay, AssayError> { self.builder.finish() }
}

/// Arithmetic/tamper fixtures reach this only inside this module's tests.
/// The production constructor and append path never accept external scores.
pub(super) struct ReplayBuilder<'a> {
    dataset: &'a Dataset,
    rows: Vec<ScoredRow>,
    first_identity: Option<ExecutionIdentity>,
    executions: Sha256,
    compatible_executions: Sha256,
    observations: Sha256,
    model_work: Int4Work,
    width: Option<usize>,
    heads: usize,
    eos: u32,
    signature: Option<(u32, [usize; 3])>,
    failed: bool,
}
impl<'a> ReplayBuilder<'a> {
    pub(super) fn new(dataset: &'a Dataset, limits: Limits) -> Result<Self, AssayError> {
        dataset.validate_limits(limits)?;
        let registry = crate::tokenizer::pinned_controls::pinned().map_err(|_| AssayError::Identity)?;
        let mut ends = registry.template_controls().entries().iter()
            .filter(|entry| entry.special && entry.surface == crate::template::IM_END);
        let eos = ends.next().ok_or(AssayError::Identity)?.id;
        if ends.next().is_some() { return Err(AssayError::Identity); }
        let hash = |domain: &[u8]| { let mut value = Sha256::new(); value.update(domain);
            value.update((dataset.replay_len() as u64).to_be_bytes()); value };
        Ok(Self { dataset, rows: reserved(dataset.replay_len())?, first_identity: None,
            executions: hash(b"fnlp.native_faithfulness.executions.v1\0"),
            compatible_executions: hash(b"fnlp.native_faithfulness.pair.v1\0"),
            observations: hash(b"fnlp.native_faithfulness.observations.v1\0"),
            model_work: Int4Work::default(), width: None, heads: 0, eos, signature: None, failed: false })
    }
    fn enter(&mut self, index: usize) -> Result<(), AssayError> {
        if self.failed { return Err(AssayError::Observation); }
        self.failed = true;
        if index != self.rows.len() || index >= self.dataset.replay_len() { return Err(AssayError::Scope); }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn append(&mut self, index: usize, identity: &ExecutionIdentity, budget: TaskBudget,
        planned: Int4Work, width: usize, heads: usize, output_bytes: u64,
        run: Int4JudgeRun, control: &mut impl DecodeStepControl) -> Result<(), ReplayError> {
        checkpoint(control)?;
        let row = self.dataset.replay_rows().get(index).ok_or(AssayError::Scope)?;
        identity.validate().map_err(|_| AssayError::Identity)?;
        budget.validate().map_err(|_| AssayError::Limits)?;
        if identity.task_spec != "judge-v1"
            || identity.numerics_profile != (NumericsProfile::StrictQuantized { version: 2 })
            || identity.thinking_mode != ThinkingMode::Disabled || identity.tool_mode != ToolMode::None
            || width == 0 || self.width.is_some_and(|prior| prior != width)
            || run.schema_version != ASSAY_SCHEMA_VERSION || run.execution != INT4_JUDGE_EXECUTION
            || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != width
            || run.head_count != heads || !(1..=32).contains(&heads)
            || run.model_work != planned || planned.forward_positions == 0 || planned.projected_logits == 0
            || planned.attention_pairs == 0 || planned.projections.dot_products == 0
            || planned.projections.multiply_accumulates == 0 || output_bytes > budget.max_output_bytes {
            return Err(AssayError::Identity.into());
        }
        let JudgeResult::Faithfulness(result) = &run.result else { return Err(AssayError::Observation.into()); };
        if result.heads.len() != heads || result.work.projected_logits != planned.projected_logits {
            return Err(AssayError::Observation.into());
        }
        // The shared independent validator recomputes the exact full source
        // partition, all score semantics, every final decision and every quote.
        validate_faithfulness_result(&row.source, result, self.dataset.manifest.policy,
            &mut self.signature, control).map_err(|error| match error {
                FaithfulnessAggregationError::Cancelled(kind) => ReplayError::Cancelled(kind),
                FaithfulnessAggregationError::Allocation => ReplayError::Assay(AssayError::Allocation),
                _ => ReplayError::Assay(AssayError::Observation),
            })?;
        if self.signature != Some((self.eos, [2; 3])) { return Err(AssayError::Observation.into()); }
        let observed = crate::canonjson::canonical_bytes(&run).map_err(|_| AssayError::Commitment)?;
        if observed.len() as u64 > output_bytes { return Err(AssayError::Output.into()); }
        checkpoint(control)?;
        let source = Sha256Digest::of_bytes(row.source.as_bytes());
        let claim = Sha256Digest::of_bytes(row.claim.as_bytes());
        #[derive(Serialize)]
        struct Binding<'a> { index: usize, id: &'a str, split: Split, source: Sha256Digest,
            claim: Sha256Digest, gold: FaithfulnessRelation, budget: TaskBudget }
        let record = crate::canonjson::canonical_bytes(&Binding { index, id: &row.id, split: row.split,
            source, claim, gold: row.relation, budget }).map_err(|_| AssayError::Commitment)?;
        let full = identity.canonical_json_bytes().map_err(|_| AssayError::Identity)?;
        let compatible = compatible_identity(identity).canonical_json_bytes().map_err(|_| AssayError::Identity)?;
        if full.len() > 16_384 || compatible.len() > 16_384 { return Err(AssayError::Identity.into()); }
        frame(&mut self.executions, &record); frame(&mut self.executions, &full);
        frame(&mut self.compatible_executions, &record); frame(&mut self.compatible_executions, &compatible);
        frame(&mut self.observations, &record); frame(&mut self.observations, &observed);
        self.model_work = self.model_work.checked_add(planned).map_err(|_| AssayError::InputLimit)?;
        self.heads = self.heads.checked_add(heads).ok_or(AssayError::InputLimit)?;
        self.rows.push(ScoredRow { id: row.id.clone(), source, claim, gold: label_index(row.relation),
            probability: std::array::from_fn(|i| result.heads[0].candidates[i].candidate_weight),
            top: label_index(result.global.top_relation), predicted: result.relation.map(label_index),
            abstention: result.abstention });
        if self.first_identity.is_none() { self.first_identity = Some(identity.clone()); }
        self.width = Some(width); checkpoint(control)?; Ok(())
    }
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn accept_fixture(&mut self, index: usize, identity: &ExecutionIdentity, budget: TaskBudget,
        planned: Int4Work, width: usize, heads: usize, run: Int4JudgeRun,
        control: &mut impl DecodeStepControl) -> Result<(), ReplayError> {
        self.enter(index)?;
        self.append(index, identity, budget, planned, width, heads, budget.max_output_bytes, run, control)?;
        self.failed = false; Ok(())
    }
    pub(super) fn finish(self) -> Result<NativeReplay, AssayError> {
        if self.failed || self.rows.len() != self.dataset.replay_len() { return Err(AssayError::Observation); }
        Ok(NativeReplay { manifest: self.dataset.manifest.clone(), manifest_digest: self.dataset.manifest_digest,
            input_digest: self.dataset.input_digest, sizes: self.dataset.sizes, rows: self.rows,
            first_identity: self.first_identity.ok_or(AssayError::Identity)?,
            executions: self.executions.finalize().into(), compatible_executions: self.compatible_executions.finalize().into(),
            observations: self.observations.finalize().into(), model_work: self.model_work,
            group_width: self.width.ok_or(AssayError::Identity)?, heads: self.heads })
    }
}
fn label_index(relation: FaithfulnessRelation) -> usize {
    match relation { FaithfulnessRelation::Contradicted => 0, FaithfulnessRelation::Entailed => 1,
        FaithfulnessRelation::Unsupported => 2 }
}
fn frame(hash: &mut Sha256, bytes: &[u8]) { hash.update((bytes.len() as u64).to_be_bytes()); hash.update(bytes); }
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), ReplayError> {
    control.checkpoint(0).map_or(Ok(()), |kind| Err(ReplayError::Cancelled(kind)))
}
fn compatible_identity(identity: &ExecutionIdentity) -> ExecutionIdentity {
    let mut value = identity.clone(); let absent = Sha256Digest::of_bytes(b"native-assay-comparison-subject-v1");
    value.logical_model_digest = absent; value.packing_set_digest = absent;
    value.artifact_format = "comparison-subject".to_owned(); value.quant_recipe = "comparison-subject".to_owned();
    value.backend_semantic_version = "comparison-subject".to_owned();
    value.host_class = None; value.compiler_identity = None; value
}
