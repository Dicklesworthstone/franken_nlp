//! Storage-neutral complete-source execution on the ordered NDJSON runner.
//! One item owns every discovery chunk and every permitted synthesis pass;
//! all native, mask, verification and reducer allowances survive flushes.
use super::*;
use crate::{
    execution_identity::Sha256Digest,
    grammar::mask::MaskWorkLimits,
    native_engine::{generation::grouped_int4::bind_candidate_model, int4::PackedInt4Kernel,
        strict_int4::{Int4ModelIdentity, Int4ModelWeights, STRICT_INT4_EXECUTION}},
    tasks::source_planning::grouped_int4::long::{
        Int4SourceMapError, Int4SourceMapLimits, Int4SourceMapPreflight, Int4SourceMapRun,
        PreparedInt4SourceMap, SourceMapTask, INT4_SOURCE_MAP_EXECUTION,
        cohort::{Int4SourceMapCohortBudget, Int4SourceMapCohortMemoryLimits,
            Int4SourceMapCohortRequirement, hierarchy::Int4HierarchyCohortSchedule},
        keyphrases::{Int4CorpusKeyphraseError, Int4CorpusKeyphraseRun, Int4KeyphraseLimits,
            INT4_CORPUS_KEYPHRASE_EXECUTION},
        summary::{Int4CorpusSummaryError, Int4CorpusSummaryRun, Int4SummaryLimits,
            INT4_CORPUS_SUMMARY_EXECUTION,
            synthesis::{Int4SummarySynthesisPreflight, Int4SummarySynthesisRun,
                PreparedInt4SummarySynthesis, SourceSummarySynthesis, INT4_SUMMARY_SYNTHESIS_EXECUTION,
                hierarchy::{Int4SummaryHierarchyPreflight, Int4SummaryHierarchyRun,
                    PreparedInt4SummaryHierarchy, SummaryHierarchyLimits, INT4_SUMMARY_HIERARCHY_EXECUTION}}},
        question::{Int4QuestionPreflight, Int4QuestionRun, PreparedInt4Question, SourceQuestion,
            INT4_QUESTION_EXECUTION,
            synthesis::{Int4QuestionSynthesisPreflight, Int4QuestionSynthesisRun,
                PreparedInt4QuestionSynthesis, SourceQuestionSynthesis, INT4_QUESTION_SYNTHESIS_EXECUTION,
                hierarchy::{Int4QuestionHierarchyPreflight, Int4QuestionHierarchyRun,
                    PreparedInt4QuestionHierarchy, QuestionHierarchyLimits, INT4_QUESTION_HIERARCHY_EXECUTION}}}},
    validation::grounded_fields::GroundingBudget,
};
use std::{fmt, sync::Arc};

pub(crate) mod task;
pub(crate) mod planning;
pub(crate) mod backend;
mod stream;
pub use task::{Int4SourceDocumentTask, Int4SourceDocumentLifetime};
pub use backend::{Int4SourceDocumentAdmission, Int4SourceDocumentBatchAdmission, Int4SourceDocumentRun};
pub use stream::{NativeInt4SourceDocumentBatch, PreparedInt4SourceDocumentBatch};

/// Frozen, reconstructed and admitted identity ownership per possible chunk.
pub const SOURCE_DOCUMENT_CHUNK_METADATA_BYTES: u64 = 32_768;

/// Every record uses the fixed invocation policy. Unknown keys and all JSON
/// sequences are rejected; absent/null arguments are handled by BatchDocument.
#[derive(Default)]
pub struct SourceDocumentArgs {}
impl<'de> Deserialize<'de> for SourceDocumentArgs {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct EmptyMap;
        impl<'de> serde::de::Visitor<'de> for EmptyMap {
            type Value = SourceDocumentArgs;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an empty object of document arguments")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                if map.next_key::<String>()?.is_some() {
                    return Err(serde::de::Error::custom("document arguments cannot override the fixed source policy"));
                }
                Ok(SourceDocumentArgs {})
            }
        }
        deserializer.deserialize_map(EmptyMap)
    }
}

/// Invocation-wide allowances, charged atomically before each execution attempt.
/// Verification and deterministic reducers reserve their whole document cap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SourceDocumentBatchLimits {
    pub max_model_work: Int4Work,
    pub max_mask_visits: u64,
    pub max_verification: GroundingBudget,
    pub max_summary_scan_steps: u64,
    pub max_keyphrase_scan_work: u64,
}

/// Actual-model-bound immutable complete-document factory. It contains no
/// storage, durable manifest, filesystem handle or user-supplied executor.
pub struct Int4SourceDocumentBatchPlanner {
    planner: Arc<SourceTaskPlanner>,
    identity: ExecutionIdentity,
    group_width: usize,
    task: Int4SourceDocumentTask,
    budget: TaskBudget,
    planning: SourcePlanningLimits,
    mapping: Int4SourceMapLimits,
    run_limits: Int4SourceDocumentBatchLimits,
    chunk_cohorts: bool,
    hierarchy_groups: Option<usize>,
    owner: Arc<()>,
}
impl Int4SourceDocumentBatchPlanner {
    #[allow(clippy::too_many_arguments)]
    pub fn for_model(model: &Int4ModelIdentity, planner: Arc<SourceTaskPlanner>, task: Int4SourceDocumentTask,
        budget: TaskBudget, planning: SourcePlanningLimits, mapping: Int4SourceMapLimits,
        run_limits: Int4SourceDocumentBatchLimits) -> Result<Self, BatchFault> {
        validate_limits(&task, budget, planning, mapping, run_limits)?;
        let absent = Sha256Digest::of_bytes(b"null");
        let identity = bind_candidate_model(ExecutionIdentity {
            schema_version: 1, source_revision: "unbound".to_owned(), logical_model_digest: absent,
            artifact_format: "unbound".to_owned(), quant_recipe: "unbound".to_owned(), packing_set_digest: absent,
            tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
            task_spec: task.kind().spec().identity(), taskir_digest: absent, prompt_digest: absent,
            grammar_compiler_version: "uncompiled".to_owned(), schema_digest: absent,
            numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
            sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
            calibration_digest: absent, decision_policy_digest: absent,
            backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None,
        }, model).map_err(|_| BatchCode::Admission)?;
        Self::configured(planner, identity, model.group_width(), task, budget, planning, mapping, run_limits)
    }
    #[allow(clippy::too_many_arguments)]
    fn configured(planner: Arc<SourceTaskPlanner>, identity: ExecutionIdentity, group_width: usize,
        task: Int4SourceDocumentTask, budget: TaskBudget, planning: SourcePlanningLimits,
        mapping: Int4SourceMapLimits, run_limits: Int4SourceDocumentBatchLimits) -> Result<Self, BatchFault> {
        validate_limits(&task, budget, planning, mapping, run_limits)?;
        super::check_configuration(&planner, &identity, budget, planning, None, group_width)?;
        if identity.task_spec != task.kind().spec().identity() {
            return Err(BatchCode::Admission.into());
        }
        let required = Int4MemoryRequirement::for_context(planning.max_context_tokens, group_width)
            .map_err(|_| BatchCode::InvalidLimits)?;
        if required.kv_bytes == 0 || required.kv_bytes != budget.max_kv_bytes {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(Self { planner, identity, group_width, task, budget, planning, mapping, run_limits,
            chunk_cohorts: false, hierarchy_groups: None, owner: Arc::new(()) })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn task_budget(&self) -> TaskBudget { self.budget }
    pub fn planning_limits(&self) -> SourcePlanningLimits { self.planning }
    pub fn document_limits(&self) -> Int4SourceMapLimits { self.mapping }
    pub fn task(&self) -> &Int4SourceDocumentTask { &self.task }
    pub fn run_limits(&self) -> Int4SourceDocumentBatchLimits { self.run_limits }
    pub fn max_result_bytes(&self) -> u64 { self.task.max_result_bytes(self.mapping) }
    pub fn output_token_ceiling(&self) -> Result<u64, BatchFault> { self.task.output_token_ceiling(self.mapping, self.budget) }
    pub fn retained_defaults_bytes(&self) -> Result<u64, BatchFault> { self.task.retained_bytes() }
    pub fn with_chunk_cohorts(mut self) -> Result<Self, BatchFault> {
        if !self.chunk_cohorts { self.chunk_cohorts = true; self.owner = Arc::new(()); }
        self.cohort_requirements()?;
        Ok(self)
    }
    pub fn with_hierarchy_cohorts(mut self, max_groups: usize) -> Result<Self, BatchFault> {
        let max_passes = match self.task() {
            Int4SourceDocumentTask::SummaryHierarchy { limits, .. } => limits.max_passes,
            Int4SourceDocumentTask::QuestionHierarchy { limits, .. } => limits.max_passes,
            _ => return Err(BatchCode::InvalidLimits.into()),
        };
        Int4HierarchyCohortSchedule { max_groups, discovery_cohorts: self.chunk_cohorts }
            .validate(max_passes).map_err(|_| BatchCode::InvalidLimits)?;
        if self.hierarchy_groups != Some(max_groups) { self.hierarchy_groups = Some(max_groups); self.owner = Arc::new(()); }
        self.cohort_requirements()?;
        Ok(self)
    }
    pub fn uses_chunk_cohorts(&self) -> bool { self.chunk_cohorts }
    pub fn hierarchy_cohort_schedule(&self) -> Option<Int4HierarchyCohortSchedule> {
        self.hierarchy_groups.map(|max_groups| Int4HierarchyCohortSchedule { max_groups, discovery_cohorts: self.chunk_cohorts })
    }
    pub fn uses_native_cohorts(&self) -> bool { self.chunk_cohorts || self.hierarchy_groups.is_some() }
    pub fn cohort_requirements(&self) -> Result<Int4SourceMapCohortRequirement, BatchFault> {
        self.configuration().cohort_requirements()
    }
    pub(crate) fn configuration(&self) -> Configuration<'_> {
        Configuration { planner: &self.planner, identity: &self.identity, group_width: self.group_width,
            task: &self.task, budget: self.budget, planning: self.planning, mapping: self.mapping,
            chunk_cohorts: self.chunk_cohorts, hierarchy: self.hierarchy_cohort_schedule() }
    }
}

/// Borrowed native policy used by both live and durable factories. The latter
/// keeps its recipe and journal authority; only native reconstruction is shared.
pub(crate) struct Configuration<'a> {
    pub(crate) planner: &'a SourceTaskPlanner,
    pub(crate) identity: &'a ExecutionIdentity,
    pub(crate) group_width: usize,
    pub(crate) task: &'a Int4SourceDocumentTask,
    pub(crate) budget: TaskBudget,
    pub(crate) planning: SourcePlanningLimits,
    pub(crate) mapping: Int4SourceMapLimits,
    pub(crate) chunk_cohorts: bool,
    pub(crate) hierarchy: Option<Int4HierarchyCohortSchedule>,
}
impl Configuration<'_> {
    pub(crate) fn uses_native_cohorts(&self) -> bool { self.chunk_cohorts || self.hierarchy.is_some() }
    pub(crate) fn cohort_requirements(&self) -> Result<Int4SourceMapCohortRequirement, BatchFault> {
        let context = self.planning.max_context_tokens;
        let scalar = Int4MemoryRequirement::for_context(context, self.group_width).map_err(|_| BatchCode::InvalidLimits)?;
        if scalar.kv_bytes != self.budget.max_kv_bytes { return Err(BatchCode::Admission.into()); }
        if let Some(schedule) = self.hierarchy {
            let discovery = if schedule.discovery_cohorts { Some(self.discovery_cohort_requirements()?) } else { None };
            return schedule.requirements(discovery, context, self.group_width, self.budget)
                .map_err(|_| BatchCode::InvalidLimits.into());
        }
        let quote = self.discovery_cohort_requirements()?;
        if matches!(self.task, Int4SourceDocumentTask::SummarySynthesis(_) | Int4SourceDocumentTask::SummaryHierarchy { .. }
            | Int4SourceDocumentTask::QuestionSynthesis(_) | Int4SourceDocumentTask::QuestionHierarchy { .. }) {
            quote.with_synthesis_context(context, self.group_width, self.budget.max_kv_bytes)
                .map_err(|_| BatchCode::Admission.into())
        } else { Ok(quote) }
    }
    fn discovery_cohort_requirements(&self) -> Result<Int4SourceMapCohortRequirement, BatchFault> {
        let max_chunks = self.mapping.reduction.map_batch_chunks;
        if max_chunks > self.mapping.chunks.max_chunks { return Err(BatchCode::InvalidLimits.into()); }
        Int4SourceMapCohortMemoryLimits { max_chunks, context_tokens: self.planning.max_context_tokens,
            max_output_tokens_per_chunk: u64::from(self.budget.max_output_tokens),
            max_result_bytes_per_chunk: self.budget.max_output_bytes,
        }.requirements(self.group_width).map_err(|_| BatchCode::InvalidLimits.into())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DocumentWork { pub(crate) model: Int4Work, pub(crate) mask_node_visits: u64 }

pub(crate) fn validate_limits(task: &Int4SourceDocumentTask, budget: TaskBudget, planning: SourcePlanningLimits,
    mapping: Int4SourceMapLimits, run: Int4SourceDocumentBatchLimits) -> Result<(), BatchFault> {
    validate_mapping(mapping, budget, planning)?;
    task.validate(planning, mapping)?;
    task.verify_lifetime(mapping, Int4SourceDocumentLifetime { verification: run.max_verification,
        summary_scan_steps: run.max_summary_scan_steps }, 1)?;
    if !fits(mapping.max_model_work, run.max_model_work) || mapping.max_mask_visits > run.max_mask_visits
        || task.keyphrase_scan_work() > run.max_keyphrase_scan_work {
        return Err(BatchCode::InvalidLimits.into());
    }
    if let Int4SourceDocumentTask::Keyphrases { max_lifetime_scan_work, .. } = task {
        // The fixed task cap and the live invocation cap are both authority;
        // require the latter to tighten the former, never silently replace it.
        if run.max_keyphrase_scan_work > *max_lifetime_scan_work { return Err(BatchCode::InvalidLimits.into()); }
    }
    task.output_token_ceiling(mapping, budget)?;
    task.retained_bytes()?;
    Ok(())
}
pub(crate) fn validate_mapping(m: Int4SourceMapLimits, budget: TaskBudget, p: SourcePlanningLimits) -> Result<(), BatchFault> {
    budget.validate().map_err(|_| BatchCode::InvalidLimits)?;
    m.chunks.effective_token_limit().map_err(|_| BatchCode::InvalidLimits)?;
    let e = m.reduction; let v = m.verification;
    if m.chunks.max_input_bytes == 0 || !(1..=256).contains(&m.chunks.max_chunks)
        || m.chunks.max_chunk_bytes > p.max_input_bytes || m.chunks.context_tokens > p.max_context_tokens
        || m.chunks.reserved_tokens < budget.max_output_tokens as usize
        || !(1..=64 * 1024 * 1024).contains(&p.max_input_bytes) || !(1..=262_144).contains(&p.max_context_tokens)
        || !(1..=1024).contains(&p.max_passages)
        || m.mask_visits_per_chunk == 0 || m.max_mask_visits < m.mask_visits_per_chunk
        || m.mask_limits.max_trie_node_visits == 0 || m.mask_limits.checkpoint_interval_nodes == 0
        || !(1..=1024).contains(&e.map_batch_chunks) || !(2..=1024).contains(&e.reduce_fan_in)
        || e.max_reduction_levels > 64 || !(1..=1_000_000).contains(&e.max_task_calls)
        || [e.max_value_bytes, e.max_live_value_bytes, e.max_total_value_bytes, e.max_result_bytes]
            .iter().any(|n| !(1..=64 * 1024 * 1024).contains(n))
        || e.max_value_bytes > e.max_live_value_bytes
        || !(1..=1_000_000).contains(&v.max_fields) || !(1..=1_000_000).contains(&v.max_matches) || v.max_scan_steps == 0 {
        return Err(BatchCode::InvalidLimits.into());
    }
    super::validate_limits(Int4SourceBatchLimits { max_model_work: m.max_model_work,
        masks: SourceMaskBudget { per_mask: m.mask_limits, max_visits_per_item: m.mask_visits_per_chunk,
            max_visits_per_run: m.max_mask_visits } })
}
fn verification_fits(a: GroundingBudget, b: GroundingBudget) -> bool {
    a.max_fields <= b.max_fields && a.max_matches <= b.max_matches && a.max_scan_steps <= b.max_scan_steps
}
pub(crate) fn poll(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}
fn map_failure(error: Int4SourceMapError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    BatchItemFailure::fatal(match error {
        Int4SourceMapError::InvalidLimits => BatchCode::InvalidLimits,
        Int4SourceMapError::EmptySource => BatchCode::DocumentLimit,
        Int4SourceMapError::WorkLimit => BatchCode::WorkLimit,
        Int4SourceMapError::Admission => BatchCode::Admission,
        Int4SourceMapError::Allocation => BatchCode::Allocation,
        Int4SourceMapError::Chunk(_) => BatchCode::Planning,
        _ => BatchCode::InvalidExecution,
    })
}
fn summary_failure(error: Int4CorpusSummaryError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    BatchItemFailure::fatal(if matches!(error, Int4CorpusSummaryError::InvalidLimits) { BatchCode::InvalidLimits } else { BatchCode::InvalidExecution })
}
pub(crate) fn keyphrase_failure(error: Int4CorpusKeyphraseError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    BatchItemFailure::fatal(if matches!(error, Int4CorpusKeyphraseError::InvalidLimits) { BatchCode::InvalidLimits } else { BatchCode::InvalidExecution })
}

#[cfg(test)] mod tests;
