//! Complete private replay projection. Only JobRunner's domain-separated HMAC
//! commitment is retained; no document, raw pseudonym key or namespace is stored.
use super::*;
use crate::{
    grammar::{CompileLimits, mask::MaskWorkLimits, runtime::{SourceRuntimeLimits, SOURCE_JSON_RUNTIME_VERSION}},
    native_engine::strict_int4::{Int4Work, STRICT_INT4_PROFILE},
    tasks::{ir::TaskBudget, ner::NerOptions, mapreduce::{ChunkLimits, ExecutionLimits, CHUNK_PROFILE},
        source_planning::{SourcePlanningLimits, SOURCE_PROMPT_VERSION,
            grouped_int4::long::{Int4SourceMapLimits, INT4_SOURCE_MAP_EXECUTION}},
        redact::{RedactionRequest, actions::ACTION_POLICY_VERSION, detectors::RULE_PROFILE,
            union::OVERLAP_POLICY, pseudonym::{PseudonymEncoding, PseudonymIdentity},
            grouped_int4::long::{Int4LongRedactionConfig, INT4_LONG_REDACTION_EXECUTION}}},
    validation::grounded_fields::GroundingBudget,
};
const INPUT_POLICY: &str = "fixed-redaction-policy-original-utf8-full256-v1";

/// A sealed serialization projection, never a deserializable task factory.
/// Fields stay private and Debug is absent because policies can be sensitive.
#[derive(Serialize)]
pub struct Int4RedactionJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    input_policy: &'static str,
    execution: &'static str,
    numerics_profile: &'static str,
    group_width: usize,
    max_source_bytes: usize,
    prompt_version: &'static str,
    source_runtime: &'static str,
    action_version: &'static str,
    rule_version: &'static str,
    overlap_version: &'static str,
    detector: DetectorRecipe,
    request: RedactionRequest,
    max_model_work: Int4Work,
    max_mask_visits: u64,
    pseudonyms: Option<PseudonymIdentity>,
    item_work: JobWork,
    max_result_bytes: u64,
}
#[derive(Serialize)]
struct DetectorRecipe {
    ner: NerOptions, per_chunk: TaskBudget, planning: PlanningRecipe,
    mapping: MapRecipe, max_result_bytes: u64,
}
#[derive(Serialize)]
struct MapRecipe {
    execution: &'static str, chunk_profile: &'static str, reduction_profile: &'static str,
    chunks: ChunkLimits, reduction: ExecutionLimits, max_model_work: Int4Work,
    mask_limits: MaskRecipe, mask_visits_per_chunk: u64, max_mask_visits: u64,
    verification: GroundingBudget,
}
#[derive(Serialize)]
struct PlanningRecipe {
    max_input_bytes: usize, max_context_tokens: usize, max_passages: usize,
    compiler: CompilerRecipe, source: SourceRuntimeLimits,
}
#[derive(Serialize)]
struct CompilerRecipe {
    max_schema_bytes: usize, max_string_bytes: usize, max_array_items: usize,
    max_output_bytes: usize, max_states: usize, max_transitions: usize, max_mask_bytes: usize,
}
#[derive(Serialize)]
struct MaskRecipe { max_trie_node_visits: usize, checkpoint_interval_nodes: usize }
impl From<SourcePlanningLimits> for PlanningRecipe {
    fn from(limits: SourcePlanningLimits) -> Self {
        let SourcePlanningLimits { max_input_bytes, max_context_tokens, max_passages, compiler, source } = limits;
        let CompileLimits { max_schema_bytes, max_string_bytes, max_array_items, max_output_bytes,
            max_states, max_transitions, max_mask_bytes } = compiler;
        Self { max_input_bytes, max_context_tokens, max_passages, source,
            compiler: CompilerRecipe { max_schema_bytes, max_string_bytes, max_array_items,
                max_output_bytes, max_states, max_transitions, max_mask_bytes } }
    }
}
impl From<MaskWorkLimits> for MaskRecipe {
    fn from(limits: MaskWorkLimits) -> Self {
        let MaskWorkLimits { max_trie_node_visits, checkpoint_interval_nodes } = limits;
        Self { max_trie_node_visits, checkpoint_interval_nodes }
    }
}
impl From<Int4SourceMapLimits> for MapRecipe {
    fn from(limits: Int4SourceMapLimits) -> Self {
        let Int4SourceMapLimits { chunks, reduction, max_model_work, mask_limits,
            mask_visits_per_chunk, max_mask_visits, verification } = limits;
        Self { execution: INT4_SOURCE_MAP_EXECUTION, chunk_profile: CHUNK_PROFILE,
            reduction_profile: crate::tasks::mapreduce::execution::EXECUTION_PROFILE,
            chunks, reduction, max_model_work, mask_limits: mask_limits.into(),
            mask_visits_per_chunk, max_mask_visits, verification }
    }
}
impl Int4RedactionJobRecipe {
    pub(super) fn new(config: &Int4RedactionJobConfig, group_width: usize, context: Option<&Pseudonyms<'_>>)
        -> Result<Self, BatchFault> {
        bounded(&config.batch.request)?; bounded(&config.batch.detector.ner)?;
        config.batch.request.actions.check_key(context).map_err(|_| BatchCode::Admission)?;
        if context.is_some_and(|p| p.identity().encoding != PseudonymEncoding::Full256) {
            return Err(BatchCode::Admission.into());
        }
        // Exhaustive projections make additions to non-serializable settings a
        // deliberate resume-contract change. ExecutionIdentity is separately
        // authenticated by JobRunner; no synthetic replacement is constructed.
        let Int4RedactionJobConfig { batch, max_source_bytes } = config;
        let Int4DocumentRedactionBatchConfig { ner_identity: _, detector, request,
            max_model_work, max_mask_visits } = batch;
        let Int4LongRedactionConfig { ner, per_chunk, planning, mapping, max_result_bytes } = detector;
        let recipe = Self { version: 1, dependency_scope: "item-local", input_policy: INPUT_POLICY,
            execution: INT4_LONG_REDACTION_EXECUTION, numerics_profile: STRICT_INT4_PROFILE,
            group_width, max_source_bytes: *max_source_bytes,
            prompt_version: SOURCE_PROMPT_VERSION, source_runtime: SOURCE_JSON_RUNTIME_VERSION,
            action_version: ACTION_POLICY_VERSION, rule_version: RULE_PROFILE, overlap_version: OVERLAP_POLICY,
            detector: DetectorRecipe { ner: ner.clone(), per_chunk: *per_chunk,
                planning: (*planning).into(), mapping: (*mapping).into(), max_result_bytes: *max_result_bytes },
            request: request.clone(), max_model_work: *max_model_work, max_mask_visits: *max_mask_visits,
            pseudonyms: context.map(|p| p.identity().clone()),
            item_work: JobWork::from_int4(batch.item_model_work(), mapping.max_mask_visits),
            max_result_bytes: *max_result_bytes };
        bounded(&recipe)?;
        Ok(recipe)
    }
    pub(super) fn max_source_bytes(&self) -> usize { self.max_source_bytes }
    pub(super) fn max_result_bytes(&self) -> u64 { self.max_result_bytes }
    pub(super) fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        validate_job_limits(limits, JobWork::from_int4(self.max_model_work, self.max_mask_visits),
            self.item_work, self.max_result_bytes)
    }
    pub(super) fn check_work(&self, batch: BatchWork, poisoned: bool) -> Result<JobWork, BatchItemFailure> {
        if poisoned || batch.forward_positions != self.item_work.model.forward_positions
            || batch.projected_logits != self.item_work.model.projected_logits {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        // Persist the entire original+fresh-verification envelope, including
        // all group dots and masks. Observed early completion never refunds it.
        Ok(self.item_work)
    }
}
pub(super) fn validate_job_limits(limits: JobLimits, lifetime: JobWork, item: JobWork,
    max_result_bytes: u64) -> Result<(), BatchFault> {
    limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
    if limits.max_work != lifetime || !item.fits(lifetime)
        || max_result_bytes == 0 || max_result_bytes > limits.max_result_bytes as u64 {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
