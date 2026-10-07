//! Private complete replay contract. Never serialize key bytes or source text;
//! JobRunner persists only the existing domain-separated keyed commitment.
use super::*;
use serde::Serialize;
use crate::{
    grammar::{CompileLimits, mask::MaskWorkLimits, runtime::{SourceRuntimeLimits, SOURCE_JSON_RUNTIME_VERSION}},
    tasks::{ir::TaskBudget, ner::NerOptions, mapreduce::{ChunkLimits, ExecutionLimits, CHUNK_PROFILE},
        source_planning::{SourcePlanningLimits, SOURCE_PROMPT_VERSION,
            quantized::long::{Int8SourceMapLimits, INT8_SOURCE_MAP_EXECUTION}},
        redact::{actions::ACTION_POLICY_VERSION, detectors::RULE_PROFILE, union::OVERLAP_POLICY,
            corpus::LongRedactionBatchConfig, long::{LongRedactionConfig, LONG_REDACTION_EXECUTION},
            pseudonym::{PseudonymIdentity, PseudonymEncoding}, quantized::{Int8RedactionConfig, INT8_REDACTION_EXECUTION}}},
};

pub(super) const RECIPE_BYTES: usize = 1024 * 1024 - 1024;
#[derive(Serialize)]
pub(super) struct RedactionJobRecipe {
    version: u32, dependency_scope: &'static str, execution: &'static str,
    prompt_version: &'static str, source_runtime: &'static str,
    action_version: &'static str, rule_version: &'static str, overlap_version: &'static str,
    detector: DetectorRecipe, request: RedactionRequest,
    max_model_work: Int8Work, max_mask_visits: u64,
    pseudonyms: Option<PseudonymIdentity>,
    item_work: JobWork,
    pub(super) max_result_bytes: u64,
}
// Untagged serialization preserves the original short-job recipe byte shape.
// The top-level execution version distinguishes short and chunked pipelines.
#[derive(Serialize)]
#[serde(untagged)]
enum DetectorRecipe { Short(ShortDetector), Long(LongDetector) }
#[derive(Serialize)]
struct ShortDetector {
    ner: NerOptions, per_pass: TaskBudget, planning: PlanningRecipe,
    max_model_work: Int8Work, mask_limits: MaskRecipe, mask_visits_per_pass: u64,
    max_mask_visits: u64, max_result_bytes: u64,
}
#[derive(Serialize)]
struct LongDetector {
    ner: NerOptions, per_chunk: TaskBudget, planning: PlanningRecipe,
    mapping: MapRecipe, max_result_bytes: u64,
}
#[derive(Serialize)]
struct MapRecipe {
    execution: &'static str, chunk_profile: &'static str, reduction_profile: &'static str,
    chunks: ChunkLimits, reduction: ExecutionLimits, max_model_work: Int8Work,
    mask_limits: MaskRecipe, mask_visits_per_chunk: u64, max_mask_visits: u64,
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
impl From<Int8SourceMapLimits> for MapRecipe {
    fn from(limits: Int8SourceMapLimits) -> Self {
        let Int8SourceMapLimits { chunks, reduction, max_model_work, mask_limits,
            mask_visits_per_chunk, max_mask_visits } = limits;
        Self { execution: INT8_SOURCE_MAP_EXECUTION, chunk_profile: CHUNK_PROFILE,
            reduction_profile: crate::tasks::mapreduce::execution::EXECUTION_PROFILE,
            chunks, reduction, max_model_work, mask_limits: mask_limits.into(), mask_visits_per_chunk, max_mask_visits }
    }
}
fn key_identity(request: &RedactionRequest, context: Option<&Pseudonyms<'_>>)
    -> Result<Option<PseudonymIdentity>, HostedError> {
    request.actions.check_key(context).map_err(|e| HostedError::Redaction(e.into()))?;
    if context.is_some_and(|c| c.identity().encoding != PseudonymEncoding::Full256) {
        return Err(HostedError::Limits("retained redaction requires one full256 scope"));
    }
    Ok(context.map(|c| c.identity().clone()))
}
impl RedactionJobRecipe {
    pub(super) fn short(config: &Int8RedactionBatchConfig, context: Option<&Pseudonyms<'_>>)
        -> Result<Self, HostedError> {
        bounded(&config.request)?; bounded(&config.detector.ner)?;
        let pseudonyms = key_identity(&config.request, context)?;
        // Destructure all non-serializable fields: new native/compiler limits
        // must be deliberately added to this authenticated projection.
        let Int8RedactionConfig { ner, per_pass, planning, max_model_work, mask_limits,
            mask_visits_per_pass, max_mask_visits, max_result_bytes } = &config.detector;
        let masks = mask_visits_per_pass.checked_mul(1 + u64::from(config.request.verify))
            .filter(|&n| n > 0 && n <= *max_mask_visits && n <= config.max_mask_visits)
            .ok_or(HostedError::Limits("redaction job complete verification work"))?;
        let recipe = Self { version: 1, dependency_scope: "item-local", execution: INT8_REDACTION_EXECUTION,
            prompt_version: SOURCE_PROMPT_VERSION, source_runtime: SOURCE_JSON_RUNTIME_VERSION,
            action_version: ACTION_POLICY_VERSION, rule_version: RULE_PROFILE, overlap_version: OVERLAP_POLICY,
            detector: DetectorRecipe::Short(ShortDetector { ner: ner.clone(), per_pass: *per_pass, planning: (*planning).into(),
                max_model_work: *max_model_work, mask_limits: (*mask_limits).into(),
                mask_visits_per_pass: *mask_visits_per_pass, max_mask_visits: *max_mask_visits, max_result_bytes: *max_result_bytes }),
            request: config.request.clone(), max_model_work: config.max_model_work, max_mask_visits: config.max_mask_visits,
            pseudonyms, item_work: JobWork { model: *max_model_work, mask_node_visits: masks }, max_result_bytes: *max_result_bytes };
        bounded(&recipe)?;
        Ok(recipe)
    }
    pub(super) fn long(config: &LongRedactionBatchConfig, context: Option<&Pseudonyms<'_>>)
        -> Result<Self, HostedError> {
        bounded(&config.request)?; bounded(&config.detector.ner)?;
        let pseudonyms = key_identity(&config.request, context)?;
        let LongRedactionConfig { ner, per_chunk, planning, mapping, max_result_bytes } = &config.detector;
        let work = config.item_model_work();
        if work.projected_logits == 0 || mapping.max_mask_visits == 0 || mapping.max_mask_visits > config.max_mask_visits {
            return Err(HostedError::Limits("redaction job complete map work"));
        }
        let recipe = Self { version: 1, dependency_scope: "item-local", execution: LONG_REDACTION_EXECUTION,
            prompt_version: SOURCE_PROMPT_VERSION, source_runtime: SOURCE_JSON_RUNTIME_VERSION,
            action_version: ACTION_POLICY_VERSION, rule_version: RULE_PROFILE, overlap_version: OVERLAP_POLICY,
            detector: DetectorRecipe::Long(LongDetector { ner: ner.clone(), per_chunk: *per_chunk,
                planning: (*planning).into(), mapping: (*mapping).into(), max_result_bytes: *max_result_bytes }),
            request: config.request.clone(), max_model_work: config.max_model_work, max_mask_visits: config.max_mask_visits,
            pseudonyms, item_work: JobWork { model: work, mask_node_visits: mapping.max_mask_visits },
            max_result_bytes: *max_result_bytes };
        bounded(&recipe)?;
        Ok(recipe)
    }
    pub(super) fn check_work(&self, batch: BatchWork, poisoned: bool) -> Result<JobWork, BatchItemFailure> {
        let work = self.item_work;
        if poisoned || batch.forward_positions != work.model.forward_positions
            || batch.projected_logits != work.model.projected_logits {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        // Reserve the WHOLE attempt, including a verification prompt that can
        // only be compiled from the actual edited text. No observed-work refund.
        Ok(work)
    }
}

pub(super) fn bounded<T: Serialize + ?Sized>(value: &T) -> Result<(), HostedError> {
    struct Counter(usize);
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.checked_sub(bytes.len()).ok_or_else(|| std::io::Error::other("private recipe bound"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    serde_json::to_writer(Counter(RECIPE_BYTES), value)
        .map_err(|_| HostedError::Limits("redaction job private recipe bound"))
}
