//! Durable INT8 extraction with complete correlated field review.
//! One journal item is the complete two-stage result. The existing JobRunner
//! reserves the full declared attempt before extraction and retains its output
//! guard through spool sync and journal acknowledgement. Neither partial
//! extraction nor a failed review becomes a successful durable record.

use std::sync::Arc;
use serde::{Serialize, Serializer};
use super::CompilerRecipe;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork,
        extract::{ExtractionBatchArgs, ExtractionMaskBudget, GuardedOutput, EXTRACTION_BATCH_PROMPT,
            quantized::Int8ExtractionBatchPlanner,
            semantic::int8::{Int8SemanticExtractionBatchAdmission, Int8SemanticExtractionBatchLimits,
                Int8SemanticExtractionBatchPlanner, NativeInt8SemanticExtractionBatch,
                PreparedInt8SemanticBatchExtraction, check_judge_limits, encoding_workspace}}},
    execution_identity::ExecutionIdentity,
    grammar::{CompileLimits, mask::MaskWorkLimits,
        runtime::{SourceRuntimeLimits, JSON_RUNTIME_VERSION, SOURCE_JSON_RUNTIME_VERSION}},
    jobs::{JobLimits, JobWork, manifest::bounded_json, runner::DurableBatchProcessor},
    native_engine::{decode::DecodeStepControl, lmhead::scoring::ScoringLimits,
        strict_int8::{Int8Work, StrictInt8Engine, prefill::Int8PrefillLimits}},
    tasks::{extract::{ExtractionVocabulary, semantic::{CLAIM_RENDER_VERSION, SEMANTIC_VERIFICATION_VERSION,
            int8::{Int8SemanticExtractionRun, Int8SemanticOptions, INT8_SEMANTIC_EXECUTION}}},
        judge::{JudgeLimits, JudgePlanner, FAITHFULNESS_VERSION, JUDGE_PROMPT_VERSION,
            quantized::INT8_JUDGE_EXECUTION}},
    tokenizer::{specials::TemplateControlIds, untrusted::{SafeBpeWorkLimits, UntrustedEncoding}},
};

/// Exact recipe and canonicalization storage bound. Hosts price two such
/// buffers simultaneously, beside the retained actual compiler/config owners.
pub const SEMANTIC_EXTRACTION_JOB_RECIPE_BYTES: usize = 1024 * 1024 - 1024;
const INPUT_POLICY: &str = "fixed-schema-and-semantic-review-no-record-overrides-v1";
const ENCODING_SCOPE: &str = "one-invocation-across-all-document-attempts-v1";
const RESERVATION_POLICY: &str = "complete-extraction-and-review-ceiling-per-durable-attempt-v1";

/// Immutable task settings. Physical worker/prompt geometry is deliberately
/// separate; changing it cannot silently change the authenticated task recipe.
/// No Debug/Deserialize: exact schemas and claim templates may be private.
#[derive(Clone)]
pub struct Int8SemanticExtractionJobConfig {
    pub extraction: ExtractionBatchArgs,
    pub compiler_limits: CompileLimits,
    pub source_limits: SourceRuntimeLimits,
    pub options: Int8SemanticOptions,
    pub judge_planning: JudgeLimits,
    pub encoding: UntrustedEncoding,
    pub encoding_work_limits: Option<SafeBpeWorkLimits>,
    /// Must equal the immutable journal's five native axes plus mask visits.
    pub work: Int8SemanticExtractionBatchLimits,
}

/// One actual immutable configuration owner. Only its keyed commitment enters
/// durable state; serialized settings are never a public executable factory.
pub struct Int8SemanticExtractionJobRecipe {
    config: Int8SemanticExtractionJobConfig,
    extraction_identity: ExecutionIdentity,
    judge_identity: ExecutionIdentity,
}
#[derive(Serialize)]
struct ScorerRecipe {
    max_candidates: usize, max_total_tokens: usize, max_nodes: usize,
    max_depth: usize, max_candidate_id_bytes: usize, max_projected_logits: u64,
}
#[derive(Serialize)]
struct JudgeRecipe {
    per_head: ScorerRecipe, max_total_prompt_tokens: usize,
    max_total_candidate_tokens: usize, max_total_projected_logits: u64,
    max_output_bytes: u64,
}
impl From<JudgeLimits> for JudgeRecipe {
    fn from(limits: JudgeLimits) -> Self {
        let JudgeLimits { per_head, max_total_prompt_tokens, max_total_candidate_tokens,
            max_total_projected_logits, max_output_bytes } = limits;
        let ScoringLimits { max_candidates, max_total_tokens, max_nodes,
            max_depth, max_candidate_id_bytes, max_projected_logits } = per_head;
        Self { per_head: ScorerRecipe { max_candidates, max_total_tokens, max_nodes,
            max_depth, max_candidate_id_bytes, max_projected_logits }, max_total_prompt_tokens,
            max_total_candidate_tokens, max_total_projected_logits, max_output_bytes }
    }
}
#[derive(Serialize)]
struct WorkRecipe {
    max_model_work: Int8Work,
    max_trie_node_visits: usize,
    checkpoint_interval_nodes: usize,
    max_visits_per_item: u64,
    max_visits_per_run: u64,
}
#[derive(Serialize)]
struct Settings<'a> {
    extraction: &'a ExtractionBatchArgs,
    compiler: CompilerRecipe,
    source: SourceRuntimeLimits,
    options: &'a Int8SemanticOptions,
    judge_planning: JudgeRecipe,
    input_encoding: &'a UntrustedEncoding,
    invocation_encoding_work_limits: Option<SafeBpeWorkLimits>,
    work: WorkRecipe,
}
impl<'a> From<&'a Int8SemanticExtractionJobConfig> for Settings<'a> {
    fn from(config: &'a Int8SemanticExtractionJobConfig) -> Self {
        // Exhaustive projections make future compiler/scorer/mask axes explicit
        // recipe changes instead of unbound settings on a resumed population.
        let CompileLimits { max_schema_bytes, max_string_bytes, max_array_items,
            max_output_bytes, max_states, max_transitions, max_mask_bytes } = config.compiler_limits;
        let Int8SemanticExtractionBatchLimits { max_model_work, masks } = config.work;
        let ExtractionMaskBudget { per_mask, max_visits_per_item, max_visits_per_run } = masks;
        let MaskWorkLimits { max_trie_node_visits, checkpoint_interval_nodes } = per_mask;
        Self { extraction: &config.extraction, compiler: CompilerRecipe { max_schema_bytes,
            max_string_bytes, max_array_items, max_output_bytes, max_states, max_transitions, max_mask_bytes },
            source: config.source_limits, options: &config.options,
            judge_planning: config.judge_planning.into(), input_encoding: &config.encoding,
            invocation_encoding_work_limits: config.encoding_work_limits,
            work: WorkRecipe { max_model_work, max_trie_node_visits, checkpoint_interval_nodes,
                max_visits_per_item, max_visits_per_run } }
    }
}
#[derive(Serialize)]
struct RecipeView<'a> {
    version: u32,
    dependency_scope: &'static str,
    input_policy: &'static str,
    reservation_policy: &'static str,
    execution: &'static str,
    extraction_prompt: &'static str,
    json_runtime: &'static str,
    source_runtime: &'static str,
    semantic_verification: &'static str,
    claim_rendering: &'static str,
    judge_execution: &'static str,
    judge_prompt: &'static str,
    faithfulness: &'static str,
    encoding_scope: &'static str,
    extraction_identity: &'a ExecutionIdentity,
    judge_identity: &'a ExecutionIdentity,
    settings: Settings<'a>,
}
impl Serialize for Int8SemanticExtractionJobRecipe {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        RecipeView { version: 1, dependency_scope: "item-local", input_policy: INPUT_POLICY,
            reservation_policy: RESERVATION_POLICY, execution: INT8_SEMANTIC_EXECUTION,
            extraction_prompt: EXTRACTION_BATCH_PROMPT, json_runtime: JSON_RUNTIME_VERSION,
            source_runtime: SOURCE_JSON_RUNTIME_VERSION, semantic_verification: SEMANTIC_VERIFICATION_VERSION,
            claim_rendering: CLAIM_RENDER_VERSION, judge_execution: INT8_JUDGE_EXECUTION,
            judge_prompt: JUDGE_PROMPT_VERSION, faithfulness: FAITHFULNESS_VERSION, encoding_scope: ENCODING_SCOPE,
            extraction_identity: &self.extraction_identity, judge_identity: &self.judge_identity,
            settings: (&self.config).into() }.serialize(serializer)
    }
}
impl Int8SemanticExtractionJobRecipe {
    fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
        let config = &self.config;
        config.work.validate_item(config.options.max_model_work)?;
        let lifetime = JobWork { model: config.work.max_model_work,
            mask_node_visits: config.work.masks.max_visits_per_run };
        if limits.max_work != lifetime || config.options.limits.max_output_bytes > limits.max_result_bytes as u64 {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}

/// Builds the genuine pinned extraction and judge compilers from the exact
/// serialized settings before opening any job storage. Resume recreates the
/// authenticated declarations; committed items are skipped by JobRunner.
/// Tokenizer work is invocation-scoped, while all six native/mask axes and
/// attempt counts remain durably spent across new invocations.
pub struct Int8SemanticExtractionJobPlanner {
    compiler: Arc<Int8SemanticExtractionBatchPlanner>,
    recipe: Arc<Int8SemanticExtractionJobRecipe>,
}
impl Int8SemanticExtractionJobPlanner {
    pub fn pinned(controls: &TemplateControlIds, eos: u32, identity: ExecutionIdentity,
        config: Int8SemanticExtractionJobConfig) -> Result<Self, BatchFault> {
        config.extraction.budget.validate().map_err(|_| BatchCode::InvalidLimits)?;
        config.options.validate().map_err(|_| BatchCode::InvalidLimits)?;
        config.work.validate_item(config.options.max_model_work)?;
        check_judge_limits(config.judge_planning)?;
        encoding_workspace(&config.encoding, config.encoding_work_limits)?;
        bounded_json(&Settings::from(&config), SEMANTIC_EXTRACTION_JOB_RECIPE_BYTES)
            .map_err(|_| BatchCode::InvalidLimits)?;
        let extraction = Int8ExtractionBatchPlanner::pinned(controls, eos, identity, config.extraction.budget,
            config.compiler_limits, config.source_limits, Some(config.extraction.clone()))?
            .with_document_encoding(config.encoding.clone())?;
        let judge = Arc::new(JudgePlanner::pinned(controls, eos)
            .and_then(|planner| planner.with_document_encoding(config.encoding.clone()))
            .map_err(|_| BatchCode::Admission)?);
        let mut judge_identity = extraction.base_execution_identity().clone();
        judge_identity.task_spec = "judge-v1".to_owned();
        judge_identity.tokenizer_digest = judge.tokenizer_digest();
        judge_identity.template_digest = *judge.template_digest();
        let compiler = Arc::new(Int8SemanticExtractionBatchPlanner::new(extraction, judge, judge_identity,
            config.judge_planning, config.options.clone(), config.work, config.encoding_work_limits)?);
        let recipe = Int8SemanticExtractionJobRecipe { config,
            extraction_identity: compiler.base_execution_identity().clone(),
            judge_identity: compiler.judge_identity().clone() };
        bounded_json(&recipe, SEMANTIC_EXTRACTION_JOB_RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(Self { compiler, recipe: Arc::new(recipe) })
    }
    pub(crate) fn retain_compiler(&self) -> Arc<Int8SemanticExtractionBatchPlanner> {
        Arc::clone(&self.compiler)
    }
    pub(crate) fn retain_recipe(&self) -> Arc<Int8SemanticExtractionJobRecipe> {
        Arc::clone(&self.recipe)
    }
    pub fn config(&self) -> &Int8SemanticExtractionJobConfig { &self.recipe.config }
    pub fn job_recipe(&self) -> &Int8SemanticExtractionJobRecipe { &self.recipe }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.recipe.extraction_identity }
    pub fn judge_identity(&self) -> &ExecutionIdentity { &self.recipe.judge_identity }
    pub fn max_result_bytes(&self) -> u64 { self.recipe.config.options.limits.max_output_bytes }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.compiler.document_encoding() }
    pub fn encoding_work_limits(&self) -> Option<SafeBpeWorkLimits> { self.compiler.encoding_work_limits() }
    pub fn encoding_workspace_bytes(&self) -> Result<u64, BatchFault> { self.compiler.encoding_workspace_bytes() }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.recipe.check_job_limits(limits) }
}

/// The same consuming native adapter, with its per-invocation prepared seal.
/// No wire result or arbitrary model callback can enter this durable adapter.
pub struct Int8SemanticExtractionJobProcessor<'e, 'w, 'v, A: Int8SemanticExtractionBatchAdmission> {
    native: NativeInt8SemanticExtractionBatch<'e, 'w, 'v, A>,
    recipe: Arc<Int8SemanticExtractionJobRecipe>,
}
impl<'e, 'w, 'v, A: Int8SemanticExtractionBatchAdmission> Int8SemanticExtractionJobProcessor<'e, 'w, 'v, A> {
    pub fn new(planner: Int8SemanticExtractionJobPlanner, engine: &'e mut StrictInt8Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        Self::scheduled(planner, engine, vocabulary, admission, None)
    }
    pub fn new_layer_major(planner: Int8SemanticExtractionJobPlanner, engine: &'e mut StrictInt8Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A, prefill: Int8PrefillLimits) -> Result<Self, BatchFault> {
        Self::scheduled(planner, engine, vocabulary, admission, Some(prefill))
    }
    fn scheduled(planner: Int8SemanticExtractionJobPlanner, engine: &'e mut StrictInt8Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A, prefill: Option<Int8PrefillLimits>)
        -> Result<Self, BatchFault> {
        let Int8SemanticExtractionJobPlanner { compiler, recipe } = planner;
        let native = match prefill {
            None => NativeInt8SemanticExtractionBatch::new_shared(compiler, engine, vocabulary, admission),
            Some(prefill) => NativeInt8SemanticExtractionBatch::new_shared_layer_major(compiler, engine,
                vocabulary, admission, prefill),
        }?;
        Ok(Self { native, recipe })
    }
}
fn shared_document(input: BatchDocument<()>) -> Result<BatchDocument<ExtractionBatchArgs>, BatchItemFailure> {
    if input.task_args.is_some() { return Err(BatchItemFailure::reject(BatchCode::InvalidEnvelope)); }
    Ok(BatchDocument { id: input.id, text: input.text, task_args: None })
}
impl<A: Int8SemanticExtractionBatchAdmission> BatchProcessor for Int8SemanticExtractionJobProcessor<'_, '_, '_, A> {
    type Args = ();
    type Prepared = PreparedInt8SemanticBatchExtraction;
    type Output = GuardedOutput<Int8SemanticExtractionRun, A::Guard>;
    fn prepare(&mut self, input: BatchDocument<()>) -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare(shared_document(input)?)
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, input: BatchDocument<()>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare_with_control(shared_document(input)?, control)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { self.native.planned_work(prepared) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> { self.native.execute(prepared, control) }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.native.execute_with_context(prepared, context, control)
    }
}
impl<A: Int8SemanticExtractionBatchAdmission> DurableBatchProcessor for Int8SemanticExtractionJobProcessor<'_, '_, '_, A> {
    type Recipe = Int8SemanticExtractionJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.recipe.extraction_identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.recipe.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        self.native.verify_prepared(prepared)?;
        let config = &self.recipe.config;
        if prepared.model_work() != config.options.max_model_work
            || prepared.max_result_bytes() != config.options.limits.max_output_bytes {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        Ok(JobWork { model: prepared.model_work(), mask_node_visits: config.work.masks.max_visits_per_item })
    }
    fn max_result_bytes(&self, prepared: &Self::Prepared) -> u64 { prepared.max_result_bytes() }
}

#[cfg(test)] mod tests;
