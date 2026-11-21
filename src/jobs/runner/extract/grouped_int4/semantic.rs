//! Authenticated durable extraction with complete correlated field reviews.
//!
//! The existing native semantic processor remains the only execution path.
//! JobRunner durably spends the entire declared attempt before extraction;
//! failure in either stage cannot retain extraction alone as a successful item.

use std::sync::Arc;
use serde::{Serialize, Serializer};
use super::{CompilerRecipe, shared_document};
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork,
        extract::{ExtractionBatchArgs, ExtractionMaskBudget, GuardedOutput, EXTRACTION_BATCH_PROMPT,
            grouped_int4::Int4ExtractionBatchPlanner,
            semantic::{Int4SemanticExtractionBatchAdmission, Int4SemanticExtractionBatchLimits,
                Int4SemanticExtractionBatchPlanner, NativeInt4SemanticExtractionBatch,
                PreparedInt4SemanticBatchExtraction}}},
    execution_identity::ExecutionIdentity,
    grammar::{CompileLimits, mask::MaskWorkLimits,
        runtime::{SourceRuntimeLimits, JSON_RUNTIME_VERSION, SOURCE_JSON_RUNTIME_VERSION}},
    jobs::{JobLimits, JobWork, manifest::bounded_json, runner::DurableBatchProcessor},
    native_engine::{decode::DecodeStepControl, lmhead::scoring::ScoringLimits,
        strict_int4::{Int4ModelIdentity, Int4Work, StrictInt4Engine}},
    tasks::{extract::{ExtractionVocabulary, semantic::{CLAIM_RENDER_VERSION, SEMANTIC_VERIFICATION_VERSION,
            grouped_int4::{Int4SemanticExtractionRun, Int4SemanticOptions, INT4_SEMANTIC_EXECUTION}}},
        judge::{JudgeLimits, FAITHFULNESS_VERSION, JUDGE_PROMPT_VERSION,
            grouped_int4::{Int4JudgePlanner, INT4_JUDGE_EXECUTION}}},
    tokenizer::untrusted::UntrustedEncoding,
};

/// Reconciled whole-document extraction with complete-source semantic reviews.
pub mod document;

/// Bound for the exact existing private recipe and its serialization workspace.
pub(crate) const SEMANTIC_EXTRACTION_JOB_RECIPE_BYTES: usize = 1024 * 1024 - 1024;
const INPUT_POLICY: &str = "fixed-schema-and-semantic-review-no-record-overrides-v1";
const ENCODING_SCOPE: &str = "one-complete-document-and-all-field-reviews-v1";

/// Exact task settings, separate from physical host and IO reservations.
/// The extraction and judge budgets own their complete native KV ceilings.
/// No Debug/Deserialize: schema and claim templates may contain private data.
#[derive(Clone)]
pub struct Int4SemanticExtractionJobConfig {
    pub extraction: ExtractionBatchArgs,
    pub compiler_limits: CompileLimits,
    pub source_limits: SourceRuntimeLimits,
    pub options: Int4SemanticOptions,
    pub judge_planning: JudgeLimits,
    pub encoding: UntrustedEncoding,
    /// Must equal the immutable job's six lifetime work axes. Each attempt
    /// spends options.max_model_work and masks.max_visits_per_item in full.
    pub work: Int4SemanticExtractionBatchLimits,
}

/// Private recipe ownership, not an executable wire factory. Serialization
/// borrows the one configuration owner; only its keyed job commitment is
/// persisted. Both actual compiler identities are frozen independently.
pub struct Int4SemanticExtractionJobRecipe {
    config: Int4SemanticExtractionJobConfig,
    extraction_identity: ExecutionIdentity,
    judge_identity: ExecutionIdentity,
    group_width: usize,
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
    max_model_work: Int4Work,
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
    options: &'a Int4SemanticOptions,
    judge_planning: JudgeRecipe,
    input_encoding: &'a UntrustedEncoding,
    work: WorkRecipe,
}
impl<'a> From<&'a Int4SemanticExtractionJobConfig> for Settings<'a> {
    fn from(config: &'a Int4SemanticExtractionJobConfig) -> Self {
        // Exhaustive projections force future non-serializable compiler,
        // scorer and native work fields into the immutable recipe.
        let CompileLimits { max_schema_bytes, max_string_bytes, max_array_items,
            max_output_bytes, max_states, max_transitions, max_mask_bytes } = config.compiler_limits;
        let Int4SemanticExtractionBatchLimits { max_model_work, masks } = config.work;
        let ExtractionMaskBudget { per_mask, max_visits_per_item, max_visits_per_run } = masks;
        let MaskWorkLimits { max_trie_node_visits, checkpoint_interval_nodes } = per_mask;
        Self { extraction: &config.extraction, compiler: CompilerRecipe { max_schema_bytes,
            max_string_bytes, max_array_items, max_output_bytes, max_states, max_transitions, max_mask_bytes },
            source: config.source_limits, options: &config.options,
            judge_planning: config.judge_planning.into(), input_encoding: &config.encoding,
            work: WorkRecipe { max_model_work, max_trie_node_visits, checkpoint_interval_nodes,
                max_visits_per_item, max_visits_per_run } }
    }
}
#[derive(Serialize)]
struct RecipeView<'a> {
    version: u32,
    dependency_scope: &'static str,
    input_policy: &'static str,
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
    group_width: usize,
    extraction_identity: &'a ExecutionIdentity,
    judge_identity: &'a ExecutionIdentity,
    settings: Settings<'a>,
}
impl Serialize for Int4SemanticExtractionJobRecipe {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        RecipeView { version: 1, dependency_scope: "item-local", input_policy: INPUT_POLICY,
            execution: INT4_SEMANTIC_EXECUTION, extraction_prompt: EXTRACTION_BATCH_PROMPT,
            json_runtime: JSON_RUNTIME_VERSION, source_runtime: SOURCE_JSON_RUNTIME_VERSION,
            semantic_verification: SEMANTIC_VERIFICATION_VERSION, claim_rendering: CLAIM_RENDER_VERSION,
            judge_execution: INT4_JUDGE_EXECUTION, judge_prompt: JUDGE_PROMPT_VERSION,
            faithfulness: FAITHFULNESS_VERSION, encoding_scope: ENCODING_SCOPE,
            group_width: self.group_width, extraction_identity: &self.extraction_identity,
            judge_identity: &self.judge_identity, settings: (&self.config).into() }.serialize(serializer)
    }
}
impl Int4SemanticExtractionJobRecipe {
    fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
        let config = &self.config;
        config.work.validate_item(config.options.max_model_work)?;
        let lifetime = JobWork::from_int4(config.work.max_model_work, config.work.masks.max_visits_per_run);
        if limits.max_work != lifetime || config.options.limits.max_output_bytes > limits.max_result_bytes as u64 {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}

/// Immutable actual-model factory. All fixed schema/review declarations are
/// checked before any job storage is opened; each source is separately planned.
pub struct Int4SemanticExtractionJobPlanner {
    compiler: Arc<Int4SemanticExtractionBatchPlanner>,
    recipe: Int4SemanticExtractionJobRecipe,
}
impl Int4SemanticExtractionJobPlanner {
    pub fn for_model(model: &Int4ModelIdentity, config: Int4SemanticExtractionJobConfig)
        -> Result<Self, BatchFault> {
        config.extraction.budget.validate().map_err(|_| BatchCode::InvalidLimits)?;
        config.options.validate().map_err(|_| BatchCode::InvalidLimits)?;
        config.work.validate_item(config.options.max_model_work)?;
        encoding_workspace(&config.encoding)?;
        check_judge_limits(config.judge_planning)?;
        // Bound the owned schema and templates before cloning them into the
        // immutable genuine compilers. No synthetic document is compiled.
        bounded_json(&Settings::from(&config), SEMANTIC_EXTRACTION_JOB_RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        let extraction = Int4ExtractionBatchPlanner::for_model(model, config.extraction.budget,
            config.compiler_limits, config.source_limits, Some(config.extraction.clone()))?
            .with_document_encoding(config.encoding.clone())?;
        let judge = Arc::new(Int4JudgePlanner::for_model(model, config.options.specification.judge_budget,
            config.judge_planning).and_then(|planner| planner.with_document_encoding(config.encoding.clone()))
            .map_err(|_| BatchCode::Admission)?);
        let compiler = Arc::new(Int4SemanticExtractionBatchPlanner::new(extraction, judge, config.options.clone(), config.work)?);
        let recipe = Int4SemanticExtractionJobRecipe { config, group_width: model.group_width(),
            extraction_identity: compiler.base_execution_identity().clone(), judge_identity: compiler.judge_identity().clone() };
        bounded_json(&recipe, SEMANTIC_EXTRACTION_JOB_RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(Self { compiler, recipe })
    }
    /// Retain the genuine tokenizer/schema/judge owner outside the worker's
    /// processor so unwind and queue discard cannot release it before join.
    pub(crate) fn retain_compiler(&self) -> Arc<Int4SemanticExtractionBatchPlanner> {
        Arc::clone(&self.compiler)
    }
    pub fn config(&self) -> &Int4SemanticExtractionJobConfig { &self.recipe.config }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.recipe.extraction_identity }
    pub fn judge_identity(&self) -> &ExecutionIdentity { &self.recipe.judge_identity }
    pub fn max_result_bytes(&self) -> u64 { self.recipe.config.options.limits.max_output_bytes }
    pub fn encoding_workspace_bytes(&self) -> Result<u64, BatchFault> { encoding_workspace(&self.recipe.config.encoding) }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.recipe.check_job_limits(limits) }
}
fn encoding_workspace(encoding: &UntrustedEncoding) -> Result<u64, BatchFault> {
    encoding.validate().map_err(|_| BatchCode::InvalidLimits)?;
    match encoding {
        UntrustedEncoding::ByteFallback => Ok(0),
        UntrustedEncoding::SafeBpeV1(limits) => limits.requirements().map(|required| required.workspace_bytes)
            .map_err(|_| BatchCode::InvalidLimits.into()),
    }
}
fn check_judge_limits(limits: JudgeLimits) -> Result<(), BatchFault> {
    let l = JudgeRecipe::from(limits); let p = l.per_head;
    if p.max_candidates == 0 || p.max_total_tokens == 0 || p.max_nodes == 0 || p.max_depth == 0
        || p.max_candidate_id_bytes == 0 || p.max_projected_logits == 0 || l.max_total_prompt_tokens == 0
        || l.max_total_candidate_tokens == 0 || l.max_total_projected_logits == 0 || l.max_output_bytes == 0 {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}

/// One genuine extraction-plus-review processor. Its guarded combined result
/// survives serialization, result-spool sync and journal acknowledgement.
pub struct Int4SemanticExtractionJobProcessor<'e, 'w, 'v, A: Int4SemanticExtractionBatchAdmission> {
    native: NativeInt4SemanticExtractionBatch<'e, 'w, 'v, A>,
    recipe: Int4SemanticExtractionJobRecipe,
}
impl<'e, 'w, 'v, A: Int4SemanticExtractionBatchAdmission> Int4SemanticExtractionJobProcessor<'e, 'w, 'v, A> {
    pub fn new(planner: Int4SemanticExtractionJobPlanner, engine: &'e mut StrictInt4Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        let Int4SemanticExtractionJobPlanner { compiler, recipe } = planner;
        let native = NativeInt4SemanticExtractionBatch::new_shared(compiler, engine, vocabulary, admission)?;
        Ok(Self { native, recipe })
    }
}
impl<A: Int4SemanticExtractionBatchAdmission> BatchProcessor for Int4SemanticExtractionJobProcessor<'_, '_, '_, A> {
    // Reject all non-null task_args while freezing the complete input stream,
    // before any job file or model forward. Config owns every shared setting.
    type Args = ();
    type Prepared = PreparedInt4SemanticBatchExtraction;
    type Output = GuardedOutput<Int4SemanticExtractionRun, A::Guard>;
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
impl<A: Int4SemanticExtractionBatchAdmission> DurableBatchProcessor for Int4SemanticExtractionJobProcessor<'_, '_, '_, A> {
    type Recipe = Int4SemanticExtractionJobRecipe;
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
        // Review claims do not exist before extraction. Reserve the complete
        // declared combined attempt, never only extraction or observed work.
        Ok(JobWork::from_int4(prepared.model_work(), config.work.masks.max_visits_per_item))
    }
    fn max_result_bytes(&self, prepared: &Self::Prepared) -> u64 { prepared.max_result_bytes() }
}

#[cfg(test)]
mod tests;
