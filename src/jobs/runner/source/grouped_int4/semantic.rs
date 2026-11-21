//! Authenticated retained summaries and passage answers with complete-source
//! correlated review. One durable item owns generation and every second-reader
//! result; a failed review cannot commit the source result on its own.

use std::sync::Arc;
use serde::{Serialize, Serializer};
use super::{CompilerRecipe, PlanningRecipe};
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, extract::{ExtractionMaskBudget, GuardedOutput},
        source::semantic::{Int4SemanticSourceBatchAdmission, Int4SemanticSourceBatchArgs,
            Int4SemanticSourceBatchLimits, Int4SemanticSourceBatchPlanner,
            Int4SemanticSourceBatchTask, NativeInt4SemanticSourceBatch, PreparedInt4SemanticBatchSource}},
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    grammar::{CompileLimits, mask::MaskWorkLimits,
        runtime::{JSON_RUNTIME_VERSION, SOURCE_JSON_RUNTIME_VERSION}},
    jobs::{JobLimits, JobWork, manifest::bounded_json, runner::DurableBatchProcessor},
    native_engine::{decode::DecodeStepControl, generation::grouped_int4::bind_candidate_model,
        lmhead::scoring::ScoringLimits,
        strict_int4::{Int4ModelIdentity, Int4Work, StrictInt4Engine, STRICT_INT4_EXECUTION}},
    tasks::{answer::{AnswerInputLimits, ANSWER_PASSAGE_LAYOUT_VERSION}, extract::ExtractionVocabulary,
        judge::{JudgeLimits, FAITHFULNESS_VERSION, JUDGE_PROMPT_VERSION,
            grouped_int4::{Int4JudgePlanner, INT4_JUDGE_EXECUTION}},
        source_planning::{SourcePlanningLimits, SourceTaskPlanner, SOURCE_PROMPT_VERSION,
            grouped_int4::{INT4_SOURCE_EXECUTION, semantic::{Int4SourceSemanticOptions,
                Int4SourceSemanticRun, INT4_SOURCE_SEMANTIC_EXECUTION, SOURCE_SEMANTIC_VERSION,
                SOURCE_SEMANTIC_CLAIM_VERSION, SOURCE_SEMANTIC_SCOPE, SOURCE_SEMANTIC_CALIBRATION}}}},
    tokenizer::untrusted::UntrustedEncoding,
};

/// The exact private recipe and each serialization buffer have this ceiling.
pub(crate) const SEMANTIC_SOURCE_JOB_RECIPE_BYTES: usize = 1024 * 1024 - 1024;
const INPUT_POLICY: &str = "fixed-summary-or-answer-complete-original-source-and-review-v1";
const ENCODING_SCOPE: &str = "one-source-or-question-manifest-passages-and-all-statement-reviews-v1";

/// Fixed source-generation and complete-statement review policy. Input records
/// provide only original documents or original questions and ordered passages.
/// No Debug/Deserialize: future source settings may contain private material.
#[derive(Clone)]
pub struct Int4SemanticSourceJobConfig {
    pub task: Int4SemanticSourceBatchTask,
    pub planning: SourcePlanningLimits,
    pub options: Int4SourceSemanticOptions,
    pub judge_planning: JudgeLimits,
    pub encoding: UntrustedEncoding,
    /// Exactly the immutable job's six lifetime work ceilings. Each attempt
    /// spends options.max_model_work and masks.max_visits_per_item in full.
    pub work: Int4SemanticSourceBatchLimits,
}

/// Private immutable recipe, never a deserialized executable. Serialization
/// borrows its single configuration owner. Only the keyed commitment enters
/// the job manifest, and both identities come from the actual model compilers.
pub struct Int4SemanticSourceJobRecipe {
    config: Int4SemanticSourceJobConfig,
    source_identity: ExecutionIdentity,
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
    task: Int4SemanticSourceBatchTask,
    planning: PlanningRecipe,
    /// Native source planning applies this one aggregate bound to question,
    /// canonical passage manifest and joined original evidence before encoding.
    answer_input: Option<AnswerInputLimits>,
    options: &'a Int4SourceSemanticOptions,
    judge_planning: JudgeRecipe,
    input_encoding: &'a UntrustedEncoding,
    work: WorkRecipe,
}
impl<'a> From<&'a Int4SemanticSourceJobConfig> for Settings<'a> {
    fn from(config: &'a Int4SemanticSourceJobConfig) -> Self {
        // Exhaustive projections make added compiler, source planning, scorer
        // and native work fields impossible to omit from the private recipe.
        let SourcePlanningLimits { max_input_bytes, max_context_tokens, max_passages, compiler, source } = config.planning;
        let CompileLimits { max_schema_bytes, max_string_bytes, max_array_items, max_output_bytes,
            max_states, max_transitions, max_mask_bytes } = compiler;
        let Int4SemanticSourceBatchLimits { max_model_work, masks } = config.work;
        let ExtractionMaskBudget { per_mask, max_visits_per_item, max_visits_per_run } = masks;
        let MaskWorkLimits { max_trie_node_visits, checkpoint_interval_nodes } = per_mask;
        let answer_input = matches!(config.task, Int4SemanticSourceBatchTask::Answer { .. }).then_some(
            AnswerInputLimits { max_passages, max_input_bytes,
                max_input_tokens: config.task.budget().max_input_tokens as usize });
        Self { task: config.task, planning: PlanningRecipe { max_input_bytes, max_context_tokens, max_passages,
                compiler: CompilerRecipe { max_schema_bytes, max_string_bytes, max_array_items,
                    max_output_bytes, max_states, max_transitions, max_mask_bytes }, source },
            answer_input, options: &config.options, judge_planning: config.judge_planning.into(),
            input_encoding: &config.encoding, work: WorkRecipe { max_model_work, max_trie_node_visits,
                checkpoint_interval_nodes, max_visits_per_item, max_visits_per_run } }
    }
}
#[derive(Serialize)]
struct RecipeView<'a> {
    version: u32,
    dependency_scope: &'static str,
    input_policy: &'static str,
    execution: &'static str,
    source_execution: &'static str,
    source_prompt: &'static str,
    answer_passage_layout: &'static str,
    json_runtime: &'static str,
    source_runtime: &'static str,
    semantic_verification: &'static str,
    claim_rendering: &'static str,
    semantic_scope: &'static str,
    semantic_calibration: &'static str,
    judge_execution: &'static str,
    judge_prompt: &'static str,
    faithfulness: &'static str,
    encoding_scope: &'static str,
    group_width: usize,
    source_identity: &'a ExecutionIdentity,
    judge_identity: &'a ExecutionIdentity,
    settings: Settings<'a>,
}
impl Serialize for Int4SemanticSourceJobRecipe {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        RecipeView { version: 1, dependency_scope: "item-local", input_policy: INPUT_POLICY,
            execution: INT4_SOURCE_SEMANTIC_EXECUTION, source_execution: INT4_SOURCE_EXECUTION,
            source_prompt: SOURCE_PROMPT_VERSION, answer_passage_layout: ANSWER_PASSAGE_LAYOUT_VERSION,
            json_runtime: JSON_RUNTIME_VERSION, source_runtime: SOURCE_JSON_RUNTIME_VERSION,
            semantic_verification: SOURCE_SEMANTIC_VERSION, claim_rendering: SOURCE_SEMANTIC_CLAIM_VERSION,
            semantic_scope: SOURCE_SEMANTIC_SCOPE, semantic_calibration: SOURCE_SEMANTIC_CALIBRATION,
            judge_execution: INT4_JUDGE_EXECUTION, judge_prompt: JUDGE_PROMPT_VERSION,
            faithfulness: FAITHFULNESS_VERSION, encoding_scope: ENCODING_SCOPE,
            group_width: self.group_width, source_identity: &self.source_identity,
            judge_identity: &self.judge_identity, settings: (&self.config).into() }.serialize(serializer)
    }
}
impl Int4SemanticSourceJobRecipe {
    fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
        let config = &self.config;
        config.work.validate_item(config.options.max_model_work)?;
        let lifetime = JobWork::from_int4(config.work.max_model_work, config.work.masks.max_visits_per_run);
        if limits.max_work != lifetime || limits.max_id_bytes > 256
            || config.planning.max_input_bytes > limits.max_input_bytes_per_item
            || config.options.limits.max_output_bytes > limits.max_result_bytes as u64 {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}

/// Own the genuine pinned source and judge compilers under one actual model.
/// All fixed policy is checked before job storage. Record-specific summary/QA
/// envelope shape, original evidence and prompt limits are checked in prepare.
pub struct Int4SemanticSourceJobPlanner {
    compiler: Arc<Int4SemanticSourceBatchPlanner<'static>>,
    recipe: Int4SemanticSourceJobRecipe,
}
impl Int4SemanticSourceJobPlanner {
    pub fn for_model(model: &Int4ModelIdentity, config: Int4SemanticSourceJobConfig)
        -> Result<Self, BatchFault> {
        config.task.validate_semantic(&config.options)?;
        config.planning.validate().map_err(|_| BatchCode::InvalidLimits)?;
        config.work.validate_item(config.options.max_model_work)?;
        encoding_workspace(&config.encoding)?;
        check_judge_limits(config.judge_planning)?;
        if let Some(answer) = Settings::from(&config).answer_input {
            answer.validate().map_err(|_| BatchCode::InvalidLimits)?;
        }
        bounded_json(&Settings::from(&config), SEMANTIC_SOURCE_JOB_RECIPE_BYTES)
            .map_err(|_| BatchCode::InvalidLimits)?;
        let controls = crate::tokenizer::pinned_controls::pinned().map_err(|_| BatchCode::Admission)?;
        let mut ends = controls.template_controls().entries().iter()
            .filter(|entry| entry.special && entry.surface == crate::template::IM_END);
        let eos = ends.next().ok_or(BatchCode::Admission)?.id;
        if ends.next().is_some() { return Err(BatchCode::Admission.into()); }
        let source = Arc::new(SourceTaskPlanner::pinned(controls.template_controls(), eos)
            .and_then(|planner| planner.with_document_encoding(config.encoding.clone()))
            .map_err(|_| BatchCode::Admission)?);
        let identity = source_identity(model, &source, config.task)?;
        let judge = Arc::new(Int4JudgePlanner::for_model(model, config.options.judge_budget, config.judge_planning)
            .and_then(|planner| planner.with_document_encoding(config.encoding.clone()))
            .map_err(|_| BatchCode::Admission)?);
        let compiler = Arc::new(Int4SemanticSourceBatchPlanner::new_owned(source, identity,
            config.task.budget(), config.planning, config.task, judge, config.options.clone(), config.work)?);
        let recipe = Int4SemanticSourceJobRecipe { config, group_width: model.group_width(),
            source_identity: compiler.base_execution_identity().clone(), judge_identity: compiler.judge_identity().clone() };
        bounded_json(&recipe, SEMANTIC_SOURCE_JOB_RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(Self { compiler, recipe })
    }
    /// Keep the actual source tokenizer, templates and judge alive through
    /// physical worker join, including processor unwind and queue discard.
    pub(crate) fn retain_compiler(&self) -> Arc<Int4SemanticSourceBatchPlanner<'static>> {
        Arc::clone(&self.compiler)
    }
    pub fn config(&self) -> &Int4SemanticSourceJobConfig { &self.recipe.config }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.recipe.source_identity }
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
fn source_identity(model: &Int4ModelIdentity, planner: &SourceTaskPlanner,
    task: Int4SemanticSourceBatchTask) -> Result<ExecutionIdentity, BatchFault> {
    let absent = Sha256Digest::of_bytes(b"null");
    // Model authority is bound from the supplied actual model; task-specific
    // prompt/schema bytes remain uncompiled until an original record is planned.
    bind_candidate_model(ExecutionIdentity {
        schema_version: 1, source_revision: "unbound".to_owned(), logical_model_digest: absent,
        artifact_format: "unbound".to_owned(), quant_recipe: "unbound".to_owned(), packing_set_digest: absent,
        tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
        task_spec: task.built_in().spec().identity(), taskir_digest: absent, prompt_digest: absent,
        grammar_compiler_version: "uncompiled".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None,
    }, model).map_err(|_| BatchCode::Admission.into())
}

/// The existing native source-plus-review processor is the sole executable.
/// Its complete output and admission guard survive durable spool sync and ack.
pub struct Int4SemanticSourceJobProcessor<'e, 'w, 'v, A: Int4SemanticSourceBatchAdmission> {
    native: NativeInt4SemanticSourceBatch<'static, 'e, 'w, 'v, A>,
    recipe: Int4SemanticSourceJobRecipe,
}
impl<'e, 'w, 'v, A: Int4SemanticSourceBatchAdmission> Int4SemanticSourceJobProcessor<'e, 'w, 'v, A> {
    pub fn new(planner: Int4SemanticSourceJobPlanner, engine: &'e mut StrictInt4Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        let Int4SemanticSourceJobPlanner { compiler, recipe } = planner;
        let native = NativeInt4SemanticSourceBatch::new_shared(compiler, engine, vocabulary, admission)?;
        Ok(Self { native, recipe })
    }
}
impl<A: Int4SemanticSourceBatchAdmission> BatchProcessor for Int4SemanticSourceJobProcessor<'_, '_, '_, A> {
    type Args = Int4SemanticSourceBatchArgs;
    type Prepared = PreparedInt4SemanticBatchSource;
    type Output = GuardedOutput<Int4SourceSemanticRun, A::Guard>;
    fn prepare(&mut self, input: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare(input)
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, input: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> { self.native.prepare_with_control(input, control) }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { self.native.planned_work(prepared) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> { self.native.execute(prepared, control) }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.native.execute_with_context(prepared, context, control)
    }
}
impl<A: Int4SemanticSourceBatchAdmission> DurableBatchProcessor for Int4SemanticSourceJobProcessor<'_, '_, '_, A> {
    type Recipe = Int4SemanticSourceJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.recipe.source_identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.recipe.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        self.native.verify_prepared(prepared)?;
        let config = &self.recipe.config;
        if prepared.model_work() != config.options.max_model_work
            || prepared.max_result_bytes() != config.options.limits.max_output_bytes {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        // Complete statements are not known before generation. Reserve the
        // whole declared attempt, with no refund for a failed or empty review.
        Ok(JobWork::from_int4(prepared.model_work(), config.work.masks.max_visits_per_item))
    }
    fn max_result_bytes(&self, prepared: &Self::Prepared) -> u64 { prepared.max_result_bytes() }
}

#[cfg(test)]
mod tests;
