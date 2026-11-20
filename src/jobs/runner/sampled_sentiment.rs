//! Authenticated retained sampled sentiment on the resident native processor.
//!
//! The immutable factory freezes the actual compiler's private seed, complete
//! sampling policy and every resource ceiling. The existing job runner owns
//! population authentication, persistent failed-attempt debits and spool-first
//! publication. No alternate generator or deserialized recipe grants execution.

use serde::Serialize;
use std::sync::Arc;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, extract::ExtractionMaskBudget, output::GuardedOutput,
        sentiment::sampled::{NativeSampledSentimentBatch, PreparedSampledSentimentRecord,
            SampledSentimentBatchAdmission, SampledSentimentBatchArgs, SampledSentimentBatchLimits,
            SampledSentimentBatchPlanner, SAMPLED_SENTIMENT_JOB_RECIPE_BYTES}},
    execution_identity::ExecutionIdentity,
    grammar::{CompileLimits, mask::MaskWorkLimits, runtime::JSON_RUNTIME_VERSION},
    jobs::{JobLimits, JobWork, manifest::bounded_json, runner::DurableBatchProcessor},
    native_engine::{constrained_int4::{INT4_JSON_SAMPLED_EXECUTION,
            INT4_JSON_SAMPLED_REUSED_PREFILL_EXECUTION, JSON_SAMPLER_VERSION},
        decode::DecodeStepControl, strict_int4::{Int4MemoryRequirement, Int4Work, StrictInt4Engine,
            prefill::{Int4PrefillLimits, INT4_LAYER_MAJOR_PREFILL_VERSION}}},
    tasks::{ir::TaskBudget, sentiment::{SentimentAxis, sampled::{SampledSentimentLimits, SampledSentimentOptions,
        SampledSentimentPlanner, SampledSentimentRun, SampledSentimentVocabulary, MAX_SENTIMENT_SAMPLES,
        SAMPLED_SENTIMENT_AGGREGATION_VERSION, SAMPLED_SENTIMENT_EXECUTION,
        SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION,
        SAMPLED_SENTIMENT_ITEM_ADDRESS_VERSION, SAMPLED_SENTIMENT_PROMPT_VERSION}}},
    tokenizer::untrusted::UntrustedEncoding,
};

pub use crate::batch::sentiment::sampled::sampled_sentiment_job_preparation_bytes;

/// Serialize only for the owned job's keyed commitment. There is no Debug,
/// Deserialize, public field or caller-supplied constructor; a recipe cannot
/// become a model factory, a public receipt or an independent sampling seed.
#[derive(Serialize)]
pub struct SampledSentimentJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    input_policy: &'static str,
    execution: &'static str,
    prompt_version: &'static str,
    aggregation_version: &'static str,
    sampler_version: &'static str,
    item_address_version: &'static str,
    sample_address_stride: usize,
    native_execution: &'static str,
    grammar_version: &'static str,
    task_spec: &'static str,
    group_width: usize,
    task_budget: TaskBudget,
    default_axes: Vec<SentimentAxis>,
    sampling: SampledSentimentOptions,
    planning: PlanningRecipe,
    native: NativeRecipe,
    input_encoding: UntrustedEncoding,
    job: JobLimits,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill: Option<PrefillRecipe>,
}

#[derive(Serialize)]
struct PrefillRecipe {
    execution: &'static str,
    max_batch_rows: usize,
    max_extra_scratch_bytes: u64,
}

#[derive(Serialize)]
struct PlanningRecipe {
    max_input_bytes: usize,
    max_total_prompt_tokens: usize,
    max_sample_json_bytes: usize,
    max_result_bytes: u64,
    compile: CompilerRecipe,
}
#[derive(Serialize)]
struct CompilerRecipe {
    max_schema_bytes: usize,
    max_string_bytes: usize,
    max_array_items: usize,
    max_output_bytes: usize,
    max_states: usize,
    max_transitions: usize,
    max_mask_bytes: usize,
}
impl From<SampledSentimentLimits> for PlanningRecipe {
    fn from(limits: SampledSentimentLimits) -> Self {
        // Exhaustive patterns make future compiler fields part of the review.
        let SampledSentimentLimits { max_input_bytes, max_total_prompt_tokens,
            max_sample_json_bytes, max_result_bytes, compile } = limits;
        let CompileLimits { max_schema_bytes, max_string_bytes, max_array_items,
            max_output_bytes, max_states, max_transitions, max_mask_bytes } = compile;
        Self { max_input_bytes, max_total_prompt_tokens, max_sample_json_bytes, max_result_bytes,
            compile: CompilerRecipe { max_schema_bytes, max_string_bytes, max_array_items,
                max_output_bytes, max_states, max_transitions, max_mask_bytes } }
    }
}
#[derive(Serialize)]
struct NativeRecipe {
    max_item_work: Int4Work,
    max_model_work: Int4Work,
    max_mask_trie_node_visits: usize,
    mask_checkpoint_interval_nodes: usize,
    max_mask_visits_per_item: u64,
    max_mask_visits_per_run: u64,
    max_sampler_bytes: u64,
}
impl From<SampledSentimentBatchLimits> for NativeRecipe {
    fn from(limits: SampledSentimentBatchLimits) -> Self {
        let SampledSentimentBatchLimits { max_item_work, max_model_work, masks, max_sampler_bytes } = limits;
        let ExtractionMaskBudget { per_mask, max_visits_per_item, max_visits_per_run } = masks;
        let MaskWorkLimits { max_trie_node_visits, checkpoint_interval_nodes } = per_mask;
        Self { max_item_work, max_model_work, max_mask_trie_node_visits: max_trie_node_visits,
            mask_checkpoint_interval_nodes: checkpoint_interval_nodes,
            max_mask_visits_per_item: max_visits_per_item, max_mask_visits_per_run: max_visits_per_run,
            max_sampler_bytes }
    }
}
impl SampledSentimentJobRecipe {
    fn new(group_width: usize, task_budget: TaskBudget, sampling: SampledSentimentOptions,
        planning: SampledSentimentLimits, native: SampledSentimentBatchLimits,
        input_encoding: UntrustedEncoding, job: JobLimits) -> Result<Self, BatchFault> {
        task_budget.validate().map_err(|_| BatchCode::InvalidLimits)?;
        sampling.validate().map_err(|_| BatchCode::InvalidLimits)?;
        planning.validate().map_err(|_| BatchCode::InvalidLimits)?;
        native.validate()?;
        input_encoding.validate().map_err(|_| BatchCode::InvalidLimits)?;
        Int4MemoryRequirement::for_context(1, group_width).map_err(|_| BatchCode::InvalidLimits)?;
        let recipe = Self { version: 2, dependency_scope: "item-local",
            input_policy: "exact-record-independent-axis-sample-addresses-v1",
            execution: SAMPLED_SENTIMENT_EXECUTION, prompt_version: SAMPLED_SENTIMENT_PROMPT_VERSION,
            aggregation_version: SAMPLED_SENTIMENT_AGGREGATION_VERSION,
            sampler_version: JSON_SAMPLER_VERSION, grammar_version: JSON_RUNTIME_VERSION,
            item_address_version: SAMPLED_SENTIMENT_ITEM_ADDRESS_VERSION,
            sample_address_stride: MAX_SENTIMENT_SAMPLES, native_execution: INT4_JSON_SAMPLED_EXECUTION,
            task_spec: "sentiment-v1", group_width, task_budget,
            default_axes: SampledSentimentBatchArgs::default().axes, sampling,
            planning: planning.into(), native: native.into(), input_encoding, job, prefill: None };
        recipe.check_job_limits(job)?;
        bounded_json(&recipe, SAMPLED_SENTIMENT_JOB_RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(recipe)
    }
    fn with_prefill(mut self, limits: Int4PrefillLimits) -> Result<Self, BatchFault> {
        limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
        let Int4PrefillLimits { max_batch_rows, max_extra_scratch_bytes } = limits;
        self.prefill = Some(PrefillRecipe { execution: INT4_LAYER_MAJOR_PREFILL_VERSION,
            max_batch_rows, max_extra_scratch_bytes });
        bounded_json(&self, SAMPLED_SENTIMENT_JOB_RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(self)
    }
    fn with_reused_prefill(mut self) -> Result<Self, BatchFault> {
        // These existing fields bind the physical execution policy without
        // changing the canonical recipe of an already retained cold job.
        self.execution = SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION;
        self.native_execution = INT4_JSON_SAMPLED_REUSED_PREFILL_EXECUTION;
        bounded_json(&self, SAMPLED_SENTIMENT_JOB_RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(self)
    }
    fn reuses_prefill(&self) -> bool { self.execution == SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION }
    fn prefill_limits(&self) -> Option<Int4PrefillLimits> {
        self.prefill.as_ref().map(|prefill| Int4PrefillLimits { max_batch_rows: prefill.max_batch_rows,
            max_extra_scratch_bytes: prefill.max_extra_scratch_bytes })
    }
    fn max_result_bytes(&self) -> u64 { self.task_budget.max_output_bytes.min(self.planning.max_result_bytes) }
    fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
        if limits != self.job || limits.max_id_bytes > 256
            || self.planning.max_input_bytes > limits.max_input_bytes_per_item
            || limits.max_work != JobWork::from_int4(self.native.max_model_work, self.native.max_mask_visits_per_run)
            || self.task_budget.max_output_bytes.min(self.planning.max_result_bytes) > limits.max_result_bytes as u64 {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}

/// The actual model-bound resident compiler moves intact into this factory.
/// Explicit resume reconstructs the same private recipe. Attempt and delivery
/// counters never change sample addresses. A chosen prompt schedule is frozen
/// separately so a resumed job cannot mix unqualified physical execution paths.
pub struct SampledSentimentJobPlanner {
    compiler: SampledSentimentBatchPlanner,
    identity: ExecutionIdentity,
    recipe: SampledSentimentJobRecipe,
    planning: SampledSentimentLimits,
    preparation_bytes: u64,
}
impl SampledSentimentJobPlanner {
    pub fn new(planner: SampledSentimentPlanner, vocabulary: Arc<SampledSentimentVocabulary>,
        native: SampledSentimentBatchLimits, limits: JobLimits) -> Result<Self, BatchFault> {
        let task = planner.task_budget();
        let compiler = SampledSentimentBatchPlanner::new(planner, vocabulary, task, native)?;
        let (options, planning) = compiler.durable_configuration();
        let recipe = SampledSentimentJobRecipe::new(compiler.group_width(), task,
            options.clone(), planning, native, compiler.document_encoding().clone(), limits)?;
        let preparation_bytes = sampled_sentiment_job_preparation_bytes(task, planning, compiler.document_encoding())?;
        let identity = compiler.base_execution_identity().clone();
        Ok(Self { compiler, identity, recipe, planning, preparation_bytes })
    }
    /// Select before storage is opened; rows and workspace enter the private
    /// recipe so changing either requires a distinct durable job.
    pub fn with_layer_major_prefill(mut self, limits: Int4PrefillLimits) -> Result<Self, BatchFault> {
        self.recipe = self.recipe.with_prefill(limits)?;
        Ok(self)
    }
    /// Reuse only the current axis's prompt within each complete document.
    /// The explicit policy is authenticated before any durable execution, so
    /// resume cannot switch between cold and reused-prefill results. Every
    /// attempt still reserves the full cold native and mask work allowance.
    pub fn with_reused_prefill(mut self) -> Result<Self, BatchFault> {
        self.recipe = self.recipe.with_reused_prefill()?;
        Ok(self)
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_budget(&self) -> TaskBudget { self.compiler.task_budget() }
    pub fn native_limits(&self) -> SampledSentimentBatchLimits { self.compiler.limits() }
    pub fn planning_limits(&self) -> SampledSentimentLimits { self.planning }
    pub fn group_width(&self) -> usize { self.compiler.group_width() }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.compiler.document_encoding() }
    pub fn preparation_requirement_bytes(&self) -> u64 { self.preparation_bytes }
    pub fn max_result_bytes(&self) -> u64 { self.recipe.max_result_bytes() }
    pub fn prefill_limits(&self) -> Option<Int4PrefillLimits> { self.recipe.prefill_limits() }
    pub fn reuses_prefill(&self) -> bool { self.recipe.reuses_prefill() }
    pub fn job_limits(&self) -> JobLimits { self.recipe.job }
    /// Private configuration for keyed commitment only; do not publish it.
    pub fn job_recipe(&self) -> &SampledSentimentJobRecipe { &self.recipe }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        self.recipe.check_job_limits(limits)
    }
}

/// One native processor, with the same complete bundle validator and guards
/// as live sampled populations. The guard remains inside the serializable
/// output until its result spool and journal acknowledgement have completed.
pub struct SampledSentimentJobProcessor<'e, 'w, A: SampledSentimentBatchAdmission> {
    native: NativeSampledSentimentBatch<'e, 'w, A>,
    identity: ExecutionIdentity,
    recipe: SampledSentimentJobRecipe,
}
impl<'e, 'w, A: SampledSentimentBatchAdmission> SampledSentimentJobProcessor<'e, 'w, A> {
    pub fn new(planner: SampledSentimentJobPlanner, engine: &'e mut StrictInt4Engine<'w>, admission: A)
        -> Result<Self, BatchFault> {
        let SampledSentimentJobPlanner { compiler, identity, recipe, .. } = planner;
        let native = if recipe.reuses_prefill() {
            NativeSampledSentimentBatch::new_reusing_prefill(compiler, engine, admission,
                recipe.prefill_limits())?
        } else { match recipe.prefill_limits() {
            Some(prefill) => NativeSampledSentimentBatch::new_layer_major(compiler, engine, admission, prefill)?,
            None => NativeSampledSentimentBatch::new(compiler, engine, admission)?,
        } };
        Ok(Self { native, identity, recipe })
    }
}
impl<A: SampledSentimentBatchAdmission> BatchProcessor for SampledSentimentJobProcessor<'_, '_, A> {
    type Args = SampledSentimentBatchArgs;
    type Prepared = PreparedSampledSentimentRecord;
    type Output = GuardedOutput<SampledSentimentRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare(document)
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> { self.native.prepare_with_control(document, control) }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { self.native.planned_work(prepared) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> { self.native.execute(prepared, control) }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.native.execute_with_context(prepared, context, control)
    }
}
impl<A: SampledSentimentBatchAdmission> DurableBatchProcessor for SampledSentimentJobProcessor<'_, '_, A> {
    type Recipe = SampledSentimentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.recipe.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        if self.native.is_poisoned() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        // Reserve the complete independent sample schedule BEFORE execution.
        // Early EOS, cancellation and failed attempts cannot refund its work.
        Ok(JobWork::from_int4(prepared.model_work(), prepared.mask_node_visits()))
    }
    fn max_result_bytes(&self, prepared: &Self::Prepared) -> u64 { prepared.max_result_bytes() }
}

#[cfg(test)] mod tests;
