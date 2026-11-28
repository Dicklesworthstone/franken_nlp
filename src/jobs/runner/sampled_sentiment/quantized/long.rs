//! Authenticated complete-document sampled INT8 jobs.
//! The journal persists all five native counters and the complete mask debit
//! before inference. Only a sealed complete map reaches the result spool; failed
//! attempts and authenticated restart never refund the journal's lifetime work.
use super::PlanningRecipe;
use serde::Serialize;
use std::sync::Arc;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, extract::ExtractionMaskBudget, generation::GuardedOutput,
        sentiment::sampled::{SampledSentimentBatchArgs, SAMPLED_SENTIMENT_JOB_RECIPE_BYTES,
            quantized::{Int8SampledSentimentBatchLimits, long::{
                Int8SampledSentimentDocumentPlanner, NativeInt8SampledSentimentDocumentBatch,
                PreparedInt8SampledSentimentDocumentRecord, Int8SampledSentimentDocumentBatchAdmission}}}},
    execution_identity::ExecutionIdentity,
    grammar::{mask::MaskWorkLimits, runtime::JSON_RUNTIME_VERSION},
    jobs::{JobLimits, JobWork, manifest::bounded_json, runner::DurableBatchProcessor},
    native_engine::{constrained_int8::{INT8_JSON_SAMPLED_EXECUTION, INT8_JSON_SAMPLED_SPARSE_EXECUTION,
            JSON_SAMPLER_VERSION, sparse::Int8JsonSparseLimits},
        decode::DecodeStepControl, strict_int8::{Int8Work, StrictInt8Engine,
            prefill::{Int8PrefillLimits, LAYER_MAJOR_PREFILL_VERSION as INT8_LAYER_MAJOR_PREFILL_VERSION}}},
    tasks::{ir::TaskBudget, mapreduce::{CHUNK_PROFILE, ChunkLimits},
        sentiment::{SentimentAxis, sampled::{SampledSentimentLimits, SampledSentimentOptions,
            SampledSentimentVocabulary, MAX_SENTIMENT_SAMPLES, SAMPLED_SENTIMENT_PROMPT_VERSION,
            SAMPLED_SENTIMENT_AGGREGATION_VERSION, long::SAMPLED_SENTIMENT_MAP_AGGREGATION,
            quantized::{Int8SampledSentimentPlanner, INT8_SAMPLED_SENTIMENT_EXECUTION,
                INT8_SAMPLED_SENTIMENT_SPARSE_EXECUTION,
                long::{Int8SampledSentimentMapLimits, Int8SampledSentimentMapRun,
                    INT8_SAMPLED_SENTIMENT_MAP_EXECUTION, INT8_SAMPLED_SENTIMENT_MAP_SPARSE_EXECUTION}}}}},
    tokenizer::untrusted::UntrustedEncoding,
};

/// Only serialize for the owner's keyed recipe commitment. No Debug,
/// Deserialize, public fields or caller-supplied recipe constructor.
#[derive(Serialize)]
pub struct Int8SampledSentimentDocumentJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    input_policy: &'static str,
    execution: &'static str,
    child_execution: &'static str,
    native_execution: &'static str,
    chunk_profile: &'static str,
    prompt_version: &'static str,
    aggregation_version: &'static str,
    child_aggregation_version: &'static str,
    grammar_version: &'static str,
    sampler_version: &'static str,
    document_address_version: &'static str,
    chunk_address_version: &'static str,
    sample_address_stride: usize,
    task_budget: TaskBudget,
    default_axes: Vec<SentimentAxis>,
    sampling: SampledSentimentOptions,
    planning: PlanningRecipe,
    mapping: MappingRecipe,
    native: NativeRecipe,
    input_encoding: UntrustedEncoding,
    job: JobLimits,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill: Option<PrefillRecipe>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_selected_rows: Option<usize>,
}
#[derive(Serialize)]
struct MappingRecipe {
    chunks: ChunkLimits,
    max_model_work: Int8Work,
    max_retained_result_bytes: u64,
    max_result_bytes: u64,
}
impl From<Int8SampledSentimentMapLimits> for MappingRecipe {
    fn from(limits: Int8SampledSentimentMapLimits) -> Self {
        let Int8SampledSentimentMapLimits { chunks, max_model_work, max_retained_result_bytes, max_result_bytes } = limits;
        Self { chunks, max_model_work, max_retained_result_bytes, max_result_bytes }
    }
}
impl MappingRecipe {
    fn limits(&self) -> Int8SampledSentimentMapLimits {
        Int8SampledSentimentMapLimits { chunks: self.chunks, max_model_work: self.max_model_work,
            max_retained_result_bytes: self.max_retained_result_bytes, max_result_bytes: self.max_result_bytes }
    }
}
#[derive(Serialize)]
struct NativeRecipe {
    max_item_work: Int8Work,
    max_model_work: Int8Work,
    max_mask_trie_node_visits: usize,
    mask_checkpoint_interval_nodes: usize,
    max_mask_visits_per_item: u64,
    max_mask_visits_per_run: u64,
    max_sampler_bytes: u64,
}
impl From<Int8SampledSentimentBatchLimits> for NativeRecipe {
    fn from(limits: Int8SampledSentimentBatchLimits) -> Self {
        let Int8SampledSentimentBatchLimits { max_item_work, max_model_work, masks, max_sampler_bytes } = limits;
        let ExtractionMaskBudget { per_mask, max_visits_per_item, max_visits_per_run } = masks;
        let MaskWorkLimits { max_trie_node_visits, checkpoint_interval_nodes } = per_mask;
        Self { max_item_work, max_model_work, max_mask_trie_node_visits: max_trie_node_visits,
            mask_checkpoint_interval_nodes: checkpoint_interval_nodes,
            max_mask_visits_per_item: max_visits_per_item, max_mask_visits_per_run: max_visits_per_run,
            max_sampler_bytes }
    }
}
#[derive(Serialize)]
struct PrefillRecipe {
    execution: &'static str,
    max_batch_rows: usize,
    max_extra_scratch_bytes: u64,
}
impl Int8SampledSentimentDocumentJobRecipe {
    #[allow(clippy::too_many_arguments)]
    fn new(task_budget: TaskBudget, sampling: SampledSentimentOptions, planning: SampledSentimentLimits,
        mapping: Int8SampledSentimentMapLimits, native: Int8SampledSentimentBatchLimits,
        input_encoding: UntrustedEncoding, job: JobLimits) -> Result<Self, BatchFault> {
        task_budget.validate().map_err(|_| BatchCode::InvalidLimits)?;
        sampling.validate().map_err(|_| BatchCode::InvalidLimits)?;
        planning.validate().map_err(|_| BatchCode::InvalidLimits)?;
        native.validate()?;
        input_encoding.validate().map_err(|_| BatchCode::InvalidLimits)?;
        let recipe = Self { version: 1, dependency_scope: "item-local",
            input_policy: "exact-complete-source-chunk-axis-sample-addresses-v1",
            execution: INT8_SAMPLED_SENTIMENT_MAP_EXECUTION, child_execution: INT8_SAMPLED_SENTIMENT_EXECUTION,
            native_execution: INT8_JSON_SAMPLED_EXECUTION, chunk_profile: CHUNK_PROFILE,
            prompt_version: SAMPLED_SENTIMENT_PROMPT_VERSION, aggregation_version: SAMPLED_SENTIMENT_MAP_AGGREGATION,
            child_aggregation_version: SAMPLED_SENTIMENT_AGGREGATION_VERSION, grammar_version: JSON_RUNTIME_VERSION,
            sampler_version: JSON_SAMPLER_VERSION,
            document_address_version: "int8-sampled-sentiment-complete-document-address-v1",
            chunk_address_version: "int8-sampled-sentiment-original-chunk-address-v1",
            sample_address_stride: MAX_SENTIMENT_SAMPLES, task_budget,
            default_axes: SampledSentimentBatchArgs::default().axes, sampling, planning: planning.into(),
            mapping: mapping.into(), native: native.into(), input_encoding, job, prefill: None,
            max_selected_rows: None };
        recipe.check_job_limits(job)?;
        recipe.check_bytes()?;
        Ok(recipe)
    }
    fn check_bytes(&self) -> Result<(), BatchFault> {
        bounded_json(self, SAMPLED_SENTIMENT_JOB_RECIPE_BYTES)
            .map(|_| ()).map_err(|_| BatchCode::InvalidLimits.into())
    }
    fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
        self.mapping.limits().validate(self.task_budget).map_err(|_| BatchCode::InvalidLimits)?;
        let (execution, child, native) = match self.max_selected_rows {
            Some(max_rows_per_step) => {
                Int8JsonSparseLimits { max_rows_per_step }.validate().map_err(|_| BatchCode::InvalidLimits)?;
                (INT8_SAMPLED_SENTIMENT_MAP_SPARSE_EXECUTION,
                    INT8_SAMPLED_SENTIMENT_SPARSE_EXECUTION, INT8_JSON_SAMPLED_SPARSE_EXECUTION)
            }
            None => (INT8_SAMPLED_SENTIMENT_MAP_EXECUTION,
                INT8_SAMPLED_SENTIMENT_EXECUTION, INT8_JSON_SAMPLED_EXECUTION),
        };
        if limits != self.job || limits.max_id_bytes > 256
            || self.planning.max_input_bytes > limits.max_input_bytes_per_item
            || self.mapping.chunks.max_input_bytes > limits.max_input_bytes_per_item
            || self.mapping.max_model_work != self.native.max_item_work
            || self.mapping.max_result_bytes > limits.max_result_bytes as u64
            || limits.max_work != (JobWork { model: self.native.max_model_work,
                mask_node_visits: self.native.max_mask_visits_per_run })
            || self.execution != execution || self.child_execution != child || self.native_execution != native {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
    fn with_projection(mut self, selected: Option<Int8JsonSparseLimits>) -> Result<Self, BatchFault> {
        if let Some(limits) = selected {
            self.max_selected_rows = Some(limits.validate().map_err(|_| BatchCode::InvalidLimits)?);
            self.execution = INT8_SAMPLED_SENTIMENT_MAP_SPARSE_EXECUTION;
            self.child_execution = INT8_SAMPLED_SENTIMENT_SPARSE_EXECUTION;
            self.native_execution = INT8_JSON_SAMPLED_SPARSE_EXECUTION;
        }
        self.check_job_limits(self.job)?;
        self.check_bytes()?;
        Ok(self)
    }
    fn with_prefill(mut self, limits: Int8PrefillLimits) -> Result<Self, BatchFault> {
        limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
        let Int8PrefillLimits { max_batch_rows, max_extra_scratch_bytes } = limits;
        self.prefill = Some(PrefillRecipe { execution: INT8_LAYER_MAJOR_PREFILL_VERSION,
            max_batch_rows, max_extra_scratch_bytes });
        self.check_bytes()?;
        Ok(self)
    }
    fn prefill_limits(&self) -> Option<Int8PrefillLimits> {
        self.prefill.as_ref().map(|prefill| Int8PrefillLimits { max_batch_rows: prefill.max_batch_rows,
            max_extra_scratch_bytes: prefill.max_extra_scratch_bytes })
    }
}

/// Owns the actual compiler and its effective seed, never a reconstructed
/// caller recipe. Physical schedules are explicitly frozen before storage IO.
pub struct Int8SampledSentimentDocumentJobPlanner {
    compiler: Int8SampledSentimentDocumentPlanner,
    identity: ExecutionIdentity,
    recipe: Int8SampledSentimentDocumentJobRecipe,
    preparation_bytes: u64,
}
impl Int8SampledSentimentDocumentJobPlanner {
    pub fn new(planner: Int8SampledSentimentPlanner, vocabulary: Arc<SampledSentimentVocabulary>,
        mapping: Int8SampledSentimentMapLimits, native: Int8SampledSentimentBatchLimits,
        limits: JobLimits) -> Result<Self, BatchFault> {
        let task = planner.task_budget();
        let compiler = Int8SampledSentimentDocumentPlanner::new(planner, vocabulary, task, mapping, native)?;
        let recipe = Int8SampledSentimentDocumentJobRecipe::new(task, compiler.sampling_options().clone(),
            compiler.planning_limits(), mapping, native, compiler.document_encoding().clone(), limits)?
            .with_projection(compiler.selected_rows())?;
        let preparation_bytes = compiler.preparation_requirement_bytes()
            .checked_add(SAMPLED_SENTIMENT_JOB_RECIPE_BYTES as u64).ok_or(BatchCode::InvalidLimits)?;
        let identity = compiler.base_execution_identity().clone();
        Ok(Self { compiler, identity, recipe, preparation_bytes })
    }
    pub fn with_layer_major_prefill(mut self, limits: Int8PrefillLimits) -> Result<Self, BatchFault> {
        self.recipe = self.recipe.with_prefill(limits)?;
        Ok(self)
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_budget(&self) -> TaskBudget { self.compiler.task_budget() }
    pub fn native_limits(&self) -> Int8SampledSentimentBatchLimits { self.compiler.limits() }
    pub fn mapping_limits(&self) -> Int8SampledSentimentMapLimits { self.compiler.mapping() }
    pub fn planning_limits(&self) -> SampledSentimentLimits { self.compiler.planning_limits() }
    pub fn sampling_options(&self) -> &SampledSentimentOptions { &self.recipe.sampling }
    pub fn selected_rows(&self) -> Option<Int8JsonSparseLimits> { self.compiler.selected_rows() }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.compiler.document_encoding() }
    pub fn preparation_requirement_bytes(&self) -> u64 { self.preparation_bytes }
    pub fn max_result_bytes(&self) -> u64 { self.recipe.mapping.max_result_bytes }
    pub fn prefill_limits(&self) -> Option<Int8PrefillLimits> { self.recipe.prefill_limits() }
    pub fn job_limits(&self) -> JobLimits { self.recipe.job }
    pub(crate) fn retained_owners(&self)
        -> (Arc<Int8SampledSentimentPlanner>, Arc<SampledSentimentVocabulary>) {
        self.compiler.retained_owners()
    }
    /// Private configuration for the owner-only keyed commitment; never log it.
    pub fn job_recipe(&self) -> &Int8SampledSentimentDocumentJobRecipe { &self.recipe }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.recipe.check_job_limits(limits) }
}

pub struct Int8SampledSentimentDocumentJobProcessor<'e, 'w, A: Int8SampledSentimentDocumentBatchAdmission> {
    native: NativeInt8SampledSentimentDocumentBatch<'e, 'w, A>,
    identity: ExecutionIdentity,
    recipe: Int8SampledSentimentDocumentJobRecipe,
}
impl<'e, 'w, A: Int8SampledSentimentDocumentBatchAdmission> Int8SampledSentimentDocumentJobProcessor<'e, 'w, A> {
    pub fn new(planner: Int8SampledSentimentDocumentJobPlanner, engine: &'e mut StrictInt8Engine<'w>, admission: A)
        -> Result<Self, BatchFault> {
        let Int8SampledSentimentDocumentJobPlanner { compiler, identity, recipe, .. } = planner;
        recipe.check_job_limits(recipe.job)?;
        let native = match recipe.prefill_limits() {
            Some(prefill) => NativeInt8SampledSentimentDocumentBatch::new_layer_major(compiler, engine, admission, prefill)?,
            None => NativeInt8SampledSentimentDocumentBatch::new(compiler, engine, admission)?,
        };
        Ok(Self { native, identity, recipe })
    }
}
impl<A: Int8SampledSentimentDocumentBatchAdmission> BatchProcessor for Int8SampledSentimentDocumentJobProcessor<'_, '_, A> {
    type Args = SampledSentimentBatchArgs;
    type Prepared = PreparedInt8SampledSentimentDocumentRecord;
    type Output = GuardedOutput<Int8SampledSentimentMapRun, A::Guard>;
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
impl<A: Int8SampledSentimentDocumentBatchAdmission> DurableBatchProcessor
    for Int8SampledSentimentDocumentJobProcessor<'_, '_, A> {
    type Recipe = Int8SampledSentimentDocumentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.recipe.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        if self.native.is_poisoned() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        if prepared.execution_version() != self.recipe.execution {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        Ok(JobWork { model: prepared.model_work(), mask_node_visits: prepared.mask_node_visits() })
    }
    fn max_result_bytes(&self, prepared: &Self::Prepared) -> u64 { prepared.max_result_bytes() }
}

#[cfg(test)] mod tests;
