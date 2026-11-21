//! Authenticated complete-document sampled sentiment jobs.
//!
//! The existing journal reserves the complete cold chunk/axis/sample attempt
//! before native work. Only a sealed, validated whole-document result reaches
//! the spool. The private recipe binds both the per-chunk sampler and the
//! complete-source partition and reducer; a restart cannot renew either cap.

use super::*;
use crate::{
    batch::sentiment::sampled::long::{NativeSampledSentimentDocumentBatch,
        PreparedSampledSentimentDocumentRecord, SampledSentimentDocumentBatchAdmission,
        SampledSentimentDocumentPlanner},
    tasks::{mapreduce::{CHUNK_PROFILE, ChunkLimits}, sentiment::sampled::long::{
        SampledSentimentMapLimits, SampledSentimentMapRun,
        SAMPLED_SENTIMENT_MAP_AGGREGATION, SAMPLED_SENTIMENT_MAP_EXECUTION,
        SAMPLED_SENTIMENT_MAP_REUSED_PREFILL_EXECUTION}},
};

/// Only the job's owner-only keyed recipe may serialize this value. It has
/// no Debug/Deserialize implementation and is never a public provenance row.
#[derive(Serialize)]
pub struct SampledSentimentDocumentJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    input_policy: &'static str,
    execution: &'static str,
    chunk_profile: &'static str,
    aggregation_version: &'static str,
    item_address_version: &'static str,
    chunk_sampling: SampledSentimentJobRecipe,
    mapping: MappingRecipe,
}

#[derive(Serialize)]
struct MappingRecipe {
    chunks: ChunkLimits,
    max_model_work: Int4Work,
    max_retained_result_bytes: u64,
    max_result_bytes: u64,
}
impl From<SampledSentimentMapLimits> for MappingRecipe {
    fn from(limits: SampledSentimentMapLimits) -> Self {
        let SampledSentimentMapLimits { chunks, max_model_work,
            max_retained_result_bytes, max_result_bytes } = limits;
        Self { chunks, max_model_work, max_retained_result_bytes, max_result_bytes }
    }
}
impl MappingRecipe {
    fn limits(&self) -> SampledSentimentMapLimits {
        SampledSentimentMapLimits { chunks: self.chunks, max_model_work: self.max_model_work,
            max_retained_result_bytes: self.max_retained_result_bytes,
            max_result_bytes: self.max_result_bytes }
    }
}

impl SampledSentimentDocumentJobRecipe {
    fn new(chunk_sampling: SampledSentimentJobRecipe, mapping: SampledSentimentMapLimits)
        -> Result<Self, BatchFault> {
        let recipe = Self { version: 1, dependency_scope: "item-local",
            input_policy: "exact-complete-source-chunk-axis-sample-addresses-v1",
            execution: SAMPLED_SENTIMENT_MAP_EXECUTION, chunk_profile: CHUNK_PROFILE,
            aggregation_version: SAMPLED_SENTIMENT_MAP_AGGREGATION,
            item_address_version: crate::tasks::sentiment::sampled::long::SAMPLED_SENTIMENT_MAP_ITEM_ADDRESS_VERSION,
            chunk_sampling, mapping: mapping.into() };
        recipe.check_job_limits(recipe.chunk_sampling.job)?;
        recipe.check_bytes()?;
        Ok(recipe)
    }
    fn check_bytes(&self) -> Result<(), BatchFault> {
        bounded_json(self, SAMPLED_SENTIMENT_JOB_RECIPE_BYTES)
            .map(|_| ()).map_err(|_| BatchCode::InvalidLimits.into())
    }
    fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        self.chunk_sampling.check_job_limits(limits)?;
        self.mapping.limits().validate(self.chunk_sampling.task_budget)
            .map_err(|_| BatchCode::InvalidLimits)?;
        let expected = if self.chunk_sampling.reuses_prefill() {
            SAMPLED_SENTIMENT_MAP_REUSED_PREFILL_EXECUTION
        } else { SAMPLED_SENTIMENT_MAP_EXECUTION };
        if self.execution != expected
            || self.mapping.chunks.max_input_bytes > limits.max_input_bytes_per_item
            || self.mapping.max_model_work != self.chunk_sampling.native.max_item_work
            || self.mapping.max_result_bytes > limits.max_result_bytes as u64 {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}

/// A real model-bound whole-document compiler, with one immutable private
/// recipe. Sample addresses depend on item IDs and exact source/configuration,
/// never on retry counters, population order or a flush epoch.
pub struct SampledSentimentDocumentJobPlanner {
    compiler: SampledSentimentDocumentPlanner,
    identity: ExecutionIdentity,
    recipe: SampledSentimentDocumentJobRecipe,
    planning: SampledSentimentLimits,
    preparation_bytes: u64,
}
impl SampledSentimentDocumentJobPlanner {
    pub fn new(planner: SampledSentimentPlanner, vocabulary: Arc<SampledSentimentVocabulary>,
        mapping: SampledSentimentMapLimits, native: SampledSentimentBatchLimits,
        limits: JobLimits) -> Result<Self, BatchFault> {
        let task = planner.task_budget();
        let compiler = SampledSentimentDocumentPlanner::new(planner, vocabulary, task, mapping, native)?;
        let (options, planning) = compiler.durable_configuration();
        let chunk_sampling = SampledSentimentJobRecipe::new(compiler.group_width(), task,
            options.clone(), planning, native, compiler.document_encoding().clone(), limits)?;
        let recipe = SampledSentimentDocumentJobRecipe::new(chunk_sampling, mapping)?;
        let preparation_bytes = compiler.preparation_requirement_bytes()
            .checked_add(SAMPLED_SENTIMENT_JOB_RECIPE_BYTES as u64).ok_or(BatchCode::InvalidLimits)?;
        let identity = compiler.base_execution_identity().clone();
        Ok(Self { compiler, identity, recipe, planning, preparation_bytes })
    }
    /// Bind this explicit physical prompt schedule before opening job storage.
    pub fn with_layer_major_prefill(mut self, limits: Int4PrefillLimits) -> Result<Self, BatchFault> {
        self.recipe.chunk_sampling = self.recipe.chunk_sampling.with_prefill(limits)?;
        self.recipe.check_bytes()?;
        Ok(self)
    }
    /// Prefix reuse remains private to one chunk and axis. Durable attempt
    /// debits continue to price the full cold schedule, including failed work.
    pub fn with_reused_prefill(mut self) -> Result<Self, BatchFault> {
        self.recipe.chunk_sampling = self.recipe.chunk_sampling.with_reused_prefill()?;
        self.recipe.execution = SAMPLED_SENTIMENT_MAP_REUSED_PREFILL_EXECUTION;
        self.recipe.check_bytes()?;
        Ok(self)
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_budget(&self) -> TaskBudget { self.compiler.task_budget() }
    pub fn native_limits(&self) -> SampledSentimentBatchLimits { self.compiler.limits() }
    pub fn mapping_limits(&self) -> SampledSentimentMapLimits { self.compiler.mapping() }
    pub fn planning_limits(&self) -> SampledSentimentLimits { self.planning }
    pub fn sampling_options(&self) -> &SampledSentimentOptions { &self.recipe.chunk_sampling.sampling }
    pub fn group_width(&self) -> usize { self.compiler.group_width() }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.compiler.document_encoding() }
    pub fn preparation_requirement_bytes(&self) -> u64 { self.preparation_bytes }
    pub fn max_result_bytes(&self) -> u64 { self.recipe.mapping.max_result_bytes }
    pub fn prefill_limits(&self) -> Option<Int4PrefillLimits> { self.recipe.chunk_sampling.prefill_limits() }
    pub fn reuses_prefill(&self) -> bool { self.recipe.chunk_sampling.reuses_prefill() }
    pub fn job_limits(&self) -> JobLimits { self.recipe.chunk_sampling.job }
    /// Private configuration for keyed storage commitment; never publish it.
    pub fn job_recipe(&self) -> &SampledSentimentDocumentJobRecipe { &self.recipe }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        self.recipe.check_job_limits(limits)
    }
}

/// The same native complete-document adapter used for live populations. Its
/// output guard survives until the result frame and journal acknowledgement.
pub struct SampledSentimentDocumentJobProcessor<'e, 'w, A: SampledSentimentDocumentBatchAdmission> {
    native: NativeSampledSentimentDocumentBatch<'e, 'w, A>,
    identity: ExecutionIdentity,
    recipe: SampledSentimentDocumentJobRecipe,
}
impl<'e, 'w, A: SampledSentimentDocumentBatchAdmission> SampledSentimentDocumentJobProcessor<'e, 'w, A> {
    pub fn new(planner: SampledSentimentDocumentJobPlanner, engine: &'e mut StrictInt4Engine<'w>, admission: A)
        -> Result<Self, BatchFault> {
        let SampledSentimentDocumentJobPlanner { compiler, identity, recipe, .. } = planner;
        recipe.check_job_limits(recipe.chunk_sampling.job)?;
        let native = if recipe.chunk_sampling.reuses_prefill() {
            NativeSampledSentimentDocumentBatch::new_reusing_prefill(compiler, engine, admission,
                recipe.chunk_sampling.prefill_limits())?
        } else { match recipe.chunk_sampling.prefill_limits() {
            Some(prefill) => NativeSampledSentimentDocumentBatch::new_layer_major(compiler, engine, admission, prefill)?,
            None => NativeSampledSentimentDocumentBatch::new(compiler, engine, admission)?,
        } };
        Ok(Self { native, identity, recipe })
    }
}
impl<A: SampledSentimentDocumentBatchAdmission> BatchProcessor for SampledSentimentDocumentJobProcessor<'_, '_, A> {
    type Args = SampledSentimentBatchArgs;
    type Prepared = PreparedSampledSentimentDocumentRecord;
    type Output = GuardedOutput<SampledSentimentMapRun, A::Guard>;
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
impl<A: SampledSentimentDocumentBatchAdmission> DurableBatchProcessor for SampledSentimentDocumentJobProcessor<'_, '_, A> {
    type Recipe = SampledSentimentDocumentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.recipe.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        if self.native.is_poisoned() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        Ok(JobWork::from_int4(prepared.model_work(), prepared.mask_node_visits()))
    }
    fn max_result_bytes(&self, prepared: &Self::Prepared) -> u64 { prepared.max_result_bytes() }
}

#[cfg(test)] mod tests;
