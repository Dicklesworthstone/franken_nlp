//! Authenticated complete-source faithfulness uses the same sealed native
//! corpus processor. Every source and its exact claim are one durable item;
//! neither individual evidence heads nor source chunks can become results.
use super::*;
use serde::Serialize;
use crate::{
    batch::judge::grouped_int4::{Int4FaithfulnessDocumentBatchAdmission,
        Int4FaithfulnessDocumentBatchPlanner, Int4FaithfulnessDocumentRun,
        NativeInt4FaithfulnessDocumentBatch, PreparedInt4FaithfulnessDocument},
    jobs::manifest::bounded_json,
    native_engine::{int4::PackedInt4Kernel, strict_int4::{Int4ModelWeights, Int4Work}},
    tasks::{mapreduce::{CHUNK_PROFILE, ChunkLimits}, judge::{FaithfulnessPolicy,
        FaithfulnessAggregationPolicy, FAITHFULNESS_AGGREGATION_PROFILE, JudgeLimits,
        grouped_int4::{Int4JudgePlanner, faithfulness_map::{Int4FaithfulnessMapLimits,
            INT4_FAITHFULNESS_MAP_EXECUTION, INT4_FAITHFULNESS_MAP_COHORT_EXECUTION,
            memory::Int4FaithfulnessMapMemoryLimits}}}},
};
pub use crate::batch::judge::grouped_int4::Int4FaithfulnessDocumentArgs;

const RECIPE_BYTES: usize = 1024 * 1024 - 1024;

/// Only the actual immutable model-bound compiler supplies this projection.
/// The journal stores its keyed commitment, never private raw settings.
#[derive(Serialize)]
pub struct Int4FaithfulnessDocumentJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    input_policy: &'static str,
    execution: &'static str,
    chunk_profile: &'static str,
    aggregation_profile: &'static str,
    reservation_policy: &'static str,
    judge: Int4ScoredJobRecipe,
    policy: FaithfulnessPolicy,
    chunks: ChunkLimits,
    per_document_model_work: Int4Work,
    max_retained_result_bytes: u64,
    max_result_bytes: u64,
    max_stored_result_bytes: u64,
    aggregation: FaithfulnessAggregationPolicy,
    cohort_chunks: Option<usize>,
    /// Absence preserves the original recipe bytes and 32-head declaration.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_heads: Option<usize>,
}
impl Int4FaithfulnessDocumentJobRecipe {
    fn new(judge: Int4ScoredJobRecipe, policy: FaithfulnessPolicy, mapping: Int4FaithfulnessMapLimits)
        -> Result<Self, BatchFault> {
        policy.validate().map_err(|_| BatchCode::InvalidLimits)?;
        // A new map field must deliberately join authenticated resume.
        let Int4FaithfulnessMapLimits { chunks, max_model_work, max_retained_result_bytes,
            max_result_bytes, aggregation } = mapping;
        let recipe = Self { version: 1, dependency_scope: "item-local-complete-source-claim",
            input_policy: "fixed-faithfulness-policy-required-claim-item-args-v1",
            execution: INT4_FAITHFULNESS_MAP_EXECUTION, chunk_profile: CHUNK_PROFILE,
            aggregation_profile: FAITHFULNESS_AGGREGATION_PROFILE,
            reservation_policy: "exact-all-chunk-five-axis-work-before-durable-attempt-v1",
            judge, policy, chunks, per_document_model_work: max_model_work,
            max_retained_result_bytes, max_result_bytes, max_stored_result_bytes: max_result_bytes,
            aggregation, cohort_chunks: None, max_heads: None };
        bounded_json(&recipe, RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(recipe)
    }
}

pub struct Int4FaithfulnessDocumentJobPlanner {
    compiler: Int4FaithfulnessDocumentBatchPlanner,
    identity: ExecutionIdentity,
    recipe: Int4FaithfulnessDocumentJobRecipe,
    limits: JobLimits,
}
impl Int4FaithfulnessDocumentJobPlanner {
    pub fn new(planner: Int4JudgePlanner, policy: FaithfulnessPolicy,
        mapping: Int4FaithfulnessMapLimits, limits: JobLimits) -> Result<Self, BatchFault> {
        check_limits(mapping, limits)?;
        let identity = planner.execution_identity().clone();
        let (task, planning) = planner.durable_configuration();
        let native = Int4ScoredBatchLimits { max_item_work: mapping.max_model_work,
            max_model_work: limits.max_work.int4_model_work() };
        let judge = Int4ScoredJobRecipe::judge(planner.group_width(), task, planning, None, native)?;
        let recipe = Int4FaithfulnessDocumentJobRecipe::new(judge, policy, mapping)?;
        let compiler = Int4FaithfulnessDocumentBatchPlanner::new(planner, policy, mapping, native.max_model_work)?;
        Ok(Self { compiler, identity, recipe, limits })
    }
    /// Narrow the physical declaration without changing the fixed policy or
    /// task ceiling. An explicit cap becomes immutable resume authority;
    /// existing jobs created without this opt-in keep their original recipe.
    pub fn with_max_heads(mut self, max_heads: usize) -> Result<Self, BatchFault> {
        self.compiler = self.compiler.with_max_heads(max_heads)?;
        self.recipe.max_heads = Some(max_heads);
        bounded_json(&self.recipe, RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(self)
    }
    pub fn with_chunk_cohorts(mut self, chunks: usize) -> Result<Self, BatchFault> {
        self.compiler = self.compiler.with_chunk_cohorts(chunks)?;
        if self.compiler.max_result_bytes() > self.limits.max_result_bytes as u64 {
            return Err(BatchCode::InvalidLimits.into());
        }
        self.recipe.execution = INT4_FAITHFULNESS_MAP_COHORT_EXECUTION;
        self.recipe.cohort_chunks = Some(chunks);
        self.recipe.max_stored_result_bytes = self.compiler.max_result_bytes();
        bounded_json(&self.recipe, RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(self)
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_budget(&self) -> TaskBudget { self.compiler.task_budget() }
    pub fn planning_limits(&self) -> JudgeLimits { self.compiler.planning_limits() }
    pub fn document_limits(&self) -> Int4FaithfulnessMapLimits { self.compiler.mapping_limits() }
    pub fn job_limits(&self) -> JobLimits { self.limits }
    pub fn group_width(&self) -> usize { self.compiler.group_width() }
    pub fn policy(&self) -> FaithfulnessPolicy { self.compiler.policy() }
    pub fn max_heads(&self) -> usize { self.compiler.max_heads() }
    pub fn cohort_chunks(&self) -> Option<usize> { self.compiler.cohort_chunks() }
    pub fn max_result_bytes(&self) -> u64 { self.compiler.max_result_bytes() }
    pub fn job_recipe(&self) -> &Int4FaithfulnessDocumentJobRecipe { &self.recipe }
    pub fn retained_defaults_bytes(&self) -> Result<u64, BatchFault> { self.compiler.retained_defaults_bytes() }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        if limits != self.limits || self.max_result_bytes() > limits.max_result_bytes as u64 {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}

/// The result guard stays owned through complete serialization, spool sync
/// and journal acknowledgement in the existing runner. Every native failure
/// keeps its already persisted whole-source work debit.
pub struct Int4FaithfulnessDocumentJobProcessor<'e, 'w, A: Int4FaithfulnessDocumentBatchAdmission> {
    native: NativeInt4FaithfulnessDocumentBatch<'e, 'w, A>,
    identity: ExecutionIdentity,
    recipe: Int4FaithfulnessDocumentJobRecipe,
    limits: JobLimits,
}
impl<'e, 'w, A: Int4FaithfulnessDocumentBatchAdmission> Int4FaithfulnessDocumentJobProcessor<'e, 'w, A> {
    pub fn new(planner: Int4FaithfulnessDocumentJobPlanner, engine: &'e mut StrictInt4Engine<'w>, admission: A)
        -> Result<Self, BatchFault> {
        let Int4FaithfulnessDocumentJobPlanner { compiler, identity, recipe, limits } = planner;
        let native = NativeInt4FaithfulnessDocumentBatch::new(compiler, engine, admission)?;
        Ok(Self { native, identity, recipe, limits })
    }
    pub fn new_cohorts(planner: Int4FaithfulnessDocumentJobPlanner, weights: &'w Int4ModelWeights,
        kernel: PackedInt4Kernel, memory: Int4FaithfulnessMapMemoryLimits, admission: A)
        -> Result<Self, BatchFault> {
        let Int4FaithfulnessDocumentJobPlanner { compiler, identity, recipe, limits } = planner;
        let native = NativeInt4FaithfulnessDocumentBatch::new_cohorts(compiler, weights, kernel, memory, admission)?;
        Ok(Self { native, identity, recipe, limits })
    }
}
impl<A: Int4FaithfulnessDocumentBatchAdmission> BatchProcessor for Int4FaithfulnessDocumentJobProcessor<'_, '_, A> {
    type Args = Int4FaithfulnessDocumentArgs;
    type Prepared = PreparedInt4FaithfulnessDocument;
    type Output = GuardedOutput<Int4FaithfulnessDocumentRun, A::Guard>;
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
impl<A: Int4FaithfulnessDocumentBatchAdmission> DurableBatchProcessor for Int4FaithfulnessDocumentJobProcessor<'_, '_, A> {
    type Recipe = Int4FaithfulnessDocumentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        if limits != self.limits { return Err(BatchCode::InvalidLimits.into()); }
        Ok(())
    }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        if self.native.is_poisoned() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        Ok(JobWork::from_int4(prepared.planned_work(), 0))
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.recipe.max_stored_result_bytes }
}
fn check_limits(mapping: Int4FaithfulnessMapLimits, limits: JobLimits) -> Result<(), BatchFault> {
    limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
    if limits.max_work.mask_node_visits != 0
        || !JobWork::from_int4(mapping.max_model_work, 0).fits(limits.max_work)
        || mapping.chunks.max_input_bytes > limits.max_input_bytes_per_item
        || mapping.max_result_bytes > limits.max_result_bytes as u64 {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}

#[cfg(test)] mod tests;
