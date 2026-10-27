//! Authenticated durable scoring on the same resident INT4 stream processor.
//!
//! The actual model-bound compiler supplies every private recipe field. The
//! existing owned-job runner authenticates the whole population and persists
//! all five work debits before execution, including attempts that later fail.
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, output::GuardedOutput,
        scored_int4::{Int4ScoredBatchAdmission, Int4ScoredBatchLimits, Int4ScoredBatchPlanner,
            Int4ScoredTask, NativeInt4ScoredBatch, PreparedInt4Scored}},
    execution_identity::ExecutionIdentity,
    jobs::{JobLimits, JobWork, runner::DurableBatchProcessor},
    native_engine::{decode::DecodeStepControl, strict_int4::StrictInt4Engine},
    tasks::ir::TaskBudget,
};

mod recipe;
pub use recipe::Int4ScoredJobRecipe;

/// Closed to the existing model-bound classification, sentiment and judge
/// compilers. Neither a caller-supplied recipe nor an alternate scorer can
/// acquire durable execution authority through this route.
pub trait Int4ScoredJobTask: Int4ScoredTask + recipe::Sealed {}

/// Immutable owned factory. Private defaults are bounded before cloning them
/// into the native stream compiler. Only their keyed recipe commitment enters
/// the metadata journal; this type exposes no raw compiler or mutable recipe.
pub struct Int4ScoredJobPlanner<P: Int4ScoredJobTask> {
    compiler: Int4ScoredBatchPlanner<P>,
    identity: ExecutionIdentity,
    recipe: Int4ScoredJobRecipe,
}
impl<P: Int4ScoredJobTask> Int4ScoredJobPlanner<P> {
    pub fn new(planner: P, defaults: Option<P::Args>, native: Int4ScoredBatchLimits)
        -> Result<Self, BatchFault> {
        native.validate()?;
        let identity = planner.base_identity().clone();
        let (recipe, defaults) = planner.freeze_recipe(defaults, native)?;
        let compiler = Int4ScoredBatchPlanner::new(planner, defaults, native)?;
        Ok(Self { compiler, identity, recipe })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_ceiling(&self) -> TaskBudget { self.recipe.task_ceiling() }
    pub fn native_limits(&self) -> Int4ScoredBatchLimits { self.compiler.limits() }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        self.recipe.check_job_limits(limits)
    }
}

/// The genuine native processor remains the sole scoring path. Its result
/// guard survives canonical encoding, spool sync and journal acknowledgement.
pub struct Int4ScoredJobProcessor<'e, 'w, P: Int4ScoredJobTask, A: Int4ScoredBatchAdmission> {
    native: NativeInt4ScoredBatch<'e, 'w, P, A>,
    identity: ExecutionIdentity,
    recipe: Int4ScoredJobRecipe,
}
impl<'e, 'w, P: Int4ScoredJobTask, A: Int4ScoredBatchAdmission> Int4ScoredJobProcessor<'e, 'w, P, A> {
    pub fn new(planner: Int4ScoredJobPlanner<P>, engine: &'e mut StrictInt4Engine<'w>, admission: A)
        -> Result<Self, BatchFault> {
        let Int4ScoredJobPlanner { compiler, identity, recipe } = planner;
        let native = NativeInt4ScoredBatch::new(compiler, engine, admission)?;
        Ok(Self { native, identity, recipe })
    }
}
impl<P: Int4ScoredJobTask, A: Int4ScoredBatchAdmission> BatchProcessor for Int4ScoredJobProcessor<'_, '_, P, A> {
    type Args = P::Args;
    type Prepared = PreparedInt4Scored<P>;
    type Output = GuardedOutput<P::Output, A::Guard>;
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
impl<P: Int4ScoredJobTask, A: Int4ScoredBatchAdmission> DurableBatchProcessor for Int4ScoredJobProcessor<'_, '_, P, A> {
    type Recipe = Int4ScoredJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.recipe.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        if self.native.is_poisoned() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        // Copy the actual group-dependent counters. Scoring has no grammar
        // masks; restarting the native ledger cannot refund persisted debits.
        Ok(JobWork::from_int4(prepared.model_work(), 0))
    }
    fn max_result_bytes(&self, prepared: &Self::Prepared) -> u64 { prepared.max_result_bytes() }
}

#[cfg(test)]
mod tests;
