//! Exact exclusive/multi-label scoring behind the existing durable runner.
use super::ClassificationJobRecipe;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor, BatchRequestContext, BatchWork,
        classify::{ClassificationBatchArgs, GuardedOutput, quantized::{Int8ClassificationAdmission,
            Int8ClassificationBatchPlanner, NativeInt8ClassificationBatch, PreparedInt8BatchClassification}}},
    execution_identity::ExecutionIdentity,
    jobs::{JobWork, runner::DurableBatchProcessor},
    native_engine::{decode::DecodeStepControl, strict_int8::{Int8Work, StrictInt8Engine}},
    tasks::{classify::{ClassificationLimits, ClassificationPlanner, quantized::Int8ClassificationRun}, ir::TaskBudget},
};

/// Model-free immutable factory. The borrowed pinned planner must be retained
/// and charged by the host; it never becomes a self-referential owned object.
pub struct Int8ClassificationJobPlanner<'p> {
    compiler: Int8ClassificationBatchPlanner<'p>,
    identity: ExecutionIdentity,
    recipe: ClassificationJobRecipe,
}
impl<'p> Int8ClassificationJobPlanner<'p> {
    pub fn new(planner: &'p ClassificationPlanner, identity: ExecutionIdentity, ceiling: TaskBudget,
        planning: ClassificationLimits, defaults: Option<ClassificationBatchArgs>, max_model_work: Int8Work)
        -> Result<Self, BatchFault> {
        let recipe = ClassificationJobRecipe::new(ceiling, planning, defaults, max_model_work)?;
        let compiler = Int8ClassificationBatchPlanner::new(planner, identity.clone(), ceiling,
            planning, recipe.defaults.clone())?;
        Ok(Self { compiler, identity, recipe })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_ceiling(&self) -> TaskBudget { self.recipe.task_ceiling }
    pub fn job_recipe(&self) -> &ClassificationJobRecipe { &self.recipe }
    pub fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<ClassificationBatchArgs>,
        control: &mut C) -> Result<PreparedInt8BatchClassification, BatchItemFailure> {
        self.compiler.prepare_with_control(document, control)
    }
}

/// Real native execution only. A guarded result stays owned until its durable
/// spool frame is synced and its journal acknowledgement is committed.
pub struct Int8ClassificationJobProcessor<'p, 'e, 'w, A: Int8ClassificationAdmission> {
    native: NativeInt8ClassificationBatch<'p, 'e, 'w, A>,
    identity: ExecutionIdentity,
    recipe: ClassificationJobRecipe,
}
impl<'p, 'e, 'w, A: Int8ClassificationAdmission> Int8ClassificationJobProcessor<'p, 'e, 'w, A> {
    pub fn new(planner: Int8ClassificationJobPlanner<'p>, engine: &'e mut StrictInt8Engine<'w>, admission: A)
        -> Result<Self, BatchFault> {
        let Int8ClassificationJobPlanner { compiler, identity, recipe } = planner;
        let native = NativeInt8ClassificationBatch::new(compiler, engine, admission, recipe.max_model_work)?;
        Ok(Self { native, identity, recipe })
    }
}
impl<A: Int8ClassificationAdmission> BatchProcessor for Int8ClassificationJobProcessor<'_, '_, '_, A> {
    type Args = ClassificationBatchArgs;
    type Prepared = PreparedInt8BatchClassification;
    type Output = GuardedOutput<Int8ClassificationRun, A::Guard>;
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
impl<A: Int8ClassificationAdmission> DurableBatchProcessor for Int8ClassificationJobProcessor<'_, '_, '_, A> {
    type Recipe = ClassificationJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        if !self.native.ready() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        // Finite scoring has no grammar-mask work. All five native axes are
        // nevertheless reserved durably before ANY admission/scoring callback.
        Ok(JobWork { model: prepared.model_work(), mask_node_visits: 0 })
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 {
        // Conservative factory ceiling. The native plan also enforces each
        // possibly smaller request budget; the host requires this whole ceiling
        // to fit the immutable job envelope before starting the population.
        self.recipe.task_ceiling.max_output_bytes
    }
}
