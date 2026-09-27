//! Complete independent-axis sentiment bundles, never partial-axis commits.
use super::SentimentJobRecipe;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor, BatchRequestContext,
        BatchWork, generation::GuardedOutput},
    execution_identity::ExecutionIdentity,
    jobs::{JobWork, runner::DurableBatchProcessor},
    native_engine::{decode::DecodeStepControl, strict_int8::StrictInt8Engine},
    tasks::{ir::TaskBudget, sentiment::{SentimentPlanner,
        batch::{SentimentBatchArgs, SentimentBatchConfig, Int8SentimentAdmission,
            Int8SentimentBatchPlanner, NativeInt8SentimentBatch, PreparedInt8BatchSentiment},
        quantized::Int8SentimentRun}},
};

pub struct Int8SentimentJobPlanner<'p> {
    compiler: Int8SentimentBatchPlanner<'p>,
    identity: ExecutionIdentity,
    recipe: SentimentJobRecipe,
}
impl<'p> Int8SentimentJobPlanner<'p> {
    pub fn new(planner: &'p SentimentPlanner, config: SentimentBatchConfig) -> Result<Self, BatchFault> {
        config.validate(planner)?;
        let recipe = SentimentJobRecipe::new(&config)?;
        let identity = config.identity.clone();
        let compiler = Int8SentimentBatchPlanner::new(planner, config)?;
        Ok(Self { compiler, identity, recipe })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_ceiling(&self) -> TaskBudget { self.recipe.task_ceiling }
    pub fn job_recipe(&self) -> &SentimentJobRecipe { &self.recipe }
    pub fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<SentimentBatchArgs>,
        control: &mut C) -> Result<PreparedInt8BatchSentiment, BatchItemFailure> {
        self.compiler.prepare_with_control(document, control)
    }
}

pub struct Int8SentimentJobProcessor<'p, 'e, 'w, A: Int8SentimentAdmission> {
    native: NativeInt8SentimentBatch<'p, 'e, 'w, A>,
    identity: ExecutionIdentity,
    recipe: SentimentJobRecipe,
}
impl<'p, 'e, 'w, A: Int8SentimentAdmission> Int8SentimentJobProcessor<'p, 'e, 'w, A> {
    pub fn new(planner: Int8SentimentJobPlanner<'p>, engine: &'e mut StrictInt8Engine<'w>, admission: A)
        -> Result<Self, BatchFault> {
        let Int8SentimentJobPlanner { compiler, identity, recipe } = planner;
        let native = NativeInt8SentimentBatch::new(compiler, engine, admission)?;
        Ok(Self { native, identity, recipe })
    }
}
impl<A: Int8SentimentAdmission> BatchProcessor for Int8SentimentJobProcessor<'_, '_, '_, A> {
    type Args = SentimentBatchArgs;
    type Prepared = PreparedInt8BatchSentiment;
    type Output = GuardedOutput<Int8SentimentRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        // The existing adapter refuses uncontrolled preparation. JobRunner
        // always calls prepare_with_control with its invocation controller.
        self.native.prepare(document)
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> { self.native.prepare_with_control(document, control) }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { self.native.planned_work(prepared) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> { self.native.execute(prepared, control) }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        // Preserve stable durable ordinal/epoch. The native adapter refuses
        // duplicate/nonmonotonic requests; no new sequence starts per axis.
        self.native.execute_with_context(prepared, context, control)
    }
}
impl<A: Int8SentimentAdmission> DurableBatchProcessor for Int8SentimentJobProcessor<'_, '_, '_, A> {
    type Recipe = SentimentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        if self.native.is_poisoned() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        Ok(JobWork { model: prepared.model_work(), mask_node_visits: 0 })
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.recipe.task_ceiling.max_output_bytes }
}
