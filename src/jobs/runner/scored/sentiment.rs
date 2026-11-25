//! Complete independent-axis sentiment bundles, never partial-axis commits.
use super::{EncodingWorkScope, SentimentJobRecipe};
use std::sync::Arc;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor, BatchRequestContext,
        BatchWork, generation::GuardedOutput},
    execution_identity::ExecutionIdentity,
    jobs::{JobWork, manifest::bounded_json, runner::DurableBatchProcessor},
    native_engine::{decode::DecodeStepControl, strict_int8::{Int8Work, StrictInt8Engine, prefill::Int8PrefillLimits}},
    tasks::{ir::TaskBudget, sentiment::{SentimentPlanner,
        batch::{SentimentBatchArgs, SentimentBatchConfig, Int8SentimentAdmission,
            Int8SentimentBatchPlanner, NativeInt8SentimentBatch, PreparedInt8BatchSentiment},
        quantized::Int8SentimentRun}},
    tokenizer::untrusted::{SafeBpeWork, SafeBpeWorkLimits, UntrustedEncoding},
};

pub struct Int8SentimentJobPlanner<'p> {
    compiler: Int8SentimentBatchPlanner<'p>,
    identity: ExecutionIdentity,
    recipe: SentimentJobRecipe,
    encoding: Arc<EncodingWorkScope>,
}
/// The same private invocation owner must survive factory preparation,
/// reconstruction and the consuming native processor handoff.
pub struct PreparedInt8SentimentJob {
    native: PreparedInt8BatchSentiment,
    encoding: Arc<EncodingWorkScope>,
}
impl PreparedInt8SentimentJob {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.native.execution_identity() }
    pub fn model_work(&self) -> Int8Work { self.native.model_work() }
    pub fn head_count(&self) -> usize { self.native.head_count() }
}
impl<'p> Int8SentimentJobPlanner<'p> {
    pub fn new(planner: &'p SentimentPlanner, config: SentimentBatchConfig) -> Result<Self, BatchFault> {
        Self::new_with_document_encoding(planner, config, None)
    }
    pub fn new_with_document_encoding(planner: &'p SentimentPlanner, config: SentimentBatchConfig,
        total: Option<SafeBpeWorkLimits>) -> Result<Self, BatchFault> {
        let encoding = EncodingWorkScope::new(planner.document_encoding(), total)?;
        Self::with_encoding_scope(planner, config, encoding)
    }
    pub(crate) fn with_encoding_scope(planner: &'p SentimentPlanner, config: SentimentBatchConfig,
        encoding: Arc<EncodingWorkScope>) -> Result<Self, BatchFault> {
        if planner.document_encoding() != &encoding.policy { return Err(BatchCode::Admission.into()); }
        encoding.workspace_bytes()?;
        config.validate_with_document_encoding(planner)?;
        let mut recipe = SentimentJobRecipe::new(&config)?;
        recipe.input_encoding = (encoding.policy != UntrustedEncoding::ByteFallback).then(|| encoding.policy.clone());
        recipe.encoding_work_limits = encoding.limits;
        bounded_json(&recipe, super::RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        let identity = config.identity.clone();
        let compiler = Int8SentimentBatchPlanner::new_with_document_encoding(planner, config)?;
        Ok(Self { compiler, identity, recipe, encoding })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_ceiling(&self) -> TaskBudget { self.recipe.task_ceiling }
    pub fn job_recipe(&self) -> &SentimentJobRecipe { &self.recipe }
    pub fn document_encoding(&self) -> &UntrustedEncoding { &self.encoding.policy }
    pub fn encoding_work_limits(&self) -> Option<SafeBpeWorkLimits> { self.encoding.limits }
    pub fn encoding_workspace_bytes(&self) -> Result<u64, BatchFault> { self.encoding.workspace_bytes() }
    pub fn encoding_work(&self) -> Result<SafeBpeWork, BatchFault> { self.encoding.work() }
    pub fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<SentimentBatchArgs>,
        control: &mut C) -> Result<PreparedInt8SentimentJob, BatchItemFailure> {
        let native = self.encoding.prepare(control, |ceiling, work, control|
            self.compiler.prepare_with_encoding_budget(document, control, work, ceiling))?;
        Ok(PreparedInt8SentimentJob { native, encoding: Arc::clone(&self.encoding) })
    }
}

pub struct Int8SentimentJobProcessor<'p, 'e, 'w, A: Int8SentimentAdmission> {
    native: NativeInt8SentimentBatch<'p, 'e, 'w, A>,
    identity: ExecutionIdentity,
    recipe: SentimentJobRecipe,
    encoding: Arc<EncodingWorkScope>,
}
impl<'p, 'e, 'w, A: Int8SentimentAdmission> Int8SentimentJobProcessor<'p, 'e, 'w, A> {
    pub fn new(planner: Int8SentimentJobPlanner<'p>, engine: &'e mut StrictInt8Engine<'w>, admission: A)
        -> Result<Self, BatchFault> {
        let Int8SentimentJobPlanner { compiler, identity, recipe, encoding } = planner;
        let native = NativeInt8SentimentBatch::new(compiler, engine, admission)?;
        Ok(Self { native, identity, recipe, encoding })
    }
    /// Preserve original semantic identity, cumulative tokenizer work and
    /// nonrefundable durable native work while scheduling prompt morsels.
    pub fn new_layer_major(planner: Int8SentimentJobPlanner<'p>, engine: &'e mut StrictInt8Engine<'w>,
        admission: A, prefill: Int8PrefillLimits) -> Result<Self, BatchFault> {
        let Int8SentimentJobPlanner { compiler, identity, recipe, encoding } = planner;
        let native = NativeInt8SentimentBatch::new_layer_major(compiler, engine, admission, prefill)?;
        Ok(Self { native, identity, recipe, encoding })
    }
}
impl<A: Int8SentimentAdmission> BatchProcessor for Int8SentimentJobProcessor<'_, '_, '_, A> {
    type Args = SentimentBatchArgs;
    type Prepared = PreparedInt8SentimentJob;
    type Output = GuardedOutput<Int8SentimentRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        // Keep the native adapter's fail-closed refusal and poison behavior.
        // JobRunner always lends its original caller control below.
        let native = self.native.prepare(document)?;
        Ok(PreparedInt8SentimentJob { native, encoding: Arc::clone(&self.encoding) })
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        let native = self.encoding.prepare(control, |ceiling, work, control|
            self.native.prepare_with_encoding_budget(document, control, work, ceiling))?;
        Ok(PreparedInt8SentimentJob { native, encoding: Arc::clone(&self.encoding) })
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { self.native.planned_work(&prepared.native) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        self.encoding.verify(&prepared.encoding)?;
        self.native.execute(prepared.native, control)
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.encoding.verify(&prepared.encoding)?;
        // Preserve stable durable ordinal/epoch and the complete axis bundle.
        self.native.execute_with_context(prepared.native, context, control)
    }
}
impl<A: Int8SentimentAdmission> DurableBatchProcessor for Int8SentimentJobProcessor<'_, '_, '_, A> {
    type Recipe = SentimentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        self.encoding.verify(&prepared.encoding)?;
        if self.native.is_poisoned() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        Ok(JobWork { model: prepared.model_work(), mask_node_visits: 0 })
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.recipe.task_ceiling.max_output_bytes }
}
