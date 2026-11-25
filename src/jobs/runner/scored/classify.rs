//! Exact exclusive/multi-label scoring behind the existing durable runner.
use super::{ClassificationJobRecipe, EncodingWorkScope};
use std::sync::Arc;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor, BatchRequestContext, BatchWork,
        classify::{ClassificationBatchArgs, GuardedOutput, quantized::{Int8ClassificationAdmission,
            Int8ClassificationBatchPlanner, NativeInt8ClassificationBatch, PreparedInt8BatchClassification}}},
    execution_identity::ExecutionIdentity,
    jobs::{JobWork, manifest::bounded_json, runner::DurableBatchProcessor},
    native_engine::{decode::DecodeStepControl, strict_int8::{Int8Work, StrictInt8Engine, prefill::Int8PrefillLimits}},
    tasks::{classify::{ClassificationLimits, ClassificationPlanner, quantized::Int8ClassificationRun}, ir::TaskBudget},
    tokenizer::untrusted::{SafeBpeWork, SafeBpeWorkLimits, UntrustedEncoding},
};

/// Model-free immutable factory. The borrowed pinned planner must be retained
/// and charged by the host; it never becomes a self-referential owned object.
pub struct Int8ClassificationJobPlanner<'p> {
    compiler: Int8ClassificationBatchPlanner<'p>,
    identity: ExecutionIdentity,
    recipe: ClassificationJobRecipe,
    encoding: Arc<EncodingWorkScope>,
}
/// The original invocation owns preparation, including tokenizer work.
/// Neither serialization nor an equivalent new factory can mint this owner.
pub struct PreparedInt8ClassificationJob {
    native: PreparedInt8BatchClassification,
    encoding: Arc<EncodingWorkScope>,
}
impl PreparedInt8ClassificationJob {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.native.execution_identity() }
    pub fn model_work(&self) -> Int8Work { self.native.model_work() }
    pub fn head_count(&self) -> usize { self.native.head_count() }
}
impl<'p> Int8ClassificationJobPlanner<'p> {
    pub fn new(planner: &'p ClassificationPlanner, identity: ExecutionIdentity, ceiling: TaskBudget,
        planning: ClassificationLimits, defaults: Option<ClassificationBatchArgs>, max_model_work: Int8Work)
        -> Result<Self, BatchFault> {
        Self::new_with_document_encoding(planner, identity, ceiling, planning, defaults, max_model_work, None)
    }
    /// Explicit compact input requires a nonrenewable invocation allowance.
    /// The actual pinned compiler owns the policy; callers cannot supply a
    /// second encoding identity that disagrees with its prompt tokens.
    pub fn new_with_document_encoding(planner: &'p ClassificationPlanner, identity: ExecutionIdentity,
        ceiling: TaskBudget, planning: ClassificationLimits, defaults: Option<ClassificationBatchArgs>,
        max_model_work: Int8Work, total: Option<SafeBpeWorkLimits>) -> Result<Self, BatchFault> {
        let encoding = EncodingWorkScope::new(planner.document_encoding(), total)?;
        Self::with_encoding_scope(planner, identity, ceiling, planning, defaults, max_model_work, encoding)
    }
    pub(crate) fn with_encoding_scope(planner: &'p ClassificationPlanner, identity: ExecutionIdentity,
        ceiling: TaskBudget, planning: ClassificationLimits, defaults: Option<ClassificationBatchArgs>,
        max_model_work: Int8Work, encoding: Arc<EncodingWorkScope>) -> Result<Self, BatchFault> {
        if planner.document_encoding() != &encoding.policy { return Err(BatchCode::Admission.into()); }
        encoding.workspace_bytes()?;
        let mut recipe = ClassificationJobRecipe::new(ceiling, planning, defaults, max_model_work)?;
        recipe.input_encoding = (encoding.policy != UntrustedEncoding::ByteFallback).then(|| encoding.policy.clone());
        recipe.encoding_work_limits = encoding.limits;
        bounded_json(&recipe, super::RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        let compiler = Int8ClassificationBatchPlanner::new_with_document_encoding(planner, identity.clone(),
            ceiling, planning, recipe.defaults.clone())?;
        Ok(Self { compiler, identity, recipe, encoding })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_ceiling(&self) -> TaskBudget { self.recipe.task_ceiling }
    pub fn job_recipe(&self) -> &ClassificationJobRecipe { &self.recipe }
    pub fn document_encoding(&self) -> &UntrustedEncoding { &self.encoding.policy }
    pub fn encoding_work_limits(&self) -> Option<SafeBpeWorkLimits> { self.encoding.limits }
    pub fn encoding_workspace_bytes(&self) -> Result<u64, BatchFault> { self.encoding.workspace_bytes() }
    pub fn encoding_work(&self) -> Result<SafeBpeWork, BatchFault> { self.encoding.work() }
    pub fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<ClassificationBatchArgs>,
        control: &mut C) -> Result<PreparedInt8ClassificationJob, BatchItemFailure> {
        let native = self.encoding.prepare(control, |ceiling, work, control|
            self.compiler.prepare_with_encoding_budget(document, control, work, ceiling))?;
        Ok(PreparedInt8ClassificationJob { native, encoding: Arc::clone(&self.encoding) })
    }
}

/// Real native execution only. A guarded result stays owned until its durable
/// spool frame is synced and its journal acknowledgement is committed.
pub struct Int8ClassificationJobProcessor<'p, 'e, 'w, A: Int8ClassificationAdmission> {
    native: NativeInt8ClassificationBatch<'p, 'e, 'w, A>,
    identity: ExecutionIdentity,
    recipe: ClassificationJobRecipe,
    encoding: Arc<EncodingWorkScope>,
}
impl<'p, 'e, 'w, A: Int8ClassificationAdmission> Int8ClassificationJobProcessor<'p, 'e, 'w, A> {
    pub fn new(planner: Int8ClassificationJobPlanner<'p>, engine: &'e mut StrictInt8Engine<'w>, admission: A)
        -> Result<Self, BatchFault> {
        let Int8ClassificationJobPlanner { compiler, identity, recipe, encoding } = planner;
        let native = NativeInt8ClassificationBatch::new(compiler, engine, admission, recipe.max_model_work)?;
        Ok(Self { native, identity, recipe, encoding })
    }
    /// Physical prompt scheduling changes neither the authenticated semantic
    /// recipe nor any durable attempt/work debit. The same spent tokenizer
    /// owner crosses the native handoff without a fresh allowance.
    pub fn new_layer_major(planner: Int8ClassificationJobPlanner<'p>, engine: &'e mut StrictInt8Engine<'w>,
        admission: A, prefill: Int8PrefillLimits) -> Result<Self, BatchFault> {
        let Int8ClassificationJobPlanner { compiler, identity, recipe, encoding } = planner;
        let native = NativeInt8ClassificationBatch::new_layer_major(compiler, engine, admission,
            recipe.max_model_work, prefill)?;
        Ok(Self { native, identity, recipe, encoding })
    }
}
impl<A: Int8ClassificationAdmission> BatchProcessor for Int8ClassificationJobProcessor<'_, '_, '_, A> {
    type Args = ClassificationBatchArgs;
    type Prepared = PreparedInt8ClassificationJob;
    type Output = GuardedOutput<Int8ClassificationRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        if self.encoding.policy != UntrustedEncoding::ByteFallback {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let native = self.native.prepare(document)?;
        Ok(PreparedInt8ClassificationJob { native, encoding: Arc::clone(&self.encoding) })
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        let native = self.encoding.prepare(control, |ceiling, work, control|
            self.native.prepare_with_encoding_budget(document, control, work, ceiling))?;
        Ok(PreparedInt8ClassificationJob { native, encoding: Arc::clone(&self.encoding) })
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
        self.native.execute_with_context(prepared.native, context, control)
    }
}
impl<A: Int8ClassificationAdmission> DurableBatchProcessor for Int8ClassificationJobProcessor<'_, '_, '_, A> {
    type Recipe = ClassificationJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        self.encoding.verify(&prepared.encoding)?;
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
