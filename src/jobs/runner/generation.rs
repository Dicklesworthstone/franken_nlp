//! Retained text generation on the existing addressed native batch adapter.
//! No replacement sampler, tokenizer, result producer or automatic retry.
use serde::{Deserialize, Serialize};
use super::DurableBatchProcessor;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor, BatchRequestContext,
        BatchWork, generation::{GenerationBatchArgs, GuardedOutput, quantized::{Int8BatchLimits,
            Int8GenerationBatchAdmission, Int8GenerationBatchPlanner, NativeInt8GenerationBatch}}},
    canonjson, execution_identity::{ExecutionIdentity, Sha256Digest},
    jobs::JobWork,
    native_engine::{decode::DecodeStepControl, generation::GenerationOptions,
        strict_int8::{Int8Work, StrictInt8Engine}},
    tasks::{chat::{ChatLimits, ChatMessage, quantized::{Int8ChatPlanner, Int8ChatResult, PreparedInt8Chat}},
        ir::TaskBudget},
    tokenizer::specials::TemplateControlIds,
};
mod recipe;
pub use recipe::GenerationJobRecipe;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationJobTask { Generate, Chat }
impl GenerationJobTask {
    pub fn identity(self) -> &'static str {
        match self { Self::Generate => "generate-v1", Self::Chat => "chat-v1" }
    }
}

/// Fixed caller-owned configuration. All effective generation options,
/// including an explicit seed, are frozen once, not read again per attempt.
/// Not Deserialize or Debug: it is neither wire authority nor safe telemetry.
pub struct GenerationJobConfig {
    pub task: GenerationJobTask,
    pub generation: GenerationOptions,
    pub budget: TaskBudget,
    pub planning: ChatLimits,
    pub native: Int8BatchLimits,
}

/// The record's `text` is the prompt/final user turn. Optional history is
/// permitted only for chat. These fields cannot change generation or budgets.
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationJobArgs {
    #[serde(default)] pub sample_index: u64,
    #[serde(default)] pub history: Vec<ChatMessage>,
}

pub struct Int8GenerationJobPlanner {
    planner: Int8ChatPlanner,
    recipe: GenerationJobRecipe,
    binding: Sha256Digest,
}
/// Sealed origin as well as native request. Plans prepared by a different
/// job contract cannot be transferred to this processor, even on one model.
pub struct PreparedGenerationJob { native: PreparedInt8Chat, binding: Sha256Digest }
impl PreparedGenerationJob {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.native.execution_identity() }
    pub fn model_work(&self) -> Int8Work { self.native.planned_work() }
}
impl Int8GenerationJobPlanner {
    pub fn pinned(controls: &TemplateControlIds, eos: u32, identity: ExecutionIdentity,
        config: GenerationJobConfig) -> Result<Self, BatchFault> {
        if identity.task_spec != config.task.identity() { return Err(BatchCode::Admission.into()); }
        recipe::validate(&config, eos)?;
        let planner = Int8ChatPlanner::pinned(controls, eos, identity, config.budget, config.planning)
            .map_err(|_| BatchCode::Admission)?;
        let recipe = GenerationJobRecipe::new(config)?;
        let bytes = canonjson::canonical_bytes(&(planner.execution_identity(), &recipe))
            .map_err(|_| BatchCode::Serialization)?;
        let binding = Sha256Digest::of_bytes(&bytes);
        Ok(Self { planner, recipe, binding })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    pub fn job_recipe(&self) -> &GenerationJobRecipe { &self.recipe }
    pub fn task_budget(&self) -> TaskBudget { self.recipe.budget }
    pub fn native_limits(&self) -> Int8BatchLimits { self.recipe.native_limits() }

    /// Actual pinned planning only. The source is never normalized or silently
    /// shortened; the shared chat compiler checks the complete role history.
    pub fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<GenerationJobArgs>,
        control: &mut C) -> Result<PreparedGenerationJob, BatchItemFailure> {
        let document = self.document(document, control)?;
        let compiler = Int8GenerationBatchPlanner::new(&self.planner, None).map_err(BatchItemFailure::fatal)?;
        self.seal(compiler.prepare(document)?, control)
    }
    fn document<C: DecodeStepControl>(&self, document: BatchDocument<GenerationJobArgs>, control: &mut C)
        -> Result<BatchDocument<GenerationBatchArgs>, BatchItemFailure> {
        poll(control)?;
        let args = document.task_args.unwrap_or_default();
        self.recipe.check_input(&document.id, &document.text, &args)?;
        // Clone ONLY bounded fixed options; history/text are moved, not copied
        // by this adapter. The native compiler independently enforces limits.
        let generation = self.recipe.generation.clone(); let budget = self.recipe.budget;
        let task_args = match self.recipe.task {
            GenerationJobTask::Generate => GenerationBatchArgs::Generate { generation, budget, sample_index: args.sample_index },
            GenerationJobTask::Chat => GenerationBatchArgs::Chat { history: args.history, generation, budget,
                sample_index: args.sample_index },
        };
        Ok(BatchDocument { id: document.id, text: document.text, task_args: Some(task_args) })
    }
    fn seal<C: DecodeStepControl>(&self, native: PreparedInt8Chat, control: &mut C)
        -> Result<PreparedGenerationJob, BatchItemFailure> {
        poll(control)?;
        if !fits(native.planned_work(), self.recipe.native_limits().max_model_work)
            || native.native_plan().sampler_bytes() > self.recipe.native_limits().max_sampler_bytes {
            return Err(BatchItemFailure::reject(BatchCode::WorkLimit));
        }
        Ok(PreparedGenerationJob { native, binding: self.binding })
    }
    fn check(&self, prepared: &PreparedGenerationJob) -> Result<(), BatchItemFailure> {
        if prepared.binding != self.binding || prepared.native.task_plan().task_spec_identity() != self.recipe.task.identity() {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        Ok(())
    }
}

pub struct Int8GenerationJobProcessor<'p, 'e, 'w, A: Int8GenerationBatchAdmission> {
    planner: &'p Int8GenerationJobPlanner,
    native: NativeInt8GenerationBatch<'p, 'e, 'w, A>,
    failed: bool,
}
impl<'p, 'e, 'w, A: Int8GenerationBatchAdmission> Int8GenerationJobProcessor<'p, 'e, 'w, A> {
    pub fn new(planner: &'p Int8GenerationJobPlanner, engine: &'e mut StrictInt8Engine<'w>, admission: A)
        -> Result<Self, BatchFault> {
        let compiler = Int8GenerationBatchPlanner::new(&planner.planner, None)?;
        let native = NativeInt8GenerationBatch::new(compiler, engine, admission, planner.native_limits())?;
        Ok(Self { planner, native, failed: false })
    }
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed || self.native.is_poisoned() { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
}
impl<A: Int8GenerationBatchAdmission> BatchProcessor for Int8GenerationJobProcessor<'_, '_, '_, A> {
    type Args = GenerationJobArgs;
    type Prepared = PreparedGenerationJob;
    type Output = GuardedOutput<Int8ChatResult, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.ready()?; self.failed = true;
        let input = self.planner.document(document, control)?;
        // Bounded compilation is polled around, not falsely claimed preemptible.
        let native = self.native.prepare(input)?;
        let prepared = self.planner.seal(native, control)?;
        self.failed = false; Ok(prepared)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.model_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.ready()?; self.failed = true;
        self.planner.check(&prepared)?; poll(control)?;
        // JobRunner supplies the ORIGINAL ordinal, not attempt count. The
        // native sampler uses item ID/sample index, not this delivery sequence.
        let output = self.native.execute_with_context(prepared.native, context, control)?;
        poll(control)?;
        self.failed = false; Ok(output)
    }
}
impl<A: Int8GenerationBatchAdmission> DurableBatchProcessor for Int8GenerationJobProcessor<'_, '_, '_, A> {
    type Recipe = GenerationJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { self.planner.job_recipe() }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        self.ready()?; self.planner.check(prepared)?;
        // Admit full worst-case work before the first token, including failed
        // attempts. EOS/stop/byte limits do not refund or renew this allowance.
        Ok(JobWork { model: prepared.model_work(), mask_node_visits: 0 })
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.planner.task_budget().max_output_bytes }
}
fn poll<C: DecodeStepControl>(control: &mut C) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}
fn fits(a: Int8Work, b: Int8Work) -> bool {
    a.forward_positions <= b.forward_positions && a.projected_logits <= b.projected_logits
        && a.attention_pairs <= b.attention_pairs && a.projections.fits(b.projections)
}
#[cfg(test)] mod tests;
