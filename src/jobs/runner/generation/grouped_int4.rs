//! Durable free text and chat on the genuine grouped-INT4 native generator.
//!
//! The immutable model-bound factory fixes the complete sampling policy,
//! context and lifetime job limits. Each original item compiles its own pinned
//! transcript and addressed sampler key; attempt and delivery numbers never
//! reseed it. No alternate generator, caller recipe or decoded receipt is an
//! execution authority.
use super::{GenerationJobArgs, GenerationJobTask};
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, output::GuardedOutput},
    canonjson,
    execution_identity::{ExecutionIdentity, Sha256Digest},
    jobs::{JobLimits, JobWork, runner::DurableBatchProcessor},
    native_engine::{constrained_int4, decode::DecodeStepControl,
        generation::{GenerationOptions, PROCESSOR_VERSION,
            grouped_int4::{Int4GenerationBudget, INT4_GENERATION_VERSION}},
        strict_int4::{Int4MemoryRequirement, Int4ModelIdentity, Int4Work,
            StrictInt4Engine, STRICT_INT4_PROFILE}},
    tasks::{chat::{ChatLimits, ChatMessage, ChatRequest, ChatRole, GenerateRequest,
        grouped_int4::{Int4ChatError, Int4ChatPlanner, Int4ChatResult, PreparedInt4Chat}},
        ir::TaskBudget},
};
use serde::Serialize;
use std::sync::Arc;

mod recipe;
mod execution;
pub use recipe::Int4GenerationJobRecipe;
pub use execution::{Int4GenerationAdmission, Int4GenerationBatchAdmission, Int4GenerationJobProcessor};

/// Fixed before population freeze. Sampling already contains its effective
/// seed; a resumed process never obtains a replacement seed or policy.
pub struct Int4GenerationJobConfig {
    pub task: GenerationJobTask,
    pub generation: GenerationOptions,
    pub budget: TaskBudget,
    pub planning: ChatLimits,
    /// Per-attempt native ceiling. The complete JobLimits additionally bounds
    /// cumulative work across this invocation and every persisted retry.
    pub native: Int4GenerationBudget,
    pub context_tokens: usize,
}

pub struct Int4GenerationJobPlanner {
    compiler: Int4ChatPlanner,
    identity: ExecutionIdentity,
    recipe: Int4GenerationJobRecipe,
    owner: Arc<()>,
}
impl Int4GenerationJobPlanner {
    pub fn for_model(model: &Int4ModelIdentity, config: Int4GenerationJobConfig, limits: JobLimits)
        -> Result<Self, BatchFault> {
        // Reject invalid/unbounded options before allocating pinned compiler
        // assets, and before the host can create or resume any job files.
        recipe::validate(&config, model.group_width(), limits)?;
        let compiler = Int4ChatPlanner::for_model(model, config.budget, config.planning)
            .map_err(|error| failure(error, BatchCode::Planning).fault)?;
        Self::configured(compiler, config, limits)
    }
    // Planning fixtures use the real pinned compiler. Production construction
    // is exclusively model-bound; there is no public identity/receipt factory.
    fn configured(compiler: Int4ChatPlanner, config: Int4GenerationJobConfig, limits: JobLimits)
        -> Result<Self, BatchFault> {
        recipe::validate(&config, compiler.group_width(), limits)?;
        if config.generation.eos_token_ids != [compiler.eos_token_id()]
            || config.generation.banned_token_ids.contains(&compiler.eos_token_id()) {
            return Err(BatchCode::InvalidLimits.into());
        }
        let mut identity = compiler.execution_identity().clone();
        identity.task_spec = config.task.identity().to_owned();
        identity.sampler_version = recipe::sampler_version(&config.generation).to_owned();
        // This is a private fixed-policy identity, not an item sampling key.
        // The actual compiler binds prompt, TaskIR, stable item ID and sample
        // index independently for every item. Only keyed commitments persist.
        let policy = canonjson::canonical_bytes(&("fnlp-int4-generation-job-policy-v1",
            INT4_GENERATION_VERSION, PROCESSOR_VERSION, config.task, &config.generation))
            .map_err(|_| BatchCode::Planning)?;
        identity.decision_policy_digest = Sha256Digest::of_bytes(&policy);
        identity.validate().map_err(|_| BatchCode::Admission)?;
        let recipe = Int4GenerationJobRecipe::new(config, compiler.group_width(), limits)?;
        Ok(Self { compiler, identity, recipe, owner: Arc::new(()) })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_budget(&self) -> TaskBudget { self.recipe.budget }
    pub fn native_limits(&self) -> Int4GenerationBudget { self.recipe.native_limits() }
    pub fn job_limits(&self) -> JobLimits { self.recipe.job }
    pub fn context_tokens(&self) -> usize { self.recipe.context_tokens }
    pub fn group_width(&self) -> usize { self.recipe.group_width }
    pub fn max_result_bytes(&self) -> u64 { self.task_budget().max_output_bytes }
    pub fn output_token_ceiling(&self) -> u64 { u64::from(self.task_budget().max_output_tokens) }
    pub fn job_recipe(&self) -> &Int4GenerationJobRecipe { &self.recipe }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        if limits != self.job_limits() { return Err(BatchCode::InvalidLimits.into()); }
        Ok(())
    }
    fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<GenerationJobArgs>, control: &mut C)
        -> Result<PreparedInt4GenerationJob, BatchItemFailure> {
        poll(control)?;
        let args = document.task_args.unwrap_or_default();
        self.recipe.check_input(&document.id, &document.text, &args)?;
        let sample_index = args.sample_index;
        // Options were validated and privately frozen before this bounded
        // clone. Exact input and history strings move into the typed request.
        let generation = self.recipe.generation.clone();
        let budget = self.task_budget();
        let native = match self.recipe.task {
            GenerationJobTask::Generate => self.compiler.plan_generate(&GenerateRequest {
                item_id: document.id, sample_index, prompt: document.text, generation, budget }),
            GenerationJobTask::Chat => {
                let mut messages = args.history;
                messages.try_reserve_exact(1).map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
                messages.push(ChatMessage { role: ChatRole::User, content: document.text });
                self.compiler.plan_chat(&ChatRequest { item_id: document.id, sample_index, messages, generation, budget })
            }
        }.map_err(|error| failure(error, BatchCode::Planning))?;
        // Compilation is bounded and polled around, not claimed preemptible.
        poll(control)?;
        let work = JobWork::from_int4(native.planned_work(), 0);
        let prepared = PreparedInt4GenerationJob { native, work, sample_index, owner: Arc::clone(&self.owner) };
        self.check(&prepared)?;
        Ok(prepared)
    }
    fn check(&self, prepared: &PreparedInt4GenerationJob) -> Result<(), BatchItemFailure> {
        if !Arc::ptr_eq(&self.owner, &prepared.owner)
            || prepared.native.task_plan().task_spec_identity() != self.recipe.task.identity()
            || prepared.native.native_plan().group_width() != self.group_width()
            || prepared.work != JobWork::from_int4(prepared.native.planned_work(), 0) {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        let work = prepared.native.planned_work();
        let budget = self.native_limits();
        if !prepared.work.fits(self.job_limits().max_work)
            || work.forward_positions > budget.native.max_forward_positions
            || work.attention_pairs > budget.native.max_attention_pairs
            || !work.projections.fits(budget.native.max_projection_work)
            || work.forward_positions > self.context_tokens() as u64
            || prepared.native.native_plan().sampler_bytes() > budget.max_sampler_bytes {
            return Err(BatchItemFailure::fatal(BatchCode::WorkLimit));
        }
        Ok(())
    }
}

/// Actual immutable native preparation plus a factory-local ownership seal.
/// It cannot be deserialized, altered, executed through a different factory,
/// or exported as a purported model receipt.
pub struct PreparedInt4GenerationJob {
    native: PreparedInt4Chat,
    work: JobWork,
    sample_index: u64,
    owner: Arc<()>,
}
impl PreparedInt4GenerationJob {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.native.execution_identity() }
    pub fn model_work(&self) -> Int4Work { self.native.planned_work() }
    pub fn sampler_bytes(&self) -> u64 { self.native.native_plan().sampler_bytes() }
}

fn check_capacity(planner: &Int4GenerationJobPlanner, engine: &StrictInt4Engine<'_>) -> Result<u64, BatchItemFailure> {
    constrained_int4::check_model(planner.execution_identity(), engine.model_identity())
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0)
        || engine.profile() != STRICT_INT4_PROFILE
        || engine.model_identity().group_width() != planner.group_width()
        || engine.kv_cache().capacity_positions() != planner.context_tokens() {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let required = Int4MemoryRequirement::for_context(planner.context_tokens(), planner.group_width())
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    if required.kv_bytes == 0 || required.kv_bytes != planner.task_budget().max_kv_bytes
        || required.kv_bytes != planner.native_limits().max_kv_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    Ok(required.kv_bytes)
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}
fn failure(error: Int4ChatError, fallback: BatchCode) -> BatchItemFailure {
    error.cancellation().map_or_else(|| BatchItemFailure::fatal(fallback),
        |cause| BatchItemFailure::fatal(BatchFault::cancelled(cause)))
}

#[cfg(test)] mod tests;
