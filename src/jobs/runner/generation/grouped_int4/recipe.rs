//! Exact private generation policy and resource commitment for resume.
use super::*;
use crate::{jobs::manifest::bounded_json,
    native_engine::{generation::{GenerationLimits, GenerationSampling},
        portable_int8::ProjectionWork, sampler::SAMPLER_VERSION, strict_int4::Int4RunBudget},
    tasks::chat::CHAT_PROMPT_VERSION};

#[derive(Serialize)]
struct PlanningRecipe {
    max_messages: usize, max_message_bytes: usize, max_total_message_bytes: usize,
    max_prompt_tokens: usize, max_new_tokens: usize, max_output_bytes: usize, max_sampler_bytes: u64,
}
impl From<ChatLimits> for PlanningRecipe {
    fn from(limits: ChatLimits) -> Self {
        let ChatLimits { max_messages, max_message_bytes, max_total_message_bytes, generation } = limits;
        let GenerationLimits { max_prompt_tokens, max_new_tokens, max_output_bytes, max_sampler_bytes } = generation;
        Self { max_messages, max_message_bytes, max_total_message_bytes, max_prompt_tokens,
            max_new_tokens, max_output_bytes, max_sampler_bytes }
    }
}
#[derive(Serialize)]
struct NativeRecipe {
    max_forward_positions: u64, max_attention_pairs: u64, max_projection_work: ProjectionWork,
    max_kv_bytes: u64, max_sampler_bytes: u64,
}
impl From<Int4GenerationBudget> for NativeRecipe {
    fn from(limits: Int4GenerationBudget) -> Self {
        let Int4GenerationBudget { native, max_kv_bytes, max_sampler_bytes } = limits;
        let Int4RunBudget { max_forward_positions, max_attention_pairs, max_projection_work } = native;
        Self { max_forward_positions, max_attention_pairs, max_projection_work, max_kv_bytes, max_sampler_bytes }
    }
}

/// Serialize-only settings created by the actual model-bound factory. Exact
/// options, including seed and byte stops, are private recipe contents; only
/// JobRunner's domain-separated keyed commitment enters metadata storage.
#[derive(Serialize)]
pub struct Int4GenerationJobRecipe {
    version: u32, input_profile: &'static str, dependency_scope: &'static str,
    execution: &'static str, prompt_version: &'static str, sampler_version: &'static str,
    processor_version: &'static str, addressing: &'static str, reservation_policy: &'static str,
    pub(super) task: GenerationJobTask,
    pub(super) generation: GenerationOptions,
    pub(super) budget: TaskBudget,
    pub(super) group_width: usize,
    pub(super) context_tokens: usize,
    pub(super) job: JobLimits,
    planning: PlanningRecipe,
    native: NativeRecipe,
}
impl Int4GenerationJobRecipe {
    pub(super) fn new(config: Int4GenerationJobConfig, group_width: usize, job: JobLimits) -> Result<Self, BatchFault> {
        // Exhaustive projections make added planner/native axes visible here.
        let Int4GenerationJobConfig { task, generation, budget, planning, native, context_tokens } = config;
        let recipe = Self { version: 1, input_profile: "id-text-history-sample-v1", dependency_scope: "item-local",
            execution: INT4_GENERATION_VERSION, prompt_version: CHAT_PROMPT_VERSION,
            sampler_version: sampler_version(&generation), processor_version: PROCESSOR_VERSION,
            addressing: "exact-item-id-and-sample-index; original-ordinal-delivery; no-attempt-reseed-v1",
            reservation_policy: "complete-native-all-five-work-per-attempt; full-resident-kv-v1",
            task, generation, budget, group_width, context_tokens, job, planning: planning.into(), native: native.into() };
        bounded_json(&recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(recipe)
    }
    pub(super) fn native_limits(&self) -> Int4GenerationBudget {
        Int4GenerationBudget { native: Int4RunBudget { max_forward_positions: self.native.max_forward_positions,
            max_attention_pairs: self.native.max_attention_pairs, max_projection_work: self.native.max_projection_work },
            max_kv_bytes: self.native.max_kv_bytes, max_sampler_bytes: self.native.max_sampler_bytes }
    }
    pub(super) fn check_input(&self, id: &str, text: &str, args: &GenerationJobArgs) -> Result<(), BatchItemFailure> {
        let reject = || BatchItemFailure::fatal(BatchCode::Planning);
        if id.trim().is_empty() || id.len() > self.job.max_id_bytes.min(256) || id.chars().any(char::is_control)
            || text.len() > self.planning.max_message_bytes || text.len() > self.job.max_input_bytes_per_item
            || args.history.len() >= self.planning.max_messages
            || self.task == GenerationJobTask::Generate && !args.history.is_empty() { return Err(reject()); }
        let mut bytes = text.len();
        for message in &args.history {
            if message.content.len() > self.planning.max_message_bytes { return Err(reject()); }
            bytes = bytes.checked_add(message.content.len()).ok_or_else(reject)?;
        }
        if bytes > self.planning.max_total_message_bytes || bytes > self.job.max_input_bytes_per_item {
            return Err(reject());
        }
        Ok(())
    }
}
pub(super) fn sampler_version(options: &GenerationOptions) -> &'static str {
    match &options.sampling { GenerationSampling::Greedy => "fnlp-greedy-v1", GenerationSampling::Seeded { .. } => SAMPLER_VERSION }
}
pub(super) fn validate(config: &Int4GenerationJobConfig, group_width: usize, job: JobLimits) -> Result<(), BatchFault> {
    job.validate().map_err(|_| BatchCode::InvalidLimits)?;
    config.budget.validate().map_err(|_| BatchCode::InvalidLimits)?;
    let required = Int4MemoryRequirement::for_context(config.context_tokens, group_width)
        .map_err(|_| BatchCode::InvalidLimits)?;
    let mut generation = config.planning.generation;
    generation.max_sampler_bytes = generation.max_sampler_bytes.min(config.native.max_sampler_bytes);
    config.generation.validate(generation).map_err(|_| BatchCode::InvalidLimits)?;
    let prompt = (config.budget.max_input_tokens as usize).min(generation.max_prompt_tokens);
    let positions = prompt.checked_add(config.generation.max_new_tokens)
        .and_then(|n| n.checked_sub(1)).ok_or(BatchCode::InvalidLimits)?;
    let native = config.native.native; let work = job.max_work.int4_model_work();
    if config.budget.max_kv_bytes != required.kv_bytes || config.native.max_kv_bytes != required.kv_bytes
        || prompt == 0 || positions > config.context_tokens
        || config.generation.max_new_tokens > config.budget.max_output_tokens as usize
        || config.generation.max_output_bytes as u64 > config.budget.max_output_bytes
        || config.budget.max_output_bytes > job.max_result_bytes as u64
        || !(1..=128).contains(&config.planning.max_messages)
        || config.planning.max_message_bytes == 0 || config.planning.max_total_message_bytes == 0
        || work.forward_positions == 0 || work.projected_logits == 0 || work.attention_pairs == 0
        || work.projections.dot_products == 0 || work.projections.multiply_accumulates == 0
        || native.max_forward_positions == 0 || native.max_attention_pairs == 0
        || native.max_projection_work.dot_products == 0 || native.max_projection_work.multiply_accumulates == 0
        || native.max_forward_positions > work.forward_positions || native.max_attention_pairs > work.attention_pairs
        || !native.max_projection_work.fits(work.projections) {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
