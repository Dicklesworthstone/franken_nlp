//! Exact private replay contract; no caller-supplied receipt or native factory.
use super::*;
use crate::{jobs::manifest::bounded_json,
    native_engine::{generation::{GenerationLimits, GenerationSampling, PROCESSOR_VERSION, quantized::INT8_GENERATION_VERSION}, sampler::SAMPLER_VERSION},
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
pub struct GenerationJobRecipe {
    version: u32, input_profile: &'static str, dependency_scope: &'static str,
    execution: &'static str, prompt_version: &'static str,
    sampler_version: &'static str, processor_version: &'static str, addressing: &'static str,
    pub(super) task: GenerationJobTask,
    pub(super) generation: GenerationOptions,
    pub(super) budget: TaskBudget,
    planning: PlanningRecipe, max_model_work: Int8Work, max_sampler_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<UntrustedEncoding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) encoding_work_limits: Option<SafeBpeWorkLimits>,
}
impl GenerationJobRecipe {
    pub(super) fn new(config: GenerationJobConfig) -> Result<Self, BatchFault> {
        let Int8BatchLimits { max_sampler_bytes, max_model_work } = config.native;
        let recipe = Self { version: 1, input_profile: "id-text-history-sample-v1", dependency_scope: "item-local",
            execution: INT8_GENERATION_VERSION, prompt_version: CHAT_PROMPT_VERSION,
            sampler_version: match &config.generation.sampling {
                GenerationSampling::Greedy => "fnlp-greedy-v1",
                GenerationSampling::Seeded { .. } => SAMPLER_VERSION,
            }, processor_version: PROCESSOR_VERSION,
            addressing: "exact-item-id-and-sample-index; original-ordinal-delivery; no-attempt-reseed-v1",
            task: config.task, generation: config.generation, budget: config.budget,
            planning: config.planning.into(), max_model_work, max_sampler_bytes,
            input_encoding: None, encoding_work_limits: None };
        // The enclosing JobRunner recipe gets its own remaining envelope room.
        // Only a domain-separated keyed commitment reaches the job manifest.
        bounded_json(&recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(recipe)
    }
    pub(super) fn with_document_encoding(mut self, encoding: &UntrustedEncoding,
        total: Option<SafeBpeWorkLimits>) -> Result<Self, BatchFault> {
        encoding_workspace(encoding, total)?;
        self.input_encoding = (encoding != &UntrustedEncoding::ByteFallback).then(|| encoding.clone());
        self.encoding_work_limits = total;
        bounded_json(&self, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(self)
    }
    pub(super) fn native_limits(&self) -> Int8BatchLimits {
        Int8BatchLimits { max_sampler_bytes: self.max_sampler_bytes, max_model_work: self.max_model_work }
    }
    pub(super) fn check_input(&self, id: &str, text: &str, args: &GenerationJobArgs) -> Result<(), BatchItemFailure> {
        let reject = || BatchItemFailure::reject(BatchCode::Planning);
        if id.trim().is_empty() || id.len() > 256 || id.chars().any(char::is_control)
            || text.len() > self.planning.max_message_bytes
            || args.history.len() >= self.planning.max_messages
            || self.task == GenerationJobTask::Generate && !args.history.is_empty() { return Err(reject()); }
        let mut bytes = text.len();
        for message in &args.history {
            if message.content.len() > self.planning.max_message_bytes { return Err(reject()); }
            bytes = bytes.checked_add(message.content.len()).ok_or_else(reject)?;
        }
        if bytes > self.planning.max_total_message_bytes { return Err(reject()); }
        Ok(())
    }
}
pub(super) fn encoding_workspace(encoding: &UntrustedEncoding,
    total: Option<SafeBpeWorkLimits>) -> Result<u64, BatchFault> {
    match (encoding, total) {
        (UntrustedEncoding::ByteFallback, None) => Ok(0),
        (UntrustedEncoding::SafeBpeV1(call), Some(total)) => {
            total.validate().map_err(|_| BatchCode::InvalidLimits)?;
            call.requirements().map_err(|_| BatchCode::InvalidLimits)?.workspace_bytes
                .checked_add(1024).ok_or_else(|| BatchCode::InvalidLimits.into())
        }
        _ => Err(BatchCode::InvalidLimits.into()),
    }
}
pub(super) fn validate(config: &GenerationJobConfig, eos: u32) -> Result<(), BatchFault> {
    config.budget.validate().map_err(|_| BatchCode::InvalidLimits)?;
    let mut generation = config.planning.generation;
    generation.max_sampler_bytes = generation.max_sampler_bytes.min(config.native.max_sampler_bytes);
    config.generation.validate(generation).map_err(|_| BatchCode::InvalidLimits)?;
    let work = config.native.max_model_work;
    if config.generation.eos_token_ids != [eos] || config.generation.banned_token_ids.contains(&eos)
        || config.generation.max_new_tokens > config.budget.max_output_tokens as usize
        || config.generation.max_output_bytes as u64 > config.budget.max_output_bytes
        || config.planning.max_messages == 0 || config.planning.max_messages > 128
        || config.planning.max_message_bytes == 0 || config.planning.max_total_message_bytes == 0
        || work.forward_positions == 0 || work.projected_logits == 0 || work.attention_pairs == 0
        || work.projections.dot_products == 0 || work.projections.multiply_accumulates == 0 {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
