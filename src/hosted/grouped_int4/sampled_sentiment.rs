//! Independent sampled sentiment generations under one admitted native call.
//! Each sample starts with empty logical KV; the complete bundle shares one
//! native allocation, sampler workspace, mask allowance and cancellation scope.

use super::*;
use crate::{
    grammar::mask::MaskWorkLimits,
    native_engine::{
        constrained::JsonWorkBudget,
        constrained_int4::{Int4JsonBudget, json_sampling_workspace_bytes},
        strict_int4::Int4Work,
    },
    tasks::{ir::TaskBudget, sentiment::sampled::{PreparedSampledSentiment,
        SampledSentimentError, SampledSentimentRun, SampledSentimentVocabulary}},
};

/// All work ceilings apply to the whole axis-by-sample bundle. Preparation
/// includes the sealed prompts and immutable vocabulary; callers also retain
/// their own preparation charge while constructing these values before transfer.
#[derive(Clone, Copy, Debug)]
pub struct SampledSentimentHostLimits {
    pub native: NativeLimits,
    pub preparation_reserve_bytes: u64,
    pub max_model_work: Int4Work,
    pub max_sampler_bytes: u64,
    pub mask_limits: MaskWorkLimits,
    pub max_mask_node_visits: u64,
}

impl NlpEngine {
    /// Run independently addressed, schema-constrained score/explanation
    /// samples on a resident candidate. Only a complete validated bundle is
    /// returned. The output guard survives caller serialization and delivery.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_sampled_sentiment(&self, model: &ResidentInt4,
        prepared: PreparedSampledSentiment, vocabulary: Arc<SampledSentimentVocabulary>,
        limits: SampledSentimentHostLimits, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<SampledSentimentRun>, HostedError> {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) {
            return Err(HostedError::ResourceDomain);
        }
        crate::native_engine::constrained_int4::check_model(
            prepared.execution_identity(), model.model_identity())
            .map_err(SampledSentimentError::from).map_err(task_error)?;
        if prepared.group_width() != model.model_identity().group_width() {
            return Err(HostedError::ModelIdentity);
        }
        let work = prepared.planned_work();
        let (required, scratch_bytes) = requirements(limits, prepared.task_budget(),
            prepared.required_context(), prepared.group_width(), work,
            prepared.preparation_requirement_bytes())?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers,
            limits.preparation_reserve_bytes)?, || Ok((prepared, vocabulary)))?;
        let output = output_claim(&lease, input.value.0.max_result_bytes(),
            u64::from(input.value.0.task_budget().max_output_tokens))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let input = input;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens,
                    Int4MemoryBudget { max_kv_bytes: required.kv_bytes,
                        max_rope_bytes: required.rope_bytes,
                        max_scratch_payload_bytes: required.scratch_payload_bound })
                    .map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let result = allocate(output, || {
                let (prepared, vocabulary) = &input.value;
                let budget = Int4JsonBudget {
                    native: Int4RunBudget::exact(work),
                    json: JsonWorkBudget { max_forward_positions: work.forward_positions,
                        max_projected_logits: work.projected_logits,
                        max_kv_bytes: required.kv_bytes,
                        max_total_mask_node_visits: limits.max_mask_node_visits,
                        mask_limits: limits.mask_limits },
                };
                let run = prepared.execute_with_control(&mut engine.value,
                    prepared.execution_identity(), vocabulary, budget,
                    limits.max_sampler_bytes, control).map_err(task_error)?;
                if !fits(run.model_work, work) || run.group_width != prepared.group_width()
                    || engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(HostedError::Limits("sampled sentiment completion accounting"));
                }
                checkpoint(control)?;
                Ok(run)
            })?;
            drop(engine);
            drop(input);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

fn task_error(error: SampledSentimentError) -> HostedError {
    HostedError::Int4(Int4HostError::SampledSentiment(error))
}

fn fits(work: Int4Work, cap: Int4Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}

fn requirements(limits: SampledSentimentHostLimits, task: TaskBudget, context: usize,
    width: usize, work: Int4Work, preparation_bytes: u64)
    -> Result<(Int4MemoryRequirement, u64), HostedError> {
    limits.native.run.validate()?;
    task.validate().map_err(|_| HostedError::Limits("sampled sentiment task budget"))?;
    let sampler_bytes = json_sampling_workspace_bytes()
        .map_err(SampledSentimentError::from).map_err(task_error)?;
    if preparation_bytes == 0 || preparation_bytes > limits.preparation_reserve_bytes
        || limits.native.allocator_reserve_bytes == 0
        || context == 0 || context > limits.native.context_tokens
        || work.forward_positions == 0 || work.projected_logits == 0
        || !fits(work, limits.max_model_work) || sampler_bytes > limits.max_sampler_bytes
        || limits.mask_limits.max_trie_node_visits == 0
        || limits.mask_limits.checkpoint_interval_nodes == 0 || limits.max_mask_node_visits == 0 {
        return Err(HostedError::Limits("sampled sentiment complete work, sampler or preparation"));
    }
    let required = Int4MemoryRequirement::for_context(limits.native.context_tokens, width)
        .map_err(native_error)?;
    if required.kv_bytes > task.max_kv_bytes {
        return Err(HostedError::Limits("sampled sentiment complete resident KV"));
    }
    let scratch = sum(&[required.rope_bytes, required.scratch_payload_bound,
        limits.native.allocator_reserve_bytes, sampler_bytes])?;
    Ok((required, scratch))
}

#[cfg(test)]
mod tests;
