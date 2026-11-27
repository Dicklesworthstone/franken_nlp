//! Complete independent INT8 score/justification bundles on one charged host.
//! Every sample has fresh logical KV and its own addressed draws. Native,
//! sampler, preparation and output ownership survive physical completion.

use super::*;
use crate::{
    grammar::mask::MaskWorkLimits,
    native_engine::{
        constrained::JsonWorkBudget,
        constrained_int8::{Int8JsonBudget, json_sampling_workspace_bytes},
        strict_int8::{Int8Work, prefill::Int8PrefillLimits},
    },
    tasks::{ir::TaskBudget, sentiment::sampled::{
        SampledSentimentVocabulary,
        quantized::{PreparedInt8SampledSentiment, Int8SampledSentimentRun,
            Int8SampledSentimentError, INT8_SAMPLED_SENTIMENT_EXECUTION},
    }},
};

/// Nonrenewable limits for the complete axis-by-sample bundle. Preparation
/// covers transferred plans and vocabulary in addition to any caller-owned
/// preparation reservation; it is modeled admission, not measured RSS.
#[derive(Clone, Copy, Debug)]
pub struct Int8SampledSentimentHostLimits {
    pub native: NativeLimits,
    pub preparation_reserve_bytes: u64,
    pub max_model_work: Int8Work,
    pub max_sampler_bytes: u64,
    pub mask_limits: MaskWorkLimits,
    pub max_mask_node_visits: u64,
}

impl NlpEngine {
    pub fn execute_int8_sampled_sentiment(&self, model: &ResidentInt8,
        prepared: PreparedInt8SampledSentiment, vocabulary: Arc<SampledSentimentVocabulary>,
        limits: Int8SampledSentimentHostLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8SampledSentimentRun>, HostedError> {
        self.execute_int8_sampled_sentiment_scheduled(model, prepared, vocabulary,
            limits, None, cancellation)
    }

    /// Only prompt scheduling changes: each sample still begins with empty KV,
    /// scores the complete vocabulary and consumes its independently addressed
    /// random sequence. The declared prompt workspace is charged before dispatch.
    pub fn execute_int8_sampled_sentiment_layer_major(&self, model: &ResidentInt8,
        prepared: PreparedInt8SampledSentiment, vocabulary: Arc<SampledSentimentVocabulary>,
        limits: Int8SampledSentimentHostLimits, prefill: Int8PrefillLimits,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8SampledSentimentRun>, HostedError> {
        self.execute_int8_sampled_sentiment_scheduled(model, prepared, vocabulary,
            limits, Some(prefill), cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_int8_sampled_sentiment_scheduled(&self, model: &ResidentInt8,
        prepared: PreparedInt8SampledSentiment, vocabulary: Arc<SampledSentimentVocabulary>,
        limits: Int8SampledSentimentHostLimits, prefill: Option<Int8PrefillLimits>,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8SampledSentimentRun>, HostedError> {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.execution_identity())?;
        let required = requirements(limits.native)?;
        let work = prepared.planned_work();
        let scratch = validate(limits, prepared.task_budget(), prepared.required_context(),
            required, work, prepared.preparation_requirement_bytes())?;
        let scratch = super::scored::scoring_scratch(scratch, prefill)?;
        let budget = execution_budget(work, required.kv_bytes, limits);
        prepared.complete_mask_charge(&vocabulary, budget, limits.max_sampler_bytes)
            .map_err(HostedError::SampledSentiment)?;
        let lease = self.resources().acquire_lease();
        let output = output_claim(&lease, prepared.max_result_bytes(),
            u64::from(prepared.task_budget().max_output_tokens))?;
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers,
            limits.preparation_reserve_bytes)?, || Ok((prepared, vocabulary)))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let input = input;
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required))
                .map_err(HostedError::Model))?;
            let result = allocate(output, || {
                let (prepared, vocabulary) = &input.value;
                let run = match prefill {
                    None => prepared.execute_with_control(&mut engine.value,
                        prepared.execution_identity(), vocabulary, budget,
                        limits.max_sampler_bytes, control),
                    Some(prefill) => prepared.execute_layer_major_with_control(&mut engine.value,
                        prepared.execution_identity(), vocabulary, budget,
                        limits.max_sampler_bytes, prefill, control),
                }.map_err(HostedError::SampledSentiment)?;
                if run.execution != INT8_SAMPLED_SENTIMENT_EXECUTION || !fits(run.model_work, work)
                    || engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(HostedError::Limits("sampled sentiment completion accounting"));
                }
                if let Some(cause) = crate::native_engine::decode::DecodeStepControl::checkpoint(control, 0) {
                    return Err(HostedError::SampledSentiment(Int8SampledSentimentError::from(
                        crate::tasks::sentiment::sampled::SampledSentimentError::Cancelled(cause))));
                }
                Ok(run)
            })?;
            drop(engine);
            drop(input);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

fn execution_budget(work: Int8Work, kv_bytes: u64, limits: Int8SampledSentimentHostLimits)
    -> Int8JsonBudget {
    Int8JsonBudget { native: Int8RunBudget::exact(work),
        json: JsonWorkBudget { max_forward_positions: work.forward_positions,
            max_projected_logits: work.projected_logits, max_kv_bytes: kv_bytes,
            max_total_mask_node_visits: limits.max_mask_node_visits,
            mask_limits: limits.mask_limits } }
}
fn fits(work: Int8Work, cap: Int8Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn validate(limits: Int8SampledSentimentHostLimits, task: TaskBudget, context: usize,
    required: Int8MemoryRequirement, work: Int8Work, preparation_bytes: u64)
    -> Result<u64, HostedError> {
    limits.native.run.validate()?;
    task.validate().map_err(|_| HostedError::Limits("sampled sentiment task budget"))?;
    let sampler = json_sampling_workspace_bytes()
        .map_err(Int8SampledSentimentError::from).map_err(HostedError::SampledSentiment)?;
    if preparation_bytes == 0 || preparation_bytes > limits.preparation_reserve_bytes
        || context == 0 || context > limits.native.context_tokens
        || required.kv_bytes > task.max_kv_bytes || limits.native.allocator_reserve_bytes == 0
        || work.forward_positions == 0 || work.projected_logits == 0
        || !fits(work, limits.max_model_work) || sampler > limits.max_sampler_bytes
        || limits.mask_limits.max_trie_node_visits == 0
        || limits.mask_limits.checkpoint_interval_nodes == 0 || limits.max_mask_node_visits == 0 {
        return Err(HostedError::Limits("sampled sentiment whole-bundle admission"));
    }
    sum(&[required.rope_bytes, required.scratch_payload_bound,
        limits.native.allocator_reserve_bytes, sampler])
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Int8SampledSentimentHostLimits, TaskBudget, Int8MemoryRequirement, Int8Work) {
        let required = Int8MemoryRequirement::for_context(128).unwrap();
        let one = crate::native_engine::constrained_int8::planned_work(32, 64).unwrap();
        let work = one.checked_add(one).unwrap();
        let limits = Int8SampledSentimentHostLimits {
            native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 1 << 20,
                run: RunLimits { max_elapsed: Duration::from_secs(30),
                    max_checkpoints: 1000, cleanup_reserve_bytes: 65_536 } },
            preparation_reserve_bytes: 1 << 20, max_model_work: work,
            max_sampler_bytes: json_sampling_workspace_bytes().unwrap(),
            mask_limits: MaskWorkLimits { max_trie_node_visits: 1000, checkpoint_interval_nodes: 256 },
            max_mask_node_visits: 128_000,
        };
        let task = TaskBudget { max_input_tokens: 64, max_output_tokens: 128,
            max_output_bytes: 65_536, max_grammar_states: 4096, max_kv_bytes: required.kv_bytes };
        (limits, task, required, work)
    }
    #[test]
    fn every_whole_bundle_resource_is_admitted_before_dispatch() {
        let (limits, task, required, work) = fixture();
        assert_eq!(validate(limits, task, 95, required, work, 1 << 20).unwrap(),
            required.rope_bytes + required.scratch_payload_bound + (1 << 20)
                + json_sampling_workspace_bytes().unwrap());
        for axis in 0..10 {
            let mut limits = limits;
            match axis {
                0 => limits.preparation_reserve_bytes -= 1,
                1 => limits.max_model_work.forward_positions -= 1,
                2 => limits.max_model_work.projected_logits -= 1,
                3 => limits.max_model_work.attention_pairs -= 1,
                4 => limits.max_model_work.projections.dot_products -= 1,
                5 => limits.max_model_work.projections.multiply_accumulates -= 1,
                6 => limits.max_sampler_bytes -= 1,
                7 => limits.mask_limits.max_trie_node_visits = 0,
                8 => limits.mask_limits.checkpoint_interval_nodes = 0,
                _ => limits.max_mask_node_visits = 0,
            }
            assert!(validate(limits, task, 95, required, work, 1 << 20).is_err());
        }
    }
    #[test]
    fn independent_samples_share_capacity_but_keep_aggregate_work() {
        let (limits, mut task, required, work) = fixture();
        assert!(work.forward_positions > limits.native.context_tokens as u64);
        validate(limits, task, 95, required, work, 1 << 20).unwrap();
        assert!(validate(limits, task, 129, required, work, 1 << 20).is_err());
        task.max_kv_bytes -= 1;
        assert!(validate(limits, task, 95, required, work, 1 << 20).is_err());
    }
    #[test]
    fn sampled_task_errors_preserve_cancellation_without_private_diagnostics() {
        let cause = crate::native_engine::decode::DecodeCancellationKind::Deadline;
        let error = HostedError::SampledSentiment(Int8SampledSentimentError::from(
            crate::tasks::sentiment::sampled::SampledSentimentError::Cancelled(cause)));
        assert_eq!(error.to_string(), "hosted native sampled sentiment failed");
        assert_eq!(format!("{error:?}"), error.to_string());
        assert!(error.source().is_some());
        let HostedError::SampledSentiment(inner) = error else { unreachable!() };
        assert_eq!(inner.cancellation(), Some(cause));
    }
    #[test]
    fn transferred_input_and_guarded_output_are_send() {
        fn send<T: Send + 'static>() {}
        send::<PreparedInt8SampledSentiment>();
        send::<Int8SampledSentimentRun>();
        send::<SampledSentimentVocabulary>();
    }
}
