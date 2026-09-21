//! Single-request, process-hosted INT8 source tasks with real memory ownership.
use super::*;
use crate::{
    grammar::mask::MaskWorkLimits,
    native_engine::{constrained::JsonWorkBudget, strict_int8::Int8Work},
    tasks::{ir::TaskBudget, source_planning::quantized::{
        Int8SourceTaskRun, PreparedInt8SourceTask}},
};

/// Complete native and caller-priced preparation envelope. The preparation
/// charge must cover the transferred compiled source/grammar/passage metadata
/// and pinned vocabulary, including allocations retained by their Arcs. It is
/// a modeled reservation, not an allocator interceptor or an observed RSS cap.
#[derive(Clone, Copy, Debug)]
pub struct SourceLimits {
    pub native: NativeLimits,
    pub preparation_reserve_bytes: u64,
    pub mask_limits: MaskWorkLimits,
    pub max_mask_node_visits: u64,
}
struct SourceInput {
    prepared: PreparedInt8SourceTask,
    vocabulary: Arc<ExtractionVocabulary>,
}

impl NlpEngine {
    /// Execute NER, keyphrases, cited summarization or passage QA on the same
    /// resident current-candidate model and the one process-owned coordinator.
    /// All input captures and native storage drain before physical completion;
    /// result memory remains attached to the returned HostedOutput. The caller
    /// prepares the exact identity first; this method never repairs/re-tokenizes
    /// it, downloads weights or activates a catalog/default-model route.
    pub fn execute_int8_source(&self, model: &ResidentInt8, prepared: PreparedInt8SourceTask,
        vocabulary: Arc<ExtractionVocabulary>, limits: SourceLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8SourceTaskRun>, HostedError> {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.execution_identity())?;
        let required = requirements(limits.native)?;
        let work = prepared.planned_work();
        validate(limits, prepared.task_budget(), work, required.kv_bytes)?;
        let lease = self.resources().acquire_lease();
        let output = output_claim(&lease, prepared.max_result_bytes(),
            u64::from(prepared.task_budget().max_output_tokens))?;
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers,
            limits.preparation_reserve_bytes)?, || Ok(SourceInput { prepared, vocabulary }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            // Capture the entire Charged aggregate, not separately captured
            // fields whose drop order could release the charge before storage.
            let input = input;
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let budget = Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
                max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
                max_kv_bytes: required.kv_bytes, max_total_mask_node_visits: limits.max_mask_node_visits,
                mask_limits: limits.mask_limits,
            } };
            // allocate() preserves the output claim through native execution,
            // source validation, typed finalization and post-result commit. A
            // commit failure drops the result before returning its charge.
            let result = allocate(output, || input.value.prepared.execute_with_control(
                input.value.prepared.execution_identity(), &mut engine.value, &input.value.vocabulary, budget, control)
                .map_err(HostedError::Source))?;
            drop(engine);
            drop(input);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

fn validate(limits: SourceLimits, budget: TaskBudget, work: Int8Work, actual_kv_bytes: u64)
    -> Result<(), HostedError> {
    if limits.preparation_reserve_bytes == 0 || limits.mask_limits.max_trie_node_visits == 0
        || limits.mask_limits.checkpoint_interval_nodes == 0 || limits.max_mask_node_visits == 0 {
        return Err(HostedError::Limits("source preparation and mask limits"));
    }
    if work.forward_positions > limits.native.context_tokens as u64 || actual_kv_bytes > budget.max_kv_bytes {
        return Err(HostedError::Limits("source whole-context admission"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::source_planning::quantized::Int8SourceError;
    use crate::native_engine::{decode::DecodeCancellationKind, lmhead::NANBEIGE_VOCAB_SIZE};
    fn limits() -> SourceLimits {
        SourceLimits { native: NativeLimits { context_tokens: 64, allocator_reserve_bytes: 1 << 20,
            run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 10000, cleanup_reserve_bytes: 65536 } },
            preparation_reserve_bytes: 1 << 20,
            mask_limits: MaskWorkLimits { max_trie_node_visits: 10000, checkpoint_interval_nodes: 64 },
            max_mask_node_visits: 100000 }
    }
    fn budget() -> TaskBudget { TaskBudget { max_input_tokens: 64, max_output_tokens: 8,
        max_output_bytes: 65536, max_grammar_states: 8192, max_kv_bytes: 4096 } }
    fn work() -> Int8Work { Int8Work::for_sequence(0, 64, 8 * NANBEIGE_VOCAB_SIZE).unwrap() }
    #[test]
    fn exact_full_context_and_full_kv_reservation_fit() {
        validate(limits(), budget(), work(), 4096).unwrap();
        let mut l = limits(); l.native.context_tokens -= 1;
        assert!(validate(l, budget(), work(), 4096).is_err());
        assert!(validate(limits(), budget(), work(), 4097).is_err());
    }
    #[test]
    fn every_preparation_and_mask_axis_requires_nonzero_authority() {
        for axis in 0..4 {
            let mut l = limits();
            match axis { 0 => l.preparation_reserve_bytes = 0, 1 => l.mask_limits.max_trie_node_visits = 0,
                2 => l.mask_limits.checkpoint_interval_nodes = 0, _ => l.max_mask_node_visits = 0 }
            assert!(validate(l, budget(), work(), 4096).is_err());
        }
    }
    #[test]
    fn hosted_error_retains_typed_cancellation_without_displaying_nested_payloads() {
        let error = HostedError::Source(Int8SourceError::Cancelled(DecodeCancellationKind::Deadline));
        let HostedError::Source(ref original) = error else { unreachable!() };
        assert_eq!(original.cancellation(), Some(DecodeCancellationKind::Deadline));
        assert!(std::error::Error::source(&error).is_some());
        assert_eq!(format!("{error}"), "hosted native source task failed");
        assert_eq!(format!("{error:?}"), "hosted native source task failed");
    }
    #[test]
    fn preparation_capture_and_native_output_can_cross_the_owned_runtime_boundary() {
        fn send<T: Send + 'static>() {}
        send::<SourceInput>(); send::<Int8SourceTaskRun>();
    }
}
