//! Live INT4 token delivery on the existing owned native invocation.
//! Tokens remain provisional until independent task validation and physical
//! drain finish. The completed result keeps the same sink and its memory charge.

use super::*;
use crate::native_engine::{decode::DecodeEventSink, strict_int4::prefill::Int4PrefillLimits};

struct StreamInput<S> { prepared: Option<PreparedInt4Chat>, sink: S }

/// A validated INT4 completion and the sink that received its provisional
/// events. Output, preparation and sink reservations remain live while the
/// caller reconciles and publishes its final frame. Neither storage can be
/// extracted without its resource guard.
pub struct HostedInt4ChatStream<S> {
    output: HostedOutput<Int4ChatResult>,
    input: Charged<StreamInput<S>>,
}
impl<S> HostedInt4ChatStream<S> {
    pub fn result(&self) -> &Int4ChatResult { self.output.result() }

    /// Called after native drain and the host's final cancellation decision.
    /// Publication may still fail; provisional events alone are never success.
    pub fn with_sink<T>(&mut self, publish: impl FnOnce(&Int4ChatResult, &mut S) -> T) -> T {
        publish(self.output.result(), &mut self.input.value.sink)
    }
}

impl NlpEngine {
    /// Generate/chat with live two-phase token events from a resident INT4
    /// model. The complete sink and prepared request move into the physical
    /// invocation, retaining their charges even if its async wrapper cancels.
    ///
    /// A reserve/permit failure aborts native execution without retrying the
    /// writer. A sink must not publish terminal success from a token callback:
    /// task finalization, cancellation and physical drain may still fail.
    /// Blocking callbacks backpressure the producer and cannot be preempted.
    /// This API adds no writer thread, detached task or independent runtime.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_chat_stream<S>(&self, model: &ResidentInt4, prepared: PreparedInt4Chat,
        request_seq: u64, limits: ChatStreamLimits, kernel: PackedInt4Kernel, sink: S,
        cancellation: CancellationToken) -> Result<HostedInt4ChatStream<S>, HostedError>
    where S: DecodeEventSink + Send + 'static {
        self.run_int4_chat_stream(model, prepared, request_seq, limits, None, kernel, sink, cancellation)
    }

    /// The same streamed task with explicitly admitted layer-major prompt
    /// execution. Additional prefill workspace is charged separately; seed,
    /// semantic identity, generation and final task validation are unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_chat_stream_layer_major<S>(&self, model: &ResidentInt4, prepared: PreparedInt4Chat,
        request_seq: u64, limits: ChatStreamLimits, prefill: Int4PrefillLimits, kernel: PackedInt4Kernel,
        sink: S, cancellation: CancellationToken) -> Result<HostedInt4ChatStream<S>, HostedError>
    where S: DecodeEventSink + Send + 'static {
        self.run_int4_chat_stream(model, prepared, request_seq, limits, Some(prefill), kernel, sink, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_chat_stream<S>(&self, model: &ResidentInt4, prepared: PreparedInt4Chat,
        request_seq: u64, limits: ChatStreamLimits, prefill: Option<Int4PrefillLimits>,
        kernel: PackedInt4Kernel, sink: S, cancellation: CancellationToken)
        -> Result<HostedInt4ChatStream<S>, HostedError>
    where S: DecodeEventSink + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let rebound = bind_candidate_model(prepared.execution_identity().clone(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        if prepared.native_plan().group_width() != model.model_identity().group_width()
            || &rebound != prepared.execution_identity() {
            return Err(HostedError::ModelIdentity);
        }
        let retained = retained_bytes(limits)?;
        let required = native_requirements(&prepared, request_seq, limits.native, limits.max_sampler_bytes)?;
        let scratch = workspace_bytes(required, limits, prefill)?;
        let work = prepared.planned_work();
        let task = *prepared.task_plan().ir().budget();
        let lease = self.resources().acquire_lease();
        // Capture the complete aggregate, including its guard, on both started
        // and discarded-while-queued paths. Disjoint field captures could drop
        // a reservation before the sink or its preparation storage is disposed.
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, retained)?,
            || Ok(StreamInput { prepared: Some(prepared), sink }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch)?;
        let output = output_claim(&lease, task.max_output_bytes, u64::from(task.max_output_tokens))?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            let prepared = input.value.prepared.take().ok_or(HostedError::CompletionMissing)?;
            let mut engine = allocate_native(kv, workspace, || {
                StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens,
                    Int4MemoryBudget { max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                        max_scratch_payload_bytes: required.scratch_payload_bound })
                    .map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let budget = Int4GenerationBudget { native: Int4RunBudget::exact(work),
                max_kv_bytes: required.kv_bytes, max_sampler_bytes: limits.max_sampler_bytes };
            let result = match prefill {
                None => prepared.execute_with_sink(prepared.execution_identity(), &mut engine.value,
                    request_seq, budget, &mut input.value.sink, control),
                Some(prefill) => prepared.execute_layer_major_with_sink(prepared.execution_identity(), &mut engine.value,
                    request_seq, budget, prefill, &mut input.value.sink, control),
            }.map_err(chat_error)?;
            // The task's independent UTF-8/control/work/output validation has
            // succeeded. Dispatch still owns final cancellation and full join;
            // only its successful return exposes this guarded completion.
            drop(engine);
            drop(prepared);
            let committed = output.commit()?;
            drop(lease);
            Ok(HostedInt4ChatStream { output: GuardedOutput::new(result, committed), input })
        })
    }
}

fn retained_bytes(limits: ChatStreamLimits) -> Result<u64, HostedError> {
    limits.native.run.validate()?;
    if limits.max_sampler_bytes == 0 || limits.preparation_reserve_bytes == 0 || limits.sink_reserve_bytes == 0 {
        return Err(HostedError::Limits("explicit INT4 stream sampler/preparation/sink reservations required"));
    }
    sum(&[limits.preparation_reserve_bytes, limits.sink_reserve_bytes])
}

fn workspace_bytes(required: Int4MemoryRequirement, limits: ChatStreamLimits,
    prefill: Option<Int4PrefillLimits>) -> Result<u64, HostedError> {
    let extra = prefill.map(Int4PrefillLimits::validate).transpose().map_err(native_error)?.unwrap_or(0);
    sum(&[required.rope_bytes, required.scratch_payload_bound, limits.max_sampler_bytes,
        limits.native.allocator_reserve_bytes, extra])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_engine::{decode::{DecodeCancellationKind, DecodeTokenEvent},
        generation::grouped_int4::Int4GenerationError};

    fn limits() -> ChatStreamLimits {
        ChatStreamLimits {
            native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 64 << 20,
                run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 100,
                    cleanup_reserve_bytes: 65536 } },
            max_sampler_bytes: 32 << 20, preparation_reserve_bytes: 256 << 20, sink_reserve_bytes: 1 << 20,
        }
    }

    #[test]
    fn stream_retained_storage_rejects_missing_authority_and_checked_overflow() {
        assert_eq!(retained_bytes(limits()).unwrap(), 257 << 20);
        for axis in 0..7 {
            let mut limits = limits();
            match axis { 0 => limits.max_sampler_bytes = 0, 1 => limits.preparation_reserve_bytes = 0,
                2 => limits.sink_reserve_bytes = 0, 3 => limits.preparation_reserve_bytes = u64::MAX,
                4 => limits.native.run.max_elapsed = Duration::ZERO,
                5 => limits.native.run.max_checkpoints = 1, _ => limits.native.run.cleanup_reserve_bytes = 0 }
            assert!(retained_bytes(limits).is_err(), "axis {axis}");
        }
    }

    #[test]
    fn prefill_workspace_is_additional_for_every_context_and_group_partition() {
        for group in [1, 3, 128, MAX_MODEL_K] { for context in [1, 128, 2048] {
            let required = Int4MemoryRequirement::for_context(context, group).unwrap();
            let limits = limits();
            let sequential = workspace_bytes(required, limits, None).unwrap();
            assert_eq!(sequential, required.rope_bytes + required.scratch_payload_bound
                + limits.max_sampler_bytes + limits.native.allocator_reserve_bytes);
            for rows in [1, 3, 64] {
                let extra = Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
                let prefill = Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: extra };
                assert_eq!(workspace_bytes(required, limits, Some(prefill)).unwrap(), sequential + extra);
                assert_eq!(retained_bytes(limits).unwrap(), 257 << 20);
            }
        } }
    }

    #[test]
    fn malformed_prefill_and_workspace_overflow_fail_before_allocation() {
        let required = Int4MemoryRequirement::for_context(128, 128).unwrap();
        for rows in [0, 65, usize::MAX] {
            assert!(workspace_bytes(required, limits(), Some(Int4PrefillLimits {
                max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX })).is_err());
        }
        let extra = Int4PrefillLimits::required_extra_scratch_bytes(4).unwrap();
        assert!(workspace_bytes(required, limits(), Some(Int4PrefillLimits {
            max_batch_rows: 4, max_extra_scratch_bytes: extra - 1 })).is_err());
        let mut overflowing = limits(); overflowing.max_sampler_bytes = u64::MAX;
        assert!(workspace_bytes(required, overflowing, None).is_err());
        overflowing = limits(); overflowing.native.allocator_reserve_bytes = u64::MAX;
        assert!(workspace_bytes(required, overflowing, None).is_err());
    }

    #[test]
    fn stream_native_cancellation_keeps_its_exact_typed_cause() {
        use DecodeCancellationKind::*;
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let error = chat_error(Int4ChatError::Native(Int4GenerationError::Native(StrictInt4Error::Cancelled(cause))));
            let HostedError::Int4(Int4HostError::Chat(error)) = error else { panic!("wrong cancellation wrapper") };
            assert_eq!(error.cancellation(), Some(cause));
        }
    }

    #[test]
    fn sink_and_guarded_completion_can_cross_the_owned_runtime_boundary() {
        struct Sink;
        impl DecodeEventSink for Sink {
            type Permit = (); type Error = &'static str;
            fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
            fn permit(&mut self, _: (), _: DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
        }
        fn send<T: Send + 'static>() {}
        send::<StreamInput<Sink>>(); send::<HostedInt4ChatStream<Sink>>();
    }
}
