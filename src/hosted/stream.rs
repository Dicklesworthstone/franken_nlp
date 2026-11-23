//! Live native token delivery with physical-completion and resource ownership.
//! Events are provisional. Only the returned, validated completion can authorize
//! a terminal success frame, and it is returned AFTER the hosted scope drains.
use super::*;
use crate::{native_engine::{decode::DecodeEventSink, simd::Int8Kernel,
    strict_int8::{Int8Work, prefill::Int8PrefillLimits}}, tasks::ir::TaskBudget};

/// Caller-priced retained preparation and sink memory, not an RSS guarantee.
/// A channel/sink must bound its own buffers and include them in this reserve.
/// Slow blocking sinks backpressure the native producer; they are not preempted.
#[derive(Clone, Copy, Debug)]
pub struct ChatStreamLimits {
    pub native: NativeLimits,
    pub max_sampler_bytes: u64,
    /// Pinned tokenizer, prompt/TaskIR graphs, options and allocator headroom.
    pub preparation_reserve_bytes: u64,
    /// Sink, transport staging, retained token/byte verification and IO buffers.
    pub sink_reserve_bytes: u64,
}
impl ChatStreamLimits {
    fn retained_bytes(self) -> Result<u64, HostedError> {
        self.native.run.validate()?;
        if self.max_sampler_bytes == 0 || self.preparation_reserve_bytes == 0 || self.sink_reserve_bytes == 0 {
            return Err(HostedError::Limits("explicit stream sampler/preparation/sink reservations required"));
        }
        sum(&[self.preparation_reserve_bytes, self.sink_reserve_bytes])
    }
}

struct StreamInput<S> { prepared: Option<PreparedInt8Chat>, sink: S }

/// The completed task and the SAME sink that accepted its provisional events.
/// Output and sink/preparation charges remain live during terminal publication.
/// No unguarded owned-result or sink extraction is exposed. The prompt is dropped
/// inside the physical invocation and is not retained in this completion.
pub struct HostedChatStream<S> {
    output: HostedOutput<Int8ChatResult>,
    input: Charged<StreamInput<S>>,
    compute: Option<Int8ComputeReceipt>,
}
impl<S> HostedChatStream<S> {
    pub fn result(&self) -> &Int8ChatResult { self.output.result() }
    /// Completed physical work, available only after every worker joined and
    /// the original cancellation allowance accepted final publication. Serial
    /// streams retain their existing semantic result and report no team.
    pub fn compute(&self) -> Option<&Int8ComputeReceipt> { self.compute.as_ref() }
    /// Publish/verify a final frame only after this completion exists. A sink
    /// may fail here; earlier token frames are not a successful completed task.
    pub fn with_sink<T>(&mut self, publish: impl FnOnce(&Int8ChatResult, &mut S) -> T) -> T {
        publish(self.output.result(), &mut self.input.value.sink)
    }
}

/// The dispatcher owns this aggregate OUTSIDE its scoped work callback.
/// Cancellation, a failed sink and panic cannot release input/sink or native
/// reservations while a child is still live. Successful completion moves the
/// charged sink and output into the post-join result; errors leave them here.
struct StreamAdmission<S> {
    input: Option<Charged<StreamInput<S>>>,
    kv: Option<Pending>, scratch: Option<Pending>, output: Option<Pending>,
    _kv: Option<CommittedMemory>, _scratch: Option<CommittedMemory>, _lease: EngineLease,
}
impl<S> StreamAdmission<S> {
    fn commit_native(&mut self) -> Result<(), HostedError> {
        self._kv = Some(self.kv.take().ok_or(HostedError::CompletionMissing)?.commit()?);
        self._scratch = Some(self.scratch.take().ok_or(HostedError::CompletionMissing)?.commit()?);
        Ok(())
    }
}

impl NlpEngine {
    /// Execute a prepared generation/chat request with live two-phase events on
    /// the existing process-owned blocking pool and native scoped invocation.
    /// The actual resident model, identity, full KV and output admission remain
    /// mandatory; callers cannot replace them with a fabricated admission guard.
    ///
    /// The sink moves into the physical invocation, including queued-cancel
    /// paths. A failed reserve/permit stops native execution without retry; the
    /// native driver clears its KV and preserves poison/cancellation semantics.
    /// An error may follow already published provisional tokens. Do not emit a
    /// success terminal from inside the sink: dispatch can still cancel/fail
    /// after the last token or during final task validation/physical drain.
    ///
    /// This synchronous API never calls a sink on a detached task or creates a
    /// separate writer thread. Re-entry from the sink is rejected by the host.
    /// An arbitrary blocking writer cannot be safely interrupted mid-write.
    pub fn execute_int8_chat_stream<S>(&self, model: &ResidentInt8, prepared: PreparedInt8Chat,
        request_seq: u64, limits: ChatStreamLimits, sink: S, cancellation: CancellationToken)
        -> Result<HostedChatStream<S>, HostedError>
    where S: DecodeEventSink + Send + 'static {
        self.run_int8_chat_stream(model, prepared, request_seq, limits, None, None, sink, cancellation)
    }

    /// Explicit candidate layer-major prefill with the ordinary streaming
    /// protocol. Extra native scratch is admitted independently of sink and
    /// preparation storage, and remains charged through physical drain.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_chat_stream_layer_major<S>(&self, model: &ResidentInt8, prepared: PreparedInt8Chat,
        request_seq: u64, limits: ChatStreamLimits, prefill: Int8PrefillLimits, sink: S,
        cancellation: CancellationToken) -> Result<HostedChatStream<S>, HostedError>
    where S: DecodeEventSink + Send + 'static {
        self.run_int8_chat_stream(model, prepared, request_seq, limits, Some(prefill), None, sink, cancellation)
    }

    /// Stream a complete request on one persistent team of 2..=64 compute
    /// threads, including the coordinator. Complete output rows use the exact
    /// scalar integer kernel and retain their ordinary BF16 boundaries. One
    /// team serves every prompt position and every generated token.
    ///
    /// The installed resource host must admit the requested width. Worker
    /// storage is additional to full KV, coordinator, sampler and sink memory.
    /// The same bounded sink applies backpressure without a writer thread or
    /// detached work. Token events remain provisional until the team drains.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_chat_stream_parallel<S>(&self, model: &ResidentInt8, prepared: PreparedInt8Chat,
        request_seq: u64, limits: ChatStreamLimits, compute_threads: usize, sink: S,
        cancellation: CancellationToken) -> Result<HostedChatStream<S>, HostedError>
    where S: DecodeEventSink + Send + 'static {
        self.run_int8_chat_stream(model, prepared, request_seq, limits, None,
            Some(compute_threads), sink, cancellation)
    }

    /// Reuse one fixed team across bounded layer-major prompt morsels and
    /// streamed decode. Coordinator prompt storage and each child's batch
    /// capacity are both charged before execution. The original work budget,
    /// deadline, sink protocol and independent task finalizer remain in force
    /// throughout prefill, decode and physical join.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_chat_stream_layer_major_parallel<S>(&self, model: &ResidentInt8,
        prepared: PreparedInt8Chat, request_seq: u64, limits: ChatStreamLimits,
        prefill: Int8PrefillLimits, compute_threads: usize, sink: S,
        cancellation: CancellationToken) -> Result<HostedChatStream<S>, HostedError>
    where S: DecodeEventSink + Send + 'static {
        self.run_int8_chat_stream(model, prepared, request_seq, limits, Some(prefill),
            Some(compute_threads), sink, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int8_chat_stream<S>(&self, model: &ResidentInt8, prepared: PreparedInt8Chat,
        request_seq: u64, limits: ChatStreamLimits, prefill: Option<Int8PrefillLimits>,
        compute_threads: Option<usize>, sink: S,
        cancellation: CancellationToken) -> Result<HostedChatStream<S>, HostedError>
    where S: DecodeEventSink + Send + 'static {
        match compute_threads {
            None => dispatch::preflight(self, limits.native.run)?,
            Some(width) => dispatch::preflight_int8_team(self, limits.native.run, width)?,
        }
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.execution_identity())?;
        let retained_bytes = limits.retained_bytes()?;
        let required = requirements(limits.native)?;
        let scratch_bytes = workspace_bytes(required, limits, prefill, compute_threads)?;
        let work = prepared.planned_work();
        capacity(request_seq, limits.native.context_tokens, required.kv_bytes,
            *prepared.task_plan().ir().budget(), work, prepared.native_plan().sampler_bytes(), limits.max_sampler_bytes)?;
        let lease = self.resources().acquire_lease();
        // Keep the entire input aggregate: capture/drop of only its fields could
        // release a ledger charge before a queued sink or prompt is disposed.
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, retained_bytes)?,
            || Ok(StreamInput { prepared: Some(prepared), sink }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
        let task = input.value.prepared.as_ref().ok_or(HostedError::CompletionMissing)?.task_plan().ir().budget();
        let output = output_claim(&lease, task.max_output_bytes, u64::from(task.max_output_tokens))?;
        let model = model.clone();
        if let Some(width) = compute_threads {
            let admission = StreamAdmission { input: Some(input), kv: Some(kv), scratch: Some(scratch),
                output: Some(output), _kv: None, _scratch: None, _lease: lease };
            let kernel = Int8Kernel::scalar();
            let max_batch_rows = prefill.map_or(1, |limits| limits.max_batch_rows);
            return dispatch::run_int8_prefill_team(self, limits.native.run, cancellation, model,
                width, kernel, max_batch_rows, admission, move |model, admission, control, team| {
                    let mut engine = model.engine(limits.native.context_tokens, memory_budget(required))
                        .map_err(HostedError::Model)?.with_kernel(kernel)
                        .with_projection_executor(team).map_err(HostedError::Native)?;
                    admission.commit_native()?;
                    // Borrow the charged aggregate until success. An early
                    // error or unwinding sink leaves ownership in admission
                    // until the dispatcher has joined every physical child.
                    let input = admission.input.as_mut().ok_or(HostedError::CompletionMissing)?;
                    let prepared = input.value.prepared.take().ok_or(HostedError::CompletionMissing)?;
                    let budget = Int8GenerationBudget { native: Int8RunBudget::exact(work),
                        max_kv_bytes: required.kv_bytes, max_sampler_bytes: limits.max_sampler_bytes };
                    let result = match prefill {
                        Some(prefill) => prepared.execute_layer_major_with_sink(prepared.execution_identity(), &mut engine,
                            request_seq, budget, prefill, &mut input.value.sink, control),
                        None => prepared.execute_with_sink(prepared.execution_identity(), &mut engine,
                            request_seq, budget, &mut input.value.sink, control),
                    }.map_err(HostedError::Chat)?;
                    drop(engine);
                    drop(prepared);
                    let compute = Some(team.stats().into());
                    let committed = admission.output.take().ok_or(HostedError::CompletionMissing)?.commit()?;
                    let input = admission.input.take().ok_or(HostedError::CompletionMissing)?;
                    // This guarded value remains inside the dispatcher until
                    // its final post-join cancellation check accepts it.
                    Ok(HostedChatStream { output: GuardedOutput::new(result, committed), input, compute })
                });
        }
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            let prepared = input.value.prepared.take().ok_or(HostedError::CompletionMissing)?;
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let budget = Int8GenerationBudget { native: Int8RunBudget::exact(work),
                max_kv_bytes: required.kv_bytes, max_sampler_bytes: limits.max_sampler_bytes };
            let result = match prefill {
                Some(prefill) => prepared.execute_layer_major_with_sink(prepared.execution_identity(), &mut engine.value,
                    request_seq, budget, prefill, &mut input.value.sink, control),
                None => prepared.execute_with_sink(prepared.execution_identity(), &mut engine.value,
                    request_seq, budget, &mut input.value.sink, control),
            }.map_err(HostedError::Chat)?;
            // Independent UTF-8/control/work/result validation has completed.
            // Still NO terminal success: dispatch owns cancellation and drain.
            drop(engine);
            drop(prepared);
            let committed = output.commit()?;
            drop(lease);
            Ok(HostedChatStream { output: GuardedOutput::new(result, committed), input, compute: None })
        })
    }
}

fn prefill_extra(prefill: Option<Int8PrefillLimits>) -> Result<u64, HostedError> {
    prefill.map(Int8PrefillLimits::validate).transpose().map(|bytes| bytes.unwrap_or(0)).map_err(HostedError::Native)
}

/// Return coordinator scratch, checking the simultaneous child storage before
/// reservation. The dispatcher separately owns that child charge through join.
fn workspace_bytes(required: Int8MemoryRequirement, limits: ChatStreamLimits,
    prefill: Option<Int8PrefillLimits>, compute_threads: Option<usize>) -> Result<u64, HostedError> {
    let scratch = sum(&[required.rope_bytes, required.scratch_payload_bound,
        limits.max_sampler_bytes, limits.native.allocator_reserve_bytes, prefill_extra(prefill)?])?;
    if let Some(width) = compute_threads {
        let children = dispatch::int8_prefill_team_workspace_bytes(width,
            prefill.map_or(1, |limits| limits.max_batch_rows))?;
        sum(&[scratch, children])?;
    }
    Ok(scratch)
}

fn capacity(sequence: u64, context: usize, kv: u64, task: TaskBudget, work: Int8Work,
    sampler: u64, max_sampler: u64) -> Result<(), HostedError> {
    task.validate().map_err(|_| HostedError::Limits("stream task budget"))?;
    if sequence == 0 || context == 0 || kv == 0 || kv > task.max_kv_bytes
        || work.forward_positions == 0 || work.forward_positions > context as u64
        || sampler == 0 || sampler > max_sampler {
        return Err(HostedError::Limits("stream full KV/context/sampler admission"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_engine::decode::DecodeTokenEvent;
    fn task() -> TaskBudget {
        TaskBudget { max_input_tokens: 32, max_output_tokens: 8, max_output_bytes: 4096,
            max_grammar_states: 1, max_kv_bytes: 1 << 20 }
    }
    fn native() -> NativeLimits {
        NativeLimits { context_tokens: 64, allocator_reserve_bytes: 4096,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 1000, cleanup_reserve_bytes: 4096 } }
    }
    fn limits() -> ChatStreamLimits {
        ChatStreamLimits { native: native(), max_sampler_bytes: 32 << 20,
            preparation_reserve_bytes: 256 << 20, sink_reserve_bytes: 1 << 20 }
    }
    #[test]
    fn zero_or_overflowed_retained_storage_cannot_enter_a_stream() {
        assert_eq!(limits().retained_bytes().unwrap(), 257 << 20);
        for axis in 0..6 {
            let mut l = limits();
            match axis { 0 => l.preparation_reserve_bytes = 0, 1 => l.sink_reserve_bytes = 0,
                2 => l.max_sampler_bytes = 0, 3 => l.preparation_reserve_bytes = u64::MAX,
                4 => l.native.run.max_checkpoints = 1, _ => l.native.run.max_elapsed = Duration::ZERO }
            assert!(l.retained_bytes().is_err(), "axis {axis}");
        }
    }
    #[test]
    fn whole_resident_kv_context_and_sampler_are_required_even_for_short_prompts() {
        let w = Int8Work::for_sequence(0, 40, 166144).unwrap();
        capacity(1, 64, 1 << 20, task(), w, 32 << 20, 32 << 20).unwrap();
        assert!(capacity(0, 64, 1 << 20, task(), w, 32 << 20, 32 << 20).is_err());
        assert!(capacity(1, 39, 1 << 20, task(), w, 32 << 20, 32 << 20).is_err());
        assert!(capacity(1, 64, (1 << 20) + 1, task(), w, 32 << 20, 32 << 20).is_err());
        assert!(capacity(1, 64, 0, task(), w, 32 << 20, 32 << 20).is_err());
        assert!(capacity(1, 64, 1 << 20, task(), w, 32 << 20, (32 << 20) - 1).is_err());
        assert!(capacity(1, 64, 1 << 20, task(), w, 0, 32 << 20).is_err());
    }
    #[test]
    fn sink_input_and_guarded_completion_cross_the_owned_runtime_boundary() {
        struct Sink;
        impl DecodeEventSink for Sink {
            type Permit = (); type Error = &'static str;
            fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
            fn permit(&mut self, _: (), _: DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
        }
        fn send<T: Send + 'static>() {}
        send::<StreamInput<Sink>>(); send::<StreamAdmission<Sink>>(); send::<HostedChatStream<Sink>>();
        let _serial = NlpEngine::execute_int8_chat_stream::<Sink>;
        let _prefill = NlpEngine::execute_int8_chat_stream_layer_major::<Sink>;
        let _team = NlpEngine::execute_int8_chat_stream_parallel::<Sink>;
        let _prefill_team = NlpEngine::execute_int8_chat_stream_layer_major_parallel::<Sink>;
    }
    #[test]
    fn grouped_stream_scratch_is_additional_to_unchanged_sink_preparation_and_native_storage() {
        let required = requirements(native()).unwrap();
        let ordinary = sum(&[required.rope_bytes, required.scratch_payload_bound,
            limits().max_sampler_bytes, native().allocator_reserve_bytes]).unwrap();
        assert_eq!(prefill_extra(None).unwrap(), 0);
        for rows in [1, 4, 64] {
            let extra = Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
            let p = Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: extra };
            let charged = sum(&[ordinary, prefill_extra(Some(p)).unwrap()]).unwrap();
            assert_eq!(charged - ordinary, extra);
            assert_eq!(limits().retained_bytes().unwrap(), 257 << 20);
        }
    }
    #[test]
    fn invalid_or_underpriced_grouped_stream_cannot_reach_native_allocation() {
        for rows in [0, 65, usize::MAX] {
            assert!(prefill_extra(Some(Int8PrefillLimits { max_batch_rows: rows,
                max_extra_scratch_bytes: u64::MAX })).is_err());
        }
        let bytes = Int8PrefillLimits::required_extra_scratch_bytes(4).unwrap();
        assert!(prefill_extra(Some(Int8PrefillLimits { max_batch_rows: 4, max_extra_scratch_bytes: bytes - 1 })).is_err());
        assert!(sum(&[u64::MAX, bytes]).is_err());
    }

    #[test]
    fn stream_team_keeps_native_sampler_sink_and_prompt_workspace_as_separate_charges() {
        let required = requirements(native()).unwrap();
        let ordinary = required.rope_bytes + required.scratch_payload_bound
            + limits().max_sampler_bytes + native().allocator_reserve_bytes;
        for rows in [1, 3, 64] {
            let prefill = Int8PrefillLimits { max_batch_rows: rows,
                max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() };
            for schedule in [None, Some(prefill)] {
                let extra = prefill_extra(schedule).unwrap();
                assert_eq!(workspace_bytes(required, limits(), schedule, None).unwrap(), ordinary + extra);
                for width in [2, 3, 64] {
                    assert_eq!(workspace_bytes(required, limits(), schedule, Some(width)).unwrap(), ordinary + extra);
                    let children = dispatch::int8_prefill_team_workspace_bytes(width,
                        schedule.map_or(1, |p| p.max_batch_rows)).unwrap();
                    assert!(children > 0);
                    assert_eq!(limits().retained_bytes().unwrap(), 257 << 20);
                    assert_eq!(memory_budget(required).max_kv_bytes, required.kv_bytes);
                }
            }
        }
    }

    #[test]
    fn invalid_stream_team_width_and_batch_capacity_refuse_without_native_allocation() {
        let required = requirements(native()).unwrap();
        for width in [0, 1, 65, usize::MAX] {
            assert!(workspace_bytes(required, limits(), None, Some(width)).is_err());
        }
        for rows in [0, 65, usize::MAX] {
            assert!(workspace_bytes(required, limits(), Some(Int8PrefillLimits {
                max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX }), Some(3)).is_err());
        }
        let bytes = Int8PrefillLimits::required_extra_scratch_bytes(3).unwrap();
        assert!(workspace_bytes(required, limits(), Some(Int8PrefillLimits {
            max_batch_rows: 3, max_extra_scratch_bytes: bytes - 1 }), Some(3)).is_err());
    }

    #[test]
    fn separately_valid_coordinator_and_worker_quotes_cannot_overflow_the_simultaneous_peak() {
        let required = requirements(native()).unwrap();
        let prefill = Int8PrefillLimits { max_batch_rows: 3,
            max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(3).unwrap() };
        let children = dispatch::int8_prefill_team_workspace_bytes(4, 3).unwrap();
        let fixed = required.rope_bytes + required.scratch_payload_bound
            + limits().max_sampler_bytes + prefill.validate().unwrap();
        let mut exact = limits();
        exact.native.allocator_reserve_bytes = u64::MAX - fixed - children;
        assert_eq!(workspace_bytes(required, exact, Some(prefill), Some(4)).unwrap(), u64::MAX - children);
        exact.native.allocator_reserve_bytes += 1;
        assert!(workspace_bytes(required, exact, Some(prefill), None).is_ok());
        assert!(matches!(workspace_bytes(required, exact, Some(prefill), Some(4)),
            Err(HostedError::Limits("memory arithmetic"))));
    }

    #[test]
    fn stream_team_native_failures_keep_all_typed_cancellation_causes() {
        use crate::native_engine::{decode::DecodeCancellationKind::*, generation::quantized::Int8GenerationError};
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let error = HostedError::Chat(Int8ChatError::Native(Int8GenerationError::Native(
                StrictInt8Error::Cancelled(cause))));
            let HostedError::Chat(error) = error else { panic!("wrong native error boundary") };
            assert_eq!(error.cancellation(), Some(cause));
        }
    }
}

#[cfg(test)] mod model_tests;
