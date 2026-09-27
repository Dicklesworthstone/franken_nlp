//! Live native token delivery with physical-completion and resource ownership.
//! Events are provisional. Only the returned, validated completion can authorize
//! a terminal success frame, and it is returned AFTER the hosted scope drains.
use super::*;
use crate::{native_engine::{decode::DecodeEventSink, strict_int8::Int8Work}, tasks::ir::TaskBudget};

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
}
impl<S> HostedChatStream<S> {
    pub fn result(&self) -> &Int8ChatResult { self.output.result() }
    /// Publish/verify a final frame only after this completion exists. A sink
    /// may fail here; earlier token frames are not a successful completed task.
    pub fn with_sink<T>(&mut self, publish: impl FnOnce(&Int8ChatResult, &mut S) -> T) -> T {
        publish(self.output.result(), &mut self.input.value.sink)
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
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.execution_identity())?;
        let retained_bytes = limits.retained_bytes()?;
        let required = requirements(limits.native)?;
        let work = prepared.planned_work();
        capacity(request_seq, limits.native.context_tokens, required.kv_bytes,
            *prepared.task_plan().ir().budget(), work, prepared.native_plan().sampler_bytes(), limits.max_sampler_bytes)?;
        let lease = self.resources().acquire_lease();
        // Keep the entire input aggregate: capture/drop of only its fields could
        // release a ledger charge before a queued sink or prompt is disposed.
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, retained_bytes)?,
            || Ok(StreamInput { prepared: Some(prepared), sink }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound,
                limits.max_sampler_bytes, limits.native.allocator_reserve_bytes])?)?;
        let task = input.value.prepared.as_ref().ok_or(HostedError::CompletionMissing)?.task_plan().ir().budget();
        let output = output_claim(&lease, task.max_output_bytes, u64::from(task.max_output_tokens))?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            let prepared = input.value.prepared.take().ok_or(HostedError::CompletionMissing)?;
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let result = prepared.execute_with_sink(prepared.execution_identity(), &mut engine.value,
                request_seq, Int8GenerationBudget { native: Int8RunBudget::exact(work),
                    max_kv_bytes: required.kv_bytes, max_sampler_bytes: limits.max_sampler_bytes },
                &mut input.value.sink, control).map_err(HostedError::Chat)?;
            // Independent UTF-8/control/work/result validation has completed.
            // Still NO terminal success: dispatch owns cancellation and drain.
            drop(engine);
            drop(prepared);
            let committed = output.commit()?;
            drop(lease);
            Ok(HostedChatStream { output: GuardedOutput::new(result, committed), input })
        })
    }
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
        send::<StreamInput<Sink>>(); send::<HostedChatStream<Sink>>();
    }
}
