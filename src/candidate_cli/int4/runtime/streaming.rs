//! The existing INT4 text planner and loader with owned provisional delivery.
use super::*;
use crate::{candidate_cli::stream::{Failure, StreamBounds, events::TokenSink},
    hosted::{ChatStreamLimits, HostedError, Int4HostError},
    native_engine::{generation::{GenerationError, grouped_int4::Int4GenerationError},
        strict_int4::{Int4ModelError, StrictInt4Error, weights::snapshot::Int4SnapshotError}},
    tasks::chat::{ChatError, grouped_int4::Int4ChatError}};

pub(in crate::candidate_cli::int4) fn execute<W: Write + Send + 'static>(task: Task,
    args: Int4Args, limits: Int4Limits, bounds: StreamBounds, input: &mut impl Read, output: W)
    -> Result<(), Failure> {
    if limits.minimum_memory_bytes > limits.memory_bytes || limits.kernel.kind() != args.kernel
        || !prefill::matches(&args, limits)? || bounds.tokens != args.max_new_tokens
        || bounds.content_bytes != args.max_output_bytes || bounds.capture_logprobs != args.logprobs {
        return Err(Failure::usage("stream_admitted_options_mismatch"));
    }
    let started = Instant::now();
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or(CandidateError::Timeout);
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: limits.memory_bytes,
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    // Covers initial private input and planner storage before the host owns
    // its moved prompt/sink. It remains live through terminal write and flush.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, limits.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    let text = if args.input.as_os_str() == "-" {
        read_input(input, args.max_input_bytes)?
    } else {
        let mut file = File::open(&args.input).map_err(|_| CandidateError::Input)?;
        read_input(&mut file, args.max_input_bytes)?
    };
    // Same role-order/control-boundary preflight, before any model access.
    let request = parse_input(task, text, args.max_input_bytes)?;
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = load_model_with_failure(&engine, &args, limits, remaining()?,
        cancellation.clone(), host_failure)?;
    remaining()?;
    let Prepared::Text(prepared) = prepare(Input::Text(request), &model, &args, limits,
        &mut DeadlineControl { started, deadline })? else {
        return Err(Failure::usage("stream_text_plan_required"));
    };
    remaining()?;
    if let Some(path) = &args.save_snapshot {
        // Exactly the buffered save contract: save after complete planning,
        // before native execution; preserve partial files on failed writes.
        let pin = engine.save_int4_snapshot(&model, path.clone(), limits.snapshot_budget(false)?,
            snapshot_run(&args, remaining()?), cancellation.clone()).map_err(host_failure)?;
        if pin != model.model_identity().snapshot_pin() {
            return Err(CandidateError::Identity.into());
        }
    }
    remaining()?;
    // The model supplies the actual independently recomputed INT4 identity.
    // The sink retains the full stop/minimum/seed policy of this exact plan.
    let sink = TokenSink::new_int4(output, task, bounds, model.model_identity(),
        prepared.native_plan().options())?;
    let stream_limits = ChatStreamLimits {
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining()?, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        max_sampler_bytes: SAMPLER_BYTES, preparation_reserve_bytes: limits.preparation_bytes,
        sink_reserve_bytes: bounds.sink_memory_bytes,
    };
    let mut completed = match limits.prefill {
        Some(prefill) => engine.execute_int4_chat_stream_layer_major(&model, prepared, 1,
            stream_limits, prefill, limits.kernel, sink, cancellation),
        None => engine.execute_int4_chat_stream(&model, prepared, 1, stream_limits,
            limits.kernel, sink, cancellation),
    }.map_err(host_failure)?;
    // Only the guarded completion returned AFTER physical drain may publish
    // success. A late deadline or final write failure leaves the stream incomplete.
    remaining()?;
    completed.with_sink(|result, sink| sink.finish_int4(result))
}

fn cancelled(cause: DecodeCancellationKind) -> Failure {
    use DecodeCancellationKind::*;
    let (exit, code) = match cause {
        Timeout | Deadline | PollQuota | CostBudget => (ErrorCode::BudgetOrTimeout, "stream_execution_budget"),
        User | ParentCancelled | Shutdown => (ErrorCode::Cancelled, "stream_cancelled"),
        ResourceUnavailable => (ErrorCode::AdmissionOrResourceLimit, "stream_resource_unavailable"),
        FailFast | LinkedExit => (ErrorCode::Generic, "stream_supervised_failure"),
        RaceLost => (ErrorCode::Generic, "stream_root_race_invariant"),
    };
    Failure { exit, code, cancellation: Some(cause) }
}

fn host_failure(error: HostedError) -> Failure {
    match error {
        HostedError::Stopped { stop, task_error } => {
            if matches!(stop.kind, DecodeCancellationKind::FailFast | DecodeCancellationKind::LinkedExit) {
                if let Some(error) = task_error {
                    let mut failure = host_failure(*error); failure.cancellation = Some(stop.kind); return failure;
                }
            }
            cancelled(stop.kind)
        }
        HostedError::Int4(Int4HostError::Chat(error)) => {
            if let Some(cause) = error.cancellation() { return cancelled(cause); }
            match error {
                Int4ChatError::Chat(ChatError::NoResult(_)) => Failure {
                    exit: ErrorCode::StructuredTaskNoResult, code: "stream_final_task_no_result", cancellation: None },
                Int4ChatError::Native(Int4GenerationError::Generation(GenerationError::Stream)) =>
                    Failure::output("stream_output_or_protocol"),
                _ => Failure::output("stream_native_task_failed"),
            }
        }
        HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause)))
            | HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause)))
            | HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))) => cancelled(cause),
        HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Budget))
            | HostedError::Int4(Int4HostError::Model(Int4ModelError::Budget)) => Failure {
                exit: ErrorCode::BudgetOrTimeout, code: "stream_model_budget", cancellation: None },
        HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Allocation))
            | HostedError::Int4(Int4HostError::Model(Int4ModelError::Allocation))
            | HostedError::Int4(Int4HostError::Native(StrictInt4Error::Allocation | StrictInt4Error::Memory))
            | HostedError::Limits(_) | HostedError::Reservation(_) | HostedError::Reentrant
            | HostedError::SingleCoordinatorRequired | HostedError::MissingRuntimeContext => Failure {
                exit: ErrorCode::AdmissionOrResourceLimit, code: "stream_host_admission", cancellation: None },
        HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Format | Int4SnapshotError::Integrity | Int4SnapshotError::Values))
            | HostedError::ResourceDomain | HostedError::ModelIdentity => Failure {
                exit: ErrorCode::ArtifactIntegrityOrFormatOrVersion, code: "stream_model_identity", cancellation: None },
        // Outer panic/join/scope failures remain failures even if a nested
        // task was cancelled. No error payload or private pathname is emitted.
        _ => Failure::output("stream_runtime_or_completion"),
    }
}

#[cfg(test)] mod tests;
