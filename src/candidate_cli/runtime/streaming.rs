//! Same pinned preparation and native generation; only delivery is different.
use super::*;
use super::source_tasks::Session;
use crate::{candidate_cli::stream::{Failure, StreamBounds, events::TokenSink},
    hosted::{ChatStreamLimits, HostedError},
    native_engine::{decode::DecodeCancellationKind, generation::quantized::Int8GenerationError},
    tasks::chat::{ChatError, quantized::{PreparedInt8Chat, Int8ChatError}}};

/// Shared with the original buffered generate/chat path. In particular, stream
/// framing never changes the stable item id, sample index, seed, source bytes,
/// template/control policy, task ceiling or admitted native work.
pub(super) fn prepare(args: &CandidateArgs, limits: Limits, text: String,
    messages: Option<Vec<ChatMessage>>, facts: &ArtifactIdentity) -> Result<PreparedInt8Chat, CandidateError> {
    let identity = candidate_identity(facts)?;
    let controls = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let eos = controls.template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END)
        .map(|entry| entry.id).ok_or(CandidateError::Identity)?;
    let budget = TaskBudget {
        max_input_tokens: u32::try_from(limits.max_prompt_tokens).map_err(|_| CandidateError::Arguments)?,
        max_output_tokens: u32::try_from(args.max_new_tokens).map_err(|_| CandidateError::Arguments)?,
        max_output_bytes: limits.result_bytes as u64, max_grammar_states: 1, max_kv_bytes: limits.kv_bytes,
    };
    let planner = Int8ChatPlanner::pinned(controls.template_controls(), eos, identity, budget,
        ChatLimits { max_messages: 128, max_message_bytes: args.max_input_bytes,
            max_total_message_bytes: args.max_input_bytes, generation: args.generation_limits(limits) })
        .map_err(|_| CandidateError::Planning)?;
    let options = args.options(eos)?;
    let prepared = match messages {
        Some(messages) => planner.plan_chat(&ChatRequest { item_id: "cli".to_owned(), sample_index: 0,
            messages, generation: options, budget }),
        None => planner.plan_generate(&GenerateRequest { item_id: "cli".to_owned(), sample_index: 0,
            prompt: text, generation: options, budget }),
    }.map_err(|_| CandidateError::Planning)?;
    if prepared.planned_work().forward_positions > args.context_tokens as u64 { return Err(CandidateError::Planning); }
    Ok(prepared)
}

pub(in crate::candidate_cli) fn execute<W: Write + Send + 'static>(task: Task, args: CandidateArgs,
    limits: Limits, bounds: StreamBounds, input: &mut impl Read, output: W) -> Result<(), Failure> {
    // Declared first: initial input, token verification buffers, metadata and
    // planner allocations remain covered before transfer to the charged host.
    let session = Session::new(&args, limits)?;
    let text = session.read(&args.input, input, args.max_input_bytes)?;
    let messages = match task {
        Task::Chat => Some(parse_messages(&text, args.max_input_bytes)?), Task::Generate => None,
    };
    let facts = session.facts(&args)?;
    let prepared = prepare(&args, limits, text, messages, &facts)?;
    session.remaining()?;
    let eos = prepared.native_plan().options().eos_token_ids.first().copied()
        .ok_or_else(|| Failure::usage("stream_eos_contract"))?;
    let sink = TokenSink::new(output, task, bounds, &facts, eos, args.seed.clone())?;
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let stream_limits = ChatStreamLimits { native: session.native(&args)?, max_sampler_bytes: SAMPLER_BYTES,
        preparation_reserve_bytes: limits.preparation_bytes, sink_reserve_bytes: bounds.sink_memory_bytes };
    let mut completed = match args.policy.prefill()? {
        Some(prefill) => session.engine.execute_int8_chat_stream_layer_major(&model, prepared, 1,
            stream_limits, prefill, sink, cancellation),
        None => session.engine.execute_int8_chat_stream(&model, prepared, 1, stream_limits, sink, cancellation),
    }.map_err(host_failure)?;
    // No native callback can mint the terminal frame. This point is AFTER the
    // physical scope joins and dispatch's cancellation/stop precedence resolves.
    session.remaining()?;
    completed.with_sink(|result, sink| sink.finish(result))
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
        HostedError::Chat(error) => {
            if let Some(cause) = error.cancellation() { return cancelled(cause); }
            match error {
                Int8ChatError::Chat(ChatError::NoResult(_)) => Failure {
                    exit: ErrorCode::StructuredTaskNoResult, code: "stream_final_task_no_result", cancellation: None },
                Int8ChatError::Native(Int8GenerationError::Generation(
                    crate::native_engine::generation::GenerationError::Stream)) => Failure::output("stream_output_or_protocol"),
                _ => Failure::output("stream_native_task_failed"),
            }
        }
        HostedError::ResourceDomain | HostedError::ModelIdentity => Failure {
            exit: ErrorCode::ArtifactIntegrityOrFormatOrVersion, code: "stream_model_identity", cancellation: None },
        HostedError::Limits(_) | HostedError::Reservation(_) | HostedError::Reentrant
            | HostedError::SingleCoordinatorRequired | HostedError::MissingRuntimeContext => Failure {
                exit: ErrorCode::AdmissionOrResourceLimit, code: "stream_host_admission", cancellation: None },
        // Panic/join/scope failure is never ordinary user cancellation or a
        // successful task with a shortened token list.
        _ => Failure::output("stream_runtime_or_completion"),
    }
}

#[cfg(test)] mod tests;
