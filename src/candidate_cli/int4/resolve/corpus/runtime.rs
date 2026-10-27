//! Owned raw NDJSON delivery; the hosted completion alone writes the footer.
use super::*;
use std::{io::{BufRead, BufReader}, sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    corpus::{entities_int4::{Int4EntityError, prepare_int4_entities},
        entity_stream::EntityStreamError, native_resolve::grouped_int4::Int4ResolveError, resolve::ResolveError},
    hosted::{CancellationToken, HostedError, Int4HostError, NativeLimits, RunLimits,
        Int4EntityCorpusHostConfig, Int4EntityCorpusHostRequirements, Int4EntityCorpusError},
    native_engine::{decode::DecodeStepControl,
        strict_int4::{Int4ModelError, StrictInt4Error, weights::snapshot::Int4SnapshotError}},
    tasks::BuiltInTask,
};

struct DeadlineControl { started: Instant, deadline: Duration }
impl DecodeStepControl for DeadlineControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        (self.started.elapsed() >= self.deadline).then_some(DecodeCancellationKind::Deadline)
    }
}
fn native(args: &Int4Args, elapsed: Duration) -> NativeLimits {
    NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
        run: RunLimits { max_elapsed: elapsed, max_checkpoints: args.max_checkpoints,
            cleanup_reserve_bytes: CLEANUP_BYTES } }
}

pub(super) fn memory_floor(args: &Int4Args, command: &CorpusArgs, common: Int4Limits,
    transport: EntityStreamLimits, io_bytes: u64, task: &Int4EntityConfig) -> Result<u64, Failure> {
    let requirement = Int4EntityCorpusHostRequirements::for_config(task, transport,
        native(args, Duration::from_secs(args.timeout_seconds)), common.preparation_bytes,
        command.resolution.graph_bytes().map_err(candidate_failure)?, io_bytes, args.group_width)
        .map_err(host_failure)?;
    // Resident model allocation and the caller's policy/planner preparation
    // coexist with every host-owned parser, graph, output and native claim.
    sum(&[common.load.weight_payload_bytes, ALLOCATOR_BYTES, common.preparation_bytes,
        requirement.total_bytes().map_err(host_failure)?]).map_err(candidate_failure)
}

pub(super) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    args: Int4Args, command: CorpusArgs, _admitted: CorpusLimits, input: R, output: W,
) -> Result<(), Failure> {
    let started = Instant::now();
    let limits = command.validate(&args)?;
    let common = limits.common;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or_else(|| cancelled(DecodeCancellationKind::Deadline));
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: common.memory_bytes,
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().map_err(|_| candidate_failure(CandidateError::Runtime))?;
    let lease = engine.resources().acquire_lease();
    let _preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes)
        .map_err(|_| candidate_failure(CandidateError::Memory))?.commit()
        .map_err(|_| candidate_failure(CandidateError::Memory))?;
    remaining()?;
    // Fixed policy is decoded and admitted before any corpus or model read.
    // A record cannot alter NER types, margins, native budgets or task identity.
    let mut policy_file = crate::local_io::open_document(&command.stream.policy)
        .map_err(|_| input_failure("entity_corpus_policy_input"))?;
    let text = read_input(&mut policy_file, POLICY_BYTES).map_err(candidate_failure)?;
    let policy = Policy::parse(&text)?;
    drop(text);
    drop(policy_file);
    let task = command.configuration(&args, common, policy)?;
    if memory_floor(&args, &command, common, limits.transport, limits.io_bytes, &task)? > common.memory_bytes {
        return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit,
            code: "entity_corpus_memory_floor", cancellation: None });
    }
    remaining()?;
    let (source, vocabulary) = crate::candidate_cli::int4::source::planner().map_err(candidate_failure)?;
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = crate::candidate_cli::int4::runtime::load_model_with_failure(&engine, &args, common,
        remaining()?, cancellation.clone(), host_failure)?;
    remaining()?;
    let source_identity = crate::candidate_cli::int4::source::identity(model.model_identity(), &source,
        BuiltInTask::Ner).map_err(candidate_failure)?;
    let (resolver, resolution_identity) = crate::candidate_cli::int4::resolve::runtime::planner(model.model_identity())
        .map_err(candidate_failure)?;
    let (source, resolver, vocabulary) = (Arc::new(source), Arc::new(resolver), Arc::new(vocabulary));
    // Authenticate the fixed recipe against the actual loaded model, including
    // the empty-epoch policy, before optional snapshot publication or corpus IO.
    drop(prepare_int4_entities(Vec::new(), Arc::clone(&source), source_identity.clone(),
        Arc::clone(&resolver), resolution_identity.clone(), task.clone(), model.model_identity(),
        &mut DeadlineControl { started, deadline }).map_err(entity_failure)?);
    remaining()?;
    if let Some(path) = &args.save_snapshot {
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false).map_err(candidate_failure)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(host_failure)?;
        if pin != model.model_identity().snapshot_pin() { return Err(candidate_failure(CandidateError::Identity)); }
    }
    remaining()?;
    let reader: Box<dyn BufRead + Send> = if args.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES,
            crate::local_io::open_document(&args.input).map_err(|_| input_failure("entity_corpus_input_open"))?))
    };
    let config = Int4EntityCorpusHostConfig {
        source_identity, resolution_identity, task, transport: limits.transport,
        native: native(&args, remaining()?), preparation_reserve_bytes: common.preparation_bytes,
        graph_reserve_bytes: command.resolution.graph_bytes().map_err(candidate_failure)?,
        io_reserve_bytes: limits.io_bytes,
    };
    let completed = engine.execute_int4_entity_corpus(&model, source, resolver, vocabulary, config,
        common.kernel, reader, output, cancellation).map_err(corpus_failure)?;
    // No frame here can certify a task that is still physically running. The
    // completion retains writer guards, and its sole footer write is fallible.
    remaining()?;
    completed.finish().map_err(corpus_failure)?;
    Ok(())
}

fn cancelled(cause: DecodeCancellationKind) -> Failure {
    use DecodeCancellationKind::*;
    let (exit, code) = match cause {
        Timeout | Deadline | PollQuota | CostBudget => (ErrorCode::BudgetOrTimeout, "entity_corpus_execution_budget"),
        User | ParentCancelled | Shutdown => (ErrorCode::Cancelled, "entity_corpus_cancelled"),
        ResourceUnavailable => (ErrorCode::AdmissionOrResourceLimit, "entity_corpus_resource_unavailable"),
        FailFast | LinkedExit => (ErrorCode::Generic, "entity_corpus_supervised_failure"),
        RaceLost => (ErrorCode::Generic, "entity_corpus_root_race_invariant"),
    };
    Failure { exit, code, cancellation: Some(cause) }
}
fn budget_failure(code: &'static str) -> Failure {
    Failure { exit: ErrorCode::BudgetOrTimeout, code, cancellation: None }
}
fn no_result() -> Failure {
    Failure { exit: ErrorCode::StructuredTaskNoResult, code: "entity_corpus_epoch_no_result", cancellation: None }
}
fn graph_failure(error: ResolveError) -> Failure {
    match error {
        ResolveError::Cancelled(cause) => cancelled(cause),
        ResolveError::InvalidLimits => Failure::usage("entity_corpus_graph_options"),
        ResolveError::InvalidInput | ResolveError::InvalidAnchor => input_failure("entity_corpus_original_documents"),
        ResolveError::InputBudget | ResolveError::BlockBudget | ResolveError::PairBudget
            | ResolveError::ScanBudget | ResolveError::ClusterBudget | ResolveError::OutputBudget =>
                budget_failure("entity_corpus_graph_budget"),
        ResolveError::AllocationRefused => candidate_failure(CandidateError::Memory),
        _ => Failure::output("entity_corpus_graph_contract"),
    }
}
fn entity_failure(error: Int4EntityError) -> Failure {
    if let Some(cause) = error.cancellation() { return cancelled(cause); }
    match error {
        Int4EntityError::InvalidLimits => Failure::usage("entity_corpus_task_options"),
        Int4EntityError::InvalidInput => input_failure("entity_corpus_original_documents"),
        Int4EntityError::Identity => candidate_failure(CandidateError::Identity),
        Int4EntityError::WorkBudget => budget_failure("entity_corpus_lifetime_budget"),
        Int4EntityError::Allocation => candidate_failure(CandidateError::Memory),
        Int4EntityError::Accounting => Failure::output("entity_corpus_native_contract"),
        Int4EntityError::Graph(error) => graph_failure(error),
        Int4EntityError::Resolution(Int4ResolveError::Identity) => candidate_failure(CandidateError::Identity),
        Int4EntityError::Resolution(Int4ResolveError::WorkBudget) => budget_failure("entity_corpus_pair_budget"),
        Int4EntityError::Resolution(Int4ResolveError::Resolution(error)) => graph_failure(error),
        Int4EntityError::Resolution(Int4ResolveError::Native(error)) =>
            host_failure(HostedError::Int4(Int4HostError::Native(error))),
        Int4EntityError::Native(error) => host_failure(HostedError::Int4(Int4HostError::Native(error))),
        _ => no_result(),
    }
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
        HostedError::Int4(Int4HostError::Entities(error)) => entity_failure(error),
        HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause)))
            | HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause)))
            | HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))) => cancelled(cause),
        HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Budget))
            | HostedError::Int4(Int4HostError::Model(Int4ModelError::Budget)) => budget_failure("entity_corpus_model_budget"),
        HostedError::Int4(Int4HostError::Native(StrictInt4Error::Context | StrictInt4Error::Work)) =>
            budget_failure("entity_corpus_native_budget"),
        HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Allocation))
            | HostedError::Int4(Int4HostError::Model(Int4ModelError::Allocation))
            | HostedError::Int4(Int4HostError::Native(StrictInt4Error::Allocation | StrictInt4Error::Memory))
            | HostedError::Limits(_) | HostedError::Reservation(_) | HostedError::Reentrant
            | HostedError::SingleCoordinatorRequired | HostedError::MissingRuntimeContext =>
                candidate_failure(CandidateError::Memory),
        HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Format | Int4SnapshotError::Integrity | Int4SnapshotError::Values))
            | HostedError::ModelIdentity | HostedError::ResourceDomain => candidate_failure(CandidateError::Identity),
        // Panic/join/scope failures cannot be hidden by a nested cancellation.
        _ => Failure::output("entity_corpus_runtime_or_completion"),
    }
}
fn corpus_failure(error: Int4EntityCorpusError) -> Failure {
    match error {
        Int4EntityCorpusError::Host(error) => host_failure(error),
        Int4EntityCorpusError::Transport(error) => match error {
            EntityStreamError::InvalidLimits => Failure::usage("entity_corpus_transport_options"),
            EntityStreamError::InputIo | EntityStreamError::InvalidRecord => input_failure("entity_corpus_record"),
            EntityStreamError::InputLimit | EntityStreamError::LineLimit | EntityStreamError::RecordLimit
                | EntityStreamError::SnapshotLimit | EntityStreamError::OutputLimit => budget_failure("entity_corpus_transport_budget"),
            EntityStreamError::Resolution(error) => graph_failure(error),
            EntityStreamError::Handler(error) => host_failure(error),
            EntityStreamError::OutputIo => Failure::output("entity_corpus_output_delivery_unknown"),
        },
        Int4EntityCorpusError::OutputIo => Failure::output("entity_corpus_output_delivery_unknown"),
        Int4EntityCorpusError::Completion => Failure::output("entity_corpus_completion_contract"),
    }
}

#[cfg(test)] mod tests;
