//! One owned extraction/review stream with complete pre-IO admission.
use super::*;
use std::{io::{BufRead, BufReader}, time::{Duration, Instant}};
use crate::{
    NlpEngine, MemoryClass,
    batch::{BatchCode, BatchFault},
    candidate_cli::batch::CandidateWriter,
    hosted::{CancellationToken, HostedError, Int4HostError, Int4SemanticCorpusConfig,
        Int4SemanticCorpusRequirements, NativeLimits, RunLimits, int4_semantic_corpus_requirements,
        corpus::CorpusLimits},
    native_engine::{decode::DecodeCancellationKind, generation::grouped_int4::INT4_SOURCE_FORMAT,
        strict_int4::{Int4ModelError, StrictInt4Error,
            weights::snapshot::{Int4SnapshotError, INT4_SNAPSHOT_VERSION}}},
    tasks::extract::semantic::{SemanticError, grouped_int4::{Int4SemanticError, INT4_SEMANTIC_EXECUTION}},
    tokenizer::untrusted::UntrustedEncoding,
};

pub(super) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    args: Int4Args, batch: BatchArgs, _admitted: Int4Limits, input: R, output: W) -> Result<(), Failure> {
    let started = Instant::now();
    let initial = batch.validate(&args)?;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or_else(|| cancelled(DecodeCancellationKind::Deadline));
    let engine = NlpEngine::builder().resource_config(compute::resource_config(initial))
        .build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    // Fixed policy/schema and writer staging remain caller-owned beside the
    // host's resident compiler, live source and complete review intermediates.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, initial.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let options = parse_options(&read_local(&batch.semantic.semantic_options, OPTIONS_BYTES, &remaining)?)?;
    let limits = batch.policy_limits(&args, initial, &options)?;
    let quote = requirements(&args, &batch, limits, &options, remaining()?)?;
    let limits = admit_memory(&args, limits, quote)?;
    let common = limits.common;
    // No schema, original population or model is opened until the complete
    // fixed policy, run work, transport and simultaneous owner peak are known.
    let schema = read_local(batch.semantic.extraction.schema_path(),
        crate::candidate_cli::extract::SCHEMA_BYTES, &remaining)?;
    let defaults = batch.load_defaults(schema, limits.task, &options)?;
    let encoding = batch.semantic.input_encoding.policy()?;
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = crate::candidate_cli::int4::runtime::load_model_with_failure(
        &engine, &args, common, remaining()?, cancellation.clone(), host_failure)?;
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(host_failure)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity.into()); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    remaining()?;
    let reader: Box<dyn BufRead + Send> = if args.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES,
            crate::local_io::open_document(&args.input).map_err(|_| CandidateError::Input)?))
    };
    remaining()?;
    let identity = model.model_identity();
    let content: String = identity.content_sha256().iter().map(|byte| format!("{byte:02x}")).collect();
    let recipe = format!("{}:group={}", identity.matrix_recipe(), identity.group_width());
    let provenance = Provenance {
        protocol: "fnlp-candidate-int4-semantic-extraction-batch-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        task: "extract-semantic", execution: INT4_SEMANTIC_EXECUTION,
        same_model_correlated: true, calibration: "experimental_uncalibrated_diagnostic_only",
        model_id: "Nanbeige4.2-3B", source_revision: identity.revision(),
        source_root_sha256: identity.source_root_sha256(), model_content_sha256: &content,
        candidate_format: INT4_SOURCE_FORMAT, quant_recipe: &recipe, group_width: args.group_width,
        integer_kernel: common.kernel.kind().id(), snapshot,
        compute_schedule: compute::Schedule::for_threads(common.compute.threads),
        input_encoding: input_encoding::reported_policy(&encoding),
    };
    let writer = CandidateWriter::new(output, &provenance, limits.transport.max_output_line_bytes,
        limits.output_bytes)?;
    let config = Int4SemanticCorpusConfig {
        extraction: defaults, compiler_limits: batch.semantic.extraction.compiler(),
        source_limits: batch.semantic.extraction.source_limits(),
        judge_planning: judge_limits(&options), options, encoding,
        work: limits.native, limits: host_limits(&args, limits, remaining()?),
    };
    let summary = if common.compute.threads > 1 {
        engine.batch_int4_semantic_extract_with_threads(&model, config, common.kernel,
            common.compute.threads, reader, writer, cancellation.clone())
    } else {
        engine.batch_int4_semantic_extract(&model, config, common.kernel, reader, writer, cancellation.clone())
    }.map_err(host_failure)?;
    if let Some(cause) = cancellation.cause() { return Err(cancelled(cause)); }
    remaining()?;
    completed(summary)
}

fn host_limits(args: &Int4Args, limits: BatchLimits, remaining: Duration) -> CorpusLimits {
    CorpusLimits {
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        transport: limits.transport, preparation_reserve_bytes: limits.common.preparation_bytes,
        io_reserve_bytes: limits.io_bytes,
    }
}
fn requirements(args: &Int4Args, batch: &BatchArgs, limits: BatchLimits, options: &Int4SemanticOptions,
    remaining: Duration) -> Result<Int4SemanticCorpusRequirements, Failure> {
    int4_semantic_corpus_requirements(args.group_width, limits.task, options, judge_limits(options),
        &batch.semantic.input_encoding.policy()?, limits.native, host_limits(args, limits, remaining),
        (limits.common.compute.threads > 1).then_some(limits.common.compute.threads)).map_err(host_failure)
}
fn admit_memory(args: &Int4Args, mut limits: BatchLimits, quote: Int4SemanticCorpusRequirements)
    -> Result<BatchLimits, Failure> {
    let mut common = limits.common;
    if quote.native != common.native || quote.team_workspace_bytes != common.compute.workspace_bytes {
        return Err(CandidateError::Identity.into());
    }
    let execution = quote.total_bytes().map_err(host_failure)?;
    let loading = if common.snapshot_pin.is_some() { common.snapshot_need.scratch_bytes }
        else { sum(&[common.source_preparation_bytes, common.load.scratch_payload_bytes])? };
    let saving = if args.save_snapshot.is_some() { common.snapshot_need.scratch_bytes } else { 0 };
    common.minimum_memory_bytes = sum(&[common.preparation_bytes, common.load.weight_payload_bytes,
        ALLOCATOR_BYTES, loading.max(saving).max(execution), CLEANUP_BYTES])?;
    if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Memory.into()); }
    limits.common = common;
    Ok(limits)
}
fn read_local(path: &std::path::Path, cap: usize, remaining: &impl Fn() -> Result<Duration, Failure>)
    -> Result<String, Failure> {
    remaining()?;
    let mut file = crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?;
    let result = read_input(&mut file, cap)?;
    remaining()?;
    Ok(result)
}
fn cancelled(cause: DecodeCancellationKind) -> Failure {
    use DecodeCancellationKind::*;
    let (exit, code) = match cause {
        Timeout | Deadline | PollQuota | CostBudget => (ErrorCode::BudgetOrTimeout, "semantic_batch_budget"),
        User | ParentCancelled | Shutdown => (ErrorCode::Cancelled, "semantic_batch_cancelled"),
        ResourceUnavailable => (ErrorCode::AdmissionOrResourceLimit, "semantic_batch_resources"),
        FailFast | LinkedExit => (ErrorCode::Generic, "semantic_batch_supervised_failure"),
        RaceLost => (ErrorCode::Generic, "semantic_batch_root_race"),
    };
    Failure { exit, code, cancellation: Some(cause) }
}
fn batch_failure(fault: BatchFault) -> Failure {
    if let Some(cause) = fault.cancellation { return cancelled(cause); }
    use BatchCode::*;
    let (exit, code) = match fault.code {
        InvalidLimits => (ErrorCode::Usage, "semantic_batch_limits"),
        InvalidUtf8 | InvalidJson | InvalidEnvelope | InvalidId | DuplicateId | Planning | InputIo =>
            (ErrorCode::InputDecodeOrParse, "semantic_batch_input"),
        LineLimit | DocumentLimit | EpochIdLimit | InputLimit | RequestLimit | OutputLineLimit | OutputLimit | WorkLimit =>
            (ErrorCode::BudgetOrTimeout, "semantic_batch_budget"),
        Admission | Allocation => (ErrorCode::AdmissionOrResourceLimit, "semantic_batch_admission"),
        Execution => (ErrorCode::StructuredTaskNoResult, "semantic_batch_no_result"),
        Cancelled => (ErrorCode::Generic, "semantic_batch_missing_cancellation"),
        SequenceOverflow | InvalidExecution | Serialization | OutputIo =>
            (ErrorCode::Generic, "semantic_batch_runtime_or_delivery"),
    };
    Failure { exit, code, cancellation: None }
}
fn semantic_failure(error: Int4SemanticError) -> Failure {
    if let Some(cause) = error.cancellation() { return cancelled(cause); }
    let (exit, code) = match error {
        Int4SemanticError::WorkBudget | Int4SemanticError::Semantic(SemanticError::Limit(_)) =>
            (ErrorCode::BudgetOrTimeout, "semantic_batch_budget"),
        Int4SemanticError::Contract => (ErrorCode::Usage, "semantic_batch_policy"),
        Int4SemanticError::Semantic(SemanticError::AllocationRefused) =>
            (ErrorCode::AdmissionOrResourceLimit, "semantic_batch_resources"),
        Int4SemanticError::Accounting => (ErrorCode::Generic, "semantic_batch_accounting"),
        _ => (ErrorCode::StructuredTaskNoResult, "semantic_batch_no_result"),
    };
    Failure { exit, code, cancellation: None }
}
fn host_failure(error: HostedError) -> Failure {
    match error {
        HostedError::BatchSetup(fault) => batch_failure(fault),
        HostedError::Batch(error) => batch_failure(error.fault),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => host_failure(*error),
        HostedError::Stopped { stop, task_error } => {
            if matches!(stop.kind, DecodeCancellationKind::FailFast | DecodeCancellationKind::LinkedExit) {
                if let Some(error) = task_error {
                    let mut failure = host_failure(*error);
                    failure.cancellation = Some(stop.kind);
                    return failure;
                }
            }
            cancelled(stop.kind)
        }
        HostedError::Int4(Int4HostError::SemanticExtraction(error)) => semantic_failure(error),
        HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause)))
            | HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause)))
            | HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))) => cancelled(cause),
        HostedError::Limits(_) | HostedError::Reservation(_) | HostedError::Reentrant
            | HostedError::MissingRuntimeContext | HostedError::SingleCoordinatorRequired => Failure {
                exit: ErrorCode::AdmissionOrResourceLimit, code: "semantic_batch_admission", cancellation: None },
        HostedError::ModelIdentity | HostedError::ResourceDomain
            | HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Format
                | Int4SnapshotError::Integrity | Int4SnapshotError::Values)) => Failure {
                exit: ErrorCode::ArtifactIntegrityOrFormatOrVersion, code: "semantic_batch_model_identity", cancellation: None },
        _ => Failure::output("semantic_batch_runtime_or_completion"),
    }
}

#[derive(Serialize)]
struct SnapshotRecord { operation: &'static str, format: &'static str, file_bytes: u64 }
impl SnapshotRecord {
    fn new(operation: &'static str, need: Int4SnapshotRequirements) -> Self {
        Self { operation, format: INT4_SNAPSHOT_VERSION, file_bytes: need.file_bytes }
    }
}
#[derive(Serialize)]
struct Provenance<'a> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str,
    task: &'static str, execution: &'static str, same_model_correlated: bool, calibration: &'static str,
    model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str, model_content_sha256: &'a str,
    candidate_format: &'static str, quant_recipe: &'a str, group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_schedule: Option<compute::Schedule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
}

#[cfg(test)]
mod tests;
