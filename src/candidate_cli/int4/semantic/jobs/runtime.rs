//! One admitted owned job; complete extraction/review results commit together.
use super::*;
use std::time::{Duration, Instant};
use crate::{
    NlpEngine, MemoryClass,
    candidate_cli::{extract::SCHEMA_BYTES, runtime::owned_jobs::{
        host_failure as owned_failure, job_failure, open_mode, progress_report}},
    hosted::{CancellationToken, HostedError, Int4HostError, Int4SemanticJobRequirements,
        NativeLimits, RunLimits, int4_semantic_job_requirements,
        corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest}},
    jobs::{JobError, JobSecret, runner::{JobRunError,
        extract::grouped_int4::semantic::Int4SemanticExtractionJobConfig}},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        strict_int4::{Int4ModelError, StrictInt4Error,
            weights::snapshot::{Int4SnapshotError, INT4_SNAPSHOT_VERSION}}},
    tasks::extract::semantic::{SemanticError,
        grouped_int4::{Int4SemanticError, INT4_SEMANTIC_EXECUTION}},
    tokenizer::untrusted::UntrustedEncoding,
};

pub(super) fn execute<R: Read + Send + 'static>(mode: RunMode, args: Int4Args,
    job: JobArgs, _admitted: Int4Limits, input: R, output: &mut impl Write) -> Result<(), Failure> {
    let started = Instant::now();
    job.storage.input_retention.validate(mode, &args.input).map_err(storage_failure)?;
    let initial = job.validate(&args)?;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or_else(|| cancelled(DecodeCancellationKind::Deadline));
    let engine = NlpEngine::builder().resource_config(compute::resource_config(initial))
        .build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    // Caller-owned configurations, key, schema, reader and completion metadata
    // stay charged beside the host's independently priced owned invocation.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, initial.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let options = parse_options(&read_config(&job.semantic.semantic_options, OPTIONS_BYTES, &remaining)?)?;
    let lifetime = job_cli::parse_limits(&read_config(&job.storage.limits_file,
        job_cli::LIMIT_BYTES, &remaining)?).map_err(storage_failure)?;
    let limits = job.lifetime_limits(&args, initial, &options, lifetime)?;
    let quote = requirements(&args, &job, limits, &options, lifetime, remaining()?)?;
    let limits = admit_memory(&args, limits, quote)?;
    let common = limits.common;
    // Complete work/result/policy and simultaneous memory bounds precede all
    // key, schema, population and model access. Only fixed bounded settings
    // have been read at this point; no durable storage has been opened.
    let key = JobSecret::read(&job.storage.key_file).map_err(|error| storage_failure(job_failure(error)))?;
    remaining()?;
    let schema = read_config(job.semantic.extraction.schema_path(), SCHEMA_BYTES, &remaining)?;
    let extraction = job.defaults(schema, limits.task, &options)?;
    let encoding = job.semantic.input_encoding.policy()?;
    let config = Int4SemanticExtractionJobConfig {
        extraction, compiler_limits: job.semantic.extraction.compiler(),
        source_limits: job.semantic.extraction.source_limits(),
        judge_planning: judge_limits(&options), options, encoding: encoding.clone(), work: limits.native,
    };
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = crate::candidate_cli::int4::runtime::load_model_with_failure(
        &engine, &args, common, remaining()?, cancellation.clone(), host_error)?;
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(host_error)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity.into()); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    remaining()?;
    let reader = job_cli::population_reader(mode, job.storage.input_retention,
        &args.input, input, IO_BUFFER_BYTES).map_err(storage_failure)?;
    let host = host_limits(&args, limits, remaining()?);
    let job_id = job.storage.job_id;
    let materialize = job.storage.materialize;
    let request = SourceJobRequest {
        root: job.storage.job_dir, key, job_id, limits: lifetime,
        mode: open_mode(mode, job.storage.input_retention).map_err(storage_failure)?, materialize,
    };
    let progress = if common.compute.threads > 1 {
        engine.job_int4_semantic_extract_with_threads(&model, config, request, host,
            common.kernel, common.compute.threads, reader, cancellation.clone())
    } else {
        engine.job_int4_semantic_extract(&model, config, request, host,
            common.kernel, reader, cancellation.clone())
    }.map_err(execution_failure)?;
    if let Some(cause) = cancellation.cause() { return Err(cancelled(cause)); }
    remaining()?;
    let report = progress_report(progress, job_id, lifetime, materialize, mode, "extract-semantic")
        .map_err(storage_failure)?;
    let identity = model.model_identity();
    let content: String = identity.content_sha256().iter().map(|byte| format!("{byte:02x}")).collect();
    let recipe = format!("{}:group={}", identity.matrix_recipe(), identity.group_width());
    let response = Response {
        protocol: "fnlp-candidate-int4-semantic-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        execution: INT4_SEMANTIC_EXECUTION, same_model_correlated: true,
        calibration: "experimental_uncalibrated_diagnostic_only",
        model_id: "Nanbeige4.2-3B", source_revision: identity.revision(),
        source_root_sha256: identity.source_root_sha256(), model_content_sha256: &content,
        candidate_format: INT4_SOURCE_FORMAT, quant_recipe: &recipe, group_width: identity.group_width(),
        integer_kernel: common.kernel.kind().id(), snapshot,
        compute_schedule: compute::Schedule::for_threads(common.compute.threads),
        input_encoding: input_encoding::reported_policy(&encoding), output: &report,
    };
    publish_report(&response, output, &remaining, &cancellation)
}

fn host_limits(args: &Int4Args, limits: JobExecutionLimits, remaining: Duration) -> JobHostLimits {
    JobHostLimits {
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        transport: limits.transport, preparation_reserve_bytes: limits.common.preparation_bytes,
        io_reserve_bytes: limits.io_reserve_bytes, journal_reserve_bytes: limits.journal_reserve_bytes,
        serialization_reserve_bytes: limits.serialization_reserve_bytes,
    }
}
fn requirements(args: &Int4Args, command: &JobArgs, limits: JobExecutionLimits,
    options: &Int4SemanticOptions, job: JobLimits, remaining: Duration)
    -> Result<Int4SemanticJobRequirements, Failure> {
    int4_semantic_job_requirements(args.group_width, limits.task, options, judge_limits(options),
        &command.semantic.input_encoding.policy()?, limits.native, host_limits(args, limits, remaining), job,
        command.storage.input_retention.input_memory().map_err(storage_failure)?,
        (limits.common.compute.threads > 1).then_some(limits.common.compute.threads))
        .map_err(execution_failure)
}
fn admit_memory(args: &Int4Args, mut limits: JobExecutionLimits, quote: Int4SemanticJobRequirements)
    -> Result<JobExecutionLimits, Failure> {
    let mut common = limits.common;
    if quote.native != common.native || quote.team_workspace_bytes != common.compute.workspace_bytes {
        return Err(CandidateError::Identity.into());
    }
    let execution = quote.total_bytes().map_err(host_error)?;
    let loading = if common.snapshot_pin.is_some() { common.snapshot_need.scratch_bytes }
        else { sum(&[common.source_preparation_bytes, common.load.scratch_payload_bytes])? };
    let saving = if args.save_snapshot.is_some() { common.snapshot_need.scratch_bytes } else { 0 };
    common.minimum_memory_bytes = sum(&[common.preparation_bytes, common.load.weight_payload_bytes,
        ALLOCATOR_BYTES, loading.max(saving).max(execution), CLEANUP_BYTES])?;
    if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Memory.into()); }
    limits.common = common;
    Ok(limits)
}
fn read_config(path: &std::path::Path, cap: usize, remaining: &impl Fn() -> Result<Duration, Failure>)
    -> Result<String, Failure> {
    remaining()?;
    let mut file = crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?;
    let text = read_input(&mut file, cap)?;
    remaining()?;
    Ok(text)
}

fn cancelled(cause: DecodeCancellationKind) -> Failure {
    use DecodeCancellationKind::*;
    let (exit, code) = match cause {
        Timeout | Deadline | PollQuota | CostBudget => (ErrorCode::BudgetOrTimeout, "semantic_job_budget"),
        User | ParentCancelled | Shutdown => (ErrorCode::Cancelled, "semantic_job_cancelled"),
        ResourceUnavailable => (ErrorCode::AdmissionOrResourceLimit, "semantic_job_resources"),
        FailFast | LinkedExit => (ErrorCode::Generic, "semantic_job_supervised_failure"),
        RaceLost => (ErrorCode::Generic, "semantic_job_root_race"),
    };
    Failure { exit, code, cancellation: Some(cause) }
}
fn host_error(error: HostedError) -> Failure { execution_failure(HostedJobError::Host(error)) }
fn execution_failure(error: HostedJobError) -> Failure {
    match error {
        HostedJobError::Job(JobRunError::Storage(JobError::Cancelled(cause))) => cancelled(cause),
        HostedJobError::Job(JobRunError::Processor(fault)) => {
            if let Some(cause) = fault.cancellation { return cancelled(cause); }
            storage_failure(owned_failure(HostedJobError::Job(JobRunError::Processor(fault))))
        }
        HostedJobError::Host(HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. }) => host_error(*error),
        HostedJobError::Host(HostedError::Stopped { stop, task_error }) => {
            if matches!(stop.kind, DecodeCancellationKind::FailFast | DecodeCancellationKind::LinkedExit) {
                if let Some(error) = task_error {
                    let mut failure = host_error(*error);
                    failure.cancellation = Some(stop.kind);
                    return failure;
                }
            }
            cancelled(stop.kind)
        }
        HostedJobError::Host(HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause)))
            | HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause)))
            | HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause)))) => cancelled(cause),
        HostedJobError::Host(HostedError::Int4(Int4HostError::SemanticExtraction(error))) => {
            if let Some(cause) = error.cancellation() { return cancelled(cause); }
            let (exit, code) = match error {
                Int4SemanticError::WorkBudget | Int4SemanticError::Semantic(SemanticError::Limit(_)) =>
                    (ErrorCode::BudgetOrTimeout, "semantic_job_budget"),
                Int4SemanticError::Contract => (ErrorCode::Usage, "semantic_job_policy"),
                Int4SemanticError::Semantic(SemanticError::AllocationRefused) =>
                    (ErrorCode::AdmissionOrResourceLimit, "semantic_job_resources"),
                Int4SemanticError::Accounting => (ErrorCode::Generic, "semantic_job_accounting"),
                _ => (ErrorCode::StructuredTaskNoResult, "semantic_job_no_result"),
            };
            Failure { exit, code, cancellation: None }
        }
        HostedJobError::Host(HostedError::ModelIdentity | HostedError::ResourceDomain
            | HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Format
                | Int4SnapshotError::Integrity | Int4SnapshotError::Values))) => Failure {
                    exit: ErrorCode::ArtifactIntegrityOrFormatOrVersion,
                    code: "semantic_job_model_identity", cancellation: None },
        other => storage_failure(owned_failure(other)),
    }
}
fn publish_report<T: Serialize>(response: &T, output: &mut impl Write,
    remaining: &impl Fn() -> Result<Duration, Failure>, cancellation: &CancellationToken)
    -> Result<(), Failure> {
    publish_checked(response, job_cli::REPORT_BYTES, output, &|| {
        if cancellation.cause().is_some() { return Err(CandidateError::Execution); }
        remaining().map(|_| ()).map_err(|_| CandidateError::Timeout)
    }).map_err(|error| match (error, cancellation.cause()) {
        (_, Some(cause)) => cancelled(cause),
        (CandidateError::Timeout, None) => cancelled(DecodeCancellationKind::Deadline),
        (error, None) => Failure::from(error),
    })
}

#[derive(Serialize)]
struct SnapshotRecord { operation: &'static str, format: &'static str, file_bytes: u64 }
impl SnapshotRecord {
    fn new(operation: &'static str, need: Int4SnapshotRequirements) -> Self {
        Self { operation, format: INT4_SNAPSHOT_VERSION, file_bytes: need.file_bytes }
    }
}
#[derive(Serialize)]
struct Response<'a, T: Serialize> {
    protocol: &'static str,
    schema_version: u32,
    scope: &'static str,
    evidence: &'static str,
    execution: &'static str,
    same_model_correlated: bool,
    calibration: &'static str,
    model_id: &'static str,
    source_revision: &'a str,
    source_root_sha256: &'a str,
    model_content_sha256: &'a str,
    candidate_format: &'static str,
    quant_recipe: &'a str,
    group_width: usize,
    integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_schedule: Option<compute::Schedule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
    output: &'a T,
}

#[cfg(test)]
mod tests;
