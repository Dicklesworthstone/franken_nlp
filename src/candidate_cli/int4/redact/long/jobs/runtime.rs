//! Authenticate one complete population before repair or durable native execution.
use super::*;
use std::{io::{BufRead, BufReader}, sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    candidate_cli::runtime::owned_jobs::{host_failure, job_failure, open_mode, progress_report},
    hosted::{CancellationToken, HostedError, Int4HostError, NativeLimits, RunLimits, RedactionPseudonyms,
        corpus::{RedactionCorpusConfig, jobs::{HostedJobError, JobHostLimits, SourceJobRequest}}},
    jobs::{JobError, JobSecret, MismatchField, runner::{JobRunError,
        redact::grouped_int4::{check_configuration, check_job_limits, INT4_REDACTION_JOB_RECIPE_BYTES}}},
    native_engine::{decode::DecodeCancellationKind, strict_int4::weights::snapshot::INT4_SNAPSHOT_VERSION},
    tasks::{BuiltInTask, redact::{detectors, pseudonym::Pseudonyms}},
};

pub(super) fn execute<R: Read + Send + 'static>(mode: RunMode, args: Int4Args,
    job: JobArgs, _admitted: DocumentLimits, mut input: R, output: &mut impl Write) -> Result<(), Failure> {
    let started = Instant::now();
    let admitted = job.validate(&args)?;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or(CandidateError::Timeout);
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: admitted.common.memory_bytes,
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    // Declared before private allocations; survives their drop and final report flush.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, admitted.common.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let lifetime = job_cli::parse_limits(&read_config(&job.storage.limits_file, job_cli::LIMIT_BYTES, &remaining)?)?;
    let limits = job.lifetime_limits(&args, admitted, lifetime)?;
    let common = limits.common;
    check_host_quote(host_limits(&args, limits, remaining()?), lifetime, limits.buffer_bytes)?;
    // Protected job-authentication key is separate from the caller's binary
    // pseudonym key. Neither is generated, reused as sampling entropy or printed.
    let key = JobSecret::read(&job.storage.key_file).map_err(job_failure)?;
    remaining()?;
    let request = job.document.request(&args, limits.document);
    let secret = key_scope(&job, &mut input, &request)?;
    remaining()?;
    let options = job.document.redaction.source.options_path()
        .map(|path| read_config(path, crate::candidate_cli::source::OPTIONS_BYTES, &remaining)).transpose()?;
    let ner = shared::ner_options(options.as_deref())?;
    drop(options);
    drop(detectors::detect("", &request.rules, request.rule_budget).map_err(|_| CandidateError::Planning)?);
    let (planner, vocabulary) = source::planner()?;
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = super::super::super::super::runtime::load_model(
        &engine, &args, common, remaining()?, cancellation.clone())?;
    remaining()?;
    let identity = source::identity(model.model_identity(), &planner, BuiltInTask::Ner)?;
    let config = job.configuration(&args, limits, identity.clone(), ner);
    check_job_limits(&config, lifetime).map_err(|_| Failure::usage("redaction_job_lifetime_contract"))?;
    check_configuration(&planner, &config, model.model_identity(), lifetime)
        .map_err(|_| Failure::usage("redaction_job_configuration"))?;
    // This separately requested model snapshot is setup output, not a job
    // commit. Population/recipe authentication happens inside the host below.
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity.into()); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    remaining()?;
    // No caller stdio lock crosses the physical boundary. All decoded original
    // text is also checked against config.max_source_bytes by the durable adapter.
    let reader: Box<dyn BufRead + Send> = if args.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES,
            crate::local_io::open_document(&args.input).map_err(|_| CandidateError::Input)?))
    };
    let host = host_limits(&args, limits, remaining()?);
    let job_id = job.storage.job_id;
    let materialize = job.storage.materialize;
    let request = SourceJobRequest { root: job.storage.job_dir, key, job_id, limits: lifetime,
        mode: open_mode(mode), materialize };
    let progress = engine.job_int4_redact_document(&model, Arc::new(planner), Arc::new(vocabulary),
        RedactionCorpusConfig { batch: config, edit_reserve_bytes: limits.document.edit_bytes },
        secret, request, host, common.kernel, reader, cancellation).map_err(execution_failure)?;
    remaining()?;
    let report = progress_report(progress, job_id, lifetime, materialize, mode, "redact")?;
    let content = identity.logical_model_digest.to_hex();
    let response = Response {
        protocol: "fnlp-candidate-int4-redact-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "redact",
        redaction_mode: "complete_document", retained_output_is_sensitive: true,
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(), model_content_sha256: &content,
        candidate_format: &identity.artifact_format, quant_recipe: &identity.quant_recipe,
        group_width: model.model_identity().group_width(), integer_kernel: common.kernel.kind().id(),
        snapshot, output: &report,
    };
    // Metadata only. Delivery failure may follow durable progress and never
    // authorizes automatic replay, tail repair or an unrequested rollback.
    publish(&response, job_cli::REPORT_BYTES, output).map_err(Failure::from)
}
fn key_scope(job: &JobArgs, input: &mut impl Read, request: &RedactionRequest)
    -> Result<Option<RedactionPseudonyms>, CandidateError> {
    let secret = job.document.redaction.key(input)?
        .map(|(key, namespace)| RedactionPseudonyms { key: Arc::new(key), namespace });
    let context = secret.as_ref().map(|s| Pseudonyms::full256(&s.key, &s.namespace,
        request.actions.expected_key_commitment.as_deref())).transpose().map_err(|_| CandidateError::Identity)?;
    request.actions.check_key(context.as_ref()).map_err(|_| CandidateError::Arguments)?;
    drop(context);
    Ok(secret)
}
fn read_config(path: &std::path::Path, cap: usize,
    remaining: &impl Fn() -> Result<Duration, CandidateError>) -> Result<String, Failure> {
    remaining()?;
    let mut file = crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?;
    let text = read_input(&mut file, cap)?;
    remaining()?;
    Ok(text)
}
fn host_limits(args: &Int4Args, limits: JobExecutionLimits, remaining: Duration) -> JobHostLimits {
    JobHostLimits {
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        transport: limits.transport, preparation_reserve_bytes: limits.preparation_reserve_bytes,
        io_reserve_bytes: limits.io_reserve_bytes, journal_reserve_bytes: limits.journal_reserve_bytes,
        serialization_reserve_bytes: limits.serialization_reserve_bytes,
    }
}
fn check_host_quote(host: JobHostLimits, job: JobLimits, expected: u64) -> Result<(), Failure> {
    if host.required_buffer_bytes(job).map_err(execution_failure)? != expected
        || (INT4_REDACTION_JOB_RECIPE_BYTES as u64).checked_mul(4) != Some(RECIPE_RESERVE_BYTES) {
        return Err(Failure::usage("job_host_memory_quote"));
    }
    Ok(())
}
fn cancelled(cause: DecodeCancellationKind) -> Failure {
    let (exit, code) = match cause {
        DecodeCancellationKind::Timeout | DecodeCancellationKind::Deadline
            | DecodeCancellationKind::PollQuota | DecodeCancellationKind::CostBudget =>
            (ErrorCode::BudgetOrTimeout, "job_execution_budget"),
        DecodeCancellationKind::User | DecodeCancellationKind::Shutdown | DecodeCancellationKind::ParentCancelled =>
            (ErrorCode::Cancelled, "job_cancelled"),
        DecodeCancellationKind::ResourceUnavailable =>
            (ErrorCode::AdmissionOrResourceLimit, "job_execution_resource"),
        DecodeCancellationKind::FailFast | DecodeCancellationKind::LinkedExit =>
            (ErrorCode::Generic, "job_supervised_failure"),
        DecodeCancellationKind::RaceLost => (ErrorCode::Generic, "job_cancellation_invariant"),
    };
    Failure { exit, code }
}
fn execution_failure(error: HostedJobError) -> Failure {
    match error {
        HostedJobError::Host(HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. }) =>
            execution_failure(HostedJobError::Host(*error)),
        HostedJobError::Host(HostedError::Stopped { stop, task_error: Some(error) })
            if matches!(stop.kind, DecodeCancellationKind::FailFast | DecodeCancellationKind::LinkedExit) =>
            execution_failure(HostedJobError::Host(*error)),
        HostedJobError::Host(HostedError::Stopped { stop, .. }) => cancelled(stop.kind),
        HostedJobError::Job(JobRunError::Storage(JobError::Cancelled(cause))) => cancelled(cause),
        HostedJobError::Job(JobRunError::Processor(fault)) => match fault.cancellation {
            Some(cause) => cancelled(cause),
            None => host_failure(HostedJobError::Job(JobRunError::Processor(fault))),
        },
        HostedJobError::Host(HostedError::Int4(Int4HostError::SourceMap(error))) =>
            error.cancellation().map(cancelled).unwrap_or(Failure {
                exit: ErrorCode::StructuredTaskNoResult, code: "job_task_failed" }),
        HostedJobError::Host(HostedError::Int4(Int4HostError::Redaction(error))) =>
            error.cancellation().map(cancelled).unwrap_or(Failure {
                exit: ErrorCode::StructuredTaskNoResult, code: "job_task_failed" }),
        HostedJobError::Job(JobRunError::Storage(JobError::Mismatch(field))) => Failure {
            exit: ErrorCode::ArtifactIntegrityOrFormatOrVersion,
            code: match field {
                MismatchField::Job => "job_mismatch_id", MismatchField::Secret => "job_mismatch_secret",
                MismatchField::Execution => "job_mismatch_execution", MismatchField::Recipe => "job_mismatch_recipe",
                MismatchField::Population => "job_mismatch_population", MismatchField::Limits => "job_mismatch_limits",
            },
        },
        other => host_failure(other),
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
struct Response<'a, T: Serialize> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str, task: &'static str,
    redaction_mode: &'static str, retained_output_is_sensitive: bool,
    model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str,
    model_content_sha256: &'a str, candidate_format: &'a str, quant_recipe: &'a str,
    group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    output: &'a T,
}
#[cfg(test)]
mod tests;
