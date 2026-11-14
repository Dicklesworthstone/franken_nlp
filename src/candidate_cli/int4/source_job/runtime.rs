//! One owned native invocation for an authenticated source-task population.
use super::*;
use crate::jobs::JobInputMemory;
use std::{sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, MemoryClass,
    batch::source::grouped_int4::MAX_SOURCE_ARGUMENT_BYTES,
    candidate_cli::runtime::owned_jobs::{host_failure, job_failure, open_mode, progress_report},
    hosted::{CancellationToken, HostedError, Int4HostError, NativeLimits, RunLimits,
        corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest}},
    jobs::{JobError, JobSecret, runner::source::grouped_int4::Int4SourceJobPlanner},
    native_engine::strict_int4::{StrictInt4Error, weights::snapshot::INT4_SNAPSHOT_VERSION},
    tokenizer::untrusted::UntrustedEncoding,
};

pub(super) fn execute<R: Read + Send + 'static>(mode: RunMode, args: Int4Args,
    job: JobArgs, _admitted: Int4Limits, input: R, output: &mut impl Write) -> Result<(), Failure> {
    let started = Instant::now();
    job.storage.input_retention.validate(mode, &args.input)?;
    let input_memory = job.storage.input_retention.input_memory()?;
    let common = job.validate(&args)?;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or(CandidateError::Timeout);
    let engine = NlpEngine::builder().resource_config(compute::resource_config(common))
        .build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    // Private options, key, pinned compiler/vocabulary, IO and report staging
    // remain charged until their storage drops, including durable-run errors.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let lifetime = job_cli::parse_limits(&read_config(&job.storage.limits_file,
        job_cli::LIMIT_BYTES, &remaining)?)?;
    let limits = job.lifetime_limits(&args, common, lifetime)?;
    let common = limits.common;
    check_host_quote(host_limits(&args, limits, remaining()?), lifetime, limits.buffer_bytes, input_memory)?;
    let key = JobSecret::read(&job.storage.key_file).map_err(job_failure)?;
    remaining()?;
    let defaults_text = job.defaults.defaults.as_deref()
        .map(|path| read_config(path, MAX_SOURCE_ARGUMENT_BYTES, &remaining)).transpose()?;
    let budget = job.source.budget(&args, common);
    let defaults = source_batch::load_defaults(job.kind, defaults_text.as_deref(), budget, args.max_input_bytes)?;
    drop(defaults_text);
    remaining()?;
    let reader = job_cli::population_reader(mode, job.storage.input_retention, &args.input, input, IO_BUFFER_BYTES)?;
    let cancellation = CancellationToken::default();
    let model = super::super::runtime::load_model(&engine, &args, common, remaining()?, cancellation.clone())?;
    remaining()?;
    let (source_planner, vocabulary) = source::planner()?;
    let encoding = args.input_encoding.policy(args.max_input_bytes, common.max_prompt_tokens)?;
    let encoding_work_limits = job.encoding.limits(&args)?;
    // This single original Arc owns the complete invocation allowance. Do not
    // fork the composed encoder at record, compiler or hosted-run boundaries.
    let source_planner = match encoding_work_limits {
        Some(work) => source_planner.with_map_document_encoding(encoding.clone(), work)
            .map_err(|_| CandidateError::Planning)?,
        None => source_planner,
    };
    let planner = Int4SourceJobPlanner::for_model_with_document_encoding(model.model_identity(),
        Arc::new(source_planner), source::task(job.kind), budget,
        job.source.planning(&args), defaults, limits.native).map_err(|_| CandidateError::Planning)?;
    planner.check_job_limits(lifetime).map_err(|_| CandidateError::Planning)?;
    check_encoding_quote(&planner, &encoding, encoding_work_limits, limits.encoding_workspace_bytes)?;
    remaining()?;
    let identity = planner.execution_identity().clone();
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity.into()); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    let host = host_limits(&args, limits, remaining()?);
    let job_id = job.storage.job_id;
    let materialize = job.storage.materialize;
    let task = job.kind.name();
    let request = SourceJobRequest { root: job.storage.job_dir, key, job_id, limits: lifetime,
        mode: open_mode(mode, job.storage.input_retention)?, materialize };
    // Host authenticates the exact complete population and private recipe,
    // skips committed items and retains each output guard through spool sync
    // and journal acknowledgement. No native result is fabricated here.
    let progress = if common.compute.threads == 1 {
        engine.job_int4_source(&model, planner, Arc::new(vocabulary), request, host,
            common.kernel, reader, cancellation.clone())
    } else {
        engine.job_int4_source_with_threads(&model, planner, Arc::new(vocabulary), request, host,
            common.kernel, common.compute.threads, reader, cancellation.clone())
    }.map_err(execution_failure)?;
    if let Some(cause) = cancellation.cause() {
        return Err(job_failure(JobError::Cancelled(cause)));
    }
    remaining()?;
    let report = progress_report(progress, job_id, lifetime, materialize, mode, task)?;
    let content = identity.logical_model_digest.to_hex();
    let response = Response {
        protocol: "fnlp-candidate-int4-source-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        task, model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(),
        model_content_sha256: &content, candidate_format: &identity.artifact_format,
        quant_recipe: &identity.quant_recipe, group_width: model.model_identity().group_width(),
        integer_kernel: common.kernel.kind().id(), snapshot,
        input_encoding: input_encoding::reported_policy(&encoding), encoding_work_limits,
        compute_schedule: compute::Schedule::for_threads(common.compute.threads), output: &report,
    };
    // Metadata only. Delivery/flush failure may follow durable commits and
    // never authorizes retry without the authenticated resume path.
    publish_report(&response, output, &remaining, &cancellation)
}

fn check_encoding_quote(planner: &Int4SourceJobPlanner, encoding: &UntrustedEncoding,
    total: Option<SafeBpeWorkLimits>, workspace: u64) -> Result<(), Failure> {
    if planner.document_encoding() != encoding || planner.encoding_work_limits() != total
        || planner.encoding_workspace_bytes() != workspace {
        return Err(Failure::usage("source_job_tokenizer_quote"));
    }
    Ok(())
}

fn execution_failure(error: HostedJobError) -> Failure {
    match error {
        HostedJobError::Host(HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. }) =>
                execution_failure(HostedJobError::Host(*error)),
        HostedJobError::Host(HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause)))) =>
            job_failure(JobError::Cancelled(cause)),
        other => host_failure(other),
    }
}

fn publish_report<T: Serialize>(response: &T, output: &mut impl Write,
    remaining: &impl Fn() -> Result<Duration, CandidateError>, cancellation: &CancellationToken)
    -> Result<(), Failure> {
    // Cancellation may arrive after durable completion, during serialization
    // or while the consumer flushes. Preserve its original typed policy and
    // never turn successful persistence into successful command delivery.
    publish_checked(response, job_cli::REPORT_BYTES, output, &|| {
        if cancellation.cause().is_some() { return Err(CandidateError::Execution); }
        remaining().map(|_| ())
    }).map_err(|error| match (error, cancellation.cause()) {
        (CandidateError::Execution, Some(cause)) => job_failure(JobError::Cancelled(cause)),
        (error, _) => Failure::from(error),
    })
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
fn check_host_quote(host: JobHostLimits, job: JobLimits, expected: u64, input: JobInputMemory) -> Result<(), Failure> {
    if host.required_buffer_bytes_for_input(job, input).map_err(host_failure)? != expected {
        return Err(Failure::usage("job_host_memory_quote"));
    }
    Ok(())
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
    task: &'static str,
    model_id: &'static str,
    source_revision: &'a str,
    source_root_sha256: &'a str,
    model_content_sha256: &'a str,
    candidate_format: &'a str,
    quant_recipe: &'a str,
    group_width: usize,
    integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_schedule: Option<compute::Schedule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_work_limits: Option<SafeBpeWorkLimits>,
    output: &'a T,
}
#[cfg(test)]
mod tests;
