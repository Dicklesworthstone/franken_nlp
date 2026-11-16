//! One authenticated population, one resident sampler and one physical drain.
use super::*;
use std::{sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, MemoryClass,
    batch::sentiment::sampled::SampledSentimentBatchPlanner,
    candidate_cli::runtime::owned_jobs::{host_failure, job_failure, open_mode, progress_report},
    hosted::{CancellationToken, HostedError, NativeLimits, RunLimits,
        corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest}},
    jobs::{JobError, JobInputMemory, JobSecret, runner::sampled_sentiment::SampledSentimentJobPlanner},
    native_engine::strict_int4::{prefill::{Int4PrefillLimits, INT4_LAYER_MAJOR_PREFILL_VERSION},
        weights::snapshot::INT4_SNAPSHOT_VERSION},
    tasks::sentiment::sampled::{SampledSentimentError, SampledSentimentPlanner, SampledSentimentVocabulary},
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
    // Every private config/key/planner and the final report drops before this
    // caller charge. The host separately owns its transferred input lifetime.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let lifetime = job_cli::parse_limits(&read_config(&job.storage.limits_file,
        job_cli::LIMIT_BYTES, &remaining)?)?;
    let limits = job.lifetime_limits(&args, common, lifetime)?;
    let common = limits.common;
    check_host_quote(host_limits(&args, limits, remaining()?), lifetime, limits.buffer_bytes, input_memory)?;
    let options = job.sampled.options(&args)?;
    let budget = job.sampled.budget(&args, common, SentimentAxis::ALL.len())?;
    let encoding = args.input_encoding.policy(args.max_input_bytes, common.max_prompt_tokens)?;
    let key = JobSecret::read(&job.storage.key_file).map_err(job_failure)?;
    remaining()?;
    let reader = job_cli::population_reader(mode, job.storage.input_retention, &args.input, input, IO_BUFFER_BYTES)?;
    let cancellation = CancellationToken::default();
    let model = crate::candidate_cli::int4::runtime::load_model_with_failure(&engine, &args, common,
        remaining()?, cancellation.clone(), native_failure)?;
    remaining()?;
    let planner = SampledSentimentPlanner::for_model(model.model_identity(), options, budget,
        job.sampled.task_limits(&args)).map_err(task_failure)?
        .with_input_encoding(encoding.clone()).map_err(task_failure)?;
    remaining()?;
    let vocabulary = Arc::new(SampledSentimentVocabulary::pinned().map_err(task_failure)?);
    remaining()?;
    let planner = SampledSentimentBatchPlanner::new(planner, vocabulary, budget, limits.native)
        .map_err(|_| CandidateError::Planning)?;
    let planner = SampledSentimentJobPlanner::new(planner, lifetime).map_err(|_| CandidateError::Planning)?;
    planner.check_job_limits(lifetime).map_err(|_| CandidateError::Planning)?;
    if planner.group_width() != args.group_width || planner.document_encoding() != &encoding
        || planner.preparation_requirement_bytes() > limits.preparation_reserve_bytes {
        return Err(CandidateError::Identity.into());
    }
    let identity = planner.execution_identity().clone();
    remaining()?;
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(native_failure)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity.into()); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    let host = host_limits(&args, limits, remaining()?);
    let job_id = job.storage.job_id;
    let materialize = job.storage.materialize;
    let request = SourceJobRequest { root: job.storage.job_dir, key, job_id, limits: lifetime,
        mode: open_mode(mode, job.storage.input_retention)?, materialize };
    let progress = if common.compute.threads > 1 {
        engine.job_int4_sampled_sentiment_with_threads(&model, planner, request, host,
            common.kernel, common.compute.threads, reader, cancellation.clone())
    } else {
        match common.prefill {
            None => engine.job_int4_sampled_sentiment(&model, planner, request, host,
                common.kernel, reader, cancellation.clone()),
            Some(prefill) => engine.job_int4_sampled_sentiment_layer_major(&model, planner,
                request, host, common.kernel, prefill, reader, cancellation.clone()),
        }
    }.map_err(execution_failure)?;
    if let Some(cause) = cancellation.cause() { return Err(job_failure(JobError::Cancelled(cause))); }
    remaining()?;
    let report = progress_report(progress, job_id, lifetime, materialize, mode, "sentiment-sampled")?;
    let content = identity.logical_model_digest.to_hex();
    let response = Response {
        protocol: "fnlp-candidate-int4-sentiment-sampled-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "sentiment-sampled",
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(),
        model_content_sha256: &content, candidate_format: &identity.artifact_format,
        quant_recipe: &identity.quant_recipe, group_width: model.model_identity().group_width(),
        integer_kernel: common.kernel.kind().id(), snapshot,
        input_encoding: input_encoding::reported_policy(&encoding),
        compute_schedule: compute::Schedule::for_threads(common.compute.threads),
        prefill_schedule: common.prefill.map(PrefillSchedule::new).transpose()?, output: &report,
    };
    // Already-committed jobs still obey the original token and deadline here.
    // Report failure never means that earlier durable commits rolled back.
    publish_report(&response, output, &remaining, &cancellation)
}

fn task_failure(error: SampledSentimentError) -> Failure {
    let failure = super::super::runtime::task_failure(error);
    Failure { exit: failure.exit, code: failure.code }
}
fn native_failure(error: HostedError) -> Failure {
    let failure = super::super::runtime::host_failure(error);
    Failure { exit: failure.exit, code: failure.code }
}
fn execution_failure(error: HostedJobError) -> Failure {
    match error {
        HostedJobError::Host(error) => native_failure(error),
        other => host_failure(other),
    }
}
fn publish_report<T: Serialize>(response: &T, output: &mut impl Write,
    remaining: &impl Fn() -> Result<Duration, CandidateError>,
    cancellation: &CancellationToken) -> Result<(), Failure> {
    let check = || {
        if cancellation.cause().is_some() { return Err(CandidateError::Execution); }
        remaining().map(|_| ())
    };
    publish_checked(response, job_cli::REPORT_BYTES, output, &check).map_err(|error| {
        match (error, cancellation.cause()) {
            (CandidateError::Execution, Some(cause)) => job_failure(JobError::Cancelled(cause)),
            (error, _) => Failure::from(error),
        }
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
fn check_host_quote(host: JobHostLimits, job: JobLimits, expected: u64, input: JobInputMemory)
    -> Result<(), Failure> {
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
struct PrefillSchedule { execution: &'static str, max_batch_rows: usize, admitted_workspace_bytes: u64 }
impl PrefillSchedule {
    fn new(limits: Int4PrefillLimits) -> Result<Self, CandidateError> {
        Ok(Self { execution: INT4_LAYER_MAJOR_PREFILL_VERSION, max_batch_rows: limits.max_batch_rows,
            admitted_workspace_bytes: limits.validate().map_err(|_| CandidateError::Arguments)? })
    }
}
#[derive(Serialize)]
struct Response<'a, T: Serialize> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str, task: &'static str,
    model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str, model_content_sha256: &'a str,
    candidate_format: &'a str, quant_recipe: &'a str, group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a crate::tokenizer::untrusted::UntrustedEncoding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_schedule: Option<compute::Schedule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_schedule: Option<PrefillSchedule>,
    output: &'a T,
}

#[cfg(test)]
mod tests;
