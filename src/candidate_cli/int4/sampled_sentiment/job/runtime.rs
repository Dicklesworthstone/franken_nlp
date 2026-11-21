//! One resident sampled executor inside the existing authenticated job host.
use super::*;
use super::super::runtime::{cancelled, host_failure as native_failure, task_failure};
use std::{cell::Cell, sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, MemoryClass,
    batch::{BatchCode, BatchFault},
    candidate_cli::runtime::owned_jobs::{self as owned, open_mode, progress_report},
    hosted::{CancellationToken, HostedError, NativeLimits, RunLimits,
        int4_sampled_sentiment_job_requirements,
        corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest}},
    jobs::{JobError, JobSecret, runner::{JobRunError, sampled_sentiment::SampledSentimentJobPlanner}},
    native_engine::{decode::DecodeCancellationKind,
        strict_int4::{prefill::INT4_LAYER_MAJOR_PREFILL_VERSION,
            weights::snapshot::INT4_SNAPSHOT_VERSION}},
    tasks::sentiment::sampled::{SampledSentimentPlanner, SampledSentimentVocabulary},
    tokenizer::untrusted::UntrustedEncoding,
};

pub(super) fn execute<R: Read + Send + 'static>(mode: RunMode, args: Int4Args, job: JobArgs,
    _admitted: Int4Limits, input: R, output: &mut impl Write) -> Result<(), Failure> {
    let started = Instant::now();
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or_else(|| cancelled(DecodeCancellationKind::Deadline));
    job.storage.input_retention.validate(mode, &args.input).map_err(storage_failure)?;
    let common = job.validate(&args)?;
    let engine = NlpEngine::builder().resource_config(compute::resource_config(common))
        .build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    // The caller's compiler/configuration/report staging remains distinct
    // from the complete host-owned population, planner and journal allowance.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let lifetime = job_cli::parse_limits(&read_config(&job.storage.limits_file, &remaining)?)
        .map_err(storage_failure)?;
    let limits = job.lifetime_limits(&args, common, lifetime)?;
    let common = limits.common;
    let encoding = args.input_encoding.policy(args.max_input_bytes, common.max_prompt_tokens)?;
    let preparation = sampled_sentiment_job_preparation_bytes(limits.task, limits.planning, &encoding)
        .map_err(batch_failure)?;
    let input_memory = job.storage.input_retention.input_memory().map_err(storage_failure)?;
    let host = host_limits(&args, limits, remaining()?);
    let required = int4_sampled_sentiment_job_requirements(args.group_width, limits.task, limits.native,
        preparation, lifetime, host, input_memory, common.prefill,
        (common.compute.threads > 1).then_some(common.compute.threads)).map_err(job_failure)?;
    if required.input_bytes != limits.buffer_bytes || required.native != common.native
        || sum(&[required.input_bytes, required.native.kv_bytes, required.workspace_bytes,
            required.output_reservation_bytes, required.team_workspace_bytes])? != limits.execution_bytes {
        return Err(Failure::usage("sampled_sentiment_job_host_quote"));
    }
    // No private key, model or population opens before the complete lifetime
    // work relation and caller+load/save/execute peak have been admitted.
    remaining()?;
    let key = JobSecret::read(&job.storage.key_file).map_err(storage_error)?;
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = super::super::super::runtime::load_model_with_failure(&engine, &args, common,
        remaining()?, cancellation.clone(), native_failure)?;
    remaining()?;
    let compiler = SampledSentimentPlanner::for_model(model.model_identity(),
        job.sampled.options(&args)?, limits.task, limits.planning).map_err(task_failure)?
        .with_input_encoding(encoding.clone()).map_err(task_failure)?;
    let compiler = job.sampled.configure_planner(compiler)?;
    let vocabulary = Arc::new(SampledSentimentVocabulary::pinned().map_err(task_failure)?);
    remaining()?;
    let planner = SampledSentimentJobPlanner::new(compiler, vocabulary, limits.native, lifetime)
        .map_err(batch_failure)?;
    let planner = match common.prefill {
        Some(prefill) => planner.with_layer_major_prefill(prefill).map_err(batch_failure)?,
        None => planner,
    };
    let planner = if job.sampled.reuse_prefill {
        planner.with_reused_prefill().map_err(batch_failure)?
    } else { planner };
    planner.check_job_limits(lifetime).map_err(batch_failure)?;
    if planner.preparation_requirement_bytes() != preparation
        || planner.max_result_bytes() != common.result_bytes as u64
        || planner.selected_rows() != job.sampled.selected_rows()?
        || planner.document_encoding() != &encoding {
        return Err(Failure::output("sampled_sentiment_job_planner_quote"));
    }
    let identity = planner.base_execution_identity().clone();
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(native_failure)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity.into()); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    remaining()?;
    let reader = job_cli::population_reader(mode, job.storage.input_retention, &args.input, input, IO_BUFFER_BYTES)
        .map_err(storage_failure)?;
    let job_id = job.storage.job_id;
    let materialize = job.storage.materialize;
    let request = SourceJobRequest { root: job.storage.job_dir, key, job_id, limits: lifetime,
        mode: open_mode(mode, job.storage.input_retention).map_err(storage_failure)?, materialize };
    let host = host_limits(&args, limits, remaining()?);
    let progress = if common.compute.threads == 1 {
        engine.job_int4_sampled_sentiment(&model, planner, request, host, common.kernel, reader, cancellation.clone())
    } else {
        engine.job_int4_sampled_sentiment_with_threads(&model, planner, request, host, common.kernel,
            common.compute.threads, reader, cancellation.clone())
    }.map_err(job_failure)?;
    if let Some(cause) = cancellation.cause() { return Err(cancelled(cause)); }
    remaining()?;
    let report = progress_report(progress, job_id, lifetime, materialize, mode, "sentiment-sampled")
        .map_err(storage_failure)?;
    let content = identity.logical_model_digest.to_hex();
    let response = Response {
        protocol: "fnlp-candidate-int4-sentiment-sampled-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        task: "sentiment-sampled", durable_unit: "complete_independent_sample_bundle",
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(),
        model_content_sha256: &content, candidate_format: &identity.artifact_format,
        quant_recipe: &identity.quant_recipe, group_width: model.model_identity().group_width(),
        integer_kernel: common.kernel.kind().id(), snapshot,
        input_encoding: input_encoding::reported_policy(&encoding),
        compute_schedule: compute::Schedule::for_threads(common.compute.threads),
        prefill_schedule: common.prefill.map(PrefillSchedule::new).transpose()?,
        prefill_reuse: job.sampled.prefill_reuse_execution(),
        max_selected_rows: job.sampled.max_selected_rows,
        output: &report,
    };
    // Staging/deadline failure can follow durable progress. Publish only one
    // bounded metadata record after the physical host has fully drained.
    publish_report(&response, output, &remaining, &cancellation)
}

fn publish_report<T: Serialize>(response: &T, output: &mut impl Write,
    remaining: &impl Fn() -> Result<Duration, Failure>, cancellation: &CancellationToken)
    -> Result<(), Failure> {
    // The three checks surround staging, the first external byte and flush.
    // A cause observed on the original token takes precedence over the wall
    // clock. Save the first observed failure instead of consulting a changed
    // token after another failure has already terminated publication.
    let first_failure = Cell::new(None);
    publish_checked(response, job_cli::REPORT_BYTES, output, &|| {
        if first_failure.get().is_some() { return Err(CandidateError::Execution); }
        let failure = cancellation.cause().map(cancelled).or_else(|| remaining().err());
        match failure {
            Some(failure) => {
                first_failure.set(Some(failure));
                Err(CandidateError::Execution)
            }
            None => Ok(()),
        }
    }).map_err(|error| first_failure.get().unwrap_or_else(|| error.into()))
}

fn read_config(path: &std::path::Path, remaining: &impl Fn() -> Result<Duration, Failure>)
    -> Result<String, Failure> {
    remaining()?;
    let mut file = crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?;
    let text = read_input(&mut file, job_cli::LIMIT_BYTES)?;
    remaining()?;
    Ok(text)
}
fn host_limits(args: &Int4Args, limits: JobExecutionLimits, remaining: Duration) -> JobHostLimits {
    JobHostLimits { native: NativeLimits { context_tokens: args.context_tokens,
        allocator_reserve_bytes: ALLOCATOR_BYTES,
        run: RunLimits { max_elapsed: remaining, max_checkpoints: args.max_checkpoints,
            cleanup_reserve_bytes: CLEANUP_BYTES } },
        transport: limits.transport, preparation_reserve_bytes: limits.preparation_reserve_bytes,
        io_reserve_bytes: limits.io_reserve_bytes, journal_reserve_bytes: limits.journal_reserve_bytes,
        serialization_reserve_bytes: limits.serialization_reserve_bytes }
}
fn storage_error(error: JobError) -> Failure {
    if let JobError::Cancelled(cause) = error { return cancelled(cause); }
    storage_failure(owned::job_failure(error))
}
fn batch_failure(fault: BatchFault) -> Failure {
    if let Some(cause) = fault.cancellation { return cancelled(cause); }
    use BatchCode::*;
    let (exit, code) = match fault.code {
        InvalidLimits => (ErrorCode::Usage, "sampled_sentiment_job_limits"),
        InvalidUtf8 | InvalidJson | InvalidEnvelope | InvalidId | DuplicateId | Planning | InputIo =>
            (ErrorCode::InputDecodeOrParse, "sampled_sentiment_job_input"),
        LineLimit | DocumentLimit | EpochIdLimit | InputLimit | RequestLimit
            | OutputLineLimit | OutputLimit | WorkLimit =>
            (ErrorCode::BudgetOrTimeout, "sampled_sentiment_job_budget"),
        Admission | Allocation => (ErrorCode::AdmissionOrResourceLimit, "sampled_sentiment_job_admission"),
        Execution => (ErrorCode::StructuredTaskNoResult, "sampled_sentiment_job_no_result"),
        Cancelled => (ErrorCode::Generic, "sampled_sentiment_job_missing_cancellation"),
        SequenceOverflow | InvalidExecution | Serialization | OutputIo =>
            (ErrorCode::Generic, "sampled_sentiment_job_runtime_or_delivery"),
    };
    Failure { exit, code, cancellation: None }
}
fn job_failure(error: HostedJobError) -> Failure {
    match error {
        HostedJobError::Host(HostedError::BatchSetup(fault)) => batch_failure(fault),
        HostedJobError::Host(error) => native_failure(error),
        HostedJobError::Job(JobRunError::Storage(error)) => storage_error(error),
        HostedJobError::Job(JobRunError::Processor(fault)) => batch_failure(fault),
        HostedJobError::Job(JobRunError::InvalidEnvelope) => Failure {
            exit: ErrorCode::InputDecodeOrParse, code: "sampled_sentiment_job_envelope", cancellation: None },
        HostedJobError::Job(JobRunError::WorkContract | JobRunError::Stopped) =>
            Failure::output("sampled_sentiment_job_completion_invariant"),
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
struct PrefillSchedule { execution: &'static str, max_batch_rows: usize, admitted_workspace_bytes: u64 }
impl PrefillSchedule {
    fn new(limits: Int4PrefillLimits) -> Result<Self, CandidateError> {
        Ok(Self { execution: INT4_LAYER_MAJOR_PREFILL_VERSION, max_batch_rows: limits.max_batch_rows,
            admitted_workspace_bytes: limits.validate().map_err(|_| CandidateError::Arguments)? })
    }
}
#[derive(Serialize)]
struct Response<'a, T: Serialize> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str,
    task: &'static str, durable_unit: &'static str, model_id: &'static str,
    source_revision: &'a str, source_root_sha256: &'a str, model_content_sha256: &'a str,
    candidate_format: &'a str, quant_recipe: &'a str, group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_schedule: Option<compute::Schedule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_schedule: Option<PrefillSchedule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_reuse: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_selected_rows: Option<usize>,
    output: &'a T,
}

#[cfg(test)] mod tests;
