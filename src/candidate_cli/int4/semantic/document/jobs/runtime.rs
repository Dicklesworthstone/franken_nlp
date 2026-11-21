//! Owned complete-document execution; one guarded combined result per commit.
use super::*;
use super::super::runtime as document_runtime;
use std::time::{Duration, Instant};
use crate::{
    MemoryClass, NlpEngine,
    candidate_cli::{extract::SCHEMA_BYTES, runtime::owned_jobs::{
        host_failure as owned_failure, job_failure, open_mode, progress_report}},
    hosted::{CancellationToken, HostedError, Int4SemanticObjectDocumentJobHostLimits,
        Int4SemanticObjectDocumentJobRequirements, RunLimits, int4_semantic_object_job_requirements,
        corpus::jobs::{HostedJobError, SourceJobRequest}},
    jobs::{JobError, JobSecret, runner::{JobRunError,
        extract::grouped_int4::semantic::document::Int4SemanticObjectDocumentJobConfig}},
    native_engine::{decode::DecodeStepControl, generation::grouped_int4::INT4_SOURCE_FORMAT,
        strict_int4::weights::snapshot::INT4_SNAPSHOT_VERSION},
    tasks::extract::semantic::grouped_int4::document::{
        INT4_SEMANTIC_OBJECT_DOCUMENT_EXECUTION, SEMANTIC_OBJECT_DOCUMENT_SCOPE},
    tokenizer::untrusted::UntrustedEncoding,
};

struct DeadlineControl { started: Instant, deadline: Duration }
impl DecodeStepControl for DeadlineControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        (self.started.elapsed() >= self.deadline).then_some(DecodeCancellationKind::Deadline)
    }
}

pub(super) fn execute<R: Read + Send + 'static>(mode: RunMode, args: Int4Args,
    job: JobArgs, _admitted: extract_map::object::ObjectMapLimits,
    input: R, output: &mut impl Write) -> Result<(), Failure> {
    let started = Instant::now();
    job.storage.input_retention.validate(mode, &args.input).map_err(storage_failure)?;
    let object = job.validate(&args)?;
    let initial = object.map.common;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or_else(|| document_runtime::cancelled(DecodeCancellationKind::Deadline));
    let engine = NlpEngine::builder().resource_config(compute::resource_config(initial))
        .build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    // Configuration, original-schema derivation, key, caller IO and completion
    // metadata coexist with the host's independently priced owned invocation.
    let caller_bytes = caller_preparation_bytes(&job, initial)?;
    let _preparation = lease.reserve(MemoryClass::JobBuffers, caller_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let options = parse_options(&document_runtime::read_local(&job.semantic.semantic_options,
        OPTIONS_BYTES, &remaining)?)?;
    let lifetime = job_cli::parse_limits(&document_runtime::read_local(&job.storage.limits_file,
        job_cli::LIMIT_BYTES, &remaining)?).map_err(storage_failure)?;
    let limits = job.lifetime_limits(&args, object, &options, lifetime)?;
    let quote = requirements(&args, &job, limits, &options, lifetime, remaining()?)?;
    let limits = admit_memory(&args, limits, quote, caller_bytes)?;
    let common = limits.object.map.common;
    // All declarations and the complete simultaneous peak have passed before
    // touching key, schema, original/retained population, source or snapshot.
    let key = JobSecret::read(&job.storage.key_file).map_err(|error| storage_failure(job_failure(error)))?;
    remaining()?;
    let schema = document_runtime::read_local(job.semantic.object.map.extraction.schema_path(),
        SCHEMA_BYTES, &remaining)?;
    let extraction = document_runtime::source_request(&job.semantic, &options, schema,
        limits.task, limits.object, &mut DeadlineControl { started, deadline })?;
    let encoding = job.semantic.object.map.encoding.policy()?;
    let host = host_limits(&args, &job, limits, &options, remaining()?);
    let config = Int4SemanticObjectDocumentJobConfig {
        extraction, compiler_limits: job.semantic.object.map.extraction.compiler(),
        source_limits: job.semantic.object.map.extraction.source_limits(),
        judge_planning: judge_limits(&options), options, encoding: encoding.clone(),
        extraction_limits: limits.object.mapping(), review_limits: limits.review, work: limits.native,
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
    let mut host = host;
    host.document.native.run.max_elapsed = remaining()?;
    let job_id = job.storage.job_id;
    let materialize = job.storage.materialize;
    let request = SourceJobRequest {
        root: job.storage.job_dir, key, job_id, limits: lifetime,
        mode: open_mode(mode, job.storage.input_retention).map_err(storage_failure)?, materialize,
    };
    let progress = if common.compute.threads > 1 {
        engine.job_int4_semantic_object_documents_with_threads(&model, config, request, host,
            common.kernel, common.compute.threads, reader, cancellation.clone())
    } else {
        engine.job_int4_semantic_object_documents(&model, config, request, host,
            common.kernel, reader, cancellation.clone())
    }.map_err(execution_failure)?;
    if let Some(cause) = cancellation.cause() { return Err(document_runtime::cancelled(cause)); }
    remaining()?;
    let report = progress_report(progress, job_id, lifetime, materialize, mode, "extract-semantic-document")
        .map_err(storage_failure)?;
    let identity = model.model_identity();
    let content: String = identity.content_sha256().iter().map(|byte| format!("{byte:02x}")).collect();
    let recipe = format!("{}:group={}", identity.matrix_recipe(), identity.group_width());
    let response = Response {
        protocol: "fnlp-candidate-int4-semantic-object-document-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        execution: INT4_SEMANTIC_OBJECT_DOCUMENT_EXECUTION, semantic_scope: SEMANTIC_OBJECT_DOCUMENT_SCOPE,
        same_model_correlated: true, calibration: "experimental_uncalibrated_diagnostic_only",
        model_id: "Nanbeige4.2-3B", source_revision: identity.revision(),
        source_root_sha256: identity.source_root_sha256(), model_content_sha256: &content,
        candidate_format: INT4_SOURCE_FORMAT, quant_recipe: &recipe, group_width: identity.group_width(),
        integer_kernel: common.kernel.kind().id(), snapshot,
        compute_schedule: compute::Schedule::for_threads(common.compute.threads),
        input_encoding: input_encoding::reported_policy(&encoding), output: &report,
    };
    publish_report(&response, output, &remaining, &cancellation)
}

fn caller_preparation_bytes(command: &JobArgs, common: Int4Limits) -> Result<u64, Failure> {
    Ok(sum(&[
        document_runtime::caller_preparation_bytes(&command.semantic, common)?,
        job_cli::LIMIT_BYTES as u64 * 4,
        job_cli::REPORT_BYTES as u64 * 2,
        IO_BUFFER_BYTES as u64 * 2,
    ])?)
}
fn host_limits(args: &Int4Args, command: &JobArgs, limits: JobExecutionLimits,
    options: &Int4SemanticOptions, remaining: Duration) -> Int4SemanticObjectDocumentJobHostLimits {
    Int4SemanticObjectDocumentJobHostLimits {
        document: document_runtime::host_limits(args, &command.semantic, limits.object,
            options, limits.review, remaining),
        transport: limits.transport, io_reserve_bytes: limits.io_reserve_bytes,
        journal_reserve_bytes: limits.journal_reserve_bytes,
        serialization_reserve_bytes: limits.serialization_reserve_bytes,
    }
}
fn requirements(args: &Int4Args, command: &JobArgs, limits: JobExecutionLimits,
    options: &Int4SemanticOptions, job: JobLimits, remaining: Duration)
    -> Result<Int4SemanticObjectDocumentJobRequirements, Failure> {
    int4_semantic_object_job_requirements(args.group_width, limits.task, options,
        &command.semantic.object.map.encoding.policy()?, limits.native,
        host_limits(args, command, limits, options, remaining), job,
        command.storage.input_retention.input_memory().map_err(storage_failure)?,
        (limits.object.map.common.compute.threads > 1).then_some(limits.object.map.common.compute.threads))
        .map_err(execution_failure)
}
fn admit_memory(args: &Int4Args, mut limits: JobExecutionLimits,
    quote: Int4SemanticObjectDocumentJobRequirements, caller_bytes: u64)
    -> Result<JobExecutionLimits, Failure> {
    let mut common = limits.object.map.common;
    if quote.native != common.native || quote.team_workspace_bytes != common.compute.workspace_bytes {
        return Err(CandidateError::Identity.into());
    }
    let execution = quote.total_bytes().map_err(host_error)?;
    let loading = if common.snapshot_pin.is_some() { common.snapshot_need.scratch_bytes }
        else { sum(&[common.source_preparation_bytes, common.load.scratch_payload_bytes])? };
    let saving = if args.save_snapshot.is_some() { common.snapshot_need.scratch_bytes } else { 0 };
    common.minimum_memory_bytes = sum(&[caller_bytes, common.load.weight_payload_bytes,
        ALLOCATOR_BYTES, loading.max(saving).max(execution), CLEANUP_BYTES])?;
    if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Memory.into()); }
    limits.object.map.common = common;
    Ok(limits)
}

fn host_error(error: HostedError) -> Failure { execution_failure(HostedJobError::Host(error)) }
fn execution_failure(error: HostedJobError) -> Failure {
    match error {
        HostedJobError::Job(JobRunError::Storage(JobError::Cancelled(cause))) =>
            document_runtime::cancelled(cause),
        HostedJobError::Job(JobRunError::Processor(fault)) => {
            if let Some(cause) = fault.cancellation { return document_runtime::cancelled(cause); }
            storage_failure(owned_failure(HostedJobError::Job(JobRunError::Processor(fault))))
        }
        HostedJobError::Host(error) => document_runtime::host_error(error),
        other => storage_failure(owned_failure(other)),
    }
}
fn publish_report<T: Serialize>(response: &T, output: &mut impl Write,
    remaining: &impl Fn() -> Result<Duration, Failure>, cancellation: &CancellationToken)
    -> Result<(), Failure> {
    document_runtime::publish(response, job_cli::REPORT_BYTES, output, remaining, cancellation)
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
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str,
    execution: &'static str, semantic_scope: &'static str,
    same_model_correlated: bool, calibration: &'static str, model_id: &'static str,
    source_revision: &'a str, source_root_sha256: &'a str, model_content_sha256: &'a str,
    candidate_format: &'static str, quant_recipe: &'a str, group_width: usize, integer_kernel: &'static str,
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
