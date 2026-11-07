//! One admitted resident model and authenticated population for durable QA.
use super::*;
use std::{sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    batch::source::grouped_int4::question::Int4QuestionDocumentBatchPlanner,
    candidate_cli::runtime::owned_jobs::{host_failure, job_failure, open_mode, progress_report},
    hosted::{CancellationToken, HostedError, Int4HostError, NativeLimits, RunLimits,
        corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest}},
    jobs::{JobError, JobSecret,
        runner::source::grouped_int4::question_document::Int4QuestionDocumentJobPlanner},
    native_engine::strict_int4::{Int4ModelError, StrictInt4Error,
        weights::snapshot::{Int4SnapshotError, INT4_SNAPSHOT_VERSION}},
    tasks::BuiltInTask,
};

pub(super) fn execute<R: Read + Send + 'static>(mode: RunMode, args: Int4Args,
    job: JobArgs, _admitted: Int4Limits, input: R, output: &mut impl Write) -> Result<(), Failure> {
    let started = Instant::now();
    job.storage.input_retention.validate(mode, &args.input)?;
    let declared = job.validate(&args)?;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or(CandidateError::Timeout);
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: declared.memory_bytes,
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    let preparation = sum(&[declared.preparation_bytes,
        2 * IO_BUFFER_BYTES as u64, job_cli::REPORT_BYTES as u64])?;
    let _preparation = lease.reserve(MemoryClass::JobBuffers, preparation)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let lifetime = job_cli::parse_limits(&read_config(&job.storage.limits_file, job_cli::LIMIT_BYTES, &remaining)?)?;
    let raw = job.source.options_path().map(|path| read_config(path, shared::OPTIONS_BYTES, &remaining)).transpose()?;
    let options = question::options(raw.as_deref())?;
    drop(raw);
    let limits = job.lifetime_limits(&args, declared, options, lifetime)?;
    if limits.cli_preparation_bytes != preparation { return Err(CandidateError::Memory.into()); }
    let common = limits.common;
    let host = host_limits(&args, limits, remaining()?);
    let quote = crate::hosted::int4_question_document_job_requirements(args.group_width,
        limits.config, lifetime, host, limits.reduction_reserve_bytes,
        job.questions.cohort_chunks, job.questions.cohort_hierarchy_groups).map_err(host_failure)?;
    if quote != limits.memory { return Err(CandidateError::Memory.into()); }
    remaining()?;
    let (source, vocabulary) = source::planner()?;
    let key = JobSecret::read(&job.storage.key_file).map_err(job_failure)?;
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = crate::candidate_cli::int4::runtime::load_model_with_failure(&engine, &args, common,
        remaining()?, cancellation.clone(), model_failure)?;
    remaining()?;
    let identity = source::identity(model.model_identity(), &source, BuiltInTask::Answer)?;
    let factory = Int4QuestionDocumentBatchPlanner::for_model(model.model_identity(), Arc::new(source),
        identity, limits.config).map_err(|fault| host_failure(HostedJobError::Host(HostedError::BatchSetup(fault))))?;
    let factory = if job.questions.cohort_chunks.is_some() {
        factory.with_chunk_cohorts().map_err(|fault| host_failure(
            HostedJobError::Job(crate::jobs::runner::JobRunError::Processor(fault))))?
    } else { factory };
    let factory = if let Some(groups) = job.questions.cohort_hierarchy_groups {
        factory.with_hierarchy_cohorts(groups).map_err(|fault| host_failure(
            HostedJobError::Job(crate::jobs::runner::JobRunError::Processor(fault))))?
    } else { factory };
    let planner = Int4QuestionDocumentJobPlanner::new(factory, lifetime).map_err(|fault| host_failure(
        HostedJobError::Job(crate::jobs::runner::JobRunError::Processor(fault))))?;
    if planner.max_result_bytes() != quote.document.max_output_bytes
        || planner.max_output_tokens() != quote.document.max_output_tokens {
        return Err(CandidateError::Identity.into());
    }
    let identity = planner.execution_identity().clone();
    remaining()?;
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(model_failure)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity.into()); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    remaining()?;
    let reader = job_cli::population_reader(mode, job.storage.input_retention, &args.input, input, IO_BUFFER_BYTES)?;
    let job_id = job.storage.job_id;
    let materialize = job.storage.materialize;
    let request = SourceJobRequest { root: job.storage.job_dir, key, job_id, limits: lifetime,
        mode: open_mode(mode, job.storage.input_retention)?, materialize };
    let progress = engine.job_int4_question_documents(&model, planner, Arc::new(vocabulary), request,
        host_limits(&args, limits, remaining()?), limits.reduction_reserve_bytes,
        common.kernel, reader, cancellation).map_err(|error| match error {
            HostedJobError::Host(error) => model_failure(error), other => host_failure(other),
        })?;
    remaining()?;
    let report = progress_report(progress, job_id, lifetime, materialize, mode, "answer")?;
    let content = identity.logical_model_digest.to_hex();
    let response = Response { protocol: "fnlp-candidate-int4-answer-document-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "answer",
        durable_unit: "complete_original_source_question", question_source: "per_record",
        answer_mode: answer_mode(limits.config), model_id: "Nanbeige4.2-3B",
        source_revision: &identity.source_revision, source_root_sha256: model.model_identity().source_root_sha256(),
        model_content_sha256: &content, candidate_format: &identity.artifact_format,
        quant_recipe: &identity.quant_recipe, group_width: model.model_identity().group_width(),
        integer_kernel: common.kernel.kind().id(), snapshot, cohort_chunks: job.questions.cohort_chunks,
        cohort_hierarchy_groups: job.questions.cohort_hierarchy_groups, output: &report };
    publish_report(&response, output, &remaining)
}

fn model_failure(error: HostedError) -> Failure {
    let cancellation = match &error {
        HostedError::Int4(Int4HostError::SourceMap(error)) => error.cancellation(),
        HostedError::Int4(Int4HostError::SourceTask(error)) => error.cancellation(),
        HostedError::BatchSetup(fault) => fault.cancellation,
        _ => None,
    };
    if let Some(cause) = cancellation { return job_failure(JobError::Cancelled(cause)); }
    match error {
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => model_failure(*error),
        HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause)))
            | HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause)))
            | HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))) =>
                job_failure(JobError::Cancelled(cause)),
        other => host_failure(HostedJobError::Host(other)),
    }
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
    JobHostLimits { native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
        run: RunLimits { max_elapsed: remaining, max_checkpoints: args.max_checkpoints,
            cleanup_reserve_bytes: CLEANUP_BYTES } }, transport: limits.transport,
        preparation_reserve_bytes: limits.common.preparation_bytes, io_reserve_bytes: limits.io_reserve_bytes,
        journal_reserve_bytes: limits.journal_reserve_bytes,
        serialization_reserve_bytes: limits.serialization_reserve_bytes }
}
fn answer_mode(config: Int4QuestionDocumentBatchConfig) -> &'static str {
    if config.hierarchy.is_some() { "hierarchical_verified_evidence_synthesis" }
    else if config.synthesis.is_some() { "verified_evidence_synthesis" }
    else { "independent_passage_answers" }
}
fn publish_report<T: Serialize>(response: &T, output: &mut impl Write,
    remaining: &impl Fn() -> Result<Duration, CandidateError>) -> Result<(), Failure> {
    remaining()?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(job_cli::REPORT_BYTES).map_err(|_| CandidateError::Memory)?;
    let mut staged = BoundedOutput { bytes, cap: job_cli::REPORT_BYTES };
    serde_json::to_writer(&mut staged, response).map_err(|_| CandidateError::Output)?;
    staged.write_all(b"\n").map_err(|_| CandidateError::Output)?;
    remaining()?;
    output.write_all(&staged.bytes).and_then(|()| output.flush()).map_err(|_| CandidateError::Output)?;
    remaining()?;
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
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str,
    task: &'static str, durable_unit: &'static str, question_source: &'static str, answer_mode: &'static str,
    model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str, model_content_sha256: &'a str,
    candidate_format: &'a str, quant_recipe: &'a str, group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cohort_chunks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cohort_hierarchy_groups: Option<usize>,
    output: &'a T,
}

#[cfg(test)] mod tests;
