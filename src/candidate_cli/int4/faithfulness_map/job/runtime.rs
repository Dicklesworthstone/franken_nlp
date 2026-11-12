//! One resident model and one authenticated lifetime for complete faithfulness maps.
use super::*;
use std::time::{Duration, Instant};
use crate::{
    NlpEngine, MemoryClass,
    candidate_cli::runtime::owned_jobs::{host_failure, job_failure, open_mode, progress_report},
    hosted::{CancellationToken, HostedError, Int4HostError, NativeLimits, RunLimits,
        corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest}},
    jobs::{JobError, JobSecret,
        runner::scored::grouped_int4::faithfulness_document::Int4FaithfulnessDocumentJobPlanner},
    native_engine::strict_int4::{Int4ModelError, StrictInt4Error,
        weights::snapshot::{Int4SnapshotError, INT4_SNAPSHOT_VERSION}},
    tasks::judge::grouped_int4::Int4JudgePlanner,
};

pub(super) fn execute<R: Read + Send + 'static>(mode: RunMode, args: Int4Args, job: JobArgs,
    _admitted: MapLimits, input: R, output: &mut impl Write) -> Result<(), Failure> {
    let started = Instant::now();
    job.storage.input_retention.validate(mode, &args.input)?;
    let map = job.validate(&args)?;
    let encoding = job.map.input_encoding(&args)?;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or(CandidateError::Timeout);
    let engine = NlpEngine::builder().resource_config(compute::resource_config(map.common))
        .build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    // Fixed settings and bounded final reporting retain this owner from
    // before private configuration IO until the entire invocation returns.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, map.cli_preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let lifetime = job_cli::parse_limits(&read_config(&job.storage.limits_file, job_cli::LIMIT_BYTES, &remaining)?)?;
    let limits = job.lifetime_limits(&args, map, lifetime)?;
    let map = limits.map; let common = map.common;
    let host = host_limits(&args, limits, remaining()?);
    if host.required_buffer_bytes_for_input(lifetime, job.storage.input_retention.input_memory()?)
        .map_err(host_failure)? != limits.buffer_bytes {
        return Err(Failure::usage("faithfulness_document_job_host_quote"));
    }
    let budget = job.map.budget_with_max_heads(common, job.faithfulness.max_evidence_heads.unwrap_or(32))?;
    let policy_text = read_config(&job.faithfulness.policy, super::super::corpus::POLICY_BYTES, &remaining)?;
    let policy = super::super::corpus::parse_policy_with_max_heads(&policy_text, job.faithfulness.max_evidence_heads)?;
    drop(policy_text);
    remaining()?;
    let key = JobSecret::read(&job.storage.key_file).map_err(job_failure)?;
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = crate::candidate_cli::int4::runtime::load_model_with_failure(&engine, &args, common,
        remaining()?, cancellation.clone(), model_failure)?;
    remaining()?;
    let compiler = Int4JudgePlanner::for_model(model.model_identity(), budget, map.planning)
        .and_then(|planner| planner.with_document_encoding(encoding.clone()))
        .map_err(|_| CandidateError::Planning)?;
    let planner = Int4FaithfulnessDocumentJobPlanner::new(compiler, policy, map.mapping, lifetime)
        .map_err(|_| CandidateError::Planning)?;
    let planner = if let Some(max_heads) = job.faithfulness.max_evidence_heads {
        planner.with_max_heads(max_heads).map_err(|_| CandidateError::Planning)?
    } else { planner };
    let cohort_chunks = job.map.mapping.cohort_chunks;
    let planner = if let Some(chunks) = cohort_chunks {
        planner.with_chunk_cohorts(chunks).map_err(|_| CandidateError::Planning)?
    } else { planner };
    planner.check_job_limits(lifetime).map_err(|_| CandidateError::Planning)?;
    if planner.max_result_bytes() != job.result_bytes(map)? || planner.document_encoding() != &encoding {
        return Err(Failure::usage("faithfulness_document_job_result_quote"));
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
    let job_id = job.storage.job_id; let materialize = job.storage.materialize;
    let request = SourceJobRequest { root: job.storage.job_dir, key, job_id, limits: lifetime,
        mode: open_mode(mode, job.storage.input_retention)?, materialize };
    let host = host_limits(&args, limits, remaining()?);
    let progress = if common.compute.threads == 1 {
        engine.job_int4_faithfulness_document(&model, planner, request, host, common.kernel, reader, cancellation)
    } else {
        engine.job_int4_faithfulness_document_with_threads(&model, planner, request, host, common.kernel,
            common.compute.threads, reader, cancellation)
    }.map_err(|error| match error { HostedJobError::Host(error) => model_failure(error),
        other => host_failure(other) })?;
    remaining()?;
    let report = progress_report(progress, job_id, lifetime, materialize, mode, "faithfulness")?;
    let content = identity.logical_model_digest.to_hex();
    let response = Response { protocol: "fnlp-candidate-int4-faithfulness-document-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "faithfulness",
        durable_unit: "complete_original_document", model_id: "Nanbeige4.2-3B",
        source_revision: &identity.source_revision, source_root_sha256: model.model_identity().source_root_sha256(),
        model_content_sha256: &content, candidate_format: &identity.artifact_format,
        quant_recipe: &identity.quant_recipe, group_width: model.model_identity().group_width(),
        integer_kernel: common.kernel.kind().id(), snapshot, cohort_chunks,
        compute_schedule: compute::Schedule::for_threads(common.compute.threads),
        max_evidence_heads: job.faithfulness.max_evidence_heads,
        input_encoding: input_encoding::reported_policy(&encoding), output: &report };
    // Failure can follow durable commits; stdout contains only metadata.
    publish_report(&response, output, &remaining)
}

fn model_failure(error: HostedError) -> Failure {
    let task_cancellation = match &error {
        HostedError::Int4(Int4HostError::FaithfulnessMap(error)) => error.cancellation(),
        HostedError::Int4(Int4HostError::Judge(error)) => error.cancellation(),
        _ => None,
    };
    if let Some(cause) = task_cancellation { return job_failure(JobError::Cancelled(cause)); }
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
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        transport: limits.transport, preparation_reserve_bytes: limits.preparation_reserve_bytes,
        io_reserve_bytes: limits.io_reserve_bytes, journal_reserve_bytes: limits.journal_reserve_bytes,
        serialization_reserve_bytes: limits.serialization_reserve_bytes }
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
    task: &'static str, durable_unit: &'static str, model_id: &'static str,
    source_revision: &'a str, source_root_sha256: &'a str, model_content_sha256: &'a str,
    candidate_format: &'a str, quant_recipe: &'a str, group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cohort_chunks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_schedule: Option<compute::Schedule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_evidence_heads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a crate::tokenizer::untrusted::UntrustedEncoding>,
    output: &'a T,
}

#[cfg(test)] mod tests;
