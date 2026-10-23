//! Owned population, real resident INT4 model and one authenticated job run.
use super::*;
use std::{io::{BufRead, BufReader}, sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    candidate_cli::{extract::SCHEMA_BYTES, runtime::owned_jobs::{
        host_failure, job_failure, open_mode, progress_report}},
    hosted::{CancellationToken, NativeLimits, RunLimits,
        corpus::jobs::{JobHostLimits, SourceJobRequest}},
    jobs::{JobSecret, runner::extract::grouped_int4::Int4ExtractionJobPlanner},
    native_engine::strict_int4::weights::snapshot::INT4_SNAPSHOT_VERSION,
    tasks::extract::ExtractionVocabulary,
};

pub(super) fn execute<R: Read + Send + 'static>(mode: RunMode, args: Int4Args,
    job: JobArgs, _admitted: Int4Limits, input: R, output: &mut impl Write) -> Result<(), Failure> {
    let started = Instant::now();
    // Recheck options at the owned boundary before private IO/runtime creation.
    let common = job.validate(&args)?;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or(CandidateError::Timeout);
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: common.memory_bytes,
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    // Config/key/schema/planner/IO and final report allocations drop BEFORE
    // their preparation charge, including errors after a durable job commit.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let lifetime = job_cli::parse_limits(&read_config(&job.storage.limits_file,
        job_cli::LIMIT_BYTES, &remaining)?)?;
    let limits = job.lifetime_limits(&args, common, lifetime)?;
    let common = limits.common;
    // Cross-check the CLI quote against the same hosted reservation calculator
    // BEFORE key/schema/model reads. Neither calculation is a memory permit.
    check_host_quote(host_limits(&args, limits, remaining()?), lifetime, limits.buffer_bytes)?;
    let key = JobSecret::read(&job.storage.key_file).map_err(job_failure)?;
    remaining()?;
    let schema = read_config(job.extraction.schema_path(), SCHEMA_BYTES, &remaining)?;
    let budget = job.extraction.budget(&args, common);
    let defaults = job.extraction.shared_defaults(schema, budget)?;
    remaining()?;
    let reader: Box<dyn BufRead + Send> = if args.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES,
            crate::local_io::open_document(&args.input).map_err(|_| CandidateError::Input)?))
    };
    let cancellation = CancellationToken::default();
    let model = super::super::runtime::load_model(&engine, &args, common, remaining()?, cancellation.clone())?;
    remaining()?;
    let planner = Int4ExtractionJobPlanner::for_model(model.model_identity(), budget,
        job.extraction.compiler(), job.extraction.source_limits(), defaults, limits.native)
        .map_err(|_| CandidateError::Planning)?;
    planner.check_job_limits(lifetime).map_err(|_| CandidateError::Planning)?;
    remaining()?;
    let controls = crate::tokenizer::pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let vocabulary = ExtractionVocabulary::pinned(controls.template_controls())
        .map(Arc::new).map_err(|_| CandidateError::Planning)?;
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
    let request = SourceJobRequest { root: job.storage.job_dir, key, job_id, limits: lifetime,
        mode: open_mode(mode), materialize };
    // Only the real host opens the protected job, authenticates the exact
    // population/recipe, skips committed ordinals and admits pending forwards.
    let progress = engine.job_int4_extract(&model, planner, vocabulary, request, host,
        common.kernel, reader, cancellation).map_err(host_failure)?;
    remaining()?;
    let report = progress_report(progress, job_id, lifetime, materialize, mode, "extract")?;
    let content = identity.logical_model_digest.to_hex();
    let response = Response {
        protocol: "fnlp-candidate-int4-extract-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(),
        model_content_sha256: &content, candidate_format: &identity.artifact_format,
        quant_recipe: &identity.quant_recipe, group_width: model.model_identity().group_width(),
        integer_kernel: common.kernel.kind().id(), snapshot, output: &report,
    };
    // Metadata only: result bytes remain in the explicit owner-only spool.
    // Delivery failure can follow a durable commit and never implies rollback.
    publish(&response, job_cli::REPORT_BYTES, output).map_err(Failure::from)
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
    if host.required_buffer_bytes(job).map_err(host_failure)? != expected {
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
    output: &'a T,
}

#[cfg(test)]
mod tests;
