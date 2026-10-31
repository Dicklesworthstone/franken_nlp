//! One authenticated population, resident engine and whole-document job runner.
use super::*;
use std::{io::{BufRead, BufReader}, sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    candidate_cli::runtime::owned_jobs::{host_failure, job_failure, open_mode, progress_report},
    hosted::{CancellationToken, NativeLimits, RunLimits, corpus::jobs::{JobHostLimits, SourceJobRequest}},
    jobs::{JobSecret, runner::source::grouped_int4::document::{Int4SourceDocumentJobPlanner, Int4SourceDocumentLifetime}},
    native_engine::strict_int4::weights::snapshot::INT4_SNAPSHOT_VERSION,
};

pub(super) fn execute<R: Read + Send + 'static>(mode: RunMode, args: Int4Args, job: JobArgs,
    _admitted: MapLimits, input: R, output: &mut impl Write) -> Result<(), Failure> {
    let started = Instant::now();
    let map = job.validate(&args)?;
    let common = map.common;
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
    // Configuration, question, tokenizer, vocabulary and publication staging
    // retain this owner while the host separately retains transferred inputs.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let lifetime = job_cli::parse_limits(&read_config(&job.storage.limits_file, job_cli::LIMIT_BYTES, &remaining)?)?;
    let limits = job.lifetime_limits(&args, map, lifetime)?;
    let map = limits.map; let common = map.common;
    check_host_quote(host_limits(&args, limits, remaining()?), lifetime, limits.buffer_bytes)?;
    let key = JobSecret::read(&job.storage.key_file).map_err(job_failure)?;
    remaining()?;
    let options = job.map.source.options_path().map(|path|
        read_config(path, crate::candidate_cli::source::OPTIONS_BYTES, &remaining)).transpose()?;
    let question = job.map.question_path().map(|path|
        read_config(path, crate::candidate_cli::source::OPTIONS_BYTES, &remaining)).transpose()?;
    let task = job.map.document_task(&args, map, options.as_deref(), question, job.keyphrase_scan_cap())?;
    drop(options);
    let document_mode = job.mode_name(map);
    let budget = job.map.source.budget(&args, common);
    let planning = job.map.planning(&args);
    let non_model_lifetime = Int4SourceDocumentLifetime { verification: job.lifetime.verification(),
        summary_scan_steps: job.lifetime.max_lifetime_summary_scan_steps };
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = super::super::runtime::load_model(&engine, &args, common, remaining()?, cancellation.clone())?;
    remaining()?;
    let (compiler, vocabulary) = source::planner()?;
    let planner = Int4SourceDocumentJobPlanner::for_model(model.model_identity(), Arc::new(compiler), task,
        budget, planning, map.mapping, lifetime, non_model_lifetime).map_err(|_| CandidateError::Planning)?;
    let planner = if map.cohort_chunks.is_some() {
        planner.with_chunk_cohorts().map_err(|_| CandidateError::Planning)?
    } else { planner };
    let cohort = planner.uses_chunk_cohorts().then(|| planner.cohort_requirements())
        .transpose().map_err(|_| CandidateError::Memory)?;
    if cohort != map.cohort { return Err(Failure::usage("document_job_cohort_quote")); }
    planner.check_job_limits(lifetime).map_err(|_| CandidateError::Planning)?;
    let identity = planner.execution_identity().clone();
    let host = host_limits(&args, limits, remaining()?);
    if planner.workspace_bytes(host.native, map.reduction_reserve_bytes)
        .map_err(|_| CandidateError::Memory)? != map.workspace_bytes {
        return Err(Failure::usage("document_job_workspace_quote"));
    }
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        // This complete model checkpoint follows immutable factory admission.
        // Whole-population/document preflight occurs in the host; this model
        // file survives a later population, authentication or execution error.
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
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
    let task_name = job.map.kind.name();
    let job_id = job.storage.job_id; let materialize = job.storage.materialize;
    let request = SourceJobRequest { root: job.storage.job_dir, key, job_id, limits: lifetime,
        mode: open_mode(mode), materialize };
    let progress = engine.job_int4_source_document(&model, planner, Arc::new(vocabulary), request,
        host_limits(&args, limits, remaining()?), map.reduction_reserve_bytes, common.kernel,
        reader, cancellation).map_err(host_failure)?;
    remaining()?;
    let report = progress_report(progress, job_id, lifetime, materialize, mode, task_name)?;
    let content = identity.logical_model_digest.to_hex();
    let response = Response {
        protocol: "fnlp-candidate-int4-document-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: task_name, document_mode,
        durable_unit: "complete_original_document", model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(), model_content_sha256: &content,
        candidate_format: &identity.artifact_format, quant_recipe: &identity.quant_recipe,
        group_width: model.model_identity().group_width(), integer_kernel: common.kernel.kind().id(),
        snapshot, chunk_schedule: ChunkSchedule::new(map), output: &report,
    };
    // A metadata write failure can follow durable commits. No private source,
    // question, quote, task options, filesystem path or key is published here.
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
struct ChunkSchedule {
    execution: &'static str,
    phase: &'static str,
    declared_max_chunks: usize,
}
impl ChunkSchedule {
    fn new(map: MapLimits) -> Option<Self> {
        map.cohort_chunks.map(|declared_max_chunks| Self {
            execution: "native_chunk_cohorts",
            phase: if map.synthesis.is_some() || map.question_synthesis.is_some() { "discovery" } else { "map" },
            declared_max_chunks,
        })
    }
}
#[derive(Serialize)]
struct Response<'a, T: Serialize> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str,
    task: &'static str, document_mode: &'static str, durable_unit: &'static str,
    model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str, model_content_sha256: &'a str,
    candidate_format: &'a str, quant_recipe: &'a str, group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chunk_schedule: Option<ChunkSchedule>,
    output: &'a T,
}

#[cfg(test)]
mod tests;
