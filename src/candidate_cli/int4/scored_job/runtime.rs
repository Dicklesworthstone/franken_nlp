//! The existing authenticated job runner owns retained results and resume.
use super::*;
use std::{io::{BufRead, BufReader}, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    candidate_cli::runtime::owned_jobs::{host_failure, job_failure, open_mode, progress_report},
    hosted::{CancellationToken, NativeLimits, ResidentInt4, RunLimits,
        corpus::jobs::{JobHostLimits, SourceJobRequest}},
    jobs::{JobSecret, runner::scored::grouped_int4::{Int4ScoredJobPlanner, Int4ScoredJobTask}},
    native_engine::strict_int4::weights::snapshot::INT4_SNAPSHOT_VERSION,
    tasks::{classify::grouped_int4::Int4ClassificationPlanner,
        sentiment::grouped_int4::Int4SentimentPlanner, judge::grouped_int4::Int4JudgePlanner},
};
use super::super::scored_batch::runtime::sentiment_options;

pub(super) fn execute<R: Read + Send + 'static>(mode: RunMode, args: Int4Args,
    job: JobArgs, _admitted: Int4Limits, input: R, output: &mut impl Write) -> Result<(), Failure> {
    let started = Instant::now();
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
    // Settings, key, factory, reader and final report drop before this charge.
    // The host independently retains its transferred inputs through drain.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let lifetime = job_cli::parse_limits(&read_config(&job.storage.limits_file,
        job_cli::LIMIT_BYTES, &remaining)?)?;
    let limits = job.lifetime_limits(&args, common, lifetime)?;
    let common = limits.common;
    check_host_quote(host_limits(&args, limits, remaining()?), lifetime, limits.buffer_bytes)?;
    let key = JobSecret::read(&job.storage.key_file).map_err(job_failure)?;
    remaining()?;
    let text = job.score.defaults.as_ref().map(|path|
        read_config(path, shared::DEFAULTS_BYTES, &remaining)).transpose()?;
    let budget = job.scoring.budget(common);
    let defaults = job.load_defaults(text.as_deref(), budget, &args)?;
    drop(text);
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
    match defaults {
        Defaults::Classify(defaults) => {
            let planner = Int4ClassificationPlanner::for_model(model.model_identity(), budget,
                job.scoring.classification_limits(&args)).map_err(|_| CandidateError::Planning)?;
            let planner = Int4ScoredJobPlanner::new(planner, defaults, limits.native)
                .map_err(|_| CandidateError::Planning)?;
            run(&engine, &model, planner, "classify", mode, &args, job, limits, lifetime,
                key, reader, output, started, deadline, cancellation)
        }
        Defaults::Sentiment { args: defaults, policy } => {
            let planner = Int4SentimentPlanner::for_model(model.model_identity(), sentiment_options(policy)?, budget,
                job.scoring.sentiment_limits(&args)).map_err(|_| CandidateError::Planning)?;
            let planner = Int4ScoredJobPlanner::new(planner, Some(defaults), limits.native)
                .map_err(|_| CandidateError::Planning)?;
            run(&engine, &model, planner, "sentiment", mode, &args, job, limits, lifetime,
                key, reader, output, started, deadline, cancellation)
        }
        Defaults::Judge(defaults) => {
            let planner = Int4JudgePlanner::for_model(model.model_identity(), budget,
                job.scoring.judge_limits(budget)).map_err(|_| CandidateError::Planning)?;
            let planner = Int4ScoredJobPlanner::new(planner, defaults, limits.native)
                .map_err(|_| CandidateError::Planning)?;
            run(&engine, &model, planner, "judge", mode, &args, job, limits, lifetime,
                key, reader, output, started, deadline, cancellation)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run<P: Int4ScoredJobTask>(engine: &NlpEngine, model: &ResidentInt4, planner: Int4ScoredJobPlanner<P>,
    task: &'static str, mode: RunMode, args: &Int4Args, job: JobArgs, limits: JobExecutionLimits,
    lifetime: JobLimits, key: JobSecret, reader: Box<dyn BufRead + Send>, output: &mut impl Write,
    started: Instant, deadline: Duration, cancellation: CancellationToken) -> Result<(), Failure> {
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or(CandidateError::Timeout);
    planner.check_job_limits(lifetime).map_err(|_| CandidateError::Planning)?;
    let identity = planner.execution_identity().clone();
    remaining()?;
    let common = limits.common;
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        // This checkpoint stores the complete model after factory admission.
        // It survives a later population/authentication/execution failure.
        let pin = engine.save_int4_snapshot(model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity.into()); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    let host = host_limits(args, limits, remaining()?);
    let job_id = job.storage.job_id;
    let materialize = job.storage.materialize;
    let request = SourceJobRequest { root: job.storage.job_dir, key, job_id, limits: lifetime,
        mode: open_mode(mode), materialize };
    let progress = engine.job_int4_scored(model, planner, request, host, common.kernel,
        reader, cancellation).map_err(host_failure)?;
    remaining()?;
    let report = progress_report(progress, job_id, lifetime, materialize, mode, task)?;
    let content = identity.logical_model_digest.to_hex();
    let response = Response {
        protocol: "fnlp-candidate-int4-score-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task,
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(),
        model_content_sha256: &content, candidate_format: &identity.artifact_format,
        quant_recipe: &identity.quant_recipe, group_width: model.model_identity().group_width(),
        integer_kernel: common.kernel.kind().id(), snapshot, output: &report,
    };
    // A report-write error can follow durable commits. No private scores,
    // labels, documents, paths or keys are included, and rollback is not implied.
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
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str, task: &'static str,
    model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str, model_content_sha256: &'a str,
    candidate_format: &'a str, quant_recipe: &'a str, group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    output: &'a T,
}

#[cfg(test)]
mod tests;
