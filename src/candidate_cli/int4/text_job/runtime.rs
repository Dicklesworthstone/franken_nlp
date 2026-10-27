//! Actual resident model, pinned text compiler and one owned retained job.
use super::*;
use std::{io::{BufRead, BufReader}, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    candidate_cli::runtime::owned_jobs::{host_failure, job_failure, open_mode, progress_report},
    hosted::{CancellationToken, NativeLimits, ResidentInt4, RunLimits,
        corpus::jobs::{JobHostLimits, SourceJobRequest}},
    jobs::{JobSecret, runner::generation::grouped_int4::{Int4GenerationJobConfig, Int4GenerationJobPlanner}},
    native_engine::strict_int4::weights::snapshot::INT4_SNAPSHOT_VERSION,
    tasks::{chat::{ChatLimits, grouped_int4::Int4ChatPlanner}, ir::TaskBudget},
};

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
    // Declare ahead of every key/config/planner/reader allocation and keep it
    // until the final report has flushed, including errors after durable work.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let lifetime = job_cli::parse_limits(&read_config(&job.storage.limits_file,
        job_cli::LIMIT_BYTES, &remaining)?)?;
    let limits = job.lifetime_limits(&args, common, lifetime)?;
    let common = limits.common;
    check_host_quote(host_limits(&args, limits, remaining()?), lifetime, limits.buffer_bytes)?;
    let key = JobSecret::read(&job.storage.key_file).map_err(job_failure)?;
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
    let planner = prepare(&model, &job, &args, limits, lifetime)?;
    remaining()?;
    // Quote the actual native factory and complete host occupancy before the
    // optional snapshot creates any file. Only the host grants reservations.
    check_native_quote(&planner, &args, limits, host_limits(&args, limits, remaining()?), lifetime)?;
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
    let task = job.task_name(); let job_id = job.storage.job_id; let materialize = job.storage.materialize;
    let request = SourceJobRequest { root: job.storage.job_dir, key, job_id, limits: lifetime,
        mode: open_mode(mode), materialize };
    // The host authenticates the complete population and exact recipe before
    // replay/repair, executes only pending whole items and drains physically.
    let progress = engine.job_int4_generation(&model, planner, request, host,
        common.kernel, reader, cancellation).map_err(host_failure)?;
    remaining()?;
    let report = progress_report(progress, job_id, lifetime, materialize, mode, task)?;
    let content = identity.logical_model_digest.to_hex();
    let response = Response {
        protocol: "fnlp-candidate-int4-text-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(),
        model_content_sha256: &content, candidate_format: &identity.artifact_format,
        quant_recipe: &identity.quant_recipe, group_width: model.model_identity().group_width(),
        integer_kernel: common.kernel.kind().id(), snapshot, output: &report,
    };
    // No prompts, histories, seeds, tokens, generated text or scores on stdout.
    // Report delivery failure can follow committed progress and is not rollback.
    publish(&response, job_cli::REPORT_BYTES, output).map_err(Failure::from)
}

fn task_limits(args: &Int4Args, common: Int4Limits) -> Result<(TaskBudget, ChatLimits), Failure> {
    let budget = TaskBudget {
        max_input_tokens: u32::try_from(common.max_prompt_tokens).map_err(|_| CandidateError::Arguments)?,
        max_output_tokens: u32::try_from(args.max_new_tokens).map_err(|_| CandidateError::Arguments)?,
        max_output_bytes: common.result_bytes as u64, max_grammar_states: 1,
        max_kv_bytes: common.native.kv_bytes,
    };
    let planning = ChatLimits { max_messages: 128, max_message_bytes: args.max_input_bytes,
        max_total_message_bytes: args.max_input_bytes, generation: args.generation_limits(common) };
    Ok((budget, planning))
}
fn configuration(job: &JobArgs, args: &Int4Args, limits: JobExecutionLimits, eos: u32)
    -> Result<Int4GenerationJobConfig, Failure> {
    let (budget, planning) = task_limits(args, limits.common)?;
    Ok(Int4GenerationJobConfig { task: job.task, generation: args.options(eos)?,
        budget, planning, native: limits.native, context_tokens: args.context_tokens })
}
fn prepare(model: &ResidentInt4, job: &JobArgs, args: &Int4Args,
    limits: JobExecutionLimits, lifetime: JobLimits) -> Result<Int4GenerationJobPlanner, Failure> {
    let (budget, planning) = task_limits(args, limits.common)?;
    // This temporary pinned planner proves the tokenizer's terminal EOS from
    // the actual loaded model contract; no unbound fabricated model identity.
    let eos = Int4ChatPlanner::for_model(model.model_identity(), budget, planning)
        .map_err(|_| CandidateError::Planning)?.eos_token_id();
    Int4GenerationJobPlanner::for_model(model.model_identity(), configuration(job, args, limits, eos)?, lifetime)
        .map_err(|_| Failure::usage("text_job_pinned_configuration"))
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
fn check_native_quote(planner: &Int4GenerationJobPlanner, args: &Int4Args,
    limits: JobExecutionLimits, host: JobHostLimits, job: JobLimits) -> Result<(), Failure> {
    check_host_quote(host, job, limits.buffer_bytes)?;
    planner.check_job_limits(job).map_err(|_| CandidateError::Planning)?;
    let (task, _) = task_limits(args, limits.common)?;
    let native = planner.native_limits();
    let work = job.max_work.int4_model_work();
    if planner.context_tokens() != host.native.context_tokens || planner.context_tokens() != args.context_tokens
        || planner.group_width() != args.group_width || planner.task_budget() != task
        || planner.max_result_bytes() != task.max_output_bytes
        || planner.output_token_ceiling() != u64::from(task.max_output_tokens)
        || native.max_kv_bytes != task.max_kv_bytes || native.max_sampler_bytes != SAMPLER_BYTES
        || native.native.max_forward_positions != work.forward_positions
        || native.native.max_attention_pairs != work.attention_pairs
        || native.native.max_projection_work != work.projections {
        return Err(Failure::usage("text_job_native_quote"));
    }
    let quoted = simultaneous_memory(args.group_width, native, task, host, job, limits.common)?;
    if quoted > limits.common.minimum_memory_bytes || quoted > limits.common.memory_bytes {
        return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "text_job_native_memory" });
    }
    Ok(())
}
/// Price the host's independent full-context KV, native workspace/sampler and
/// output_claim while CLI preparation and complete job buffers coexist.
fn simultaneous_memory(width: usize, native: Int4GenerationBudget, task: TaskBudget,
    host: JobHostLimits, job: JobLimits, common: Int4Limits) -> Result<u64, Failure> {
    let need = Int4MemoryRequirement::for_context(host.native.context_tokens, width)
        .map_err(|_| Failure::usage("text_job_native_quote"))?;
    if need != common.native || native.max_kv_bytes != need.kv_bytes || task.max_kv_bytes != need.kv_bytes
        || native.max_sampler_bytes == 0 || host.native.allocator_reserve_bytes == 0 {
        return Err(Failure::usage("text_job_native_quote"));
    }
    sum(&[common.preparation_bytes, common.load.weight_payload_bytes, ALLOCATOR_BYTES,
        host.required_buffer_bytes(job).map_err(host_failure)?,
        need.kv_bytes, need.rope_bytes, need.scratch_payload_bound,
        host.native.allocator_reserve_bytes, native.max_sampler_bytes,
        product(task.max_output_bytes, 4)?, product(u64::from(task.max_output_tokens), 8)?, 4096,
        host.native.run.cleanup_reserve_bytes])
        .map_err(|_| Failure::usage("job_memory_arithmetic"))
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
    model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str,
    model_content_sha256: &'a str, candidate_format: &'a str, quant_recipe: &'a str,
    group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    output: &'a T,
}

#[cfg(test)]
mod tests;
