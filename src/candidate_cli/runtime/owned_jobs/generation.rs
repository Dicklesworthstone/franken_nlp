//! Retained text uses the same pinned native compiler and protected job owner.
use super::*;
use crate::{
    batch::generation::quantized::Int8BatchLimits,
    candidate_cli::{jobs::generation::TextJobArgs, text_batch::TextTask},
    jobs::runner::generation::{GenerationJobConfig, GenerationJobTask, Int8GenerationJobPlanner},
    native_engine::strict_int8::prefill::{Int8PrefillLimits, LAYER_MAJOR_PREFILL_VERSION},
    tasks::chat::ChatLimits,
};

fn prepare(args: &TextJobArgs, limits: Limits, job: JobLimits, facts: &ArtifactIdentity)
    -> Result<Int8GenerationJobPlanner, Failure> {
    args.check_lifetime(job, limits)?;
    let controls = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let controls = controls.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END)
        .map(|e| e.id).ok_or(CandidateError::Identity)?;
    let task = match args.task { TextTask::Generate => GenerationJobTask::Generate, TextTask::Chat => GenerationJobTask::Chat };
    let mut identity = candidate_identity(facts)?; identity.task_spec = task.identity().into();
    let generation = args.common.options(eos)?;
    let budget = TaskBudget { max_input_tokens: limits.max_prompt_tokens as u32,
        max_output_tokens: args.common.max_new_tokens as u32, max_output_bytes: limits.result_bytes as u64,
        max_grammar_states: 4096, max_kv_bytes: limits.kv_bytes };
    let config = GenerationJobConfig { task, generation, budget,
        planning: ChatLimits { max_messages: 128, max_message_bytes: args.common.max_input_bytes,
            max_total_message_bytes: args.common.max_input_bytes, generation: args.common.generation_limits(limits) },
        // The immutable lifetime authority also bounds one physical invocation.
        // Native per-item estimates still debit all five lifetime counters.
        native: Int8BatchLimits { max_sampler_bytes: SAMPLER_BYTES, max_model_work: job.max_work.model } };
    Int8GenerationJobPlanner::pinned(controls, eos, identity, config)
        .map_err(|_| Failure::usage("text_job_pinned_configuration"))
}
fn host_limits(args: &TextJobArgs, limits: Limits, session: &Session) -> Result<JobHostLimits, Failure> {
    Ok(JobHostLimits { native: session.native(&args.common)?,
        transport: PopulationReadLimits { max_stream_bytes: args.max_stream_mib.checked_mul(MIB)
            .ok_or_else(|| Failure::usage("text_job_transport"))?, max_lines: args.max_input_lines },
        preparation_reserve_bytes: limits.preparation_bytes, io_reserve_bytes: (IO_BYTES * 2) as u64,
        journal_reserve_bytes: args.journal_memory_mib.checked_mul(MIB).ok_or_else(|| Failure::usage("text_job_memory"))?,
        serialization_reserve_bytes: args.serialization_memory_mib.checked_mul(MIB)
            .ok_or_else(|| Failure::usage("text_job_memory"))? })
}
pub(in crate::candidate_cli) fn execute<R: Read + Send + 'static>(mode: RunMode, args: TextJobArgs,
    limits: Limits, input: R, output: &mut impl Write) -> Result<(), Failure> {
    args.input_retention.validate(mode, &args.common.input)?;
    // First allocation owner survives all input, planner and report locals.
    let session = Session::new(&args.common, limits)?;
    let lifetime = command::parse_limits(&read_config(&session, &args.limits_file, command::LIMIT_BYTES)?)?;
    args.check_lifetime(lifetime, limits)?;
    let buffers = host_limits(&args, limits, &session)?.required_buffer_bytes_for_input(lifetime,
        args.input_retention.input_memory()?).map_err(host_failure)?;
    let complete = args.execution_memory_bytes(limits)?.checked_add(buffers)
        .ok_or_else(|| Failure::usage("text_job_memory_arithmetic"))?;
    if complete > limits.memory_bytes {
        return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "text_job_population_memory" });
    }
    let prefill = args.common.policy.prefill()?;
    let key = JobSecret::read(&args.key_file).map_err(job_failure)?;
    session.remaining()?;
    let facts = session.facts(&args.common)?;
    let planner = prepare(&args, limits, lifetime, &facts)?;
    session.remaining()?;
    let reader = command::population_reader(mode, args.input_retention, &args.common.input, input, IO_BYTES)?;
    let cancellation = CancellationToken::default();
    let model = session.load(&args.common, limits, &facts, cancellation.clone())?;
    let host = host_limits(&args, limits, &session)?;
    let task = args.task_name(); let job_id = args.job_id; let materialize = args.materialize;
    let request = SourceJobRequest { root: args.job_dir, key, job_id, limits: lifetime,
        mode: open_mode(mode, args.input_retention)?, materialize };
    let progress = match (limits.compute.threads, prefill) {
        (1, None) => session.engine.job_int8_text(&model, planner, request, host, reader, cancellation.clone()),
        (1, Some(prefill)) => session.engine.job_int8_text_layer_major(&model, planner, request,
            host, prefill, reader, cancellation.clone()),
        (threads, None) => session.engine.job_int8_text_with_threads(&model, planner, request, host,
            crate::native_engine::simd::config::process_kernel(), threads, reader, cancellation.clone()),
        (threads, Some(prefill)) => session.engine.job_int8_text_layer_major_with_threads(
            &model, planner, request, host, crate::native_engine::simd::config::process_kernel(),
            threads, prefill, reader, cancellation.clone()),
    }.map_err(host_failure)?;
    if let Some(cause) = cancellation.cause() { return Err(job_failure(JobError::Cancelled(cause))); }
    session.remaining()?;
    let report = progress_report(progress, job_id, lifetime, materialize, mode, task)?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: &report };
    // No generated content, prompt, token IDs, seed or raw logprobs on stdout.
    // A report delivery failure may follow committed durable progress.
    let response = ScheduledResponse { response: &response,
        prefill_schedule: prefill.map(PrefillSchedule::new).transpose()?,
        compute_schedule: ComputeSchedule::for_admission(limits.compute) };
    publish_report(&response, output, &|| session.remaining(), &cancellation)
}

#[derive(Serialize)]
struct PrefillSchedule { execution: &'static str, max_batch_rows: usize, admitted_workspace_bytes: u64 }
impl PrefillSchedule {
    fn new(limits: Int8PrefillLimits) -> Result<Self, Failure> {
        Ok(Self { execution: LAYER_MAJOR_PREFILL_VERSION, max_batch_rows: limits.max_batch_rows,
            admitted_workspace_bytes: limits.validate().map_err(|_| Failure::usage("text_job_prefill_quote"))? })
    }
}
/// Chosen physical schedule, not measured throughput or completed work. This
/// appears only after the hosted job returns from its physical worker drain.
#[derive(Serialize)]
struct ComputeSchedule {
    execution: &'static str,
    compute_threads: usize,
    admitted_workspace_bytes: u64,
}
impl ComputeSchedule {
    fn for_admission(admitted: compute::Admission) -> Option<Self> {
        (admitted.threads > 1).then_some(Self { execution: "int8-output-row-team-v1",
            compute_threads: admitted.threads, admitted_workspace_bytes: admitted.workspace_bytes })
    }
}
#[derive(Serialize)]
struct ScheduledResponse<'a, T: Serialize> {
    #[serde(flatten)] response: &'a T,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_schedule: Option<PrefillSchedule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_schedule: Option<ComputeSchedule>,
}
#[cfg(test)] mod tests;
