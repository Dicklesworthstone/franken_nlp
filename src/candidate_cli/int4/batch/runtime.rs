//! Complete fixed-cohort CLI execution through the existing charged INT4 host.
//! Input/plans and publication staging remain charged through write and flush.
use super::*;
use std::{fs::File, io::BufReader, time::{Duration, Instant}};
use crate::{NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    hosted::{CancellationToken, Int4LoadLimits, RunLimits, ResidentInt4,
        grouped_int4::batch::{Int4BatchHostLimits, Int4ChatBatchMember}},
    native_engine::strict_int4::{Int4ModelLoadBudget, weights::snapshot::INT4_SNAPSHOT_VERSION},
    tasks::{ir::TaskBudget, chat::{ChatLimits, GenerateRequest, ChatRequest,
        grouped_int4::{Int4ChatPlanner, batch::Int4ChatBatchLimits}}}};

pub(super) fn execute(args: Int4Args, batch: BatchArgs, limits: BatchLimits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    // Recompute the private quote before IO, so stale programmatic admission
    // cannot weaken the expanded resource envelope or select another kernel.
    let fresh = batch.validate(&args)?;
    if fresh.common.minimum_memory_bytes != limits.common.minimum_memory_bytes
        || fresh.publication_bytes != limits.publication_bytes
        || limits.common.kernel.kind() != args.kernel { return Err(CandidateError::Arguments); }
    let limits = fresh;
    let started = Instant::now(); let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|d| !d.is_zero()).ok_or(CandidateError::Timeout);
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: limits.common.memory_bytes,
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    let _preparation = lease.reserve(MemoryClass::JobBuffers, limits.common.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let records = if args.input.as_os_str() == "-" {
        batch.read(&mut BufReader::new(input), args.max_input_bytes, || remaining().map(|_| ()))?
    } else {
        remaining()?;
        let file = File::open(&args.input).map_err(|_| CandidateError::Input)?;
        batch.read(&mut BufReader::new(file), args.max_input_bytes, || remaining().map(|_| ()))?
    };
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = load(&engine, &args, limits.common, remaining()?, cancellation.clone())?;
    remaining()?;
    let planner = planner(&model, &args, limits.common)?;
    let (members, items) = prepare(records, &planner, &args, limits.common, || remaining().map(|_| ()))?;
    let identity = members.first().ok_or(CandidateError::Input)?.prepared.execution_identity().clone();
    let mut host_limits = Int4BatchHostLimits {
        task: Int4ChatBatchLimits { generation: limits.generation, max_result_bytes: limits.result_bytes },
        allocator_reserve_bytes: ALLOCATOR_BYTES,
        run: stage_run(remaining()?, args.max_checkpoints),
    };
    // Native/task bounds are checked on ALL real compiled plans before save
    // or execution. The host independently repeats these checks at reservation.
    host_limits.requirements(&members).map_err(|_| CandidateError::Planning)?;
    let mut snapshot = limits.common.snapshot_pin.map(|_| Snapshot { operation: "loaded",
        format: INT4_SNAPSHOT_VERSION, file_bytes: limits.common.snapshot_need.file_bytes });
    if let Some(path) = &args.save_snapshot {
        let pin = engine.save_int4_snapshot(&model, path.clone(), limits.common.snapshot_budget(false)?,
            stage_run(remaining()?, args.snapshot_checkpoints), cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(Snapshot { operation: "saved", format: INT4_SNAPSHOT_VERSION,
            file_bytes: limits.common.snapshot_need.file_bytes });
    }
    host_limits.run.max_elapsed = remaining()?;
    let result = engine.execute_int4_chat_batch(&model, members, host_limits, limits.common.kernel, cancellation)
        .map_err(|_| CandidateError::Execution)?;
    remaining()?;
    // The guard is live while its result is borrowed for serialization. No
    // prefix is emitted if any member fails planning/execution/finalization.
    let content = identity.logical_model_digest.to_hex();
    let response = Response {
        schema_version: 1, protocol: PROTOCOL, scope: "real-source-current-candidate-int4",
        evidence: "non_authoritative", model_id: "Nanbeige4.2-3B",
        source_revision: &identity.source_revision, source_root_sha256: model.model_identity().source_root_sha256(),
        model_content_sha256: &content, candidate_format: &identity.artifact_format, quant_recipe: &identity.quant_recipe,
        group_width: args.group_width, integer_kernel: limits.common.kernel.kind().id(),
        items: &items, output: result.result(), snapshot,
    };
    publish(&response, limits.publication_bytes, output)
}

fn stage_run(remaining: Duration, checkpoints: u64) -> RunLimits {
    RunLimits { max_elapsed: remaining, max_checkpoints: checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES }
}
fn load(engine: &NlpEngine, args: &Int4Args, limits: Int4Limits, remaining: Duration,
    cancellation: CancellationToken) -> Result<ResidentInt4, CandidateError> {
    match (&args.snapshot, limits.snapshot_pin) {
        (Some(path), Some(pin)) => engine.load_int4_snapshot(path.clone(), pin, limits.snapshot_budget(true)?,
            ALLOCATOR_BYTES, stage_run(remaining, args.snapshot_checkpoints), cancellation),
        (None, None) => engine.load_source_candidate_int4(
            args.source_dir.clone().ok_or(CandidateError::Arguments)?,
            args.source_manifest.clone().ok_or(CandidateError::Arguments)?, args.group_width,
            Int4LoadLimits { model: Int4ModelLoadBudget { max_payload_bytes: limits.max_load_bytes,
                max_source_values: limits.load.source_values, max_groups: limits.load.groups,
                max_shard_revalidation_bytes: limits.load.shard_revalidation_bytes },
                source_preparation_bytes: limits.source_preparation_bytes, allocator_reserve_bytes: ALLOCATOR_BYTES,
                run: stage_run(remaining, args.load_checkpoints) }, cancellation),
        _ => return Err(CandidateError::Arguments),
    }.map_err(|_| CandidateError::Model)
}
fn task_budget(args: &Int4Args, limits: Int4Limits) -> Result<TaskBudget, CandidateError> {
    Ok(TaskBudget {
        max_input_tokens: u32::try_from(limits.max_prompt_tokens).map_err(|_| CandidateError::Arguments)?,
        max_output_tokens: u32::try_from(args.max_new_tokens).map_err(|_| CandidateError::Arguments)?,
        max_output_bytes: limits.result_bytes as u64, max_grammar_states: 1, max_kv_bytes: limits.native.kv_bytes,
    })
}
fn chat_limits(args: &Int4Args, limits: Int4Limits) -> ChatLimits {
    ChatLimits { max_messages: 128, max_message_bytes: args.max_input_bytes,
        max_total_message_bytes: args.max_input_bytes, generation: args.generation_limits(limits) }
}
fn planner(model: &ResidentInt4, args: &Int4Args, limits: Int4Limits) -> Result<Int4ChatPlanner, CandidateError> {
    Int4ChatPlanner::for_model(model.model_identity(), task_budget(args, limits)?, chat_limits(args, limits))
        .map_err(|_| CandidateError::Planning)
}
#[derive(Serialize)]
struct Item { id: String, sample_index: u64, request_seq: u64 }
fn prepare(records: Vec<Record>, planner: &Int4ChatPlanner, args: &Int4Args, limits: Int4Limits,
    mut checkpoint: impl FnMut() -> Result<(), CandidateError>)
    -> Result<(Vec<Int4ChatBatchMember>, Vec<Item>), CandidateError> {
    if records.is_empty() || records.len() > MAX_INT4_SEQUENCES { return Err(CandidateError::Input); }
    let mut members = Vec::new(); let mut items = Vec::new();
    members.try_reserve_exact(records.len()).map_err(|_| CandidateError::Memory)?;
    items.try_reserve_exact(records.len()).map_err(|_| CandidateError::Memory)?;
    let options = args.options(planner.eos_token_id())?; let budget = task_budget(args, limits)?;
    for (index, record) in records.into_iter().enumerate() {
        checkpoint()?;
        let request_seq = index as u64 + 1;
        let prepared = match record.content {
            wire::Content::Generate(prompt) => planner.plan_generate(&GenerateRequest {
                item_id: record.id.clone(), sample_index: record.sample_index, prompt, generation: options.clone(), budget }),
            wire::Content::Chat(messages) => planner.plan_chat(&ChatRequest {
                item_id: record.id.clone(), sample_index: record.sample_index, messages, generation: options.clone(), budget }),
        }.map_err(|_| CandidateError::Planning)?;
        items.push(Item { id: record.id, sample_index: record.sample_index, request_seq });
        members.push(Int4ChatBatchMember { prepared, request_seq, context_tokens: args.context_tokens });
    }
    checkpoint()?;
    Ok((members, items))
}
#[derive(Serialize)]
struct Snapshot { operation: &'static str, format: &'static str, file_bytes: u64 }
#[derive(Serialize)]
struct Response<'a, T: Serialize> {
    schema_version: u32, protocol: &'static str, scope: &'static str, evidence: &'static str,
    model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str,
    model_content_sha256: &'a str, candidate_format: &'a str, quant_recipe: &'a str,
    group_width: usize, integer_kernel: &'static str, items: &'a [Item], output: &'a T,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<Snapshot>,
}

#[cfg(test)] mod tests;
