//! Whole-document preflight, one owned native map and complete publication.
use super::*;
use std::time::{Duration, Instant};
use crate::{NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    hosted::{CancellationToken, HostedError, Int4HostError, Int4SentimentMapHostConfig,
        NativeLimits, RunLimits},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        generation::grouped_int4::INT4_SOURCE_FORMAT,
        strict_int4::weights::snapshot::INT4_SNAPSHOT_VERSION},
    tasks::sentiment::grouped_int4::{Int4SentimentError, Int4SentimentPlanner,
        long::Int4SentimentMapError}};
use super::super::scored_batch::runtime::sentiment_options;

struct DeadlineControl { started: Instant, deadline: Duration }
impl DecodeStepControl for DeadlineControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        (self.started.elapsed() >= self.deadline).then_some(DecodeCancellationKind::Deadline)
    }
}
fn stopped(cause: DecodeCancellationKind) -> CandidateError {
    match cause {
        DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout => CandidateError::Timeout,
        _ => CandidateError::Execution,
    }
}
fn planning_error(error: Int4SentimentMapError) -> CandidateError {
    error.cancellation().map(stopped).unwrap_or(CandidateError::Planning)
}
fn sentiment_error(error: Int4SentimentError) -> CandidateError {
    error.cancellation().map(stopped).unwrap_or(CandidateError::Planning)
}
fn execution_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => stopped(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => execution_error(*error),
        HostedError::Int4(Int4HostError::SentimentMap(error)) =>
            error.cancellation().map(stopped).unwrap_or(CandidateError::Execution),
        HostedError::Reservation(_) => CandidateError::Memory,
        HostedError::ModelIdentity | HostedError::ResourceDomain => CandidateError::Identity,
        _ => CandidateError::Execution,
    }
}

pub(super) fn execute(args: Int4Args, options: MapArgs, _admitted: MapLimits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let started = Instant::now();
    let limits = options.validate(&args)?;
    let common = limits.common;
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
    let _preparation = lease.reserve(MemoryClass::JobBuffers, limits.cli_preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    let text = if args.input.as_os_str() == "-" { read_input(input, args.max_input_bytes)? }
        else { read_input(&mut crate::local_io::open_document(&args.input)
            .map_err(|_| CandidateError::Input)?, args.max_input_bytes)? };
    remaining()?;
    let budget = options.budget(common)?;
    let (request, policy) = sentiment::parse_request(&text, budget, args.max_input_bytes)?;
    options.admit_request(&request)?;
    drop(text);
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = super::super::runtime::load_model(&engine, &args, common, remaining()?, cancellation.clone())?;
    remaining()?;
    let mut config = Int4SentimentMapHostConfig {
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining()?, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        planning: limits.planning, mapping: limits.mapping, options: sentiment_options(policy)?,
        preparation_reserve_bytes: common.preparation_bytes,
    };
    // Ordinary execution plans each chunk once, inside the charged host. An
    // explicit snapshot save additionally requires complete preflight before
    // creating the requested file; that temporary preparation is then dropped.
    let expected = if args.save_snapshot.is_some() {
        let preflight_bytes = sum(&[limits.memory.input_bytes, limits.memory.workspace_bytes])?;
        let _preflight = lease.reserve(MemoryClass::JobBuffers, preflight_bytes)
            .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
        let planner = Int4SentimentPlanner::for_model(model.model_identity(), config.options, budget,
            config.planning).map_err(sentiment_error)?;
        let prepared = planner.plan_document_with_control(&request, config.mapping,
            &mut DeadlineControl { started, deadline }).map_err(planning_error)?;
        if prepared.head_count() != request.axes.len() || prepared.required_context() > args.context_tokens {
            return Err(CandidateError::Identity);
        }
        Some(prepared.preflight_metadata())
    } else { None };
    remaining()?;
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    config.native.run.max_elapsed = remaining()?;
    let model_identity = model.model_identity();
    let content: String = model_identity.content_sha256().iter().map(|byte| format!("{byte:02x}")).collect();
    let recipe = format!("{}:group={}", model_identity.matrix_recipe(), model_identity.group_width());
    let provenance = Provenance { protocol: "fnlp-candidate-int4-sentiment-map-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "sentiment-map",
        model_id: "Nanbeige4.2-3B", source_revision: model_identity.revision(),
        source_root_sha256: model_identity.source_root_sha256(), model_content_sha256: &content,
        candidate_format: INT4_SOURCE_FORMAT, quant_recipe: &recipe,
        group_width: model.model_identity().group_width(), integer_kernel: common.kernel.kind().id(), snapshot };
    if let Some(chunks) = options.mapping.cohort_chunks {
        let result = engine.map_int4_sentiment_cohorts(&model, request, config, chunks,
            common.kernel, cancellation).map_err(execution_error)?;
        if let Some(expected) = expected {
            expected.verify_completed(&result.result().output).map_err(|_| CandidateError::Execution)?;
        }
        remaining()?;
        let cohort = Cohort { schema_version: result.result().schema_version,
            execution: result.result().execution.clone(), max_chunks: chunks, chunk_cohorts: result.result().chunk_cohorts,
            group_steps: result.result().group_steps, projection_groups: result.result().projection_groups };
        publish_complete(&Response { provenance: &provenance, output: &result.result().output, cohort: Some(cohort) },
            limits.envelope_bytes, output, &remaining)
    } else {
        let result = engine.map_int4_sentiment(&model, request, config, common.kernel, cancellation)
            .map_err(execution_error)?;
        if let Some(expected) = expected {
            expected.verify_completed(result.result()).map_err(|_| CandidateError::Execution)?;
        }
        remaining()?;
        publish_complete(&Response { provenance: &provenance, output: result.result(), cohort: None },
            limits.envelope_bytes, output, &remaining)
    }
}

/// A slow final writer/flush still consumes the end-to-end deadline. Callers
/// retain their guarded native result and publication reservation until return.
fn publish_complete<T: Serialize>(value: &T, cap: usize, output: &mut impl Write,
    remaining: &impl Fn() -> Result<Duration, CandidateError>) -> Result<(), CandidateError> {
    publish(value, cap, output)?;
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
struct Provenance<'a> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str, task: &'static str,
    model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str, model_content_sha256: &'a str,
    candidate_format: &'a str, quant_recipe: &'a str, group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
}
#[derive(Serialize)]
struct Cohort {
    schema_version: u32, execution: String,
    max_chunks: usize, chunk_cohorts: usize, group_steps: u64, projection_groups: u64,
}
#[derive(Serialize)]
struct Response<'a, 'b, T: Serialize> {
    #[serde(flatten)]
    provenance: &'a Provenance<'b>,
    output: &'a T,
    #[serde(skip_serializing_if = "Option::is_none")]
    cohort: Option<Cohort>,
}

#[cfg(test)]
mod tests;
