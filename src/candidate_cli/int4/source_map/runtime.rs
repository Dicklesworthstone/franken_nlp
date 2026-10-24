//! Whole-source preflight followed by one guarded native map, summary or QA.
use super::*;
use std::{sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    execution_identity::ExecutionIdentity,
    hosted::{CancellationToken, HostedError, Int4HostError, Int4SourceMapConfig, NativeLimits, RunLimits},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        strict_int4::weights::snapshot::INT4_SNAPSHOT_VERSION},
    tasks::{ir::PlanContext, source_planning::{SourceTaskPlanner,
        grouped_int4::long::{Int4SourceMapError, Int4SourceMapPreflight,
            question::{SourceQuestion, Int4QuestionPreflight}}}},
};

enum Request { Map(SourceMapTask), Question(SourceQuestion) }
enum Prepared {
    Map(Int4SourceMapConfig, Int4SourceMapPreflight),
    Question(Int4SourceMapConfig<SourceQuestion>, Int4QuestionPreflight),
}
fn configuration<T>(identity: ExecutionIdentity, task: T, args: &Int4Args,
    options: &MapArgs, limits: MapLimits, elapsed: Duration) -> Int4SourceMapConfig<T> {
    Int4SourceMapConfig { identity, task, budget: options.source.budget(args, limits.common),
        planning: options.source.planning(args), mapping: limits.mapping,
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: elapsed, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        preparation_reserve_bytes: limits.common.preparation_bytes,
        reduction_reserve_bytes: limits.reduction_reserve_bytes }
}
struct DeadlineControl { started: Instant, deadline: Duration }
impl DecodeStepControl for DeadlineControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        (self.started.elapsed() >= self.deadline).then_some(DecodeCancellationKind::Deadline)
    }
}
fn preflight(source: &str, planner: &SourceTaskPlanner, config: &Int4SourceMapConfig,
    width: usize, summary: Option<Int4SummaryLimits>, control: &mut impl DecodeStepControl)
    -> Result<Int4SourceMapPreflight, CandidateError> {
    if source.is_empty() { return Err(CandidateError::Input); }
    let context = PlanContext::new(&config.identity, config.budget).map_err(|_| CandidateError::Planning)?;
    let prepared = planner.plan_int4_map_with_control(source, &config.task, config.budget, &context,
        config.planning, config.mapping, width, control).map_err(planning_error)?;
    if let Some(summary) = summary { prepared.check_summary(summary).map_err(|_| CandidateError::Planning)?; }
    // Only bounded geometry/work metadata survives. The host replans the same
    // original document under its own charge before native work can start.
    Ok(prepared.preflight_metadata())
}
fn question_preflight(source: &str, planner: &SourceTaskPlanner,
    config: &Int4SourceMapConfig<SourceQuestion>, width: usize, control: &mut impl DecodeStepControl)
    -> Result<Int4QuestionPreflight, CandidateError> {
    if source.is_empty() { return Err(CandidateError::Input); }
    let context = PlanContext::new(&config.identity, config.budget).map_err(|_| CandidateError::Planning)?;
    planner.preflight_int4_question_with_control(source, &config.task, config.budget, &context,
        config.planning, config.mapping, width, control).map_err(planning_error)
}
fn cancelled(cause: DecodeCancellationKind) -> CandidateError {
    match cause {
        DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout => CandidateError::Timeout,
        _ => CandidateError::Execution,
    }
}
fn planning_error(error: Int4SourceMapError) -> CandidateError {
    error.cancellation().map(cancelled).unwrap_or(CandidateError::Planning)
}
fn execution_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => cancelled(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => execution_error(*error),
        HostedError::Int4(Int4HostError::SourceMap(error)) =>
            error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution),
        HostedError::Int4(Int4HostError::SourceSummary(error)) =>
            error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution),
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
    let _preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    let option_json = options.source.options_path().map(|path| {
        read_input(&mut crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?,
            shared::OPTIONS_BYTES)
    }).transpose()?;
    remaining()?;
    let text = if args.input.as_os_str() == "-" { read_input(input, args.max_input_bytes)? }
        else { read_input(&mut crate::local_io::open_document(&args.input)
            .map_err(|_| CandidateError::Input)?, args.max_input_bytes)? };
    remaining()?;
    let budget = options.source.budget(&args, common);
    let (text, task) = if let Some(question) = &options.question {
        if text.trim().is_empty() { return Err(CandidateError::Input); }
        let question_text = read_input(&mut crate::local_io::open_document(question.path())
            .map_err(|_| CandidateError::Input)?, question::QUESTION_BYTES)?;
        remaining()?;
        (text, Request::Question(question.task(question_text, option_json.as_deref())?))
    } else {
        let (text, task) = options.request(text, option_json.as_deref(), budget, args.max_input_bytes)?;
        (text, Request::Map(task))
    };
    drop(option_json);
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = super::super::runtime::load_model(&engine, &args, common, remaining()?, cancellation.clone())?;
    remaining()?;
    let (planner, vocabulary) = source::planner()?;
    let identity = source::identity(model.model_identity(), &planner, source::task(options.kind))?;
    let prepared = match task {
        Request::Map(task) => {
            let config = configuration(identity.clone(), task, &args, &options, limits, remaining()?);
            if config.workspace_bytes(limits.mapping.chunks.max_chunks).map_err(|_| CandidateError::Memory)?
                != limits.workspace_bytes { return Err(CandidateError::Memory); }
            let expected = preflight(&text, &planner, &config, model.model_identity().group_width(),
                limits.summary, &mut DeadlineControl { started, deadline })?;
            Prepared::Map(config, expected)
        }
        Request::Question(task) => {
            let config = configuration(identity.clone(), task, &args, &options, limits, remaining()?);
            if config.question_workspace_bytes(limits.mapping.chunks.max_chunks)
                .map_err(|_| CandidateError::Memory)? != limits.workspace_bytes {
                return Err(CandidateError::Memory);
            }
            let expected = question_preflight(&text, &planner, &config,
                model.model_identity().group_width(), &mut DeadlineControl { started, deadline })?;
            Prepared::Question(config, expected)
        }
    };
    remaining()?;
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        // Persist only after the COMPLETE original source and all native work
        // preflight. This complete model file survives later task failure.
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    let content = identity.logical_model_digest.to_hex();
    let provenance = Provenance {
        protocol: if options.kind == Kind::Answer { "fnlp-candidate-int4-document-qa-v1" }
            else { "fnlp-candidate-int4-source-map-v1" }, schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: options.kind.name(),
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(), model_content_sha256: &content,
        candidate_format: &identity.artifact_format, quant_recipe: &identity.quant_recipe,
        group_width: model.model_identity().group_width(), integer_kernel: common.kernel.kind().id(), snapshot,
    };
    match prepared {
        Prepared::Map(mut config, expected) => {
            config.native.run.max_elapsed = remaining()?;
            if let Some(summary) = limits.summary {
                let result = engine.summarize_int4_source(&model, text, Arc::new(planner), Arc::new(vocabulary),
                    config, summary, common.kernel, cancellation).map_err(execution_error)?;
                expected.verify_summary_completed(result.result()).map_err(|_| CandidateError::Execution)?;
                remaining()?;
                publish(&Response { provenance: &provenance, output: result.result() },
                    limits.max_result_bytes + 4096, output)
            } else {
                let result = engine.map_int4_source(&model, text, Arc::new(planner), Arc::new(vocabulary),
                    config, common.kernel, cancellation).map_err(execution_error)?;
                expected.verify_completed(result.result()).map_err(|_| CandidateError::Execution)?;
                remaining()?;
                publish(&Response { provenance: &provenance, output: result.result() },
                    limits.max_result_bytes + 4096, output)
            }
        }
        Prepared::Question(mut config, expected) => {
            config.native.run.max_elapsed = remaining()?;
            let result = engine.answer_int4_document(&model, text, Arc::new(planner), Arc::new(vocabulary),
                config, common.kernel, cancellation).map_err(execution_error)?;
            expected.verify_completed(result.result()).map_err(|_| CandidateError::Execution)?;
            remaining()?;
            // The hosted output guard survives staging, write AND flush.
            // Native failure cannot publish a successful prefix or abstention.
            publish(&Response { provenance: &provenance, output: result.result() },
                limits.max_result_bytes + 4096, output)
        }
    }
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
struct Response<'a, 'b, T: Serialize> {
    #[serde(flatten)]
    provenance: &'a Provenance<'b>,
    output: &'a T,
}

#[cfg(test)]
mod tests;
