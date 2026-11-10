//! Preflight the exact original/partial schema pair and all actual chunks,
//! then publish one complete map-plus-object result under the original deadline.
use super::*;
use std::{sync::Arc, time::{Duration, Instant}};
use crate::{NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    batch::extract::grouped_int4::{Int4ExtractionBatchPlanner, long::{
        cohort::Int4ExtractionMapCohortStats,
        object::{Int4ObjectExtractionPlanner, Int4ObjectExtractionMapPreflight,
            Int4ObjectExtractionMapRun, Int4ObjectExtractionMapError}}},
    hosted::{CancellationToken, HostedError, Int4HostError, Int4ExtractionMapConfig,
        Int4ObjectExtractionMapConfig, NativeLimits, RunLimits},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        strict_int4::weights::snapshot::INT4_SNAPSHOT_VERSION},
    tasks::extract::{ExtractionVocabulary, object_reconcile::{ROOT_OBJECT_DERIVATION, ROOT_OBJECT_RECONCILIATION}},
};
struct DeadlineControl { started: Instant, deadline: Duration }
impl DecodeStepControl for DeadlineControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        (self.started.elapsed() >= self.deadline).then_some(DecodeCancellationKind::Deadline)
    }
}
fn cancelled(cause: DecodeCancellationKind) -> CandidateError {
    match cause { DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout => CandidateError::Timeout,
        _ => CandidateError::Execution }
}
fn map_planning_error(error: Int4ObjectExtractionMapError) -> CandidateError {
    error.cancellation().map(cancelled).unwrap_or(CandidateError::Planning)
}
fn execution_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => cancelled(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => execution_error(*error),
        HostedError::Int4(Int4HostError::ObjectExtractionMap(error)) =>
            error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution),
        HostedError::Reservation(_) => CandidateError::Memory,
        HostedError::ModelIdentity | HostedError::ResourceDomain => CandidateError::Identity,
        _ => CandidateError::Execution,
    }
}
fn preflight(source: &str, request: &ExtractionBatchArgs, planner: &Int4ObjectExtractionPlanner,
    limits: ObjectMapLimits, context_limit: usize, control: &mut impl DecodeStepControl)
    -> Result<Int4ObjectExtractionMapPreflight, CandidateError> {
    if source.is_empty() { return Err(CandidateError::Input); }
    let prepared = planner.plan_document_with_control(source, request, limits.mapping(), control).map_err(map_planning_error)?;
    if let Some(ceiling) = limits.map.cohort {
        let required = prepared.cohort_requirements(context_limit).map_err(map_planning_error)?;
        check_cohort_peak(required, ceiling)?;
    }
    Ok(prepared.preflight_metadata())
}
fn check_cohort_peak(required: Int4ExtractionMapCohortRequirement, ceiling: Int4ExtractionMapCohortRequirement)
    -> Result<(), CandidateError> {
    if required.native.kv_bytes > ceiling.native.kv_bytes
        || required.native.rope_bytes > ceiling.native.rope_bytes
        || required.native.scratch_payload_bound > ceiling.native.scratch_payload_bound
        || required.workspace_bytes > ceiling.workspace_bytes || required.max_result_bytes > ceiling.max_result_bytes {
        return Err(CandidateError::Memory);
    }
    Ok(())
}
pub(super) fn execute(args: Int4Args, options: ObjectMapArgs, _admitted: ObjectMapLimits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let started = Instant::now();
    let limits = options.validate(&args)?;
    let common = limits.map.common;
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
    // This bounded workspace funds original/partial schema parses as well as
    // preflight. Its later hosted replacement does not overlap this charge.
    let object_preparation = lease.reserve(MemoryClass::JobBuffers, limits.object_workspace_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    let schema = read_local(options.map.extraction.schema_path(), crate::candidate_cli::extract::SCHEMA_BYTES)?;
    let budget = options.map.extraction.budget(&args, common);
    let request = options.request(schema, budget, limits, &mut DeadlineControl { started, deadline })?;
    remaining()?;
    let text = if args.input.as_os_str() == "-" { read_input(input, args.max_input_bytes)? }
        else { read_local(&args.input, args.max_input_bytes)? };
    if text.is_empty() { return Err(CandidateError::Input); }
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = super::super::super::runtime::load_model(&engine, &args, common, remaining()?, cancellation.clone())?;
    remaining()?;
    let base = Int4ExtractionBatchPlanner::for_model(model.model_identity(), budget,
        options.map.extraction.compiler(), options.map.extraction.source_limits(), None).map_err(|_| CandidateError::Planning)?;
    let base = options.map.extraction.configure_planner(base)?;
    let planner = Int4ObjectExtractionPlanner::new(base).map_err(map_planning_error)?;
    let expected = preflight(&text, &request, &planner, limits, args.context_tokens,
        &mut DeadlineControl { started, deadline })?;
    let controls = crate::tokenizer::pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let vocabulary = ExtractionVocabulary::pinned(controls.template_controls()).map(Arc::new).map_err(|_| CandidateError::Planning)?;
    let identity = planner.base_execution_identity().clone();
    let mut config = Int4ObjectExtractionMapConfig { mapping: Int4ExtractionMapConfig {
        request, limits: limits.map.mapping, native: NativeLimits { context_tokens: args.context_tokens,
            allocator_reserve_bytes: ALLOCATOR_BYTES, run: RunLimits { max_elapsed: remaining()?,
                max_checkpoints: args.max_checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES } },
        preparation_reserve_bytes: common.preparation_bytes, reduction_reserve_bytes: limits.map.reduction_reserve_bytes,
    }, reconciliation: limits.object };
    if config.workspace_bytes(limits.map.mapping.mapping.chunks.max_chunks, options.map.extraction.compiler()).map_err(|_| CandidateError::Memory)?
        != limits.map.workspace_bytes { return Err(CandidateError::Memory); }
    if let Some(quote) = limits.map.cohort {
        if config.cohort_requirements(args.group_width).map_err(|_| CandidateError::Memory)? != quote {
            return Err(CandidateError::Memory);
        }
    }
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        // A valid saved candidate is independent of later data-dependent
        // conflicts or missing required fields; no output object is promised.
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    config.mapping.native.run.max_elapsed = remaining()?;
    drop(object_preparation);
    let content = identity.logical_model_digest.to_hex();
    let finish = |document: &Int4ObjectExtractionMapRun, cohort_schedule: Option<ObjectCohortSchedule<'_>>| {
        expected.verify_completed(document).map_err(|_| CandidateError::Execution)?;
        remaining()?;
        let response = ObjectResponse { schema_version: 1, scope: "real-source-current-candidate-int4",
            evidence: "non_authoritative", model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
            source_root_sha256: model.model_identity().source_root_sha256(), model_content_sha256: &content,
            candidate_format: &identity.artifact_format, quant_recipe: &identity.quant_recipe,
            group_width: model.model_identity().group_width(), integer_kernel: common.kernel.kind().id(),
            policy: ROOT_OBJECT_RECONCILIATION, schema_derivation: ROOT_OBJECT_DERIVATION,
            limits: limits.object, output: document, snapshot, cohort_schedule };
        // Keep the complete output guard alive until checked write and flush,
        // including reconciliation origins and the bounded cohort statistics.
        publish_checked(&response, limits.map.max_result_bytes + 4096, output, &|| remaining().map(|_| ()))
    };
    if let Some(max_chunks) = limits.map.cohort_chunks {
        let result = engine.extract_int4_object_document_cohorts(&model, text, Arc::new(planner), vocabulary,
            config, common.kernel, cancellation).map_err(execution_error)?;
        expected.verify_cohorts_completed(result.result()).map_err(|_| CandidateError::Execution)?;
        finish(&result.result().document, Some(ObjectCohortSchedule::new(max_chunks, &result.result().cohorts)))
    } else {
        let result = engine.extract_int4_object_document(&model, text, Arc::new(planner), vocabulary,
            config, common.kernel, cancellation).map_err(execution_error)?;
        finish(result.result(), None)
    }
}
fn read_local(path: &std::path::Path, cap: usize) -> Result<String, CandidateError> {
    read_input(&mut crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?, cap)
}
#[derive(Serialize)]
struct SnapshotRecord { operation: &'static str, format: &'static str, file_bytes: u64 }
impl SnapshotRecord {
    fn new(operation: &'static str, need: Int4SnapshotRequirements) -> Self {
        Self { operation, format: INT4_SNAPSHOT_VERSION, file_bytes: need.file_bytes }
    }
}
#[derive(Serialize)]
struct ObjectResponse<'a, T: Serialize> {
    schema_version: u32, scope: &'static str, evidence: &'static str, model_id: &'static str,
    source_revision: &'a str, source_root_sha256: &'a str, model_content_sha256: &'a str,
    candidate_format: &'a str, quant_recipe: &'a str, group_width: usize, integer_kernel: &'static str,
    policy: &'static str, schema_derivation: &'static str, limits: RootObjectLimits,
    output: &'a T,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cohort_schedule: Option<ObjectCohortSchedule<'a>>,
}
#[derive(Serialize)]
struct ObjectCohortSchedule<'a> {
    execution: &'static str,
    declared_max_chunks: usize,
    physical: &'a Int4ExtractionMapCohortStats,
}
impl<'a> ObjectCohortSchedule<'a> {
    fn new(declared_max_chunks: usize, physical: &'a Int4ExtractionMapCohortStats) -> Self {
        Self { execution: "cross_chunk_partial_object_cohorts", declared_max_chunks, physical }
    }
}

#[cfg(test)] mod tests;
