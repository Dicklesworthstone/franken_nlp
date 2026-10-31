//! Full-document preflight and one guarded hosted INT4 map/reduce invocation.
use super::*;
use std::{fs::File, sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    batch::extract::grouped_int4::{Int4ExtractionBatchPlanner,
        long::{Int4ExtractionMapPreflight, Int4ExtractionMapError, Int4ExtractionMapRun,
            cohort::Int4ExtractionMapCohortStats}},
    hosted::{CancellationToken, Int4ExtractionMapConfig, NativeLimits, RunLimits},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        strict_int4::weights::snapshot::INT4_SNAPSHOT_VERSION},
    tasks::extract::ExtractionVocabulary,
};

struct DeadlineControl { started: Instant, deadline: Duration }
impl DecodeStepControl for DeadlineControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        (self.started.elapsed() >= self.deadline).then_some(DecodeCancellationKind::Deadline)
    }
}
fn preflight(source: &str, request: &ExtractionBatchArgs, planner: &Int4ExtractionBatchPlanner,
    mapping: Int4ExtractionMapLimits, control: &mut impl DecodeStepControl)
    -> Result<Int4ExtractionMapPreflight, CandidateError> {
    if source.is_empty() { return Err(CandidateError::Input); }
    let prepared = planner.plan_document_with_control(source, request, mapping, control).map_err(planning_error)?;
    // Retain geometry and work only. The host recompiles under its own charge
    // and verifies every chunk before any native work starts.
    Ok(prepared.preflight_metadata())
}
fn planning_error(error: Int4ExtractionMapError) -> CandidateError {
    match error.cancellation() {
        Some(DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout) => CandidateError::Timeout,
        Some(_) => CandidateError::Execution,
        None => CandidateError::Planning,
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
    let schema = read_input(&mut File::open(options.extraction.schema_path())
        .map_err(|_| CandidateError::Input)?, crate::candidate_cli::extract::SCHEMA_BYTES)?;
    remaining()?;
    let text = if args.input.as_os_str() == "-" { read_input(input, args.max_input_bytes)? }
        else { read_input(&mut File::open(&args.input).map_err(|_| CandidateError::Input)?, args.max_input_bytes)? };
    if text.is_empty() { return Err(CandidateError::Input); }
    let budget = options.extraction.budget(&args, common);
    let request = options.request(schema, budget)?;
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = super::super::runtime::load_model(&engine, &args, common, remaining()?, cancellation.clone())?;
    remaining()?;
    let planner = Int4ExtractionBatchPlanner::for_model(model.model_identity(), budget,
        options.extraction.compiler(), options.extraction.source_limits(), None).map_err(|_| CandidateError::Planning)?;
    let expected = preflight(&text, &request, &planner, limits.mapping, &mut DeadlineControl { started, deadline })?;
    let controls = crate::tokenizer::pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let vocabulary = ExtractionVocabulary::pinned(controls.template_controls())
        .map(Arc::new).map_err(|_| CandidateError::Planning)?;
    remaining()?;
    let identity = planner.base_execution_identity().clone();
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        // Save only after the complete original document preflights. A later
        // execution failure preserves this complete independent candidate file.
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    let config = Int4ExtractionMapConfig { request, limits: limits.mapping,
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining()?, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        preparation_reserve_bytes: common.preparation_bytes, reduction_reserve_bytes: limits.reduction_reserve_bytes };
    // The CLI pre-IO quote and the actual host use the same complete frontier
    // quantities. A mismatch must not quietly weaken the memory declaration.
    if config.workspace_bytes(limits.mapping.mapping.chunks.max_chunks).map_err(|_| CandidateError::Memory)?
        != limits.workspace_bytes { return Err(CandidateError::Memory); }
    if let Some(quote) = limits.cohort {
        if config.cohort_requirements(args.group_width).map_err(|_| CandidateError::Memory)? != quote {
            return Err(CandidateError::Memory);
        }
    }
    let content = identity.logical_model_digest.to_hex();
    let finish = |document: &Int4ExtractionMapRun, cohort_schedule: Option<CohortSchedule<'_>>| {
        expected.verify_completed(document).map_err(|_| CandidateError::Execution)?;
        remaining()?;
        let response = SourceResponse { schema_version: 1, scope: "real-source-current-candidate-int4",
            evidence: "non_authoritative", model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
            source_root_sha256: model.model_identity().source_root_sha256(), model_content_sha256: &content,
            candidate_format: &identity.artifact_format, quant_recipe: &identity.quant_recipe,
            group_width: model.model_identity().group_width(), integer_kernel: common.kernel.kind().id(),
            output: document, snapshot, cohort_schedule };
        // No parsing/re-serialization of each exact JSON string, object merge
        // or successful-prefix publication. The result guard survives both
        // write and flush, including the complete cohort wrapper's statistics.
        publish(&response, limits.max_result_bytes + 4096, output)
    };
    if let Some(max_chunks) = limits.cohort_chunks {
        let result = engine.extract_int4_document_cohorts(&model, text, Arc::new(planner), vocabulary,
            config, common.kernel, cancellation).map_err(|_| CandidateError::Execution)?;
        finish(&result.result().document, Some(CohortSchedule::new(max_chunks, &result.result().cohorts)))
    } else {
        let result = engine.extract_int4_document(&model, text, Arc::new(planner), vocabulary,
            config, common.kernel, cancellation).map_err(|_| CandidateError::Execution)?;
        finish(result.result(), None)
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
struct SourceResponse<'a, T: Serialize> {
    schema_version: u32, scope: &'static str, evidence: &'static str, model_id: &'static str,
    source_revision: &'a str, source_root_sha256: &'a str, model_content_sha256: &'a str,
    candidate_format: &'a str, quant_recipe: &'a str, group_width: usize, integer_kernel: &'static str,
    output: &'a T,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cohort_schedule: Option<CohortSchedule<'a>>,
}
#[derive(Serialize)]
struct CohortSchedule<'a> {
    execution: &'static str,
    declared_max_chunks: usize,
    physical: &'a Int4ExtractionMapCohortStats,
}
impl<'a> CohortSchedule<'a> {
    fn new(declared_max_chunks: usize, physical: &'a Int4ExtractionMapCohortStats) -> Self {
        Self { execution: "cross_chunk_constrained_cohorts", declared_max_chunks, physical }
    }
}

#[cfg(test)]
mod tests;
