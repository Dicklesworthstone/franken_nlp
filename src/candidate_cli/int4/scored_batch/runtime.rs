//! One owned ordered stream, one resident model, one finite native work ledger.
use super::*;
use std::{io::{BufRead, BufReader}, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    batch::scored_int4::{Int4ScoredBatchPlanner, Int4ScoredTask},
    candidate_cli::batch::{CandidateWriter, completed},
    hosted::{CancellationToken, HostedError, NativeLimits, ResidentInt4, RunLimits, corpus::CorpusLimits},
    native_engine::{decode::DecodeCancellationKind, lmhead::scoring::ScoringMode,
        strict_int4::weights::snapshot::INT4_SNAPSHOT_VERSION},
    tasks::{classify::grouped_int4::Int4ClassificationPlanner,
        judge::grouped_int4::Int4JudgePlanner,
        sentiment::{SentimentOptions, SentimentPolicy, grouped_int4::Int4SentimentPlanner}},
};

fn sentiment_options(policy: SentimentPolicy) -> Result<SentimentOptions, CandidateError> {
    let controls = crate::tokenizer::pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let mut ends = controls.template_controls().entries().iter()
        .filter(|entry| entry.special && entry.surface == crate::template::IM_END);
    let eos_token_id = ends.next().ok_or(CandidateError::Identity)?.id;
    if ends.next().is_some() { return Err(CandidateError::Identity); }
    Ok(SentimentOptions { mode: ScoringMode::FullVocabulary, eos_token_id, policy })
}
fn stopped(cause: DecodeCancellationKind) -> CandidateError {
    match cause {
        DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout => CandidateError::Timeout,
        _ => CandidateError::Execution,
    }
}
fn execution_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => stopped(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => execution_error(*error),
        HostedError::BatchSetup(fault) => fault.cancellation.map(stopped).unwrap_or(CandidateError::Batch),
        HostedError::Batch(error) => error.fault.cancellation.map(stopped).unwrap_or(CandidateError::Batch),
        HostedError::Reservation(_) => CandidateError::Memory,
        HostedError::ModelIdentity | HostedError::ResourceDomain => CandidateError::Identity,
        _ => CandidateError::Batch,
    }
}

pub(super) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    args: Int4Args, batch: BatchArgs, _admitted: BatchLimits, input: R, output: W)
    -> Result<(), CandidateError> {
    let started = Instant::now();
    let limits = batch.validate(&args)?;
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
    let text = batch.stream.defaults.as_ref().map(|path| {
        read_input(&mut crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?, shared::DEFAULTS_BYTES)
    }).transpose()?;
    let budget = batch.scoring.budget(common);
    let defaults = batch.load_defaults(text.as_deref(), budget, &args)?;
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
                batch.scoring.classification_limits(&args)).map_err(|_| CandidateError::Planning)?;
            let planner = Int4ScoredBatchPlanner::new(planner, defaults, limits.native).map_err(|_| CandidateError::Planning)?;
            stream(&engine, &model, planner, "classify", &args, limits, started, deadline, reader, output, cancellation)
        }
        Defaults::Sentiment { args: defaults, policy } => {
            let planner = Int4SentimentPlanner::for_model(model.model_identity(), sentiment_options(policy)?, budget,
                batch.scoring.sentiment_limits(&args)).map_err(|_| CandidateError::Planning)?;
            let planner = Int4ScoredBatchPlanner::new(planner, Some(defaults), limits.native).map_err(|_| CandidateError::Planning)?;
            stream(&engine, &model, planner, "sentiment", &args, limits, started, deadline, reader, output, cancellation)
        }
        Defaults::Judge(defaults) => {
            let planner = Int4JudgePlanner::for_model(model.model_identity(), budget,
                batch.scoring.judge_limits(budget)).map_err(|_| CandidateError::Planning)?;
            let planner = Int4ScoredBatchPlanner::new(planner, defaults, limits.native).map_err(|_| CandidateError::Planning)?;
            stream(&engine, &model, planner, "judge", &args, limits, started, deadline, reader, output, cancellation)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn stream<P: Int4ScoredTask + Send + 'static, W: Write + Send + 'static>(engine: &NlpEngine,
    model: &ResidentInt4, planner: Int4ScoredBatchPlanner<P>, task: &'static str, args: &Int4Args,
    limits: BatchLimits, started: Instant, deadline: Duration, reader: Box<dyn BufRead + Send>,
    output: W, cancellation: CancellationToken) -> Result<(), CandidateError> {
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or(CandidateError::Timeout);
    remaining()?;
    let common = limits.common;
    let identity = planner.base_execution_identity().clone();
    if planner.group_width() != model.model_identity().group_width() { return Err(CandidateError::Identity); }
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        // Defaults and the model-bound factory are admitted first. Future
        // records remain incremental; a saved complete model survives failure.
        let pin = engine.save_int4_snapshot(model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    remaining()?;
    let content = identity.logical_model_digest.to_hex();
    let provenance = Provenance {
        protocol: "fnlp-candidate-int4-score-batch-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task,
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(), model_content_sha256: &content,
        candidate_format: &identity.artifact_format, quant_recipe: &identity.quant_recipe,
        group_width: model.model_identity().group_width(), integer_kernel: common.kernel.kind().id(), snapshot,
    };
    let writer = CandidateWriter::new(output, &provenance, limits.transport.max_output_line_bytes, limits.output_bytes)?;
    let corpus = CorpusLimits {
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining()?, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        transport: limits.transport, preparation_reserve_bytes: limits.preparation_reserve_bytes,
        io_reserve_bytes: limits.io_bytes,
    };
    let summary = engine.batch_int4_scored(model, planner, corpus, common.kernel, reader, writer, cancellation)
        .map_err(execution_error)?;
    // The host emitted the sole terminal record. A drained stream containing
    // rejected items still has nonzero process status, with no extra stdout.
    completed(summary)
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

#[cfg(test)]
mod tests;
