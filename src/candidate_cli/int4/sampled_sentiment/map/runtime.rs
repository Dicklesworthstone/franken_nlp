//! One charged complete-document map and deadline-checked final publication.
use super::*;
use std::time::{Duration, Instant};
use crate::{
    NlpEngine, MemoryClass,
    hosted::{CancellationToken, HostedError, Int4HostError, NativeLimits, RunLimits,
        SampledSentimentMapHostConfig},
    native_engine::{decode::DecodeCancellationKind,
        generation::grouped_int4::INT4_SOURCE_FORMAT,
        strict_int4::{prefill::INT4_LAYER_MAJOR_PREFILL_VERSION,
            weights::snapshot::INT4_SNAPSHOT_VERSION}},
    tasks::{mapreduce::MapReduceError, sentiment::sampled::long::{SampledSentimentMapError,
        SAMPLED_SENTIMENT_MAP_EXECUTION, SAMPLED_SENTIMENT_MAP_REUSED_PREFILL_EXECUTION}},
    validation::grounded_fields::VerifiedSourceSpan,
};
use super::super::runtime::{cancelled, task_failure};

fn map_failure(error: SampledSentimentMapError) -> Failure {
    if let Some(cause) = error.cancellation() { return cancelled(cause); }
    use SampledSentimentMapError::*;
    let (exit, code) = match error {
        Sampled(error) => return task_failure(error),
        Limits | Work | Allocation | Chunk(MapReduceError::AllocationRefused) =>
            (ErrorCode::AdmissionOrResourceLimit, "sampled_sentiment_map_resources"),
        Output | Chunk(MapReduceError::TokenizerBudget) =>
            (ErrorCode::BudgetOrTimeout, "sampled_sentiment_map_budget"),
        EmptySource | Chunk(MapReduceError::InvalidLimits | MapReduceError::InputBudget
            | MapReduceError::ChunkBudget | MapReduceError::TokenBudget) =>
            (ErrorCode::AdmissionOrResourceLimit, "sampled_sentiment_map_partition"),
        _ => (ErrorCode::Generic, "sampled_sentiment_map_invariant"),
    };
    Failure { exit, code, cancellation: None }
}

fn host_failure(error: HostedError) -> Failure {
    match error {
        HostedError::Int4(Int4HostError::SampledSentimentMap(error)) => map_failure(error),
        HostedError::Stopped { stop, task_error } => {
            if matches!(stop.kind, DecodeCancellationKind::FailFast | DecodeCancellationKind::LinkedExit) {
                if let Some(error) = task_error {
                    let mut failure = host_failure(*error);
                    failure.cancellation = Some(stop.kind);
                    return failure;
                }
            }
            cancelled(stop.kind)
        }
        error => super::super::runtime::host_failure(error),
    }
}

fn host_config(args: &Int4Args, options: MapArgs, limits: MapLimits, remaining: Duration)
    -> Result<SampledSentimentMapHostConfig, CandidateError> {
    Ok(SampledSentimentMapHostConfig {
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        planning: limits.planning, mapping: limits.mapping, options: options.sampled.options(args)?,
        preparation_reserve_bytes: limits.common.preparation_bytes,
        max_sampler_bytes: json_sampling_workspace_bytes().map_err(|_| CandidateError::Arguments)?,
        mask_limits: options.sampled.masks(), max_mask_node_visits: options.sampled.max_mask_node_visits,
        prefill: limits.common.prefill, reuse_prefill: options.sampled.reuse_prefill,
    })
}

fn check_host_quote(args: &Int4Args, options: MapArgs, limits: MapLimits,
    config: &SampledSentimentMapHostConfig, encoding: &UntrustedEncoding) -> Result<(), Failure> {
    let task = options.sampled.budget(args, limits.common, SentimentAxis::ALL.len())?;
    let required = config.requirements_with_encoding(args.group_width, task,
        SentimentAxis::ALL.len(), encoding).map_err(host_failure)?;
    let prefill = limits.common.prefill.map(Int4PrefillLimits::validate).transpose()
        .map_err(|_| CandidateError::Arguments)?.unwrap_or(0);
    if required.native != limits.memory.native || required.input_bytes != limits.memory.input_bytes
        || required.workspace_bytes != sum(&[limits.memory.workspace_bytes, prefill])?
        || required.output_reservation_bytes != limits.memory.output_reservation_bytes
        || required.max_output_bytes != limits.memory.max_output_bytes
        || required.max_output_tokens != limits.memory.max_output_tokens {
        return Err(Failure::usage("sampled_sentiment_map_memory_quote"));
    }
    Ok(())
}

pub(super) fn execute(args: Int4Args, options: MapArgs, _admitted: MapLimits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), Failure> {
    let started = Instant::now();
    let limits = options.validate(&args)?;
    let common = limits.common;
    let encoding = options.input_encoding(&args)?;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or(CandidateError::Timeout);
    let mut config = host_config(&args, options, limits, remaining()?)?;
    check_host_quote(&args, options, limits, &config, &encoding)?;
    let engine = NlpEngine::builder().resource_config(compute::resource_config(common))
        .build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    let _preparation = lease.reserve(MemoryClass::JobBuffers, limits.cli_preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    let text = if args.input.as_os_str() == "-" { read_input(input, args.max_input_bytes)? }
        else { read_input(&mut crate::local_io::open_document(&args.input)
            .map_err(|_| CandidateError::Input)?, args.max_input_bytes)? };
    remaining()?;
    let mut request = parse_request(&text,
        options.sampled.budget(&args, common, SentimentAxis::ALL.len())?, args.max_input_bytes)?;
    request.budget = options.sampled.budget(&args, common, request.axes.len())?;
    let source_span = VerifiedSourceSpan { byte_start: 0, byte_end: request.document.len(),
        scalar_start: 0, scalar_end: request.document.chars().count() };
    let axes = request.axes.len();
    drop(text);
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = super::super::super::runtime::load_model_with_failure(&engine, &args, common,
        remaining()?, cancellation.clone(), host_failure)?;
    config.native.run.max_elapsed = remaining()?;
    let result = if common.compute.threads > 1 {
        engine.map_int4_sampled_sentiment_with_encoding_and_threads(&model, request, config,
            encoding, common.kernel, common.compute.threads, cancellation.clone()).map(compute::Completion::Team)
    } else {
        engine.map_int4_sampled_sentiment_with_encoding(&model, request, config, encoding,
            common.kernel, cancellation.clone()).map(compute::Completion::serial)
    }.map_err(host_failure)?;
    let run = result.result();
    let execution = if options.sampled.reuse_prefill { SAMPLED_SENTIMENT_MAP_REUSED_PREFILL_EXECUTION }
        else { SAMPLED_SENTIMENT_MAP_EXECUTION };
    if run.execution != execution || run.group_width != args.group_width
        || run.single_context_equivalence || run.source_span != source_span
        || run.samples_per_axis != options.sampled.samples
        || run.chunks.is_empty() || run.chunks.len() > options.mapping.max_chunks
        || run.chunks.iter().any(|chunk| chunk.run.dimensions.len() != axes)
        || !fits_work(run.planned_model_work, options.sampled.work_ceiling())
        || !fits_work(run.model_work, run.planned_model_work)
        || run.planned_mask_node_visit_charge > options.sampled.max_mask_node_visits
        || run.mask_node_visit_charge > run.planned_mask_node_visit_charge {
        return Err(Failure::output("sampled_sentiment_map_completion"));
    }
    remaining()?;
    let model_identity = model.model_identity();
    let content: String = model_identity.content_sha256().iter().map(|byte| format!("{byte:02x}")).collect();
    let recipe = format!("{}:group={}", model_identity.matrix_recipe(), model_identity.group_width());
    let response = Response {
        protocol: "fnlp-candidate-int4-sentiment-sampled-map-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        task: "sentiment-sampled-map", model_id: "Nanbeige4.2-3B",
        source_revision: model_identity.revision(), source_root_sha256: model_identity.source_root_sha256(),
        model_content_sha256: &content, candidate_format: INT4_SOURCE_FORMAT, quant_recipe: &recipe,
        group_width: args.group_width, integer_kernel: common.kernel.kind().id(),
        snapshot: common.snapshot_pin.map(|_| SnapshotRecord { operation: "loaded",
            format: INT4_SNAPSHOT_VERSION, file_bytes: common.snapshot_need.file_bytes }),
        prefill_schedule: common.prefill.map(PrefillSchedule::new).transpose()?,
        prefill_reuse: options.sampled.prefill_reuse_execution(), output: run,
    };
    publish_complete(&compute::Response { response: &response, compute: result.compute() },
        limits.envelope_bytes, output, &remaining, &cancellation)
}

fn fits_work(work: Int4Work, ceiling: Int4Work) -> bool {
    work.forward_positions <= ceiling.forward_positions && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs && work.projections.fits(ceiling.projections)
}

fn publish_complete<T: Serialize>(value: &T, cap: usize, output: &mut impl Write,
    remaining: &impl Fn() -> Result<Duration, CandidateError>, cancellation: &CancellationToken)
    -> Result<(), Failure> {
    publish_checked(value, cap, output, &|| {
        if cancellation.cause().is_some() { return Err(CandidateError::Execution); }
        remaining().map(|_| ())
    }).map_err(|error| match (error, cancellation.cause()) {
        (CandidateError::Execution, Some(cause)) => cancelled(cause),
        (CandidateError::Timeout, _) => cancelled(DecodeCancellationKind::Deadline),
        (error, _) => Failure::from(error),
    })
}

#[derive(Serialize)]
struct SnapshotRecord { operation: &'static str, format: &'static str, file_bytes: u64 }
#[derive(Serialize)]
struct PrefillSchedule { execution: &'static str, max_batch_rows: usize, admitted_workspace_bytes: u64 }
impl PrefillSchedule {
    fn new(limits: Int4PrefillLimits) -> Result<Self, CandidateError> {
        Ok(Self { execution: INT4_LAYER_MAJOR_PREFILL_VERSION, max_batch_rows: limits.max_batch_rows,
            admitted_workspace_bytes: limits.validate().map_err(|_| CandidateError::Arguments)? })
    }
}
#[derive(Serialize)]
struct Response<'a, T: Serialize> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str,
    task: &'static str, model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str,
    model_content_sha256: &'a str, candidate_format: &'a str, quant_recipe: &'a str,
    group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_schedule: Option<PrefillSchedule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_reuse: Option<&'static str>,
    output: &'a T,
}

#[cfg(test)]
mod tests;
