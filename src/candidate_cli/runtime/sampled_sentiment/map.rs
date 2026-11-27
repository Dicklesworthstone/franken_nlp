//! Complete-document sampled INT8 execution through one charged host request.
pub(in crate::candidate_cli) mod corpus;
#[cfg(all(feature = "metadata-store", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(in crate::candidate_cli) mod job;
use super::*;
use crate::{
    candidate_cli::sampled_sentiment::map::{MapLimits, SampledSentimentMapCommand},
    hosted::{Int8SampledSentimentMapHostConfig, NativeLimits, RunLimits},
    tasks::{mapreduce::MapReduceError, sentiment::sampled::quantized::long::{
        Int8SampledSentimentMapError, Int8SampledSentimentMapRun,
        INT8_SAMPLED_SENTIMENT_MAP_EXECUTION,
    }},
    tokenizer::untrusted::UntrustedEncoding,
    validation::grounded_fields::VerifiedSourceSpan,
};
use std::time::Duration;

fn map_failure(error: Int8SampledSentimentMapError) -> Failure {
    if let Some(cause) = error.cancellation() { return cancelled(cause); }
    use Int8SampledSentimentMapError::*;
    let (exit, code) = match error {
        Sampled(error) => return task_failure(error),
        Shared(error) => return shared_map_failure(error),
        Limits | Work | Allocation | Chunk(MapReduceError::AllocationRefused) =>
            (ErrorCode::AdmissionOrResourceLimit, "int8_sampled_sentiment_map_resources"),
        Output | Chunk(MapReduceError::TokenizerBudget) =>
            (ErrorCode::BudgetOrTimeout, "int8_sampled_sentiment_map_budget"),
        EmptySource | Chunk(MapReduceError::InvalidLimits | MapReduceError::InputBudget
            | MapReduceError::ChunkBudget | MapReduceError::TokenBudget) =>
            (ErrorCode::AdmissionOrResourceLimit, "int8_sampled_sentiment_map_partition"),
        _ => (ErrorCode::Generic, "int8_sampled_sentiment_map_invariant"),
    };
    Failure { exit, code, cancellation: None }
}

fn shared_map_failure(error: crate::tasks::sentiment::sampled::long::SampledSentimentMapError)
    -> Failure {
    use crate::tasks::sentiment::sampled::long::SampledSentimentMapError as Shared;
    use Int8SampledSentimentMapError as Map;
    let error = match error {
        Shared::Limits => Map::Limits,
        Shared::EmptySource => Map::EmptySource,
        Shared::Work => Map::Work,
        Shared::Allocation => Map::Allocation,
        Shared::Output => Map::Output,
        Shared::Chunk(error) => Map::Chunk(error),
        Shared::Sampled(error) => Map::Sampled(error.into()),
        Shared::Cancelled(cause) => Map::Cancelled(cause),
        _ => return Failure::output("int8_sampled_sentiment_map_invariant"),
    };
    map_failure(error)
}

fn map_host_failure(error: HostedError) -> Failure {
    match error {
        HostedError::SampledSentimentMap(error) => map_failure(error),
        HostedError::Stopped { stop, task_error } => {
            if matches!(stop.kind, DecodeCancellationKind::FailFast | DecodeCancellationKind::LinkedExit) {
                if let Some(error) = task_error {
                    let mut failure = map_host_failure(*error);
                    failure.cancellation = Some(stop.kind);
                    return failure;
                }
            }
            cancelled(stop.kind)
        }
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => map_host_failure(*error),
        error => host_failure(error),
    }
}

fn config(command: &SampledSentimentMapCommand, limits: &MapLimits)
    -> Result<Int8SampledSentimentMapHostConfig, CandidateError> {
    Ok(Int8SampledSentimentMapHostConfig {
        native: NativeLimits { context_tokens: limits.common.context_tokens,
            allocator_reserve_bytes: 64 * MIB,
            run: RunLimits { max_elapsed: Duration::from_secs(limits.common.timeout_seconds),
                max_checkpoints: limits.common.max_checkpoints, cleanup_reserve_bytes: 65_536 } },
        planning: command.sampled.task_limits(), mapping: limits.mapping,
        options: command.sampled.options()?,
        preparation_reserve_bytes: limits.shared.preparation_bytes,
        max_sampler_bytes: json_sampling_workspace_bytes().map_err(|_| CandidateError::Arguments)?,
        mask_limits: command.sampled.masks(),
        max_mask_node_visits: command.sampled.max_mask_node_visits,
        prefill: limits.prefill,
    })
}

fn check_host_quote(limits: &MapLimits, host: &Int8SampledSentimentMapHostConfig) -> Result<(), Failure> {
    let need = host.requirements_with_encoding(limits.task, SentimentAxis::ALL.len(), &limits.encoding)
        .map_err(map_host_failure)?;
    let extra = limits.prefill.map(|p| p.validate()).transpose()
        .map_err(|_| setup_failure(CandidateError::Arguments))?.unwrap_or(0);
    let workspace = limits.memory.workspace_bytes.checked_add(extra)
        .ok_or_else(|| setup_failure(CandidateError::Arguments))?;
    if need.native != limits.memory.native || need.input_bytes != limits.memory.input_bytes
        || need.workspace_bytes != workspace
        || need.output_reservation_bytes != limits.memory.output_reservation_bytes
        || need.max_output_tokens != limits.memory.max_output_tokens
        || need.max_output_bytes != limits.memory.max_output_bytes
        || limits.minimum_memory_bytes > limits.shared.memory_bytes {
        return Err(Failure::usage("int8_sampled_sentiment_map_memory_quote"));
    }
    Ok(())
}

pub(in crate::candidate_cli) fn execute(command: SampledSentimentMapCommand, limits: MapLimits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), Failure> {
    let mut host = config(&command, &limits).map_err(setup_failure)?;
    check_host_quote(&limits, &host)?;
    // Session is first so its preparation ownership outlives every request,
    // held artifact, guarded result and final serialization allocation.
    let session = Session::new_with_prefill(&limits.common, limits.shared, limits.prefill)
        .map_err(setup_failure)?;
    let check = || session.remaining().map(|_| ());
    let text = checked_stage(session.read(&limits.common.input, input, limits.common.max_input_bytes),
        &check, setup_failure)?;
    let mut request = checked_stage(command::request(&text, limits.task, limits.common.max_input_bytes),
        &check, setup_failure)?;
    request.budget = checked_stage(command.sampled.budget(limits.shared, request.axes.len()),
        &check, setup_failure)?;
    let source_span = VerifiedSourceSpan { byte_start: 0, byte_end: request.document.len(),
        scalar_start: 0, scalar_end: request.document.chars().count() };
    let mut axes = request.axes.clone();
    axes.sort_unstable();
    let samples = host.options.samples;
    drop(text);
    let facts = checked_stage(session.facts_with_error_detail(&limits.common), &check, session_failure)?;
    let cancellation = CancellationToken::default();
    let model = checked_stage(session.load_with_error_detail(&limits.common, limits.shared, &facts, cancellation.clone()),
        &check, session_failure)?;
    host.native.run.max_elapsed = checked_stage(session.remaining(), &check, setup_failure)?;
    // Planning, native work and deterministic reduction happen together under
    // the transferred request's deadline and nonrenewable work allowances.
    let result = if limits.shared.compute.threads == 1 {
        compute::Completion::Serial(checked_stage(
            session.engine.map_int8_sampled_sentiment_with_encoding(&model, request, host,
                limits.encoding.clone(), cancellation), &check, map_host_failure)?)
    } else {
        compute::Completion::Team(checked_stage(
            session.engine.map_int8_sampled_sentiment_with_encoding_and_threads(&model, request, host,
                limits.encoding.clone(), crate::native_engine::simd::config::process_kernel(),
                limits.shared.compute.threads, cancellation), &check, map_host_failure)?)
    };
    checked_stage(check_completion(result.result(), source_span, &axes, samples, &limits,
        command.sampled.max_mask_node_visits), &check, |failure| failure)?;
    let response = CandidateResponse { schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative",
        model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: result.result() };
    let response = compute::Response { response: &response, compute: result.compute() };
    // The map already reports its exact compact policy and cumulative work.
    // Both the output and preparation charges survive external write and flush.
    publish_checked(&response, limits.envelope_bytes, output, &check).map_err(setup_failure)
}

fn fits(work: Int8Work, cap: Int8Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}

fn check_completion(run: &Int8SampledSentimentMapRun, span: VerifiedSourceSpan,
    axes: &[SentimentAxis], samples: usize, limits: &MapLimits, mask_ceiling: u64) -> Result<(), Failure> {
    let expected_encoding = match &limits.encoding {
        UntrustedEncoding::ByteFallback => run.input_encoding.is_none(),
        policy => run.input_encoding.as_ref().is_some_and(|record| &record.policy == policy),
    };
    if run.schema_version != 1 || run.execution != INT8_SAMPLED_SENTIMENT_MAP_EXECUTION
        || run.numerics_profile != crate::native_engine::strict_int8::STRICT_INT8_PROFILE
        || run.single_context_equivalence || run.source_span != span || run.samples_per_axis != samples
        || !partition_within_declared(run.chunk_limits, limits.mapping.chunks) || !expected_encoding
        || run.chunks.is_empty() || run.chunks.len() > limits.mapping.chunks.max_chunks
        || run.chunks.iter().any(|chunk| chunk.run.dimensions.len() != axes.len()
            || chunk.run.dimensions.iter().zip(axes).any(|(dimension, axis)| dimension.axis != *axis
                || dimension.samples.len() != samples))
        || !fits(run.planned_model_work, limits.mapping.max_model_work)
        || !fits(run.model_work, run.planned_model_work)
        || run.generated_tokens > limits.memory.max_output_tokens
        || run.retained_result_bytes > limits.mapping.max_retained_result_bytes
        || run.planned_mask_node_visit_charge > mask_ceiling
        || run.mask_node_visit_charge > run.planned_mask_node_visit_charge {
        return Err(Failure::output("int8_sampled_sentiment_map_completion"));
    }
    Ok(())
}

/// Planning reserves the largest trusted axis scaffold and may tighten the
/// source-token allowance. The host authenticates the exact resulting plan;
/// the CLI checks it cannot widen any declared document boundary.
fn partition_within_declared(actual: crate::tasks::mapreduce::ChunkLimits,
    declared: crate::tasks::mapreduce::ChunkLimits) -> bool {
    actual.max_input_bytes == declared.max_input_bytes
        && actual.max_chunk_bytes == declared.max_chunk_bytes
        && actual.context_tokens == declared.context_tokens
        && actual.max_chunks == declared.max_chunks
        && actual.max_tokenizer_calls == declared.max_tokenizer_calls
        && actual.max_chunk_tokens <= declared.max_chunk_tokens
        && actual.reserved_tokens >= declared.reserved_tokens
        && actual.effective_token_limit().is_ok()
}

#[cfg(test)] mod tests;
