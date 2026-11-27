//! Independent INT8 samples on the existing checked-file session and host.
//! All requested axes and samples are admitted before model materialization;
//! preparation and guarded output remain charged through checked publication.
use super::*;
pub(in crate::candidate_cli) mod corpus;
pub(in crate::candidate_cli) mod map;
use super::source_tasks::Session;
use std::sync::Arc;
use crate::{
    candidate_cli::{sampled_sentiment::{self as command, SampledSentimentCommand}, stream::Failure},
    grammar::mask::MaskOracleError,
    hosted::{HostedError, Int8SampledSentimentHostLimits},
    native_engine::{
        artifact_bridge::ArtifactBridgeError,
        constrained::{JsonDecodeError, JsonWorkBudget},
        constrained_int8::{Int8JsonBudget, Int8JsonError, json_sampling_workspace_bytes},
        decode::DecodeCancellationKind,
        strict_int8::{CurrentCandidateInt8Error, Int8RunBudget, Int8Work, StrictInt8Error},
    },
    tasks::sentiment::{SentimentAxis, sampled::{SampledSentimentError, SampledSentimentVocabulary,
        quantized::{Int8SampledSentimentError, Int8SampledSentimentPlanner,
            Int8SampledSentimentRun, INT8_SAMPLED_SENTIMENT_EXECUTION}}},
    tokenizer::untrusted::UntrustedEncoding,
};

pub(in crate::candidate_cli) fn cancelled(cause: DecodeCancellationKind) -> Failure {
    use DecodeCancellationKind::*;
    let (exit, code) = match cause {
        Timeout | Deadline | PollQuota | CostBudget =>
            (ErrorCode::BudgetOrTimeout, "int8_sampled_sentiment_budget"),
        User | ParentCancelled | Shutdown =>
            (ErrorCode::Cancelled, "int8_sampled_sentiment_cancelled"),
        ResourceUnavailable =>
            (ErrorCode::AdmissionOrResourceLimit, "int8_sampled_sentiment_resources"),
        FailFast | LinkedExit =>
            (ErrorCode::Generic, "int8_sampled_sentiment_supervised_failure"),
        RaceLost => (ErrorCode::Generic, "int8_sampled_sentiment_root_race"),
    };
    Failure { exit, code, cancellation: Some(cause) }
}

pub(in crate::candidate_cli) fn task_failure(error: Int8SampledSentimentError) -> Failure {
    if let Some(cause) = error.cancellation() { return cancelled(cause); }
    use Int8SampledSentimentError::{Native, Shared};
    use SampledSentimentError as TaskError;
    let (exit, code) = match error {
        Shared(TaskError::TokenizerWork) =>
            (ErrorCode::BudgetOrTimeout, "int8_sampled_sentiment_encoding_budget"),
        Shared(TaskError::Limits | TaskError::WorkBudget | TaskError::Allocation)
            | Native(Int8JsonError::Decode(JsonDecodeError::AllocationRefused))
            | Native(Int8JsonError::Native(StrictInt8Error::Context
                | StrictInt8Error::Memory | StrictInt8Error::Allocation)) =>
            (ErrorCode::AdmissionOrResourceLimit, "int8_sampled_sentiment_resources"),
        Shared(TaskError::OutputBudget)
            | Native(Int8JsonError::Decode(JsonDecodeError::BudgetExceeded(_)))
            | Native(Int8JsonError::Decode(JsonDecodeError::Mask(MaskOracleError::WorkBudgetExceeded { .. })))
            | Native(Int8JsonError::Native(StrictInt8Error::Work)) =>
            (ErrorCode::BudgetOrTimeout, "int8_sampled_sentiment_budget"),
        Shared(TaskError::NoResult)
            | Native(Int8JsonError::Decode(JsonDecodeError::NoLegalToken | JsonDecodeError::IndependentValidation)) =>
            (ErrorCode::StructuredTaskNoResult, "int8_sampled_sentiment_no_result"),
        Shared(TaskError::Identity | TaskError::Tokenizer | TaskError::Template)
            | Native(Int8JsonError::ModelIdentity) =>
            (ErrorCode::ArtifactIntegrityOrFormatOrVersion, "int8_sampled_sentiment_identity"),
        Shared(TaskError::Schema) =>
            (ErrorCode::SchemaOrRecipeCompile, "int8_sampled_sentiment_schema"),
        _ => (ErrorCode::Generic, "int8_sampled_sentiment_task_invariant"),
    };
    Failure { exit, code, cancellation: None }
}

pub(in crate::candidate_cli) fn host_failure(error: HostedError) -> Failure {
    match error {
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
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => host_failure(*error),
        HostedError::BatchSetup(fault) => corpus::failure(fault),
        HostedError::Batch(error) => corpus::failure(error.fault),
        HostedError::SampledSentiment(error) => task_failure(error),
        HostedError::Native(error) => task_failure(error.into()),
        HostedError::Model(CurrentCandidateInt8Error::Artifact(ArtifactBridgeError::Cancelled(cause)))
            | HostedError::Model(CurrentCandidateInt8Error::Engine(StrictInt8Error::Cancelled(cause))) => cancelled(cause),
        HostedError::Limits(_) | HostedError::Reservation(_) | HostedError::Spawn(_)
            | HostedError::Reentrant | HostedError::MissingRuntimeContext
            | HostedError::SingleCoordinatorRequired => Failure {
                exit: ErrorCode::AdmissionOrResourceLimit,
                code: "int8_sampled_sentiment_admission", cancellation: None },
        HostedError::ModelIdentity | HostedError::ResourceDomain | HostedError::Model(_) => Failure {
            exit: ErrorCode::ArtifactIntegrityOrFormatOrVersion,
            code: "int8_sampled_sentiment_model_identity", cancellation: None },
        _ => Failure::output("int8_sampled_sentiment_runtime_or_completion"),
    }
}

fn setup_failure(error: CandidateError) -> Failure {
    if error == CandidateError::Timeout { return cancelled(DecodeCancellationKind::Deadline); }
    let (exit, code) = match error {
        CandidateError::Arguments => (ErrorCode::Usage, "int8_sampled_sentiment_options"),
        CandidateError::Input => (ErrorCode::InputDecodeOrParse, "int8_sampled_sentiment_input"),
        CandidateError::Planning | CandidateError::Memory | CandidateError::Runtime =>
            (ErrorCode::AdmissionOrResourceLimit, "int8_sampled_sentiment_plan_admission"),
        CandidateError::Identity | CandidateError::Model =>
            (ErrorCode::ArtifactIntegrityOrFormatOrVersion, "int8_sampled_sentiment_model_identity"),
        _ => (ErrorCode::Generic, "int8_sampled_sentiment_setup_or_delivery"),
    };
    Failure { exit, code, cancellation: None }
}

/// A refused stage still owes the original invocation its terminal poll.
/// Preserve an already reported cancellation cause across physical cleanup;
/// otherwise an expired invocation wins over incidental stage errors.
pub(in crate::candidate_cli) fn checked_stage<T, E>(result: Result<T, E>,
    check: impl FnOnce() -> Result<(), CandidateError>, classify: impl FnOnce(E) -> Failure)
    -> Result<T, Failure> {
    let checked = check().map_err(setup_failure);
    match result {
        Ok(value) => { checked?; Ok(value) }
        Err(error) => {
            let failure = classify(error);
            if failure.cancellation.is_some() { return Err(failure); }
            checked?;
            Err(failure)
        }
    }
}

pub(in crate::candidate_cli) fn execute(command: SampledSentimentCommand,
    common: CandidateArgs, limits: Limits, input: &mut impl Read, output: &mut impl Write)
    -> Result<(), Failure> {
    let prefill = command.prefill().map_err(setup_failure)?;
    if limits.compute != common.compute.admit_prefill(prefill).map_err(setup_failure)? {
        return Err(setup_failure(CandidateError::Arguments));
    }
    let encoding = common.input_encoding_policy(limits).map_err(setup_failure)?;
    let options = command.options().map_err(setup_failure)?;
    let samples_per_axis = options.samples;
    // Session is declared first so every request, compiler, vocabulary and
    // response allocation drops before the caller's preparation reservation.
    let session = Session::new_with_prefill(&common, limits, prefill).map_err(setup_failure)?;
    let check = || session.remaining().map(|_| ());
    let text = checked_stage(session.read(&common.input, input, common.max_input_bytes), &check, setup_failure)?;
    let budget = checked_stage(command.budget(limits, SentimentAxis::ALL.len()), &check, setup_failure)?;
    let mut request = checked_stage(command::request(&text, budget, common.max_input_bytes), &check, setup_failure)?;
    request.budget = checked_stage(command.budget(limits, request.axes.len()), &check, setup_failure)?;
    checked_stage(if matches!(encoding, UntrustedEncoding::ByteFallback)
        && request.document.len() > limits.max_prompt_tokens {
        Err(CandidateError::Planning)
    } else { Ok(()) }, &check, setup_failure)?;
    let mut axes = request.axes.clone(); axes.sort_unstable();
    let sample_count = checked_stage(axes.len().checked_mul(samples_per_axis)
        .ok_or(CandidateError::Arguments), &check, setup_failure)?;
    let output_tokens = u64::from(request.budget.max_output_tokens);
    drop(text);
    check().map_err(setup_failure)?;
    // facts() holds the authenticated file open; load() consumes that same
    // source after the complete sampled task has been compiled and admitted.
    let facts = checked_stage(session.facts(&common), &check, setup_failure)?;
    let planner = checked_stage(Int8SampledSentimentPlanner::for_model_with_encoding(&facts, options,
        request.budget, command.task_limits(), encoding.clone()), &check, task_failure)?;
    let prepared = checked_stage(planner.plan_with_control(&request, &mut session.control()), &check, task_failure)?;
    checked_stage(command.admit_plan(prepared.required_context(), prepared.planned_work()), &check, setup_failure)?;
    checked_stage(if prepared.sample_count() != sample_count || prepared.max_output_tokens() != output_tokens
        || prepared.input_encoding() != &encoding
        || prepared.max_result_bytes() != limits.result_bytes as u64
        || prepared.preparation_requirement_bytes() > limits.preparation_bytes {
        Err(CandidateError::Planning)
    } else { Ok(()) }, &check, setup_failure)?;
    let expected_work = prepared.planned_work();
    let sampler_bytes = checked_stage(json_sampling_workspace_bytes(), &check,
        |error| task_failure(error.into()))?;
    let vocabulary = Arc::new(checked_stage(SampledSentimentVocabulary::pinned(), &check,
        |error| task_failure(error.into()))?);
    let budget = Int8JsonBudget { native: Int8RunBudget::exact(expected_work),
        json: JsonWorkBudget { max_forward_positions: expected_work.forward_positions,
            max_projected_logits: expected_work.projected_logits, max_kv_bytes: limits.kv_bytes,
            max_total_mask_node_visits: command.max_mask_node_visits, mask_limits: command.masks() } };
    let mask_charge = checked_stage(prepared.complete_mask_charge(&vocabulary, budget, sampler_bytes),
        &check, task_failure)?;
    drop(planner);
    drop(request);
    check().map_err(setup_failure)?;
    let cancellation = CancellationToken::default();
    let model = checked_stage(session.load(&common, limits, &facts, cancellation.clone()), &check, setup_failure)?;
    let host = Int8SampledSentimentHostLimits {
        native: checked_stage(session.native(&common), &check, setup_failure)?,
        preparation_reserve_bytes: limits.preparation_bytes, max_model_work: command.work_ceiling(),
        max_sampler_bytes: sampler_bytes, mask_limits: command.masks(),
        max_mask_node_visits: command.max_mask_node_visits,
    };
    let guarded = if limits.compute.threads == 1 {
        compute::Completion::Serial(checked_stage(match prefill {
            None => session.engine.execute_int8_sampled_sentiment(&model, prepared, vocabulary, host, cancellation),
            Some(prefill) => session.engine.execute_int8_sampled_sentiment_layer_major(
                &model, prepared, vocabulary, host, prefill, cancellation),
        }, &check, host_failure)?)
    } else {
        let kernel = crate::native_engine::simd::config::process_kernel();
        compute::Completion::Team(checked_stage(match prefill {
            None => session.engine.execute_int8_sampled_sentiment_with_threads(
                &model, prepared, vocabulary, host, kernel, limits.compute.threads, cancellation),
            Some(prefill) => session.engine.execute_int8_sampled_sentiment_layer_major_with_threads(
                &model, prepared, vocabulary, host, kernel, limits.compute.threads, prefill, cancellation),
        }, &check, host_failure)?)
    };
    checked_stage(check_completion(guarded.result(), &axes, samples_per_axis,
        expected_work, output_tokens, mask_charge), &check, |failure| failure)?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: guarded.result() };
    let response = compute::Response { response: &response, compute: guarded.compute() };
    let response = input_encoding::Response { response: &response,
        input_encoding: input_encoding::reported_policy(&encoding) };
    let cap = checked_stage(limits.result_bytes.checked_add(4096)
        .ok_or(CandidateError::Output), &check, setup_failure)?;
    // Neither the output guard nor Session's preparation charge is released
    // until staging, the external write and flush, and the final deadline check.
    publish_checked(&response, cap, output, &check).map_err(setup_failure)
}

fn check_completion(run: &Int8SampledSentimentRun, axes: &[SentimentAxis], samples: usize,
    expected: Int8Work, output_tokens: u64, mask_charge: u64) -> Result<(), Failure> {
    let work = run.model_work;
    if run.execution != INT8_SAMPLED_SENTIMENT_EXECUTION || run.samples_per_axis != samples
        || run.dimensions.len() != axes.len()
        || run.dimensions.iter().zip(axes).any(|(dimension, axis)| dimension.axis != *axis
            || dimension.samples.len() != samples
            || dimension.samples.iter().enumerate().any(|(ordinal, sample)| sample.sample_index != ordinal))
        || work.forward_positions > expected.forward_positions || work.projected_logits > expected.projected_logits
        || work.attention_pairs > expected.attention_pairs || !work.projections.fits(expected.projections)
        || run.generated_tokens > output_tokens || run.mask_node_visit_charge > mask_charge {
        return Err(Failure::output("int8_sampled_sentiment_completion_accounting"));
    }
    Ok(())
}
