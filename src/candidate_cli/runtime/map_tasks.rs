//! Long-document candidate inference on the existing charged map/merge host.
use super::*;
mod summary;
mod question;
mod extraction;
mod synthesis;
mod keyphrase_reduction;
use std::sync::Arc;
use super::source_tasks::{Session, planner, source_identity};
use crate::{candidate_cli::map::{MapCommand, encoding},
    hosted::{HostedError, SourceMapConfig},
    native_engine::decode::{DecodeCancellationKind, DecodeStepControl},
    tasks::{ir::PlanContext,
        source_planning::{SourceTaskPlanner, quantized::{Int8SourceError,
            long::{Int8SourceMapLimits, Int8SourceMapRun, Int8SourceMapPreflight, SourceMapTask}}}}};

type Preflight = Int8SourceMapPreflight;

#[allow(clippy::too_many_arguments)]
fn preflight(source: &str, planner: &SourceTaskPlanner, command: &MapCommand,
    task: &SourceMapTask, context: &PlanContext<'_>, mapping: Int8SourceMapLimits, budget: TaskBudget,
    control: &mut impl DecodeStepControl) -> Result<Preflight, CandidateError> {
    poll(control)?;
    if source.is_empty() { return Err(CandidateError::Input); }
    let expected = planner.preflight_int8_map_with_control(source, task, budget, context,
        command.host.planning(), mapping, control)
        .map_err(|error| error.cancellation().map(cancelled).unwrap_or(CandidateError::Planning))?;
    command.admit_work(expected.planned_work(), expected.reserved_mask_visits())?;
    poll(control)?;
    Ok(expected)
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), CandidateError> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(cancelled(cause)), None => Ok(()) }
}
fn cancelled(cause: DecodeCancellationKind) -> CandidateError {
    match cause { DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout => CandidateError::Timeout,
        _ => CandidateError::Execution }
}
fn source_error(error: Int8SourceError) -> CandidateError {
    error.cancellation().map(cancelled).unwrap_or(CandidateError::Planning)
}
fn host_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => cancelled(stop.kind),
        HostedError::Native(crate::native_engine::strict_int8::StrictInt8Error::Cancelled(cause)) => cancelled(cause),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => host_error(*error),
        HostedError::SourceMap(error) => error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution),
        HostedError::Source(error) => error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution),
        HostedError::Reservation(_) => CandidateError::Memory,
        _ => CandidateError::Execution,
    }
}

pub(in crate::candidate_cli) fn execute(command: MapCommand, args: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    if command.kind()? == crate::tasks::BuiltInTask::Answer {
        return question::execute(command, args, limits, input, output);
    }
    if command.kind()? == crate::tasks::BuiltInTask::Extract {
        return extraction::execute(command, args, limits, input, output);
    }
    if command.summary.synthesis.synthesize_summary {
        return synthesis::execute(command, args, limits, input, output);
    }
    let session = Session::new(&args, limits)?;
    let text = session.read(&command.input, input, args.max_input_bytes)?;
    let option_text = command.options.as_ref()
        .map(|path| session.read(path, input, crate::candidate_cli::source::OPTIONS_BYTES)).transpose()?;
    let task = command.map_task(option_text.as_deref())?;
    if text.is_empty() { return Err(CandidateError::Input); }
    session.remaining()?;
    let facts = session.facts(&args)?;
    let (planner, vocabulary) = planner()?;
    let planner = match command.encoding.limits(&command)? {
        Some(encoding) => planner.with_map_document_encoding(encoding.policy(), encoding.complete)
            .map_err(|_| CandidateError::Planning)?,
        None => planner,
    };
    let budget = command.host.task_budget(limits);
    let identity = source_identity(&facts, &planner, command.kind()?)?;
    let context = PlanContext::new(&identity, budget).map_err(|_| CandidateError::Planning)?;
    let capacity = planner.int8_map_capacity_with_control(&task, budget, &context,
        command.host.planning(), &mut session.control()).map_err(source_error)?;
    let mapping = command.mapping(capacity)?;
    let summary = command.summary.limits(command.max_map_result_bytes)?;
    if let Some(limits) = summary { limits.validate(mapping).map_err(|_| CandidateError::Planning)?; }
    let keyphrases = command.keyphrases.limits(command.max_map_result_bytes)?;
    if let Some(limits) = keyphrases { limits.validate(mapping).map_err(|_| CandidateError::Planning)?; }
    let expected = preflight(&text, &planner, &command, &task, &context, mapping, budget, &mut session.control())?;
    let expected_summary = summary.map(|limits| expected.summary_with_control(limits, &mut session.control())
        .map_err(|error| error.cancellation().map(cancelled).unwrap_or(CandidateError::Planning))).transpose()?;
    let expected_keyphrases = keyphrases.map(|limits| expected.keyphrases_with_control(limits, &mut session.control())
        .map_err(|error| error.cancellation().map(cancelled).unwrap_or(CandidateError::Planning))).transpose()?;
    session.remaining()?;
    // No weights until the complete partition and all five work axes fit.
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let config = SourceMapConfig { identity, task, budget, planning: command.host.planning(), mapping,
        native: session.native(&args)?, preparation_reserve_bytes: limits.preparation_bytes,
        reduction_reserve_bytes: command.reduction_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)? };
    // Retain the exact planner, not a fresh ledger, through hosted recompilation
    // and final usage reporting. All private source storage stays charged.
    let planner = Arc::new(planner);
    if let (Some(limits), Some(expected)) = (summary, expected_summary) {
        let result = session.engine.summarize_int8_source(&model, text, Arc::clone(&planner), Arc::new(vocabulary),
            config, limits, cancellation).map_err(host_error)?;
        summary::check_completed(&expected, result.result(), &mut session.control())?;
        return deliver_encoded(&session, &facts, &result, encoding::report(&planner)?.as_ref(),
            command.max_map_result_bytes + 4096, output);
    }
    if let (Some(limits), Some(expected)) = (keyphrases, expected_keyphrases) {
        let result = session.engine.keyphrases_int8_source(&model, text, Arc::clone(&planner), Arc::new(vocabulary),
            config, limits, cancellation).map_err(host_error)?;
        keyphrase_reduction::check_completed(&expected, result.result(), &mut session.control())?;
        return deliver_encoded(&session, &facts, &result, encoding::report(&planner)?.as_ref(),
            command.max_map_result_bytes + 4096, output);
    }
    let result = session.engine.map_int8_source(&model, text, Arc::clone(&planner), Arc::new(vocabulary),
        config, cancellation).map_err(host_error)?;
    check_completed(&expected, result.result(), &mut session.control())?;
    deliver_encoded(&session, &facts, &result, encoding::report(&planner)?.as_ref(),
        command.max_map_result_bytes + 4096, output)
}
fn deliver<T: Serialize>(session: &Session, facts: &ArtifactIdentity, result: &T,
    cap: usize, output: &mut impl Write) -> Result<(), CandidateError> {
    deliver_encoded(session, facts, result, None, cap, output)
}
fn deliver_encoded<T: Serialize>(session: &Session, facts: &ArtifactIdentity, result: &T,
    input_encoding: Option<&encoding::EncodingReport>, cap: usize, output: &mut impl Write)
    -> Result<(), CandidateError> {
    deliver_scheduled(session, facts, result, input_encoding, None, cap, output)
}
fn deliver_scheduled<T: Serialize>(session: &Session, facts: &ArtifactIdentity, result: &T,
    input_encoding: Option<&encoding::EncodingReport>, compute: Option<&crate::hosted::Int8ComputeReceipt>,
    cap: usize, output: &mut impl Write) -> Result<(), CandidateError> {
    session.remaining()?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: result };
    // Both host output ownership and the CLI's aggregate staging reservation
    // remain live through complete serialization, external write and flush.
    let response = encoding::EncodedResponse { response, input_encoding };
    let response = compute::Response { response: &response, compute };
    publish_checked(&response, cap, output, &|| session.remaining().map(|_| ()))
}

fn check_completed(expected: &Preflight, result: &Int8SourceMapRun,
    control: &mut impl DecodeStepControl) -> Result<(), CandidateError> {
    expected.verify_completed_with_control(result, control)
        .map_err(|error| error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution))
}

#[cfg(test)] mod tests;
