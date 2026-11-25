//! Complete judge requests on the existing candidate model and process host.
use super::*;
use super::source_tasks::Session;
use crate::{candidate_cli::{judge::{self as command, JudgeCommand}, scored::ScoredArgs},
    hosted::HostedError,
    native_engine::decode::{DecodeCancellationKind, DecodeStepControl},
    tasks::{ir::PlanContext, judge::{JudgePlanner, JudgeRequest,
        quantized::{Int8JudgeError, PreparedInt8Judge}}},
    tokenizer::untrusted::{SafeBpeWork, UntrustedEncoding}};

fn prepare(request: &JudgeRequest, facts: &ArtifactIdentity, args: &ScoredArgs,
    budget: TaskBudget, encoding: &UntrustedEncoding, control: &mut impl DecodeStepControl)
    -> Result<PreparedInt8Judge, CandidateError> {
    poll(control)?;
    let mut identity = candidate_identity(facts)?;
    let registry = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END)
        .map(|e| e.id).ok_or(CandidateError::Identity)?;
    let planner = JudgePlanner::pinned(controls, eos)
        .and_then(|planner| planner.with_document_encoding(encoding.clone()));
    poll(control)?;
    let planner = planner.map_err(|_| CandidateError::Planning)?;
    identity.task_spec = "judge-v1".to_owned();
    identity.tokenizer_digest = planner.tokenizer_digest();
    identity.template_digest = *planner.template_digest();
    let context = PlanContext::new(&identity, budget).map_err(|_| CandidateError::Planning)?;
    let prepared = planner.plan_int8_with_control(request, &context,
        command::planning_limits(args, budget), control);
    if let Some(cause) = prepared.as_ref().err().and_then(|error| error.cancellation()) {
        return Err(stopped(cause));
    }
    // The same original refusal remains authoritative even when compilation
    // also reports an ordinary planning or tokenizer-cap failure.
    poll(control)?;
    let prepared = prepared.map_err(planning_error)?;
    // Independent heads reuse their largest context, but all five work axes
    // are summed. Nothing executes or loads until the COMPLETE bundle fits.
    let admitted = args.admit_plan(prepared.required_context(), prepared.planned_work());
    poll(control)?;
    admitted?;
    Ok(prepared)
}
fn stopped(cause: DecodeCancellationKind) -> CandidateError {
    match cause {
        DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout => CandidateError::Timeout,
        _ => CandidateError::Execution,
    }
}
fn planning_error(error: Int8JudgeError) -> CandidateError {
    error.cancellation().map(stopped).unwrap_or(CandidateError::Planning)
}
fn host_cancellation(error: &HostedError) -> Option<DecodeCancellationKind> {
    match error {
        HostedError::Stopped { stop, .. } => Some(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => host_cancellation(error),
        HostedError::Judge(error) => error.cancellation(),
        HostedError::Native(crate::native_engine::strict_int8::StrictInt8Error::Cancelled(cause)) => Some(*cause),
        _ => None,
    }
}
fn host_error(error: HostedError) -> CandidateError {
    if let Some(cause) = host_cancellation(&error) { return stopped(cause); }
    match error {
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => host_error(*error),
        HostedError::Reservation(_) => CandidateError::Memory,
        HostedError::ModelIdentity | HostedError::ResourceDomain => CandidateError::Identity,
        _ => CandidateError::Execution,
    }
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), CandidateError> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(stopped(cause)), None => Ok(()) }
}

pub(in crate::candidate_cli) fn execute(command: JudgeCommand, args: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let prefill = command.prefill.limits()?;
    if command.validate()?.1 != limits || args.compute != command.compute {
        return Err(CandidateError::Arguments);
    }
    let encoding = command.encoding_limits.policy(&command.encoding)?;
    // Session owns preparation before any request/tokenizer allocations and
    // survives native execution and the completed response's external delivery.
    let session = Session::new_with_prefill(&args, limits, prefill)?;
    let text = session.read(&command.args.input, input, args.max_input_bytes)?;
    let budget = command.args.budget(limits);
    let request = command::request(&text, budget, args.max_input_bytes)?;
    session.remaining()?;
    let facts = session.facts(&args)?;
    let plan = prepare(&request, &facts, &command.args, budget, &encoding, &mut session.control())?;
    let encoding_work = plan.input_encoding_work();
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let native = session.native(&args)?;
    let result = if limits.compute.threads == 1 {
        match prefill {
            Some(prefill) => session.engine.execute_int8_judge_layer_major(&model, plan, native, prefill, cancellation),
            None => session.engine.execute_int8_judge(&model, plan, native, cancellation),
        }.map(compute::Completion::Serial)
    } else {
        let kernel = crate::native_engine::simd::config::process_kernel();
        match prefill {
            Some(prefill) => session.engine.execute_int8_judge_layer_major_with_threads(
                &model, plan, native, kernel, limits.compute.threads, prefill, cancellation),
            None => session.engine.execute_int8_judge_with_threads(
                &model, plan, native, kernel, limits.compute.threads, cancellation),
        }.map(compute::Completion::Team)
    };
    if let Some(cause) = result.as_ref().err().and_then(host_cancellation) { return Err(stopped(cause)); }
    session.remaining()?;
    let result = result.map_err(host_error)?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: result.result() };
    // A failure in either order, any criterion or any evidence head has no
    // output. Hosted result ownership remains attached through write AND flush.
    let response = compute::Response { response: &response, compute: result.compute() };
    let response = EncodingResponse { response: &response,
        input_encoding: input_encoding::reported_policy(&encoding), input_encoding_work: encoding_work };
    publish_checked(&response, command.args.max_result_bytes + 4096, output,
        &|| session.remaining().map(|_| ()))
}

#[derive(Serialize)]
struct EncodingResponse<'a, T: Serialize> {
    #[serde(flatten)] response: &'a T,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding_work: Option<SafeBpeWork>,
}

#[cfg(test)] mod tests;
