//! Complete judge requests on the existing candidate model and process host.
use super::*;
use super::source_tasks::Session;
use crate::{candidate_cli::{judge::{self as command, JudgeCommand}, scored::ScoredArgs},
    native_engine::decode::DecodeStepControl,
    tasks::{ir::PlanContext, judge::{JudgePlanner, JudgeRequest, quantized::PreparedInt8Judge}}};

fn prepare(request: &JudgeRequest, facts: &ArtifactIdentity, args: &ScoredArgs,
    budget: TaskBudget, control: &mut impl DecodeStepControl) -> Result<PreparedInt8Judge, CandidateError> {
    poll(control)?;
    let mut identity = candidate_identity(facts)?;
    let registry = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END)
        .map(|e| e.id).ok_or(CandidateError::Identity)?;
    let planner = JudgePlanner::pinned(controls, eos).map_err(|_| CandidateError::Planning)?;
    identity.task_spec = "judge-v1".to_owned();
    identity.tokenizer_digest = planner.tokenizer_digest();
    identity.template_digest = *planner.template_digest();
    let context = PlanContext::new(&identity, budget).map_err(|_| CandidateError::Planning)?;
    let prepared = planner.plan_int8_with_control(request, &context,
        command::planning_limits(args, budget), control).map_err(|_| CandidateError::Planning)?;
    // Independent heads reuse their largest context, but all five work axes
    // are summed. Nothing executes or loads until the COMPLETE bundle fits.
    args.admit_plan(prepared.required_context(), prepared.planned_work())?;
    poll(control)?;
    Ok(prepared)
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), CandidateError> {
    if control.prefill_checkpoint(0).is_some() { Err(CandidateError::Planning) } else { Ok(()) }
}

pub(in crate::candidate_cli) fn execute(command: JudgeCommand, args: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    // Session owns preparation before any request/tokenizer allocations and
    // survives native execution and the completed response's external delivery.
    let session = Session::new(&args, limits)?;
    let text = session.read(&command.args.input, input, args.max_input_bytes)?;
    let budget = command.args.budget(limits);
    let request = command::request(&text, budget, args.max_input_bytes)?;
    session.remaining()?;
    let facts = session.facts(&args)?;
    let plan = prepare(&request, &facts, &command.args, budget, &mut session.control())?;
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let result = session.engine.execute_int8_judge(&model, plan, session.native(&args)?, cancellation)
        .map_err(|_| CandidateError::Execution)?;
    session.remaining()?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: &result };
    // A failure in either order, any criterion or any evidence head has no
    // output. Hosted result ownership remains attached through write AND flush.
    publish(&response, command.args.max_result_bytes + 4096, output)
}

#[cfg(test)] mod tests;
