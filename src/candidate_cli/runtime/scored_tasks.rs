//! Candidate classification/sentiment reuse the same admitted model and host.
use super::*;
use super::source_tasks::Session;
use crate::{
    candidate_cli::scored::{self as command, Request, ScoredArgs, ScoredCommand},
    hosted::SentimentHostLimits,
    native_engine::decode::DecodeStepControl,
    tasks::{ir::PlanContext,
        classify::{ClassificationPlanner, quantized::PreparedInt8Classification},
        sentiment::{SentimentPlanner, SentimentOptions, quantized::PreparedInt8Sentiment}},
};
use crate::native_engine::lmhead::scoring::ScoringMode;

enum Prepared { Classify(PreparedInt8Classification), Sentiment(PreparedInt8Sentiment) }
fn prepare(request: &Request, facts: &ArtifactIdentity, args: &ScoredArgs, control: &mut impl DecodeStepControl)
    -> Result<Prepared, CandidateError> {
    if control.prefill_checkpoint(0).is_some() { return Err(CandidateError::Planning); }
    let mut identity = candidate_identity(facts)?;
    let registry = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END)
        .map(|e| e.id).ok_or(CandidateError::Identity)?;
    let prepared = match request {
        Request::Classify(request) => {
            let planner = ClassificationPlanner::pinned(controls, eos).map_err(|_| CandidateError::Planning)?;
            identity.task_spec = "classify-v1".to_owned();
            identity.template_digest = *planner.template_digest();
            identity.tokenizer_digest = planner.tokenizer_digest();
            let context = PlanContext::new(&identity, request.budget).map_err(|_| CandidateError::Planning)?;
            let plan = planner.plan_int8_with_control(request, &context, args.classification_limits(), control)
                .map_err(|_| CandidateError::Planning)?;
            args.admit_plan(plan.required_context(), plan.planned_work())?;
            Prepared::Classify(plan)
        }
        Request::Sentiment { request, policy } => {
            let planner = SentimentPlanner::pinned(controls, SentimentOptions {
                mode: ScoringMode::FullVocabulary, eos_token_id: eos, policy: *policy,
            }).map_err(|_| CandidateError::Planning)?;
            identity.task_spec = "sentiment-v1".to_owned();
            identity.template_digest = *planner.template_digest();
            identity.tokenizer_digest = planner.tokenizer_digest();
            let context = PlanContext::new(&identity, request.budget).map_err(|_| CandidateError::Planning)?;
            let plan = planner.plan_int8_with_control(request, &context, args.sentiment_limits(), control)
                .map_err(|_| CandidateError::Planning)?;
            args.admit_plan(plan.required_context(), plan.planned_work())?;
            Prepared::Sentiment(plan)
        }
    };
    if control.prefill_checkpoint(0).is_some() { return Err(CandidateError::Planning); }
    Ok(prepared)
}

pub(in crate::candidate_cli) fn execute(command: ScoredCommand, common: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    // Session precedes every input, planner and output allocation. Its modeled
    // preparation charge survives all returned failures and final delivery.
    let session = Session::new(&common, limits)?;
    let text = session.read(&command.args.input, input, common.max_input_bytes)?;
    let request = command::request(command.kind, &text, command.args.budget(limits), common.max_input_bytes)?;
    session.remaining()?;
    let facts = session.facts(&common)?;
    let prepared = prepare(&request, &facts, &command.args, &mut session.control())?;
    session.remaining()?;
    // Only a complete bound plan with ALL work axes admitted can load weights.
    let cancellation = CancellationToken::default();
    let model = session.load(&common, limits, &facts, cancellation.clone())?;
    let cap = command.args.max_result_bytes + 4096;
    match prepared {
        Prepared::Classify(plan) => {
            let result = session.engine.execute_int8_classify(&model, plan, session.native(&common)?, cancellation)
                .map_err(|_| CandidateError::Execution)?;
            deliver(&session, &facts, &result, cap, output)
        }
        Prepared::Sentiment(plan) => {
            let result = session.engine.execute_int8_sentiment(&model, plan, SentimentHostLimits {
                native: session.native(&common)?, preparation_reserve_bytes: limits.preparation_bytes,
                max_model_work: command.args.work_ceiling(),
            }, cancellation).map_err(|_| CandidateError::Execution)?;
            deliver(&session, &facts, &result, cap, output)
        }
    }
}
fn deliver<T: Serialize>(session: &Session, facts: &ArtifactIdentity, guarded: &T,
    cap: usize, output: &mut impl Write) -> Result<(), CandidateError> {
    session.remaining()?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: guarded };
    // guarded is the caller's HostedOutput; it and Session are still alive
    // through serialization, external write and flush. No partial head output.
    publish(&response, cap, output)
}

#[cfg(test)] mod tests;
