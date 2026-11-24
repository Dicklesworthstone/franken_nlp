//! Complete-document INT8 scoring on the existing resident model and host.
use super::source_tasks::Session;
use super::*;
use crate::{
    candidate_cli::{
        scored::{self, Kind, Request},
        sentiment_map::{MapLimits, SentimentMapCommand},
    },
    hosted::{HostedError, Int8SentimentMapHostConfig},
    native_engine::{decode::DecodeCancellationKind, lmhead::scoring::ScoringMode},
    tasks::sentiment::{SentimentOptions, SentimentPlanner},
};
use std::sync::Arc;

pub(in crate::candidate_cli) fn execute(
    command: SentimentMapCommand,
    admitted: MapLimits,
    input: &mut impl Read,
    output: &mut impl Write,
) -> Result<(), CandidateError> {
    let session = Session::new_with_prefill(&admitted.common, admitted.shared, admitted.prefill)?;
    let text = session.read(&command.args.input, input, command.args.max_input_bytes)?;
    let Request::Sentiment { request, policy } = scored::request(
        Kind::Sentiment,
        &text,
        admitted.task,
        command.args.max_input_bytes,
    )?
    else {
        return Err(CandidateError::Input);
    };
    drop(text);
    session.remaining()?;
    let facts = session.facts(&admitted.common)?;
    let registry = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let controls = registry.template_controls();
    let eos = controls
        .entries()
        .iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END)
        .map(|entry| entry.id)
        .ok_or(CandidateError::Identity)?;
    let planner = Arc::new(
        SentimentPlanner::pinned(
            controls,
            SentimentOptions {
                mode: ScoringMode::FullVocabulary,
                eos_token_id: eos,
                policy,
            },
        )
        .and_then(|planner| planner.with_document_encoding(admitted.encoding.clone()))
        .map_err(|_| CandidateError::Planning)?,
    );
    let mut identity = candidate_identity(&facts)?;
    identity.task_spec = "sentiment-v1".to_owned();
    identity.tokenizer_digest = planner.tokenizer_digest();
    identity.template_digest = *planner.template_digest();
    let cancellation = CancellationToken::default();
    let model = session.load(
        &admitted.common,
        admitted.shared,
        &facts,
        cancellation.clone(),
    )?;
    let config = Int8SentimentMapHostConfig {
        identity,
        native: session.native(&admitted.common)?,
        planning: command.args.sentiment_limits(),
        mapping: admitted.mapping,
        preparation_reserve_bytes: admitted.shared.preparation_bytes,
    };
    if config
        .requirements(admitted.task, 4, &admitted.encoding)
        .map_err(execution_error)?
        != admitted.memory
    {
        return Err(CandidateError::Arguments);
    }
    // All chunks are prepared exactly once, inside the charged host. Compact
    // tokenization receives one nonrenewable ledger for that entire planning.
    let result = if admitted.shared.compute.threads == 1 {
        compute::Completion::Serial(match admitted.prefill {
            None => session.engine.map_int8_sentiment(&model, request, planner, config, cancellation),
            Some(prefill) => session.engine.map_int8_sentiment_layer_major(
                &model, request, planner, config, prefill, cancellation),
        }.map_err(execution_error)?)
    } else {
        let kernel = crate::native_engine::simd::config::process_kernel();
        compute::Completion::Team(match admitted.prefill {
            None => session.engine.map_int8_sentiment_with_threads(&model, request, planner,
                config, kernel, admitted.shared.compute.threads, cancellation),
            Some(prefill) => session.engine.map_int8_sentiment_layer_major_with_threads(
                &model, request, planner, config, kernel, admitted.shared.compute.threads,
                prefill, cancellation),
        }.map_err(execution_error)?)
    };
    session.remaining()?;
    let response = CandidateResponse {
        schema_version: 1,
        scope: "real-artifact-current-candidate",
        evidence: "non_authoritative",
        model_id: &facts.model_id,
        source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256,
        logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id,
        output: result.result(),
    };
    let response = compute::Response { response: &response, compute: result.compute() };
    // The map reports its exact encoder policy/work; preserve the ordinary
    // provenance wrapper and retain both ownership guards through flush.
    publish_checked(&response, admitted.envelope_bytes, output, &|| {
        session.remaining().map(|_| ())
    })
}
fn stopped(cause: DecodeCancellationKind) -> CandidateError {
    match cause {
        DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout => {
            CandidateError::Timeout
        }
        _ => CandidateError::Execution,
    }
}
fn execution_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => stopped(stop.kind),
        HostedError::Join {
            physical_error: Some(error),
            ..
        }
        | HostedError::Scope {
            physical_error: Some(error),
            ..
        } => execution_error(*error),
        HostedError::Native(crate::native_engine::strict_int8::StrictInt8Error::Cancelled(cause)) => {
            stopped(cause)
        }
        HostedError::SentimentMap(error) => error
            .cancellation()
            .map(stopped)
            .unwrap_or(CandidateError::Execution),
        HostedError::Reservation(_) => CandidateError::Memory,
        HostedError::ModelIdentity | HostedError::ResourceDomain => CandidateError::Identity,
        _ => CandidateError::Execution,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::sentiment::quantized::{Int8SentimentError, long::Int8SentimentMapError};
    #[test]
    fn typed_document_cancellation_keeps_deadline_exit_semantics() {
        for kind in [
            DecodeCancellationKind::Deadline,
            DecodeCancellationKind::Timeout,
        ] {
            assert_eq!(
                execution_error(HostedError::SentimentMap(Int8SentimentMapError::Sentiment(
                    Int8SentimentError::Cancelled(kind)
                ))),
                CandidateError::Timeout
            );
        }
        assert_eq!(
            execution_error(HostedError::SentimentMap(Int8SentimentMapError::Sentiment(
                Int8SentimentError::Cancelled(DecodeCancellationKind::PollQuota)
            ))),
            CandidateError::Execution
        );
    }

    #[test]
    fn direct_native_cancellation_keeps_every_cause_and_deadline_exit_semantics() {
        use crate::native_engine::strict_int8::StrictInt8Error;
        for cause in [
            DecodeCancellationKind::User,
            DecodeCancellationKind::Deadline,
            DecodeCancellationKind::Timeout,
            DecodeCancellationKind::PollQuota,
            DecodeCancellationKind::CostBudget,
            DecodeCancellationKind::RaceLost,
            DecodeCancellationKind::ParentCancelled,
            DecodeCancellationKind::ResourceUnavailable,
            DecodeCancellationKind::Shutdown,
            DecodeCancellationKind::FailFast,
            DecodeCancellationKind::LinkedExit,
        ] {
            assert_eq!(
                execution_error(HostedError::Native(StrictInt8Error::Cancelled(cause))),
                if matches!(cause, DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout) {
                    CandidateError::Timeout
                } else {
                    CandidateError::Execution
                }
            );
        }
    }

    #[test]
    fn mismatched_worker_or_prompt_quotes_refuse_before_installing_a_runtime() {
        let parsed = crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "sentiment-map", "--model", "never-opened.fnlpq", "--memory-mib", "16384",
            "--compute-threads", "4", "--prefill-rows", "7",
        ]).unwrap();
        let CandidateCommand::SentimentMap(command) = CandidateCommand::from_matches(&parsed).unwrap()
            else { panic!("sentiment document required"); };
        let admitted = command.validate().unwrap();
        let mut wrong_workers = admitted.shared;
        wrong_workers.compute.workspace_bytes -= 1;
        assert_eq!(Session::new_with_prefill(&admitted.common, wrong_workers, admitted.prefill).err(),
            Some(CandidateError::Arguments));
        assert_eq!(Session::new_with_prefill(&admitted.common, admitted.shared, None).err(),
            Some(CandidateError::Arguments));
    }

}
