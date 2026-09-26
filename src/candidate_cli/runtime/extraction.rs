//! A sealed raw-schema plan on the existing process-hosted INT8 decoder.
use super::*;
use std::sync::Arc;
use super::source_tasks::Session;
use crate::{
    batch::{BatchDocument, extract::{ExtractionBatchArgs,
        quantized::{Int8ExtractionBatchPlanner, PreparedInt8BatchExtraction}}},
    candidate_cli::extract::{self as command, ExtractCommand},
    native_engine::decode::{DecodeCancellationKind, DecodeStepControl},
    tasks::extract::ExtractionVocabulary,
};

pub(super) fn planner(facts: &ArtifactIdentity, host: &source::SourceHostArgs, limits: Limits,
    defaults: Option<ExtractionBatchArgs>) -> Result<Int8ExtractionBatchPlanner, CandidateError> {
    let controls = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let eos = controls.template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END)
        .map(|entry| entry.id).ok_or(CandidateError::Identity)?;
    let mut identity = candidate_identity(facts)?;
    identity.task_spec = "extract-v1".to_owned();
    Int8ExtractionBatchPlanner::pinned(controls.template_controls(), eos, identity,
        host.task_budget(limits), command::compiler(host), host.planning().source, defaults)
        .map_err(|_| CandidateError::Planning)
}

pub(super) fn vocabulary() -> Result<Arc<ExtractionVocabulary>, CandidateError> {
    let controls = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    ExtractionVocabulary::pinned(controls.template_controls()).map(Arc::new)
        .map_err(|_| CandidateError::Planning)
}

fn prepare(planner: &Int8ExtractionBatchPlanner, document: String, request: ExtractionBatchArgs,
    host: &source::SourceHostArgs, control: &mut impl DecodeStepControl)
    -> Result<PreparedInt8BatchExtraction, CandidateError> {
    let prepared = planner.prepare_with_control(BatchDocument {
        id: "cli".to_owned(), text: document, task_args: Some(request),
    }, control).map_err(|error| match error.fault.cancellation {
        Some(DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout) => CandidateError::Timeout,
        Some(_) => CandidateError::Execution,
        None => CandidateError::Planning,
    })?;
    // The compiler's task ceiling reserves output space before it sees input.
    // Check the actual compiled work as well, without truncation or fallback.
    if prepared.planned_work().forward_positions > host.context_tokens as u64 {
        return Err(CandidateError::Planning);
    }
    prepared.verify_identity(prepared.execution_identity()).map_err(|_| CandidateError::Identity)?;
    Ok(prepared)
}

pub(in crate::candidate_cli) fn execute(command: ExtractCommand, args: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    // Storage-before-charge lifetime: this owner is declared before schema,
    // document, tokenizer, program, executable and output-staging allocations.
    let session = Session::new(&args, limits)?;
    let schema = session.read(&command.schema, input, command::SCHEMA_BYTES)?;
    let document = session.read(&command.input, input, args.max_input_bytes)?;
    let request = command::arguments(schema, command.source_membership, command.host.task_budget(limits))?;
    command::check_schema(&request, &document, &command.host)?;
    session.remaining()?;
    let facts = session.facts(&args)?;
    let compiler = planner(&facts, &command.host, limits, None)?;
    let prepared = prepare(&compiler, document, request, &command.host, &mut session.control())?;
    drop(compiler);
    session.remaining()?;
    let vocabulary = vocabulary()?;
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let result = session.engine.execute_int8_extract(&model, prepared.into_extraction_plan(), vocabulary,
        session.native(&args)?, command.host.masks(), command.host.max_mask_node_visits, cancellation)
        .map_err(|_| CandidateError::Execution)?;
    session.remaining()?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: &result };
    // The raw extracted JSON remains a STRING in the typed result. Converting
    // it through Value here would destroy the exact decimal/scalar contract.
    // Output and preparation guards remain live until external flush completes.
    publish(&response, command.host.max_result_bytes + 4096, output)
}

#[cfg(test)] mod tests;
