//! One admitted resident INT8 invocation for extraction and every field judge.
use super::*;
use std::sync::Arc;
use super::source_tasks::Session;
use crate::{
    batch::{BatchDocument, extract::{ExtractionBatchArgs, ExtractionBatchGrounding}},
    candidate_cli::semantic::{self as command, SemanticCommand},
    grammar::runtime::JsonProgram,
    hosted::{HostedError, Int8SemanticHostLimits, Int8SemanticHostRequirements},
    native_engine::{decode::DecodeCancellationKind, strict_int8::{Int8Work, STRICT_INT8_PROFILE}},
    tasks::{extract::semantic::int8::{Int8SemanticError, Int8SemanticOptions,
        Int8SemanticExtractionRun, PreparedInt8SemanticExtraction, INT8_SEMANTIC_EXECUTION},
        judge::JudgePlanner},
};

fn cancelled(cause: DecodeCancellationKind) -> CandidateError {
    match cause { DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout => CandidateError::Timeout,
        _ => CandidateError::Execution }
}
fn planning_error(error: Int8SemanticError) -> CandidateError {
    error.cancellation().map(cancelled).unwrap_or(CandidateError::Planning)
}
fn execution_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => cancelled(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => execution_error(*error),
        HostedError::SemanticExtraction(error) => error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution),
        HostedError::Native(crate::native_engine::strict_int8::StrictInt8Error::Cancelled(cause)) => cancelled(cause),
        HostedError::Reservation(_) => CandidateError::Memory,
        HostedError::ModelIdentity | HostedError::ResourceDomain => CandidateError::Identity,
        _ => CandidateError::Execution,
    }
}

pub(in crate::candidate_cli) fn execute(command: SemanticCommand, args: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    // This owner is established before any private options/source/schema or
    // output allocation and lives through the final external flush.
    let session = Session::new(&args, limits)?;
    let options = command::parse_options(&session.read(&command.semantic_options, input, command::OPTIONS_BYTES)?)?;
    let envelope_bytes = command.admit_options(&args, limits, &options)?;
    let budget = command.host.task_budget(limits);
    let judge_budget = options.specification.judge_budget;
    let semantic_limits = options.limits;
    let planning = command::judge_limits(&options);
    let max_work = options.max_model_work;
    let host = host_limits(&session, &args, &command, limits)?;
    let upper = host.requirements(budget, judge_budget, semantic_limits, planning, args.max_input_bytes)
        .map_err(|_| CandidateError::Arguments)?;
    admit_memory(&args, limits, upper)?;
    // Fixed policy and whole-source capacity are admitted before these reads.
    let schema = session.read(&command.schema, input, extract::SCHEMA_BYTES)?;
    let document = session.read(&command.input, input, args.max_input_bytes)?;
    let source_bytes = document.len();
    let request = source_schema(&command, &options, schema, &document, budget)?;
    session.remaining()?;
    let quote = host.requirements(budget, judge_budget, semantic_limits, planning, source_bytes)
        .map_err(|_| CandidateError::Arguments)?;
    if !quote_fits(quote, upper) { return Err(CandidateError::Identity); }
    admit_memory(&args, limits, quote)?;
    let facts = session.facts(&args)?;
    let compiler = super::extraction::planner(&facts, &command.host, limits, None)?;
    let extraction = compiler.prepare_with_control(BatchDocument {
        id: "cli".to_owned(), text: document, task_args: Some(request),
    }, &mut session.control()).map_err(|error|
        error.fault.cancellation.map(cancelled).unwrap_or(CandidateError::Planning))?;
    if extraction.planned_work().forward_positions > args.context_tokens as u64 {
        return Err(CandidateError::Planning);
    }
    extraction.verify_identity(extraction.execution_identity()).map_err(|_| CandidateError::Identity)?;
    drop(compiler);
    let registry = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let controls = registry.template_controls();
    let eos = controls.entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END)
        .map(|entry| entry.id).ok_or(CandidateError::Identity)?;
    let judge = Arc::new(JudgePlanner::pinned(controls, eos).map_err(|_| CandidateError::Planning)?);
    let mut judge_identity = extraction.execution_identity().clone();
    judge_identity.task_spec = "judge-v1".to_owned();
    judge_identity.tokenizer_digest = judge.tokenizer_digest();
    judge_identity.template_digest = *judge.template_digest();
    let prepared = PreparedInt8SemanticExtraction::new(extraction, judge, judge_identity,
        planning, options, &mut session.control()).map_err(planning_error)?;
    let actual = host.requirements(prepared.extraction_budget(), prepared.judge_budget(),
        prepared.semantic_limits(), prepared.judge_planning_limits(), prepared.source_bytes())
        .map_err(|_| CandidateError::Arguments)?;
    if prepared.extraction_budget() != budget || prepared.judge_budget() != judge_budget
        || prepared.semantic_limits() != semantic_limits || prepared.source_bytes() != source_bytes
        || prepared.max_model_work() != max_work || actual != quote {
        return Err(CandidateError::Identity);
    }
    admit_memory(&args, limits, actual)?;
    let reserved = prepared.extraction_reserved_work();
    let vocabulary = super::extraction::vocabulary()?;
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let result = session.engine.execute_int8_semantic_extract(&model, prepared, vocabulary,
        host_limits(&session, &args, &command, limits)?, cancellation).map_err(execution_error)?;
    check_result(result.result(), reserved, max_work)?;
    session.remaining()?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: result.result() };
    // The original JSON remains an exact string, never a floating-point Value.
    // Result and caller preparation guards remain live through write and flush.
    publish_checked(&response, envelope_bytes, output, &|| session.remaining().map(|_| ()))
}

fn source_schema(command: &SemanticCommand, options: &Int8SemanticOptions, schema: String,
    source: &str, budget: TaskBudget) -> Result<ExtractionBatchArgs, CandidateError> {
    if source.len() > command.host.max_input_bytes { return Err(CandidateError::Input); }
    let request = extract::arguments(schema, command.source_membership, budget)?;
    let program = match request.grounding {
        ExtractionBatchGrounding::Structural => JsonProgram::compile(&request.schema, extract::compiler(&command.host)),
        ExtractionBatchGrounding::SourceMembership => JsonProgram::compile_with_source(&request.schema,
            source, extract::compiler(&command.host), command.host.planning().source),
    }.map_err(|_| CandidateError::Planning)?;
    if program.requires_source() != command.source_membership { return Err(CandidateError::Planning); }
    options.validate_source_schema(program.declarative_schema(), source).map_err(planning_error)?;
    Ok(request)
}

fn host_limits(session: &Session, args: &CandidateArgs, command: &SemanticCommand, limits: Limits)
    -> Result<Int8SemanticHostLimits, CandidateError> {
    Ok(Int8SemanticHostLimits { native: session.native(args)?, masks: command.host.masks(),
        max_mask_node_visits: command.host.max_mask_node_visits,
        preparation_reserve_bytes: limits.preparation_bytes })
}

fn minimum_memory(args: &CandidateArgs, limits: Limits, quote: Int8SemanticHostRequirements)
    -> Result<u64, CandidateError> {
    let required = Int8MemoryRequirement::for_context(args.context_tokens).map_err(|_| CandidateError::Arguments)?;
    if quote.native != required || limits.kv_bytes != required.kv_bytes
        || limits.compute != compute::Args::default().admit_prefill(None)?
        || args.compute.threads() != 1 {
        return Err(CandidateError::Arguments);
    }
    let sum = |values: &[u64]| values.iter().try_fold(0_u64,
        |total, &bytes| total.checked_add(bytes).ok_or(CandidateError::Arguments));
    let load = sum(&[limits.weight_bytes, 192 * MIB, 64 * 1024, 65_536])?;
    let execution = sum(&[limits.weight_bytes, 192 * MIB,
        quote.total_bytes().map_err(|_| CandidateError::Arguments)?, 65_536])?;
    sum(&[limits.preparation_bytes, load.max(execution)])
}
fn admit_memory(args: &CandidateArgs, limits: Limits, quote: Int8SemanticHostRequirements)
    -> Result<(), CandidateError> {
    if minimum_memory(args, limits, quote)? > limits.memory_bytes { return Err(CandidateError::Arguments); }
    Ok(())
}
fn quote_fits(actual: Int8SemanticHostRequirements, ceiling: Int8SemanticHostRequirements) -> bool {
    actual.native == ceiling.native && actual.native_workspace_bytes <= ceiling.native_workspace_bytes
        && actual.preparation_bytes <= ceiling.preparation_bytes && actual.output_bytes <= ceiling.output_bytes
}
fn fits(work: Int8Work, cap: Int8Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn check_result(run: &Int8SemanticExtractionRun, extraction: Int8Work, cap: Int8Work)
    -> Result<(), CandidateError> {
    let reserved = extraction.checked_add(run.semantic_model_work).map_err(|_| CandidateError::Execution)?;
    let actual = run.extraction.model_work.checked_add(run.semantic_model_work).map_err(|_| CandidateError::Execution)?;
    if run.schema_version != 1 || run.execution != INT8_SEMANTIC_EXECUTION || run.numerics_profile != STRICT_INT8_PROFILE
        || run.extraction.schema_version != 1 || run.extraction.execution != crate::tasks::extract::quantized::INT8_EXTRACT_VERSION
        || run.extraction_reserved_work != extraction || run.reserved_model_work != reserved || run.model_work != actual
        || !fits(run.extraction.model_work, extraction) || !fits(actual, reserved) || !fits(reserved, cap)
        || !run.semantic.same_model_correlated || run.semantic.calibration != "experimental_uncalibrated_diagnostic_only" {
        return Err(CandidateError::Execution);
    }
    Ok(())
}

#[cfg(test)] mod tests;
