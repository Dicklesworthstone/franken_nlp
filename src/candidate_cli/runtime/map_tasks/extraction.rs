//! Exact-schema document extraction; all actual chunks compile before weights.
use super::*;
use super::super::extraction as compiler;
use crate::{
    batch::extract::{ExtractionBatchArgs, quantized::{Int8ExtractionBatchPlanner,
        long::{Int8ExtractionMapLimits, Int8ExtractionMapPreflight}}},
    candidate_cli::extract as schema,
    hosted::ExtractionMapConfig,
    native_engine::decode::DecodeCancellationKind,
    tasks::source_planning::quantized::long::Int8SourceMapError,
};

/// Retain only geometry/work metadata, not a second set of live schema plans.
/// The host independently rebuilds and seals all actual chunks under its charge.
fn preflight(source: &str, request: &ExtractionBatchArgs, planner: &Int8ExtractionBatchPlanner,
    mapping: Int8ExtractionMapLimits, control: &mut impl DecodeStepControl)
    -> Result<Int8ExtractionMapPreflight, CandidateError> {
    if source.is_empty() { return Err(CandidateError::Input); }
    let prepared = planner.plan_document_with_control(source, request, mapping, control).map_err(planning_error)?;
    Ok(prepared.preflight_metadata())
}
fn planning_error(error: Int8SourceMapError) -> CandidateError {
    match error.cancellation() {
        Some(DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout) => CandidateError::Timeout,
        Some(_) => CandidateError::Execution,
        None => CandidateError::Planning,
    }
}

pub(super) fn execute(command: MapCommand, args: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    // Declared before source/schema/compiler allocations: this preparation
    // charge remains live until external serialization, write and flush finish.
    let session = Session::new(&args, limits)?;
    let schema_text = session.read(command.extraction.path()?, input, schema::SCHEMA_BYTES)?;
    let text = session.read(&command.input, input, args.max_input_bytes)?;
    if text.is_empty() { return Err(CandidateError::Input); }
    let request = schema::arguments(schema_text, command.extraction.source_membership, command.host.task_budget(limits))?;
    let mapping = command.extraction_mapping()?;
    session.remaining()?;
    let facts = session.facts(&args)?;
    let planner = compiler::planner(&facts, &command.host, limits, None)?;
    // Do not compile one whole-document source automaton here: the source may
    // exceed a single-context source limit. Validate the exact schema against
    // EVERY real chunk instead, never a placeholder and never truncated input.
    let expected = preflight(&text, &request, &planner, mapping, &mut session.control())?;
    command.admit_work(expected.planned_work(), expected.reserved_mask_visits())?;
    let vocabulary = compiler::vocabulary()?;
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let config = ExtractionMapConfig { request, limits: mapping, native: session.native(&args)?,
        preparation_reserve_bytes: limits.preparation_bytes,
        reduction_reserve_bytes: command.reduction_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)? };
    let result = session.engine.extract_int8_document(&model, text, Arc::new(planner), vocabulary,
        config, cancellation).map_err(|_| CandidateError::Execution)?;
    expected.verify_completed(result.result()).map_err(|_| CandidateError::Execution)?;
    // Exact native JSON remains a STRING. No serde Value roundtrip, repair,
    // object merge, source-field synthesis or partially successful publication.
    deliver(&session, &facts, &result, command.max_map_result_bytes + 4096, output)
}

#[cfg(test)] mod tests;
