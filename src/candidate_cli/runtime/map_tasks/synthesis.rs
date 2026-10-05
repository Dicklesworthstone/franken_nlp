//! End-to-end candidate CLI over the shared pinned compiler and charged host.
use super::*;
use crate::tasks::source_planning::quantized::long::{SourceMapTask,
    summary::synthesis::{SourceSummarySynthesis, Int8SummarySynthesisPreflight,
        hierarchy::Int8SummaryHierarchyPreflight}};

#[allow(clippy::too_many_arguments)]
fn prepare_metadata(source: &str, planner: &SourceTaskPlanner, request: SourceSummarySynthesis,
    budget: TaskBudget, context: &PlanContext<'_>, command: &MapCommand, mapping: Int8SourceMapLimits,
    control: &mut impl DecodeStepControl) -> Result<Int8SummarySynthesisPreflight, CandidateError> {
    // The actual chunk programs are compiled here, not placeholder text. Drop
    // every temporary program before weights load; the charged host recompiles
    // the same complete commitment before its one physical invocation proceeds.
    let prepared = planner.plan_int8_summary_synthesis_with_control(source, request,
        budget, context, command.host.planning(), mapping, control).map_err(planning_error)?;
    let expected = prepared.preflight_metadata();
    command.admit_work(expected.reserved_model_work().map_err(planning_error)?,
        expected.reserved_mask_visits().map_err(planning_error)?)?;
    Ok(expected)
}
enum Expected { Single(Int8SummarySynthesisPreflight), Hierarchical(Int8SummaryHierarchyPreflight) }
#[allow(clippy::too_many_arguments)]
fn prepare_mode(source: &str, planner: &SourceTaskPlanner, request: SourceSummarySynthesis,
    budget: TaskBudget, context: &PlanContext<'_>, command: &MapCommand, mapping: Int8SourceMapLimits,
    control: &mut impl DecodeStepControl) -> Result<Expected, CandidateError> {
    let Some(hierarchy) = command.summary.synthesis.hierarchy.limits(command.summary.synthesis.synthesize_summary)? else {
        return prepare_metadata(source, planner, request, budget, context, command, mapping, control).map(Expected::Single);
    };
    let prepared = planner.plan_int8_summary_hierarchy_with_control(source, request, hierarchy, budget,
        context, command.host.planning(), mapping, control).map_err(planning_error)?;
    let expected = prepared.preflight_metadata();
    command.admit_work(expected.reserved_model_work(), expected.reserved_mask_visits())?;
    Ok(Expected::Hierarchical(expected))
}
fn planning_error(error: crate::tasks::source_planning::quantized::long::Int8SourceMapError) -> CandidateError {
    use crate::native_engine::decode::DecodeCancellationKind;
    match error.cancellation() {
        Some(DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout) => CandidateError::Timeout,
        Some(_) => CandidateError::Execution,
        None => CandidateError::Planning,
    }
}
pub(super) fn execute(command: MapCommand, args: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    command.summary.synthesis.check_planning(command.host.planning())?;
    let session = Session::new(&args, limits)?;
    let text = session.read(&command.input, input, args.max_input_bytes)?;
    let option_text = command.options.as_ref()
        .map(|path| session.read(path, input, crate::candidate_cli::source::OPTIONS_BYTES)).transpose()?;
    let SourceMapTask::Summarize(options) = command.map_task(option_text.as_deref())? else { return Err(CandidateError::Arguments); };
    let request = command.summary.synthesis.request(options, command.host.planning())?;
    if text.is_empty() { return Err(CandidateError::Input); }
    session.remaining()?;
    let facts = session.facts(&args)?;
    let (planner, vocabulary) = planner()?;
    let budget = command.host.task_budget(limits);
    let identity = source_identity(&facts, &planner, crate::tasks::BuiltInTask::Summarize)?;
    let context = PlanContext::new(&identity, budget).map_err(|_| CandidateError::Planning)?;
    let capacity = planner.int8_map_capacity_with_control(&SourceMapTask::Summarize(options), budget,
        &context, command.host.planning(), &mut session.control()).map_err(|_| CandidateError::Planning)?;
    let mapping = command.mapping(capacity)?;
    let expected = prepare_mode(&text, &planner, request, budget, &context, &command, mapping, &mut session.control())?;
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let config = SourceMapConfig { identity, task: request, budget, planning: command.host.planning(), mapping,
        native: session.native(&args)?, preparation_reserve_bytes: limits.preparation_bytes,
        reduction_reserve_bytes: command.reduction_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)? };
    match expected {
        Expected::Single(expected) => {
            let result = session.engine.synthesize_int8_summary(&model, text, Arc::new(planner), Arc::new(vocabulary),
                config, cancellation).map_err(|_| CandidateError::Execution)?;
            expected.verify_completed(result.result()).map_err(|_| CandidateError::Execution)?;
            deliver(&session, &facts, &result, command.max_map_result_bytes + 4096, output)
        }
        Expected::Hierarchical(expected) => {
            let result = session.engine.synthesize_int8_summary_hierarchical(&model, text, Arc::new(planner), Arc::new(vocabulary),
                config, expected.limits(), cancellation).map_err(|_| CandidateError::Execution)?;
            expected.verify_completed(result.result()).map_err(|_| CandidateError::Execution)?;
            deliver(&session, &facts, &result, command.max_map_result_bytes + 4096, output)
        }
    }
}
#[cfg(test)] mod tests;
