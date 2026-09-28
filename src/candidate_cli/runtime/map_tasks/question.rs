//! Complete-document QA through one existing process host and native model.
use super::*;
use crate::{candidate_cli::map::question::QUESTION_BYTES,
    tasks::BuiltInTask};
mod synthesis;

pub(super) fn execute(command: MapCommand, args: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    if command.question.synthesis.synthesize_answer { return synthesis::execute(command, args, limits, input, output); }
    // First local owner: all question/source/planner/result storage drops before
    // the CLI's preparation charge, including any error before model loading.
    let session = Session::new(&args, limits)?;
    let text = session.read(&command.input, input, args.max_input_bytes)?;
    let question = session.read(command.question.path()?, input, QUESTION_BYTES)?;
    let option_text = command.options.as_ref()
        .map(|path| session.read(path, input, crate::candidate_cli::source::OPTIONS_BYTES)).transpose()?;
    let task = command.question.task(question, option_text.as_deref())?;
    let mapping = command.question_mapping()?;
    let budget = command.host.task_budget(limits);
    let facts = session.facts(&args)?;
    let (planner, vocabulary) = planner()?;
    let identity = source_identity(&facts, &planner, BuiltInTask::Answer)?;
    let context = PlanContext::new(&identity, budget).map_err(|_| CandidateError::Planning)?;
    // Actual question + actual manifest + actual source ids for EVERY nonblank
    // passage. No second prepared grammar set or weights are needed to refuse
    // a whole-document work, passage count, input or context overflow here.
    let expected = planner.preflight_int8_question_with_control(&text, &task, budget, &context,
        command.host.planning(), mapping, &mut session.control()).map_err(|_| CandidateError::Planning)?;
    command.admit_work(expected.planned_work(), expected.reserved_mask_visits())?;
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let config = SourceMapConfig { identity, task, budget, planning: command.host.planning(), mapping,
        native: session.native(&args)?, preparation_reserve_bytes: limits.preparation_bytes,
        reduction_reserve_bytes: command.reduction_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)? };
    let result = session.engine.answer_int8_document(&model, text, Arc::new(planner), Arc::new(vocabulary),
        config, cancellation).map_err(|_| CandidateError::Execution)?;
    session.remaining()?;
    // Verify exact work, original source coverage, blank/native status counts,
    // outcome and verification ceilings, not just successful serialization.
    expected.verify_completed(result.result()).map_err(|_| CandidateError::Execution)?;
    // Output and preparation guards remain alive through write AND flush.
    deliver(&session, &facts, &result, command.max_map_result_bytes + 4096, output)
}

#[cfg(test)] mod tests;
