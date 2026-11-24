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
    let (session, prefill) = scheduled_session(&command, &args, limits)?;
    let text = session.read(&command.input, input, args.max_input_bytes)?;
    let question = session.read(command.question.path()?, input, QUESTION_BYTES)?;
    let option_text = command.options.as_ref()
        .map(|path| session.read(path, input, crate::candidate_cli::source::OPTIONS_BYTES)).transpose()?;
    let task = command.question.task(question, option_text.as_deref())?;
    let mapping = command.question_mapping()?;
    let budget = command.host.task_budget(limits);
    let facts = session.facts(&args)?;
    let (planner, vocabulary) = planner()?;
    let planner = match command.encoding.limits(&command)? {
        Some(encoding) => planner.with_map_document_encoding(encoding.policy(), encoding.complete)
            .map_err(|_| CandidateError::Planning)?,
        None => planner,
    };
    let identity = source_identity(&facts, &planner, BuiltInTask::Answer)?;
    let context = PlanContext::new(&identity, budget).map_err(|_| CandidateError::Planning)?;
    // Actual question + actual manifest + actual source ids for EVERY nonblank
    // passage. No second prepared grammar set or weights are needed to refuse
    // a whole-document work, passage count, input or context overflow here.
    let expected = planner.preflight_int8_question_with_control(&text, &task, budget, &context,
        command.host.planning(), mapping, &mut session.control()).map_err(question_error)?;
    command.admit_work(expected.planned_work(), expected.reserved_mask_visits())?;
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let config = SourceMapConfig { identity, task, budget, planning: command.host.planning(), mapping,
        native: session.native(&args)?, preparation_reserve_bytes: limits.preparation_bytes,
        reduction_reserve_bytes: command.reduction_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)? };
    // Keep the same operation owner across preflight and hosted reconstruction;
    // actual question/manifest/source plans spend the remaining shared ledger.
    let planner = Arc::new(planner);
    let vocabulary = Arc::new(vocabulary);
    let width = limits.compute.threads;
    let kernel = crate::native_engine::simd::config::process_kernel();
    let result = if width == 1 {
        compute::Completion::Serial(match prefill {
            None => session.engine.answer_int8_document(&model, text, Arc::clone(&planner),
                vocabulary, config, cancellation),
            Some(prefill) => session.engine.answer_int8_document_layer_major(&model, text,
                Arc::clone(&planner), vocabulary, config, prefill, cancellation),
        }.map_err(host_error)?)
    } else {
        compute::Completion::Team(match prefill {
            None => session.engine.answer_int8_document_with_threads(&model, text, Arc::clone(&planner),
                vocabulary, config, kernel, width, cancellation),
            Some(prefill) => session.engine.answer_int8_document_layer_major_with_threads(&model, text,
                Arc::clone(&planner), vocabulary, config, kernel, width, prefill, cancellation),
        }.map_err(host_error)?)
    };
    session.remaining()?;
    // Verify exact work, original source coverage, blank/native status counts,
    // outcome and verification ceilings, not just successful serialization.
    expected.verify_completed_with_control(result.result(), &mut session.control())
        .map_err(|error| error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution))?;
    // Output and preparation guards remain alive through write AND flush.
    deliver_scheduled(&session, &facts, result.result(), encoding::report(&planner)?.as_ref(),
        result.compute(), command.max_map_result_bytes + 4096, output)
}

fn question_error(error: crate::tasks::source_planning::quantized::long::Int8SourceMapError) -> CandidateError {
    error.cancellation().map(cancelled).unwrap_or(CandidateError::Planning)
}

#[cfg(test)] mod tests;
