//! Long-document candidate inference on the existing charged map/merge host.
use super::*;
mod summary;
mod question;
mod extraction;
mod synthesis;
mod keyphrase_reduction;
use std::{cell::RefCell, sync::Arc};
use super::source_tasks::{Session, planner, source_identity};
use crate::{candidate_cli::map::{MapCommand, encoding},
    hosted::{HostedError, SourceMapConfig},
    native_engine::{constrained_int8, decode::{DecodeCancellationKind, DecodeStepControl}, strict_int8::Int8Work},
    tasks::{ir::PlanContext, mapreduce::{ChunkPlan, MapReduceError},
        source_planning::{SourceTaskPlanner, quantized::{Int8SourceError, capacity::Int8SourceMapCapacity,
            long::{Int8SourceMapLimits, Int8SourceMapRun}}}}};

/// Geometry/work only; no source bytes, prompt commitments or native receipt.
struct Preflight { chunks: usize, source_bytes: usize, source_scalars: usize, work: Int8Work, masks: u64 }

fn preflight(source: &str, planner: &SourceTaskPlanner, command: &MapCommand,
    capacity: Int8SourceMapCapacity, mapping: Int8SourceMapLimits, budget: TaskBudget,
    control: &mut impl DecodeStepControl) -> Result<Preflight, CandidateError> {
    poll(control)?;
    if source.is_empty() { return Err(CandidateError::Input); }
    // Same partitioner and exact source counter as the native map compiler.
    // This temporary metadata plan drops before loading and retains no source
    // grammar or second prepared task. The host independently compiles every
    // actual chunk before the first native forward.
    planner.map_document_encoding_workspace_bytes().map_err(|_| CandidateError::Planning)?;
    let mut failure = None;
    let mut cancellation = None;
    let chunks = {
        let control = RefCell::new(&mut *control);
        ChunkPlan::build_with_checkpoints(source, mapping.chunks, |text| {
            planner.source_encoder().encode_with_control(text, mapping.chunks.max_chunk_bytes,
                mapping.chunks.max_chunk_bytes, &mut || control.borrow_mut().prefill_checkpoint(0))
                .map(|s| s.total_token_count()).map_err(|error| {
                    failure = Some(Int8SourceError::from(error)); MapReduceError::Tokenizer
                })
        }, || match control.borrow_mut().prefill_checkpoint(0) {
            Some(cause) => { cancellation = Some(cause); Err(MapReduceError::Cancelled) }, None => Ok(()),
        })
    };
    if let Some(cause) = cancellation { return Err(cancelled(cause)); }
    if let Some(error) = failure { return Err(source_error(error)); }
    let chunks = chunks.map_err(|_| CandidateError::Planning)?;
    let mut work = Int8Work::default();
    let mut masks = 0_u64;
    for chunk in chunks.chunks() {
        poll(control)?;
        let prompt = capacity.scaffold_tokens().checked_add(chunk.tokens()).ok_or(CandidateError::Planning)?;
        if prompt > budget.max_input_tokens as usize
            || prompt.checked_add(budget.max_output_tokens as usize)
                .is_none_or(|n| n > command.host.context_tokens) { return Err(CandidateError::Planning); }
        work = work.checked_add(constrained_int8::planned_work(prompt, budget.max_output_tokens as usize)
            .map_err(|_| CandidateError::Planning)?).map_err(|_| CandidateError::Planning)?;
        masks = masks.checked_add(mapping.mask_visits_per_chunk).ok_or(CandidateError::Planning)?;
        command.admit_work(work, masks)?;
    }
    poll(control)?;
    Ok(Preflight { chunks: chunks.chunks().len(), source_bytes: source.len(),
        source_scalars: source.chars().count(), work, masks })
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), CandidateError> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(cancelled(cause)), None => Ok(()) }
}
fn cancelled(cause: DecodeCancellationKind) -> CandidateError {
    match cause { DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout => CandidateError::Timeout,
        _ => CandidateError::Execution }
}
fn source_error(error: Int8SourceError) -> CandidateError {
    error.cancellation().map(cancelled).unwrap_or(CandidateError::Planning)
}
fn host_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => cancelled(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => host_error(*error),
        HostedError::SourceMap(error) => error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution),
        HostedError::Source(error) => error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution),
        HostedError::Reservation(_) => CandidateError::Memory,
        _ => CandidateError::Execution,
    }
}

pub(in crate::candidate_cli) fn execute(command: MapCommand, args: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    if command.kind()? == crate::tasks::BuiltInTask::Answer {
        return question::execute(command, args, limits, input, output);
    }
    if command.kind()? == crate::tasks::BuiltInTask::Extract {
        return extraction::execute(command, args, limits, input, output);
    }
    if command.summary.synthesis.synthesize_summary {
        return synthesis::execute(command, args, limits, input, output);
    }
    let session = Session::new(&args, limits)?;
    let text = session.read(&command.input, input, args.max_input_bytes)?;
    let option_text = command.options.as_ref()
        .map(|path| session.read(path, input, crate::candidate_cli::source::OPTIONS_BYTES)).transpose()?;
    let task = command.map_task(option_text.as_deref())?;
    if text.is_empty() { return Err(CandidateError::Input); }
    session.remaining()?;
    let facts = session.facts(&args)?;
    let (planner, vocabulary) = planner()?;
    let planner = match command.encoding.limits(&command)? {
        Some(encoding) => planner.with_map_document_encoding(encoding.policy(), encoding.complete)
            .map_err(|_| CandidateError::Planning)?,
        None => planner,
    };
    let budget = command.host.task_budget(limits);
    let identity = source_identity(&facts, &planner, command.kind()?)?;
    let context = PlanContext::new(&identity, budget).map_err(|_| CandidateError::Planning)?;
    let capacity = planner.int8_map_capacity_with_control(&task, budget, &context,
        command.host.planning(), &mut session.control()).map_err(source_error)?;
    let mapping = command.mapping(capacity)?;
    let summary = command.summary.limits(command.max_map_result_bytes)?;
    if let Some(limits) = summary { limits.validate(mapping).map_err(|_| CandidateError::Planning)?; }
    let keyphrases = command.keyphrases.limits(command.max_map_result_bytes)?;
    if let Some(limits) = keyphrases { limits.validate(mapping).map_err(|_| CandidateError::Planning)?; }
    let expected = preflight(&text, &planner, &command, capacity, mapping, budget, &mut session.control())?;
    session.remaining()?;
    // No weights until the complete partition and all five work axes fit.
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let config = SourceMapConfig { identity, task, budget, planning: command.host.planning(), mapping,
        native: session.native(&args)?, preparation_reserve_bytes: limits.preparation_bytes,
        reduction_reserve_bytes: command.reduction_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)? };
    // Retain the exact planner, not a fresh ledger, through hosted recompilation
    // and final usage reporting. All private source storage stays charged.
    let planner = Arc::new(planner);
    if let Some(limits) = summary {
        let result = session.engine.summarize_int8_source(&model, text, Arc::clone(&planner), Arc::new(vocabulary),
            config, limits, cancellation).map_err(host_error)?;
        summary::check_completed(&expected, result.result(), limits.max_bullets)?;
        return deliver_encoded(&session, &facts, &result, encoding::report(&planner)?.as_ref(),
            command.max_map_result_bytes + 4096, output);
    }
    if let Some(limits) = keyphrases {
        let result = session.engine.keyphrases_int8_source(&model, text, Arc::clone(&planner), Arc::new(vocabulary),
            config, limits, cancellation).map_err(host_error)?;
        keyphrase_reduction::check_completed(&expected, result.result(), limits)?;
        return deliver_encoded(&session, &facts, &result, encoding::report(&planner)?.as_ref(),
            command.max_map_result_bytes + 4096, output);
    }
    let result = session.engine.map_int8_source(&model, text, Arc::clone(&planner), Arc::new(vocabulary),
        config, cancellation).map_err(host_error)?;
    check_completed(&expected, result.result())?;
    deliver_encoded(&session, &facts, &result, encoding::report(&planner)?.as_ref(),
        command.max_map_result_bytes + 4096, output)
}
fn deliver<T: Serialize>(session: &Session, facts: &ArtifactIdentity, result: &T,
    cap: usize, output: &mut impl Write) -> Result<(), CandidateError> {
    deliver_encoded(session, facts, result, None, cap, output)
}
fn deliver_encoded<T: Serialize>(session: &Session, facts: &ArtifactIdentity, result: &T,
    input_encoding: Option<&encoding::EncodingReport>, cap: usize, output: &mut impl Write)
    -> Result<(), CandidateError> {
    session.remaining()?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: result };
    // Both host output ownership and the CLI's aggregate staging reservation
    // remain live through complete serialization, external write and flush.
    let response = encoding::EncodedResponse { response, input_encoding };
    publish_checked(&response, cap, output, &|| session.remaining().map(|_| ()))
}

fn check_completed(expected: &Preflight, result: &Int8SourceMapRun) -> Result<(), CandidateError> {
    let root = result.mapped.root(); let span = root.source_span();
    if result.planned_model_work != expected.work || result.reserved_mask_node_visits != expected.masks
        || root.value().len() != expected.chunks || root.chunk_range() != (0..expected.chunks)
        || span.byte_start != 0 || span.byte_end != expected.source_bytes
        || span.scalar_start != 0 || span.scalar_end != expected.source_scalars {
        return Err(CandidateError::Execution);
    }
    Ok(())
}

#[cfg(test)] mod tests;
