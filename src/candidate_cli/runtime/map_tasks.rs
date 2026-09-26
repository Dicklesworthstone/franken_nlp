//! Long-document candidate inference on the existing charged map/merge host.
use super::*;
use std::sync::Arc;
use super::source_tasks::{Session, planner, source_identity};
use crate::{candidate_cli::map::MapCommand,
    hosted::SourceMapConfig,
    native_engine::{constrained_int8, decode::DecodeStepControl, strict_int8::Int8Work},
    tasks::{ir::PlanContext, mapreduce::{ChunkPlan, MapReduceError},
        source_planning::{SourceTaskPlanner, quantized::{capacity::Int8SourceMapCapacity,
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
    let chunks = ChunkPlan::build_with_checkpoints(source, mapping.chunks, |text| {
        planner.source_encoder().encode(text, mapping.chunks.max_chunk_bytes, mapping.chunks.max_chunk_bytes)
            .map(|s| s.total_token_count()).map_err(|_| MapReduceError::Tokenizer)
    }, || {
        if control.prefill_checkpoint(0).is_some() { Err(MapReduceError::Cancelled) } else { Ok(()) }
    }).map_err(|_| CandidateError::Planning)?;
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
    if control.prefill_checkpoint(0).is_some() { Err(CandidateError::Planning) } else { Ok(()) }
}

pub(in crate::candidate_cli) fn execute(command: MapCommand, args: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let session = Session::new(&args, limits)?;
    let text = session.read(&command.input, input, args.max_input_bytes)?;
    let option_text = command.options.as_ref()
        .map(|path| session.read(path, input, crate::candidate_cli::source::OPTIONS_BYTES)).transpose()?;
    let task = command.map_task(option_text.as_deref())?;
    if text.is_empty() { return Err(CandidateError::Input); }
    session.remaining()?;
    let facts = session.facts(&args)?;
    let (planner, vocabulary) = planner()?;
    let budget = command.host.task_budget(limits);
    let identity = source_identity(&facts, &planner, command.kind()?)?;
    let context = PlanContext::new(&identity, budget).map_err(|_| CandidateError::Planning)?;
    let capacity = planner.int8_map_capacity_with_control(&task, budget, &context,
        command.host.planning(), &mut session.control()).map_err(|_| CandidateError::Planning)?;
    let mapping = command.mapping(capacity)?;
    let expected = preflight(&text, &planner, &command, capacity, mapping, budget, &mut session.control())?;
    session.remaining()?;
    // No weights until the complete partition and all five work axes fit.
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let config = SourceMapConfig { identity, task, budget, planning: command.host.planning(), mapping,
        native: session.native(&args)?, preparation_reserve_bytes: limits.preparation_bytes,
        reduction_reserve_bytes: command.reduction_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)? };
    let result = session.engine.map_int8_source(&model, text, Arc::new(planner), Arc::new(vocabulary),
        config, cancellation).map_err(|_| CandidateError::Execution)?;
    session.remaining()?;
    check_completed(&expected, result.result())?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: &result };
    // Both host output ownership and the CLI's aggregate staging reservation
    // remain live through complete serialization, external write and flush.
    publish(&response, command.max_map_result_bytes + 4096, output)
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
