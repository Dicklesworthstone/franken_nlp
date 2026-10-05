//! Opt-in cross-document execution; preserve the existing ordered wire frames.
//! Bounded read-ahead, all-row admission and physical completion precede the
//! first result of a cohort. No row is retried after a native/delivery failure.
use super::*;
use crate::{hosted::ChatCohortLimits, native_engine::portable_int8::batch::MAX_BATCH_ROWS};

pub(super) fn run(session: &Session, command: &TextBatchCommand, limits: Limits,
    input: &mut impl BufRead, output: &mut impl Write, width: usize) -> Result<(), CandidateError> {
    if width == 0 || width > MAX_BATCH_ROWS { return Err(CandidateError::Arguments); }
    command.refill_strategy(width)?;
    let prefill = command.common.policy.prefill()?;
    let args = &command.common;
    let mut input_bytes = 0;
    let mut transport = Output::new(output, command.max_total_output_bytes);
    let mut ledger = Ledger::default();
    // Empty/malformed first input still cannot open model metadata or weights.
    let first = read_record(input, command, &mut input_bytes)?;
    session.remaining()?;
    let Some(first) = first else { return complete(&ledger, input_bytes, &mut transport); };
    let facts = session.facts(args)?;
    let planner = Planner::new(args, limits, &facts)?;
    let cancellation = CancellationToken::default();
    let mut model = None;
    let mut pending = Some(first);
    let frame_cap = limits.result_bytes.checked_add(8192).ok_or(CandidateError::Output)?;
    while let Some(first) = pending.take() {
        let batch = collect(first, input, &planner, command, &mut ledger, &mut input_bytes,
            width, &mut || session.remaining().map(|_| ()))?;
        let count = batch.ids.len();
        let refill = command.refill_strategy(count)?;
        let active_count = refill.as_ref().map_or(count, |(active, _)| *active);
        // Admit the WHOLE queued window, never just the smaller live-slot set.
        // The footer allowance is retained exactly once across all results.
        transport.admit_frame(delivery_bytes(count, frame_cap)?)?;
        let max_sampler_bytes = SAMPLER_BYTES.checked_mul(active_count as u64).ok_or(CandidateError::Memory)?;
        let max_result_bytes = (limits.result_bytes as u64).checked_mul(count as u64)
            .and_then(|bytes| bytes.checked_add(4096)).ok_or(CandidateError::Memory)?;
        if model.is_none() { model = Some(session.load(args, limits, &facts, cancellation.clone())?); }
        let resident = model.as_ref().ok_or(CandidateError::Model)?;
        let cohort_limits = ChatCohortLimits { native: session.native(args)?, max_sampler_bytes,
            preparation_reserve_bytes: limits.preparation_bytes, max_result_bytes };
        let result = match refill {
            Some((active, prefill)) => session.engine.execute_int8_chat_refilling(resident, batch.prepared,
                batch.first_request_seq, cohort_limits, active, prefill, cancellation.clone()),
            None => match prefill {
                Some(prefill) => session.engine.execute_int8_chat_cohort_packed(resident, batch.prepared,
                    batch.first_request_seq, cohort_limits, prefill, cancellation.clone()),
                None => session.engine.execute_int8_chat_cohort(resident, batch.prepared,
                    batch.first_request_seq, cohort_limits, cancellation.clone()),
            },
        }.map_err(|_| CandidateError::Execution)?;
        session.remaining()?;
        let completed = &result.result().results;
        if completed.len() != count || completed.iter().enumerate()
            .any(|(index, row)| row.result.request_seq != batch.first_request_seq + index as u64) {
            return Err(CandidateError::Execution);
        }
        for (index, (id, row)) in batch.ids.iter().zip(completed).enumerate() {
            session.remaining()?;
            let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
                evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
                source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
                quant_recipe: &facts.recipe_id, output: row };
            let frame = ResultFrame { protocol: PROTOCOL, schema_version: 1, event: "result",
                id, request_seq: batch.first_request_seq + index as u64, response };
            // Keep the one guard covering ALL completed rows through each
            // canonical staging, write and flush. Do not clone out bare rows.
            publish(&frame, frame_cap, &mut transport)?;
        }
        drop(result);
        session.remaining()?;
        if batch.eof { break; }
        pending = read_record(input, command, &mut input_bytes)?;
    }
    session.remaining()?;
    complete(&ledger, input_bytes, &mut transport)
}

struct PreparedCohort {
    prepared: Vec<PreparedInt8Chat>, ids: Vec<String>, first_request_seq: u64, eof: bool,
}
/// Real pinned planning seam: production supplies the invocation deadline.
/// Returning a cohort requires every ID/work claim to enter the SAME corpus
/// ledger. A rejected later record cannot trigger native execution of siblings.
#[allow(clippy::too_many_arguments)]
fn collect(first: Record, input: &mut impl BufRead, planner: &Planner, command: &TextBatchCommand,
    ledger: &mut Ledger, input_bytes: &mut u64, width: usize,
    checkpoint: &mut impl FnMut() -> Result<(), CandidateError>) -> Result<PreparedCohort, CandidateError> {
    if width == 0 || width > MAX_BATCH_ROWS { return Err(CandidateError::Arguments); }
    let first_request_seq = ledger.records.checked_add(1).ok_or(CandidateError::Batch)?;
    let mut prepared = Vec::new(); let mut ids = Vec::new();
    prepared.try_reserve_exact(width).map_err(|_| CandidateError::Memory)?;
    ids.try_reserve_exact(width).map_err(|_| CandidateError::Memory)?;
    let mut next = Some(first);
    let mut eof = false;
    while let Some(record) = next.take() {
        checkpoint()?;
        let (id, plan) = planner.prepare(record)?;
        checkpoint()?;
        let work = plan.planned_work();
        if work.forward_positions > command.common.context_tokens as u64 { return Err(CandidateError::Planning); }
        ledger.admit(&id, ReservedWork { forward_positions: work.forward_positions,
            projected_logits: work.projected_logits }, command)?;
        ids.push(id); prepared.push(plan);
        // Do not consume a record beyond a full group or the record ceiling.
        // After publishing this admitted group, any excess record will fail
        // the unchanged corpus ledger rather than being silently dropped.
        if prepared.len() == width || ledger.records == command.max_records { break; }
        checkpoint()?;
        next = read_record(input, command, input_bytes)?;
        if next.is_none() { eof = true; }
    }
    checkpoint()?;
    Ok(PreparedCohort { prepared, ids, first_request_seq, eof })
}
fn read_record(input: &mut impl BufRead, command: &TextBatchCommand, input_bytes: &mut u64)
    -> Result<Option<Record>, CandidateError> {
    wire::read_line(input, command.common.max_input_bytes, input_bytes, command.max_total_input_bytes)?
        .map(|line| wire::parse_record(command.task, &line, command.common.max_input_bytes)).transpose()
}
fn delivery_bytes(count: usize, frame_cap: usize) -> Result<usize, CandidateError> {
    if count == 0 || count > MAX_BATCH_ROWS { return Err(CandidateError::Arguments); }
    count.checked_mul(frame_cap).ok_or(CandidateError::Output)
}
#[cfg(test)] mod tests;
#[cfg(test)] mod refill_tests;
