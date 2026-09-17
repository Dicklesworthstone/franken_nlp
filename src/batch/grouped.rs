//! Bounded, ordered NDJSON cohorts for an actual multi-row processor.
//!
//! Unlike `run_ndjson`, a window executes through ONE `execute_group` call.
//! Transport never loops over scalar `execute` calls and calls that batching.
//! The existing serial entrypoint/protocol schema and robot CLI are unchanged.
//! No threads, runtime, timeout polling, admission broker or retries are added.
//!
//! A window contains at most `max_records` nonempty documents/errors. The host
//! must reserve that many bounded prepared items, one framing/planning scratch
//! item, result storage and canonical serialization overhead. This row bound is
//! NOT a process-memory certificate. Processors must bound each prepared item.
//! Input can wait for a full window: an interactive producer must send a flush
//! command (or select width one); this blocking API has no hidden latency timer.

use super::*;

pub const GROUPED_EXECUTION: &str = "ordered-cohort-no-retry-v1";
pub const MAX_GROUP_RECORDS: usize = 128;

#[derive(Clone, Copy, Debug)]
pub struct GroupLimits {
    /// Includes malformed/rejected documents, not only model-ready rows.
    pub max_records: usize,
}
impl Default for GroupLimits {
    fn default() -> Self { Self { max_records: 1 } }
}

/// Plans and delivery coordinates are owned until the processor has quiesced.
/// Coordinates never grant resource authority or become sampling addresses.
pub struct GroupedRequest<P> { pub prepared: P, pub context: BatchRequestContext }
pub struct GroupedRow<T> {
    pub request_seq: u64,
    pub result: Result<T, BatchItemFailure>,
}

/// A single host-owned cohort reservation survives ALL result delivery.
/// Do not distribute an unshared guard to only the first result. Fields drop
/// in declaration order: all retained result storage is freed before the guard.
/// The guard needs neither Clone, Debug nor Serialize and never enters JSON.
pub struct GroupedOutput<T, G> { rows: Vec<GroupedRow<T>>, _guard: G }
impl<T, G> GroupedOutput<T, G> {
    pub fn new(rows: Vec<GroupedRow<T>>, guard: G) -> Self { Self { rows, _guard: guard } }
    pub fn rows(&self) -> &[GroupedRow<T>] { &self.rows }
}

/// Trusted embedding boundary, not a task recipe or executable input option.
/// Preparation must be model-free and bounded per item. Planned work is a sound
/// ceiling enforced by execution. A group shares one compatible loaded model;
/// per-row attention/sampling identities and task finalizers stay independent.
/// A returned error means fatal cohort failure, never a successful partial set.
/// Per-item failures may continue only after native cleanup has completed.
pub trait GroupedBatchProcessor {
    type Args: DeserializeOwned;
    type Prepared;
    type Output: Serialize;
    type Guard;
    fn max_group_size(&self) -> usize;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure>;
    fn planned_work(&self, prepared: &Self::Prepared) -> Result<BatchWork, BatchItemFailure>;
    fn execute_group<C: DecodeStepControl>(&mut self, requests: Vec<GroupedRequest<Self::Prepared>>, control: &mut C)
        -> Result<GroupedOutput<Self::Output, Self::Guard>, BatchFault>;
}

/// Canonical ordered output over finite cohorts. Every explicit flush and EOF
/// first drains the preceding cohort. Duplicate-ID epochs reset only after the
/// flush event itself has been written and flushed. No read crosses that barrier.
/// Group width one is available; the older serial runner remains the default.
///
/// Work is charged before enqueue and never refunded, even if subsequent input
/// or cancellation aborts a not-yet-executed cohort. Fatal read/prepare/control
/// failures discard pending work and emit run_error, not fake document results.
/// Output failure poisons the stream: no further input, inference or writes.
/// Up to one bounded window may already have been read when delivery fails.
/// Panics unwind owned plans/results/guards; they are not caught as doc errors.
pub fn run_ndjson<R: BufRead, W: Write, P: GroupedBatchProcessor, C: DecodeStepControl>(
    reader: &mut R, writer: &mut W, processor: &mut P, limits: BatchLimits,
    groups: GroupLimits, control: &mut C,
) -> Result<BatchSummary, BatchRunError> {
    let mut summary = BatchSummary::default();
    limits.validate().map_err(|fault| BatchRunError { fault, summary })?;
    if groups.max_records == 0 || groups.max_records > MAX_GROUP_RECORDS
        || groups.max_records > processor.max_group_size() {
        return Err(BatchRunError { fault: BatchCode::InvalidLimits.into(), summary });
    }
    let mut sink = Sink::new(writer, limits);
    let mut epoch = 1;
    let result = run(reader, &mut sink, processor, limits, groups, control, &mut summary, &mut epoch);
    match result {
        Ok(()) => {
            let mut event = event::<()>("run_complete", epoch, summary.requests);
            event.summary = Some(&summary);
            sink.emit(&event, true).map_err(|fault| BatchRunError { fault, summary })?;
            Ok(summary)
        }
        Err(fault) => {
            if !sink.poisoned() {
                let mut event = event::<()>("run_error", epoch, summary.requests);
                event.error = Some(fault); event.summary = Some(&summary);
                sink.emit(&event, true).map_err(|fault| BatchRunError { fault, summary })?;
            }
            Err(BatchRunError { fault, summary })
        }
    }
}

struct Pending<P> {
    context: BatchRequestContext,
    id: Option<String>,
    prepared: Option<P>,
    fault: Option<BatchFault>,
    charge: Option<BatchWork>,
}
impl<P> Pending<P> {
    fn rejected(context: BatchRequestContext, id: Option<String>, fault: BatchFault) -> Self {
        Self { context, id, prepared: None, fault: Some(fault), charge: None }
    }
}

fn run<R: BufRead, W: Write, P: GroupedBatchProcessor, C: DecodeStepControl>(reader: &mut R,
    sink: &mut Sink<'_, W>, processor: &mut P, limits: BatchLimits, groups: GroupLimits,
    control: &mut C, summary: &mut BatchSummary, epoch: &mut u64) -> Result<(), BatchFault> {
    checkpoint(control)?;
    sink.emit(&event::<()>("run_start", *epoch, 0), false)?;
    let mut pending = reserve(groups.max_records)?;
    let mut seen = BTreeSet::new(); let mut id_bytes = 0;
    loop {
        let Some(frame) = read_frame(reader, limits, summary, control)? else {
            drain(&mut pending, processor, sink, summary, control)?;
            barrier(sink, summary, epoch, true)?;
            return Ok(());
        };
        if frame.bytes.is_empty() && !frame.oversized { continue; }
        let document = match parse::<P::Args>(&frame, limits) {
            Ok(Command::Flush) => {
                drain(&mut pending, processor, sink, summary, control)?;
                barrier(sink, summary, epoch, false)?;
                seen.clear(); id_bytes = 0; continue;
            }
            Ok(Command::Document(document)) => Ok(document),
            Err(fault) => Err(fault),
        };
        let context = BatchRequestContext { request_seq: summary.requests, epoch: *epoch,
            input_line: frame.line, byte_offset: frame.offset };
        pending.push(prepare(document, context, processor, limits, control, summary, &mut seen, &mut id_bytes)?);
        if pending.len() == groups.max_records { drain(&mut pending, processor, sink, summary, control)?; }
    }
}

fn prepare<P: GroupedBatchProcessor, C: DecodeStepControl>(document: Result<BatchDocument<P::Args>, BatchFault>,
    context: BatchRequestContext, processor: &mut P, limits: BatchLimits, control: &mut C,
    summary: &mut BatchSummary, seen: &mut BTreeSet<String>, id_bytes: &mut usize)
    -> Result<Pending<P::Prepared>, BatchFault> {
    let document = match document {
        Ok(document) => document,
        Err(fault) => return Ok(Pending::rejected(context, None, fault)),
    };
    if document.id.is_empty() || document.id.len() > limits.max_id_bytes || document.id.chars().any(char::is_control) {
        return Ok(Pending::rejected(context, None, BatchCode::InvalidId.into()));
    }
    if seen.contains(&document.id) {
        return Ok(Pending::rejected(context, Some(document.id), BatchCode::DuplicateId.into()));
    }
    let next_bytes = id_bytes.checked_add(document.id.len()).ok_or(BatchCode::EpochIdLimit)?;
    if seen.len() >= limits.max_epoch_ids || next_bytes > limits.max_epoch_id_bytes {
        return Ok(Pending::rejected(context, Some(document.id), BatchCode::EpochIdLimit.into()));
    }
    let id = document.id.clone(); seen.insert(id.clone()); *id_bytes = next_bytes;
    if document.text.len() > limits.max_document_bytes {
        return Ok(Pending::rejected(context, Some(id), BatchCode::DocumentLimit.into()));
    }
    checkpoint(control)?;
    let prepared = match processor.prepare(document) {
        Ok(prepared) => prepared,
        Err(failure) if fatal(failure) => return Err(failure.fault),
        Err(failure) => return Ok(Pending::rejected(context, Some(id), failure.fault)),
    };
    let charge = match processor.planned_work(&prepared) {
        Ok(charge) => charge,
        Err(failure) if fatal(failure) => return Err(failure.fault),
        Err(failure) => return Ok(Pending::rejected(context, Some(id), failure.fault)),
    };
    let total = match summary.reserved_work.add(charge) {
        Ok(total) if total.fits(limits.max_work) => total,
        _ => return Ok(Pending::rejected(context, Some(id), BatchCode::WorkLimit.into())),
    };
    checkpoint(control)?;
    summary.reserved_work = total;
    Ok(Pending { context, id: Some(id), prepared: Some(prepared), fault: None, charge: Some(charge) })
}

fn drain<W: Write, P: GroupedBatchProcessor, C: DecodeStepControl>(pending: &mut Vec<Pending<P::Prepared>>,
    processor: &mut P, sink: &mut Sink<'_, W>, summary: &mut BatchSummary, control: &mut C) -> Result<(), BatchFault> {
    if pending.is_empty() { return Ok(()); }
    checkpoint(control)?;
    let mut requests = reserve(pending.len())?;
    for row in pending.iter_mut() {
        if let Some(prepared) = row.prepared.take() { requests.push(GroupedRequest { prepared, context: row.context }); }
    }
    let output = if requests.is_empty() { None } else { Some(processor.execute_group(requests, control)?) };
    // Refuse missing, duplicated, reordered or foreign rows before publishing
    // ANY result. Errors in the input window stay in their original positions.
    if let Some(output) = &output {
        let ready = pending.iter().filter(|row| row.fault.is_none());
        if output.rows.len() != ready.clone().count()
            || output.rows.iter().zip(ready).any(|(result, request)| result.request_seq != request.context.request_seq) {
            return Err(BatchCode::InvalidExecution.into());
        }
    }
    let mut index = 0;
    for row in pending.iter() {
        checkpoint(control)?;
        if let Some(fault) = row.fault {
            emit_error(sink, summary, row, fault)?;
            continue;
        }
        let result = output.as_ref().and_then(|output| output.rows.get(index)).ok_or(BatchCode::InvalidExecution)?;
        index += 1;
        match &result.result {
            Ok(value) => {
                let mut event = event("doc", row.context.epoch, row.context.request_seq);
                event.caller_id = row.id.as_deref(); event.input_line = Some(row.context.input_line);
                event.byte_offset = Some(row.context.byte_offset); event.reserved_work = row.charge; event.result = Some(value);
                match sink.emit(&event, false) {
                    Ok(()) => summary.succeeded += 1,
                    Err(fault) if matches!(fault.code, BatchCode::OutputLineLimit | BatchCode::Serialization) => {
                        emit_error(sink, summary, row, fault)?;
                    }
                    Err(fault) => return Err(fault),
                }
            }
            Err(failure) => {
                emit_error(sink, summary, row, failure.fault)?;
                if fatal(*failure) { return Err(failure.fault); }
            }
        }
    }
    // `output` is deliberately still alive here, including on every early exit
    // above. Its guard is not released between first and last result flush.
    pending.clear();
    Ok(())
}
fn fatal(failure: BatchItemFailure) -> bool {
    failure.stop || failure.fault.cancellation.is_some()
        || matches!(failure.fault.code, BatchCode::Cancelled | BatchCode::InputIo | BatchCode::OutputIo | BatchCode::InvalidExecution)
}
fn emit_error<W: Write, P>(sink: &mut Sink<'_, W>, summary: &mut BatchSummary,
    row: &Pending<P>, fault: BatchFault) -> Result<(), BatchFault> {
    let mut event = event::<()>("doc_error", row.context.epoch, row.context.request_seq);
    event.caller_id = row.id.as_deref(); event.input_line = Some(row.context.input_line);
    event.byte_offset = Some(row.context.byte_offset); event.reserved_work = row.charge; event.error = Some(fault);
    sink.emit(&event, false)?; summary.failed += 1; Ok(())
}
fn barrier<W: Write>(sink: &mut Sink<'_, W>, summary: &mut BatchSummary, epoch: &mut u64, eof: bool) -> Result<(), BatchFault> {
    let next = epoch.checked_add(1).ok_or(BatchCode::SequenceOverflow)?;
    let mut event = event::<()>("flush", *epoch, summary.requests); event.eof = Some(eof);
    sink.emit(&event, false)?;
    summary.flushes = summary.flushes.checked_add(1).ok_or(BatchCode::SequenceOverflow)?;
    if !eof { *epoch = next; }
    Ok(())
}
fn event<'a, T: Serialize>(kind: &'static str, epoch: u64, sequence: u64) -> Event<'a, T> {
    Event::new(kind, epoch, sequence).with_execution(GROUPED_EXECUTION)
}
fn reserve<T>(count: usize) -> Result<Vec<T>, BatchFault> {
    let mut values = Vec::new(); values.try_reserve_exact(count).map_err(|_| BatchCode::Allocation)?; Ok(values)
}

#[cfg(test)] mod tests;
