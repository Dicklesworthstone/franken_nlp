//! Bounded, blocking, ordered item-local NDJSON execution.
//!
//! This is a serial baseline, NOT continuous/GEMM batching or a durable job.
//! One task and one response are live; delivery/flush finishes before the next
//! record is read. The caller owns model/process admission and I/O lifetimes.
//! No worker, runtime, model loader, journal, signal handler or retry is created.

use std::{collections::BTreeSet, error::Error, fmt, io::{BufRead, Write}};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use crate::{canonjson, native_engine::decode::{DecodeCancellationKind, DecodeStepControl}};

pub mod classify;
pub mod extract;
pub mod generation;
pub mod grouped;
pub mod judge;
pub mod source;
mod framing;
mod output;
use framing::{Frame, read_frame};
use output::{Event, Sink};

/// Separate, explicitly named library protocol; not an unreviewed extension
/// of the frozen `robot schema` envelope. CLI activation remains caller-owned.
pub const BATCH_PROTOCOL: &str = "fnlp-item-local-batch-v1";
pub const BATCH_EXECUTION: &str = "ordered-serial-no-retry-v1";

/// Original UTF-8 text, not normalized, trimmed or retokenized by the runner.
/// Intentionally no Debug: text/task arguments are private request content.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchDocument<A> {
    pub id: String,
    pub text: String,
    pub task_args: Option<A>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BatchWork {
    pub forward_positions: u64,
    pub projected_logits: u64,
}
impl BatchWork {
    fn add(self, other: Self) -> Result<Self, BatchFault> {
        Ok(Self {
            forward_positions: self.forward_positions.checked_add(other.forward_positions).ok_or(BatchCode::WorkLimit)?,
            projected_logits: self.projected_logits.checked_add(other.projected_logits).ok_or(BatchCode::WorkLimit)?,
        })
    }
    fn fits(self, ceiling: Self) -> bool {
        self.forward_positions <= ceiling.forward_positions && self.projected_logits <= ceiling.projected_logits
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BatchLimits {
    /// Bytes before LF, including a CR in CRLF. Oversized records are drained.
    pub max_line_bytes: usize,
    pub max_document_bytes: usize,
    pub max_id_bytes: usize,
    pub max_epoch_ids: usize,
    pub max_epoch_id_bytes: usize,
    pub max_json_depth: usize,
    /// Includes blank lines and delimiters, so endless oversized input stops.
    pub max_input_bytes: u64,
    /// Nonempty records, including explicit flush commands and malformed data.
    pub max_requests: u64,
    /// Includes the complete envelope and its final newline.
    pub max_output_line_bytes: usize,
    pub max_output_bytes: u64,
    /// Reservation is charged before execution and NEVER refunded on failure.
    pub max_work: BatchWork,
}
impl Default for BatchLimits {
    fn default() -> Self {
        Self { max_line_bytes: 1024 * 1024, max_document_bytes: 512 * 1024,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024,
            max_json_depth: 64, max_input_bytes: 1024 * 1024 * 1024, max_requests: 100_000,
            max_output_line_bytes: 4 * 1024 * 1024, max_output_bytes: 1024 * 1024 * 1024,
            max_work: BatchWork { forward_positions: 1_000_000, projected_logits: 100_000_000 } }
    }
}
impl BatchLimits {
    pub fn validate(self) -> Result<(), BatchFault> {
        if self.max_line_bytes == 0 || self.max_document_bytes > self.max_line_bytes
            || self.max_id_bytes == 0 || self.max_id_bytes > 256 || self.max_epoch_ids == 0
            || self.max_epoch_id_bytes < self.max_id_bytes || self.max_json_depth == 0 || self.max_json_depth > 64
            || self.max_input_bytes == 0 || self.max_requests == 0
            || self.max_output_line_bytes < output::TERMINAL_RESERVE
            || self.max_output_bytes < (2 * output::TERMINAL_RESERVE) as u64 {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}

/// Fixed categories only: never serialize parser paths, provider errors,
/// source snippets, raw JSON, model identities or content-derived digests.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchCode {
    InvalidLimits, InvalidUtf8, InvalidJson, InvalidEnvelope, LineLimit,
    DocumentLimit, InvalidId, DuplicateId, EpochIdLimit, InputLimit, RequestLimit,
    SequenceOverflow, OutputLineLimit, OutputLimit, WorkLimit, Planning,
    Admission, Execution, InvalidExecution, Allocation, Serialization,
    InputIo, OutputIo, Cancelled,
}
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BatchFault {
    pub code: BatchCode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancellation: Option<DecodeCancellationKind>,
}
impl From<BatchCode> for BatchFault {
    fn from(code: BatchCode) -> Self { Self { code, cancellation: None } }
}
impl BatchFault {
    pub fn cancelled(cause: DecodeCancellationKind) -> Self {
        Self { code: BatchCode::Cancelled, cancellation: Some(cause) }
    }
}
impl fmt::Display for BatchFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "batch {:?}", self.code)?;
        if let Some(cause) = self.cancellation { write!(f, " ({cause:?})")?; }
        Ok(())
    }
}
impl Error for BatchFault {}

/// A task may reject a document and continue only after its native resources
/// are quiescent. Cancellation, admission-host failure and broken engine state
/// must stop admission; they must not become a stream of document failures.
#[derive(Clone, Copy, Debug)]
pub struct BatchItemFailure { pub fault: BatchFault, pub stop: bool }
impl BatchItemFailure {
    pub fn reject(code: BatchCode) -> Self { Self { fault: code.into(), stop: false } }
    pub fn fatal(fault: impl Into<BatchFault>) -> Self { Self { fault: fault.into(), stop: true } }
}

/// Engine-assigned delivery coordinates. Not deserializable from task_args,
/// and NEVER a semantic sampling address, cache key, or admission certificate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchRequestContext {
    pub request_seq: u64,
    pub epoch: u64,
    pub input_line: u64,
    pub byte_offset: u64,
}

/// Trusted embedding boundary, not an executable recipe deserialized from input.
/// Implementations must be ITEM-LOCAL, prepare without model work, return a
/// sound work ceiling, enforce it on execution, and bound their result storage.
/// `execute` returns only after native cleanup/completion. Panics are NOT caught
/// and mislabeled as per-document failures. The embedding host supervises them.
pub trait BatchProcessor {
    type Args: DeserializeOwned;
    type Prepared;
    type Output: Serialize;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure>;
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork;
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure>;
    /// The runner uses this entrypoint after reserving aggregate work. Existing
    /// processors retain their behavior; generation can echo the real sequence
    /// without deriving it from caller IDs or adding it to the sampling key.
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        _context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.execute(prepared, control)
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BatchSummary {
    pub input_bytes: u64,
    pub input_lines: u64,
    pub requests: u64,
    pub succeeded: u64,
    pub failed: u64,
    pub flushes: u64,
    /// Conservative charged ceilings, including failed attempts. Not measured
    /// work: a native task may expose its observed work in its own result.
    pub reserved_work: BatchWork,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchRunError { pub fault: BatchFault, pub summary: BatchSummary }
impl fmt::Display for BatchRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.fault) }
}
impl Error for BatchRunError {}

/// Consume requests `{id,text,task_args?}` and exact `{"flush":true}` controls.
/// Output is canonical and ordered, with a per-line sequence assigned BEFORE
/// JSON/UTF-8 parsing. Only empty LF/CRLF records are ignored; whitespace-only
/// input is malformed. The last record need not end in LF. Duplicate IDs are
/// retained for the numbered epoch even if planning/execution rejects an item.
///
/// A successful return means EOF was drained and run_complete was flushed, NOT
/// that every item succeeded: inspect `failed`. A failed/partial output write
/// poisons the stream; no further input or output is attempted. A normal I/O
/// flush acknowledges the local writer, not remote processing or durability.
/// Blocking I/O itself cannot be preempted by these cooperative checkpoints.
pub fn run_ndjson<R: BufRead, W: Write, P: BatchProcessor, C: DecodeStepControl>(
    reader: &mut R, writer: &mut W, processor: &mut P, limits: BatchLimits, control: &mut C,
) -> Result<BatchSummary, BatchRunError> {
    let mut summary = BatchSummary::default();
    limits.validate().map_err(|fault| BatchRunError { fault, summary })?;
    let mut sink = Sink::new(writer, limits);
    let mut epoch = 1_u64;
    let result = run(reader, &mut sink, processor, limits, control, &mut summary, &mut epoch);
    match result {
        Ok(()) => {
            let mut event = Event::<()>::new("run_complete", epoch, summary.requests);
            event.summary = Some(&summary);
            sink.emit(&event, true).map_err(|fault| BatchRunError { fault, summary })?;
            Ok(summary)
        }
        Err(fault) => {
            // Never append an error after a partial write or failed flush.
            if !sink.poisoned() {
                let mut event = Event::<()>::new("run_error", epoch, summary.requests);
                event.error = Some(fault); event.summary = Some(&summary);
                sink.emit(&event, true).map_err(|fault| BatchRunError { fault, summary })?;
            }
            Err(BatchRunError { fault, summary })
        }
    }
}
fn run<R: BufRead, W: Write, P: BatchProcessor, C: DecodeStepControl>(reader: &mut R,
    sink: &mut Sink<'_, W>, processor: &mut P, limits: BatchLimits, control: &mut C,
    summary: &mut BatchSummary, epoch: &mut u64) -> Result<(), BatchFault> {
    checkpoint(control)?;
    sink.emit(&Event::<()>::new("run_start", *epoch, 0), false)?;
    let mut seen = BTreeSet::new(); let mut id_bytes = 0_usize;
    loop {
        let frame = read_frame(reader, limits, summary, control)?;
        let Some(frame) = frame else {
            flush(sink, summary, epoch, true)?;
            return Ok(());
        };
        if frame.bytes.is_empty() && !frame.oversized { continue; }
        // Framing assigns the sequence as soon as nonempty payload is known,
        // so even a later read/budget error cannot masquerade as the prior item.
        let document = match parse::<P::Args>(&frame, limits) {
            Ok(Command::Flush) => {
                flush(sink, summary, epoch, false)?;
                // Emission/flush is the drain acknowledgement. Never reset
                // the duplicate set merely upon enqueue or failed delivery.
                seen.clear(); id_bytes = 0; continue;
            }
            Ok(Command::Document(document)) => document,
            Err(fault) => { reject(sink, summary, *epoch, &frame, None, fault)?; continue; }
        };
        let id = &document.id;
        if id.is_empty() || id.len() > limits.max_id_bytes || id.chars().any(char::is_control) {
            reject(sink, summary, *epoch, &frame, None, BatchCode::InvalidId.into())?; continue;
        }
        if seen.contains(id) {
            reject(sink, summary, *epoch, &frame, Some(id), BatchCode::DuplicateId.into())?; continue;
        }
        let next_bytes = id_bytes.checked_add(id.len()).ok_or(BatchCode::EpochIdLimit)?;
        if seen.len() >= limits.max_epoch_ids || next_bytes > limits.max_epoch_id_bytes {
            reject(sink, summary, *epoch, &frame, Some(id), BatchCode::EpochIdLimit.into())?; continue;
        }
        let id = document.id.clone(); seen.insert(id.clone()); id_bytes = next_bytes;
        if document.text.len() > limits.max_document_bytes {
            reject(sink, summary, *epoch, &frame, Some(&id), BatchCode::DocumentLimit.into())?; continue;
        }
        checkpoint(control)?;
        let prepared = match processor.prepare(document) {
            Ok(plan) => plan,
            Err(error) => {
                reject(sink, summary, *epoch, &frame, Some(&id), error.fault)?;
                if error.stop { return Err(error.fault); } else { continue; }
            }
        };
        let charge = processor.planned_work(&prepared);
        let total = summary.reserved_work.add(charge);
        let total = match total {
            Ok(total) if total.fits(limits.max_work) => total,
            _ => { reject(sink, summary, *epoch, &frame, Some(&id), BatchCode::WorkLimit.into())?; continue; }
        };
        checkpoint(control)?;
        summary.reserved_work = total; // No refunds, including failed attempts.
        let context = BatchRequestContext { request_seq: summary.requests, epoch: *epoch,
            input_line: frame.line, byte_offset: frame.offset };
        match processor.execute_with_context(prepared, context, control) {
            Ok(result) => {
                let mut event = Event::new("doc", *epoch, summary.requests);
                event.caller_id = Some(&id); event.input_line = Some(frame.line);
                event.byte_offset = Some(frame.offset); event.result = Some(&result); event.reserved_work = Some(charge);
                match sink.emit(&event, false) {
                    Ok(()) => summary.succeeded += 1,
                    Err(fault) if matches!(fault.code, BatchCode::OutputLineLimit | BatchCode::Serialization) => {
                        reject(sink, summary, *epoch, &frame, Some(&id), fault)?;
                    }
                    Err(fault) => return Err(fault),
                }
            }
            Err(error) => {
                reject(sink, summary, *epoch, &frame, Some(&id), error.fault)?;
                if error.stop { return Err(error.fault); }
            }
        }
    }
}
fn reject<W: Write>(sink: &mut Sink<'_, W>, summary: &mut BatchSummary, epoch: u64,
    frame: &Frame, id: Option<&str>, fault: BatchFault) -> Result<(), BatchFault> {
    let mut event = Event::<()>::new("doc_error", epoch, summary.requests);
    event.caller_id = id; event.input_line = Some(frame.line); event.byte_offset = Some(frame.offset); event.error = Some(fault);
    sink.emit(&event, false)?;
    summary.failed += 1; Ok(())
}
fn flush<W: Write>(sink: &mut Sink<'_, W>, summary: &mut BatchSummary, epoch: &mut u64, eof: bool) -> Result<(), BatchFault> {
    let next = epoch.checked_add(1).ok_or(BatchCode::SequenceOverflow)?;
    let mut event = Event::<()>::new("flush", *epoch, summary.requests);
    event.eof = Some(eof); sink.emit(&event, false)?;
    summary.flushes = summary.flushes.checked_add(1).ok_or(BatchCode::SequenceOverflow)?;
    if !eof { *epoch = next; }
    Ok(())
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), BatchFault> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(BatchFault::cancelled(cause)), None => Ok(()) }
}
enum Command<A> { Document(BatchDocument<A>), Flush }
fn parse<A: DeserializeOwned>(frame: &Frame, limits: BatchLimits) -> Result<Command<A>, BatchFault> {
    if frame.oversized { return Err(BatchCode::LineLimit.into()); }
    let text = std::str::from_utf8(&frame.bytes).map_err(|_| BatchCode::InvalidUtf8)?;
    let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits {
        max_depth: limits.max_json_depth, max_string_bytes: limits.max_line_bytes,
    }).map_err(|_| BatchCode::InvalidJson)?;
    if let Some(object) = value.as_object() {
        if object.len() == 1 && object.get("flush").and_then(|v| v.as_bool()) == Some(true) { return Ok(Command::Flush); }
    }
    serde_json::from_value(value).map(Command::Document).map_err(|_| BatchCode::InvalidEnvelope.into())
}

#[cfg(test)]
mod tests;
