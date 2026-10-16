//! Corpus execution and completion framing, separate from model ownership.
//! The only production processor is the charged shared-model INT4 host.
use super::*;

pub(super) struct Prepared<T> { pub value: T, pub planned: Int4Work }
pub(super) struct Completed<T> { pub value: T, pub planned: Int4Work, pub actual: Int4Work, pub records: usize }
/// Private static seam. No embedding/CLI caller can substitute a processor.
pub(super) trait Processor {
    type Prepared;
    type Output: Serialize;
    fn checkpoint(&mut self) -> Result<(), CandidateError>;
    fn prepare(&mut self, window: InputWindow) -> Result<Prepared<Self::Prepared>, CandidateError>;
    /// All work/output admission has succeeded before execute can save or infer.
    fn execute(&mut self, prepared: Self::Prepared) -> Result<Completed<Self::Output>, CandidateError>;
}

pub(super) fn drive<P: Processor>(processor: &mut P, input: &mut impl BufRead, output: &mut impl Write,
    options: CorpusArgs, limits: CorpusLimits, line_cap: usize, checkpoints: u64) -> Result<(), CandidateError> {
    let mut reader = CorpusInput::new(options)?;
    let mut ledger = CorpusLedger::default();
    let mut output = wire::Output::new(output, options.max_total_output_bytes);
    loop {
        let window = reader.next_window(input, line_cap, || processor.checkpoint())?;
        if window.records.is_empty() { break; }
        let first = window.first_sequence; let count = window.records.len();
        // Reserve transport room for the complete DECLARED window and footer
        // before model loading/planning, not after an expensive inference.
        output.admit_frame(limits.frame_bytes)?;
        processor.checkpoint()?;
        let prepared = processor.prepare(window)?;
        ledger.admit(first, count, prepared.planned, checkpoints, options)?;
        processor.checkpoint()?;
        let completed = processor.execute(prepared.value)?;
        if completed.records != count { return Err(CandidateError::Batch); }
        ledger.complete(completed.planned, completed.actual)?;
        processor.checkpoint()?;
        let frame = WindowFrame { schema_version: 1, protocol: CORPUS_PROTOCOL, event: "window_complete",
            window: ledger.windows, first_request_seq: first, records: count as u64, batch: &completed.value };
        publish_canonical(&frame, limits.frame_bytes, &mut output)?;
        // Completed owns the real result guard. It drops only after serialization,
        // external write AND flush, and before allocating the next input window.
        drop(completed);
    }
    processor.checkpoint()?;
    let complete = ledger.summary(&reader, output.written)?;
    publish_canonical(&complete, wire::FOOTER_BYTES, &mut output)
}

#[derive(Serialize)]
struct WindowFrame<'a, T: Serialize> {
    schema_version: u32, protocol: &'static str, event: &'static str,
    window: u64, first_request_seq: u64, records: u64, batch: &'a T,
}
/// The canonical writer rejects nonfinite numbers and sorts typed object keys.
/// Both tree/byte staging are covered by preparation/output admission. No
/// external write occurs until the entire frame and its newline fit the cap.
fn publish_canonical<T: Serialize>(value: &T, cap: usize, output: &mut impl Write) -> Result<(), CandidateError> {
    let mut bytes = canonjson::canonical_bytes(value).map_err(|_| CandidateError::Output)?;
    if bytes.len().checked_add(1).is_none_or(|length| length > cap) { return Err(CandidateError::Output); }
    bytes.try_reserve_exact(1).map_err(|_| CandidateError::Memory)?;
    bytes.push(b'\n');
    output.write_all(&bytes).and_then(|()| output.flush()).map_err(|_| CandidateError::Output)
}
