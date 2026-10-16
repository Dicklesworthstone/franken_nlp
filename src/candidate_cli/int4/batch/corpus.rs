//! Bounded INT4 corpus windows with one monotone, whole-invocation ledger.
//! This is a fail-fast pipe, not a durable snapshot, retry loop or daemon.
use super::*;
use std::collections::BTreeSet;
use crate::native_engine::portable_int8::ProjectionWork;

pub(in crate::candidate_cli::int4) const CORPUS_PROTOCOL: &str = "fnlp-source-int4-text-corpus-v1";
const MAX_CORPUS_RECORDS: u64 = 1_000_000;
const MAX_TRANSPORT_BYTES: u64 = 16 * 1024 * MIB;
const FRAME_ALLOWANCE: usize = 4096;

#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::candidate_cli::int4) struct CorpusArgs {
    #[arg(long, value_enum, default_value = "generate")]
    task: BatchTask,
    /// Simultaneously prepared/executed documents, 1..=64, not total records.
    #[arg(long, default_value_t = 8)]
    cohort_rows: usize,
    /// Whole-invocation record/seen-ID bound. IDs never reset between windows.
    #[arg(long, default_value_t = 1000)]
    max_records: u64,
    /// Whole input transport, including JSON framing and CR/LF bytes.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_input_bytes: u64,
    /// Whole output transport, including the required corpus_complete frame.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_output_bytes: u64,
    #[arg(long, default_value_t = 1_000_000)]
    max_total_forward_positions: u64,
    #[arg(long, default_value_t = 100_000_000_000)]
    max_total_projected_logits: u64,
    #[arg(long, default_value_t = 1_000_000_000_000)]
    max_total_attention_pairs: u64,
    #[arg(long, default_value_t = 10_000_000_000_000)]
    max_total_group_dots: u64,
    #[arg(long, default_value_t = 1_000_000_000_000_000)]
    max_total_multiply_accumulates: u64,
    /// Sum of reserved max-checkpoints allowances, charged once per window.
    /// Early completion does not refund unused checkpoints to a later window.
    #[arg(long, default_value_t = 1_000_000_000_000)]
    max_total_execution_checkpoints: u64,
}

#[derive(Clone, Copy, Debug)]
struct CorpusLimits { window: BatchLimits, frame_bytes: usize }
impl CorpusArgs {
    fn check(self) -> Result<(), CandidateError> {
        if self.cohort_rows == 0 || self.cohort_rows > MAX_INT4_SEQUENCES
            || self.max_records == 0 || self.max_records > MAX_CORPUS_RECORDS
            || self.max_total_input_bytes == 0 || self.max_total_input_bytes > MAX_TRANSPORT_BYTES
            || self.max_total_output_bytes < wire::FOOTER_BYTES as u64
            || self.max_total_output_bytes > MAX_TRANSPORT_BYTES
            || self.max_total_forward_positions == 0 || self.max_total_projected_logits == 0
            || self.max_total_attention_pairs == 0 || self.max_total_group_dots == 0
            || self.max_total_multiply_accumulates == 0 || self.max_total_execution_checkpoints == 0 {
            return Err(CandidateError::Arguments);
        }
        Ok(())
    }
    fn window_args(self) -> BatchArgs {
        BatchArgs { task: self.task, max_records: self.cohort_rows, max_total_input_bytes: 64 * MIB }
    }
    fn ceiling(self) -> Int4Work {
        Int4Work { forward_positions: self.max_total_forward_positions,
            projected_logits: self.max_total_projected_logits, attention_pairs: self.max_total_attention_pairs,
            projections: ProjectionWork { dot_products: self.max_total_group_dots,
                multiply_accumulates: self.max_total_multiply_accumulates } }
    }
    fn validate(self, args: &Int4Args) -> Result<CorpusLimits, CandidateError> {
        self.check()?;
        if args.max_checkpoints > self.max_total_execution_checkpoints { return Err(CandidateError::Arguments); }
        // One bounded input window and two frame stages, plus the full finite
        // seen-ID population. Native/KV/sampler charges scale with cohort_rows,
        // NOT max_records. The ID-set allowance is modeled payload, not RSS.
        let extra = sum(&[times(self.max_records, 1024)?, (2 * FRAME_ALLOWANCE + 8192) as u64])?;
        let window = self.window_args().validate_with_preparation(args, extra)?;
        let frame_bytes = window.publication_bytes.checked_add(FRAME_ALLOWANCE).ok_or(CandidateError::Arguments)?;
        Ok(CorpusLimits { window, frame_bytes })
    }
}

/// Contents are retained only for the current window. IDs are retained for
/// the complete finite invocation. No next-window read occurs during inference.
struct CorpusInput {
    options: CorpusArgs, ids: BTreeSet<String>, records: u64, consumed: u64,
    eof: bool, failed: bool,
}
struct InputWindow { first_sequence: u64, records: Vec<Record> }
impl CorpusInput {
    fn new(options: CorpusArgs) -> Result<Self, CandidateError> {
        options.check()?;
        Ok(Self { options, ids: BTreeSet::new(), records: 0, consumed: 0, eof: false, failed: false })
    }
    fn next_window(&mut self, input: &mut impl BufRead, line_cap: usize,
        mut checkpoint: impl FnMut() -> Result<(), CandidateError>) -> Result<InputWindow, CandidateError> {
        if self.failed { return Err(CandidateError::Batch); }
        let result = self.read_window(input, line_cap, &mut checkpoint);
        if result.is_err() { self.failed = true; }
        result
    }
    fn read_window(&mut self, input: &mut impl BufRead, line_cap: usize,
        checkpoint: &mut impl FnMut() -> Result<(), CandidateError>) -> Result<InputWindow, CandidateError> {
        if line_cap == 0 || line_cap > MAX_INPUT_BYTES { return Err(CandidateError::Arguments); }
        let first_sequence = self.records.checked_add(1).ok_or(CandidateError::Batch)?;
        let mut records = Vec::new();
        if self.eof { return Ok(InputWindow { first_sequence, records }); }
        records.try_reserve_exact(self.options.cohort_rows).map_err(|_| CandidateError::Memory)?;
        while records.len() < self.options.cohort_rows {
            checkpoint()?;
            // Probe EOF at the exact record ceiling without consuming another
            // record. A full window otherwise returns immediately, not after
            // blocking for the next line. BufRead may prefetch bounded bytes.
            if self.records == self.options.max_records {
                if !input.fill_buf().map_err(|_| CandidateError::Input)?.is_empty() { return Err(CandidateError::Input); }
                self.eof = true; break;
            }
            let Some(line) = wire::read_line(input, line_cap, &mut self.consumed, self.options.max_total_input_bytes)? else {
                self.eof = true; break;
            };
            let task = match self.options.task { BatchTask::Generate => wire::TextTask::Generate, BatchTask::Chat => wire::TextTask::Chat };
            let record = wire::parse_record(task, &line, line_cap)?;
            if self.ids.contains(&record.id) { return Err(CandidateError::Input); }
            let next = self.records.checked_add(1).ok_or(CandidateError::Batch)?;
            self.ids.insert(record.id.clone()); self.records = next;
            records.push(record);
        }
        checkpoint()?;
        Ok(InputWindow { first_sequence, records })
    }
}

#[derive(Clone, Copy)]
struct WindowCharge { records: u64, planned: Int4Work }
#[derive(Default)]
struct CorpusLedger {
    reserved: Int4Work, actual: Int4Work, checkpoints: u64,
    completed_records: u64, windows: u64, pending: Option<WindowCharge>,
}
fn fits(work: Int4Work, ceiling: Int4Work) -> bool {
    work.forward_positions <= ceiling.forward_positions && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs && work.projections.fits(ceiling.projections)
}
impl CorpusLedger {
    /// Every axis is checked before changing any ledger field. This reserves
    /// the compiled worst case, not actual early-stopping work. No refund API.
    fn admit(&mut self, first_sequence: u64, records: usize, planned: Int4Work,
        checkpoints: u64, options: CorpusArgs) -> Result<(), CandidateError> {
        options.check()?;
        let records = u64::try_from(records).map_err(|_| CandidateError::Batch)?;
        if self.pending.is_some() || records == 0 || records > options.cohort_rows as u64
            || first_sequence != self.completed_records.checked_add(1).ok_or(CandidateError::Batch)?
            || self.completed_records.checked_add(records).is_none_or(|n| n > options.max_records)
            || planned.forward_positions < records || planned.projected_logits == 0 || checkpoints == 0 {
            return Err(CandidateError::Batch);
        }
        let reserved = self.reserved.checked_add(planned).map_err(|_| CandidateError::Batch)?;
        let charged = self.checkpoints.checked_add(checkpoints).filter(|&n| n <= options.max_total_execution_checkpoints)
            .ok_or(CandidateError::Batch)?;
        if !fits(reserved, options.ceiling()) { return Err(CandidateError::Batch); }
        self.reserved = reserved; self.checkpoints = charged;
        self.pending = Some(WindowCharge { records, planned });
        Ok(())
    }
    /// Called only for a completely finalized window, before publication.
    /// A broken output write ends the caller immediately, without a footer.
    fn complete(&mut self, planned: Int4Work, actual: Int4Work) -> Result<(), CandidateError> {
        let pending = self.pending.ok_or(CandidateError::Batch)?;
        if planned != pending.planned || !fits(actual, planned)
            || actual.forward_positions < pending.records || actual.projected_logits == 0 {
            return Err(CandidateError::Batch);
        }
        let actual = self.actual.checked_add(actual).map_err(|_| CandidateError::Batch)?;
        let records = self.completed_records.checked_add(pending.records).ok_or(CandidateError::Batch)?;
        let windows = self.windows.checked_add(1).ok_or(CandidateError::Batch)?;
        self.actual = actual; self.completed_records = records; self.windows = windows; self.pending = None;
        Ok(())
    }
    fn summary(&self, input: &CorpusInput, output_bytes_before_footer: u64) -> Result<CorpusSummary, CandidateError> {
        if !input.eof || input.failed || self.pending.is_some() || self.completed_records != input.records {
            return Err(CandidateError::Batch);
        }
        Ok(CorpusSummary { schema_version: 1, protocol: CORPUS_PROTOCOL, event: "corpus_complete",
            evidence: "non_authoritative", records: self.completed_records, windows: self.windows,
            input_bytes: input.consumed, output_bytes_before_footer, reserved_work: self.reserved,
            actual_work: self.actual, reserved_execution_checkpoints: self.checkpoints })
    }
}
#[derive(Serialize)]
struct CorpusSummary {
    schema_version: u32, protocol: &'static str, event: &'static str, evidence: &'static str,
    records: u64, windows: u64, input_bytes: u64, output_bytes_before_footer: u64,
    reserved_work: Int4Work, actual_work: Int4Work, reserved_execution_checkpoints: u64,
}

#[cfg(test)] mod tests;
