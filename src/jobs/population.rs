//! Bounded NDJSON ingestion for an immutable, complete owned-job population.
//! This retains private input in memory only; it opens no job files and runs
//! no planner/model. Hosts must admit the population and parser allocations
//! before calling it. Large populations fail closed instead of using the live
//! batch runner's weaker epoch-local duplicate window.
use super::{FrozenManifest, JobContract, JobError, JobId, JobInput, JobLimits, JobSecret, checkpoint};
use crate::{batch::BatchDocument, canonjson, native_engine::decode::DecodeStepControl};
use serde::Serialize;
use std::{collections::BTreeSet, io::BufRead};

/// Transport ceilings include blank lines and LF delimiters. They do not
/// change item semantics: records retain every byte BEFORE LF (including CR).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PopulationReadLimits {
    pub max_stream_bytes: u64,
    pub max_lines: u64,
}
impl PopulationReadLimits {
    pub fn validate(self) -> Result<(), JobError> {
        if self.max_stream_bytes == 0 || self.max_stream_bytes > i64::MAX as u64
            || self.max_lines == 0 || self.max_lines > i64::MAX as u64 {
            return Err(JobError::InvalidLimits);
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PopulationStats {
    pub items: u64,
    pub input_bytes: u64,
    pub input_lines: u64,
    /// Same logical accounting as FrozenManifest: ID + original + normalized.
    /// Original and normalized borrow ONE retained record in this profile.
    pub snapshot_bytes: u64,
}
struct Envelope { id: String, bytes: Vec<u8> }

/// No Debug/Clone/Deserialize or mutable content access. Task arguments remain
/// inside the exact original envelope, not a reserialized serde Value. The
/// complete native recipe and typed arguments are still verified by JobRunner.
pub struct JobPopulation { items: Vec<Envelope>, stats: PopulationStats }
impl JobPopulation {
    /// Consume a finite NDJSON stream before any durable execution. Only empty
    /// LF/CRLF records are ignored; whitespace-only input and flush controls
    /// are invalid. EOF may terminate a final record without LF. Input errors
    /// discard the whole candidate population, never return a usable prefix.
    ///
    /// Checkpoints bound bytes processed BETWEEN BufRead calls, not the time
    /// spent in arbitrary blocking IO. The supplied reader's own buffers and
    /// allocator overhead remain the embedding host's accounting obligation.
    pub fn read_ndjson<R: BufRead, C: DecodeStepControl>(reader: &mut R, key: &JobSecret,
        job: JobId, limits: JobLimits, transport: PopulationReadLimits, control: &mut C)
        -> Result<Self, JobError> {
        checkpoint(control)?; limits.validate()?; transport.validate()?;
        let mut result = Self { items: Vec::new(), stats: PopulationStats::default() };
        let mut ids = BTreeSet::new();
        while let Some(bytes) = read_record(reader, &mut result.stats, limits.max_input_bytes_per_item, transport, control)? {
            checkpoint(control)?;
            if bytes.is_empty() { continue; }
            if result.stats.items >= limits.max_items { return Err(JobError::Limit); }
            let id = envelope_id(&bytes, limits)?;
            let snapshot = result.stats.snapshot_bytes.checked_add(id.len() as u64)
                .and_then(|n| n.checked_add(bytes.len() as u64))
                .and_then(|n| n.checked_add(bytes.len() as u64))
                .filter(|&n| n <= limits.max_snapshot_bytes).ok_or(JobError::Limit)?;
            // The transient uniqueness index contains no extra plaintext IDs.
            let commitment = key.commit(b"population-reader-id", &[&job.0, id.as_bytes()]);
            if !ids.insert(commitment.0) { return Err(JobError::DuplicateId); }
            result.items.try_reserve(1).map_err(|_| JobError::Allocation)?;
            result.items.push(Envelope { id, bytes });
            result.stats.items += 1;
            result.stats.snapshot_bytes = snapshot;
        }
        if result.items.is_empty() { return Err(JobError::InvalidInput); }
        if result.stats.items > limits.max_attempts { return Err(JobError::InvalidLimits); }
        checkpoint(control)?;
        Ok(result)
    }
    pub fn stats(&self) -> PopulationStats { self.stats }
    pub fn inputs(&self) -> impl ExactSizeIterator<Item = JobInput<'_>> + '_ {
        self.items.iter().map(|item| JobInput { id: &item.id, original: &item.bytes, normalized: &item.bytes })
    }
    /// Bounded borrow-only index for JobRunner. It owns no copy of input text.
    /// Keep this population alive until the runner (and its index) are dropped.
    pub fn borrowed_inputs(&self) -> Result<Vec<JobInput<'_>>, JobError> {
        let mut inputs = Vec::new();
        inputs.try_reserve_exact(self.items.len()).map_err(|_| JobError::Allocation)?;
        inputs.extend(self.inputs());
        Ok(inputs)
    }
    pub fn freeze<R: Serialize, C: DecodeStepControl>(&self, key: &JobSecret,
        contract: JobContract<'_, R>, control: &mut C) -> Result<FrozenManifest, JobError> {
        FrozenManifest::freeze(key, contract, self.inputs(), control)
    }
}
fn envelope_id(bytes: &[u8], limits: JobLimits) -> Result<String, JobError> {
    let text = std::str::from_utf8(bytes).map_err(|_| JobError::InvalidInput)?;
    let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits {
        max_depth: 64, max_string_bytes: limits.max_input_bytes_per_item,
    }).map_err(|_| JobError::InvalidInput)?;
    let document: BatchDocument<serde_json::Value> = serde_json::from_value(value).map_err(|_| JobError::InvalidInput)?;
    if document.id.is_empty() || document.id.len() > limits.max_id_bytes { return Err(JobError::InvalidInput); }
    Ok(document.id)
}
fn read_record<R: BufRead, C: DecodeStepControl>(reader: &mut R, stats: &mut PopulationStats,
    max_record: usize, limits: PopulationReadLimits, control: &mut C) -> Result<Option<Vec<u8>>, JobError> {
    let mut record = Vec::new();
    loop {
        checkpoint(control)?;
        let available = reader.fill_buf().map_err(|_| JobError::Io)?;
        if available.is_empty() {
            if record.is_empty() { return Ok(None); }
            charge_line(stats, limits)?;
            return Ok(Some(record));
        }
        // Never copy an unbounded fill_buf slice or defer cancellation until
        // the end of a giant line. No drain/recovery loop exists for bad input.
        let window = &available[..available.len().min(8192)];
        let newline = window.iter().position(|&b| b == b'\n');
        let length = newline.unwrap_or(window.len());
        let consumed = length + usize::from(newline.is_some());
        let total = stats.input_bytes.checked_add(consumed as u64)
            .filter(|&n| n <= limits.max_stream_bytes).ok_or(JobError::Limit)?;
        let next_length = record.len().checked_add(length).filter(|&n| n <= max_record).ok_or(JobError::Limit)?;
        record.try_reserve(length).map_err(|_| JobError::Allocation)?;
        record.extend_from_slice(&window[..length]);
        debug_assert_eq!(record.len(), next_length);
        reader.consume(consumed);
        stats.input_bytes = total;
        if newline.is_some() {
            charge_line(stats, limits)?;
            // Only a complete empty CRLF is ignored. A lone CR at EOF remains
            // an invalid whitespace-only record; real envelope CR is retained.
            if record == b"\r" { record.clear(); }
            return Ok(Some(record));
        }
    }
}
fn charge_line(stats: &mut PopulationStats, limits: PopulationReadLimits) -> Result<(), JobError> {
    stats.input_lines = stats.input_lines.checked_add(1).filter(|&n| n <= limits.max_lines).ok_or(JobError::Limit)?;
    Ok(())
}

#[cfg(test)] mod tests;
