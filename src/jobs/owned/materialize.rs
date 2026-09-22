//! Owned output publication, not an exactly-once claim about arbitrary pipes.
use super::*;
use crate::local_io::LocalIoError;
use std::io::{Read, Write};

impl OwnedJob {
    /// Write one canonical result plus LF per frozen ordinal to the fixed
    /// owner-only `materialized.ndjson`. Stage, sync, and no-replace publish via
    /// the existing local IO transaction; then journal `materialized` states.
    /// A crash after publication is recovered by byte-for-byte verification,
    /// never by overwriting an existing destination or rerunning inference.
    pub fn materialize_ordered<C: DecodeStepControl>(&mut self, control: &mut C) -> Result<JobProgress, JobError> {
        self.ready()?; checkpoint(control)?;
        if self.header.committed != self.header.items { return Err(JobError::Incomplete); }
        self.poisoned = true;
        self.verify_prefix(control)?;
        if self.files.spool.metadata().map_err(|_| JobError::Io)?.len() != self.header.spool_end {
            return Err(JobError::UncommittedTail);
        }
        if self.header.materialized { self.poisoned = false; return Ok(self.progress()); }
        if self.files.materialized(self.manifest.limits.max_materialized_bytes)?.is_some() {
            // An existing destination is acceptable only as exact committed
            // output from an interrupted publication, never as caller input.
            self.verify_materialized(control)?;
            self.files.materialized(self.manifest.limits.max_materialized_bytes)?
                .ok_or(JobError::Corrupt)?.sync_all().map_err(|_| JobError::PublicationUncertain)?;
            self.files.sync_directory()?;
        } else {
            let destination = self.files.destination()?;
            let mut producer_error = None;
            let stage = destination.stage_with(|file| {
                self.write_materialized(file, control).map_err(|error| {
                    producer_error = Some(error); LocalIoError::Io
                })
            }).map_err(|error| producer_error.unwrap_or_else(|| local_error(error)))?;
            checkpoint(control)?;
            stage.publish().map_err(local_error)?;
        }
        self.fault(Fault::Published)?;
        checkpoint(control)?;
        let mut header = self.header.clone(); header.materialized = true;
        self.journal.transaction(|journal| {
            for ordinal in 0..header.items {
                checkpoint(control)?;
                let mut row = self.item(ordinal)?;
                row.stage = Stage::Materialized;
                journal.write(Table::Item, ordinal, &self.key, &row, false)?;
            }
            journal.write(Table::Header, 0, &self.key, &header, false)
        })?;
        self.files.check_database(self.manifest.limits.max_journal_bytes)?;
        self.files.sync_database()?;
        self.header = header;
        self.fault(Fault::Materialized)?;
        self.poisoned = false; Ok(self.progress())
    }
    fn write_materialized<W: Write, C: DecodeStepControl>(&mut self, writer: &mut W, control: &mut C) -> Result<(), JobError> {
        let mut total = 0_u64;
        for ordinal in 0..self.header.items {
            checkpoint(control)?;
            let row = self.item(ordinal)?;
            let bytes = self.read_row(&row)?;
            total = materialized_total(total, bytes.len(), self.manifest.limits.max_materialized_bytes)?;
            writer.write_all(&bytes).and_then(|_| writer.write_all(b"\n")).map_err(|_| JobError::Io)?;
        }
        checkpoint(control)
    }
    pub(super) fn verify_materialized<C: DecodeStepControl>(&mut self, control: &mut C) -> Result<(), JobError> {
        if self.header.committed != self.header.items { return Err(JobError::Corrupt); }
        let mut file = self.files.materialized(self.manifest.limits.max_materialized_bytes)?.ok_or(JobError::Corrupt)?;
        let mut buffer = [0_u8; 8192]; let mut total = 0_u64;
        for ordinal in 0..self.header.items {
            checkpoint(control)?;
            let row = self.item(ordinal)?;
            let bytes = self.read_row(&row)?;
            total = materialized_total(total, bytes.len(), self.manifest.limits.max_materialized_bytes)?;
            for chunk in bytes.chunks(buffer.len()) {
                file.read_exact(&mut buffer[..chunk.len()]).map_err(|_| JobError::Corrupt)?;
                if &buffer[..chunk.len()] != chunk { return Err(JobError::Corrupt); }
            }
            file.read_exact(&mut buffer[..1]).map_err(|_| JobError::Corrupt)?;
            if buffer[0] != b'\n' { return Err(JobError::Corrupt); }
        }
        if file.read(&mut buffer[..1]).map_err(|_| JobError::Io)? != 0 { return Err(JobError::Corrupt); }
        checkpoint(control)
    }
}
fn materialized_total(current: u64, result: usize, max: u64) -> Result<u64, JobError> {
    current.checked_add(result as u64).and_then(|n| n.checked_add(1)).filter(|&n| n <= max).ok_or(JobError::Limit)
}
fn local_error(error: LocalIoError) -> JobError {
    match error {
        LocalIoError::AlreadyExists => JobError::AlreadyExists,
        LocalIoError::PublicationUncertain => JobError::PublicationUncertain,
        LocalIoError::Io => JobError::Io,
        LocalIoError::UnsupportedProfile => JobError::Platform,
        _ => JobError::UnsafeStorage,
    }
}
