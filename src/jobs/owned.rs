//! Serial durable operations on one exclusively held owned job. Runtime,
//! resource admission, model/vocabulary ownership and actual native cleanup
//! remain the embedding host's responsibilities, not journal capabilities.
use super::{FrozenManifest, JobError, JobId, JobInput, JobSecret, JobWork, checkpoint,
    frame::{self, Pointer}, journal::{Journal, Table}, manifest::{Binding, ItemBinding}};
use crate::{local_io::JobFiles, native_engine::decode::DecodeStepControl};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TailPolicy {
    /// Diagnose without modifying an uncommitted/torn suffix.
    Refuse,
    /// Truncate the uncommitted spool suffix and discard reserved-name output
    /// stages ONLY after every journal-authorized frame has authenticated.
    DiscardUncommitted,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JobProgress {
    pub job_id: JobId, pub items: u64, pub committed: u64,
    pub attempts: u64, pub reserved_work: JobWork, pub spool_bytes: u64,
    pub materialized: bool,
}
#[derive(Clone, Copy, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Stage { Pending, Admitted, Running, ResultCommitted, Materialized }
#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Header {
    schema_version: u32, binding: Binding, items: u64, committed: u64,
    attempts: u64, used: JobWork, spool_end: u64, materialized: bool,
}
#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Item {
    binding: ItemBinding, stage: Stage, attempts: u64, used: JobWork, pointer: Option<Pointer>,
}

/// Holds the kernel lock until after the database and spool handles close.
/// There is no Clone, public connection, raw file handle or Deserialize path.
/// All input/result buffers and database-engine allocations must be priced by
/// the host when used inside a process-admitted NLP invocation.
pub struct OwnedJob {
    journal: Journal,
    files: JobFiles,
    key: JobSecret,
    manifest: FrozenManifest,
    header: Header,
    poisoned: bool,
    #[cfg(test)] fail_at: Option<Fault>,
}
impl OwnedJob {
    /// Explicit result-storage opt-in. `root` must already be an owner-only
    /// directory; reserved filenames must not exist. Store the fresh random
    /// job secret through the host's protected-key policy BEFORE calling this.
    /// Failed initialization is never silently adopted or overwritten.
    pub fn create<C: DecodeStepControl>(root: &Path, key: JobSecret, manifest: FrozenManifest, control: &mut C)
        -> Result<Self, JobError> {
        checkpoint(control)?;
        check_secret(&key, &manifest)?;
        let limits = manifest.limits;
        let files = JobFiles::open(root, true, limits.max_journal_bytes, limits.max_spool_bytes)?;
        let journal = Journal::open(&files.database_path(), limits.max_journal_bytes)?;
        let header = Header { schema_version: 1, binding: manifest.binding.clone(), items: manifest.item_count(),
            committed: 0, attempts: 0, used: JobWork::default(), spool_end: 0, materialized: false };
        journal.transaction(|journal| {
            journal.create_schema()?;
            for binding in &manifest.items {
                checkpoint(control)?;
                let row = Item { binding: binding.clone(), stage: Stage::Pending, attempts: 0,
                    used: JobWork::default(), pointer: None };
                journal.write(Table::Item, binding.ordinal, &key, &row, true)?;
            }
            journal.write(Table::Header, 0, &key, &header, true)
        })?;
        files.check_database(limits.max_journal_bytes)?; files.sync_database()?;
        Ok(Self { journal, files, key, manifest, header, poisoned: false, #[cfg(test)] fail_at: None })
    }

    /// Recompute `manifest` from the complete original population first. Key,
    /// semantic and population mismatches refuse before spool repair or work.
    /// Running/admitted attempts stay charged and may be retried explicitly;
    /// committed records are never inferred again to reconstruct their bytes.
    pub fn resume<C: DecodeStepControl>(root: &Path, key: JobSecret, manifest: FrozenManifest,
        tail: TailPolicy, control: &mut C) -> Result<Self, JobError> {
        checkpoint(control)?; check_secret(&key, &manifest)?;
        let limits = manifest.limits;
        let files = JobFiles::open(root, false, limits.max_journal_bytes, limits.max_spool_bytes)?;
        let journal = Journal::open(&files.database_path(), limits.max_journal_bytes)?;
        let header: Header = journal.read(Table::Header, 0, &key)?;
        manifest.binding.compare(&header.binding)?;
        let mut job = Self { journal, files, key, manifest, header, poisoned: true, #[cfg(test)] fail_at: None };
        job.verify_prefix(control)?;
        let length = job.files.spool.metadata().map_err(|_| JobError::Io)?.len();
        if length != job.header.spool_end {
            if tail == TailPolicy::Refuse { return Err(JobError::UncommittedTail); }
            checkpoint(control)?;
            job.files.spool.set_len(job.header.spool_end).map_err(|_| JobError::Io)?;
            job.files.spool.sync_all().map_err(|_| JobError::Io)?;
        }
        if job.files.has_stages()? {
            if tail == TailPolicy::Refuse { return Err(JobError::UncommittedTail); }
            checkpoint(control)?;
            job.files.discard_stages(job.manifest.limits.max_materialized_bytes)?;
        }
        job.poisoned = false;
        Ok(job)
    }
    pub fn progress(&self) -> JobProgress {
        JobProgress { job_id: self.header.binding.job, items: self.header.items,
            committed: self.header.committed, attempts: self.header.attempts,
            reserved_work: self.header.used, spool_bytes: self.header.spool_end, materialized: self.header.materialized }
    }
    pub fn is_poisoned(&self) -> bool { self.poisoned }
    fn ready(&self) -> Result<(), JobError> {
        if self.poisoned { Err(JobError::Poisoned) } else { Ok(()) }
    }
    fn item(&self, ordinal: u64) -> Result<Item, JobError> {
        let row: Item = self.journal.read(Table::Item, ordinal, &self.key)?;
        if self.manifest.items.get(ordinal as usize) != Some(&row.binding) { return Err(JobError::Corrupt); }
        Ok(row)
    }
    fn persist(&self, row: &Item, header: &Header) -> Result<(), JobError> {
        self.files.check_database(self.manifest.limits.max_journal_bytes)?;
        self.journal.transaction(|journal| {
            journal.write(Table::Item, row.binding.ordinal, &self.key, row, false)?;
            journal.write(Table::Header, 0, &self.key, header, false)
        })?;
        self.files.check_database(self.manifest.limits.max_journal_bytes)?;
        // FULL transaction commit precedes these additional file/directory
        // barriers. Failure is uncertain, never an invitation to refund/retry.
        self.files.sync_database()
    }

    /// Reserve ALL work axes before the host executes the next exact input.
    /// The borrow prevents concurrent mutation, replayed tokens and cross-job
    /// token use. Dropping an unfinished attempt poisons this session; reopen
    /// explicitly to retry with a NEW debit. No automatic retry exists.
    pub fn begin<C: DecodeStepControl>(&mut self, input: &JobInput<'_>, work: JobWork, control: &mut C)
        -> Result<Attempt<'_>, JobError> {
        self.ready()?; checkpoint(control)?;
        let ordinal = self.header.committed;
        self.manifest.verify_input(&self.key, ordinal, input)?;
        let mut row = self.item(ordinal)?;
        if row.pointer.is_some() || !matches!(row.stage, Stage::Pending | Stage::Admitted | Stage::Running) {
            return Err(JobError::InvalidTransition);
        }
        let mut header = self.header.clone();
        header.attempts = header.attempts.checked_add(1).filter(|&n| n <= self.manifest.limits.max_attempts).ok_or(JobError::WorkLimit)?;
        header.used = header.used.checked_add(work)?;
        if !header.used.fits(self.manifest.limits.max_work) { return Err(JobError::WorkLimit); }
        row.attempts = row.attempts.checked_add(1).ok_or(JobError::WorkLimit)?;
        row.used = row.used.checked_add(work)?;
        self.poisoned = true;
        row.stage = Stage::Admitted;
        self.persist(&row, &header)?;
        self.header = header;
        self.fault(Fault::Admitted)?;
        checkpoint(control)?;
        row.stage = Stage::Running;
        self.persist(&row, &self.header)?;
        self.fault(Fault::Running)?;
        Ok(Attempt { job: self, row, finished: false })
    }

    /// Independently validate the entire journal-authorized prefix. A valid
    /// looking tail remains an error here; only explicit resume policy repairs.
    pub fn verify<C: DecodeStepControl>(&mut self, control: &mut C) -> Result<JobProgress, JobError> {
        self.ready()?; self.poisoned = true;
        self.verify_prefix(control)?;
        if self.files.spool.metadata().map_err(|_| JobError::Io)?.len() != self.header.spool_end {
            return Err(JobError::UncommittedTail);
        }
        self.poisoned = false; Ok(self.progress())
    }
    fn verify_prefix<C: DecodeStepControl>(&mut self, control: &mut C) -> Result<(), JobError> {
        checkpoint(control)?;
        let header: Header = self.journal.read(Table::Header, 0, &self.key)?;
        if header != self.header || header.schema_version != 1 || header.items != self.manifest.item_count()
            || header.committed > header.items || header.attempts > self.manifest.limits.max_attempts
            || !header.used.fits(self.manifest.limits.max_work)
            || (header.materialized && header.committed != header.items) { return Err(JobError::Corrupt); }
        self.manifest.binding.compare(&header.binding)?;
        self.journal.check_count(header.items)?;
        self.files.check_database(self.manifest.limits.max_journal_bytes)?;
        let physical = self.files.spool.metadata().map_err(|_| JobError::Io)?.len();
        if physical < header.spool_end || physical > self.manifest.limits.max_spool_bytes { return Err(JobError::Corrupt); }
        let mut end = 0_u64; let mut attempts = 0_u64; let mut used = JobWork::default();
        for ordinal in 0..header.items {
            checkpoint(control)?;
            let row = self.item(ordinal)?;
            attempts = attempts.checked_add(row.attempts).ok_or(JobError::Corrupt)?;
            used = used.checked_add(row.used).map_err(|_| JobError::Corrupt)?;
            if ordinal < header.committed {
                let expected_stage = if header.materialized { Stage::Materialized } else { Stage::ResultCommitted };
                if row.attempts == 0 || row.stage != expected_stage {
                    return Err(JobError::Corrupt);
                }
                let pointer = row.pointer.as_ref().ok_or(JobError::Corrupt)?;
                if pointer.offset != end { return Err(JobError::Corrupt); }
                let bytes = frame::read(&mut self.files.spool, &self.key, &header.binding,
                    &row.binding, pointer, self.manifest.limits.max_result_bytes)?;
                drop(bytes);
                end = pointer.end()?;
            } else {
                if row.pointer.is_some() || !matches!(row.stage, Stage::Pending | Stage::Admitted | Stage::Running)
                    || (row.stage == Stage::Pending && (row.attempts != 0 || row.used != JobWork::default()))
                    || (row.stage != Stage::Pending && row.attempts == 0)
                    || (ordinal > header.committed && row.stage != Stage::Pending) { return Err(JobError::Corrupt); }
            }
        }
        if end != header.spool_end || attempts != header.attempts || used != header.used { return Err(JobError::Corrupt); }
        if header.materialized { self.verify_materialized(control)?; }
        Ok(())
    }

    /// Bounded, authenticated bytes for an already committed result. This is
    /// storage data, not native execution authority or an external delivery ack.
    /// The caller owns accounting and lifetime of the returned buffer.
    pub fn read_committed<C: DecodeStepControl>(&mut self, ordinal: u64, control: &mut C) -> Result<Vec<u8>, JobError> {
        self.ready()?; checkpoint(control)?;
        if ordinal >= self.header.committed { return Err(JobError::Incomplete); }
        self.poisoned = true;
        let row = self.item(ordinal)?;
        let result = self.read_row(&row)?;
        self.poisoned = false; Ok(result)
    }
    fn read_row(&mut self, row: &Item) -> Result<Vec<u8>, JobError> {
        frame::read(&mut self.files.spool, &self.key, &self.header.binding, &row.binding,
            row.pointer.as_ref().ok_or(JobError::Corrupt)?, self.manifest.limits.max_result_bytes)
    }

    fn fault(&mut self, point: Fault) -> Result<(), JobError> {
        #[cfg(test)] if self.fail_at == Some(point) { return Err(JobError::Io); }
        #[cfg(not(test))] let _ = point;
        Ok(())
    }
}

/// An exclusively borrowed, durably debited running attempt. Result guards
/// owned by the host stay live through `commit` because the result is borrowed.
/// The receipt proves storage commit, never native accuracy or model quality.
pub struct Attempt<'a> { job: &'a mut OwnedJob, row: Item, finished: bool }
impl Attempt<'_> {
    pub fn ordinal(&self) -> u64 { self.row.binding.ordinal }
    pub fn number(&self) -> u64 { self.row.attempts }
    pub fn commit<T: Serialize, C: DecodeStepControl>(mut self, result: &T, control: &mut C) -> Result<JobProgress, JobError> {
        checkpoint(control)?;
        let limits = self.job.manifest.limits;
        let pointer = frame::append(&mut self.job.files.spool, &self.job.key, &self.job.header.binding,
            &self.row.binding, self.job.header.spool_end, result, limits.max_result_bytes, limits.max_spool_bytes)?;
        self.job.fault(Fault::SpoolWritten)?;
        self.job.files.spool.sync_all().map_err(|_| JobError::Io)?;
        self.job.fault(Fault::SpoolSynced)?;
        checkpoint(control)?;
        let mut header = self.job.header.clone();
        header.committed = header.committed.checked_add(1).ok_or(JobError::Limit)?;
        header.spool_end = pointer.end()?;
        self.row.pointer = Some(pointer); self.row.stage = Stage::ResultCommitted;
        self.job.persist(&self.row, &header)?;
        self.job.header = header;
        self.job.fault(Fault::ResultCommitted)?;
        self.job.poisoned = false; self.finished = true;
        Ok(self.job.progress())
    }
}
impl Drop for Attempt<'_> {
    fn drop(&mut self) { if !self.finished { self.job.poisoned = true; } }
}
fn check_secret(key: &JobSecret, manifest: &FrozenManifest) -> Result<(), JobError> {
    manifest.limits.validate()?;
    if !key.commit(b"key-check", &[&manifest.binding.job.0]).matches(manifest.binding.secret) {
        return Err(JobError::Mismatch(super::MismatchField::Secret));
    }
    Ok(())
}
#[derive(Clone, Copy, Eq, PartialEq)]
enum Fault { Admitted, Running, SpoolWritten, SpoolSynced, ResultCommitted, Published, Materialized }

// Kept beside the owner so publication and journal-state transitions share
// the same exclusive session, poisoned-error policy, and cancellation control.
mod materialize;
#[cfg(test)] mod tests;
