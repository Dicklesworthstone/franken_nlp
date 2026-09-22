//! Management-only authority over existing authenticated job state.
//!
//! Stored commitments can authenticate stored outputs without the original
//! corpus. They cannot authorize a new attempt: this type exposes no input,
//! manifest, processor, connection, inner owner, or conversion to OwnedJob.
use super::*;
use crate::jobs::{manifest::bounded_json, MismatchField};
use std::collections::BTreeSet;

/// A bounded metadata report, never an inference or original-input receipt.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct StoredJobReport {
    pub schema_version: u32,
    pub verification_scope: &'static str,
    pub job_id: JobId,
    pub items: u64,
    pub committed: u64,
    pub attempts: u64,
    pub reserved_work: JobWork,
    pub committed_spool_bytes: u64,
    pub uncommitted_spool_bytes: u64,
    pub staged_output_present: bool,
    /// Journal-authorized publication; its bytes were independently checked.
    pub materialized: bool,
    /// A destination exists without a publication acknowledgement. Status
    /// reports presence only; verify/materialize compare it with committed data.
    pub unacknowledged_output_present: bool,
}

/// Exclusively locks an existing job without granting new execution authority.
/// Even the original JobLimits must match their keyed stored commitment.
/// Open authenticates every metadata row and journal-authorized result frame.
/// Orphans are reported, never truncated or promoted by this interface.
pub struct StoredJob {
    job: OwnedJob,
    report: StoredJobReport,
}
impl StoredJob {
    pub fn open<C: DecodeStepControl>(root: &Path, key: JobSecret, expected_job: JobId,
        limits: super::super::JobLimits, control: &mut C) -> Result<Self, JobError> {
        checkpoint(control)?;
        limits.validate()?;
        let files = JobFiles::open(root, false, limits.max_journal_bytes, limits.max_spool_bytes)?;
        let journal = Journal::open(&files.database_path(), limits.max_journal_bytes)?;
        let header: Header = journal.read(Table::Header, 0, &key)?;
        if header.binding.job != expected_job { return Err(JobError::Mismatch(MismatchField::Job)); }
        if !key.commit(b"key-check", &[&expected_job.0]).matches(header.binding.secret) {
            return Err(JobError::Mismatch(MismatchField::Secret));
        }
        let encoded_limits = bounded_json(&limits, 16 * 1024)?;
        if !key.commit(b"limits", &[&expected_job.0, &encoded_limits]).matches(header.binding.limits) {
            return Err(JobError::Mismatch(MismatchField::Limits));
        }
        if header.schema_version != 1 || header.items == 0 || header.items > limits.max_items
            || header.items > limits.max_attempts || header.committed > header.items {
            return Err(JobError::Corrupt);
        }
        journal.check_count(header.items)?;
        let count = usize::try_from(header.items).map_err(|_| JobError::Limit)?;
        let mut items = Vec::new();
        items.try_reserve_exact(count).map_err(|_| JobError::Allocation)?;
        let mut seen = BTreeSet::new();
        let mut population = key.commit(b"population-start", &[&expected_job.0]);
        for ordinal in 0..header.items {
            checkpoint(control)?;
            let row: Item = journal.read(Table::Item, ordinal, &key)?;
            if row.binding.ordinal != ordinal || !seen.insert(row.binding.id.0) {
                return Err(JobError::Corrupt);
            }
            let encoded = bounded_json(&row.binding, 4096)?;
            population = key.commit(b"population-next", &[&expected_job.0, &population.0, &encoded]);
            items.push(row.binding);
        }
        population = key.commit(b"population-end", &[&expected_job.0, &population.0,
            &header.items.to_le_bytes()]);
        if !population.matches(header.binding.population) { return Err(JobError::Corrupt); }
        drop(seen);
        // Only this private management owner is reconstructed. No caller can
        // retrieve its manifest to pretend stored hashes prove current inputs.
        let manifest = FrozenManifest { binding: header.binding.clone(), items, limits };
        let mut job = OwnedJob { journal, files, key, manifest, header, poisoned: true,
            #[cfg(test)] fail_at: None };
        job.verify_prefix(control)?;
        let report = describe(&job, control)?;
        job.poisoned = false;
        Ok(Self { job, report })
    }

    /// Snapshot from the last successful authenticated open/verification while
    /// this handle retains the exclusive kernel lock. Reports orphans honestly.
    pub fn status(&self) -> Result<StoredJobReport, JobError> {
        self.job.ready()?;
        Ok(self.report)
    }

    /// Check complete stored-state consistency without repairing anything.
    /// The presence of an uncommitted tail or stage is a typed failure. An
    /// unacknowledged published file must equal the committed result stream;
    /// verification never marks it materialized or authorizes fresh inference.
    pub fn verify<C: DecodeStepControl>(&mut self, control: &mut C) -> Result<StoredJobReport, JobError> {
        self.job.ready()?;
        self.job.poisoned = true;
        self.job.verify_prefix(control)?;
        let report = describe(&self.job, control)?;
        reject_orphans(&report)?;
        if report.unacknowledged_output_present { self.job.verify_materialized(control)?; }
        checkpoint(control)?;
        self.report = report;
        self.job.poisoned = false;
        Ok(report)
    }

    /// Explicit publication of already committed results only, via the same
    /// owner-only, synced, no-replace output transaction as OwnedJob. Orphans
    /// require the original explicit recovery path; this method never deletes
    /// them. No destination or overwrite option is accepted.
    pub fn materialize_ordered<C: DecodeStepControl>(&mut self, control: &mut C)
        -> Result<StoredJobReport, JobError> {
        self.verify(control)?;
        self.job.materialize_ordered(control)?;
        // Failure during this final report must close the management session
        // even if publication committed: reopen to reconcile acknowledgement.
        self.job.poisoned = true;
        let report = describe(&self.job, control)?;
        self.report = report;
        self.job.poisoned = false;
        Ok(report)
    }
}
fn reject_orphans(report: &StoredJobReport) -> Result<(), JobError> {
    if report.uncommitted_spool_bytes != 0 || report.staged_output_present {
        return Err(JobError::UncommittedTail);
    }
    Ok(())
}
fn describe<C: DecodeStepControl>(job: &OwnedJob, control: &mut C) -> Result<StoredJobReport, JobError> {
    checkpoint(control)?;
    let progress = job.progress();
    let length = job.files.spool.metadata().map_err(|_| JobError::Io)?.len();
    let tail = length.checked_sub(progress.spool_bytes).ok_or(JobError::Corrupt)?;
    let staged = job.files.has_stages()?;
    let published = job.files.materialized(job.manifest.limits.max_materialized_bytes)?.is_some();
    checkpoint(control)?;
    Ok(StoredJobReport { schema_version: 1,
        verification_scope: "authenticated-stored-state-not-input-replay-v1",
        job_id: progress.job_id, items: progress.items, committed: progress.committed,
        attempts: progress.attempts, reserved_work: progress.reserved_work,
        committed_spool_bytes: progress.spool_bytes, uncommitted_spool_bytes: tail,
        staged_output_present: staged, materialized: progress.materialized,
        unacknowledged_output_present: published && !progress.materialized })
}

#[cfg(test)] mod tests;
