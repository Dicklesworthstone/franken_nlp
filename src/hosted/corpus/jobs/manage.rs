//! No-model job management through the existing process-owned runtime/ledger.
use super::*;
use crate::jobs::{StoredJob, StoredJobReport};
use serde::Serialize;

/// Fixed operations over stored data. There is no execution/recovery variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredJobOperation { Status, Verify, MaterializeOrdered }

/// Explicit access to existing private result storage. No original corpus or
/// recipe is supplied, and no new native work can be authorized by this type.
pub struct StoredJobRequest {
    pub root: PathBuf,
    pub key: JobSecret,
    pub job_id: JobId,
    pub limits: JobLimits,
    pub operation: StoredJobOperation,
}

/// Per-invocation modeled RAM reservations, separate from immutable JobLimits.
/// Database file size is not a database-engine RAM bound; the host prices that
/// engine and allocator overhead explicitly. No native workspace is requested.
#[derive(Clone, Copy, Debug)]
pub struct JobManagementLimits {
    pub run: RunLimits,
    pub journal_reserve_bytes: u64,
    pub serialization_reserve_bytes: u64,
    pub io_reserve_bytes: u64,
}
impl JobManagementLimits {
    fn reservation_bytes(self, job: JobLimits) -> Result<u64, HostedJobError> {
        self.run.validate()?; job.validate()?;
        if self.journal_reserve_bytes == 0 || self.serialization_reserve_bytes == 0 || self.io_reserve_bytes == 0 {
            return Err(HostedError::Limits("explicit stored-job memory reservations required").into());
        }
        let index = job.max_items.checked_mul(1024).ok_or(HostedError::Limits("stored-job index arithmetic"))?;
        let frames = (job.max_result_bytes as u64).checked_mul(4)
            .ok_or(HostedError::Limits("stored-job frame arithmetic"))?;
        Ok(sum(&[index, frames, 2 * 1024 * 1024, self.journal_reserve_bytes,
            self.serialization_reserve_bytes, self.io_reserve_bytes])?)
    }
}
impl NlpEngine {
    /// Authenticate status, verify, or publish stored results without loading a
    /// model or reading original inputs. Key, locked journal/spool, preparation
    /// and IO remain owned until the existing physical-completion handoff.
    pub fn manage_owned_job(&self, request: StoredJobRequest, limits: JobManagementLimits,
        cancellation: CancellationToken) -> Result<HostedOutput<StoredJobReport>, HostedJobError> {
        dispatch::preflight(self, limits.run)?;
        let bytes = sum(&[limits.reservation_bytes(request.limits)?, request.root.capacity() as u64])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?, || Ok(Some(request)))?;
        // The report is fixed metadata, not retained result content. Its own
        // charge survives the journal/input cleanup and caller serialization.
        let output = output_claim(&lease, 16 * 1024, 0)?;
        let completed = dispatch::run(self, limits.run, cancellation, move |control| {
            let mut input = input; // Whole aggregate, including queued drop order.
            let request = input.value.take().ok_or(HostedError::CompletionMissing)?;
            let result = perform(request, control);
            let result = match result {
                Ok(report) => {
                    let charged = allocate(output, || Ok(report))?;
                    Ok(GuardedOutput::new(charged.value, charged._memory))
                }
                Err(error) => { drop(output); Err(error) }
            };
            drop(input);
            drop(lease);
            Ok(result)
        })?;
        completed.map_err(HostedJobError::from)
    }
}
fn perform<C: DecodeStepControl>(request: StoredJobRequest, control: &mut C) -> Result<StoredJobReport, JobError> {
    let StoredJobRequest { root, key, job_id, limits, operation } = request;
    let mut job = StoredJob::open(&root, key, job_id, limits, control)?;
    match operation {
        StoredJobOperation::Status => job.status(),
        StoredJobOperation::Verify => job.verify(control),
        StoredJobOperation::MaterializeOrdered => job.materialize_ordered(control),
    }
    // The management owner closes its journal/spool/lock and drops the key
    // before returning metadata; no live storage authority crosses dispatch.
}

#[cfg(test)] mod tests;
