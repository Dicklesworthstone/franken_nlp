//! Opt-in, item-local owned jobs. Content never enters the metadata-only store.
//!
//! Freeze the full replay contract and ordered input population before opening
//! a job. An owned job reserves work durably before execution, syncs each result
//! frame before committing its journal pointer, and materializes only verified
//! committed pointers. It creates no runtime, worker, scheduler, or retry loop.
//! Arbitrary stdout is not an exactly-once destination. See
//! `docs/owned-jobs.md` for the platform, privacy and host-admission boundaries.

use std::{error::Error, fmt};
mod commitment;
mod manifest;
mod frame;
pub mod population;
pub use commitment::{Commitment, JobId, JobSecret};
pub use manifest::{FrozenManifest, JobContract, JobInput, JobLimits, JobWork};

#[cfg(all(feature = "metadata-store", target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
mod journal;
#[cfg(all(feature = "metadata-store", target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
mod owned;
#[cfg(all(feature = "metadata-store", target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use owned::{Attempt, JobProgress, OwnedJob, StoredJob, StoredJobReport, TailPolicy};
#[cfg(all(feature = "metadata-store", target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub mod runner;

/// No filesystem is touched merely by importing or constructing a manifest.
pub const OWNED_JOBS_AVAILABLE: bool = cfg!(all(feature = "metadata-store", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")));

/// Field names only: diagnostics never expose old/new private commitments.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MismatchField { Job, Secret, Execution, Recipe, Population, Limits }

/// Closed failures; no paths, SQL, content, keys, parser excerpts or private
/// digests are retained by Display, Debug, or an error-source chain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobError {
    InvalidLimits, InvalidIdentity, InvalidInput, DuplicateId, Limit, WorkLimit,
    Allocation, Serialization, Authentication, Mismatch(MismatchField),
    Platform, UnsafeStorage, Busy, AlreadyExists, Io, Journal, Corrupt,
    UncommittedTail, Incomplete, Finished, Poisoned, InvalidTransition,
    PublicationUncertain, Cancelled(crate::native_engine::decode::DecodeCancellationKind),
}
impl fmt::Display for JobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "owned job refused: {self:?}") }
}
impl Error for JobError {}

pub(super) fn checkpoint<C: crate::native_engine::decode::DecodeStepControl>(control: &mut C) -> Result<(), JobError> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(JobError::Cancelled(cause)), None => Ok(()) }
}

#[cfg(test)] mod tests;
