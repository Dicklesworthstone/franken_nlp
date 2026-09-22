//! Bounded replay contracts and authenticated framing for opt-in owned jobs.
//! Complete population freeze and private commitments create no runtime or
//! journal authority; durable storage is supplied by the owning job layer.

use std::{error::Error, fmt};
mod commitment;
mod manifest;
mod frame;
pub use commitment::{Commitment, JobId, JobSecret};
pub use manifest::{FrozenManifest, JobContract, JobInput, JobLimits, JobWork};

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
