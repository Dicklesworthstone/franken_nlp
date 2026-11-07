//! Retained replay shares the original exclusive owner for the entire
//! authenticate → refreeze → result-recovery → execution sequence. No unlocked
//! snapshot, stored-hash manifest, or mutable file handle escapes this module.
use super::*;
use crate::jobs::{JobLimits, manifest::ManifestBuilder, retained::RetainedInputCursor};
use std::fs::File;

impl OwnedJob {
    /// A fresh builder binds the real processor identity/recipe; the private
    /// validator parses every actual envelope into that processor's argument
    /// type. Neither receives a raw file or stored-hash execution authority.
    /// Both finish before recovery, under the returned owner's existing lock.
    pub(in crate::jobs) fn resume_with_retained_inputs<C, E>(root: &Path, key: JobSecret,
        expected_job: JobId, limits: JobLimits, retention: InputRetentionLimits,
        tail: TailPolicy, mut fresh: ManifestBuilder, control: &mut C,
        mut validate: impl FnMut(&JobInput<'_>, &mut C) -> Result<(), E>)
        -> Result<(Self, RetainedInputCursor<File>), E>
    where C: DecodeStepControl, E: From<JobError> {
        checkpoint(control)?;
        retention.validate()?;
        let mut job = stored::load_metadata(root, key, expected_job, limits, control)?;
        let receipt = job.header.retained_inputs.as_ref().ok_or(JobError::InputsNotRetained)?;
        receipt.validate(limits, job.header.items)?;
        if receipt.spool_bytes > retention.max_spool_bytes { return Err(JobError::Limit.into()); }
        let read_cap = retention.max_spool_bytes.min(receipt.max_spool_bytes);
        let file = job.files.retained_inputs(read_cap)?.ok_or(JobError::Corrupt)?;
        let mut cursor = RetainedInputCursor::new(file, &job.key, &job.manifest, receipt, control)?;
        let mut pending = None;
        loop {
            if cursor.ordinal() == job.header.committed { pending = Some(cursor.bookmark()?); }
            let Some(input) = cursor.read_next(&job.key, &job.manifest, control)? else { break; };
            validate(&input, control)?;
            fresh.push(&job.key, &input, control)?;
        }
        let manifest = fresh.finish(&job.key)?;
        // This fresh freeze is the sole transition from authenticated storage
        // metadata to current execution authority. Recipe, identity, limits,
        // population and key must all agree before any result repair.
        check_secret(&job.key, &manifest)?;
        manifest.binding.compare(&job.header.binding)?;
        cursor.replay_from(pending.ok_or(JobError::Corrupt)?, control)?;
        job.manifest = manifest;
        job.recover(tail, control)?;
        Ok((job, cursor))
    }

    /// The cursor lends one exact record through native execution and result
    /// commit. Its borrow is tied only to the cursor; the job remains available
    /// to persist the attempt. Any failed read poisons this owner as well as
    /// the cursor, so there is no recovery or retry inside the same session.
    pub(in crate::jobs) fn read_retained_input<'a, C: DecodeStepControl>(&mut self,
        cursor: &'a mut RetainedInputCursor<File>, control: &mut C) -> Result<JobInput<'a>, JobError> {
        self.ready()?;
        self.poisoned = true;
        if cursor.ordinal() != self.header.committed { return Err(JobError::InvalidTransition); }
        let input = cursor.read_next(&self.key, &self.manifest, control)?.ok_or(JobError::Finished)?;
        self.poisoned = false;
        Ok(input)
    }

    /// Streaming storage verification; at most one input record plus strict
    /// JSON-parser storage is transient. Never construct a whole population
    /// during status/verify/materialize. Returns orphan presence separately so
    /// management status can report even an empty unreceipted file honestly.
    pub(super) fn check_retained_inputs<C: DecodeStepControl>(&self, control: &mut C) -> Result<bool, JobError> {
        checkpoint(control)?;
        match &self.header.retained_inputs {
            Some(receipt) => {
                receipt.validate(self.manifest.limits, self.header.items)?;
                let mut file = self.files.retained_inputs(receipt.max_spool_bytes)?.ok_or(JobError::Corrupt)?;
                retained::read(&mut file, &self.key, &self.manifest, receipt, control, |_| Ok(()))?;
                Ok(false)
            },
            None => Ok(self.files.retained_inputs(retained::storage_ceiling(self.manifest.limits)?)?.is_some()),
        }
    }
    pub(super) fn verify_retained_inputs<C: DecodeStepControl>(&self, control: &mut C) -> Result<(), JobError> {
        if self.check_retained_inputs(control)? { return Err(JobError::UncommittedTail); }
        Ok(())
    }
}
