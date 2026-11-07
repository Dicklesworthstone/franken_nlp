//! Input restoration shares the original exclusive owner for the entire
//! authenticate → restore → refreeze → result-recovery sequence. No unlocked
//! snapshot, stored-hash manifest, or mutable file handle escapes this module.
use super::*;
use crate::jobs::{JobLimits, population::JobPopulation};

impl OwnedJob {
    /// A private callback must freeze the real processor identity/recipe from
    /// these exact restored bytes. It runs before any recovery mutation, under
    /// the same lock that remains owned by the returned inference session.
    pub(in crate::jobs) fn resume_with_retained_inputs<C, E>(root: &Path, key: JobSecret,
        expected_job: JobId, limits: JobLimits, retention: InputRetentionLimits,
        tail: TailPolicy, control: &mut C,
        freeze: impl FnOnce(&JobSecret, &JobPopulation, &mut C) -> Result<FrozenManifest, E>)
        -> Result<(Self, JobPopulation), E>
    where C: DecodeStepControl, E: From<JobError> {
        checkpoint(control)?;
        retention.validate()?;
        let mut job = stored::load_metadata(root, key, expected_job, limits, control)?;
        let receipt = job.header.retained_inputs.as_ref().ok_or(JobError::InputsNotRetained)?;
        receipt.validate(limits, job.header.items)?;
        if receipt.spool_bytes > retention.max_spool_bytes { return Err(JobError::Limit.into()); }
        let read_cap = retention.max_spool_bytes.min(receipt.max_spool_bytes);
        let mut file = job.files.retained_inputs(read_cap)?.ok_or(JobError::Corrupt)?;
        let mut population = JobPopulation::retained();
        retained::read(&mut file, &job.key, &job.manifest, receipt, control,
            |id, bytes| population.push_retained(id, bytes, limits))?;
        drop(file);
        let manifest = freeze(&job.key, &population, control)?;
        // This fresh freeze is the sole transition from authenticated storage
        // metadata to current execution authority. Recipe, identity, limits,
        // population and key must all agree before any result repair.
        check_secret(&job.key, &manifest)?;
        manifest.binding.compare(&job.header.binding)?;
        job.manifest = manifest;
        job.recover(tail, control)?;
        Ok((job, population))
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
                retained::read(&mut file, &self.key, &self.manifest, receipt, control, |_, _| Ok(()))?;
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
