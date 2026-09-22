//! A bounded complete-population freeze, not the live batch epoch ID window.
use super::{Commitment, JobError, JobId, JobSecret, MismatchField};
use crate::{canonjson, execution_identity::ExecutionIdentity, native_engine::strict_int8::Int8Work};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, io::Write};

/// All native work axes plus grammar-mask work. These are nonrenewable work
/// ceilings, never deserializable memory permits or inference receipts.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JobWork { pub model: Int8Work, pub mask_node_visits: u64 }
impl JobWork {
    pub fn checked_add(self, other: Self) -> Result<Self, JobError> {
        Ok(Self { model: self.model.checked_add(other.model).map_err(|_| JobError::WorkLimit)?,
            mask_node_visits: self.mask_node_visits.checked_add(other.mask_node_visits).ok_or(JobError::WorkLimit)? })
    }
    pub fn fits(self, limit: Self) -> bool {
        self.model.forward_positions <= limit.model.forward_positions
            && self.model.projected_logits <= limit.model.projected_logits
            && self.model.attention_pairs <= limit.model.attention_pairs
            && self.model.projections.fits(limit.model.projections)
            && self.mask_node_visits <= limit.mask_node_visits
    }
}

/// Immutable across resume, including retry and storage ceilings. This first
/// implementation deliberately bounds the in-memory manifest index; larger
/// populations are rejected, never silently reduced to epoch-local uniqueness.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JobLimits {
    pub max_items: u64,
    pub max_id_bytes: usize,
    pub max_input_bytes_per_item: usize,
    pub max_snapshot_bytes: u64,
    pub max_result_bytes: usize,
    pub max_spool_bytes: u64,
    pub max_materialized_bytes: u64,
    /// Database page budget, rounded down to whole 4096-byte pages. Rollback
    /// journal allowance is additional: at most twice this ceiling plus 64 KiB.
    pub max_journal_bytes: u64,
    pub max_attempts: u64,
    pub max_work: JobWork,
}
impl JobLimits {
    pub fn validate(self) -> Result<(), JobError> {
        if !(1..=100_000).contains(&self.max_items) || !(1..=4096).contains(&self.max_id_bytes)
            || !(1..=64 * 1024 * 1024).contains(&self.max_input_bytes_per_item)
            || !(1..=64 * 1024 * 1024).contains(&self.max_result_bytes)
            || self.max_attempts == 0 || self.max_attempts > i64::MAX as u64
            || self.max_snapshot_bytes == 0 || self.max_snapshot_bytes > i64::MAX as u64
            || self.max_spool_bytes < super::frame::HEADER_BYTES as u64 + 1
            || self.max_spool_bytes > i64::MAX as u64 || self.max_materialized_bytes == 0
            || self.max_materialized_bytes > i64::MAX as u64
            || !(65_536..=4 * 1024 * 1024 * 1024_u64).contains(&self.max_journal_bytes) {
            return Err(JobError::InvalidLimits);
        }
        Ok(())
    }
}

/// The full item-local semantic recipe: include defaults, effective sampling
/// seed/address version, normalization, schema, dependency scopes and any other
/// meaning-affecting setting in `recipe`. Native adapters must still verify
/// their actual admitted ExecutionIdentity. This is not a task factory.
pub struct JobContract<'a, R> {
    pub job_id: JobId,
    pub execution: &'a ExecutionIdentity,
    pub recipe: &'a R,
    pub limits: JobLimits,
}

/// Exact replay input. `normalized` is the host's explicitly selected
/// representation (the original bytes for identity normalization). Neither is
/// stored by this module. IDs must be unique across the complete population.
pub struct JobInput<'a> { pub id: &'a str, pub original: &'a [u8], pub normalized: &'a [u8] }

#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ItemBinding {
    pub ordinal: u64, pub id: Commitment, pub original: Commitment, pub normalized: Commitment,
}
#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Binding {
    pub job: JobId, pub secret: Commitment, pub execution: Commitment, pub recipe: Commitment,
    pub population: Commitment, pub limits: Commitment,
}
impl Binding {
    pub fn compare(&self, other: &Self) -> Result<(), JobError> {
        let field = if self.job != other.job { Some(MismatchField::Job) }
            else if !self.secret.matches(other.secret) { Some(MismatchField::Secret) }
            else if !self.execution.matches(other.execution) { Some(MismatchField::Execution) }
            else if !self.recipe.matches(other.recipe) { Some(MismatchField::Recipe) }
            else if !self.population.matches(other.population) { Some(MismatchField::Population) }
            else if !self.limits.matches(other.limits) { Some(MismatchField::Limits) } else { None };
        field.map_or(Ok(()), |f| Err(JobError::Mismatch(f)))
    }
}

/// No Deserialize: recompute it from the complete original population before
/// every create/resume. It retains only bounded keyed metadata, never text.
pub struct FrozenManifest {
    pub(super) binding: Binding,
    pub(super) items: Vec<ItemBinding>,
    pub(super) limits: JobLimits,
}
impl FrozenManifest {
    pub fn freeze<'a, R: Serialize, C: crate::native_engine::decode::DecodeStepControl>(
        key: &JobSecret, contract: JobContract<'_, R>, inputs: impl IntoIterator<Item = JobInput<'a>>,
        control: &mut C) -> Result<Self, JobError> {
        super::checkpoint(control)?;
        contract.limits.validate()?;
        contract.execution.validate().map_err(|_| JobError::InvalidIdentity)?;
        let execution = bounded_json(contract.execution, 64 * 1024)?;
        let recipe = bounded_json(contract.recipe, 1024 * 1024)?;
        let limits = bounded_json(&contract.limits, 16 * 1024)?;
        let job = &contract.job_id.0;
        let mut population = key.commit(b"population-start", &[job]);
        let mut items = Vec::new(); let mut seen = BTreeSet::new(); let mut bytes = 0_u64;
        for input in inputs {
            super::checkpoint(control)?;
            if items.len() as u64 >= contract.limits.max_items { return Err(JobError::Limit); }
            let ordinal = items.len() as u64;
            let item = bind_input(key, contract.job_id, ordinal, &input, contract.limits)?;
            // Keyed, fixed-size IDs keep private identifier strings out of the
            // retained index. Hash collision fails closed as a duplicate.
            if !seen.insert(item.id.0) { return Err(JobError::DuplicateId); }
            bytes = bytes.checked_add(input.id.len() as u64).and_then(|n| n.checked_add(input.original.len() as u64))
                .and_then(|n| n.checked_add(input.normalized.len() as u64)).filter(|&n| n <= contract.limits.max_snapshot_bytes)
                .ok_or(JobError::Limit)?;
            let encoded = bounded_json(&item, 4096)?;
            population = key.commit(b"population-next", &[job, &population.0, &encoded]);
            items.try_reserve(1).map_err(|_| JobError::Allocation)?; items.push(item);
        }
        if items.is_empty() || items.len() as u64 > contract.limits.max_attempts { return Err(JobError::InvalidLimits); }
        population = key.commit(b"population-end", &[job, &population.0, &(items.len() as u64).to_le_bytes()]);
        Ok(Self { binding: Binding { job: contract.job_id, secret: key.commit(b"key-check", &[job]),
            execution: key.commit(b"execution", &[job, &execution]), recipe: key.commit(b"recipe", &[job, &recipe]),
            population, limits: key.commit(b"limits", &[job, &limits]) }, items, limits: contract.limits })
    }
    pub fn item_count(&self) -> u64 { self.items.len() as u64 }
    pub fn job_id(&self) -> JobId { self.binding.job }
    pub fn population_commitment(&self) -> Commitment { self.binding.population }
    pub fn limits(&self) -> JobLimits { self.limits }
    pub(super) fn verify_input(&self, key: &JobSecret, ordinal: u64, input: &JobInput<'_>) -> Result<(), JobError> {
        let expected = self.items.get(usize::try_from(ordinal).map_err(|_| JobError::Limit)?).ok_or(JobError::Finished)?;
        let actual = bind_input(key, self.binding.job, ordinal, input, self.limits)?;
        if &actual != expected { return Err(JobError::Mismatch(MismatchField::Population)); }
        Ok(())
    }
}
fn bind_input(key: &JobSecret, job: JobId, ordinal: u64, input: &JobInput<'_>, limits: JobLimits)
    -> Result<ItemBinding, JobError> {
    if input.id.is_empty() || input.id.len() > limits.max_id_bytes
        || input.original.len() > limits.max_input_bytes_per_item || input.normalized.len() > limits.max_input_bytes_per_item {
        return Err(JobError::InvalidInput);
    }
    let id = key.commit(b"item-id", &[&job.0, input.id.as_bytes()]);
    Ok(ItemBinding { ordinal, id,
        original: key.commit(b"input-original", &[&job.0, &id.0, input.original]),
        normalized: key.commit(b"input-normalized", &[&job.0, &id.0, input.normalized]) })
}

/// Count before canonical tree allocation. Serialize the original value after
/// sizing so canonical nonfinite rejection is not bypassed by serde's nulls.
pub(super) fn bounded_json<T: Serialize + ?Sized>(value: &T, max: usize) -> Result<Vec<u8>, JobError> {
    struct Counter { left: usize, limit: bool }
    impl Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > self.left { self.limit = true; return Err(std::io::Error::other("job output bound")); }
            self.left -= bytes.len(); Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    let mut counter = Counter { left: max, limit: false };
    serde_json::to_writer(&mut counter, value).map_err(|_| if counter.limit { JobError::Limit } else { JobError::Serialization })?;
    let bytes = canonjson::canonical_bytes(value).map_err(|_| JobError::Serialization)?;
    if bytes.len() > max { return Err(JobError::Limit); }
    Ok(bytes)
}
