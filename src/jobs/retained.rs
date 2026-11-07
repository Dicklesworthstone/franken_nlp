//! Exact envelope retention with a separately admitted, immutable disk cap.
//! The authenticated journal receipt is the only authority for this spool.
//! No frame, filename, or stored input can authorize native work by itself.
use super::{Commitment, FrozenManifest, JobError, JobInput, JobLimits, JobSecret,
    checkpoint, manifest::Binding, population::envelope_id};
use crate::native_engine::decode::DecodeStepControl;
use serde::{Deserialize, Serialize};
use std::io::{Read, Seek, SeekFrom, Write};

pub(super) const HEADER_BYTES: usize = 56;
const MAGIC: &[u8; 8] = b"FNLPINP1";

/// Explicit consent and byte ceiling for private input retention. This does
/// not change JobLimits, the native recipe, or their existing commitments.
/// On replay it is a current read ceiling; tightening it is allowed only when
/// the complete authenticated spool fits. The original creation cap remains
/// sealed in the receipt and can never be expanded by a later caller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InputRetentionLimits { pub max_spool_bytes: u64 }
impl InputRetentionLimits {
    pub fn validate(self) -> Result<(), JobError> {
        if self.max_spool_bytes == 0 || self.max_spool_bytes > i64::MAX as u64 {
            return Err(JobError::InvalidLimits);
        }
        Ok(())
    }
}

/// Content-free status for an independently authenticated retained snapshot.
/// This is storage evidence, never a model execution or quality receipt.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RetainedInputReport {
    pub items: u64,
    pub spool_bytes: u64,
    pub max_spool_bytes: u64,
}

#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct InputReceipt {
    pub schema_version: u32,
    pub items: u64,
    pub spool_bytes: u64,
    pub max_spool_bytes: u64,
    pub commitment: Commitment,
}
impl InputReceipt {
    pub fn report(&self) -> RetainedInputReport {
        RetainedInputReport { items: self.items, spool_bytes: self.spool_bytes,
            max_spool_bytes: self.max_spool_bytes }
    }
    pub fn validate(&self, limits: JobLimits, expected_items: u64) -> Result<(), JobError> {
        InputRetentionLimits { max_spool_bytes: self.max_spool_bytes }.validate()
            .map_err(|_| JobError::Corrupt)?;
        let minimum = self.items.checked_mul(HEADER_BYTES as u64 + 1).ok_or(JobError::Corrupt)?;
        if self.schema_version != 1 || self.items == 0 || self.items != expected_items
            || self.items > limits.max_items || self.items > limits.max_attempts
            || self.spool_bytes < minimum || self.spool_bytes > self.max_spool_bytes
            || self.spool_bytes > storage_ceiling(limits)? {
            return Err(JobError::Corrupt);
        }
        Ok(())
    }
}

/// A mathematical upper bound used only to inspect an unreceipted file. It
/// grants no write/retention consent and no authority to adopt an orphan.
pub(super) fn storage_ceiling(limits: JobLimits) -> Result<u64, JobError> {
    limits.max_items.checked_mul(HEADER_BYTES as u64)
        .and_then(|n| n.checked_add(limits.max_snapshot_bytes / 2))
        .map(|n| n.min(i64::MAX as u64)).ok_or(JobError::Limit)
}

/// Validate the complete write size before creating any job files. There is
/// exactly one copy of each identity-normalized envelope on disk.
pub(super) fn preflight<C: DecodeStepControl>(key: &JobSecret, manifest: &FrozenManifest,
    limits: InputRetentionLimits, inputs: &[JobInput<'_>], control: &mut C) -> Result<u64, JobError> {
    limits.validate()?;
    if inputs.len() as u64 != manifest.item_count() { return Err(JobError::InvalidInput); }
    let mut length = 0_u64;
    for (ordinal, input) in inputs.iter().enumerate() {
        checkpoint(control)?;
        if input.original != input.normalized || input.original.is_empty() {
            return Err(JobError::InvalidInput);
        }
        if envelope_id(input.original, manifest.limits)? != input.id { return Err(JobError::InvalidInput); }
        manifest.verify_input(key, ordinal as u64, input)?;
        length = length.checked_add(HEADER_BYTES as u64)
            .and_then(|n| n.checked_add(input.original.len() as u64))
            .filter(|&n| n <= limits.max_spool_bytes).ok_or(JobError::Limit)?;
    }
    Ok(length)
}

fn start(key: &JobSecret, binding: &Binding, cap: u64) -> Commitment {
    key.commit(b"retained-input-start-v1", &[&binding.job.0, &binding.population.0,
        &binding.execution.0, &binding.recipe.0, &binding.limits.0, &cap.to_le_bytes()])
}
fn next(key: &JobSecret, prior: Commitment, ordinal: u64, offset: u64, bytes: &[u8]) -> Commitment {
    key.commit(b"retained-input-next-v1", &[&prior.0, &ordinal.to_le_bytes(),
        &offset.to_le_bytes(), bytes])
}
fn finish(key: &JobSecret, prior: Commitment, items: u64, length: u64) -> Commitment {
    key.commit(b"retained-input-end-v1", &[&prior.0, &items.to_le_bytes(), &length.to_le_bytes()])
}

/// The caller must sync this file and its parent before journaling the returned
/// receipt. A write result alone never acknowledges retained durable input.
pub(super) fn write<W: Write + Seek, C: DecodeStepControl>(writer: &mut W, key: &JobSecret,
    manifest: &FrozenManifest, limits: InputRetentionLimits, inputs: &[JobInput<'_>],
    control: &mut C) -> Result<InputReceipt, JobError> {
    let expected = preflight(key, manifest, limits, inputs, control)?;
    if writer.seek(SeekFrom::End(0)).map_err(|_| JobError::Io)? != 0 { return Err(JobError::AlreadyExists); }
    let mut chain = start(key, &manifest.binding, limits.max_spool_bytes);
    let mut offset = 0_u64;
    for (ordinal, input) in inputs.iter().enumerate() {
        checkpoint(control)?;
        chain = next(key, chain, ordinal as u64, offset, input.original);
        let mut header = [0_u8; HEADER_BYTES];
        header[..8].copy_from_slice(MAGIC);
        header[8..16].copy_from_slice(&(ordinal as u64).to_le_bytes());
        header[16..24].copy_from_slice(&(input.original.len() as u64).to_le_bytes());
        header[24..].copy_from_slice(&chain.0);
        writer.write_all(&header).map_err(|_| JobError::Io)?;
        for chunk in input.original.chunks(8192) {
            checkpoint(control)?;
            writer.write_all(chunk).map_err(|_| JobError::Io)?;
        }
        offset = offset.checked_add(HEADER_BYTES as u64)
            .and_then(|n| n.checked_add(input.original.len() as u64)).ok_or(JobError::Limit)?;
    }
    if offset != expected { return Err(JobError::Corrupt); }
    checkpoint(control)?;
    Ok(InputReceipt { schema_version: 1, items: manifest.item_count(), spool_bytes: offset,
        max_spool_bytes: limits.max_spool_bytes,
        commitment: finish(key, chain, manifest.item_count(), offset) })
}

/// Validate framing, every envelope, every keyed item, and the final chain.
/// Only one bounded frame and its strict JSON parser are transient. A visitor
/// may retain that owned buffer; no additional population-sized copy exists.
/// Visitors must discard all state if the final receipt check fails.
pub(super) fn read<R: Read + Seek, C: DecodeStepControl>(reader: &mut R, key: &JobSecret,
    manifest: &FrozenManifest, receipt: &InputReceipt, control: &mut C,
    mut visit: impl FnMut(String, Vec<u8>) -> Result<(), JobError>) -> Result<(), JobError> {
    receipt.validate(manifest.limits, manifest.item_count())?;
    if reader.seek(SeekFrom::End(0)).map_err(|_| JobError::Io)? != receipt.spool_bytes {
        return Err(JobError::Corrupt);
    }
    reader.seek(SeekFrom::Start(0)).map_err(|_| JobError::Io)?;
    let mut chain = start(key, &manifest.binding, receipt.max_spool_bytes);
    let mut offset = 0_u64;
    let mut snapshot_bytes = 0_u64;
    for ordinal in 0..receipt.items {
        checkpoint(control)?;
        let mut header = [0_u8; HEADER_BYTES];
        reader.read_exact(&mut header).map_err(|_| JobError::Corrupt)?;
        let length = u64::from_le_bytes(header[16..24].try_into().map_err(|_| JobError::Corrupt)?);
        let end = offset.checked_add(HEADER_BYTES as u64).and_then(|n| n.checked_add(length))
            .filter(|&n| n <= receipt.spool_bytes).ok_or(JobError::Corrupt)?;
        let next_snapshot = length.checked_mul(2).and_then(|n| n.checked_add(snapshot_bytes))
            .filter(|&n| n <= manifest.limits.max_snapshot_bytes).ok_or(JobError::Limit)?;
        if &header[..8] != MAGIC || header[8..16] != ordinal.to_le_bytes()
            || length == 0 || length > manifest.limits.max_input_bytes_per_item as u64 {
            return Err(JobError::Corrupt);
        }
        let n = usize::try_from(length).map_err(|_| JobError::Limit)?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(n).map_err(|_| JobError::Allocation)?;
        bytes.resize(n, 0);
        for chunk in bytes.chunks_mut(8192) {
            checkpoint(control)?;
            reader.read_exact(chunk).map_err(|_| JobError::Corrupt)?;
        }
        let actual = next(key, chain, ordinal, offset, &bytes);
        let expected = Commitment(header[24..].try_into().map_err(|_| JobError::Corrupt)?);
        if !actual.matches(expected) { return Err(JobError::Authentication); }
        let id = envelope_id(&bytes, manifest.limits)?;
        snapshot_bytes = next_snapshot.checked_add(id.len() as u64)
            .filter(|&n| n <= manifest.limits.max_snapshot_bytes).ok_or(JobError::Limit)?;
        manifest.verify_input(key, ordinal, &JobInput { id: &id, original: &bytes, normalized: &bytes })?;
        visit(id, bytes)?;
        chain = actual;
        offset = end;
    }
    if offset != receipt.spool_bytes || !finish(key, chain, receipt.items, offset).matches(receipt.commitment) {
        return Err(JobError::Authentication);
    }
    checkpoint(control)
}

#[cfg(test)] mod tests;
