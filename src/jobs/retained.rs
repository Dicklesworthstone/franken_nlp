//! Exact envelope retention with a separately admitted, immutable disk cap.
//! The authenticated journal receipt is the only authority for this spool.
//! No frame, filename, or stored input can authorize native work by itself.
use super::{Commitment, FrozenManifest, JobError, JobInput, JobLimits, JobSecret,
    checkpoint, manifest::Binding, population::envelope_id};
use crate::native_engine::decode::DecodeStepControl;
use serde::{Deserialize, Serialize};
use std::io::{Read, Seek, SeekFrom, Write};

pub(super) const HEADER_BYTES: usize = 56;
pub(super) const MAGIC: &[u8; 8] = b"FNLPINP1";

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

pub(super) fn start(key: &JobSecret, binding: &Binding, cap: u64) -> Commitment {
    key.commit(b"retained-input-start-v1", &[&binding.job.0, &binding.population.0,
        &binding.execution.0, &binding.recipe.0, &binding.limits.0, &cap.to_le_bytes()])
}
pub(super) fn next(key: &JobSecret, prior: Commitment, ordinal: u64, offset: u64, bytes: &[u8]) -> Commitment {
    key.commit(b"retained-input-next-v1", &[&prior.0, &ordinal.to_le_bytes(),
        &offset.to_le_bytes(), bytes])
}
pub(super) fn finish(key: &JobSecret, prior: Commitment, items: u64, length: u64) -> Commitment {
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

/// Opaque position from this exact receipt. A bookmark is useful only after
/// its owner has verified the WHOLE spool; it is never independent authority.
#[derive(Clone, Copy)]
pub(super) struct InputBookmark {
    receipt: Commitment,
    ordinal: u64,
    offset: u64,
    snapshot_bytes: u64,
    chain: Commitment,
}

/// One reusable input buffer and one sequential file handle. Lending the
/// current envelope prevents another read while planning/execution still uses
/// its bytes. No input population, frame-offset index, or caller-controlled
/// seek handle escapes. The enclosing runner drops this owner before its job
/// lock. Hosts price the maximum record plus strict parser storage separately
/// from the bounded keyed manifest/uniqueness index.
pub(super) struct RetainedInputCursor<R> {
    reader: R,
    receipt: InputReceipt,
    binding: Binding,
    limits: JobLimits,
    ordinal: u64,
    offset: u64,
    snapshot_bytes: u64,
    chain: Commitment,
    bytes: Vec<u8>,
    id: String,
    whole_spool_verified: bool,
    poisoned: bool,
}
impl<R: Read + Seek> RetainedInputCursor<R> {
    pub(super) fn new<C: DecodeStepControl>(mut reader: R, key: &JobSecret,
        manifest: &FrozenManifest, receipt: &InputReceipt, control: &mut C) -> Result<Self, JobError> {
        checkpoint(control)?;
        receipt.validate(manifest.limits, manifest.item_count())?;
        if reader.seek(SeekFrom::End(0)).map_err(|_| JobError::Io)? != receipt.spool_bytes {
            return Err(JobError::Corrupt);
        }
        reader.seek(SeekFrom::Start(0)).map_err(|_| JobError::Io)?;
        Ok(Self { reader, receipt: receipt.clone(), binding: manifest.binding.clone(), limits: manifest.limits,
            ordinal: 0, offset: 0, snapshot_bytes: 0,
            chain: start(key, &manifest.binding, receipt.max_spool_bytes), bytes: Vec::new(), id: String::new(),
            whole_spool_verified: false, poisoned: false })
    }
    pub(super) fn ordinal(&self) -> u64 { self.ordinal }
    pub(super) fn bookmark(&self) -> Result<InputBookmark, JobError> {
        if self.poisoned { return Err(JobError::Poisoned); }
        Ok(InputBookmark { receipt: self.receipt.commitment, ordinal: self.ordinal,
            offset: self.offset, snapshot_bytes: self.snapshot_bytes, chain: self.chain })
    }
    /// Resume the same sequential chain at the next pending ordinal, without
    /// rereading committed inputs during execution. Full initial verification
    /// and a freshly reconstructed native contract must precede this call.
    pub(super) fn replay_from<C: DecodeStepControl>(&mut self, position: InputBookmark,
        control: &mut C) -> Result<(), JobError> {
        if self.poisoned { return Err(JobError::Poisoned); }
        self.poisoned = true;
        checkpoint(control)?;
        if !self.whole_spool_verified || !position.receipt.matches(self.receipt.commitment)
            || position.ordinal > self.receipt.items || position.offset > self.receipt.spool_bytes
            || position.snapshot_bytes > self.limits.max_snapshot_bytes
            || (position.ordinal == self.receipt.items) != (position.offset == self.receipt.spool_bytes) {
            return Err(JobError::InvalidTransition);
        }
        self.reader.seek(SeekFrom::Start(position.offset)).map_err(|_| JobError::Io)?;
        self.ordinal = position.ordinal;
        self.offset = position.offset;
        self.snapshot_bytes = position.snapshot_bytes;
        self.chain = position.chain;
        self.bytes.clear();
        self.id.clear();
        self.poisoned = false;
        Ok(())
    }
    pub(super) fn read_next<C: DecodeStepControl>(&mut self, key: &JobSecret,
        manifest: &FrozenManifest, control: &mut C) -> Result<Option<JobInput<'_>>, JobError> {
        if self.poisoned { return Err(JobError::Poisoned); }
        self.poisoned = true;
        checkpoint(control)?;
        manifest.binding.compare(&self.binding)?;
        if manifest.limits != self.limits { return Err(JobError::Mismatch(super::MismatchField::Limits)); }
        if self.ordinal == self.receipt.items {
            self.check_end(key)?;
            self.poisoned = false;
            return Ok(None);
        }
        let mut header = [0_u8; HEADER_BYTES];
        self.reader.read_exact(&mut header).map_err(|_| JobError::Corrupt)?;
        let length = u64::from_le_bytes(header[16..24].try_into().map_err(|_| JobError::Corrupt)?);
        let end = self.offset.checked_add(HEADER_BYTES as u64).and_then(|n| n.checked_add(length))
            .filter(|&n| n <= self.receipt.spool_bytes).ok_or(JobError::Corrupt)?;
        let next_snapshot = length.checked_mul(2).and_then(|n| n.checked_add(self.snapshot_bytes))
            .filter(|&n| n <= self.limits.max_snapshot_bytes).ok_or(JobError::Limit)?;
        if &header[..8] != MAGIC || header[8..16] != self.ordinal.to_le_bytes()
            || length == 0 || length > self.limits.max_input_bytes_per_item as u64 {
            return Err(JobError::Corrupt);
        }
        let n = usize::try_from(length).map_err(|_| JobError::Limit)?;
        self.id.clear();
        self.bytes.try_reserve_exact(n.saturating_sub(self.bytes.len())).map_err(|_| JobError::Allocation)?;
        self.bytes.resize(n, 0);
        for chunk in self.bytes.chunks_mut(8192) {
            checkpoint(control)?;
            self.reader.read_exact(chunk).map_err(|_| JobError::Corrupt)?;
        }
        let actual = next(key, self.chain, self.ordinal, self.offset, &self.bytes);
        let expected = Commitment(header[24..].try_into().map_err(|_| JobError::Corrupt)?);
        if !actual.matches(expected) { return Err(JobError::Authentication); }
        self.id = envelope_id(&self.bytes, self.limits)?;
        self.snapshot_bytes = next_snapshot.checked_add(self.id.len() as u64)
            .filter(|&n| n <= self.limits.max_snapshot_bytes).ok_or(JobError::Limit)?;
        manifest.verify_input(key, self.ordinal, &JobInput { id: &self.id,
            original: &self.bytes, normalized: &self.bytes })?;
        self.chain = actual;
        self.offset = end;
        self.ordinal += 1;
        // Do not release the terminal record before the end commitment agrees.
        if self.ordinal == self.receipt.items { self.check_end(key)?; }
        checkpoint(control)?;
        self.poisoned = false;
        Ok(Some(JobInput { id: &self.id, original: &self.bytes, normalized: &self.bytes }))
    }
    fn check_end(&mut self, key: &JobSecret) -> Result<(), JobError> {
        if self.ordinal != self.receipt.items || self.offset != self.receipt.spool_bytes
            || !finish(key, self.chain, self.receipt.items, self.offset).matches(self.receipt.commitment) {
            return Err(JobError::Authentication);
        }
        self.whole_spool_verified = true;
        Ok(())
    }
}

/// Authenticate the complete spool with one reusable record buffer. Visitors
/// borrow each exact envelope and must discard derived state on ANY failure.
pub(super) fn read<R: Read + Seek, C: DecodeStepControl>(reader: &mut R, key: &JobSecret,
    manifest: &FrozenManifest, receipt: &InputReceipt, control: &mut C,
    mut visit: impl FnMut(&JobInput<'_>) -> Result<(), JobError>) -> Result<(), JobError> {
    let mut cursor = RetainedInputCursor::new(reader, key, manifest, receipt, control)?;
    while let Some(input) = cursor.read_next(key, manifest, control)? { visit(&input)?; }
    Ok(())
}

#[cfg(test)] mod tests;
