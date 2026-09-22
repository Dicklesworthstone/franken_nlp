//! A journal pointer is the only authority to read a frame. Valid-looking
//! bytes in an uncommitted tail are never promoted to committed output.
use super::{Commitment, JobError, JobSecret, manifest::{Binding, ItemBinding, bounded_json}};
use serde::{Deserialize, Serialize};
use std::io::{Read, Seek, SeekFrom, Write};

pub(super) const HEADER_BYTES: usize = 120;
const MAGIC: &[u8; 8] = b"FNLPJOB1";
pub(super) const GENERATION: u64 = 1;
#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Pointer {
    pub generation: u64, pub offset: u64, pub length: u64, pub commitment: Commitment,
}
impl Pointer {
    pub fn end(&self) -> Result<u64, JobError> { self.offset.checked_add(self.length).ok_or(JobError::Corrupt) }
}
fn result_key(key: &JobSecret, binding: &Binding, item: &ItemBinding, offset: u64, bytes: &[u8]) -> Commitment {
    key.commit(b"result-frame", &[&binding.job.0, &binding.population.0, &binding.execution.0,
        &binding.recipe.0, &binding.limits.0, &item.id.0, &item.original.0, &item.normalized.0,
        &item.ordinal.to_le_bytes(), &GENERATION.to_le_bytes(), &offset.to_le_bytes(), bytes])
}
pub(super) fn append<T: Serialize, W: Write + Seek>(writer: &mut W, key: &JobSecret,
    binding: &Binding, item: &ItemBinding, offset: u64, value: &T, max_result: usize, max_spool: u64)
    -> Result<Pointer, JobError> {
    let bytes = bounded_json(value, max_result)?;
    let length = (HEADER_BYTES as u64).checked_add(bytes.len() as u64).ok_or(JobError::Limit)?;
    offset.checked_add(length).filter(|&n| n <= max_spool).ok_or(JobError::Limit)?;
    let commitment = result_key(key, binding, item, offset, &bytes);
    let mut header = [0_u8; HEADER_BYTES];
    header[..8].copy_from_slice(MAGIC);
    header[8..40].copy_from_slice(&binding.population.0);
    header[40..72].copy_from_slice(&item.id.0);
    header[72..80].copy_from_slice(&item.ordinal.to_le_bytes());
    header[80..88].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
    header[88..120].copy_from_slice(&commitment.0);
    if writer.seek(SeekFrom::End(0)).map_err(|_| JobError::Io)? != offset { return Err(JobError::UncommittedTail); }
    writer.write_all(&header).and_then(|_| writer.write_all(&bytes)).map_err(|_| JobError::Io)?;
    // Deliberately no receipt here: the owner must sync the file, then commit
    // the pointer in its database before acknowledging anything.
    Ok(Pointer { generation: GENERATION, offset, length, commitment })
}
pub(super) fn read<R: Read + Seek>(reader: &mut R, key: &JobSecret, binding: &Binding,
    item: &ItemBinding, pointer: &Pointer, max_result: usize) -> Result<Vec<u8>, JobError> {
    if pointer.generation != GENERATION || pointer.length <= HEADER_BYTES as u64
        || pointer.length - HEADER_BYTES as u64 > max_result as u64 { return Err(JobError::Corrupt); }
    let mut header = [0_u8; HEADER_BYTES];
    reader.seek(SeekFrom::Start(pointer.offset)).and_then(|_| reader.read_exact(&mut header)).map_err(|_| JobError::Corrupt)?;
    let length = pointer.length - HEADER_BYTES as u64;
    if &header[..8] != MAGIC || header[8..40] != binding.population.0 || header[40..72] != item.id.0
        || header[72..80] != item.ordinal.to_le_bytes() || header[80..88] != length.to_le_bytes()
        || header[88..120] != pointer.commitment.0 { return Err(JobError::Corrupt); }
    let n = usize::try_from(length).map_err(|_| JobError::Limit)?;
    let mut bytes = Vec::new(); bytes.try_reserve_exact(n).map_err(|_| JobError::Allocation)?; bytes.resize(n, 0);
    reader.read_exact(&mut bytes).map_err(|_| JobError::Corrupt)?;
    if !result_key(key, binding, item, pointer.offset, &bytes).matches(pointer.commitment) { return Err(JobError::Authentication); }
    Ok(bytes)
}
