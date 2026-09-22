//! Domain-separated HMAC commitments. No unkeyed private-content digest is
//! persisted or exported. Public artifact identities retain their own contract.
use super::JobError;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Public, opaque job identity. Supply 128 fresh random bits from the host's
/// admitted entropy source; this constructor is not a randomness certificate.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct JobId(pub [u8; 16]);

/// Fixed-width keyed commitment. Deliberately no Debug or string formatting.
/// Serialization is used only in the owner-only journal and explicit receipts.
#[derive(Clone, Copy, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Commitment(pub(crate) [u8; 32]);
impl Commitment {
    pub(crate) fn matches(self, other: Self) -> bool {
        self.0.iter().zip(other.0.iter()).fold(0_u8, |diff, (a, b)| diff | (a ^ b)) == 0
    }
}

/// One caller-owned 256-bit secret per job. It is neither cloneable nor
/// serializable. The embedding host generates it once and persists it, when
/// requested, using its explicit protected-key policy. Never regenerate it on
/// resume. This type does not claim compiler-proof erasure of all copies.
pub struct JobSecret([u8; 32]);
impl JobSecret {
    pub fn from_bytes(bytes: [u8; 32]) -> Self { Self(bytes) }

    /// Read exactly 32 bytes using the existing private-parent/no-symlink key
    /// reader. Missing, wrong-sized and insecure key files fail closed.
    pub fn read(path: &std::path::Path) -> Result<Self, JobError> {
        use std::io::Read;
        let mut file = crate::local_io::open_key(path).map_err(|_| JobError::UnsafeStorage)?;
        let mut bytes = [0_u8; 32];
        file.read_exact(&mut bytes).map_err(|_| JobError::Io)?;
        let mut extra = [0_u8; 1];
        if file.read(&mut extra).map_err(|_| JobError::Io)? != 0 {
            bytes.fill(0); return Err(JobError::InvalidInput);
        }
        Ok(Self(bytes))
    }

    pub(crate) fn commit(&self, domain: &[u8], parts: &[&[u8]]) -> Commitment {
        let mut inner = hmac_inner(&self.0);
        inner.update(b"fnlp-owned-job-hmac-v1\0");
        inner.update((domain.len() as u64).to_le_bytes()); inner.update(domain);
        inner.update((parts.len() as u64).to_le_bytes());
        for part in parts { inner.update((part.len() as u64).to_le_bytes()); inner.update(part); }
        Commitment(hmac_finish(&self.0, inner))
    }
}
impl Drop for JobSecret { fn drop(&mut self) { self.0.fill(0); } }

fn key_block(key: &[u8]) -> [u8; 64] {
    let mut block = [0_u8; 64];
    if key.len() > block.len() { block[..32].copy_from_slice(&Sha256::digest(key)); }
    else { block[..key.len()].copy_from_slice(key); }
    block
}
fn hmac_inner(key: &[u8]) -> Sha256 {
    let mut block = key_block(key); for byte in &mut block { *byte ^= 0x36; }
    let mut hash = Sha256::new(); hash.update(block); block.fill(0); hash
}
fn hmac_finish(key: &[u8], inner: Sha256) -> [u8; 32] {
    let mut block = key_block(key); for byte in &mut block { *byte ^= 0x5c; }
    let mut outer = Sha256::new(); outer.update(block); outer.update(inner.finalize());
    block.fill(0); outer.finalize().into()
}
#[cfg(test)]
pub(super) fn rfc_hmac(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut inner = hmac_inner(key); inner.update(data); hmac_finish(key, inner)
}
