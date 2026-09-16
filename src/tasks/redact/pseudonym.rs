//! HMAC-SHA-256 pseudonyms (RFC 2104 construction, RFC 4231 test vectors).
//! Caller-owned high-entropy keys never implement Debug/Clone/Serialize. A
//! streaming context uses all 256 bits. Short output requires a sealed complete
//! value-set preflight; an unseen value can never extend that context lazily.

use std::io::Read;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use super::{PiiKind, RedactError};

pub const PSEUDONYM_VERSION: &str = "fnlp-pseudonym-v1";
pub const CANONICALIZATION: &str = "exact-utf8-v1";
const MAX_KEY_BYTES: usize = 4096;
const MAX_VALUE_BYTES: usize = 1024 * 1024;

/// The caller must provide random key material from an authorized source;
/// checking byte length does not establish entropy. No CLI argv key surface.
pub struct PseudonymKey { block: [u8; 64], id: String }
impl PseudonymKey {
    pub fn from_bytes(bytes: &[u8], public_key_id: &str) -> Result<Self, RedactError> {
        if !(32..=MAX_KEY_BYTES).contains(&bytes.len()) || public_key_id.is_empty()
            || public_key_id.len() > 128 || !public_key_id.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)) {
            return Err(RedactError::InvalidOptions);
        }
        Ok(Self { block: key_block(bytes), id: public_key_id.to_owned() })
    }
    /// Bounded read from an already-authorized inherited descriptor or private
    /// stream. Opening files and establishing owner/ACL authority belong to
    /// the platform layer, not this method. Binary key bytes are not trimmed.
    pub fn from_reader<R: Read>(reader: &mut R, public_key_id: &str) -> Result<Self, RedactError> {
        let mut bytes = SecretInput([0; MAX_KEY_BYTES + 1]);
        let mut used = 0;
        loop {
            match reader.read(&mut bytes.0[used..]) {
                Ok(0) => break,
                Ok(n) if n <= bytes.0.len() - used => { used += n; if used == bytes.0.len() { return Err(RedactError::InvalidOptions); } }
                Ok(_) => return Err(RedactError::InvalidOptions),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => return Err(RedactError::MissingKey),
            }
        }
        Self::from_bytes(&bytes.0[..used], public_key_id)
    }
    pub fn commitment(&self) -> String {
        hex(&hmac_block(&self.block, |h| h.update(b"fnlp-pseudonym-key-commit-v1")))
    }
    /// Resume callers provide the saved key commitment, not merely its label.
    pub fn require_commitment(&self, expected: &str) -> Result<(), RedactError> {
        let actual = self.commitment();
        let equal = expected.len() == actual.len() && expected.bytes().zip(actual.bytes()).fold(0_u8, |x, (a, b)| x | (a ^ b)) == 0;
        if equal { Ok(()) } else { Err(RedactError::KeyMismatch) }
    }
}
// Best effort only: no claim of compiler-proof zeroization or debugger safety.
impl Drop for PseudonymKey { fn drop(&mut self) { self.block.fill(0); } }
struct SecretInput([u8; MAX_KEY_BYTES + 1]);
impl Drop for SecretInput { fn drop(&mut self) { self.0.fill(0); } }

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PseudonymEncoding { Full256, Preflighted128 }

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PseudonymIdentity {
    pub version: String,
    pub canonicalization: String,
    pub key_id: String,
    pub key_commitment: String,
    pub encoding: PseudonymEncoding,
    /// Keyed binding of namespace and, in 128-bit mode, the sealed value set.
    pub scope_commitment: String,
}

#[derive(Clone, Copy, Debug)]
pub struct PseudonymBudget { pub max_values: usize, pub max_value_bytes: usize }
impl Default for PseudonymBudget {
    fn default() -> Self { Self { max_values: 16_384, max_value_bytes: 8 * 1024 * 1024 } }
}
struct Entry { kind: PiiKind, value: String, digest: [u8; 32] }

/// Private session state can contain the canonical value dictionary. It never
/// implements serialization or Debug. Callers retain one sealed context for
/// the entire job; creating per-item 128-bit contexts is not job-wide preflight.
pub struct Pseudonyms<'a> {
    key: &'a PseudonymKey,
    namespace: &'a str,
    entries: Option<Vec<Entry>>,
    identity: PseudonymIdentity,
}
impl<'a> Pseudonyms<'a> {
    /// Explicit full-digest mode for streaming or jobs without complete preflight.
    pub fn full256(key: &'a PseudonymKey, namespace: &'a str, expected_key_commitment: Option<&str>) -> Result<Self, RedactError> {
        Self::new(key, namespace, expected_key_commitment, None)
    }
    /// Admit the COMPLETE job's (type, exact value) set before exposing any
    /// truncated pseudonym. No external-sort fallback is implied: exhaustion
    /// fails closed; the caller may explicitly restart in full256 mode.
    pub fn preflight128(
        key: &'a PseudonymKey, namespace: &'a str, values: &[(PiiKind, &str)],
        budget: PseudonymBudget, expected_key_commitment: Option<&str>,
    ) -> Result<Self, RedactError> {
        check_namespace(namespace)?;
        if let Some(expected) = expected_key_commitment { key.require_commitment(expected)?; }
        if budget.max_values > 1_000_000 || values.len() > budget.max_values { return Err(RedactError::DetectionBudget); }
        let bytes = values.iter().try_fold(0_usize, |sum, (_, v)| {
            if v.is_empty() || v.len() > MAX_VALUE_BYTES { return None; }
            sum.checked_add(v.len())
        }).ok_or(RedactError::InputBudget)?;
        if bytes > budget.max_value_bytes { return Err(RedactError::InputBudget); }
        let mut entries = Vec::new();
        entries.try_reserve_exact(values.len()).map_err(|_| RedactError::AllocationRefused)?;
        for &(kind, value) in values {
            entries.push(Entry { kind, value: value.to_owned(), digest: value_mac(key, namespace, kind, value) });
        }
        entries.sort_unstable_by(|a, b| (a.kind, a.value.as_str()).cmp(&(b.kind, b.value.as_str())));
        entries.dedup_by(|a, b| a.kind == b.kind && a.value == b.value);
        check_collisions(&entries)?;
        Self::new(key, namespace, expected_key_commitment, Some(entries))
    }
    fn new(key: &'a PseudonymKey, namespace: &'a str, expected: Option<&str>, entries: Option<Vec<Entry>>) -> Result<Self, RedactError> {
        check_namespace(namespace)?;
        if let Some(expected) = expected { key.require_commitment(expected)?; }
        let encoding = if entries.is_some() { PseudonymEncoding::Preflighted128 } else { PseudonymEncoding::Full256 };
        let scope = hmac_block(&key.block, |h| {
            h.update(b"fnlp-pseudonym-scope-v1"); framed(h, namespace.as_bytes());
            h.update([u8::from(entries.is_some())]);
            if let Some(values) = &entries {
                h.update((values.len() as u64).to_be_bytes());
                for e in values { framed(h, e.kind.label().as_bytes()); framed(h, e.value.as_bytes()); }
            }
        });
        let identity = PseudonymIdentity { version: PSEUDONYM_VERSION.to_owned(), canonicalization: CANONICALIZATION.to_owned(),
            key_id: key.id.clone(), key_commitment: key.commitment(), encoding, scope_commitment: hex(&scope) };
        Ok(Self { key, namespace, entries, identity })
    }
    pub fn identity(&self) -> &PseudonymIdentity { &self.identity }
    pub fn pseudonym(&self, kind: PiiKind, value: &str) -> Result<String, RedactError> {
        if value.is_empty() || value.len() > MAX_VALUE_BYTES { return Err(RedactError::InputBudget); }
        let digest = match &self.entries {
            Some(entries) => {
                let i = entries.binary_search_by(|e| (e.kind, e.value.as_str()).cmp(&(kind, value)))
                    .map_err(|_| RedactError::InvalidOptions)?;
                entries[i].digest
            }
            None => value_mac(self.key, self.namespace, kind, value),
        };
        let count = if self.entries.is_some() { 16 } else { 32 };
        Ok(format!("[{}_{}]", kind.label(), hex(&digest[..count])))
    }
}
fn check_namespace(namespace: &str) -> Result<(), RedactError> {
    if namespace.is_empty() || namespace.len() > 256 { Err(RedactError::InvalidOptions) } else { Ok(()) }
}
fn check_collisions(entries: &[Entry]) -> Result<(), RedactError> {
    let mut order = Vec::new();
    order.try_reserve_exact(entries.len()).map_err(|_| RedactError::AllocationRefused)?;
    order.extend(entries);
    order.sort_unstable_by(|a, b| a.digest[..16].cmp(&b.digest[..16]));
    if order.windows(2).any(|p| p[0].digest[..16] == p[1].digest[..16]
        && (p[0].kind != p[1].kind || p[0].value != p[1].value)) { return Err(RedactError::Collision); }
    Ok(())
}
fn value_mac(key: &PseudonymKey, namespace: &str, kind: PiiKind, value: &str) -> [u8; 32] {
    hmac_block(&key.block, |h| {
        h.update(PSEUDONYM_VERSION.as_bytes());
        for part in [namespace.as_bytes(), kind.label().as_bytes(), CANONICALIZATION.as_bytes(), value.as_bytes()] { framed(h, part); }
    })
}
fn framed(hash: &mut Sha256, bytes: &[u8]) { hash.update((bytes.len() as u64).to_be_bytes()); hash.update(bytes); }
fn key_block(bytes: &[u8]) -> [u8; 64] {
    let mut block = [0_u8; 64];
    if bytes.len() > 64 { block[..32].copy_from_slice(&Sha256::digest(bytes)); }
    else { block[..bytes.len()].copy_from_slice(bytes); }
    block
}
fn hmac_block(block: &[u8; 64], feed: impl FnOnce(&mut Sha256)) -> [u8; 32] {
    let mut pad = [0_u8; 64];
    for i in 0..64 { pad[i] = block[i] ^ 0x36; }
    let mut inner = Sha256::new(); inner.update(pad); feed(&mut inner);
    let digest = inner.finalize();
    for i in 0..64 { pad[i] = block[i] ^ 0x5c; }
    let mut outer = Sha256::new(); outer.update(pad); outer.update(digest);
    pad.fill(0); outer.finalize().into()
}
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &b in bytes { output.push(DIGITS[(b >> 4) as usize] as char); output.push(DIGITS[(b & 15) as usize] as char); }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    // RFC 4231 sections 4.2-4.8; vectors, not copied implementation code.
    #[test]
    fn rfc4231_all_seven_sha256_vectors() {
        let cases: Vec<(Vec<u8>, Vec<u8>, &str)> = vec![
            (vec![0x0b;20], b"Hi There".to_vec(), "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"),
            (b"Jefe".to_vec(), b"what do ya want for nothing?".to_vec(), "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"),
            (vec![0xaa;20], vec![0xdd;50], "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe"),
            ((1_u8..=25).collect(), vec![0xcd;50], "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b"),
            (vec![0x0c;20], b"Test With Truncation".to_vec(), "a3b6167473100ee06e0c796c2955552b"),
            (vec![0xaa;131], b"Test Using Larger Than Block-Size Key - Hash Key First".to_vec(), "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"),
            (vec![0xaa;131], b"This is a test using a larger than block-size key and a larger than block-size data. The key needs to be hashed before being used by the HMAC algorithm.".to_vec(), "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2"),
        ];
        for (key, data, expected) in cases {
            let actual = hex(&hmac_block(&key_block(&key), |h| h.update(&data)));
            assert_eq!(&actual[..expected.len()], expected);
        }
    }
    #[test]
    fn key_length_reader_and_resume_commitment_are_enforced() {
        assert!(PseudonymKey::from_bytes(&[1;31], "k1").is_err());
        assert!(PseudonymKey::from_reader(&mut &vec![1;4097][..], "k1").is_err());
        let key = PseudonymKey::from_reader(&mut &[1;32][..], "k1").unwrap();
        let other = PseudonymKey::from_bytes(&[2;32], "k1").unwrap();
        key.require_commitment(&key.commitment()).unwrap();
        assert_eq!(key.require_commitment(&other.commitment()), Err(RedactError::KeyMismatch));
    }
    #[test]
    fn namespace_type_value_and_canonicalization_are_unambiguous() {
        let key = PseudonymKey::from_bytes(&[1;32], "k1").unwrap();
        let a = Pseudonyms::full256(&key, "ab", None).unwrap(); let b = Pseudonyms::full256(&key, "a", None).unwrap();
        assert_ne!(a.pseudonym(PiiKind::Person, "c").unwrap(), b.pseudonym(PiiKind::Person, "bc").unwrap());
        assert_ne!(a.pseudonym(PiiKind::Person, "Alice").unwrap(), a.pseudonym(PiiKind::Organization, "Alice").unwrap());
        assert_ne!(a.pseudonym(PiiKind::Person, "é").unwrap(), a.pseudonym(PiiKind::Person, "e\u{301}").unwrap());
    }
    #[test]
    fn full_digest_is_stable_but_cross_key_distinct() {
        let a = PseudonymKey::from_bytes(&[1;32], "a").unwrap(); let b = PseudonymKey::from_bytes(&[2;32], "b").unwrap();
        let x = Pseudonyms::full256(&a, "job", None).unwrap(); let y = Pseudonyms::full256(&b, "job", None).unwrap();
        let token = x.pseudonym(PiiKind::Person, "Alice").unwrap();
        assert_eq!(token.len(), "[person_]".len() + 64);
        assert_eq!(token, x.pseudonym(PiiKind::Person, "Alice").unwrap());
        assert_ne!(token, y.pseudonym(PiiKind::Person, "Alice").unwrap());
    }
    #[test]
    fn sealed_short_mode_refuses_unseen_values() {
        let key = PseudonymKey::from_bytes(&[1;32], "k").unwrap();
        let values = [(PiiKind::Person, "Alice"), (PiiKind::Person, "Bob")];
        let x = Pseudonyms::preflight128(&key, "job", &values, PseudonymBudget::default(), None).unwrap();
        assert_eq!(x.pseudonym(PiiKind::Person, "Alice").unwrap().len(), "[person_]".len() + 32);
        assert!(x.pseudonym(PiiKind::Person, "Carol").is_err());
    }
    #[test]
    fn truncated_collision_between_distinct_values_fails_before_output() {
        let mut other = [1_u8;32]; other[31] = 2;
        let entries = [Entry { kind: PiiKind::Person, value: "Alice".to_owned(), digest: [1;32] },
            Entry { kind: PiiKind::Person, value: "Bob".to_owned(), digest: other }];
        assert_eq!(check_collisions(&entries), Err(RedactError::Collision));
    }
    #[test]
    fn preflight_budgets_and_input_order_are_explicit() {
        let k = PseudonymKey::from_bytes(&[1;32], "k").unwrap(); let v = [(PiiKind::Person, "Alice"), (PiiKind::Person, "Bob")];
        assert!(Pseudonyms::preflight128(&k, "j", &v, PseudonymBudget { max_values: 1, max_value_bytes: 100 }, None).is_err());
        assert!(Pseudonyms::preflight128(&k, "j", &v, PseudonymBudget { max_values: 2, max_value_bytes: 7 }, None).is_err());
        let a = Pseudonyms::preflight128(&k, "j", &v, PseudonymBudget::default(), None).unwrap();
        let b = Pseudonyms::preflight128(&k, "j", &[v[1], v[0]], PseudonymBudget::default(), None).unwrap();
        assert_eq!(a.identity(), b.identity());
    }
}
