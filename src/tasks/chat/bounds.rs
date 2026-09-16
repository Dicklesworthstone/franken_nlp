//! Reject amplified complete responses before building canonical JSON storage.
use super::*;
use std::io::{self, Write};

pub(super) fn result<T: Serialize>(value: &T, cap: u64) -> Result<(), ChatError> {
    struct Counter { remaining: u64, overflow: bool }
    impl Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() as u64 > self.remaining {
                self.overflow = true; return Err(io::Error::other("bounded response"));
            }
            self.remaining -= bytes.len() as u64; Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    let mut count = Counter { remaining: cap, overflow: false };
    if serde_json::to_writer(&mut count, value).is_err() {
        return Err(if count.overflow { ChatError::Limit("complete result bytes") } else { ChatError::Serialization });
    }
    // Preserve canonical nonfinite-number rejection and verify the actual
    // canonical size. Never feed serde's serialized bytes back as authority.
    if canonjson::canonical_bytes(value).map_err(|_| ChatError::Serialization)?.len() as u64 > cap {
        return Err(ChatError::Limit("complete result bytes"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, collections::BTreeMap};
    #[test]
    fn oversized_value_stops_after_the_nonallocating_size_pass() {
        struct Counted<'a> { calls: &'a Cell<usize>, value: &'a str }
        impl Serialize for Counted<'_> {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                self.calls.set(self.calls.get() + 1); serializer.serialize_str(self.value)
            }
        }
        let calls = Cell::new(0); let text = "x".repeat(4096);
        assert!(matches!(result(&Counted { calls: &calls, value: &text }, 64), Err(ChatError::Limit("complete result bytes"))));
        assert_eq!(calls.get(), 1, "canonical storage must not be constructed for the oversized value");
    }
    #[test]
    fn complete_canonical_size_and_nonfinite_rejection_are_preserved() {
        let value = BTreeMap::from([("b", 1), ("a", 2)]);
        let bytes = canonjson::canonical_bytes(&value).unwrap();
        result(&value, bytes.len() as u64).unwrap();
        assert!(result(&value, bytes.len() as u64 - 1).is_err());
        assert!(matches!(result(&f64::NAN, 100), Err(ChatError::Serialization)));
    }
}
