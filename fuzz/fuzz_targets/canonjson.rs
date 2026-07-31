//! Dev-only `cargo fuzz run canonjson` target.
//!
//! The fuzz workspace is intentionally not part of the release graph.  Its
//! `fuzz/Cargo.toml` should name this target and depend on `libfuzzer-sys` plus
//! the local `franken_nlp` crate.

#![no_main]

use franken_nlp::canonjson::{canonical_bytes, parse_str};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(input) = std::str::from_utf8(data) else {
        return;
    };
    let Ok(value) = parse_str(input) else {
        return;
    };

    let bytes = canonical_bytes(&value).expect("parsed JSON must have canonical bytes");
    let canonical = std::str::from_utf8(&bytes).expect("canonical JSON must be UTF-8");
    let reparsed = parse_str(canonical).expect("writer output must pass rejecting parser");
    assert_eq!(
        reparsed, value,
        "round-trip mismatch for minimized input: {input:?}"
    );

    let duplicate = format!(r#"{{"duplicate":0,"duplicate":{input}}}"#);
    assert!(
        parse_str(&duplicate).is_err(),
        "duplicate injection accepted for minimized input: {input:?}"
    );
});
