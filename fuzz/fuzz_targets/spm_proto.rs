#![no_main]

use franken_nlp::tokenizer::sp_model::parse_spm_model;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The contract is binary: parse the bounded accepted subset or return a
    // typed error. Neither result may panic, hang, or allocate from a declared
    // hostile length prefix.
    let _ = parse_spm_model(data);
});
