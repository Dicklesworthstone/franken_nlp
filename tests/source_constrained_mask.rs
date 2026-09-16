//! Model-free integration of the pinned detokenizer, vocabulary trie, and
//! request-owned source × JSON byte language. Control-token exclusion remains
//! the native decoder's separate policy; this checks exact payload legality.

use franken_nlp::{
    grammar::{CompileLimits,
        mask::{MaskWorkLimits, VocabMaskOracle, VocabTrie},
        runtime::{JsonProgram, SourceRuntimeLimits}},
    native_engine::lmhead::NANBEIGE_VOCAB_SIZE,
    tokenizer::embedded::EmbeddedTokenizer,
};

#[test]
fn pinned_source_masks_equal_direct_replay_at_json_and_unicode_boundaries() {
    let embedded = EmbeddedTokenizer::pinned().unwrap();
    let oracle = VocabMaskOracle::new(VocabTrie::from_tokenizer(embedded.tokenizer(), NANBEIGE_VOCAB_SIZE).unwrap());
    let source = "Alice \"A\"\n上海 Alice";
    let schema = r#"{"type":"object","properties":{"name":{"type":"string","maxLength":12,"x-fnlp-source":"verbatim"},"type":{"type":"string","enum":["person","location"]}},"required":["name","type"],"additionalProperties":false}"#;
    let program = JsonProgram::compile_with_source(schema, source, CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
    let limits = MaskWorkLimits { max_trie_node_visits: oracle.trie().node_count(), checkpoint_interval_nodes: 1024 };
    for prefix in ["", "{\"name\":\"", "{\"name\":\"Ali", "{\"name\":\"\\", "{\"name\":\"上", "{\"name\":\"上海\",\"type\":\""] {
        let mut state = program.initial_state();
        assert!(state.consume_bytes(prefix.as_bytes()), "{prefix:?}");
        let mask = oracle.materialize(&state, limits, |_| true).unwrap();
        for id in 0..NANBEIGE_VOCAB_SIZE as u32 {
            let mut direct = state.clone();
            let expected = oracle.trie().token_bytes(id).is_some_and(|bytes| !bytes.is_empty() && direct.consume_bytes(bytes));
            assert_eq!(mask.contains(id), expected, "prefix={prefix:?} token={id}");
            if expected && direct.is_accepting() {
                let mut complete = prefix.as_bytes().to_vec();
                complete.extend_from_slice(oracle.trie().token_bytes(id).unwrap());
                program.validate_json(std::str::from_utf8(&complete).unwrap()).unwrap();
            }
        }
    }
    // Real byte-fallback ids make the source constraint observable before
    // decoding: Z is absent, whereas A is a viable source prefix.
    let mut state = program.initial_state(); assert!(state.consume_bytes(b"{\"name\":\""));
    let mask = oracle.materialize(&state, limits, |_| true).unwrap();
    for (byte, expected) in [(b'Z', false), (b'A', true)] {
        let ids = embedded.tokenizer().encode_byte_fallback_only(&[byte]).unwrap();
        assert_eq!(ids.len(), 1); assert_eq!(mask.contains(ids[0]), expected);
    }
}
