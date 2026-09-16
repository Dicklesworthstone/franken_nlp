//! Model-free integration of the pinned tokenizer, byte runtime, and trie mask.
use franken_nlp::{
    grammar::{CompileLimits, mask::{MaskWorkLimits, VocabMaskOracle, VocabTrie}, runtime::JsonProgram},
    native_engine::lmhead::NANBEIGE_VOCAB_SIZE,
    tokenizer::embedded::EmbeddedTokenizer,
};

#[test]
fn pinned_vocabulary_masks_equal_direct_byte_replay_at_json_boundaries() {
    let embedded = EmbeddedTokenizer::pinned().expect("committed pinned tokenizer");
    let trie = VocabTrie::from_tokenizer(embedded.tokenizer(), NANBEIGE_VOCAB_SIZE).unwrap();
    let limits = MaskWorkLimits { max_trie_node_visits: trie.node_count(), checkpoint_interval_nodes: 1024 };
    let oracle = VocabMaskOracle::new(trie);
    let program = JsonProgram::compile(
        r#"{"type":"object","properties":{"flag":{"type":"boolean"}},"required":["flag"],"additionalProperties":false}"#,
        CompileLimits { max_output_bytes: 64, ..CompileLimits::default() },
    ).unwrap();
    for prefix in ["", "{", "{\"flag\":", "{\"flag\":true", "{\"flag\":true}"] {
        let mut state = program.initial_state();
        assert!(state.consume_bytes(prefix.as_bytes()));
        let mask = oracle.materialize(&state, limits, |_| true).unwrap();
        for id in 0..NANBEIGE_VOCAB_SIZE as u32 {
            let direct = oracle.trie().token_bytes(id).is_some_and(|bytes| {
                !bytes.is_empty() && state.clone().consume_bytes(bytes)
            });
            assert_eq!(mask.contains(id), direct, "prefix={prefix:?} token={id}");
        }
        if state.is_accepting() { assert_eq!(mask.legal_ids().count(), 0); }
        else { assert!(mask.legal_ids().next().is_some()); }
    }
}
