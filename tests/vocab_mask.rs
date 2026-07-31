//! Model-free coverage for the exact-byte vocabulary trie and schema-only cache.

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    thread,
};

use franken_nlp::{
    grammar::mask::{
        ByteState, MaskCacheError, MaskOracleError, MaskWorkLimits, NANBEIGE_MASK_BYTES_PER_STATE,
        SchemaMaskCache, SchemaMaskKey, VocabMaskOracle, VocabTrie,
    },
    tokenizer::{
        bpe::{AddedToken, SpBpeTokenizer},
        sp_model::{
            ModelType, NormalizerFacts, PieceType, SpecialPieceIds, SpmModel, SpmPiece,
            parse_spm_model,
        },
    },
};

const PINNED_TOKENIZER_MODEL: &[u8] = include_bytes!("fixtures/reference/tokenizer.model");
const PINNED_ADDED_TOKENS: &[u8] = include_bytes!("fixtures/reference/added_tokens.json");
const NANBEIGE_VOCAB_SIZE: usize = 166_144;

fn piece(surface: &str, piece_type: PieceType) -> SpmPiece {
    SpmPiece {
        piece: surface.to_owned(),
        score: 0.0,
        piece_type,
    }
}

fn tokenizer() -> SpBpeTokenizer {
    SpBpeTokenizer::from_model(SpmModel {
        pieces: vec![
            piece("<unk>", PieceType::Unknown),
            piece("<s>", PieceType::Control),
            piece("<0x00>", PieceType::Byte),
            piece("a", PieceType::Normal),
            piece("▁a", PieceType::Normal),
            piece("ab", PieceType::Normal),
            piece("<0xFF>", PieceType::Byte),
            piece("λ", PieceType::Normal),
        ],
        model_type: ModelType::Bpe,
        normalizer: NormalizerFacts {
            name: "identity".to_owned(),
            is_identity: true,
            precompiled_charsmap_is_empty: true,
        },
        special_ids: SpecialPieceIds {
            unk_id: 0,
            bos_id: 1,
            eos_id: -1,
            pad_id: -1,
        },
    })
    .expect("toy tokenizer is internally valid")
}

fn real_tokenizer() -> SpBpeTokenizer {
    let model = parse_spm_model(PINNED_TOKENIZER_MODEL).expect("pinned tokenizer fixture parses");
    let added: BTreeMap<String, u32> =
        serde_json::from_slice(PINNED_ADDED_TOKENS).expect("pinned added-token fixture parses");
    SpBpeTokenizer::with_added_tokens(
        model,
        added
            .into_iter()
            .map(|(surface, id)| AddedToken::new(surface, id)),
    )
    .expect("pinned tokenizer plus added-token table is internally valid")
}

#[derive(Clone)]
struct PrefixState {
    expected: Vec<u8>,
    cursor: usize,
}

impl PrefixState {
    fn new(expected: &[u8]) -> Self {
        Self {
            expected: expected.to_vec(),
            cursor: 0,
        }
    }
}

impl ByteState for PrefixState {
    fn consume(&mut self, byte: u8) -> bool {
        let matches = self.expected.get(self.cursor).copied() == Some(byte);
        if matches {
            self.cursor += 1;
        }
        matches
    }
}

fn candidate_is_legal(initial: &PrefixState, bytes: &[u8]) -> bool {
    let mut state = initial.clone();
    bytes.iter().copied().all(|byte| state.consume(byte))
}

#[test]
fn trie_uses_detokenized_bytes_and_keeps_model_padding_illegal() {
    let trie = VocabTrie::from_tokenizer(&tokenizer(), 10).expect("trie builds from exact decoder");
    assert_eq!(trie.vocab_size(), 10);
    assert_eq!(trie.token_bytes(4), Some(&b" a"[..]));
    assert_eq!(
        trie.token_bytes(1),
        None,
        "control pieces cannot advance grammar"
    );
    assert_eq!(
        trie.token_bytes(8),
        None,
        "padded model rows have no tokenizer token"
    );
    assert_eq!(trie.token_bytes(7), Some("λ".as_bytes()));
    assert!(trie.indexed_token_count() < trie.vocab_size());
    assert!(trie.node_count() > 1);
}

#[test]
fn pinned_real_vocab_trie_covers_decodable_rows_and_excludes_embedding_padding() {
    let trie = VocabTrie::from_tokenizer(&real_tokenizer(), NANBEIGE_VOCAB_SIZE)
        .expect("pinned real vocabulary trie builds without model weights");
    assert_eq!(trie.vocab_size(), NANBEIGE_VOCAB_SIZE);
    assert_eq!(
        trie.token_bytes(166_143),
        None,
        "alignment padding is not a token"
    );
    assert_eq!(trie.token_bytes(166_103), Some(&b"<think>"[..]));
    assert!(trie.indexed_token_count() > 166_000);
    assert_eq!(
        (trie.vocab_size() + 7) / 8,
        NANBEIGE_MASK_BYTES_PER_STATE,
        "the model width retains the documented dense-mask footprint"
    );
}

#[test]
fn shared_trie_walk_equals_bruteforce_per_token_transition() {
    let trie = VocabTrie::from_tokenizer(&tokenizer(), 10).expect("trie builds");
    let oracle = VocabMaskOracle::new(trie.clone());
    let initial = PrefixState::new(b" ab");
    let mask = oracle
        .materialize(&initial, MaskWorkLimits::default(), |_| true)
        .expect("bounded trie walk succeeds");

    for token_id in 0..u32::try_from(trie.vocab_size()).expect("toy width fits u32") {
        let brute_force = trie
            .token_bytes(token_id)
            .is_some_and(|bytes| candidate_is_legal(&initial, bytes));
        assert_eq!(
            mask.contains(token_id),
            brute_force,
            "trie and brute force differ for token id={token_id}"
        );
    }
    assert_eq!(mask.legal_ids().collect::<Vec<_>>(), vec![4]);
}

#[test]
fn randomized_prefixes_match_bruteforce_including_byte_fallback_and_utf8_boundaries() {
    let trie = VocabTrie::from_tokenizer(&tokenizer(), 10).expect("trie builds");
    let oracle = VocabMaskOracle::new(trie.clone());
    let alphabet = [0, b' ', b'a', b'b', 0xff, b'<', 0xce, 0xbb];
    let mut seed = 0x4d41_534b_0000_0001_u64;
    for case in 0..128 {
        let length = usize::try_from(seed & 7).expect("small random length fits usize");
        let mut expected = Vec::with_capacity(length);
        for _ in 0..length {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let index = usize::try_from((seed >> 32) % alphabet.len() as u64)
                .expect("alphabet index fits usize");
            expected.push(alphabet[index]);
        }
        let initial = PrefixState::new(&expected);
        let mask = oracle
            .materialize(&initial, MaskWorkLimits::default(), |_| true)
            .expect("finite toy trie always materializes under default work limits");
        for token_id in 0..u32::try_from(trie.vocab_size()).expect("toy width fits u32") {
            let brute_force = trie
                .token_bytes(token_id)
                .is_some_and(|bytes| candidate_is_legal(&initial, bytes));
            assert_eq!(
                mask.contains(token_id),
                brute_force,
                "random case={case} differs for token id={token_id} expected_bytes={expected:?}"
            );
        }
    }
}

#[test]
fn mask_work_limits_and_checkpoints_refuse_partial_success() {
    let oracle = VocabMaskOracle::new(VocabTrie::from_tokenizer(&tokenizer(), 10).unwrap());
    let state = PrefixState::new(b" a");
    let budget = oracle
        .materialize(
            &state,
            MaskWorkLimits {
                max_trie_node_visits: 1,
                checkpoint_interval_nodes: 1,
            },
            |_| true,
        )
        .expect_err("a one-node budget cannot expand the trie");
    assert!(matches!(budget, MaskOracleError::WorkBudgetExceeded { .. }));

    let cancelled = oracle
        .materialize(
            &state,
            MaskWorkLimits {
                max_trie_node_visits: 100,
                checkpoint_interval_nodes: 1,
            },
            |_| false,
        )
        .expect_err("checkpoint cancellation must not yield a partial mask");
    assert!(matches!(
        cancelled,
        MaskOracleError::CancelledAtCheckpoint { .. }
    ));
}

#[test]
fn schema_cache_is_byte_bounded_and_has_no_document_key_surface() {
    let oracle = VocabMaskOracle::new(VocabTrie::from_tokenizer(&tokenizer(), 10).unwrap());
    let mask = oracle
        .materialize(&PrefixState::new(b" a"), MaskWorkLimits::default(), |_| {
            true
        })
        .unwrap();
    let key = SchemaMaskKey {
        tokenizer_digest: [1; 32],
        schema_digest: [2; 32],
        grammar_version: 1,
    };
    let mut cache = SchemaMaskCache::new(mask.byte_len());
    cache
        .insert(key, 7, mask.clone())
        .expect("one mask fits exactly");
    assert_eq!(cache.get(key, 7), Some(mask.clone()));
    assert_eq!(cache.stats().schema_entries, 1);
    assert_eq!(cache.stats().state_masks, 1);
    assert_eq!(cache.stats().used_bytes, mask.byte_len());

    let error = cache
        .insert(key, 8, mask)
        .expect_err("second state exceeds the strict byte budget");
    assert!(matches!(error, MaskCacheError::ByteBudgetExceeded { .. }));
}

#[test]
fn concurrent_schema_keys_share_only_aggregate_cache_state() {
    let oracle = VocabMaskOracle::new(VocabTrie::from_tokenizer(&tokenizer(), 10).unwrap());
    let mask = oracle
        .materialize(&PrefixState::new(b" a"), MaskWorkLimits::default(), |_| {
            true
        })
        .unwrap();
    let cache = Arc::new(Mutex::new(SchemaMaskCache::new(mask.byte_len() * 2)));
    let mut joins = Vec::new();
    for schema_digest in [[3; 32], [4; 32]] {
        let cache = Arc::clone(&cache);
        let mask = mask.clone();
        joins.push(thread::spawn(move || {
            cache
                .lock()
                .expect("cache mutex is not poisoned")
                .insert(
                    SchemaMaskKey {
                        tokenizer_digest: [1; 32],
                        schema_digest,
                        grammar_version: 1,
                    },
                    7,
                    mask,
                )
                .expect("independent schema mask fits the aggregate budget");
        }));
    }
    for join in joins {
        join.join().expect("cache admission thread completes");
    }
    let stats = cache.lock().expect("cache mutex is not poisoned").stats();
    assert_eq!(stats.schema_entries, 2);
    assert_eq!(stats.state_masks, 2);
    assert_eq!(stats.used_bytes, stats.capacity_bytes);
}
