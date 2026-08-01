#![deny(unsafe_code)]

//! Model-gated greedy-loop coverage for the eager source engine.
//!
//! This is a code-first transcript exercise, not L4 parity authority.  The
//! oracle executable-code closure and fixture-promotion lineage remain open;
//! an armed run proves only that this loop feeds the eager engine and preserves
//! the recorded token sequence under its explicit L4 fixture.

use std::{
    convert::Infallible,
    env, fs,
    path::{Path, PathBuf},
};

use franken_nlp::{
    native_engine::{
        decode::{
            DECODE_TOKEN_EVENT_SCHEMA_VERSION, DecodeEventSink, DecodeFinishReason, DecodeParams,
            DecodeTokenEvent, greedy_decode_with_sink,
        },
        hf_bf16_eager::{HF_BF16_EAGER_PROFILE, HfBf16EagerEngine, HfBf16EagerWeights},
    },
    tokenizer::embedded::EmbeddedTokenizer,
};
use serde::Deserialize;

const PINNED_MODEL_SOURCE_ENV: &str = "FNLP_PINNED_MODEL_SOURCE";
const L4_GREEDY_TRACE: &str = "tests/fixtures/reference/hf-bf16-eager/prompt-000/trace.json";

#[derive(Debug, Deserialize)]
struct L4Trace {
    attention_backend: String,
    dtype: String,
    greedy_contract: GreedyContract,
    greedy_tokens: Vec<u32>,
    model_id: String,
    prefill_input_ids: Vec<u32>,
    profile: String,
    revision: String,
    source_closure_verification: String,
    variance_only: bool,
}

#[derive(Debug, Deserialize)]
struct GreedyContract {
    stable_prefix_length: usize,
    status: String,
}

#[derive(Default)]
struct RecordingSink {
    events: Vec<DecodeTokenEvent>,
}

impl DecodeEventSink for RecordingSink {
    type Permit = usize;
    type Error = Infallible;

    fn reserve(&mut self, event: &DecodeTokenEvent) -> Result<Self::Permit, Self::Error> {
        Ok(event.token_index)
    }

    fn permit(&mut self, permit: Self::Permit, event: DecodeTokenEvent) -> Result<(), Self::Error> {
        assert_eq!(permit, self.events.len());
        self.events.push(event);
        Ok(())
    }
}

fn armed_pinned_model_source() -> Option<PathBuf> {
    let Some(source) = env::var_os(PINNED_MODEL_SOURCE_ENV).map(PathBuf::from) else {
        eprintln!(
            "DECODE_LOOP RESULT=SKIPPED_NO_MODEL authority=non_authoritative reason={PINNED_MODEL_SOURCE_ENV}-unset"
        );
        return None;
    };
    if !source.is_dir() {
        eprintln!(
            "DECODE_LOOP RESULT=SKIPPED_NO_MODEL authority=non_authoritative reason={PINNED_MODEL_SOURCE_ENV}-not-directory"
        );
        return None;
    }
    Some(source)
}

#[test]
fn eager_greedy_loop_matches_l4_fixture_stable_prefix_and_stream_bytes() {
    let Some(source) = armed_pinned_model_source() else {
        return;
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let trace: L4Trace = serde_json::from_slice(
        &fs::read(root.join(L4_GREEDY_TRACE)).expect("read eager decode L4 trace"),
    )
    .expect("parse eager decode L4 trace");
    assert_eq!(trace.profile, HF_BF16_EAGER_PROFILE);
    assert_eq!(trace.attention_backend, "eager");
    assert_eq!(trace.dtype, "bfloat16");
    assert_eq!(trace.model_id, "Nanbeige/Nanbeige4.2-3B");
    assert_eq!(trace.revision, "f56ec5a9650268aa098496734743c25ea778bd2d");
    assert_eq!(
        trace.source_closure_verification,
        "verified-full-ten-file-sha256"
    );
    assert!(!trace.variance_only);
    assert_eq!(trace.greedy_contract.status, "frozen");
    assert!(!trace.prefill_input_ids.is_empty());
    assert!(trace.greedy_contract.stable_prefix_length > 0);
    assert!(trace.greedy_contract.stable_prefix_length <= trace.greedy_tokens.len());
    let expected_tokens = &trace.greedy_tokens[..trace.greedy_contract.stable_prefix_length];

    let weights = HfBf16EagerWeights::from_pinned_source(&source)
        .expect("armed source must satisfy the eager tensor census");
    let context_cap = trace
        .prefill_input_ids
        .len()
        .checked_add(expected_tokens.len().saturating_sub(1))
        .expect("fixture context arithmetic must fit usize");
    let mut engine = HfBf16EagerEngine::new(weights, context_cap)
        .expect("fixture context must admit every required K/V feedback position");
    let tokenizer = EmbeddedTokenizer::pinned().expect("embedded tokenizer must parse");
    let params = DecodeParams::hf_bf16_greedy(
        41,
        expected_tokens.len(),
        expected_tokens
            .len()
            .checked_mul(256)
            .expect("fixture byte budget arithmetic must fit usize"),
    );
    let mut sink = RecordingSink::default();
    let output = greedy_decode_with_sink(
        &mut engine,
        &trace.prefill_input_ids,
        &params,
        tokenizer.tokenizer(),
        &mut sink,
    )
    .expect("eager greedy loop must execute the recorded L4 stable prefix");

    assert_eq!(output.numerics_profile, HF_BF16_EAGER_PROFILE);
    assert_eq!(output.emitted_token_ids, expected_tokens);
    assert_eq!(output.finish_reason, DecodeFinishReason::Budget);
    assert_eq!(sink.events.len(), output.emitted_token_ids.len());
    assert!(sink.events.iter().enumerate().all(|(index, event)| {
        event.schema_version == DECODE_TOKEN_EVENT_SCHEMA_VERSION
            && event.request_seq == params.request_seq
            && event.token_index == index
            && event.token_id == output.emitted_token_ids[index]
            && event.logprob.is_none()
    }));
    let streamed_bytes = sink
        .events
        .iter()
        .flat_map(|event| event.decoded_bytes.iter().copied())
        .collect::<Vec<_>>();
    assert_eq!(streamed_bytes, output.decoded_bytes);

    eprintln!(
        "DECODE_LOOP RESULT=CODE_FIRST_L4_TRANSCRIPT_MATCH authority=non_authoritative tokens={} profile={} fixture={}",
        output.emitted_token_ids.len(),
        output.numerics_profile,
        L4_GREEDY_TRACE,
    );
}
