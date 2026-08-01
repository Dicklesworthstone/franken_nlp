#![deny(unsafe_code)]

//! Model-gated greedy-loop coverage for the eager source engine.
//!
//! This is a code-first transcript exercise, not L4 parity authority.  The
//! oracle executable-code closure and fixture-promotion lineage remain open;
//! an armed run proves only that this loop feeds the eager engine and preserves
//! the recorded token sequence under its explicit source fixture.

use std::{
    convert::Infallible,
    env, fs,
    path::{Path, PathBuf},
};

use franken_nlp::{
    native_engine::{
        decode::{
            DECODE_TOKEN_EVENT_SCHEMA_VERSION, DecodeEventSink, DecodeParams, DecodeTokenEvent,
            greedy_decode_with_sink,
        },
        hf_bf16_eager::{HF_BF16_EAGER_PROFILE, HfBf16EagerEngine, HfBf16EagerWeights},
    },
    tokenizer::embedded::EmbeddedTokenizer,
};
use serde::Deserialize;

const PINNED_MODEL_SOURCE_ENV: &str = "FNLP_PINNED_MODEL_SOURCE";
const ORACLE_SMOKE: &str = "docs/truth-pack/oracle_smoke_source_closure_f56ec5a.json";

#[derive(Debug, Deserialize)]
struct OracleSmoke {
    execution: OracleSmokeExecution,
}

#[derive(Debug, Deserialize)]
struct OracleSmokeExecution {
    greedy_new_token_ids: Vec<u32>,
    prompt_token_ids: Vec<u32>,
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
fn eager_greedy_loop_matches_the_recorded_smoke_prefix_and_stream_bytes() {
    let Some(source) = armed_pinned_model_source() else {
        return;
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let oracle: OracleSmoke = serde_json::from_slice(
        &fs::read(root.join(ORACLE_SMOKE)).expect("read eager decode oracle smoke"),
    )
    .expect("parse eager decode oracle smoke");
    assert!(!oracle.execution.prompt_token_ids.is_empty());
    assert!(!oracle.execution.greedy_new_token_ids.is_empty());

    let weights = HfBf16EagerWeights::from_pinned_source(&source)
        .expect("armed source must satisfy the eager tensor census");
    let context_cap = oracle
        .execution
        .prompt_token_ids
        .len()
        .checked_add(
            oracle
                .execution
                .greedy_new_token_ids
                .len()
                .saturating_sub(1),
        )
        .expect("fixture context arithmetic must fit usize");
    let mut engine = HfBf16EagerEngine::new(weights, context_cap)
        .expect("fixture context must admit every required K/V feedback position");
    let tokenizer = EmbeddedTokenizer::pinned().expect("embedded tokenizer must parse");
    let params = DecodeParams::hf_bf16_greedy(
        41,
        oracle.execution.greedy_new_token_ids.len(),
        oracle
            .execution
            .greedy_new_token_ids
            .len()
            .checked_mul(256)
            .expect("fixture byte budget arithmetic must fit usize"),
    );
    let mut sink = RecordingSink::default();
    let output = greedy_decode_with_sink(
        &mut engine,
        &oracle.execution.prompt_token_ids,
        &params,
        tokenizer.tokenizer(),
        &mut sink,
    )
    .expect("eager greedy loop must execute the recorded smoke prefix");

    assert_eq!(output.numerics_profile, HF_BF16_EAGER_PROFILE);
    assert_eq!(
        output.emitted_token_ids,
        oracle.execution.greedy_new_token_ids
    );
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
        "DECODE_LOOP RESULT=CODE_FIRST_TRANSCRIPT_MATCH authority=non_authoritative tokens={} profile={}",
        output.emitted_token_ids.len(),
        output.numerics_profile,
    );
}
