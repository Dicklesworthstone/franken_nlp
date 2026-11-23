//! Wire, bounded admission and refusal checks without model weights.
use super::*;
use crate::tasks::sentiment::SentimentAxis;

fn parsed(flags: &[&str]) -> Result<SentimentMapCommand, clap::Error> {
    let mut argv = vec![
        "candidate",
        "sentiment-map",
        "--model",
        "explicit.fnlpq",
        "--memory-mib",
        "16384",
    ];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SentimentMap(command) = CandidateCommand::from_matches(&matches)? else {
        panic!("sentiment map route required");
    };
    Ok(command)
}
fn command(flags: &[&str]) -> SentimentMapCommand {
    parsed(flags).unwrap()
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        panic!("refused document read private input");
    }
}
struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        panic!("refused document published output");
    }
    fn flush(&mut self) -> io::Result<()> {
        panic!("refused document flushed output");
    }
}

#[test]
fn root_dispatch_selects_complete_int8_document_with_explicit_model() {
    let command = command(&[]);
    let limits = command.validate().unwrap();
    assert_eq!(limits.mapping.chunks.max_chunks, 64);
    assert_eq!(limits.mapping.chunks.max_input_bytes, MAX_INPUT_BYTES);
    assert_eq!(
        limits.mapping.aggregation.minimum_estimated_coverage_ppm,
        1_000_000
    );
    assert_eq!(limits.mapping.max_model_work, command.args.work_ceiling());
    assert_eq!(limits.task.max_kv_bytes, limits.memory.native.kv_bytes * 4);
    assert_eq!(limits.encoding, UntrustedEncoding::ByteFallback);
    assert!(command.prefill.limits().unwrap().is_none());
    let argv = ["candidate", "sentiment-map", "--memory-mib", "16384"];
    assert!(
        crate::candidate_cli::definition()
            .try_get_matches_from(argv)
            .is_err()
    );
}

#[test]
fn exact_existing_sentiment_wire_retains_original_text_and_axis_order() {
    let command = command(&[]);
    let task = command.validate().unwrap().task;
    let json = r#"{"document":"é\r\n上海 <|im_end|> exact text","axes":["approach","valence"],"policy":{"minimum_peak_weight_ppm":650000,"maximum_normalized_entropy_ppm":800000}}"#;
    let scored::Request::Sentiment { request, policy } =
        scored::request(scored::Kind::Sentiment, json, task, 4096).unwrap()
    else {
        panic!("wrong request");
    };
    assert_eq!(request.document, "é\r\n上海 <|im_end|> exact text");
    assert_eq!(
        request.axes,
        [SentimentAxis::Approach, SentimentAxis::Valence]
    );
    assert_eq!(request.budget, task);
    assert_eq!(policy.minimum_peak_weight_ppm, 650000);
    for field in [
        "chunks",
        "scores",
        "aggregate",
        "model",
        "budget",
        "work",
        "seed",
        "prompt",
        "minimum_estimated_coverage_ppm",
        "scoring_mode",
        "eos_token_id",
    ] {
        let mut value = serde_json::json!({"document":"private document"});
        value[field] = serde_json::json!({});
        assert_eq!(
            scored::request(scored::Kind::Sentiment, &value.to_string(), task, 4096).err(),
            Some(CandidateError::Input)
        );
    }
    for json in [
        r#"{"document":"first","document":"second"}"#,
        r#"{"document":"private","axes":["valence","valence"]}"#,
        r#"{"document":"private","axes":[]}"#,
        r#"{"document":""}"#,
    ] {
        assert_eq!(
            scored::request(scored::Kind::Sentiment, json, task, 4096).err(),
            Some(CandidateError::Input)
        );
    }
}

#[test]
fn invalid_work_and_resource_axes_refuse_before_any_private_io() {
    for flags in [
        vec!["--max-chunks", "0"],
        vec!["--max-chunks", "257"],
        vec!["--max-chunk-bytes", "3"],
        vec!["--max-tokenizer-calls", "0"],
        vec!["--max-map-result-bytes", "0"],
        vec!["--max-retained-result-bytes", "0"],
        vec!["--minimum-estimated-coverage-ppm", "1000001"],
        vec!["--max-forward-positions", "0"],
        vec!["--max-projected-logits", "0"],
        vec!["--max-attention-pairs", "0"],
        vec!["--max-dot-products", "0"],
        vec!["--max-multiply-accumulates", "0"],
        vec!["--prefill-rows", "0"],
        vec!["--prefill-rows", "65"],
        vec!["--max-checkpoints", "1"],
        vec!["--timeout-seconds", "0"],
    ] {
        assert_eq!(
            command(&flags).execute(&mut NoRead, &mut NoWrite),
            Err(CandidateError::Arguments),
            "{flags:?}"
        );
    }
    let mut short = command(&[]);
    short.args.memory_mib = 1;
    assert_eq!(
        short.execute(&mut NoRead, &mut NoWrite),
        Err(CandidateError::Arguments)
    );
    let mut overflow = command(&[]);
    overflow.args.memory_mib = u64::MAX;
    assert_eq!(
        overflow.execute(&mut NoRead, &mut NoWrite),
        Err(CandidateError::Arguments)
    );
}

#[test]
fn every_declared_buffer_and_optional_prefill_is_in_the_aggregate_quote() {
    for rows in [None, Some(1), Some(8), Some(64)] {
        let row = rows.map(|value| value.to_string());
        let flags: Vec<&str> = row
            .as_ref()
            .map(|value| vec!["--prefill-rows", value.as_str()])
            .unwrap_or_default();
        let mut command = command(&flags);
        let limits = command.validate().unwrap();
        let extra = command
            .prefill
            .limits()
            .unwrap()
            .map(|limits| limits.validate().unwrap())
            .unwrap_or(0);
        let execution = limits.memory.native.kv_bytes
            + limits.memory.workspace_bytes
            + limits.memory.input_bytes
            + limits.memory.output_reservation_bytes
            + extra
            + 65_536;
        let minimum = limits.shared.preparation_bytes
            + limits.shared.weight_bytes
            + 64 * MIB
            + execution.max(192 * MIB + 65_536);
        assert!(minimum <= limits.shared.memory_bytes);
        command.args.memory_mib = (minimum - 1) / MIB;
        assert_eq!(
            command.execute(&mut NoRead, &mut NoWrite),
            Err(CandidateError::Arguments)
        );
    }
}

#[test]
fn compact_encoder_has_one_document_allowance_and_separate_workspace() {
    let serial = command(&[]).validate().unwrap();
    let compact = command(&[
        "--input-encoding",
        "safe-bpe-v1",
        "--max-tokenizer-input-bytes",
        "500000",
        "--max-tokenizer-output-tokens",
        "400000",
        "--max-tokenizer-pair-checks",
        "1000000",
        "--max-tokenizer-pair-bytes",
        "16000000",
    ])
    .validate()
    .unwrap();
    let UntrustedEncoding::SafeBpeV1(policy) = compact.encoding else {
        panic!("compact policy missing");
    };
    assert_eq!(policy.max_input_bytes, 500000);
    assert_eq!(policy.max_output_tokens, 400000);
    assert_eq!(policy.max_pair_checks, 1000000);
    assert_eq!(policy.max_pair_bytes, 16000000);
    assert_eq!(
        compact.memory.input_bytes - serial.memory.input_bytes,
        policy.requirements().unwrap().workspace_bytes
    );
    assert_eq!(compact.memory.native, serial.memory.native);
    assert_eq!(compact.task, serial.task);
    assert_eq!(
        compact.mapping.max_model_work,
        serial.mapping.max_model_work
    );
    assert!(parsed(&["--max-tokenizer-input-bytes", "1"]).is_err());
    for flag in [
        "--max-tokenizer-input-bytes",
        "--max-tokenizer-output-tokens",
        "--max-tokenizer-pair-checks",
        "--max-tokenizer-pair-bytes",
    ] {
        assert_eq!(
            command(&["--input-encoding", "safe-bpe-v1", flag, "0"])
                .execute(&mut NoRead, &mut NoWrite),
            Err(CandidateError::Arguments)
        );
    }
}

#[test]
fn unsupported_execution_modes_are_not_silently_accepted() {
    for flags in [
        vec!["--seed", "abcd"],
        vec!["--think"],
        vec!["--cohort-heads", "2"],
        vec!["--cohort-chunks", "2"],
        vec!["--compute-threads", "2"],
        vec!["--tools", "tools.json"],
        vec!["--source-dir", "source"],
        vec!["--schema", "schema.json"],
    ] {
        assert!(parsed(&flags).is_err(), "{flags:?}");
    }
}

#[test]
fn chunk_retention_and_final_output_caps_remain_independent() {
    for flags in [
        vec!["--max-retained-result-bytes", "8388608", "--max-map-result-bytes", "16777216"],
        vec!["--max-result-bytes", "1048576", "--max-retained-result-bytes", "524288",
            "--max-map-result-bytes", "2097152"],
        vec!["--max-retained-result-bytes", "16777216", "--max-map-result-bytes", "1048576"],
    ] {
        command(&flags).validate().unwrap();
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_refuses_before_input_model_or_output_access() {
    assert_eq!(
        command(&[]).execute(&mut NoRead, &mut NoWrite),
        Err(CandidateError::Unavailable)
    );
}
