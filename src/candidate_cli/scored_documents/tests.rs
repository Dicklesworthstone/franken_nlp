//! Command admission is independent of model files and private corpus IO.
use super::*;

pub(in crate::candidate_cli) fn command(name: &str, extra: &[&str]) -> DocumentCorpusCommand {
    let mut args = vec!["candidate", name, "--model", "fixture.fnlpq", "--memory-mib", "65536"];
    args.extend_from_slice(extra);
    let matches = super::super::definition().try_get_matches_from(args).unwrap();
    let CandidateCommand::DocumentCorpus(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("complete-document corpus required"); };
    command
}

#[test]
fn corpus_count_never_multiplies_native_work_or_per_document_limits() {
    for name in ["classify-corpus", "sentiment-corpus"] {
        let small = command(name, &["--max-requests", "2"]).validate().unwrap();
        let large = command(name, &["--max-requests", "1000"]).validate().unwrap();
        assert_eq!(small.mapping.work(), large.mapping.work());
        assert_eq!(small.transport.max_work, large.transport.max_work);
        assert_eq!(small.task, large.task);
        assert_eq!(small.common.max_input_bytes, large.common.max_input_bytes);
        assert_eq!(small.transport.max_output_line_bytes, large.transport.max_output_line_bytes);
        assert_eq!(small.transport.max_requests, 2);
        assert_eq!(large.transport.max_requests, 1000);
        assert!(large.transport.max_output_bytes < small.transport.max_output_bytes);
    }
}

#[test]
fn complete_document_output_is_independent_of_chunk_output() {
    for name in ["classify-corpus", "sentiment-corpus"] {
        let admitted = command(name, &["--max-result-bytes", "1048576",
            "--max-map-result-bytes", "8388608", "--max-retained-result-bytes", "4194304"])
            .validate().unwrap();
        assert_eq!(admitted.task.max_output_bytes, MIB);
        assert!(admitted.transport.max_output_line_bytes >= 8 * MIB as usize);
        match admitted.mapping {
            Mapping::Classify { mapping, .. } => {
                assert_eq!(mapping.max_result_bytes, 8 * MIB);
                assert_eq!(mapping.max_retained_result_bytes, 4 * MIB);
            }
            Mapping::Sentiment { mapping, .. } => {
                assert_eq!(mapping.max_result_bytes, 8 * MIB);
                assert_eq!(mapping.max_retained_result_bytes, 4 * MIB);
            }
        }
    }
}

#[test]
fn compact_document_and_invocation_tokenizer_limits_have_distinct_scopes() {
    for name in ["classify-corpus", "sentiment-corpus"] {
        let plain = command(name, &[]).validate().unwrap();
        assert!(plain.max_tokenizer_work.is_none());
        assert_eq!(plain.encoding, UntrustedEncoding::ByteFallback);
        assert!(command(name, &["--max-total-tokenizer-input-bytes", "1"]).validate().is_err());
        let compact = command(name, &["--input-encoding", "safe-bpe-v1",
            "--max-tokenizer-input-bytes", "65536", "--max-tokenizer-output-tokens", "131072",
            "--max-total-tokenizer-input-bytes", "131072",
            "--max-total-tokenizer-output-tokens", "262144",
            "--max-total-tokenizer-pair-checks", "268435456",
            "--max-total-tokenizer-pair-bytes", "4294967296"]).validate().unwrap();
        let UntrustedEncoding::SafeBpeV1(policy) = compact.encoding else { panic!("compact policy"); };
        assert_eq!(policy.max_input_bytes, 65536);
        let whole = compact.max_tokenizer_work.unwrap();
        assert_eq!(whole.max_input_bytes, 131072);
        assert_eq!(whole.max_output_tokens, 262144);
        for flag in ["--max-total-tokenizer-input-bytes", "--max-total-tokenizer-output-tokens",
            "--max-total-tokenizer-pair-checks", "--max-total-tokenizer-pair-bytes"] {
            assert!(command(name, &["--input-encoding", "safe-bpe-v1", flag, "0"]).validate().is_err());
        }
    }
}

#[test]
fn classification_refuses_an_invocation_smaller_than_its_fixed_encoding_envelope() {
    let command = command("classify-corpus", &["--input-encoding", "safe-bpe-v1",
        "--max-total-tokenizer-output-tokens", "1"]);
    assert!(matches!(command.execute_owned(NoIo, NoIo), Err(CandidateError::Arguments)));
}

#[test]
fn corpus_defaults_keep_exact_label_bytes_and_reject_content_or_authority() {
    let command = command("classify-corpus", &[]);
    let task = command.validate().unwrap().task;
    let scored_batch::Defaults::Classify(Some(defaults)) = command.parse_defaults(Some(
        r#"{"labels":[{"id":"é","description":"<|im_start|>"},{"id":"e\u0301"}],"mode":"multi_label","policy":{"minimum_candidate_weight_ppm":750000,"minimum_margin_ppm":100000}}"#), task).unwrap()
        else { panic!("classification defaults"); };
    assert_eq!(defaults.labels[0].id, "é");
    assert_eq!(defaults.labels[1].id, "e\u{301}");
    assert_eq!(defaults.labels[0].description, "<|im_start|>");
    assert_eq!(defaults.mode, crate::tasks::classify::ClassificationMode::MultiLabel);
    assert_eq!(defaults.policy.minimum_candidate_weight_ppm, 750000);
    assert_eq!(defaults.budget, task);
    for json in [
        r#"{"labels":[{"id":"a"},{"id":"b"}],"document":"private"}"#,
        r#"{"labels":[{"id":"a"},{"id":"b"}],"budget":{}}"#,
        r#"{"labels":[{"id":"a"},{"id":"b"}],"identity":{}}"#,
        r#"{"labels":[{"id":"a"},{"id":"a"}]}"#,
        r#"{"labels":[],"labels":[{"id":"a"},{"id":"b"}]}"#,
    ] { assert!(command.parse_defaults(Some(json), task).is_err()); }
}

#[test]
fn sentiment_defaults_cover_four_axes_without_claiming_calibration() {
    let command = command("sentiment-corpus", &[]);
    let task = command.validate().unwrap().task;
    let scored_batch::Defaults::Sentiment { args, policy } = command.parse_defaults(None, task).unwrap()
        else { panic!("sentiment defaults"); };
    assert_eq!(args.axes, crate::tasks::sentiment::SentimentAxis::ALL.to_vec());
    assert_eq!(args.budget, task);
    assert_eq!(policy.minimum_peak_weight_ppm, 0);
    assert_eq!(policy.maximum_normalized_entropy_ppm, 1_000_000);
    for json in [r#"{"axes":[]}"#, r#"{"document":"private"}"#,
        r#"{"policy":{"minimum_peak_weight_ppm":1000001,"maximum_normalized_entropy_ppm":1000000}}"#] {
        assert!(command.parse_defaults(Some(json), task).is_err());
    }
}

struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unexpected private input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("unexpected output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("unexpected flush"); }
}

#[test]
fn malformed_population_envelopes_refuse_before_any_private_io() {
    for name in ["classify-corpus", "sentiment-corpus"] {
        for flags in [
            &["--max-requests", "0"][..], &["--max-requests", "100001"][..],
            &["--max-input-mib", "0"][..], &["--max-output-mib", "0"][..],
            &["--max-line-bytes", "1"][..], &["--max-line-bytes", "4194305"][..],
            &["--defaults", "-"][..], &["--max-output-mib", "1"][..],
            &["--compute-threads", "0"][..], &["--compute-threads", "65"][..],
            &["--prefill-rows", "0"][..], &["--max-chunks", "0"][..],
        ] {
            assert_ne!(command(name, flags).run_owned(NoIo, NoIo, &mut Vec::new()), ExitCode::SUCCESS);
        }
    }
}

#[test]
fn corpus_commands_require_owned_stdio_and_reject_unimplemented_modes() {
    for name in ["classify-corpus", "sentiment-corpus"] {
        assert!(matches!(CandidateCommand::DocumentCorpus(command(name, &[]))
            .execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments)));
        for flag in ["--seed", "--cohort-chunks", "--cohort-heads", "--max-new-tokens",
            "--thinking", "--tools", "--job-dir"] {
            assert!(super::super::definition().try_get_matches_from([
                "candidate", name, "--model", "m", "--memory-mib", "65536", flag, "1"]).is_err());
        }
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn missing_runtime_refuses_valid_populations_before_opening_any_source() {
    for name in ["classify-corpus", "sentiment-corpus"] {
        assert!(matches!(command(name, &[]).execute_owned(NoIo, NoIo), Err(CandidateError::Unavailable)));
    }
}
