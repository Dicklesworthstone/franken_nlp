use super::*;
use crate::tokenizer::untrusted::UntrustedEncoding;

fn limits(command: &Int4Command) -> Result<Int4Limits, CandidateError> {
    match command.task {
        Int4Task::Classify(scoring) => scoring.validate_one_off_with_prefill(&command.args),
        Int4Task::Sentiment(sentiment) => sentiment.validate(&command.args),
        _ => panic!("expected one-off scored task"),
    }
}

#[test]
fn compact_scored_input_is_explicit_and_independent_of_head_scheduling() {
    for task in ["classify", "sentiment"] {
        for schedule in [&[][..], &["--prefill-rows", "4"][..], &["--cohort-heads", "4"][..]] {
            let mut flags = vec!["--input-encoding", "safe-bpe-v1",
                "--max-tokenizer-pair-checks", "123456", "--max-tokenizer-pair-bytes", "987654"];
            flags.extend_from_slice(schedule);
            let compact = command(task, &flags);
            let admitted = limits(&compact).unwrap();
            let UntrustedEncoding::SafeBpeV1(policy) = compact.args.input_encoding
                .policy(compact.args.max_input_bytes, admitted.max_prompt_tokens).unwrap()
                else { panic!("compact policy was lost"); };
            assert_eq!(policy.max_pair_checks, 123456);
            assert_eq!(policy.max_pair_bytes, 987654);
            assert_eq!(policy.max_input_bytes, compact.args.max_input_bytes);
            assert_eq!(policy.max_output_tokens, admitted.max_prompt_tokens);
            assert!(policy.requirements().unwrap().workspace_bytes > 0);
        }
        let default = command(task, &[]);
        let explicit_byte = command(task, &["--input-encoding", "byte-fallback"]);
        assert!(default.args.input_encoding.is_default());
        assert!(explicit_byte.args.input_encoding.is_default());
        assert_eq!(limits(&default).unwrap().minimum_memory_bytes,
            limits(&explicit_byte).unwrap().minimum_memory_bytes);
    }
}

#[test]
fn compact_scored_workspace_and_invalid_pair_limits_refuse_before_private_io() {
    for task in ["classify", "sentiment"] {
        // The ordinary byte policy fits the default preparation reserve. The
        // additional one-MiB safe-BPE workspace must be admitted separately.
        limits(&command(task, &["--max-input-bytes", "1048576"])).unwrap();
        for flags in [
            &["--input-encoding", "safe-bpe-v1", "--max-input-bytes", "1048576"][..],
            &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "0"][..],
            &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-bytes", "0"][..],
            &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "18446744073709551615"][..],
        ] {
            let mut output = Vec::new();
            assert_eq!(command(task, flags).execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
            assert!(output.is_empty());
        }
        limits(&command(task, &["--input-encoding", "safe-bpe-v1", "--max-input-bytes", "1048576",
            "--preparation-mib", "1024"])).unwrap();
    }
}

#[test]
fn compact_scoring_survives_pinned_snapshot_selection() {
    for task in ["classify", "sentiment"] {
        let pin = "ab".repeat(32);
        let matches = crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "source-int4", task, "--snapshot", "private.snapshot",
            "--expected-model-sha256", &pin, "--memory-mib", "8192",
            "--input-encoding", "safe-bpe-v1",
        ]).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong snapshot dispatcher"); };
        assert!(limits(&command).unwrap().snapshot_pin.is_some());
        assert!(!command.args.input_encoding.is_default());
    }
}

#[test]
fn unwired_scored_workflows_cannot_inherit_a_compact_input_contract() {
    let compact = command("classify", &["--input-encoding", "safe-bpe-v1"]);
    let Int4Task::Classify(scoring) = compact.task else { panic!("wrong task"); };
    assert!(scoring.validate(&compact.args).is_err());
    assert!(scoring.validate_batch(&compact.args).is_err());
    assert!(scoring.validate_with_prefill(&compact.args).is_err());
    for task in ["resolve", "redact", "faithfulness-map", "faithfulness-corpus"] {
        let error = crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "source-int4", task, "--source-dir", "private-source",
            "--source-manifest", "private-manifest", "--memory-mib", "8192",
            "--input-encoding", "safe-bpe-v1",
        ]).unwrap_err();
        assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
    }
    for task in ["classify", "sentiment"] {
        let matches = crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "source-int4", task, "--source-dir", "private-source",
            "--source-manifest", "private-manifest", "--memory-mib", "8192",
            "--max-tokenizer-pair-checks", "100",
        ]).unwrap();
        assert!(CandidateCommand::from_matches(&matches).is_err());
    }
}
