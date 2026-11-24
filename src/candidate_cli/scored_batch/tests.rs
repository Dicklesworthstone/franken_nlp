//! Parser/resource contracts only; no fixture performs neural inference.
use super::*;

pub(in crate::candidate_cli) fn command(task: &str, extra: &[&str]) -> ScoreBatchCommand {
    let mut argv = vec!["candidate", "score-batch", "--task", task, "--model", "local.fnlpq", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    let matches = super::super::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::ScoreBatch(c) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("scored stream routed as a single request") };
    c
}
fn task(c: &ScoreBatchCommand) -> TaskBudget { c.host.budget(c.validate().unwrap().1) }
const LABELS: &str = r#"{"labels":[{"id":"complaint"},{"id":"question"}]}"#;
#[test]
fn owned_dispatch_requires_explicit_task_model_and_memory_authority() {
    for name in ["classify", "sentiment"] { command(name, &[]).validate().unwrap(); }
    for argv in [vec!["candidate", "score-batch"], vec!["candidate", "score-batch", "--task", "classify"],
        vec!["candidate", "score-batch", "--model", "m", "--memory-mib", "8192"]] {
        assert!(super::super::definition().try_get_matches_from(argv).is_err());
    }
    assert!(super::super::definition().try_get_matches_from(["candidate", "batch", "--task", "ner",
        "--model", "m", "--memory-mib", "8192"]).is_ok());
}
#[test]
fn score_streams_do_not_accept_generation_or_source_mask_overrides() {
    for flag in ["--seed", "--temperature-milli", "--max-new-tokens", "--max-mask-node-visits", "--schema"] {
        assert!(definition().try_get_matches_from(["score-batch", "--task", "classify", "--model", "m",
            "--memory-mib", "8192", flag, "1"]).is_err());
    }
}
#[test]
fn more_records_do_not_multiply_or_refresh_compute_authority() {
    let a = command("sentiment", &["--max-requests", "10"]);
    let b = command("sentiment", &["--max-requests", "1000"]);
    assert_eq!(a.host.work_ceiling(), b.host.work_ceiling());
    assert_eq!(a.validate().unwrap().2.transport.max_work, b.validate().unwrap().2.transport.max_work);
    assert_eq!(a.validate().unwrap().2.transport.max_work.forward_positions, a.host.max_forward_positions);
}
#[test]
fn every_native_and_transport_axis_is_checked_before_io() {
    for (flag, value) in [("--max-forward-positions", "0"), ("--max-projected-logits", "0"),
        ("--max-attention-pairs", "0"), ("--max-dot-products", "0"), ("--max-multiply-accumulates", "0"),
        ("--max-requests", "0"), ("--max-requests", "100001"), ("--max-input-mib", "0"),
        ("--max-output-mib", "1"), ("--max-line-bytes", "1"), ("--max-line-bytes", "4194305"),
        ("--defaults", "-"), ("--preparation-mib", "1")] {
        assert!(command("classify", &[flag, value]).validate().is_err(), "{flag}");
    }
}
#[test]
fn candidate_framing_and_owned_buffers_are_priced_separately() {
    let c = command("sentiment", &[]); let e = c.validate().unwrap().2;
    assert_eq!(e.transport.max_output_bytes + (c.max_requests + 4) * FRAME_ALLOWANCE, e.output_bytes);
    assert!(e.io_bytes >= e.transport.max_output_line_bytes as u64 + 2 * IO_BUFFER_BYTES as u64);
    assert_eq!(e.transport.max_document_bytes, c.host.max_input_bytes);
}
#[test]
fn compact_score_streams_keep_whole_run_authority_and_reserve_encoder_memory() {
    for name in ["classify", "sentiment"] {
        let byte = command(name, &[]);
        let compact = command(name, &["--input-encoding", "safe-bpe-v1"]);
        let (common, _, envelope) = compact.validate().unwrap();
        assert_eq!(common.input_encoding, compact.encoding);
        assert_eq!(compact.host.work_ceiling(), byte.host.work_ceiling());
        assert_eq!(envelope.transport.max_work, byte.validate().unwrap().2.transport.max_work);
        let mut larger = command(name, &["--input-encoding", "safe-bpe-v1", "--max-input-bytes", "1048576"]);
        assert!(larger.validate().is_err());
        larger.host.preparation_mib = 768;
        assert!(larger.validate().is_ok());
    }
}
#[test]
fn compact_judgment_and_invalid_tokenizer_work_refuse_before_any_io() {
    struct Never;
    impl Read for Never { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("input read") } }
    impl Write for Never {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output write") }
        fn flush(&mut self) -> io::Result<()> { panic!("output flush") }
    }
    let c = command("judge", &["--input-encoding", "safe-bpe-v1", "--defaults", "never-open.json"]);
    assert_eq!(c.execute_owned(Never, Never), Err(CandidateError::Arguments));
    for name in ["classify", "sentiment"] {
        for flag in ["--max-tokenizer-pair-checks", "--max-tokenizer-pair-bytes"] {
            let c = command(name, &["--input-encoding", "safe-bpe-v1", flag, "0"]);
            assert_eq!(c.execute_owned(Never, Never), Err(CandidateError::Arguments));
        }
    }
}
#[test]
fn classification_defaults_keep_exact_labels_and_host_owned_budget() {
    let c = command("classify", &[]); let budget = task(&c);
    let Defaults::Classify(Some(d)) = c.parse_defaults(Some(LABELS), budget).unwrap() else { panic!("wrong settings") };
    assert_eq!(d.labels[0].id, "complaint"); assert_eq!(d.mode, ClassificationMode::Exclusive);
    assert_eq!(d.budget, budget); assert_eq!(d.policy, ClassificationPolicy::default());
    assert!(matches!(c.parse_defaults(None, budget).unwrap(), Defaults::Classify(None)));
    let one = r#"{"labels":[{"id":"é"}],"mode":"multi_label"}"#;
    assert!(c.parse_defaults(Some(one), budget).is_ok());
    let one_exclusive = r#"{"labels":[{"id":"é"}]}"#;
    assert!(c.parse_defaults(Some(one_exclusive), budget).is_err());
}
#[test]
fn duplicate_and_invalid_labels_or_policies_cannot_enter_defaults() {
    let c = command("classify", &[]); let b = task(&c);
    for json in [r#"{"labels":[]}"#, r#"{"labels":[{"id":"a"},{"id":"a"}]}"#,
        r#"{"labels":[{"id":" "},{"id":"b"}]}"#, r#"{"labels":[{"id":"a\n"},{"id":"b"}]}"#,
        r#"{"labels":[{"id":"a"},{"id":"b"}],"policy":{"minimum_candidate_weight_ppm":1000001,"minimum_margin_ppm":0}}"#] {
        assert!(c.parse_defaults(Some(json), b).is_err());
    }
    let json = serde_json::json!({"labels":[{"id":"a","description":"x".repeat(4097)},{"id":"b"}]}).to_string();
    assert!(c.parse_defaults(Some(&json), b).is_err());
}
#[test]
fn sentiment_defaults_bind_axes_and_run_policy_without_document_or_budget() {
    let c = command("sentiment", &[]); let b = task(&c);
    let Defaults::Sentiment { args, policy: p } = c.parse_defaults(None, b).unwrap() else { panic!("wrong settings") };
    assert_eq!(args.axes, SentimentAxis::ALL); assert_eq!(args.budget, b); assert_eq!(p, policy());
    let d = r#"{"axes":["valence"],"policy":{"minimum_peak_weight_ppm":700000,"maximum_normalized_entropy_ppm":800000}}"#;
    let Defaults::Sentiment { args, policy: p } = c.parse_defaults(Some(d), b).unwrap() else { panic!("wrong settings") };
    assert_eq!(args.axes, [SentimentAxis::Valence]); assert_eq!(p.minimum_peak_weight_ppm, 700000);
    for json in [r#"{"axes":[]}"#, r#"{"axes":["valence","valence"]}"#, r#"{"axes":["diagnosis"]}"#,
        r#"{"policy":{"minimum_peak_weight_ppm":0,"maximum_normalized_entropy_ppm":1000001}}"#] {
        assert!(c.parse_defaults(Some(json), b).is_err());
    }
}
#[test]
fn duplicate_keys_and_injected_identity_budget_or_source_fail_closed() {
    let c = command("sentiment", &[]); let b = task(&c);
    for json in [r#"{"axes":["valence"],"axes":["arousal"]}"#,
        r#"{"axes":["valence"],"\u0061xes":["arousal"]}"#,
        r#"{"document":"private"}"#, r#"{"budget":{}}"#, r#"{"identity":{}}"#,
        r#"{"labels":[{"id":"x"}]}"#] {
        assert!(c.parse_defaults(Some(json), b).is_err());
    }
    assert!(c.parse_defaults(Some(&" ".repeat(DEFAULTS_BYTES + 1)), b).is_err());
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_runtime_never_opens_defaults_or_reads_or_writes_streams() {
    struct Never;
    impl Read for Never { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("input read") } }
    impl Write for Never {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output write") }
        fn flush(&mut self) -> io::Result<()> { panic!("output flush") }
    }
    for name in ["classify", "sentiment"] {
        let c = command(name, &["--defaults", "never-open.json"]);
        assert_eq!(c.execute_owned(Never, Never), Err(CandidateError::Unavailable));
    }
}

#[test]
fn every_scored_stream_composes_threads_and_prompt_rows_without_changing_authority() {
    for task in ["classify", "sentiment", "judge"] {
        let serial = command(task, &[]); let (_, baseline, base_envelope) = serial.validate().unwrap();
        assert_eq!(serial.compute.threads(), 1); assert!(serial.prefill_limits().unwrap().is_none());
        for threads in ["1", "4", "64"] {
            for rows in [None, Some("1"), Some("4"), Some("64")] {
                let mut flags = vec!["--compute-threads", threads];
                if let Some(rows) = rows { flags.extend_from_slice(&["--prefill-rows", rows]); }
                let mut selected = command(task, &flags); selected.host.memory_mib = 32768;
                let (args, limits, envelope) = selected.validate().unwrap();
                let prefill = selected.prefill_limits().unwrap();
                assert_eq!(args.compute, selected.compute);
                assert_eq!(limits.compute, selected.compute.admit_prefill(prefill).unwrap());
                assert_eq!(limits.compute.threads, threads.parse::<usize>().unwrap());
                assert_eq!(prefill.map(|p| p.max_batch_rows), rows.map(|r| r.parse::<usize>().unwrap()));
                assert_eq!(selected.host.budget(limits), serial.host.budget(baseline));
                assert_eq!(selected.host.work_ceiling(), serial.host.work_ceiling());
                assert_eq!(envelope.transport.max_work, base_envelope.transport.max_work);
                assert_eq!(envelope.transport.max_requests, base_envelope.transport.max_requests);
            }
        }
    }
}

#[test]
fn combined_stream_admission_accepts_the_exact_peak_and_refuses_one_byte_less() {
    for task in ["classify", "sentiment", "judge"] {
        let c = command(task, &["--compute-threads", "4", "--prefill-rows", "4"]);
        let (_, mut limits, envelope) = c.validate().unwrap(); let prefill = c.prefill_limits().unwrap();
        let peak = c.peak_bytes(limits, envelope, prefill).unwrap();
        limits.memory_bytes = peak;
        c.check_scheduling_memory(limits, envelope, prefill).unwrap();
        limits.memory_bytes = peak - 1;
        assert_eq!(c.check_scheduling_memory(limits, envelope, prefill), Err(CandidateError::Arguments));
        limits.memory_bytes = peak;
        limits.compute.workspace_bytes -= 1;
        assert_eq!(c.check_scheduling_memory(limits, envelope, prefill), Err(CandidateError::Arguments));
    }
}

#[test]
fn packed_prompt_and_fixed_worker_memory_are_both_added_to_the_complete_stream_quote() {
    let serial = command("classify", &[]); let (_, base, envelope) = serial.validate().unwrap();
    let grouped = command("classify", &["--prefill-rows", "4"]);
    let (_, grouped_limits, _) = grouped.validate().unwrap(); let prefill = grouped.prefill_limits().unwrap();
    let threaded = command("classify", &["--compute-threads", "4", "--prefill-rows", "4"]);
    let (_, threaded_limits, _) = threaded.validate().unwrap();
    let baseline = serial.peak_bytes(base, envelope, None).unwrap();
    let prompt_bytes = prefill.unwrap().validate().unwrap();
    assert_eq!(grouped.peak_bytes(grouped_limits, envelope, prefill).unwrap(), baseline + prompt_bytes);
    assert_eq!(threaded.peak_bytes(threaded_limits, envelope, prefill).unwrap(),
        baseline + prompt_bytes + threaded_limits.compute.workspace_bytes);
    let mut larger = envelope; larger.io_bytes += 31;
    assert_eq!(threaded.peak_bytes(threaded_limits, larger, prefill).unwrap(),
        baseline + prompt_bytes + threaded_limits.compute.workspace_bytes + 31);
    larger.io_bytes = u64::MAX;
    assert!(threaded.peak_bytes(threaded_limits, larger, prefill).is_err());
}

#[test]
fn invalid_team_and_combined_memory_limits_refuse_before_defaults_model_or_stream_io() {
    struct Never;
    impl Read for Never { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("input read"); } }
    impl Write for Never {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output write"); }
        fn flush(&mut self) -> io::Result<()> { panic!("output flush"); }
    }
    for task in ["classify", "sentiment", "judge"] {
        for threads in ["0", "65", "999999"] {
            let c = command(task, &["--compute-threads", threads, "--prefill-rows", "4",
                "--defaults", "never-open.json"]);
            assert_eq!(c.execute_owned(Never, Never), Err(CandidateError::Arguments));
        }
        let mut c = command(task, &["--compute-threads", "4", "--prefill-rows", "4",
            "--defaults", "never-open.json"]);
        let (_, limits, envelope) = c.validate().unwrap();
        let peak = c.peak_bytes(limits, envelope, c.prefill_limits().unwrap()).unwrap();
        c.host.memory_mib = (peak - 1) / MIB;
        assert_eq!(c.execute_owned(Never, Never), Err(CandidateError::Arguments));
    }
}

#[test]
fn malformed_thread_counts_never_silently_select_a_serial_stream() {
    for task in ["classify", "sentiment", "judge"] {
        for count in ["-1", "1.5", "NaN", "999999999999999999999999999999999999"] {
            assert!(crate::candidate_cli::definition().try_get_matches_from(["candidate", "score-batch",
                "--task", task, "--model", "m", "--memory-mib", "8192",
                "--compute-threads", count]).is_err());
        }
    }
}

#[test]
fn live_and_retained_scoring_adapters_share_explicit_bounded_prefill_capability() {
    for task in ["classify", "sentiment", "judge"] {
        for rows in ["1", "4", "64"] {
            let c = command(task, &["--prefill-rows", rows]);
            let retained = corpus_prefill(c.kind().unwrap(), &c.prefill).unwrap().unwrap();
            assert_eq!(retained.max_batch_rows, c.prefill_limits().unwrap().unwrap().max_batch_rows);
            assert_eq!(retained.max_batch_rows, rows.parse::<usize>().unwrap());
        }
        for rows in ["0", "65"] {
            let c = command(task, &["--prefill-rows", rows]);
            assert!(corpus_prefill(c.kind().unwrap(), &c.prefill).is_err());
        }
    }
}
