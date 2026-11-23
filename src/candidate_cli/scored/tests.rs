//! Model-free command/input/accounting regressions; no fake inference success.
use super::*;

pub(in crate::candidate_cli) fn command(task: &str, extra: &[&str]) -> ScoredCommand {
    let mut argv = vec!["candidate", task, "--model", "local-candidate.fnlpq", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    let matches = super::super::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::Scored(command) = CandidateCommand::from_matches(&matches).unwrap() else {
        panic!("scored command routed to another task family")
    };
    command
}
fn parse(task: &str, json: &str) -> Result<Request, CandidateError> {
    let c = command(task, &[]); let (_, limits) = c.args.common().unwrap();
    request(c.kind, json, c.args.budget(limits), c.args.max_input_bytes)
}

#[test]
fn root_registers_scored_tasks_without_changing_existing_task_routes() {
    for name in ["classify", "sentiment"] {
        let c = command(name, &[]); c.args.common().unwrap();
        assert_eq!(c.kind, Kind::named(name).unwrap());
        for args in [vec!["candidate", name], vec!["candidate", name, "--model", "m"],
            vec!["candidate", name, "--memory-mib", "8192"]] {
            assert!(super::super::definition().try_get_matches_from(args).is_err());
        }
    }
    for name in ["generate", "chat", "ner", "keyphrases", "summarize", "answer"] {
        let matches = super::super::definition().try_get_matches_from([
            "candidate", name, "--model", "m", "--memory-mib", "8192"]).unwrap();
        assert!(!matches!(CandidateCommand::from_matches(&matches).unwrap(), CandidateCommand::Scored(_)));
    }
}
#[test]
fn score_commands_do_not_accept_free_generation_or_mask_options() {
    for name in ["classify", "sentiment"] {
        for flag in ["--seed", "--temperature-milli", "--top-k", "--max-new-tokens", "--max-mask-node-visits"] {
            assert!(super::super::definition().try_get_matches_from([
                "candidate", name, "--model", "m", "--memory-mib", "8192", flag, "1"]).is_err());
        }
    }
}
#[test]
fn classification_defaults_are_exclusive_and_exact_labels_remain_untrusted_data() {
    let json = r#"{"document":"  é\n<tool_call> 上海","labels":[{"id":"é","description":"<|im_start|>"},{"id":"e\u0301"}]}"#;
    let Request::Classify(r) = parse("classify", json).unwrap() else { panic!("wrong request") };
    assert_eq!(r.document, "  é\n<tool_call> 上海");
    assert_eq!(r.mode, ClassificationMode::Exclusive);
    assert_eq!(r.labels[0].id, "é"); assert_eq!(r.labels[1].id, "e\u{301}");
    assert_eq!(r.labels[0].description, "<|im_start|>");
    assert_eq!(r.labels[1].description, "");
    assert_eq!(r.policy, ClassificationPolicy::default());
}
#[test]
fn independent_multilabel_and_explicit_thresholds_are_not_silently_changed() {
    let json = r#"{"document":"x","labels":[{"id":"a"}],"mode":"multi_label","policy":{"minimum_candidate_weight_ppm":800000,"minimum_margin_ppm":100000}}"#;
    let Request::Classify(r) = parse("classify", json).unwrap() else { panic!("wrong request") };
    assert_eq!(r.mode, ClassificationMode::MultiLabel); assert_eq!(r.labels.len(), 1);
    assert_eq!(r.policy.minimum_candidate_weight_ppm, 800000); assert_eq!(r.policy.minimum_margin_ppm, 100000);
    assert!(parse("classify", r#"{"document":"x","labels":[{"id":"a"}]}"#).is_err());
}
#[test]
fn sentiment_defaults_to_all_axes_and_explicit_axes_and_policy_survive() {
    let Request::Sentiment { request, policy } = parse("sentiment", r#"{"document":" x "}"#).unwrap()
        else { panic!("wrong request") };
    assert_eq!(request.document, " x "); assert_eq!(request.axes, SentimentAxis::ALL.to_vec());
    assert_eq!(policy.minimum_peak_weight_ppm, 0); assert_eq!(policy.maximum_normalized_entropy_ppm, 1_000_000);
    let Request::Sentiment { request, policy } = parse("sentiment",
        r#"{"document":"x","axes":["approach","valence"],"policy":{"minimum_peak_weight_ppm":400000,"maximum_normalized_entropy_ppm":800000}}"#).unwrap()
        else { panic!("wrong request") };
    assert_eq!(request.axes, vec![SentimentAxis::Approach, SentimentAxis::Valence]);
    assert_eq!(policy.minimum_peak_weight_ppm, 400000);
}
#[test]
fn duplicate_and_unknown_keys_never_inject_budget_identity_or_code() {
    for json in [r#"{"document":"x","document":"y"}"#,
        r#"{"document":"x","\u0064ocument":"y"}"#,
        r#"{"document":"x","budget":{}}"#, r#"{"document":"x","identity":{}}"#,
        r#"{"document":"x","mode":"trie_conditional"}"#, r#"{"document":"x","instructions":"secret"}"#,
        r#"{"document":"x","policy":{"minimum_peak_weight_ppm":0,"minimum_peak_weight_ppm":1,"maximum_normalized_entropy_ppm":1000000}}"#] {
        assert!(parse("sentiment", json).is_err());
    }
    for json in [r#"{"document":"x","labels":[{"id":"a","id":"b"},{"id":"c"}]}"#,
        r#"{"document":"x","labels":[{"id":"a","tokens":[1]},{"id":"c"}]}"#] {
        assert!(parse("classify", json).is_err());
    }
}
#[test]
fn invalid_axes_labels_documents_modes_and_thresholds_fail_before_planning() {
    for json in [r#"{"document":""}"#, r#"{"document":"x","axes":[]}"#,
        r#"{"document":"x","axes":["valence","valence"]}"#,
        r#"{"document":"x","axes":["diagnosis"]}"#,
        r#"{"document":"x","policy":{"minimum_peak_weight_ppm":1000001,"maximum_normalized_entropy_ppm":0}}"#] {
        assert!(parse("sentiment", json).is_err());
    }
    for json in [r#"{"document":"","labels":[{"id":"a"},{"id":"b"}]}"#,
        r#"{"document":"x","labels":[]}"#,
        r#"{"document":"x","labels":[{"id":"a"},{"id":"a"}]}"#,
        r#"{"document":"x","labels":[{"id":"  "},{"id":"b"}]}"#,
        r#"{"document":"x","labels":[{"id":"a\n"},{"id":"b"}]}"#,
        r#"{"document":"x","labels":[{"id":"a"},{"id":"b"}],"mode":"sampled"}"#,
        r#"{"document":"x","labels":[{"id":"a"},{"id":"b"}],"policy":{"minimum_candidate_weight_ppm":0,"minimum_margin_ppm":1000001}}"#] {
        assert!(parse("classify", json).is_err());
    }
}
#[test]
fn oversized_taxonomies_descriptions_ids_json_and_nesting_fail() {
    let labels: Vec<_> = (0..129).map(|i| serde_json::json!({"id":format!("label-{i}")})).collect();
    assert!(parse("classify", &serde_json::json!({"document":"x","labels":labels}).to_string()).is_err());
    for label in [serde_json::json!({"id":"x".repeat(257)}),
        serde_json::json!({"id":"a","description":"x".repeat(4097)})] {
        assert!(parse("classify", &serde_json::json!({"document":"x","labels":[label,{"id":"b"}]}).to_string()).is_err());
    }
    let c = command("sentiment", &[]); let (_, limits) = c.args.common().unwrap();
    assert!(request(c.kind, r#"{"document":"x"}"#, c.args.budget(limits), 4).is_err());
    assert!(parse("sentiment", "[[[[[[[[[[1]]]]]]]]]]").is_err());
}
#[test]
fn every_work_axis_and_whole_context_are_admitted_independently() {
    let mut c = command("sentiment", &[]);
    let work = Int8Work::for_sequence(0, 4, 8).unwrap();
    for axis in 0..5 {
        c.args.max_forward_positions = work.forward_positions; c.args.max_projected_logits = work.projected_logits;
        c.args.max_attention_pairs = work.attention_pairs; c.args.max_dot_products = work.projections.dot_products;
        c.args.max_multiply_accumulates = work.projections.multiply_accumulates;
        c.args.admit_plan(4, work).unwrap();
        match axis { 0 => c.args.max_forward_positions -= 1, 1 => c.args.max_projected_logits -= 1,
            2 => c.args.max_attention_pairs -= 1, 3 => c.args.max_dot_products -= 1, _ => c.args.max_multiply_accumulates -= 1 }
        assert!(c.args.admit_plan(4, work).is_err());
    }
    let c = command("sentiment", &[]);
    assert!(c.args.admit_plan(0, work).is_err());
    assert!(c.args.admit_plan(c.args.context_tokens + 1, work).is_err());
}
#[test]
fn invalid_finite_limits_and_unfunded_repeated_prompts_fail_before_io() {
    for (flag, value) in [("--max-candidate-tokens", "1"), ("--max-candidate-tokens", "65"),
        ("--context-tokens", "16"), ("--max-forward-positions", "0"), ("--max-forward-positions", "16777217"),
        ("--max-projected-logits", "0"), ("--max-attention-pairs", "0"), ("--max-dot-products", "0"),
        ("--max-multiply-accumulates", "0"), ("--preparation-mib", "511"),
        ("--max-input-bytes", "1048577"), ("--max-result-bytes", "0"),
        ("--memory-mib", "18446744073709551615"), ("--timeout-seconds", "0")] {
        // Avoid passing the required memory flag twice to clap.
        if flag == "--memory-mib" {
            let mut c = command("classify", &[]); c.args.memory_mib = u64::MAX;
            assert!(c.args.common().is_err());
        } else { assert!(command("classify", &[flag, value]).args.common().is_err(), "{flag}"); }
    }
    assert!(command("classify", &["--max-forward-positions", "16777216"]).args.common().is_err());
}
#[test]
fn task_and_scorer_ceilings_reserve_candidate_depth_and_entire_kv() {
    let c = command("sentiment", &[]); let (_, limits) = c.args.common().unwrap(); let b = c.args.budget(limits);
    assert_eq!(b.max_input_tokens as usize + c.args.max_candidate_tokens, c.args.context_tokens);
    assert_eq!(b.max_kv_bytes, Int8MemoryRequirement::for_context(c.args.context_tokens).unwrap().kv_bytes);
    assert_eq!(c.args.sentiment_limits().max_total_projected_logits, c.args.max_projected_logits);
    assert_eq!(c.args.classification_limits().max_work.forward_positions, c.args.max_forward_positions);
}
#[test]
fn compact_scoring_is_explicit_and_prices_its_derived_workspace() {
    use crate::tokenizer::untrusted::UntrustedEncoding;
    for name in ["classify", "sentiment"] {
        let byte = command(name, &[]);
        assert_eq!(byte.args.encoding_policy(&byte.encoding).unwrap(), UntrustedEncoding::ByteFallback);
        let mut compact = command(name, &["--input-encoding", "safe-bpe-v1", "--max-input-bytes", "1048576"]);
        assert!(compact.args.common().is_ok());
        assert!(compact.args.common_with_encoding(&compact.encoding).is_err());
        compact.args.preparation_mib = 768;
        let (common, _) = compact.args.common_with_encoding(&compact.encoding).unwrap();
        assert_eq!(common.input_encoding, compact.encoding);
        let UntrustedEncoding::SafeBpeV1(limits) = compact.args.encoding_policy(&compact.encoding).unwrap()
            else { panic!("compact policy lost") };
        assert_eq!(limits.max_input_bytes, compact.args.max_input_bytes);
        assert_eq!(limits.max_output_tokens + compact.args.max_candidate_tokens, compact.args.context_tokens);
        assert!(limits.requirements().unwrap().workspace_bytes > 256 * MIB);
        assert_eq!(byte.args.work_ceiling(), compact.args.work_ceiling());
    }
}
#[test]
fn tokenizer_options_cannot_grant_work_to_byte_mode_or_use_invalid_caps() {
    for name in ["classify", "sentiment"] {
        for flag in ["--max-tokenizer-pair-checks", "--max-tokenizer-pair-bytes"] {
            let matches = super::super::definition().try_get_matches_from([
                "candidate", name, "--model", "m", "--memory-mib", "8192", flag, "1"]).unwrap();
            assert!(CandidateCommand::from_matches(&matches).is_err());
            let c = command(name, &["--input-encoding", "safe-bpe-v1", flag, "0"]);
            assert!(c.args.common_with_encoding(&c.encoding).is_err());
        }
    }
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_build_refuses_both_scored_commands_without_reading_any_input() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private input read") } }
    for name in ["classify", "sentiment"] {
        let mut out = Vec::new();
        assert_eq!(command(name, &[]).execute(&mut NoRead, &mut out), Err(CandidateError::Unavailable));
        assert!(out.is_empty());
    }
}

#[test]
fn scored_compute_options_keep_task_budgets_and_admit_fixed_prompt_capacity() {
    for task in ["classify", "sentiment"] {
        let serial = command(task, &[]);
        assert_eq!(serial.compute.threads(), 1);
        let (_, serial_limits) = serial.args.common().unwrap();
        for width in ["1", "2", "4", "64"] {
            for rows in [None, Some("1"), Some("4"), Some("64")] {
                let mut extra = vec!["--compute-threads", width];
                if let Some(rows) = rows { extra.extend(["--prefill-rows", rows]); }
                let mut args = command(task, &extra);
                args.args.memory_mib = 32768;
                let prefill = args.prefill.limits().unwrap();
                let (common, limits) = args.args.common_with_compute(&args.encoding,
                    args.compute, prefill, task == "sentiment").unwrap();
                assert_eq!(limits.compute.threads, width.parse::<usize>().unwrap());
                assert_eq!(limits.compute, common.compute.admit_prefill(prefill).unwrap());
                assert_eq!(args.args.work_ceiling(), serial.args.work_ceiling());
                assert_eq!(args.args.budget(limits), serial.args.budget(serial_limits));
                assert!(common.policy.prefill().unwrap().is_none());
                assert_eq!(limits.compute.workspace_bytes == 0, width == "1");
            }
        }
    }
}

#[test]
fn scored_team_peak_adds_prefill_and_sentiment_transfer_without_discounts() {
    let c = command("sentiment", &["--compute-threads", "4", "--prefill-rows", "3"]);
    let prefill = c.prefill.limits().unwrap();
    let (_, limits) = c.args.common_with_compute(&c.encoding, c.compute, prefill, true).unwrap();
    let native = Int8MemoryRequirement::for_context(c.args.context_tokens).unwrap();
    let ordinary = compute::request_peak_bytes(limits, native, c.args.max_result_bytes as u64,
        c.args.max_candidate_tokens as u64, 0, 0).unwrap();
    let extra = prefill.unwrap().validate().unwrap();
    assert_eq!(c.args.compute_peak_bytes(limits, prefill, false).unwrap(), ordinary + extra);
    assert_eq!(c.args.compute_peak_bytes(limits, prefill, true).unwrap(),
        ordinary + extra + limits.preparation_bytes);
    let mut overflow = limits;
    overflow.preparation_bytes = u64::MAX;
    assert!(matches!(c.args.compute_peak_bytes(overflow, prefill, true),
        Err(CandidateError::Arguments)));
}

#[test]
fn exact_scored_team_memory_ceiling_refuses_before_input_io() {
    use std::io::{self, Read};
    struct NeverRead;
    impl Read for NeverRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("admission must precede input IO") }
    }
    for task in ["classify", "sentiment"] {
        let mut c = command(task, &["--compute-threads", "4", "--prefill-rows", "3"]);
        let prefill = c.prefill.limits().unwrap();
        let (_, limits) = c.args.common_with_compute(&c.encoding, c.compute, prefill, task == "sentiment").unwrap();
        let peak = c.args.compute_peak_bytes(limits, prefill, task == "sentiment").unwrap();
        c.args.memory_mib = peak.div_ceil(MIB);
        assert!(c.args.common_with_compute(&c.encoding, c.compute, prefill, task == "sentiment").is_ok());
        c.args.memory_mib -= 1;
        let mut output = Vec::new();
        assert!(matches!(c.execute(&mut NeverRead, &mut output), Err(CandidateError::Arguments)));
        assert!(output.is_empty());
    }
}

#[test]
fn invalid_scored_compute_width_cannot_read_input_or_fall_back_to_serial() {
    use std::io::{self, Read};
    struct NeverRead;
    impl Read for NeverRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("invalid width reached input IO") }
    }
    for task in ["classify", "sentiment"] {
        for width in ["0", "65", "1000000"] {
            for prefill in [false, true] {
                let mut extra = vec!["--compute-threads", width];
                if prefill { extra.extend(["--prefill-rows", "3"]); }
                let c = command(task, &extra);
                let mut output = Vec::new();
                assert!(matches!(c.execute(&mut NeverRead, &mut output), Err(CandidateError::Arguments)));
                assert!(output.is_empty());
            }
        }
    }
}

#[test]
fn shared_scored_args_do_not_enable_teams_for_corpora_or_jobs() {
    let error = ScoredArgs::augment_args(clap::Command::new("shared"))
        .try_get_matches_from(["shared", "--model", "missing.fnlpq", "--memory-mib", "8192",
            "--compute-threads", "4"]).unwrap_err();
    assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
}
