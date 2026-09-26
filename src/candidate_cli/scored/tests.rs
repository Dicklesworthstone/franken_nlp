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
