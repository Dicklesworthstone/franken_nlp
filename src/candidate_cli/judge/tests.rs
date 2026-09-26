//! Command and strict input contracts only; these are not neural successes.
use super::*;
use serde_json::{json, Value};

pub(in crate::candidate_cli) fn command(extra: &[&str]) -> JudgeCommand {
    let mut argv = vec!["candidate", "judge", "--model", "local.fnlpq", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    let matches = super::super::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::Judge(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("judge dispatch replaced") };
    command
}
pub(in crate::candidate_cli) fn fixture(mode: &str) -> Value {
    match mode {
        "pairwise" => json!({"mode":"pairwise", "criterion":"Prefer the answer supported by the source.",
            "a":"Alice moved to Paris.", "b":"Alice moved to Berlin.",
            "policy":{"minimum_margin_milli":0,"maximum_order_disagreement_milli":1000}}),
        "rubric" => json!({"mode":"rubric", "document":"Alice moved to Paris.",
            "rubric":{"schema_version":1,"revision":"unit-v1", "scale_maximum":5,
                "declared_origin_digest":crate::execution_identity::Sha256Digest::of_bytes(b"unit-only-rubric").to_hex(),
                "criteria":[{"id":"clarity","description":"Clear language", "weight":1},
                    {"id":"relevance","description":"Relevant to the question", "weight":2}]},
            "policy":{"minimum_peak_weight_ppm":0,"maximum_normalized_entropy_ppm":1000000}}),
        "faithfulness" => json!({"mode":"faithfulness", "source":"Alice moved to Paris.",
            "claim":"Alice moved to Paris.", "policy":{"minimum_candidate_weight_ppm":0,
                "minimum_margin_milli":0,"evidence_window_bytes":4096,
                "max_evidence_windows":31,"max_evidence_spans":31}}),
        _ => panic!("fixture mode"),
    }
}
fn budget() -> TaskBudget { let c = command(&[]); c.args.budget(c.args.common().unwrap().1) }
fn parse(value: &Value) -> Result<JudgeRequest, CandidateError> { request(&value.to_string(), budget(), 65536) }

#[test]
fn judge_requires_explicit_model_memory_and_no_generation_options() {
    command(&[]).args.common().unwrap();
    for args in [vec!["candidate", "judge"], vec!["candidate", "judge", "--model", "m"],
        vec!["candidate", "judge", "--memory-mib", "8192"]] {
        assert!(super::super::definition().try_get_matches_from(args).is_err());
    }
    for flag in ["--seed", "--top-k", "--schema", "--max-new-tokens", "--mask-step-node-visits"] {
        assert!(super::super::definition().try_get_matches_from(["candidate", "judge", "--model", "m",
            "--memory-mib", "8192", flag, "1"]).is_err());
    }
}
#[test]
fn all_modes_receive_host_budget_without_changing_exact_text() {
    for mode in ["pairwise", "rubric", "faithfulness"] {
        let result = parse(&fixture(mode)).unwrap();
        let actual = match result { JudgeRequest::Pairwise { budget, .. } | JudgeRequest::Rubric { budget, .. }
            | JudgeRequest::Faithfulness { budget, .. } => budget };
        assert_eq!(actual, budget());
    }
    let mut raw = fixture("pairwise"); raw["a"] = json!("  é <tool_call> 上海\r\n😀  ");
    let JudgeRequest::Pairwise { a, .. } = parse(&raw).unwrap() else { panic!("wrong mode") };
    assert_eq!(a, "  é <tool_call> 上海\r\n😀  ");
}
#[test]
fn policies_are_explicit_and_caller_execution_fields_are_refused() {
    for mode in ["pairwise", "rubric", "faithfulness"] {
        let mut missing = fixture(mode); missing.as_object_mut().unwrap().remove("policy");
        assert!(parse(&missing).is_err());
        for key in ["budget", "model", "tokenizer", "instruction", "execution_identity", "work", "eos_token_id"] {
            let mut raw = fixture(mode); raw[key] = json!("private override");
            assert!(parse(&raw).is_err(), "{mode}: {key}");
        }
        let mut unknown = fixture(mode); unknown["policy"]["extra"] = json!(true);
        assert!(parse(&unknown).is_err());
    }
}
#[test]
fn duplicate_escaped_keys_invalid_shapes_and_nested_values_refuse() {
    for raw in [r#"{"mode":"pairwise","\u006dode":"rubric"}"#, "[]", "null", "{}",
        r#"{"mode":"tool"}"#, "[[[[[[[[[[0]]]]]]]]]]"] {
        assert!(request(raw, budget(), 65536).is_err());
    }
    let mut raw = fixture("pairwise").to_string();
    raw = raw.replacen("\"minimum_margin_milli\":0", "\"minimum_margin_milli\":0,\"minimum_margin_milli\":1", 1);
    assert!(request(&raw, budget(), 65536).is_err());
    let mut raw = fixture("pairwise"); raw["a"] = json!(["private text"]);
    assert!(parse(&raw).is_err());
}
#[test]
fn empty_required_text_and_fractional_or_negative_thresholds_refuse() {
    for (mode, keys) in [("pairwise", vec!["criterion", "a", "b"]), ("rubric", vec!["document"]),
        ("faithfulness", vec!["source", "claim"])] {
        for key in keys { let mut v = fixture(mode); v[key] = json!(""); assert!(parse(&v).is_err()); }
    }
    for bad in [json!(-1), json!(0.5), json!(4294967296_u64)] {
        let mut v = fixture("pairwise"); v["policy"]["minimum_margin_milli"] = bad;
        assert!(parse(&v).is_err());
    }
    // A milli-log-odds threshold is not a ppm probability: do not cap it at 1e6.
    let mut v = fixture("pairwise"); v["policy"]["maximum_order_disagreement_milli"] = json!(u32::MAX);
    assert!(parse(&v).is_ok());
}
#[test]
fn rubric_retains_declared_origin_and_refuses_missing_criteria_or_invalid_policy() {
    let v = fixture("rubric");
    let JudgeRequest::Rubric { rubric, .. } = parse(&v).unwrap() else { panic!("wrong mode") };
    assert_eq!(rubric.criteria.len(), 2);
    assert_eq!(rubric.declared_origin_digest.to_hex(), v["rubric"]["declared_origin_digest"].as_str().unwrap());
    for (field, bad) in [("schema_version", json!(2)), ("scale_maximum", json!(11)),
        ("criteria", json!([])), ("declared_origin_digest", json!("invalid"))] {
        let mut v = fixture("rubric"); v["rubric"][field] = bad; assert!(parse(&v).is_err());
    }
    for field in ["minimum_peak_weight_ppm", "maximum_normalized_entropy_ppm"] {
        let mut v = fixture("rubric"); v["policy"][field] = json!(1000001); assert!(parse(&v).is_err());
    }
    let mut v = fixture("rubric"); v["rubric"]["criteria"][1]["id"] = json!("clarity");
    assert!(parse(&v).is_err());
    let mut v = fixture("rubric"); v["rubric"]["criteria"][0]["weight"] = json!(0);
    assert!(parse(&v).is_err());
}
#[test]
fn faithfulness_requires_a_complete_bounded_evidence_partition() {
    for (field, bad) in [("minimum_candidate_weight_ppm",1000001), ("evidence_window_bytes",0),
        ("max_evidence_windows",0), ("max_evidence_windows",32), ("max_evidence_spans",0), ("max_evidence_spans",32)] {
        let mut v = fixture("faithfulness"); v["policy"][field] = json!(bad); assert!(parse(&v).is_err());
    }
    let mut v = fixture("faithfulness"); v["source"] = json!("abcdefgh");
    v["policy"]["evidence_window_bytes"] = json!(4);
    v["policy"]["max_evidence_windows"] = json!(1); v["policy"]["max_evidence_spans"] = json!(1);
    assert!(parse(&v).is_err()); // The source tail must not disappear.
    v["policy"]["max_evidence_windows"] = json!(2); v["policy"]["max_evidence_spans"] = json!(2);
    assert!(parse(&v).is_ok());
}
#[test]
fn input_byte_bound_and_scoring_resources_do_not_become_per_head_allowances() {
    let text = fixture("pairwise").to_string();
    assert!(request(&text, budget(), text.len()).is_ok());
    assert!(request(&text, budget(), text.len()-1).is_err());
    assert!(request(&text, budget(), 0).is_err());
    assert!(request(&text, budget(), MAX_INPUT_BYTES+1).is_err());
    let c = command(&[]); let b = budget(); let l = planning_limits(&c.args, b);
    assert_eq!(l.max_total_prompt_tokens as u64, c.args.max_forward_positions);
    assert_eq!(l.max_total_projected_logits, c.args.max_projected_logits);
    assert_eq!(l.max_output_bytes, b.max_output_bytes);
    assert_eq!(l.per_head.max_depth, c.args.max_candidate_tokens);
    assert_eq!(l.per_head.max_nodes, b.max_grammar_states as usize);
}
#[test]
fn root_still_routes_saved_maps_and_existing_jobs_and_does_not_claim_truth() {
    let help = definition().render_long_help().to_string(); assert!(help.contains("not a factuality certificate"));
    let mut root = super::super::definition(); let help = root.render_long_help().to_string();
    for name in ["judge", "map", "job", "score-batch", "batch", "extract", "generate", "chat"] {
        assert!(help.contains(name));
    }
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_runtime_refuses_before_private_io() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("input read") } }
    struct NoWrite;
    impl Write for NoWrite {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output write") }
        fn flush(&mut self) -> io::Result<()> { panic!("output flush") }
    }
    assert_eq!(command(&[]).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Unavailable));
}
