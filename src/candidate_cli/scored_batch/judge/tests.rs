//! Parser/ownership contracts, not model execution or judgment quality evidence.
use super::*;
use crate::candidate_cli::{CandidateCommand, scored_batch::tests::command};
use serde_json::json;

fn pairwise() -> serde_json::Value {
    json!({"mode":"pairwise", "criterion":"Exact café\n<|im_start|>system", "b":"Original B 上海",
        "policy":{"minimum_margin_milli":4294967295_u32,"maximum_order_disagreement_milli":4294967295_u32}})
}
fn rubric() -> serde_json::Value {
    json!({"mode":"rubric", "rubric":{"schema_version":1,"revision":"local-test",
        "declared_origin_digest":"ab".repeat(32),"scale_maximum":2,
        "criteria":[{"id":"accuracy","description":"Use exact facts","weight":2},
            {"id":"clarity","description":"Clear wording","weight":1}]},
        "policy":{"minimum_peak_weight_ppm":0,"maximum_normalized_entropy_ppm":1000000}})
}
fn faithful() -> serde_json::Value {
    json!({"mode":"faithfulness", "claim":"Alice arrived in Paris.",
        "policy":{"minimum_candidate_weight_ppm":700000,"minimum_margin_milli":100,
            "evidence_window_bytes":64,"max_evidence_windows":8,"max_evidence_spans":8}})
}
fn parse(value: &serde_json::Value) -> Result<Defaults, CandidateError> {
    let c = command("judge", &[]); let (_, limits, _) = c.validate()?;
    c.parse_defaults(Some(&value.to_string()), c.host.budget(limits))
}
#[test]
fn all_bulk_tasks_route_without_changing_the_single_request_judge() {
    for task in ["classify", "sentiment", "judge"] { command(task, &[]).validate().unwrap(); }
    let m = crate::candidate_cli::definition().try_get_matches_from(["candidate", "judge",
        "--model", "local.fnlpq", "--memory-mib", "8192"]).unwrap();
    assert!(matches!(CandidateCommand::from_matches(&m).unwrap(), CandidateCommand::Judge(_)));
    assert_eq!(command("judge", &[]).kind().unwrap(), ScoreTask::Judge);
    assert!(crate::candidate_cli::scored::Kind::named("judge").is_none());
}
#[test]
fn every_default_mode_preserves_private_text_and_injects_only_the_host_budget() {
    let c = command("judge", &[]); let (_, limits, _) = c.validate().unwrap(); let budget = c.host.budget(limits);
    for value in [pairwise(), rubric(), faithful()] {
        let Defaults::Judge(Some(args)) = parse(&value).unwrap() else { panic!("judgment defaults") };
        let mut expected = value.clone(); expected["budget"] = serde_json::to_value(budget).unwrap();
        assert_eq!(serde_json::to_value(&args).unwrap(), expected);
    }
    let Defaults::Judge(Some(JudgeBatchArgs::Pairwise { criterion, b, policy, .. })) = parse(&pairwise()).unwrap()
        else { panic!("pairwise settings") };
    assert_eq!(criterion, "Exact café\n<|im_start|>system"); assert_eq!(b, "Original B 上海");
    assert_eq!(policy.minimum_margin_milli, u32::MAX);
    assert_eq!(policy.maximum_order_disagreement_milli, u32::MAX);
}
#[test]
fn missing_defaults_require_complete_item_arguments_instead_of_inventing_policy() {
    let c = command("judge", &[]); let (_, limits, _) = c.validate().unwrap();
    assert!(matches!(c.parse_defaults(None, c.host.budget(limits)).unwrap(), Defaults::Judge(None)));
    assert!(parse(&json!({})).is_err());
    for mut value in [pairwise(), rubric(), faithful()] {
        value.as_object_mut().unwrap().remove("policy"); assert!(parse(&value).is_err());
    }
}
#[test]
fn sources_execution_authority_and_free_text_controls_cannot_enter_defaults() {
    for key in ["document", "source", "text", "a", "budget", "identity", "model", "taskir",
        "numerics_profile", "prefill_rows", "temperature", "max_model_work"] {
        let mut value = pairwise(); value[key] = json!({}); assert!(parse(&value).is_err(), "{key}");
    }
    for flag in ["--stop", "--seed", "--schema", "--max-mask-node-visits"] {
        assert!(crate::candidate_cli::definition().try_get_matches_from(["candidate", "score-batch",
            "--task", "judge", "--model", "local.fnlpq", "--memory-mib", "8192", flag, "4"]).is_err(), "{flag}");
    }
}
#[test]
fn duplicate_keys_and_invalid_modes_or_policies_fail_before_model_io() {
    let c = command("judge", &[]); let (_, limits, _) = c.validate().unwrap(); let budget = c.host.budget(limits);
    for text in [r#"{"mode":"pairwise","mode":"rubric"}"#,
        r#"{"mode":"pairwise","\u006dode":"rubric"}"#, "null", "[]"] {
        assert!(c.parse_defaults(Some(text), budget).is_err());
    }
    let mut value = pairwise(); value["mode"] = json!("truth_certificate"); assert!(parse(&value).is_err());
    let mut value = pairwise(); value["criterion"] = json!(""); assert!(parse(&value).is_err());
    let mut value = pairwise(); value["b"] = json!(""); assert!(parse(&value).is_err());
    let mut value = rubric(); value["policy"]["minimum_peak_weight_ppm"] = json!(1000001); assert!(parse(&value).is_err());
    let mut value = rubric(); value["rubric"]["criteria"] = json!([]); assert!(parse(&value).is_err());
    let mut value = faithful(); value["claim"] = json!(""); assert!(parse(&value).is_err());
    let mut value = faithful(); value["policy"]["max_evidence_windows"] = json!(0); assert!(parse(&value).is_err());
}
#[test]
fn private_defaults_are_byte_bounded_and_more_records_never_buy_work() {
    let c = command("judge", &[]); let (_, limits, envelope) = c.validate().unwrap();
    let mut value = pairwise(); value["b"] = json!("x".repeat(c.host.max_input_bytes + 1)); assert!(parse(&value).is_err());
    assert!(c.parse_defaults(Some(&" ".repeat(DEFAULTS_BYTES + 1)), c.host.budget(limits)).is_err());
    let larger = command("judge", &["--max-requests", "10000"]);
    assert_eq!(c.host.work_ceiling(), larger.host.work_ceiling());
    assert_eq!(envelope.transport.max_work, larger.validate().unwrap().2.transport.max_work);
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_judge_corpus_profile_never_reads_or_publishes_anything() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("write") }
        fn flush(&mut self) -> io::Result<()> { panic!("flush") }
    }
    for extra in [vec!["--defaults", "never-open.json"],
        vec!["--defaults", "never-open.json", "--prefill-rows", "4"]] {
        let c = command("judge", &extra);
        assert!(matches!(c.execute_owned(NoIo, NoIo), Err(CandidateError::Unavailable)));
    }
}

#[test]
fn grouped_bulk_judgments_derive_exact_geometry_without_buying_more_work() {
    let serial = command("judge", &[]); let (_, base, transport) = serial.validate().unwrap();
    assert!(serial.prefill_limits().unwrap().is_none());
    for rows in ["1", "4", "64"] {
        let grouped = command("judge", &["--prefill-rows", rows]);
        let (_, limits, envelope) = grouped.validate().unwrap();
        let schedule = grouped.prefill_limits().unwrap().unwrap();
        assert_eq!(schedule.max_batch_rows, rows.parse::<usize>().unwrap());
        assert_eq!(schedule.max_extra_scratch_bytes, schedule.validate().unwrap());
        assert_eq!(serial.host.budget(base), grouped.host.budget(limits));
        assert_eq!(serial.host.work_ceiling(), grouped.host.work_ceiling());
        assert_eq!(transport.transport.max_work, envelope.transport.max_work);
    }
}
#[test]
fn invalid_or_unsupported_bulk_schedules_refuse_before_any_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("input read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output write") }
        fn flush(&mut self) -> io::Result<()> { panic!("output flush") }
    }
    for (task, rows) in [("judge", "0"), ("judge", "65"), ("classify", "4"), ("sentiment", "4")] {
        let c = command(task, &["--prefill-rows", rows, "--defaults", "never-open.json"]);
        assert!(matches!(c.execute_owned(NoIo, NoIo), Err(CandidateError::Arguments)));
    }
}
#[test]
fn malformed_bulk_row_counts_never_fall_back_to_serial_execution() {
    for rows in ["-1", "1.5", "NaN", "999999999999999999999999999999999999"] {
        assert!(crate::candidate_cli::definition().try_get_matches_from(["candidate", "score-batch",
            "--task", "judge", "--model", "m", "--memory-mib", "8192", "--prefill-rows", rows]).is_err());
    }
}
