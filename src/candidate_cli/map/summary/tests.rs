//! Explicit-mode, bounded-policy and disabled-feature contracts, not inference.
use super::*;
use crate::candidate_cli::map::tests::command;

#[test]
fn default_maps_are_unchanged_and_reduction_requires_the_summary_task() {
    for task in ["ner", "keyphrases", "summarize"] {
        let cmd = command(task, &[]); cmd.validate().unwrap();
        assert!(cmd.summary.limits(cmd.max_map_result_bytes).unwrap().is_none());
    }
    command("summarize", &["--reduce-summary"]).validate().unwrap();
    for task in ["ner", "keyphrases"] {
        assert!(command(task, &["--reduce-summary"]).validate().is_err());
    }
}
#[test]
fn summary_policy_flags_cannot_be_silently_ignored() {
    for flag in ["--summary-bullets", "--max-unique-summary-bullets", "--max-summary-citations",
        "--max-summary-evidence-spans", "--max-summary-scan-steps"] {
        assert!(crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "map", "--task", "summarize", "--model", "local.fnlpq", "--memory-mib", "8192", flag, "1",
        ]).is_err(), "{flag}");
    }
}
#[test]
fn final_bullet_selection_is_independent_of_each_native_chunk_and_compute_authority() {
    let cmd = command("summarize", &["--reduce-summary", "--summary-bullets", "3"]);
    let (_, budget) = cmd.validate().unwrap();
    let raw = r#"{"max_bullets":2,"max_bullet_scalars":64,"max_citations_per_bullet":2,"max_quote_scalars":32}"#;
    let SourceMapTask::Summarize(options) = cmd.map_task(Some(raw)).unwrap() else { panic!("summary") };
    let summary = cmd.summary.limits(cmd.max_map_result_bytes).unwrap().unwrap();
    assert_eq!(options.max_bullets, 2); assert_eq!(summary.max_bullets, 3);
    assert_eq!(summary.max_result_bytes, cmd.max_map_result_bytes);
    assert_eq!(summary.aggregation.max_value_bytes, cmd.max_map_result_bytes);
    let plain = command("summarize", &[]);
    assert_eq!(cmd.work_ceiling(), plain.work_ceiling());
    assert_eq!(cmd.max_total_mask_node_visits, plain.max_total_mask_node_visits);
    assert_eq!(cmd.host.task_budget(budget).max_output_tokens, plain.host.max_new_tokens as u32);
}
#[test]
fn finite_preselection_and_verification_limits_refuse_invalid_values() {
    for (flag, value) in [("--summary-bullets", "0"), ("--summary-bullets", "1025"),
        ("--max-unique-summary-bullets", "0"), ("--max-unique-summary-bullets", "65537"),
        ("--max-summary-citations", "1000001"), ("--max-summary-evidence-spans", "0"),
        ("--max-summary-scan-steps", "0"), ("--max-summary-scan-steps", "1000000000001")] {
        assert!(command("summarize", &["--reduce-summary", flag, value]).validate().is_err(), "{flag}");
    }
    let cmd = command("summarize", &["--reduce-summary", "--max-unique-summary-bullets", "99",
        "--max-summary-citations", "100", "--max-summary-evidence-spans", "200", "--max-summary-scan-steps", "12345"]);
    cmd.validate().unwrap(); let limits = cmd.summary.limits(cmd.max_map_result_bytes).unwrap().unwrap();
    assert_eq!(limits.aggregation.max_unique_bullets, 99); assert_eq!(limits.aggregation.max_citations, 100);
    assert_eq!(limits.aggregation.max_evidence_spans, 200); assert_eq!(limits.aggregation.max_scan_steps, 12345);
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_runtime_and_wrong_task_refuse_before_any_source_or_output_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("write") }
        fn flush(&mut self) -> io::Result<()> { panic!("flush") }
    }
    assert_eq!(command("summarize", &["--reduce-summary"]).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Unavailable));
    assert_eq!(command("ner", &["--reduce-summary"]).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
}
