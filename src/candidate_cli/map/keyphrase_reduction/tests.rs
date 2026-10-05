//! Explicit mode, policy limits and disabled-feature behavior; not inference.
use super::*;
use crate::candidate_cli::map::tests::command;

#[test]
fn default_maps_stay_independent_and_global_ranking_requires_keyphrases() {
    for task in ["ner", "keyphrases", "summarize"] {
        let cmd = command(task, &[]); cmd.validate().unwrap();
        assert!(cmd.keyphrases.limits(cmd.max_map_result_bytes).unwrap().is_none());
    }
    command("keyphrases", &["--reduce-keyphrases"]).validate().unwrap();
    for task in ["ner", "summarize"] {
        assert!(command(task, &["--reduce-keyphrases"]).validate().is_err());
    }
}
#[test]
fn policy_flags_cannot_be_silently_ignored_or_mix_reducers() {
    for flag in ["--document-keyphrases", "--max-unique-keyphrases",
        "--max-keyphrase-evidence-spans", "--max-keyphrase-scan-work"] {
        assert!(crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "map", "--task", "keyphrases", "--model", "local.fnlpq", "--memory-mib", "8192", flag, "1",
        ]).is_err(), "{flag}");
    }
    for other in ["--reduce-summary", "--synthesize-summary"] {
        assert!(crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "map", "--task", "keyphrases", "--model", "local.fnlpq",
            "--memory-mib", "8192", "--reduce-keyphrases", other,
        ]).is_err());
    }
}
#[test]
fn document_top_k_does_not_change_native_chunk_options_or_work_authority() {
    let cmd = command("keyphrases", &["--reduce-keyphrases", "--document-keyphrases", "3"]);
    let (_, budget) = cmd.validate().unwrap();
    let SourceMapTask::Keyphrases(options) = cmd.map_task(Some(r#"{"max_phrases":2,"max_phrase_scalars":64}"#)).unwrap()
        else { panic!("keyphrases") };
    let policy = cmd.keyphrases.limits(cmd.max_map_result_bytes).unwrap().unwrap();
    assert_eq!(options.max_phrases, 2); assert_eq!(policy.max_phrases, 3);
    assert_eq!(policy.max_result_bytes, cmd.max_map_result_bytes);
    assert_eq!(policy.aggregation.max_value_bytes, cmd.max_map_result_bytes);
    let plain = command("keyphrases", &[]);
    assert_eq!(cmd.work_ceiling(), plain.work_ceiling());
    assert_eq!(cmd.max_total_mask_node_visits, plain.max_total_mask_node_visits);
    assert_eq!(cmd.host.task_budget(budget).max_output_tokens, plain.host.max_new_tokens as u32);
}
#[test]
fn invalid_preselection_and_verification_limits_fail_before_source_io() {
    for (flag, value) in [("--document-keyphrases", "0"), ("--document-keyphrases", "4097"),
        ("--max-unique-keyphrases", "0"), ("--max-unique-keyphrases", "65537"),
        ("--max-keyphrase-evidence-spans", "0"), ("--max-keyphrase-evidence-spans", "1000001"),
        ("--max-keyphrase-scan-work", "0"), ("--max-keyphrase-scan-work", "1000000000001")] {
        assert!(command("keyphrases", &["--reduce-keyphrases", flag, value]).validate().is_err(), "{flag}");
    }
    let cmd = command("keyphrases", &["--reduce-keyphrases", "--max-unique-keyphrases", "99",
        "--max-keyphrase-evidence-spans", "200", "--max-keyphrase-scan-work", "12345"]);
    cmd.validate().unwrap(); let limits = cmd.keyphrases.limits(cmd.max_map_result_bytes).unwrap().unwrap();
    assert_eq!(limits.aggregation.max_unique_phrases, 99);
    assert_eq!(limits.aggregation.max_evidence_spans, 200); assert_eq!(limits.aggregation.max_scan_work, 12345);
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_runtime_and_wrong_task_do_not_read_write_or_flush() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("write") }
        fn flush(&mut self) -> io::Result<()> { panic!("flush") }
    }
    assert_eq!(command("keyphrases", &["--reduce-keyphrases"]).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Unavailable));
    assert_eq!(command("ner", &["--reduce-keyphrases"]).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
}
