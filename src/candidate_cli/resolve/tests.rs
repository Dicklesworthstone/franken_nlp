//! Strict source-snapshot input and resource boundaries, no model execution.
use super::*;
pub(in crate::candidate_cli) fn command(extra: &[&str]) -> ResolveCommand {
    let mut argv = vec!["candidate", "resolve", "--model", "local.fnlpq", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    let matches = super::super::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::Resolve(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong dispatch") };
    command
}
pub(in crate::candidate_cli) fn input(second: &str) -> String {
    serde_json::json!({"documents":[
        {"id":"a","text":"é Alice","mentions":[{"entity_type":"person","surface":"Alice",
            "span":{"byte_start":3,"byte_end":8,"scalar_start":2,"scalar_end":7}}]},
        {"id":"b","text":second,"mentions":[{"entity_type":"person","surface":second,
            "span":{"byte_start":0,"byte_end":second.len(),"scalar_start":0,"scalar_end":second.chars().count()}}]}],
        "options":{"blocking":"ascii_word_overlap","context_scalars":32,"minimum_margin_milli":1000}}).to_string()
}
#[test]
fn command_exposes_a_bounded_snapshot_not_generation_or_schema_controls() {
    let cmd = command(&[]); let (_, limits) = cmd.validate().unwrap();
    assert_eq!(cmd.scoring(limits).planning.max_pairs, cmd.graph().max_candidate_pairs);
    assert_eq!(cmd.scoring(limits).max_model_work, cmd.host.work_ceiling());
    for flag in ["--seed", "--schema", "--max-mask-node-visits"] {
        assert!(super::super::definition().try_get_matches_from(["candidate", "resolve", flag, "private"]).is_err());
    }
}
#[test]
fn exact_source_and_coordinate_domains_are_preserved() {
    let cmd = command(&[]); let parsed = cmd.input(&input("Alice")).unwrap();
    let first = &parsed.documents[0];
    assert_eq!(first.text, "é Alice");
    assert_eq!(first.mentions[0].span.byte_start, 3);
    assert_eq!(first.mentions[0].span.scalar_start, 2);
}
#[test]
fn unknown_duplicate_keys_and_missing_policies_are_not_coerced() {
    let cmd = command(&[]);
    for text in ["{}", r#"{"documents":[],"documents":[]}"#,
        r#"{"documents":[],"\u0064ocuments":[]}"#] { assert!(cmd.input(text).is_err()); }
    let original: serde_json::Value = serde_json::from_str(&input("Alice")).unwrap();
    for field in ["budget", "execution_identity", "prompt", "scores"] {
        let mut changed = original.clone(); changed[field] = serde_json::json!({});
        assert!(cmd.input(&changed.to_string()).is_err());
    }
    let mut missing = original.clone(); missing.as_object_mut().unwrap().remove("options");
    assert!(cmd.input(&missing.to_string()).is_err());
    for (name, number) in [("minimum_margin_milli",0), ("minimum_margin_milli",1_000_001), ("context_scalars",2049)] {
        let mut changed = original.clone(); changed["options"][name] = serde_json::json!(number);
        assert!(cmd.input(&changed.to_string()).is_err());
    }
}
#[test]
fn graph_and_memory_limits_are_finite_and_pair_count_does_not_multiply_work() {
    for (flag, value) in [("--max-documents","0"), ("--max-mentions","16385"),
        ("--max-pairs","65537"), ("--max-pair-visits","0"), ("--max-cluster-checks","0"),
        ("--max-scan-steps","0"), ("--graph-reserve-mib","1"), ("--graph-reserve-mib","18446744073709551615")] {
        assert!(command(&[flag,value]).validate().is_err(), "{flag}");
    }
    let a = command(&["--max-pairs","1"]); let b = command(&["--max-pairs","256"]);
    assert_eq!(a.host.work_ceiling(), b.host.work_ceiling());
    command(&["--max-pairs","0"]).validate().unwrap();
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn feature_disabled_refuses_before_any_source_or_model_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("write") }
        fn flush(&mut self) -> io::Result<()> { panic!("flush") }
    }
    assert_eq!(command(&[]).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Unavailable));
}
