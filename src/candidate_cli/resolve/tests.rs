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

#[test]
fn compute_threads_reach_every_resolution_route_and_keep_serial_defaults() {
    assert_eq!(command(&[]).compute.threads(), 1);
    for mode in [vec![], vec!["--discover-entities"], vec!["--discover-entities", "--chunked"]] {
        let mut flags = mode;
        flags.extend(["--compute-threads", "4", "--prefill-rows", "3", "--graph-reserve-mib", "256"]);
        let mut cmd = command(&flags); cmd.host.memory_mib = 16_384;
        let (common, limits) = cmd.validate().unwrap();
        let prefill = cmd.prefill.limits().unwrap();
        assert_eq!(cmd.compute.threads(), 4); assert_eq!(common.compute.threads(), 4);
        assert_eq!(limits.compute, cmd.compute.admit_prefill(prefill).unwrap());
        assert_eq!(prefill.unwrap().max_batch_rows, 3);
        assert!(limits.compute.workspace_bytes > cmd.compute.admit_prefill(None).unwrap().workspace_bytes);
    }
}
#[test]
fn complete_threaded_peak_includes_graph_and_chunk_preflight_owners() {
    let mut plain = command(&["--compute-threads", "4", "--prefill-rows", "3", "--graph-reserve-mib", "256"]);
    plain.host.memory_mib = 16_384;
    let (_, a) = plain.validate().unwrap(); let schedule = plain.prefill.limits().unwrap();
    let plain_peak = plain.compute_peak_bytes(a, schedule).unwrap();
    let mut discovery = command(&["--discover-entities", "--compute-threads", "4", "--prefill-rows", "3",
        "--graph-reserve-mib", "256"]);
    discovery.host.memory_mib = 16_384;
    let (_, b) = discovery.validate().unwrap();
    let discovery_peak = discovery.compute_peak_bytes(b, schedule).unwrap();
    assert!(discovery_peak > plain_peak);
    let mut chunked = command(&["--discover-entities", "--chunked", "--compute-threads", "4",
        "--prefill-rows", "3", "--graph-reserve-mib", "256"]);
    chunked.host.memory_mib = 16_384;
    let (_, mut c) = chunked.validate().unwrap();
    let chunked_peak = chunked.compute_peak_bytes(c, schedule).unwrap();
    assert!(chunked_peak >= discovery_peak + chunked.graph_reserve_mib * MIB);
    c.memory_bytes = chunked_peak;
    chunked.check_compute_memory(c, schedule).unwrap();
    c.memory_bytes -= 1;
    assert_eq!(chunked.check_compute_memory(c, schedule), Err(CandidateError::Arguments));
    c.memory_bytes = chunked_peak; c.compute = chunked.compute.admit_prefill(None).unwrap();
    assert_eq!(chunked.check_compute_memory(c, schedule), Err(CandidateError::Arguments));
}
#[test]
fn worker_schedule_does_not_multiply_pair_or_whole_snapshot_allowances() {
    let serial = command(&[]);
    let mut threaded = command(&["--compute-threads", "4", "--prefill-rows", "3"]);
    threaded.host.memory_mib = 16_384;
    let (_, a) = serial.validate().unwrap(); let (_, b) = threaded.validate().unwrap();
    assert_eq!(serial.scoring(a).max_model_work, threaded.scoring(b).max_model_work);
    assert_eq!(serial.scoring(a).planning.max_work, threaded.scoring(b).planning.max_work);
    assert_eq!(serial.graph().max_candidate_pairs, threaded.graph().max_candidate_pairs);
    assert_eq!(crate::canonjson::canonical_bytes(&serial.host.budget(a)).unwrap(),
        crate::canonjson::canonical_bytes(&threaded.host.budget(b)).unwrap());
}
#[test]
fn invalid_workers_or_complete_peak_refuse_before_input_or_output_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("write") }
        fn flush(&mut self) -> io::Result<()> { panic!("flush") }
    }
    for mode in [vec![], vec!["--discover-entities"], vec!["--discover-entities", "--chunked"]] {
        for width in ["0", "65", "1000000"] {
            let mut flags = mode.clone(); flags.extend(["--compute-threads", width]);
            assert_eq!(command(&flags).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
        }
        let mut flags = mode; flags.extend(["--compute-threads", "4", "--prefill-rows", "64",
            "--graph-reserve-mib", "256"]);
        let mut underfunded = command(&flags);
        underfunded.host.memory_mib = 1024;
        assert_eq!(underfunded.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
}
