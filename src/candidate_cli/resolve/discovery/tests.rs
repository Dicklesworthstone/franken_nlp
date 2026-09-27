use super::*;
use crate::candidate_cli::resolve::tests::command;
fn raw() -> String {
    serde_json::json!({"documents":[{"id":"a","text":"é Alice Alice"},{"id":"b","text":"Bob"}],
        "options":{"blocking":"ascii_word_overlap","context_scalars":32,"minimum_margin_milli":1000}}).to_string()
}
#[test]
fn discovery_is_explicit_and_preserves_raw_unicode_without_accepting_mentions() {
    let manual = command(&[]); assert!(!manual.discovery.discover_entities);
    assert!(manual.input(&raw()).is_err());
    let cmd = command(&["--discover-entities"]); cmd.validate().unwrap();
    let source = cmd.discovery.input(&cmd, &raw()).unwrap();
    assert_eq!(source.documents[0].text, "é Alice Alice"); assert_eq!(source.ner, NerOptions::default());
    let mut injected: serde_json::Value = serde_json::from_str(&raw()).unwrap();
    for field in ["mentions", "ner", "scores", "budget", "execution_identity"] {
        injected["documents"][0][field] = serde_json::json!([]);
        assert!(cmd.discovery.input(&cmd, &injected.to_string()).is_err());
        injected["documents"][0].as_object_mut().unwrap().remove(field);
    }
}
#[test]
fn duplicate_keys_ids_invalid_ner_scope_and_missing_policy_fail_before_metadata() {
    let cmd = command(&["--discover-entities"]);
    for text in ["{}", r#"{"documents":[],"\u0064ocuments":[]}"#] { assert!(cmd.discovery.input(&cmd,text).is_err()); }
    let original: serde_json::Value = serde_json::from_str(&raw()).unwrap();
    let mut repeated = original.clone(); repeated["documents"][1]["id"] = serde_json::json!("a");
    assert!(cmd.discovery.input(&cmd,&repeated.to_string()).is_err());
    let mut invalid = original.clone(); invalid["ner"] = serde_json::json!({"types":["person","person"],"max_entities":4,"max_mention_scalars":32});
    assert!(cmd.discovery.input(&cmd,&invalid.to_string()).is_err());
    let mut missing = original; missing.as_object_mut().unwrap().remove("options");
    assert!(cmd.discovery.input(&cmd,&missing.to_string()).is_err());
}
#[test]
fn discovery_limits_and_memory_floor_are_finite_and_shared() {
    for (flag,value) in [("--max-ner-tokens","0"),("--max-ner-tokens","2048"),
        ("--max-ner-mask-node-visits","1"),("--max-snapshot-mask-node-visits","1"),
        ("--max-expanded-bytes","0"),("--max-expanded-bytes","67108865"),("--graph-reserve-mib","32")] {
        assert!(command(&["--discover-entities",flag,value]).validate().is_err(),"{flag}");
    }
    // Wire-sized input does not admit a much larger expanded graph for free.
    assert!(command(&["--discover-entities","--max-expanded-bytes","67108864"]).validate().is_err());
    let cmd = command(&["--discover-entities","--max-snapshot-mask-node-visits","1000000000"]);
    cmd.validate().unwrap(); assert!(cmd.discovery.input(&cmd,&raw()).is_err());
}
#[test]
fn ner_tokens_and_pair_depth_are_distinct_and_do_not_multiply_model_authority() {
    let cmd = command(&["--discover-entities","--max-ner-tokens","64","--max-candidate-tokens","8"]);
    let (_, limits) = cmd.validate().unwrap(); let source = cmd.discovery.input(&cmd,&raw()).unwrap();
    let config = cmd.discovery.config(&cmd,limits,source.ner,source.options);
    assert_eq!(config.ner_budget.max_output_tokens,64); assert_eq!(config.scoring.planning.per_head.max_output_tokens,8);
    assert_eq!(config.max_model_work,cmd.host.work_ceiling()); assert_eq!(config.scoring.max_model_work,config.max_model_work);
    assert_eq!(config.graph.max_input_bytes,16*1024*1024);
    assert_eq!(config.source_planning.max_input_bytes,cmd.host.max_input_bytes);
}
#[test]
fn empty_raw_snapshot_is_allowed_only_by_the_explicit_raw_mode() {
    let cmd = command(&["--discover-entities"]); let mut value: serde_json::Value = serde_json::from_str(&raw()).unwrap();
    value["documents"] = serde_json::json!([]);
    assert!(cmd.discovery.input(&cmd,&value.to_string()).unwrap().documents.is_empty());
    assert!(command(&[]).input(&value.to_string()).is_err());
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_feature_never_reads_or_writes_in_discovery_mode() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("write") }
        fn flush(&mut self) -> io::Result<()> { panic!("flush") }
    }
    assert_eq!(command(&["--discover-entities"]).execute(&mut NoIo,&mut NoIo),Err(CandidateError::Unavailable));
}
