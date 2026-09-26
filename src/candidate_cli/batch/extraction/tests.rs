//! Model-free command/default/work tests; no synthetic native-success producer.
use super::*;

fn command(task: &str, extra: &[&str]) -> BatchCommand {
    let mut argv = vec!["candidate", "batch", "--task", task, "--model", "local.fnlpq", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
    match CandidateCommand::from_matches(&matches).unwrap() {
        CandidateCommand::Batch(command) => command,
        _ => panic!("wrong corpus route"),
    }
}
fn ceiling(command: &BatchCommand) -> TaskBudget {
    let (_, limits, _) = command.validate().unwrap(); command.host.task_budget(limits)
}
fn defaults_json(schema: &str, budget: TaskBudget) -> String {
    serde_json::to_string(&ExtractionBatchArgs { schema: schema.to_owned(), budget,
        grounding: ExtractionBatchGrounding::Structural }).unwrap()
}

#[test]
fn extract_is_a_distinct_corpus_task_with_shared_schema_or_per_record_arguments() {
    let cmd = command("extract", &["--schema", "schema.json"]);
    cmd.validate().unwrap(); assert_eq!(cmd.task().unwrap(), BuiltInTask::Extract);
    assert!(cmd.kind().is_err()); // Never dispatch through a fixed source task.
    let cap = ceiling(&cmd);
    let defaults = cmd.load_extract_defaults(Some(r#"{"type":"string"}"#.to_owned()), None, cap).unwrap().unwrap();
    assert_eq!(defaults.budget, cap);
    assert_eq!(defaults.grounding, ExtractionBatchGrounding::Structural);
    let cmd = command("extract", &[]);
    assert!(cmd.load_extract_defaults(None, None, ceiling(&cmd)).unwrap().is_none());
}

#[test]
fn schema_switches_cannot_be_ignored_by_other_tasks_or_conflict_with_defaults() {
    for task in ["ner", "keyphrases", "summarize", "answer"] {
        assert!(command(task, &["--schema", "s"]).validate().is_err());
        assert!(command(task, &["--schema", "s", "--source-membership"]).validate().is_err());
        assert!(command(task, &[]).validate().is_ok());
    }
    for extra in [vec!["--source-membership"], vec!["--schema", "s", "--defaults", "d"]] {
        let mut argv = vec!["candidate", "batch", "--task", "extract", "--model", "m", "--memory-mib", "8192"];
        argv.extend(extra);
        assert!(crate::candidate_cli::definition().try_get_matches_from(argv).is_err());
    }
    assert!(command("extract", &["--schema", "-"]).validate().is_err());
    let mut cmd = command("extract", &[]); cmd.source_membership = true;
    assert!(cmd.validate().is_err());
}

#[test]
fn complete_default_configuration_keeps_exact_schema_numbers_and_bytes() {
    let cmd = command("extract", &["--defaults", "defaults.json"]); let cap = ceiling(&cmd);
    let schema = " {\"type\":\"number\",\"const\":0.12345678901234567890123456789012345678}\n";
    let json = defaults_json(schema, cap);
    let parsed = cmd.load_extract_defaults(None, Some(&json), cap).unwrap().unwrap();
    assert_eq!(parsed.schema.as_bytes(), schema.as_bytes());
    assert_eq!(parsed.budget, cap);
}

#[test]
fn outer_and_embedded_schema_duplicates_and_unknown_configuration_are_rejected() {
    let cmd = command("extract", &["--defaults", "d"]); let cap = ceiling(&cmd);
    for json in [r#"{}"#.to_owned(),
        r#"{"schema":"{}","\u0073chema":"{}"}"#.to_owned(),
        defaults_json(r#"{"type":"string","type":"number"}"#, cap),
        defaults_json(r#"{"type":"string","\u0074ype":"number"}"#, cap),
        defaults_json(r#"{"$ref":"https://example.invalid/schema"}"#, cap)] {
        assert!(cmd.load_extract_defaults(None, Some(&json), cap).is_err());
    }
    let mut value = serde_json::to_value(ExtractionBatchArgs { schema: r#"{"type":"string"}"#.to_owned(),
        grounding: ExtractionBatchGrounding::Structural, budget: cap }).unwrap();
    value["instruction"] = serde_json::json!("private injected instruction");
    assert!(cmd.load_extract_defaults(None, Some(&value.to_string()), cap).is_err());
}

#[test]
fn every_default_budget_axis_is_checked_and_full_allocated_kv_cannot_be_underpriced() {
    let cmd = command("extract", &["--defaults", "d"]); let cap = ceiling(&cmd);
    let original = serde_json::to_value(ExtractionBatchArgs { schema: r#"{"type":"string"}"#.to_owned(),
        grounding: ExtractionBatchGrounding::Structural, budget: cap }).unwrap();
    for field in ["max_input_tokens", "max_output_tokens", "max_output_bytes", "max_grammar_states", "max_kv_bytes"] {
        for replacement in [0, u64::MAX] {
            let mut changed = original.clone(); changed["budget"][field] = serde_json::json!(replacement);
            assert!(cmd.load_extract_defaults(None, Some(&changed.to_string()), cap).is_err(), "{field}");
        }
    }
    let mut lower = cap; lower.max_kv_bytes -= 1;
    assert!(cmd.load_extract_defaults(None, Some(&defaults_json(r#"{"type":"string"}"#, lower)), cap).is_err());
    let mut lower = cap; lower.max_output_tokens = 16;
    let parsed = cmd.load_extract_defaults(None, Some(&defaults_json(r#"{"type":"string"}"#, lower)), cap).unwrap().unwrap();
    assert_eq!(parsed.budget.max_output_tokens, 16);
    assert_eq!(ceiling(&cmd), cap);
}

#[test]
fn shared_source_schema_is_explicit_and_never_bound_to_invented_default_evidence() {
    let cmd = command("extract", &["--schema", "s", "--source-membership"]); let cap = ceiling(&cmd);
    // This constant may be valid in a future document. An empty fabricated
    // source would wrongly reject it while claiming to preflight membership.
    let schema = r#"{"type":"string","const":"Alice","x-fnlp-source":"verbatim"}"#;
    let defaults = cmd.load_extract_defaults(Some(schema.to_owned()), None, cap).unwrap().unwrap();
    assert_eq!(defaults.grounding, ExtractionBatchGrounding::SourceMembership);
    assert_eq!(defaults.schema, schema);
    let structural = command("extract", &["--schema", "s"]);
    assert!(structural.load_extract_defaults(Some(schema.to_owned()), None, ceiling(&structural)).is_err());
}

#[test]
fn schema_and_configuration_byte_limits_do_not_depend_on_file_metadata() {
    let cmd = command("extract", &["--schema", "s"]); let cap = ceiling(&cmd);
    for schema in [String::new(), " ".repeat(schema_cli::SCHEMA_BYTES + 1)] {
        assert!(cmd.load_extract_defaults(Some(schema), None, cap).is_err());
    }
    let cmd = command("extract", &["--defaults", "d"]);
    assert!(cmd.load_extract_defaults(None, Some(&" ".repeat(MAX_SOURCE_ARGUMENT_BYTES + 1)), ceiling(&cmd)).is_err());
}

#[test]
fn model_masks_and_full_wire_budgets_apply_unchanged_to_extraction() {
    let cmd = command("extract", &["--max-requests", "3"]); let (_, _, envelope) = cmd.validate().unwrap();
    let one = constrained_int8::planned_work(cmd.host.context_tokens - cmd.host.max_new_tokens, cmd.host.max_new_tokens).unwrap();
    assert_eq!(envelope.native.max_model_work, one.checked_add(one).unwrap().checked_add(one).unwrap());
    assert_eq!(envelope.native.masks.max_visits_per_run, 3 * cmd.host.max_mask_node_visits);
    assert_eq!(envelope.transport.max_output_bytes + (cmd.max_requests + 4) * FRAME_ALLOWANCE, envelope.output_bytes);
    assert_eq!(completed(BatchSummary { failed: 1, ..BatchSummary::default() }), Err(CandidateError::Batch));
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_feature_refuses_all_extract_input_forms_before_any_io() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("input read") } }
    struct NoWrite;
    impl Write for NoWrite {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output written") }
        fn flush(&mut self) -> io::Result<()> { panic!("output flushed") }
    }
    for flags in [vec![], vec!["--schema", "not-opened"], vec!["--defaults", "not-opened"]] {
        assert_eq!(command("extract", &flags).execute_owned(NoRead, NoWrite), Err(CandidateError::Unavailable));
    }
}
