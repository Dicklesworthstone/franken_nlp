//! Fixed policy, complete admission and owned-routing checks without a model.
use super::*;
use serde_json::json;

fn parse(extra: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", "resolve-corpus", "--discover-entities",
        "--policy", "private-policy.json", "--source-dir", "private-source",
        "--source-manifest", "private-manifest", "--memory-mib", "16384"];
    argv.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("INT4 entity corpus required"); };
    Ok(command)
}
pub(super) fn args(extra: &[&str]) -> (CorpusArgs, Int4Args) {
    let command = parse(extra).unwrap();
    let Int4Task::ResolveCorpus(options) = command.task else { panic!("INT4 entity corpus required"); };
    (options, command.args)
}
fn policy() -> serde_json::Value {
    json!({"ner":{"types":["person","organization"],"max_entities":8,"max_mention_scalars":64},
        "options":{"blocking":"exact_surface","context_scalars":128,"minimum_margin_milli":1000}})
}
pub(super) struct NoIo;
impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private input accessed"); } }
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("stdout accessed"); }
    fn flush(&mut self) -> io::Result<()> { panic!("stdout flushed"); }
}

#[test]
fn corpus_requires_explicit_discovery_and_policy_without_generation_controls() {
    for required in [vec!["--discover-entities"], vec!["--policy", "private-policy.json"], vec![]] {
        let mut argv = vec!["candidate", "source-int4", "resolve-corpus", "--source-dir", "s",
            "--source-manifest", "m", "--memory-mib", "16384"];
        argv.extend(required);
        assert!(crate::candidate_cli::definition().try_get_matches_from(argv).is_err());
    }
    for flags in [vec!["--seed", "00"], vec!["--max-new-tokens", "128"], vec!["--prefill-rows", "4"],
        vec!["--max-candidate-tokens", "2"], vec!["--stop", "END"], vec!["--think"],
        vec!["--model", "int8.fnlpq"], vec!["--schema", "schema.json"]] {
        assert!(parse(&flags).is_err());
    }
    let (command, args) = args(&[]);
    let limits = command.validate(&args).unwrap();
    assert_eq!(limits.transport.max_corpus_bytes, 1_048_576);
    assert_eq!(limits.transport.max_document_bytes, 1_048_576);
    assert_eq!(format!("{command:?}"), "EntityCorpusArgs");
}

#[test]
fn snapshot_selection_keeps_independent_pin_and_forbids_source_fallback() {
    let pin = "38".repeat(32);
    let matches = crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "source-int4", "resolve-corpus", "--discover-entities", "--chunked",
        "--policy", "private-policy.json", "--snapshot", "private-model.snapshot",
        "--expected-model-sha256", &pin, "--memory-mib", "16384",
    ]).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("INT4 route"); };
    let Int4Task::ResolveCorpus(options) = command.task else { panic!("entity corpus route"); };
    let limits = options.validate(&command.args).unwrap();
    assert_eq!(limits.common.snapshot_pin.unwrap().model_content_sha256, [0x38; 32]);
    assert!(command.args.source_dir.is_none());
    assert!(parse(&["--snapshot", "private-model.snapshot", "--expected-model-sha256", &pin]).is_err());
}

#[test]
fn fixed_policy_rejects_duplicate_keys_omissions_and_authority_overrides() {
    let parsed = Policy::parse(&policy().to_string()).unwrap();
    assert_eq!(parsed.ner.max_entities, 8);
    assert_eq!(parsed.options.blocking, crate::corpus::resolve::BlockingPolicy::ExactSurface);
    for field in ["documents", "budget", "work", "model", "execution_identity", "seed", "thinking", "tools"] {
        let mut value = policy(); value[field] = json!({});
        assert!(Policy::parse(&value.to_string()).is_err());
    }
    for text in ["{}", "[]", "null", r#"{"ner":{},"ner":{},"options":{}}"#,
        r#"{"ner":{},"\u006eer":{},"options":{}}"#] {
        assert!(Policy::parse(text).is_err());
    }
    for missing in ["ner", "options"] {
        let mut value = policy(); value.as_object_mut().unwrap().remove(missing);
        assert!(Policy::parse(&value.to_string()).is_err());
    }
    let text = policy().to_string();
    let duplicate = text.replacen("\"max_entities\":", "\"max_entities\":1,\"max_entities\":", 1);
    assert!(Policy::parse(&duplicate).is_err());
    let mut large = text; large.push_str(&" ".repeat(POLICY_BYTES));
    assert_eq!(Policy::parse(&large).err().unwrap().code, "entity_corpus_policy_bytes");
}

#[test]
fn fixed_policy_validates_ner_schema_and_explicit_resolution_margin() {
    for (field, value) in [("types", json!(["person", "person"])), ("types", json!(["invented"])),
        ("max_entities", json!(0)), ("max_mention_scalars", json!(0))] {
        let mut invalid = policy(); invalid["ner"][field] = value;
        assert!(Policy::parse(&invalid.to_string()).is_err());
    }
    for (field, value) in [("context_scalars", 2049), ("minimum_margin_milli", 0),
        ("minimum_margin_milli", 1_000_001)] {
        let mut invalid = policy(); invalid["options"][field] = json!(value);
        assert_eq!(Policy::parse(&invalid.to_string()).err().unwrap().code, "entity_corpus_resolution_policy");
    }
}

#[test]
fn epochs_reuse_one_work_mask_and_grounding_allowance_without_multiplication() {
    for flags in [&[][..], &["--chunked", "--max-ner-chunks", "8", "--max-snapshot-ner-chunks", "32"][..]] {
        let (mut command, args) = args(flags);
        let first = command.validate(&args).unwrap();
        let config = command.configuration(&args, first.common, Policy::parse(&policy().to_string()).unwrap()).unwrap();
        assert_eq!(config.max_model_work, command.resolution.scoring.work_ceiling());
        assert_eq!(config.scoring.max_model_work, config.max_model_work);
        assert_eq!(config.masks.max_visits_per_run, 1_000_000_000_000);
        assert_eq!(config.verification.max_matches, 1_000_000);
        assert_eq!(config.verification.max_scan_steps, 268_435_456);
        assert_eq!(config.scoring.planning.per_head.max_output_tokens, 2);
        assert_eq!(config.ner_budget.max_output_tokens, 256);
        assert_eq!(config.chunks.is_some(), !flags.is_empty());
        command.stream.max_snapshots = 1;
        let second = command.validate(&args).unwrap();
        let changed = command.configuration(&args, second.common, Policy::parse(&policy().to_string()).unwrap()).unwrap();
        assert_eq!(config.max_model_work, changed.max_model_work);
        assert_eq!(config.masks.max_visits_per_run, changed.masks.max_visits_per_run);
        assert_eq!(config.verification.max_matches, changed.verification.max_matches);
        assert_eq!(config.graph.max_documents, second.transport.max_documents);
        assert_eq!(first.common.minimum_memory_bytes, second.common.minimum_memory_bytes);
    }
}

#[test]
fn total_output_always_reserves_a_complete_epoch_and_the_terminal_footer() {
    let (mut command, args) = args(&[]);
    let limits = command.validate(&args).unwrap();
    command.stream.max_total_output_bytes = limits.transport.max_output_line_bytes as u64 + ENVELOPE_BYTES as u64;
    command.validate(&args).unwrap();
    command.stream.max_total_output_bytes -= 1;
    assert_eq!(command.validate(&args).err().unwrap().code, "entity_corpus_output_too_small");
}

#[test]
fn malformed_limits_and_borrowed_dispatch_refuse_before_any_private_io() {
    assert_eq!(parse(&[]).unwrap().execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    for flags in [vec!["--max-records", "0"], vec!["--max-snapshots", "0"],
        vec!["--max-total-input-bytes", "0"], vec!["--max-total-output-bytes", "1"],
        vec!["--max-line-bytes", "1048575"], vec!["--max-document-bytes", "0"],
        vec!["--max-document-bytes", "1048577"], vec!["--max-grounding-fields", "0"],
        vec!["--max-grounding-matches", "1000001"], vec!["--max-grounding-scan-steps", "0"]] {
        let mut diagnostics = Vec::new();
        let status = parse(&flags).unwrap().run_owned(NoIo, NoIo, &mut diagnostics);
        assert_eq!(status, ErrorCode::Usage.as_process_exit());
        let report: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
        assert_eq!(report["protocol"], ENTITY_STREAM_PROTOCOL);
        assert_eq!(report["event"], "run_error");
        assert_eq!(report["status"], "incomplete");
        assert_eq!(report["cluster_scope"], "epoch_local");
        assert_eq!(report["completed_epochs_may_exist"], true);
        assert!(!String::from_utf8(diagnostics).unwrap().contains("private"));
    }
    let (mut command, args) = args(&[]);
    command.resolution.discovery.discover_entities = false;
    assert_eq!(command.validate(&args).err().unwrap().code, "entity_corpus_policy_required");
    command.resolution.discovery.discover_entities = true;
    command.stream.policy = PathBuf::from("-");
    assert_eq!(command.validate(&args).err().unwrap().code, "entity_corpus_policy_required");
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_does_not_open_policy_model_input_or_output() {
    let (command, args) = args(&[]);
    assert_eq!(command.execute_owned(args, NoIo, NoIo).unwrap_err().code, "entity_corpus_profile_unavailable");
}
