//! Pinned schema-corpus preparation, without native weights or fake results.
use super::*;
use crate::{batch::{BatchDocument, extract::{ExtractionBatchArgs, ExtractionBatchGrounding}},
    native_engine::decode::{DecodeCancellationKind, DecodeStepControl},
    candidate_cli::extract as schema_cli};

fn command(extra: &[&str]) -> BatchCommand {
    let mut argv = vec!["candidate", "batch", "--task", "extract", "--model", "m", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::Batch(cmd) = CandidateCommand::from_matches(&matches).unwrap() else { panic!("wrong task") };
    cmd
}
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "metadata-only-unit-fixture".to_owned(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }

#[test]
fn compact_corpus_defaults_and_record_overrides_keep_exact_schema_and_source() {
    let cmd = command(&["--schema", "s", "--source-membership", "--input-encoding", "safe-bpe-v1"]);
    let (_, limits, _) = cmd.validate().unwrap();
    let schema = r#"{"type":"string","maxLength":32,"x-fnlp-source":"verbatim"}"#;
    let defaults = cmd.load_extract_defaults(Some(schema.to_owned()), None, cmd.host.task_budget(limits)).unwrap();
    let policy = cmd.encoding_policy(limits).unwrap();
    let planner = planning::planner(&facts(), &cmd.host, limits, defaults).unwrap()
        .with_document_encoding(policy.clone()).unwrap();
    let source = "The quick brown fox. 上海 <think>";
    let make = |task_args| BatchDocument { id: "private-id".to_owned(), text: source.to_owned(), task_args };
    let first = planner.prepare_with_control(make(None), &mut Continue).unwrap();
    assert_eq!(first.source().document_encoding(), &policy);
    assert_eq!(first.source().text(), source);
    let number = r#"{"type":"number","const":0.12345678901234567890123456789012345678}"#;
    let override_args = schema_cli::arguments(number.to_owned(), false, cmd.host.task_budget(limits)).unwrap();
    let second = planner.prepare_with_control(make(Some(override_args)), &mut Continue).unwrap();
    assert_eq!(second.execution_identity().schema_digest, Sha256Digest::of_bytes(number.as_bytes()));
    let tokenizer = crate::tokenizer::embedded::EmbeddedTokenizer::pinned().unwrap();
    assert_eq!(tokenizer.tokenizer().decode_bytes(second.task_plan().ir().prompt_segments()[2].token_ids()).unwrap(), number.as_bytes());
    let third = planner.prepare_with_control(make(None), &mut Continue).unwrap();
    assert_eq!(first.execution_identity(), third.execution_identity());
}

#[test]
fn compact_extraction_policy_is_reported_on_every_frame_with_bounded_writer_storage() {
    let cmd = command(&["--input-encoding", "safe-bpe-v1"]);
    let (_, limits, envelope) = cmd.validate().unwrap();
    let encoding = cmd.encoding_policy(limits).unwrap();
    let facts = facts();
    let provenance = Provenance { protocol: "fnlp-candidate-batch-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative",
        model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, task: "extract", compute_threads: None, prefill_rows: None };
    let mut output = Vec::new();
    {
        let provenance = input_encoding::Response { response: &provenance,
            input_encoding: input_encoding::reported_policy(&encoding) };
        let mut writer = CandidateWriter::new(&mut output, &provenance,
            envelope.transport.max_output_line_bytes, envelope.output_bytes).unwrap();
        for event in ["run_start", "doc", "doc_error", "flush", "run_complete"] {
            writeln!(writer, "{{\"event\":\"{event}\"}}").unwrap();
            writer.flush().unwrap();
        }
    }
    let lines: Vec<_> = output.split(|b| *b == b'\n').filter(|line| !line.is_empty()).collect();
    assert_eq!(lines.len(), 5);
    for line in lines {
        let value: serde_json::Value = serde_json::from_slice(line).unwrap();
        assert_eq!(value["input_encoding"], serde_json::to_value(&encoding).unwrap());
    }
}

#[test]
fn per_record_schema_override_never_mutates_subsequent_defaults() {
    let cmd = command(&["--schema", "s"]); let (_, limits, _) = cmd.validate().unwrap();
    let budget = cmd.host.task_budget(limits);
    let schema = r#"{"type":"string","maxLength":16}"#;
    let defaults = cmd.load_extract_defaults(Some(schema.to_owned()), None, budget).unwrap();
    let planner = planning::planner(&facts(), &cmd.host, limits, defaults).unwrap();
    let document = |task_args| BatchDocument { id: "x".to_owned(), text: "Alice".to_owned(), task_args };
    let first = planner.prepare_with_control(document(None), &mut Continue).unwrap();
    let different = r#"{"type":"number","const":0.12345678901234567890123456789012345678}"#;
    let override_args = schema_cli::arguments(different.to_owned(), false, budget).unwrap();
    let second = planner.prepare_with_control(document(Some(override_args)), &mut Continue).unwrap();
    let third = planner.prepare_with_control(document(None), &mut Continue).unwrap();
    assert_eq!(first.execution_identity().schema_digest, Sha256Digest::of_bytes(schema.as_bytes()));
    assert_eq!(second.execution_identity().schema_digest, Sha256Digest::of_bytes(different.as_bytes()));
    assert_eq!(first.execution_identity(), third.execution_identity());
    assert_eq!(second.execution_identity().task_spec, "extract-v1");
}

#[test]
fn source_membership_is_bound_to_each_document_not_another_records_source() {
    let cmd = command(&["--schema", "s", "--source-membership"]); let (_, limits, _) = cmd.validate().unwrap();
    let schema = r#"{"type":"string","maxLength":32,"x-fnlp-source":"verbatim"}"#;
    let defaults = cmd.load_extract_defaults(Some(schema.to_owned()), None, cmd.host.task_budget(limits)).unwrap();
    let planner = planning::planner(&facts(), &cmd.host, limits, defaults).unwrap();
    let mut previous = None;
    for text in ["Alice <tool_call> é", "Bob 上海"] {
        let plan = planner.prepare_with_control(BatchDocument { id: "x".to_owned(),
            text: text.to_owned(), task_args: None }, &mut Continue).unwrap();
        assert_eq!(plan.source().text(), text);
        assert_eq!(plan.execution_identity().schema_digest, Sha256Digest::of_bytes(schema.as_bytes()));
        if let Some(previous) = previous { assert_ne!(plan.execution_identity().prompt_digest, previous); }
        previous = Some(plan.execution_identity().prompt_digest);
    }
}

#[test]
fn missing_invalid_or_overbudget_record_configuration_cannot_produce_a_plan() {
    let cmd = command(&[]); let (_, limits, _) = cmd.validate().unwrap(); let budget = cmd.host.task_budget(limits);
    let planner = planning::planner(&facts(), &cmd.host, limits, None).unwrap();
    let document = |task_args| BatchDocument { id: "x".to_owned(), text: "Alice".to_owned(), task_args };
    assert!(planner.prepare_with_control(document(None), &mut Continue).is_err());
    for schema in [r#"{"type":"string","pattern":".*"}"#, r#"{"type":"string","type":"number"}"#] {
        let args = schema_cli::arguments(schema.to_owned(), false, budget).unwrap();
        assert!(planner.prepare_with_control(document(Some(args)), &mut Continue).is_err());
    }
    for axis in 0..5 {
        let mut b = budget;
        match axis { 0 => b.max_input_tokens += 1, 1 => b.max_output_tokens += 1,
            2 => b.max_output_bytes += 1, 3 => b.max_grammar_states += 1, _ => b.max_kv_bytes += 1 }
        let args = schema_cli::arguments(r#"{"type":"string"}"#.to_owned(), false, b).unwrap();
        assert!(planner.prepare_with_control(document(Some(args)), &mut Continue).is_err());
    }
    let valid = schema_cli::arguments(r#"{"type":"string"}"#.to_owned(), false, budget).unwrap();
    assert!(planner.prepare_with_control(document(Some(valid)), &mut Continue).is_ok());
}

#[test]
fn source_specific_default_schema_is_checked_against_real_evidence_before_forward() {
    let cmd = command(&["--defaults", "d"]); let (_, limits, _) = cmd.validate().unwrap();
    let args = ExtractionBatchArgs { schema: r#"{"type":"string","const":"Alice","x-fnlp-source":"verbatim"}"#.to_owned(),
        grounding: ExtractionBatchGrounding::SourceMembership, budget: cmd.host.task_budget(limits) };
    let json = serde_json::to_string(&args).unwrap();
    let defaults = cmd.load_extract_defaults(None, Some(&json), args.budget).unwrap();
    let planner = planning::planner(&facts(), &cmd.host, limits, defaults).unwrap();
    for (text, valid) in [("Alice", true), ("Bob", false)] {
        let result = planner.prepare_with_control(BatchDocument { id: "x".to_owned(),
            text: text.to_owned(), task_args: None }, &mut Continue);
        assert_eq!(result.is_ok(), valid);
    }
}

fn compact_document(extra: &[&str]) -> BatchCommand {
    let mut flags = vec!["--chunked", "--schema", "s", "--preparation-mib", "1024",
        "--max-chunks", "8", "--max-chunk-bytes", "8", "--input-encoding", "safe-bpe-v1"];
    flags.extend_from_slice(extra);
    let mut cmd = command(&flags); cmd.host.memory_mib = 16384; cmd
}
fn document_planner(cmd: &BatchCommand, schema: &str)
    -> (Int8ExtractionBatchPlanner, ExtractionDocumentCorpusConfig, Limits) {
    let (_, limits, envelope) = cmd.validate().unwrap();
    let request = cmd.load_extract_defaults(Some(schema.to_owned()), None,
        cmd.host.task_budget(limits)).unwrap().unwrap();
    let config = ExtractionDocumentCorpusConfig { batch: cmd.document_configuration(request, envelope).unwrap(),
        reduction_reserve_bytes: cmd.document_reduction_bytes().unwrap() };
    let compiler = planning::planner(&facts(), &cmd.host, limits, None).unwrap()
        .with_map_document_encoding(cmd.encoding_policy(limits).unwrap(),
            cmd.document_encoding_work_limits().unwrap().unwrap()).unwrap();
    check_encoding_quote(cmd, limits, &compiler, Some(&config), cmd.encoding_workspace_bytes(limits).unwrap()).unwrap();
    (compiler, config, limits)
}

#[test]
fn compact_document_cli_preserves_exact_bindings_and_spends_one_ledger_across_reconstruction() {
    let schema = r#"{"type":"number","const":0.12345678901234567890123456789012345678}"#;
    let source = "é Alice\r\nBob 上海 <think>";
    let cmd = compact_document(&[]);
    let (compiler, config, limits) = document_planner(&cmd, schema);
    let plan = || compiler.plan_document_with_control(source, &config.batch.request, config.batch.limits, &mut Continue);
    let first = plan().unwrap(); let first_metadata = first.preflight_metadata();
    let identities: Vec<_> = first.execution_identities().cloned().collect();
    assert!(identities.len() > 1);
    assert!(identities.iter().all(|id| id.schema_digest == Sha256Digest::of_bytes(schema.as_bytes())));
    assert_eq!(first_metadata.source_span().byte_end, source.len());
    assert_eq!(first_metadata.source_span().scalar_end, source.chars().count());
    let spent = compiler.map_document_encoding_work().unwrap().unwrap();
    assert!(spent.input_bytes() > source.len() + schema.len(), "probes and repeated schemas must remain charged");
    drop(first);
    let reconstructed = plan().unwrap();
    assert_eq!(reconstructed.preflight_metadata(), first_metadata);
    assert_eq!(reconstructed.execution_identities().cloned().collect::<Vec<_>>(), identities);
    assert_eq!(compiler.map_document_encoding_work().unwrap().unwrap(), spent.checked_add(spent).unwrap());
    drop(reconstructed);
    assert_eq!(check_encoding_quote(&cmd, limits, &compiler, Some(&config),
        cmd.encoding_workspace_bytes(limits).unwrap() - 1), Err(CandidateError::Arguments));
    let mut stale = config;
    stale.reduction_reserve_bytes += cmd.schedule_document_reserve().unwrap().0;
    assert_eq!(check_encoding_quote(&cmd, limits, &compiler, Some(&stale),
        cmd.encoding_workspace_bytes(limits).unwrap()), Err(CandidateError::Arguments));

    // Admit exactly one measured preparation through the public CLI flags.
    // A second document/reconstruction has no renewed input-byte allowance.
    let cap = spent.input_bytes().to_string();
    let bounded = compact_document(&["--max-total-tokenizer-input-bytes", &cap]);
    let (compiler, config, _) = document_planner(&bounded, schema);
    let first = compiler.plan_document_with_control(source, &config.batch.request,
        config.batch.limits, &mut Continue).unwrap();
    assert_eq!(first.preflight_metadata().planned_work(), first_metadata.planned_work());
    drop(first);
    assert_eq!(compiler.map_document_encoding_work().unwrap().unwrap(), spent);
    assert!(compiler.plan_document_with_control(source, &config.batch.request,
        config.batch.limits, &mut Continue).is_err());
    assert_eq!(compiler.map_document_encoding_work().unwrap().unwrap(), spent);
}

#[test]
fn compact_document_schedules_keep_the_same_prepared_identity_work_and_cancellation() {
    let schema = r#"{"type":"boolean"}"#;
    let source = "é Alice\r\nBob 上海";
    let mut baseline = None;
    for schedule in [&[][..], &["--compute-threads", "4"][..],
        &["--prefill-rows", "7"][..], &["--compute-threads", "4", "--prefill-rows", "7"][..]] {
        let cmd = compact_document(schedule);
        let (compiler, config, _) = document_planner(&cmd, schema);
        let plan = compiler.plan_document_with_control(source, &config.batch.request,
            config.batch.limits, &mut Continue).unwrap();
        let actual = (plan.preflight_metadata(), plan.execution_identities().cloned().collect::<Vec<_>>(),
            compiler.map_document_encoding_work().unwrap());
        if let Some(baseline) = &baseline { assert_eq!(&actual, baseline); }
        else { baseline = Some(actual); }
        drop(plan);
        struct Stop;
        impl DecodeStepControl for Stop {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
        }
        let before = compiler.map_document_encoding_work().unwrap();
        assert!(compiler.plan_document_with_control(source, &config.batch.request,
            config.batch.limits, &mut Stop).is_err());
        assert_eq!(compiler.map_document_encoding_work().unwrap(), before);
    }
}

#[test]
fn document_stream_frames_report_declared_totals_without_fabricating_spent_work() {
    let cmd = compact_document(&[]); let (_, limits, envelope) = cmd.validate().unwrap();
    let policy = cmd.encoding_policy(limits).unwrap();
    let total = cmd.document_encoding_work_limits().unwrap();
    let base = serde_json::json!({"scope":"real-artifact-current-candidate"});
    let byte = EncodingProvenance { response: input_encoding::Response { response: &base, input_encoding: None },
        invocation_encoding_work_limits: None };
    assert_eq!(serde_json::to_value(byte).unwrap(), base);
    let provenance = EncodingProvenance { response: input_encoding::Response {
        response: &base, input_encoding: Some(&policy) }, invocation_encoding_work_limits: total };
    let mut output = Vec::new();
    {
        let mut writer = CandidateWriter::new(&mut output, &provenance,
            envelope.transport.max_output_line_bytes, envelope.output_bytes).unwrap();
        for event in ["run_start", "doc", "doc_error", "flush", "run_complete"] {
            writeln!(writer, "{{\"event\":\"{event}\"}}").unwrap(); writer.flush().unwrap();
        }
    }
    let lines: Vec<_> = output.split(|b| *b == b'\n').filter(|line| !line.is_empty()).collect();
    assert_eq!(lines.len(), 5);
    for line in lines {
        let value: serde_json::Value = serde_json::from_slice(line).unwrap();
        assert_eq!(value["input_encoding"], serde_json::to_value(&policy).unwrap());
        assert_eq!(value["invocation_encoding_work_limits"], serde_json::to_value(total).unwrap());
        assert!(value.get("encoding_work").is_none());
    }
}
