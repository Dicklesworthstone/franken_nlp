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
