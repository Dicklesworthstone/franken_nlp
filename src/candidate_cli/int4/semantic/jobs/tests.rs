//! Source-only parser, lifetime-authority and private-IO regressions.
//! No model execution, storage acknowledgement or quality evidence is implied.
use super::*;
use crate::{jobs::JobWork, tokenizer::untrusted::UntrustedEncoding};

const PIN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn argv(operation: &str, snapshot: bool, extra: &[&str]) -> Vec<String> {
    let mut words = vec!["candidate", "source-int4", "extract-semantic-job", operation,
        "--memory-mib", "65536", "--schema", "private-schema.json",
        "--semantic-options", "private-options.json", "--job-dir", "private/job",
        "--key-file", "private/key", "--limits", "private-limits.json",
        "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"];
    if snapshot { words.extend(["--snapshot", "private.snapshot", "--expected-model-sha256", PIN]); }
    else { words.extend(["--source-dir", "private-source", "--source-manifest", "private-manifest.json"]); }
    words.extend_from_slice(extra);
    words.into_iter().map(str::to_owned).collect()
}
fn parse(words: Vec<String>) -> Result<Int4Command, clap::Error> {
    let matches = crate::candidate_cli::definition().try_get_matches_from(words)?;
    match CandidateCommand::from_matches(&matches)? {
        CandidateCommand::SourceInt4(command) => Ok(command),
        _ => panic!("wrong candidate dispatcher"),
    }
}
pub(super) fn command(operation: &str, snapshot: bool, extra: &[&str]) -> (JobCommand, Int4Args) {
    let command = parse(argv(operation, snapshot, extra)).unwrap();
    assert!(matches!(command.task, Int4Task::SemanticJob));
    assert!(command.batch.is_none());
    let Some(Int4JobCommand::Semantic(job)) = command.job else { panic!("wrong retained semantic route"); };
    (job, command.args)
}
pub(super) fn options() -> Int4SemanticOptions { super::super::tests::fixture().3 }
pub(super) fn lifetime(command: &JobArgs, args: &Int4Args, options: &Int4SemanticOptions) -> JobLimits {
    let twice = options.max_model_work.checked_add(options.max_model_work).unwrap();
    JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: args.max_input_bytes,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: options.limits.max_output_bytes as usize,
        max_spool_bytes: 32 * MIB, max_materialized_bytes: 32 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 5,
        max_work: JobWork::from_int4(twice, command.semantic.extraction.mask_visits() * 3) }
}
pub(super) struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private population read before admission"); }
}
pub(super) struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused semantic job published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused semantic job flushed output"); }
}

#[test]
fn retained_semantic_routes_keep_explicit_model_policy_and_storage_identity() {
    let options = options();
    for operation in ["start", "resume"] { for snapshot in [false, true] { for compact in [false, true] {
        let mut flags = vec!["--compute-threads", "4", "--materialize",
            if operation == "start" { "--spool-input" } else { "--use-retained-input" },
            "--max-retained-input-mib", "8"];
        if compact { flags.extend(["--input-encoding", "safe-bpe-v1"]); }
        let (command, args) = command(operation, snapshot, &flags);
        command.args.storage.input_retention.validate(command.mode, &args.input).unwrap();
        let initial = command.args.validate(&args).unwrap();
        let job = lifetime(&command.args, &args, &options);
        let admitted = command.args.lifetime_limits(&args, initial, &options, job).unwrap();
        assert_eq!(command.mode.name(), operation);
        assert_eq!(args.input, PathBuf::from("-"));
        assert_eq!(initial.snapshot_pin.is_some(), snapshot);
        assert_eq!(admitted.common.compute.threads, 4);
        assert!(admitted.common.prefill.is_none());
        assert_eq!(admitted.common.native, initial.native);
        assert_eq!(command.args.semantic.extraction.schema_path(), std::path::Path::new("private-schema.json"));
        assert_eq!(command.args.semantic.semantic_options, PathBuf::from("private-options.json"));
        assert_eq!(job_cli::job_id_hex(command.args.storage.job_id), "0123456789abcdef0123456789abcdef");
        assert!(command.args.storage.store_results && command.args.storage.materialize);
        assert!(args.input_encoding.is_default(), "semantic encoding owns the whole-attempt ledger");
        assert_eq!(matches!(command.args.semantic.input_encoding.policy().unwrap(),
            UntrustedEncoding::SafeBpeV1(_)), compact);
        assert_eq!(admitted.task.max_output_bytes, 1 << 20);
        assert_eq!(admitted.common.result_bytes, options.limits.max_output_bytes as usize);
    } } }
    let (serial, args) = command("start", false, &["--save-snapshot", "private-new.snapshot"]);
    let common = serial.args.validate(&args).unwrap();
    assert_eq!(common.compute.threads, 1);
    assert_eq!(common.compute.workspace_bytes, 0);
    assert_eq!(args.save_snapshot, Some(PathBuf::from("private-new.snapshot")));
}

#[test]
fn root_dispatch_uses_owned_job_io_and_refuses_borrowed_execution() {
    let command = parse(argv("start", false, &[])).unwrap();
    assert_eq!(command.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    let mut command = parse(argv("resume", true, &[])).unwrap();
    let Some(Int4JobCommand::Semantic(job)) = &mut command.job else { panic!("wrong job dispatcher"); };
    job.args.storage.store_results = false;
    let mut diagnostics = Vec::new();
    assert_eq!(command.run_owned(NoRead, NoWrite, &mut diagnostics), ErrorCode::Usage.as_process_exit());
    let report: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
    assert_eq!(report["operation"], "resume");
    assert_eq!(report["code"], "semantic_job_storage_options");
    assert_eq!(report["durable_progress_may_exist"], true);
    assert!(report.get("cancellation").is_none());
    assert!(diagnostics.len() < job_cli::REPORT_BYTES);
    let text = String::from_utf8(diagnostics).unwrap();
    for private in ["private-schema", "private-options", "private.snapshot", "private/job", "private/key", PIN] {
        assert!(!text.contains(private));
    }
}

#[test]
fn all_lifetime_work_axes_are_copied_without_population_or_attempt_multipliers() {
    let (mut command, args) = command("start", false, &[]);
    let options = options();
    let initial = command.args.validate(&args).unwrap();
    let job = lifetime(&command.args, &args, &options);
    let first = command.args.lifetime_limits(&args, initial, &options, job).unwrap();
    assert_eq!(first.native.max_model_work, job.max_work.int4_model_work());
    assert_eq!(first.native.masks.max_visits_per_run, job.max_work.mask_node_visits);
    assert_eq!(first.native.masks.max_visits_per_item, command.args.semantic.extraction.mask_visits());
    assert_eq!(first.native.masks.per_mask.max_trie_node_visits,
        command.args.semantic.extraction.masks().max_trie_node_visits);
    let mut larger = job; larger.max_items = 100_000; larger.max_attempts = 1_000_000;
    command.args.storage.max_stream_mib *= 2;
    command.args.storage.max_input_lines *= 2;
    let changed = command.args.lifetime_limits(&args, initial, &options, larger).unwrap();
    assert_eq!(changed.native.max_model_work, first.native.max_model_work);
    assert_eq!(changed.native.masks.max_visits_per_run, first.native.masks.max_visits_per_run);
    assert_eq!(changed.native.masks.max_visits_per_item, first.native.masks.max_visits_per_item);
    assert_eq!(changed.task, first.task);
    assert_eq!(changed.common.native, first.common.native);
    assert_eq!(changed.transport.max_stream_bytes, first.transport.max_stream_bytes * 2);
    assert_eq!(changed.transport.max_lines, first.transport.max_lines * 2);
    assert_eq!(changed.io_reserve_bytes, first.io_reserve_bytes);
    assert_eq!(changed.journal_reserve_bytes, first.journal_reserve_bytes);
    assert_eq!(changed.serialization_reserve_bytes, first.serialization_reserve_bytes);
}

#[test]
fn every_lifetime_axis_must_cover_one_complete_extraction_and_review_attempt() {
    let (command, args) = command("start", false, &[]);
    let options = options();
    let initial = command.args.validate(&args).unwrap();
    let job = lifetime(&command.args, &args, &options);
    for axis in 0..6 {
        let mut short = job;
        match axis {
            0 => short.max_work.model.forward_positions = options.max_model_work.forward_positions - 1,
            1 => short.max_work.model.projected_logits = options.max_model_work.projected_logits - 1,
            2 => short.max_work.model.attention_pairs = options.max_model_work.attention_pairs - 1,
            3 => short.max_work.model.projections.dot_products = options.max_model_work.projections.dot_products - 1,
            4 => short.max_work.model.projections.multiply_accumulates = options.max_model_work.projections.multiply_accumulates - 1,
            _ => short.max_work.mask_node_visits = command.args.semantic.extraction.mask_visits() - 1,
        }
        assert_eq!(command.args.lifetime_limits(&args, initial, &options, short).unwrap_err().code,
            "semantic_job_lifetime_work", "axis={axis}");
    }
}

#[test]
fn stored_combined_result_and_record_caps_cannot_exceed_admission() {
    let (command, args) = command("resume", true, &[]);
    let options = options();
    let initial = command.args.validate(&args).unwrap();
    let job = lifetime(&command.args, &args, &options);
    let limits = command.args.lifetime_limits(&args, initial, &options, job).unwrap();
    assert_eq!(limits.task.max_output_bytes, 1 << 20);
    assert!(limits.common.result_bytes as u64 > limits.task.max_output_bytes);
    let mut first_stage_only = job; first_stage_only.max_result_bytes = limits.task.max_output_bytes as usize;
    assert_eq!(command.args.lifetime_limits(&args, initial, &options, first_stage_only).unwrap_err().code,
        "semantic_job_input_or_combined_result_limit");
    let mut short = job; short.max_result_bytes -= 1;
    assert_eq!(command.args.lifetime_limits(&args, initial, &options, short).unwrap_err().code,
        "semantic_job_input_or_combined_result_limit");
    let mut input = job; input.max_input_bytes_per_item += 1;
    assert_eq!(command.args.lifetime_limits(&args, initial, &options, input).unwrap_err().code,
        "semantic_job_input_or_combined_result_limit");
    let mut invalid = job; invalid.max_attempts = 0;
    assert_eq!(command.args.lifetime_limits(&args, initial, &options, invalid).unwrap_err().code,
        "semantic_job_limits");
}

#[test]
fn exact_schema_grounding_and_claim_rules_are_checked_without_population_io() {
    let schema = r#"{ "type":"object", "properties":{"amount":{"type":"integer","const":9007199254740993}}, "required":["amount"], "additionalProperties":false }"#;
    let (command, args) = command("start", false, &[]);
    let options = options();
    let initial = command.args.validate(&args).unwrap();
    let job = lifetime(&command.args, &args, &options);
    let limits = command.args.lifetime_limits(&args, initial, &options, job).unwrap();
    let defaults = command.args.defaults(schema.to_owned(), limits.task, &options).unwrap();
    assert_eq!(defaults.schema, schema);
    assert_eq!(defaults.budget, limits.task);
    let mut wrong = options.clone();
    wrong.specification.claims[0].path = vec![
        crate::tasks::extract::semantic::ClaimPathStep::Property("absent".to_owned())];
    assert!(command.args.defaults(schema.to_owned(), limits.task, &wrong).is_err());
    assert!(command.args.defaults(r#"{"type":"object","type":"object"}"#.to_owned(), limits.task, &options).is_err());

    let (grounded, _) = self::command("resume", false, &["--source-membership"]);
    assert!(grounded.args.defaults(schema.to_owned(), limits.task, &options).is_err());
    let verbatim = r#"{"type":"object","properties":{"amount":{"type":"integer"},"excerpt":{"type":"string","enum":["上海"],"maxLength":2,"x-fnlp-source":"verbatim"}},"required":["amount","excerpt"],"additionalProperties":false}"#;
    let defaults = grounded.args.defaults(verbatim.to_owned(), limits.task, &options).unwrap();
    assert_eq!(defaults.schema, verbatim);
    assert_eq!(defaults.grounding, ExtractionBatchGrounding::SourceMembership);
    assert!(command.args.defaults(verbatim.to_owned(), limits.task, &options).is_err());
}

#[test]
fn invalid_private_paths_storage_and_memory_options_refuse_before_io() {
    for case in 0..10 {
        let (mut command, args) = command("start", false, &[]);
        match case {
            0 => command.args.storage.store_results = false,
            1 => command.args.storage.job_dir = PathBuf::from("-"),
            2 => command.args.storage.key_file = PathBuf::new(),
            3 => command.args.storage.limits_file = PathBuf::from("-"),
            4 => command.args.storage.max_stream_mib = 0,
            5 => command.args.storage.max_input_lines = 0,
            6 => command.args.storage.journal_memory_mib = 0,
            7 => command.args.storage.serialization_memory_mib = 0,
            8 => command.args.storage.journal_memory_mib = u64::MAX,
            _ => command.args.storage.serialization_memory_mib = args.memory_mib + 1,
        }
        assert_eq!(command.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().exit,
            ErrorCode::Usage, "case={case}");
    }
    for path in ["", "-", "https://example.invalid/private-options"] {
        let (mut command, args) = command("resume", true, &[]);
        command.args.semantic.semantic_options = PathBuf::from(path);
        assert_eq!(command.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().exit, ErrorCode::Usage);
    }
}

#[test]
fn retained_input_and_tail_repair_require_the_matching_explicit_operation() {
    assert!(matches!(command("resume", false, &[]).0.mode,
        RunMode::Resume { discard_uncommitted: false }));
    assert!(matches!(command("resume", false, &["--discard-uncommitted-tail"]).0.mode,
        RunMode::Resume { discard_uncommitted: true }));
    assert!(parse(argv("start", false, &["--discard-uncommitted-tail"])).is_err());
    for (operation, flag) in [("start", "--use-retained-input"), ("resume", "--spool-input")] {
        let (command, args) = command(operation, false, &[flag, "--max-retained-input-mib", "8"]);
        assert_eq!(command.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().code,
            "job_input_retention_options");
    }
    for flags in [&["--use-retained-input"][..],
        &["--use-retained-input", "--max-retained-input-mib", "8", "population.ndjson"][..],
        &["--use-retained-input", "--max-retained-input-mib", "8", "-"][..]] {
        assert!(parse(argv("resume", true, flags)).is_err());
    }
    assert!(definition().try_get_matches_from(["extract-semantic-job"]).is_err());
    assert!(definition().try_get_matches_from(["extract-semantic-job", "verify"]).is_err());
}

#[test]
fn unsupported_generation_projection_and_batch_controls_are_not_inherited() {
    for flags in [
        &["--seed", PIN][..], &["--logprobs"][..], &["--prefill-rows", "1"][..],
        &["--max-selected-rows", "32"][..], &["--cohort-documents", "2"][..],
        &["--defaults", "private-defaults"][..], &["--think"][..],
        &["--max-output-bytes", "65536"][..], &["--min-new-tokens", "0"][..],
        &["--max-total-forward-positions", "99999"][..],
        &["--max-total-tokenizer-input-bytes", "99999"][..],
    ] {
        assert!(parse(argv("start", false, flags)).is_err(), "{flags:?}");
    }
    for required in ["--schema", "--semantic-options", "--job-dir", "--key-file", "--limits", "--job-id"] {
        let mut words = argv("start", false, &[]);
        let at = words.iter().position(|word| word == required).unwrap();
        words.drain(at..at + 2);
        assert!(parse(words).is_err(), "{required}");
    }
}

#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unavailable_owned_job_profile_refuses_before_private_configuration_or_population() {
    let (command, args) = command("start", false, &[]);
    assert_eq!(command.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().code,
        "owned_job_profile_unavailable");
}
