//! Parser and immutable-lifetime regressions; no native or storage execution.
use super::*;
use crate::{jobs::JobWork, tokenizer::untrusted::UntrustedEncoding};
const PIN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn argv(operation: &str, snapshot: bool, extra: &[&str]) -> Vec<String> {
    let mut words = vec!["candidate", "source-int4", "extract-semantic-document-job", operation,
        "--memory-mib", "65536", "--schema", "private-schema.json",
        "--semantic-options", "private-options.json", "--job-dir", "private/job",
        "--key-file", "private/key", "--limits", "private-limits.json",
        "--job-id", "0123456789abcdef0123456789abcdef", "--store-results",
        "--max-input-bytes", "4096", "--max-chunks", "4", "--max-chunk-bytes", "1024",
        "--max-result-bytes", "65536", "--max-map-result-bytes", "262144",
        "--max-live-value-bytes", "1048576", "--max-total-value-bytes", "1048576",
        "--max-object-json-bytes", "65536", "--max-object-result-bytes", "262144",
        "--max-grammar-states", "512", "--max-new-tokens", "64",
        "--review-max-chunks", "4", "--review-max-chunk-bytes", "2048",
        "--review-max-result-bytes", "262144", "--review-max-retained-result-bytes", "524288"];
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
    assert!(matches!(command.task, Int4Task::SemanticObjectDocumentJob));
    assert!(command.batch.is_none());
    let Some(Int4JobCommand::SemanticObjectDocument(job)) = command.job else { panic!("wrong document job"); };
    (job, command.args)
}
pub(super) fn options() -> Int4SemanticOptions { super::super::tests::fixture().3 }
pub(super) fn lifetime(command: &JobArgs, args: &Int4Args, options: &Int4SemanticOptions) -> JobLimits {
    let twice = options.max_model_work.checked_add(options.max_model_work).unwrap();
    let map = command.semantic.validate(args).unwrap().mapping().mapping.mapping;
    JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: args.max_input_bytes,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: options.limits.max_output_bytes as usize,
        max_spool_bytes: 32 * MIB, max_materialized_bytes: 32 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 5,
        max_work: JobWork::from_int4(twice, map.max_mask_visits * 3) }
}
pub(super) struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private population before admission"); }
}
pub(super) struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused document job published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused document job flushed output"); }
}

#[test]
fn source_snapshot_retained_input_and_compute_share_the_complete_document_contract() {
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
        assert_eq!(initial.map.common.snapshot_pin.is_some(), snapshot);
        assert_eq!(admitted.object.map.common.compute.threads, 4);
        assert!(admitted.object.map.common.prefill.is_none());
        assert_eq!(admitted.object.map.common.native, initial.map.common.native);
        assert_eq!(admitted.object.mapping().mapping.mapping.chunks.max_input_bytes, 4096);
        assert_eq!(admitted.object.mapping().mapping.mapping.chunks.max_chunks, 4);
        assert_eq!(admitted.object.object.max_json_bytes, 65536);
        assert_eq!(admitted.review.chunks.max_input_bytes, 4096 + options.limits.max_claim_bytes);
        assert_eq!(admitted.review.chunks.max_chunks, 4);
        assert!(command.args.storage.store_results && command.args.storage.materialize);
        assert!(args.input_encoding.is_default());
        assert_eq!(matches!(command.args.semantic.object.map.encoding.policy().unwrap(),
            UntrustedEncoding::SafeBpeV1(_)), compact);
        assert_eq!(admitted.task.max_output_bytes, 65536);
    } } }
}

#[test]
fn native_and_mask_authority_never_multiply_while_grounding_funds_every_lifetime_attempt() {
    let (command, args) = command("start", false, &[]);
    let options = options();
    let initial = command.args.validate(&args).unwrap();
    let job = lifetime(&command.args, &args, &options);
    let first = command.args.lifetime_limits(&args, initial, &options, job).unwrap();
    let per = initial.mapping().mapping.verification;
    assert_eq!(first.native.max_model_work, job.max_work.int4_model_work());
    assert_eq!(first.native.max_mask_node_visits, job.max_work.mask_node_visits);
    assert_eq!(first.native.max_verification, lifetime_grounding(per, job.max_attempts).unwrap());
    let mut larger = job; larger.max_items = 100_000; larger.max_attempts *= 2;
    let second = command.args.lifetime_limits(&args, initial, &options, larger).unwrap();
    assert_eq!(second.native.max_model_work, first.native.max_model_work);
    assert_eq!(second.native.max_mask_node_visits, first.native.max_mask_node_visits);
    assert_eq!(second.native.max_verification.max_fields, first.native.max_verification.max_fields * 2);
    assert_eq!(second.native.max_verification.max_matches, first.native.max_verification.max_matches * 2);
    assert_eq!(second.native.max_verification.max_scan_steps, first.native.max_verification.max_scan_steps * 2);
    assert_eq!(second.task, first.task);
    assert_eq!(second.review.chunks.max_input_bytes, first.review.chunks.max_input_bytes);
    assert_eq!(second.review.max_model_work, first.review.max_model_work);
    assert_eq!(second.review.max_result_bytes, first.review.max_result_bytes);
    for axis in 0..3 {
        let mut cap = GroundingBudget { max_fields: 1, max_matches: 1, max_scan_steps: 1 };
        match axis { 0 => cap.max_fields = usize::MAX, 1 => cap.max_matches = usize::MAX,
            _ => cap.max_scan_steps = u64::MAX }
        assert_eq!(lifetime_grounding(cap, 2).unwrap_err().code, "semantic_object_job_grounding_lifetime");
    }
    assert!(lifetime_grounding(per, 0).is_err());
}

#[test]
fn all_six_persisted_axes_and_combined_result_cap_cover_one_whole_attempt() {
    let (command, args) = command("resume", true, &[]);
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
            4 => short.max_work.model.projections.multiply_accumulates =
                options.max_model_work.projections.multiply_accumulates - 1,
            _ => short.max_work.mask_node_visits = initial.mapping().mapping.mapping.max_mask_visits - 1,
        };
        assert_eq!(command.args.lifetime_limits(&args, initial, &options, short).unwrap_err().code,
            "semantic_object_job_lifetime_work", "axis={axis}");
    }
    let mut first_stage_only = job;
    first_stage_only.max_result_bytes = initial.map.common.result_bytes;
    assert_eq!(command.args.lifetime_limits(&args, initial, &options, first_stage_only).unwrap_err().code,
        "semantic_object_job_input_or_combined_result_limit");
    let mut oversize = job; oversize.max_input_bytes_per_item += 1;
    assert_eq!(command.args.lifetime_limits(&args, initial, &options, oversize).unwrap_err().code,
        "semantic_object_job_input_or_combined_result_limit");
    let mut missing = job; missing.max_attempts = 0;
    assert_eq!(command.args.lifetime_limits(&args, initial, &options, missing).unwrap_err().code,
        "semantic_object_job_limits");
}

#[test]
fn invalid_storage_review_and_schedule_options_refuse_before_private_io() {
    for case in 0..8 {
        let (mut command, args) = command("start", false, &[]);
        match case {
            0 => command.args.storage.store_results = false,
            1 => command.args.storage.job_dir = PathBuf::from("-"),
            2 => command.args.storage.key_file = PathBuf::new(),
            3 => command.args.storage.limits_file = PathBuf::from("-"),
            4 => command.args.storage.max_stream_mib = 0,
            5 => command.args.storage.max_input_lines = 0,
            6 => command.args.storage.journal_memory_mib = u64::MAX,
            _ => command.args.storage.serialization_memory_mib = 0,
        }
        assert_eq!(command.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().exit, ErrorCode::Usage);
    }
    for flags in [
        &["--max-selected-rows", "32"][..], &["--cohort-chunks", "2"][..],
        &["--compute-threads", "0"][..], &["--compute-threads", "65"][..],
        &["--review-max-tokenizer-calls", "0"][..],
        &["--review-minimum-classified-coverage-ppm", "1000001"][..],
        &["--prefill-rows", "2"][..], &["--thinking", "on"][..],
    ] {
        if let Ok(parsed) = parse(argv("start", false, flags)) {
            let Some(Int4JobCommand::SemanticObjectDocument(command)) = parsed.job else { panic!("wrong route"); };
            assert!(command.execute_owned(parsed.args, NoRead, &mut NoWrite).is_err(), "{flags:?}");
        }
    }
}

#[test]
fn root_owned_dispatch_reports_bounded_errors_without_private_inputs_or_partial_results() {
    let parsed = parse(argv("start", false, &[])).unwrap();
    assert_eq!(parsed.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    let mut parsed = parse(argv("resume", true, &[])).unwrap();
    let Some(Int4JobCommand::SemanticObjectDocument(job)) = &mut parsed.job else { panic!("wrong job"); };
    job.args.storage.store_results = false;
    let mut diagnostics = Vec::new();
    assert_eq!(parsed.run_owned(NoRead, NoWrite, &mut diagnostics), ErrorCode::Usage.as_process_exit());
    let report: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
    assert_eq!(report["operation"], "resume");
    assert_eq!(report["code"], "semantic_object_job_storage_options");
    assert_eq!(report["durable_progress_may_exist"], true);
    assert!(report.get("cancellation").is_none());
    assert!(diagnostics.len() < job_cli::REPORT_BYTES);
    let text = std::str::from_utf8(&diagnostics).unwrap();
    for private in ["private-schema", "private-options", "private.snapshot", "private/job", "private/key", PIN] {
        assert!(!text.contains(private));
    }
}

#[test]
fn retained_replay_and_tail_repair_require_the_matching_explicit_operation() {
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
    for required in ["--schema", "--semantic-options", "--job-dir", "--key-file", "--limits", "--job-id"] {
        let mut words = argv("start", false, &[]);
        let at = words.iter().position(|word| word == required).unwrap();
        words.drain(at..at + 2);
        assert!(parse(words).is_err(), "{required}");
    }
    assert!(definition().try_get_matches_from(["extract-semantic-document-job"]).is_err());
    assert!(definition().try_get_matches_from(["extract-semantic-document-job", "verify"]).is_err());
}

#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unsupported_owned_profile_refuses_before_any_private_configuration_or_population() {
    let (command, args) = command("start", false, &[]);
    assert_eq!(command.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().code,
        "owned_job_profile_unavailable");
}
