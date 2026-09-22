//! CLI contract tests; parser/help paths do not construct a model or runtime.
use super::*;
use clap::FromArgMatches;
use crate::jobs::{StoredJobReport, JobWork};
fn common_args(operation: &str) -> Vec<String> {
    ["job", operation, "--job-dir", "private-job", "--job-id", "07070707070707070707070707070707",
        "--key-file", "private-key", "--limits", "limits.json", "--memory-bytes", "134217728"]
        .into_iter().map(str::to_owned).collect()
}
fn report() -> StoredJobReport {
    StoredJobReport { schema_version: 1, verification_scope: "authenticated-stored-state-not-input-replay-v1",
        job_id: JobId([7; 16]), items: 2, committed: 1, attempts: 2, reserved_work: JobWork::default(),
        committed_spool_bytes: 512, uncommitted_spool_bytes: 0, staged_output_present: false,
        materialized: false, unacknowledged_output_present: false }
}
#[test]
fn all_management_commands_parse_without_any_model_arguments() {
    for operation in ["status", "verify", "materialize"] {
        let mut args = common_args(operation); args.push("--json".to_owned());
        if operation == "materialize" { args.push("--ordered".to_owned()); }
        let matches = definition().try_get_matches_from(args).unwrap();
        JobCommand::from_arg_matches(&matches).unwrap();
    }
}
#[test]
fn no_inference_overwrite_repair_or_secret_argv_surface_is_registered() {
    for operation in ["start", "resume", "purge"] {
        assert!(definition().try_get_matches_from(common_args(operation)).is_err());
    }
    for flag in ["--key", "--secret", "--model", "--overwrite", "--discard-tail", "--output"] {
        let mut args = common_args("materialize"); args.extend([flag.to_owned(), "private-value".to_owned()]);
        assert!(definition().try_get_matches_from(args).is_err());
    }
}
#[test]
fn help_and_schema_require_no_paths_keys_or_runtime() {
    assert_eq!(definition().try_get_matches_from(["job", "--help"]).unwrap_err().kind(), clap::error::ErrorKind::DisplayHelp);
    let matches = definition().try_get_matches_from(["job", "schema"]).unwrap();
    let mut out = Vec::new(); let mut err = Vec::new();
    assert_eq!(JobCommand::from_arg_matches(&matches).unwrap().run(&mut out, &mut err), ExitCode::SUCCESS);
    assert!(err.is_empty());
    let schema = canonjson::parse_str(std::str::from_utf8(&out).unwrap()).unwrap();
    assert_eq!(schema["can_execute_or_resume"], false);
    assert_eq!(schema["model_required"], false); assert_eq!(schema["original_inputs_required"], false);
    assert_eq!(schema["report_fields"].as_array().unwrap().len(), 12);
}
#[test]
fn job_id_has_one_explicit_canonical_representation() {
    assert_eq!(parse_job_id("00112233445566778899aabbccddeeff").unwrap(),
        JobId([0,17,34,51,68,85,102,119,136,153,170,187,204,221,238,255]));
    for bad in ["", "0011", "00112233445566778899AABBCCDDEEFF", "0x00112233445566778899aabbccddeeff",
        "00112233445566778899aabbccddeeff\n", "zz112233445566778899aabbccddeeff"] {
        assert!(parse_job_id(bad).is_err());
    }
}
#[test]
fn duplicate_unknown_and_oversized_limit_documents_are_rejected() {
    let limits = JobLimits { max_items: 16, max_id_bytes: 128, max_input_bytes_per_item: 65536,
        max_snapshot_bytes: 1 << 20, max_result_bytes: 16384, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 32 << 20,
        max_attempts: 16, max_work: JobWork::default() };
    let original = canonjson::canonical_bytes(&limits).unwrap();
    parse_limits(&original).unwrap();
    let mut duplicate = original.clone(); duplicate.pop(); duplicate.extend_from_slice(b",\"max_items\":16}");
    assert!(parse_limits(&duplicate).is_err());
    let mut unknown = original; unknown.pop(); unknown.extend_from_slice(b",\"private-key\":\"secret\"}");
    assert!(parse_limits(&unknown).is_err());
    assert!(parse_limits(&vec![b' '; LIMIT_FILE_BYTES + 1]).is_err());
    assert!(parse_limits(&[0xff]).is_err());
}
#[test]
fn report_is_one_canonical_line_with_no_retained_result_contents() {
    let r = report(); let mut bytes = Vec::new();
    emit(&mut bytes, &Success { schema_version: 1, operation: StoredJobOperation::Status, status: "ok", report: &r }).unwrap();
    assert_eq!(bytes.iter().filter(|&&c| c == b'\n').count(), 1);
    let value = canonjson::parse_str(std::str::from_utf8(&bytes).unwrap()).unwrap();
    assert_eq!(value["report"]["attempts"], 2);
    assert_eq!(value["report"]["materialized"], false);
    assert_eq!(value["report"]["verification_scope"], schema().report_scope);
}
#[test]
fn all_cancellation_causes_keep_budget_versus_cancel_classification() {
    use DecodeCancellationKind::*;
    for kind in [Timeout, Deadline, PollQuota, CostBudget] {
        assert_eq!(cancelled(kind).exit, ErrorCode::BudgetOrTimeout);
    }
    for kind in [User, FailFast, RaceLost, ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        assert_eq!(cancelled(kind).exit, ErrorCode::Cancelled);
    }
}
#[test]
fn integrity_orphans_and_incompleteness_have_distinct_machine_codes() {
    assert_eq!(job_failure(JobError::Corrupt).exit, ErrorCode::ArtifactIntegrityOrFormatOrVersion);
    assert_eq!(job_failure(JobError::UncommittedTail).code, "uncommitted_tail_or_stage");
    assert_eq!(job_failure(JobError::Incomplete).code, "job_incomplete");
    assert_eq!(job_failure(JobError::PublicationUncertain).code, "publication_uncertain");
    assert_eq!(job_failure(JobError::Busy).exit, ErrorCode::AdmissionOrResourceLimit);
}
#[test]
fn default_failure_response_contains_only_closed_metadata() {
    let mut bytes = Vec::new(); let f = job_failure(JobError::Mismatch(crate::jobs::MismatchField::Recipe));
    assert_eq!(report_error("verify", f, &mut bytes), ErrorCode::ArtifactIntegrityOrFormatOrVersion.as_process_exit());
    let value = canonjson::parse_str(std::str::from_utf8(&bytes).unwrap()).unwrap();
    assert_eq!(value["code"], "stored_contract_mismatch"); assert_eq!(value["exit_code"], 7);
    assert_eq!(value.as_object().unwrap().len(), 5);
}
#[test]
fn output_failure_is_terminal_without_appending_a_second_record() {
    struct Partial { writes: usize, bytes: Vec<u8> }
    impl Write for Partial {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.writes += 1;
            if self.writes > 1 { return Err(std::io::Error::other("sink failed")); }
            self.bytes.extend_from_slice(&bytes[..1]); Ok(1)
        }
        fn flush(&mut self) -> std::io::Result<()> { panic!("failed write must not report a flush") }
    }
    let mut out = Partial { writes: 0, bytes: Vec::new() };
    assert_eq!(emit(&mut out, &schema()).unwrap_err().code, "output_io");
    assert_eq!(out.bytes.len(), 1); assert_eq!(out.writes, 2);
}
