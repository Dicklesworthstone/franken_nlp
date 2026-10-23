//! End-to-end dispatch and pure runtime-contract checks without model IO.
use super::*;
use crate::jobs::{JobId, JobProgress, JobWork};

struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unexpected private input access"); }
}
struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("unexpected report delivery"); }
    fn flush(&mut self) -> io::Result<()> { panic!("unexpected report flush"); }
}
fn command(retain: bool) -> Int4Command {
    let mut argv = vec!["candidate", "source-int4", "extract-job", "resume",
        "--source-dir", "private-source", "--source-manifest", "private-manifest.json",
        "--memory-mib", "65536", "--schema", "private-schema.json",
        "--job-dir", "private-job", "--key-file", "private-key", "--limits", "private-limits.json",
        "--job-id", "0123456789abcdef0123456789abcdef"];
    if retain { argv.push("--store-results"); }
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong dispatcher"); };
    command
}
fn parts() -> (JobCommand, Int4Args) {
    let Int4Command { task: Int4Task::ExtractJob, args, batch: None, job: Some(Int4JobCommand::Extraction(job)) } = command(true)
        else { panic!("job route lost its owned command"); };
    (job, args)
}
fn lifetime(job: &JobArgs, args: &Int4Args, common: Int4Limits) -> JobLimits {
    let work = crate::native_engine::constrained_int4::planned_work(args.group_width,
        common.max_prompt_tokens, args.max_new_tokens).unwrap();
    JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: args.max_input_bytes,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: common.result_bytes,
        max_spool_bytes: 16 * MIB, max_materialized_bytes: 16 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 5,
        max_work: JobWork::from_int4(work, job.extraction.mask_visits() * 3) }
}

#[test]
fn root_dispatch_requires_owned_job_input_and_preserves_metadata_failure_contract() {
    assert_eq!(command(true).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    let mut diagnostics = Vec::new();
    let result = command(false).run_owned(NoRead, NoWrite, &mut diagnostics);
    assert_eq!(result, ErrorCode::Usage.as_process_exit());
    let report: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
    assert_eq!(report["operation"], "resume");
    assert_eq!(report["code"], "explicit_storage_or_job_options");
    assert_eq!(report["durable_progress_may_exist"], true);
    assert!(!std::str::from_utf8(&diagnostics).unwrap().contains("private-"));
}

#[test]
fn runtime_revalidates_the_owned_request_before_process_or_file_access() {
    let (mut command, args) = parts();
    let admitted = command.args.validate(&args).unwrap();
    command.args.storage.max_stream_mib = 0;
    let error = execute(command.mode, args, command.args, admitted, NoRead, &mut NoWrite).unwrap_err();
    assert_eq!(error.code, "explicit_storage_or_job_options");
}

#[test]
fn cli_memory_quote_agrees_with_the_host_and_cannot_replace_host_validation() {
    let (command, args) = parts();
    let common = command.args.validate(&args).unwrap();
    let job = lifetime(&command.args, &args, common);
    let limits = command.args.lifetime_limits(&args, common, job).unwrap();
    let host = host_limits(&args, limits, Duration::from_secs(30));
    check_host_quote(host, job, limits.buffer_bytes).unwrap();
    assert_eq!(check_host_quote(host, job, limits.buffer_bytes - 1).unwrap_err().code, "job_host_memory_quote");
    let mut invalid = host; invalid.journal_reserve_bytes = 0;
    assert_eq!(check_host_quote(invalid, job, limits.buffer_bytes).unwrap_err().code, "job_host_admission");
    assert_eq!(host.native.context_tokens, args.context_tokens);
    assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
    assert_eq!(host.native.run.max_elapsed, Duration::from_secs(30));
}

#[test]
fn completion_envelope_contains_candidate_identity_and_authenticated_progress_only() {
    let (command, args) = parts();
    let common = command.args.validate(&args).unwrap();
    let limits = lifetime(&command.args, &args, common);
    let id = JobId([0x12; 16]);
    let progress = JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
        reserved_work: limits.max_work, spool_bytes: 1024, materialized: true };
    let report = progress_report(progress, id, limits, true,
        RunMode::Resume { discard_uncommitted: true }, "extract").unwrap();
    let content = "ab".repeat(32);
    let source = "cd".repeat(32);
    let response = Response {
        protocol: "fnlp-candidate-int4-extract-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture-public-revision",
        source_root_sha256: &source, model_content_sha256: &content,
        candidate_format: "fixture-candidate-format", quant_recipe: "fixture-grouped-int4",
        group_width: args.group_width, integer_kernel: common.kernel.kind().id(),
        snapshot: Some(SnapshotRecord::new("loaded", common.snapshot_need)), output: &report,
    };
    let mut bytes = Vec::new();
    publish(&response, job_cli::REPORT_BYTES, &mut bytes).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["protocol"], "fnlp-candidate-int4-extract-job-v1");
    assert_eq!(value["model_content_sha256"], content);
    assert_eq!(value["group_width"], args.group_width);
    assert_eq!(value["snapshot"]["format"], INT4_SNAPSHOT_VERSION);
    assert_eq!(value["output"]["operation"], "resume");
    assert_eq!(value["output"]["attempts"], 3);
    assert_eq!(value["output"]["committed"], 2);
    assert_eq!(value["output"]["materialized"], true);
    assert_eq!(value["output"]["reserved_work"], serde_json::to_value(limits.max_work).unwrap());
    for forbidden in ["text", "schema", "key", "job_dir", "results", "documents"] {
        assert!(value.get(forbidden).is_none());
        assert!(value["output"].get(forbidden).is_none());
    }
    assert!(bytes.ends_with(b"\n"));
    assert_eq!(publish(&response, 1, &mut NoWrite), Err(CandidateError::Output));
}
