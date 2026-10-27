//! Owned dispatch, runtime memory quote and metadata publication fixtures.
use super::*;
use crate::jobs::{JobId, JobProgress, JobWork};

struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unexpected private input access"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("unexpected metadata publication"); }
    fn flush(&mut self) -> io::Result<()> { panic!("unexpected metadata flush"); }
}
fn command(task: &str, retain: bool) -> Int4Command {
    let mut words = vec!["candidate", "source-int4", "score-job", "resume", "--task", task,
        "--source-dir", "private-source", "--source-manifest", "private-manifest.json",
        "--memory-mib", "65536", "--defaults", "private-defaults.json", "--job-dir", "private-job",
        "--key-file", "private-key", "--limits", "private-limits.json",
        "--job-id", "0123456789abcdef0123456789abcdef"];
    if retain { words.push("--store-results"); }
    let matches = crate::candidate_cli::definition().try_get_matches_from(words).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong INT4 dispatcher"); };
    command
}
fn parts() -> (JobCommand, Int4Args) {
    let Int4Command { task: Int4Task::ScoredJob, args, batch: None, job: Some(Int4JobCommand::Scored(job)) } =
        command("classify", true) else { panic!("lost owned scoring job"); };
    (job, args)
}
fn lifetime(job: &JobArgs, args: &Int4Args, common: Int4Limits) -> JobLimits {
    JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: args.max_input_bytes,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: common.result_bytes,
        max_spool_bytes: 16 * MIB, max_materialized_bytes: 16 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 5,
        max_work: JobWork::from_int4(job.scoring.work_ceiling(), 0) }
}

#[test]
fn root_dispatch_requires_owned_input_for_every_fixed_scoring_task() {
    for task in ["classify", "sentiment", "judge"] {
        assert_eq!(command(task, true).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
        let mut diagnostics = Vec::new();
        assert_eq!(command(task, false).run_owned(NoIo, NoIo, &mut diagnostics), ErrorCode::Usage.as_process_exit());
        let value: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
        assert_eq!(value["operation"], "resume"); assert_eq!(value["code"], "explicit_scored_job_storage");
        assert_eq!(value["durable_progress_may_exist"], true);
        assert!(!std::str::from_utf8(&diagnostics).unwrap().contains("private-"));
    }
}
#[test]
fn runtime_revalidates_the_owned_request_before_runtime_or_private_file_access() {
    let (mut c, args) = parts(); let common = c.args.validate(&args).unwrap();
    c.args.storage.max_stream_mib = 0;
    assert_eq!(execute(c.mode, args, c.args, common, NoIo, &mut NoIo).unwrap_err().code,
        "explicit_scored_job_storage");
}
#[test]
fn cli_quote_matches_the_host_population_reservation_and_preserves_the_invocation_budget() {
    let (c, args) = parts(); let common = c.args.validate(&args).unwrap();
    let fixed = lifetime(&c.args, &args, common); let limits = c.args.lifetime_limits(&args, common, fixed).unwrap();
    let host = host_limits(&args, limits, Duration::from_secs(30));
    check_host_quote(host, fixed, limits.buffer_bytes).unwrap();
    assert_eq!(check_host_quote(host, fixed, limits.buffer_bytes - 1).unwrap_err().code, "job_host_memory_quote");
    let mut invalid = host; invalid.journal_reserve_bytes = 0;
    assert_eq!(check_host_quote(invalid, fixed, limits.buffer_bytes).unwrap_err().code, "job_host_admission");
    assert_eq!(host.native.context_tokens, args.context_tokens);
    assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
    assert_eq!(host.native.run.max_elapsed, Duration::from_secs(30));
    assert_eq!(host.transport.max_lines, c.args.storage.max_input_lines);
}
#[test]
fn completion_metadata_is_task_and_candidate_bound_without_private_scores_or_recipe() {
    let (c, args) = parts(); let common = c.args.validate(&args).unwrap();
    let fixed = lifetime(&c.args, &args, common); let id = JobId([0x12; 16]);
    for task in ["classify", "sentiment", "judge"] {
        let progress = JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
            reserved_work: fixed.max_work, spool_bytes: 1024, materialized: true };
        let report = progress_report(progress, id, fixed, true,
            RunMode::Resume { discard_uncommitted: true }, task).unwrap();
        let content = "ab".repeat(32); let source = "cd".repeat(32);
        let response = Response {
            protocol: "fnlp-candidate-int4-score-job-v1", schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task,
            model_id: "Nanbeige4.2-3B", source_revision: "fixture-public-revision",
            source_root_sha256: &source, model_content_sha256: &content,
            candidate_format: "fixture-candidate-format", quant_recipe: "fixture-grouped-int4",
            group_width: args.group_width, integer_kernel: common.kernel.kind().id(),
            snapshot: Some(SnapshotRecord::new("loaded", common.snapshot_need)), output: &report,
        };
        let mut bytes = Vec::new(); publish(&response, job_cli::REPORT_BYTES, &mut bytes).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["protocol"], "fnlp-candidate-int4-score-job-v1"); assert_eq!(value["task"], task);
        assert_eq!(value["model_content_sha256"], content); assert_eq!(value["group_width"], args.group_width);
        assert_eq!(value["snapshot"]["format"], INT4_SNAPSHOT_VERSION);
        assert_eq!(value["output"]["operation"], "resume"); assert_eq!(value["output"]["attempts"], 3);
        assert_eq!(value["output"]["committed"], 2); assert_eq!(value["output"]["materialized"], true);
        assert_eq!(value["output"]["reserved_work"], serde_json::to_value(fixed.max_work).unwrap());
        for forbidden in ["text", "labels", "scores", "policy", "recipe", "key", "job_dir", "results", "documents"] {
            assert!(value.get(forbidden).is_none()); assert!(value["output"].get(forbidden).is_none());
        }
        assert!(bytes.ends_with(b"\n"));
        assert_eq!(publish(&response, 1, &mut NoIo), Err(CandidateError::Output));
    }
}
