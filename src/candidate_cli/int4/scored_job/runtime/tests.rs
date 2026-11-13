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
    check_host_quote(host, fixed, limits.buffer_bytes, JobInputMemory::CompletePopulation).unwrap();
    assert_eq!(check_host_quote(host, fixed, limits.buffer_bytes - 1, JobInputMemory::CompletePopulation)
        .unwrap_err().code, "job_host_memory_quote");
    let mut invalid = host; invalid.journal_reserve_bytes = 0;
    assert_eq!(check_host_quote(invalid, fixed, limits.buffer_bytes, JobInputMemory::CompletePopulation)
        .unwrap_err().code, "job_host_admission");
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
            snapshot: Some(SnapshotRecord::new("loaded", common.snapshot_need)), input_encoding: None,
            compute_schedule: None, output: &report,
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
        assert!(value.get("input_encoding").is_none());
        let encoding = crate::tokenizer::untrusted::UntrustedEncoding::SafeBpeV1(
            crate::tokenizer::untrusted::SafeBpeLimits {
                max_input_bytes: 8192, max_output_tokens: 4096,
                max_pair_checks: 123456, max_pair_bytes: 2345678,
            });
        let mut compact = response;
        compact.input_encoding = Some(&encoding);
        let mut compact_value = serde_json::to_value(&compact).unwrap();
        assert_eq!(compact_value.as_object_mut().unwrap().remove("input_encoding"),
            Some(serde_json::to_value(&encoding).unwrap()));
        assert_eq!(compact_value, value);
        assert!(value.get("compute_schedule").is_none());
        compact.compute_schedule = compute::Schedule::for_threads(4);
        let mut threaded = serde_json::to_value(&compact).unwrap();
        assert_eq!(threaded.as_object_mut().unwrap().remove("compute_schedule"),
            compute::Schedule::for_threads(4).map(serde_json::to_value).transpose().unwrap());
        threaded.as_object_mut().unwrap().remove("input_encoding");
        assert_eq!(threaded, value);
    }
}


#[test]
fn immutable_compiler_policy_and_workspace_must_match_the_cli_before_snapshot_or_job_io() {
    use crate::native_engine::strict_int4::weights::projection_fixture;
    use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};
    let (c, args) = parts(); let common = c.args.validate(&args).unwrap();
    let fixed = lifetime(&c.args, &args, common);
    let limits = c.args.lifetime_limits(&args, common, fixed).unwrap();
    let (model, _) = projection_fixture(args.group_width);
    let policy = UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 8192, max_output_tokens: 4096,
        max_pair_checks: 4 * 1024 * 1024, max_pair_bytes: 64 * 1024 * 1024,
    });
    let planner = Int4ClassificationPlanner::for_model(model.identity(), c.args.scoring.budget(common),
        c.args.scoring.classification_limits(&args)).unwrap().with_document_encoding(policy.clone()).unwrap();
    let planner = Int4ScoredJobPlanner::new_with_document_encoding(planner, None, limits.native).unwrap();
    let workspace = planner.encoding_workspace_bytes();
    check_planner_quote(&planner, &policy, workspace).unwrap();
    assert_eq!(check_planner_quote(&planner, &UntrustedEncoding::ByteFallback, workspace), Err(CandidateError::Planning));
    assert_eq!(check_planner_quote(&planner, &policy, workspace - 1), Err(CandidateError::Planning));
    assert_eq!(check_planner_quote(&planner, &policy, workspace + 1), Err(CandidateError::Planning));
    let UntrustedEncoding::SafeBpeV1(mut changed) = policy else { unreachable!() };
    changed.max_pair_checks += 1;
    assert_eq!(check_planner_quote(&planner, &UntrustedEncoding::SafeBpeV1(changed), workspace),
        Err(CandidateError::Planning));
    planner.check_job_limits(fixed).unwrap();
}


#[test]
fn original_native_cancellation_causes_keep_the_durable_policy_boundary() {
    use crate::native_engine::decode::DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = job_failure(JobError::Cancelled(cause));
        let actual = execution_failure(HostedJobError::Host(
            HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause)))));
        assert_eq!(actual.exit, expected.exit);
        assert_eq!(actual.code, expected.code);
    }
}

#[test]
fn scored_metadata_must_still_fit_and_have_time_before_and_after_delivery() {
    let available = || Ok(Duration::from_secs(1));
    let oversized = "x".repeat(job_cli::REPORT_BYTES);
    assert!(publish_report(&oversized, &mut NoIo, &available).is_err());
    let checks = std::cell::Cell::new(0);
    let expired_before_write = || {
        checks.set(checks.get() + 1);
        if checks.get() == 1 { available() } else { Err(CandidateError::Timeout) }
    };
    assert_eq!(publish_report(&true, &mut NoIo, &expired_before_write).unwrap_err().exit,
        ErrorCode::BudgetOrTimeout);
    let checks = std::cell::Cell::new(0);
    let expired_after_flush = || {
        checks.set(checks.get() + 1);
        if checks.get() < 3 { available() } else { Err(CandidateError::Timeout) }
    };
    let mut delivered = Vec::new();
    assert_eq!(publish_report(&true, &mut delivered, &expired_after_flush).unwrap_err().exit,
        ErrorCode::BudgetOrTimeout);
    assert_eq!(delivered, b"true\n");
    struct BrokenFlush(Vec<u8>);
    impl Write for BrokenFlush {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.0.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("fixture flush")) }
    }
    let mut broken = BrokenFlush(Vec::new());
    assert!(publish_report(&true, &mut broken, &available).is_err());
    assert_eq!(broken.0, b"true\n");
}
