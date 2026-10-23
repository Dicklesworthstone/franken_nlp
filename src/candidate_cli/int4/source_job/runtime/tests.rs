//! Real dispatch and hosted contract fixtures; no model or durable IO is used.
use super::*;
use crate::{
    hosted::corpus::jobs::HostedJobError,
    jobs::{JobError, JobId, JobProgress, JobWork, runner::JobRunError},
    native_engine::decode::DecodeCancellationKind,
};

struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unexpected private population read"); }
}
struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("unexpected result delivery"); }
    fn flush(&mut self) -> io::Result<()> { panic!("unexpected result flush"); }
}
fn command(kind: Kind, retain: bool) -> Int4Command {
    let name = format!("{}-job", kind.name());
    let mut words = vec!["candidate", "source-int4", name.as_str(), "resume",
        "--source-dir", "private-source", "--source-manifest", "private-manifest",
        "--memory-mib", "65536", "--job-dir", "private-job", "--key-file", "private-key",
        "--limits", "private-limits", "--job-id", "0123456789abcdef0123456789abcdef"];
    if retain { words.push("--store-results"); }
    let matches = crate::candidate_cli::definition().try_get_matches_from(words).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong dispatcher"); };
    command
}
fn parts(kind: Kind) -> (JobCommand, Int4Args) {
    let Int4Command { task: Int4Task::SourceJob, args, batch: None, job: Some(Int4JobCommand::Source(job)) }
        = command(kind, true) else { panic!("source job lost its owned input route"); };
    (job, args)
}
fn lifetime(job: &JobArgs, args: &Int4Args, common: Int4Limits) -> JobLimits {
    let work = crate::native_engine::constrained_int4::planned_work(args.group_width,
        common.max_prompt_tokens, args.max_new_tokens).unwrap();
    JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: args.max_input_bytes,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: common.result_bytes,
        max_spool_bytes: 16 * MIB, max_materialized_bytes: 16 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 5,
        max_work: JobWork::from_int4(work, job.source.mask_visits() * 3) }
}
fn completed(id: JobId, limits: JobLimits) -> JobProgress {
    JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
        reserved_work: limits.max_work, spool_bytes: 1024, materialized: true }
}

#[test]
fn every_source_job_requires_owned_input_and_keeps_metadata_only_failure_reporting() {
    for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer] {
        assert_eq!(command(kind, true).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
        let mut diagnostics = Vec::new();
        assert_eq!(command(kind, false).run_owned(NoRead, NoWrite, &mut diagnostics), ErrorCode::Usage.as_process_exit());
        let value: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
        assert_eq!(value["operation"], "resume"); assert_eq!(value["task"], kind.name());
        assert_eq!(value["durable_progress_may_exist"], true);
        assert!(!std::str::from_utf8(&diagnostics).unwrap().contains("private-"));
    }
}

#[test]
fn owned_boundary_rechecks_admission_before_any_runtime_or_private_io() {
    let (mut command, args) = parts(Kind::Ner);
    let admitted = command.args.validate(&args).unwrap();
    command.args.storage.max_stream_mib = 0;
    assert_eq!(execute(command.mode, args, command.args, admitted, NoRead, &mut NoWrite).unwrap_err().code,
        "explicit_storage_or_job_options");
    let (command, mut args) = parts(Kind::Summarize);
    let admitted = command.args.validate(&args).unwrap();
    args.seed = Some("ab".repeat(32));
    assert_eq!(execute(command.mode, args, command.args, admitted, NoRead, &mut NoWrite).unwrap_err().code,
        "invalid_job_options");
}

#[test]
fn complete_population_quote_matches_host_for_every_source_task() {
    for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer] {
        let (command, args) = parts(kind);
        let common = command.args.validate(&args).unwrap();
        let job = lifetime(&command.args, &args, common);
        let limits = command.args.lifetime_limits(&args, common, job).unwrap();
        let host = host_limits(&args, limits, Duration::from_secs(19));
        check_host_quote(host, job, limits.buffer_bytes).unwrap();
        assert_eq!(check_host_quote(host, job, limits.buffer_bytes - 1).unwrap_err().code, "job_host_memory_quote");
        assert_eq!(host.native.context_tokens, args.context_tokens);
        assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
        assert_eq!(host.native.run.max_elapsed, Duration::from_secs(19));
        let mut invalid = host; invalid.serialization_reserve_bytes = 0;
        assert_eq!(check_host_quote(invalid, job, limits.buffer_bytes).unwrap_err().code, "job_host_admission");
    }
}

#[test]
fn all_tasks_publish_only_authenticated_progress_and_public_model_provenance() {
    for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer] {
        let (command, args) = parts(kind);
        let common = command.args.validate(&args).unwrap();
        let limits = lifetime(&command.args, &args, common);
        let id = JobId([0x12; 16]);
        let report = progress_report(completed(id, limits), id, limits, true,
            RunMode::Resume { discard_uncommitted: true }, kind.name()).unwrap();
        let content = "ab".repeat(32); let source = "cd".repeat(32);
        let response = Response {
            protocol: "fnlp-candidate-int4-source-job-v1", schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
            task: kind.name(), model_id: "Nanbeige4.2-3B", source_revision: "fixture-public-revision",
            source_root_sha256: &source, model_content_sha256: &content,
            candidate_format: "fixture-candidate-format", quant_recipe: "fixture-grouped-int4",
            group_width: args.group_width, integer_kernel: common.kernel.kind().id(),
            snapshot: Some(SnapshotRecord::new("loaded", common.snapshot_need)), output: &report,
        };
        let mut bytes = Vec::new(); publish(&response, job_cli::REPORT_BYTES, &mut bytes).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["protocol"], "fnlp-candidate-int4-source-job-v1");
        assert_eq!(value["task"], kind.name()); assert_eq!(value["output"]["task"], kind.name());
        assert_eq!(value["model_content_sha256"], content);
        assert_eq!(value["group_width"], args.group_width);
        assert_eq!(value["snapshot"]["format"], INT4_SNAPSHOT_VERSION);
        assert_eq!(value["output"]["attempts"], 3); assert_eq!(value["output"]["committed"], 2);
        assert_eq!(value["output"]["reserved_work"], serde_json::to_value(limits.max_work).unwrap());
        for private in ["text", "passages", "options", "key", "job_dir", "defaults", "results", "documents"] {
            assert!(value.get(private).is_none()); assert!(value["output"].get(private).is_none());
        }
        assert!(bytes.ends_with(b"\n"));
        assert_eq!(publish(&response, 1, &mut NoWrite), Err(CandidateError::Output));
    }
}

#[test]
fn partial_progress_and_excess_group_work_cannot_be_reported_as_complete() {
    let (command, args) = parts(Kind::Answer);
    let common = command.args.validate(&args).unwrap();
    let limits = lifetime(&command.args, &args, common);
    let id = JobId([0x12; 16]);
    for case in 0..5 {
        let mut progress = completed(id, limits);
        match case {
            0 => progress.committed -= 1,
            1 => progress.job_id = JobId([0x13; 16]),
            2 => progress.reserved_work.model.projections.dot_products += 1,
            3 => progress.reserved_work.mask_node_visits += 1,
            _ => progress.materialized = false,
        }
        let result = progress_report(progress, id, limits, true, RunMode::Start, "answer");
        assert!(matches!(result, Err(Failure { code: "job_completion_contract", .. })));
    }
}

#[test]
fn typed_cancellation_integrity_and_lifetime_failures_retain_existing_policy_boundary() {
    for (cause, exit, code) in [
        (DecodeCancellationKind::User, ErrorCode::Cancelled, "job_cancelled"),
        (DecodeCancellationKind::Deadline, ErrorCode::BudgetOrTimeout, "job_execution_budget"),
        (DecodeCancellationKind::CostBudget, ErrorCode::BudgetOrTimeout, "job_execution_budget"),
    ] {
        let failure = host_failure(HostedJobError::Job(JobRunError::Storage(JobError::Cancelled(cause))));
        assert_eq!(failure.exit, exit); assert_eq!(failure.code, code);
    }
    for (error, exit, code) in [
        (JobError::WorkLimit, ErrorCode::BudgetOrTimeout, "job_lifetime_work_or_attempts"),
        (JobError::Authentication, ErrorCode::ArtifactIntegrityOrFormatOrVersion, "job_integrity"),
        (JobError::UncommittedTail, ErrorCode::StructuredTaskNoResult, "job_uncommitted_tail_or_stage"),
    ] {
        let failure = host_failure(HostedJobError::Job(JobRunError::Storage(error)));
        assert_eq!(failure.exit, exit); assert_eq!(failure.code, code);
    }
}

#[test]
fn final_report_delivery_failure_never_changes_the_durable_completion_record() {
    struct Broken;
    impl Write for Broken {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { Err(io::Error::other("closed consumer")) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("closed consumer")) }
    }
    let (command, args) = parts(Kind::Keyphrases);
    let common = command.args.validate(&args).unwrap();
    let limits = lifetime(&command.args, &args, common);
    let id = JobId([0x12; 16]);
    let report = progress_report(completed(id, limits), id, limits, false, RunMode::Start, "keyphrases").unwrap();
    let before = serde_json::to_value(&report).unwrap();
    let failure = Failure::from(publish(&report, job_cli::REPORT_BYTES, &mut Broken).unwrap_err());
    assert_eq!(failure.code, "report_delivery");
    assert_eq!(serde_json::to_value(&report).unwrap(), before);
    assert_eq!(before["committed"], 2); assert_eq!(before["status"], "complete");
}
