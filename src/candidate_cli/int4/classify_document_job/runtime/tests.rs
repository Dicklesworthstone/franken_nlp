//! Host quotation, typed cancellation and metadata-publication boundaries.
use super::*;
use super::super::tests::{command, lifetime, NoIo};
use crate::{jobs::{JobId, JobProgress}, native_engine::decode::DecodeCancellationKind};

#[test]
fn host_population_quote_matches_the_admitted_complete_document_memory() {
    for flags in [vec![], vec!["--cohort-chunks", "3"],
        vec!["--input-encoding", "safe-bpe-v1"],
        vec!["--input-encoding", "safe-bpe-v1", "--cohort-chunks", "3"]] {
        let (c, args) = command("resume", &flags); let map = c.args.validate(&args).unwrap();
        let fixed = lifetime(&args, map); let limits = c.args.lifetime_limits(&args, map, fixed).unwrap();
        let host = host_limits(&args, limits, Duration::from_secs(30));
        assert_eq!(host.required_buffer_bytes(fixed).unwrap(), limits.buffer_bytes);
        assert_eq!(host.native.context_tokens, args.context_tokens);
        assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
        assert_eq!(host.native.run.max_elapsed, Duration::from_secs(30));
        assert_eq!(host.preparation_reserve_bytes, map.common.preparation_bytes);
        assert_eq!(limits.map.common.minimum_memory_bytes, map.common.minimum_memory_bytes + limits.buffer_bytes);
    }
}

#[test]
fn runtime_rechecks_programmatic_storage_refusal_before_any_private_io() {
    let (mut c, args) = command("resume", &[]); let map = c.args.validate(&args).unwrap();
    c.args.storage.store_results = false;
    assert_eq!(execute(c.mode, args, c.args, map, NoIo, &mut NoIo).unwrap_err().code,
        "explicit_classification_document_job_storage");
}

#[test]
fn all_typed_load_and_snapshot_cancellation_causes_survive_nested_physical_errors() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost, ParentCancelled,
        ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = job_failure(JobError::Cancelled(cause));
        for error in [HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))),
            HostedError::Scope { source: asupersync::cx::ScopedCpuError::WorkerCapExceeded { cap: 1 },
                physical_error: Some(Box::new(HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause))))) }] {
            let actual = model_failure(error); assert_eq!(actual.exit, expected.exit); assert_eq!(actual.code, expected.code);
        }
    }
}

#[test]
fn completion_reports_only_public_model_and_durable_progress_metadata() {
    let (c, args) = command("resume", &["--cohort-chunks", "3"]); let map = c.args.validate(&args).unwrap();
    let fixed = lifetime(&args, map); let id = JobId([0x12; 16]);
    let progress = JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
        reserved_work: fixed.max_work, spool_bytes: 1024, materialized: true };
    let report = progress_report(progress, id, fixed, true, c.mode, "classify").unwrap();
    let response = Response { protocol: "fnlp-candidate-int4-classify-document-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "classify",
        durable_unit: "complete_original_document", model_id: "Nanbeige4.2-3B",
        source_revision: "public-revision", source_root_sha256: "public-source", model_content_sha256: "public-model",
        candidate_format: "candidate-int4", quant_recipe: "strict-int4", group_width: args.group_width,
        integer_kernel: map.common.kernel.kind().id(), snapshot: None, cohort_chunks: Some(3),
        input_encoding: None, output: &report };
    let mut bytes = Vec::new(); publish_report(&response, &mut bytes, &available).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["durable_unit"], "complete_original_document"); assert_eq!(value["cohort_chunks"], 3);
    assert_eq!(value["output"]["committed"], 2); assert_eq!(value["output"]["attempts"], 3);
    assert_eq!(value["output"]["reserved_work"], serde_json::to_value(fixed.max_work).unwrap());
    for forbidden in ["text", "labels", "scores", "policy", "recipe", "key", "job_dir", "documents", "aggregate"] {
        assert!(value.get(forbidden).is_none()); assert!(value["output"].get(forbidden).is_none());
    }
    assert!(bytes.ends_with(b"\n"));
}

fn available() -> Result<Duration, CandidateError> { Ok(Duration::from_secs(1)) }
#[test]
fn oversize_or_expired_metadata_never_reaches_the_writer() {
    let oversized = "x".repeat(job_cli::REPORT_BYTES);
    assert_eq!(publish_report(&oversized, &mut NoIo, &available).unwrap_err().exit, ErrorCode::Generic);
    let checks = std::cell::Cell::new(0);
    let deadline = || {
        checks.set(checks.get() + 1);
        if checks.get() == 1 { available() } else { Err(CandidateError::Timeout) }
    };
    assert_eq!(publish_report(&serde_json::json!({"status":"complete"}), &mut NoIo, &deadline)
        .unwrap_err().exit, ErrorCode::BudgetOrTimeout);
    assert_eq!(checks.get(), 2, "serialization cannot renew the original publication deadline");
}

#[test]
fn flush_failure_or_a_deadline_during_delivery_never_reports_success() {
    struct FlushFails { bytes: Vec<u8> }
    impl Write for FlushFails {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.bytes.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("fixture flush")) }
    }
    let mut writer = FlushFails { bytes: Vec::new() };
    assert!(publish_report(&true, &mut writer, &available).is_err()); assert!(!writer.bytes.is_empty());
    let checks = std::cell::Cell::new(0);
    let deadline = || { checks.set(checks.get() + 1);
        if checks.get() < 3 { available() } else { Err(CandidateError::Timeout) } };
    let mut bytes = Vec::new();
    assert_eq!(publish_report(&true, &mut bytes, &deadline).unwrap_err().exit, ErrorCode::BudgetOrTimeout);
    assert_eq!(bytes, b"true\n"); assert_eq!(checks.get(), 3);
}
