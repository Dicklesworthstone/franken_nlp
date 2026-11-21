//! Pure ownership, cancellation and completion-publication contracts.
//! Constructed metadata is not native inference or durable execution evidence.
use super::*;
use super::super::tests::{command, options, lifetime, NoRead, NoWrite};
use crate::{
    batch::BatchFault,
    hosted::{Int4HostError, RunStop},
    jobs::{JobId, JobProgress},
    native_engine::strict_int4::{Int4ModelError, StrictInt4Error,
        weights::snapshot::Int4SnapshotError},
    tasks::extract::semantic::grouped_int4::document::Int4SemanticObjectDocumentError,
};

#[test]
fn runtime_rechecks_private_io_admission_before_initializing_the_process() {
    let (mut command, args) = command("start", false, &[]);
    let admitted = command.args.validate(&args).unwrap();
    command.args.storage.max_stream_mib = 0;
    let failure = execute(command.mode, args, command.args, admitted, NoRead, &mut NoWrite).unwrap_err();
    assert_eq!(failure.code, "semantic_object_job_storage_options");
    assert_eq!(failure.exit, ErrorCode::Usage);
}

#[test]
fn caller_and_owned_job_peaks_cover_source_snapshot_team_and_retained_replay() {
    for snapshot in [false, true] { for compact in [false, true] { for team in [false, true] {
        let mut flags = Vec::new();
        if compact { flags.extend(["--input-encoding", "safe-bpe-v1"]); }
        if team { flags.extend(["--compute-threads", "4"]); }
        let (command, args) = command("resume", snapshot, &flags);
        let options = options();
        let initial = command.args.validate(&args).unwrap();
        let job = lifetime(&command.args, &args, &options);
        let limits = command.args.lifetime_limits(&args, initial, &options, job).unwrap();
        let caller = caller_preparation_bytes(&command.args, initial.map.common).unwrap();
        assert!(caller > document_runtime::caller_preparation_bytes(
            &command.args.semantic, initial.map.common).unwrap());
        let time = Duration::from_secs(19);
        let complete = requirements(&args, &command.args, limits, &options, job, time).unwrap();
        let host = host_limits(&args, &command.args, limits, &options, time);
        assert_eq!(host.document.extraction.mapping.mapping.chunks.max_input_bytes, args.max_input_bytes);
        assert_eq!(host.document.review.chunks.max_input_bytes, args.max_input_bytes + options.limits.max_claim_bytes);
        assert_eq!(host.document.native.run.max_elapsed, time);
        assert_eq!(host.document.native.run.max_checkpoints, args.max_checkpoints);
        assert_eq!(complete.native, initial.map.common.native);
        assert_eq!(complete.team_workspace_bytes, initial.map.common.compute.workspace_bytes);
        assert!(complete.input_bytes > complete.preparation_bytes);
        let common = limits.object.map.common;
        let loading = if snapshot { common.snapshot_need.scratch_bytes }
            else { common.source_preparation_bytes + common.load.scratch_payload_bytes };
        let execution = complete.total_bytes().unwrap();
        let admitted = admit_memory(&args, limits, complete, caller).unwrap();
        assert_eq!(admitted.object.map.common.minimum_memory_bytes,
            caller + common.load.weight_payload_bytes + ALLOCATOR_BYTES
                + loading.max(execution) + CLEANUP_BYTES);
        if !snapshot {
            let mut saving_args = args.clone();
            saving_args.save_snapshot = Some(PathBuf::from("private-new.snapshot"));
            let saving = admit_memory(&saving_args, limits, complete, caller).unwrap();
            assert_eq!(saving.object.map.common.minimum_memory_bytes,
                caller + common.load.weight_payload_bytes + ALLOCATOR_BYTES
                    + loading.max(common.snapshot_need.scratch_bytes).max(execution) + CLEANUP_BYTES);
        }
        let mut exact = limits;
        exact.object.map.common.memory_bytes = admitted.object.map.common.minimum_memory_bytes;
        assert!(admit_memory(&args, exact, complete, caller).is_ok());
        exact.object.map.common.memory_bytes -= 1;
        assert_eq!(admit_memory(&args, exact, complete, caller).unwrap_err().exit, ErrorCode::AdmissionOrResourceLimit);
        for mismatch in 0..2 {
            let mut wrong = complete;
            if mismatch == 0 { wrong.native.kv_bytes += 1; } else { wrong.team_workspace_bytes += 1; }
            assert_eq!(admit_memory(&args, limits, wrong, caller).unwrap_err().exit,
                ErrorCode::ArtifactIntegrityOrFormatOrVersion);
        }
        flags.extend(["--use-retained-input", "--max-retained-input-mib", "8"]);
        let (retained, retained_args) = self::command("resume", snapshot, &flags);
        let retained_limits = retained.args.lifetime_limits(&retained_args,
            retained.args.validate(&retained_args).unwrap(), &options, job).unwrap();
        let replay = requirements(&retained_args, &retained.args, retained_limits,
            &options, job, time).unwrap();
        assert_eq!(complete.input_bytes - replay.input_bytes, 2 * job.max_snapshot_bytes);
        assert_eq!(complete.total_bytes().unwrap() - replay.total_bytes().unwrap(), 2 * job.max_snapshot_bytes);
        assert_eq!(complete.preparation_bytes, replay.preparation_bytes);
        assert_eq!(complete.native, replay.native);
        assert_eq!(complete.workspace_bytes, replay.workspace_bytes);
        assert_eq!(complete.output_reservation_bytes, replay.output_reservation_bytes);
        assert_eq!(complete.team_workspace_bytes, replay.team_workspace_bytes);
        assert_eq!(retained_limits.native.max_model_work, limits.native.max_model_work);
        assert_eq!(retained_limits.native.max_verification, limits.native.max_verification);
    } } }
}

#[test]
fn durable_and_native_wrapper_failures_preserve_every_original_cancellation_cause() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = job_failure(JobError::Cancelled(cause)).exit;
        for error in [
            HostedJobError::Job(JobRunError::Storage(JobError::Cancelled(cause))),
            HostedJobError::Job(JobRunError::Processor(BatchFault::cancelled(cause))),
            HostedJobError::Host(HostedError::Stopped { stop: RunStop { kind: cause,
                runtime_reason: None, runtime_checkpoint_error: None }, task_error: None }),
            HostedJobError::Host(HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause)))),
            HostedJobError::Host(HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause)))),
            HostedJobError::Host(HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause)))),
            HostedJobError::Host(HostedError::Int4(Int4HostError::SemanticObjectDocument(
                Int4SemanticObjectDocumentError::Cancelled(cause)))),
            HostedJobError::Host(HostedError::Scope {
                source: asupersync::cx::ScopedCpuError::WorkerCapExceeded { cap: 1 },
                physical_error: Some(Box::new(HostedError::Scope {
                    source: asupersync::cx::ScopedCpuError::WorkerCapExceeded { cap: 1 },
                    physical_error: Some(Box::new(HostedError::Int4(Int4HostError::SemanticObjectDocument(
                        Int4SemanticObjectDocumentError::Cancelled(cause))))),
                })),
            }),
        ] {
            let failure = execution_failure(error);
            assert_eq!(failure.cancellation, Some(cause));
            assert_eq!(failure.exit, expected);
        }
        let token = CancellationToken::default(); token.cancel(cause);
        let failure = publish_report(&true, &mut NoWrite,
            &|| Err(document_runtime::cancelled(Deadline)), &token).unwrap_err();
        assert_eq!(failure.cancellation, Some(cause));
    }
    for cause in [FailFast, LinkedExit] {
        let failure = host_error(HostedError::Stopped {
            stop: RunStop { kind: cause, runtime_reason: None, runtime_checkpoint_error: None },
            task_error: Some(Box::new(HostedError::Int4(Int4HostError::SemanticObjectDocument(
                Int4SemanticObjectDocumentError::WorkBudget)))),
        });
        assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
        assert_eq!(failure.cancellation, Some(cause));
    }
}

#[test]
fn completion_is_bounded_metadata_and_never_contains_private_document_results() {
    let (command, args) = command("resume", true, &[]);
    let options = options();
    let initial = command.args.validate(&args).unwrap().map.common;
    let limits = lifetime(&command.args, &args, &options);
    let id = JobId([0x12; 16]);
    let progress = JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
        reserved_work: limits.max_work, spool_bytes: 1024, materialized: true };
    let report = progress_report(progress, id, limits, true,
        RunMode::Resume { discard_uncommitted: true }, "extract-semantic-document").unwrap();
    let content = "ab".repeat(32); let source = "cd".repeat(32);
    let response = Response {
        protocol: "fnlp-candidate-int4-semantic-object-document-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        execution: INT4_SEMANTIC_OBJECT_DOCUMENT_EXECUTION, semantic_scope: SEMANTIC_OBJECT_DOCUMENT_SCOPE,
        same_model_correlated: true, calibration: "experimental_uncalibrated_diagnostic_only",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture-public-revision",
        source_root_sha256: &source, model_content_sha256: &content, candidate_format: INT4_SOURCE_FORMAT,
        quant_recipe: "fixture-grouped-int4", group_width: args.group_width,
        integer_kernel: initial.kernel.kind().id(), snapshot: Some(SnapshotRecord::new("loaded", initial.snapshot_need)),
        compute_schedule: None, input_encoding: None, output: &report,
    };
    let mut bytes = Vec::new();
    publish_report(&response, &mut bytes, &|| Ok(Duration::from_secs(1)), &CancellationToken::default()).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["execution"], INT4_SEMANTIC_OBJECT_DOCUMENT_EXECUTION);
    assert_eq!(value["semantic_scope"], SEMANTIC_OBJECT_DOCUMENT_SCOPE);
    assert_eq!(value["same_model_correlated"], true);
    assert_eq!(value["output"]["task"], "extract-semantic-document");
    assert_eq!(value["output"]["operation"], "resume");
    assert_eq!(value["output"]["committed"], 2);
    assert_eq!(value["output"]["attempts"], 3);
    assert_eq!(value["output"]["reserved_work"], serde_json::to_value(limits.max_work).unwrap());
    assert!(bytes.ends_with(b"\n")); assert!(bytes.len() < job_cli::REPORT_BYTES);
    for private in ["text", "schema", "claims", "semantic_options", "key", "job_dir", "results", "documents"] {
        assert!(value.get(private).is_none()); assert!(value["output"].get(private).is_none());
    }
    let text = std::str::from_utf8(&bytes).unwrap();
    for marker in ["private-schema", "private-options", "private.snapshot", "amount-check-v1", "The amount is "] {
        assert!(!text.contains(marker));
    }
}

#[test]
fn cancelled_or_oversize_completion_staging_publishes_no_bytes() {
    let available = || Ok(Duration::from_secs(1));
    assert!(publish_report(&"x".repeat(job_cli::REPORT_BYTES), &mut NoWrite,
        &available, &CancellationToken::default()).is_err());
    struct CancelWhileSerializing(CancellationToken);
    impl Serialize for CancelWhileSerializing {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.0.cancel(DecodeCancellationKind::Shutdown);
            serializer.serialize_bool(true)
        }
    }
    let token = CancellationToken::default();
    let failure = publish_report(&CancelWhileSerializing(token.clone()), &mut NoWrite,
        &available, &token).unwrap_err();
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::Shutdown));
    struct CancelOnFlush { bytes: Vec<u8>, token: CancellationToken }
    impl Write for CancelOnFlush {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.bytes.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { self.token.cancel(DecodeCancellationKind::Shutdown); Ok(()) }
    }
    let token = CancellationToken::default();
    let mut writer = CancelOnFlush { bytes: Vec::new(), token: token.clone() };
    let failure = publish_report(&true, &mut writer, &available, &token).unwrap_err();
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::Shutdown));
    assert_eq!(writer.bytes, b"true\n");
}
