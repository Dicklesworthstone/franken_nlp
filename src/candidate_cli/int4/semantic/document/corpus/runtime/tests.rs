//! Pure full-stream admission and typed cancellation boundary definitions.
use super::*;
use super::super::tests::fixture;
use crate::hosted::RunStop;

#[test]
fn caller_schema_and_transport_stage_overlap_with_the_full_host_peak() {
    for route in 0..3 {
        let (command, mut args, _limits, options) = fixture();
        if route == 1 { args.save_snapshot = Some(PathBuf::from("new.snapshot")); }
        if route == 2 {
            args.source_dir = None; args.source_manifest = None;
            args.snapshot = Some(PathBuf::from("pinned.snapshot"));
            args.expected_model_sha256 = Some("ab".repeat(32));
        }
        let object = command.validate(&args).unwrap();
        let limits = command.policy_limits(&args, object, &options).unwrap();
        let initial = limits.document.map.common;
        let host = document::host_limits(&args, &command.document, limits.document,
            &options, limits.review, Duration::from_secs(30));
        let quote = int4_semantic_object_document_corpus_requirements(args.group_width,
            limits.task, &options, &UntrustedEncoding::ByteFallback, host, limits.work,
            limits.transport, limits.io_bytes, None).unwrap();
        let caller = document::caller_preparation_bytes(&command.document, initial).unwrap() + limits.io_bytes;
        let mut available = initial;
        available.memory_bytes = u64::MAX;
        let admitted = admit_memory(&args, available, quote, caller).unwrap();
        let loading = if initial.snapshot_pin.is_some() { initial.snapshot_need.scratch_bytes }
            else { initial.source_preparation_bytes + initial.load.scratch_payload_bytes };
        let saving = if args.save_snapshot.is_some() { initial.snapshot_need.scratch_bytes } else { 0 };
        assert_eq!(admitted.minimum_memory_bytes, caller + initial.load.weight_payload_bytes
            + ALLOCATOR_BYTES + loading.max(saving).max(quote.total_bytes().unwrap()) + CLEANUP_BYTES);
        let mut short = initial;
        short.memory_bytes = admitted.minimum_memory_bytes - 1;
        assert!(admit_memory(&args, short, quote, caller).is_err());
        assert!(limits.transport.max_document_bytes > 0);
    }
}

#[test]
fn complete_review_source_and_actual_team_geometry_are_shared_by_both_admission_layers() {
    let (command, args, limits, options) = fixture();
    let mut host = document::host_limits(&args, &command.document, limits.document,
        &options, limits.review, Duration::from_secs(30));
    assert_eq!(host.review.chunks.max_input_bytes, args.max_input_bytes + options.limits.max_claim_bytes);
    let bytes = int4_semantic_object_document_corpus_requirements(args.group_width, limits.task,
        &options, &UntrustedEncoding::ByteFallback, host, limits.work, limits.transport,
        limits.io_bytes, None).unwrap();
    let team = int4_semantic_object_document_corpus_requirements(args.group_width, limits.task,
        &options, &UntrustedEncoding::ByteFallback, host, limits.work, limits.transport,
        limits.io_bytes, Some(2)).unwrap();
    assert_eq!(bytes.input_bytes, team.input_bytes);
    assert!(team.team_workspace_bytes > 0);
    assert!(admit_memory(&args, limits.document.map.common, team, 0).is_err(),
        "a two-thread quote cannot enter an admitted serial command");
    host.review.chunks.max_input_bytes -= 1;
    assert!(int4_semantic_object_document_corpus_requirements(args.group_width, limits.task,
        &options, &UntrustedEncoding::ByteFallback, host, limits.work, limits.transport,
        limits.io_bytes, None).is_err());
}

#[test]
fn all_eleven_cancellation_causes_survive_the_stream_and_supervisor_boundaries() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = document::cancelled(cause);
        let fault = BatchFault::cancelled(cause);
        let direct = batch_failure(fault);
        assert_eq!(direct.cancellation, Some(cause));
        assert_eq!(direct.exit, expected.exit);
        let wrapped = host_failure(HostedError::Stopped { stop: RunStop { kind: cause,
            runtime_reason: None, runtime_checkpoint_error: None }, task_error: None });
        assert_eq!(wrapped.cancellation, Some(cause));
        assert_eq!(wrapped.exit, expected.exit);
    }
    for cause in [FailFast, LinkedExit] {
        let failure = host_failure(HostedError::Stopped { stop: RunStop { kind: cause,
            runtime_reason: None, runtime_checkpoint_error: None },
            task_error: Some(Box::new(HostedError::BatchSetup(BatchCode::WorkLimit.into()))) });
        assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
        assert_eq!(failure.cancellation, Some(cause));
    }
}

#[test]
fn stream_limits_and_failed_items_do_not_become_success() {
    for code in [BatchCode::LineLimit, BatchCode::DocumentLimit, BatchCode::InputLimit,
        BatchCode::RequestLimit, BatchCode::OutputLineLimit, BatchCode::OutputLimit, BatchCode::WorkLimit] {
        assert_eq!(batch_failure(code.into()).exit, ErrorCode::BudgetOrTimeout);
    }
    for code in [BatchCode::Admission, BatchCode::Allocation] {
        assert_eq!(batch_failure(code.into()).exit, ErrorCode::AdmissionOrResourceLimit);
    }
    assert_eq!(batch_failure(BatchCode::Execution.into()).exit, ErrorCode::StructuredTaskNoResult);
    assert_eq!(batch_failure(BatchCode::OutputIo.into()).exit, ErrorCode::Generic);
}

#[test]
fn provenance_contains_explicit_diagnostic_scope_and_no_private_task_configuration() {
    let provenance = Provenance {
        protocol: "fnlp-candidate-int4-semantic-object-corpus-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        task: "extract-semantic-document", execution: INT4_SEMANTIC_OBJECT_DOCUMENT_EXECUTION,
        same_model_correlated: true, calibration: "experimental_uncalibrated_diagnostic_only",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture-revision", source_root_sha256: "fixture-root",
        model_content_sha256: "fixture-content", candidate_format: INT4_SOURCE_FORMAT,
        quant_recipe: "fixture-recipe", group_width: 32, integer_kernel: "scalar", snapshot: None,
        compute_schedule: None, input_encoding: None,
    };
    let value = serde_json::to_value(provenance).unwrap();
    assert_eq!(value["task"], "extract-semantic-document");
    assert_eq!(value["same_model_correlated"], true);
    for name in ["schema", "semantic_options", "source", "question", "input_encoding", "snapshot", "compute_schedule"] {
        assert!(value.get(name).is_none(), "private or unselected field {name}");
    }
}
