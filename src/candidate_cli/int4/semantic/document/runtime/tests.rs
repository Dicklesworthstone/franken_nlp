//! UNRUN pure source/admission/publication fixtures; no model execution.
use super::*;
use super::super::tests::fixture;
use crate::hosted::RunStop;
use crate::batch::{BatchFault, extract::grouped_int4::long::object::Int4ObjectExtractionMapError};

struct NeverStop;
impl DecodeStepControl for NeverStop {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused result reached stdout"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused result reached flush"); }
}

#[test]
fn original_schema_rules_validate_without_inventing_chunk_or_source_authority() {
    let (command, args, limits, options) = fixture();
    let task = command.object.map.extraction.budget(&args, limits.map.common);
    let schema = r#"{"type":"object","properties":{"amount":{"type":"integer"}},"required":["amount"],"additionalProperties":false}"#;
    let request = source_request(&command, &options, schema.to_owned(), task, limits, &mut NeverStop).unwrap();
    assert_eq!(request.schema, schema, "the original required set must survive CLI preparation");
    let wrong = r#"{"type":"object","properties":{"amount":{"type":"string","maxLength":8}},"required":["amount"],"additionalProperties":false}"#;
    assert!(source_request(&command, &options, wrong.to_owned(), task, limits, &mut NeverStop).is_err());
}

#[test]
fn peak_includes_caller_staging_and_complete_host_ownership_for_all_load_routes() {
    for route in 0..3 {
        let (command, mut args, limits, options) = fixture();
        let mut common = limits.map.common;
        if route == 1 { args.save_snapshot = Some(PathBuf::from("new.snapshot")); }
        if route == 2 {
            args.source_dir = None; args.source_manifest = None;
            args.snapshot = Some(PathBuf::from("pinned.snapshot"));
            args.expected_model_sha256 = Some("ab".repeat(32));
            common = command.validate(&args).unwrap().map.common;
        }
        let review = command.review_limits(&args, common, &options).unwrap();
        let host = host_limits(&args, &command, limits, &options, review, Duration::from_secs(17));
        let task = command.object.map.extraction.budget(&args, common);
        let quote = int4_semantic_object_document_requirements(args.group_width, task,
            &options, &UntrustedEncoding::ByteFallback, host, None).unwrap();
        let schema_bytes = RootObjectSchema::workspace_bytes(command.object.map.extraction.compiler()).unwrap();
        let preparation = caller_preparation_bytes(&command, common).unwrap();
        assert!(preparation >= common.preparation_bytes + schema_bytes + OPTIONS_BYTES as u64 * 16);
        let caller = preparation + (options.limits.max_output_bytes + 4096) * 2;
        // Test exact arithmetic independently of the chosen process ceiling.
        common.memory_bytes = u64::MAX;
        let admitted = admit_memory(&args, common, quote, caller).unwrap();
        let load = if route == 2 { common.snapshot_need.scratch_bytes }
            else { common.source_preparation_bytes + common.load.scratch_payload_bytes };
        let save = if route == 1 { common.snapshot_need.scratch_bytes } else { 0 };
        assert_eq!(admitted.minimum_memory_bytes, caller + common.load.weight_payload_bytes
            + ALLOCATOR_BYTES + load.max(save).max(quote.total_bytes().unwrap()) + CLEANUP_BYTES);
        common.memory_bytes = admitted.minimum_memory_bytes;
        assert!(admit_memory(&args, common, quote, caller).is_ok());
        common.memory_bytes -= 1;
        assert_eq!(admit_memory(&args, common, quote, caller).unwrap_err().exit,
            ErrorCode::AdmissionOrResourceLimit);
        for axis in 0..2 {
            let mut changed = quote;
            if axis == 0 { changed.native.kv_bytes += 1; } else { changed.team_workspace_bytes += 1; }
            assert_eq!(admit_memory(&args, common, changed, caller).unwrap_err().exit,
                ErrorCode::ArtifactIntegrityOrFormatOrVersion);
        }
    }
}

#[test]
fn all_typed_cancellations_survive_native_and_physical_error_wrappers() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = cancelled(cause);
        for error in [
            HostedError::Stopped { stop: RunStop { kind: cause,
                runtime_reason: None, runtime_checkpoint_error: None }, task_error: None },
            HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))),
            HostedError::Int4(Int4HostError::ObjectExtractionMap(Int4ObjectExtractionMapError::Reconciliation(
                RootObjectError::Cancelled(cause)))),
            HostedError::Int4(Int4HostError::Judge(Int4JudgeError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::FaithfulnessMap(Int4FaithfulnessMapError::Judge(Int4JudgeError::Cancelled(cause)))),
            HostedError::Int4(Int4HostError::SemanticExtraction(Int4SemanticError::Cancelled(cause))),
            HostedError::BatchSetup(BatchFault::cancelled(cause)),
            HostedError::Int4(Int4HostError::SemanticObjectDocument(Int4SemanticObjectDocumentError::Cancelled(cause))),
            HostedError::Scope {
                source: asupersync::cx::ScopedCpuError::WorkerCapExceeded { cap: 1 },
                physical_error: Some(Box::new(HostedError::Int4(Int4HostError::SemanticObjectDocument(
                    Int4SemanticObjectDocumentError::Cancelled(cause))))),
            },
        ] {
            let failure = host_error(error);
            assert_eq!(failure.cancellation, Some(cause));
            assert_eq!(failure.exit, expected.exit);
        }
        let stop = CancellationToken::default(); stop.cancel(cause);
        let failure = publish(&true, 1024, &mut NoWrite,
            &|| Err(cancelled(Deadline)), &stop).unwrap_err();
        assert_eq!(failure.cancellation, Some(cause));
        assert_eq!(failure.exit, expected.exit);
    }
    for cause in [FailFast, LinkedExit] {
        let failure = host_error(HostedError::Stopped { stop: RunStop { kind: cause,
            runtime_reason: None, runtime_checkpoint_error: None },
            task_error: Some(Box::new(HostedError::Int4(Int4HostError::SemanticObjectDocument(
                Int4SemanticObjectDocumentError::WorkBudget)))) });
        assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
        assert_eq!(failure.cancellation, Some(cause));
    }
}

#[test]
fn nested_review_resource_limits_keep_their_actionable_error_class() {
    for error in [Int4FaithfulnessMapError::Work, Int4FaithfulnessMapError::Output] {
        let failure = host_error(HostedError::Int4(Int4HostError::SemanticObjectDocument(
            Int4SemanticObjectDocumentError::JudgeMap(error))));
        assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
        assert_eq!(failure.cancellation, None);
    }
    let failure = host_error(HostedError::Int4(Int4HostError::SemanticObjectDocument(
        Int4SemanticObjectDocumentError::JudgeMap(Int4FaithfulnessMapError::Allocation))));
    assert_eq!(failure.exit, ErrorCode::AdmissionOrResourceLimit);
}

#[test]
fn checked_publication_preserves_no_prefix_failure_and_post_flush_cancellation() {
    let available = || Ok(Duration::from_secs(1));
    assert!(publish(&"oversized", 4, &mut NoWrite, &available, &CancellationToken::default()).is_err());
    struct CancelWhileSerializing(CancellationToken);
    impl Serialize for CancelWhileSerializing {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.0.cancel(DecodeCancellationKind::Shutdown);
            serializer.serialize_bool(true)
        }
    }
    let stop = CancellationToken::default();
    let failure = publish(&CancelWhileSerializing(stop.clone()), 1024,
        &mut NoWrite, &available, &stop).unwrap_err();
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::Shutdown));
    struct CancelOnFlush { bytes: Vec<u8>, stop: CancellationToken, flushes: usize }
    impl Write for CancelOnFlush {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.bytes.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { self.flushes += 1; self.stop.cancel(DecodeCancellationKind::Shutdown); Ok(()) }
    }
    let stop = CancellationToken::default();
    let mut output = CancelOnFlush { bytes: Vec::new(), stop: stop.clone(), flushes: 0 };
    let failure = publish(&true, 1024, &mut output, &available, &stop).unwrap_err();
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::Shutdown));
    assert_eq!(output.bytes, b"true\n");
    assert_eq!(output.flushes, 1);
}

#[test]
fn metadata_distinguishes_correlated_review_without_serializing_private_configuration() {
    let response = Response { protocol: "fnlp-candidate-int4-semantic-object-document-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        same_model_correlated: true, calibration: "experimental_uncalibrated_diagnostic_only",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture-revision", source_root_sha256: "fixture-root",
        model_content_sha256: "fixture-content", candidate_format: INT4_SOURCE_FORMAT,
        quant_recipe: "fixture-recipe", group_width: 32, integer_kernel: "scalar", snapshot: None,
        input_encoding: None, output: &true };
    let value = serde_json::to_value(response).unwrap();
    assert_eq!(value["same_model_correlated"], true);
    assert_eq!(value["calibration"], "experimental_uncalibrated_diagnostic_only");
    assert!(value.get("semantic_options").is_none());
    assert!(value.get("snapshot").is_none());
    assert!(value.get("input_encoding").is_none());
}
