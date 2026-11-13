//! Typed cancellation, completion and publication fixtures; no native inference.
use super::*;
use crate::{corpus::native_resolve::NativeResolveError, hosted::RunStop,
    native_engine::strict_int4::StrictInt4Error};

fn causes() -> [DecodeCancellationKind; 11] {
    [DecodeCancellationKind::User, DecodeCancellationKind::Timeout, DecodeCancellationKind::Deadline,
        DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown, DecodeCancellationKind::LinkedExit]
}
fn stopped(kind: DecodeCancellationKind, task_error: Option<HostedError>) -> HostedError {
    HostedError::Stopped { stop: RunStop { kind, runtime_reason: None, runtime_checkpoint_error: None },
        task_error: task_error.map(Box::new) }
}

#[test]
fn all_cancellation_variants_keep_their_policy_boundary_through_nested_failures() {
    for cause in causes() {
        let expected = if matches!(cause, DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout) {
            CandidateError::Timeout
        } else { CandidateError::Execution };
        assert_eq!(graph_error(ResolveError::Cancelled(cause)), expected);
        assert_eq!(planning_error(Int4ResolveError::Resolution(ResolveError::Cancelled(cause))), expected);
        assert_eq!(planning_error(Int4ResolveError::Planning(NativeResolveError::Resolution(
            ResolveError::Cancelled(cause)))), expected);
        assert_eq!(planning_error(Int4ResolveError::Native(StrictInt4Error::Cancelled(cause))), expected);
        assert_eq!(execution_error(HostedError::Int4(Int4HostError::Resolution(
            Int4ResolveError::Resolution(ResolveError::Cancelled(cause))))), expected);
        assert_eq!(execution_error(stopped(cause, None)), expected);
    }
    assert_eq!(planning_error(Int4ResolveError::Identity), CandidateError::Identity);
    assert_eq!(planning_error(Int4ResolveError::WorkBudget), CandidateError::Planning);
    assert_eq!(graph_error(ResolveError::InvalidAnchor), CandidateError::Input);
    assert_eq!(graph_error(ResolveError::PairBudget), CandidateError::Planning);
    let mut deadline = DeadlineControl { started: Instant::now(), deadline: Duration::ZERO };
    assert_eq!(deadline.checkpoint(0), Some(DecodeCancellationKind::Deadline));
}

#[test]
fn wrapper_fail_fast_and_linked_exit_keep_physical_error_but_deadline_keeps_precedence() {
    for cause in [DecodeCancellationKind::FailFast, DecodeCancellationKind::LinkedExit] {
        assert_eq!(execution_error(stopped(cause, Some(HostedError::ModelIdentity))), CandidateError::Identity);
        assert_eq!(execution_error(stopped(cause, Some(HostedError::Int4(Int4HostError::Resolution(
            Int4ResolveError::Native(StrictInt4Error::Cancelled(DecodeCancellationKind::Deadline))))))),
            CandidateError::Timeout);
    }
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout] {
        assert_eq!(execution_error(stopped(cause, Some(HostedError::ModelIdentity))), CandidateError::Timeout);
    }
    assert_eq!(execution_error(HostedError::Int4(Int4HostError::Resolution(Int4ResolveError::Identity))),
        CandidateError::Identity);
    assert_eq!(execution_error(HostedError::ResourceDomain), CandidateError::Identity);
    assert_eq!(execution_error(HostedError::Int4(Int4HostError::Resolution(Int4ResolveError::Accounting))),
        CandidateError::Execution);
}

fn no_neural_result() -> (Int4ResolutionRun, Expected) {
    struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    }
    let documents = vec![ResolutionDocument { id: "empty".to_owned(), text: "Private source".to_owned(), mentions: Vec::new() }];
    let plan = ResolutionPlan::prepare(&documents, ResolveOptions::default(), ResolveLimits::default(), &mut Continue).unwrap();
    let result = plan.finalize(Vec::new(), &mut Continue).unwrap();
    // This only exercises the publication checker. It is not a native receipt,
    // a public scoring provider or a claim that a model was executed.
    (Int4ResolutionRun { schema_version: 1, execution: INT4_RESOLUTION_EXECUTION,
        numerics_profile: STRICT_INT4_PROFILE, scoring_execution: INT4_SCORING_EXECUTION,
        group_width: 128, model_evaluated: false, head_count: 0,
        planned_model_work: Int4Work::default(), model_work: Int4Work::default(), rewound_positions: 0, result },
        Expected { documents: 1, mentions: 0, pairs: 0, group_width: 128, work: Int4Work::default() })
}

#[test]
fn publication_checker_refuses_incomplete_or_relabelled_snapshot_accounting() {
    let (valid, expected) = no_neural_result(); check_result(&valid, expected).unwrap();
    for mutation in 0..14 {
        let (mut run, expected) = no_neural_result();
        match mutation {
            0 => run.schema_version = 2,
            1 => run.execution = "different-executor",
            2 => run.numerics_profile = "strict-quantized-v1",
            3 => run.scoring_execution = "different-scorer",
            4 => run.group_width = 32,
            5 => run.model_evaluated = true,
            6 => run.head_count = 2,
            7 => run.result.document_count = 0,
            8 => run.planned_model_work.forward_positions = 1,
            9 => run.model_work.forward_positions = 1,
            10 => run.model_work.projected_logits = 1,
            11 => run.model_work.attention_pairs = 1,
            12 => run.model_work.projections.dot_products = 1,
            _ => run.model_work.projections.multiply_accumulates = 1,
        }
        assert_eq!(check_result(&run, expected), Err(CandidateError::Execution));
    }
    let (run, mut expected) = no_neural_result(); expected.mentions = 1;
    assert_eq!(check_result(&run, expected), Err(CandidateError::Execution));
    expected.mentions = 0; expected.pairs = 1;
    assert_eq!(check_result(&run, expected), Err(CandidateError::Execution));
}

#[test]
fn completed_publication_keeps_cluster_data_and_hides_private_provenance() {
    let result = serde_json::json!({"fixture_only":true,"result":{"calibration":"uncalibrated",
        "mentions":[{"document_id":"a","surface":"Élan"}],"clusters":[{"id":0,"mentions":[0]}]}});
    let provenance = Provenance { protocol: "fnlp-candidate-int4-resolve-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "resolve",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "model-source-root",
        model_content_sha256: "model-content", candidate_format: "candidate-format", quant_recipe: "candidate:group=128",
        group_width: 128, integer_kernel: "s_scalar_i32", snapshot: None,
        cohort_pairs: None, cohort_chunks: None };
    let response = Response { provenance: &provenance, output: &result };
    let mut bytes = Vec::new(); publish(&response, 4096, &mut bytes).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["output"], result);
    assert_eq!(value["protocol"], "fnlp-candidate-int4-resolve-v1");
    for absent in ["documents", "text", "prompt", "prompt_digest", "corpus_snapshot_digest", "binding",
        "source_dir", "source_manifest", "snapshot_path"] { assert!(value.get(absent).is_none()); }
    let mut refused = Vec::new();
    assert_eq!(publish(&response, 1, &mut refused), Err(CandidateError::Output));
    assert!(refused.is_empty());
}

#[test]
fn cohort_provenance_reports_only_explicit_physical_schedule() {
    let provenance = Provenance { protocol: "fnlp-candidate-int4-resolve-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "resolve",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "source-root",
        model_content_sha256: "model-content", candidate_format: "candidate-format", quant_recipe: "group=128",
        group_width: 128, integer_kernel: "s_scalar_i32", snapshot: None,
        cohort_pairs: Some(3), cohort_chunks: None };
    let value = serde_json::to_value(&provenance).unwrap();
    assert_eq!(value["cohort_pairs"], 3);
    assert!(value.get("cohort_chunks").is_none());
    for field in ["source", "documents", "question", "surface", "identity", "prompt", "scores"] {
        assert!(value.get(field).is_none());
    }
}

#[test]
fn complete_resolution_publication_refuses_deadline_expiry_during_serialization() {
    use std::cell::Cell;
    struct Expire<'a>(&'a Cell<bool>);
    impl Serialize for Expire<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.0.set(true);
            serializer.serialize_bool(true)
        }
    }
    struct NoOutput;
    impl Write for NoOutput {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("expired result reached stdout"); }
        fn flush(&mut self) -> io::Result<()> { panic!("expired result flushed stdout"); }
    }
    let expired = Cell::new(false);
    let remaining = || if expired.get() { Err(CandidateError::Timeout) } else { Ok(Duration::from_secs(1)) };
    assert_eq!(publish_complete(&Expire(&expired), 64, &mut NoOutput, &remaining), Err(CandidateError::Timeout));
}

#[test]
fn complete_resolution_publication_keeps_deadline_authority_through_flush() {
    use std::cell::Cell;
    struct ExpiringWriter<'a> { expired: &'a Cell<bool>, bytes: Vec<u8> }
    impl Write for ExpiringWriter<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(bytes); Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { self.expired.set(true); Ok(()) }
    }
    let expired = Cell::new(false);
    let remaining = || if expired.get() { Err(CandidateError::Timeout) } else { Ok(Duration::from_secs(1)) };
    let mut writer = ExpiringWriter { expired: &expired, bytes: Vec::new() };
    assert_eq!(publish_complete(&true, 64, &mut writer, &remaining), Err(CandidateError::Timeout));
    assert_eq!(writer.bytes, b"true\n");
}

#[test]
fn flush_failure_remains_output_failure_after_complete_staging() {
    struct FlushFailure(Vec<u8>);
    impl Write for FlushFailure {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.0.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("fixture flush failure")) }
    }
    let value = serde_json::json!({"fixture_only":true,"complete_resolution":true});
    let mut writer = FlushFailure(Vec::new());
    assert_eq!(publish(&value, 4096, &mut writer), Err(CandidateError::Output));
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&writer.0).unwrap(), value);
}

#[test]
fn row_team_receipt_is_physical_and_keeps_the_unchanged_complete_resolution_output() {
    let (result, expected) = no_neural_result();
    check_result(&result, expected).unwrap();
    let provenance = Provenance { protocol: "fnlp-candidate-int4-resolve-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "resolve",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "source-root",
        model_content_sha256: "model-content", candidate_format: "candidate-format", quant_recipe: "group=32",
        group_width: 32, integer_kernel: "s_scalar_i32", snapshot: None,
        cohort_pairs: None, cohort_chunks: None };
    let response = Response { provenance: &provenance, output: &result };
    let scalar = serde_json::to_value(&response).unwrap();
    // Zero-work receipt is a wire fixture; it makes no neural-execution claim.
    let compute = crate::hosted::Int4ComputeReceipt { schedule: "int4-output-row-team-v1",
        compute_threads: 3, completed_projections: 0, completed_rows: 0, completed_worker_rows: 0 };
    let mut bytes = Vec::new();
    publish_complete(&compute::Response { response: &response, compute: Some(&compute) },
        4096, &mut bytes, &|| Ok(Duration::from_secs(1))).unwrap();
    let mut actual: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(actual.as_object_mut().unwrap().remove("compute"), Some(serde_json::to_value(compute).unwrap()));
    assert_eq!(actual, scalar);
    assert_eq!(serde_json::to_value(compute::Response { response: &response, compute: None }).unwrap(), scalar);
}
