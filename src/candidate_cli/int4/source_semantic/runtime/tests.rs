//! Source preflight, admission and publication fixtures; no neural inference.
use super::*;
use std::cell::Cell;

mod model;

struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unadmitted semantic runtime read input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("unadmitted semantic runtime wrote output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("unadmitted semantic runtime flushed output"); }
}
struct NeverCancelled;
impl DecodeStepControl for NeverCancelled {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

#[test]
fn owned_runtime_rechecks_task_kind_and_changed_options_before_runtime_paths_or_io() {
    for kind in [Kind::Ner, Kind::Keyphrases] {
        let (command, args, quoted, _) = super::super::tests::fixture(Kind::Summarize);
        assert_eq!(execute(kind, args, command, quoted, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
    for kind in [Kind::Summarize, Kind::Answer] {
        for axis in 0..5 {
            let (mut command, mut args, quoted, _) = super::super::tests::fixture(kind);
            match axis {
                0 => args.seed = Some("ab".repeat(32)),
                1 => args.context_tokens = 0,
                2 => command.semantic_options = PathBuf::from("-"),
                3 => args.snapshot = Some(PathBuf::from("conflicting-private.snapshot")),
                _ => args.input = PathBuf::new(),
            }
            assert_eq!(execute(kind, args, command, quoted, &mut NoIo, &mut NoIo),
                Err(CandidateError::Arguments), "{kind:?} axis {axis}");
        }
    }
}

#[test]
fn complete_original_evidence_is_admitted_before_model_loading_and_question_is_not_evidence() {
    let (command, args, common, options) = super::super::tests::fixture(Kind::Answer);
    let budget = command.source.budget(&args, common);
    let text = serde_json::json!({"question":"Where does Alice work?", "passages":[
        {"id":"private-label-one", "text":"Alice works in 上海."},
        {"id":"private-label-two", "text":"Bob stays in Paris."}]}).to_string();
    let request = crate::candidate_cli::source::request(Kind::Answer, text, None, budget, args.max_input_bytes).unwrap();
    let facts = options.validate_evidence_with_control(&request, &mut NeverCancelled).unwrap();
    assert_eq!(facts.source_bytes, "Alice works in 上海.\n\nBob stays in Paris.".len());
    assert_eq!(facts.question_bytes, "Where does Alice work?".len());
    assert_eq!(facts.passage_count, 2);
    assert!(input_fits(facts, input_ceiling(Kind::Answer, &args, &options).unwrap()));
    let mut incomplete = options.clone();
    incomplete.judge_policy.evidence_window_bytes = 1;
    incomplete.judge_policy.max_evidence_windows = 1;
    incomplete.judge_policy.max_evidence_spans = 1;
    assert!(incomplete.validate_evidence_with_control(&request, &mut NeverCancelled).is_err());
    let mut too_small = options.clone(); too_small.limits.max_source_bytes = facts.source_bytes - 1;
    assert!(too_small.validate_evidence_with_control(&request, &mut NeverCancelled).is_err());
    let mut too_small = options.clone(); too_small.limits.max_claim_bytes = facts.question_bytes - 1;
    assert!(too_small.validate_evidence_with_control(&request, &mut NeverCancelled).is_err());
    let mut expired = DeadlineControl { started: Instant::now(), deadline: Duration::ZERO };
    let error = options.validate_evidence_with_control(&request, &mut expired).unwrap_err();
    assert_eq!(planning_error(error), CandidateError::Timeout);
}

#[test]
fn declared_quote_bounds_actual_evidence_and_cannot_hide_a_larger_population_or_storage_axis() {
    for kind in [Kind::Summarize, Kind::Answer] {
        let (command, args, common, options) = super::super::tests::fixture(kind);
        let common = command.admit_options(&args, common, &options).unwrap();
        let budget = command.source.budget(&args, common);
        let host = host_limits(&args, &command, common, Duration::from_secs(1));
        let ceiling = input_ceiling(kind, &args, &options).unwrap();
        let actual = SourceSemanticInputFacts { source_bytes: 64,
            question_bytes: if kind == Kind::Answer { 20 } else { 0 },
            passage_count: if kind == Kind::Answer { 2 } else { 0 } };
        assert!(input_fits(actual, ceiling));
        let upper = host.requirements(args.group_width, budget, options.judge_budget,
            options.limits, judge_limits(&options), ceiling).unwrap();
        let quote = host.requirements(args.group_width, budget, options.judge_budget,
            options.limits, judge_limits(&options), actual).unwrap();
        assert!(quote_fits(quote, upper)); assert_eq!(quote.native, upper.native);
        assert_eq!(quote.output_bytes, upper.output_bytes);
        assert!(quote.preparation_bytes < upper.preparation_bytes);
        for axis in 0..3 {
            let mut larger = ceiling;
            match axis { 0 => larger.source_bytes += 1, 1 => larger.question_bytes += 1,
                _ => larger.passage_count += 1 }
            assert!(!input_fits(larger, ceiling));
        }
        for axis in 0..4 {
            let mut larger = upper;
            match axis { 0 => larger.native.kv_bytes += 1, 1 => larger.native_workspace_bytes += 1,
                2 => larger.preparation_bytes += 1, _ => larger.output_bytes += 1 }
            assert!(!quote_fits(larger, upper));
        }
    }
}

#[test]
fn complete_memory_admission_keeps_both_preparation_owners_and_prices_the_peak_stage() {
    for snapshot in [false, true] {
        for save in [false, true] {
            let (command, mut args, _, options) = super::super::tests::fixture(Kind::Answer);
            if snapshot {
                args.source_dir = None; args.source_manifest = None;
                args.snapshot = Some(PathBuf::from("private.snapshot"));
                args.expected_model_sha256 = Some("ab".repeat(32));
            }
            if save { args.save_snapshot = Some(PathBuf::from("private-output.snapshot")); }
            if snapshot && save {
                assert_eq!(command.validate(&args).err(), Some(CandidateError::Arguments));
                continue;
            }
            let common = command.admit_options(&args, command.validate(&args).unwrap(), &options).unwrap();
            let budget = command.source.budget(&args, common);
            let host = host_limits(&args, &command, common, Duration::from_secs(1));
            let quote = host.requirements(args.group_width, budget, options.judge_budget, options.limits,
                judge_limits(&options), SourceSemanticInputFacts { source_bytes: 128, question_bytes: 20, passage_count: 2 }).unwrap();
            let admitted = admit_memory(&args, common, quote).unwrap();
            let loading = if snapshot { common.snapshot_need.scratch_bytes }
                else { common.source_preparation_bytes + common.load.scratch_payload_bytes };
            let saving = if save { common.snapshot_need.scratch_bytes } else { 0 };
            let execution = quote.total_bytes().unwrap();
            assert_eq!(admitted.minimum_memory_bytes, common.preparation_bytes + common.load.weight_payload_bytes
                + ALLOCATOR_BYTES + loading.max(saving).max(execution) + CLEANUP_BYTES);
            assert!(quote.preparation_bytes >= host.preparation_reserve_bytes);
            assert_eq!(command.source.budget(&args, admitted), budget);
            assert_eq!(admitted.result_bytes as u64, options.limits.max_output_bytes);
            let mut insufficient = common; insufficient.memory_bytes = admitted.minimum_memory_bytes - 1;
            assert_eq!(admit_memory(&args, insufficient, quote).err(), Some(CandidateError::Memory));
            let mut mismatched = quote; mismatched.native.kv_bytes += 1;
            assert_eq!(admit_memory(&args, common, mismatched).err(), Some(CandidateError::Identity));
            let mut overflow = common; overflow.preparation_bytes = u64::MAX;
            assert!(admit_memory(&args, overflow, quote).is_err());
        }
    }
}

#[test]
fn publication_preserves_the_original_source_result_and_separate_correlated_diagnostics() {
    // Wire-format fixture only: these values are not fabricated inference evidence.
    let original = serde_json::json!({"result":{"task":"summarize", "result":{
        "semantic_support":"not_assessed", "bullets":[{"text":"上海", "citations":[{
            "byte_start":4, "byte_end":10, "scalar_start":3, "scalar_end":5}]}]}}});
    let value = serde_json::json!({"source_task":original, "semantic":{
        "same_model_correlated":true, "calibration":SOURCE_SEMANTIC_CALIBRATION,
        "evidence_scope":SOURCE_SEMANTIC_SCOPE, "statements":[{"status":"contradicted"}]}});
    let response = Response { protocol: "fnlp-candidate-int4-semantic-source-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "summarize",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "fixture-source",
        model_content_sha256: "fixture-model", candidate_format: "fixture-format", quant_recipe: "fixture-group",
        group_width: 128, integer_kernel: "s_scalar_i32", snapshot: None, output: &value };
    let mut bytes = Vec::new();
    publish_checked(&response, 8192, &mut bytes, &|| Ok(())).unwrap();
    let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(wire["output"], value); assert_eq!(wire["output"]["source_task"], original);
    assert_eq!(wire["evidence"], "non_authoritative"); assert_eq!(wire["task"], "summarize");
    for absent in ["semantic_options", "options_path", "source_dir", "source_manifest", "prompt_digest",
        "judge_identity", "decision_policy_digest", "snapshot"] { assert!(wire.get(absent).is_none()); }
}

#[test]
fn deadline_crossed_during_combined_serialization_publishes_no_partial_result() {
    struct Expires<'a> { expired: &'a Cell<bool> }
    impl Serialize for Expires<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.expired.set(true);
            serializer.serialize_str("complete original and review fixture")
        }
    }
    let expired = Cell::new(false); let mut bytes = Vec::new();
    let result = publish_checked(&Expires { expired: &expired }, 4096, &mut bytes,
        &|| if expired.get() { Err(CandidateError::Timeout) } else { Ok(()) });
    assert_eq!(result, Err(CandidateError::Timeout)); assert!(bytes.is_empty());
    assert_eq!(publish_checked(&serde_json::json!({"source_task":{},"semantic":{}}), 1,
        &mut bytes, &|| Ok(())), Err(CandidateError::Output));
    assert!(bytes.is_empty());
}

#[test]
fn typed_cancellation_memory_and_identity_failures_keep_their_public_error_class() {
    for (cause, expected) in [(DecodeCancellationKind::Deadline, CandidateError::Timeout),
        (DecodeCancellationKind::Timeout, CandidateError::Timeout),
        (DecodeCancellationKind::User, CandidateError::Execution),
        (DecodeCancellationKind::PollQuota, CandidateError::Execution)] {
        assert_eq!(planning_error(Int4SourceSemanticError::Cancelled(cause)), expected);
        assert_eq!(execution_error(HostedError::Int4(Int4HostError::SourceSemantic(
            Int4SourceSemanticError::Cancelled(cause)))), expected);
        assert_eq!(execution_error(HostedError::Stopped { stop: crate::hosted::RunStop {
            kind: cause, runtime_reason: None, runtime_checkpoint_error: None }, task_error: None }), expected);
    }
    assert_eq!(planning_error(Int4SourceSemanticError::WorkBudget), CandidateError::Planning);
    assert_eq!(execution_error(HostedError::Reservation(crate::ReservationError::ArithmeticOverflow)),
        CandidateError::Memory);
    assert_eq!(execution_error(HostedError::ModelIdentity), CandidateError::Identity);
    assert_eq!(execution_error(HostedError::ResourceDomain), CandidateError::Identity);
    assert_eq!(execution_error(HostedError::Limits("refused")), CandidateError::Execution);
}
