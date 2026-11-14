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
fn compact_qa_snapshot_quote_admits_one_encoder_and_one_team_for_the_complete_review() {
    let (_, _, _, options) = super::super::tests::fixture(Kind::Answer);
    let parsed = super::super::tests::command("answer-semantic",
        &["--input-encoding", "safe-bpe-v1", "--compute-threads", "4"]);
    let Int4Task::SourceSemantic(Kind::Answer, command) = parsed.task else { panic!("wrong semantic route"); };
    let mut args = parsed.args;
    args.source_dir = None; args.source_manifest = None;
    args.snapshot = Some(PathBuf::from("private.snapshot"));
    args.expected_model_sha256 = Some("ab".repeat(32));
    let common = command.admit_options(&args, command.validate(&args).unwrap(), &options).unwrap();
    assert!(common.snapshot_pin.is_some());
    let budget = command.source.budget(&args, common);
    let policy = command.input_encoding.policy().unwrap();
    let ceiling = input_ceiling(Kind::Answer, &args, &options).unwrap();
    let host = host_limits(&args, &command, common, Duration::from_secs(1));
    let byte = host.requirements(args.group_width, budget, options.judge_budget,
        options.limits, judge_limits(&options), ceiling).unwrap();
    let upper = host.requirements_with_encoding(args.group_width, budget, options.judge_budget,
        options.limits, judge_limits(&options), ceiling, &policy).unwrap();
    assert_eq!(upper.preparation_bytes - byte.preparation_bytes, command.input_encoding.workspace_bytes().unwrap());
    assert_eq!(upper.native, byte.native); assert_eq!(upper.native, common.native);
    assert_eq!(upper.native_workspace_bytes, byte.native_workspace_bytes);
    assert_eq!(upper.output_bytes, byte.output_bytes);
    let facts = SourceSemanticInputFacts { source_bytes: 100, question_bytes: 20, passage_count: 2 };
    assert!(input_fits(facts, ceiling));
    let actual = host.requirements_with_encoding(args.group_width, budget, options.judge_budget,
        options.limits, judge_limits(&options), facts, &policy).unwrap();
    assert!(quote_fits(actual, upper));
    assert!(actual.preparation_bytes < upper.preparation_bytes);
    let team = crate::hosted::int4_task_team_workspace_bytes(common.compute.threads, args.group_width).unwrap();
    assert_eq!(common.compute.workspace_bytes, team);
    let admitted = admit_memory(&args, common, upper).unwrap();
    let execution = upper.total_bytes().unwrap() + team;
    assert_eq!(admitted.minimum_memory_bytes, common.preparation_bytes + common.load.weight_payload_bytes
        + ALLOCATOR_BYTES + common.snapshot_need.scratch_bytes.max(execution) + CLEANUP_BYTES);
    assert_eq!(command.source.budget(&args, admitted), budget);
    let mut insufficient = common; insufficient.memory_bytes = admitted.minimum_memory_bytes - 1;
    assert_eq!(admit_memory(&args, insufficient, upper).err(), Some(CandidateError::Memory));
    let mut uncharged = common; uncharged.compute.workspace_bytes = 0;
    assert_eq!(admit_memory(&args, uncharged, upper).err(), Some(CandidateError::Identity));
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
    assert!(wire["output"].get("input_encoding").is_none());
    // Real pinned encoding supplies the receipt; the result/citations remain
    // publication fixtures and do not assert native generation or review.
    use crate::{tasks::extract::{grounded::SourceDocumentEncoder,
        semantic::grouped_int4::Int4SemanticEncoding}, tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding}};
    let policy = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 8192,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 });
    let encoder = SourceDocumentEncoder::pinned(crate::tokenizer::pinned_controls::pinned().unwrap().template_controls())
        .unwrap().with_document_encoding(policy.clone()).unwrap();
    let source = "éa 上海 <think>";
    let document = encoder.encode_with_control(source, 8192, 8192, &mut || None).unwrap();
    let work = document.input_encoding_work().unwrap();
    assert_eq!(work.input_bytes(), source.len());
    let receipt = Int4SemanticEncoding::from_work(&policy, work).unwrap().unwrap();
    let mut compact = value.clone();
    compact["input_encoding"] = serde_json::to_value(&receipt).unwrap();
    let compact_response = Response { output: &compact, ..response };
    let mut compact_bytes = Vec::new();
    publish_checked(&compact_response, 8192, &mut compact_bytes, &|| Ok(())).unwrap();
    let compact_wire: serde_json::Value = serde_json::from_slice(&compact_bytes).unwrap();
    assert_eq!(compact_wire["output"]["source_task"], original);
    assert_eq!(compact_wire["output"]["semantic"], value["semantic"]);
    assert_eq!(compact_wire["output"]["input_encoding"], serde_json::to_value(receipt).unwrap());
    let mut refused = Vec::new();
    assert_eq!(publish_checked(&compact_response, 1, &mut refused, &|| Ok(())), Err(CandidateError::Output));
    assert!(refused.is_empty());
    assert_eq!(publish_checked(&compact_response, 8192, &mut refused, &|| Err(CandidateError::Timeout)),
        Err(CandidateError::Timeout));
    assert!(refused.is_empty());
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

#[test]
fn semantic_source_teams_price_one_workspace_and_the_installed_thread_envelope() {
    for kind in [Kind::Summarize, Kind::Answer] {
        for width in ["2", "4"] {
            let (command, mut args, _, options) = super::super::tests::fixture(kind);
            let matches = compute::definition(clap::Command::new("team"))
                .try_get_matches_from(["team", "--compute-threads", width]).unwrap();
            args.compute = compute::Args::from_matches(&matches).unwrap();
            let common = command.admit_options(&args, command.validate(&args).unwrap(), &options).unwrap();
            let facts = SourceSemanticInputFacts { source_bytes: 100,
                question_bytes: if kind == Kind::Answer { 20 } else { 0 },
                passage_count: if kind == Kind::Answer { 2 } else { 0 } };
            let quote = host_limits(&args, &command, common, Duration::from_secs(1))
                .requirements(args.group_width, command.source.budget(&args, common), options.judge_budget,
                    options.limits, judge_limits(&options), facts).unwrap();
            let admitted = admit_memory(&args, common, quote).unwrap();
            let extra = crate::hosted::int4_task_team_workspace_bytes(common.compute.threads, args.group_width).unwrap();
            assert_eq!(common.compute.workspace_bytes, extra);
            let execution = quote.total_bytes().unwrap() + extra;
            let loading = common.source_preparation_bytes + common.load.scratch_payload_bytes;
            assert_eq!(admitted.minimum_memory_bytes, common.preparation_bytes + common.load.weight_payload_bytes
                + ALLOCATOR_BYTES + loading.max(execution) + CLEANUP_BYTES);
            let config = compute::resource_config(admitted);
            assert_eq!(config.scoped_cpu_children_per_coordinator + 1, common.compute.threads);
            assert_eq!(config.thread_ceiling, common.compute.threads + 1);
            let mut insufficient = common; insufficient.memory_bytes = admitted.minimum_memory_bytes - 1;
            assert_eq!(admit_memory(&args, insufficient, quote).err(), Some(CandidateError::Memory));
            let mut uncharged = common; uncharged.compute.workspace_bytes = 0;
            assert_eq!(admit_memory(&args, uncharged, quote).err(), Some(CandidateError::Identity));
        }
    }
}

#[test]
fn physical_review_statistics_do_not_change_serial_payloads_or_allow_expired_publication() {
    let value = serde_json::json!({"output":{"source_task":{"result":{"semantic_support":"not_assessed"}},
        "semantic":{"same_model_correlated":true,"statements":[{"status":"unsupported"}]}}});
    let mut baseline = Vec::new(); publish_checked(&value, 4096, &mut baseline, &|| Ok(())).unwrap();
    let mut serial = Vec::new();
    publish_checked(&compute::Response { response: &value, compute: None }, 4096, &mut serial, &|| Ok(())).unwrap();
    assert_eq!(serial, baseline);
    let receipt = crate::hosted::Int4ComputeReceipt { schedule: "int4-output-row-team-v1",
        compute_threads: 2, completed_projections: 17, completed_rows: 100, completed_worker_rows: 50 };
    let response = compute::Response { response: &value, compute: Some(&receipt) };
    let mut threaded = Vec::new(); publish_checked(&response, 4096, &mut threaded, &|| Ok(())).unwrap();
    let wire: serde_json::Value = serde_json::from_slice(&threaded).unwrap();
    assert_eq!(wire["output"], value["output"]);
    assert_eq!(wire["compute"], serde_json::to_value(receipt).unwrap());
    let mut refused = Vec::new();
    assert_eq!(publish_checked(&response, 4096, &mut refused, &|| Err(CandidateError::Timeout)),
        Err(CandidateError::Timeout));
    assert!(refused.is_empty());
}
