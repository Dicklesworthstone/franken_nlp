use super::*;
use super::super::tests::{dataset, encode, replay, replay_changed, wire};
use crate::{batch::extract::ExtractionBatchGrounding, grammar::{CompileLimits, runtime::SourceRuntimeLimits}};

fn key() -> CommitmentKey { CommitmentKey::new("native-extract-tests", [43; 32]).unwrap() }
fn options() -> EvalOptions {
    EvalOptions { minimum_samples: 5,
        bootstrap: crate::assay::BootstrapOptions { resamples: 24, confidence_level: 0.9, seed: 47 } }
}
fn gates() -> Gates {
    Gates { schema_version: 1, max_field_f1_decrease: 0.0, max_field_recall_decrease: 0.0,
        max_document_accuracy_decrease: 0.0 }
}
fn report(replay: &NativeReplay) -> Evaluation {
    evaluate(replay, &key(), "test-population", options(), Limits::default()).unwrap()
}
fn grounded_dataset(locked: usize) -> Dataset {
    let (mut manifest, rows) = wire(locked);
    manifest.schemas[0].schema_json = r#"{"type":"object","properties":{"name":{"type":"string","x-fnlp-source":"verbatim"},"amount":{"type":"number"}},"required":["name","amount"],"additionalProperties":false}"#.to_owned();
    manifest.schemas[0].grounding = ExtractionBatchGrounding::SourceMembership;
    let (manifest, input) = encode(manifest, &rows); let extract = ExtractLimits::default();
    let compiler = CompileLimits { max_output_bytes: 100_000, ..CompileLimits::default() };
    let mut source = SourceRuntimeLimits::default();
    source.verification.max_fields = extract.max_fields_per_row;
    source.verification.max_matches = extract.max_occurrences_per_row;
    source.verification.max_scan_steps = extract.max_verification_steps;
    Dataset::parse(&manifest, &input, Limits::default(), extract, compiler, source).unwrap()
}

#[test]
fn complete_native_report_is_sealed_without_disclosing_private_json_or_cases() {
    let dataset = dataset(5); let replay = replay(&dataset, "candidate"); let result = report(&replay);
    assert_eq!(result.body.task, NATIVE_TASK); assert_eq!(result.body.evidence, NATIVE_EVIDENCE);
    assert_eq!(result.body.score_space, ScoreSpace::NotComputed);
    assert_eq!(result.body.receipt_grade, ReceiptGrade::AuditOnly); assert!(!result.body.activation_authority);
    assert_eq!(result.body.validated_rows, 7); assert_eq!(result.body.executed_rows, 5);
    assert_eq!(result.body.bindings.metadata.schema_ids, ["record"]);
    assert_eq!(result.body.bindings.execution_collection_rows, 5);
    assert_eq!(result.body.bindings.observation_collection_rows, 5);
    let Measurement::Measured { metrics } = &result.body.measurement else { panic!("measured"); };
    assert_eq!(metrics.fields.support, 10); assert_eq!(metrics.fields.true_positive, 10);
    assert_eq!(metrics.fields.f1, Some(1.0)); assert_eq!(metrics.document_accuracy, 1.0);
    assert_eq!(metrics.by_schema.len(), 1); assert_eq!(metrics.by_language.len(), 2);
    let text = String::from_utf8(crate::assay::render(&result, Limits::default()).unwrap()).unwrap();
    for private in ["PRIVATE_0", "private-test-000", "private-development", "Ana", "\"amount\"", "\"properties\"",
        &replay.input_digest.to_hex(), &replay.rows[0].unit.to_hex(), &replay.first_identity.prompt_digest.to_hex()] {
        assert!(!text.contains(private), "private value escaped report");
    }
    assert_eq!(result.seal, commit(&key(), "test-population", "native_extract_evaluation_v1", &result.body, Limits::default()).unwrap());
}

#[test]
fn identical_quality_passes_paired_gates_without_activating_a_model() {
    let dataset = dataset(5); let candidate = replay(&dataset, "candidate"); let baseline = replay(&dataset, "baseline");
    let result = compare(&candidate, &baseline, &key(), "scope", "declared-host", gates(), options(), Limits::default()).unwrap();
    let Decision::Passed { comparisons, paired } = &result.body.decision else { panic!("equal quality"); };
    assert_eq!(comparisons.len(), 3); assert!(comparisons.iter().all(|value| value.passed));
    assert_eq!(paired.equal, 5); assert_eq!(paired.field_f1_change, Some(0.0));
    let interval = paired.field_f1_change_interval.unwrap(); assert_eq!((interval.lower, interval.upper), (0.0, 0.0));
    assert_eq!(result.body.gate_basis, "point_estimates"); assert_eq!(result.body.host_scope_evidence, "caller_declared");
    assert!(!result.body.activation_authority); assert_eq!(result.body.receipt_grade, ReceiptGrade::AuditOnly);
    assert_ne!(result.body.candidate.body.bindings.execution_collection, result.body.baseline.body.bindings.execution_collection);
    assert_eq!(result.body.total_model_work, candidate.model_work.checked_add(baseline.model_work).unwrap());
    assert_eq!(result.body.total_mask_node_visits, candidate.mask_node_visits + baseline.mask_node_visits);
}

#[test]
fn wrong_values_fail_field_and_document_gates_and_negative_bounds_require_improvement() {
    let dataset = dataset(5); let good = replay(&dataset, "good"); let bad = replay_changed(&dataset, "bad");
    let result = compare(&bad, &good, &key(), "scope", "host", gates(), options(), Limits::default()).unwrap();
    let Decision::Rejected { comparisons, paired } = result.body.decision else { panic!("wrong values must reject"); };
    assert_eq!(paired.candidate_worse, 5); assert_eq!(paired.mean_field_error_change, 2.0);
    assert_eq!(paired.field_f1_change, Some(-0.5)); assert_eq!(paired.field_recall_change, Some(-0.5));
    assert_eq!(paired.document_accuracy_change, -1.0); assert!(comparisons.iter().all(|value| !value.passed));
    let improve = Gates { max_field_f1_decrease: -0.4, max_field_recall_decrease: -0.4,
        max_document_accuracy_decrease: -0.9, ..gates() };
    assert!(matches!(compare(&good, &bad, &key(), "scope", "host", improve, options(), Limits::default()).unwrap().body.decision,
        Decision::Passed { .. }));
    let require_more = Gates { max_field_f1_decrease: -0.6, ..improve };
    assert!(matches!(compare(&good, &bad, &key(), "scope", "host", require_more, options(), Limits::default()).unwrap().body.decision,
        Decision::Rejected { .. }));
}

#[test]
fn shared_gold_source_work_is_counted_once_and_native_verification_three_times() {
    let dataset = grounded_dataset(5); let candidate = replay(&dataset, "candidate"); let baseline = replay(&dataset, "baseline");
    let one = report(&candidate); let gold = dataset.gold_verification_steps(); assert!(gold > 0);
    assert_eq!(one.body.gold_verification_fields, 7); assert_eq!(one.body.gold_verification_matches, 12);
    assert_eq!(one.body.total_source_verification_steps, gold + 3 * one.body.independent_verification_steps);
    let paired = compare(&candidate, &baseline, &key(), "scope", "host", gates(), options(), Limits::default()).unwrap();
    assert_eq!(paired.body.total_source_verification_steps, 2 * one.body.total_source_verification_steps - gold);
    assert_eq!(paired.body.candidate.body.gold_verification_steps, paired.body.baseline.body.gold_verification_steps);
}

#[test]
fn paired_scope_binds_input_schema_language_policy_and_shared_gold_preparation() {
    let dataset = dataset(5); let original = replay(&dataset, "candidate");
    for field in 0..8 {
        let mut changed = replay(&dataset, "baseline");
        let digest = crate::execution_identity::Sha256Digest::of_bytes(b"changed scope");
        match field {
            0 => changed.input_digest = digest,
            1 => changed.rows[0].schema_id = "another-schema".to_owned(),
            2 => changed.rows[0].language = "und".to_owned(),
            3 => changed.rows[0].unit = digest,
            4 => changed.compatible_executions = [99; 32],
            5 => changed.extract_limits.max_verification_steps -= 1,
            6 => changed.gold_verification_fields += 1,
            _ => changed.gold_verification_steps += 1,
        }
        assert_eq!(compare(&original, &changed, &key(), "scope", "host", gates(), options(), Limits::default()).err(),
            Some(AssayError::Scope), "field {field}");
    }
}

#[test]
fn complete_observation_commitments_are_independent_of_equal_metric_values() {
    let dataset = dataset(5); let original = replay(&dataset, "candidate"); let mut changed = replay(&dataset, "candidate");
    changed.observations = [97; 32];
    let a = report(&original); let b = report(&changed);
    assert_eq!(a.body.measurement, b.body.measurement);
    assert_eq!(a.body.bindings.execution_collection, b.body.bindings.execution_collection);
    assert_ne!(a.body.bindings.observation_collection, b.body.bindings.observation_collection); assert_ne!(a.seal, b.seal);
}

#[test]
fn insufficient_populations_and_two_arm_draw_cost_are_explicit() {
    let tiny = dataset(2); let tiny = replay(&tiny, "candidate");
    assert!(matches!(report(&tiny).body.measurement, Measurement::InsufficientData { rows: 2, .. }));
    assert!(matches!(compare(&tiny, &tiny, &key(), "scope", "host", gates(), options(), Limits::default()).unwrap().body.decision,
        Decision::InsufficientData { rows: 2, .. }));
    let dataset = dataset(5); let replay = replay(&dataset, "candidate");
    let limits = Limits { max_bootstrap_draws: 5 * options().bootstrap.resamples as u64, ..Limits::default() };
    assert!(evaluate(&replay, &key(), "scope", options(), limits).is_ok());
    assert_eq!(compare(&replay, &replay, &key(), "scope", "host", gates(), options(), limits).err(), Some(AssayError::InputLimit));
}

#[test]
fn paired_source_totals_cannot_renew_a_complete_operation_allowance() {
    let dataset = dataset(5); let mut candidate = replay(&dataset, "candidate"); let mut baseline = replay(&dataset, "baseline");
    candidate.total_source_verification_steps = candidate.extract_limits.max_verification_steps / 2 + 1;
    baseline.total_source_verification_steps = candidate.total_source_verification_steps;
    assert!(evaluate(&candidate, &key(), "scope", options(), Limits::default()).is_ok());
    assert_eq!(compare(&candidate, &baseline, &key(), "scope", "host", gates(), options(), Limits::default()).err(),
        Some(AssayError::InputLimit));
}

#[test]
fn ambiguous_gate_documents_invalid_scopes_and_output_overruns_are_refused() {
    assert_eq!(parse_gates(&serde_json::to_vec(&gates()).unwrap(), Limits::default()).unwrap(), gates());
    for invalid in [br#"{"schema_version":1,"schema_version":1}"#.as_slice(), br#"{"schema_version":1}"#.as_slice(),
        br#"{"schema_version":1,"max_field_f1_decrease":0,"max_field_recall_decrease":0,"max_document_accuracy_decrease":0,"extra":1}"#.as_slice()] {
        assert!(parse_gates(invalid, Limits::default()).is_err());
    }
    assert_eq!(Gates { max_field_f1_decrease: f64::NAN, ..gates() }.validate(Limits::default()).err(), Some(AssayError::Qualification));
    assert_eq!(Gates { max_field_recall_decrease: 1.01, ..gates() }.validate(Limits::default()).err(), Some(AssayError::Qualification));
    let dataset = dataset(5); let replay = replay(&dataset, "candidate");
    assert_eq!(evaluate(&replay, &key(), "bad scope", options(), Limits::default()).err(), Some(AssayError::Commitment));
    assert_eq!(compare(&replay, &replay, &key(), "scope", "bad host", gates(), options(), Limits::default()).err(), Some(AssayError::Scope));
    assert_eq!(evaluate(&replay, &key(), "scope", options(), Limits { max_output_bytes: 2, ..Limits::default() }).err(), Some(AssayError::Output));
}

#[test]
fn report_finalization_preserves_checkpoint_failure() {
    let dataset = dataset(5); let replay = replay(&dataset, "candidate"); let mut calls = 0;
    let result = evaluate_with_checkpoint::<AssayError>(&replay, &key(), "scope", options(), Limits::default(), &mut || {
        calls += 1; if calls == 10 { Err(AssayError::Allocation) } else { Ok(()) }
    });
    assert_eq!(result.err(), Some(AssayError::Allocation));
}
