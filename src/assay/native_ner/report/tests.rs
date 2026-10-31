use super::*;
use super::super::tests::{dataset, encode, replay, replay_changed, wire};
use crate::tasks::{ner::EntityType, source_planning::SourceTaskResult};

fn key() -> CommitmentKey { CommitmentKey::new("native-ner-tests", [37; 32]).unwrap() }
fn options() -> EvalOptions {
    EvalOptions { minimum_samples: 5,
        bootstrap: crate::assay::BootstrapOptions { resamples: 24, confidence_level: 0.9, seed: 47 } }
}
fn gates() -> Gates {
    Gates { schema_version: 1, max_exact_f1_decrease: 0.0, max_type_relaxed_f1_decrease: 0.0,
        max_exact_recall_decrease: 0.0, max_document_accuracy_decrease: 0.0 }
}
fn report(replay: &NativeReplay) -> Evaluation {
    evaluate(replay, &key(), "test-population", options(), Limits::default()).unwrap()
}

#[test]
fn native_ner_report_binds_complete_observations_and_keeps_private_content_out() {
    let dataset = dataset(5); let replay = replay(&dataset, "candidate");
    let result = report(&replay);
    assert_eq!(result.body.task, NATIVE_TASK); assert_eq!(result.body.evidence, NATIVE_EVIDENCE);
    assert_eq!(result.body.score_space, ScoreSpace::NotComputed);
    assert_eq!(result.body.receipt_grade, ReceiptGrade::AuditOnly); assert!(!result.body.activation_authority);
    assert_eq!(result.body.executed_rows, 5); assert_eq!(result.body.validated_rows, 7);
    assert_eq!(result.body.bindings.execution_collection_rows, 5);
    assert_eq!(result.body.bindings.observation_collection_rows, 5);
    assert_eq!(result.body.bindings.metadata.entity_types, ["location", "person"]);
    assert_eq!(result.body.total_source_verification_steps, 3 * result.body.independent_verification_steps);
    let Measurement::Measured { metrics } = &result.body.measurement else { panic!("expected measured"); };
    assert_eq!(metrics.exact.true_positive, 10); assert_eq!(metrics.exact.f1, Some(1.0));
    assert_eq!(metrics.type_relaxed.f1, Some(1.0)); assert_eq!(metrics.by_language.len(), 2);
    let text = String::from_utf8(crate::assay::render(&result, Limits::default()).unwrap()).unwrap();
    for private in ["PRIVATE_0", "private-test-000", "private-development", "é上海", "Ana",
        &replay.input_digest.to_hex(), &replay.rows[0].unit.to_hex(), &replay.first_identity.prompt_digest.to_hex()] {
        assert!(!text.contains(private));
    }
    assert_eq!(result.seal, commit(&key(), "test-population", "native_ner_evaluation_v1", &result.body, Limits::default()).unwrap());
}

#[test]
fn identical_observed_quality_has_zero_paired_change_without_model_activation() {
    let dataset = dataset(5); let candidate = replay(&dataset, "candidate"); let baseline = replay(&dataset, "baseline");
    let result = compare(&candidate, &baseline, &key(), "scope", "declared-cpu-host", gates(), options(), Limits::default()).unwrap();
    let Decision::Passed { comparisons, paired } = &result.body.decision else { panic!("equal quality"); };
    assert_eq!(comparisons.len(), 4); assert!(comparisons.iter().all(|gate| gate.passed));
    assert_eq!(paired.equal, 5); assert_eq!(paired.exact_f1_change, Some(0.0));
    assert_eq!(paired.exact_f1_change_interval.unwrap().lower, 0.0);
    assert_eq!(paired.exact_f1_change_interval.unwrap().upper, 0.0);
    assert_eq!(result.body.gate_basis, "point_estimates"); assert!(!result.body.activation_authority);
    assert_ne!(result.body.candidate.body.bindings.execution_collection, result.body.baseline.body.bindings.execution_collection);
    assert_eq!(result.body.total_model_work, candidate.model_work.checked_add(baseline.model_work).unwrap());
}

#[test]
fn wrong_entity_type_fails_exact_gates_and_preserves_relaxed_boundary_credit() {
    let dataset = dataset(5); let baseline = replay(&dataset, "baseline");
    let candidate = replay_changed(&dataset, "candidate", |index, _, output| {
        if index == 0 { let SourceTaskResult::Ner(result) = &mut output.result else { panic!("NER fixture"); };
            result.entities[0].entity_type = EntityType::Person; }
    });
    let result = compare(&candidate, &baseline, &key(), "scope", "host", gates(), options(), Limits::default()).unwrap();
    let Decision::Rejected { comparisons, paired } = &result.body.decision else { panic!("wrong type must reject"); };
    assert_eq!(paired.candidate_worse, 1); assert_eq!(paired.equal, 4);
    assert!(paired.exact_f1_change.unwrap() < 0.0); assert_eq!(paired.type_relaxed_f1_change, Some(0.0));
    assert!(comparisons.iter().any(|gate| gate.name == "exact_f1_decrease" && !gate.passed));
    assert!(comparisons.iter().any(|gate| gate.name == "type_relaxed_f1_decrease" && gate.passed));
}

#[test]
fn missing_predictions_remain_false_negatives_and_reduce_document_accuracy() {
    let dataset = dataset(5); let baseline = replay(&dataset, "baseline");
    let candidate = replay_changed(&dataset, "candidate", |index, _, output| {
        if index == 0 { let SourceTaskResult::Ner(result) = &mut output.result else { panic!("NER fixture"); };
            result.entities.clear(); }
    });
    let result = compare(&candidate, &baseline, &key(), "scope", "host", gates(), options(), Limits::default()).unwrap();
    let Decision::Rejected { paired, .. } = result.body.decision else { panic!("missed spans must reject"); };
    assert_eq!(paired.mean_exact_error_change, 0.4);
    assert!((paired.document_accuracy_change + 0.2).abs() < 1e-12);
}

#[test]
fn observed_tokens_have_a_separate_commitment_even_when_span_metrics_match() {
    let dataset = dataset(5); let original = replay(&dataset, "candidate");
    let changed = replay_changed(&dataset, "candidate", |index, _, output| {
        if index == 0 { let SourceTaskResult::Ner(result) = &mut output.result else { panic!("NER fixture"); };
            result.generated_token_ids[0] = 3; }
    });
    let a = report(&original); let b = report(&changed);
    assert_eq!(a.body.measurement, b.body.measurement);
    assert_eq!(a.body.bindings.execution_collection, b.body.bindings.execution_collection);
    assert_ne!(a.body.bindings.observation_collection, b.body.bindings.observation_collection);
    assert_ne!(a.seal, b.seal);
}

#[test]
fn paired_scope_refuses_changed_prompt_policy_source_and_limits() {
    let dataset = dataset(5); let original = replay(&dataset, "candidate");
    for field in 0..3 {
        let changed = replay_changed(&dataset, "baseline", |index, identity, _| {
            if index == 0 { let digest = crate::execution_identity::Sha256Digest::of_bytes(b"changed policy");
                match field { 0 => identity.prompt_digest = digest, 1 => identity.decision_policy_digest = digest,
                    _ => identity.schema_digest = digest } }
        });
        assert_eq!(compare(&original, &changed, &key(), "scope", "host", gates(), options(), Limits::default()).err(), Some(AssayError::Scope));
    }
    let (manifest, mut rows) = wire(5); rows[2]["text"] = serde_json::json!("é上海 and Ana CHANGED_0");
    let (manifest, input) = encode(manifest, &rows);
    let changed = Dataset::parse(&manifest, &input, Limits::default(), NerLimits::default()).unwrap();
    let changed = replay(&changed, "baseline");
    assert_eq!(compare(&original, &changed, &key(), "scope", "host", gates(), options(), Limits::default()).err(), Some(AssayError::Scope));
    let mut changed = replay(&dataset, "baseline"); changed.ner_limits.max_verification_steps -= 1;
    assert_eq!(compare(&original, &changed, &key(), "scope", "host", gates(), options(), Limits::default()).err(), Some(AssayError::Scope));
}

#[test]
fn empty_span_corpus_cannot_pass_undefined_f1_and_recall_gates() {
    let (manifest, mut rows) = wire(5);
    for row in &mut rows { row["spans"] = serde_json::json!([]); }
    let (manifest, input) = encode(manifest, &rows);
    let dataset = Dataset::parse(&manifest, &input, Limits::default(), NerLimits::default()).unwrap();
    let replay = replay(&dataset, "candidate");
    let result = compare(&replay, &replay, &key(), "scope", "host", gates(), options(), Limits::default()).unwrap();
    let Decision::Rejected { comparisons, .. } = result.body.decision else { panic!("undefined scores must reject"); };
    assert_eq!(comparisons.iter().filter(|gate| gate.observed.is_none() && !gate.passed).count(), 3);
}

#[test]
fn tiny_population_is_insufficient_and_paired_work_never_renews_the_draw_budget() {
    let tiny = dataset(2); let tiny = replay(&tiny, "candidate");
    assert!(matches!(report(&tiny).body.measurement, Measurement::InsufficientData { rows: 2, .. }));
    let result = compare(&tiny, &tiny, &key(), "scope", "host", gates(), options(), Limits::default()).unwrap();
    assert!(matches!(result.body.decision, Decision::InsufficientData { rows: 2, .. }));
    let dataset = dataset(5); let replay = replay(&dataset, "candidate");
    let limits = Limits { max_bootstrap_draws: 5 * options().bootstrap.resamples as u64, ..Limits::default() };
    assert!(evaluate(&replay, &key(), "scope", options(), limits).is_ok());
    assert_eq!(compare(&replay, &replay, &key(), "scope", "host", gates(), options(), limits).err(), Some(AssayError::InputLimit));
}

#[test]
fn strict_gate_documents_and_scopes_refuse_ambiguous_policy() {
    let bytes = serde_json::to_vec(&gates()).unwrap(); assert_eq!(parse_gates(&bytes, Limits::default()).unwrap(), gates());
    for invalid in [br#"{"schema_version":1,"schema_version":1}"#.as_slice(),
        br#"{"schema_version":1}"#.as_slice(),
        br#"{"schema_version":1,"max_exact_f1_decrease":0,"max_type_relaxed_f1_decrease":0,"max_exact_recall_decrease":0,"max_document_accuracy_decrease":0,"extra":1}"#.as_slice()] {
        assert!(parse_gates(invalid, Limits::default()).is_err());
    }
    let dataset = dataset(5); let replay = replay(&dataset, "candidate");
    assert_eq!(evaluate(&replay, &key(), "bad scope", options(), Limits::default()).err(), Some(AssayError::Commitment));
    assert_eq!(compare(&replay, &replay, &key(), "scope", "bad host", gates(), options(), Limits::default()).err(), Some(AssayError::Scope));
    assert_eq!(evaluate(&replay, &key(), "scope", options(), Limits { max_output_bytes: 2, ..Limits::default() }).err(), Some(AssayError::Output));
}

#[test]
fn report_generation_preserves_a_checkpoint_failure() {
    let dataset = dataset(5); let replay = replay(&dataset, "candidate");
    let mut calls = 0;
    let result = evaluate_with_checkpoint::<AssayError>(&replay, &key(), "scope", options(), Limits::default(), &mut || {
        calls += 1; if calls == 10 { Err(AssayError::Allocation) } else { Ok(()) }
    });
    assert_eq!(result.err(), Some(AssayError::Allocation));
}
