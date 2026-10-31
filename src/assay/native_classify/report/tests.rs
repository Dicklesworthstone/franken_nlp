use super::*;
use super::super::{tests as binary, taxonomy_tests::{vector_dataset, vector_replay}};

fn gold() -> Vec<Vec<usize>> { vec![vec![0], vec![1], vec![2], vec![0], vec![2]] }
fn weights() -> Vec<Vec<f64>> {
    vec![vec![0.8, 0.1, 0.1], vec![0.1, 0.8, 0.1], vec![0.1, 0.1, 0.8],
        vec![0.8, 0.1, 0.1], vec![0.1, 0.1, 0.8]]
}
fn report(replay: &NativeReplay) -> Evaluation {
    evaluate(replay, &binary::key(), "scope", binary::options(), Limits::default()).unwrap()
}

#[test]
fn vector_reports_bind_full_native_collections_and_keep_private_rows_out_of_output() {
    for kind in [TaskKind::Multiclass, TaskKind::Multilabel] {
        let dataset = vector_dataset(kind, &gold());
        let replay = vector_replay(&dataset, &weights(), "candidate");
        let evaluation = report(&replay);
        assert_eq!(evaluation.body.task, dataset.manifest().task);
        assert!(evaluation.body.classification.is_none());
        let quality = evaluation.body.label_set.as_ref().unwrap();
        assert_eq!(quality.rows, 5); assert_eq!(quality.exact_match_accuracy, 1.0);
        assert_eq!(quality.macro_f1, Some(1.0));
        assert_eq!(evaluation.body.bindings.metadata.labels, ["alpha", "beta", "gamma"]);
        assert_eq!(evaluation.body.bindings.metadata.positive_label, None);
        assert_eq!(evaluation.body.bindings.observation_collection_format, VECTOR_OBSERVATION_COLLECTION_FORMAT);
        assert_eq!(evaluation.body.bindings.execution_collection_rows, 7);
        assert!(!evaluation.body.activation_authority);
        let text = String::from_utf8(crate::assay::render(&evaluation, Limits::default()).unwrap()).unwrap();
        for private in ["PRIVATE_VECTOR_SOURCE", "vector-test-000", "vector-calibration"] {
            assert!(!text.contains(private));
        }
        assert!(evaluation.body.score_semantics.contains("bootstrap_whole_documents"));
        let expected = if kind == TaskKind::Multiclass { 0.03 } else { 0.02 };
        let Measurement::Measured { metrics } = &evaluation.body.measurement else { panic!("expected measured"); };
        assert!((metrics.brier - expected).abs() < 1e-12);
    }
}

#[test]
fn paired_vector_comparison_uses_document_loss_and_exact_set_accuracy_gates() {
    for kind in [TaskKind::Multiclass, TaskKind::Multilabel] {
        let dataset = vector_dataset(kind, &gold());
        let a = vector_replay(&dataset, &weights(), "candidate");
        let b = vector_replay(&dataset, &weights(), "baseline");
        let mut gates = binary::gates();
        gates.max_accuracy_decrease = 0.0; gates.max_macro_f1_decrease = 0.0;
        gates.reliability.max_brier_increase = 0.0;
        let compared = compare(&a, &b, &binary::key(), "scope", "host", gates,
            binary::options(), Limits::default()).unwrap();
        let Decision::Passed { paired, .. } = &compared.body.decision else { panic!("identical observed quality"); };
        assert_eq!(paired.rows, 5); assert_eq!(paired.equal, 5); assert_eq!(paired.mean_brier_change, 0.0);
        assert_eq!(compared.body.total_model_work, a.model_work.checked_add(b.model_work).unwrap());
        assert_ne!(compared.body.candidate.body.bindings.execution_collection,
            compared.body.baseline.body.bindings.execution_collection);
        let mut worse = weights(); worse[0] = vec![0.1, 0.8, 0.1];
        let worse = vector_replay(&dataset, &worse, "candidate");
        let rejected = compare(&worse, &b, &binary::key(), "scope", "host", gates,
            binary::options(), Limits::default()).unwrap();
        let Decision::Rejected { comparisons, paired } = rejected.body.decision else { panic!("regression must reject"); };
        assert_eq!(paired.candidate_worse, 1);
        let expected = if kind == TaskKind::Multiclass { 0.7 / 5.0 } else { (1.4 / 3.0) / 5.0 };
        assert!((paired.mean_brier_change - expected).abs() < 1e-12);
        assert!(comparisons.iter().any(|gate| gate.name == "accuracy_decrease" && !gate.passed));
    }
}

#[test]
fn mismatched_task_gold_and_full_head_work_are_not_pairable() {
    let dataset = vector_dataset(TaskKind::Multiclass, &gold());
    let original = vector_replay(&dataset, &weights(), "candidate");
    let mut changed_gold = gold(); changed_gold[0] = vec![1];
    let changed_dataset = vector_dataset(TaskKind::Multiclass, &changed_gold);
    let changed = vector_replay(&changed_dataset, &weights(), "baseline");
    assert_eq!(compare(&original, &changed, &binary::key(), "scope", "host", binary::gates(),
        binary::options(), Limits::default()).err(), Some(AssayError::Scope));
    let multi_dataset = vector_dataset(TaskKind::Multilabel, &gold());
    let multi = vector_replay(&multi_dataset, &weights(), "baseline");
    assert_eq!(compare(&original, &multi, &binary::key(), "scope", "host", binary::gates(),
        binary::options(), Limits::default()).err(), Some(AssayError::Scope));
}

#[test]
fn calibration_callbacks_cannot_transform_multiclass_or_independent_label_evidence() {
    for kind in [TaskKind::Multiclass, TaskKind::Multilabel] {
        let dataset = vector_dataset(kind, &gold());
        let replay = vector_replay(&dataset, &weights(), "candidate");
        assert_eq!(evaluate_transformed_with_checkpoint::<AssayError>(&replay, &binary::key(), "scope",
            binary::options(), Limits::default(), &|_| panic!("must refuse before transform"), "not-authorized",
            &mut || Ok(())).err(), Some(AssayError::CalibrationBindingMismatch));
        assert_eq!(compare_transformed_with_checkpoint::<AssayError>(&replay, &replay, &binary::key(), "scope", "host",
            binary::gates(), binary::options(), Limits::default(), &Ok, &Ok, "raw", "raw", &mut || Ok(())).err(),
            Some(AssayError::CalibrationBindingMismatch));
    }
}

#[test]
fn vector_insufficient_data_and_paired_bootstrap_limits_are_explicit() {
    let dataset = vector_dataset(TaskKind::Multilabel, &[vec![], vec![0]]);
    let replay = vector_replay(&dataset, &[vec![0.1; 3], vec![0.8, 0.1, 0.1]], "candidate");
    let evaluation = report(&replay);
    assert!(matches!(evaluation.body.measurement, Measurement::InsufficientData { rows: 2, .. }));
    assert!(evaluation.body.label_set.is_none());
    let compared = compare(&replay, &replay, &binary::key(), "scope", "host", binary::gates(),
        binary::options(), Limits::default()).unwrap();
    assert!(matches!(compared.body.decision, Decision::InsufficientData { rows: 2, .. }));
    let dataset = vector_dataset(TaskKind::Multiclass, &gold()); let replay = vector_replay(&dataset, &weights(), "candidate");
    let options = binary::options();
    let limits = Limits { max_bootstrap_draws: 5 * options.bootstrap.resamples as u64, ..Limits::default() };
    assert!(evaluate(&replay, &binary::key(), "scope", options, limits).is_ok());
    assert_eq!(compare(&replay, &replay, &binary::key(), "scope", "host", binary::gates(), options, limits).err(),
        Some(AssayError::InputLimit));
}
