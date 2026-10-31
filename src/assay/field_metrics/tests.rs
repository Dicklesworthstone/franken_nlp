use super::*;
use crate::{assay::BootstrapOptions, validation::parse_json};

fn counts(gold: &str, predicted: &str) -> FieldCounts {
    FieldCounts::observe_with_checkpoint::<AssayError>(&parse_json(gold).unwrap(), &parse_json(predicted).unwrap(),
        4096, 65_536, 128, &mut || Ok(())).unwrap()
}
fn options() -> EvalOptions {
    EvalOptions { minimum_samples: 5, bootstrap: BootstrapOptions { resamples: 32, confidence_level: 0.9, seed: 47 } }
}
fn observations(counts: &[FieldCounts]) -> Vec<Observation<'_>> {
    counts.iter().enumerate().map(|(index, counts)| Observation { schema_id: if index % 2 == 0 { "a" } else { "b" },
        language: if index % 2 == 0 { "en" } else { "zh" }, counts }).collect()
}
fn measure(counts: &[FieldCounts]) -> Measurement {
    measure_with_checkpoint::<AssayError>(&["a", "b", "unused"], &observations(counts), options(), Limits::default(), &mut || Ok(())).unwrap()
}

#[test]
fn exact_decimal_equality_never_rounds_large_integers() {
    assert!(counts("{\"x\":1.00,\"zero\":-0e20}", "{\"zero\":0,\"x\":1e0}").exact_match());
    let large = counts("12345678901234567890123456789012345678", "12345678901234567890123456789012345679");
    assert_eq!(large, FieldCounts { true_positive: 0, false_positive: 1, false_negative: 1 });
    assert!(counts("1e-308", "0.1e-307").exact_match());
}

#[test]
fn nested_fields_align_keys_and_array_positions_without_value_multisets() {
    let value = counts(r#"{"a":{"v":2,"n":null},"b":[1,2],"c":[]}"#,
        r#"{"c":[],"b":[2,1],"a":{"v":2,"n":null}}"#);
    assert_eq!(value, FieldCounts { true_positive: 3, false_positive: 2, false_negative: 2 });
    assert_eq!(value.scores().f1, Some(0.6));
    assert_eq!(counts(r#"{"a":1,"b":2}"#, r#"{"a":2,"b":1}"#).true_positive, 0);
}

#[test]
fn null_missing_empty_and_container_kinds_have_distinct_semantics() {
    for (gold, predicted) in [("null", "{}"), ("[]", "{}"), (r#"{"x":null}"#, "{}"),
        (r#"{"0":"x"}"#, r#"["x"]"#), ("1", "\"1\""), ("false", "0"), ("[[]]", "[]")] {
        let value = counts(gold, predicted);
        assert_eq!(value, FieldCounts { true_positive: 0, false_positive: 1, false_negative: 1 }, "{gold} vs {predicted}");
    }
    for value in ["null", "{}", "[]", "[[]]", "{\"\":{}}"] {
        assert_eq!(counts(value, value), FieldCounts { true_positive: 1, false_positive: 0, false_negative: 0 });
    }
}

#[test]
fn unmatched_subtrees_count_each_terminal_once() {
    let value = counts(r#"{"x":[1,2,3],"z":7}"#, r#"{"x":{"a":1,"b":2},"z":7,"new":null}"#);
    assert_eq!(value, FieldCounts { true_positive: 1, false_positive: 3, false_negative: 3 });
    assert_eq!(counts("[1,2,3]", "[1]"), FieldCounts { true_positive: 1, false_positive: 0, false_negative: 2 });
}

#[test]
fn pointer_looking_keys_are_literal_and_never_collide_with_nested_paths() {
    let value = counts(r#"{"a/b":1,"a":{"b":2},"~":3}"#, r#"{"a/b":2,"a":{"b":1},"~":3}"#);
    assert_eq!(value.true_positive, 1); assert_eq!(value.errors(), 4);
}

#[test]
fn field_node_and_depth_caps_cover_both_matching_and_unmatched_sides() {
    let pair = (parse_json("[[1,2]]").unwrap(), parse_json("[[1,2]]").unwrap());
    for (fields, nodes, depth) in [(1, 10, 10), (2, 3, 10), (2, 4, 1)] {
        assert_eq!(FieldCounts::observe_with_checkpoint::<AssayError>(&pair.0, &pair.1, fields, nodes, depth,
            &mut || Ok(())).err(), Some(AssayError::InputLimit));
        assert_eq!(count_with_checkpoint::<AssayError>(&pair.0, fields, nodes, depth,
            &mut || Ok(())).err(), Some(AssayError::InputLimit));
    }
    assert_eq!(count_with_checkpoint::<AssayError>(&pair.0, 2, 4, 2, &mut || Ok(())).unwrap(), 2);
    let short = parse_json("{}").unwrap();
    assert!(FieldCounts::observe_with_checkpoint::<AssayError>(&short, &pair.0, 1, 10, 10, &mut || Ok(())).is_err());
    assert!(count_with_checkpoint::<AssayError>(&short, 0, 10, 10, &mut || Ok(())).is_err());
}

#[test]
fn micro_f1_and_exact_document_accuracy_weight_different_units() {
    let many = counts("[1,2,3,4,5,6,7,8,9]", "[1,2,3,4,5,6,7,8,9]");
    let bad = counts("1", "2");
    let Measurement::Measured { metrics } = measure(&[many, bad, bad, bad, bad]) else { panic!("measured"); };
    assert_eq!(metrics.fields.support, 13); assert_eq!(metrics.fields.f1, Some(9.0 / 13.0));
    assert_eq!(metrics.exact_documents, 1); assert_eq!(metrics.document_accuracy, 0.2);
    assert_eq!(metrics.by_schema.iter().map(|row| row.fields.support).sum::<usize>(), 13);
    let absent = metrics.by_schema.iter().find(|row| row.schema_id == "unused").unwrap();
    assert_eq!(absent.rows, 0); assert_eq!(absent.fields.f1, None); assert_eq!(absent.document_accuracy, None);
    assert_eq!(metrics.by_language.iter().map(|row| row.rows).sum::<usize>(), 5);
}

#[test]
fn paired_draws_are_correlated_and_marginals_match_standalone() {
    let good = counts("1", "1"); let bad = counts("1", "2");
    let current = [good, bad, good, bad, good]; let prior = [bad, bad, good, bad, good];
    let (a, b, paired) = paired_with_checkpoint::<AssayError>(&["a", "b", "unused"], &observations(&current),
        &observations(&prior), options(), Limits::default(), &mut || Ok(())).unwrap();
    assert_eq!(a, measure(&current)); assert_eq!(b, measure(&prior));
    let paired = paired.unwrap(); assert_eq!(paired.candidate_better, 1); assert_eq!(paired.candidate_worse, 0);
    assert_eq!(paired.mean_field_error_change, -0.4);
    let (_, _, same) = paired_with_checkpoint::<AssayError>(&["a", "b"], &observations(&current),
        &observations(&current), options(), Limits::default(), &mut || Ok(())).unwrap();
    let same = same.unwrap();
    for interval in [same.field_precision_change_interval, same.field_recall_change_interval,
        same.field_f1_change_interval, Some(same.document_accuracy_change_interval)] {
        let interval = interval.unwrap(); assert_eq!((interval.lower, interval.upper), (0.0, 0.0));
    }
}

#[test]
fn paired_scope_and_two_arm_cost_are_checked_before_resampling() {
    let values = [counts("1", "1"); 5]; let rows = observations(&values);
    let mut changed = observations(&values); changed[0].schema_id = "b";
    assert_eq!(paired_with_checkpoint::<AssayError>(&["a", "b"], &rows, &changed, options(), Limits::default(),
        &mut || Ok(())).err(), Some(AssayError::Scope));
    let limits = Limits { max_bootstrap_draws: 5 * options().bootstrap.resamples as u64, ..Limits::default() };
    assert!(measure_with_checkpoint::<AssayError>(&["a", "b"], &rows, options(), limits, &mut || Ok(())).is_ok());
    assert_eq!(paired_with_checkpoint::<AssayError>(&["a", "b"], &rows, &rows, options(), limits,
        &mut || Ok(())).err(), Some(AssayError::InputLimit));
    assert!(matches!(measure(&values[..4]), Measurement::InsufficientData { rows: 4, .. }));
}

#[derive(Debug, PartialEq)]
enum Stop { Cancelled, Assay(AssayError) }
impl From<AssayError> for Stop { fn from(value: AssayError) -> Self { Self::Assay(value) } }
#[test]
fn deep_scoring_and_aggregation_preserve_typed_cancellation() {
    let large = parse_json(&format!("[{}]", vec!["1"; 256].join(","))).unwrap(); let mut calls = 0;
    let error = FieldCounts::observe_with_checkpoint::<Stop>(&large, &large, 4096, 65_536, 128, &mut || {
        calls += 1; if calls == 5 { Err(Stop::Cancelled) } else { Ok(()) }
    }).unwrap_err();
    assert_eq!(error, Stop::Cancelled);
    let values = [counts("1", "1"); 5];
    assert_eq!(measure_with_checkpoint::<Stop>(&["a", "b"], &observations(&values), options(), Limits::default(),
        &mut || Err(Stop::Cancelled)).err(), Some(Stop::Cancelled));
}
