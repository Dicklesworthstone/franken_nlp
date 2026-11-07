//! Packed envelopes keep each original source/schema finalizer authoritative.
use super::*;

fn limits() -> Int4PrefillLimits {
    Int4PrefillLimits { max_batch_rows: 64,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(64).unwrap() }
}
fn packed_raw(requests: &[Int4ExtractCohortRequest<'_>], json: &[&str]) -> Int4JsonCohortRun {
    let mut run = raw(requests, json);
    run.execution = INT4_JSON_PACKED_COHORT_EXECUTION.to_owned();
    run.group_steps = run.model_work.forward_positions.div_ceil(limits().max_batch_rows as u64)
        .max(run.projection_groups);
    run
}

#[test]
fn packed_source_rows_preserve_exact_values_order_occurrences_and_work() {
    let plans = [source_plan("é Alice Alice", &[]), source_plan("Bob", &[]),
        plan(r#"{"type":"boolean"}"#, 127)];
    let requests = requests(&plans); let json = ["\"Alice\"", "\"Bob\"", "true"];
    let ordinary = finish(&requests, raw(&requests, &json), 1 << 20).unwrap();
    let raw = packed_raw(&requests, &json);
    assert!(finish(&requests, raw.clone(), 1 << 20).is_err());
    let packed = finish_scheduled(&requests, raw, 1 << 20, Some(limits())).unwrap();
    assert_eq!(packed.execution, INT4_EXTRACT_PACKED_COHORT_EXECUTION);
    assert_eq!(packed.sequences, ordinary.sequences);
    assert_eq!(packed.planned_work, ordinary.planned_work); assert_eq!(packed.model_work, ordinary.model_work);
    assert!(packed.group_steps < ordinary.group_steps);
    let evidence = &packed.sequences[0].result.source_fields[0];
    assert_eq!(evidence.occurrence, SourceOccurrence::Ambiguous);
    assert_eq!(evidence.spans[0].byte_start, 3); assert_eq!(evidence.spans[0].scalar_start, 2);
}

#[test]
fn packed_marker_work_and_physical_limits_cannot_bypass_independent_finalization() {
    let plans = [source_plan("Alice", &[]), source_plan("Bob", &[])]; let requests = requests(&plans);
    for axis in 0..10 {
        let mut run = packed_raw(&requests, &["\"Alice\"", "\"Bob\""]);
        match axis {
            0 => run.execution = INT4_JSON_COHORT_EXECUTION.to_owned(),
            1 => run.planned_work.projected_logits += 1, 2 => run.model_work.attention_pairs += 1,
            3 => run.group_steps = 0, 4 => run.group_steps = run.model_work.forward_positions + 1,
            5 => run.group_steps = run.model_work.forward_positions.div_ceil(64) - 1,
            6 => run.projection_groups = 1, 7 => run.projection_groups = 5,
            8 => run.sequences.swap(0, 1), _ => run.sequences[1].group_width = 32,
        }
        assert!(finish_scheduled(&requests, run, 1 << 20, Some(limits())).is_err(), "axis {axis}");
    }
    let bad_source = packed_raw(&requests, &["\"Alice\"", "\"Mallory\""]);
    assert!(finish_scheduled(&requests, bad_source, 1 << 20, Some(limits())).is_err());
    let mut insufficient = limits(); insufficient.max_extra_scratch_bytes -= 1;
    assert!(finish_scheduled(&requests, packed_raw(&requests, &["\"Alice\"", "\"Bob\""]),
        1 << 20, Some(insufficient)).is_err());
}

#[test]
fn packed_selection_keeps_the_full_projection_contract_and_exact_envelope_bound() {
    let plans = [plan(r#"{"type":"boolean"}"#, 127)
        .with_selected_rows(selected_head::Int4JsonSparseLimits { max_rows_per_step: NANBEIGE_VOCAB_SIZE }).unwrap()];
    let selected = requests(&plans);
    assert!(finish_scheduled(&selected, packed_raw(&selected, &["true"]), 1 << 20, Some(limits())).is_err());
    let plans = [source_plan("Alice Alice", &[])]; let requests = requests(&plans);
    let output = finish_scheduled(&requests, packed_raw(&requests, &["\"Alice\""]), 1 << 20, Some(limits())).unwrap();
    let cap = canonjson::canonical_bytes(&output).unwrap().len() as u64;
    assert!(finish_scheduled(&requests, packed_raw(&requests, &["\"Alice\""]), cap, Some(limits())).is_ok());
    assert!(finish_scheduled(&requests, packed_raw(&requests, &["\"Alice\""]), cap - 1, Some(limits())).is_err());
}
