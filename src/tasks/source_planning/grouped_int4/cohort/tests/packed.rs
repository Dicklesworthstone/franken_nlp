//! The packed outer envelope still runs all four sealed source finalizers.
use super::*;

fn limits() -> Int4PrefillLimits {
    Int4PrefillLimits { max_batch_rows: 64,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(64).unwrap() }
}
fn packed_raw(plans: &[PreparedInt4SourceTask], fixtures: &[(SourceTaskRequest, &str, &str)]) -> Int4ExtractCohortRun {
    let mut run = raw(plans, fixtures);
    run.execution = INT4_EXTRACT_PACKED_COHORT_EXECUTION.to_owned();
    run.group_steps = run.model_work.forward_positions.div_ceil(limits().max_batch_rows as u64)
        .max(run.projection_groups);
    run
}

#[test]
fn all_source_tasks_keep_complete_semantics_with_distinct_packed_physical_counts() {
    let planner = planner(); let fixtures = fixtures();
    let plans: Vec<_> = fixtures.iter().map(|(request, _, _)| prepare(&planner, request)).collect();
    let requests = requests(&plans);
    let ordinary = finish(&requests, raw(&plans, &fixtures), 8 << 20).unwrap();
    let raw = packed_raw(&plans, &fixtures);
    assert!(finish(&requests, raw.clone(), 8 << 20).is_err());
    let packed = finish_scheduled(&requests, raw, 8 << 20, Some(limits())).unwrap();
    assert_eq!(packed.execution, INT4_SOURCE_PACKED_COHORT_EXECUTION);
    assert_eq!(packed.sequences, ordinary.sequences);
    assert_eq!(packed.planned_work, ordinary.planned_work); assert_eq!(packed.model_work, ordinary.model_work);
    assert!(packed.group_steps < ordinary.group_steps);
    for plan in &plans { plan.verify_identity(plan.execution_identity()).unwrap(); }
    let cap = canonjson::canonical_bytes(&packed).unwrap().len() as u64;
    assert!(finish_scheduled(&requests, packed_raw(&plans, &fixtures), cap, Some(limits())).is_ok());
    assert!(finish_scheduled(&requests, packed_raw(&plans, &fixtures), cap - 1, Some(limits())).is_err());
}

#[test]
fn packed_source_finalizers_reject_work_drift_impossible_counts_and_late_semantic_failure() {
    let planner = planner(); let mut fixtures = fixtures();
    let plans: Vec<_> = fixtures.iter().map(|(request, _, _)| prepare(&planner, request)).collect();
    let requests = requests(&plans);
    for axis in 0..9 {
        let mut run = packed_raw(&plans, &fixtures);
        match axis {
            0 => run.execution = INT4_EXTRACT_COHORT_EXECUTION.to_owned(),
            1 => run.planned_work.projected_logits += 1, 2 => run.model_work.attention_pairs += 1,
            3 => run.group_steps = 0, 4 => run.group_steps = run.model_work.forward_positions + 1,
            5 => run.group_steps = run.model_work.forward_positions.div_ceil(64) - 1,
            6 => run.projection_groups = 1, 7 => run.projection_groups = 9,
            _ => run.sequences.swap(0, 1),
        }
        assert!(finish_scheduled(&requests, run, 8 << 20, Some(limits())).is_err(), "axis {axis}");
    }
    fixtures[2].2 = r#"[{"citations":[],"text":"A claim."}]"#;
    assert!(finish_scheduled(&requests, packed_raw(&plans, &fixtures), 8 << 20, Some(limits())).is_err());
}

#[test]
fn packed_source_execution_never_reinterprets_selected_rows_as_full_head_or_repairs_identity() {
    use crate::native_engine::{constrained_int4::sparse::Int4JsonSparseLimits, lmhead::NANBEIGE_VOCAB_SIZE};
    let planner = planner();
    let fixtures = vec![(ner("Alice"), "Alice", "[]")];
    let plans = vec![prepare(&planner, &fixtures[0].0)
        .with_selected_rows(Int4JsonSparseLimits { max_rows_per_step: NANBEIGE_VOCAB_SIZE }).unwrap()];
    assert!(finish_scheduled(&requests(&plans), packed_raw(&plans, &fixtures), 8 << 20, Some(limits())).is_err());
    let plans = vec![prepare(&planner, &fixtures[0].0)]; let mut requests = requests(&plans);
    let mut wrong = plans[0].execution_identity().clone(); wrong.prompt_digest = Sha256Digest::of_bytes(b"foreign source");
    requests[0].admitted_identity = &wrong;
    assert!(finish_scheduled(&requests, packed_raw(&plans, &fixtures), 8 << 20, Some(limits())).is_err());
}

#[test]
fn packed_selected_source_cohorts_keep_caps_results_and_chunk_counts_independent() {
    use crate::native_engine::{constrained_int4::sparse::Int4JsonSparseLimits, lmhead::NANBEIGE_VOCAB_SIZE};
    let planner = planner(); let fixtures = fixtures();
    let plans: Vec<_> = fixtures.iter().zip([7, 33, 65, NANBEIGE_VOCAB_SIZE])
        .map(|((request, _, _), max_rows_per_step)| prepare(&planner, request)
            .with_selected_rows(Int4JsonSparseLimits { max_rows_per_step }).unwrap()).collect();
    let requests = requests(&plans);
    let ordinary = finish(&requests, selected_raw(&plans, &fixtures), 8 << 20).unwrap();
    let mut raw = selected_raw(&plans, &fixtures);
    raw.execution = INT4_EXTRACT_SELECTED_PACKED_COHORT_EXECUTION.to_owned();
    raw.group_steps = raw.model_work.forward_positions.div_ceil(limits().max_batch_rows as u64).max(2);
    assert!(raw.projection_groups > raw.group_steps);
    assert!(finish(&requests, raw.clone(), 8 << 20).is_err());
    let packed = finish_scheduled(&requests, raw.clone(), 8 << 20, Some(limits())).unwrap();
    assert_eq!(packed.execution, INT4_SOURCE_SELECTED_PACKED_COHORT_EXECUTION);
    assert_eq!(packed.sequences, ordinary.sequences);
    assert_eq!(packed.planned_work, ordinary.planned_work);
    assert_eq!(packed.model_work, ordinary.model_work);
    assert_eq!(packed.projection_groups, ordinary.projection_groups);
    for axis in 0..4 {
        let mut changed = raw.clone();
        match axis { 0 => changed.group_steps = 1,
            1 => changed.group_steps = changed.model_work.forward_positions + 1,
            2 => changed.execution = INT4_EXTRACT_PACKED_COHORT_EXECUTION.to_owned(),
            _ => changed.projection_groups = 0 }
        assert!(finish_scheduled(&requests, changed, 8 << 20, Some(limits())).is_err(), "axis {axis}");
    }
    let mut wrong = plans[3].execution_identity().clone();
    wrong.schema_digest = Sha256Digest::of_bytes(b"another independently sealed schema");
    let mut requests = requests; requests[3].admitted_identity = &wrong;
    assert!(finish_scheduled(&requests, raw, 8 << 20, Some(limits())).is_err());
}
