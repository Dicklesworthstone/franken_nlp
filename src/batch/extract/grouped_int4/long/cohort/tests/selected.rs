//! Scheduling and rejection fixtures only, not a native equality claim.
use super::*;

fn selected(cap: usize) -> Int4ExtractionBatchPlanner {
    planner().with_selected_rows(Int4JsonSparseLimits { max_rows_per_step: cap }).unwrap()
}

#[test]
fn ragged_selected_windows_preserve_complete_documents_and_count_physical_row_calls() {
    let a = source_args(); let vocabulary = vocabulary(); let source = "éAéA      éAéA Alice Bob Carol";
    for cap in [7, 65, NANBEIGE_VOCAB_SIZE] {
        let p = selected(cap); let serial_plan = prepare(&p, source, &a, limits(1));
        let ids: Vec<_> = serial_plan.execution_identities().cloned().collect();
        let serial = serial_plan.execute_with_driver(&ids, Script::new(&a), &mut Continue).unwrap();
        let count = serial.mapped.root().chunk_range().len(); assert!(count > 2);
        for width in [1, 2, 4] {
            let prepared = prepare(&p, source, &a, limits(width));
            let expected = prepared.preflight_metadata();
            let dense = prepare(&planner(), source, &a, limits(width));
            assert_eq!(prepared.cohort_requirements(8192).unwrap(), dense.cohort_requirements(8192).unwrap());
            let result = run(prepared, Scripted::new(&a), &vocabulary).unwrap();
            expected.verify_completed(&result.document).unwrap();
            assert_eq!(result.execution, INT4_EXTRACTION_MAP_SELECTED_COHORT_EXECUTION);
            assert_eq!(without_schedule(&result.document), without_schedule(&serial));
            assert_eq!(result.cohorts.cohorts, count.div_ceil(width) as u64);
            assert_eq!(result.cohorts.chunks, count as u64);
            assert_eq!(result.cohorts.model_work, serial.model_work);
            assert_eq!(result.cohorts.projection_groups, count as u64 * 2 * cap.div_ceil(32) as u64);
            if cap == NANBEIGE_VOCAB_SIZE { assert!(result.cohorts.projection_groups > result.cohorts.group_steps); }
        }
    }
}

#[test]
fn selected_cohorts_refuse_dense_labels_invalid_row_work_and_dense_head_counts() {
    let p = selected(65); let a = source_args(); let vocabulary = vocabulary();
    for axis in [11, 12, 13] {
        let mut script = Scripted::new(&a); script.scalar.corrupt = Some(axis);
        assert!(run(prepare(&p, "Alice Bob Carol", &a, limits(2)), script, &vocabulary).is_err(), "row {axis}");
    }
    for axis in 0..10 {
        let mut script = Scripted::new(&a); script.corrupt_group = Some(axis);
        assert!(run(prepare(&p, "Alice Bob Carol", &a, limits(2)), script, &vocabulary).is_err(), "group {axis}");
    }
    let mut prepared = prepare(&selected(NANBEIGE_VOCAB_SIZE), "Alice Bob Carol", &a, limits(2));
    let text = prepared.chunks.chunks().last().unwrap().text().to_owned();
    *prepared.plans.last_mut().unwrap() = planner().prepare(BatchDocument {
        id: "tail".to_owned(), text, task_args: Some(a) }).unwrap();
    let script = Scripted::new(&source_args()); let calls = Rc::clone(&script.scalar.calls);
    assert!(run(prepared, script, &vocabulary).is_err()); assert_eq!(calls.get(), 0);
}

#[test]
fn selected_cohorts_retain_exact_array_union_and_late_cancellation() {
    let p = selected(65); let vocabulary = vocabulary();
    let a = args(r#"{"type":"array","items":{"type":"integer"},"maxItems":8}"#,
        ExtractionBatchGrounding::Structural);
    let source = "Alice Bob Carol Dora Eve"; let mut l = limits(2);
    l.root_array_union = Some(super::super::super::tests::union_limits());
    let prepared = prepare(&p, source, &a, l); let expected = prepared.preflight_metadata();
    let mut script = Scripted::new(&a); script.scalar.json = Some("[1,2]".to_owned());
    let result = run(prepared, script, &vocabulary).unwrap(); expected.verify_completed(&result.document).unwrap();
    assert_eq!(result.document.root_array_union.as_ref().unwrap().json(), "[1,2]");
    assert_eq!(result.document.root_array_union.as_ref().unwrap().records()[0].origins.len(), expected.chunk_count());

    let mut script = Scripted::new(&a); script.scalar.json = Some("[1,2]".to_owned()); script.scalar.fail_at = Some(3);
    let calls = Rc::clone(&script.scalar.calls);
    let error = run(prepare(&p, source, &a, l), script, &vocabulary).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); assert_eq!(calls.get(), 3);
}
