//! Closed scripted contract checks; these are not native sparse/full receipts.
use super::*;

fn selected(cap: usize) -> Int4ExtractionBatchPlanner {
    planner().with_selected_rows(Int4JsonSparseLimits { max_rows_per_step: cap }).unwrap()
}

#[test]
fn complete_documents_bind_the_exact_head_policy_even_when_work_matches_the_full_head() {
    let a = source_args(); let source = "é上海 and 上海";
    let full = prepare(&planner(), source, &a, limits());
    let full_work = full.preflight_metadata().planned_work();
    for cap in [1, 7, 65, NANBEIGE_VOCAB_SIZE] {
        let p = selected(cap); let prepared = prepare(&p, source, &a, limits());
        let expected = prepared.preflight_metadata();
        assert_eq!(expected.selected_rows(), p.selected_rows());
        assert_eq!(expected.execution_version(), INT4_EXTRACTION_MAP_SELECTED_EXECUTION);
        assert_eq!(expected.planned_work().forward_positions, full_work.forward_positions);
        assert_eq!(expected.planned_work().attention_pairs, full_work.attention_pairs);
        assert_eq!(expected.reserved_mask_visits(), full.preflight_metadata().reserved_mask_visits());
        assert_eq!(expected.planned_work().projected_logits,
            expected.chunk_count() as u64 * u64::from(a.budget.max_output_tokens) * cap as u64);
        for (dense, sparse) in full.execution_identities().zip(prepared.execution_identities()) {
            assert_eq!(dense.prompt_digest, sparse.prompt_digest);
            assert_eq!(dense.schema_digest, sparse.schema_digest);
            assert_ne!(dense.decision_policy_digest, sparse.decision_policy_digest);
        }
        if cap == NANBEIGE_VOCAB_SIZE { assert_eq!(expected.planned_work(), full_work); }
        else { assert!(expected.planned_work().projections.dot_products < full_work.projections.dot_products); }
        let result = run(prepared, Script::new(&a)).unwrap();
        expected.verify_completed(&result).unwrap();
        assert_eq!(result.max_selected_rows, Some(cap));
        assert_eq!(result.model_work.projected_logits, expected.chunk_count() as u64 * 2 * cap as u64);
        assert!(full.preflight_metadata().verify_completed(&result).is_err());
        let mut wrong = result; wrong.max_selected_rows = None;
        assert!(expected.verify_completed(&wrong).is_err());
    }
    let result = run(full, Script::new(&a)).unwrap();
    assert_eq!(result.execution, INT4_EXTRACTION_MAP_EXECUTION);
    assert!(serde_json::to_value(result).unwrap().get("max_selected_rows").is_none());
}

#[test]
fn selected_maps_precharge_every_work_axis_and_mask_visit_without_renewal() {
    let p = selected(65); let a = source_args(); let source = "Alice Bob Carol";
    let expected = prepare(&p, source, &a, limits()).preflight_metadata();
    let mut exact = limits(); exact.mapping.max_model_work = expected.planned_work();
    exact.mapping.max_mask_visits = expected.reserved_mask_visits();
    let result = run(prepare(&p, source, &a, exact), Script::new(&a)).unwrap();
    expected.verify_completed(&result).unwrap();
    for axis in 0..6 {
        let mut short = exact;
        match axis {
            0 => short.mapping.max_model_work.forward_positions -= 1,
            1 => short.mapping.max_model_work.projected_logits -= 1,
            2 => short.mapping.max_model_work.attention_pairs -= 1,
            3 => short.mapping.max_model_work.projections.dot_products -= 1,
            4 => short.mapping.max_model_work.projections.multiply_accumulates -= 1,
            _ => short.mapping.max_mask_visits -= 1,
        }
        assert!(p.plan_document_with_control(source, &a, short, &mut Continue).is_err(), "axis {axis}");
    }
    for axis in 0..6 {
        let mut short = limits();
        match axis {
            0 => short.verification.max_fields = 1,
            1 => short.verification.max_matches = 1,
            2 => short.verification.max_scan_steps = 1,
            3 => short.mapping.reduction.max_live_value_bytes = 1,
            4 => short.mapping.reduction.max_total_value_bytes = 1,
            _ => short.mapping.reduction.max_result_bytes = 1,
        }
        match p.plan_document_with_control(source, &a, short, &mut Continue) {
            Ok(prepared) => assert!(run(prepared, Script::new(&a)).is_err(), "axis {axis}"),
            Err(error) => assert!(matches!(error, Int4ExtractionMapError::InvalidLimits)),
        }
    }
}

#[test]
fn a_mixed_final_chunk_or_dense_admission_is_refused_before_the_first_forward() {
    let a = source_args(); let source = "Alice Bob Carol";
    let dense = planner(); let p = selected(NANBEIGE_VOCAB_SIZE);
    let full = prepare(&dense, source, &a, limits());
    let dense_ids: Vec<_> = full.execution_identities().cloned().collect();
    let prepared = prepare(&p, source, &a, limits());
    let script = Script::new(&a); let calls = Rc::clone(&script.calls);
    assert!(prepared.execute_with_driver(&dense_ids, script, &mut Continue).is_err());
    assert_eq!(calls.get(), 0);

    for replacement in [dense, selected(NANBEIGE_VOCAB_SIZE - 1)] {
        let mut prepared = prepare(&p, source, &a, limits());
        let text = prepared.chunks.chunks().last().unwrap().text().to_owned();
        *prepared.plans.last_mut().unwrap() = replacement.prepare(BatchDocument {
            id: "tail".to_owned(), text, task_args: Some(a.clone()) }).unwrap();
        let script = Script::new(&a); let calls = Rc::clone(&script.calls);
        assert!(run(prepared, script).is_err()); assert_eq!(calls.get(), 0);
    }
}

#[test]
fn selected_late_rows_cannot_relabel_skip_eos_projection_or_exceed_the_bound() {
    let p = selected(65); let a = source_args();
    for axis in 0..14 {
        let prepared = prepare(&p, "Alice Bob Carol", &a, limits());
        let mut script = Script::new(&a); script.corrupt = Some(axis);
        let calls = Rc::clone(&script.calls);
        assert!(run(prepared, script).is_err(), "axis {axis}"); assert_eq!(calls.get(), 2);
    }
}

#[test]
fn selected_array_union_preserves_exact_unicode_origins_and_binds_the_head_policy() {
    let p = selected(65); let a = args(r#"{"type":"array","items":{"type":"string","x-fnlp-source":"verbatim"},"maxItems":8}"#,
        ExtractionBatchGrounding::SourceMembership);
    let source = "éAéAéAéA"; let mut l = limits(); l.root_array_union = Some(union_limits());
    let prepared = prepare(&p, source, &a, l); let expected = prepared.preflight_metadata();
    let full = prepare(&planner(), source, &a, l).preflight_metadata();
    assert_ne!(expected.root_array_binding, full.root_array_binding);
    let mut script = Script::new(&a); script.json = Some(r#"["é"]"#.to_owned());
    let result = run(prepared, script).unwrap(); expected.verify_completed(&result).unwrap();
    let union = result.root_array_union.as_ref().unwrap();
    assert_eq!(union.json(), r#"["é"]"#); assert_eq!(union.records()[0].origins.len(), 2);
    assert_eq!(result.verification_used.matches, 4);
    for chunk in result.mapped.root().value().chunks() {
        for span in &chunk.original_fields[0].spans {
            assert_eq!(&source[span.byte_start..span.byte_end], "é");
            assert_eq!(source[..span.byte_start].chars().count(), span.scalar_start);
        }
    }
}

#[test]
fn selected_native_and_final_checkpoint_cancellation_keep_the_original_cause() {
    struct Count { calls: usize, stop: Option<usize> }
    impl DecodeStepControl for Count {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1; (self.stop == Some(self.calls)).then_some(DecodeCancellationKind::CostBudget)
        }
    }
    let p = selected(65); let a = source_args(); let source = "Alice Bob Carol";
    let mut script = Script::new(&a); script.fail_at = Some(2);
    assert_eq!(run(prepare(&p, source, &a, limits()), script).err().unwrap().cancellation(),
        Some(DecodeCancellationKind::Deadline));
    let prepared = prepare(&p, source, &a, limits());
    let ids: Vec<_> = prepared.execution_identities().cloned().collect();
    let mut count = Count { calls: 0, stop: None };
    prepared.execute_with_driver(&ids, Script::new(&a), &mut count).unwrap();
    let error = prepare(&p, source, &a, limits()).execute_with_driver(&ids, Script::new(&a),
        &mut Count { calls: 0, stop: Some(count.calls) }).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::CostBudget));
    assert!(!format!("{error:?}").contains(source));
}
