//! Source-only regressions: fixtures exercise sealed policy/accounting, not
//! real-model fidelity. Native dense/selected draws have their own gate.
use super::*;

fn cap(rows: usize) -> Int4JsonSparseLimits { Int4JsonSparseLimits { max_rows_per_step: rows } }

fn identities(plan: &PreparedSampledSentiment) -> Vec<Vec<u8>> {
    plan.axes.iter().map(|axis| canonjson::canonical_bytes(&axis.identity).unwrap()).collect()
}

#[test]
fn selected_admission_is_distinct_while_semantic_axis_and_item_addresses_are_unchanged() {
    for encoding in [UntrustedEncoding::ByteFallback, UntrustedEncoding::SafeBpeV1(compact_limits())] {
        let dense = encoding_planner(encoding.clone());
        let selected = encoding_planner(encoding).with_selected_rows(cap(97)).unwrap();
        assert_ne!(canonjson::canonical_bytes(dense.execution_identity()).unwrap(),
            canonjson::canonical_bytes(selected.execution_identity()).unwrap());
        for item in [None, Some("record-é"), Some("second-record")] {
            let prepare = |planner: &SampledSentimentPlanner| match item {
                None => planner.plan(&request()),
                Some(id) => planner.plan_for_item_with_control(&request(), id, &mut Continue),
            }.unwrap();
            let a = prepare(&dense); let b = prepare(&selected);
            assert_eq!(identities(&a), identities(&b));
            assert_eq!(a.program.sampling_digest(), b.program.sampling_digest());
            for (a, b) in a.axes.iter().zip(&b.axes) { assert_eq!(a.prompt, b.prompt); }
            assert!(b.verify_identity(a.execution_identity()).is_err());
            assert!(a.verify_identity(b.execution_identity()).is_err());
            b.verify_identity(b.execution_identity()).unwrap();
            assert_eq!(b.selected_rows(), Some(cap(97)));
        }
    }
}

#[test]
fn selected_ceiling_changes_only_projection_work_and_keeps_complete_memory_and_mask_authority() {
    let dense = prepare(&request()).unwrap();
    for rows in [1, 32, 97, NANBEIGE_VOCAB_SIZE] {
        let selected = prepare(&request()).unwrap().with_projection(Some(cap(rows))).unwrap();
        assert_eq!(dense.required_context(), selected.required_context());
        assert_eq!(dense.task_budget(), selected.task_budget());
        assert_eq!(dense.preparation_requirement_bytes(), selected.preparation_requirement_bytes());
        assert_eq!(dense.max_result_bytes(), selected.max_result_bytes());
        assert_eq!(dense.planned_work().forward_positions, selected.planned_work().forward_positions);
        assert_eq!(dense.planned_work().attention_pairs, selected.planned_work().attention_pairs);
        assert_eq!(selected.planned_work().projected_logits,
            (selected.sample_count() * options().max_sample_tokens * rows) as u64);
        let mut total = Int4Work::default();
        for axis in &selected.axes {
            let expected = constrained_int4::sparse::planned_work(32, axis.prompt.len(),
                options().max_sample_tokens, cap(rows)).unwrap();
            assert_eq!(axis.work, expected);
            for _ in 0..options().samples { total = total.checked_add(expected).unwrap(); }
        }
        assert_eq!(selected.planned_work(), total);
        assert!(fits_work(total, dense.planned_work()));
        if rows == NANBEIGE_VOCAB_SIZE { assert_eq!(total, dense.planned_work()); }
    }
    for rows in [0, NANBEIGE_VOCAB_SIZE + 1, usize::MAX] {
        assert!(encoding_planner(UntrustedEncoding::ByteFallback).with_selected_rows(cap(rows)).is_err());
    }
}

fn selected_bundle(plan: &PreparedSampledSentiment, reuse: bool) -> SampledSentimentRun {
    let dense = prepare(&request()).unwrap();
    let mut run = complete_bundle(&dense);
    run.execution = plan.execution_version(reuse).to_owned();
    run.model_work = Int4Work::default();
    for (axis, dimension) in plan.axes.iter().zip(&mut run.dimensions) {
        for (index, sample) in dimension.samples.iter_mut().enumerate() {
            let rows = sample.generated_tokens as u64 * 17;
            sample.model_work = plan.sample_work_for_rows(axis, index, sample.generated_tokens, reuse, rows).unwrap();
            run.model_work = run.model_work.checked_add(sample.model_work).unwrap();
        }
    }
    run
}

#[test]
fn complete_selected_bundles_validate_exact_actual_rows_and_reused_causal_work() {
    let plan = prepare(&request()).unwrap().with_projection(Some(cap(17))).unwrap();
    for reuse in [false, true] {
        let run = selected_bundle(&plan, reuse);
        plan.verify_completed_with_control(&run, COMPLETE_MASK_CHARGE_PER_TOKEN,
            run.mask_node_visit_charge, reuse, &mut Continue).unwrap();
        for error in 0..5 {
            let mut bad = run.clone();
            match error {
                0 => bad.execution = projection::execution(None, reuse).to_owned(),
                1 => bad.dimensions[0].samples[0].model_work.projected_logits += 1,
                2 => bad.dimensions[0].samples[0].model_work.projections.dot_products -= 1,
                3 => bad.dimensions[0].samples[0].mask_node_visit_charge -= 1,
                _ => bad.model_work.forward_positions -= 1,
            }
            assert!(plan.verify_completed_with_control(&bad, COMPLETE_MASK_CHARGE_PER_TOKEN,
                run.mask_node_visit_charge, reuse, &mut Continue).is_err());
        }
        let axis = &plan.axes[0]; let generated = 5;
        for projected in [0, 4, 86, u64::MAX] {
            assert!(plan.sample_work_for_rows(axis, 0, generated, reuse, projected).is_err());
        }
        let work = plan.sample_work_for_rows(axis, 1, generated, reuse, 65).unwrap();
        let (start, count) = if reuse { (axis.prompt.len() - 1, generated) }
            else { (0, axis.prompt.len() + generated - 1) };
        assert_eq!(work, Int4Work::for_sequence(32, start, count, 65).unwrap());
    }
}

#[test]
fn native_selected_finalizer_refuses_dense_tags_and_unreconciled_rows() {
    let plan = prepare(&request()).unwrap().with_projection(Some(cap(97))).unwrap();
    let axis = &plan.axes[0];
    for reuse in [false, true] {
        let mut run = synthetic(&plan, axis, r#"{"score":45,"justification":"Helpful wording."}"#);
        run.execution = projection::native_execution(Some(cap(97)), reuse).to_owned();
        run.model_work = plan.sample_work_for_rows(axis, 1, run.output.token_ids.len(), reuse,
            run.output.token_ids.len() as u64 * 40).unwrap();
        run.output.forward_positions = run.model_work.forward_positions;
        run.output.projected_logits = run.model_work.projected_logits;
        plan.finalize_sample(axis, 1, run.clone(), reuse).unwrap();
        let mut wrong = run.clone(); wrong.execution = projection::native_execution(None, reuse).to_owned();
        assert!(plan.finalize_sample(axis, 1, wrong, reuse).is_err());
        let mut wrong = run.clone(); wrong.output.projected_logits += 1;
        assert!(plan.finalize_sample(axis, 1, wrong, reuse).is_err());
        let mut wrong = run; wrong.model_work.projections.multiply_accumulates -= 1;
        assert!(plan.finalize_sample(axis, 1, wrong, reuse).is_err());
    }
}

#[test]
fn encoding_and_selected_builder_order_preserve_the_same_two_identity_authorities() {
    let encoding = UntrustedEncoding::SafeBpeV1(compact_limits());
    let a = encoding_planner(UntrustedEncoding::ByteFallback).with_selected_rows(cap(97)).unwrap()
        .with_input_encoding(encoding.clone()).unwrap();
    let b = encoding_planner(encoding).with_selected_rows(cap(97)).unwrap();
    assert_eq!(canonjson::canonical_bytes(a.execution_identity()).unwrap(),
        canonjson::canonical_bytes(b.execution_identity()).unwrap());
    let a = a.plan(&request()).unwrap(); let b = b.plan(&request()).unwrap();
    assert_eq!(identities(&a), identities(&b));
    assert_eq!(a.planned_work(), b.planned_work());
    let different_cap = encoding_planner(UntrustedEncoding::SafeBpeV1(compact_limits()))
        .with_selected_rows(cap(98)).unwrap().plan(&request()).unwrap();
    assert_eq!(identities(&a), identities(&different_cap));
    assert!(a.verify_identity(different_cap.execution_identity()).is_err());
}
