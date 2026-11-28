//! Model-free policy and accounting checks. Synthetic receipts do not certify
//! neural parity; the native decoder separately checks the complete legal set.
use super::*;

fn cap(rows: usize) -> Int8JsonSparseLimits { Int8JsonSparseLimits { max_rows_per_step: rows } }

fn planner(encoding: UntrustedEncoding) -> Int8SampledSentimentPlanner {
    Int8SampledSentimentPlanner::for_model_with_encoding(&facts(), options(), budget(),
        SampledSentimentLimits::default(), encoding).unwrap()
}

fn compact() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 32_768, max_output_tokens: 32_768,
        max_pair_checks: 1 << 22, max_pair_bytes: 1 << 26 })
}

#[test]
fn selected_admission_binds_the_cap_without_renumbering_axis_or_item_draws() {
    for encoding in [UntrustedEncoding::ByteFallback, compact()] {
        let dense = planner(encoding.clone());
        let selected = planner(encoding).with_selected_rows(cap(97)).unwrap();
        assert_ne!(dense.execution_identity(), selected.execution_identity());
        assert_eq!(selected.selected_rows(), Some(cap(97)));
        for item in [None, Some("record-é"), Some("another-record")] {
            let prepare = |planner: &Int8SampledSentimentPlanner| match item {
                None => planner.plan(&request()),
                Some(id) => planner.plan_for_item_with_control(&request(), id, &mut Continue),
            }.unwrap();
            let a = prepare(&dense); let b = prepare(&selected);
            assert_eq!(a.inner.program.sampling_digest(), b.inner.program.sampling_digest());
            assert_eq!(a.encoding_work(), b.encoding_work());
            for (a, b) in a.inner.axes.iter().zip(&b.inner.axes) {
                assert_eq!(a.axis, b.axis);
                assert_eq!(a.prompt, b.prompt);
                assert_eq!(a.identity, b.identity);
            }
            assert_ne!(a.execution_identity(), b.execution_identity());
            assert!(a.verify_identity(b.execution_identity()).is_err());
            assert!(b.verify_identity(a.execution_identity()).is_err());
            b.verify_identity(b.execution_identity()).unwrap();
            assert_eq!(b.selected_rows(), Some(cap(97)));
        }
        let first = selected.plan_for_item_with_control(&request(), "first", &mut Continue).unwrap();
        let second = selected.plan_for_item_with_control(&request(), "second", &mut Continue).unwrap();
        assert_ne!(first.inner.axes[0].identity, second.inner.axes[0].identity);
    }
}

#[test]
fn selected_policy_setter_order_preserves_encoding_and_axis_order_is_irrelevant() {
    let a = planner(UntrustedEncoding::ByteFallback).with_selected_rows(cap(97)).unwrap()
        .with_input_encoding(compact()).unwrap();
    let b = planner(compact()).with_selected_rows(cap(97)).unwrap()
        .with_selected_rows(cap(97)).unwrap();
    assert_eq!(a.execution_identity(), b.execution_identity());
    let planned = a.plan(&request()).unwrap();
    let mut reversed = request(); reversed.axes.reverse();
    let reordered = b.plan(&reversed).unwrap();
    assert_eq!(planned.execution_identity(), reordered.execution_identity());
    assert_eq!(planned.planned_work(), reordered.planned_work());
    for (a, b) in planned.inner.axes.iter().zip(&reordered.inner.axes) {
        assert_eq!(a.identity, b.identity); assert_eq!(a.prompt, b.prompt);
    }
    let changed = planner(compact()).with_selected_rows(cap(98)).unwrap().plan(&request()).unwrap();
    assert_ne!(planned.execution_identity(), changed.execution_identity());
    for (a, b) in planned.inner.axes.iter().zip(&changed.inner.axes) { assert_eq!(a.identity, b.identity); }
}

#[test]
fn selected_planning_prices_every_cold_sample_without_discounting_memory_or_masks() {
    let dense = prepare();
    for rows in [1, 32, 97, NANBEIGE_VOCAB_SIZE] {
        let selected = prepare().with_projection(Some(cap(rows))).unwrap();
        assert_eq!(selected.required_context(), dense.required_context());
        assert_eq!(selected.task_budget(), dense.task_budget());
        assert_eq!(selected.preparation_requirement_bytes(), dense.preparation_requirement_bytes());
        assert_eq!(selected.max_result_bytes(), dense.max_result_bytes());
        assert_eq!(selected.max_output_tokens(), dense.max_output_tokens());
        assert_eq!(selected.reuse_sampler_requirement_bytes().unwrap(), dense.reuse_sampler_requirement_bytes().unwrap());
        let work = selected.planned_work();
        assert_eq!(work.forward_positions, dense.planned_work().forward_positions);
        assert_eq!(work.attention_pairs, dense.planned_work().attention_pairs);
        assert_eq!(work.projected_logits, selected.max_output_tokens() * rows as u64);
        let mut total = Int8Work::default();
        for axis in &selected.inner.axes {
            let expected = constrained_int8::sparse::planned_work(axis.prompt.len(),
                options().max_sample_tokens, cap(rows)).unwrap();
            assert_eq!(axis.work, expected);
            for _ in 0..options().samples { total = total.checked_add(expected).unwrap(); }
        }
        assert_eq!(total, work);
        assert!(fits_work(work, dense.planned_work()));
        if rows == NANBEIGE_VOCAB_SIZE { assert_eq!(work, dense.planned_work()); }
        let budget = admission(&selected);
        let sampler = json_sampling_workspace_bytes().unwrap();
        assert_eq!(selected.preflight_bounds(MASK_CHARGE, budget, sampler).unwrap(),
            MASK_CHARGE * options().max_sample_tokens as u64);
        assert!(selected.preflight_bounds(MASK_CHARGE, budget, sampler - 1).is_err());
        for field in 0..8 {
            let mut bad = budget;
            match field {
                0 => bad.native.max_forward_positions -= 1,
                1 => bad.native.max_attention_pairs -= 1,
                2 => bad.native.max_projection_work.dot_products -= 1,
                3 => bad.native.max_projection_work.multiply_accumulates -= 1,
                4 => bad.json.max_projected_logits -= 1,
                5 => bad.json.max_total_mask_node_visits -= 1,
                6 => bad.json.max_kv_bytes = selected.required_context() as u64 * KV_BYTES_PER_TOKEN as u64 - 1,
                _ => bad.json.mask_limits.checkpoint_interval_nodes = 0,
            }
            assert!(selected.preflight_bounds(MASK_CHARGE, bad, sampler).is_err(), "field {field}");
        }
    }
    for rows in [0, NANBEIGE_VOCAB_SIZE + 1, usize::MAX] {
        assert!(planner(UntrustedEncoding::ByteFallback).with_selected_rows(cap(rows)).is_err());
    }
}

fn admission(plan: &PreparedInt8SampledSentiment) -> Int8JsonBudget {
    let work = plan.planned_work();
    Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: plan.task_budget().max_kv_bytes,
        max_total_mask_node_visits: MASK_CHARGE * plan.max_output_tokens(),
        mask_limits: MaskWorkLimits { max_trie_node_visits: MASK_CHARGE as usize, checkpoint_interval_nodes: 1 } } }
}

fn selected_native(plan: &PreparedInt8SampledSentiment, axis: &PreparedAxis<Int8Work>,
    index: usize, reuse: bool, json: &str) -> Int8JsonRun {
    let mut run = synthetic(plan, axis, json);
    let tokens = run.output.token_ids.len();
    let retained = if reuse && index > 0 { axis.prompt.len() - 1 } else { 0 };
    let positions = if retained == 0 { axis.prompt.len() + tokens - 1 } else { tokens };
    // Different valid row totals on each sample test the actual head receipt,
    // without treating the declared maximum as work that really occurred.
    let work = Int8Work::for_sequence(retained, positions, tokens * (index + 1)).unwrap();
    run.execution = projection::native_execution(plan.selected_rows(), reuse).to_owned();
    run.model_work = work;
    run.output.forward_positions = work.forward_positions;
    run.output.projected_logits = work.projected_logits;
    run
}

fn selected_bundle(plan: &PreparedInt8SampledSentiment, reuse: bool) -> Int8SampledSentimentRun {
    let mut run = completed(&prepare());
    let mut dimensions = Vec::new(); let mut work = Int8Work::default();
    for axis in &plan.inner.axes {
        let mut samples = Vec::new();
        for index in 0..plan.inner.options.samples {
            let json = format!(r#"{{"score":{},"justification":"Fixture explanation."}}"#, index as i16 * 20 - 20);
            let sample = plan.finalize_sample_scheduled(axis, index,
                selected_native(plan, axis, index, reuse, &json), reuse).unwrap();
            work = work.checked_add(sample.model_work).unwrap(); samples.push(sample);
        }
        dimensions.push(aggregate(axis.axis, samples).unwrap());
    }
    run.execution = plan.execution_version(reuse).to_owned();
    run.dimensions = dimensions; run.model_work = work; run
}

#[test]
fn selected_actual_rows_and_prefix_reuse_keep_exact_causal_geometry_and_aggregates() {
    let plan = prepare().with_projection(Some(cap(17))).unwrap();
    let dense = completed(&prepare());
    for reuse in [false, true] {
        let run = selected_bundle(&plan, reuse);
        plan.verify_completed_scheduled_with_control(&run, MASK_CHARGE,
            run.mask_node_visit_charge, reuse, &mut Continue).unwrap();
        assert_eq!(run.generated_tokens, dense.generated_tokens);
        assert_eq!(run.mask_node_visit_charge, dense.mask_node_visit_charge);
        assert!(run.model_work.projected_logits < dense.model_work.projected_logits);
        for (a, b) in dense.dimensions.iter().zip(&run.dimensions) {
            assert_eq!(a.mean, b.mean); assert_eq!(a.population_stddev, b.population_stddev);
            for (a, b) in a.samples.iter().zip(&b.samples) {
                assert_eq!(a.score, b.score); assert_eq!(a.justification, b.justification);
                assert_eq!(a.sample_index, b.sample_index); assert_eq!(a.generated_tokens, b.generated_tokens);
            }
        }
        if reuse { assert!(run.model_work.forward_positions < dense.model_work.forward_positions); }
        else { assert_eq!(run.model_work.forward_positions, dense.model_work.forward_positions); }
        assert!(plan.verify_completed_scheduled_with_control(&run, MASK_CHARGE,
            u64::MAX, !reuse, &mut Continue).is_err());
        assert!(prepare().verify_completed_scheduled_with_control(&run, MASK_CHARGE,
            u64::MAX, reuse, &mut Continue).is_err());
    }
}

#[test]
fn selected_finalizers_reject_wrong_execution_row_counts_and_native_receipt_drift() {
    let plan = prepare().with_projection(Some(cap(17))).unwrap(); let axis = &plan.inner.axes[0];
    for reuse in [false, true] {
        let good = selected_native(&plan, axis, 1, reuse, r#"{"score":45,"justification":"Helpful wording."}"#);
        assert_eq!(plan.finalize_sample_scheduled(axis, 1, good.clone(), reuse).unwrap().score, 45);
        for field in 0..7 {
            let mut bad = good.clone();
            match field {
                0 => bad.execution = projection::native_execution(None, reuse).to_owned(),
                1 => bad.output.projected_logits = 0,
                2 => bad.output.projected_logits = bad.output.token_ids.len() as u64 * 17 + 1,
                3 => bad.model_work.projected_logits += 1,
                4 => bad.model_work.projections.multiply_accumulates -= 1,
                5 => bad.output.forward_positions += 1,
                _ => { bad.output.token_ids.pop(); }
            }
            assert!(plan.finalize_sample_scheduled(axis, 1, bad, reuse).is_err(), "field {field}");
        }
        for projected in [0, 4, 86, u64::MAX] {
            assert!(plan.sample_work_for_rows(axis, 1, 5, reuse, projected).is_err());
        }
        assert!(plan.sample_work_for_rows(axis, 1, options().max_sample_tokens + 1, reuse, 256).is_err());
    }
}

#[test]
fn selected_complete_results_retain_per_sample_accounting_output_caps_and_cancellation() {
    let mut plan = prepare().with_projection(Some(cap(17))).unwrap();
    for reuse in [false, true] {
        let good = selected_bundle(&plan, reuse);
        for field in 0..6 {
            let mut bad = good.clone();
            match field {
                0 => bad.execution = projection::execution(None, reuse).to_owned(),
                1 => bad.dimensions[0].samples[0].model_work.projected_logits += 1,
                2 => bad.dimensions[0].samples[1].model_work.projections.dot_products -= 1,
                3 => {
                    bad.dimensions[0].samples[0].mask_node_visit_charge += 1;
                    bad.dimensions[0].samples[1].mask_node_visit_charge -= 1;
                }
                4 => bad.model_work.attention_pairs -= 1,
                _ => bad.dimensions[0].mean += 1.0,
            }
            assert!(plan.verify_completed_scheduled_with_control(&bad, MASK_CHARGE,
                u64::MAX, reuse, &mut Continue).is_err(), "field {field}");
        }
        let size = canonjson::canonical_bytes(&good).unwrap().len() as u64;
        plan.inner.max_result_bytes = size;
        plan.verify_completed_scheduled_with_control(&good, MASK_CHARGE,
            good.mask_node_visit_charge, reuse, &mut Continue).unwrap();
        plan.inner.max_result_bytes -= 1;
        assert!(matches!(plan.verify_completed_scheduled_with_control(&good, MASK_CHARGE,
            good.mask_node_visit_charge, reuse, &mut Continue),
            Err(Int8SampledSentimentError::Shared(SampledSentimentError::OutputBudget))));
        plan.inner.max_result_bytes = budget().max_output_bytes;
        for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout, DecodeCancellationKind::Deadline,
            DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
            DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled, DecodeCancellationKind::ResourceUnavailable,
            DecodeCancellationKind::Shutdown, DecodeCancellationKind::LinkedExit] {
            let error = plan.verify_completed_scheduled_with_control(&good, MASK_CHARGE,
                good.mask_node_visit_charge, reuse, &mut StopAfter { remaining: 6, cause }).err().unwrap();
            assert_eq!(error.cancellation(), Some(cause));
        }
    }
}

#[test]
fn selected_reuse_does_not_turn_saved_prefix_work_into_extra_admission() {
    let plan = prepare().with_projection(Some(cap(17))).unwrap();
    let mut warm = Int8Work::default();
    for axis in &plan.inner.axes {
        for index in 0..plan.inner.options.samples {
            let tokens = plan.inner.options.max_sample_tokens;
            let (retained, positions) = if index == 0 { (0, axis.prompt.len() + tokens - 1) }
                else { (axis.prompt.len() - 1, tokens) };
            warm = warm.checked_add(Int8Work::for_sequence(retained, positions, tokens * 17).unwrap()).unwrap();
        }
    }
    assert!(warm.forward_positions < plan.planned_work().forward_positions);
    assert_eq!(warm.projected_logits, plan.planned_work().projected_logits);
    let sampler = plan.reuse_sampler_requirement_bytes().unwrap();
    let mut insufficient = admission(&plan);
    insufficient.native = Int8RunBudget::exact(warm);
    insufficient.json.max_forward_positions = warm.forward_positions;
    assert!(plan.preflight_bounds(MASK_CHARGE, insufficient, sampler).is_err());
    assert!(plan.preflight_bounds(MASK_CHARGE, admission(&plan), sampler).is_ok());
}
