//! Reuse receipts stay private test fixtures, never a public model backend.
use super::*;

fn replayed_native(plan: &PreparedInt8SampledSentiment, axis: &PreparedAxis<Int8Work>,
    index: usize, json: &str) -> Int8JsonRun {
    let mut run = synthetic(plan, axis, json);
    run.execution = constrained_int8::INT8_JSON_SAMPLED_REUSED_PREFILL_EXECUTION.to_owned();
    if index > 0 {
        run.model_work = Int8Work::for_sequence(axis.prompt.len() - 1, run.output.token_ids.len(),
            run.output.token_ids.len() * NANBEIGE_VOCAB_SIZE).unwrap();
        run.output.forward_positions = run.model_work.forward_positions;
    }
    run
}

fn replayed_bundle(plan: &PreparedInt8SampledSentiment) -> Int8SampledSentimentRun {
    let mut run = completed(plan);
    let mut dimensions = Vec::new(); let mut work = Int8Work::default();
    for axis in &plan.inner.axes {
        let mut samples = Vec::new();
        for index in 0..plan.inner.options.samples {
            let json = format!(r#"{{"score":{},"justification":"Fixture explanation."}}"#, index as i16 * 20 - 20);
            let sample = plan.finalize_sample_scheduled(axis, index,
                replayed_native(plan, axis, index, &json), true).unwrap();
            work = work.checked_add(sample.model_work).unwrap(); samples.push(sample);
        }
        dimensions.push(aggregate(axis.axis, samples).unwrap());
    }
    run.execution = INT8_SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION.to_owned();
    run.dimensions = dimensions; run.model_work = work; run
}

#[test]
fn replay_bundle_preserves_independent_scores_and_charges_only_executed_prefix_work() {
    let plan = prepare(); let cold = completed(&plan); let replayed = replayed_bundle(&plan);
    plan.verify_completed_scheduled_with_control(&replayed, MASK_CHARGE,
        replayed.mask_node_visit_charge, true, &mut Continue).unwrap();
    let mut saved = Int8Work::default();
    for ((axis, cold), replayed) in plan.inner.axes.iter().zip(&cold.dimensions).zip(&replayed.dimensions) {
        let prefix = Int8Work::for_sequence(0, axis.prompt.len() - 1, 0).unwrap();
        for _ in 1..plan.inner.options.samples { saved = saved.checked_add(prefix).unwrap(); }
        assert_eq!(cold.axis, replayed.axis); assert_eq!(cold.mean, replayed.mean);
        assert_eq!(cold.population_stddev, replayed.population_stddev);
        for (cold, replayed) in cold.samples.iter().zip(&replayed.samples) {
            assert_eq!(cold.sample_index, replayed.sample_index); assert_eq!(cold.score, replayed.score);
            assert_eq!(cold.justification, replayed.justification);
            assert_eq!(cold.generated_tokens, replayed.generated_tokens);
            assert_eq!(cold.mask_node_visit_charge, replayed.mask_node_visit_charge);
            if cold.sample_index == 0 { assert_eq!(cold.model_work, replayed.model_work); }
            else { assert_eq!(replayed.model_work.checked_add(prefix).unwrap(), cold.model_work); }
        }
    }
    assert_eq!(replayed.model_work.checked_add(saved).unwrap(), cold.model_work);
    assert_eq!(replayed.generated_tokens, cold.generated_tokens);
    assert_eq!(replayed.mask_node_visit_charge, cold.mask_node_visit_charge);
    assert!(plan.verify_completed_with_control(&replayed, MASK_CHARGE, u64::MAX, &mut Continue).is_err());
    assert!(plan.verify_completed_scheduled_with_control(&cold, MASK_CHARGE, u64::MAX, true, &mut Continue).is_err());
}

#[test]
fn replay_finalizer_rejects_cold_or_refunded_work_and_a_different_physical_schedule() {
    let plan = prepare(); let axis = &plan.inner.axes[0];
    let json = r#"{"score":45,"justification":"Helpful wording."}"#;
    let good = replayed_native(&plan, axis, 1, json);
    assert_eq!(plan.finalize_sample_scheduled(axis, 1, good.clone(), true).unwrap().score, 45);
    assert!(plan.finalize_sample(axis, 1, good.clone()).is_err());
    for field in 0..6 {
        let mut bad = good.clone();
        match field {
            0 => bad.execution = INT8_JSON_SAMPLED_EXECUTION.to_owned(),
            1 => bad.model_work = synthetic(&plan, axis, json).model_work,
            2 => bad.model_work.attention_pairs = Int8Work::for_sequence(0, bad.output.token_ids.len(), 0).unwrap().attention_pairs,
            3 => bad.output.forward_positions += 1,
            4 => bad.model_work.projections.dot_products -= 1,
            _ => bad.output.projected_logits -= NANBEIGE_VOCAB_SIZE as u64,
        }
        assert!(plan.finalize_sample_scheduled(axis, 1, bad, true).is_err(), "field {field}");
    }
    assert!(plan.finalize_sample_scheduled(axis, plan.inner.options.samples, good, true).is_err());
    // Even the first sample of a reused axis reports the explicit schedule,
    // although its physical forwards equal the original cold execution.
    assert!(plan.finalize_sample_scheduled(axis, 0, synthetic(&plan, axis, json), true).is_err());
}

#[test]
fn reused_task_still_requires_the_complete_cold_work_ceiling() {
    let plan = prepare(); let mut reused = Int8Work::default();
    for axis in &plan.inner.axes {
        for index in 0..plan.inner.options.samples {
            let (start, positions) = if index == 0 { (0, axis.prompt.len() + plan.inner.options.max_sample_tokens - 1) }
                else { (axis.prompt.len() - 1, plan.inner.options.max_sample_tokens) };
            reused = reused.checked_add(Int8Work::for_sequence(start, positions,
                plan.inner.options.max_sample_tokens * NANBEIGE_VOCAB_SIZE).unwrap()).unwrap();
        }
    }
    assert!(reused.forward_positions < plan.planned_work().forward_positions);
    let masks = MASK_CHARGE * plan.inner.options.max_sample_tokens as u64 * plan.sample_count() as u64;
    let mut budget = Int8JsonBudget { native: Int8RunBudget::exact(reused), json: JsonWorkBudget {
        max_forward_positions: reused.forward_positions, max_projected_logits: reused.projected_logits,
        max_kv_bytes: plan.task_budget().max_kv_bytes, max_total_mask_node_visits: masks,
        mask_limits: MaskWorkLimits { max_trie_node_visits: MASK_CHARGE as usize, checkpoint_interval_nodes: 1 } } };
    let sampler = plan.reuse_sampler_requirement_bytes().unwrap();
    assert!(plan.preflight_bounds(MASK_CHARGE, budget, sampler).is_err());
    let cold = plan.planned_work(); budget.native = Int8RunBudget::exact(cold);
    budget.json.max_forward_positions = cold.forward_positions;
    budget.json.max_projected_logits = cold.projected_logits;
    assert!(plan.preflight_bounds(MASK_CHARGE, budget, sampler).is_ok());
    assert_eq!(sampler, json_sampling_workspace_bytes().unwrap()
        + (plan.inner.options.samples * std::mem::size_of::<Int8JsonSamplingOptions>()) as u64);
}

#[test]
fn reused_bundle_refuses_cross_sample_credit_even_when_total_work_is_preserved() {
    let plan = prepare(); let good = replayed_bundle(&plan);
    for field in 0..5 {
        let mut bad = good.clone();
        match field {
            0 => {
                let first = bad.dimensions[0].samples[0].model_work;
                bad.dimensions[0].samples[0].model_work = bad.dimensions[0].samples[1].model_work;
                bad.dimensions[0].samples[1].model_work = first;
            }
            1 => {
                bad.dimensions[0].samples[0].mask_node_visit_charge += 1;
                bad.dimensions[0].samples[1].mask_node_visit_charge -= 1;
            }
            2 => bad.dimensions[0].samples[1].sample_index = 0,
            3 => bad.execution = INT8_SAMPLED_SENTIMENT_EXECUTION.to_owned(),
            _ => bad.model_work.attention_pairs -= 1,
        }
        assert!(plan.verify_completed_scheduled_with_control(&bad, MASK_CHARGE,
            u64::MAX, true, &mut Continue).is_err(), "field {field}");
    }
}

#[test]
fn reused_complete_result_keeps_output_limits_and_late_cancellation() {
    let mut plan = prepare(); let run = replayed_bundle(&plan);
    let size = canonjson::canonical_bytes(&run).unwrap().len() as u64;
    plan.inner.max_result_bytes = size;
    plan.verify_completed_scheduled_with_control(&run, MASK_CHARGE,
        run.mask_node_visit_charge, true, &mut Continue).unwrap();
    plan.inner.max_result_bytes -= 1;
    assert!(matches!(plan.verify_completed_scheduled_with_control(&run, MASK_CHARGE,
        run.mask_node_visit_charge, true, &mut Continue),
        Err(Int8SampledSentimentError::Shared(SampledSentimentError::OutputBudget))));
    plan.inner.max_result_bytes = size;
    let mut stop = StopAfter { remaining: 6, cause: DecodeCancellationKind::Shutdown };
    let error = plan.verify_completed_scheduled_with_control(&run, MASK_CHARGE,
        run.mask_node_visit_charge, true, &mut stop).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Shutdown));
}
