//! Packed scheduling fixtures exercise real-call shape, not model fidelity.
use super::*;

fn limits(rows: usize) -> Int4PrefillLimits {
    Int4PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
fn execute_packed(requests: &[Int4JsonCohortRequest<'_>], budget: Int4JsonCohortBudget,
    limits: Int4PrefillLimits, driver: &mut Script) -> Result<Int4JsonCohortRun, Int4JsonError> {
    execution::drive_scheduled(GROUP, requests, &vocabulary(), budget, Some(limits), driver, Ok)
}

#[test]
fn long_ragged_prompts_pack_real_calls_without_changing_complete_rows_or_work() {
    let id = identity(); let p = programs(); let o = options(4);
    let requests = [request(&id, &[6; 32], &p[0], &o), request(&id, &[6; 17], &p[1], &o),
        request(&id, &[6; 11], &p[2], &o)];
    let mut ordinary = Script::new(3);
    let expected = execute(&requests, budget(&requests), &mut ordinary).unwrap();
    let mut packed = Script::new(3);
    let output = execute_packed(&requests, budget(&requests), limits(8), &mut packed).unwrap();
    assert_eq!(ordinary.packed_calls, 0);
    assert_eq!(output.execution, INT4_JSON_PACKED_COHORT_EXECUTION);
    assert_eq!(output.sequences, expected.sequences);
    assert_eq!(output.planned_work, expected.planned_work); assert_eq!(output.model_work, expected.model_work);
    assert_eq!(packed.seen, ordinary.seen); assert_eq!(packed.projected, ordinary.projected);
    assert_eq!(output.group_steps as usize, packed.packed_calls);
    assert_eq!(packed.packed_calls, packed.groups.len());
    assert_eq!(output.projection_groups as usize, packed.heads.len());
    assert!(output.group_steps < expected.group_steps);
    assert!(packed.groups.iter().any(|group| group.windows(2).any(|pair| pair[0].0 == pair[1].0)));
    assert!(!packed.aborted);
}

#[test]
fn narrow_packs_make_progress_and_never_append_unsampled_feedback_or_eos() {
    let id = identity(); let p = programs(); let o = options(4);
    let requests = [request(&id, &[6], &p[0], &o), request(&id, &[6; 9], &p[1], &o),
        request(&id, &[6; 5], &p[2], &o)];
    let expected = execute(&requests, budget(&requests), &mut Script::new(3)).unwrap();
    for capacity in [1, 2, 3, 5, 64] {
        let mut driver = Script::new(3);
        let output = execute_packed(&requests, budget(&requests), limits(capacity), &mut driver).unwrap();
        assert_eq!(output.sequences, expected.sequences, "capacity {capacity}");
        assert_eq!(output.model_work, expected.model_work);
        for (slot, request) in requests.iter().enumerate() {
            let mut tokens = request.prompt.to_vec();
            let generated = &output.sequences[slot].output.token_ids;
            tokens.extend_from_slice(&generated[..generated.len() - 1]);
            assert_eq!(driver.seen[slot], tokens);
            assert!(!driver.seen[slot].contains(&0));
            for group in &driver.groups {
                assert!(group.len() <= capacity);
                assert!(group.iter().filter(|&&(row, token)| row == slot && token != 6).count() <= 1);
            }
        }
        if capacity == 1 { assert_eq!(output.group_steps, output.model_work.forward_positions); }
        assert!(!driver.aborted);
    }
}

#[test]
fn invalid_extra_scratch_and_all_work_ceilings_refuse_before_any_pack() {
    let id = identity(); let p = programs(); let o = options(4);
    for axis in 0..13 {
        let mut requests = [request(&id, &[6; 8], &p[0], &o), request(&id, &[6; 4], &p[1], &o)];
        let mut cap = budget(&requests); let mut prefill = limits(8);
        match axis {
            0 => prefill.max_batch_rows = 0, 1 => prefill.max_batch_rows = 65,
            2 => prefill.max_extra_scratch_bytes -= 1,
            3 => cap.native.max_forward_positions -= 1, 4 => cap.native.max_attention_pairs -= 1,
            5 => cap.native.max_projection_work.dot_products -= 1,
            6 => cap.native.max_projection_work.multiply_accumulates -= 1,
            7 => cap.max_mask_node_visits -= 1,
            8 => requests[1].budget.native.max_forward_positions -= 1,
            9 => requests[1].budget.native.max_attention_pairs -= 1,
            10 => requests[1].budget.json.max_projected_logits -= 1,
            11 => requests[1].budget.json.max_total_mask_node_visits -= 1,
            _ => cap.max_result_bytes = 255,
        }
        let mut driver = Script::new(2);
        assert!(execute_packed(&requests, cap, prefill, &mut driver).is_err(), "axis {axis}");
        assert!(driver.groups.is_empty()); assert_eq!(driver.packed_calls, 0); assert!(driver.aborted);
    }
}

#[test]
fn cancellation_covers_interior_prompt_positions_and_every_post_sample_checkpoint() {
    let id = identity(); let p = programs(); let o = options(4);
    let requests = [request(&id, &[6; 8], &p[0], &o)]; let cap = budget(&requests);
    let mut baseline = Script::new(1);
    execute_packed(&requests, cap, limits(8), &mut baseline).unwrap();
    for position in 0..8 {
        let mut driver = Script::new(1); driver.control.prefill = Some(position);
        assert_eq!(execute_packed(&requests, cap, limits(8), &mut driver).unwrap_err().cancellation(),
            Some(DecodeCancellationKind::PollQuota));
        assert!(driver.groups.is_empty()); assert!(driver.aborted);
    }
    for stop in 1..=baseline.control.calls {
        let mut driver = Script::new(1); driver.control.stop = Some(stop);
        assert_eq!(execute_packed(&requests, cap, limits(8), &mut driver).unwrap_err().cancellation(),
            Some(DecodeCancellationKind::Deadline)); assert!(driver.aborted);
    }
    let mut driver = Script::new(1); driver.control.stop = Some(baseline.control.calls);
    let mut finalized = false;
    let result: Result<_, Int4JsonError> = execution::drive_scheduled(GROUP, &requests, &vocabulary(), cap,
        Some(limits(8)), &mut driver, |run| { finalized = true; Ok(run) });
    assert!(result.unwrap_err().cancellation().is_some()); assert!(finalized); assert!(driver.aborted);
}

#[test]
fn partial_packs_invalid_full_logits_and_wrong_work_abort_every_sequence() {
    let id = identity(); let p = programs(); let o = options(4);
    let requests = [request(&id, &[6; 8], &p[0], &o), request(&id, &[6; 4], &p[1], &o)];
    for mode in 0..6 {
        let mut driver = Script::new(2);
        match mode { 0 => driver.fail_group = Some(1), 1 => driver.bad_head = true,
            2 => driver.bad_work = true, 3 => driver.scores[1][6] = f32::NAN,
            4 => driver.scores[1][6] = f32::INFINITY, _ => driver.scores[1][6] = f32::NEG_INFINITY }
        assert!(execute_packed(&requests, budget(&requests), limits(8), &mut driver).is_err());
        assert!(driver.aborted);
    }
}

#[test]
fn packed_finalizer_and_complete_envelope_remain_inside_native_ownership() {
    let id = identity(); let p = programs(); let o = options(4);
    let requests = [request(&id, &[6; 8], &p[0], &o)]; let mut driver = Script::new(1);
    let result: Result<(), Int4JsonError> = execution::drive_scheduled(GROUP, &requests, &vocabulary(),
        budget(&requests), Some(limits(8)), &mut driver,
        |_| Err(JsonDecodeError::IndependentValidation.into()));
    assert!(result.is_err()); assert!(driver.aborted);
    let raw = execute_packed(&requests, budget(&requests), limits(8), &mut Script::new(1)).unwrap();
    let mut cap = budget(&requests); cap.max_result_bytes = serde_json::to_vec(&raw).unwrap().len() as u64;
    execute_packed(&requests, cap, limits(8), &mut Script::new(1)).unwrap();
    cap.max_result_bytes -= 1; let mut driver = Script::new(1);
    assert!(execute_packed(&requests, cap, limits(8), &mut driver).is_err()); assert!(driver.aborted);
}

#[test]
fn physical_counter_bounds_do_not_mistake_token_positions_for_invocations() {
    check_group_counts(Some(limits(8)), 33, 20, 5, 3, 5, 3).unwrap();
    assert!(check_group_counts(None, 33, 20, 5, 3, 5, 3).is_err());
    for (steps, heads) in [(4, 3), (34, 3), (5, 0), (5, 2), (5, 6)] {
        assert!(check_group_counts(Some(limits(8)), 33, 20, 5, 3, steps, heads).is_err());
    }
}
