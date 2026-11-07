//! Synthetic execution comparisons use the existing independent single-row
//! selected decoder. These are regression definitions, not model parity proof.
use super::*;
use crate::native_engine::constrained_int4::sparse::{self, Int4JsonSparseLimits};

fn limits(cap: usize) -> Int4JsonSparseLimits { Int4JsonSparseLimits { max_rows_per_step: cap } }
fn prefill(rows: usize) -> Int4PrefillLimits {
    Int4PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
fn selected_request<'a>(id: &'a ExecutionIdentity, prompt: &'a [u32], program: &'a JsonProgram,
    options: &'a JsonDecodeOptions, limits: Int4JsonSparseLimits) -> Int4JsonCohortRequest<'a> {
    let mut row = request(id, prompt, program, options);
    let work = sparse::planned_work(GROUP, prompt.len(), options.max_new_tokens, limits).unwrap();
    row.budget.native = Int4RunBudget::exact(work); row.budget.json.max_projected_logits = work.projected_logits;
    row
}
fn selected_budget(requests: &[Int4JsonCohortRequest<'_>], limits: &[Int4JsonSparseLimits]) -> Int4JsonCohortBudget {
    let mut cap = budget(requests); let mut work = Int4Work::default();
    for (request, &limits) in requests.iter().zip(limits) {
        work = work.checked_add(sparse::planned_work(GROUP, request.prompt.len(),
            request.options.max_new_tokens, limits).unwrap()).unwrap();
    }
    cap.native = Int4RunBudget::exact(work); cap
}
fn execute_selected(requests: &[Int4JsonCohortRequest<'_>], limits: &[Int4JsonSparseLimits],
    prefill: Option<Int4PrefillLimits>, cap: Int4JsonCohortBudget, driver: &mut Script)
    -> Result<Int4JsonCohortRun, Int4JsonError> {
    execution::drive_with_projection(GROUP, requests, &vocabulary(), cap,
        Projection::Selected(limits), prefill, driver, Ok)
}
struct SparseSingle<'a> { driver: &'a mut Script, slot: usize }
impl sparse::Driver for SparseSingle<'_> {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.driver.control }
    fn append(&mut self, token: u32) -> Result<(), Int4JsonError> {
        self.driver.append_group(&[Int4CohortToken { sequence: self.slot, token }])
    }
    fn logits(&mut self, rows: &[u32]) -> Result<Vec<f32>, Int4JsonError> { self.driver.logits_selected(self.slot, rows) }
    fn work(&self) -> Int4Work { GroupDriver::work(self.driver, self.slot).unwrap() }
    fn abort(&mut self) { self.driver.abort(); }
}

#[test]
fn selected_cohorts_and_every_packed_width_equal_independent_serial_decoding() {
    let id = identity(); let programs = programs(); let options = options(4);
    let limits = [limits(2), limits(2), limits(1)];
    let requests = [selected_request(&id, &[6; 11], &programs[0], &options, limits[0]),
        selected_request(&id, &[6; 7], &programs[1], &options, limits[1]),
        selected_request(&id, &[6; 3], &programs[2], &options, limits[2])];
    let mut expected = Vec::new();
    for (slot, request) in requests.iter().enumerate() {
        let mut driver = Script::new(3);
        expected.push(sparse::drive(GROUP, request.prompt, request.program, &vocabulary(), request.options,
            request.budget, limits[slot], &mut SparseSingle { driver: &mut driver, slot }, Ok::<_, Int4JsonError>).unwrap());
    }
    for schedule in [None, Some(prefill(1)), Some(prefill(2)), Some(prefill(8)), Some(prefill(64))] {
        let mut driver = Script::new(3);
        let run = execute_selected(&requests, &limits, schedule, selected_budget(&requests, &limits), &mut driver).unwrap();
        assert_eq!(run.sequences, expected);
        assert_eq!(run.execution, Projection::Selected(&limits).execution(schedule.is_some()));
        assert_eq!(run.projection_groups as usize, driver.selected_heads.len());
        assert_eq!(run.group_steps as usize, driver.groups.len());
        assert_eq!(run.projection_groups, 7);
        for (slot, request) in requests.iter().enumerate() {
            let mut tokens = request.prompt.to_vec();
            tokens.extend_from_slice(&expected[slot].output.token_ids[..expected[slot].output.token_ids.len() - 1]);
            assert_eq!(driver.seen[slot], tokens);
            assert_eq!(run.sequences[slot].output.mask_node_visit_charge,
                7 * run.sequences[slot].output.token_ids.len() as u64);
        }
        if schedule.is_some_and(|limits| limits.max_batch_rows == 64) { assert!(run.projection_groups > run.group_steps); }
        assert!(!driver.aborted);
    }
}

#[test]
fn cap_refuses_the_complete_language_before_any_selected_projection_and_eos_remains_scored() {
    let id = identity(); let programs = programs(); let mut options = options(4); let limits = [limits(1)];
    let requests = [selected_request(&id, &[6], &programs[0], &options, limits[0])];
    let mut driver = Script::new(1);
    assert!(matches!(execute_selected(&requests, &limits, None, selected_budget(&requests, &limits), &mut driver),
        Err(Int4JsonError::Decode(JsonDecodeError::BudgetExceeded("complete legal row set")))));
    assert!(driver.selected_heads.is_empty()); assert!(driver.aborted);
    options.excluded_token_ids.insert(1);
    let requests = [selected_request(&id, &[6], &programs[0], &options, limits[0])];
    let mut driver = Script::new(1);
    let run = execute_selected(&requests, &limits, Some(prefill(4)), selected_budget(&requests, &limits), &mut driver).unwrap();
    assert_eq!(run.sequences[0].output.json, "false");
    assert_eq!(driver.selected_heads, [(0, vec![2]), (0, vec![0])]);
    assert_eq!(driver.seen[0], [6, 2]);
    assert_eq!(run.model_work.projected_logits, 2);
}

#[test]
fn selected_heads_cover_large_legal_sets_with_stable_chunk_boundary_ties() {
    let id = identity(); let options = options(3); let limits = [limits(69)];
    let program = JsonProgram::compile(r#"{"type":"string","maxLength":8}"#, CompileLimits::default()).unwrap();
    let mut vocabulary = Table(vec![Vec::new()]);
    vocabulary.0.extend((1..70).map(|id| format!("\"x{id}\"").into_bytes()));
    let mut requests = [selected_request(&id, &[69], &program, &options, limits[0])];
    requests[0].budget.json.max_total_mask_node_visits = 70 * options.max_new_tokens as u64;
    let mut driver = Script::new(1); driver.scores[0] = vec![0.0; 70];
    let result: Int4JsonCohortRun = execution::drive_with_projection(GROUP, &requests, &vocabulary,
        selected_budget(&requests, &limits), Projection::Selected(&limits), Some(prefill(64)), &mut driver,
        Ok::<_, Int4JsonError>).unwrap();
    assert_eq!(result.sequences[0].output.token_ids, [1, 0]);
    assert_eq!(driver.selected_heads.iter().map(|(_, ids)| ids.len()).collect::<Vec<_>>(), [32, 32, 5, 1]);
    assert_eq!(driver.selected_heads[..3].iter().flat_map(|(_, ids)| ids.iter().copied()).collect::<Vec<_>>(),
        (1..70).collect::<Vec<_>>());
    assert_eq!(result.model_work.projected_logits, 70); assert_eq!(result.projection_groups, 4);
    assert_eq!(result.group_steps, 2);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut driver = Script::new(1); driver.scores[0] = vec![0.0; 70]; driver.scores[0][69] = bad;
        let result: Result<Int4JsonCohortRun, Int4JsonError> = execution::drive_with_projection(GROUP, &requests,
            &vocabulary, selected_budget(&requests, &limits), Projection::Selected(&limits), None, &mut driver, Ok);
        assert!(result.is_err()); assert!(driver.aborted); assert_eq!(driver.selected_heads.len(), 3);
    }
}

#[test]
fn selected_projection_never_claims_finiteness_of_unrequested_logits() {
    let id = identity(); let programs = programs(); let options = options(4); let limits = [limits(2)];
    let requests = [selected_request(&id, &[6], &programs[0], &options, limits[0])];
    let mut driver = Script::new(1); driver.scores[0][6] = f32::NAN;
    execute_selected(&requests, &limits, None, selected_budget(&requests, &limits), &mut driver).unwrap();
    assert!(driver.selected_heads.iter().all(|(_, ids)| !ids.contains(&6)));
    for mode in 0..3 {
        let mut driver = Script::new(1);
        match mode { 0 => driver.bad_head = true, 1 => driver.bad_work = true, _ => driver.fail_group = Some(2) }
        assert!(execute_selected(&requests, &limits, Some(prefill(2)), selected_budget(&requests, &limits), &mut driver).is_err());
        assert!(driver.aborted);
    }
}

#[test]
fn selected_all_axis_admission_keeps_mask_work_and_extra_scratch_before_the_first_forward() {
    let id = identity(); let programs = programs(); let options = options(4); let limits = [limits(2)];
    for axis in 0..13 {
        let mut requests = [selected_request(&id, &[6; 4], &programs[0], &options, limits[0])];
        let mut cap = selected_budget(&requests, &limits); let mut packing = prefill(4);
        match axis {
            0 => cap.native.max_forward_positions -= 1, 1 => cap.native.max_attention_pairs -= 1,
            2 => cap.native.max_projection_work.dot_products -= 1,
            3 => cap.native.max_projection_work.multiply_accumulates -= 1,
            4 => cap.max_mask_node_visits -= 1, 5 => requests[0].budget.json.max_projected_logits -= 1,
            6 => requests[0].budget.native.max_forward_positions -= 1,
            7 => requests[0].budget.native.max_attention_pairs -= 1,
            8 => requests[0].budget.native.max_projection_work.dot_products -= 1,
            9 => requests[0].budget.native.max_projection_work.multiply_accumulates -= 1,
            10 => requests[0].budget.json.max_total_mask_node_visits -= 1,
            11 => packing.max_extra_scratch_bytes -= 1, _ => cap.max_result_bytes = 127,
        }
        let mut driver = Script::new(1);
        assert!(execute_selected(&requests, &limits, Some(packing), cap, &mut driver).is_err(), "axis {axis}");
        assert!(driver.groups.is_empty()); assert!(driver.heads.is_empty()); assert!(driver.aborted);
    }
    let requests = [selected_request(&id, &[6], &programs[0], &options, limits[0])];
    for invalid in [Vec::new(), vec![self::limits(0)], vec![self::limits(8)], vec![self::limits(2); 2]] {
        let mut driver = Script::new(1);
        assert!(execute_selected(&requests, &invalid, None, selected_budget(&requests, &limits), &mut driver).is_err());
        assert!(driver.groups.is_empty());
    }
}

#[test]
fn selected_cancellation_and_late_finalization_abort_the_complete_collection() {
    let id = identity(); let programs = programs(); let options = options(4); let limits = [limits(2)];
    let requests = [selected_request(&id, &[6; 4], &programs[0], &options, limits[0])];
    let cap = selected_budget(&requests, &limits);
    for schedule in [None, Some(prefill(4))] {
        let mut baseline = Script::new(1); execute_selected(&requests, &limits, schedule, cap, &mut baseline).unwrap();
        for stop in 1..=baseline.control.calls {
            let mut driver = Script::new(1); driver.control.stop = Some(stop);
            assert!(execute_selected(&requests, &limits, schedule, cap, &mut driver).unwrap_err().cancellation().is_some());
            assert!(driver.aborted);
        }
        let mut driver = Script::new(1); driver.control.stop = Some(baseline.control.calls);
        let mut finalized = false;
        let result: Result<Int4JsonCohortRun, Int4JsonError> = execution::drive_with_projection(GROUP, &requests,
            &vocabulary(), cap, Projection::Selected(&limits), schedule, &mut driver, |run| { finalized = true; Ok(run) });
        assert!(result.unwrap_err().cancellation().is_some()); assert!(finalized); assert!(driver.aborted);
        let mut driver = Script::new(1);
        let result: Result<(), Int4JsonError> = execution::drive_with_projection(GROUP, &requests,
            &vocabulary(), cap, Projection::Selected(&limits), schedule, &mut driver,
            |_| Err(JsonDecodeError::IndependentValidation.into()));
        assert!(result.is_err()); assert!(driver.aborted);
    }
    for position in 0..4 {
        let mut driver = Script::new(1); driver.control.prefill = Some(position);
        assert!(execute_selected(&requests, &limits, Some(prefill(4)), cap, &mut driver).unwrap_err().cancellation().is_some());
        assert!(driver.groups.is_empty()); assert!(driver.aborted);
    }
}

#[test]
fn selected_complete_envelope_is_bounded_after_every_independent_row_finishes() {
    let id = identity(); let programs = programs(); let options = options(4); let limits = [limits(2)];
    let requests = [selected_request(&id, &[6], &programs[0], &options, limits[0])];
    let mut cap = selected_budget(&requests, &limits);
    let run = execute_selected(&requests, &limits, Some(prefill(2)), cap, &mut Script::new(1)).unwrap();
    cap.max_result_bytes = serde_json::to_vec(&run).unwrap().len() as u64;
    execute_selected(&requests, &limits, Some(prefill(2)), cap, &mut Script::new(1)).unwrap();
    cap.max_result_bytes -= 1; let mut driver = Script::new(1);
    assert!(execute_selected(&requests, &limits, Some(prefill(2)), cap, &mut driver).is_err());
    assert!(driver.aborted);
}

#[test]
fn selected_physical_head_counts_cover_chunks_without_weakening_causal_forward_bounds() {
    // One 65-row selection needs three native head calls at one causal state.
    check_selected_group_counts(None, 1, 1, 1, 1, 65, 1, 3).unwrap();
    assert!(check_group_counts(None, 1, 1, 1, 1, 1, 3).is_err());
    check_selected_group_counts(Some(prefill(8)), 33, 20, 5, 3, 69, 5, 7).unwrap();
    for (steps, heads) in [(4, 7), (34, 7), (5, 4), (5, 8)] {
        assert!(check_selected_group_counts(Some(prefill(8)), 33, 20, 5, 3, 69, steps, heads).is_err());
    }
    assert!(check_selected_group_counts(Some(prefill(64)), 10, 10, 5, 5, 5, 4, 5).is_err());
    assert!(check_selected_group_counts(None, 3, 3, 4, 2, 4, 3, 4).is_err());
    assert!(check_selected_group_counts(None, 3, 3, 2, 2, 1, 3, 2).is_err());
}
