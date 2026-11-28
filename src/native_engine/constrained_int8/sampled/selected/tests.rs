//! Closed-driver contracts, not executed native-model parity or speed evidence.
use super::*;
use super::super::tests::{Table, identity, options, program, sampling};

#[derive(Default)]
struct Control { calls: usize, stop_at: Option<usize>, cause: Option<DecodeCancellationKind>,
    stop_prefill: Option<usize>, prefills: Vec<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        if self.stop_at.is_some_and(|at| self.calls >= at) { self.cause } else { None }
    }
    fn prefill_checkpoint(&mut self, position: usize) -> Option<DecodeCancellationKind> {
        self.prefills.push(position);
        if self.stop_prefill == Some(position) { self.cause } else { self.checkpoint(position) }
    }
}
struct Script {
    row: Vec<f32>, control: Control, prefix: Vec<u32>, total: Int8Work,
    rows: Vec<Vec<u32>>, contexts: Vec<Vec<u32>>, forwards: Vec<u32>, rewinds: Vec<usize>,
    morsels: Vec<usize>, aborted: bool, corrupt: Option<usize>, fail_row: Option<u32>, hidden_valid: bool,
}
impl Script {
    fn new(row: Vec<f32>) -> Self {
        Self { row, control: Control::default(), prefix: Vec::new(), total: Int8Work::default(),
            rows: Vec::new(), contexts: Vec::new(), forwards: Vec::new(), rewinds: Vec::new(),
            morsels: Vec::new(), aborted: false, corrupt: None, fail_row: None, hidden_valid: false }
    }
    fn forward(&mut self, token: u32) -> Result<(), Int8JsonError> {
        self.total = self.total.checked_add(Int8Work::for_sequence(self.prefix.len(), 1, 0)?)?;
        self.prefix.push(token); self.forwards.push(token); self.hidden_valid = true; Ok(())
    }
    fn project(&mut self, rows: &[u32]) -> Result<Vec<f32>, Int8JsonError> {
        if !self.hidden_valid { return Err(StrictInt8Error::EmptyHidden.into()); }
        self.total = self.total.checked_add(Int8Work::for_sequence(0, 0, rows.len())?)?;
        self.rows.push(rows.to_vec()); self.contexts.push(self.prefix.clone());
        if self.fail_row.is_some_and(|id| rows.contains(&id)) { return Err(StrictInt8Error::Primitive.into()); }
        Ok(rows.iter().map(|&id| self.row[id as usize]).collect())
    }
    fn receipt(&self) -> Int8Work {
        let mut work = self.total;
        match self.corrupt {
            Some(0) => work.forward_positions += 1, Some(1) => work.projected_logits += 1,
            Some(2) => work.attention_pairs += 1, Some(3) => work.projections.dot_products += 1,
            Some(_) => work.projections.multiply_accumulates += 1, None => (),
        }
        work
    }
}
impl selected_head::Driver for Script {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn append(&mut self, token: u32) -> Result<(), Int8JsonError> { self.forward(token) }
    fn logits(&mut self, rows: &[u32]) -> Result<Vec<f32>, Int8JsonError> { self.project(rows) }
    fn work(&self) -> Result<Int8Work, Int8JsonError> { Ok(self.receipt()) }
    fn abort(&mut self) { self.aborted = true; }
}
impl super::super::super::Driver for Script {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn append(&mut self, token: u32) -> Result<(), Int8JsonError> { self.forward(token) }
    fn logits(&mut self) -> Result<Vec<f32>, Int8JsonError> {
        let rows: Vec<_> = (0..self.row.len() as u32).collect();
        self.project(&rows)
    }
    fn work(&self) -> Result<Int8Work, Int8JsonError> { Ok(self.receipt()) }
    fn abort(&mut self) { self.aborted = true; }
}
impl Replay for Script {
    fn append_layer_major(&mut self, prompt: &[u32], prefill: Int8PrefillLimits) -> Result<(), Int8JsonError> {
        for chunk in prompt.chunks(prefill.max_batch_rows) {
            self.morsels.push(chunk.len());
            for &token in chunk { self.forward(token)?; }
        }
        Ok(())
    }
    fn rewind_for_replay(&mut self, retain: usize) -> Result<(), Int8JsonError> {
        if retain >= self.prefix.len() { return Err(Int8JsonError::WorkMismatch); }
        self.prefix.truncate(retain); self.rewinds.push(retain); self.hidden_valid = false; Ok(())
    }
}
fn cap(rows: usize) -> Int8JsonSparseLimits { Int8JsonSparseLimits { max_rows_per_step: rows } }
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
fn budget(prompt: usize, tokens: usize, width: usize, rows: usize, samples: usize) -> Int8JsonBudget {
    let one = selected_head::planned_work(prompt, tokens, cap(rows)).unwrap();
    let work = (0..samples).try_fold(Int8Work::default(), |sum, _| sum.checked_add(one)).unwrap();
    Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: 8192 * KV_BYTES_PER_TOKEN as u64,
        max_total_mask_node_visits: (tokens * width * samples) as u64, mask_limits: MaskWorkLimits::default(),
    } }
}
#[allow(clippy::too_many_arguments)]
fn execute(prompt: &[u32], p: &JsonProgram, v: &Table, o: &JsonDecodeOptions, s: &Int8JsonSamplingOptions,
    limits: Int8JsonSparseLimits, b: Int8JsonBudget, d: &mut Script, prefill: Option<Int8PrefillLimits>)
    -> Result<Int8JsonRun, Int8JsonError> {
    let mut selection = Selected::new(&identity(), prompt, p, o, s, json_sampling_workspace_bytes().unwrap(), false)?;
    let mut driver = Prompt { inner: d, baseline: Int8Work::default(), retained: 0, replay: false, prefill };
    selected_head::drive_selected(prompt, p, v, o, b, limits, &mut driver, &mut selection, Ok)
}
fn dense(prompt: &[u32], p: &JsonProgram, v: &Table, o: &JsonDecodeOptions, s: &Int8JsonSamplingOptions,
    d: &mut Script) -> Result<Int8JsonRun, Int8JsonError> {
    let mut selection = Sampled::new(&identity(), prompt, p, o, s, json_sampling_workspace_bytes().unwrap())?;
    let b = budget(prompt.len(), o.max_new_tokens, v.width(), v.width(), 1);
    super::super::super::drive_selected(prompt, p, v, o, b, d, &mut selection, Ok)
}

#[test]
fn exact_legal_sampling_matches_dense_tokens_across_processors_and_prompt_morsels() {
    let p = program(r#"{"type":"boolean"}"#);
    let v = Table::new(&[b"", b"true", b"false", b"invalid", b"<think>"]);
    let prompt = [3, 3, 3]; let o = options(2);
    for (temperature_milli, top_k, top_p_ppm) in [(1000, None, 1_000_000), (700, None, 600_000),
        (1200, Some(2), 750_000), (1, Some(1), 1_000_000), (100_000, None, 1_000_000)] {
        for sample_index in 0..12 {
            let s = Int8JsonSamplingOptions { sample_index, temperature_milli, top_k, top_p_ppm, ..sampling() };
            let row = vec![100.0, -0.0, 0.0, 10_000.0, 10_000.0];
            let expected = dense(&prompt, &p, &v, &o, &s, &mut Script::new(row.clone())).unwrap();
            for schedule in [None, Some(prefill(1)), Some(prefill(2)), Some(prefill(4))] {
                let mut d = Script::new(row.clone());
                let actual = execute(&prompt, &p, &v, &o, &s, cap(2), budget(3, 2, 5, 2, 1), &mut d, schedule).unwrap();
                assert_eq!(actual.output.token_ids, expected.output.token_ids);
                assert_eq!(actual.output.json, expected.output.json);
                assert_eq!(actual.output.mask_node_visit_charge, expected.output.mask_node_visit_charge);
                assert_eq!(actual.model_work.forward_positions, expected.model_work.forward_positions);
                assert_eq!(actual.model_work.attention_pairs, expected.model_work.attention_pairs);
                assert_eq!(actual.model_work, Int8Work::for_sequence(0, 4, 3).unwrap());
                assert_eq!(d.rows, [vec![1, 2], vec![0]]);
                assert_eq!(actual.execution, INT8_JSON_SAMPLED_SPARSE_EXECUTION);
                assert!(!d.aborted);
            }
        }
    }
}

#[test]
fn original_ids_and_exact_nucleus_survive_multiple_projection_blocks() {
    let p = program(r#"{"type":"boolean"}"#); let o = options(2);
    let mut mask = DenseTokenMask::empty(100);
    for id in 1..100 { mask.set_legal(id).unwrap(); }
    for row in [vec![0.0; 100], (0..100).map(|i| ((i * 19 % 71) as f32 - 35.0) / 5.0).collect()] {
        for top_k in [None, Some(1), Some(MAX_TOP_K)] { for p_ppm in [1, 600_000, 1_000_000] {
            for sample_index in 0..8 {
                let s = Int8JsonSamplingOptions { sample_index, top_k, top_p_ppm: p_ppm, ..sampling() };
                let mut full = Sampled::new(&identity(), &[1], &p, &o, &s, json_sampling_workspace_bytes().unwrap()).unwrap();
                let expected = full.select(&row, &mask, false, &o, 0, &mut Control::default()).unwrap();
                let mut selected = Selected::new(&identity(), &[1], &p, &o, &s,
                    json_sampling_workspace_bytes().unwrap(), false).unwrap();
                let mut d = Script::new(row.clone()); d.forward(1).unwrap();
                let (actual, rows) = selected_head::Selection::project(&mut selected, &mask, false, &o, 99, 99, &mut d, 0).unwrap();
                assert_eq!(actual, expected); assert_eq!(rows, 99);
                assert_eq!(d.rows.iter().map(Vec::len).collect::<Vec<_>>(), [32, 32, 32, 3]);
                assert_eq!(d.rows.into_iter().flatten().collect::<Vec<_>>(), (1..100).collect::<Vec<u32>>());
            }
        } }
    }
}

#[test]
fn complete_cap_refusal_precedes_projection_and_accepting_eos_overrides_exclusion() {
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true", b"false", b"bad"]);
    let mut d = Script::new(vec![100.0, 0.0, 0.0, 10_000.0]);
    assert!(matches!(execute(&[3], &p, &v, &options(2), &sampling(), cap(1),
        budget(1, 2, 4, 1, 1), &mut d, None), Err(Int8JsonError::Decode(JsonDecodeError::BudgetExceeded(_)))));
    assert!(d.rows.is_empty()); assert_eq!(d.forwards, [3]); assert!(d.aborted);
    let mut o = options(2); o.excluded_token_ids.extend([0, 1]);
    let mut d = Script::new(vec![100.0, 10_000.0, 0.0, 100_000.0]);
    let run = execute(&[3], &p, &v, &o, &sampling(), cap(1), budget(1, 2, 4, 1, 1), &mut d, None).unwrap();
    assert_eq!(run.output.token_ids, [2, 0]); assert_eq!(d.rows, [vec![2], vec![0]]);
    assert_eq!(run.output.projected_logits, 2); assert_eq!(run.output.mask_node_visit_charge, 8);
}

#[test]
fn nonfinite_selected_rows_refuse_late_blocks_but_unprojected_logits_are_not_claimed() {
    let p = program(r#"{"type":"boolean"}"#); let o = options(2);
    let mut mask = DenseTokenMask::empty(100);
    for id in 1..100 { mask.set_legal(id).unwrap(); }
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut row = vec![0.0; 100]; row[99] = bad;
        let mut d = Script::new(row); d.forward(1).unwrap();
        let mut selected = Selected::new(&identity(), &[1], &p, &o, &sampling(),
            json_sampling_workspace_bytes().unwrap(), false).unwrap();
        assert!(matches!(selected_head::Selection::project(&mut selected, &mask, false, &o, 99, 99, &mut d, 0),
            Err(Int8JsonError::Decode(JsonDecodeError::InvalidLogits))));
        assert_eq!(d.rows.len(), 4);
    }
    let v = Table::new(&[b"", b"true", b"false", b"bad"]);
    let row = vec![100.0, 0.0, 0.0, f32::NAN];
    assert!(dense(&[1], &p, &v, &o, &sampling(), &mut Script::new(row.clone())).is_err());
    execute(&[1], &p, &v, &o, &sampling(), cap(2), budget(1, 2, 4, 2, 1), &mut Script::new(row), None).unwrap();
}

#[test]
fn every_native_and_full_mask_preflight_axis_refuses_before_forward() {
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true", b"false"]);
    let o = options(2); let b = budget(3, 2, 3, 2, 1);
    for axis in 0..7 {
        let mut b = b;
        match axis { 0 => b.native.max_forward_positions -= 1,
            1 => b.native.max_attention_pairs -= 1, 2 => b.native.max_projection_work.dot_products -= 1,
            3 => b.native.max_projection_work.multiply_accumulates -= 1,
            4 => b.json.max_forward_positions -= 1, 5 => b.json.max_projected_logits -= 1,
            _ => b.json.max_total_mask_node_visits -= 1 }
        let mut d = Script::new(vec![100.0, 0.0, 0.0]);
        assert!(execute(&[1, 1, 1], &p, &v, &o, &sampling(), cap(2), b, &mut d, None).is_err());
        assert!(d.forwards.is_empty()); assert!(d.rows.is_empty()); assert!(d.aborted);
    }
}

#[test]
fn replay_matches_independent_draws_and_causal_contexts_with_exact_live_address_quote() {
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true", b"false"]);
    let o = options(2); let samples: Vec<_> = (0..4).rev().map(|sample_index|
        Int8JsonSamplingOptions { sample_index, ..sampling() }).collect();
    let bytes = json_sampling_reuse_workspace_bytes(samples.len()).unwrap();
    for prompt in [vec![1], vec![1, 1, 1, 1, 1]] { for schedule in [None, Some(prefill(2))] {
        let b = budget(prompt.len(), 2, 3, 2, samples.len());
        let mut contexts = Vec::new();
        let expected: Vec<_> = samples.iter().map(|sample| {
            let mut d = Script::new(vec![100.0, 0.0, 0.0]);
            let run = execute(&prompt, &p, &v, &o, sample, cap(2), b, &mut d, schedule).unwrap();
            contexts.extend(d.contexts); run
        }).collect();
        let mut d = Script::new(vec![100.0, 0.0, 0.0]);
        let actual = drive_reused(&identity(), &prompt, &p, &v, &o, &samples, b,
            bytes, cap(2), schedule, &mut d, |_, run| Ok::<_, Int8JsonError>(run)).unwrap();
        for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(a.output.token_ids, e.output.token_ids); assert_eq!(a.output.json, e.output.json);
            assert_eq!(a.output.mask_node_visit_charge, e.output.mask_node_visit_charge);
            assert_eq!(a.execution, INT8_JSON_SAMPLED_SPARSE_REUSED_PREFILL_EXECUTION);
            assert_eq!(a.model_work, Int8Work::for_sequence(
                if index == 0 { 0 } else { prompt.len() - 1 },
                if index == 0 { prompt.len() + 1 } else { 2 }, 3).unwrap());
        }
        assert_eq!(d.contexts, contexts);
        assert_eq!(d.total, actual.iter().try_fold(Int8Work::default(), |sum, run| sum.checked_add(run.model_work)).unwrap());
        assert_eq!(d.rewinds, vec![prompt.len() - 1; samples.len() - 1]); assert!(!d.aborted);
        let expected_prefills: Vec<_> = (0..prompt.len()).chain(std::iter::repeat_n(prompt.len() - 1, samples.len() - 1)).collect();
        assert_eq!(d.control.prefills, expected_prefills);
        for (b, bytes) in [({ let mut short = b; short.json.max_total_mask_node_visits -= 1; short }, bytes), (b, bytes - 1)] {
            let mut refused = Script::new(vec![100.0, 0.0, 0.0]);
            assert!(drive_reused(&identity(), &prompt, &p, &v, &o, &samples, b,
                bytes, cap(2), schedule, &mut refused, |_, run| Ok::<_, Int8JsonError>(run)).is_err());
            assert!(refused.forwards.is_empty()); assert!(refused.rows.is_empty());
        }
    } }
}

#[test]
fn typed_cancellation_finalizer_and_forged_work_never_complete_a_selected_sample() {
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true", b"false"]);
    let o = options(2); let b = budget(1, 2, 3, 2, 1);
    let mut baseline = Script::new(vec![100.0, 0.0, 0.0]);
    execute(&[1], &p, &v, &o, &sampling(), cap(2), b, &mut baseline, None).unwrap();
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, LinkedExit, Shutdown] {
        for stop_at in [1, 4, baseline.control.calls] {
            let mut d = Script::new(vec![100.0, 0.0, 0.0]);
            d.control = Control { stop_at: Some(stop_at), cause: Some(cause), ..Control::default() };
            let error = execute(&[1], &p, &v, &o, &sampling(), cap(2), b, &mut d, None).unwrap_err();
            assert_eq!(error.cancellation(), Some(cause)); assert!(d.aborted);
        }
        let mut d = Script::new(vec![100.0, 0.0, 0.0]);
        let mut selection = Selected::new(&identity(), &[1], &p, &o, &sampling(),
            json_sampling_workspace_bytes().unwrap(), false).unwrap();
        let failed = selected_head::drive_selected(&[1], &p, &v, &o, b, cap(2), &mut d,
            &mut selection, |_| Err::<Int8JsonRun, _>(Int8JsonError::Decode(JsonDecodeError::Cancelled(cause))));
        assert_eq!(failed.unwrap_err().cancellation(), Some(cause)); assert!(d.aborted);
    }
    for axis in 0..5 {
        let mut d = Script::new(vec![100.0, 0.0, 0.0]); d.corrupt = Some(axis);
        assert!(matches!(execute(&[1], &p, &v, &o, &sampling(), cap(2), b, &mut d, None), Err(Int8JsonError::WorkMismatch)));
        assert!(d.aborted);
    }
    let mut d = Script::new(vec![100.0, 0.0, 0.0]); d.fail_row = Some(0);
    assert!(execute(&[1], &p, &v, &o, &sampling(), cap(2), b, &mut d, None).is_err());
    assert!(d.aborted); assert_eq!(d.rows, [vec![1, 2], vec![0]]);
}

#[test]
fn replay_tail_cancellation_and_late_finalizer_failure_refuse_the_complete_bundle() {
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true", b"false"]);
    let prompt = [1, 1, 1]; let o = options(2);
    let samples: Vec<_> = (0..3).map(|sample_index| Int8JsonSamplingOptions { sample_index, ..sampling() }).collect();
    let bytes = json_sampling_reuse_workspace_bytes(samples.len()).unwrap();
    let b = budget(prompt.len(), 2, 3, 2, samples.len());
    let mut baseline = Script::new(vec![100.0, 0.0, 0.0]);
    drive_reused(&identity(), &prompt, &p, &v, &o, &samples, b, bytes, cap(2), Some(prefill(2)),
        &mut baseline, |_, run| Ok::<_, Int8JsonError>(run)).unwrap();
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, LinkedExit, Shutdown] {
        let mut late = Script::new(vec![100.0, 0.0, 0.0]);
        late.control = Control { stop_at: Some(baseline.control.calls), cause: Some(cause), ..Control::default() };
        let mut checked = 0;
        let result = drive_reused(&identity(), &prompt, &p, &v, &o, &samples, b,
            bytes, cap(2), Some(prefill(2)), &mut late, |_, run| { checked += 1; Ok::<_, Int8JsonError>(run) });
        assert_eq!(result.unwrap_err().cancellation(), Some(cause));
        assert_eq!(checked, samples.len()); assert!(late.aborted); assert_eq!(late.rewinds, [2, 2]);
        let mut late = Script::new(vec![100.0, 0.0, 0.0]);
        let mut checked = 0;
        let result = drive_reused(&identity(), &prompt, &p, &v, &o, &samples, b,
            bytes, cap(2), None, &mut late, |index, run| {
                checked += 1;
                if index == 1 { Err(Int8JsonError::Decode(JsonDecodeError::Cancelled(cause))) }
                else { Ok(run) }
            });
        assert_eq!(result.unwrap_err().cancellation(), Some(cause));
        assert_eq!(checked, 2); assert!(late.aborted); assert_eq!(late.rewinds, [2]);
    }
}

#[test]
fn selected_sampling_cancellation_after_complete_projection_never_returns_a_token() {
    let p = program(r#"{"type":"boolean"}"#); let o = options(2);
    let mut mask = DenseTokenMask::empty(1025);
    for id in 1..1025 { mask.set_legal(id).unwrap(); }
    let row: Vec<_> = (0..1025).map(|i| ((i * 19 % 71) as f32 - 35.0) / 5.0).collect();
    let mut selection = Selected::new(&identity(), &[1], &p, &o, &sampling(),
        json_sampling_workspace_bytes().unwrap(), false).unwrap();
    let mut baseline = Script::new(row.clone()); baseline.forward(1).unwrap();
    selected_head::Selection::project(&mut selection, &mask, false, &o, 1024, 1024, &mut baseline, 0).unwrap();
    let last_checkpoint = baseline.control.calls;
    assert_eq!(baseline.rows.iter().map(Vec::len).sum::<usize>(), 1024);
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let mut d = Script::new(row.clone()); d.forward(1).unwrap();
        d.control = Control { stop_at: Some(last_checkpoint), cause: Some(cause), ..Default::default() };
        let error = selected_head::Selection::project(&mut selection, &mask, false, &o,
            1024, 1024, &mut d, 0).unwrap_err();
        assert_eq!(error.cancellation(), Some(cause));
        assert_eq!(d.rows.iter().map(Vec::len).sum::<usize>(), 1024);
        assert_eq!(d.forwards, [1], "cancelled selection must not feed a token back");
    }
}
