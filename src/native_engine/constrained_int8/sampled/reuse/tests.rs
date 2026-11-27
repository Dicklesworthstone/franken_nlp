//! Closed-driver replay contracts; these are not native-model parity evidence.
use super::*;
use std::{cell::Cell, collections::BTreeSet, rc::Rc};
use crate::grammar::{CompileLimits, runtime::SourceRuntimeLimits};

struct Table(Vec<Vec<u8>>);
impl Table {
    fn new(bytes: &[&[u8]]) -> Self { Self(bytes.iter().map(|bytes| bytes.to_vec()).collect()) }
}
impl Vocabulary for Table {
    fn width(&self) -> usize { self.0.len() }
    fn bytes(&self, id: u32) -> Option<&[u8]> { self.0.get(id as usize).map(Vec::as_slice) }
    fn mask_charge(&self, _: MaskWorkLimits) -> usize { self.width() }
    fn mask<C: DecodeStepControl>(&self, state: &JsonState<'_>, _: MaskWorkLimits, _: &mut C, _: usize)
        -> Result<DenseTokenMask, Int8JsonError> {
        let mut mask = DenseTokenMask::empty(self.width());
        for (id, bytes) in self.0.iter().enumerate() {
            if !bytes.is_empty() && state.clone().consume_bytes(bytes) { mask.set_legal(id as u32).unwrap(); }
        }
        Ok(mask)
    }
}
#[derive(Default)]
struct Control {
    cancellation: Rc<Cell<Option<DecodeCancellationKind>>>,
    prefill_stop: Option<usize>, prefills: Vec<usize>,
}
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { self.cancellation.get() }
    fn prefill_checkpoint(&mut self, index: usize) -> Option<DecodeCancellationKind> {
        self.prefills.push(index);
        if self.prefill_stop == Some(index) { Some(DecodeCancellationKind::Deadline) }
        else { self.cancellation.get() }
    }
}
// Current branch and cumulative executed work are independent: truncation
// never returns a forward, attention pair or vocabulary projection.
struct Script {
    control: Control, row: Vec<f32>, seen: Vec<u32>, work: Int8Work,
    contexts: Vec<Vec<u32>>, rewinds: Vec<usize>, prompt_calls: usize,
    aborted: bool, rewind_cancellation: Option<DecodeCancellationKind>, corrupt: bool,
    hidden_valid: bool,
}
impl Script {
    fn new(row: Vec<f32>) -> Self {
        Self { control: Control::default(), row, seen: Vec::new(), work: Int8Work::default(),
            contexts: Vec::new(), rewinds: Vec::new(), prompt_calls: 0, aborted: false,
            rewind_cancellation: None, corrupt: false, hidden_valid: false }
    }
}
impl Driver for Script {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn append(&mut self, token: u32) -> Result<(), Int8JsonError> {
        self.work = self.work.checked_add(Int8Work::for_sequence(self.seen.len(), 1, 0)?)?;
        self.seen.push(token); self.hidden_valid = true; Ok(())
    }
    fn logits(&mut self) -> Result<Vec<f32>, Int8JsonError> {
        if !self.hidden_valid { return Err(StrictInt8Error::EmptyHidden.into()); }
        self.work = self.work.checked_add(Int8Work::for_sequence(self.seen.len(), 0, self.row.len())?)?;
        self.contexts.push(self.seen.clone()); Ok(self.row.clone())
    }
    fn work(&self) -> Result<Int8Work, Int8JsonError> {
        let mut work = self.work;
        if self.corrupt && !self.rewinds.is_empty() { work.projections.dot_products += 1; }
        Ok(work)
    }
    fn abort(&mut self) { self.aborted = true; }
}
impl prefill::PromptDriver for Script {
    fn append_layer_major(&mut self, prompt: &[u32], limits: Int8PrefillLimits) -> Result<(), Int8JsonError> {
        limits.validate()?; self.prompt_calls += 1;
        for chunk in prompt.chunks(limits.max_batch_rows) {
            for &token in chunk { self.append(token)?; }
        }
        Ok(())
    }
}
impl ReplayDriver for Script {
    fn rewind_for_replay(&mut self, retain: usize) -> Result<(), Int8JsonError> {
        if let Some(cause) = self.rewind_cancellation { return Err(StrictInt8Error::Cancelled(cause).into()); }
        if retain > self.seen.len() { return Err(StrictInt8Error::Cache.into()); }
        self.seen.truncate(retain); self.rewinds.push(retain); self.hidden_valid = false; Ok(())
    }
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"sampled-int8-json-replay-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "portable-quant-v1".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "sentiment-sampled-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "fixture".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: JSON_SAMPLER_VERSION.to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn sampling() -> Int8JsonSamplingOptions {
    Int8JsonSamplingOptions { effective_seed: [37; 32], temperature_milli: 1000,
        top_k: None, top_p_ppm: 1_000_000, sample_index: 0 }
}
fn options(tokens: usize) -> JsonDecodeOptions {
    JsonDecodeOptions { max_new_tokens: tokens, eos_token_id: 0, excluded_token_ids: BTreeSet::new() }
}
fn program(schema: &str) -> JsonProgram { JsonProgram::compile(schema, CompileLimits::default()).unwrap() }
fn limits(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
fn budget(prompt: usize, tokens: usize, width: usize, samples: usize) -> Int8JsonBudget {
    let mut work = work_for_width(prompt, tokens, width).unwrap();
    let warm = Int8Work::for_sequence(prompt - 1, tokens, tokens * width).unwrap();
    for _ in 1..samples { work = work.checked_add(warm).unwrap(); }
    Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: 8192 * KV_BYTES_PER_TOKEN as u64,
        max_total_mask_node_visits: (tokens * width * samples) as u64, mask_limits: MaskWorkLimits::default(),
    } }
}
#[allow(clippy::too_many_arguments)]
fn cold(prompt: &[u32], p: &JsonProgram, v: &Table, o: &JsonDecodeOptions,
    sample: &Int8JsonSamplingOptions, driver: &mut Script) -> Result<Int8JsonRun, Int8JsonError> {
    let mut selection = Sampled::new(&identity(), prompt, p, o, sample, json_sampling_workspace_bytes()?)?;
    drive_selected(prompt, p, v, o, budget(prompt.len(), o.max_new_tokens, v.width(), 1),
        driver, &mut selection, Ok)
}
#[allow(clippy::too_many_arguments)]
fn reused(prompt: &[u32], p: &JsonProgram, v: &Table, o: &JsonDecodeOptions,
    samples: &[Int8JsonSamplingOptions], b: Int8JsonBudget, driver: &mut Script,
    prefill: Option<Int8PrefillLimits>) -> Result<Vec<Int8JsonRun>, Int8JsonError> {
    drive_reused_prefill(&identity(), prompt, p, v, o, samples, b,
        json_sampling_reuse_workspace_bytes(32)?, prefill, driver, |_, run| Ok(run))
}

#[test]
fn reuse_matches_cold_draws_and_restores_the_exact_prompt_before_every_sample() {
    let p = program(r#"{"type":"boolean"}"#);
    let v = Table::new(&[b"", b"true", b"false", b"invalid"]); let o = options(2);
    let samples: Vec<_> = [31, 4, 17, 0, 63, 2, 9, 11].into_iter().map(|index|
        Int8JsonSamplingOptions { sample_index: index, ..sampling() }).collect();
    for prompt in [vec![3], vec![3, 2, 3, 1, 2]] {
        let expected: Vec<_> = samples.iter().map(|sample| cold(&prompt, &p, &v, &o,
            sample, &mut Script::new(vec![100.0, 0.0, 0.0, 10_000.0])).unwrap()).collect();
        for schedule in [None, Some(limits(4))] {
            let mut driver = Script::new(vec![100.0, 0.0, 0.0, 10_000.0]);
            let actual = reused(&prompt, &p, &v, &o, &samples,
                budget(prompt.len(), 2, v.width(), samples.len()), &mut driver, schedule).unwrap();
            let mut total = Int8Work::default();
            for (index, (run, expected)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(run.output.token_ids, expected.output.token_ids);
                assert_eq!(run.output.json, expected.output.json);
                assert_eq!(run.output.projected_logits, expected.output.projected_logits);
                assert_eq!(run.output.mask_node_visit_charge, expected.output.mask_node_visit_charge);
                assert_eq!(run.execution, INT8_JSON_SAMPLED_REUSED_PREFILL_EXECUTION);
                let retained = if index == 0 { 0 } else { prompt.len() - 1 };
                let work = Int8Work::for_sequence(retained, prompt.len() + 1 - retained, 2 * v.width()).unwrap();
                assert_eq!(run.model_work, work); assert_eq!(run.output.forward_positions, work.forward_positions);
                assert_eq!(driver.contexts[2 * index], prompt);
                let mut branch = prompt.clone(); branch.push(run.output.token_ids[0]);
                assert_eq!(driver.contexts[2 * index + 1], branch);
                total = total.checked_add(work).unwrap();
            }
            assert_eq!(driver.work, total); assert!(!driver.aborted);
            assert_eq!(driver.rewinds, vec![prompt.len() - 1; samples.len() - 1]);
            assert_eq!(driver.prompt_calls, usize::from(schedule.is_some()));
            let saved = Int8Work::for_sequence(0, prompt.len() - 1, 0).unwrap();
            for run in &actual[1..] { assert_eq!(run.model_work.checked_add(saved).unwrap(), expected[0].model_work); }
        }
    }
}

#[test]
fn replay_keeps_exact_unicode_source_membership() {
    let p = JsonProgram::compile_with_source(
        r#"{"type":"string","maxLength":16,"x-fnlp-source":"verbatim"}"#,
        "Alice é😀 and Alice", CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
    let v = Table::new(&[b"", br#""Mallory""#, "\"é😀\"".as_bytes()]); let o = options(2);
    let samples = [sampling(); 2]; let mut driver = Script::new(vec![100.0, 1000.0, 1.0]);
    let runs = reused(&[1, 2, 1], &p, &v, &o, &samples, budget(3, 2, 3, 2), &mut driver, None).unwrap();
    for run in runs {
        assert_eq!(run.output.json, "\"é😀\"");
        assert!(!p.source_fields(&run.output.json).unwrap().is_empty());
    }
}

#[test]
fn complete_replay_work_and_all_sample_policies_are_checked_before_native_entry() {
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true"]); let o = options(2);
    let samples = [sampling(); 3]; let b = budget(5, 2, v.width(), samples.len());
    for axis in 0..7 {
        let mut bad = b;
        match axis { 0 => bad.native.max_forward_positions -= 1, 1 => bad.native.max_attention_pairs -= 1,
            2 => bad.native.max_projection_work.dot_products -= 1,
            3 => bad.native.max_projection_work.multiply_accumulates -= 1,
            4 => bad.json.max_forward_positions -= 1, 5 => bad.json.max_projected_logits -= 1,
            _ => bad.json.max_total_mask_node_visits -= 1 }
        let mut driver = Script::new(vec![100.0, 1.0]);
        assert!(reused(&[1; 5], &p, &v, &o, &samples, bad, &mut driver, None).is_err());
        assert!(driver.seen.is_empty()); assert!(driver.rewinds.is_empty());
        assert_eq!(driver.work, Int8Work::default()); assert!(!driver.aborted);
    }
    for samples in [vec![], vec![sampling(); 33],
        vec![sampling(), Int8JsonSamplingOptions { top_p_ppm: 0, ..sampling() }]] {
        let mut driver = Script::new(vec![100.0, 1.0]);
        assert!(reused(&[1; 5], &p, &v, &o, &samples, b, &mut driver, None).is_err());
        assert!(driver.seen.is_empty()); assert_eq!(driver.work, Int8Work::default());
    }
    let mut short = limits(4); short.max_extra_scratch_bytes -= 1;
    let mut driver = Script::new(vec![100.0, 1.0]);
    assert!(reused(&[1; 5], &p, &v, &o, &samples, b, &mut driver, Some(short)).is_err());
    assert!(driver.seen.is_empty()); assert_eq!(driver.prompt_calls, 0);
    assert!(reused(&[], &p, &v, &o, &samples, b, &mut driver, None).is_err());
    assert!(reused(&[1], &p, &v, &options(0), &samples, b, &mut driver, None).is_err());
}

#[test]
fn first_layer_major_prompt_keeps_position_checkpoints_before_native_entry() {
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true"]); let o = options(2);
    for at in 0..5 {
        let mut driver = Script::new(vec![100.0, 1.0]); driver.control.prefill_stop = Some(at);
        let error = reused(&[1; 5], &p, &v, &o, &[sampling(); 2], budget(5, 2, 2, 2),
            &mut driver, Some(limits(4))).unwrap_err();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
        assert_eq!(driver.control.prefills, (0..=at).collect::<Vec<_>>());
        assert!(driver.seen.is_empty()); assert_eq!(driver.prompt_calls, 0); assert!(driver.aborted);
    }
}

#[test]
fn replay_and_finalization_keep_every_cancellation_cause_without_partial_results() {
    use DecodeCancellationKind::*;
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true"]); let o = options(2);
    let samples = [sampling(); 2]; let b = budget(3, 2, v.width(), samples.len());
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let mut driver = Script::new(vec![100.0, 1.0]); driver.rewind_cancellation = Some(cause);
        let error = reused(&[1; 3], &p, &v, &o, &samples, b, &mut driver, None).unwrap_err();
        assert_eq!(error.cancellation(), Some(cause)); assert!(driver.aborted);
        assert_eq!(driver.work, work_for_width(3, 2, v.width()).unwrap());
        let mut driver = Script::new(vec![100.0, 1.0]);
        let cancel = Rc::clone(&driver.control.cancellation); let mut finalized = 0;
        let result: Result<Vec<Int8JsonRun>, Int8JsonError> = drive_reused_prefill(&identity(),
            &[1; 3], &p, &v, &o, &samples, b, json_sampling_reuse_workspace_bytes(2).unwrap(),
            None, &mut driver, |_, run| { finalized += 1; cancel.set(Some(cause)); Ok(run) });
        assert_eq!(result.unwrap_err().cancellation(), Some(cause));
        assert_eq!(finalized, 1); assert!(driver.aborted); assert!(driver.rewinds.is_empty());
    }
}

#[test]
fn forged_work_finalizer_refusal_and_counter_underflow_are_fatal() {
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true"]); let o = options(2);
    let samples = [sampling(); 2]; let b = budget(3, 2, v.width(), samples.len());
    let mut driver = Script::new(vec![100.0, 1.0]); driver.corrupt = true;
    assert!(matches!(reused(&[1; 3], &p, &v, &o, &samples, b, &mut driver, None), Err(Int8JsonError::WorkMismatch)));
    assert!(driver.aborted);
    let mut driver = Script::new(vec![100.0, 1.0]); let mut finalized = 0;
    let result: Result<Vec<Int8JsonRun>, Int8JsonError> = drive_reused_prefill(&identity(),
        &[1; 3], &p, &v, &o, &samples, b, json_sampling_reuse_workspace_bytes(2).unwrap(),
        None, &mut driver, |index, run| {
            finalized += 1;
            if index == 1 { Err(JsonDecodeError::IndependentValidation.into()) } else { Ok(run) }
        });
    assert!(matches!(result, Err(Int8JsonError::Decode(JsonDecodeError::IndependentValidation))));
    assert_eq!(finalized, 2); assert!(driver.aborted); assert_eq!(driver.rewinds, [2]);
    let full = work_for_width(5, 2, 3).unwrap();
    assert_eq!(work_delta(full, full).unwrap(), Int8Work::default());
    for axis in 0..5 {
        let mut bad = full;
        match axis { 0 => bad.forward_positions += 1, 1 => bad.projected_logits += 1,
            2 => bad.attention_pairs += 1, 3 => bad.projections.dot_products += 1,
            _ => bad.projections.multiply_accumulates += 1 }
        assert!(matches!(work_delta(full, bad), Err(Int8JsonError::WorkMismatch)));
    }
}

#[test]
fn reuse_quote_prices_the_live_address_array_and_refuses_one_byte_short() {
    let base = json_sampling_workspace_bytes().unwrap();
    for samples in [1, 3, 32] {
        assert_eq!(json_sampling_reuse_workspace_bytes(samples).unwrap(),
            base + (samples * std::mem::size_of::<Int8JsonSamplingOptions>()) as u64);
    }
    for samples in [0, 33, usize::MAX] { assert!(json_sampling_reuse_workspace_bytes(samples).is_err()); }
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true"]); let o = options(2);
    let mut driver = Script::new(vec![100.0, 1.0]);
    let result: Result<Vec<Int8JsonRun>, Int8JsonError> = drive_reused_prefill(&identity(),
        &[1; 3], &p, &v, &o, &[sampling(); 3], budget(3, 2, 2, 3),
        json_sampling_reuse_workspace_bytes(3).unwrap() - 1, None, &mut driver, |_, run| Ok(run));
    assert!(matches!(result, Err(Int8JsonError::Decode(JsonDecodeError::BudgetExceeded(_)))));
    assert_eq!(driver.work, Int8Work::default()); assert!(driver.seen.is_empty()); assert!(!driver.aborted);
}
