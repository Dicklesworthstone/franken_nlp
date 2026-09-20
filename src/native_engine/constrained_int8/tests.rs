use super::*;
use std::collections::BTreeSet;
use crate::grammar::{CompileLimits, runtime::SourceRuntimeLimits};

struct Table(Vec<Vec<u8>>);
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
struct Control { prefill: Option<usize>, step: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, step: usize) -> Option<DecodeCancellationKind> {
        (self.step == Some(step)).then_some(DecodeCancellationKind::Deadline)
    }
    fn prefill_checkpoint(&mut self, position: usize) -> Option<DecodeCancellationKind> {
        (self.prefill == Some(position)).then_some(DecodeCancellationKind::Deadline)
    }
}
struct Script {
    control: Control, row: Vec<f32>, seen: Vec<u32>, head_calls: usize, width: usize,
    aborted: bool, fail_append: bool, corrupt_work: bool,
}
impl Script {
    fn new(row: Vec<f32>) -> Self {
        Self { width: row.len(), row, control: Control::default(), seen: Vec::new(), head_calls: 0,
            aborted: false, fail_append: false, corrupt_work: false }
    }
}
impl Driver for Script {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn append(&mut self, token: u32) -> Result<(), Int8JsonError> {
        if self.fail_append { return Err(StrictInt8Error::Primitive.into()); }
        self.seen.push(token); Ok(())
    }
    fn logits(&mut self) -> Result<Vec<f32>, Int8JsonError> { self.head_calls += 1; Ok(self.row.clone()) }
    fn work(&self) -> Int8Work {
        let mut work = Int8Work::for_sequence(0, self.seen.len(), self.head_calls * self.width).unwrap();
        if self.corrupt_work { work.attention_pairs += 1; }
        work
    }
    fn abort(&mut self) { self.aborted = true; }
}
fn options(n: usize) -> JsonDecodeOptions {
    JsonDecodeOptions { max_new_tokens: n, eos_token_id: 0, excluded_token_ids: BTreeSet::new() }
}
fn program(schema: &str) -> JsonProgram { JsonProgram::compile(schema, CompileLimits::default()).unwrap() }
fn table(values: &[&[u8]]) -> Table { Table(values.iter().map(|v| v.to_vec()).collect()) }
fn budget(prompt: usize, output: usize, width: usize) -> Int8JsonBudget {
    let work = work_for_width(prompt, output, width).unwrap();
    Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: 8192 * KV_BYTES_PER_TOKEN as u64, max_total_mask_node_visits: (output * width) as u64,
        mask_limits: MaskWorkLimits::default(),
    } }
}
fn execute(prompt: &[u32], p: &JsonProgram, v: &Table, o: &JsonDecodeOptions, b: Int8JsonBudget, d: &mut Script)
    -> Result<Int8JsonRun, Int8JsonError> {
    drive(prompt, p, v, o, b, d, Ok)
}

#[test]
fn prefill_projects_only_the_last_prompt_then_scores_explicit_eos() {
    let p = program(r#"{"type":"boolean"}"#);
    let v = table(&[b"", b"true", b"malformed"]);
    let mut d = Script::new(vec![100.0, 1.0, 1000.0]);
    let out = execute(&[2, 2, 2], &p, &v, &options(2), budget(3, 2, 3), &mut d).unwrap();
    assert_eq!(d.seen, vec![2, 2, 2, 1]); assert_eq!(d.head_calls, 2); assert!(!d.aborted);
    assert_eq!(out.output.json, "true"); assert_eq!(out.output.token_ids, vec![1, 0]);
    assert_eq!(out.output.forward_positions, 4); assert_eq!(out.output.projected_logits, 6);
    assert_eq!(out.output.mask_node_visit_charge, 6);
    assert_eq!(out.model_work, Int8Work::for_sequence(0, 4, 6).unwrap());
    assert_eq!(out.output.numerics_profile, STRICT_INT8_PROFILE); assert_eq!(out.execution, INT8_JSON_EXECUTION);
}

#[test]
fn source_membership_masks_higher_scoring_off_source_json() {
    let p = JsonProgram::compile_with_source(r#"{"type":"string","maxLength":16,"x-fnlp-source":"verbatim"}"#,
        "Alice", CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
    let v = table(&[b"", br#""Mallory""#, br#""Alice""#]);
    let mut d = Script::new(vec![10.0, 100.0, 1.0]);
    let out = execute(&[1], &p, &v, &options(2), budget(1, 2, 3), &mut d).unwrap();
    assert_eq!(out.output.json, r#""Alice""#);
    assert!(!p.source_fields(&out.output.json).unwrap().is_empty());
}

#[test]
fn one_token_can_cross_json_boundaries_and_preserve_multibyte_text() {
    let p = program(r#"{"type":"object","properties":{"x":{"type":"string","maxLength":8}},"required":["x"],"additionalProperties":false}"#);
    let value = r#"{"x":"é😀"}"#;
    let v = table(&[b"", value.as_bytes()]); let mut d = Script::new(vec![0.0, 1.0]);
    let out = execute(&[1], &p, &v, &options(2), budget(1, 2, 2), &mut d).unwrap();
    assert_eq!(out.output.json.as_bytes(), value.as_bytes());
}

#[test]
fn accepting_numeric_prefix_does_not_force_eos_or_shorter_number() {
    let p = program(r#"{"type":"integer","enum":[1,10]}"#);
    let v = table(&[b"", b"1", b"0"]); let mut d = Script::new(vec![0.0, 2.0, 1.0]);
    let out = execute(&[1], &p, &v, &options(3), budget(1, 3, 3), &mut d).unwrap();
    assert_eq!(out.output.json, "10"); assert_eq!(out.output.token_ids, vec![1, 2, 0]);
}

#[test]
fn budget_exhaustion_returns_no_even_structurally_complete_eosless_value() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    let mut d = Script::new(vec![0.0, 1.0]);
    assert!(matches!(execute(&[1], &p, &v, &options(1), budget(1, 1, 2), &mut d),
        Err(Int8JsonError::Decode(JsonDecodeError::BudgetExceeded(_)))));
    assert!(d.aborted); assert_eq!(d.seen, vec![1]); assert_eq!(d.head_calls, 1);
}

#[test]
fn controls_remain_excluded_but_terminal_eos_has_its_exact_acceptance_rule() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true", b"false"]);
    let mut o = options(2); o.excluded_token_ids.extend([0, 1]);
    let mut d = Script::new(vec![20.0, 100.0, 1.0]);
    let out = execute(&[1], &p, &v, &o, budget(1, 2, 3), &mut d).unwrap();
    assert_eq!(out.output.json, "false"); assert_eq!(out.output.token_ids, vec![2, 0]);
    o.excluded_token_ids.insert(2);
    assert!(matches!(execute(&[1], &p, &v, &o, budget(1, 2, 3), &mut Script::new(vec![0.0; 3])),
        Err(Int8JsonError::Decode(JsonDecodeError::NoLegalToken))));
}

#[test]
fn every_logit_must_be_finite_even_for_a_masked_out_token() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true", b"bad"]);
    for row in [vec![0.0, 1.0, f32::NAN], vec![0.0, 1.0, f32::INFINITY],
        vec![0.0, 1.0, f32::NEG_INFINITY], vec![0.0, 1.0]] {
        let mut d = Script::new(row);
        assert!(matches!(execute(&[1], &p, &v, &options(2), budget(1, 2, 3), &mut d),
            Err(Int8JsonError::Decode(JsonDecodeError::InvalidLogits))));
        assert!(d.aborted);
    }
}

#[test]
fn first_token_id_wins_ties_including_signed_zero() {
    let mut mask = DenseTokenMask::empty(3); mask.set_legal(1).unwrap(); mask.set_legal(2).unwrap();
    assert_eq!(select(&[100.0, -0.0, 0.0], &mask, false, &options(2)).unwrap(), 1);
    assert_eq!(select(&[-0.0, 0.0, 0.0], &mask, true, &options(2)).unwrap(), 0);
}

#[test]
fn all_work_axes_are_preflighted_before_any_native_append() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    for axis in 0..7 {
        let mut b = budget(2, 2, 2);
        match axis {
            0 => b.json.max_forward_positions -= 1, 1 => b.json.max_projected_logits -= 1,
            2 => b.json.max_total_mask_node_visits -= 1, 3 => b.native.max_forward_positions -= 1,
            4 => b.native.max_attention_pairs -= 1, 5 => b.native.max_projection_work.dot_products -= 1,
            _ => b.native.max_projection_work.multiply_accumulates -= 1,
        }
        let mut d = Script::new(vec![0.0, 1.0]);
        assert!(execute(&[1, 1], &p, &v, &options(2), b, &mut d).is_err());
        assert!(d.seen.is_empty()); assert_eq!(d.head_calls, 0);
    }
}

#[test]
fn kv_checks_resident_capacity_and_context_not_only_live_positions() {
    let work = planned_work(1, 2).unwrap(); let all = 8 * KV_BYTES_PER_TOKEN as u64;
    assert!(check_capacity(work, 8, all).is_ok());
    assert!(check_capacity(work, 8, all - 1).is_err()); assert!(check_capacity(work, 1, all).is_err());
    assert!(planned_work(0, 1).is_err()); assert!(planned_work(1, 0).is_err());
    assert!(planned_work(usize::MAX, 2).is_err());
    assert!(planned_work(DEFAULT_ADMITTED_CONTEXT_CAP, 2).is_err());
    assert_eq!(planned_work(17, 3).unwrap().projected_logits, 3 * NANBEIGE_VOCAB_SIZE as u64);
}

#[test]
fn both_prefill_and_decode_cancellation_abort_without_partial_success() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    for prefill in [true, false] {
        let mut d = Script::new(vec![0.0, 1.0]);
        if prefill { d.control.prefill = Some(1); } else { d.control.step = Some(1); }
        let error = execute(&[1, 1], &p, &v, &options(2), budget(2, 2, 2), &mut d).unwrap_err();
        assert!(error.cancellation().is_some()); assert!(d.aborted);
        if prefill { assert_eq!(d.seen.len(), 1); assert_eq!(d.head_calls, 0); }
    }
}

#[test]
fn native_failure_and_accounting_disagreement_are_fatal() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    for native in [true, false] {
        let mut d = Script::new(vec![0.0, 1.0]); d.fail_append = native; d.corrupt_work = !native;
        let error = execute(&[1], &p, &v, &options(2), budget(1, 2, 2), &mut d).unwrap_err();
        assert!(matches!(error, Int8JsonError::Native(_) | Int8JsonError::WorkMismatch)); assert!(d.aborted);
    }
}

#[test]
fn finalization_failure_aborts_the_still_owned_session() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    let mut d = Script::new(vec![0.0, 1.0]); let mut finalized = false;
    let result: Result<(), Int8JsonError> = drive(&[1], &p, &v, &options(2), budget(1, 2, 2), &mut d, |run| {
        finalized = true; assert_eq!(run.output.json, "true");
        Err(JsonDecodeError::IndependentValidation.into())
    });
    assert!(result.is_err()); assert!(finalized); assert!(d.aborted);
}

#[test]
fn successful_replay_has_identical_content_and_complete_work() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    let mut outputs = Vec::new();
    for _ in 0..2 {
        let out = execute(&[1], &p, &v, &options(2), budget(1, 2, 2), &mut Script::new(vec![0.0, 1.0])).unwrap();
        outputs.push(crate::canonjson::canonical_bytes(&out).unwrap());
    }
    assert_eq!(outputs[0], outputs[1]);
}
