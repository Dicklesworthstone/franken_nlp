//! Model-free driver and exact selection regressions; NOT real-model parity.
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
struct Control { prefill: Option<usize>, step: Option<usize>, cancel_poll: Option<usize>, polls: usize }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, step: usize) -> Option<DecodeCancellationKind> {
        let poll = self.polls; self.polls += 1;
        (self.step == Some(step) || self.cancel_poll == Some(poll)).then_some(DecodeCancellationKind::Deadline)
    }
    fn prefill_checkpoint(&mut self, position: usize) -> Option<DecodeCancellationKind> {
        (self.prefill == Some(position)).then_some(DecodeCancellationKind::Deadline)
    }
}
struct Script {
    row: Vec<f32>, seen: Vec<u32>, heads: Vec<Vec<u32>>, control: Control,
    aborted: bool, fail_append: bool, fail_head: bool, corrupt_work: bool, short_head: bool,
}
impl Script {
    fn new(row: Vec<f32>) -> Self {
        Self { row, seen: Vec::new(), heads: Vec::new(), control: Control::default(),
            aborted: false, fail_append: false, fail_head: false, corrupt_work: false, short_head: false }
    }
    fn forward(&mut self, token: u32) -> Result<(), Int8JsonError> {
        if self.fail_append { return Err(StrictInt8Error::Primitive.into()); }
        self.seen.push(token); Ok(())
    }
    fn project(&mut self, rows: &[u32]) -> Result<Vec<f32>, Int8JsonError> {
        if self.fail_head { return Err(StrictInt8Error::Primitive.into()); }
        self.heads.push(rows.to_vec());
        let mut values: Vec<_> = rows.iter().map(|&id| self.row[id as usize]).collect();
        if self.short_head { values.pop(); }
        Ok(values)
    }
    fn receipt(&self) -> Int8Work {
        let projected = self.heads.iter().map(Vec::len).sum();
        let mut work = Int8Work::for_sequence(0, self.seen.len(), projected).unwrap();
        if self.corrupt_work { work.attention_pairs += 1; }
        work
    }
}
impl Driver for Script {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn append(&mut self, token: u32) -> Result<(), Int8JsonError> { self.forward(token) }
    fn logits(&mut self, rows: &[u32]) -> Result<Vec<f32>, Int8JsonError> { self.project(rows) }
    fn work(&self) -> Int8Work { self.receipt() }
    fn abort(&mut self) { self.aborted = true; }
}
impl super::super::Driver for Script {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn append(&mut self, token: u32) -> Result<(), Int8JsonError> { self.forward(token) }
    fn logits(&mut self) -> Result<Vec<f32>, Int8JsonError> {
        let rows: Vec<_> = (0..self.row.len() as u32).collect(); self.project(&rows)
    }
    fn work(&self) -> Int8Work { self.receipt() }
    fn abort(&mut self) { self.aborted = true; }
}
fn options(n: usize) -> JsonDecodeOptions {
    JsonDecodeOptions { max_new_tokens: n, eos_token_id: 0, excluded_token_ids: BTreeSet::new() }
}
fn program(schema: &str) -> JsonProgram { JsonProgram::compile(schema, CompileLimits::default()).unwrap() }
fn table(values: &[&[u8]]) -> Table { Table(values.iter().map(|v| v.to_vec()).collect()) }
fn limits(cap: usize) -> Int8JsonSparseLimits { Int8JsonSparseLimits { max_rows_per_step: cap } }
fn budget(prompt: usize, output: usize, width: usize, cap: usize) -> Int8JsonBudget {
    let work = planned_work(prompt, output, limits(cap)).unwrap();
    Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: 8192 * KV_BYTES_PER_TOKEN as u64, max_total_mask_node_visits: (output * width) as u64,
        mask_limits: MaskWorkLimits::default(),
    } }
}
fn execute(prompt: &[u32], p: &JsonProgram, v: &Table, o: &JsonDecodeOptions,
    cap: usize, d: &mut Script) -> Result<Int8JsonRun, Int8JsonError> {
    drive(prompt, p, v, o, budget(prompt.len(), o.max_new_tokens, v.width(), cap), limits(cap), d, Ok)
}

#[test]
fn grammar_precedes_head_and_singleton_choices_still_score_and_forward() {
    let p = program(r#"{"type":"boolean"}"#);
    let v = table(&[b"", b"true", b"bad", b"also bad"]);
    let mut d = Script::new(vec![100.0, 1.0, 1000.0, 2000.0]);
    let out = execute(&[2, 2, 2], &p, &v, &options(2), 1, &mut d).unwrap();
    assert_eq!(d.seen, [2, 2, 2, 1]); assert_eq!(d.heads, [vec![1], vec![0]]);
    assert_eq!(out.output.token_ids, [1, 0]); assert_eq!(out.output.json, "true");
    assert_eq!(out.output.projected_logits, 2); assert_eq!(out.output.mask_node_visit_charge, 8);
    assert_eq!(out.model_work, Int8Work::for_sequence(0, 4, 2).unwrap());
    assert_eq!(out.execution, INT8_SPARSE_JSON_EXECUTION); assert!(!d.aborted);
}

#[test]
fn every_legal_choice_is_kept_and_a_small_cap_never_prunes() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true", b"false"]);
    let mut d = Script::new(vec![100.0, 1.0, 2.0]);
    let out = execute(&[1], &p, &v, &options(2), 2, &mut d).unwrap();
    assert_eq!(d.heads, [vec![1, 2], vec![0]]); assert_eq!(out.output.json, "false");
    let mut refused = Script::new(vec![100.0, 1.0, 2.0]);
    assert!(matches!(execute(&[1], &p, &v, &options(2), 1, &mut refused),
        Err(Int8JsonError::Decode(JsonDecodeError::BudgetExceeded(_)))));
    assert!(refused.heads.is_empty()); assert!(refused.aborted);
}

#[test]
fn accepting_numeric_prefix_still_competes_with_longer_legal_numbers() {
    let p = program(r#"{"type":"integer","enum":[1,10]}"#);
    let v = table(&[b"", b"1", b"0"]); let mut d = Script::new(vec![0.0, 2.0, 1.0]);
    let out = execute(&[1], &p, &v, &options(3), 2, &mut d).unwrap();
    assert_eq!(out.output.json, "10"); assert_eq!(out.output.token_ids, [1, 2, 0]);
    assert_eq!(d.heads, [vec![1], vec![0, 2], vec![0]]);
}

#[test]
fn exclusions_apply_before_row_count_but_eos_keeps_its_acceptance_rule() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true", b"false"]);
    let mut o = options(2); o.excluded_token_ids.extend([0, 1]);
    let mut d = Script::new(vec![20.0, 100.0, 1.0]);
    let out = execute(&[1], &p, &v, &o, 1, &mut d).unwrap();
    assert_eq!(out.output.token_ids, [2, 0]); assert_eq!(d.heads, [vec![2], vec![0]]);
    o.excluded_token_ids.insert(2);
    assert!(matches!(execute(&[1], &p, &v, &o, 1, &mut d),
        Err(Int8JsonError::Decode(JsonDecodeError::NoLegalToken))));
}

#[test]
fn source_masks_reject_high_scoring_off_source_strings() {
    let p = JsonProgram::compile_with_source(r#"{"type":"string","maxLength":16,"x-fnlp-source":"verbatim"}"#,
        "Alice é", CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
    let v = table(&[b"", br#""Mallory""#, "\"Alice é\"".as_bytes()]);
    let mut d = Script::new(vec![10.0, 100.0, 1.0]);
    let out = execute(&[1], &p, &v, &options(2), 1, &mut d).unwrap();
    assert_eq!(out.output.json, "\"Alice é\""); assert_eq!(d.heads[0], [2]);
    assert!(!p.source_fields(&out.output.json).unwrap().is_empty());
}

#[test]
fn finite_full_and_sparse_scripted_decodes_have_identical_semantics() {
    let fixtures: &[(&str, &[&[u8]], &[f32], usize)] = &[
        (r#"{"type":"boolean"}"#, &[b"", b"true", b"false", b"bad"], &[10.0, -0.0, 0.0, 100.0], 2),
        (r#"{"type":"integer","enum":[1,10]}"#, &[b"", b"1", b"0", b"bad"], &[0.0, 2.0, 1.0, 100.0], 3),
        (r#"{"type":"string","maxLength":8}"#, &[b"", "\"é😀\"".as_bytes(), b"bad"], &[10.0, 1.0, 100.0], 2),
    ];
    for &(schema, bytes, scores, count) in fixtures {
        let p = program(schema); let v = table(bytes); let o = options(count);
        let mut sparse = Script::new(scores.to_vec()); let mut dense = Script::new(scores.to_vec());
        let a = execute(&[1, 1], &p, &v, &o, v.width(), &mut sparse).unwrap();
        let b = super::super::drive(&[1, 1], &p, &v, &o, budget(2, count, v.width(), v.width()),
            &mut dense, Ok::<_, Int8JsonError>).unwrap();
        assert_eq!(a.output.token_ids, b.output.token_ids); assert_eq!(a.output.json, b.output.json);
        assert_eq!(a.output.forward_positions, b.output.forward_positions);
        assert_eq!(a.output.mask_node_visit_charge, b.output.mask_node_visit_charge);
        assert_eq!(sparse.seen, dense.seen); assert!(a.output.projected_logits < b.output.projected_logits);
    }
}

#[test]
fn exhaustive_small_masks_match_full_argmax_including_signed_zero() {
    let scores = [-0.0, 0.0, 3.0, 3.0, -5.0, 2.0];
    for bits in 0..64 {
        let mut mask = DenseTokenMask::empty(6);
        for id in 0..6 { if bits & (1 << id) != 0 { mask.set_legal(id).unwrap(); } }
        for excluded in 0..64 {
            let mut o = options(1);
            for id in 0..6 { if excluded & (1 << id) != 0 { o.excluded_token_ids.insert(id); } }
            for accepting in [false, true] {
                let full = super::super::select(&scores, &mask, accepting, &o);
                let sparse = legal_rows(&mask, accepting, &o, 6, &mut Control::default(), 0).and_then(|rows| {
                    let logits: Vec<_> = rows.iter().map(|&id| scores[id as usize]).collect(); select(&rows, &logits)
                });
                assert_eq!(sparse.ok(), full.ok());
            }
        }
    }
}

#[test]
fn only_requested_logits_are_checked_and_malformed_selected_outputs_fail() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true", b"bad"]);
    let mut d = Script::new(vec![0.0, 1.0, f32::NAN]);
    assert!(execute(&[1], &p, &v, &options(2), 1, &mut d).is_ok());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut d = Script::new(vec![0.0, bad, 1.0]);
        assert!(matches!(execute(&[1], &p, &v, &options(2), 1, &mut d),
            Err(Int8JsonError::Decode(JsonDecodeError::InvalidLogits))));
        assert!(d.aborted);
    }
    let mut d = Script::new(vec![0.0, 1.0, 1.0]); d.short_head = true;
    assert!(execute(&[1], &p, &v, &options(2), 1, &mut d).is_err()); assert!(d.aborted);
    assert!(select(&[2, 1], &[0.0, 1.0]).is_err()); assert!(select(&[1, 1], &[0.0, 1.0]).is_err());
}

#[test]
fn all_work_axes_are_admitted_before_prefill_without_full_head_overcharging() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true", b"bad", b"worse"]);
    let b = budget(2, 2, 4, 1);
    assert_eq!(b.json.max_projected_logits, 2); assert_eq!(b.json.max_total_mask_node_visits, 8);
    for axis in 0..7 {
        let mut b = b;
        match axis {
            0 => b.json.max_forward_positions -= 1, 1 => b.json.max_projected_logits -= 1,
            2 => b.json.max_total_mask_node_visits -= 1, 3 => b.native.max_forward_positions -= 1,
            4 => b.native.max_attention_pairs -= 1, 5 => b.native.max_projection_work.dot_products -= 1,
            _ => b.native.max_projection_work.multiply_accumulates -= 1,
        }
        let mut d = Script::new(vec![10.0, 1.0, 100.0, 100.0]);
        let result: Result<Int8JsonRun, Int8JsonError> = drive(&[1, 1], &p, &v, &options(2), b, limits(1), &mut d, Ok);
        assert!(result.is_err()); assert!(d.seen.is_empty()); assert!(d.heads.is_empty());
    }
}

#[test]
fn bad_geometry_and_ids_cannot_enter_native_work() {
    assert!(limits(0).validate().is_err()); assert!(limits(NANBEIGE_VOCAB_SIZE + 1).validate().is_err());
    assert!(planned_work(usize::MAX, 2, limits(1)).is_err());
    assert!(planned_work(DEFAULT_ADMITTED_CONTEXT_CAP, 2, limits(1)).is_err());
    let v = table(&[b"", b"true"]); let b = budget(1, 2, 2, 1);
    assert!(preflight(&[], &v, &options(2), b, limits(1)).is_err());
    assert!(preflight(&[2], &v, &options(2), b, limits(1)).is_err());
    assert!(preflight(&[1], &v, &options(2), b, limits(3)).is_err());
    let mut o = options(2); o.excluded_token_ids.insert(2);
    assert!(preflight(&[1], &v, &o, b, limits(1)).is_err());
}

#[test]
fn prefill_selection_and_late_cancellation_abort_the_same_driver() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    for stage in 0..3 {
        let mut d = Script::new(vec![10.0, 1.0]);
        match stage { 0 => d.control.prefill = Some(1), 1 => d.control.step = Some(0), _ => d.control.step = Some(2) }
        let error = execute(&[1, 1], &p, &v, &options(2), 1, &mut d).unwrap_err();
        assert!(error.cancellation().is_some()); assert!(d.aborted);
        if stage == 0 { assert_eq!(d.seen, [1]); assert!(d.heads.is_empty()); }
        if stage == 2 { assert_eq!(d.heads.len(), 2); }
    }
}

#[test]
fn legal_row_traversal_has_bounded_cancellation_checkpoints() {
    let mut mask = DenseTokenMask::empty(4096);
    for id in 1..4096 { mask.set_legal(id).unwrap(); }
    let mut control = Control { cancel_poll: Some(1), ..Control::default() };
    assert!(legal_rows(&mask, false, &options(1), 4096, &mut control, 0).unwrap_err().cancellation().is_some());
    assert_eq!(control.polls, 2);
}

#[test]
fn eosless_budget_exhaustion_native_errors_and_bad_receipts_never_succeed() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    let mut d = Script::new(vec![0.0, 1.0]);
    assert!(execute(&[1], &p, &v, &options(1), 1, &mut d).is_err()); assert!(d.aborted);
    for fault in 0..3 {
        let mut d = Script::new(vec![0.0, 1.0]);
        match fault { 0 => d.fail_append = true, 1 => d.fail_head = true, _ => d.corrupt_work = true }
        assert!(execute(&[1], &p, &v, &options(2), 1, &mut d).is_err()); assert!(d.aborted);
    }
}

#[test]
fn task_finalization_failure_aborts_before_session_release() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    let mut d = Script::new(vec![0.0, 1.0]);
    let result: Result<(), Int8JsonError> = drive(&[1], &p, &v, &options(2), budget(1, 2, 2, 1),
        limits(1), &mut d, |_| Err(Int8JsonError::WorkMismatch));
    assert!(result.is_err()); assert!(d.aborted); assert_eq!(d.heads.len(), 2);
}
