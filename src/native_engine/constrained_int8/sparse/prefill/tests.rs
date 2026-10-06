//! Synthetic driver integration tests, not real-model numerical qualification.
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
struct Control { prefill: Option<usize>, step: Option<usize>, polls: usize, cancel_poll: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, step: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1;
        (self.step == Some(step) || self.cancel_poll == Some(self.polls)).then_some(DecodeCancellationKind::Deadline)
    }
    fn prefill_checkpoint(&mut self, position: usize) -> Option<DecodeCancellationKind> {
        (self.prefill == Some(position)).then_some(DecodeCancellationKind::Deadline)
    }
}
struct Model {
    control: Control, logits: Vec<f32>, seen: Vec<u32>, heads: usize, projected: usize, groups: Vec<usize>,
    aborted: bool, fail_at: Option<usize>, forged_work: bool,
}
impl Model {
    fn new(logits: &[f32]) -> Self {
        Self { control: Control::default(), logits: logits.to_vec(), seen: Vec::new(), heads: 0, projected: 0,
            groups: Vec::new(), aborted: false, fail_at: None, forged_work: false }
    }
}
impl Driver for Model {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn append(&mut self, token: u32) -> Result<(), Int8JsonError> {
        if self.fail_at == Some(self.seen.len()) { return Err(StrictInt8Error::Primitive.into()); }
        self.seen.push(token); Ok(())
    }
    fn logits(&mut self, rows: &[u32]) -> Result<Vec<f32>, Int8JsonError> {
        self.heads += 1; self.projected += rows.len();
        Ok(rows.iter().map(|&id| self.logits[id as usize]).collect())
    }
    fn work(&self) -> Int8Work {
        let mut work = Int8Work::for_sequence(0, self.seen.len(), self.projected).unwrap();
        if self.forged_work { work.attention_pairs += 1; }
        work
    }
    fn abort(&mut self) { self.aborted = true; }
}
impl PromptDriver for Model {
    fn append_layer_major(&mut self, prompt: &[u32], limits: Int8PrefillLimits) -> Result<(), Int8JsonError> {
        for chunk in prompt.chunks(limits.max_batch_rows) {
            self.groups.push(chunk.len());
            for &token in chunk {
                if let Some(cause) = self.control.checkpoint(usize::MAX) {
                    return Err(StrictInt8Error::Cancelled(cause).into());
                }
                self.append(token)?;
            }
        }
        Ok(())
    }
}
fn sparse_limits() -> Int8JsonSparseLimits { Int8JsonSparseLimits { max_rows_per_step: 1 } }
fn limits(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
fn options(n: usize) -> JsonDecodeOptions {
    JsonDecodeOptions { max_new_tokens: n, eos_token_id: 0, excluded_token_ids: BTreeSet::new() }
}
fn budget(prompt: usize, output: usize, width: usize) -> Int8JsonBudget {
    let work = planned_work(prompt, output, sparse_limits()).unwrap();
    Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: 8192 * KV_BYTES_PER_TOKEN as u64, max_total_mask_node_visits: (output * width) as u64,
        mask_limits: MaskWorkLimits::default(),
    } }
}
fn fixture() -> (JsonProgram, Table, Model) {
    (JsonProgram::compile(r#"{"type":"boolean"}"#, CompileLimits::default()).unwrap(),
        Table(vec![Vec::new(), b"true".to_vec(), b"malformed".to_vec()]), Model::new(&[100.0, 1.0, 1000.0]))
}
fn execute(prompt: &[u32], program: &JsonProgram, vocabulary: &Table, options: &JsonDecodeOptions,
    budget: Int8JsonBudget, prefill: Int8PrefillLimits, model: &mut Model) -> Result<Int8JsonRun, Int8JsonError> {
    drive_layer_major(prompt, program, vocabulary, options, budget, sparse_limits(), prefill, model, Ok)
}

#[test]
fn prompt_grouping_preserves_exact_output_and_work_including_tail_morsels() {
    let (p, v, mut serial) = fixture();
    let prompt = [2; 5]; let o = options(2); let b = budget(5, 2, 3);
    let expected: Int8JsonRun = drive(&prompt, &p, &v, &o, b, sparse_limits(), &mut serial, Ok::<_, Int8JsonError>).unwrap();
    for width in [1, 2, 4, 64] {
        let (_, _, mut grouped) = fixture();
        let actual = execute(&prompt, &p, &v, &o, b, limits(width), &mut grouped).unwrap();
        assert_eq!(actual, expected); assert_eq!(grouped.seen, serial.seen);
        assert_eq!(grouped.heads, 2); assert!(!grouped.aborted);
        assert_eq!(grouped.groups, prompt.chunks(width).map(|c| c.len()).collect::<Vec<_>>());
        assert_eq!(actual.output.token_ids, vec![1, 0]);
        assert_eq!(actual.model_work, Int8Work::for_sequence(0, 6, 2).unwrap());
    }
}

#[test]
fn source_grammar_still_rejects_higher_scoring_off_source_multibyte_values() {
    let p = JsonProgram::compile_with_source(r#"{"type":"string","maxLength":16,"x-fnlp-source":"verbatim"}"#,
        "Éve met 上海", CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
    let v = Table(vec![Vec::new(), br#""Mallory""#.to_vec(), "\"Éve\"".as_bytes().to_vec()]);
    let mut model = Model::new(&[100.0, 1000.0, 1.0]);
    let result = execute(&[1; 3], &p, &v, &options(2), budget(3, 2, 3), limits(2), &mut model).unwrap();
    assert_eq!(result.output.json, "\"Éve\"");
    assert!(!p.source_fields(&result.output.json).unwrap().is_empty());
}

#[test]
fn all_work_axes_refuse_before_prompt_or_head_execution() {
    let (p, v, _) = fixture();
    for axis in 0..7 {
        let (_, _, mut model) = fixture(); let mut b = budget(5, 2, 3);
        match axis {
            0 => b.json.max_forward_positions -= 1, 1 => b.json.max_projected_logits -= 1,
            2 => b.json.max_total_mask_node_visits -= 1, 3 => b.native.max_forward_positions -= 1,
            4 => b.native.max_attention_pairs -= 1, 5 => b.native.max_projection_work.dot_products -= 1,
            _ => b.native.max_projection_work.multiply_accumulates -= 1,
        }
        assert!(execute(&[2; 5], &p, &v, &options(2), b, limits(4), &mut model).is_err());
        assert!(model.groups.is_empty()); assert!(model.seen.is_empty()); assert_eq!(model.heads, 0);
    }
}

#[test]
fn invalid_row_counts_and_short_scratch_fail_before_driver_entry() {
    let (p, v, _) = fixture(); let mut short = limits(4); short.max_extra_scratch_bytes -= 1;
    for limit in [short, Int8PrefillLimits { max_batch_rows: 0, max_extra_scratch_bytes: u64::MAX },
        Int8PrefillLimits { max_batch_rows: 65, max_extra_scratch_bytes: u64::MAX }] {
        let (_, _, mut model) = fixture();
        assert!(execute(&[2; 5], &p, &v, &options(2), budget(5, 2, 3), limit, &mut model).is_err());
        assert!(model.groups.is_empty()); assert!(model.seen.is_empty()); assert_eq!(model.heads, 0);
        assert!(!model.aborted);
    }
}

#[test]
fn cancellation_before_and_during_prefill_aborts_without_a_projection() {
    let (p, v, _) = fixture();
    for before in [false, true] {
        let (_, _, mut model) = fixture();
        if before { model.control.prefill = Some(2); } else { model.control.cancel_poll = Some(3); }
        let error = execute(&[2; 5], &p, &v, &options(2), budget(5, 2, 3), limits(4), &mut model).unwrap_err();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
        assert!(model.aborted); assert_eq!(model.heads, 0);
        assert_eq!(model.seen.len(), if before { 0 } else { 2 });
    }
}

#[test]
fn native_tail_failure_never_enters_the_decode_loop() {
    let (p, v, mut model) = fixture(); model.fail_at = Some(4);
    assert!(execute(&[2; 5], &p, &v, &options(2), budget(5, 2, 3), limits(4), &mut model).is_err());
    assert!(model.aborted); assert_eq!(model.heads, 0); assert_eq!(model.groups, vec![4, 1]);
}

#[test]
fn finalization_failure_and_late_cancellation_abort_the_owned_driver() {
    let (p, v, _) = fixture();
    for cancel in [false, true] {
        let (_, _, mut model) = fixture(); if cancel { model.control.step = Some(2); }
        let mut finalized = false;
        let result: Result<(), Int8JsonError> = drive_layer_major(&[2; 5], &p, &v, &options(2),
            budget(5, 2, 3), sparse_limits(), limits(4), &mut model, |_| {
                finalized = true;
                if cancel { Ok(()) } else { Err(JsonDecodeError::IndependentValidation.into()) }
            });
        assert!(finalized); assert!(result.is_err()); assert!(model.aborted);
        if cancel { assert_eq!(result.unwrap_err().cancellation(), Some(DecodeCancellationKind::Deadline)); }
    }
}

#[test]
fn exhausted_eos_budget_and_forged_native_work_are_not_successes() {
    let (p, v, _) = fixture();
    for forged in [false, true] {
        let (_, _, mut model) = fixture(); model.forged_work = forged;
        let tokens = if forged { 2 } else { 1 };
        let error = execute(&[2; 5], &p, &v, &options(tokens), budget(5, tokens, 3), limits(4), &mut model).unwrap_err();
        assert!(matches!(error, Int8JsonError::WorkMismatch | Int8JsonError::Decode(JsonDecodeError::BudgetExceeded(_))));
        assert!(model.aborted);
    }
}

#[test]
fn grouping_does_not_turn_the_row_ceiling_into_top_k_pruning() {
    let (p, _, mut model) = fixture();
    let v = Table(vec![Vec::new(), b"true".to_vec(), b"false".to_vec()]);
    let error = execute(&[2; 5], &p, &v, &options(2), budget(5, 2, 3), limits(4), &mut model).unwrap_err();
    assert!(matches!(error, Int8JsonError::Decode(JsonDecodeError::BudgetExceeded(_))));
    assert!(model.aborted); assert_eq!(model.heads, 0); assert_eq!(model.projected, 0);
    assert_eq!(model.seen.len(), 5);
}
