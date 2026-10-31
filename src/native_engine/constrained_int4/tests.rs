//! Executable grammar/driver contracts with synthetic logits, not model parity.
use super::*;
use std::collections::BTreeSet;
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{NumericsProfile, Sha256Digest},
    grammar::{CompileLimits, runtime::SourceRuntimeLimits},
    native_engine::{portable_int8::int4::convert::GROUPED_INT4_RECIPE,
        quant_algebra::MAX_MODEL_K, strict_int4::STRICT_INT4_EXECUTION},
};
mod prefill;

const GROUP: usize = 127;
struct Table(Vec<Vec<u8>>);
impl Vocabulary for Table {
    fn width(&self) -> usize { self.0.len() }
    fn bytes(&self, id: u32) -> Option<&[u8]> { self.0.get(id as usize).map(Vec::as_slice) }
    fn mask_charge(&self, _: MaskWorkLimits) -> usize { self.width() }
    fn mask<C: DecodeStepControl>(&self, state: &JsonState<'_>, _: MaskWorkLimits, _: &mut C, _: usize)
        -> Result<DenseTokenMask, Int4JsonError> {
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
        (self.prefill == Some(position)).then_some(DecodeCancellationKind::CostBudget)
    }
}
struct Script {
    control: Control, row: Vec<f32>, seen: Vec<u32>, heads: usize,
    group: usize, aborted: bool, fail_append: bool, corrupt_work: bool,
    prompt_calls: usize, prompt_morsels: Vec<usize>,
}
impl Script {
    fn new(row: Vec<f32>) -> Self {
        Self { row, control: Control::default(), seen: Vec::new(), heads: 0,
            group: GROUP, aborted: false, fail_append: false, corrupt_work: false,
            prompt_calls: 0, prompt_morsels: Vec::new() }
    }
}
impl Driver for Script {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn append(&mut self, token: u32) -> Result<(), Int4JsonError> {
        if self.fail_append { return Err(StrictInt4Error::Primitive.into()); }
        self.seen.push(token); Ok(())
    }
    fn logits(&mut self) -> Result<Vec<f32>, Int4JsonError> { self.heads += 1; Ok(self.row.clone()) }
    fn work(&self) -> Int4Work {
        let mut work = Int4Work::for_sequence(self.group, 0, self.seen.len(), self.heads * self.row.len()).unwrap();
        if self.corrupt_work { work.projections.dot_products += 1; }
        work
    }
    fn abort(&mut self) { self.aborted = true; }
}
fn options(n: usize) -> JsonDecodeOptions {
    JsonDecodeOptions { max_new_tokens: n, eos_token_id: 0, excluded_token_ids: BTreeSet::new() }
}
fn program(schema: &str) -> JsonProgram { JsonProgram::compile(schema, CompileLimits::default()).unwrap() }
fn table(values: &[&[u8]]) -> Table { Table(values.iter().map(|bytes| bytes.to_vec()).collect()) }
fn budget(prompt: usize, output: usize, width: usize) -> Int4JsonBudget {
    let work = work_for_width(GROUP, prompt, output, width).unwrap();
    Int4JsonBudget { native: Int4RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: 8192 * KV_BYTES_PER_TOKEN as u64, max_total_mask_node_visits: (output * width) as u64,
        mask_limits: MaskWorkLimits::default(),
    } }
}
fn execute(prompt: &[u32], p: &JsonProgram, v: &Table, o: &JsonDecodeOptions, b: Int4JsonBudget, d: &mut Script)
    -> Result<Int4JsonRun, Int4JsonError> {
    drive(GROUP, prompt, p, v, o, b, d, Ok)
}

#[test]
fn only_final_prefill_position_projects_and_content_is_fed_before_scored_eos() {
    let p = program(r#"{"type":"boolean"}"#);
    let v = table(&[b"", b"true", b"malformed"]);
    let mut d = Script::new(vec![100.0, 1.0, 1000.0]);
    let out = execute(&[2, 2, 2], &p, &v, &options(2), budget(3, 2, 3), &mut d).unwrap();
    assert_eq!(d.seen, vec![2, 2, 2, 1]); assert_eq!(d.heads, 2); assert!(!d.aborted);
    assert_eq!(out.output.json, "true"); assert_eq!(out.output.token_ids, vec![1, 0]);
    assert_eq!(out.output.forward_positions, 4); assert_eq!(out.output.projected_logits, 6);
    assert_eq!(out.output.mask_node_visit_charge, 6);
    assert_eq!(out.model_work, Int4Work::for_sequence(GROUP, 0, 4, 6).unwrap());
    assert_eq!(out.group_width, GROUP); assert_eq!(out.output.numerics_profile, STRICT_INT4_PROFILE);
    assert_eq!(out.execution, INT4_JSON_EXECUTION);
}

#[test]
fn source_product_excludes_higher_scoring_off_source_value() {
    let p = JsonProgram::compile_with_source(r#"{"type":"string","maxLength":16,"x-fnlp-source":"verbatim"}"#,
        "Alice and Alice", CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
    let v = table(&[b"", br#""Mallory""#, br#""Alice""#]);
    let mut d = Script::new(vec![10.0, 100.0, 1.0]);
    let out = execute(&[1], &p, &v, &options(2), budget(1, 2, 3), &mut d).unwrap();
    assert_eq!(out.output.json, r#""Alice""#);
    assert!(!p.source_fields(&out.output.json).unwrap().is_empty());
}

#[test]
fn byte_fallback_tokens_can_complete_one_multibyte_json_scalar() {
    let p = program(r#"{"type":"string","maxLength":1,"enum":["é"]}"#);
    let v = table(&[b"", b"\"", &[0xc3], &[0xa9]]);
    let mut d = Script::new(vec![100.0, 3.0, 2.0, 1.0]);
    let out = execute(&[1], &p, &v, &options(5), budget(1, 5, 4), &mut d).unwrap();
    assert_eq!(out.output.json, "\"é\""); assert_eq!(out.output.token_ids, vec![1, 2, 3, 1, 0]);
}

#[test]
fn tokens_can_cross_multiple_schema_boundaries() {
    let p = program(r#"{"type":"object","properties":{"x":{"type":"string","maxLength":8}},"required":["x"],"additionalProperties":false}"#);
    let value = r#"{"x":"é😀"}"#;
    let v = table(&[b"", value.as_bytes()]); let mut d = Script::new(vec![0.0, 1.0]);
    let out = execute(&[1], &p, &v, &options(2), budget(1, 2, 2), &mut d).unwrap();
    assert_eq!(out.output.json.as_bytes(), value.as_bytes());
}

#[test]
fn accepting_numeric_prefix_does_not_force_shorter_number_or_eos() {
    let p = program(r#"{"type":"integer","enum":[1,10]}"#);
    let v = table(&[b"", b"1", b"0"]); let mut d = Script::new(vec![0.0, 2.0, 1.0]);
    let out = execute(&[1], &p, &v, &options(3), budget(1, 3, 3), &mut d).unwrap();
    assert_eq!(out.output.json, "10"); assert_eq!(out.output.token_ids, vec![1, 2, 0]);
}

#[test]
fn complete_json_without_an_eos_budget_is_not_success() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    let mut d = Script::new(vec![0.0, 1.0]);
    assert!(matches!(execute(&[1], &p, &v, &options(1), budget(1, 1, 2), &mut d),
        Err(Int4JsonError::Decode(JsonDecodeError::BudgetExceeded(_)))));
    assert!(d.aborted); assert_eq!(d.seen, vec![1]); assert_eq!(d.heads, 1);
}

#[test]
fn excluded_controls_and_terminal_eos_keep_separate_legality() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true", b"false"]);
    let mut o = options(2); o.excluded_token_ids.extend([0, 1]);
    let mut d = Script::new(vec![20.0, 100.0, 1.0]);
    let out = execute(&[1], &p, &v, &o, budget(1, 2, 3), &mut d).unwrap();
    assert_eq!(out.output.json, "false"); assert_eq!(out.output.token_ids, vec![2, 0]);
    o.excluded_token_ids.insert(2);
    assert!(matches!(execute(&[1], &p, &v, &o, budget(1, 2, 3), &mut Script::new(vec![0.0; 3])),
        Err(Int4JsonError::Decode(JsonDecodeError::NoLegalToken))));
}

#[test]
fn nonfinite_logits_cannot_hide_behind_the_mask() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true", b"bad"]);
    for row in [vec![0.0, 1.0, f32::NAN], vec![0.0, 1.0, f32::INFINITY],
        vec![0.0, 1.0, f32::NEG_INFINITY], vec![0.0, 1.0]] {
        let mut d = Script::new(row);
        assert!(matches!(execute(&[1], &p, &v, &options(2), budget(1, 2, 3), &mut d),
            Err(Int4JsonError::Decode(JsonDecodeError::InvalidLogits))));
        assert!(d.aborted);
    }
}

#[test]
fn first_token_id_wins_native_ties_including_signed_zero() {
    let mut mask = DenseTokenMask::empty(3); mask.set_legal(1).unwrap(); mask.set_legal(2).unwrap();
    assert_eq!(select(&[100.0, -0.0, 0.0], &mask, false, &options(2)).unwrap(), 1);
    assert_eq!(select(&[-0.0, 0.0, 0.0], &mask, true, &options(2)).unwrap(), 0);
}

#[test]
fn every_work_axis_is_preflighted_before_native_work() {
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
        assert!(d.seen.is_empty()); assert_eq!(d.heads, 0);
    }
}

#[test]
fn request_refusals_include_all_ids_and_bounded_mask_work() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    for axis in 0..5 {
        let mut o = options(2); let mut b = budget(1, 2, 2); let mut prompt = vec![1];
        match axis { 0 => prompt[0] = 2, 1 => o.eos_token_id = 2,
            2 => { o.excluded_token_ids.insert(2); },
            3 => b.json.mask_limits.max_trie_node_visits = 0,
            _ => b.json.mask_limits.checkpoint_interval_nodes = 0 }
        let mut d = Script::new(vec![0.0, 1.0]);
        assert!(execute(&prompt, &p, &v, &o, b, &mut d).is_err());
        assert!(d.seen.is_empty()); assert_eq!(d.heads, 0);
    }
}

#[test]
fn kv_prices_resident_capacity_and_work_prices_actual_group_tails() {
    let work = planned_work(GROUP, 1, 2).unwrap(); let all = 8 * KV_BYTES_PER_TOKEN as u64;
    assert!(check_capacity(work, 8, all).is_ok());
    assert!(check_capacity(work, 8, all - 1).is_err()); assert!(check_capacity(work, 1, all).is_err());
    for (group, prompt, output) in [(0, 1, 1), (MAX_MODEL_K + 1, 1, 1), (GROUP, 0, 1),
        (GROUP, 1, 0), (GROUP, usize::MAX, 2), (GROUP, DEFAULT_ADMITTED_CONTEXT_CAP, 2)] {
        assert!(planned_work(group, prompt, output).is_err());
    }
    let small = planned_work(16, 17, 3).unwrap(); let large = planned_work(32, 17, 3).unwrap();
    assert_eq!(small.projected_logits, 3 * NANBEIGE_VOCAB_SIZE as u64);
    assert_eq!(small.projections.multiply_accumulates, large.projections.multiply_accumulates);
    assert!(small.projections.dot_products > large.projections.dot_products);
}

#[test]
fn cancellation_during_prefill_and_decode_aborts_without_partial_success() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    for prefill in [true, false] {
        let mut d = Script::new(vec![0.0, 1.0]);
        if prefill { d.control.prefill = Some(1); } else { d.control.step = Some(1); }
        let error = execute(&[1, 1], &p, &v, &options(2), budget(2, 2, 2), &mut d).unwrap_err();
        assert_eq!(error.cancellation(), Some(if prefill { DecodeCancellationKind::CostBudget } else { DecodeCancellationKind::Deadline }));
        assert!(d.aborted);
        if prefill { assert_eq!(d.seen.len(), 1); assert_eq!(d.heads, 0); }
    }
}

#[test]
fn native_failure_or_wrong_group_accounting_invalidates_the_session() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    for axis in 0..3 {
        let mut d = Script::new(vec![0.0, 1.0]);
        match axis { 0 => d.fail_append = true, 1 => d.corrupt_work = true, _ => d.group = 32 }
        let error = execute(&[1], &p, &v, &options(2), budget(1, 2, 2), &mut d).unwrap_err();
        assert!(matches!(error, Int4JsonError::Native(_) | Int4JsonError::WorkMismatch)); assert!(d.aborted);
    }
}

#[test]
fn finalizer_failure_aborts_before_session_lifetime_ends() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    let mut d = Script::new(vec![0.0, 1.0]); let mut finalized = false;
    let result: Result<(), Int4JsonError> = drive(GROUP, &[1], &p, &v, &options(2), budget(1, 2, 2), &mut d, |run| {
        finalized = true; assert_eq!(run.output.json, "true");
        Err(JsonDecodeError::IndependentValidation.into())
    });
    assert!(result.is_err()); assert!(finalized); assert!(d.aborted);
}

#[test]
fn late_cancellation_after_successful_finalization_still_has_no_result() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    let mut d = Script::new(vec![0.0, 1.0]); d.control.step = Some(2);
    let mut finalized = false;
    let result: Result<(), Int4JsonError> = drive(GROUP, &[1], &p, &v, &options(2), budget(1, 2, 2), &mut d, |_| {
        finalized = true; Ok(())
    });
    assert!(finalized); assert!(d.aborted); assert_eq!(result.unwrap_err().cancellation(), Some(DecodeCancellationKind::Deadline));
}

#[test]
fn complete_candidate_output_has_deterministic_canonical_replay() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    let result = || execute(&[1], &p, &v, &options(2), budget(1, 2, 2), &mut Script::new(vec![0.0, 1.0])).unwrap();
    assert_eq!(crate::canonjson::canonical_bytes(&result()).unwrap(), crate::canonjson::canonical_bytes(&result()).unwrap());
}

fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"synthetic-int4-constrained-model");
    let packing = Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
        grouped_int4::INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, GROUP)).unwrap());
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: grouped_int4::INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={GROUP}"),
        packing_set_digest: packing, tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "fixture".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}

#[test]
fn profile_contract_cannot_relabel_another_model_partition_or_execution() {
    check_profile(&identity(), GROUP).unwrap();
    for axis in 0..12 {
        let mut id = identity();
        match axis { 0 => id.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            1 => id.numerics_profile = NumericsProfile::HfBf16Eager,
            2 => id.artifact_format = "fnlpq-v1".to_owned(), 3 => id.quant_recipe.push('x'),
            4 => id.source_revision.push('x'), 5 => id.backend_semantic_version.push('x'),
            6 => id.packing_set_digest = Sha256Digest::of_bytes(b"wrong-packing"),
            7 => id.kv_dtype = "f32".to_owned(), 8 => id.logical_model_digest = Sha256Digest::of_bytes(b"wrong-content"),
            9 => id.thinking_mode = ThinkingMode::Enabled, 10 => id.tool_mode = ToolMode::Xml,
            _ => id.tool_mode = ToolMode::Json }
        assert!(matches!(check_profile(&id, GROUP), Err(Int4JsonError::ModelIdentity)));
    }
    assert!(check_profile(&identity(), 32).is_err());
    assert!(check_profile(&identity(), 0).is_err());
}
