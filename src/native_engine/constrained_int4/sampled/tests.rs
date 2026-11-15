//! Synthetic driver contracts, not native model-quality or parity evidence.
use super::*;
use std::collections::BTreeSet;
use crate::{
    execution_identity::{NumericsProfile, Sha256Digest},
    grammar::{CompileLimits, runtime::SourceRuntimeLimits},
};

const GROUP: usize = 127;

struct Table { bytes: Vec<Vec<u8>>, lie: bool }
impl Table {
    fn new(bytes: &[&[u8]]) -> Self {
        Self { bytes: bytes.iter().map(|bytes| bytes.to_vec()).collect(), lie: false }
    }
}
impl Vocabulary for Table {
    fn width(&self) -> usize { self.bytes.len() }
    fn bytes(&self, id: u32) -> Option<&[u8]> { self.bytes.get(id as usize).map(Vec::as_slice) }
    fn mask_charge(&self, _: MaskWorkLimits) -> usize { self.width() }
    fn mask<C: DecodeStepControl>(&self, state: &JsonState<'_>, _: MaskWorkLimits, _: &mut C, _: usize)
        -> Result<DenseTokenMask, Int4JsonError> {
        let mut mask = DenseTokenMask::empty(self.width());
        for (id, bytes) in self.bytes.iter().enumerate() {
            if !bytes.is_empty() && (self.lie || state.clone().consume_bytes(bytes)) {
                mask.set_legal(id as u32).unwrap();
            }
        }
        Ok(mask)
    }
}

#[derive(Default)]
struct Control { prefill: Option<usize>, step: Option<usize>, cause: Option<DecodeCancellationKind> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, step: usize) -> Option<DecodeCancellationKind> {
        if self.step == Some(step) { self.cause } else { None }
    }
    fn prefill_checkpoint(&mut self, index: usize) -> Option<DecodeCancellationKind> {
        if self.prefill == Some(index) { self.cause } else { None }
    }
}

struct Script {
    control: Control, row: Vec<f32>, seen: Vec<u32>, heads: usize, aborted: bool, corrupt: bool,
    prompt_calls: usize, prompt_morsels: Vec<usize>,
}
impl Script {
    fn new(row: Vec<f32>) -> Self {
        Self { control: Control::default(), row, seen: Vec::new(), heads: 0, aborted: false, corrupt: false,
            prompt_calls: 0, prompt_morsels: Vec::new() }
    }
}
impl Driver for Script {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn append(&mut self, token: u32) -> Result<(), Int4JsonError> { self.seen.push(token); Ok(()) }
    fn logits(&mut self) -> Result<Vec<f32>, Int4JsonError> { self.heads += 1; Ok(self.row.clone()) }
    fn work(&self) -> Int4Work {
        let mut work = Int4Work::for_sequence(GROUP, 0, self.seen.len(), self.heads * self.row.len()).unwrap();
        if self.corrupt { work.projections.dot_products += 1; }
        work
    }
    fn abort(&mut self) { self.aborted = true; }
}
impl prefill::PrefillDriver for Script {
    fn append_layer_major(&mut self, prompt: &[u32], limits: Int4PrefillLimits) -> Result<(), Int4JsonError> {
        self.prompt_calls += 1;
        for chunk in prompt.chunks(limits.max_batch_rows) {
            self.prompt_morsels.push(chunk.len());
            for &token in chunk {
                if let Some(cause) = self.control.prefill_checkpoint(self.seen.len()) {
                    return Err(StrictInt4Error::Cancelled(cause).into());
                }
                self.append(token)?;
            }
        }
        Ok(())
    }
}

fn sampling() -> JsonSamplingOptions {
    JsonSamplingOptions { effective_seed: [37; 32], temperature_milli: 1000,
        top_k: None, top_p_ppm: 1_000_000, sample_index: 0 }
}
fn options(tokens: usize) -> JsonDecodeOptions {
    JsonDecodeOptions { max_new_tokens: tokens, eos_token_id: 0, excluded_token_ids: BTreeSet::new() }
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"sampled-json-synthetic-identity");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "sentiment-sampled-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "fixture".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: JSON_SAMPLER_VERSION.to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None }
}
fn program(schema: &str) -> JsonProgram { JsonProgram::compile(schema, CompileLimits::default()).unwrap() }
fn budget(prompt: usize, tokens: usize, width: usize) -> Int4JsonBudget {
    let work = work_for_width(GROUP, prompt, tokens, width).unwrap();
    Int4JsonBudget { native: Int4RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: 8192 * KV_BYTES_PER_TOKEN as u64,
        max_total_mask_node_visits: (tokens * width) as u64, mask_limits: MaskWorkLimits::default(),
    } }
}
#[allow(clippy::too_many_arguments)]
fn execute(prompt: &[u32], p: &JsonProgram, v: &Table, o: &JsonDecodeOptions, s: &JsonSamplingOptions,
    b: Int4JsonBudget, d: &mut Script) -> Result<Int4JsonRun, Int4JsonError> {
    preflight(GROUP, prompt, v, o, b)?;
    let mut select = Sampled::new(&identity(), prompt, p, o, s, json_sampling_workspace_bytes().unwrap())?;
    drive_selected(GROUP, prompt, p, v, o, b, d, &mut select, Ok)
}
fn prefill_limits(rows: usize) -> Int4PrefillLimits {
    Int4PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
#[allow(clippy::too_many_arguments)]
fn execute_layer_major(prompt: &[u32], p: &JsonProgram, v: &Table, o: &JsonDecodeOptions,
    s: &JsonSamplingOptions, b: Int4JsonBudget, d: &mut Script, limits: Int4PrefillLimits)
    -> Result<Int4JsonRun, Int4JsonError> {
    let mut select = Sampled::new(&identity(), prompt, p, o, s, json_sampling_workspace_bytes().unwrap())?;
    prefill::drive_layer_major_selected(GROUP, prompt, p, v, o, b, d, &mut select, limits, Ok)
}

#[test]
fn sampled_json_keeps_exact_mask_scored_eos_and_final_prefill_work() {
    let p = program(r#"{"type":"boolean"}"#);
    let v = Table::new(&[b"", b"true", b"false", b"invalid", b"<think>"]);
    let mut o = options(2); o.excluded_token_ids.extend([0, 1]);
    let mut d = Script::new(vec![100.0, 1000.0, 1.0, 10_000.0, 10_000.0]);
    let run = execute(&[3, 3, 3], &p, &v, &o, &sampling(), budget(3, 2, 5), &mut d).unwrap();
    assert_eq!(run.output.json, "false");
    assert_eq!(run.output.token_ids, vec![2, 0]);
    assert_eq!(d.seen, vec![3, 3, 3, 2]);
    assert_eq!(d.heads, 2);
    assert!(!d.aborted);
    assert_eq!(run.execution, INT4_JSON_SAMPLED_EXECUTION);
    assert_eq!(run.output.mask_node_visit_charge, 10);
    assert_eq!(run.model_work, Int4Work::for_sequence(GROUP, 0, 4, 10).unwrap());
    p.validate_json(&run.output.json).unwrap();
}

#[test]
fn sample_addresses_replay_independently_of_other_samples_or_call_order() {
    let p = program(r#"{"type":"boolean"}"#);
    let v = Table::new(&[b"", b"true", b"false"]);
    let o = options(2); let b = budget(1, 2, 3);
    let one = |index| {
        let s = JsonSamplingOptions { sample_index: index, ..sampling() };
        execute(&[1], &p, &v, &o, &s, b, &mut Script::new(vec![100.0, 0.0, 0.0])).unwrap()
    };
    let forward: Vec<_> = (0..24).map(one).collect();
    let reverse: Vec<_> = (0..24).rev().map(one).rev().collect();
    assert_eq!(crate::canonjson::canonical_bytes(&forward).unwrap(),
        crate::canonjson::canonical_bytes(&reverse).unwrap());
    assert!(forward.iter().any(|r| r.output.json == "true"));
    assert!(forward.iter().any(|r| r.output.json == "false"));
}

#[test]
fn top_k_precedes_nucleus_after_exact_mask_and_preserves_original_ids() {
    let p = program(r#"{"type":"boolean"}"#); let o = options(2);
    let s = JsonSamplingOptions { top_k: Some(2), top_p_ppm: 500_000, ..sampling() };
    let mut select = Sampled::new(&identity(), &[1], &p, &o, &s, json_sampling_workspace_bytes().unwrap()).unwrap();
    let mut mask = DenseTokenMask::empty(6);
    for id in [1, 3, 5] { mask.set_legal(id).unwrap(); }
    // The two equal best LEGAL logits are 3 and 5, so their conditional
    // nucleus at 0.5 keeps exactly original ID 3. Illegal high logits lose.
    assert_eq!(select.select(&[100.0, -10.0, 1000.0, 0.0, 1000.0, 0.0], &mask, false, &o, 0).unwrap(), 3);
    let s = JsonSamplingOptions { top_k: Some(1), ..sampling() };
    let mut select = Sampled::new(&identity(), &[1], &p, &o, &s, json_sampling_workspace_bytes().unwrap()).unwrap();
    assert_eq!(select.select(&[100.0, -0.0, 1000.0, 0.0, 1000.0, 0.0], &mask, false, &o, 0).unwrap(), 3);
}

#[test]
fn sampled_source_product_never_selects_high_scoring_off_source_text() {
    let p = JsonProgram::compile_with_source(r#"{"type":"string","maxLength":16,"x-fnlp-source":"verbatim"}"#,
        "Alice and Alice", CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
    let v = Table::new(&[b"", br#""Mallory""#, br#""Alice""#]);
    let run = execute(&[1], &p, &v, &options(2), &sampling(), budget(1, 2, 3),
        &mut Script::new(vec![100.0, 1000.0, 1.0])).unwrap();
    assert_eq!(run.output.json, r#""Alice""#);
    assert!(!p.source_fields(&run.output.json).unwrap().is_empty());
}

#[test]
fn greedy_wire_and_first_id_ties_are_unchanged_by_sampled_support() {
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true", b"false"]);
    let mut d = Script::new(vec![100.0, -0.0, 0.0]);
    let run: Int4JsonRun = drive(GROUP, &[1], &p, &v, &options(2), budget(1, 2, 3), &mut d, Ok::<_, Int4JsonError>).unwrap();
    assert_eq!(run.execution, INT4_JSON_EXECUTION);
    assert_eq!(run.output.token_ids, vec![1, 0]);
    let wire = serde_json::to_value(&run).unwrap();
    assert_eq!(wire.as_object().unwrap().len(), 5);
    assert!(wire.get("sampling").is_none());
}

#[test]
fn sampling_options_and_workspace_refuse_before_native_work() {
    let p = program(r#"{"type":"boolean"}"#); let o = options(2);
    for axis in 0..6 {
        let mut s = sampling();
        match axis { 0 => s.temperature_milli = 0, 1 => s.temperature_milli = 100_001,
            2 => s.top_k = Some(0), 3 => s.top_k = Some(MAX_TOP_K + 1),
            4 => s.top_p_ppm = 0, _ => s.top_p_ppm = 1_000_001 }
        assert!(s.validate().is_err());
        assert!(Sampled::new(&identity(), &[1], &p, &o, &s, u64::MAX).is_err());
    }
    let bytes = json_sampling_workspace_bytes().unwrap();
    assert!(Sampled::new(&identity(), &[1], &p, &o, &sampling(), bytes - 1).is_err());
    assert!(Sampled::new(&identity(), &[1], &p, &o, &sampling(), bytes).is_ok());
    let mut wrong = identity(); wrong.sampler_version = "fnlp-greedy-v1".to_owned();
    assert!(matches!(Sampled::new(&wrong, &[1], &p, &o, &sampling(), bytes), Err(Int4JsonError::ModelIdentity)));
    let mut oversized = identity(); oversized.task_spec = "x".repeat(MAX_IDENTITY_TEXT_BYTES + 1);
    assert!(Sampled::new(&oversized, &[1], &p, &o, &sampling(), bytes).is_err());
    let mut wire = serde_json::to_value(sampling()).unwrap(); wire["extra"] = true.into();
    assert!(serde_json::from_value::<JsonSamplingOptions>(wire).is_err());
    let debug = format!("{:?}", sampling());
    assert!(debug.contains("[redacted]"));
    assert!(!debug.contains("37, 37"));
}

#[test]
fn exact_identity_prompt_options_and_sample_settings_bind_the_private_draw_key() {
    let id = identity(); let o = options(2); let s = sampling(); let p = program(r#"{"type":"boolean"}"#);
    let expected = request_key(&id, &[1, 2], &p, &o, &s).unwrap();
    assert!(expected == request_key(&id, &[1, 2], &p, &o, &s).unwrap());
    for axis in 0..10 {
        let mut id = id.clone(); let mut o = o.clone(); let mut s = s;
        let mut prompt = vec![1, 2];
        match axis { 0 => id.calibration_digest = Sha256Digest::of_bytes(b"new calibration"),
            1 => id.logical_model_digest = Sha256Digest::of_bytes(b"new model"),
            2 => prompt.swap(0, 1), 3 => o.max_new_tokens += 1,
            4 => { o.excluded_token_ids.insert(1); }, 5 => s.effective_seed[0] += 1,
            6 => s.temperature_milli += 1, 7 => s.top_k = Some(1),
            8 => s.top_p_ppm -= 1, _ => s.sample_index += 1 }
        assert!(expected != request_key(&id, &prompt, &p, &o, &s).unwrap());
    }
}

#[test]
fn actual_schema_source_and_program_bounds_change_draws_despite_unchanged_declared_identity() {
    let id = identity(); let o = options(2); let s = sampling();
    let p = program(r#"{"type":"string","maxLength":1}"#);
    let expected = request_key(&id, &[1], &p, &o, &s).unwrap();
    for p in [program(r#"{"type":"string","maxLength":2}"#),
        program(r#"{"type":"string","enum":["x"]}"#),
        JsonProgram::compile(r#"{"type":"string","maxLength":1}"#,
            CompileLimits { max_output_bytes: 16, ..CompileLimits::default() }).unwrap()] {
        assert!(expected != request_key(&id, &[1], &p, &o, &s).unwrap());
    }
    let schema = r#"{"type":"string","x-fnlp-source":"verbatim"}"#;
    let bound = |source, limits| JsonProgram::compile_with_source(schema, source,
        CompileLimits::default(), limits).unwrap();
    let original = bound("Alice", SourceRuntimeLimits::default());
    let expected = request_key(&id, &[1], &original, &o, &s).unwrap();
    let changed_source = bound("Bob", SourceRuntimeLimits::default());
    assert!(expected != request_key(&id, &[1], &changed_source, &o, &s).unwrap());
    let mut limits = SourceRuntimeLimits::default(); limits.verification.max_matches -= 1;
    assert!(expected != request_key(&id, &[1], &bound("Alice", limits), &o, &s).unwrap());
    let mut limits = SourceRuntimeLimits::default(); limits.verification.max_scan_steps -= 1;
    assert!(expected != request_key(&id, &[1], &bound("Alice", limits), &o, &s).unwrap());
}

#[test]
fn all_cancellation_kinds_survive_prefill_decode_and_finalization_without_result() {
    use DecodeCancellationKind::*;
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true"]);
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for boundary in 0..3 {
            let mut d = Script::new(vec![100.0, 1.0]); d.control.cause = Some(cause);
            match boundary { 0 => d.control.prefill = Some(0), 1 => d.control.step = Some(0),
                _ => d.control.step = Some(2) }
            let error = execute(&[1], &p, &v, &options(2), &sampling(), budget(1, 2, 2), &mut d).unwrap_err();
            assert_eq!(error.cancellation(), Some(cause));
            assert!(d.aborted);
            if boundary == 0 { assert!(d.seen.is_empty()); assert_eq!(d.heads, 0); }
        }
    }
}

#[test]
fn accepting_value_without_scored_eos_or_enough_output_bytes_is_no_result() {
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true"]);
    let mut d = Script::new(vec![100.0, 1.0]);
    assert!(matches!(execute(&[1], &p, &v, &options(1), &sampling(), budget(1, 1, 2), &mut d),
        Err(Int4JsonError::Decode(JsonDecodeError::BudgetExceeded("output tokens before EOS")))));
    assert!(d.aborted); assert_eq!(d.heads, 1);
    let p = JsonProgram::compile(r#"{"type":"boolean"}"#, CompileLimits { max_output_bytes: 4,
        ..CompileLimits::default() }).unwrap();
    let mut lying = Table::new(&[b"", b"truefalse"]); lying.lie = true;
    let mut d = Script::new(vec![100.0, 1.0]);
    assert!(matches!(execute(&[1], &p, &lying, &options(2), &sampling(), budget(1, 2, 2), &mut d),
        Err(Int4JsonError::Decode(JsonDecodeError::BudgetExceeded("JSON bytes")))));
    assert!(d.aborted);
}

#[test]
fn masking_every_non_eos_candidate_refuses_instead_of_finishing_early() {
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true", b"false"]);
    let mut o = options(2); o.excluded_token_ids.extend([1, 2]);
    let mut d = Script::new(vec![1000.0, 1.0, 1.0]);
    assert!(matches!(execute(&[1], &p, &v, &o, &sampling(), budget(1, 2, 3), &mut d),
        Err(Int4JsonError::Decode(JsonDecodeError::NoLegalToken))));
    assert!(d.aborted); assert_eq!(d.heads, 1); assert_eq!(d.seen, vec![1]);
}

#[test]
fn finalizer_validation_failure_poisoning_and_invalid_logits_never_publish() {
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true", b"invalid"]);
    let o = options(2); let b = budget(1, 2, 3);
    for row in [vec![100.0, 1.0, f32::NAN], vec![100.0, 1.0, f32::INFINITY],
        vec![100.0, 1.0, f32::NEG_INFINITY], vec![100.0, 1.0]] {
        let mut d = Script::new(row);
        assert!(matches!(execute(&[1], &p, &v, &o, &sampling(), b, &mut d),
            Err(Int4JsonError::Decode(JsonDecodeError::InvalidLogits))));
        assert!(d.aborted);
    }
    let mut d = Script::new(vec![100.0, 1.0, 1.0]); d.corrupt = true;
    assert!(matches!(execute(&[1], &p, &v, &o, &sampling(), b, &mut d), Err(Int4JsonError::WorkMismatch)));
    assert!(d.aborted);
    let mut d = Script::new(vec![100.0, 1.0, 1.0]); let mut finalized = false;
    let mut select = Sampled::new(&identity(), &[1], &p, &o, &sampling(), json_sampling_workspace_bytes().unwrap()).unwrap();
    let result: Result<(), Int4JsonError> = drive_selected(GROUP, &[1], &p, &v, &o, b, &mut d, &mut select, |run| {
        p.validate_json(&run.output.json).unwrap(); finalized = true;
        Err(JsonDecodeError::IndependentValidation.into())
    });
    assert!(finalized); assert!(d.aborted);
    assert!(matches!(result, Err(Int4JsonError::Decode(JsonDecodeError::IndependentValidation))));
}

#[test]
fn sampled_path_reuses_every_existing_native_and_mask_admission_axis() {
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true"]);
    for axis in 0..7 {
        let mut b = budget(2, 2, 2);
        match axis { 0 => b.json.max_forward_positions -= 1, 1 => b.json.max_projected_logits -= 1,
            2 => b.json.max_total_mask_node_visits -= 1, 3 => b.native.max_forward_positions -= 1,
            4 => b.native.max_attention_pairs -= 1, 5 => b.native.max_projection_work.dot_products -= 1,
            _ => b.native.max_projection_work.multiply_accumulates -= 1 }
        let mut d = Script::new(vec![100.0, 1.0]);
        assert!(execute(&[1, 1], &p, &v, &options(2), &sampling(), b, &mut d).is_err());
        assert!(d.seen.is_empty()); assert_eq!(d.heads, 0);
    }
}

#[test]
fn bounded_prompt_morsels_preserve_sample_addresses_masks_scored_eos_and_all_work() {
    let p = program(r#"{"type":"boolean"}"#);
    let v = Table::new(&[b"", b"true", b"false", b"invalid", b"<think>"]);
    let prompt = [3; 9];
    let mut o = options(2); o.excluded_token_ids.insert(0);
    let b = budget(prompt.len(), 2, v.width());
    for index in 0..24 {
        let s = JsonSamplingOptions { sample_index: index, ..sampling() };
        let mut serial = Script::new(vec![100.0, 0.0, 0.0, 10_000.0, 10_000.0]);
        let expected = execute(&prompt, &p, &v, &o, &s, b, &mut serial).unwrap();
        for rows in [1, 2, 4, 8, 64] {
            let mut grouped = Script::new(serial.row.clone());
            let actual = execute_layer_major(&prompt, &p, &v, &o, &s, b,
                &mut grouped, prefill_limits(rows)).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(actual.execution, INT4_JSON_SAMPLED_EXECUTION);
            assert_eq!(actual.output.token_ids.last(), Some(&0));
            assert!(actual.output.json == "true" || actual.output.json == "false");
            assert_eq!(actual.model_work, Int4Work::for_sequence(GROUP, 0, 10, 10).unwrap());
            assert_eq!(actual.output.mask_node_visit_charge, 10);
            assert_eq!(grouped.seen, serial.seen);
            assert_eq!(grouped.heads, 2);
            assert_eq!(grouped.prompt_calls, 1);
            assert_eq!(grouped.prompt_morsels, prompt.chunks(rows).map(|chunk| chunk.len()).collect::<Vec<_>>());
            assert!(!grouped.aborted);
        }
    }
}

#[test]
fn grouped_sampling_keeps_source_product_numeric_continuations_and_split_utf8() {
    let source = JsonProgram::compile_with_source(
        r#"{"type":"string","maxLength":16,"x-fnlp-source":"verbatim"}"#,
        "Alice and Alice", CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
    let cases = [
        (source, Table::new(&[b"", br#""Mallory""#, br#""Alice""#]), vec![100.0, 1000.0, 1.0], 2),
        (program(r#"{"type":"integer","enum":[1,10]}"#),
            Table::new(&[b"", b"1", b"0"]), vec![0.0, 2.0, 1.0], 3),
        (program(r#"{"type":"string","maxLength":1,"enum":["é"]}"#),
            Table::new(&[b"", b"\"", &[0xc3], &[0xa9]]), vec![100.0, 3.0, 2.0, 1.0], 5),
    ];
    for (p, v, row, tokens) in cases {
        let prompt = [1; 5]; let o = options(tokens); let b = budget(5, tokens, v.width());
        for s in [sampling(), JsonSamplingOptions { top_k: Some(1), top_p_ppm: 500_000,
            sample_index: 31, ..sampling() }] {
            let mut serial = Script::new(row.clone());
            let expected = execute(&prompt, &p, &v, &o, &s, b, &mut serial).unwrap();
            for rows in [1, 4, 64] {
                let mut grouped = Script::new(row.clone());
                let actual = execute_layer_major(&prompt, &p, &v, &o, &s, b,
                    &mut grouped, prefill_limits(rows)).unwrap();
                assert_eq!(actual, expected);
                assert_eq!(grouped.seen, serial.seen);
                assert_eq!(grouped.heads, serial.heads);
                assert_eq!(grouped.prompt_calls, 1);
                p.validate_json(&actual.output.json).unwrap();
            }
        }
    }
}

#[test]
fn grouped_sampler_refuses_invalid_prefill_or_complete_work_before_prompt_entry() {
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true"]);
    let mut short = prefill_limits(4); short.max_extra_scratch_bytes -= 1;
    for limits in [short, Int4PrefillLimits { max_batch_rows: 0, max_extra_scratch_bytes: u64::MAX },
        Int4PrefillLimits { max_batch_rows: 65, max_extra_scratch_bytes: u64::MAX },
        Int4PrefillLimits { max_batch_rows: usize::MAX, max_extra_scratch_bytes: u64::MAX }] {
        let mut driver = Script::new(vec![100.0, 1.0]);
        assert!(execute_layer_major(&[1; 5], &p, &v, &options(2), &sampling(), budget(5, 2, 2),
            &mut driver, limits).is_err());
        assert!(driver.seen.is_empty());
        assert_eq!(driver.prompt_calls, 0);
        assert_eq!(driver.heads, 0);
        assert!(!driver.aborted);
    }
    for axis in 0..7 {
        let mut b = budget(5, 2, 2);
        match axis { 0 => b.json.max_forward_positions -= 1, 1 => b.json.max_projected_logits -= 1,
            2 => b.json.max_total_mask_node_visits -= 1, 3 => b.native.max_forward_positions -= 1,
            4 => b.native.max_attention_pairs -= 1, 5 => b.native.max_projection_work.dot_products -= 1,
            _ => b.native.max_projection_work.multiply_accumulates -= 1 }
        let mut driver = Script::new(vec![100.0, 1.0]);
        assert!(execute_layer_major(&[1; 5], &p, &v, &options(2), &sampling(), b,
            &mut driver, prefill_limits(4)).is_err());
        assert!(driver.seen.is_empty());
        assert_eq!(driver.prompt_calls, 0);
        assert_eq!(driver.heads, 0);
    }
}

#[test]
fn grouped_sampler_preserves_all_cancel_causes_through_prompt_decode_and_finalization() {
    use DecodeCancellationKind::*;
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true"]);
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for boundary in 0..3 {
            let mut driver = Script::new(vec![100.0, 1.0]); driver.control.cause = Some(cause);
            match boundary { 0 => driver.control.prefill = Some(2), 1 => driver.control.step = Some(0),
                _ => driver.control.step = Some(2) }
            let error = execute_layer_major(&[1; 5], &p, &v, &options(2), &sampling(), budget(5, 2, 2),
                &mut driver, prefill_limits(2)).unwrap_err();
            assert_eq!(error.cancellation(), Some(cause));
            assert!(driver.aborted);
            assert_eq!(driver.prompt_calls, 1);
            if boundary == 0 {
                assert_eq!(driver.seen, [1; 2]);
                assert_eq!(driver.heads, 0);
            } else if boundary == 2 {
                assert_eq!(driver.seen.len(), 6);
                assert_eq!(driver.heads, 2);
            }
        }
    }
}

#[test]
fn grouped_sampler_finalizer_refusal_or_invalid_logits_never_publish_a_bundle() {
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true"]);
    let o = options(2); let b = budget(5, 2, 2);
    for corrupt in [false, true] {
        let mut driver = Script::new(vec![100.0, 1.0]);
        if corrupt { driver.corrupt = true; } else { driver.row[0] = f32::NAN; }
        assert!(execute_layer_major(&[1; 5], &p, &v, &o, &sampling(), b,
            &mut driver, prefill_limits(4)).is_err());
        assert!(driver.aborted);
    }
    let mut driver = Script::new(vec![100.0, 1.0]);
    let mut select = Sampled::new(&identity(), &[1; 5], &p, &o, &sampling(),
        json_sampling_workspace_bytes().unwrap()).unwrap();
    let mut finalized = false;
    let result: Result<(), Int4JsonError> = prefill::drive_layer_major_selected(GROUP, &[1; 5],
        &p, &v, &o, b, &mut driver, &mut select, prefill_limits(4), |run| {
            finalized = true;
            assert_eq!(run.output.token_ids, [1, 0]);
            Err(JsonDecodeError::IndependentValidation.into())
        });
    assert!(matches!(result, Err(Int4JsonError::Decode(JsonDecodeError::IndependentValidation))));
    assert!(finalized);
    assert!(driver.aborted);
    assert_eq!(driver.prompt_calls, 1);
    assert_eq!(driver.heads, 2);
}
