//! Closed-driver integration checks, not native model-quality/parity evidence.
use super::*;
use std::collections::BTreeSet;
use crate::grammar::{CompileLimits, runtime::SourceRuntimeLimits};

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
        -> Result<DenseTokenMask, Int8JsonError> {
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
struct Control {
    prefill: Option<usize>, step: Option<usize>, cause: Option<DecodeCancellationKind>,
    calls: usize, cancel_at: Option<usize>,
}
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, step: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        if self.step == Some(step) || self.cancel_at == Some(self.calls) { self.cause } else { None }
    }
    fn prefill_checkpoint(&mut self, index: usize) -> Option<DecodeCancellationKind> {
        if self.prefill == Some(index) { self.cause } else { None }
    }
}
struct Script {
    control: Control, row: Vec<f32>, seen: Vec<u32>, heads: usize,
    aborted: bool, corrupt: bool, morsels: Vec<usize>,
}
impl Script {
    fn new(row: Vec<f32>) -> Self {
        Self { control: Control::default(), row, seen: Vec::new(), heads: 0,
            aborted: false, corrupt: false, morsels: Vec::new() }
    }
}
impl Driver for Script {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn append(&mut self, token: u32) -> Result<(), Int8JsonError> { self.seen.push(token); Ok(()) }
    fn logits(&mut self) -> Result<Vec<f32>, Int8JsonError> { self.heads += 1; Ok(self.row.clone()) }
    fn work(&self) -> Int8Work {
        let mut work = Int8Work::for_sequence(0, self.seen.len(), self.heads * self.row.len()).unwrap();
        if self.corrupt { work.projections.dot_products += 1; }
        work
    }
    fn abort(&mut self) { self.aborted = true; }
}
impl prefill::PromptDriver for Script {
    fn append_layer_major(&mut self, prompt: &[u32], limits: Int8PrefillLimits) -> Result<(), Int8JsonError> {
        for chunk in prompt.chunks(limits.max_batch_rows) {
            self.morsels.push(chunk.len());
            for &token in chunk {
                if let Some(cause) = self.control.prefill_checkpoint(self.seen.len()) {
                    return Err(StrictInt8Error::Cancelled(cause).into());
                }
                self.append(token)?;
            }
        }
        Ok(())
    }
}
fn sampling() -> Int8JsonSamplingOptions {
    Int8JsonSamplingOptions { effective_seed: [37; 32], temperature_milli: 1000,
        top_k: None, top_p_ppm: 1_000_000, sample_index: 0 }
}
fn options(tokens: usize) -> JsonDecodeOptions {
    JsonDecodeOptions { max_new_tokens: tokens, eos_token_id: 0, excluded_token_ids: BTreeSet::new() }
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"sampled-int8-json-test-identity");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "portable-quant-v1".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "sentiment-sampled-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "fixture".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: JSON_SAMPLER_VERSION.to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn program(schema: &str) -> JsonProgram { JsonProgram::compile(schema, CompileLimits::default()).unwrap() }
fn budget(prompt: usize, tokens: usize, width: usize) -> Int8JsonBudget {
    let work = work_for_width(prompt, tokens, width).unwrap();
    Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: 8192 * KV_BYTES_PER_TOKEN as u64,
        max_total_mask_node_visits: (tokens * width) as u64, mask_limits: MaskWorkLimits::default(),
    } }
}
fn prefill_limits(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
#[allow(clippy::too_many_arguments)]
fn execute(prompt: &[u32], p: &JsonProgram, v: &Table, o: &JsonDecodeOptions,
    s: &Int8JsonSamplingOptions, b: Int8JsonBudget, d: &mut Script)
    -> Result<Int8JsonRun, Int8JsonError> {
    preflight(prompt, v, o, b)?;
    let mut selection = Sampled::new(&identity(), prompt, p, o, s, json_sampling_workspace_bytes()?)?;
    drive_selected(prompt, p, v, o, b, d, &mut selection, Ok)
}
#[allow(clippy::too_many_arguments)]
fn execute_layer_major(prompt: &[u32], p: &JsonProgram, v: &Table, o: &JsonDecodeOptions,
    s: &Int8JsonSamplingOptions, b: Int8JsonBudget, limits: Int8PrefillLimits, d: &mut Script)
    -> Result<Int8JsonRun, Int8JsonError> {
    let mut selection = Sampled::new(&identity(), prompt, p, o, s, json_sampling_workspace_bytes()?)?;
    prefill::drive_layer_major_selected(prompt, p, v, o, b, limits, d, &mut selection, Ok)
}

#[test]
fn exact_grammar_controls_scored_eos_and_final_prefill_head_are_preserved() {
    let p = program(r#"{"type":"boolean"}"#);
    let v = Table::new(&[b"", b"true", b"false", b"invalid", b"<think>"]);
    let mut o = options(2); o.excluded_token_ids.extend([0, 1]);
    let mut d = Script::new(vec![100.0, 1000.0, 1.0, 10_000.0, 10_000.0]);
    let run = execute(&[3; 3], &p, &v, &o, &sampling(), budget(3, 2, 5), &mut d).unwrap();
    assert_eq!(run.output.json, "false");
    assert_eq!(run.output.token_ids, vec![2, 0]);
    assert_eq!(d.seen, vec![3, 3, 3, 2]);
    assert_eq!(d.heads, 2); assert!(!d.aborted);
    assert_eq!(run.execution, INT8_JSON_SAMPLED_EXECUTION);
    assert_eq!(run.output.numerics_profile, STRICT_INT8_PROFILE);
    assert_eq!(run.output.mask_node_visit_charge, 10);
    assert_eq!(run.model_work, Int8Work::for_sequence(0, 4, 10).unwrap());
    p.validate_json(&run.output.json).unwrap();
}

#[test]
fn independent_sample_addresses_replay_in_any_order_and_prefill_schedule() {
    let p = program(r#"{"type":"boolean"}"#);
    let v = Table::new(&[b"", b"true", b"false", b"invalid"]);
    let prompt = [3; 9]; let o = options(2); let b = budget(9, 2, v.width());
    let one = |index| {
        let s = Int8JsonSamplingOptions { sample_index: index, ..sampling() };
        execute(&prompt, &p, &v, &o, &s, b, &mut Script::new(vec![100.0, 0.0, 0.0, 1000.0])).unwrap()
    };
    let forward: Vec<_> = (0..16).map(one).collect();
    let reverse: Vec<_> = (0..16).rev().map(one).rev().collect();
    assert_eq!(forward, reverse);
    for (index, expected) in forward.iter().enumerate() {
        let s = Int8JsonSamplingOptions { sample_index: index as u64, ..sampling() };
        for rows in [1, 2, 4, 8, 64] {
            let mut d = Script::new(vec![100.0, 0.0, 0.0, 1000.0]);
            let actual = execute_layer_major(&prompt, &p, &v, &o, &s, b, prefill_limits(rows), &mut d).unwrap();
            assert_eq!(&actual, expected);
            assert_eq!(d.morsels, prompt.chunks(rows).map(|chunk| chunk.len()).collect::<Vec<_>>());
            assert_eq!(d.seen.len(), 10); assert_eq!(d.heads, 2); assert!(!d.aborted);
        }
    }
}

#[test]
fn temperature_top_k_and_nucleus_apply_after_masking_with_original_id_ties() {
    let p = program(r#"{"type":"boolean"}"#); let o = options(2);
    let s = Int8JsonSamplingOptions { temperature_milli: 500, top_k: Some(2),
        top_p_ppm: 500_000, ..sampling() };
    let mut selection = Sampled::new(&identity(), &[1], &p, &o, &s, u64::MAX).unwrap();
    let mut mask = DenseTokenMask::empty(6);
    for id in [1, 3, 5] { mask.set_legal(id).unwrap(); }
    // Equal highest legal rows 3 and 5 form top-k, and p=0.5 includes
    // exactly row 3. Illegal high scores never enter the denominator.
    assert_eq!(selection.select(&[100.0, -10.0, 1000.0, 0.0, 1000.0, 0.0],
        &mask, false, &o, 0, &mut Control::default()).unwrap(), 3);
    let s = Int8JsonSamplingOptions { top_k: Some(1), ..sampling() };
    let mut selection = Sampled::new(&identity(), &[1], &p, &o, &s, u64::MAX).unwrap();
    assert_eq!(selection.select(&[100.0, -0.0, 1000.0, 0.0, 1000.0, 0.0],
        &mask, false, &o, 0, &mut Control::default()).unwrap(), 1);
}

#[test]
fn source_membership_numeric_continuations_and_split_utf8_keep_their_grammar() {
    let source = JsonProgram::compile_with_source(
        r#"{"type":"string","maxLength":16,"x-fnlp-source":"verbatim"}"#,
        "Alice and Alice", CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
    let cases = [
        (source, Table::new(&[b"", br#""Mallory""#, br#""Alice""#]), vec![100.0, 1000.0, 1.0], 2, r#""Alice""#),
        (program(r#"{"type":"integer","enum":[1,10]}"#),
            Table::new(&[b"", b"1", b"0"]), vec![0.0, 2.0, 1.0], 3, "10"),
        (program(r#"{"type":"string","maxLength":1,"enum":["é"]}"#),
            Table::new(&[b"", b"\"", &[0xc3], &[0xa9]]), vec![100.0, 3.0, 2.0, 1.0], 5, "\"é\""),
    ];
    let s = Int8JsonSamplingOptions { top_k: Some(1), ..sampling() };
    for (p, v, row, tokens, expected) in cases {
        for rows in [1, 4, 64] {
            let mut d = Script::new(row.clone());
            let run = execute_layer_major(&[1; 5], &p, &v, &options(tokens), &s,
                budget(5, tokens, v.width()), prefill_limits(rows), &mut d).unwrap();
            assert_eq!(run.output.json, expected);
            p.validate_json(&run.output.json).unwrap();
        }
    }
}

#[test]
fn no_legal_token_or_eosless_value_never_becomes_completed_json() {
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true", b"false"]);
    let mut o = options(2); o.excluded_token_ids.extend([1, 2]);
    let mut d = Script::new(vec![1000.0, 1.0, 1.0]);
    assert!(matches!(execute(&[1], &p, &v, &o, &sampling(), budget(1, 2, 3), &mut d),
        Err(Int8JsonError::Decode(JsonDecodeError::NoLegalToken))));
    assert!(d.aborted);
    let v = Table::new(&[b"", b"true"]); let mut d = Script::new(vec![1000.0, 1.0]);
    assert!(matches!(execute(&[1], &p, &v, &options(1), &sampling(), budget(1, 1, 2), &mut d),
        Err(Int8JsonError::Decode(JsonDecodeError::BudgetExceeded("output tokens before EOS")))));
    assert!(d.aborted); assert_eq!(d.heads, 1);
}

#[test]
fn sampler_policy_storage_and_identity_fail_before_allocation_or_model_entry() {
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
    for axis in 0..4 {
        let mut invalid = identity();
        match axis { 0 => invalid.sampler_version = "fnlp-greedy-v1".to_owned(),
            1 => invalid.numerics_profile = NumericsProfile::StrictQuantized { version: 2 },
            2 => invalid.backend_semantic_version = "other".to_owned(),
            _ => invalid.task_spec = "x".repeat(MAX_IDENTITY_TEXT_BYTES + 1) }
        assert!(Sampled::new(&invalid, &[1], &p, &o, &sampling(), bytes).is_err());
    }
    let mut wire = serde_json::to_value(sampling()).unwrap(); wire["extra"] = true.into();
    assert!(serde_json::from_value::<Int8JsonSamplingOptions>(wire).is_err());
    let debug = format!("{:?}", sampling());
    assert!(debug.contains("[redacted]")); assert!(!debug.contains("37, 37"));
}

#[test]
fn every_model_mask_and_prefill_admission_axis_refuses_before_prompt_work() {
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true"]);
    for axis in 0..7 {
        let mut b = budget(5, 2, 2);
        match axis { 0 => b.json.max_forward_positions -= 1, 1 => b.json.max_projected_logits -= 1,
            2 => b.json.max_total_mask_node_visits -= 1, 3 => b.native.max_forward_positions -= 1,
            4 => b.native.max_attention_pairs -= 1, 5 => b.native.max_projection_work.dot_products -= 1,
            _ => b.native.max_projection_work.multiply_accumulates -= 1 }
        let mut d = Script::new(vec![100.0, 1.0]);
        assert!(execute_layer_major(&[1; 5], &p, &v, &options(2), &sampling(), b,
            prefill_limits(4), &mut d).is_err());
        assert!(d.seen.is_empty()); assert!(d.morsels.is_empty()); assert_eq!(d.heads, 0);
    }
    let mut short = prefill_limits(4); short.max_extra_scratch_bytes -= 1;
    for limits in [short, Int8PrefillLimits { max_batch_rows: 0, max_extra_scratch_bytes: u64::MAX },
        Int8PrefillLimits { max_batch_rows: 65, max_extra_scratch_bytes: u64::MAX },
        Int8PrefillLimits { max_batch_rows: usize::MAX, max_extra_scratch_bytes: u64::MAX }] {
        let mut d = Script::new(vec![100.0, 1.0]);
        assert!(execute_layer_major(&[1; 5], &p, &v, &options(2), &sampling(), budget(5, 2, 2),
            limits, &mut d).is_err());
        assert!(d.seen.is_empty()); assert!(d.morsels.is_empty()); assert_eq!(d.heads, 0);
    }
}

#[test]
fn actual_request_source_program_limits_and_sample_parameters_bind_draws() {
    let id = identity(); let o = options(2); let s = sampling();
    let p = program(r#"{"type":"string","maxLength":1}"#);
    let expected = request_key(&id, &[1, 2], &p, &o, &s).unwrap();
    assert!(expected == request_key(&id, &[1, 2], &p, &o, &s).unwrap());
    for axis in 0..11 {
        let mut id = id.clone(); let mut o = o.clone(); let mut s = s; let mut prompt = vec![1, 2];
        match axis { 0 => id.logical_model_digest = Sha256Digest::of_bytes(b"other model"),
            1 => id.calibration_digest = Sha256Digest::of_bytes(b"other calibration"),
            2 => prompt.swap(0, 1), 3 => o.max_new_tokens += 1, 4 => { o.excluded_token_ids.insert(1); },
            5 => s.effective_seed[0] += 1, 6 => s.temperature_milli += 1,
            7 => s.top_k = Some(1), 8 => s.top_p_ppm -= 1, 9 => s.sample_index += 1,
            _ => o.eos_token_id = 1 }
        assert!(expected != request_key(&id, &prompt, &p, &o, &s).unwrap(), "axis {axis}");
    }
    let different_limit = JsonProgram::compile(r#"{"type":"string","maxLength":1}"#,
        CompileLimits { max_output_bytes: 16, ..CompileLimits::default() }).unwrap();
    assert!(expected != request_key(&id, &[1, 2], &different_limit, &o, &s).unwrap());
    let source_program = |source| JsonProgram::compile_with_source(
        r#"{"type":"string","x-fnlp-source":"verbatim"}"#, source,
        CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
    assert!(request_key(&id, &[1, 2], &source_program("Alice"), &o, &s).unwrap()
        != request_key(&id, &[1, 2], &source_program("Bob"), &o, &s).unwrap());
}

#[test]
fn terminal_causes_survive_prefill_sampling_and_finalization() {
    use DecodeCancellationKind::*;
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true"]);
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for boundary in 0..3 {
            let mut d = Script::new(vec![100.0, 1.0]); d.control.cause = Some(cause);
            match boundary { 0 => d.control.prefill = Some(0), 1 => d.control.step = Some(0),
                _ => d.control.step = Some(2) }
            let error = execute(&[1], &p, &v, &options(2), &sampling(), budget(1, 2, 2), &mut d).unwrap_err();
            assert_eq!(error.cancellation(), Some(cause)); assert!(d.aborted);
            if boundary == 0 { assert!(d.seen.is_empty()); assert_eq!(d.heads, 0); }
        }
        // A failed task finalizer must still observe the original terminal
        // control, rather than turning cancellation into a validation error.
        let mut d = Script::new(vec![100.0, 1.0]);
        d.control.cause = Some(cause); d.control.step = Some(2);
        let mut finalized = false;
        let mut selection = Sampled::new(&identity(), &[1], &p, &options(2), &sampling(), u64::MAX).unwrap();
        let result: Result<(), Int8JsonError> = drive_selected(&[1], &p, &v, &options(2),
            budget(1, 2, 2), &mut d, &mut selection, |_| {
                finalized = true;
                Err(JsonDecodeError::IndependentValidation.into())
            });
        assert!(finalized); assert!(d.aborted);
        assert_eq!(result.unwrap_err().cancellation(), Some(cause));
        // Force cancellation inside the shared vocabulary processor rather
        // than only at the outer decoder's token boundaries.
        let mut selection = Sampled::new(&identity(), &[1], &p, &options(2), &sampling(), u64::MAX).unwrap();
        let mut control = Control { cause: Some(cause), cancel_at: Some(2), ..Control::default() };
        let mut mask = DenseTokenMask::empty(1025);
        for id in 1..1025 { mask.set_legal(id).unwrap(); }
        let error = selection.select(&vec![0.0; 1025], &mask, false, &options(2), 0, &mut control).unwrap_err();
        assert_eq!(error.cancellation(), Some(cause));
    }
}

#[test]
fn nonfinite_hidden_logits_accounting_mismatch_and_failed_validation_never_publish() {
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true", b"invalid"]);
    let o = options(2); let b = budget(1, 2, 3);
    for row in [vec![100.0, 1.0, f32::NAN], vec![100.0, 1.0, f32::INFINITY],
        vec![100.0, 1.0, f32::NEG_INFINITY], vec![100.0, 1.0]] {
        let mut d = Script::new(row);
        assert!(matches!(execute(&[1], &p, &v, &o, &sampling(), b, &mut d),
            Err(Int8JsonError::Decode(JsonDecodeError::InvalidLogits))));
        assert!(d.aborted);
    }
    let mut d = Script::new(vec![100.0, 1.0, 1.0]); d.corrupt = true;
    assert!(matches!(execute(&[1], &p, &v, &o, &sampling(), b, &mut d), Err(Int8JsonError::WorkMismatch)));
    assert!(d.aborted);
    let mut d = Script::new(vec![100.0, 1.0, 1.0]); let mut finalized = false;
    let mut selection = Sampled::new(&identity(), &[1], &p, &o, &sampling(), u64::MAX).unwrap();
    let result: Result<(), Int8JsonError> = drive_selected(&[1], &p, &v, &o, b, &mut d, &mut selection, |run| {
        p.validate_json(&run.output.json).unwrap(); finalized = true;
        Err(JsonDecodeError::IndependentValidation.into())
    });
    assert!(finalized); assert!(d.aborted);
    assert!(matches!(result, Err(Int8JsonError::Decode(JsonDecodeError::IndependentValidation))));
}

#[test]
fn an_incorrect_mask_cannot_bypass_output_bounds_or_transition_validation() {
    let p = JsonProgram::compile(r#"{"type":"boolean"}"#,
        CompileLimits { max_output_bytes: 4, ..CompileLimits::default() }).unwrap();
    let mut v = Table::new(&[b"", b"truefalse"]); v.lie = true;
    let mut d = Script::new(vec![100.0, 1.0]);
    assert!(matches!(execute(&[1], &p, &v, &options(2), &sampling(), budget(1, 2, 2), &mut d),
        Err(Int8JsonError::Decode(JsonDecodeError::BudgetExceeded("JSON bytes")))));
    assert!(d.aborted);
    let mut v = Table::new(&[b"", b"bad"]); v.lie = true;
    let mut d = Script::new(vec![100.0, 1.0]);
    assert!(matches!(execute(&[1], &p, &v, &options(2), &sampling(), budget(1, 2, 2), &mut d),
        Err(Int8JsonError::Decode(JsonDecodeError::IllegalTransition))));
    assert!(d.aborted);
}

#[test]
fn greedy_dispatch_and_wire_contract_are_unchanged() {
    let p = program(r#"{"type":"boolean"}"#); let v = Table::new(&[b"", b"true", b"false"]);
    let mut d = Script::new(vec![100.0, -0.0, 0.0]);
    let run: Int8JsonRun = drive(&[1], &p, &v, &options(2), budget(1, 2, 3),
        &mut d, Ok::<_, Int8JsonError>).unwrap();
    assert_eq!(run.execution, INT8_JSON_EXECUTION);
    assert_eq!(run.output.token_ids, vec![1, 0]);
    assert_eq!(serde_json::to_value(&run).unwrap().as_object().unwrap().len(), 4);
}
