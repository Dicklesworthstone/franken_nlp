//! Synthetic scheduling/grammar fixtures, not native-model parity evidence.
use super::*;
use crate::grammar::{CompileLimits, runtime::SourceRuntimeLimits};
use crate::execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode};

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
struct Control { calls: usize, stop: Option<usize>, prefill: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        (self.stop == Some(self.calls)).then_some(DecodeCancellationKind::Deadline)
    }
    fn prefill_checkpoint(&mut self, position: usize) -> Option<DecodeCancellationKind> {
        (self.prefill == Some(position)).then_some(DecodeCancellationKind::PollQuota)
    }
}
struct Script {
    control: Control, scores: Vec<Vec<f32>>, seen: Vec<Vec<u32>>, projected: Vec<usize>,
    groups: Vec<Vec<(usize, u32)>>, heads: Vec<(usize, Vec<u32>)>, aborted: bool,
    fail_group: Option<usize>, bad_work: bool, bad_head: bool,
}
impl Script {
    fn new(count: usize) -> Self {
        Self { control: Control::default(), scores: vec![vec![0.0, 2.0, 1.0, 4.0, 3.0, 1.0, 100.0]; count],
            seen: vec![Vec::new(); count], projected: vec![0; count], groups: Vec::new(), heads: Vec::new(),
            aborted: false, fail_group: None, bad_work: false, bad_head: false }
    }
}
impl GroupDriver for Script {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn append_group(&mut self, steps: &[CohortToken]) -> Result<(), Int8JsonError> {
        assert!(!steps.is_empty());
        assert!(steps.windows(2).all(|pair| pair[0].sequence < pair[1].sequence));
        self.groups.push(steps.iter().map(|s| (s.sequence, s.token)).collect());
        for step in steps {
            self.seen[step.sequence].push(step.token);
            if self.fail_group == Some(self.groups.len()) { return Err(StrictInt8Error::Primitive.into()); }
        }
        Ok(())
    }
    fn logits(&mut self, slot: usize, rows: &[u32]) -> Result<Vec<f32>, Int8JsonError> {
        self.heads.push((slot, rows.to_vec())); self.projected[slot] += rows.len();
        if self.bad_head { return Ok(vec![f32::NAN; rows.len()]); }
        Ok(rows.iter().map(|&id| self.scores[slot][id as usize]).collect())
    }
    fn work(&self, slot: usize) -> Result<Int8Work, Int8JsonError> {
        let mut work = Int8Work::for_sequence(0, self.seen[slot].len(), self.projected[slot])?;
        if self.bad_work { work.attention_pairs += 1; }
        Ok(work)
    }
    fn abort(&mut self) { self.aborted = true; }
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"cohort fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "fixture".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn vocabulary() -> Table {
    Table([b"".as_slice(), b"true", b"false", b"1", b"0", br#""Alice""#, br#""Mallory""#]
        .into_iter().map(<[u8]>::to_vec).collect())
}
fn programs() -> Vec<JsonProgram> {
    vec![JsonProgram::compile(r#"{"type":"boolean"}"#, CompileLimits::default()).unwrap(),
        JsonProgram::compile(r#"{"type":"integer","enum":[1,10]}"#, CompileLimits::default()).unwrap(),
        JsonProgram::compile_with_source(r#"{"type":"string","maxLength":16,"x-fnlp-source":"verbatim"}"#,
            "Alice", CompileLimits::default(), SourceRuntimeLimits::default()).unwrap()]
}
fn options(tokens: usize) -> JsonDecodeOptions {
    JsonDecodeOptions { max_new_tokens: tokens, eos_token_id: 0, excluded_token_ids: [0].into_iter().collect() }
}
fn request<'a>(id: &'a ExecutionIdentity, prompt: &'a [u32], program: &'a JsonProgram,
    options: &'a JsonDecodeOptions, cap: usize) -> Int8JsonCohortRequest<'a> {
    let limits = Int8JsonSparseLimits { max_rows_per_step: cap };
    let work = planned_work(prompt.len(), options.max_new_tokens, limits).unwrap();
    Int8JsonCohortRequest { identity: id, prompt, program, options, limits,
        budget: Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
            max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
            max_kv_bytes: 8192 * KV_BYTES_PER_TOKEN as u64,
            max_total_mask_node_visits: options.max_new_tokens as u64 * 7,
            mask_limits: MaskWorkLimits::default() } } }
}
fn budget(requests: &[Int8JsonCohortRequest<'_>]) -> Int8JsonCohortBudget {
    let mut work = Int8Work::default(); let mut masks = 0;
    for r in requests {
        work = work.checked_add(planned_work(r.prompt.len(), r.options.max_new_tokens, r.limits).unwrap()).unwrap();
        masks += r.budget.json.max_total_mask_node_visits;
    }
    Int8JsonCohortBudget { native: Int8RunBudget::exact(work), max_kv_bytes: u64::MAX,
        max_mask_node_visits: masks, max_result_bytes: 1 << 20 }
}
fn execute(requests: &[Int8JsonCohortRequest<'_>], budget: Int8JsonCohortBudget, driver: &mut Script)
    -> Result<Int8JsonCohortRun, Int8JsonError> {
    execution::drive(requests, &vocabulary(), budget, driver, Ok)
}
#[test]
fn ragged_prefill_and_decode_share_groups_without_crossing_source_languages() {
    let id = identity(); let p = programs(); let o = options(4);
    let requests = [request(&id, &[6], &p[0], &o, 2), request(&id, &[6,6,6], &p[1], &o, 2),
        request(&id, &[6,6], &p[2], &o, 2)];
    let mut d = Script::new(3); let result = execute(&requests, budget(&requests), &mut d).unwrap();
    assert_eq!(result.sequences.iter().map(|r| r.output.json.as_str()).collect::<Vec<_>>(), ["true", "10", "\"Alice\""]);
    assert_eq!(d.groups, [vec![(0,6),(1,6),(2,6)], vec![(0,1),(1,6),(2,6)],
        vec![(1,6),(2,5)], vec![(1,3)], vec![(1,4)]]);
    assert_eq!(result.group_steps, 5); assert_eq!(result.model_work.forward_positions, 10);
    assert_eq!(d.projected, [3,4,2]); assert_eq!(result.model_work.projected_logits, 9);
    assert!(!d.aborted); assert!(result.sequences[2].output.token_ids == [5,0]);
    assert!(d.heads.iter().all(|(_, rows)| !rows.contains(&6)));
}
#[test]
fn each_cohort_row_matches_the_existing_dense_masked_argmax_on_finite_scores() {
    let id = identity(); let p = programs(); let o = options(4); let v = vocabulary();
    let requests = [request(&id, &[6], &p[0], &o, 7), request(&id, &[6,6], &p[1], &o, 7),
        request(&id, &[6,6,6], &p[2], &o, 7)];
    let mut d = Script::new(3); let result = execute(&requests, budget(&requests), &mut d).unwrap();
    for (slot, request) in requests.iter().enumerate() {
        let mut state = request.program.initial_state(); let mut tokens = Vec::new();
        for index in 0..o.max_new_tokens {
            let mask = v.mask(&state, MaskWorkLimits::default(), &mut Control::default(), index).unwrap();
            let token = crate::native_engine::constrained_int8::select(&d.scores[slot], &mask, state.is_accepting(), &o).unwrap();
            tokens.push(token); if token == o.eos_token_id { break; }
            assert!(state.consume_bytes(v.bytes(token).unwrap()));
        }
        assert_eq!(tokens, result.sequences[slot].output.token_ids);
    }
}
#[test]
fn complete_cohort_and_row_work_is_checked_before_first_native_group() {
    let id = identity(); let p = programs(); let o = options(4);
    for axis in 0..10 {
        let mut requests = [request(&id, &[6], &p[0], &o, 7), request(&id, &[6], &p[1], &o, 7)];
        let mut b = budget(&requests);
        match axis {
            0 => b.native.max_forward_positions -= 1, 1 => b.native.max_attention_pairs -= 1,
            2 => b.native.max_projection_work.dot_products -= 1,
            3 => b.native.max_projection_work.multiply_accumulates -= 1, 4 => b.max_mask_node_visits -= 1,
            5 => requests[1].budget.native.max_forward_positions -= 1,
            6 => requests[1].budget.json.max_projected_logits -= 1,
            7 => requests[1].budget.json.max_total_mask_node_visits -= 1,
            8 => requests[1].budget.native.max_attention_pairs -= 1, _ => b.max_result_bytes = 0,
        }
        let mut d = Script::new(2);
        assert!(execute(&requests, b, &mut d).is_err()); assert!(d.groups.is_empty()); assert!(d.aborted);
    }
}
#[test]
fn complete_legal_set_over_cap_never_projects_a_pruned_prefix() {
    let id = identity(); let p = programs(); let o = options(4);
    let requests = [request(&id, &[6], &p[0], &o, 1)]; let mut d = Script::new(1);
    assert!(execute(&requests, budget(&requests), &mut d).is_err());
    assert!(d.heads.is_empty()); assert!(d.aborted);
}
#[test]
fn missing_eos_in_one_row_discards_an_already_completed_other_row() {
    let id = identity(); let p = programs(); let a = options(4); let b = options(1);
    let requests = [request(&id, &[6], &p[0], &a, 7), request(&id, &[6,6,6], &p[1], &b, 7)];
    let mut d = Script::new(2);
    assert!(execute(&requests, budget(&requests), &mut d).is_err()); assert!(d.aborted);
    assert_eq!(d.seen[0], [6,1]); assert_eq!(d.seen[1], [6,6,6]);
}
#[test]
fn a_native_partial_group_head_failure_or_work_disagreement_aborts_every_row() {
    let id = identity(); let p = programs(); let o = options(4);
    let requests = [request(&id, &[6], &p[0], &o, 7), request(&id, &[6], &p[1], &o, 7)];
    for axis in 0..3 {
        let mut d = Script::new(2);
        match axis { 0 => d.fail_group = Some(2), 1 => d.bad_head = true, _ => d.bad_work = true }
        assert!(execute(&requests, budget(&requests), &mut d).is_err()); assert!(d.aborted);
    }
}
#[test]
fn independent_control_exclusions_and_illegal_nonfinite_rows_are_preserved() {
    let id = identity(); let p = programs(); let mut a = options(4); a.excluded_token_ids.insert(1);
    let b = options(4);
    let requests = [request(&id, &[6], &p[0], &a, 7), request(&id, &[6], &p[0], &b, 7)];
    let mut d = Script::new(2); for row in &mut d.scores { row[6] = f32::NAN; }
    let result = execute(&requests, budget(&requests), &mut d).unwrap();
    assert_eq!(result.sequences[0].output.json, "false"); assert_eq!(result.sequences[1].output.json, "true");
}
#[test]
fn cancellation_covers_prefill_selection_and_late_finalization() {
    let id = identity(); let p = programs(); let o = options(4);
    let requests = [request(&id, &[6,6], &p[0], &o, 7)];
    for prefill in [true, false] {
        let mut d = Script::new(1);
        if prefill { d.control.prefill = Some(1); } else { d.control.stop = Some(2); }
        assert!(execute(&requests, budget(&requests), &mut d).unwrap_err().cancellation().is_some()); assert!(d.aborted);
    }
    let mut reference = Script::new(1); execute(&requests, budget(&requests), &mut reference).unwrap();
    let mut d = Script::new(1); d.control.stop = Some(reference.control.calls);
    let mut finalized = false;
    let result: Result<_, Int8JsonError> = execution::drive(&requests, &vocabulary(), budget(&requests), &mut d,
        |run| { finalized = true; Ok(run) });
    assert!(result.unwrap_err().cancellation().is_some()); assert!(finalized); assert!(d.aborted);
}
#[test]
fn typed_finalization_and_complete_envelope_failures_never_escape_session_ownership() {
    let id = identity(); let p = programs(); let o = options(4);
    let requests = [request(&id, &[6], &p[0], &o, 7)]; let mut d = Script::new(1);
    let error: Result<(), Int8JsonError> = execution::drive(&requests, &vocabulary(), budget(&requests), &mut d,
        |_| Err(JsonDecodeError::IndependentValidation.into()));
    assert!(error.is_err()); assert!(d.aborted);
    let result = execute(&requests, budget(&requests), &mut Script::new(1)).unwrap();
    let exact = serde_json::to_vec(&result).unwrap().len() as u64;
    let mut b = budget(&requests); b.max_result_bytes = exact;
    assert!(execute(&requests, b, &mut Script::new(1)).is_ok());
    b.max_result_bytes -= 1; let mut d = Script::new(1);
    assert!(execute(&requests, b, &mut d).is_err()); assert!(d.aborted);
}
#[test]
fn empty_or_out_of_vocabulary_rows_fail_without_native_mutation() {
    let mut d = Script::new(0); assert!(execute(&[], budget(&[]), &mut d).is_err());
    let id = identity(); let p = programs(); let o = options(4);
    let requests = [request(&id, &[7], &p[0], &o, 7)]; let mut d = Script::new(1);
    assert!(execute(&requests, budget(&requests), &mut d).is_err()); assert!(d.groups.is_empty());
}
