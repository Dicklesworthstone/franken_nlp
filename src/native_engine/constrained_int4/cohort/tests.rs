//! Deterministic native-driver/grammar fixtures; no model parity claim.
use super::*;
use crate::{execution_identity::{NumericsProfile, Sha256Digest},
    grammar::{CompileLimits, runtime::SourceRuntimeLimits}};

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
    groups: Vec<Vec<(usize, u32)>>, heads: Vec<Vec<usize>>, aborted: bool,
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
    fn append_group(&mut self, steps: &[Int4CohortToken]) -> Result<(), Int4JsonError> {
        assert!(!steps.is_empty());
        assert!(steps.windows(2).all(|pair| pair[0].sequence < pair[1].sequence));
        self.groups.push(steps.iter().map(|step| (step.sequence, step.token)).collect());
        for step in steps {
            self.seen[step.sequence].push(step.token);
            if self.fail_group == Some(self.groups.len()) { return Err(StrictInt4Error::Primitive.into()); }
        }
        Ok(())
    }
    fn logits_group(&mut self, slots: &[usize]) -> Result<Vec<f32>, Int4JsonError> {
        assert!(slots.windows(2).all(|pair| pair[0] < pair[1]));
        self.heads.push(slots.to_vec()); let mut logits = Vec::new();
        for &slot in slots {
            self.projected[slot] += self.scores[slot].len();
            logits.extend_from_slice(&self.scores[slot]);
        }
        if self.bad_head { logits.pop(); }
        Ok(logits)
    }
    fn work(&self, slot: usize) -> Result<Int4Work, Int4JsonError> {
        let mut work = Int4Work::for_sequence(GROUP, 0, self.seen[slot].len(), self.projected[slot])?;
        if self.bad_work { work.attention_pairs += 1; }
        Ok(work)
    }
    fn abort(&mut self) { self.aborted = true; }
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"private structured cohort fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "fixture".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: crate::native_engine::strict_int4::STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn vocabulary() -> Table {
    Table([b"".as_slice(), b"true", b"false", b"1", b"0", br#""Alice""#, br#""Mallory""#]
        .into_iter().map(<[u8]>::to_vec).collect())
}
fn programs() -> Vec<JsonProgram> {
    let limits = CompileLimits { max_output_bytes: 128, ..CompileLimits::default() };
    vec![JsonProgram::compile(r#"{"type":"boolean"}"#, limits).unwrap(),
        JsonProgram::compile(r#"{"type":"integer","enum":[1,10]}"#, limits).unwrap(),
        JsonProgram::compile_with_source(r#"{"type":"string","maxLength":16,"x-fnlp-source":"verbatim"}"#,
            "Alice", limits, SourceRuntimeLimits::default()).unwrap()]
}
fn options(tokens: usize) -> JsonDecodeOptions {
    JsonDecodeOptions { max_new_tokens: tokens, eos_token_id: 0, excluded_token_ids: [0].into_iter().collect() }
}
fn request<'a>(id: &'a ExecutionIdentity, prompt: &'a [u32], program: &'a JsonProgram,
    options: &'a JsonDecodeOptions) -> Int4JsonCohortRequest<'a> {
    let work = work_for_width(GROUP, prompt.len(), options.max_new_tokens, 7).unwrap();
    Int4JsonCohortRequest { identity: id, prompt, program, options,
        budget: Int4JsonBudget { native: Int4RunBudget::exact(work), json: JsonWorkBudget {
            max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
            max_kv_bytes: 8192 * KV_BYTES_PER_TOKEN as u64,
            max_total_mask_node_visits: options.max_new_tokens as u64 * 7,
            mask_limits: MaskWorkLimits::default() } } }
}
fn budget(requests: &[Int4JsonCohortRequest<'_>]) -> Int4JsonCohortBudget {
    let mut work = Int4Work::default(); let mut masks = 0;
    for request in requests {
        work = work.checked_add(work_for_width(GROUP, request.prompt.len(), request.options.max_new_tokens, 7).unwrap()).unwrap();
        masks += request.budget.json.max_total_mask_node_visits;
    }
    Int4JsonCohortBudget { native: Int4RunBudget::exact(work), max_kv_bytes: u64::MAX,
        max_mask_node_visits: masks, max_result_bytes: 1 << 20 }
}
fn execute(requests: &[Int4JsonCohortRequest<'_>], budget: Int4JsonCohortBudget, driver: &mut Script)
    -> Result<Int4JsonCohortRun, Int4JsonError> {
    execution::drive(GROUP, requests, &vocabulary(), budget, driver, Ok)
}

#[test]
fn ragged_decoder_and_full_head_groups_preserve_independent_source_languages() {
    let id = identity(); let p = programs(); let o = options(4);
    let requests = [request(&id, &[6], &p[0], &o), request(&id, &[6, 6, 6], &p[1], &o),
        request(&id, &[6, 6], &p[2], &o)];
    let mut driver = Script::new(3); let result = execute(&requests, budget(&requests), &mut driver).unwrap();
    assert_eq!(result.sequences.iter().map(|row| row.output.json.as_str()).collect::<Vec<_>>(), ["true", "10", "\"Alice\""]);
    assert_eq!(driver.groups, [vec![(0, 6), (1, 6), (2, 6)], vec![(0, 1), (1, 6), (2, 6)],
        vec![(1, 6), (2, 5)], vec![(1, 3)], vec![(1, 4)]]);
    assert_eq!(driver.heads, [vec![0], vec![0, 2], vec![1, 2], vec![1], vec![1]]);
    assert_eq!(result.group_steps, 5); assert_eq!(result.projection_groups, 5);
    assert_eq!(result.model_work.forward_positions, 10); assert_eq!(result.model_work.projected_logits, 49);
    assert_eq!(driver.projected, [14, 21, 14]); assert!(!driver.aborted);
    assert_eq!(result.sequences[2].output.token_ids, [5, 0]);
    assert!(result.sequences.iter().all(|row| row.execution == INT4_JSON_EXECUTION && row.group_width == GROUP));
}

struct Single<'a> { driver: &'a mut Script, slot: usize }
impl Driver for Single<'_> {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.driver.control }
    fn append(&mut self, token: u32) -> Result<(), Int4JsonError> {
        self.driver.append_group(&[Int4CohortToken { sequence: self.slot, token }])
    }
    fn logits(&mut self) -> Result<Vec<f32>, Int4JsonError> { self.driver.logits_group(&[self.slot]) }
    fn work(&self) -> Int4Work { GroupDriver::work(self.driver, self.slot).unwrap() }
    fn abort(&mut self) { self.driver.abort(); }
}
#[test]
fn each_complete_row_equals_the_existing_serial_decoder_including_native_work() {
    let id = identity(); let p = programs(); let o = options(4); let vocabulary = vocabulary();
    let requests = [request(&id, &[6], &p[0], &o), request(&id, &[6, 6, 6], &p[1], &o),
        request(&id, &[6, 6], &p[2], &o)];
    let grouped = execute(&requests, budget(&requests), &mut Script::new(3)).unwrap();
    for (slot, request) in requests.iter().enumerate() {
        let mut script = Script::new(3); let mut single = Single { driver: &mut script, slot };
        let serial: Int4JsonRun = super::super::drive(GROUP, request.prompt, request.program,
            &vocabulary, request.options, request.budget, &mut single, Ok::<_, Int4JsonError>).unwrap();
        assert_eq!(grouped.sequences[slot], serial);
    }
}
#[test]
fn simultaneous_ready_rows_share_the_full_head_instead_of_serial_projection_calls() {
    let id = identity(); let p = programs(); let mut a = options(4); a.excluded_token_ids.insert(1);
    let b = options(4);
    let requests = [request(&id, &[6], &p[0], &a), request(&id, &[6], &p[0], &b)];
    let mut driver = Script::new(2); let result = execute(&requests, budget(&requests), &mut driver).unwrap();
    assert_eq!(result.sequences[0].output.json, "false"); assert_eq!(result.sequences[1].output.json, "true");
    assert_eq!(driver.heads, [vec![0, 1], vec![0, 1]]);
    assert_eq!(result.projection_groups, 2); assert_eq!(result.model_work.projected_logits, 28);
}
#[test]
fn every_row_and_aggregate_work_axis_refuses_before_the_first_native_group() {
    let id = identity(); let p = programs(); let o = options(4);
    for axis in 0..13 {
        let mut requests = [request(&id, &[6], &p[0], &o), request(&id, &[6], &p[1], &o)];
        let mut cap = budget(&requests);
        match axis {
            0 => cap.native.max_forward_positions -= 1, 1 => cap.native.max_attention_pairs -= 1,
            2 => cap.native.max_projection_work.dot_products -= 1,
            3 => cap.native.max_projection_work.multiply_accumulates -= 1, 4 => cap.max_mask_node_visits -= 1,
            5 => requests[1].budget.native.max_forward_positions -= 1,
            6 => requests[1].budget.json.max_projected_logits -= 1,
            7 => requests[1].budget.json.max_total_mask_node_visits -= 1,
            8 => requests[1].budget.native.max_attention_pairs -= 1,
            9 => requests[1].budget.native.max_projection_work.dot_products -= 1,
            10 => requests[1].budget.native.max_projection_work.multiply_accumulates -= 1,
            11 => cap.max_result_bytes = 0, _ => cap.max_result_bytes = 255,
        }
        let mut driver = Script::new(2);
        assert!(execute(&requests, cap, &mut driver).is_err(), "axis {axis}");
        assert!(driver.groups.is_empty()); assert!(driver.aborted);
    }
}
#[test]
fn allocated_capacity_is_charged_per_row_and_as_an_aggregate() {
    let id = identity(); let p = programs(); let o = options(4);
    let mut requests = [request(&id, &[6], &p[0], &o), request(&id, &[6], &p[1], &o)];
    let total = 12 * KV_BYTES_PER_TOKEN as u64;
    check_capacities(GROUP, &requests, &[4, 8], total).unwrap();
    assert!(check_capacities(GROUP, &requests, &[4, 8], total - 1).is_err());
    requests[1].budget.json.max_kv_bytes = 7 * KV_BYTES_PER_TOKEN as u64;
    assert!(check_capacities(GROUP, &requests, &[4, 8], total).is_err());
    assert!(check_capacities(GROUP, &requests, &[3, 8], total).is_err());
    assert!(check_capacities(GROUP, &requests, &[4], total).is_err());
}
#[test]
fn a_late_missing_eos_discards_an_already_completed_sibling() {
    let id = identity(); let p = programs(); let a = options(4); let b = options(1);
    let requests = [request(&id, &[6], &p[0], &a), request(&id, &[6, 6, 6], &p[1], &b)];
    let mut driver = Script::new(2);
    assert!(execute(&requests, budget(&requests), &mut driver).is_err()); assert!(driver.aborted);
    assert_eq!(driver.seen[0], [6, 1]); assert_eq!(driver.seen[1], [6, 6, 6]);
}
#[test]
fn invalid_full_vocabulary_logits_do_not_hide_behind_illegal_masks() {
    let id = identity(); let p = programs(); let o = options(4);
    let requests = [request(&id, &[6], &p[0], &o), request(&id, &[6], &p[1], &o)];
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut driver = Script::new(2); driver.scores[1][6] = value;
        assert!(matches!(execute(&requests, budget(&requests), &mut driver),
            Err(Int4JsonError::Decode(JsonDecodeError::InvalidLogits))));
        assert!(driver.aborted);
    }
}
#[test]
fn native_partial_group_bad_shape_and_wrong_work_poison_the_entire_collection() {
    let id = identity(); let p = programs(); let o = options(4);
    let requests = [request(&id, &[6], &p[0], &o), request(&id, &[6], &p[1], &o)];
    for mode in 0..3 {
        let mut driver = Script::new(2);
        match mode { 0 => driver.fail_group = Some(2), 1 => driver.bad_head = true, _ => driver.bad_work = true }
        assert!(execute(&requests, budget(&requests), &mut driver).is_err()); assert!(driver.aborted);
    }
}
#[test]
fn cancellation_covers_prefill_every_checkpoint_and_post_finalization() {
    let id = identity(); let p = programs(); let o = options(4);
    let requests = [request(&id, &[6, 6], &p[0], &o)];
    let mut baseline = Script::new(1); execute(&requests, budget(&requests), &mut baseline).unwrap();
    for stop in 1..=baseline.control.calls {
        let mut driver = Script::new(1); driver.control.stop = Some(stop);
        assert_eq!(execute(&requests, budget(&requests), &mut driver).unwrap_err().cancellation(),
            Some(DecodeCancellationKind::Deadline)); assert!(driver.aborted);
    }
    let mut driver = Script::new(1); driver.control.prefill = Some(1);
    assert_eq!(execute(&requests, budget(&requests), &mut driver).unwrap_err().cancellation(),
        Some(DecodeCancellationKind::PollQuota)); assert!(driver.aborted);
    let mut driver = Script::new(1); driver.control.stop = Some(baseline.control.calls);
    let mut finalized = false;
    let result: Result<_, Int4JsonError> = execution::drive(GROUP, &requests, &vocabulary(), budget(&requests), &mut driver,
        |run| { finalized = true; Ok(run) });
    assert!(result.unwrap_err().cancellation().is_some()); assert!(finalized); assert!(driver.aborted);
}
#[test]
fn finalizer_and_exact_envelope_failures_do_not_escape_native_ownership() {
    let id = identity(); let p = programs(); let o = options(4);
    let requests = [request(&id, &[6], &p[0], &o)]; let mut driver = Script::new(1);
    let result: Result<(), Int4JsonError> = execution::drive(GROUP, &requests, &vocabulary(), budget(&requests), &mut driver,
        |_| Err(JsonDecodeError::IndependentValidation.into()));
    assert!(result.is_err()); assert!(driver.aborted);
    let raw = execute(&requests, budget(&requests), &mut Script::new(1)).unwrap();
    let exact = serde_json::to_vec(&raw).unwrap().len() as u64;
    let mut cap = budget(&requests); cap.max_result_bytes = exact;
    execute(&requests, cap, &mut Script::new(1)).unwrap();
    cap.max_result_bytes -= 1; let mut driver = Script::new(1);
    assert!(execute(&requests, cap, &mut driver).is_err()); assert!(driver.aborted);
}
#[test]
fn task_fields_can_differ_but_model_tokenizer_numerics_and_host_fields_cannot() {
    let a = identity(); let mut b = a.clone();
    b.task_spec = "ner-v1".to_owned(); b.prompt_digest = Sha256Digest::of_bytes(b"independent private document");
    b.schema_digest = Sha256Digest::of_bytes(b"independent schema"); same_model(&a, &b).unwrap();
    for field in 0..5 {
        let mut bad = b.clone(); let other = Sha256Digest::of_bytes(b"foreign model setting");
        match field { 0 => bad.tokenizer_digest = other, 1 => bad.logical_model_digest = other,
            2 => bad.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            3 => bad.host_class = Some("foreign-host".to_owned()), _ => bad.packing_set_digest = other }
        assert!(same_model(&a, &bad).is_err());
    }
}
#[test]
fn workspace_quotes_all_rows_tokens_buffers_and_refuses_overflow() {
    let mask = NANBEIGE_VOCAB_SIZE.div_ceil(8) as u64;
    assert_eq!(required_workspace_bytes(3, 12, 4096).unwrap(), 64 * 12 + 8 * 4096 + 65536 * 3 + mask);
    for (count, tokens, bytes) in [(0, 1, 1), (65, 1, 1), (1, 0, 1), (1, 1, 0),
        (1, DEFAULT_ADMITTED_CONTEXT_CAP as u64 + 1, 1), (1, 1, u64::MAX)] {
        assert!(required_workspace_bytes(count, tokens, bytes).is_err());
    }
}
#[test]
fn empty_and_out_of_vocabulary_requests_fail_before_any_native_mutation() {
    let mut driver = Script::new(0); assert!(execute(&[], budget(&[]), &mut driver).is_err());
    let id = identity(); let p = programs(); let o = options(4);
    let requests = [request(&id, &[7], &p[0], &o)]; let mut driver = Script::new(1);
    assert!(execute(&requests, budget(&requests), &mut driver).is_err()); assert!(driver.groups.is_empty());
}
