//! Scheduler/cursor tests use synthetic logits, not full-model parity evidence.
use super::*;

fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"int4-batch-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: recipe(128), packing_set_digest: packing_digest(d, 128).unwrap(),
        tokenizer_digest: d, template_digest: d, task_spec: "generate-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn plan(prompt: &[u32], item: &str, options: GenerationOptions) -> Int4GenerationPlan {
    Int4GenerationPlan::compile(prompt.to_vec(), options, identity(), 128, item, 0, GenerationLimits::default()).unwrap()
}
fn unlimited() -> Int4BatchGenerationBudget {
    Int4BatchGenerationBudget { native: Int4RunBudget { max_forward_positions: u64::MAX,
        max_attention_pairs: u64::MAX, max_projection_work: crate::native_engine::portable_int8::ProjectionWork {
            dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
        max_native_payload_bytes: u64::MAX, max_sampler_payload_bytes: u64::MAX, max_output_payload_bytes: u64::MAX }
}
fn requests(plans: &[Int4GenerationPlan]) -> Vec<Int4BatchGenerationRequest<'_>> {
    plans.iter().enumerate().map(|(row, plan)| Int4BatchGenerationRequest {
        plan, admitted_identity: plan.execution_identity(), request_seq: row as u64 + 11,
        budget: Int4GenerationBudget { native: Int4RunBudget::exact(plan.work),
            max_kv_bytes: plan.work.forward_positions * KV_BYTES_PER_TOKEN as u64,
            max_sampler_bytes: plan.sampler_bytes() } }).collect()
}
fn admission(plans: &[Int4GenerationPlan]) -> Int4BatchGenerationRequirements {
    requirements(&requests(plans), &plans.iter().map(|p| p.work.forward_positions as usize).collect::<Vec<_>>(), unlimited()).unwrap()
}
#[derive(Default)] struct Control { polls: usize, cancel: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1; (self.cancel == Some(self.polls)).then_some(DecodeCancellationKind::Deadline)
    }
}
struct Model {
    control: Control, tokens: Vec<Vec<u32>>, heads: Vec<usize>, scripts: Vec<Vec<u32>>,
    ticks: Vec<Vec<(usize, bool)>>, aborted: bool, flat: bool,
    bad_route: bool, bad_logits: bool, bad_work: bool, fail_tick: Option<usize>,
}
impl Model {
    fn new(scripts: &[&[u32]]) -> Self {
        Self { control: Control::default(), tokens: vec![Vec::new(); scripts.len()], heads: vec![0; scripts.len()],
            scripts: scripts.iter().map(|s| s.to_vec()).collect(), ticks: Vec::new(), aborted: false, flat: false,
            bad_route: false, bad_logits: false, bad_work: false, fail_tick: None }
    }
    fn logits(&mut self, row: usize) -> Vec<f32> {
        let script = &self.scripts[row]; let id = script[self.heads[row].min(script.len() - 1)];
        self.heads[row] += 1;
        let mut values = vec![-100.0; NANBEIGE_VOCAB_SIZE];
        if self.flat { values[1..21].fill(0.0); } else { values[id as usize] = 10.0; }
        values
    }
}
impl BatchDriver for Model {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn abort(&mut self) { self.aborted = true; }
    fn work(&self, row: usize) -> Result<Int4Work, Int4GenerationError> {
        let mut work = Int4Work::for_sequence(128, 0, self.tokens[row].len(), self.heads[row] * NANBEIGE_VOCAB_SIZE)?;
        if self.bad_work && row + 1 == self.tokens.len() { work.projections.dot_products += 1; }
        Ok(work)
    }
    fn step(&mut self, tokens: &[Step]) -> Result<StepOutput, Int4GenerationError> {
        if self.fail_tick == Some(self.ticks.len()) { return Err(StrictInt4Error::Primitive.into()); }
        self.ticks.push(tokens.iter().map(|t| (t.row, t.project)).collect());
        let mut rows = Vec::new(); let mut logits = Vec::new();
        for token in tokens {
            rows.push(Forwarded { row: token.row, position: self.tokens[token.row].len(), project: token.project });
            self.tokens[token.row].push(token.token);
            if token.project { logits.extend(self.logits(token.row)); }
        }
        if self.bad_route { rows.last_mut().unwrap().position += 1; }
        if self.bad_logits { *logits.last_mut().unwrap() = f32::NAN; }
        Ok(StepOutput { rows, logits })
    }
}
struct Decoder;
impl DecodeByteDecoder for Decoder {
    type Error = ();
    fn decode_token_ids(&self, ids: &[u32]) -> Result<Vec<u8>, ()> {
        Ok(ids.iter().map(|&id| b'a' + id as u8).collect())
    }
}
#[derive(Default)] struct Sink(Vec<DecodeTokenEvent>);
impl DecodeEventSink for Sink {
    type Permit = (); type Error = ();
    fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), ()> { Ok(()) }
    fn permit(&mut self, _: (), event: DecodeTokenEvent) -> Result<(), ()> { self.0.push(event); Ok(()) }
}
fn run(plans: &[Int4GenerationPlan], model: &mut Model, sink: &mut impl DecodeEventSink)
    -> Result<Int4BatchGenerationOutput, Int4GenerationError> {
    let required = admission(plans);
    let cursors = plans.iter().enumerate().map(|(i, p)|
        cursor::Cursor::new(&p.plan, i as u64 + 11, INT4_GENERATION_VERSION).unwrap()).collect();
    drive(cursors, required, model, &Decoder, sink)
}
struct Serial<'a> { model: &'a mut Model, row: usize }
impl super::super::Driver for Serial<'_> {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.model.control }
    fn append(&mut self, token: u32) -> Result<(), Int4GenerationError> { self.model.tokens[self.row].push(token); Ok(()) }
    fn logits(&mut self) -> Result<Vec<f32>, Int4GenerationError> { Ok(self.model.logits(self.row)) }
    fn work(&self) -> Int4Work { BatchDriver::work(&*self.model, self.row).unwrap() }
    fn abort(&mut self) { self.model.aborted = true; }
}

#[test]
fn ragged_prefill_and_decode_share_ticks_without_extra_heads_or_finished_feedback() {
    let plans = [plan(&[5], "short", GenerationOptions::greedy(6, 100, 0)),
        plan(&[6, 7, 8], "long", GenerationOptions::greedy(6, 100, 0))];
    let mut model = Model::new(&[&[1, 0], &[2, 3, 0]]); let mut sink = Sink::default();
    let out = run(&plans, &mut model, &mut sink).unwrap();
    assert_eq!(out.group_steps, 5);
    assert_eq!(model.tokens, [vec![5, 1], vec![6, 7, 8, 2, 3]]);
    assert_eq!(model.heads, [2, 3]);
    assert_eq!(model.ticks[0], [(0, true), (1, false)]);
    assert_eq!(model.ticks[2], [(1, true)]);
    assert_eq!(out.runs[0].sequence.token_ids, [1, 0]);
    assert_eq!(out.runs[1].sequence.token_ids, [2, 3, 0]);
    assert_eq!(sink.0.iter().map(|e| e.request_seq).collect::<Vec<_>>(), [11, 11, 12, 12, 12]);
    assert_eq!(out.actual_work, out.runs[0].model_work.checked_add(out.runs[1].model_work).unwrap());
    assert!(!model.aborted);
}

#[test]
fn seeded_and_greedy_rows_match_the_existing_serial_driver_including_logprobs() {
    let mut seeded = GenerationOptions::greedy(4, 100, 0);
    seeded.banned_token_ids = vec![0]; seeded.capture_logprobs = true;
    seeded.sampling = GenerationSampling::Seeded { effective_seed: [17; 32], temperature_milli: 800,
        top_k: Some(20), top_p_ppm: 950000 };
    let plans = [plan(&[5, 6], "first", seeded), plan(&[7], "second", GenerationOptions::greedy(3, 100, 0))];
    let mut batch = Model::new(&[&[1], &[2]]); batch.flat = true;
    let result = run(&plans, &mut batch, &mut Sink::default()).unwrap();
    for (row, plan) in plans.iter().enumerate() {
        let mut model = Model::new(&[&[1], &[2]]); model.flat = true;
        let serial = plan.drive(&Decoder, row as u64 + 11, &mut Discard, &mut Serial { model: &mut model, row }).unwrap();
        assert!(result.runs[row] == serial);
    }
}

#[test]
fn row_permutation_and_delivery_numbers_do_not_change_the_semantic_sampler() {
    let mut options = GenerationOptions::greedy(5, 100, 0); options.banned_token_ids = vec![0];
    options.sampling = GenerationSampling::Seeded { effective_seed: [23; 32], temperature_milli: 1000,
        top_k: Some(20), top_p_ppm: 1000000 };
    let mut plans = vec![plan(&[5], "one", options.clone()), plan(&[6, 7, 8], "two", options)];
    let mut a = Model::new(&[&[1], &[1]]); a.flat = true;
    let before = run(&plans, &mut a, &mut Discard).unwrap();
    plans.reverse(); let mut b = Model::new(&[&[1], &[1]]); b.flat = true;
    let after = run(&plans, &mut b, &mut Discard).unwrap();
    for row in 0..2 {
        assert_eq!(before.runs[row].sequence.token_ids, after.runs[1 - row].sequence.token_ids);
        assert_eq!(before.runs[row].model_work, after.runs[1 - row].model_work);
    }
}

#[test]
fn byte_refusal_and_suffix_stop_retire_independently_without_extra_native_steps() {
    let mut suffix = GenerationOptions::greedy(8, 100, 0); suffix.stop_suffixes = vec![b"bc".to_vec()];
    let plans = [plan(&[5], "bytes", GenerationOptions::greedy(8, 1, 0)), plan(&[6], "suffix", suffix)];
    let mut model = Model::new(&[&[1], &[1, 2]]);
    let out = run(&plans, &mut model, &mut Discard).unwrap();
    assert_eq!(out.group_steps, 2); assert_eq!(model.tokens, [vec![5, 1], vec![6, 1]]);
    assert_eq!(out.runs[0].sequence.finish_reason, GenerationFinish::ByteLimit);
    assert_eq!(out.runs[0].sequence.token_ids, [1]);
    assert_eq!(out.runs[1].sequence.finish_reason, GenerationFinish::StopSuffix);
    assert_eq!(out.runs[1].sequence.token_ids, [1, 2]);
}

#[test]
fn late_invalid_native_row_is_rejected_before_any_token_of_the_tick() {
    let plans = [plan(&[5], "a", GenerationOptions::greedy(2, 100, 0)), plan(&[6], "b", GenerationOptions::greedy(2, 100, 0))];
    for route in [true, false] {
        let mut model = Model::new(&[&[1], &[2]]); model.bad_route = route; model.bad_logits = !route;
        let mut sink = Sink::default(); assert!(run(&plans, &mut model, &mut sink).is_err());
        assert!(sink.0.is_empty()); assert!(model.aborted);
    }
}

#[test]
fn native_cancellation_delivery_and_work_failures_abort_the_entire_run() {
    let plans = [plan(&[5], "a", GenerationOptions::greedy(3, 100, 0)), plan(&[6], "b", GenerationOptions::greedy(3, 100, 0))];
    for axis in 0..3 {
        let mut model = Model::new(&[&[1], &[2]]);
        match axis { 0 => model.fail_tick = Some(1), 1 => model.control.cancel = Some(1), _ => model.bad_work = true }
        assert!(run(&plans, &mut model, &mut Discard).is_err()); assert!(model.aborted);
    }
    struct FailedSink(usize);
    impl DecodeEventSink for FailedSink {
        type Permit = (); type Error = ();
        fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), ()> { Ok(()) }
        fn permit(&mut self, _: (), _: DecodeTokenEvent) -> Result<(), ()> { self.0 += 1; Err(()) }
    }
    let mut model = Model::new(&[&[1], &[2]]); let mut sink = FailedSink(0);
    assert!(run(&plans, &mut model, &mut sink).is_err()); assert_eq!(sink.0, 1); assert!(model.aborted);
}

#[test]
fn whole_cohort_admission_prices_independent_attention_and_all_allocated_capacity() {
    let plans = [plan(&[1], "a", GenerationOptions::greedy(2, 100, 0)),
        plan(&[2, 3, 4], "b", GenerationOptions::greedy(3, 200, 0))];
    let required = admission(&plans);
    assert_eq!(required.planned_work, plans[0].work.checked_add(plans[1].work).unwrap());
    assert_eq!(required.native.kv_bytes, (2 + 5) * KV_BYTES_PER_TOKEN as u64);
    assert_eq!(required.sampler_payload_bytes, plans[0].sampler_bytes() + plans[1].sampler_bytes());
    let contexts = [2, 5]; let requests = requests(&plans);
    let exact = Int4BatchGenerationBudget { native: Int4RunBudget::exact(required.planned_work),
        max_native_payload_bytes: required.native_payload_bytes, max_sampler_payload_bytes: required.sampler_payload_bytes,
        max_output_payload_bytes: required.output_payload_upper_bytes };
    assert_eq!(requirements(&requests, &contexts, exact).unwrap(), required);
    for axis in 0..7 {
        let mut b = exact;
        match axis { 0 => b.native.max_forward_positions -= 1, 1 => b.native.max_attention_pairs -= 1,
            2 => b.native.max_projection_work.dot_products -= 1, 3 => b.native.max_projection_work.multiply_accumulates -= 1,
            4 => b.max_native_payload_bytes -= 1, 5 => b.max_sampler_payload_bytes -= 1, _ => b.max_output_payload_bytes -= 1 }
        assert!(requirements(&requests, &contexts, b).is_err());
    }
}

#[test]
fn last_member_cannot_borrow_surplus_budget_or_substitute_identity_or_delivery_id() {
    let plans = [plan(&[1], "a", GenerationOptions::greedy(2, 100, 0)), plan(&[2], "b", GenerationOptions::greedy(2, 100, 0))];
    for axis in 0..4 {
        let mut req = requests(&plans); let mut changed = plans[1].execution_identity().clone();
        changed.tokenizer_digest = Sha256Digest::of_bytes(b"different-tokenizer");
        match axis { 0 => req[1].budget.native.max_projection_work.dot_products -= 1,
            1 => req[1].admitted_identity = &changed, 2 => req[1].request_seq = req[0].request_seq, _ => req[1].request_seq = 0 }
        assert!(requirements(&req, &[2, 2], unlimited()).is_err());
    }
    assert!(requirements(&requests(&plans), &[2, 3], unlimited()).is_err());
    assert!(requirements(&requests(&plans), &[2], unlimited()).is_err());
    assert!(requirements(&[], &[], unlimited()).is_err());
}

#[test]
fn all_sixty_four_rows_are_scheduled_and_sixty_five_are_refused() {
    let plans: Vec<_> = (0..64).map(|i| plan(&[1], &format!("row-{i}"), GenerationOptions::greedy(1, 10, 0))).collect();
    let scripts = vec![&[0_u32][..]; 64]; let mut model = Model::new(&scripts);
    let out = run(&plans, &mut model, &mut Discard).unwrap();
    assert_eq!(out.group_steps, 1); assert_eq!(out.runs.len(), 64); assert_eq!(model.ticks[0].len(), 64);
    let mut req = requests(&plans); req.push(Int4BatchGenerationRequest { plan: &plans[0],
        admitted_identity: plans[0].execution_identity(), request_seq: 100, budget: req[0].budget });
    assert!(requirements(&req, &[1; 65], unlimited()).is_err());
}
