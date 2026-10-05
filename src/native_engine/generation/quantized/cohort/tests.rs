//! Synthetic logits test cursor/scheduler wiring, NOT full-model parity.
use super::*;
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"cohort-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture-revision".into(), logical_model_digest: d,
        artifact_format: "synthetic-only".into(), quant_recipe: "fixture-int8".into(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "generate-v1".into(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".into(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".into(),
        sampler_version: "fixture".into(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.into(),
        host_class: None, compiler_identity: None }
}
fn plan(prompt: &[u32], id: &str, options: GenerationOptions) -> Int8GenerationPlan {
    Int8GenerationPlan::compile(prompt.to_vec(), options, identity(), id, 0, GenerationLimits::default()).unwrap()
}
struct Decoder;
impl DecodeByteDecoder for Decoder {
    type Error = &'static str;
    fn decode_token_ids(&self, tokens: &[u32]) -> Result<Vec<u8>, Self::Error> {
        Ok(tokens.iter().map(|&id| b'a' + id as u8).collect())
    }
}
#[derive(Default)] struct Control { stop_prompt: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    fn prefill_checkpoint(&mut self, index: usize) -> Option<DecodeCancellationKind> {
        (self.stop_prompt == Some(index)).then_some(DecodeCancellationKind::CostBudget)
    }
}
#[derive(Clone)] struct Script { tokens: Vec<u32>, heads: usize, preferred: Vec<u32>, flat: bool }
impl Script {
    fn new(preferred: &[u32]) -> Self { Self { tokens: Vec::new(), heads: 0, preferred: preferred.to_vec(), flat: false } }
    fn logits(&mut self) -> Vec<f32> {
        let token = self.preferred[self.heads.min(self.preferred.len() - 1)]; self.heads += 1;
        let mut logits = vec![-100.0; NANBEIGE_VOCAB_SIZE];
        if self.flat { logits[1..21].fill(0.0); } else { logits[token as usize] = 10.0; }
        logits
    }
    fn work(&self) -> Int8Work { Int8Work::for_sequence(0, self.tokens.len(), self.heads * NANBEIGE_VOCAB_SIZE).unwrap() }
}
struct Model {
    rows: Vec<Script>, control: Control, groups: Vec<Vec<usize>>, head_groups: Vec<Vec<usize>>,
    aborted: bool, fail_step: Option<usize>, corrupt: bool, bad_shape: bool, bad_work: bool,
}
impl Model {
    fn new(rows: Vec<Script>) -> Self { Self { rows, control: Control::default(), groups: Vec::new(), head_groups: Vec::new(),
        aborted: false, fail_step: None, corrupt: false, bad_shape: false, bad_work: false } }
}
impl GroupDriver for Model {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn append_group(&mut self, steps: &[CohortToken]) -> Result<(), Int8GenerationError> {
        if self.fail_step == Some(self.groups.len()) { return Err(StrictInt8Error::Primitive.into()); }
        self.groups.push(steps.iter().map(|step| step.sequence).collect());
        for step in steps { self.rows[step.sequence].tokens.push(step.token); } Ok(())
    }
    fn logits_group(&mut self, sequences: &[usize]) -> Result<Vec<f32>, Int8GenerationError> {
        self.head_groups.push(sequences.to_vec());
        let mut output = Vec::new(); for &slot in sequences { output.extend(self.rows[slot].logits()); }
        if self.corrupt { *output.last_mut().unwrap() = f32::NAN; }
        if self.bad_shape { output.pop(); } Ok(output)
    }
    fn work(&self, sequence: usize) -> Result<Int8Work, Int8GenerationError> {
        let mut work = self.rows[sequence].work(); if self.bad_work { work.attention_pairs += 1; } Ok(work)
    }
    fn abort(&mut self) { self.aborted = true; }
}
struct Scalar { script: Script, control: Control, aborted: bool }
impl Driver for Scalar {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn append(&mut self, token: u32) -> Result<(), Int8GenerationError> { self.script.tokens.push(token); Ok(()) }
    fn logits(&mut self) -> Result<Vec<f32>, Int8GenerationError> { Ok(self.script.logits()) }
    fn work(&self) -> Int8Work { self.script.work() }
    fn abort(&mut self) { self.aborted = true; }
}
#[derive(Default)] struct Sink(Vec<DecodeTokenEvent>);
impl DecodeEventSink for Sink {
    type Permit = (); type Error = &'static str;
    fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
    fn permit(&mut self, _: (), event: DecodeTokenEvent) -> Result<(), Self::Error> { self.0.push(event); Ok(()) }
}
fn request(plan: &Int8GenerationPlan, request_seq: u64) -> Int8CohortRequest<'_, Decoder> {
    Int8CohortRequest { plan, admitted_identity: plan.execution_identity(), decoder: &Decoder, request_seq,
        budget: Int8GenerationBudget { native: Int8RunBudget::exact(plan.work), max_kv_bytes: u64::MAX,
            max_sampler_bytes: plan.sampler_bytes() } }
}
#[test]
fn ragged_prompts_mix_decode_and_drop_finished_rows_without_extra_heads() {
    let plans = [plan(&[5], "a", GenerationOptions::greedy(4, 100, 0)),
        plan(&[5, 6, 7], "b", GenerationOptions::greedy(4, 100, 0))];
    let mut model = Model::new(vec![Script::new(&[1, 0]), Script::new(&[2, 0])]);
    let mut sink = Sink::default();
    let run = drive(&[request(&plans[0], 19), request(&plans[1], 3)], &mut sink, &mut model).unwrap();
    assert_eq!(model.groups, [vec![0, 1], vec![0, 1], vec![1], vec![1]]);
    assert_eq!(model.head_groups, [vec![0], vec![0], vec![1], vec![1]]);
    assert_eq!(model.rows[0].tokens, [5, 1]); assert_eq!(model.rows[1].tokens, [5, 6, 7, 2]);
    assert_eq!(run.sequences[0].sequence.request_seq, 19); assert_eq!(run.sequences[1].sequence.request_seq, 3);
    assert_eq!(run.group_steps, 4); assert!(!model.aborted);
    assert_eq!(sink.0.iter().map(|event| event.request_seq).collect::<Vec<_>>(), [19, 19, 3, 3]);
    assert_eq!(run.model_work, model.rows[0].work().checked_add(model.rows[1].work()).unwrap());
}
#[test]
fn seeded_rows_match_scalar_and_ignore_cohort_permutation_and_delivery_numbers() {
    let mut options = GenerationOptions::greedy(5, 100, 0); options.banned_token_ids = vec![0];
    options.sampling = GenerationSampling::Seeded { effective_seed: [29; 32], temperature_milli: 900,
        top_k: Some(20), top_p_ppm: 950_000 };
    let plans = [plan(&[5, 6], "a", options.clone()), plan(&[7], "b", options)];
    let mut script = Script::new(&[1]); script.flat = true;
    let run = drive(&[request(&plans[0], 9), request(&plans[1], 2)], &mut Discard,
        &mut Model::new(vec![script.clone(), script.clone()])).unwrap();
    for index in 0..2 {
        let mut scalar = Scalar { script: script.clone(), control: Control::default(), aborted: false };
        let reference = plans[index].drive(&Decoder, [9, 2][index], &mut Discard, &mut scalar).unwrap();
        assert!(run.sequences[index] == reference);
    }
    let permuted = drive(&[request(&plans[1], 77), request(&plans[0], 88)], &mut Discard,
        &mut Model::new(vec![script.clone(), script])).unwrap();
    assert_eq!(run.sequences[0].sequence.token_ids, permuted.sequences[1].sequence.token_ids);
    assert_eq!(run.sequences[1].sequence.token_ids, permuted.sequences[0].sequence.token_ids);
}
#[test]
fn rejected_byte_proposal_is_charged_and_never_emitted() {
    let p = plan(&[5], "a", GenerationOptions::greedy(4, 1, 0)); let mut sink = Sink::default();
    let run = drive(&[request(&p, 1)], &mut sink, &mut Model::new(vec![Script::new(&[1])])).unwrap();
    assert_eq!(run.sequences[0].sequence.finish_reason, GenerationFinish::ByteLimit);
    assert_eq!(run.sequences[0].sequence.token_ids, [1]); assert_eq!(sink.0.len(), 1);
    assert_eq!(run.model_work, Int8Work::for_sequence(0, 2, 2 * NANBEIGE_VOCAB_SIZE).unwrap());
}
#[test]
fn invalid_delivery_or_identity_refuses_before_any_driver_work() {
    let p = plan(&[5], "a", GenerationOptions::greedy(1, 100, 0));
    for ids in [[0, 2], [1, 1]] {
        let mut model = Model::new(vec![Script::new(&[1]), Script::new(&[1])]);
        assert!(drive(&[request(&p, ids[0]), request(&p, ids[1])], &mut Discard, &mut model).is_err());
        assert!(model.groups.is_empty());
    }
    let mut id = p.execution_identity().clone(); id.tokenizer_digest = Sha256Digest::of_bytes(b"other");
    let mut r = request(&p, 1); r.admitted_identity = &id;
    assert!(validate_requests(&[r]).is_err());
}
#[test]
fn every_aggregate_ceiling_is_independent_of_per_row_ceiling() {
    let required = Int8CohortRequirements { planned_work: Int8Work::for_sequence(0, 3, NANBEIGE_VOCAB_SIZE).unwrap(),
        kv_bytes: 4096, sampler_bytes: 8192 };
    let budget = Int8CohortBudget { native: Int8RunBudget::exact(required.planned_work), max_kv_bytes: 4096, max_sampler_bytes: 8192 };
    check_aggregate(required, budget).unwrap();
    for axis in 0..6 { let mut short = budget;
        match axis { 0 => short.max_kv_bytes -= 1, 1 => short.max_sampler_bytes -= 1,
            2 => short.native.max_forward_positions -= 1, 3 => short.native.max_attention_pairs -= 1,
            4 => short.native.max_projection_work.dot_products -= 1, _ => short.native.max_projection_work.multiply_accumulates -= 1 }
        assert!(check_aggregate(required, short).is_err());
    }
}
#[test]
fn cancellation_and_native_fault_abort_without_retry_or_completion() {
    let p = plan(&[5, 6, 7], "a", GenerationOptions::greedy(2, 100, 0));
    let mut model = Model::new(vec![Script::new(&[1])]); model.control.stop_prompt = Some(1);
    let error = drive(&[request(&p, 1)], &mut Discard, &mut model).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::CostBudget));
    assert_eq!(model.groups.len(), 1); assert!(model.aborted);
    let mut model = Model::new(vec![Script::new(&[1])]); model.fail_step = Some(1);
    assert!(drive(&[request(&p, 1)], &mut Discard, &mut model).is_err());
    assert_eq!(model.groups.len(), 1); assert!(model.aborted);
}
#[test]
fn malformed_head_and_forged_work_cannot_become_success() {
    let p = plan(&[5], "a", GenerationOptions::greedy(1, 100, 0));
    for axis in 0..3 {
        let mut model = Model::new(vec![Script::new(&[1])]);
        match axis { 0 => model.corrupt = true, 1 => model.bad_shape = true, _ => model.bad_work = true }
        assert!(drive(&[request(&p, 1)], &mut Discard, &mut model).is_err()); assert!(model.aborted);
    }
}
#[test]
fn uncertain_sink_delivery_is_not_retried() {
    struct Broken(usize);
    impl DecodeEventSink for Broken {
        type Permit = (); type Error = &'static str;
        fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
        fn permit(&mut self, _: (), _: DecodeTokenEvent) -> Result<(), Self::Error> { self.0 += 1; Err("uncertain") }
    }
    let p = plan(&[5], "a", GenerationOptions::greedy(2, 100, 0));
    let mut model = Model::new(vec![Script::new(&[1])]); let mut sink = Broken(0);
    assert!(drive(&[request(&p, 1)], &mut sink, &mut model).is_err()); assert_eq!(sink.0, 1); assert!(model.aborted);
}
