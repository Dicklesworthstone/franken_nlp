//! Synthetic native seam exercises the real cursor; not full-model parity.
use super::*;
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"refill-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".into(), logical_model_digest: d,
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
fn limits(width: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: width,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(width).unwrap() }
}
struct Decoder;
impl DecodeByteDecoder for Decoder {
    type Error = &'static str;
    fn decode_token_ids(&self, tokens: &[u32]) -> Result<Vec<u8>, Self::Error> { Ok(tokens.iter().map(|&id| b'a' + id as u8).collect()) }
}
fn request(plan: &Int8GenerationPlan, seq: u64) -> Int8CohortRequest<'_, Decoder> {
    Int8CohortRequest { plan, admitted_identity: plan.execution_identity(), decoder: &Decoder, request_seq: seq,
        budget: Int8GenerationBudget { native: Int8RunBudget::exact(plan.work), max_kv_bytes: u64::MAX,
            max_sampler_bytes: plan.sampler_bytes() } }
}
#[derive(Default)] struct Control { polls: usize, stop: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1; (self.stop == Some(self.polls)).then_some(DecodeCancellationKind::CostBudget)
    }
    fn prefill_checkpoint(&mut self, i: usize) -> Option<DecodeCancellationKind> { self.checkpoint(i) }
}
#[derive(Clone)] struct Script { tokens: Vec<u32>, heads: usize, preferred: Vec<u32>, flat: bool }
impl Script {
    fn new(tokens: &[u32]) -> Self { Self { tokens: Vec::new(), heads: 0, preferred: tokens.to_vec(), flat: false } }
    fn logits(&mut self) -> Vec<f32> {
        let token = self.preferred[self.heads.min(self.preferred.len() - 1)]; self.heads += 1;
        let mut logits = vec![-100.0; NANBEIGE_VOCAB_SIZE];
        if self.flat { logits[1..21].fill(0.0); } else { logits[token as usize] = 10.0; } logits
    }
    fn work(&self) -> Int8Work { Int8Work::for_sequence(0, self.tokens.len(), self.heads * NANBEIGE_VOCAB_SIZE).unwrap() }
}
struct Model {
    rows: Vec<Script>, slots: Vec<Option<usize>>, queued: usize, control: Control,
    groups: Vec<Vec<(usize, usize)>>, retired: Vec<usize>, aborted: bool, fault: u8,
}
impl Model {
    fn new(rows: Vec<Script>, slots: usize) -> Self {
        Self { rows, slots: vec![None; slots], queued: 0, control: Control::default(),
            groups: Vec::new(), retired: Vec::new(), aborted: false, fault: 0 }
    }
}
impl RefillDriver for Model {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn admit_next(&mut self) -> Result<Option<RefillAdmission>, Int8GenerationError> {
        if self.queued == self.rows.len() { return Ok(None); }
        let sequence = self.slots.iter().position(Option::is_none).unwrap(); let request = self.queued;
        self.slots[sequence] = Some(request); self.queued += 1;
        Ok(Some(RefillAdmission { sequence, request: request + usize::from(self.fault == 5) }))
    }
    fn append_packed(&mut self, runs: &[CohortTokenRun<'_>], _: Int8PrefillLimits) -> Result<(), Int8GenerationError> {
        if self.fault == 6 && self.groups.len() == 1 { return Err(StrictInt8Error::Primitive.into()); }
        let mut group = Vec::new();
        for run in runs {
            let request = self.slots[run.sequence].unwrap(); group.push((request, run.tokens.len()));
            self.rows[request].tokens.extend_from_slice(run.tokens);
        }
        self.groups.push(group); Ok(())
    }
    fn logits_group(&mut self, slots: &[usize]) -> Result<Vec<f32>, Int8GenerationError> {
        let mut logits = Vec::new();
        for &slot in slots { logits.extend(self.rows[self.slots[slot].unwrap()].logits()); }
        if self.fault == 1 { *logits.last_mut().unwrap() = f32::NAN; }
        if self.fault == 2 { logits.pop(); } Ok(logits)
    }
    fn retire(&mut self, slot: usize) -> Result<RefillRetirement, Int8GenerationError> {
        let request = self.slots[slot].take().unwrap(); self.retired.push(request);
        let mut work = self.rows[request].work(); if self.fault == 3 { work.attention_pairs += 1; }
        Ok(RefillRetirement { request, work })
    }
    fn completed_work(&self) -> Result<Int8Work, Int8GenerationError> {
        assert!(self.slots.iter().all(Option::is_none)); assert_eq!(self.retired.len(), self.rows.len());
        let mut work = self.rows.iter().try_fold(Int8Work::default(), |sum, row| sum.checked_add(row.work()))?;
        if self.fault == 4 { work.forward_positions += 1; } Ok(work)
    }
    fn abort(&mut self) { self.aborted = true; }
}
struct Scalar { row: Script, control: Control }
impl Driver for Scalar {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn append(&mut self, token: u32) -> Result<(), Int8GenerationError> { self.row.tokens.push(token); Ok(()) }
    fn logits(&mut self) -> Result<Vec<f32>, Int8GenerationError> { Ok(self.row.logits()) }
    fn work(&self) -> Int8Work { self.row.work() }
    fn abort(&mut self) {}
}
#[derive(Default)] struct Sink(Vec<DecodeTokenEvent>);
impl DecodeEventSink for Sink {
    type Permit = (); type Error = &'static str;
    fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
    fn permit(&mut self, _: (), event: DecodeTokenEvent) -> Result<(), Self::Error> { self.0.push(event); Ok(()) }
}
#[test]
fn a_finished_slot_refills_before_the_straggler_finishes_its_prompt() {
    let options = GenerationOptions::greedy(4, 100, 0);
    let plans = [plan(&[5], "a", options.clone()), plan(&[5; 12], "b", options.clone()), plan(&[6], "c", options)];
    let mut model = Model::new(vec![Script::new(&[0]), Script::new(&[2, 0]), Script::new(&[1, 0])], 2);
    let mut sink = Sink::default();
    let result = drive(&[request(&plans[0], 90), request(&plans[1], 3), request(&plans[2], 7)],
        2, limits(4), &mut sink, &mut model).unwrap();
    assert_eq!(model.groups[0], [(0, 1), (1, 3)]);
    assert_eq!(model.groups[1], [(2, 1), (1, 3)]);
    assert_eq!(model.retired, [0, 2, 1]);
    assert_eq!(result.sequences.iter().map(|row| row.sequence.request_seq).collect::<Vec<_>>(), [90, 3, 7]);
    assert_eq!(model.rows[2].tokens, [6, 1]); assert!(!model.aborted);
    assert_eq!(sink.0.iter().filter(|event| event.request_seq == 7).count(), 2);
}
#[test]
fn seeded_results_match_scalar_across_slot_and_token_widths() {
    let mut options = GenerationOptions::greedy(3, 100, 0); options.banned_token_ids = vec![0];
    options.sampling = GenerationSampling::Seeded { effective_seed: [31; 32], temperature_milli: 900,
        top_k: Some(20), top_p_ppm: 950_000 };
    let plans = [plan(&[5], "a", options.clone()), plan(&[7, 8, 9], "b", options.clone()), plan(&[6, 7], "c", options)];
    let mut script = Script::new(&[1]); script.flat = true;
    let requests = [request(&plans[0], 19), request(&plans[1], 2), request(&plans[2], 99)];
    for slots in [1, 2, 3] { for width in [1, 4] {
        let result = drive(&requests, slots, limits(width), &mut Discard,
            &mut Model::new(vec![script.clone(); 3], slots)).unwrap();
        for (index, request) in requests.iter().enumerate() {
            let reference = request.plan.drive(&Decoder, request.request_seq, &mut Discard,
                &mut Scalar { row: script.clone(), control: Control::default() }).unwrap();
            assert!(result.sequences[index] == reference);
        }
    }}
}
#[test]
fn a_byte_refused_proposal_is_charged_then_the_slot_is_reused() {
    let a = plan(&[5], "a", GenerationOptions::greedy(4, 1, 0));
    let b = plan(&[6], "b", GenerationOptions::greedy(1, 100, 0));
    let mut model = Model::new(vec![Script::new(&[1]), Script::new(&[0])], 1);
    let result = drive(&[request(&a, 1), request(&b, 2)], 1, limits(4), &mut Discard, &mut model).unwrap();
    assert_eq!(result.sequences[0].sequence.finish_reason, GenerationFinish::ByteLimit);
    assert_eq!(result.sequences[0].sequence.token_ids, [1]);
    assert_eq!(result.sequences[0].model_work, Int8Work::for_sequence(0, 2, 2 * NANBEIGE_VOCAB_SIZE).unwrap());
    assert_eq!(model.rows[1].tokens, [6]); assert_eq!(model.retired, [0, 1]);
}
#[test]
fn malformed_logits_routing_or_retirement_cannot_publish_an_epoch() {
    let a = plan(&[5], "a", GenerationOptions::greedy(1, 100, 0));
    let b = plan(&[6], "b", GenerationOptions::greedy(1, 100, 0));
    for fault in 1..=6 {
        let mut model = Model::new(vec![Script::new(&[0]); 2], 1); model.fault = fault;
        assert!(drive(&[request(&a, 9), request(&b, 3)], 1, limits(4), &mut Discard, &mut model).is_err());
        assert!(model.aborted);
    }
}
#[test]
fn cancellation_and_uncertain_delivery_abort_without_retry() {
    let a = plan(&[5; 3], "a", GenerationOptions::greedy(3, 100, 0));
    let mut model = Model::new(vec![Script::new(&[1]); 2], 1); model.control.stop = Some(5);
    let error = drive(&[request(&a, 1), request(&a, 2)], 1, limits(1), &mut Discard, &mut model).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::CostBudget)); assert!(model.aborted);
    struct Broken(usize);
    impl DecodeEventSink for Broken {
        type Permit = (); type Error = &'static str;
        fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
        fn permit(&mut self, _: (), _: DecodeTokenEvent) -> Result<(), Self::Error> { self.0 += 1; Err("uncertain") }
    }
    let mut model = Model::new(vec![Script::new(&[1])], 1); let mut sink = Broken(0);
    assert!(drive(&[request(&a, 1)], 1, limits(4), &mut sink, &mut model).is_err());
    assert_eq!(sink.0, 1); assert!(model.aborted);
}
#[test]
fn duplicate_delivery_and_invalid_slots_refuse_before_admission() {
    let a = plan(&[5], "a", GenerationOptions::greedy(1, 100, 0));
    for (slots, ids) in [(0, [1, 2]), (3, [1, 2]), (1, [1, 1]), (1, [0, 2])] {
        let mut model = Model::new(vec![Script::new(&[0]); 2], 2);
        assert!(drive(&[request(&a, ids[0]), request(&a, ids[1])], slots, limits(2), &mut Discard, &mut model).is_err());
        assert_eq!(model.queued, 0); assert!(model.groups.is_empty());
    }
}
#[test]
fn replay_handles_refill_and_degenerate_serial_geometry() {
    assert_eq!(expected_steps(&[1, 12, 1], &[1, 13, 2], 2, 4).unwrap(), 5);
    for width in [1, 2, 4, 64] {
        let expected: u64 = [2_usize, 5, 1].into_iter().map(|p| p.div_ceil(width) as u64 + 2).sum();
        assert_eq!(expected_steps(&[2, 5, 1], &[4, 7, 3], 1, width).unwrap(), expected);
    }
    assert_eq!(expected_steps(&[2, 5, 1], &[4, 7, 3], 2, 1).unwrap(), 14);
    for (prompts, ends, slots, width) in [(vec![0], vec![1], 1, 2), (vec![2], vec![1], 1, 2),
        (vec![1], vec![u64::MAX], 1, 2), (vec![1], vec![1], 2, 2), (vec![1], vec![1], 1, 0)] {
        assert!(expected_steps(&prompts, &ends, slots, width).is_err());
    }
}
#[test]
fn sampler_reserve_depends_on_live_slots_not_entire_epoch_length() {
    assert_eq!(sampler_bound(8192, 2).unwrap(), 16384);
    assert!(sampler_bound(u64::MAX, 2).is_err()); assert!(sampler_bound(1, 0).is_err());
}
