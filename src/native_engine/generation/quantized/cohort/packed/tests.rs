//! Synthetic-logit differential fixtures for the real generation cursor.
//! These do not execute model weights or certify quantized-model parity.
use super::*;
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"packed-cohort-fixture");
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
    fn decode_token_ids(&self, tokens: &[u32]) -> Result<Vec<u8>, Self::Error> {
        Ok(tokens.iter().map(|&id| b'a' + id as u8).collect())
    }
}
#[derive(Default)] struct Control { polls: usize, stop: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1; (self.stop == Some(self.polls)).then_some(DecodeCancellationKind::CostBudget)
    }
    fn prefill_checkpoint(&mut self, index: usize) -> Option<DecodeCancellationKind> { self.checkpoint(index) }
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
    rows: Vec<Script>, control: Control, packs: Vec<Vec<(usize, usize)>>, heads: Vec<Vec<usize>>,
    aborted: bool, fail_pack: Option<usize>, corrupt: bool, bad_shape: bool, bad_work: bool,
}
impl Model {
    fn new(rows: Vec<Script>) -> Self { Self { rows, control: Control::default(), packs: Vec::new(), heads: Vec::new(),
        aborted: false, fail_pack: None, corrupt: false, bad_shape: false, bad_work: false } }
}
impl GroupDriver for Model {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn append_group(&mut self, _: &[CohortToken]) -> Result<(), Int8GenerationError> {
        panic!("packed driver must not silently fall back to the old scheduler")
    }
    fn logits_group(&mut self, sequences: &[usize]) -> Result<Vec<f32>, Int8GenerationError> {
        self.heads.push(sequences.to_vec());
        let mut output = Vec::new(); for &slot in sequences { output.extend(self.rows[slot].logits()); }
        if self.corrupt { *output.last_mut().unwrap() = f32::NAN; }
        if self.bad_shape { output.pop(); } Ok(output)
    }
    fn work(&self, sequence: usize) -> Result<Int8Work, Int8GenerationError> {
        let mut work = self.rows[sequence].work(); if self.bad_work { work.attention_pairs += 1; } Ok(work)
    }
    fn abort(&mut self) { self.aborted = true; }
}
impl PackedDriver for Model {
    fn append_packed(&mut self, runs: &[CohortTokenRun<'_>], limits: Int8PrefillLimits)
        -> Result<(), Int8GenerationError> {
        if self.fail_pack == Some(self.packs.len()) { return Err(StrictInt8Error::Primitive.into()); }
        assert!(runs.iter().map(|run| run.tokens.len()).sum::<usize>() <= limits.max_batch_rows);
        self.packs.push(runs.iter().map(|run| (run.sequence, run.tokens.len())).collect());
        for run in runs { self.rows[run.sequence].tokens.extend_from_slice(run.tokens); } Ok(())
    }
}
struct Scalar { script: Script, control: Control }
impl Driver for Scalar {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn append(&mut self, token: u32) -> Result<(), Int8GenerationError> { self.script.tokens.push(token); Ok(()) }
    fn logits(&mut self) -> Result<Vec<f32>, Int8GenerationError> { Ok(self.script.logits()) }
    fn work(&self) -> Int8Work { self.script.work() }
    fn abort(&mut self) {}
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
fn ragged_prompt_morsels_mix_with_decode_without_a_whole_prompt_barrier() {
    let plans = [plan(&[5], "a", GenerationOptions::greedy(4, 100, 0)),
        plan(&[6, 7, 8, 9, 10, 11, 12], "b", GenerationOptions::greedy(4, 100, 0))];
    let mut model = Model::new(vec![Script::new(&[1, 0]), Script::new(&[2, 0])]); let mut sink = Sink::default();
    let run = drive(&[request(&plans[0], 19), request(&plans[1], 3)], limits(4), &mut sink, &mut model).unwrap();
    assert_eq!(model.packs, [vec![(0, 1), (1, 3)], vec![(0, 1), (1, 3)], vec![(1, 1)], vec![(1, 1)]]);
    assert_eq!(model.heads, [vec![0], vec![0], vec![1], vec![1]]);
    assert_eq!(model.rows[0].tokens, [5, 1]); assert_eq!(model.rows[1].tokens, [6, 7, 8, 9, 10, 11, 12, 2]);
    assert_eq!(run.group_steps, 4); assert_eq!(run.model_work.forward_positions, 10);
    assert_eq!(run.execution, INT8_PACKED_COHORT_EXECUTION); assert!(!model.aborted);
    assert_eq!(sink.0.iter().map(|event| event.request_seq).collect::<Vec<_>>(), [19, 19, 3, 3]);
}
#[test]
fn every_pack_width_preserves_scalar_seeded_tokens_scores_and_complete_work() {
    let mut options = GenerationOptions::greedy(4, 100, 0); options.capture_logprobs = true;
    options.sampling = GenerationSampling::Seeded { effective_seed: [29; 32], temperature_milli: 900,
        top_k: Some(20), top_p_ppm: 950_000 };
    let plans = [plan(&[5, 6, 7, 8, 9], "a", options.clone()), plan(&[7], "b", options)];
    let mut script = Script::new(&[1]); script.flat = true;
    let mut references = Vec::new();
    for (index, plan) in plans.iter().enumerate() {
        references.push(plan.drive(&Decoder, [9, 2][index], &mut Discard,
            &mut Scalar { script: script.clone(), control: Control::default() }).unwrap());
    }
    for width in [1, 2, 3, 4, 8, 64] {
        let run = drive(&[request(&plans[0], 9), request(&plans[1], 2)], limits(width), &mut Discard,
            &mut Model::new(vec![script.clone(), script.clone()])).unwrap();
        assert!(run.sequences == references);
    }
    let permuted = drive(&[request(&plans[1], 77), request(&plans[0], 88)], limits(3), &mut Discard,
        &mut Model::new(vec![script.clone(), script])).unwrap();
    assert_eq!(references[0].sequence.token_ids, permuted.sequences[1].sequence.token_ids);
    assert_eq!(references[1].sequence.token_logprobs, permuted.sequences[0].sequence.token_logprobs);
}
#[test]
fn byte_refused_proposal_is_counted_but_not_delivered_after_bulk_prefill() {
    let p = plan(&[5, 6, 7, 8, 9], "a", GenerationOptions::greedy(4, 1, 0)); let mut sink = Sink::default();
    let run = drive(&[request(&p, 1)], limits(64), &mut sink, &mut Model::new(vec![Script::new(&[1])])).unwrap();
    assert_eq!(run.sequences[0].sequence.finish_reason, GenerationFinish::ByteLimit);
    assert_eq!(run.sequences[0].sequence.token_ids, [1]); assert_eq!(sink.0.len(), 1);
    assert_eq!(run.model_work, Int8Work::for_sequence(0, 6, 2 * NANBEIGE_VOCAB_SIZE).unwrap());
    assert_eq!(run.group_steps, 2);
}
#[test]
fn stop_suffix_and_minimum_length_use_the_existing_cursor() {
    let mut options = GenerationOptions::greedy(5, 100, 0);
    options.min_new_tokens = 2; options.stop_suffixes = vec![b"bc".to_vec()];
    let p = plan(&[5, 6, 7], "a", options);
    let run = drive(&[request(&p, 1)], limits(8), &mut Discard, &mut Model::new(vec![Script::new(&[1, 2, 3])])).unwrap();
    assert_eq!(run.sequences[0].sequence.finish_reason, GenerationFinish::StopSuffix);
    assert_eq!(run.sequences[0].sequence.token_ids, [1, 2]); assert_eq!(run.group_steps, 2);
}
#[test]
fn malformed_head_or_work_aborts_the_entire_pack() {
    let p = plan(&[5, 6], "a", GenerationOptions::greedy(1, 100, 0));
    for axis in 0..3 {
        let mut model = Model::new(vec![Script::new(&[1]), Script::new(&[2])]); let mut sink = Sink::default();
        match axis { 0 => model.corrupt = true, 1 => model.bad_shape = true, _ => model.bad_work = true }
        assert!(drive(&[request(&p, 1), request(&p, 2)], limits(8), &mut sink, &mut model).is_err());
        assert!(model.aborted);
        if axis < 2 { assert!(sink.0.is_empty()); }
    }
}
#[test]
fn cancellation_and_native_failure_do_not_retry_a_pack() {
    let p = plan(&[5, 6, 7], "a", GenerationOptions::greedy(2, 100, 0));
    let mut model = Model::new(vec![Script::new(&[1])]); model.control.stop = Some(1);
    let error = drive(&[request(&p, 1)], limits(2), &mut Discard, &mut model).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::CostBudget));
    assert!(model.packs.is_empty()); assert!(model.aborted);
    let mut model = Model::new(vec![Script::new(&[1])]); model.fail_pack = Some(1);
    assert!(drive(&[request(&p, 1)], limits(2), &mut Discard, &mut model).is_err());
    assert_eq!(model.packs.len(), 1); assert!(model.aborted);
}
#[test]
fn invalid_workspace_and_duplicate_delivery_refuse_before_native_work() {
    let p = plan(&[5], "a", GenerationOptions::greedy(1, 100, 0));
    let mut short = limits(4); short.max_extra_scratch_bytes -= 1;
    let mut model = Model::new(vec![Script::new(&[1])]);
    assert!(drive(&[request(&p, 1)], short, &mut Discard, &mut model).is_err()); assert!(model.packs.is_empty());
    let mut model = Model::new(vec![Script::new(&[1]), Script::new(&[1])]);
    assert!(drive(&[request(&p, 1), request(&p, 1)], limits(4), &mut Discard, &mut model).is_err());
    assert!(model.packs.is_empty());
}
#[test]
fn uncertain_delivery_is_not_retried_after_a_valid_packed_forward() {
    struct Broken(usize);
    impl DecodeEventSink for Broken {
        type Permit = (); type Error = &'static str;
        fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
        fn permit(&mut self, _: (), _: DecodeTokenEvent) -> Result<(), Self::Error> { self.0 += 1; Err("uncertain") }
    }
    let p = plan(&[5, 6], "a", GenerationOptions::greedy(2, 100, 0));
    let mut model = Model::new(vec![Script::new(&[1])]); let mut sink = Broken(0);
    assert!(drive(&[request(&p, 1)], limits(4), &mut sink, &mut model).is_err());
    assert_eq!(sink.0, 1); assert_eq!(model.packs.len(), 1); assert!(model.aborted);
}
