//! Synthetic-logit cursor fixtures, NOT real-model numerical qualification.
use super::*;
fn plan(options: GenerationOptions, width: usize) -> Int4GenerationPlan {
    let d = Sha256Digest::of_bytes(b"synthetic-int4-prefill-fixture");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
        logical_model_digest: d, artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: recipe(width),
        packing_set_digest: packing_digest(d, width).unwrap(), tokenizer_digest: d, template_digest: d,
        task_spec: "generate-v1".to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    Int4GenerationPlan::compile(vec![5, 6, 7, 8, 9], options, identity, width,
        "stable-item", 3, GenerationLimits::default()).unwrap()
}
fn limits(rows: usize) -> Int4PrefillLimits {
    Int4PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
#[derive(Default)] struct Control { polls: usize, cancel: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1; (self.cancel == Some(self.polls)).then_some(DecodeCancellationKind::Deadline)
    }
}
struct Model { control: Control, width: usize, tokens: Vec<u32>, preferred: Vec<u32>, heads: usize,
    prompt_calls: usize, flat: bool, fail_prompt: bool, forged: bool, corrupt: bool, aborted: bool }
impl Model {
    fn new(preferred: &[u32], width: usize) -> Self {
        Self { control: Control::default(), width, tokens: Vec::new(), preferred: preferred.to_vec(),
            heads: 0, prompt_calls: 0, flat: false, fail_prompt: false, forged: false, corrupt: false, aborted: false }
    }
}
impl Driver for Model {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn append(&mut self, token: u32) -> Result<(), Int4GenerationError> { self.tokens.push(token); Ok(()) }
    fn logits(&mut self) -> Result<Vec<f32>, Int4GenerationError> {
        let mut output = vec![-100.0; NANBEIGE_VOCAB_SIZE];
        let selected = self.preferred[self.heads.min(self.preferred.len() - 1)] as usize; self.heads += 1;
        if self.flat { output[1..5].fill(0.0); } else { output[selected] = 10.0; }
        if self.corrupt { output[0] = f32::NAN; }
        Ok(output)
    }
    fn work(&self) -> Int4Work {
        let mut work = Int4Work::for_sequence(self.width, 0, self.tokens.len(), self.heads * NANBEIGE_VOCAB_SIZE).unwrap();
        if self.forged { work.projections.dot_products += 1; } work
    }
    fn abort(&mut self) { self.aborted = true; }
}
impl PrefillDriver for Model {
    fn append_prompt(&mut self, tokens: &[u32], limits: Int4PrefillLimits) -> Result<(), Int4GenerationError> {
        self.prompt_calls += 1;
        if self.fail_prompt { return Err(StrictInt4Error::Primitive.into()); }
        for chunk in tokens.chunks(limits.max_batch_rows) { for &token in chunk {
            if let Some(cause) = self.control.checkpoint(self.tokens.len()) { return Err(StrictInt4Error::Cancelled(cause).into()); }
            self.append(token)?;
        } }
        Ok(())
    }
}
struct Decoder;
impl DecodeByteDecoder for Decoder {
    type Error = &'static str;
    fn decode_token_ids(&self, tokens: &[u32]) -> Result<Vec<u8>, Self::Error> {
        Ok(tokens.iter().map(|&id| b'a' + id as u8).collect())
    }
}
#[derive(Default)] struct Sink { events: Vec<DecodeTokenEvent>, fail: bool }
impl DecodeEventSink for Sink {
    type Permit = (); type Error = &'static str;
    fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
    fn permit(&mut self, _: (), event: DecodeTokenEvent) -> Result<(), Self::Error> {
        if self.fail { return Err("fixture delivery failure"); } self.events.push(event); Ok(())
    }
}
#[test]
fn prompt_scheduling_preserves_eos_stops_byte_limits_logprobs_and_group_work() {
    for group in [3, 128] { for case in 0..4 {
        let mut options = GenerationOptions::greedy(4, if case == 2 { 1 } else { 100 }, 0);
        options.capture_logprobs = true;
        if case == 1 { options.stop_suffixes = vec![b"bc".to_vec()]; }
        let preferred = if case == 0 { vec![1, 0] } else { vec![1, 2, 3, 4] };
        let p = plan(options, group); let mut serial = Model::new(&preferred, group);
        let mut expected_sink = Sink::default();
        let baseline = p.drive(&Decoder, 7, &mut expected_sink, &mut serial).unwrap();
        for rows in [1, 3, 4, 64] {
            let mut grouped = Model::new(&preferred, group); let mut sink = Sink::default();
            let actual = p.drive_layer_major(&Decoder, 7, &mut sink, &mut grouped, limits(rows)).unwrap();
            assert_eq!(canonjson::canonical_bytes(&baseline).unwrap(), canonjson::canonical_bytes(&actual).unwrap());
            assert_eq!(expected_sink.events.len(), sink.events.len());
            assert_eq!(expected_sink.events.iter().map(|e| &e.decoded_bytes).collect::<Vec<_>>(),
                sink.events.iter().map(|e| &e.decoded_bytes).collect::<Vec<_>>());
            assert_eq!(grouped.tokens, serial.tokens); assert_eq!(grouped.heads, serial.heads);
            assert_eq!(grouped.prompt_calls, 1); assert!(!grouped.aborted);
        }
    } }
}
#[test]
fn prefill_does_not_consume_seeded_draws_or_change_the_policy() {
    let mut options = GenerationOptions::greedy(5, 100, 0);
    options.sampling = GenerationSampling::Seeded { effective_seed: [17; 32], temperature_milli: 800,
        top_k: Some(4), top_p_ppm: 950_000 };
    options.repetition_penalty_milli = 1200; options.presence_penalty_milli = 250;
    options.frequency_penalty_milli = -125; options.logit_bias_milli.insert(3, 400);
    let p = plan(options, 127); let mut serial = Model::new(&[1], 127); serial.flat = true;
    let expected = p.drive(&Decoder, 9, &mut Sink::default(), &mut serial).unwrap();
    for rows in [1, 4, 64] {
        let mut model = Model::new(&[1], 127); model.flat = true;
        let actual = p.drive_layer_major(&Decoder, 9, &mut Sink::default(), &mut model, limits(rows)).unwrap();
        assert_eq!(canonjson::canonical_bytes(&expected).unwrap(), canonjson::canonical_bytes(&actual).unwrap());
        assert_eq!(actual.sequence.native_work.sampled_steps, 5);
    }
}
#[test]
fn prompt_failure_or_cancellation_aborts_before_a_head_or_event() {
    let p = plan(GenerationOptions::greedy(4, 100, 0), 128);
    for cancelled in [false, true] {
        let mut model = Model::new(&[1], 128); model.fail_prompt = !cancelled;
        if cancelled { model.control.cancel = Some(2); }
        let mut sink = Sink::default();
        let error = p.drive_layer_major(&Decoder, 1, &mut sink, &mut model, limits(4)).err().unwrap();
        if cancelled { assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); }
        assert!(model.aborted); assert!(sink.events.is_empty()); assert_eq!(model.heads, 0);
    }
}
#[test]
fn delivery_nonfinite_logits_and_forged_group_work_fail_without_retry() {
    let p = plan(GenerationOptions::greedy(1, 100, 0), 3);
    for axis in 0..3 {
        let mut model = Model::new(&[1], 3); model.forged = axis == 1; model.corrupt = axis == 2;
        let mut sink = Sink { events: Vec::new(), fail: axis == 0 };
        assert!(p.drive_layer_major(&Decoder, 1, &mut sink, &mut model, limits(4)).is_err());
        assert!(model.aborted); assert_eq!(model.heads, 1); assert_eq!(model.prompt_calls, 1);
    }
}
#[test]
fn insufficient_workspace_refuses_without_driver_entry_or_abort() {
    let p = plan(GenerationOptions::greedy(1, 100, 0), 128); let mut model = Model::new(&[1], 128);
    let mut limit = limits(4); limit.max_extra_scratch_bytes -= 1;
    assert!(p.drive_layer_major(&Decoder, 1, &mut Sink::default(), &mut model, limit).is_err());
    assert_eq!(model.prompt_calls, 0); assert_eq!(model.heads, 0); assert!(!model.aborted);
}
