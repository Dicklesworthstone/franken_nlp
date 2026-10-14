//! Compiler and real cursor wiring with synthetic logits, NOT model parity.
use super::*;

fn identity(width: usize) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"synthetic-int4-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: recipe(width), packing_set_digest: packing_digest(d, width).unwrap(),
        tokenizer_digest: d, template_digest: d, task_spec: "generate-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn plan(options: GenerationOptions) -> Int4GenerationPlan {
    Int4GenerationPlan::compile(vec![5, 6, 7], options, identity(127), 127, "stable-item", 0, GenerationLimits::default()).unwrap()
}
#[derive(Default)]
struct Control { stop_prompt: Option<usize>, stop_token: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, index: usize) -> Option<DecodeCancellationKind> {
        (self.stop_token == Some(index)).then_some(DecodeCancellationKind::Deadline)
    }
    fn prefill_checkpoint(&mut self, index: usize) -> Option<DecodeCancellationKind> {
        (self.stop_prompt == Some(index)).then_some(DecodeCancellationKind::CostBudget)
    }
}
struct Model {
    control: Control, tokens: Vec<u32>, preferred: Vec<u32>, heads: usize,
    aborted: bool, flat: bool, corrupt: bool, forged: bool, fail_append: Option<usize>,
}
impl Model {
    fn new(preferred: &[u32]) -> Self {
        Self { control: Control::default(), tokens: Vec::new(), preferred: preferred.to_vec(), heads: 0,
            aborted: false, flat: false, corrupt: false, forged: false, fail_append: None }
    }
}
impl Driver for Model {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn append(&mut self, token: u32) -> Result<(), Int4GenerationError> {
        if self.fail_append == Some(self.tokens.len()) { return Err(StrictInt4Error::Primitive.into()); }
        self.tokens.push(token); Ok(())
    }
    fn logits(&mut self) -> Result<Vec<f32>, Int4GenerationError> {
        let selected = self.preferred[self.heads.min(self.preferred.len() - 1)]; self.heads += 1;
        let mut logits = vec![-100.0; NANBEIGE_VOCAB_SIZE];
        if self.flat { logits[1..21].fill(0.0); } else { logits[selected as usize] = 10.0; }
        if self.corrupt { logits[0] = f32::NAN; }
        Ok(logits)
    }
    fn work(&self) -> Int4Work {
        let mut work = Int4Work::for_sequence(127, 0, self.tokens.len(), self.heads * NANBEIGE_VOCAB_SIZE).unwrap();
        if self.forged { work.projections.dot_products += 1; } work
    }
    fn abort(&mut self) { self.aborted = true; }
}
struct Decoder;
impl DecodeByteDecoder for Decoder {
    type Error = &'static str;
    fn decode_token_ids(&self, ids: &[u32]) -> Result<Vec<u8>, Self::Error> {
        Ok(ids.iter().map(|&id| b'a' + id as u8).collect())
    }
}
#[derive(Default)] struct Sink(Vec<DecodeTokenEvent>);
impl DecodeEventSink for Sink {
    type Permit = (); type Error = &'static str;
    fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
    fn permit(&mut self, _: (), event: DecodeTokenEvent) -> Result<(), Self::Error> { self.0.push(event); Ok(()) }
}

#[test]
fn int4_profile_roundtrips_centrally_and_never_enters_int8_or_eager() {
    assert_eq!(NumericsProfile::from_label(STRICT_INT4_PROFILE), Some(NumericsProfile::StrictQuantized { version: 2 }));
    let options = GenerationOptions::greedy(2, 100, 0);
    assert!(GenerationPlan::compile(vec![1], options.clone(), identity(127), "id", 0, GenerationLimits::default()).is_err());
    assert!(quantized::Int8GenerationPlan::compile(vec![1], options.clone(), identity(127), "id", 0, GenerationLimits::default()).is_err());
    for profile in [NumericsProfile::HfBf16Eager, NumericsProfile::DiagnosticF32,
        NumericsProfile::StrictQuantized { version: 1 }, NumericsProfile::StrictQuantized { version: 3 }] {
        let mut id = identity(127); id.numerics_profile = profile;
        assert!(Int4GenerationPlan::compile(vec![1], options.clone(), id, 127, "id", 0, GenerationLimits::default()).is_err());
    }
}
#[test]
fn compilation_refuses_wrong_partition_namespace_revision_backend_and_packing() {
    for axis in 0..6 {
        let mut id = identity(127);
        match axis { 0 => id.artifact_format = "fnlpq-v1".to_owned(), 1 => id.quant_recipe = recipe(128),
            2 => id.source_revision.push('x'), 3 => id.backend_semantic_version.push('x'),
            4 => id.packing_set_digest = Sha256Digest::of_bytes(b"different-layout"), _ => id.kv_dtype = "f32".to_owned() }
        assert!(Int4GenerationPlan::compile(vec![1], GenerationOptions::greedy(2, 100, 0), id,
            127, "id", 0, GenerationLimits::default()).is_err());
    }
    for width in [0, MAX_MODEL_K + 1, usize::MAX] {
        assert!(Int4GenerationPlan::compile(vec![1], GenerationOptions::greedy(2, 100, 0), identity(127),
            width, "id", 0, GenerationLimits::default()).is_err());
    }
}
#[test]
fn actual_model_content_and_partition_are_checked_separately_from_admission() {
    let id = identity(127); let actual = id.logical_model_digest;
    check_model_facts(&id, 127, 127, STRICT_INT4_PROFILE, actual).unwrap();
    assert!(check_model_facts(&id, 127, 128, STRICT_INT4_PROFILE, actual).is_err());
    assert!(check_model_facts(&id, 127, 127, "strict-quantized-v1", actual).is_err());
    assert!(check_model_facts(&id, 127, 127, STRICT_INT4_PROFILE, Sha256Digest::of_bytes(b"other-weights")).is_err());
    let p = plan(GenerationOptions::greedy(2, 100, 0));
    p.verify_identity(p.execution_identity()).unwrap();
    for axis in 0..4 {
        let mut id = p.execution_identity().clone(); let d = Sha256Digest::of_bytes(b"substitution");
        match axis { 0 => id.tokenizer_digest = d, 1 => id.template_digest = d,
            2 => id.decision_policy_digest = d, _ => id.logical_model_digest = d }
        assert!(p.verify_identity(&id).is_err());
    }
}
#[test]
fn bounds_price_groups_and_final_prompt_head_not_every_prompt_position() {
    let p = plan(GenerationOptions::greedy(4, 100, 0));
    assert_eq!(p.planned_work(), Int4Work::for_sequence(127, 0, 6, 4 * NANBEIGE_VOCAB_SIZE).unwrap());
    assert_eq!(p.prompt_tokens(), 3); assert_eq!(p.group_width(), 127);
    let budget = Int4GenerationBudget { native: Int4RunBudget::exact(p.work),
        max_kv_bytes: 8 * KV_BYTES_PER_TOKEN as u64, max_sampler_bytes: p.sampler_bytes() };
    check_bounds(p.work, p.sampler_bytes(), 8, budget).unwrap();
    assert!(check_bounds(p.work, p.sampler_bytes(), 5, budget).is_err());
    assert!(check_bounds(p.work, p.sampler_bytes(), 9, budget).is_err());
    for axis in 0..6 {
        let mut b = budget;
        match axis { 0 => b.max_kv_bytes -= 1, 1 => b.max_sampler_bytes -= 1,
            2 => b.native.max_forward_positions -= 1, 3 => b.native.max_attention_pairs -= 1,
            4 => b.native.max_projection_work.dot_products -= 1, _ => b.native.max_projection_work.multiply_accumulates -= 1 }
        assert!(check_bounds(p.work, p.sampler_bytes(), 8, b).is_err());
    }
    assert!(Int4GenerationPlan::compile(vec![1; DEFAULT_ADMITTED_CONTEXT_CAP], GenerationOptions::greedy(2, 100, 0),
        identity(127), 127, "id", 0, GenerationLimits::default()).is_err());
}
#[test]
fn model_partition_item_sample_and_prompt_bind_the_sampling_key() {
    let p = plan(GenerationOptions::greedy(2, 100, 0));
    for axis in 0..5 {
        let width = if axis == 0 { 128 } else { 127 };
        let mut id = identity(width);
        if axis == 1 { id.logical_model_digest = Sha256Digest::of_bytes(b"other-model");
            id.packing_set_digest = packing_digest(id.logical_model_digest, width).unwrap(); }
        let other = Int4GenerationPlan::compile(if axis == 2 { vec![8, 6, 7] } else { vec![5, 6, 7] },
            p.options().clone(), id, width, if axis == 3 { "other-item" } else { "stable-item" },
            if axis == 4 { 1 } else { 0 }, GenerationLimits::default()).unwrap();
        assert!(p.plan.key != other.plan.key);
    }
}
#[test]
fn eos_is_scored_without_bytes_or_feedback_and_stream_equals_content() {
    let mut options = GenerationOptions::greedy(4, 100, 0); options.capture_logprobs = true;
    let p = plan(options); let mut model = Model::new(&[1, 0]); let mut sink = Sink::default();
    let run = p.drive(&Decoder, 17, &mut sink, &mut model).unwrap();
    assert_eq!(model.tokens, [5, 6, 7, 1]); assert_eq!(model.heads, 2); assert!(!model.aborted);
    assert_eq!(run.sequence.token_ids, [1, 0]); assert_eq!(run.sequence.content_bytes, b"b");
    assert_eq!(run.sequence.finish_reason, GenerationFinish::Eos);
    assert_eq!(run.sequence.numerics_profile, STRICT_INT4_PROFILE);
    assert_eq!(run.sequence.execution, INT4_GENERATION_VERSION);
    assert_eq!(run.sequence.token_logprobs.as_ref().unwrap().len(), 2);
    assert_eq!(run.model_work, Int4Work::for_sequence(127, 0, 4, 2 * NANBEIGE_VOCAB_SIZE).unwrap());
    assert_eq!(sink.0.len(), 2); assert!(sink.0[1].decoded_bytes.is_empty());
    assert_eq!(sink.0.iter().flat_map(|event| event.decoded_bytes.iter().copied()).collect::<Vec<_>>(), run.sequence.content_bytes);
    let bytes = serde_json::to_vec(&run).unwrap();
    let restored: Int4GenerationRun = serde_json::from_slice(&bytes).unwrap();
    assert!(restored == run);
}
#[test]
fn byte_budget_and_stop_suffix_keep_actual_work_without_extra_feedback() {
    let p = plan(GenerationOptions::greedy(4, 1, 0)); let mut model = Model::new(&[1]); let mut sink = Sink::default();
    let run = p.drive(&Decoder, 1, &mut sink, &mut model).unwrap();
    assert_eq!(run.sequence.finish_reason, GenerationFinish::ByteLimit);
    assert_eq!(run.sequence.token_ids, [1]); assert_eq!(model.heads, 2); assert_eq!(sink.0.len(), 1);
    let mut options = GenerationOptions::greedy(8, 100, 0); options.stop_suffixes = vec![b"bc".to_vec()];
    let p = plan(options); let mut model = Model::new(&[1, 2]);
    let run = p.drive(&Decoder, 1, &mut Discard, &mut model).unwrap();
    assert_eq!(run.sequence.finish_reason, GenerationFinish::StopSuffix);
    assert_eq!(run.sequence.content_bytes, b"bc"); assert_eq!(model.tokens, [5, 6, 7, 1]);
}
#[test]
fn seeded_selection_ignores_delivery_sequence_and_intervening_runs() {
    let mut options = GenerationOptions::greedy(6, 100, 0); options.banned_token_ids = vec![0];
    options.sampling = GenerationSampling::Seeded { effective_seed: [19; 32], temperature_milli: 800,
        top_k: Some(20), top_p_ppm: 950000 };
    let p = plan(options);
    let execute = |seq| { let mut model = Model::new(&[1]); model.flat = true;
        p.drive(&Decoder, seq, &mut Discard, &mut model).unwrap() };
    let a = execute(1); let _other = execute(99); let b = execute(7);
    assert_eq!(a.sequence.token_ids, b.sequence.token_ids); assert_eq!(a.model_work, b.model_work);
    assert_eq!(a.sequence.native_work.sampled_steps, 6);
}
#[test]
fn cancellation_and_native_failure_abort_without_successful_partial_result() {
    let p = plan(GenerationOptions::greedy(4, 100, 0));
    let mut model = Model::new(&[1]); model.control.stop_prompt = Some(1);
    let error = p.drive(&Decoder, 1, &mut Discard, &mut model).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::CostBudget));
    assert_eq!(model.tokens, [5]); assert_eq!(model.heads, 0); assert!(model.aborted);
    let mut model = Model::new(&[1]); model.fail_append = Some(3); let mut sink = Sink::default();
    assert!(matches!(p.drive(&Decoder, 1, &mut sink, &mut model), Err(Int4GenerationError::Native(StrictInt4Error::Primitive))));
    assert!(model.aborted); assert_eq!(sink.0.len(), 1); assert_eq!(model.heads, 1);
}
#[test]
fn invalid_logits_and_forged_native_work_cannot_return_success() {
    let p = plan(GenerationOptions::greedy(1, 100, 0));
    let mut model = Model::new(&[1]); model.corrupt = true;
    assert!(p.drive(&Decoder, 1, &mut Discard, &mut model).is_err()); assert!(model.aborted);
    let mut model = Model::new(&[1]); model.forged = true;
    assert!(matches!(p.drive(&Decoder, 1, &mut Discard, &mut model), Err(Int4GenerationError::WorkMismatch)));
    assert!(model.aborted);
}
#[test]
fn uncertain_sink_delivery_and_decoder_failure_are_not_retried() {
    struct BadSink(usize);
    impl DecodeEventSink for BadSink {
        type Permit = (); type Error = ();
        fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), ()> { Ok(()) }
        fn permit(&mut self, _: (), _: DecodeTokenEvent) -> Result<(), ()> { self.0 += 1; Err(()) }
    }
    struct BadDecoder;
    impl DecodeByteDecoder for BadDecoder {
        type Error = ();
        fn decode_token_ids(&self, _: &[u32]) -> Result<Vec<u8>, ()> { Err(()) }
    }
    let p = plan(GenerationOptions::greedy(4, 100, 0)); let mut model = Model::new(&[1]); let mut sink = BadSink(0);
    assert!(p.drive(&Decoder, 1, &mut sink, &mut model).is_err());
    assert_eq!(sink.0, 1); assert!(model.aborted); assert_eq!(model.heads, 1);
    let mut model = Model::new(&[1]);
    assert!(p.drive(&BadDecoder, 1, &mut Discard, &mut model).is_err());
    assert!(model.aborted); assert_eq!(model.heads, 1);
}
