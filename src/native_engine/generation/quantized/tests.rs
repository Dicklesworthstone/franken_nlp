//! Real compiler/cursor/driver wiring with synthetic logits; no model parity.
use super::*;

fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"synthetic-int8-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture-revision".to_owned(), logical_model_digest: d,
        artifact_format: "synthetic-only".to_owned(), quant_recipe: "fixture-int8".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "generate-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn plan(options: GenerationOptions) -> Int8GenerationPlan {
    Int8GenerationPlan::compile(vec![5, 6, 7], options, identity(), "stable-item", 0, GenerationLimits::default()).unwrap()
}
fn source(id: &ExecutionIdentity) -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: id.source_revision.clone(),
        recipe_id: id.quant_recipe.clone(), source_root_sha256: Sha256Digest::of_bytes(b"source").to_hex(),
        logical_model_sha256: id.logical_model_digest.to_hex() }
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
    aborted: bool, flat: bool, corrupt: bool, forged_work: bool, fail_append: Option<usize>,
}
impl Model {
    fn new(preferred: &[u32]) -> Self {
        assert!(!preferred.is_empty());
        Self { control: Control::default(), tokens: Vec::new(), preferred: preferred.to_vec(), heads: 0,
            aborted: false, flat: false, corrupt: false, forged_work: false, fail_append: None }
    }
}
impl Driver for Model {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn append(&mut self, token: u32) -> Result<(), Int8GenerationError> {
        if self.fail_append == Some(self.tokens.len()) { return Err(StrictInt8Error::Primitive.into()); }
        self.tokens.push(token); Ok(())
    }
    fn logits(&mut self) -> Result<Vec<f32>, Int8GenerationError> {
        let token = self.preferred[self.heads.min(self.preferred.len() - 1)]; self.heads += 1;
        let mut row = vec![-100.0; NANBEIGE_VOCAB_SIZE];
        if self.flat { row[1..21].fill(0.0); } else { row[token as usize] = 10.0; }
        if self.corrupt { row[0] = f32::NAN; }
        Ok(row)
    }
    fn work(&self) -> Int8Work {
        let mut work = Int8Work::for_sequence(0, self.tokens.len(), self.heads * NANBEIGE_VOCAB_SIZE).unwrap();
        if self.forged_work { work.attention_pairs += 1; } work
    }
    fn abort(&mut self) { self.aborted = true; }
}
struct Decoder;
impl DecodeByteDecoder for Decoder {
    type Error = &'static str;
    fn decode_token_ids(&self, tokens: &[u32]) -> Result<Vec<u8>, Self::Error> {
        Ok(tokens.iter().map(|&id| b'a' + id as u8).collect())
    }
}
#[derive(Default)] struct Sink(Vec<DecodeTokenEvent>);
impl DecodeEventSink for Sink {
    type Permit = (); type Error = &'static str;
    fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
    fn permit(&mut self, _: (), event: DecodeTokenEvent) -> Result<(), Self::Error> { self.0.push(event); Ok(()) }
}

#[test]
fn profile_and_backend_are_explicit_and_never_repaired() {
    let options = GenerationOptions::greedy(2, 100, 0);
    assert!(GenerationPlan::compile(vec![1], options.clone(), identity(), "id", 0, GenerationLimits::default()).is_err());
    for profile in [NumericsProfile::HfBf16Eager, NumericsProfile::DiagnosticF32, NumericsProfile::StrictQuantized { version: 2 }] {
        let mut id = identity(); id.numerics_profile = profile;
        assert!(Int8GenerationPlan::compile(vec![1], options.clone(), id, "id", 0, GenerationLimits::default()).is_err());
    }
    let mut id = identity(); id.backend_semantic_version = "different-cast-program".to_owned();
    assert!(Int8GenerationPlan::compile(vec![1], options, id, "id", 0, GenerationLimits::default()).is_err());
}
#[test]
fn exact_native_bounds_skip_intermediate_prompt_heads() {
    let p = plan(GenerationOptions::greedy(4, 100, 0));
    assert_eq!(p.planned_work(), Int8Work::for_sequence(0, 6, 4 * NANBEIGE_VOCAB_SIZE).unwrap());
    assert_eq!(p.plan.planned_work().projected_logits, 4 * NANBEIGE_VOCAB_SIZE as u64);
    assert_eq!(p.prompt_tokens(), 3);
}
#[test]
fn source_identity_substitution_and_malformed_digest_refuse() {
    let id = identity(); let original = source(&id); check_model(&id, &original).unwrap();
    for field in 0..5 {
        let mut changed = original.clone();
        match field { 0 => changed.model_id = "other-model".to_owned(), 1 => changed.revision.push('x'),
            2 => changed.recipe_id.push('x'), 3 => changed.logical_model_sha256 = "not-a-digest".to_owned(),
            _ => changed.logical_model_sha256 = Sha256Digest::of_bytes(b"other").to_hex() }
        assert!(matches!(check_model(&id, &changed), Err(Int8GenerationError::ModelIdentity)));
    }
}
#[test]
fn admitted_identity_must_match_all_fields_not_only_the_profile() {
    let p = plan(GenerationOptions::greedy(1, 100, 0));
    p.verify_identity(p.execution_identity()).unwrap();
    for field in 0..4 {
        let mut changed = p.execution_identity().clone(); let d = Sha256Digest::of_bytes(b"other");
        match field { 0 => changed.template_digest = d, 1 => changed.tokenizer_digest = d,
            2 => changed.packing_set_digest = d, _ => changed.decision_policy_digest = d }
        assert!(p.verify_identity(&changed).is_err());
    }
}
#[test]
fn eager_compilation_keeps_legacy_identity_and_sampling_key_bytes() {
    let mut id = identity(); id.numerics_profile = NumericsProfile::HfBf16Eager;
    let options = GenerationOptions::greedy(2, 100, 0);
    let p = GenerationPlan::compile(vec![5, 6, 7], options, id, "stable-item", 4, GenerationLimits::default()).unwrap();
    let expected_policy = digest(&(GENERATION_VERSION, PROCESSOR_VERSION, &p.options, "stable-item", 4_u64)).unwrap();
    assert_eq!(p.identity.decision_policy_digest, expected_policy);
    let bytes = canonjson::canonical_bytes(&(GENERATION_VERSION, PROCESSOR_VERSION, &p.identity, &p.prompt, &p.options, "stable-item")).unwrap();
    let expected = StableRequestKey::from_canonical_digest(Sha256::digest(&bytes).into());
    assert!(p.key == expected);
    assert_eq!(p.bound.projected_logits, 4 * NANBEIGE_VOCAB_SIZE as u64);
}
#[test]
fn item_sample_prompt_and_quantized_strategy_bind_the_private_key() {
    let p = plan(GenerationOptions::greedy(2, 100, 0));
    for field in 0..3 {
        let mut prompt = vec![5, 6, 7]; if field == 0 { prompt[0] = 8; }
        let other = Int8GenerationPlan::compile(prompt, p.options().clone(), identity(),
            if field == 1 { "different-item" } else { "stable-item" }, if field == 2 { 1 } else { 0 }, GenerationLimits::default()).unwrap();
        assert!(p.plan.key != other.plan.key);
    }
    let expected = digest(&(INT8_GENERATION_VERSION, PROCESSOR_VERSION, p.options(), "stable-item", 0_u64)).unwrap();
    assert_eq!(p.execution_identity().decision_policy_digest, expected);
}
#[test]
fn eos_is_scored_without_content_or_feedback_and_only_final_prefill_projects() {
    let mut options = GenerationOptions::greedy(4, 100, 0); options.min_new_tokens = 1; options.capture_logprobs = true;
    let p = plan(options); let mut model = Model::new(&[1, 0]); let mut sink = Sink::default();
    let run = p.drive(&Decoder, 17, &mut sink, &mut model).unwrap();
    assert_eq!(model.tokens, [5, 6, 7, 1]); assert_eq!(model.heads, 2); assert!(!model.aborted);
    assert_eq!(run.sequence.token_ids, [1, 0]); assert_eq!(run.sequence.content_bytes, b"b");
    assert_eq!(run.sequence.finish_reason, GenerationFinish::Eos);
    assert_eq!(run.sequence.numerics_profile, STRICT_INT8_PROFILE);
    assert_eq!(run.sequence.execution, INT8_GENERATION_VERSION);
    assert_eq!(run.sequence.token_logprobs.as_ref().unwrap().len(), 2);
    assert_eq!(run.model_work, Int8Work::for_sequence(0, 4, 2 * NANBEIGE_VOCAB_SIZE).unwrap());
    assert_eq!(sink.0.len(), 2); assert_eq!(sink.0[1].decoded_bytes, b"");
    assert_eq!(sink.0.iter().flat_map(|e| e.decoded_bytes.iter().copied()).collect::<Vec<_>>(), run.sequence.content_bytes);
}
#[test]
fn byte_refused_proposal_is_charged_but_never_committed_or_delivered() {
    let p = plan(GenerationOptions::greedy(4, 1, 0)); let mut model = Model::new(&[1]); let mut sink = Sink::default();
    let run = p.drive(&Decoder, 1, &mut sink, &mut model).unwrap();
    assert_eq!(run.sequence.finish_reason, GenerationFinish::ByteLimit);
    assert_eq!(run.sequence.token_ids, [1]); assert_eq!(sink.0.len(), 1);
    assert_eq!(model.tokens, [5, 6, 7, 1]); assert_eq!(model.heads, 2);
    assert_eq!(run.model_work.projected_logits, 2 * NANBEIGE_VOCAB_SIZE as u64);
}
#[test]
fn cross_token_stop_suffix_preserves_all_streamed_bytes() {
    let mut options = GenerationOptions::greedy(8, 100, 0); options.stop_suffixes = vec![b"bc".to_vec()];
    let p = plan(options); let mut model = Model::new(&[1, 2]); let mut sink = Sink::default();
    let run = p.drive(&Decoder, 1, &mut sink, &mut model).unwrap();
    assert_eq!(run.sequence.finish_reason, GenerationFinish::StopSuffix);
    assert_eq!(run.sequence.content_bytes, b"bc"); assert_eq!(model.tokens, [5, 6, 7, 1]); assert_eq!(sink.0.len(), 2);
}
#[test]
fn seeded_selection_ignores_delivery_sequence_and_other_runs() {
    let mut options = GenerationOptions::greedy(6, 100, 0); options.banned_token_ids = vec![0];
    options.sampling = GenerationSampling::Seeded { effective_seed: [19; 32], temperature_milli: 800, top_k: Some(20), top_p_ppm: 950000 };
    let p = plan(options);
    let execute = |seq| { let mut model = Model::new(&[1]); model.flat = true;
        p.drive(&Decoder, seq, &mut Discard, &mut model).unwrap() };
    let a = execute(1); let _unrelated = execute(999); let b = execute(7);
    assert_eq!(a.sequence.token_ids, b.sequence.token_ids);
    assert_eq!(a.sequence.native_work.sampled_steps, 6);
    assert_eq!(a.sequence.effective_seed.as_deref(), Some("13".repeat(32).as_str()));
    assert_eq!(a.model_work, b.model_work);
}
#[test]
fn every_work_memory_and_context_axis_is_enforced() {
    let p = plan(GenerationOptions::greedy(4, 100, 0)); let work = p.planned_work();
    let budget = Int8GenerationBudget { native: Int8RunBudget::exact(work), max_kv_bytes: 8 * KV_BYTES_PER_TOKEN as u64,
        max_sampler_bytes: p.sampler_bytes() };
    check_bounds(work, p.sampler_bytes(), 8, budget).unwrap();
    assert!(check_bounds(work, p.sampler_bytes(), 5, budget).is_err());
    assert!(check_bounds(work, p.sampler_bytes(), 9, budget).is_err());
    for field in 0..6 {
        let mut b = budget;
        match field { 0 => b.max_sampler_bytes -= 1, 1 => b.max_kv_bytes -= 1,
            2 => b.native.max_forward_positions -= 1, 3 => b.native.max_attention_pairs -= 1,
            4 => b.native.max_projection_work.dot_products -= 1, _ => b.native.max_projection_work.multiply_accumulates -= 1 }
        assert!(check_bounds(work, p.sampler_bytes(), 8, b).is_err());
    }
}
#[test]
fn complete_context_refuses_during_planning_before_engine_access() {
    let options = GenerationOptions::greedy(2, 100, 0);
    assert!(Int8GenerationPlan::compile(vec![1; DEFAULT_ADMITTED_CONTEXT_CAP], options, identity(),
        "id", 0, GenerationLimits::default()).is_err());
}
#[test]
fn cancellation_stops_prefill_preserves_cause_and_aborts_driver() {
    let p = plan(GenerationOptions::greedy(2, 100, 0)); let mut model = Model::new(&[1]);
    model.control.stop_prompt = Some(1); let mut sink = Sink::default();
    let error = p.drive(&Decoder, 1, &mut sink, &mut model).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::CostBudget));
    assert_eq!(model.tokens, [5]); assert_eq!(model.heads, 0); assert!(model.aborted); assert!(sink.0.is_empty());
}
#[test]
fn native_failure_is_not_a_partial_success_or_retry() {
    let p = plan(GenerationOptions::greedy(4, 100, 0)); let mut model = Model::new(&[1]);
    model.fail_append = Some(3); let mut sink = Sink::default();
    let error = p.drive(&Decoder, 1, &mut sink, &mut model).err().unwrap();
    assert!(matches!(error, Int8GenerationError::Native(StrictInt8Error::Primitive)));
    assert_eq!(model.tokens, [5, 6, 7]); assert_eq!(sink.0.len(), 1); assert!(model.aborted);
}
#[test]
fn uncertain_stream_delivery_is_not_retried_and_aborts_driver() {
    struct Bad(usize);
    impl DecodeEventSink for Bad {
        type Permit = (); type Error = &'static str;
        fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
        fn permit(&mut self, _: (), _: DecodeTokenEvent) -> Result<(), Self::Error> { self.0 += 1; Err("private transport detail") }
    }
    let p = plan(GenerationOptions::greedy(4, 100, 0)); let mut model = Model::new(&[1]); let mut sink = Bad(0);
    let error = p.drive(&Decoder, 1, &mut sink, &mut model).err().unwrap();
    assert!(matches!(&error, Int8GenerationError::Generation(GenerationError::Stream)));
    assert!(!error.to_string().contains("private")); assert_eq!(sink.0, 1);
    assert_eq!(model.tokens, [5, 6, 7]); assert!(model.aborted);
}
#[test]
fn invalid_logits_and_forged_native_counters_are_fatal() {
    let p = plan(GenerationOptions::greedy(1, 100, 0));
    let mut model = Model::new(&[1]); model.corrupt = true;
    assert!(matches!(p.drive(&Decoder, 1, &mut Discard, &mut model), Err(Int8GenerationError::Generation(GenerationError::InvalidLogits))));
    assert!(model.aborted);
    let mut model = Model::new(&[1]); model.forged_work = true;
    assert!(matches!(p.drive(&Decoder, 1, &mut Discard, &mut model), Err(Int8GenerationError::WorkMismatch)));
    assert!(model.aborted);
}
