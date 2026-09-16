//! Synthetic projection/transport tests, not native-model qualification.
use super::*;
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "bf16-verbatim".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "generate-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::HfBf16Eager, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None }
}
fn plan(options: GenerationOptions) -> GenerationPlan {
    GenerationPlan::compile(vec![5, 6], options, identity(), "stable-item", 0, GenerationLimits::default()).unwrap()
}
struct Decoder;
impl DecodeByteDecoder for Decoder {
    type Error = &'static str;
    fn decode_token_ids(&self, tokens: &[u32]) -> Result<Vec<u8>, Self::Error> { Ok(tokens.iter().map(|&id| b'a' + id as u8).collect()) }
}
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
#[derive(Default)] struct Sink(Vec<DecodeTokenEvent>);
impl DecodeEventSink for Sink {
    type Permit = (); type Error = &'static str;
    fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
    fn permit(&mut self, _: (), event: DecodeTokenEvent) -> Result<(), Self::Error> { self.0.push(event); Ok(()) }
}
fn row(selected: usize) -> Vec<f32> { let mut row = vec![-100.0; NANBEIGE_VOCAB_SIZE]; row[selected] = 10.0; row }
#[test]
fn eos_is_scored_but_not_decoded_or_fed_back() {
    let mut options = GenerationOptions::greedy(4, 100, 0); options.min_new_tokens = 1; options.capture_logprobs = true;
    let p = plan(options); let mut sink = Sink::default(); let mut calls = 0;
    let out = p.run(&Decoder, 7, &mut sink, &mut Continue, |_| {
        calls += 1; Ok(row(if calls <= 2 { 1 } else { 0 }))
    }).unwrap();
    assert_eq!(out.token_ids, vec![1, 0]); assert_eq!(out.content_bytes, b"b"); assert_eq!(calls, 3);
    assert_eq!(out.finish_reason, GenerationFinish::Eos); assert_eq!(out.token_logprobs.as_ref().unwrap().len(), 2);
    assert_eq!(out.native_work.forward_positions, 3); assert_eq!(sink.0[1].decoded_bytes, b"");
    assert_eq!(sink.0.iter().flat_map(|e| e.decoded_bytes.iter().copied()).collect::<Vec<_>>(), out.content_bytes);
}
#[test]
fn byte_limit_refuses_pending_token_and_does_not_deliver_it() {
    let p = plan(GenerationOptions::greedy(4, 1, 0)); let mut sink = Sink::default(); let mut calls = 0;
    let out = p.run(&Decoder, 1, &mut sink, &mut Continue, |_| { calls += 1; Ok(row(1)) }).unwrap();
    assert_eq!(out.finish_reason, GenerationFinish::ByteLimit); assert_eq!(out.token_ids, vec![1]);
    assert_eq!(out.content_bytes, b"b"); assert_eq!(sink.0.len(), 1); assert_eq!(calls, 3);
}
#[test]
fn stop_suffix_matches_across_tokens_without_retracting_stream_bytes() {
    let mut options = GenerationOptions::greedy(8, 100, 0); options.stop_suffixes = vec![b"bb".to_vec()];
    let mut sink = Sink::default(); let out = plan(options).run(&Decoder, 1, &mut sink, &mut Continue, |_| Ok(row(1))).unwrap();
    assert_eq!(out.content_bytes, b"bb"); assert_eq!(out.finish_reason, GenerationFinish::StopSuffix); assert_eq!(sink.0.len(), 2);
}
#[test]
fn sampled_tokens_are_independent_of_transport_sequence_and_prior_runs() {
    let mut options = GenerationOptions::greedy(8, 100, 0); options.banned_token_ids = vec![0];
    options.sampling = GenerationSampling::Seeded { effective_seed: [9; 32], temperature_milli: 800, top_k: Some(20), top_p_ppm: 950000 };
    let p = plan(options); let mut logits = vec![-100.0; NANBEIGE_VOCAB_SIZE]; logits[1..21].fill(0.0);
    let run = |seq| p.run(&Decoder, seq, &mut Discard, &mut Continue, |_| Ok(logits.clone())).unwrap();
    let a = run(1); let _unrelated = run(99); let b = run(7);
    assert_eq!(a.token_ids, b.token_ids); assert_eq!(a.content_bytes, b.content_bytes);
    assert_eq!(a.native_work.sampled_steps, 8); assert_eq!(a.effective_seed.unwrap(), "09".repeat(32));
}
#[test]
fn full_model_and_request_options_bind_identity_before_execution() {
    let p = plan(GenerationOptions::greedy(1, 100, 0)); let mut changed = p.execution_identity().clone();
    changed.logical_model_digest = Sha256Digest::of_bytes(b"different"); assert!(p.verify_identity(&changed).is_err());
    changed = p.execution_identity().clone(); changed.decision_policy_digest = Sha256Digest::of_bytes(b"different");
    assert!(p.verify_identity(&changed).is_err()); p.verify_identity(p.execution_identity()).unwrap();
}
#[test]
fn byte_decoding_and_stream_errors_never_become_success() {
    struct Bad;
    impl DecodeEventSink for Bad {
        type Permit = (); type Error = &'static str;
        fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
        fn permit(&mut self, _: (), _: DecodeTokenEvent) -> Result<(), Self::Error> { Err("private sink detail") }
    }
    let p = plan(GenerationOptions::greedy(2, 100, 0)); let mut calls = 0;
    let error = p.run(&Decoder, 1, &mut Bad, &mut Continue, |_| { calls += 1; Ok(row(1)) }).err().unwrap();
    assert!(matches!(error, GenerationError::Stream)); assert_eq!(calls, 2); assert!(!error.to_string().contains("private"));
}
#[test]
fn cancellation_keeps_original_cause_and_does_not_continue_prefill() {
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
    }
    let p = plan(GenerationOptions::greedy(2, 100, 0)); let mut calls = 0;
    let error = p.run(&Decoder, 1, &mut Discard, &mut Stop, |_| { calls += 1; Ok(row(1)) }).err().unwrap();
    assert!(matches!(error, GenerationError::Cancelled(DecodeCancellationKind::Deadline))); assert_eq!(calls, 0);
}
#[test]
fn context_and_sampler_work_are_declared_before_model_calls() {
    let p = plan(GenerationOptions::greedy(4, 100, 0)); assert_eq!(p.planned_work().forward_positions, 5);
    let budget = GenerationBudget { max_forward_positions: 4, max_projected_logits: u64::MAX, max_kv_bytes: u64::MAX,
        max_sampler_bytes: u64::MAX }; assert!(p.check_budget(budget).is_err());
    let budget = GenerationBudget { max_forward_positions: 5, max_sampler_bytes: 0, ..budget }; assert!(p.check_budget(budget).is_err());
}
