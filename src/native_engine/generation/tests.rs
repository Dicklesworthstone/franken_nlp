//! Synthetic projection/transport tests, not native-model qualification.
use super::*;

pub(super) const CANCELLATION_KINDS: [DecodeCancellationKind; 11] = [
    DecodeCancellationKind::User, DecodeCancellationKind::Timeout,
    DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota,
    DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
    DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
    DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown,
    DecodeCancellationKind::LinkedExit,
];

/// Share a cancellation signal with an external decoder/sink. Tests trigger
/// semantic boundaries rather than depending on a particular poll count.
#[derive(Clone, Default)]
pub(super) struct BoundaryCancellation(std::rc::Rc<std::cell::Cell<Option<DecodeCancellationKind>>>);
impl DecodeStepControl for BoundaryCancellation {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { self.0.get() }
}
pub(super) struct CancellingDecoder {
    pub(super) signal: BoundaryCancellation,
    pub(super) cause: DecodeCancellationKind,
    pub(super) bytes: Vec<u8>,
}
impl DecodeByteDecoder for CancellingDecoder {
    type Error = &'static str;
    fn decode_token_ids(&self, _: &[u32]) -> Result<Vec<u8>, Self::Error> {
        self.signal.0.set(Some(self.cause));
        Ok(self.bytes.clone())
    }
}
pub(super) struct CancellingSink {
    pub(super) signal: BoundaryCancellation,
    pub(super) cause: DecodeCancellationKind,
    pub(super) events: Vec<DecodeTokenEvent>,
}
impl DecodeEventSink for CancellingSink {
    type Permit = (); type Error = &'static str;
    fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
    fn permit(&mut self, _: (), event: DecodeTokenEvent) -> Result<(), Self::Error> {
        self.events.push(event);
        self.signal.0.set(Some(self.cause));
        Ok(())
    }
}

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
fn decoding_cancellation_precedes_byte_limit_or_pending_token_delivery() {
    for cause in CANCELLATION_KINDS {
        for max_bytes in [1, 100] {
            let mut control = BoundaryCancellation::default();
            let decoder = CancellingDecoder { signal: control.clone(), cause, bytes: b"bb".to_vec() };
            let p = plan(GenerationOptions::greedy(1, max_bytes, 0));
            let mut sink = Sink::default(); let mut calls = 0;
            let error = p.run(&decoder, 1, &mut sink, &mut control, |_| {
                calls += 1; Ok(row(1))
            }).err().unwrap();
            assert!(matches!(error, GenerationError::Cancelled(actual) if actual == cause));
            assert_eq!(calls, p.prompt_tokens());
            assert!(sink.0.is_empty(), "cancelled decoding must not deliver a token");
        }
    }
}
#[test]
fn terminal_delivery_cancellation_never_becomes_a_successful_sequence() {
    for cause in CANCELLATION_KINDS {
        for finish in [GenerationFinish::Eos, GenerationFinish::StopSuffix, GenerationFinish::TokenLimit] {
            let mut options = GenerationOptions::greedy(if finish == GenerationFinish::TokenLimit { 1 } else { 4 }, 100, 0);
            if finish == GenerationFinish::StopSuffix { options.stop_suffixes = vec![b"b".to_vec()]; }
            let p = plan(options); let mut control = BoundaryCancellation::default();
            let mut sink = CancellingSink { signal: control.clone(), cause, events: Vec::new() };
            let mut calls = 0;
            let error = p.run(&Decoder, 7, &mut sink, &mut control, |_| {
                calls += 1; Ok(row(if finish == GenerationFinish::Eos { 0 } else { 1 }))
            }).err().unwrap();
            assert!(matches!(error, GenerationError::Cancelled(actual) if actual == cause));
            assert_eq!(calls, p.prompt_tokens(), "terminal delivery must not feed a token back");
            assert_eq!(sink.events.len(), 1, "a delivered event remains provisional and is never retried");
            assert_eq!(sink.events[0].request_seq, 7);
        }
    }
}
#[test]
fn context_and_sampler_work_are_declared_before_model_calls() {
    let p = plan(GenerationOptions::greedy(4, 100, 0)); assert_eq!(p.planned_work().forward_positions, 5);
    let budget = GenerationBudget { max_forward_positions: 4, max_projected_logits: u64::MAX, max_kv_bytes: u64::MAX,
        max_sampler_bytes: u64::MAX }; assert!(p.check_budget(budget).is_err());
    let budget = GenerationBudget { max_forward_positions: 5, max_sampler_bytes: 0, ..budget }; assert!(p.check_budget(budget).is_err());
}


#[derive(Default)]
struct SelectionControl { calls: usize, stop_at: Option<usize>, cause: Option<DecodeCancellationKind> }
impl DecodeStepControl for SelectionControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        if self.stop_at == Some(self.calls) { self.cause } else { None }
    }
}

#[test]
fn processor_cancellation_never_decodes_delivers_or_counts_a_sampled_token() {
    struct UnreachableDecoder;
    impl DecodeByteDecoder for UnreachableDecoder {
        type Error = &'static str;
        fn decode_token_ids(&self, _: &[u32]) -> Result<Vec<u8>, Self::Error> {
            panic!("cancelled token selection must not reach decoding");
        }
    }
    for sampling in [GenerationSampling::Greedy, GenerationSampling::Seeded {
        effective_seed: [9; 32], temperature_milli: 1000, top_k: None, top_p_ppm: 1_000_000,
    }, GenerationSampling::Seeded {
        effective_seed: [9; 32], temperature_milli: 1000, top_k: Some(20), top_p_ppm: 950_000,
    }] {
        let mut options = GenerationOptions::greedy(1, 100, 0); options.sampling = sampling;
        let p = plan(options);
        for cause in CANCELLATION_KINDS {
            let mut cursor = cursor::Cursor::new(&p, 3, GENERATION_VERSION).unwrap();
            for _ in 0..p.prompt_tokens() { cursor.record_forward(true).unwrap(); }
            let mut control = SelectionControl { stop_at: Some(3), cause: Some(cause), ..Default::default() };
            let mut sink = Sink::default();
            let error = cursor.emit_next(&row(1), &UnreachableDecoder, &mut sink, &mut control).unwrap_err();
            assert!(matches!(error, GenerationError::Cancelled(actual) if actual == cause));
            assert!(sink.0.is_empty()); assert!(cursor.output.token_ids.is_empty());
            assert_eq!(cursor.output.native_work.sampled_steps, 0);
        }
    }
}

#[test]
fn controlled_raw_logprobs_keep_full_denominator_bits_and_terminal_cancel_causes() {
    let logits: Vec<_> = (0..NANBEIGE_VOCAB_SIZE).map(|index| ((index * 17 % 101) as f32 - 50.0) / 7.0).collect();
    let maximum = logits.iter().copied().map(f64::from).fold(f64::NEG_INFINITY, f64::max);
    let sum = logits.iter().fold(0.0_f64, |sum, &value| sum + (f64::from(value) - maximum).exp());
    let mut completed = SelectionControl::default();
    for selected in [0, NANBEIGE_VOCAB_SIZE / 2, NANBEIGE_VOCAB_SIZE - 1] {
        let expected = ((f64::from(logits[selected]) - maximum) - sum.ln()) as f32;
        let actual = raw_logprob(&logits, selected as u32, &mut Continue, 0).unwrap();
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    raw_logprob(&logits, 0, &mut completed, 0).unwrap();
    for cause in CANCELLATION_KINDS {
        for stop_at in [1, 2, completed.calls / 2, completed.calls] {
            let mut control = SelectionControl { stop_at: Some(stop_at), cause: Some(cause), ..Default::default() };
            assert!(matches!(raw_logprob(&logits, 0, &mut control, 0),
                Err(GenerationError::Cancelled(actual)) if actual == cause));
            assert_eq!(control.calls, stop_at);
        }
    }
}

// Exercise the actual generation cursor, including zero-byte non-EOS events
// and partial UTF-8 pieces. Opaque decoders retain the full-prefix contract.
#[test]
fn native_piece_delivery_matches_prefix_replay_at_every_commit() {
    use crate::tokenizer::{
        bpe::{AddedToken, DecodeBytesError, SpBpeTokenizer},
        sp_model::{ModelType, NormalizerFacts, PieceType, SpecialPieceIds, SpmModel, SpmPiece},
    };

    struct DecoderRoute<'a> { tokenizer: &'a SpBpeTokenizer, incremental: bool }
    impl DecodeByteDecoder for DecoderRoute<'_> {
        type Error = DecodeBytesError;
        fn decode_token_ids(&self, ids: &[u32]) -> Result<Vec<u8>, Self::Error> {
            assert!(!self.incremental, "native cursor replayed a complete prefix");
            self.tokenizer.decode_bytes(ids)
        }
        fn sentencepiece_decoder(&self) -> Option<&SpBpeTokenizer> {
            self.incremental.then_some(self.tokenizer)
        }
    }

    let pieces = [
        ("<unk>", PieceType::Unknown), ("<s>", PieceType::Control),
        ("</s>", PieceType::Control), ("▁", PieceType::Normal),
        ("▁x", PieceType::Normal), ("<0xC3>", PieceType::Byte),
        ("<0xA9>", PieceType::Byte), ("<extra>", PieceType::Control),
    ].into_iter().map(|(piece, piece_type)| SpmPiece { piece: piece.to_owned(), score: 0.0, piece_type }).collect();
    let tokenizer = SpBpeTokenizer::with_added_tokens(SpmModel {
        pieces, model_type: ModelType::Bpe,
        normalizer: NormalizerFacts { name: "identity".to_owned(), is_identity: true,
            precompiled_charsmap_is_empty: true },
        special_ids: SpecialPieceIds { unk_id: 0, bos_id: 1, eos_id: 2, pad_id: -1 },
    }, [AddedToken::new("<added>", 7)]).unwrap();

    let script = [1_usize, 3, 4, 5, 6, 7, 2];
    let deltas: [&[u8]; 7] = [b"", b"", b" x", &[0xc3], &[0xa9], b"<added>", b""];
    for (limit, stop, finish, count) in [
        (64, false, GenerationFinish::Eos, 7),
        (3, false, GenerationFinish::ByteLimit, 4),
        (64, true, GenerationFinish::StopSuffix, 5),
    ] {
        let mut options = GenerationOptions::greedy(script.len(), limit, 2);
        if stop { options.stop_suffixes = vec!["é".as_bytes().to_vec()]; }
        let p = plan(options);
        for incremental in [false, true] {
            let mut calls = 0usize;
            let mut sink = Sink::default();
            let actual = p.run(&DecoderRoute { tokenizer: &tokenizer, incremental }, 7,
                &mut sink, &mut Continue, |_| {
                    let choice = script[calls.saturating_sub(p.prompt_tokens() - 1)];
                    calls += 1;
                    Ok(row(choice))
                }).unwrap();
            assert_eq!(actual.finish_reason, finish);
            assert_eq!(actual.token_ids, script[..count].iter().map(|&id| id as u32).collect::<Vec<_>>());
            assert_eq!(actual.content_bytes, deltas[..count].concat());
            assert_eq!(sink.0.len(), count);
            for (index, event) in sink.0.iter().enumerate() {
                assert_eq!(event.token_index, index);
                assert_eq!(event.decoded_bytes, deltas[index]);
            }
        }
    }
}
