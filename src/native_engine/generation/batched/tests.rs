//! Synthetic logits exercise the real shared Cursor/driver, not native quality.
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
fn plan(item: &str, prompt: Vec<u32>, options: GenerationOptions) -> GenerationPlan {
    GenerationPlan::compile(prompt, options, identity(), item, 0, GenerationLimits::default()).unwrap()
}
struct Decoder;
impl DecodeByteDecoder for Decoder {
    type Error = &'static str;
    fn decode_token_ids(&self, ids: &[u32]) -> Result<Vec<u8>, Self::Error> {
        if ids.contains(&0) { return Err("EOS must not be decoded"); }
        Ok(ids.iter().map(|&id| b'a' + id as u8).collect())
    }
}
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
#[derive(Default)] struct Sink(Vec<DecodeTokenEvent>);
impl DecodeEventSink for Sink {
    type Permit = (); type Error = &'static str;
    fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
    fn permit(&mut self, _: (), event: DecodeTokenEvent) -> Result<(), Self::Error> { self.0.push(event); Ok(()) }
}
fn logits(token: u32, position: usize, force: Option<usize>) -> Vec<f32> {
    let mut values = vec![-100.0; NANBEIGE_VOCAB_SIZE];
    for id in 1..=4 { values[id] = ((token as usize + position + id * 7) % 5) as f32 / 4.0; }
    if let Some(id) = force { values.fill(-100.0); values[id] = 10.0; }
    values
}
struct Fake {
    positions: Vec<usize>, closed: Vec<bool>, forced: Vec<Option<usize>>,
    groups: Vec<Vec<(usize, u32, bool)>>,
}
impl Fake {
    fn new(rows: usize) -> Self {
        Self { positions: vec![0; rows], closed: vec![false; rows], forced: vec![None; rows], groups: Vec::new() }
    }
}
impl ForwardGroup for Fake {
    fn step<C: DecodeStepControl>(&mut self, input: &[ScheduledToken], _: &mut C) -> Result<Vec<ForwardedRow>, GenerationError> {
        self.groups.push(input.iter().map(|t| (t.row, t.token, t.project)).collect());
        let mut output = Vec::new();
        for token in input {
            assert!(!self.closed[token.row], "finished rows must never receive feedback");
            let position = self.positions[token.row]; self.positions[token.row] += 1;
            output.push(ForwardedRow { row: token.row, position,
                logits: token.project.then(|| logits(token.token, position, self.forced[token.row])) });
        }
        Ok(output)
    }
    fn close(&mut self, row: usize) -> Result<(), GenerationError> {
        assert!(!self.closed[row]); self.closed[row] = true; Ok(())
    }
}
#[test]
fn ragged_prefill_and_decode_share_ticks_without_intermediate_projections() {
    let a = plan("a", vec![5], GenerationOptions::greedy(3, 100, 0));
    let b = plan("b", vec![6, 7, 8], GenerationOptions::greedy(3, 100, 0));
    let mut cursors = vec![Cursor::new(&a, 11, BATCH_GENERATION_VERSION).unwrap(), Cursor::new(&b, 22, BATCH_GENERATION_VERSION).unwrap()];
    let mut forward = Fake::new(2); forward.forced.fill(Some(1)); let mut sink = Sink::default();
    let steps = drive(&mut cursors, &mut forward, &Decoder, &mut sink, &mut Continue).unwrap();
    assert_eq!(steps, 5); assert_eq!(forward.positions, [3, 5]); assert_eq!(forward.closed, [true, true]);
    assert_eq!(forward.groups[0], [(0, 5, true), (1, 6, false)]);
    assert_eq!(forward.groups[1], [(0, 1, true), (1, 7, false)]);
    assert_eq!(cursors[0].output.native_work.projected_logits, 3 * NANBEIGE_VOCAB_SIZE as u64);
    assert_eq!(cursors[1].output.native_work.projected_logits, 3 * NANBEIGE_VOCAB_SIZE as u64);
    for seq in [11, 22] {
        let events: Vec<_> = sink.0.iter().filter(|e| e.request_seq == seq).collect();
        assert_eq!(events.iter().map(|e| e.token_index).collect::<Vec<_>>(), [0, 1, 2]);
    }
}
#[test]
fn seeded_batch_matches_scalar_and_reordering_without_reseeding() {
    let mut options = GenerationOptions::greedy(5, 100, 0); options.banned_token_ids = vec![0]; options.capture_logprobs = true;
    options.sampling = GenerationSampling::Seeded { effective_seed: [19; 32], temperature_milli: 800, top_k: Some(4), top_p_ppm: 900000 };
    let a = plan("a", vec![5, 6], options.clone()); let b = plan("b", vec![6, 7, 8, 9], options);
    let solo = |p: &GenerationPlan| {
        let mut position = 0;
        p.run(&Decoder, 999, &mut Discard, &mut Continue, |token| {
            let result = logits(token, position, None); position += 1; Ok(result)
        }).unwrap()
    };
    let expected = [solo(&a), solo(&b)];
    for order in [[&a, &b], [&b, &a]] {
        let mut cursors: Vec<_> = order.iter().enumerate().map(|(i, p)| Cursor::new(p, 100 + i as u64, BATCH_GENERATION_VERSION).unwrap()).collect();
        drive(&mut cursors, &mut Fake::new(2), &Decoder, &mut Discard, &mut Continue).unwrap();
        for (p, cursor) in order.iter().zip(cursors) {
            let target = if std::ptr::eq(*p, &a) { &expected[0] } else { &expected[1] };
            let actual = cursor.finish().unwrap();
            assert_eq!(actual.token_ids, target.token_ids); assert_eq!(actual.content_bytes, target.content_bytes);
            assert_eq!(actual.effective_seed, target.effective_seed); assert_eq!(actual.token_logprobs, target.token_logprobs);
            assert_eq!(actual.native_work.forward_positions, target.native_work.forward_positions);
            assert_eq!(actual.native_work.projected_logits, 5 * NANBEIGE_VOCAB_SIZE as u64);
            assert_eq!(actual.native_work.sampled_steps, 5);
        }
    }
}
#[test]
fn eos_and_byte_limits_close_only_the_finished_row_without_extra_forward() {
    let a = plan("a", vec![5], GenerationOptions::greedy(4, 100, 0));
    let b = plan("b", vec![6, 7], GenerationOptions::greedy(4, 1, 0));
    let mut cursors = vec![Cursor::new(&a, 1, BATCH_GENERATION_VERSION).unwrap(), Cursor::new(&b, 2, BATCH_GENERATION_VERSION).unwrap()];
    let mut forward = Fake::new(2); forward.forced = vec![Some(0), Some(1)]; let mut sink = Sink::default();
    drive(&mut cursors, &mut forward, &Decoder, &mut sink, &mut Continue).unwrap();
    assert_eq!(forward.positions, [1, 3]);
    assert_eq!(cursors[0].output.finish_reason, GenerationFinish::Eos); assert_eq!(cursors[0].output.token_ids, [0]);
    assert!(cursors[0].output.content_bytes.is_empty());
    assert_eq!(cursors[1].output.finish_reason, GenerationFinish::ByteLimit); assert_eq!(cursors[1].output.token_ids, [1]);
    assert_eq!(cursors[1].output.native_work.projected_logits, 2 * NANBEIGE_VOCAB_SIZE as u64);
    assert_eq!(sink.0.len(), 2, "the byte-refused proposal is never delivered");
}
#[test]
fn cancellation_keeps_cause_and_stops_before_the_next_native_group() {
    struct Cancel;
    impl DecodeStepControl for Cancel {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
        fn prefill_checkpoint(&mut self, index: usize) -> Option<DecodeCancellationKind> {
            (index == 1).then_some(DecodeCancellationKind::Deadline)
        }
    }
    let p = plan("p", vec![5, 6, 7], GenerationOptions::greedy(2, 100, 0));
    let mut cursors = vec![Cursor::new(&p, 1, BATCH_GENERATION_VERSION).unwrap()]; let mut forward = Fake::new(1);
    assert!(matches!(drive(&mut cursors, &mut forward, &Decoder, &mut Discard, &mut Cancel),
        Err(GenerationError::Cancelled(DecodeCancellationKind::Deadline))));
    assert_eq!(forward.groups.len(), 1);
}
#[test]
fn cursor_does_not_accept_double_forward_or_an_unfinished_result() {
    let p = plan("p", vec![5], GenerationOptions::greedy(2, 100, 0));
    let mut row = Cursor::new(&p, 1, BATCH_GENERATION_VERSION).unwrap();
    assert!(row.record_forward(false).is_err()); row.record_forward(true).unwrap();
    assert!(row.next_token().is_err()); assert!(row.record_forward(true).is_err()); assert!(row.finish().is_err());
}
#[test]
fn complete_budget_counts_selected_logits_all_rows_and_pending_output() {
    let p = plan("p", vec![5, 6, 7], GenerationOptions::greedy(4, 100, 0));
    let req = BatchGenerationRequest { plan: &p, admitted_identity: p.execution_identity(), slot: 0, request_seq: 1 };
    let r = requirements(&[req], 123).unwrap();
    assert_eq!(r.planned_work.forward_positions, 6); assert_eq!(r.planned_work.projected_logits, 4 * NANBEIGE_VOCAB_SIZE as u64);
    let exact = BatchGenerationBudget { max_forward_positions: r.planned_work.forward_positions,
        max_projected_logits: r.planned_work.projected_logits, max_native_payload_bytes: r.native_payload_bytes,
        max_sampler_payload_bytes: r.sampler_payload_bytes, max_output_payload_bytes: r.output_payload_upper_bytes };
    check_budget(r, exact).unwrap();
    assert!(check_budget(r, BatchGenerationBudget { max_forward_positions: 5, ..exact }).is_err());
    assert!(check_budget(r, BatchGenerationBudget { max_projected_logits: exact.max_projected_logits - 1, ..exact }).is_err());
    assert!(check_budget(r, BatchGenerationBudget { max_native_payload_bytes: 122, ..exact }).is_err());
    assert!(check_budget(r, BatchGenerationBudget { max_sampler_payload_bytes: 0, ..exact }).is_err());
    assert!(check_budget(r, BatchGenerationBudget { max_output_payload_bytes: 0, ..exact }).is_err());
}
#[test]
fn common_model_binding_excludes_row_policy_but_not_loaded_model_semantics() {
    let original = identity(); let mut changed = original.clone();
    changed.task_spec = "chat-v1".to_owned(); changed.decision_policy_digest = Sha256Digest::of_bytes(b"other request");
    assert_eq!(model_binding(&original).unwrap(), model_binding(&changed).unwrap());
    changed.logical_model_digest = Sha256Digest::of_bytes(b"other model");
    assert_ne!(model_binding(&original).unwrap(), model_binding(&changed).unwrap());
    changed = original.clone(); changed.tokenizer_digest = Sha256Digest::of_bytes(b"other tokenizer");
    assert_ne!(model_binding(&original).unwrap(), model_binding(&changed).unwrap());
    changed = original.clone(); changed.backend_semantic_version = "other".to_owned();
    assert_ne!(model_binding(&original).unwrap(), model_binding(&changed).unwrap());
}
