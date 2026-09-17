//! Weightless tests of the real refill driver and shared generation Cursor.
//! Synthetic logits are mechanism evidence, not model parity or host speed.
use super::*;
use std::{cell::Cell, rc::Rc};

fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"refill test only");
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
    slots: Vec<usize>, owners: Vec<Option<usize>>, positions: Vec<usize>, forced: Vec<Option<usize>>,
    opened: Vec<usize>, closed: Vec<usize>, groups: Vec<Vec<usize>>, corrupt_tick: Option<usize>,
    fail_open: Option<usize>, fail_close: Option<usize>, cancel_on_close: Option<Rc<Cell<bool>>>,
}
impl Fake {
    fn new(slots: &[usize]) -> Self {
        Self { slots: slots.to_vec(), owners: vec![None; MAX_BATCH_ROWS], positions: vec![0; slots.len()],
            forced: vec![Some(1); slots.len()], opened: Vec::new(), closed: Vec::new(), groups: Vec::new(),
            corrupt_tick: None, fail_open: None, fail_close: None, cancel_on_close: None }
    }
}
impl RefillForward for Fake {
    fn open(&mut self, row: usize) -> Result<(), GenerationError> {
        if self.fail_open == Some(row) { return Err(GenerationError::Allocation); }
        let slot = self.slots[row];
        assert!(self.owners[slot].is_none(), "replacement must wait for actual close");
        assert!(!self.opened.contains(&row)); assert_eq!(self.positions[row], 0);
        self.owners[slot] = Some(row); self.opened.push(row); Ok(())
    }
}
impl ForwardGroup for Fake {
    fn step<C: DecodeStepControl>(&mut self, tokens: &[ScheduledToken], _: &mut C) -> Result<Vec<ForwardedRow>, GenerationError> {
        self.groups.push(tokens.iter().map(|t| t.row).collect());
        let mut output = Vec::new();
        for token in tokens {
            assert_eq!(self.owners[self.slots[token.row]], Some(token.row));
            let position = self.positions[token.row]; self.positions[token.row] += 1;
            output.push(ForwardedRow { row: token.row, position,
                logits: token.project.then(|| logits(token.token, position, self.forced[token.row])) });
        }
        if self.corrupt_tick == Some(self.groups.len()) { output.last_mut().unwrap().position += 100; }
        Ok(output)
    }
    fn close(&mut self, row: usize) -> Result<(), GenerationError> {
        if self.fail_close == Some(row) { return Err(GenerationError::Contract("synthetic close failure")); }
        assert_eq!(self.owners[self.slots[row]], Some(row));
        self.owners[self.slots[row]] = None; self.closed.push(row);
        if let Some(flag) = &self.cancel_on_close { flag.set(true); }
        Ok(())
    }
}
fn cursors(plans: &[GenerationPlan]) -> Vec<Cursor<'_>> {
    plans.iter().enumerate().map(|(i, p)| Cursor::new(p, i as u64 + 1, BATCH_GENERATION_VERSION).unwrap()).collect()
}
#[test]
fn free_slot_refills_while_a_sibling_is_still_prefilling_and_decoding() {
    let plans = [plan("a", vec![5], GenerationOptions::greedy(1, 100, 0)),
        plan("b", vec![6, 7, 8], GenerationOptions::greedy(3, 100, 0)),
        plan("c", vec![9], GenerationOptions::greedy(4, 100, 0)),
        plan("d", vec![10], GenerationOptions::greedy(1, 100, 0))];
    let slots = [0, 1, 0, 1]; let mut rows = cursors(&plans); let mut forward = Fake::new(&slots); let mut sink = Sink::default();
    let steps = drive(&mut rows, &mut SlotQueue::new(&slots).unwrap(), &mut forward, &Decoder, &mut sink, &mut Continue).unwrap();
    assert_eq!(steps, 6, "the whole-wave barrier would require max(1,5)+max(4,1)=9 ticks");
    assert_eq!(forward.positions, [1, 5, 4, 1]);
    assert_eq!(forward.groups[0], [0, 1]); assert_eq!(forward.groups[1], [1, 2]); assert_eq!(forward.groups[5], [3]);
    assert!(forward.owners.iter().all(Option::is_none));
    for (row, expected) in rows.into_iter().zip([1, 3, 4, 1]) {
        let output = row.finish().unwrap();
        assert_eq!(output.token_ids.len(), expected);
        assert_eq!(output.native_work.projected_logits, expected as u64 * NANBEIGE_VOCAB_SIZE as u64);
        let tokens: Vec<_> = sink.0.iter().filter(|e| e.request_seq == output.request_seq).map(|e| e.token_index).collect();
        assert_eq!(tokens, (0..expected).collect::<Vec<_>>());
    }
}
#[test]
fn addressed_sampling_matches_scalar_across_slot_reuse_and_request_reordering() {
    let mut options = GenerationOptions::greedy(4, 100, 0); options.banned_token_ids = vec![0]; options.capture_logprobs = true;
    options.sampling = GenerationSampling::Seeded { effective_seed: [37; 32], temperature_milli: 800, top_k: Some(4), top_p_ppm: 900000 };
    let plans: Vec<_> = (0..4).map(|i| plan(&format!("row-{i}"), vec![5 + i as u32; i + 1], options.clone())).collect();
    let expected: Vec<_> = plans.iter().map(|p| {
        let mut position = 0;
        p.run(&Decoder, 99, &mut Discard, &mut Continue, |token| {
            let result = logits(token, position, None); position += 1; Ok(result)
        }).unwrap()
    }).collect();
    for order in [[0, 1, 2, 3], [3, 1, 0, 2]] {
        for slots in [[0, 0, 0, 0], [1, 0, 1, 0], [3, 2, 1, 0]] {
            let mut rows: Vec<_> = order.iter().map(|&i| Cursor::new(&plans[i], i as u64 + 1, BATCH_GENERATION_VERSION).unwrap()).collect();
            let mut forward = Fake::new(&slots); forward.forced.fill(None);
            drive(&mut rows, &mut SlotQueue::new(&slots).unwrap(), &mut forward, &Decoder, &mut Discard, &mut Continue).unwrap();
            for (row, i) in rows.into_iter().zip(order) {
                let actual = row.finish().unwrap(); let target = &expected[i];
                assert_eq!(actual.token_ids, target.token_ids); assert_eq!(actual.content_bytes, target.content_bytes);
                assert_eq!(actual.token_logprobs, target.token_logprobs); assert_eq!(actual.effective_seed, target.effective_seed);
                assert_eq!(actual.native_work.forward_positions, target.native_work.forward_positions);
                assert_eq!(actual.native_work.projected_logits, 4 * NANBEIGE_VOCAB_SIZE as u64);
            }
        }
    }
}
#[test]
fn eos_byte_refusal_and_stop_suffix_each_release_the_slot_without_feedback() {
    let mut stop = GenerationOptions::greedy(5, 100, 0); stop.stop_suffixes = vec![b"b".to_vec()];
    let plans = [plan("eos", vec![5], GenerationOptions::greedy(4, 100, 0)),
        plan("bytes", vec![6], GenerationOptions::greedy(4, 1, 0)), plan("stop", vec![7], stop)];
    let slots = [0, 0, 0]; let mut rows = cursors(&plans); let mut forward = Fake::new(&slots); forward.forced[0] = Some(0);
    let mut sink = Sink::default();
    let steps = drive(&mut rows, &mut SlotQueue::new(&slots).unwrap(), &mut forward, &Decoder, &mut sink, &mut Continue).unwrap();
    assert_eq!(steps, 4); assert_eq!(forward.positions, [1, 2, 1]); assert_eq!(forward.closed, [0, 1, 2]);
    assert_eq!(rows[0].output.finish_reason, GenerationFinish::Eos);
    assert_eq!(rows[1].output.finish_reason, GenerationFinish::ByteLimit);
    assert_eq!(rows[2].output.finish_reason, GenerationFinish::StopSuffix);
    assert_eq!(sink.0.len(), 3, "the byte-refused proposal never acquires delivery");
}
#[test]
fn cancellation_after_retirement_prevents_replacement_open() {
    struct Control(Rc<Cell<bool>>);
    impl DecodeStepControl for Control {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { self.0.get().then_some(DecodeCancellationKind::Deadline) }
    }
    let flag = Rc::new(Cell::new(false));
    let plans = [plan("a", vec![5], GenerationOptions::greedy(1, 100, 0)), plan("b", vec![6], GenerationOptions::greedy(1, 100, 0))];
    let slots = [0, 0]; let mut rows = cursors(&plans); let mut forward = Fake::new(&slots); forward.cancel_on_close = Some(Rc::clone(&flag));
    let error = drive(&mut rows, &mut SlotQueue::new(&slots).unwrap(), &mut forward, &Decoder, &mut Discard, &mut Control(flag)).unwrap_err();
    assert!(matches!(error, GenerationError::Cancelled(DecodeCancellationKind::Deadline)));
    assert_eq!(forward.opened, [0]); assert_eq!(forward.groups.len(), 1); assert_eq!(forward.closed, [0]);
}
#[test]
fn corrupted_replacement_position_is_rejected_before_any_tick_token_delivery() {
    let plans = [plan("a", vec![5], GenerationOptions::greedy(1, 100, 0)),
        plan("b", vec![6], GenerationOptions::greedy(3, 100, 0)), plan("c", vec![7], GenerationOptions::greedy(1, 100, 0))];
    let slots = [0, 1, 0]; let mut rows = cursors(&plans); let mut forward = Fake::new(&slots); forward.corrupt_tick = Some(2);
    let mut sink = Sink::default();
    assert!(drive(&mut rows, &mut SlotQueue::new(&slots).unwrap(), &mut forward, &Decoder, &mut sink, &mut Continue).is_err());
    assert_eq!(sink.0.len(), 2, "neither row may publish a token from the corrupt second reply");
    assert_eq!(forward.groups.len(), 2); assert_eq!(forward.opened, [0, 1, 2]);
}
#[test]
fn failed_close_or_replacement_open_cannot_restart_or_retry_a_request() {
    let plans = [plan("a", vec![5], GenerationOptions::greedy(1, 100, 0)), plan("b", vec![6], GenerationOptions::greedy(1, 100, 0))];
    for fail_close in [false, true] {
        let slots = [0, 0]; let mut rows = cursors(&plans); let mut forward = Fake::new(&slots);
        if fail_close { forward.fail_close = Some(0); } else { forward.fail_open = Some(1); }
        assert!(drive(&mut rows, &mut SlotQueue::new(&slots).unwrap(), &mut forward, &Decoder, &mut Discard, &mut Continue).is_err());
        assert_eq!(forward.opened, [0]); assert_eq!(forward.groups.len(), 1);
    }
}
#[test]
fn sink_failure_aborts_before_slot_reuse_and_is_not_a_document_success() {
    struct Broken;
    impl DecodeEventSink for Broken {
        type Permit = (); type Error = ();
        fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), ()> { Err(()) }
        fn permit(&mut self, _: (), _: DecodeTokenEvent) -> Result<(), ()> { panic!("failed reserve must not deliver") }
    }
    let plans = [plan("a", vec![5], GenerationOptions::greedy(1, 100, 0)), plan("b", vec![6], GenerationOptions::greedy(1, 100, 0))];
    let slots = [0, 0]; let mut rows = cursors(&plans); let mut forward = Fake::new(&slots);
    assert!(matches!(drive(&mut rows, &mut SlotQueue::new(&slots).unwrap(), &mut forward, &Decoder, &mut Broken, &mut Continue), Err(GenerationError::Stream)));
    assert_eq!(forward.opened, [0]); assert!(forward.closed.is_empty()); assert!(rows[0].output.token_ids.is_empty());
}
#[test]
fn queued_identity_and_duplicate_sequence_are_checked_before_native_state() {
    let a = plan("a", vec![5], GenerationOptions::greedy(1, 100, 0));
    let b = plan("b", vec![6], GenerationOptions::greedy(1, 100, 0));
    let mut requests = vec![BatchGenerationRequest { plan: &a, admitted_identity: a.execution_identity(), slot: 0, request_seq: 1 },
        BatchGenerationRequest { plan: &b, admitted_identity: b.execution_identity(), slot: 0, request_seq: 2 }];
    validate_requests(a.execution_identity(), &requests).unwrap();
    requests[1].request_seq = 1; assert!(validate_requests(a.execution_identity(), &requests).is_err());
    requests[1].request_seq = 2;
    let mut foreign = b.execution_identity().clone(); foreign.backend_semantic_version = "foreign".to_owned();
    requests[1].admitted_identity = &foreign;
    assert!(matches!(validate_requests(a.execution_identity(), &requests), Err(GenerationError::Identity)));
}
#[test]
fn queued_rows_are_included_in_the_complete_window_resource_price() {
    let p = plan("p", vec![5, 6], GenerationOptions::greedy(3, 100, 0));
    let requests: Vec<_> = (1..=3).map(|sequence| BatchGenerationRequest { plan: &p, admitted_identity: p.execution_identity(), slot: 0, request_seq: sequence }).collect();
    let price = requirements(&requests, 99).unwrap(); let one = requirements(&requests[..1], 99).unwrap();
    assert_eq!(price.native_payload_bytes, 99);
    assert_eq!(price.planned_work.forward_positions, 3 * one.planned_work.forward_positions);
    assert_eq!(price.sampler_payload_bytes, 3 * one.sampler_payload_bytes);
    assert_eq!(price.output_payload_upper_bytes, 3 * one.output_payload_upper_bytes);
    let insufficient = BatchGenerationBudget { max_forward_positions: u64::MAX, max_projected_logits: u64::MAX,
        max_native_payload_bytes: 99, max_sampler_payload_bytes: one.sampler_payload_bytes, max_output_payload_bytes: u64::MAX };
    assert!(check_budget(price, insufficient).is_err());
}
#[test]
fn queue_rejects_invalid_shape_and_stale_retirement() {
    assert!(SlotQueue::new(&[]).is_err()); assert!(SlotQueue::new(&[MAX_BATCH_ROWS]).is_err());
    assert!(SlotQueue::new(&[0; MAX_REFILL_REQUESTS + 1]).is_err());
    let mut queue = SlotQueue::new(&[0, 0]).unwrap(); let mut forward = Fake::new(&[0, 0]);
    queue.fill(&mut forward, &mut Continue).unwrap();
    assert!(queue.retire(1).is_err()); forward.close(0).unwrap(); queue.retire(0).unwrap();
    queue.fill(&mut forward, &mut Continue).unwrap(); assert!(queue.retire(0).is_err());
    assert_eq!(queue.active[0], Some(1));
}
#[test]
fn exhaustive_fifo_service_schedules_have_no_idle_gap_or_lost_request() {
    let mut cases = 0;
    for count in 1..=5_usize {
        for mut code in 0..9_usize.pow(count as u32) {
            let mut slots = Vec::new(); let mut lengths = Vec::new();
            for _ in 0..count { slots.push((code % 9) / 3); lengths.push(code % 3 + 1); code /= 9; }
            let mut queue = SlotQueue::new(&slots).unwrap(); let mut forward = Fake::new(&slots);
            let mut remaining = lengths.clone(); let mut ticks = 0;
            loop {
                queue.fill(&mut forward, &mut Continue).unwrap();
                let active: Vec<_> = queue.active.iter().flatten().copied().collect();
                if active.is_empty() { break; }
                ticks += 1;
                for row in active {
                    assert!(remaining[row] > 0); remaining[row] -= 1;
                    if remaining[row] == 0 { forward.close(row).unwrap(); queue.retire(row).unwrap(); }
                }
                assert!(ticks <= lengths.iter().sum::<usize>());
            }
            assert!(remaining.iter().all(|&v| v == 0));
            let expected = (0..3).map(|slot| (0..count).filter(|&row| slots[row] == slot).map(|row| lengths[row]).sum::<usize>()).max().unwrap();
            assert_eq!(ticks, expected);
            for slot in 0..3 {
                let opened: Vec<_> = forward.opened.iter().copied().filter(|&row| slots[row] == slot).collect();
                assert_eq!(opened, (0..count).filter(|&row| slots[row] == slot).collect::<Vec<_>>());
            }
            cases += 1;
        }
    }
    assert_eq!(cases, 66_429);
}
