//! Real cursor/planner tests with synthetic native rows, not model parity.
use super::*;

fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"packed-int4-test");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: recipe(128), packing_set_digest: packing_digest(d, 128).unwrap(),
        tokenizer_digest: d, template_digest: d, task_spec: "generate-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn limits(rows: usize) -> Int4PrefillLimits {
    Int4PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
fn plan(length: usize, item: usize, seeded: bool) -> Int4GenerationPlan {
    let mut options = GenerationOptions::greedy(4, 64, 0); options.capture_logprobs = true;
    if seeded {
        options.banned_token_ids = vec![0];
        options.sampling = GenerationSampling::Seeded { effective_seed: [7; 32], temperature_milli: 900,
            top_k: Some(3), top_p_ppm: 950000 };
    }
    Int4GenerationPlan::compile(vec![5; length], options, identity(), 128,
        &format!("item-{item}"), item as u64, GenerationLimits::default()).unwrap()
}
fn budget() -> Int4BatchGenerationBudget {
    Int4BatchGenerationBudget { native: Int4RunBudget { max_forward_positions: u64::MAX,
        max_attention_pairs: u64::MAX, max_projection_work: crate::native_engine::portable_int8::ProjectionWork {
            dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
        max_native_payload_bytes: u64::MAX, max_sampler_payload_bytes: u64::MAX, max_output_payload_bytes: u64::MAX }
}
fn requests(plans: &[Int4GenerationPlan]) -> Vec<Int4BatchGenerationRequest<'_>> {
    plans.iter().enumerate().map(|(row, plan)| Int4BatchGenerationRequest { plan,
        admitted_identity: plan.execution_identity(), request_seq: row as u64 + 1,
        budget: Int4GenerationBudget { native: Int4RunBudget::exact(plan.planned_work()),
            max_kv_bytes: 32 * KV_BYTES_PER_TOKEN as u64, max_sampler_bytes: plan.sampler_bytes() } }).collect()
}
#[derive(Default)] struct Control { polls: usize, cancel: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1; (self.cancel == Some(self.polls)).then_some(DecodeCancellationKind::CostBudget)
    }
}
struct Model { control: Control, tokens: Vec<Vec<u32>>, heads: Vec<usize>, packs: Vec<Vec<(usize, usize)>>,
    aborted: bool, fault: usize, flat: bool }
impl Model {
    fn new(rows: usize, flat: bool) -> Self {
        Self { control: Control::default(), tokens: vec![Vec::new(); rows], heads: vec![0; rows],
            packs: Vec::new(), aborted: false, fault: 0, flat }
    }
}
impl BatchDriver for Model {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn abort(&mut self) { self.aborted = true; }
    fn work(&self, row: usize) -> Result<Int4Work, Int4GenerationError> {
        let mut work = Int4Work::for_sequence(128, 0, self.tokens[row].len(), self.heads[row] * NANBEIGE_VOCAB_SIZE)?;
        if self.fault == 6 { work.projected_logits += 1; } Ok(work)
    }
    fn step(&mut self, tokens: &[Step]) -> Result<StepOutput, Int4GenerationError> {
        let runs: Vec<_> = tokens.iter().map(|token| ScheduledRun { row: token.row,
            tokens: std::slice::from_ref(&token.token), project: token.project }).collect();
        let output = self.packed_step(&runs, limits(tokens.len()))?;
        Ok(StepOutput { rows: output.rows.into_iter().map(|row| Forwarded {
            row: row.row, position: row.position, project: row.project }).collect(), logits: output.logits })
    }
}
impl PackedDriver for Model {
    fn packed_step(&mut self, runs: &[ScheduledRun<'_>], limits: Int4PrefillLimits)
        -> Result<PackedOutput, Int4GenerationError> {
        assert!(runs.iter().map(|r| r.tokens.len()).sum::<usize>() <= limits.max_batch_rows);
        self.packs.push(runs.iter().map(|r| (r.row, r.tokens.len())).collect());
        if self.fault == 7 { return Err(StrictInt4Error::Primitive.into()); }
        let mut rows = Vec::new(); let mut logits = Vec::new();
        for run in runs {
            rows.push(ForwardedRun { row: run.row, position: self.tokens[run.row].len(),
                count: run.tokens.len(), project: run.project });
            self.tokens[run.row].extend_from_slice(run.tokens);
            if run.project {
                let head = self.heads[run.row]; self.heads[run.row] += 1;
                let mut values = vec![-100.0; NANBEIGE_VOCAB_SIZE];
                if self.flat { values[1..4].fill(0.0); }
                else { values[if head < 2 { head + 1 } else { 0 }] = 10.0; }
                logits.extend(values);
            }
        }
        match self.fault {
            1 => { rows.last_mut().unwrap().row += 1; }
            2 => { rows.last_mut().unwrap().position += 1; }
            3 => { rows.last_mut().unwrap().count += 1; }
            4 => { *logits.last_mut().unwrap() = f32::NAN; }
            5 => { logits.pop(); }
            _ => (),
        }
        Ok(PackedOutput { rows, logits })
    }
}
struct Decoder;
impl DecodeByteDecoder for Decoder {
    type Error = ();
    fn decode_token_ids(&self, ids: &[u32]) -> Result<Vec<u8>, ()> { Ok(ids.iter().map(|&id| b'a' + id as u8).collect()) }
}
#[derive(Default)] struct Sink { events: Vec<DecodeTokenEvent>, fail: bool }
impl DecodeEventSink for Sink {
    type Error = (); type Permit = ();
    fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), ()> { Ok(()) }
    fn permit(&mut self, _: (), event: DecodeTokenEvent) -> Result<(), ()> {
        self.events.push(event); if self.fail { Err(()) } else { Ok(()) }
    }
}
fn run(plans: &[Int4GenerationPlan], width: Option<usize>, model: &mut Model, sink: &mut Sink)
    -> Result<Int4BatchGenerationOutput, Int4GenerationError> {
    let requests = requests(plans); let contexts = vec![32; plans.len()];
    let cursors = plans.iter().enumerate().map(|(i, p)| cursor::Cursor::new(&p.plan, i as u64 + 1, INT4_GENERATION_VERSION).unwrap()).collect();
    match width {
        Some(width) => drive(cursors, requirements(&requests, &contexts, budget(), limits(width))?, limits(width), model, &Decoder, sink),
        None => super::super::drive(cursors, super::super::requirements(&requests, &contexts, budget())?, model, &Decoder, sink),
    }
}

#[test]
fn packed_and_ordinary_batch_share_exact_tokens_scores_stops_and_group_work() {
    for seeded in [false, true] {
        let plans: Vec<_> = [2, 9, 1].into_iter().enumerate().map(|(i, n)| plan(n, i, seeded)).collect();
        let mut reference = Model::new(3, seeded);
        let expected = run(&plans, None, &mut reference, &mut Sink::default()).unwrap();
        for width in [1, 2, 4, 16, 64] {
            let mut model = Model::new(3, seeded); let mut sink = Sink::default();
            let actual = run(&plans, Some(width), &mut model, &mut sink).unwrap();
            assert!(actual.runs == expected.runs); assert_eq!(actual.actual_work, expected.actual_work);
            assert_eq!(model.tokens, reference.tokens); assert_eq!(model.heads, reference.heads);
            assert!(model.packs.iter().all(|p| p.iter().map(|(_, n)| n).sum::<usize>() <= width));
            for result in &actual.runs {
                let bytes: Vec<_> = sink.events.iter().filter(|e| e.request_seq == result.sequence.request_seq)
                    .flat_map(|e| e.decoded_bytes.iter().copied()).collect();
                assert_eq!(bytes, result.sequence.content_bytes);
            }
            if width == 64 { assert!(actual.group_steps < expected.group_steps); }
        }
    }
}
#[test]
fn packing_width_smaller_than_sequence_count_does_not_starve_rows_or_change_seeded_output() {
    let mut plans: Vec<_> = (0..64).map(|i| plan(1 + i % 3, i, true)).collect();
    let expected = run(&plans, Some(4), &mut Model::new(64, true), &mut Sink::default()).unwrap();
    plans.reverse();
    let actual = run(&plans, Some(7), &mut Model::new(64, true), &mut Sink::default()).unwrap();
    for (a, b) in expected.runs.iter().zip(actual.runs.iter().rev()) {
        assert_eq!(a.sequence.token_ids, b.sequence.token_ids);
        assert_eq!(a.sequence.token_logprobs, b.sequence.token_logprobs);
        assert_eq!(a.model_work, b.model_work);
    }
}
#[test]
fn complete_extra_workspace_is_counted_before_execution() {
    let plans = [plan(3, 0, false), plan(7, 1, false)]; let requests = requests(&plans);
    let base = super::super::requirements(&requests, &[32, 32], budget()).unwrap();
    for width in [1, 4, 64] {
        let mut exact = budget(); let extra = limits(width).validate().unwrap();
        exact.max_native_payload_bytes = base.native_payload_bytes + extra;
        let required = requirements(&requests, &[32, 32], exact, limits(width)).unwrap();
        assert_eq!(required.native.kv_bytes, base.native.kv_bytes);
        assert_eq!(required.native.rope_bytes, base.native.rope_bytes);
        assert_eq!(required.native.scratch_payload_bound, base.native.scratch_payload_bound + extra);
        assert_eq!(required.planned_work, base.planned_work);
        exact.max_native_payload_bytes -= 1;
        assert!(requirements(&requests, &[32, 32], exact, limits(width)).is_err());
    }
}
#[test]
fn late_native_routing_shape_or_logit_failure_cannot_publish_an_earlier_token() {
    let plans = [plan(1, 0, false), plan(1, 1, false)];
    for fault in 1..=5 {
        let mut model = Model::new(2, false); model.fault = fault; let mut sink = Sink::default();
        assert!(run(&plans, Some(64), &mut model, &mut sink).is_err());
        assert!(sink.events.is_empty()); assert!(model.aborted); assert_eq!(model.packs.len(), 1);
    }
}
#[test]
fn native_failure_cancellation_uncertain_delivery_and_work_mismatch_never_retry() {
    let plans = [plan(1, 0, false), plan(1, 1, false)];
    for fault in [6, 7] {
        let mut model = Model::new(2, false); model.fault = fault;
        assert!(run(&plans, Some(64), &mut model, &mut Sink::default()).is_err()); assert!(model.aborted);
    }
    let mut model = Model::new(2, false); model.control.cancel = Some(1);
    let error = run(&plans, Some(64), &mut model, &mut Sink::default()).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::CostBudget)); assert!(model.packs.is_empty());
    let mut model = Model::new(2, false); let mut sink = Sink { fail: true, ..Sink::default() };
    assert!(run(&plans, Some(64), &mut model, &mut sink).is_err());
    assert!(model.aborted); assert_eq!(sink.events.len(), 1); assert_eq!(model.packs.len(), 1);
}
#[test]
fn byte_stop_does_not_feed_back_the_rejected_proposal_or_advance_finished_neighbors() {
    let mut options = GenerationOptions::greedy(4, 1, 0);
    options.capture_logprobs = true;
    let first = Int4GenerationPlan::compile(vec![5], options, identity(), 128, "bytes", 0, GenerationLimits::default()).unwrap();
    let plans = [first, plan(9, 1, false)]; let mut model = Model::new(2, false);
    let output = run(&plans, Some(4), &mut model, &mut Sink::default()).unwrap();
    assert_eq!(output.runs[0].sequence.finish_reason, GenerationFinish::ByteLimit);
    assert_eq!(output.runs[0].sequence.token_ids, [1]); assert_eq!(model.tokens[0], [5, 1]);
    assert_eq!(model.heads[0], 2); assert_eq!(model.tokens[1].len(), 11);
}
