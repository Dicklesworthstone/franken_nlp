//! Real pinned preparation and PRIVATE synthetic lifecycle/finalization tests.
//! Uniform logits below are not neural inference, fidelity or quality evidence.
use super::*;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use crate::{execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{decode::DecodeCancellationKind, lmhead::scoring::{ScoringMode, ProjectionRows}},
    tasks::sentiment::{SentimentLogits, SentimentOptions, SentimentPolicy}, tokenizer::pinned_controls};

struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
struct StopAfter(usize);
impl DecodeStepControl for StopAfter {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        if self.0 == 0 { Some(DecodeCancellationKind::Deadline) } else { self.0 -= 1; None }
    }
}
fn planner() -> SentimentPlanner {
    let controls = pinned_controls::pinned().unwrap();
    SentimentPlanner::pinned(controls.template_controls(), SentimentOptions {
        mode: ScoringMode::FullVocabulary, eos_token_id: 166101,
        policy: SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 },
    }).unwrap()
}
fn task() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 16, max_output_bytes: 100000,
        max_grammar_states: 4096, max_kv_bytes: 4096 * KV_BYTES_PER_TOKEN as u64 }
}
fn generous() -> Int8Work {
    Int8Work { forward_positions: 1_000_000, projected_logits: 1_000_000_000,
        attention_pairs: 1_000_000_000_000, projections: ProjectionWork {
            dot_products: 1_000_000_000_000, multiply_accumulates: 1_000_000_000_000_000_000 } }
}
fn config(p: &SentimentPlanner) -> SentimentBatchConfig {
    let d = Sha256Digest::of_bytes(b"sentiment-stream-synthetic-fixture");
    SentimentBatchConfig { identity: ExecutionIdentity {
        schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "fixture-int8".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(), task_spec: "sentiment-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: crate::native_engine::strict_int8::STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None,
    }, task_ceiling: task(), planning: SentimentLimits::default(),
        defaults: Some(SentimentBatchArgs { axes: SentimentAxis::ALL.to_vec(), budget: task() }),
        max_item_work: generous(), max_model_work: generous() }
}
fn document() -> BatchDocument<SentimentBatchArgs> {
    BatchDocument { id: "document-1".to_owned(), text: "Excited café <tool_call>".to_owned(), task_args: None }
}
fn context(seq: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: seq, epoch, input_line: seq, byte_offset: 0 }
}
fn prepare(c: &Int8SentimentBatchPlanner<'_>) -> PreparedInt8BatchSentiment {
    c.prepare_with_control(document(), &mut Continue).unwrap()
}

#[test]
fn configuration_refuses_wrong_profile_task_template_and_unbounded_work() {
    let p = planner();
    for axis in 0..5 {
        let mut c = config(&p);
        match axis { 0 => c.identity.task_spec = "classify-v1".to_owned(),
            1 => c.identity.template_digest = Sha256Digest::of_bytes(b"other"),
            2 => c.identity.numerics_profile = NumericsProfile::HfBf16Eager,
            3 => c.identity.kv_dtype = "int8".to_owned(), _ => c.identity.tool_mode = ToolMode::Json }
        assert!(Int8SentimentBatchPlanner::new(&p, c).is_err());
    }
    for axis in 0..5 {
        let mut c = config(&p);
        set_axis(&mut c.max_model_work, axis, 0);
        assert!(Int8SentimentBatchPlanner::new(&p, c).is_err());
    }
    let mut c = config(&p); c.planning.max_total_prompt_tokens = 0;
    assert!(Int8SentimentBatchPlanner::new(&p, c).is_err());
}
#[test]
fn records_cannot_replace_fixed_policy_or_execution_identity() {
    let value = serde_json::json!({"axes":["valence"],"budget":task(),"policy":{"minimum_peak_weight_ppm":0}});
    assert!(serde_json::from_value::<SentimentBatchArgs>(value).is_err());
    let value = serde_json::json!({"axes":["valence"],"budget":task(),"identity":{}});
    assert!(serde_json::from_value::<SentimentBatchArgs>(value).is_err());
    assert!(check_args(&SentimentBatchArgs { axes: vec![], budget: task() }, task()).is_err());
    assert!(check_args(&SentimentBatchArgs { axes: vec![SentimentAxis::Valence; 2], budget: task() }, task()).is_err());
}
#[test]
fn per_document_axes_override_does_not_mutate_defaults() {
    let p = planner(); let c = Int8SentimentBatchPlanner::new(&p, config(&p)).unwrap();
    let mut one = document();
    one.task_args = Some(SentimentBatchArgs { axes: vec![SentimentAxis::Approach], budget: task() });
    let a = c.prepare_with_control(one, &mut Continue).unwrap();
    let b = prepare(&c);
    assert_eq!(a.head_count(), 1); assert_eq!(b.head_count(), 4);
    assert_ne!(a.execution_identity().taskir_digest, b.execution_identity().taskir_digest);
    let mut missing = config(&p); missing.defaults = None;
    let c = Int8SentimentBatchPlanner::new(&p, missing).unwrap();
    assert!(c.prepare_with_control(document(), &mut Continue).is_err());
}
#[test]
fn preparation_observes_the_same_controller_before_and_after_compilation() {
    let p = planner(); let c = Int8SentimentBatchPlanner::new(&p, config(&p)).unwrap();
    for calls in [0, 2, 5] {
        let e = c.prepare_with_control(document(), &mut StopAfter(calls)).err().unwrap();
        assert!(e.stop); assert_eq!(e.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    }
}
#[test]
fn every_task_budget_axis_is_bounded_by_the_run_ceiling() {
    for axis in 0..5 {
        let mut b = task();
        match axis { 0 => b.max_input_tokens += 1, 1 => b.max_output_tokens += 1,
            2 => b.max_output_bytes += 1, 3 => b.max_grammar_states += 1, _ => b.max_kv_bytes += 1 }
        assert!(check_args(&SentimentBatchArgs { axes: vec![SentimentAxis::Valence], budget: b }, task()).is_err());
    }
}
fn set_axis(w: &mut Int8Work, axis: usize, value: u64) {
    match axis { 0 => w.forward_positions = value, 1 => w.projected_logits = value,
        2 => w.attention_pairs = value, 3 => w.projections.dot_products = value,
        _ => w.projections.multiply_accumulates = value }
}
fn axis(w: Int8Work, i: usize) -> u64 {
    [w.forward_positions, w.projected_logits, w.attention_pairs,
        w.projections.dot_products, w.projections.multiply_accumulates][i]
}
#[test]
fn all_five_per_item_work_caps_are_enforced_before_native_execution() {
    let p = planner(); let original = Int8SentimentBatchPlanner::new(&p, config(&p)).unwrap();
    let work = prepare(&original).model_work();
    for i in 0..5 {
        let mut config = config(&p); set_axis(&mut config.max_item_work, i, axis(work, i) - 1);
        let c = Int8SentimentBatchPlanner::new(&p, config).unwrap();
        let e = c.prepare_with_control(document(), &mut Continue).err().unwrap();
        assert!(!e.stop); assert_eq!(e.fault.code, BatchCode::WorkLimit);
    }
}
#[test]
fn failed_subtraction_is_atomic_and_no_epoch_can_refill_work() {
    let one = Int8Work { forward_positions: 3, projected_logits: 4, attention_pairs: 5,
        projections: ProjectionWork { dot_products: 6, multiply_accumulates: 7 } };
    for i in 0..5 {
        let mut too_large = one; set_axis(&mut too_large, i, axis(one, i) + 1);
        let mut run = RunState::new(one);
        assert!(run.begin(too_large, context(1, 1)).is_err());
        assert_eq!(run.remaining, one); assert!(run.failed);
    }
    let mut run = RunState::new(one.checked_add(one).unwrap());
    run.begin(one, context(1, 1)).unwrap(); run.failed = false;
    run.begin(one, context(5, 9)).unwrap(); run.failed = false;
    assert_eq!(run.remaining, Int8Work::default());
    assert!(run.begin(one, context(6, 10)).is_err());
}
#[test]
fn stale_delivery_coordinates_are_not_new_execution_authority() {
    let mut run = RunState::new(generous());
    run.begin(Int8Work::default(), context(4, 1)).unwrap(); run.failed = false;
    assert!(run.begin(Int8Work::default(), context(4, 2)).is_err());
    assert!(run.failed);
}

struct Uniform;
impl SentimentLogits for Uniform {
    type Error = &'static str;
    fn project(&mut self, _: SentimentAxis, _: &[crate::tasks::ir::PromptSegment], _: &[u32],
        rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let width = match rows { ProjectionRows::FullVocabulary { vocabulary_size } => vocabulary_size,
            ProjectionRows::Selected(rows) => rows.len() };
        Ok(vec![0.0; width])
    }
}
fn synthetic(c: &Int8SentimentBatchPlanner<'_>, prepared: &PreparedInt8BatchSentiment) -> Int8SentimentRun {
    let request = SentimentRequest { document: document().text, axes: prepared.axes.clone(), budget: task() };
    let context = PlanContext::new(&c.config.identity, task()).unwrap();
    let plan = c.planner.plan_for_profile(&request, &context, c.config.planning,
        NumericsProfile::StrictQuantized { version: 1 }).unwrap();
    Int8SentimentRun { schema_version: 1, execution: INT8_SENTIMENT_EXECUTION.to_owned(),
        numerics_profile: STRICT_INT8_PROFILE.to_owned(), result: plan.execute(&mut Uniform).unwrap(),
        head_count: prepared.head_count(), model_work: prepared.model_work(), rewound_positions: 0 }
}
struct Fixture { output: Int8SentimentRun, calls: usize, capacity: usize, clean: bool, fail: bool }
impl Driver for Fixture {
    fn clean(&self) -> bool { self.clean }
    fn capacity(&self) -> usize { self.capacity }
    fn execute<C: DecodeStepControl>(&mut self, p: &PreparedInt8Sentiment, _: &ExecutionIdentity,
        budget: Int8ScoringBudget, _: &mut C) -> Result<Int8SentimentRun, Int8SentimentError> {
        self.calls += 1;
        assert_eq!(budget.native.max_forward_positions, p.planned_work().forward_positions);
        assert_eq!(budget.native.max_attention_pairs, p.planned_work().attention_pairs);
        if self.fail { self.clean = false; return Err(Int8SentimentError::Native(StrictInt8Error::Attention)); }
        Ok(self.output.clone())
    }
}
struct Guard(Arc<AtomicUsize>);
impl Drop for Guard { fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); } }
struct Admission { drops: Arc<AtomicUsize>, wrong_identity: bool }
impl Int8SentimentAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, r: Int8SentimentAdmissionRequest<'_>) -> Result<(ExecutionIdentity, Guard), BatchItemFailure> {
        assert_eq!(r.kv_reservation_bytes, 4096 * KV_BYTES_PER_TOKEN as u64);
        assert_eq!(r.max_result_bytes, 100000);
        let mut id = r.identity.clone();
        if self.wrong_identity { id.prompt_digest = Sha256Digest::of_bytes(b"wrong"); }
        Ok((id, Guard(Arc::clone(&self.drops))))
    }
}
fn driver(c: &Int8SentimentBatchPlanner<'_>, prepared: &PreparedInt8BatchSentiment) -> Fixture {
    Fixture { output: synthetic(c, prepared), calls: 0, capacity: 4096, clean: true, fail: false }
}
fn admission() -> Admission { Admission { drops: Arc::new(AtomicUsize::new(0)), wrong_identity: false } }
#[test]
fn complete_bundle_keeps_its_admission_guard_until_delivery_owner_drops() {
    let p = planner(); let c = Int8SentimentBatchPlanner::new(&p, config(&p)).unwrap();
    let prepared = prepare(&c); let work = prepared.model_work();
    let mut d = driver(&c, &prepared); let mut a = admission(); let mut state = RunState::new(work);
    let output = execute_admitted(prepared, c.binding, &mut d, &mut a, &mut state, context(1, 1), &mut Continue).unwrap();
    assert_eq!(output.result().head_count, 4); assert_eq!(d.calls, 1);
    assert_eq!(a.drops.load(Ordering::SeqCst), 0); assert!(!state.failed);
    assert_eq!(state.remaining, Int8Work::default());
    drop(output); assert_eq!(a.drops.load(Ordering::SeqCst), 1);
}
#[test]
fn admission_identity_mismatch_never_calls_native_or_refunds_reserved_work() {
    let p = planner(); let c = Int8SentimentBatchPlanner::new(&p, config(&p)).unwrap();
    let prepared = prepare(&c); let mut state = RunState::new(prepared.model_work());
    let mut d = driver(&c, &prepared); let mut a = admission(); a.wrong_identity = true;
    let e = execute_admitted(prepared, c.binding, &mut d, &mut a, &mut state, context(1, 1), &mut Continue).err().unwrap();
    assert!(e.stop); assert_eq!(e.fault.code, BatchCode::Admission); assert_eq!(d.calls, 0);
    assert_eq!(a.drops.load(Ordering::SeqCst), 1); assert_eq!(state.remaining, Int8Work::default());
}
#[test]
fn live_context_is_not_aggregate_forward_work_and_full_kv_is_priced() {
    let p = planner(); let c = Int8SentimentBatchPlanner::new(&p, config(&p)).unwrap();
    let prepared = prepare(&c); let mut d = driver(&c, &prepared);
    d.capacity = prepared.plan.required_context();
    assert!(prepared.model_work().forward_positions > d.capacity as u64);
    assert!(preflight(&prepared, &d).is_ok());
    d.capacity = 4097; assert!(preflight(&prepared, &d).is_err());
    d.capacity = 1; assert!(preflight(&prepared, &d).is_err());
}
#[test]
fn incomplete_or_mislabeled_native_envelopes_poison_without_partial_results() {
    let p = planner(); let c = Int8SentimentBatchPlanner::new(&p, config(&p)).unwrap();
    let original = prepare(&c); let output = synthetic(&c, &original);
    for field in 0..5 {
        let prepared = prepare(&c); let mut state = RunState::new(prepared.model_work());
        let mut d = Fixture { output: output.clone(), calls: 0, capacity: 4096, clean: true, fail: false };
        match field { 0 => { d.output.result.dimensions.pop(); }, 1 => d.output.head_count -= 1,
            2 => d.output.model_work.attention_pairs += 1, 3 => d.output.numerics_profile = "hf-bf16-eager".to_owned(),
            _ => d.output.result.dimensions.swap(0, 1) }
        let mut a = admission();
        let e = execute_admitted(prepared, c.binding, &mut d, &mut a, &mut state, context(1, 1), &mut Continue).err().unwrap();
        assert_eq!(e.fault.code, BatchCode::InvalidExecution); assert!(state.failed);
        assert_eq!(a.drops.load(Ordering::SeqCst), 1);
    }
}
#[test]
fn late_cancellation_discards_a_completed_bundle_and_retains_its_cause() {
    let p = planner(); let c = Int8SentimentBatchPlanner::new(&p, config(&p)).unwrap();
    let prepared = prepare(&c); let mut state = RunState::new(prepared.model_work());
    let mut d = driver(&c, &prepared); let mut a = admission();
    let e = execute_admitted(prepared, c.binding, &mut d, &mut a, &mut state, context(1, 1), &mut StopAfter(2)).err().unwrap();
    assert_eq!(d.calls, 1); assert!(e.stop);
    assert_eq!(e.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    assert_eq!(a.drops.load(Ordering::SeqCst), 1); assert!(state.failed);
}
#[test]
fn native_failure_and_foreign_factory_seals_never_become_successful_abstention() {
    let p = planner(); let c = Int8SentimentBatchPlanner::new(&p, config(&p)).unwrap();
    for foreign in [false, true] {
        let prepared = prepare(&c); let mut state = RunState::new(prepared.model_work());
        let mut d = driver(&c, &prepared); d.fail = true; let mut a = admission();
        let binding = if foreign { Sha256Digest::of_bytes(b"foreign") } else { c.binding };
        let e = execute_admitted(prepared, binding, &mut d, &mut a, &mut state, context(1, 1), &mut Continue).err().unwrap();
        assert!(e.stop); assert!(state.failed); assert_eq!(d.calls, usize::from(!foreign));
    }
}
