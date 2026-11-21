//! Protocol/accounting fixtures only; no native inference or quality claim.
mod planning;
use super::*;
use std::{cell::Cell, panic::{catch_unwind, AssertUnwindSafe}, rc::Rc};
use serde::{Serialize, Serializer};
use crate::{execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{decode::DecodeCancellationKind, portable_int8::ProjectionWork}};

fn work(value: u64) -> Int4Work {
    Int4Work { forward_positions: value, projected_logits: value, attention_pairs: value,
        projections: ProjectionWork { dot_products: value, multiply_accumulates: value } }
}
fn limits() -> Int4SemanticSourceBatchLimits {
    Int4SemanticSourceBatchLimits { max_model_work: work(40), masks: ExtractionMaskBudget {
        per_mask: MaskWorkLimits { max_trie_node_visits: 16, checkpoint_interval_nodes: 1 },
        max_visits_per_item: 3, max_visits_per_run: 6 } }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 128, max_output_tokens: 16, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn identity(task: &str) -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"private-semantic-source-protocol-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(),
        logical_model_digest: digest, artifact_format: "fixture".to_owned(), quant_recipe: "fixture".to_owned(),
        packing_set_digest: digest, tokenizer_digest: digest, template_digest: digest,
        task_spec: task.to_owned(), taskir_digest: digest, prompt_digest: digest,
        grammar_compiler_version: "fixture".to_owned(), schema_digest: digest,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: digest, decision_policy_digest: digest,
        backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None }
}
fn context(sequence: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: sequence, epoch, input_line: sequence, byte_offset: 0 }
}
#[derive(Default)]
struct Events { held: Cell<usize>, admitted: Cell<usize>, executed: Cell<usize>, dropped: Cell<usize> }
struct Guard(Rc<Events>);
impl Drop for Guard {
    fn drop(&mut self) {
        self.0.held.set(self.0.held.get() - 1);
        self.0.dropped.set(self.0.dropped.get() + 1);
    }
}
struct Output { valid: bool, events: Rc<Events>, fail_serialization: bool }
impl Serialize for Output {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        assert_eq!(self.events.held.get(), 1);
        if self.fail_serialization { return Err(serde::ser::Error::custom("fixture")); }
        self.valid.serialize(serializer)
    }
}
impl Drop for Output {
    fn drop(&mut self) { assert_eq!(self.events.held.get(), 1, "result storage must drop before its guard"); }
}
struct FixturePlan { source: ExecutionIdentity, judge: ExecutionIdentity, encoding: UntrustedEncoding }
impl FixturePlan {
    fn new() -> Self { Self { source: identity("summarize-v1"), judge: identity("judge-v1"),
        encoding: UntrustedEncoding::ByteFallback } }
}
impl Plan for FixturePlan {
    type Output = Output;
    type Completion = ();
    fn admission(&self, kv: u64, masks: ExtractionMaskBudget) -> Int4SemanticSourceAdmission<'_> {
        Int4SemanticSourceAdmission { identity: &self.source, judge_identity: &self.judge,
            group_width: 32, model_work: work(20), source_reserved_work: work(5),
            mask_node_visits: masks.max_visits_per_item, mask_limits: masks.per_mask,
            kv_reservation_bytes: kv, max_result_bytes: 1 << 20, source_budget: budget(), judge_budget: budget(),
            semantic_limits: SourceSemanticLimits::default(), judge_planning_limits: JudgeLimits::default(),
            input_facts: SourceSemanticInputFacts { source_bytes: 16, question_bytes: 0, passage_count: 0 },
            input_encoding: &self.encoding }
    }
    fn completion(&self) {}
    fn verify_completion(_: (), output: &Output, masks: u64) -> Result<(), BatchItemFailure> {
        if output.valid && masks == 3 { Ok(()) } else { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) }
    }
}
#[derive(Clone, Copy)]
enum Behavior { Complete, Refuse, Identity, Dirty, Corrupt, Panic, Serialization }
struct Admission { events: Rc<Events>, behavior: Behavior }
impl Int4SemanticSourceBatchAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, request: Int4SemanticSourceAdmission<'_>) -> Result<(ExecutionIdentity, Guard), BatchItemFailure> {
        self.events.admitted.set(self.events.admitted.get() + 1);
        assert_eq!(request.model_work, work(20));
        assert_eq!(request.source_reserved_work, work(5));
        assert_eq!(request.mask_node_visits, 3);
        assert!(request.kv_reservation_bytes > 0);
        match self.behavior {
            Behavior::Refuse => return Err(BatchItemFailure::reject(BatchCode::Admission)),
            Behavior::Panic => panic!("injected admission panic"), _ => {}
        }
        let mut identity = request.identity.clone();
        if matches!(self.behavior, Behavior::Identity) { identity.prompt_digest = Sha256Digest::of_bytes(b"replacement"); }
        self.events.held.set(self.events.held.get() + 1);
        Ok((identity, Guard(Rc::clone(&self.events))))
    }
}
struct FixtureDriver { events: Rc<Events>, behavior: Behavior, dirty: bool }
impl Driver<FixturePlan> for FixtureDriver {
    fn clean(&self) -> bool { !self.dirty }
    fn capacity(&self) -> usize { 128 }
    fn group_width(&self) -> usize { 32 }
    fn execute<C: DecodeStepControl>(&mut self, _: FixturePlan, masks: ExtractionMaskBudget, control: &mut C)
        -> Result<Output, BatchItemFailure> {
        assert_eq!(self.events.held.get(), 1); assert_eq!(masks.max_visits_per_item, 3);
        self.events.executed.set(self.events.executed.get() + 1);
        checkpoint(control)?;
        match self.behavior {
            Behavior::Refuse => return Err(BatchItemFailure::reject(BatchCode::Execution)),
            Behavior::Panic => panic!("injected native panic"),
            Behavior::Dirty => self.dirty = true, _ => {}
        }
        Ok(Output { valid: !matches!(self.behavior, Behavior::Corrupt), events: Rc::clone(&self.events),
            fail_serialization: matches!(self.behavior, Behavior::Serialization) })
    }
}
#[derive(Default)]
struct Control { calls: usize, stop: Option<(usize, DecodeCancellationKind)> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1; self.stop.and_then(|(at, cause)| (self.calls >= at).then_some(cause))
    }
}
fn fixture(admit: Behavior, native: Behavior) -> (RunState, FixtureDriver, Admission, Arc<()>, Rc<Events>) {
    let events = Rc::new(Events::default());
    (RunState::new(limits()), FixtureDriver { events: Rc::clone(&events), behavior: native, dirty: false },
        Admission { events: Rc::clone(&events), behavior: admit }, Arc::new(()), events)
}
fn failure<T>(result: Result<T, BatchItemFailure>) -> BatchItemFailure {
    match result { Err(error) => error, Ok(_) => panic!("expected refusal") }
}
fn run(state: &mut RunState, driver: &mut FixtureDriver, admission: &mut Admission,
    seal: &Arc<()>, sequence: u64, epoch: u64, control: &mut Control) -> Result<GuardedOutput<Output, Guard>, BatchItemFailure> {
    execute(FixturePlan::new(), Arc::clone(seal), seal, driver, admission, state, context(sequence, epoch), control)
}

#[test]
fn all_six_caps_and_overflow_are_atomic_without_partial_debits() {
    for axis in 0..6 {
        let mut short = limits();
        match axis {
            0 => short.max_model_work.forward_positions = 19, 1 => short.max_model_work.projected_logits = 19,
            2 => short.max_model_work.attention_pairs = 19, 3 => short.max_model_work.projections.dot_products = 19,
            4 => short.max_model_work.projections.multiply_accumulates = 19, _ => short.masks.max_visits_per_run = 2,
        }
        assert!(short.validate_item(work(20)).is_err());
        let mut state = RunState::new(short);
        assert_eq!(failure(state.begin(work(20), context(1, 1))).fault.code, BatchCode::WorkLimit);
        assert_eq!(state.reserved, Int4Work::default()); assert_eq!((state.masks, state.last_sequence), (0, 0));
        assert!(state.failed);
    }
    for masks_overflow in [false, true] {
        let mut state = RunState::new(Int4SemanticSourceBatchLimits { max_model_work: work(u64::MAX),
            masks: ExtractionMaskBudget { max_visits_per_run: u64::MAX, ..limits().masks } });
        if masks_overflow { state.masks = u64::MAX - 1; } else { state.reserved = work(u64::MAX - 1); }
        let before = (state.reserved, state.masks);
        assert_eq!(failure(state.begin(work(20), context(1, 1))).fault.code, BatchCode::WorkLimit);
        assert_eq!((state.reserved, state.masks), before);
    }
}

#[test]
fn complete_attempts_charge_declared_review_work_and_flush_never_renews_it() {
    let (mut state, mut driver, mut admission, seal, events) = fixture(Behavior::Complete, Behavior::Complete);
    for (sequence, epoch) in [(1, 1), (2, 17)] {
        let output = run(&mut state, &mut driver, &mut admission, &seal, sequence, epoch, &mut Control::default()).unwrap();
        assert_eq!(state.reserved, work(sequence * 20)); assert_eq!(state.masks, sequence * 3);
        assert!(!state.failed); assert_eq!(events.held.get(), 1); drop(output);
    }
    assert_eq!(failure(run(&mut state, &mut driver, &mut admission, &seal, 3, 18,
        &mut Control::default())).fault.code, BatchCode::WorkLimit);
    assert_eq!(events.executed.get(), 2); assert_eq!(state.reserved, work(40));
    assert_eq!(events.dropped.get(), 2); assert_eq!(events.held.get(), 0); assert!(state.failed);
}

#[test]
fn refusal_identity_corruption_dirty_state_and_panics_fail_stop_without_refunds() {
    for (admit, native) in [(Behavior::Refuse, Behavior::Complete), (Behavior::Identity, Behavior::Complete),
        (Behavior::Panic, Behavior::Complete), (Behavior::Complete, Behavior::Refuse),
        (Behavior::Complete, Behavior::Dirty), (Behavior::Complete, Behavior::Corrupt), (Behavior::Complete, Behavior::Panic)] {
        let (mut state, mut driver, mut admission, seal, events) = fixture(admit, native);
        let result = catch_unwind(AssertUnwindSafe(|| run(&mut state, &mut driver, &mut admission,
            &seal, 1, 1, &mut Control::default())));
        if let Ok(result) = result { assert!(failure(result).stop); }
        assert!(state.failed); assert_eq!(state.reserved, work(20)); assert_eq!(state.masks, 3);
        assert_eq!(events.held.get(), 0);
        let calls = events.executed.get();
        assert_eq!(failure(run(&mut state, &mut driver, &mut admission, &seal, 2, 2,
            &mut Control::default())).fault.code, BatchCode::InvalidExecution);
        assert_eq!(events.executed.get(), calls); assert_eq!(state.reserved, work(20));
    }
}

#[test]
fn all_original_cancellation_causes_survive_each_execution_boundary() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for at in 1..=4 {
            let (mut state, mut driver, mut admission, seal, events) = fixture(Behavior::Complete, Behavior::Complete);
            let error = failure(run(&mut state, &mut driver, &mut admission, &seal, 1, 1,
                &mut Control { calls: 0, stop: Some((at, cause)) }));
            assert_eq!(error.fault.cancellation, Some(cause)); assert!(error.stop);
            assert!(state.failed); assert_eq!(state.reserved, work(20)); assert_eq!(state.masks, 3);
            assert_eq!(events.held.get(), 0);
        }
    }
}

#[test]
fn owner_replay_and_invalid_delivery_context_cannot_reenter_execution() {
    let (mut state, mut driver, mut admission, seal, events) = fixture(Behavior::Complete, Behavior::Complete);
    let error = failure(execute(FixturePlan::new(), Arc::new(()), &seal, &mut driver,
        &mut admission, &mut state, context(1, 1), &mut Control::default()));
    assert_eq!(error.fault.code, BatchCode::Admission); assert_eq!(state.reserved, Int4Work::default());
    assert_eq!(events.executed.get(), 0);
    for invalid in [context(0, 1), context(1, 0), BatchRequestContext { input_line: 0, ..context(1, 1) }] {
        let mut state = RunState::new(limits());
        assert_eq!(failure(state.begin(work(20), invalid)).fault.code, BatchCode::InvalidExecution);
        assert_eq!(state.reserved, Int4Work::default());
    }
    let mut state = RunState::new(limits()); state.begin(work(20), context(2, 1)).unwrap(); state.failed = false;
    assert_eq!(failure(state.begin(work(20), context(2, 99))).fault.code, BatchCode::InvalidExecution);
    assert_eq!(state.reserved, work(20));
}

#[test]
fn result_guard_survives_failed_serialization_and_drops_after_result_storage() {
    let (mut state, mut driver, mut admission, seal, events) = fixture(Behavior::Complete, Behavior::Serialization);
    let output = run(&mut state, &mut driver, &mut admission, &seal, 1, 1, &mut Control::default()).unwrap();
    assert!(serde_json::to_vec(&output).is_err()); assert_eq!(events.held.get(), 1);
    drop(output); assert_eq!(events.held.get(), 0); assert_eq!(events.dropped.get(), 1);
}
