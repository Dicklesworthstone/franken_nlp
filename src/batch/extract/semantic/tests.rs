//! Private accounting/ownership fixtures, not native inference or quality proof.
mod planning;
use super::*;
use std::{cell::Cell, panic::{catch_unwind, AssertUnwindSafe}, rc::Rc};
use serde::{Serialize, Serializer};
use crate::{
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{decode::DecodeCancellationKind, portable_int8::ProjectionWork},
};

fn work(value: u64) -> Int4Work {
    Int4Work { forward_positions: value, projected_logits: value, attention_pairs: value,
        projections: ProjectionWork { dot_products: value, multiply_accumulates: value } }
}
fn limits() -> Int4SemanticExtractionBatchLimits {
    Int4SemanticExtractionBatchLimits { max_model_work: work(40), masks: ExtractionMaskBudget {
        per_mask: MaskWorkLimits { max_trie_node_visits: 16, checkpoint_interval_nodes: 1 },
        max_visits_per_item: 3, max_visits_per_run: 6 } }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 128, max_output_tokens: 16, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn identity(task: &str) -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"private-semantic-batch-protocol-fixture");
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
struct Output { value: u64, events: Rc<Events>, refuse_serialization: bool }
impl Serialize for Output {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        assert_eq!(self.events.held.get(), 1, "output guard must cover serialization");
        if self.refuse_serialization { return Err(serde::ser::Error::custom("fixture")); }
        self.value.serialize(serializer)
    }
}
impl Drop for Output {
    fn drop(&mut self) { assert_eq!(self.events.held.get(), 1, "result storage drops before its guard"); }
}
struct FixturePlan { extraction: ExecutionIdentity, judge: ExecutionIdentity, encoding: UntrustedEncoding }
impl FixturePlan {
    fn new() -> Self { Self { extraction: identity("extract-v1"), judge: identity("judge-v1"),
        encoding: UntrustedEncoding::ByteFallback } }
}
impl Plan for FixturePlan {
    type Output = Output;
    type Completion = u64;
    fn admission(&self, kv: u64, masks: ExtractionMaskBudget) -> Int4SemanticExtractionAdmission<'_> {
        Int4SemanticExtractionAdmission { identity: &self.extraction, judge_identity: &self.judge,
            group_width: 32, model_work: work(20), extraction_reserved_work: work(5),
            mask_node_visits: masks.max_visits_per_item, mask_limits: masks.per_mask,
            kv_reservation_bytes: kv, max_result_bytes: 1 << 20, extraction_budget: budget(),
            judge_budget: budget(), semantic_limits: SemanticLimits::default(),
            judge_planning_limits: JudgeLimits::default(), source_bytes: 16, input_encoding: &self.encoding }
    }
    fn completion(&self) -> u64 { 7 }
    fn verify_completion(expected: u64, result: &Output, masks: u64) -> Result<(), BatchItemFailure> {
        if result.value != expected || masks != 3 {
            Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
        } else { Ok(()) }
    }
}
#[derive(Clone, Copy)]
enum Behavior { Complete, Refuse, Identity, Dirty, Corrupt, Panic, Serialization }
struct Admission { events: Rc<Events>, behavior: Behavior }
impl Int4SemanticExtractionBatchAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, request: Int4SemanticExtractionAdmission<'_>)
        -> Result<(ExecutionIdentity, Guard), BatchItemFailure> {
        self.events.admitted.set(self.events.admitted.get() + 1);
        assert_eq!(request.model_work, work(20));
        assert_eq!(request.extraction_reserved_work, work(5));
        assert_eq!(request.mask_node_visits, 3);
        assert!(request.kv_reservation_bytes > 0);
        match self.behavior {
            Behavior::Refuse => return Err(BatchItemFailure::reject(BatchCode::Admission)),
            Behavior::Panic => panic!("injected admission unwind"),
            _ => {}
        }
        let mut identity = request.identity.clone();
        if matches!(self.behavior, Behavior::Identity) { identity.prompt_digest = Sha256Digest::of_bytes(b"changed"); }
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
        assert_eq!(self.events.held.get(), 1);
        assert_eq!(masks.max_visits_per_item, 3);
        self.events.executed.set(self.events.executed.get() + 1);
        checkpoint(control)?;
        match self.behavior {
            Behavior::Refuse => return Err(BatchItemFailure::reject(BatchCode::Execution)),
            Behavior::Panic => panic!("injected native driver unwind"),
            Behavior::Dirty => self.dirty = true,
            _ => {}
        }
        Ok(Output { value: if matches!(self.behavior, Behavior::Corrupt) { 8 } else { 7 },
            events: Rc::clone(&self.events), refuse_serialization: matches!(self.behavior, Behavior::Serialization) })
    }
}
#[derive(Default)]
struct Control { calls: usize, stop: Option<(usize, DecodeCancellationKind)> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        self.stop.and_then(|(at, cause)| (self.calls >= at).then_some(cause))
    }
}
fn fixture(admit: Behavior, native: Behavior) -> (RunState, FixtureDriver, Admission, Arc<()>, Rc<Events>) {
    let events = Rc::new(Events::default());
    (RunState::new(limits()), FixtureDriver { events: Rc::clone(&events), behavior: native, dirty: false },
        Admission { events: Rc::clone(&events), behavior: admit }, Arc::new(()), events)
}
fn failure<T>(result: Result<T, BatchItemFailure>) -> BatchItemFailure {
    match result { Err(failure) => failure, Ok(_) => panic!("expected complete-item refusal") }
}
fn run(run: &mut RunState, driver: &mut FixtureDriver, admission: &mut Admission,
    seal: &Arc<()>, sequence: u64, epoch: u64, control: &mut Control)
    -> Result<GuardedOutput<Output, Guard>, BatchItemFailure> {
    execute(FixturePlan::new(), Arc::clone(seal), seal, driver, admission, run,
        context(sequence, epoch), control)
}

#[test]
fn every_native_and_mask_axis_is_checked_and_overflow_is_atomic() {
    for axis in 0..6 {
        let mut one_short = limits();
        match axis {
            0 => one_short.max_model_work.forward_positions = 19,
            1 => one_short.max_model_work.projected_logits = 19,
            2 => one_short.max_model_work.attention_pairs = 19,
            3 => one_short.max_model_work.projections.dot_products = 19,
            4 => one_short.max_model_work.projections.multiply_accumulates = 19,
            _ => one_short.masks.max_visits_per_run = 2,
        }
        assert!(one_short.validate_item(work(20)).is_err());
        let mut state = RunState::new(one_short);
        assert_eq!(failure(state.begin(work(20), context(1, 1))).fault.code, BatchCode::WorkLimit);
        assert_eq!(state.reserved, Int4Work::default());
        assert_eq!((state.masks, state.last_sequence), (0, 0));
        assert!(state.failed);
    }
    let mut state = RunState::new(Int4SemanticExtractionBatchLimits {
        max_model_work: work(u64::MAX), masks: ExtractionMaskBudget {
            max_visits_per_run: u64::MAX, ..limits().masks } });
    state.reserved = work(u64::MAX - 1);
    let before = state.reserved;
    assert_eq!(failure(state.begin(work(20), context(1, 1))).fault.code, BatchCode::WorkLimit);
    assert_eq!(state.reserved, before);
    assert_eq!(state.masks, 0);
    state.failed = false; state.reserved = Int4Work::default(); state.masks = u64::MAX - 1;
    assert_eq!(failure(state.begin(work(20), context(1, 1))).fault.code, BatchCode::WorkLimit);
    assert_eq!(state.reserved, Int4Work::default());
    assert_eq!(state.masks, u64::MAX - 1);
}

#[test]
fn complete_document_ceiling_is_spent_without_early_completion_or_epoch_refunds() {
    let (mut state, mut driver, mut admission, seal, events) = fixture(Behavior::Complete, Behavior::Complete);
    for (sequence, epoch) in [(1, 1), (2, 17)] {
        let output = run(&mut state, &mut driver, &mut admission, &seal, sequence, epoch, &mut Control::default()).unwrap();
        assert_eq!(serde_json::to_vec(&output).unwrap(), b"7");
        assert_eq!(state.reserved, work(sequence * 20));
        assert_eq!(state.masks, sequence * 3);
        assert!(!state.failed); assert_eq!(events.held.get(), 1);
        drop(output); assert_eq!(events.held.get(), 0);
    }
    let error = failure(run(&mut state, &mut driver, &mut admission, &seal, 3, 18, &mut Control::default()));
    assert_eq!(error.fault.code, BatchCode::WorkLimit); assert!(error.stop);
    assert_eq!(events.executed.get(), 2); assert_eq!(events.admitted.get(), 2);
    assert_eq!(state.reserved, work(40)); assert_eq!(state.masks, 6);
}

#[test]
fn refused_or_panicking_admission_spends_the_attempt_before_any_native_call() {
    for behavior in [Behavior::Refuse, Behavior::Identity, Behavior::Panic] {
        let (mut state, mut driver, mut admission, seal, events) = fixture(behavior, Behavior::Complete);
        let outcome = catch_unwind(AssertUnwindSafe(||
            run(&mut state, &mut driver, &mut admission, &seal, 1, 1, &mut Control::default())));
        if matches!(behavior, Behavior::Panic) { assert!(outcome.is_err()); }
        else { assert!(failure(outcome.unwrap()).stop); }
        assert_eq!(state.reserved, work(20)); assert_eq!(state.masks, 3);
        assert_eq!(events.admitted.get(), 1); assert_eq!(events.executed.get(), 0);
        assert_eq!(events.held.get(), 0); assert!(state.failed);
        assert_eq!(failure(run(&mut state, &mut driver, &mut admission, &seal, 2, 1, &mut Control::default())).fault.code,
            BatchCode::InvalidExecution);
    }
}

#[test]
fn failed_dirty_corrupted_or_unwinding_native_completion_never_reopens_the_adapter() {
    for behavior in [Behavior::Refuse, Behavior::Dirty, Behavior::Corrupt, Behavior::Panic] {
        let (mut state, mut driver, mut admission, seal, events) = fixture(Behavior::Complete, behavior);
        let outcome = catch_unwind(AssertUnwindSafe(||
            run(&mut state, &mut driver, &mut admission, &seal, 1, 1, &mut Control::default())));
        if matches!(behavior, Behavior::Panic) { assert!(outcome.is_err()); }
        else { assert!(failure(outcome.unwrap()).stop); }
        assert_eq!(state.reserved, work(20)); assert_eq!(state.masks, 3);
        assert_eq!(events.executed.get(), 1); assert_eq!(events.held.get(), 0);
        assert_eq!(events.dropped.get(), 1); assert!(state.failed);
    }
}

#[test]
fn every_original_cancellation_cause_survives_admission_native_and_final_boundaries() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for at in 1..=4 {
            let (mut state, mut driver, mut admission, seal, events) = fixture(Behavior::Complete, Behavior::Complete);
            let error = failure(run(&mut state, &mut driver, &mut admission, &seal, 1, 1,
                &mut Control { calls: 0, stop: Some((at, cause)) }));
            assert_eq!(error.fault.cancellation, Some(cause)); assert!(error.stop);
            assert_eq!(state.reserved, work(20)); assert_eq!(state.masks, 3);
            assert_eq!(events.held.get(), 0); assert!(state.failed);
            assert_eq!(events.executed.get(), usize::from(at >= 3));
        }
    }
}

#[test]
fn prepared_owner_and_delivery_sequence_cannot_be_replaced() {
    let (mut state, mut driver, mut admission, seal, events) = fixture(Behavior::Complete, Behavior::Complete);
    let error = failure(execute(FixturePlan::new(), Arc::new(()), &seal, &mut driver, &mut admission,
        &mut state, context(1, 1), &mut Control::default()));
    assert_eq!(error.fault.code, BatchCode::Admission); assert_eq!(state.reserved, Int4Work::default());
    assert_eq!(events.admitted.get(), 0);
    for replacement in [context(1, 1), context(2, 0), BatchRequestContext {
        request_seq: 2, epoch: 1, input_line: 0, byte_offset: 0 }] {
        let (mut state, mut driver, mut admission, seal, events) = fixture(Behavior::Complete, Behavior::Complete);
        drop(run(&mut state, &mut driver, &mut admission, &seal, 1, 1, &mut Control::default()).unwrap());
        let error = failure(execute(FixturePlan::new(), Arc::clone(&seal), &seal, &mut driver, &mut admission,
            &mut state, replacement, &mut Control::default()));
        assert_eq!(error.fault.code, BatchCode::InvalidExecution);
        assert_eq!(state.reserved, work(20)); assert_eq!(events.executed.get(), 1);
    }
}

#[test]
fn output_guard_survives_failed_serialization_until_the_complete_result_is_dropped() {
    let (mut state, mut driver, mut admission, seal, events) = fixture(Behavior::Complete, Behavior::Serialization);
    let output = run(&mut state, &mut driver, &mut admission, &seal, 1, 1, &mut Control::default()).unwrap();
    assert!(serde_json::to_vec(&output).is_err());
    assert_eq!(events.held.get(), 1); assert_eq!(events.dropped.get(), 0);
    drop(output);
    assert_eq!(events.held.get(), 0); assert_eq!(events.dropped.get(), 1);
    assert_eq!(state.reserved, work(20)); assert_eq!(state.masks, 3);
}

