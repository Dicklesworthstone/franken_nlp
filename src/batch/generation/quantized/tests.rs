//! Private-driver lifecycle tests plus real pinned request compilation.
//! No synthetic driver is exposed by the public native-batch constructor.
use super::*;
use std::{cell::{Cell, RefCell}, io::{self, Cursor, Write}, rc::Rc};
use crate::{batch::{self, BatchLimits},
    native_engine::{decode::DecodeCancellationKind, portable_int8::ProjectionWork},
    tasks::chat::quantized::tests::{fixture, request, task_budget, completed_result}};

fn args(eos: u32) -> GenerationBatchArgs {
    GenerationBatchArgs::Generate { generation: request(eos).generation, budget: task_budget(), sample_index: 0 }
}
fn document(id: &str) -> BatchDocument<GenerationBatchArgs> {
    BatchDocument { id: id.to_owned(), text: "private input <think>".to_owned(), task_args: None }
}
fn context(seq: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: seq, epoch, input_line: seq, byte_offset: 0 }
}
fn room() -> Int8Work {
    Int8Work { forward_positions: u64::MAX, projected_logits: u64::MAX, attention_pairs: u64::MAX,
        projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } }
}
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
#[derive(Default)]
struct State {
    live: Cell<bool>, running: Cell<bool>, dropped: Cell<usize>,
    admitted: RefCell<Option<(Int8Work, u64, u64)>>,
}
struct Guard(Rc<State>);
impl Drop for Guard {
    fn drop(&mut self) {
        assert!(!self.0.running.get(), "admission released before physical completion");
        assert!(self.0.live.replace(false)); self.0.dropped.set(self.0.dropped.get() + 1);
    }
}
struct Admission { state: Rc<State>, reject: bool, change_identity: bool, calls: usize }
impl Admission {
    fn new(state: &Rc<State>) -> Self { Self { state: Rc::clone(state), reject: false, change_identity: false, calls: 0 } }
}
impl Int8GenerationBatchAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, req: Int8GenerationAdmission<'_>) -> Result<(ExecutionIdentity, Guard), BatchItemFailure> {
        self.calls += 1;
        if self.reject { return Err(BatchItemFailure::reject(BatchCode::Admission)); }
        assert!(!self.state.live.replace(true));
        *self.state.admitted.borrow_mut() = Some((req.model_work, req.kv_reservation_bytes, req.sampler_bytes));
        let mut identity = req.identity.clone();
        if self.change_identity { identity.source_revision.push('x'); }
        Ok((identity, Guard(Rc::clone(&self.state))))
    }
}
#[derive(Clone, Copy)]
enum Mode { Success, Cancel, Panic, WorkFailure, DirtySuccess, PoisonedSuccess, WrongSequence, WrongWork }
struct Engine { state: Rc<State>, mode: Mode, calls: usize, empty: bool, poisoned: bool }
impl Engine {
    fn new(state: &Rc<State>) -> Self { Self { state: Rc::clone(state), mode: Mode::Success, calls: 0, empty: true, poisoned: false } }
}
struct PhysicalCall(Rc<State>);
impl Drop for PhysicalCall { fn drop(&mut self) { self.0.running.set(false); } }
impl Driver for Engine {
    fn capacity(&self) -> usize { 512 }
    fn empty(&self) -> bool { self.empty }
    fn poisoned(&self) -> bool { self.poisoned }
    fn preflight(&self, prepared: &PreparedInt8Chat, identity: &ExecutionIdentity, budget: Int8GenerationBudget)
        -> Result<(), Int8ChatError> {
        prepared.verify_identity(identity)?;
        if self.poisoned || !self.empty { return Err(Int8GenerationError::Native(StrictInt8Error::EngineUnavailable).into()); }
        if prepared.planned_work().forward_positions > self.capacity() as u64 {
            return Err(Int8GenerationError::Native(StrictInt8Error::Context).into());
        }
        if budget.max_kv_bytes < self.capacity() as u64 * KV_BYTES_PER_TOKEN as u64
            || budget.max_sampler_bytes < prepared.native_plan().sampler_bytes() {
            return Err(Int8GenerationError::Native(StrictInt8Error::Memory).into());
        }
        Ok(())
    }
    fn run<C: DecodeStepControl>(&mut self, prepared: &PreparedInt8Chat, identity: &ExecutionIdentity,
        seq: u64, _: Int8GenerationBudget, _: &mut C) -> Result<Int8ChatResult, Int8ChatError> {
        prepared.verify_identity(identity)?;
        assert!(self.state.live.get()); assert!(!self.state.running.replace(true));
        let _physical = PhysicalCall(Rc::clone(&self.state)); self.calls += 1;
        match self.mode {
            Mode::Panic => panic!("private simulated native panic"),
            Mode::Cancel => { self.poisoned = true; return Err(Int8GenerationError::Native(
                StrictInt8Error::Cancelled(DecodeCancellationKind::Deadline)).into()); }
            Mode::WorkFailure => { self.poisoned = true; return Err(Int8GenerationError::Native(StrictInt8Error::Work).into()); }
            Mode::DirtySuccess => self.empty = false,
            Mode::PoisonedSuccess => self.poisoned = true,
            _ => {}
        }
        let mut output = completed_result(prepared, seq);
        match self.mode { Mode::WrongSequence => output.result.request_seq += 1,
            Mode::WrongWork => output.model_work.projections.multiply_accumulates += 1, _ => {} }
        Ok(output)
    }
}

#[test]
fn defaults_and_item_overrides_compile_real_int8_plans_without_state_leak() {
    let (planner, eos) = fixture(); let compiler = Int8GenerationBatchPlanner::new(&planner, Some(args(eos))).unwrap();
    let base = compiler.prepare(document("same")).unwrap();
    let mut changed = document("same"); let mut options = request(eos).generation; options.max_new_tokens = 3;
    changed.task_args = Some(GenerationBatchArgs::Generate { generation: options, budget: task_budget(), sample_index: 7 });
    let override_plan = compiler.prepare(changed).unwrap();
    assert!(base.verify_identity(override_plan.execution_identity()).is_err());
    assert!(base.verify_identity(compiler.prepare(document("same")).unwrap().execution_identity()).is_ok());
    assert!(Int8GenerationBatchPlanner::new(&planner, None).unwrap().prepare(document("missing")).is_err());
    assert_eq!(base.execution_identity().numerics_profile, crate::execution_identity::NumericsProfile::StrictQuantized { version: 1 });
}

#[test]
fn chat_batch_appends_the_exact_last_user_and_keeps_generate_identity_separate() {
    let (planner, eos) = fixture();
    let history = vec![ChatMessage { role: ChatRole::User, content: "earlier".to_owned() },
        ChatMessage { role: ChatRole::Assistant, content: "prior response".to_owned() }];
    let generation = request(eos).generation;
    let compiler = Int8GenerationBatchPlanner::new(&planner, Some(GenerationBatchArgs::Chat {
        history: history.clone(), generation: generation.clone(), budget: task_budget(), sample_index: 0 })).unwrap();
    let plan = compiler.prepare(document("x")).unwrap();
    let mut messages = history; messages.push(ChatMessage { role: ChatRole::User, content: document("x").text });
    let direct = planner.plan_chat(&ChatRequest { item_id: "x".to_owned(), sample_index: 0, messages, generation, budget: task_budget() }).unwrap();
    plan.verify_identity(direct.execution_identity()).unwrap();
    assert_eq!(plan.execution_identity().task_spec, "chat-v1");
}

#[test]
fn malformed_arguments_and_oversized_default_history_are_refused() {
    let (planner, eos) = fixture();
    let mut invalid = serde_json::to_value(args(eos)).unwrap();
    invalid["command"] = serde_json::json!("never execute me");
    assert!(serde_json::from_value::<GenerationBatchArgs>(invalid).is_err());
    let bad = GenerationBatchArgs::Chat { history: vec![ChatMessage { role: ChatRole::User,
        content: "x".repeat(super::super::MAX_GENERATION_ARGUMENT_BYTES) }], generation: request(eos).generation,
        budget: task_budget(), sample_index: 0 };
    assert!(Int8GenerationBatchPlanner::new(&planner, Some(bad)).is_err());
    for mut limits in [Int8BatchLimits { max_sampler_bytes: 0, max_model_work: room() },
        Int8BatchLimits { max_sampler_bytes: 1, max_model_work: Int8Work::default() }] {
        assert!(limits.validate().is_err()); limits.max_sampler_bytes = 1; limits.max_model_work = room();
        assert!(limits.validate().is_ok());
    }
}

#[test]
fn every_complete_native_work_axis_is_admitted_and_overflow_checked() {
    let work = Int8Work { forward_positions: 2, projected_logits: 3, attention_pairs: 4,
        projections: ProjectionWork { dot_products: 5, multiply_accumulates: 6 } };
    for axis in 0..5 {
        let mut limit = work;
        match axis { 0 => limit.forward_positions -= 1, 1 => limit.projected_logits -= 1,
            2 => limit.attention_pairs -= 1, 3 => limit.projections.dot_products -= 1,
            _ => limit.projections.multiply_accumulates -= 1 }
        let mut run = RunState::new(limit);
        let error = run.begin(work, context(1, 1)).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::WorkLimit);
        assert!(run.failed); assert_eq!(run.reserved, Int8Work::default());
        let mut already = Int8Work::default();
        match axis { 0 => already.forward_positions = u64::MAX, 1 => already.projected_logits = u64::MAX,
            2 => already.attention_pairs = u64::MAX, 3 => already.projections.dot_products = u64::MAX,
            _ => already.projections.multiply_accumulates = u64::MAX }
        assert!(add_work(already, work).is_none());
    }
    let mut exact = RunState::new(work); exact.begin(work, context(1, 1)).unwrap(); assert_eq!(exact.reserved, work);
}

#[test]
fn real_admission_quantities_and_guard_survive_the_native_call_and_serialization() {
    let (planner, eos) = fixture(); let compiler = Int8GenerationBatchPlanner::new(&planner, Some(args(eos))).unwrap();
    let prepared = compiler.prepare(document("x")).unwrap(); let work = prepared.planned_work();
    let state = Rc::new(State::default()); let mut admission = Admission::new(&state); let mut engine = Engine::new(&state);
    let mut run = RunState::new(room());
    let output = execute_admitted(prepared, &mut engine, &mut admission, &mut run, u64::MAX, context(1, 1), &mut Continue).unwrap();
    assert!(state.live.get()); assert!(!state.running.get()); assert_eq!(engine.calls, 1); assert!(!run.failed);
    let admitted = state.admitted.borrow().unwrap();
    assert_eq!(admitted.0, work); assert_eq!(admitted.1, 512 * KV_BYTES_PER_TOKEN as u64); assert!(admitted.2 > 0);
    let wire = crate::canonjson::canonical_string(&output).unwrap();
    assert!(state.live.get()); assert!(!wire.contains("private input")); assert!(!wire.contains("guard"));
    assert_eq!(run.reserved, work); drop(output); assert!(!state.live.get()); assert_eq!(state.dropped.get(), 1);
}

#[test]
fn changed_admission_identity_never_reaches_a_physical_forward() {
    let (planner, eos) = fixture(); let compiler = Int8GenerationBatchPlanner::new(&planner, Some(args(eos))).unwrap();
    let state = Rc::new(State::default()); let mut admission = Admission::new(&state); admission.change_identity = true;
    let mut engine = Engine::new(&state); let mut run = RunState::new(room());
    let error = execute_admitted(compiler.prepare(document("x")).unwrap(), &mut engine, &mut admission,
        &mut run, u64::MAX, context(1, 1), &mut Continue).err().unwrap();
    assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission); assert_eq!(engine.calls, 0);
    assert!(!state.live.get()); assert_eq!(state.dropped.get(), 1); assert!(run.failed);
}

#[test]
fn rejected_admission_is_not_refunded_and_new_epoch_does_not_replenish_work() {
    let (planner, eos) = fixture(); let compiler = Int8GenerationBatchPlanner::new(&planner, Some(args(eos))).unwrap();
    let prepared = compiler.prepare(document("x")).unwrap(); let work = prepared.planned_work();
    let state = Rc::new(State::default()); let mut admission = Admission::new(&state); admission.reject = true;
    let mut engine = Engine::new(&state); let mut run = RunState::new(work);
    let first = execute_admitted(prepared, &mut engine, &mut admission, &mut run,
        u64::MAX, context(1, 1), &mut Continue).err().unwrap();
    assert!(!first.stop); assert!(!run.failed); assert_eq!(run.reserved, work);
    admission.reject = false;
    let second = execute_admitted(compiler.prepare(document("x")).unwrap(), &mut engine, &mut admission,
        &mut run, u64::MAX, context(3, 2), &mut Continue).err().unwrap();
    assert!(second.stop); assert_eq!(second.fault.code, BatchCode::WorkLimit);
    assert_eq!(engine.calls, 0); assert_eq!(admission.calls, 1); assert_eq!(run.reserved, work);
}

#[test]
fn native_cancellation_preserves_cause_and_permanently_stops_admission() {
    let (planner, eos) = fixture(); let compiler = Int8GenerationBatchPlanner::new(&planner, Some(args(eos))).unwrap();
    let state = Rc::new(State::default()); let mut admission = Admission::new(&state); let mut engine = Engine::new(&state);
    engine.mode = Mode::Cancel; let mut run = RunState::new(room());
    let error = execute_admitted(compiler.prepare(document("x")).unwrap(), &mut engine, &mut admission,
        &mut run, u64::MAX, context(1, 1), &mut Continue).err().unwrap();
    assert!(error.stop); assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    assert!(!state.live.get()); assert!(!state.running.get()); assert!(engine.empty); assert!(run.failed);
    engine.poisoned = false; engine.mode = Mode::Success;
    assert!(execute_admitted(compiler.prepare(document("y")).unwrap(), &mut engine, &mut admission,
        &mut run, u64::MAX, context(2, 1), &mut Continue).is_err());
    assert_eq!(engine.calls, 1);
}

#[test]
fn panic_drains_physical_work_before_guard_drop_and_cannot_reopen_the_run() {
    let (planner, eos) = fixture(); let compiler = Int8GenerationBatchPlanner::new(&planner, Some(args(eos))).unwrap();
    let state = Rc::new(State::default()); let mut admission = Admission::new(&state); let mut engine = Engine::new(&state);
    engine.mode = Mode::Panic; let mut run = RunState::new(room());
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        execute_admitted(compiler.prepare(document("x")).unwrap(), &mut engine, &mut admission,
            &mut run, u64::MAX, context(1, 1), &mut Continue)
    }));
    assert!(caught.is_err()); assert!(run.failed); assert!(!state.live.get()); assert!(!state.running.get());
    assert_eq!(state.dropped.get(), 1); assert!(run.reserved.forward_positions > 0);
}

#[test]
fn dirty_or_poisoned_native_state_and_invalid_results_never_become_soft_success() {
    let (planner, eos) = fixture(); let compiler = Int8GenerationBatchPlanner::new(&planner, Some(args(eos))).unwrap();
    for mode in [Mode::DirtySuccess, Mode::PoisonedSuccess, Mode::WrongSequence, Mode::WrongWork, Mode::WorkFailure] {
        let state = Rc::new(State::default()); let mut admission = Admission::new(&state); let mut engine = Engine::new(&state);
        engine.mode = mode; let mut run = RunState::new(room());
        let error = execute_admitted(compiler.prepare(document("x")).unwrap(), &mut engine, &mut admission,
            &mut run, u64::MAX, context(1, 1), &mut Continue).err().unwrap();
        assert!(error.stop); assert!(run.failed); assert!(!state.live.get()); assert_eq!(state.dropped.get(), 1);
    }
}

#[test]
fn missing_or_replayed_delivery_coordinates_are_fatal_before_admission() {
    let work = Int8Work::for_sequence(0, 1, 1).unwrap();
    for coordinates in [context(0, 1), context(1, 0), BatchRequestContext { input_line: 0, ..context(1, 1) }] {
        let mut run = RunState::new(room()); assert!(run.begin(work, coordinates).is_err());
        assert!(run.failed); assert_eq!(run.reserved, Int8Work::default());
    }
    let mut run = RunState::new(room()); run.begin(work, context(5, 1)).unwrap(); run.failed = false;
    assert!(run.begin(work, context(5, 2)).is_err()); assert!(run.failed);
}

// Exercise the unchanged real NDJSON framing/output runner with the PRIVATE
// driver. Public NativeInt8GenerationBatch still has no generic-driver input.
struct Harness<'a> { compiler: Int8GenerationBatchPlanner<'a>, engine: Engine, admission: Admission, run: RunState }
impl BatchProcessor for Harness<'_> {
    type Args = GenerationBatchArgs;
    type Prepared = PreparedInt8Chat;
    type Output = GuardedOutput<Int8ChatResult, Guard>;
    fn prepare(&mut self, doc: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> { self.compiler.prepare(doc) }
    fn planned_work(&self, plan: &Self::Prepared) -> BatchWork {
        let w = plan.planned_work(); BatchWork { forward_positions: w.forward_positions, projected_logits: w.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, ctx: BatchRequestContext, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        execute_admitted(prepared, &mut self.engine, &mut self.admission, &mut self.run, u64::MAX, ctx, control)
    }
}
struct Writer { state: Rc<State>, documents: usize, flushed: usize, current_document: bool, fail_partial: bool, broken: bool }
impl Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.broken { return Err(io::Error::other("private writer failure")); }
        let v: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        self.current_document = v.get("result").is_some();
        if self.current_document {
            assert!(self.state.live.get()); assert!(!self.state.running.get()); self.documents += 1;
            if self.fail_partial { self.broken = true; return Ok(3.min(bytes.len())); }
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.current_document { assert!(self.state.live.get()); self.flushed += 1; }
        Ok(())
    }
}

#[test]
fn real_ndjson_runner_keeps_each_guard_through_write_flush_and_then_reuses_engine() {
    let (planner, eos) = fixture(); let state = Rc::new(State::default());
    let mut harness = Harness { compiler: Int8GenerationBatchPlanner::new(&planner, Some(args(eos))).unwrap(),
        engine: Engine::new(&state), admission: Admission::new(&state), run: RunState::new(room()) };
    let mut writer = Writer { state: Rc::clone(&state), documents: 0, flushed: 0,
        current_document: false, fail_partial: false, broken: false };
    let mut input = Cursor::new(b"{\"id\":\"a\",\"text\":\"one\"}\n{\"flush\":true}\n{\"id\":\"a\",\"text\":\"two\"}\n");
    let summary = batch::run_ndjson(&mut input, &mut writer, &mut harness, BatchLimits::default(), &mut Continue).unwrap();
    assert_eq!(summary.succeeded, 2); assert_eq!(summary.failed, 0);
    assert_eq!((writer.documents, writer.flushed, harness.engine.calls, state.dropped.get()), (2, 2, 2, 2));
    assert!(!state.live.get()); assert!(!harness.run.failed);
    assert_eq!(summary.reserved_work.forward_positions, harness.run.reserved.forward_positions);
    assert_eq!(summary.reserved_work.projected_logits, harness.run.reserved.projected_logits);
}

#[test]
fn partial_output_failure_drops_guard_and_stops_before_next_document_is_read() {
    let (planner, eos) = fixture(); let state = Rc::new(State::default());
    let mut harness = Harness { compiler: Int8GenerationBatchPlanner::new(&planner, Some(args(eos))).unwrap(),
        engine: Engine::new(&state), admission: Admission::new(&state), run: RunState::new(room()) };
    let mut writer = Writer { state: Rc::clone(&state), documents: 0, flushed: 0,
        current_document: false, fail_partial: true, broken: false };
    let bytes = b"{\"id\":\"a\",\"text\":\"one\"}\n{\"id\":\"b\",\"text\":\"two\"}\n";
    let mut input = Cursor::new(bytes);
    let error = batch::run_ndjson(&mut input, &mut writer, &mut harness, BatchLimits::default(), &mut Continue).unwrap_err();
    assert_eq!(error.fault.code, BatchCode::OutputIo); assert_eq!(harness.engine.calls, 1);
    assert!(input.position() < bytes.len() as u64); assert_eq!(state.dropped.get(), 1); assert!(!state.live.get());
}
