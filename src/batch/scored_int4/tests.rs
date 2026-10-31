//! Private synthetic lifecycle fixtures for the shared adapter. No model,
//! native scoring, task quality, parity or throughput evidence is produced.
use super::*;
use serde::{Deserialize, Serialize};
use std::{cell::Cell, io::{self, Cursor, Write}, rc::Rc};
use crate::{canonjson, artifact::converter::PINNED_REVISION,
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{decode::DecodeCancellationKind, generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    batch::{BatchLimits, run_ndjson}};

fn work(units: u64) -> Int4Work {
    Int4Work { forward_positions: 40 * units, projected_logits: 50 * units, attention_pairs: 60 * units,
        projections: ProjectionWork { dot_products: 70 * units, multiply_accumulates: 80 * units } }
}
fn limits(item: u64, total: u64) -> Int4ScoredBatchLimits {
    Int4ScoredBatchLimits { max_item_work: work(item), max_model_work: work(total) }
}
pub(super) fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"private-scored-stream-lifecycle-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group=32"),
        packing_set_digest: Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, 32_usize,
        )).unwrap()), tokenizer_digest: d, template_digest: d, task_spec: "classify-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn task_budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 16, max_output_tokens: 4, max_output_bytes: 1024,
        max_grammar_states: 16, max_kv_bytes: Int4MemoryRequirement::for_context(32, 32).unwrap().kv_bytes }
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Args { units: u64 }
struct Planner { identity: ExecutionIdentity }
struct Plan { identity: ExecutionIdentity, units: u64, work: Int4Work, heads: usize,
    context: usize, budget: TaskBudget, result_bytes: u64 }
#[derive(Serialize)]
struct Output { units: u64, heads: usize, work: Int4Work }
impl tasks::sealed::Sealed for Planner {}
impl Int4ScoredTask for Planner {
    type Args = Args; type Plan = Plan; type Output = Output;
    const TASK: &'static str = "classify-v1";
    fn base_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn group_width(&self) -> usize { 32 }
    fn plan_document<C: DecodeStepControl>(&self, text: String, args: Args, control: &mut C) -> Result<Plan, BatchItemFailure> {
        checkpoint(control)?;
        if text == "panic-plan" { panic!("private planning fixture"); }
        if text == "reject-plan" || !(1..=8).contains(&args.units) { return Err(BatchItemFailure::reject(BatchCode::Planning)); }
        let mut identity = self.identity.clone();
        identity.prompt_digest = Sha256Digest::of_bytes(text.as_bytes());
        identity.decision_policy_digest = Sha256Digest::of_bytes(&args.units.to_le_bytes());
        let heads = match text.as_str() { "65-heads" => 65, "128-heads" => 128, _ => 4 };
        Ok(Plan { identity, units: args.units, work: work(args.units), heads,
            context: 8, budget: task_budget(), result_bytes: 1024 })
    }
    fn facts(plan: &Plan) -> Int4ScoredPlanFacts<'_> {
        Int4ScoredPlanFacts { identity: &plan.identity, model_work: plan.work, group_width: 32,
            head_count: plan.heads, required_context: plan.context, task_budget: plan.budget, max_result_bytes: plan.result_bytes }
    }
    fn verify_identity(plan: &Plan, admitted: &ExecutionIdentity) -> Result<(), BatchItemFailure> {
        if &plan.identity != admitted { Err(BatchItemFailure::fatal(BatchCode::Admission)) } else { Ok(()) }
    }
    fn preflight(_: &Plan, _: &StrictInt4Engine<'_>, _: Int4ScoringBudget) -> Result<(), BatchItemFailure> { unreachable!() }
    fn execute<C: DecodeStepControl>(_: &Plan, _: &ExecutionIdentity, _: &mut StrictInt4Engine<'_>,
        _: Int4ScoringBudget, _: &mut C) -> Result<Output, BatchItemFailure> { unreachable!() }
    fn completed(plan: &Plan, output: &Output) -> bool {
        plan.units == output.units && plan.heads == output.heads && plan.work == output.work
    }
    fn cohort_contexts(plan: &Plan) -> Result<Vec<usize>, BatchItemFailure> {
        assert!(plan.heads <= 64, "overwide documents must be rejected before native context extraction");
        Ok(vec![plan.context; plan.heads])
    }
    fn execute_document_cohort<C: DecodeStepControl>(_: &[(&Plan, &ExecutionIdentity)],
        _: &mut crate::native_engine::strict_int4::cohort::Int4CohortEngine<'_>,
        _: crate::native_engine::strict_int4::scoring::cohort::Int4ScoringCohortBudget, _: &mut C)
        -> Result<crate::native_engine::strict_int4::scoring::cohort::Int4ScoredCohort<Vec<Output>>, BatchItemFailure> {
        unreachable!("serial lifecycle fixtures never replace native cohort execution")
    }
}
fn compiler(limits: Int4ScoredBatchLimits) -> Int4ScoredBatchPlanner<Planner> {
    Int4ScoredBatchPlanner::new(Planner { identity: identity() }, Some(Args { units: 1 }), limits).unwrap()
}
fn document() -> BatchDocument<Args> { BatchDocument { id: "a".to_owned(), text: "private source".to_owned(), task_args: None } }
fn context(seq: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: seq, epoch, input_line: seq, byte_offset: 0 }
}
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
struct StopAt { calls: usize, stop: usize }
impl DecodeStepControl for StopAt {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1; (self.calls >= self.stop).then_some(DecodeCancellationKind::Deadline)
    }
}
#[derive(Default)]
struct Probe { live: Cell<bool>, running: Cell<bool>, drops: Cell<usize>, calls: Cell<usize>, admissions: Cell<usize>, preflights: Cell<usize> }
struct Guard(Rc<Probe>);
impl Drop for Guard {
    fn drop(&mut self) {
        assert!(!self.0.running.get()); assert!(self.0.live.replace(false)); self.0.drops.set(self.0.drops.get() + 1);
    }
}
struct Physical(Rc<Probe>);
impl Drop for Physical { fn drop(&mut self) { self.0.running.set(false); } }
struct Admission { probe: Rc<Probe>, reject: bool, wrong: bool }
impl Int4ScoredBatchAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, request: Int4ScoredAdmission<'_>) -> Result<(ExecutionIdentity, Guard), BatchItemFailure> {
        self.probe.admissions.set(self.probe.admissions.get() + 1);
        assert_eq!(request.kv_reservation_bytes, Int4MemoryRequirement::for_context(32, 32).unwrap().kv_bytes);
        assert_eq!(request.facts.head_count, 4); assert_eq!(request.facts.required_context, 8);
        if self.reject { return Err(BatchItemFailure::reject(BatchCode::Admission)); }
        assert!(!self.probe.live.replace(true)); let mut identity = request.facts.identity.clone();
        if self.wrong { identity.prompt_digest = Sha256Digest::of_bytes(b"wrong admitted source"); }
        Ok((identity, Guard(Rc::clone(&self.probe))))
    }
}
#[derive(Clone, Copy)]
enum Mode { Success, Cancel, Panic, Failure, Dirty, WrongWork }
struct Engine { probe: Rc<Probe>, mode: Mode, clean: bool, capacity: usize, width: usize, reject_preflight: bool }
impl Driver<Planner> for Engine {
    fn clean(&self) -> bool { self.clean }
    fn capacity(&self) -> usize { self.capacity }
    fn group_width(&self) -> usize { self.width }
    fn preflight(&self, plan: &Plan, budget: Int4ScoringBudget) -> Result<(), BatchItemFailure> {
        self.probe.preflights.set(self.probe.preflights.get() + 1);
        assert_eq!(budget.native.max_forward_positions, plan.work.forward_positions);
        assert_eq!(budget.native.max_attention_pairs, plan.work.attention_pairs);
        assert_eq!(budget.native.max_projection_work, plan.work.projections);
        if self.reject_preflight { return Err(BatchItemFailure::reject(BatchCode::Admission)); }
        Ok(())
    }
    fn execute<C: DecodeStepControl>(&mut self, plan: &Plan, id: &ExecutionIdentity,
        _: Int4ScoringBudget, _: &mut C) -> Result<Output, BatchItemFailure> {
        Planner::verify_identity(plan, id)?;
        assert!(self.probe.live.get()); assert!(!self.probe.running.replace(true));
        let _physical = Physical(Rc::clone(&self.probe)); self.probe.calls.set(self.probe.calls.get() + 1);
        match self.mode {
            Mode::Panic => panic!("private scored lifecycle fixture"),
            Mode::Cancel => return Err(BatchItemFailure::fatal(BatchFault::cancelled(DecodeCancellationKind::Deadline))),
            Mode::Failure => return Err(BatchItemFailure::fatal(BatchCode::Execution)),
            Mode::Dirty => self.clean = false, _ => (),
        }
        let mut output = Output { units: plan.units, heads: plan.heads, work: plan.work };
        if matches!(self.mode, Mode::WrongWork) { output.work.projections.multiply_accumulates += 1; }
        Ok(output)
    }
}
fn host(probe: &Rc<Probe>, mode: Mode, total: u64) -> (Engine, Admission, RunState) {
    (Engine { probe: Rc::clone(probe), mode, clean: true, capacity: 32, width: 32, reject_preflight: false },
        Admission { probe: Rc::clone(probe), reject: false, wrong: false }, RunState::new(work(total)))
}
fn decrement(work: &mut Int4Work, axis: usize) {
    match axis { 0 => work.forward_positions -= 1, 1 => work.projected_logits -= 1, 2 => work.attention_pairs -= 1,
        3 => work.projections.dot_products -= 1, _ => work.projections.multiply_accumulates -= 1 }
}

fn cohort_limits() -> Int4ScoredCohortBatchLimits {
    let memory = Int4MemoryRequirement::for_cohort(&[32; 64], 32).unwrap();
    Int4ScoredCohortBatchLimits { max_documents: 4, max_heads: 64, context_limit: 32,
        native_memory: crate::native_engine::strict_int4::Int4MemoryBudget {
            max_kv_bytes: memory.kv_bytes, max_rope_bytes: memory.rope_bytes,
            max_scratch_payload_bytes: memory.scratch_payload_bound },
        max_workspace_bytes: 1 << 24, max_item_output_bytes: 1024, max_output_bytes: 4096 }
}

#[test]
fn overwide_cohort_plans_never_reach_native_contexts_and_do_not_poison_following_plans() {
    let compiler = compiler(limits(1, 2)); let mut remaining = work(2);
    for text in ["65-heads", "128-heads"] {
        let mut oversized = document(); oversized.text = text.to_owned();
        let error = cohort::prepare_document(&compiler, oversized, cohort_limits(), &mut remaining, &mut Continue)
            .err().expect("overwide document rejected before context extraction");
        assert_eq!(error.fault.code, BatchCode::Admission); assert!(!error.stop);
        assert_eq!(remaining, work(2));
    }
    for after in [work(1), work(0)] {
        let plan = cohort::prepare_document(&compiler, document(), cohort_limits(), &mut remaining, &mut Continue).unwrap();
        assert_eq!(plan.head_count(), 4); assert_eq!(remaining, after);
    }
    let exhausted = cohort::prepare_document(&compiler, document(), cohort_limits(), &mut remaining, &mut Continue)
        .err().expect("queued documents already spent the invocation allowance");
    assert_eq!(exhausted.fault.code, BatchCode::WorkLimit); assert!(!exhausted.stop);
    assert_eq!(remaining, work(0));
}

#[test]
fn cohort_preparation_checks_all_five_work_axes_before_queueing_without_partial_debits() {
    let compiler = compiler(limits(1, 1));
    for axis in 0..5 {
        let mut remaining = work(1); decrement(&mut remaining, axis); let before = remaining;
        let error = cohort::prepare_document(&compiler, document(), cohort_limits(), &mut remaining, &mut Continue)
            .err().expect("whole document cannot fit the remaining work");
        assert_eq!(error.fault.code, BatchCode::WorkLimit); assert!(!error.stop);
        assert_eq!(remaining, before, "axis {axis} must not partially debit other axes");
    }
}

#[test]
fn every_item_and_invocation_work_axis_is_nonzero_and_item_cannot_exceed_run() {
    for axis in 0..5 {
        let mut bad = limits(1, 1); decrement(&mut bad.max_model_work, axis); assert!(bad.validate().is_err());
        let mut bad = limits(1, 2);
        match axis { 0 => bad.max_item_work.forward_positions = 0, 1 => bad.max_item_work.projected_logits = 0,
            2 => bad.max_item_work.attention_pairs = 0, 3 => bad.max_item_work.projections.dot_products = 0,
            _ => bad.max_item_work.projections.multiply_accumulates = 0 }
        assert!(bad.validate().is_err());
    }
}
#[test]
fn invalid_profile_partition_or_task_cannot_create_a_factory() {
    for axis in 0..4 {
        let mut id = identity();
        match axis { 0 => id.task_spec = "sentiment-v1".to_owned(),
            1 => id.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            2 => id.quant_recipe = format!("{GROUPED_INT4_RECIPE}:group=16"),
            _ => id.backend_semantic_version.push('x') }
        assert!(Int4ScoredBatchPlanner::new(Planner { identity: id }, None, limits(1, 2)).is_err());
    }
}
#[test]
fn explicit_arguments_replace_defaults_and_missing_or_oversized_items_reject_before_admission() {
    let c = compiler(limits(2, 3)); let a = c.prepare(document(), &mut Continue).unwrap();
    let mut doc = document(); doc.task_args = Some(Args { units: 2 });
    let b = c.prepare(doc, &mut Continue).unwrap();
    assert_eq!(a.model_work(), work(1)); assert_eq!(b.model_work(), work(2));
    assert_ne!(a.execution_identity(), b.execution_identity());
    let empty = Int4ScoredBatchPlanner::new(Planner { identity: identity() }, None, limits(1, 2)).unwrap();
    assert_eq!(empty.prepare(document(), &mut Continue).err().unwrap().fault.code, BatchCode::Planning);
    let mut doc = document(); doc.task_args = Some(Args { units: 3 });
    assert_eq!(c.prepare(doc, &mut Continue).err().unwrap().fault.code, BatchCode::WorkLimit);
}
#[test]
fn multihead_total_work_can_exceed_one_reused_context_but_full_allocated_kv_is_required() {
    let c = compiler(limits(1, 2)); let plan = c.prepare(document(), &mut Continue).unwrap();
    let probe = Rc::new(Probe::default()); let (driver, _, _) = host(&probe, Mode::Success, 2);
    assert!(plan.model_work().forward_positions > driver.capacity as u64);
    assert_eq!(preflight::<Planner, _>(&plan.plan, &driver).unwrap(), task_budget().max_kv_bytes);
    let mut plan = c.prepare(document(), &mut Continue).unwrap(); plan.plan.budget.max_kv_bytes -= 1;
    let error = preflight::<Planner, _>(&plan.plan, &driver).unwrap_err();
    assert_eq!(error.fault.code, BatchCode::Admission); assert!(!error.stop);
}
#[test]
fn invalid_plan_geometry_capacity_and_result_claims_fail_before_native_work() {
    let c = compiler(limits(1, 2));
    for axis in 0..7 {
        let mut plan = c.prepare(document(), &mut Continue).unwrap();
        let probe = Rc::new(Probe::default()); let (mut driver, _, _) = host(&probe, Mode::Success, 2);
        match axis { 0 => plan.plan.heads = 0, 1 => plan.plan.context = 0, 2 => plan.plan.context = 33,
            3 => plan.plan.result_bytes = 0, 4 => plan.plan.result_bytes += 1, 5 => driver.width = 16,
            _ => driver.clean = false }
        assert!(preflight::<Planner, _>(&plan.plan, &driver).is_err()); assert_eq!(probe.calls.get(), 0);
    }
}
#[test]
fn preparation_checks_every_remaining_work_axis_without_consuming_it() {
    let c = compiler(limits(1, 2));
    for axis in 0..5 {
        let probe = Rc::new(Probe::default()); let (driver, _, mut run) = host(&probe, Mode::Success, 1);
        decrement(&mut run.remaining, axis); let before = run.remaining;
        let error = prepare_owned(&c, document(), &driver, &mut run, &mut Continue).err().unwrap();
        assert_eq!(error.fault.code, BatchCode::WorkLimit); assert!(!error.stop);
        assert_eq!(run.state, State::Ready); assert_eq!(run.remaining, before); assert_eq!(probe.calls.get(), 0);
    }
}
#[test]
fn foreign_factory_seal_never_reaches_admission_or_spends_model_work() {
    let first = compiler(limits(1, 2)); let second = compiler(limits(1, 2));
    let prepared = first.prepare(document(), &mut Continue).unwrap(); let probe = Rc::new(Probe::default());
    let (mut driver, mut admission, mut run) = host(&probe, Mode::Success, 2);
    let error = execute_owned(prepared, &second.seal, &mut driver, &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
    assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission); assert_eq!(run.state, State::Failed);
    assert_eq!(run.remaining, work(2)); assert_eq!(probe.admissions.get(), 0); assert_eq!(probe.calls.get(), 0);
}
#[test]
fn failed_admission_spends_all_five_counters_and_flush_epoch_never_refunds_them() {
    let c = compiler(limits(1, 2)); let probe = Rc::new(Probe::default());
    let (mut driver, mut admission, mut run) = host(&probe, Mode::Success, 1); admission.reject = true;
    let error = execute_owned(c.prepare(document(), &mut Continue).unwrap(), &c.seal,
        &mut driver, &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
    assert!(!error.stop); assert_eq!(run.state, State::Ready); assert_eq!(run.remaining, Int4Work::default());
    admission.reject = false;
    let error = execute_owned(c.prepare(document(), &mut Continue).unwrap(), &c.seal,
        &mut driver, &mut admission, &mut run, context(3, 2), &mut Continue).err().unwrap();
    assert!(error.stop); assert_eq!(error.fault.code, BatchCode::WorkLimit); assert_eq!(run.state, State::Failed);
    assert_eq!(run.remaining, Int4Work::default()); assert_eq!(probe.admissions.get(), 1); assert_eq!(probe.calls.get(), 0);
}
#[test]
fn repeated_or_invalid_delivery_coordinates_cannot_reenter_a_run() {
    for ctx in [context(0, 1), context(1, 0), BatchRequestContext { input_line: 0, ..context(1, 1) }] {
        let c = compiler(limits(1, 2)); let probe = Rc::new(Probe::default());
        let (mut driver, mut admission, mut run) = host(&probe, Mode::Success, 2);
        let error = execute_owned(c.prepare(document(), &mut Continue).unwrap(), &c.seal,
            &mut driver, &mut admission, &mut run, ctx, &mut Continue).err().unwrap();
        assert!(error.stop); assert_eq!(run.remaining, work(2)); assert_eq!(probe.calls.get(), 0);
    }
    let mut run = RunState::new(work(2)); run.enter().unwrap(); run.charge(work(1), context(4, 1)).unwrap();
    run.state = State::Ready; run.enter().unwrap(); let before = run.remaining;
    assert!(run.charge(work(1), context(4, 2)).is_err()); assert_eq!(run.remaining, before);
}
#[test]
fn guard_survives_completion_and_serialization_and_drops_after_result_owner() {
    let c = compiler(limits(1, 2)); let probe = Rc::new(Probe::default());
    let (mut driver, mut admission, mut run) = host(&probe, Mode::Success, 2);
    let output = execute_owned(c.prepare(document(), &mut Continue).unwrap(), &c.seal,
        &mut driver, &mut admission, &mut run, context(1, 1), &mut Continue).unwrap();
    assert_eq!(run.remaining, work(1)); assert_eq!(run.state, State::Ready);
    assert!(probe.live.get()); assert!(!probe.running.get());
    let json = canonjson::canonical_string(&output).unwrap();
    assert!(!json.contains("private source") && !json.contains("prompt_digest") && !json.contains("guard"));
    assert!(probe.live.get()); drop(output); assert_eq!(probe.drops.get(), 1);
}

#[test]
fn each_execution_ceiling_is_checked_atomically_before_admission() {
    let c = compiler(limits(1, 2));
    for axis in 0..5 {
        let probe = Rc::new(Probe::default()); let (mut driver, mut admission, mut run) = host(&probe, Mode::Success, 1);
        decrement(&mut run.remaining, axis); let before = run.remaining;
        let error = execute_owned(c.prepare(document(), &mut Continue).unwrap(), &c.seal,
            &mut driver, &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
        assert_eq!(error.fault.code, BatchCode::WorkLimit); assert!(error.stop);
        assert_eq!(run.remaining, before); assert_eq!(run.last_sequence, 0); assert_eq!(run.state, State::Failed);
        assert_eq!(probe.admissions.get(), 0); assert_eq!(probe.calls.get(), 0);
    }
}
#[test]
fn wrong_admitted_identity_releases_guard_without_any_native_execution() {
    let c = compiler(limits(1, 2)); let probe = Rc::new(Probe::default());
    let (mut driver, mut admission, mut run) = host(&probe, Mode::Success, 2); admission.wrong = true;
    let error = execute_owned(c.prepare(document(), &mut Continue).unwrap(), &c.seal,
        &mut driver, &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
    assert_eq!(error.fault.code, BatchCode::Admission); assert!(error.stop);
    assert_eq!(run.remaining, work(1)); assert_eq!(run.state, State::Failed);
    assert_eq!(probe.calls.get(), 0); assert_eq!(probe.drops.get(), 1); assert!(!probe.live.get());
}
#[test]
fn cancelled_failed_dirty_or_incomplete_native_bundles_cannot_publish_or_restart() {
    for (mode, code) in [(Mode::Cancel, BatchCode::Cancelled), (Mode::Failure, BatchCode::Execution),
        (Mode::Dirty, BatchCode::InvalidExecution), (Mode::WrongWork, BatchCode::InvalidExecution)] {
        let c = compiler(limits(1, 2)); let probe = Rc::new(Probe::default());
        let (mut driver, mut admission, mut run) = host(&probe, mode, 2);
        let error = execute_owned(c.prepare(document(), &mut Continue).unwrap(), &c.seal,
            &mut driver, &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
        assert_eq!(error.fault.code, code); assert!(error.stop); assert_eq!(run.state, State::Failed);
        assert_eq!(run.remaining, work(1)); assert_eq!(probe.calls.get(), 1);
        assert_eq!(probe.drops.get(), 1); assert!(!probe.live.get()); assert!(!probe.running.get());
        driver.clean = true; driver.mode = Mode::Success;
        assert!(prepare_owned(&c, document(), &driver, &mut run, &mut Continue).is_err());
        assert_eq!(probe.calls.get(), 1); assert_eq!(probe.admissions.get(), 1);
    }
}
#[test]
fn panics_in_planning_or_execution_leave_the_invocation_unavailable() {
    for in_native in [false, true] {
        let c = compiler(limits(1, 2)); let probe = Rc::new(Probe::default());
        let (mut driver, mut admission, mut run) = host(&probe, Mode::Panic, 2);
        if in_native {
            let prepared = c.prepare(document(), &mut Continue).unwrap();
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                execute_owned(prepared, &c.seal, &mut driver, &mut admission, &mut run, context(1, 1), &mut Continue)
            })).is_err());
        } else {
            let mut doc = document(); doc.text = "panic-plan".to_owned();
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                prepare_owned(&c, doc, &driver, &mut run, &mut Continue)
            })).is_err());
        }
        // Unwinding never turns Running back into Ready. Physical cleanup
        // precedes guard release, even when a fixture scorer panics.
        assert_eq!(run.state, State::Running); assert!(run.enter().is_err());
        assert_eq!(run.remaining, work(if in_native { 1 } else { 2 }));
        assert_eq!(probe.calls.get(), usize::from(in_native)); assert_eq!(probe.drops.get(), usize::from(in_native));
        assert!(!probe.live.get()); assert!(!probe.running.get());
    }
}
#[test]
fn the_original_control_can_cancel_before_planning_or_at_final_publication() {
    let c = compiler(limits(1, 2)); let probe = Rc::new(Probe::default());
    let (mut driver, mut admission, mut run) = host(&probe, Mode::Success, 2);
    let mut stop = StopAt { calls: 0, stop: 1 };
    let error = prepare_owned(&c, document(), &driver, &mut run, &mut stop).err().unwrap();
    assert_eq!(error.fault.code, BatchCode::Cancelled); assert!(error.stop); assert_eq!(stop.calls, 1);
    assert_eq!(run.state, State::Failed); assert_eq!(run.remaining, work(2)); assert_eq!(probe.preflights.get(), 0);

    let mut run = RunState::new(work(2)); let mut stop = StopAt { calls: 0, stop: 3 };
    let error = execute_owned(c.prepare(document(), &mut Continue).unwrap(), &c.seal,
        &mut driver, &mut admission, &mut run, context(1, 1), &mut stop).err().unwrap();
    assert_eq!(error.fault.code, BatchCode::Cancelled); assert!(error.stop); assert_eq!(stop.calls, 3);
    assert_eq!(run.state, State::Failed); assert_eq!(run.remaining, work(1));
    assert_eq!(probe.calls.get(), 1); assert_eq!(probe.drops.get(), 1); assert!(!probe.live.get());
}
#[test]
fn recoverable_preparation_rejections_leave_the_same_invocation_usable_and_unspent() {
    let c = compiler(limits(1, 2)); let probe = Rc::new(Probe::default());
    let (mut driver, _, mut run) = host(&probe, Mode::Success, 2);
    let mut doc = document(); doc.text = "reject-plan".to_owned();
    let error = prepare_owned(&c, doc, &driver, &mut run, &mut Continue).err().unwrap();
    assert_eq!(error.fault.code, BatchCode::Planning); assert!(!error.stop);
    driver.reject_preflight = true;
    let error = prepare_owned(&c, document(), &driver, &mut run, &mut Continue).err().unwrap();
    assert_eq!(error.fault.code, BatchCode::Admission); assert!(!error.stop);
    driver.reject_preflight = false;
    assert!(prepare_owned(&c, document(), &driver, &mut run, &mut Continue).is_ok());
    assert_eq!(run.state, State::Ready); assert_eq!(run.remaining, work(2));
    assert_eq!(probe.admissions.get(), 0); assert_eq!(probe.calls.get(), 0);
}

// This harness substitutes only the private driver; the real ordered reader,
// accounting, framing and guarded output sink remain in run_ndjson.
struct Harness {
    compiler: Int4ScoredBatchPlanner<Planner>, driver: Engine, admission: Admission,
    run: RunState, contexts: Vec<BatchRequestContext>,
}
impl Harness {
    fn new(probe: &Rc<Probe>, total: u64) -> Self {
        let (driver, admission, run) = host(probe, Mode::Success, total);
        Self { compiler: compiler(limits(1, total)), driver, admission, run, contexts: Vec::new() }
    }
}
impl BatchProcessor for Harness {
    type Args = Args; type Prepared = PreparedInt4Scored<Planner>; type Output = GuardedOutput<Output, Guard>;
    fn prepare(&mut self, _: BatchDocument<Args>) -> Result<Self::Prepared, BatchItemFailure> {
        panic!("runner bypassed original planning control")
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, doc: BatchDocument<Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        prepare_owned(&self.compiler, doc, &self.driver, &mut self.run, control)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.model_work(); BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        panic!("runner bypassed delivery coordinates")
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, context: BatchRequestContext,
        control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.contexts.push(context);
        execute_owned(prepared, &self.compiler.seal, &mut self.driver, &mut self.admission, &mut self.run, context, control)
    }
}
struct Writer {
    probe: Rc<Probe>, failure: usize, broken: bool, document: bool,
    flushed_documents: usize, events: Vec<serde_json::Value>,
}
impl Writer {
    fn new(probe: &Rc<Probe>, failure: usize) -> Self {
        Self { probe: Rc::clone(probe), failure, broken: false, document: false, flushed_documents: 0, events: Vec::new() }
    }
}
impl Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.broken { return Err(io::Error::other("private sink failure")); }
        let event: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        self.document = event.get("result").is_some();
        if self.document {
            assert!(self.probe.live.get()); assert!(!self.probe.running.get());
            if self.failure == 1 { self.broken = true; return Ok(bytes.len().min(3)); }
        }
        self.events.push(event); Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.document {
            assert!(self.probe.live.get()); assert!(!self.probe.running.get()); self.flushed_documents += 1;
            if self.failure == 2 { self.broken = true; return Err(io::Error::other("private flush failure")); }
        }
        Ok(())
    }
}
#[test]
fn real_stream_flush_preserves_all_five_budgets_and_delivery_sequence() {
    let input = b"{\"id\":\"a\",\"text\":\"one\"}\n{\"flush\":true}\n{\"id\":\"a\",\"text\":\"two\"}\n{\"id\":\"b\",\"text\":\"three\"}\n";
    let mut reader = Cursor::new(input); let probe = Rc::new(Probe::default());
    let mut processor = Harness::new(&probe, 2); let mut writer = Writer::new(&probe, 0);
    let summary = run_ndjson(&mut reader, &mut writer, &mut processor, BatchLimits::default(), &mut Continue).unwrap();
    assert_eq!((summary.requests, summary.succeeded, summary.failed, summary.flushes), (4, 2, 1, 2));
    assert_eq!(summary.reserved_work, BatchWork { forward_positions: 80, projected_logits: 100 });
    assert_eq!(processor.run.remaining, Int4Work::default()); assert_eq!(processor.run.state, State::Ready);
    assert_eq!(processor.contexts.len(), 2);
    assert_eq!((processor.contexts[0].request_seq, processor.contexts[0].epoch), (1, 1));
    assert_eq!((processor.contexts[1].request_seq, processor.contexts[1].epoch), (3, 2));
    assert_eq!(processor.contexts[1].input_line, 3); assert!(processor.contexts[1].byte_offset > 0);
    assert_eq!(probe.calls.get(), 2); assert_eq!(probe.drops.get(), 2); assert_eq!(writer.flushed_documents, 2);
    assert!(!probe.live.get()); assert_eq!(writer.events.last().unwrap()["event"], "run_complete");
    let error = writer.events.iter().find(|event| event["event"] == "doc_error").unwrap();
    assert_eq!(error["error"]["code"], "work_limit"); assert_eq!(error["request_seq"], 4);
}
#[test]
fn sink_partial_write_or_failed_flush_keeps_guard_until_delivery_stops_and_reads_no_next_item() {
    let input = b"{\"id\":\"a\",\"text\":\"one\"}\n{\"id\":\"b\",\"text\":\"two\"}\n";
    for failure in [1, 2] {
        let mut reader = Cursor::new(input); let probe = Rc::new(Probe::default());
        let mut processor = Harness::new(&probe, 2); let mut writer = Writer::new(&probe, failure);
        let error = run_ndjson(&mut reader, &mut writer, &mut processor, BatchLimits::default(), &mut Continue).unwrap_err();
        assert_eq!(error.fault.code, BatchCode::OutputIo); assert!(writer.broken);
        assert_eq!(probe.calls.get(), 1); assert_eq!(probe.drops.get(), 1); assert!(!probe.live.get());
        assert!(reader.position() < input.len() as u64); assert_eq!(processor.run.remaining, work(1));
        assert!(!writer.events.iter().any(|event| event["event"] == "run_error" || event["event"] == "run_complete"));
    }
}
#[test]
fn real_stream_stops_after_a_native_failure_and_never_reports_provisional_success() {
    let input = b"{\"id\":\"a\",\"text\":\"one\"}\n{\"id\":\"b\",\"text\":\"two\"}\n";
    let mut reader = Cursor::new(input); let probe = Rc::new(Probe::default());
    let mut processor = Harness::new(&probe, 2); processor.driver.mode = Mode::Cancel;
    let mut writer = Writer::new(&probe, 0);
    let error = run_ndjson(&mut reader, &mut writer, &mut processor, BatchLimits::default(), &mut Continue).unwrap_err();
    assert_eq!(error.fault.code, BatchCode::Cancelled); assert_eq!(error.summary.succeeded, 0); assert_eq!(error.summary.failed, 1);
    assert_eq!(error.summary.reserved_work, BatchWork { forward_positions: 40, projected_logits: 50 });
    assert_eq!(processor.run.remaining, work(1)); assert_eq!(processor.run.state, State::Failed);
    assert_eq!(probe.calls.get(), 1); assert_eq!(probe.drops.get(), 1); assert!(reader.position() < input.len() as u64);
    assert!(!writer.events.iter().any(|event| event.get("result").is_some()));
    assert_eq!(writer.events.last().unwrap()["event"], "run_error");
}
#[test]
fn all_three_native_tasks_keep_their_existing_typed_arguments_plans_and_complete_results() {
    use crate::tasks::{classify::grouped_int4 as classification, sentiment::grouped_int4 as sentiment,
        judge::grouped_int4 as judgment};
    fn classification_binding<P: Int4ScoredTask<Args = crate::batch::classify::ClassificationBatchArgs,
        Plan = classification::PreparedInt4Classification, Output = classification::Int4ClassificationRun>>() {}
    fn sentiment_binding<P: Int4ScoredTask<Args = crate::tasks::sentiment::batch::SentimentBatchArgs,
        Plan = sentiment::PreparedInt4Sentiment, Output = sentiment::Int4SentimentRun>>() {}
    fn judgment_binding<P: Int4ScoredTask<Args = crate::batch::judge::JudgeBatchArgs,
        Plan = judgment::PreparedInt4Judge, Output = judgment::Int4JudgeRun>>() {}
    classification_binding::<classification::Int4ClassificationPlanner>();
    sentiment_binding::<sentiment::Int4SentimentPlanner>(); judgment_binding::<judgment::Int4JudgePlanner>();
    assert_eq!(<classification::Int4ClassificationPlanner as Int4ScoredTask>::TASK, "classify-v1");
    assert_eq!(<sentiment::Int4SentimentPlanner as Int4ScoredTask>::TASK, "sentiment-v1");
    assert_eq!(<judgment::Int4JudgePlanner as Int4ScoredTask>::TASK, "judge-v1");
}
#[test]
fn task_argument_schemas_reject_wire_identity_tokens_and_foreign_task_fields() {
    use crate::{batch::{classify::ClassificationBatchArgs, judge::JudgeBatchArgs},
        tasks::{classify::{ClassificationMode, ClassificationPolicy}, sentiment::{SentimentAxis, batch::SentimentBatchArgs},
            judge::PairwisePolicy}};
    fn closed<A: Serialize + serde::de::DeserializeOwned>(args: A) {
        let value = serde_json::to_value(args).unwrap(); assert!(serde_json::from_value::<A>(value.clone()).is_ok());
        for field in ["execution_identity", "prompt_tokens", "request_seq", "foreign_task"] {
            let mut value = value.clone(); value.as_object_mut().unwrap().insert(field.to_owned(), serde_json::json!(7));
            assert!(serde_json::from_value::<A>(value).is_err());
        }
    }
    closed(ClassificationBatchArgs { labels: Vec::new(), mode: ClassificationMode::Exclusive,
        policy: ClassificationPolicy::default(), budget: task_budget() });
    closed(SentimentBatchArgs { axes: vec![SentimentAxis::Valence, SentimentAxis::Arousal], budget: task_budget() });
    closed(JudgeBatchArgs::Pairwise { criterion: "clarity".to_owned(), b: "candidate B".to_owned(),
        policy: PairwisePolicy { minimum_margin_milli: 0, maximum_order_disagreement_milli: 1000 }, budget: task_budget() });
}
