//! Fixed-task, ordered scoring streams on one admitted native INT4 engine.
//! Every document is one complete classification, sentiment or judgment bundle.
//! Heads and documents execute serially; no shared-layer batching is claimed.
use std::sync::Arc;
use crate::{
    execution_identity::ExecutionIdentity,
    native_engine::{constrained_int4,
        decode::DecodeStepControl, portable_int8::ProjectionWork,
        strict_int4::{Int4MemoryRequirement, Int4RunBudget, Int4Work, StrictInt4Engine,
            scoring::Int4ScoringBudget}},
    tasks::ir::TaskBudget,
};
use super::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
    BatchRequestContext, BatchWork, output::GuardedOutput};
mod tasks;
pub use tasks::Int4ScoredTask;

/// All five counters are finite, invocation-wide and nonrenewable at flush.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4ScoredBatchLimits {
    pub max_item_work: Int4Work,
    pub max_model_work: Int4Work,
}
impl Int4ScoredBatchLimits {
    pub fn validate(self) -> Result<(), BatchFault> {
        if !nonzero(self.max_item_work) || !nonzero(self.max_model_work)
            || !fits(self.max_item_work, self.max_model_work) { return Err(BatchCode::InvalidLimits.into()); }
        Ok(())
    }
}

/// Read-only facts from the existing sealed task plan, never wire authority.
pub struct Int4ScoredPlanFacts<'a> {
    pub identity: &'a ExecutionIdentity,
    pub model_work: Int4Work,
    pub group_width: usize,
    pub head_count: usize,
    pub required_context: usize,
    pub task_budget: TaskBudget,
    pub max_result_bytes: u64,
}

/// Owns a model-bound compiler and optional typed defaults. Defaults replace no
/// part of a supplied task_args object. The embedding caller prices compiler,
/// defaults and preparation storage before transferring them into the host.
pub struct Int4ScoredBatchPlanner<P: Int4ScoredTask> {
    planner: P,
    defaults: Option<P::Args>,
    limits: Int4ScoredBatchLimits,
    // An unforgeable in-process factory identity, not a digest or certificate.
    seal: Arc<()>,
}
impl<P: Int4ScoredTask> Int4ScoredBatchPlanner<P> {
    pub fn new(planner: P, defaults: Option<P::Args>, limits: Int4ScoredBatchLimits) -> Result<Self, BatchFault> {
        limits.validate()?;
        constrained_int4::check_profile(planner.base_identity(), planner.group_width())
            .map_err(|_| BatchCode::Admission)?;
        if planner.base_identity().task_spec != P::TASK { return Err(BatchCode::Admission.into()); }
        Ok(Self { planner, defaults, limits, seal: Arc::new(()) })
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { self.planner.base_identity() }
    pub fn group_width(&self) -> usize { self.planner.group_width() }
    pub fn limits(&self) -> Int4ScoredBatchLimits { self.limits }
    fn prepare<C: DecodeStepControl>(&self, document: BatchDocument<P::Args>, control: &mut C)
        -> Result<PreparedInt4Scored<P>, BatchItemFailure> {
        checkpoint(control)?;
        let args = document.task_args.or_else(|| self.defaults.clone())
            .ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        let plan = self.planner.plan_document(document.text, args, control)?;
        let facts = P::facts(&plan);
        if !fits(facts.model_work, self.limits.max_item_work) {
            return Err(BatchItemFailure::reject(BatchCode::WorkLimit));
        }
        checkpoint(control)?;
        Ok(PreparedInt4Scored { plan, seal: Arc::clone(&self.seal) })
    }
}

/// Public inspection cannot construct or replace the inner native plan/seal.
pub struct PreparedInt4Scored<P: Int4ScoredTask> { plan: P::Plan, seal: Arc<()> }
impl<P: Int4ScoredTask> PreparedInt4Scored<P> {
    pub fn execution_identity(&self) -> &ExecutionIdentity { P::facts(&self.plan).identity }
    pub fn model_work(&self) -> Int4Work { P::facts(&self.plan).model_work }
    pub fn head_count(&self) -> usize { P::facts(&self.plan).head_count }
    pub fn max_result_bytes(&self) -> u64 { P::facts(&self.plan).max_result_bytes }
}

/// A genuine embedding admission hook supplies a guard through sink delivery.
/// Scored bundles reuse KV across heads, so summed forward work is deliberately
/// separate from required_context and the complete allocated KV capacity.
pub struct Int4ScoredAdmission<'a> {
    pub facts: Int4ScoredPlanFacts<'a>,
    pub kv_reservation_bytes: u64,
}
pub trait Int4ScoredBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int4ScoredAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure>;
}

pub struct NativeInt4ScoredBatch<'e, 'w, P: Int4ScoredTask, A: Int4ScoredBatchAdmission> {
    compiler: Int4ScoredBatchPlanner<P>,
    driver: NativeDriver<'e, 'w>,
    admission: A,
    run: RunState,
}
impl<'e, 'w, P: Int4ScoredTask, A: Int4ScoredBatchAdmission> NativeInt4ScoredBatch<'e, 'w, P, A> {
    pub fn new(compiler: Int4ScoredBatchPlanner<P>, engine: &'e mut StrictInt4Engine<'w>, admission: A)
        -> Result<Self, BatchFault> {
        compiler.limits.validate()?;
        constrained_int4::check_model(compiler.base_execution_identity(), engine.model_identity())
            .map_err(|_| BatchCode::Admission)?;
        if compiler.group_width() != engine.model_identity().group_width()
            || engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
            return Err(BatchCode::InvalidExecution.into());
        }
        let run = RunState::new(compiler.limits.max_model_work);
        Ok(Self { compiler, driver: NativeDriver(engine), admission, run })
    }
    pub fn remaining_work(&self) -> Int4Work { self.run.remaining }
    pub fn is_poisoned(&self) -> bool { self.run.state != State::Ready || !clean(self.driver.0) }
}
impl<P: Int4ScoredTask, A: Int4ScoredBatchAdmission> BatchProcessor for NativeInt4ScoredBatch<'_, '_, P, A> {
    type Args = P::Args;
    type Prepared = PreparedInt4Scored<P>;
    type Output = GuardedOutput<P::Output, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.state = State::Failed;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        prepare_owned(&self.compiler, document, &self.driver, &mut self.run, control)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.model_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.run.state = State::Failed;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        execute_owned(prepared, &self.compiler.seal, &mut self.driver,
            &mut self.admission, &mut self.run, context, control)
    }
}

// Only private fixtures can replace native execution. The public constructor
// requires actual model weights and always installs NativeDriver.
trait Driver<P: Int4ScoredTask> {
    fn clean(&self) -> bool;
    fn capacity(&self) -> usize;
    fn group_width(&self) -> usize;
    fn preflight(&self, plan: &P::Plan, budget: Int4ScoringBudget) -> Result<(), BatchItemFailure>;
    fn execute<C: DecodeStepControl>(&mut self, plan: &P::Plan, admitted: &ExecutionIdentity,
        budget: Int4ScoringBudget, control: &mut C) -> Result<P::Output, BatchItemFailure>;
}
struct NativeDriver<'e, 'w>(&'e mut StrictInt4Engine<'w>);
fn clean(engine: &StrictInt4Engine<'_>) -> bool { !engine.is_poisoned() && engine.kv_cache().all_slots_have_len(0) }
impl<P: Int4ScoredTask> Driver<P> for NativeDriver<'_, '_> {
    fn clean(&self) -> bool { clean(self.0) }
    fn capacity(&self) -> usize { self.0.kv_cache().capacity_positions() }
    fn group_width(&self) -> usize { self.0.model_identity().group_width() }
    fn preflight(&self, plan: &P::Plan, budget: Int4ScoringBudget) -> Result<(), BatchItemFailure> {
        P::preflight(plan, self.0, budget)
    }
    fn execute<C: DecodeStepControl>(&mut self, plan: &P::Plan, admitted: &ExecutionIdentity,
        budget: Int4ScoringBudget, control: &mut C) -> Result<P::Output, BatchItemFailure> {
        P::execute(plan, admitted, self.0, budget, control)
    }
}
fn preflight<P: Int4ScoredTask, D: Driver<P>>(plan: &P::Plan, driver: &D) -> Result<u64, BatchItemFailure> {
    if !driver.clean() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
    let facts = P::facts(plan);
    if facts.group_width != driver.group_width() || facts.identity.task_spec != P::TASK {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    if facts.head_count == 0 || facts.required_context == 0 || !nonzero(facts.model_work) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    let required = Int4MemoryRequirement::for_context(driver.capacity(), driver.group_width())
        .map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    // Do not compare aggregate forward positions to a single reused KV context.
    if facts.required_context > driver.capacity() || required.kv_bytes > facts.task_budget.max_kv_bytes {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    if facts.max_result_bytes == 0 || facts.max_result_bytes > facts.task_budget.max_output_bytes {
        return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit));
    }
    driver.preflight(plan, budget(facts.model_work, required.kv_bytes))?;
    Ok(required.kv_bytes)
}
fn prepare_owned<P: Int4ScoredTask, D: Driver<P>, C: DecodeStepControl>(compiler: &Int4ScoredBatchPlanner<P>,
    document: BatchDocument<P::Args>, driver: &D, run: &mut RunState, control: &mut C)
    -> Result<PreparedInt4Scored<P>, BatchItemFailure> {
    run.enter()?;
    let result = (|| {
        let prepared = compiler.prepare(document, control)?;
        subtract(run.remaining, prepared.model_work())?;
        preflight::<P, D>(&prepared.plan, driver)?;
        checkpoint(control)?;
        Ok(prepared)
    })();
    settle(result, driver.clean(), run)
}
#[allow(clippy::too_many_arguments)]
fn execute_owned<P: Int4ScoredTask, D: Driver<P>, A: Int4ScoredBatchAdmission, C: DecodeStepControl>(
    prepared: PreparedInt4Scored<P>, seal: &Arc<()>, driver: &mut D, admission: &mut A,
    run: &mut RunState, context: BatchRequestContext, control: &mut C)
    -> Result<GuardedOutput<P::Output, A::Guard>, BatchItemFailure> {
    run.enter()?;
    let result = (|| {
        if !Arc::ptr_eq(&prepared.seal, seal) { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        let facts = P::facts(&prepared.plan);
        run.charge(facts.model_work, context)?;
        checkpoint(control)?;
        let kv_reservation_bytes = preflight::<P, D>(&prepared.plan, driver)?;
        let (identity, guard) = admission.admit(Int4ScoredAdmission { facts, kv_reservation_bytes })?;
        P::verify_identity(&prepared.plan, &identity)?;
        checkpoint(control)?;
        let value = driver.execute(&prepared.plan, &identity,
            budget(prepared.model_work(), kv_reservation_bytes), control)?;
        if !P::completed(&prepared.plan, &value) { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        checkpoint(control)?;
        Ok(GuardedOutput::new(value, guard))
    })();
    settle(result, driver.clean(), run)
}
fn budget(work: Int4Work, max_kv_bytes: u64) -> Int4ScoringBudget {
    Int4ScoringBudget { native: Int4RunBudget::exact(work), max_kv_bytes }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State { Ready, Running, Failed }
struct RunState { remaining: Int4Work, last_sequence: u64, state: State }
impl RunState {
    fn new(remaining: Int4Work) -> Self { Self { remaining, last_sequence: 0, state: State::Ready } }
    fn enter(&mut self) -> Result<(), BatchItemFailure> {
        if self.state != State::Ready { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        self.state = State::Running;
        Ok(())
    }
    fn charge(&mut self, work: Int4Work, context: BatchRequestContext) -> Result<(), BatchItemFailure> {
        if self.state != State::Running || context.request_seq <= self.last_sequence
            || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let remaining = subtract(self.remaining, work).map_err(|e| BatchItemFailure::fatal(e.fault))?;
        self.remaining = remaining;
        self.last_sequence = context.request_seq;
        Ok(())
    }
}
fn settle<T>(result: Result<T, BatchItemFailure>, clean: bool, run: &mut RunState) -> Result<T, BatchItemFailure> {
    match result {
        Ok(value) if clean && run.state == State::Running => { run.state = State::Ready; Ok(value) }
        Ok(_) => { run.state = State::Failed; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) }
        Err(mut error) => {
            if !clean || run.state != State::Running { error.stop = true; }
            run.state = if error.stop { State::Failed } else { State::Ready };
            Err(error)
        }
    }
}
fn nonzero(work: Int4Work) -> bool {
    work.forward_positions != 0 && work.projected_logits != 0 && work.attention_pairs != 0
        && work.projections.dot_products != 0 && work.projections.multiply_accumulates != 0
}
fn fits(work: Int4Work, cap: Int4Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn subtract(a: Int4Work, b: Int4Work) -> Result<Int4Work, BatchItemFailure> {
    let sub = |a: u64, b: u64| a.checked_sub(b).ok_or_else(|| BatchItemFailure::reject(BatchCode::WorkLimit));
    Ok(Int4Work { forward_positions: sub(a.forward_positions, b.forward_positions)?,
        projected_logits: sub(a.projected_logits, b.projected_logits)?, attention_pairs: sub(a.attention_pairs, b.attention_pairs)?,
        projections: ProjectionWork { dot_products: sub(a.projections.dot_products, b.projections.dot_products)?,
            multiply_accumulates: sub(a.projections.multiply_accumulates, b.projections.multiply_accumulates)? } })
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}

#[cfg(test)]
mod tests;
