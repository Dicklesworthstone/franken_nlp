//! Ordered raw-text classification batches on one admitted int8 engine.
//! Reuses the existing NDJSON transport and argument schema. Five native work
//! counters remain charged across flushes and separate runner invocations.

use crate::{canonjson,
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor, BatchWork},
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        portable_int8::ProjectionWork,
        strict_int8::{Int8RunBudget, Int8Work, StrictInt8Engine, StrictInt8Error, STRICT_INT8_EXECUTION,
            scoring::{Int8ScoringBudget, Int8ScoringError}}},
    tasks::{classify::{ClassificationLimits, ClassificationPlanner, ClassificationPlanningError,
        quantized::{Int8ClassificationError, Int8ClassificationRun, PreparedInt8Classification}},
        ir::{PlanContext, TaskBudget}}};
use super::{ClassificationBatchArgs, GuardedOutput};

/// The host's existing admission authority must price the WHOLE model work,
/// not only head rows. This hook creates no broker, runtime, or model loader.
/// The returned real guard covers task/output storage through sink delivery.
pub trait Int8ClassificationAdmission {
    type Guard;
    fn admit(&mut self, proposed: &ExecutionIdentity, work: Int8Work)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure>;
}
pub struct Int8ClassificationBatchPlanner<'p> {
    planner: &'p ClassificationPlanner, identity: ExecutionIdentity,
    ceiling: TaskBudget, limits: ClassificationLimits,
    defaults: Option<ClassificationBatchArgs>, binding: Sha256Digest,
}
pub struct PreparedInt8BatchClassification {
    plan: PreparedInt8Classification, factory_binding: Sha256Digest,
}
impl PreparedInt8BatchClassification {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.plan.execution_identity() }
    pub fn model_work(&self) -> Int8Work { self.plan.planned_work() }
    pub fn head_count(&self) -> usize { self.plan.head_count() }
}
impl<'p> Int8ClassificationBatchPlanner<'p> {
    pub fn new(planner: &'p ClassificationPlanner, identity: ExecutionIdentity, ceiling: TaskBudget,
        limits: ClassificationLimits, defaults: Option<ClassificationBatchArgs>) -> Result<Self, BatchFault> {
        identity.validate().map_err(|_| BatchCode::Admission)?;
        ceiling.validate().map_err(|_| BatchCode::InvalidLimits)?;
        if identity.task_spec != "classify-v1" || identity.template_digest != *planner.template_digest()
            || identity.tokenizer_digest != planner.tokenizer_digest()
            || identity.numerics_profile != (NumericsProfile::StrictQuantized { version: 1 })
            || identity.backend_semantic_version != STRICT_INT8_EXECUTION || identity.kv_dtype != "bf16"
            || identity.thinking_mode != ThinkingMode::Disabled || identity.tool_mode != ToolMode::None {
            return Err(BatchCode::Admission.into());
        }
        let binding = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            "int8-classification-batch-factory-v1", &identity, ceiling,
            (limits.max_labels, limits.max_input_bytes, limits.max_label_id_bytes, limits.max_label_description_bytes,
                limits.max_total_label_bytes, limits.max_context_tokens, limits.max_total_prompt_tokens, limits.max_work),
            (limits.scoring.max_candidates, limits.scoring.max_total_tokens, limits.scoring.max_nodes,
                limits.scoring.max_depth, limits.scoring.max_candidate_id_bytes, limits.scoring.max_projected_logits),
        )).map_err(|_| BatchCode::Serialization)?);
        Ok(Self { planner, identity, ceiling, limits, defaults, binding })
    }
    pub fn prepare(&self, document: BatchDocument<ClassificationBatchArgs>) -> Result<PreparedInt8BatchClassification, BatchItemFailure> {
        self.prepare_with_control(document, &mut Continue)
    }
    /// Corpus callers lend their actual controller; standalone prepare retains
    /// its explicitly uncontrolled compatibility behavior.
    pub fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<ClassificationBatchArgs>,
        control: &mut C) -> Result<PreparedInt8BatchClassification, BatchItemFailure> {
        checkpoint(control)?;
        let args = document.task_args.or_else(|| self.defaults.clone()).ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        let context = PlanContext::new(&self.identity, self.ceiling).map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        let plan = self.planner.plan_int8_with_control(&args.into_request(document.text), &context, self.limits, control)
            .map_err(planning_failure)?;
        checkpoint(control)?;
        Ok(PreparedInt8BatchClassification { plan, factory_binding: self.binding })
    }
    fn verify(&self, prepared: &PreparedInt8BatchClassification) -> Result<(), BatchItemFailure> {
        if prepared.factory_binding != self.binding { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        Ok(())
    }
}

pub struct NativeInt8ClassificationBatch<'p, 'e, 'w, A: Int8ClassificationAdmission> {
    compiler: Int8ClassificationBatchPlanner<'p>, engine: &'e mut StrictInt8Engine<'w>,
    admission: A, ledger: WorkLedger,
}
impl<'p, 'e, 'w, A: Int8ClassificationAdmission> NativeInt8ClassificationBatch<'p, 'e, 'w, A> {
    pub fn new(compiler: Int8ClassificationBatchPlanner<'p>, engine: &'e mut StrictInt8Engine<'w>,
        admission: A, total_work: Int8Work) -> Result<Self, BatchFault> {
        if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) { return Err(BatchCode::InvalidExecution.into()); }
        Ok(Self { compiler, engine, admission, ledger: WorkLedger { remaining: total_work, state: State::Ready } })
    }
    pub fn remaining_work(&self) -> Int8Work { self.ledger.remaining }
    pub fn ready(&self) -> bool { self.ledger.state == State::Ready }
}
impl<A: Int8ClassificationAdmission> BatchProcessor for NativeInt8ClassificationBatch<'_, '_, '_, A> {
    type Args = ClassificationBatchArgs;
    type Prepared = PreparedInt8BatchClassification;
    type Output = GuardedOutput<Int8ClassificationRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.prepare_with_control(document, &mut Continue)
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.ready()?;
        // A panic during compilation must not leave the adapter reusable.
        self.ledger.state = State::Running;
        let result = (|| {
            let prepared = self.compiler.prepare_with_control(document, control)?;
            subtract(self.ledger.remaining, prepared.model_work())?;
            prepared.plan.preflight(prepared.execution_identity(), self.engine, budget(&prepared.plan))
                .map_err(planning_failure)?;
            checkpoint(control)?;
            Ok(prepared)
        })();
        self.ledger.finish(result.as_ref().err());
        result
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { transport_work(prepared.model_work()) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        self.compiler.verify(&prepared)?;
        self.ledger.begin(prepared.model_work())?;
        // Set Running before all fallible host/native callbacks. Failed work
        // remains charged; an unwind never restores a reusable adapter.
        let result = with_admission(&mut self.admission, &prepared.plan, control, |admitted, control| {
            let run = prepared.plan.execute_with_control(admitted, self.engine, budget(&prepared.plan), control)
                .map_err(execution_failure)?;
            if self.engine.is_poisoned() || !self.engine.kv_cache().all_slots_have_len(0)
                || run.model_work != prepared.model_work() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
            Ok(run)
        });
        self.ledger.finish(result.as_ref().err()); result
    }
}
fn with_admission<A: Int8ClassificationAdmission, C: DecodeStepControl, T, F>(admission: &mut A,
    plan: &PreparedInt8Classification, control: &mut C, execute: F) -> Result<GuardedOutput<T, A::Guard>, BatchItemFailure>
where F: FnOnce(&ExecutionIdentity, &mut C) -> Result<T, BatchItemFailure> {
    checkpoint(control)?;
    let (admitted, guard) = admission.admit(plan.execution_identity(), plan.planned_work())?;
    plan.verify_identity(&admitted).map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    checkpoint(control)?;
    let value = execute(&admitted, control)?;
    // A deadline expiring during finalization suppresses the completed value.
    checkpoint(control)?;
    Ok(GuardedOutput::new(value, guard))
}
fn budget(plan: &PreparedInt8Classification) -> Int8ScoringBudget {
    Int8ScoringBudget { native: Int8RunBudget::exact(plan.planned_work()), max_kv_bytes: plan.task_budget().max_kv_bytes }
}
fn transport_work(w: Int8Work) -> BatchWork { BatchWork { forward_positions: w.forward_positions, projected_logits: w.projected_logits } }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State { Ready, Running, Failed }
struct WorkLedger { remaining: Int8Work, state: State }
impl WorkLedger {
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.state == State::Ready { Ok(()) } else { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) }
    }
    fn begin(&mut self, work: Int8Work) -> Result<(), BatchItemFailure> {
        self.ready()?;
        let remaining = subtract(self.remaining, work)?;
        self.remaining = remaining; self.state = State::Running; Ok(())
    }
    fn finish(&mut self, failure: Option<&BatchItemFailure>) {
        self.state = if self.state != State::Running || failure.is_some_and(|e| e.stop) { State::Failed } else { State::Ready };
    }
}
fn subtract(a: Int8Work, b: Int8Work) -> Result<Int8Work, BatchItemFailure> {
    let sub = |a: u64, b: u64| a.checked_sub(b).ok_or_else(|| BatchItemFailure::reject(BatchCode::WorkLimit));
    Ok(Int8Work { forward_positions: sub(a.forward_positions, b.forward_positions)?,
        projected_logits: sub(a.projected_logits, b.projected_logits)?, attention_pairs: sub(a.attention_pairs, b.attention_pairs)?,
        projections: ProjectionWork { dot_products: sub(a.projections.dot_products, b.projections.dot_products)?,
            multiply_accumulates: sub(a.projections.multiply_accumulates, b.projections.multiply_accumulates)? } })
}
fn planning_failure(e: Int8ClassificationError) -> BatchItemFailure {
    if let Some(cause) = e.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match e {
        Int8ClassificationError::Planning(e) => super::planning_failure(e),
        Int8ClassificationError::WorkBudget | Int8ClassificationError::Native(StrictInt8Error::Work)
            => BatchItemFailure::reject(BatchCode::WorkLimit),
        Int8ClassificationError::Native(StrictInt8Error::Context | StrictInt8Error::Memory)
            | Int8ClassificationError::Scoring(Int8ScoringError::Native(StrictInt8Error::Context | StrictInt8Error::Memory))
            => BatchItemFailure::reject(BatchCode::Admission),
        Int8ClassificationError::Identity => BatchItemFailure::fatal(BatchCode::Admission),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    }
}
fn execution_failure(e: Int8ClassificationError) -> BatchItemFailure {
    if let Some(cause) = e.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match e {
        Int8ClassificationError::Identity | Int8ClassificationError::Planning(ClassificationPlanningError::Identity)
            => BatchItemFailure::fatal(BatchCode::Admission),
        Int8ClassificationError::Accounting => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        _ => BatchItemFailure::fatal(BatchCode::Execution),
    }
}
fn checkpoint<C: DecodeStepControl>(c: &mut C) -> Result<(), BatchItemFailure> {
    match c.prefill_checkpoint(0) { Some(reason) => Err(BatchItemFailure::fatal(BatchFault::cancelled(reason))), None => Ok(()) }
}
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }

#[cfg(test)] mod tests;

#[cfg(test)] mod control_tests;
