//! Raw exclusive and independent multi-label classification on the existing
//! bounded NDJSON runner. One document is ONE complete classification bundle,
//! not a collection of provisional per-label results. No loader or runtime.

use crate::{
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::hf_bf16_eager::{HfBf16EagerEngine,
        candidate_scoring::{PrefixBudget, PrefixScoringError}},
    tasks::{classify::{ClassificationError, ClassificationLabel, ClassificationLimits,
        ClassificationMode, ClassificationNativeError, ClassificationPlanner, ClassificationPlanningError,
        ClassificationPolicy, ClassificationRequest, EagerClassificationRun, PreparedClassification},
        ir::{PlanContext, TaskBudget}},
};
use super::*;
pub use super::judge::{GuardedOutput, JudgeBatchAdmission as ClassificationBatchAdmission};

/// `BatchDocument.text` is the original document. Arguments contain only task
/// data and shrinking ceilings; callers cannot supply tokens, EOS, identity,
/// template code, tools, or an alternate scoring-probability interpretation.
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClassificationBatchArgs {
    pub labels: Vec<ClassificationLabel>,
    pub mode: ClassificationMode,
    pub policy: ClassificationPolicy,
    pub budget: TaskBudget,
}
impl ClassificationBatchArgs {
    fn into_request(self, document: String) -> ClassificationRequest {
        ClassificationRequest { document, labels: self.labels, mode: self.mode,
            policy: self.policy, budget: self.budget }
    }
}

pub struct ClassificationBatchPlanner<'p> {
    planner: &'p ClassificationPlanner,
    identity: ExecutionIdentity,
    ceiling: TaskBudget,
    limits: ClassificationLimits,
    defaults: Option<ClassificationBatchArgs>,
    binding: Sha256Digest,
}
pub struct PreparedBatchClassification { plan: PreparedClassification, work: BatchWork, factory_binding: Sha256Digest }
impl PreparedBatchClassification {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.plan.execution_identity() }
    pub fn planned_work(&self) -> BatchWork { self.work }
    pub fn head_count(&self) -> usize { self.plan.head_count() }
}
impl<'p> ClassificationBatchPlanner<'p> {
    /// Freeze the host identity and ceilings. Per-record data, including an
    /// explicitly supplied default, still passes the full raw task planner.
    pub fn new(planner: &'p ClassificationPlanner, identity: ExecutionIdentity, ceiling: TaskBudget,
        limits: ClassificationLimits, defaults: Option<ClassificationBatchArgs>) -> Result<Self, BatchFault> {
        identity.validate().map_err(|_| BatchCode::Admission)?;
        ceiling.validate().map_err(|_| BatchCode::InvalidLimits)?;
        if identity.task_spec != "classify-v1" || identity.template_digest != *planner.template_digest()
            || identity.tokenizer_digest != planner.tokenizer_digest() || identity.numerics_profile != NumericsProfile::HfBf16Eager
            || identity.kv_dtype != "bf16" || identity.thinking_mode != ThinkingMode::Disabled || identity.tool_mode != ToolMode::None {
            return Err(BatchCode::Admission.into());
        }
        // Private factory binding, NOT authority. It prevents a prepared
        // bundle compiled under another host/limit configuration from being
        // fed directly into this adapter through the public Rust API.
        let binding = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            "classification-batch-factory-v1", &identity, ceiling,
            (limits.max_labels, limits.max_input_bytes, limits.max_label_id_bytes,
                limits.max_label_description_bytes, limits.max_total_label_bytes,
                limits.max_context_tokens, limits.max_total_prompt_tokens, limits.max_work),
            (limits.scoring.max_candidates, limits.scoring.max_total_tokens, limits.scoring.max_nodes,
                limits.scoring.max_depth, limits.scoring.max_candidate_id_bytes, limits.scoring.max_projected_logits),
        )).map_err(|_| BatchCode::Serialization)?);
        Ok(Self { planner, identity, ceiling, limits, defaults, binding })
    }
    pub fn prepare(&self, document: BatchDocument<ClassificationBatchArgs>) -> Result<PreparedBatchClassification, BatchItemFailure> {
        let args = document.task_args.or_else(|| self.defaults.clone()).ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        let request = args.into_request(document.text);
        let context = PlanContext::new(&self.identity, self.ceiling).map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        let plan = self.planner.plan(&request, &context, self.limits).map_err(planning_failure)?;
        let work = plan.planned_work();
        Ok(PreparedBatchClassification { plan, work, factory_binding: self.binding })
    }
    fn verify_prepared(&self, prepared: &PreparedBatchClassification) -> Result<(), BatchItemFailure> {
        if prepared.factory_binding != self.binding || prepared.work != prepared.plan.planned_work() {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        Ok(())
    }
}

/// The host supplies its existing genuine admission hook and one admitted
/// eager engine. The guard must cover task/output storage through delivery.
/// The runner also applies its independent stream limits; these TWO ceilings
/// constrain the SAME work, not two allocations or a renewable per-item grant.
pub struct NativeClassificationBatch<'p, 'e, A: ClassificationBatchAdmission> {
    compiler: ClassificationBatchPlanner<'p>,
    engine: &'e mut HfBf16EagerEngine,
    admission: A,
    ledger: WorkLedger,
}
impl<'p, 'e, A: ClassificationBatchAdmission> NativeClassificationBatch<'p, 'e, A> {
    pub fn new(compiler: ClassificationBatchPlanner<'p>, engine: &'e mut HfBf16EagerEngine,
        admission: A, max_total_work: BatchWork) -> Result<Self, BatchFault> {
        if !engine.kv_cache().all_slots_have_len(0) { return Err(BatchCode::InvalidExecution.into()); }
        Ok(Self { compiler, engine, admission, ledger: WorkLedger { remaining: max_total_work, state: State::Ready } })
    }
    pub fn remaining_work(&self) -> BatchWork { self.ledger.remaining }
    pub fn ready(&self) -> bool { self.ledger.state == State::Ready }
}
impl<A: ClassificationBatchAdmission> BatchProcessor for NativeClassificationBatch<'_, '_, A> {
    type Args = ClassificationBatchArgs;
    type Prepared = PreparedBatchClassification;
    type Output = GuardedOutput<EagerClassificationRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.check_ready()?;
        let prepared = self.compiler.prepare(document)?;
        if !prepared.work.fits(self.ledger.remaining) { return Err(BatchItemFailure::reject(BatchCode::WorkLimit)); }
        prepared.plan.preflight_eager(prepared.execution_identity(), self.engine, prefix_budget(prepared.work)).map_err(native_failure)?;
        Ok(prepared)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { prepared.work }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.compiler.verify_prepared(&prepared)?;
        self.ledger.begin(prepared.work)?;
        // Running is set BEFORE admission and native work. Unwinding cannot
        // leave a reusable adapter, and neither failure nor flush refunds work.
        let result = (|| {
            checkpoint(control).map_err(BatchItemFailure::fatal)?;
            let (identity, guard) = self.admission.admit(prepared.execution_identity(), prepared.work)?;
            prepared.plan.verify_identity(&identity).map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
            let run = prepared.plan.execute_eager_with_control(&identity, self.engine, prefix_budget(prepared.work), control);
            if !self.engine.kv_cache().all_slots_have_len(0) { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
            let run = run.map_err(native_failure)?;
            if run.native_work.forward_positions != prepared.work.forward_positions
                || run.native_work.projected_logits != prepared.work.projected_logits {
                return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
            }
            Ok(GuardedOutput::new(run, guard))
        })();
        self.ledger.finish(result.as_ref().err());
        result
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State { Ready, Running, Failed }
struct WorkLedger { remaining: BatchWork, state: State }
impl WorkLedger {
    fn check_ready(&self) -> Result<(), BatchItemFailure> {
        if self.state == State::Ready { Ok(()) } else { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) }
    }
    fn begin(&mut self, work: BatchWork) -> Result<(), BatchItemFailure> {
        self.check_ready()?;
        let next = BatchWork {
            forward_positions: self.remaining.forward_positions.checked_sub(work.forward_positions)
                .ok_or_else(|| BatchItemFailure::reject(BatchCode::WorkLimit))?,
            projected_logits: self.remaining.projected_logits.checked_sub(work.projected_logits)
                .ok_or_else(|| BatchItemFailure::reject(BatchCode::WorkLimit))?,
        };
        self.remaining = next; self.state = State::Running; Ok(())
    }
    fn finish(&mut self, failure: Option<&BatchItemFailure>) {
        self.state = if self.state != State::Running || failure.is_some_and(|f| f.stop) { State::Failed } else { State::Ready };
    }
}
fn prefix_budget(work: BatchWork) -> PrefixBudget {
    PrefixBudget { max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits }
}
fn planning_failure(e: ClassificationPlanningError) -> BatchItemFailure {
    match e {
        ClassificationPlanningError::Cancelled(reason) => BatchItemFailure::fatal(BatchFault::cancelled(reason)),
        ClassificationPlanningError::AllocationRefused | ClassificationPlanningError::Task(ClassificationError::AllocationRefused)
            | ClassificationPlanningError::Task(ClassificationError::Scoring(crate::native_engine::lmhead::scoring::ScoringError::AllocationRefused))
            => BatchItemFailure::fatal(BatchCode::Allocation),
        ClassificationPlanningError::Identity => BatchItemFailure::fatal(BatchCode::Admission),
        ClassificationPlanningError::Accounting => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        ClassificationPlanningError::Serialization | ClassificationPlanningError::Task(ClassificationError::Serialization)
            => BatchItemFailure::fatal(BatchCode::Serialization),
        ClassificationPlanningError::WorkBudget => BatchItemFailure::reject(BatchCode::WorkLimit),
        ClassificationPlanningError::OutputBudget | ClassificationPlanningError::Task(ClassificationError::OutputBudgetExceeded)
            => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        _ => BatchItemFailure::reject(BatchCode::Planning),
    }
}
fn native_failure(e: ClassificationNativeError) -> BatchItemFailure {
    match e {
        ClassificationNativeError::Cancelled(reason) => BatchItemFailure::fatal(BatchFault::cancelled(reason)),
        ClassificationNativeError::ContextBudget | ClassificationNativeError::KvBudget
            | ClassificationNativeError::Native(PrefixScoringError::ContextBudget | PrefixScoringError::KvBudget)
            => BatchItemFailure::reject(BatchCode::Admission),
        ClassificationNativeError::WorkBudget | ClassificationNativeError::Native(PrefixScoringError::WorkBudget)
            => BatchItemFailure::reject(BatchCode::WorkLimit),
        ClassificationNativeError::Planning(ClassificationPlanningError::OutputBudget
            | ClassificationPlanningError::Task(ClassificationError::OutputBudgetExceeded))
            | ClassificationNativeError::Native(PrefixScoringError::OutputBudget)
            => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        ClassificationNativeError::Planning(ClassificationPlanningError::AllocationRefused
            | ClassificationPlanningError::Task(ClassificationError::AllocationRefused))
            | ClassificationNativeError::Native(PrefixScoringError::AllocationRefused)
            | ClassificationNativeError::Planning(ClassificationPlanningError::Task(ClassificationError::Scoring(
                crate::native_engine::lmhead::scoring::ScoringError::AllocationRefused)))
            => BatchItemFailure::fatal(BatchCode::Allocation),
        ClassificationNativeError::Native(PrefixScoringError::Cancelled(reason))
            | ClassificationNativeError::Planning(ClassificationPlanningError::Cancelled(reason))
            => BatchItemFailure::fatal(BatchFault::cancelled(reason)),
        ClassificationNativeError::Planning(ClassificationPlanningError::Identity) => BatchItemFailure::fatal(BatchCode::Admission),
        ClassificationNativeError::Planning(ClassificationPlanningError::Serialization
            | ClassificationPlanningError::Task(ClassificationError::Serialization)) => BatchItemFailure::fatal(BatchCode::Serialization),
        ClassificationNativeError::Native(PrefixScoringError::Engine(_) | PrefixScoringError::Head(_))
            | ClassificationNativeError::Planning(ClassificationPlanningError::Task(ClassificationError::Scoring(_)))
            => BatchItemFailure::fatal(BatchCode::Execution),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    }
}

#[cfg(test)]
mod tests;
pub mod quantized;
