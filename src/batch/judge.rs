//! Stream raw pairwise/rubric/faithfulness requests through the existing pinned
//! task planner and one ALREADY admitted eager engine. No loader or alternate
//! judge implementation. The host's actual admission guard remains live until
//! native completion, result validation AND delivery by the batch writer.

use crate::{
    execution_identity::{ExecutionIdentity, NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{decode::DecodeStepControl, hf_bf16_eager::{HfBf16EagerEngine,
        candidate_scoring::{PrefixBudget, PrefixScoringError}}},
    tasks::{ir::{PlanContext, TaskBudget}, judge::{JudgePlanner, JudgeRequest, JudgeResult,
        PreparedJudge, JudgeLimits, JudgeError, JudgeNativeError, EagerJudgeRun,
        PairwisePolicy, RubricDefinition, RubricPolicy, FaithfulnessPolicy}},
};
use super::*;
pub use super::output::GuardedOutput;

/// `text` supplies A in pairwise mode, the rubric document, or the complete
/// faithfulness source. All other task data have the SAME types and semantics
/// as JudgeRequest. Data never enters a trusted renderer as instructions.
#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum JudgeBatchArgs {
    Pairwise { criterion: String, b: String, policy: PairwisePolicy, budget: TaskBudget },
    Rubric { rubric: RubricDefinition, policy: RubricPolicy, budget: TaskBudget },
    Faithfulness { claim: String, policy: FaithfulnessPolicy, budget: TaskBudget },
}
impl JudgeBatchArgs {
    pub(crate) fn into_request(self, text: String) -> JudgeRequest {
        match self {
            Self::Pairwise { criterion, b, policy, budget } => JudgeRequest::Pairwise { criterion, a: text, b, policy, budget },
            Self::Rubric { rubric, policy, budget } => JudgeRequest::Rubric { document: text, rubric, policy, budget },
            Self::Faithfulness { claim, policy, budget } => JudgeRequest::Faithfulness { source: text, claim, policy, budget },
        }
    }
}

/// Reusable model-free planner with one fixed host ceiling and identity. It
/// retains no item after prepare returns. No task may raise that host ceiling.
pub struct JudgeBatchPlanner<'a> {
    planner: &'a JudgePlanner,
    identity: ExecutionIdentity,
    ceiling: TaskBudget,
    limits: JudgeLimits,
    defaults: Option<JudgeBatchArgs>,
}
/// Private prepared content and an exact cold-head work ceiling. No Debug or
/// wire constructor; a stream record cannot smuggle executable TaskIR/identity.
pub struct PreparedBatchJudge { plan: PreparedJudge, work: BatchWork }
impl PreparedBatchJudge {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.plan.execution_identity() }
    pub fn planned_work(&self) -> BatchWork { self.work }
}
impl<'a> JudgeBatchPlanner<'a> {
    pub fn new(planner: &'a JudgePlanner, identity: ExecutionIdentity, ceiling: TaskBudget,
        limits: JudgeLimits, defaults: Option<JudgeBatchArgs>) -> Result<Self, BatchFault> {
        identity.validate().map_err(|_| BatchCode::Admission)?;
        ceiling.validate().map_err(|_| BatchCode::InvalidLimits)?;
        if identity.task_spec != "judge-v1" || identity.template_digest != *planner.template_digest()
            || identity.tokenizer_digest != planner.tokenizer_digest() || identity.numerics_profile != NumericsProfile::HfBf16Eager
            || identity.kv_dtype != "bf16" || identity.thinking_mode != ThinkingMode::Disabled || identity.tool_mode != ToolMode::None {
            return Err(BatchCode::Admission.into());
        }
        Ok(Self { planner, identity, ceiling, limits, defaults })
    }
    pub fn prepare(&self, document: BatchDocument<JudgeBatchArgs>) -> Result<PreparedBatchJudge, BatchItemFailure> {
        let args = document.task_args.or_else(|| self.defaults.clone())
            .ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        let request = args.into_request(document.text);
        let context = PlanContext::new(&self.identity, self.ceiling)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        let plan = self.planner.plan(&request, &context, self.limits).map_err(|error| match error {
            JudgeError::AllocationRefused => BatchItemFailure::fatal(BatchCode::Allocation),
            _ => BatchItemFailure::reject(BatchCode::Planning),
        })?;
        let budget = plan.planned_native_budget().map_err(native_failure)?;
        let work = BatchWork { forward_positions: budget.max_forward_positions, projected_logits: budget.max_projected_logits };
        Ok(PreparedBatchJudge { plan, work })
    }
}

/// Embedding hook, NOT a substitute PermitBroker or an activation certificate.
/// Implement using the embedding host's ratified admission path and actual
/// guard type. `admit` returns the exact identity that host admitted plus the
/// guard it acquired. No default implementation fabricates either authority.
/// Failure is recoverable only when the host has fully drained/released work.
pub trait JudgeBatchAdmission {
    type Guard;
    fn admit(&mut self, proposed: &ExecutionIdentity, work: BatchWork)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure>;
}

pub struct NativeJudgeBatch<'p, 'e, A: JudgeBatchAdmission> {
    compiler: JudgeBatchPlanner<'p>,
    engine: &'e mut HfBf16EagerEngine,
    admission: A,
}
impl<'p, 'e, A: JudgeBatchAdmission> NativeJudgeBatch<'p, 'e, A> {
    pub fn new(compiler: JudgeBatchPlanner<'p>, engine: &'e mut HfBf16EagerEngine, admission: A)
        -> Result<Self, BatchFault> {
        if !engine.kv_cache().all_slots_have_len(0) { return Err(BatchCode::InvalidExecution.into()); }
        Ok(Self { compiler, engine, admission })
    }
}
impl<A: JudgeBatchAdmission> BatchProcessor for NativeJudgeBatch<'_, '_, A> {
    type Args = JudgeBatchArgs;
    type Prepared = PreparedBatchJudge;
    type Output = GuardedOutput<EagerJudgeRun<JudgeResult>, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        let prepared = self.compiler.prepare(document)?;
        // Capacity/full-reservation checks have no model calls or mutations.
        // The host still performs its real admission later, after the stream's
        // aggregate work has been reserved rather than renewed for each item.
        prepared.plan.preflight_eager(prepared.execution_identity(), self.engine, prefix_budget(prepared.work))
            .map_err(native_failure)?;
        Ok(prepared)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { prepared.work }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        let (identity, guard) = self.admission.admit(prepared.execution_identity(), prepared.work)?;
        prepared.plan.verify_identity(&identity).map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        let budget = prefix_budget(prepared.work);
        prepared.plan.preflight_eager(&identity, self.engine, budget).map_err(native_failure)?;
        let result = prepared.plan.execute_eager_with_control(&identity, self.engine, budget, control);
        if !self.engine.kv_cache().all_slots_have_len(0) {
            // Do not clear/reuse a possibly incomplete branch or keep admitting.
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let run = result.map_err(native_failure)?;
        if run.native_work.forward_positions != prepared.work.forward_positions
            || run.native_work.projected_logits != prepared.work.projected_logits {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        // Preserve output-memory/admission ownership until the stream has
        // serialized, written and flushed this result, including sink errors.
        Ok(GuardedOutput::new(run, guard))
    }
}
fn prefix_budget(work: BatchWork) -> PrefixBudget {
    PrefixBudget { max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits }
}
fn native_failure(error: JudgeNativeError) -> BatchItemFailure {
    match error {
        JudgeNativeError::Native(PrefixScoringError::Cancelled(cause)) => BatchItemFailure::fatal(BatchFault::cancelled(cause)),
        JudgeNativeError::Native(PrefixScoringError::ContextBudget | PrefixScoringError::KvBudget) => BatchItemFailure::reject(BatchCode::Admission),
        JudgeNativeError::Native(PrefixScoringError::WorkBudget) => BatchItemFailure::reject(BatchCode::WorkLimit),
        JudgeNativeError::Native(PrefixScoringError::OutputBudget) | JudgeNativeError::Task(JudgeError::Limit(_)) => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        JudgeNativeError::Native(PrefixScoringError::AllocationRefused) | JudgeNativeError::Task(JudgeError::AllocationRefused) => BatchItemFailure::fatal(BatchCode::Allocation),
        JudgeNativeError::Native(PrefixScoringError::Engine(_) | PrefixScoringError::Head(_)) => BatchItemFailure::fatal(BatchCode::Execution),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    }
}

#[cfg(test)]
mod tests;
