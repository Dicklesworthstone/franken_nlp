//! Closed adapters for the existing three model-bound scoring planners.
use super::*;
use serde::{Serialize, de::DeserializeOwned};
use crate::{
    batch::{classify::ClassificationBatchArgs, judge::JudgeBatchArgs},
    native_engine::strict_int4::{StrictInt4Error, STRICT_INT4_PROFILE, scoring::Int4ScoringError},
    tasks::{
        classify::{ClassificationRequest, ClassificationPlanningError,
            grouped_int4::{Int4ClassificationPlanner, PreparedInt4Classification,
                Int4ClassificationRun, Int4ClassificationError, INT4_CLASSIFICATION_EXECUTION}},
        sentiment::{SentimentRequest, SentimentPlanningError, SentimentError,
            batch::SentimentBatchArgs,
            grouped_int4::{Int4SentimentPlanner, PreparedInt4Sentiment,
                Int4SentimentRun, Int4SentimentError, INT4_SENTIMENT_EXECUTION}},
        judge::{JudgeError, grouped_int4::{Int4JudgePlanner, PreparedInt4Judge,
            Int4JudgeRun, Int4JudgeError, INT4_JUDGE_EXECUTION}},
    },
};
pub(super) mod sealed { pub trait Sealed {} }

/// Closed set of native task compilers. External implementations cannot inject
/// an alternative scorer or fabricate a prepared plan through the batch route.
/// The associated Args preserve each existing task's strict NDJSON shape.
pub trait Int4ScoredTask: sealed::Sealed + Send + 'static {
    type Args: Clone + DeserializeOwned + Send + 'static;
    type Plan: Send + 'static;
    type Output: Serialize + Send + 'static;
    const TASK: &'static str;
    fn base_identity(&self) -> &ExecutionIdentity;
    fn group_width(&self) -> usize;
    fn plan_document<C: DecodeStepControl>(&self, text: String, args: Self::Args, control: &mut C)
        -> Result<Self::Plan, BatchItemFailure>;
    fn facts(plan: &Self::Plan) -> Int4ScoredPlanFacts<'_>;
    fn verify_identity(plan: &Self::Plan, admitted: &ExecutionIdentity) -> Result<(), BatchItemFailure>;
    fn preflight(plan: &Self::Plan, engine: &StrictInt4Engine<'_>, budget: Int4ScoringBudget)
        -> Result<(), BatchItemFailure>;
    fn execute<C: DecodeStepControl>(plan: &Self::Plan, admitted: &ExecutionIdentity,
        engine: &mut StrictInt4Engine<'_>, budget: Int4ScoringBudget, control: &mut C)
        -> Result<Self::Output, BatchItemFailure>;
    fn completed(plan: &Self::Plan, output: &Self::Output) -> bool;
}

// The three sealed native plan APIs have the same ownership/accounting shape.
// Only request construction, task-specific result cap and typed error mapping
// differ; semantic finalization stays in the existing task implementation.
macro_rules! native_plan_methods {
    ($cap:expr, $error:ident, $execution:ident) => {
        fn base_identity(&self) -> &ExecutionIdentity { self.execution_identity() }
        fn group_width(&self) -> usize { self.group_width() }
        fn facts(plan: &Self::Plan) -> Int4ScoredPlanFacts<'_> {
            Int4ScoredPlanFacts { identity: plan.execution_identity(), model_work: plan.planned_work(),
                group_width: plan.group_width(), head_count: plan.head_count(),
                required_context: plan.required_context(), task_budget: plan.task_budget(),
                max_result_bytes: ($cap)(plan) }
        }
        fn verify_identity(plan: &Self::Plan, admitted: &ExecutionIdentity) -> Result<(), BatchItemFailure> {
            plan.verify_identity(admitted).map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))
        }
        fn preflight(plan: &Self::Plan, engine: &StrictInt4Engine<'_>, budget: Int4ScoringBudget)
            -> Result<(), BatchItemFailure> {
            plan.preflight(plan.execution_identity(), engine, budget).map_err(|e| $error(e, false))
        }
        fn execute<C: DecodeStepControl>(plan: &Self::Plan, admitted: &ExecutionIdentity,
            engine: &mut StrictInt4Engine<'_>, budget: Int4ScoringBudget, control: &mut C)
            -> Result<Self::Output, BatchItemFailure> {
            plan.execute_with_control(admitted, engine, budget, control).map_err(|e| $error(e, true))
        }
        fn completed(plan: &Self::Plan, output: &Self::Output) -> bool {
            output.schema_version == 1 && output.execution == $execution
                && output.numerics_profile == STRICT_INT4_PROFILE && output.group_width == plan.group_width()
                && output.head_count == plan.head_count() && output.model_work == plan.planned_work()
        }
    };
}
impl sealed::Sealed for Int4ClassificationPlanner {}
impl Int4ScoredTask for Int4ClassificationPlanner {
    type Args = ClassificationBatchArgs;
    type Plan = PreparedInt4Classification;
    type Output = Int4ClassificationRun;
    const TASK: &'static str = "classify-v1";
    fn plan_document<C: DecodeStepControl>(&self, text: String, args: Self::Args, control: &mut C)
        -> Result<Self::Plan, BatchItemFailure> {
        let request = ClassificationRequest { document: text, labels: args.labels, mode: args.mode,
            policy: args.policy, budget: args.budget };
        self.plan_with_control(&request, control).map_err(|e| classification_error(e, false))
    }
    native_plan_methods!(|p: &PreparedInt4Classification| p.task_budget().max_output_bytes,
        classification_error, INT4_CLASSIFICATION_EXECUTION);
}
impl sealed::Sealed for Int4SentimentPlanner {}
impl Int4ScoredTask for Int4SentimentPlanner {
    type Args = SentimentBatchArgs;
    type Plan = PreparedInt4Sentiment;
    type Output = Int4SentimentRun;
    const TASK: &'static str = "sentiment-v1";
    fn plan_document<C: DecodeStepControl>(&self, text: String, args: Self::Args, control: &mut C)
        -> Result<Self::Plan, BatchItemFailure> {
        let request = SentimentRequest { document: text, axes: args.axes, budget: args.budget };
        self.plan_with_control(&request, control).map_err(|e| sentiment_error(e, false))
    }
    native_plan_methods!(|p: &PreparedInt4Sentiment| p.max_result_bytes(), sentiment_error, INT4_SENTIMENT_EXECUTION);
}
impl sealed::Sealed for Int4JudgePlanner {}
impl Int4ScoredTask for Int4JudgePlanner {
    type Args = JudgeBatchArgs;
    type Plan = PreparedInt4Judge;
    type Output = Int4JudgeRun;
    const TASK: &'static str = "judge-v1";
    fn plan_document<C: DecodeStepControl>(&self, text: String, args: Self::Args, control: &mut C)
        -> Result<Self::Plan, BatchItemFailure> {
        self.plan_with_control(&args.into_request(text), control).map_err(|e| judge_error(e, false))
    }
    native_plan_methods!(|p: &PreparedInt4Judge| p.max_result_bytes(), judge_error, INT4_JUDGE_EXECUTION);
}

fn phase(mut failure: BatchItemFailure, executing: bool) -> BatchItemFailure {
    // Once any native bundle starts, a failure cannot be reinterpreted as a
    // recoverable planning rejection or a successful semantic abstention.
    if executing { failure.stop = true; }
    failure
}
fn classification_error(error: Int4ClassificationError, executing: bool) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    let failure = match error {
        Int4ClassificationError::Identity | Int4ClassificationError::Planning(ClassificationPlanningError::Identity)
            => BatchItemFailure::fatal(BatchCode::Admission),
        Int4ClassificationError::Planning(ClassificationPlanningError::AllocationRefused)
            => BatchItemFailure::fatal(BatchCode::Allocation),
        Int4ClassificationError::WorkBudget | Int4ClassificationError::Native(StrictInt4Error::Work)
            | Int4ClassificationError::Planning(ClassificationPlanningError::WorkBudget)
            => BatchItemFailure::reject(BatchCode::WorkLimit),
        Int4ClassificationError::Native(StrictInt4Error::Context | StrictInt4Error::Memory)
            | Int4ClassificationError::Scoring(Int4ScoringError::Native(StrictInt4Error::Context | StrictInt4Error::Memory))
            => BatchItemFailure::reject(BatchCode::Admission),
        Int4ClassificationError::Planning(ClassificationPlanningError::OutputBudget)
            => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        Int4ClassificationError::Planning(ClassificationPlanningError::InvalidLimits | ClassificationPlanningError::InvalidRequest
            | ClassificationPlanningError::InputBudget | ClassificationPlanningError::ContextBudget | ClassificationPlanningError::Task(_))
            => BatchItemFailure::reject(BatchCode::Planning),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    };
    phase(failure, executing)
}
fn sentiment_error(error: Int4SentimentError, executing: bool) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    let failure = match error {
        Int4SentimentError::Identity | Int4SentimentError::Planning(SentimentPlanningError::Identity)
            => BatchItemFailure::fatal(BatchCode::Admission),
        Int4SentimentError::Allocation | Int4SentimentError::Planning(SentimentPlanningError::AllocationRefused)
            | Int4SentimentError::Task(SentimentError::AllocationRefused)
            => BatchItemFailure::fatal(BatchCode::Allocation),
        Int4SentimentError::WorkBudget | Int4SentimentError::Native(StrictInt4Error::Work)
            => BatchItemFailure::reject(BatchCode::WorkLimit),
        Int4SentimentError::Native(StrictInt4Error::Context | StrictInt4Error::Memory)
            | Int4SentimentError::Scoring(Int4ScoringError::Native(StrictInt4Error::Context | StrictInt4Error::Memory))
            => BatchItemFailure::reject(BatchCode::Admission),
        Int4SentimentError::OutputBudget => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        Int4SentimentError::Planning(SentimentPlanningError::InvalidAxes | SentimentPlanningError::EmptyDocument
            | SentimentPlanningError::Budget | SentimentPlanningError::Task(_))
            => BatchItemFailure::reject(BatchCode::Planning),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    };
    phase(failure, executing)
}
fn judge_error(error: Int4JudgeError, executing: bool) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    let failure = match error {
        Int4JudgeError::Identity => BatchItemFailure::fatal(BatchCode::Admission),
        Int4JudgeError::Task(JudgeError::AllocationRefused) => BatchItemFailure::fatal(BatchCode::Allocation),
        Int4JudgeError::WorkBudget | Int4JudgeError::Native(StrictInt4Error::Work)
            => BatchItemFailure::reject(BatchCode::WorkLimit),
        Int4JudgeError::Native(StrictInt4Error::Context | StrictInt4Error::Memory)
            | Int4JudgeError::Scoring(Int4ScoringError::Native(StrictInt4Error::Context | StrictInt4Error::Memory))
            => BatchItemFailure::reject(BatchCode::Admission),
        Int4JudgeError::Task(JudgeError::Contract(_) | JudgeError::Limit(_))
            => BatchItemFailure::reject(BatchCode::Planning),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    };
    phase(failure, executing)
}
