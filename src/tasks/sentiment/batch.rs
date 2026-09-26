//! Resident, ordered INT8 sentiment on the existing bounded NDJSON runner.
//!
//! One complete independent-axis bundle is one document result. Policy and
//! score space belong to the run's pinned planner; records can narrow axes and
//! budgets, never replace model/template identity or normalize across axes.
use serde::{Deserialize, Serialize};
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, generation::GuardedOutput},
    canonjson,
    execution_identity::{ExecutionIdentity, Sha256Digest},
    native_engine::{constrained_int8,
        decode::DecodeStepControl,
        kv::KV_BYTES_PER_TOKEN,
        portable_int8::ProjectionWork,
        strict_int8::{Int8RunBudget, Int8Work, StrictInt8Engine, StrictInt8Error,
            STRICT_INT8_PROFILE, scoring::{Int8ScoringBudget, Int8ScoringError}}},
    tasks::ir::{PlanContext, TaskBudget},
};
use super::{SentimentAxis, SentimentError, SentimentLimits, SentimentPlanner,
    SentimentPlanningError, SentimentRequest,
    quantized::{Int8SentimentError, Int8SentimentRun, PreparedInt8Sentiment, INT8_SENTIMENT_EXECUTION}};

/// The document is BatchDocument.text. Policy/score space are run-wide, fixed
/// in SentimentPlanner; a per-item override replaces only these two fields.
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SentimentBatchArgs {
    pub axes: Vec<SentimentAxis>,
    pub budget: TaskBudget,
}

/// Owned configuration, not a wire admission capability. Whole-run native
/// ceilings are independent of per-document ceilings and never renew at flush.
/// No Debug/Serialize: base identity can contain private commitments.
pub struct SentimentBatchConfig {
    pub identity: ExecutionIdentity,
    pub task_ceiling: TaskBudget,
    pub planning: SentimentLimits,
    pub defaults: Option<SentimentBatchArgs>,
    pub max_item_work: Int8Work,
    pub max_model_work: Int8Work,
}
impl SentimentBatchConfig {
    pub fn validate(&self, planner: &SentimentPlanner) -> Result<(), BatchFault> {
        constrained_int8::check_profile(&self.identity).map_err(|_| BatchCode::Admission)?;
        if self.identity.task_spec != "sentiment-v1"
            || self.identity.template_digest != *planner.template_digest()
            || self.identity.tokenizer_digest != planner.tokenizer_digest() {
            return Err(BatchCode::Admission.into());
        }
        self.task_ceiling.validate().map_err(|_| BatchCode::InvalidLimits)?;
        validate_work(self.max_item_work)?;
        validate_work(self.max_model_work)?;
        if !fits(self.max_item_work, self.max_model_work) { return Err(BatchCode::InvalidLimits.into()); }
        let l = self.planning; let s = l.per_axis;
        if s.max_candidates == 0 || s.max_total_tokens == 0 || s.max_nodes == 0
            || s.max_depth == 0 || s.max_candidate_id_bytes == 0 || s.max_projected_logits == 0
            || l.max_total_prompt_tokens == 0 || l.max_total_candidate_tokens == 0
            || l.max_total_nodes == 0 || l.max_total_projected_logits == 0 || l.max_output_bytes == 0 {
            return Err(BatchCode::InvalidLimits.into());
        }
        if let Some(args) = &self.defaults { check_args(args, self.task_ceiling)?; }
        Ok(())
    }
}

pub struct Int8SentimentBatchPlanner<'p> {
    planner: &'p SentimentPlanner,
    config: SentimentBatchConfig,
    binding: Sha256Digest,
}
/// Only the actual compiler can construct a plan. The factory seal prevents
/// plans prepared under a different policy/ceiling from entering this adapter.
pub struct PreparedInt8BatchSentiment {
    plan: PreparedInt8Sentiment,
    axes: Vec<SentimentAxis>,
    binding: Sha256Digest,
}
impl PreparedInt8BatchSentiment {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.plan.execution_identity() }
    pub fn model_work(&self) -> Int8Work { self.plan.planned_work() }
    pub fn head_count(&self) -> usize { self.plan.head_count() }
}
impl<'p> Int8SentimentBatchPlanner<'p> {
    pub fn new(planner: &'p SentimentPlanner, config: SentimentBatchConfig) -> Result<Self, BatchFault> {
        config.validate(planner)?;
        let l = config.planning; let s = l.per_axis;
        let limits = (s.max_candidates, s.max_total_tokens, s.max_nodes, s.max_depth,
            s.max_candidate_id_bytes, s.max_projected_logits, l.max_total_prompt_tokens,
            l.max_total_candidate_tokens, l.max_total_nodes, l.max_total_projected_logits, l.max_output_bytes);
        let binding = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            "resident-int8-sentiment-factory-v1", &config.identity, config.task_ceiling,
            limits, config.max_item_work, config.max_model_work,
        )).map_err(|_| BatchCode::Serialization)?);
        Ok(Self { planner, config, binding })
    }
    pub fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<SentimentBatchArgs>,
        control: &mut C) -> Result<PreparedInt8BatchSentiment, BatchItemFailure> {
        checkpoint(control)?;
        // Defaults are validated before any per-document clone (at most four axes).
        let args = document.task_args.or_else(|| self.config.defaults.clone())
            .ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        check_args(&args, self.config.task_ceiling).map_err(BatchItemFailure::reject)?;
        let mut axes = args.axes; axes.sort_unstable();
        let request = SentimentRequest { document: document.text, axes: axes.clone(), budget: args.budget };
        let context = PlanContext::new(&self.config.identity, self.config.task_ceiling)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        let plan = self.planner.plan_int8_with_control(&request, &context, self.config.planning, control)
            .map_err(planning_failure)?;
        if !fits(plan.planned_work(), self.config.max_item_work) {
            return Err(BatchItemFailure::reject(BatchCode::WorkLimit));
        }
        checkpoint(control)?;
        Ok(PreparedInt8BatchSentiment { plan, axes, binding: self.binding })
    }
}

/// Actual process/output authority is supplied by the embedding host. The
/// entire native KV allocation is priced, not just a short input's used slots.
pub struct Int8SentimentAdmissionRequest<'a> {
    pub identity: &'a ExecutionIdentity,
    pub model_work: Int8Work,
    pub kv_reservation_bytes: u64,
    pub max_result_bytes: u64,
}
pub trait Int8SentimentAdmission {
    type Guard;
    fn admit(&mut self, request: Int8SentimentAdmissionRequest<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure>;
}

pub struct NativeInt8SentimentBatch<'p, 'e, 'w, A: Int8SentimentAdmission> {
    compiler: Int8SentimentBatchPlanner<'p>,
    driver: NativeDriver<'e, 'w>,
    admission: A,
    run: RunState,
}
impl<'p, 'e, 'w, A: Int8SentimentAdmission> NativeInt8SentimentBatch<'p, 'e, 'w, A> {
    pub fn new(compiler: Int8SentimentBatchPlanner<'p>, engine: &'e mut StrictInt8Engine<'w>,
        admission: A) -> Result<Self, BatchFault> {
        let driver = NativeDriver(engine);
        if !driver.clean() { return Err(BatchCode::InvalidExecution.into()); }
        let run = RunState::new(compiler.config.max_model_work);
        Ok(Self { compiler, driver, admission, run })
    }
    pub fn remaining_work(&self) -> Int8Work { self.run.remaining }
    pub fn is_poisoned(&self) -> bool { self.run.failed || !self.driver.clean() }
}
impl<A: Int8SentimentAdmission> BatchProcessor for NativeInt8SentimentBatch<'_, '_, '_, A> {
    type Args = SentimentBatchArgs;
    type Prepared = PreparedInt8BatchSentiment;
    type Output = GuardedOutput<Int8SentimentRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        // Never create a fresh/no-op controller for a run-owned preparation.
        self.run.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.ready()?; self.run.failed = true;
        let result = (|| {
            let prepared = self.compiler.prepare_with_control(document, control)?;
            preflight(&prepared, &self.driver)?;
            subtract(self.run.remaining, prepared.model_work())?;
            Ok(prepared)
        })();
        if self.driver.clean() && !result.as_ref().is_err_and(|e: &BatchItemFailure| e.stop) {
            self.run.failed = false;
        }
        result
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let w = prepared.model_work(); BatchWork { forward_positions: w.forward_positions, projected_logits: w.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.run.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        execute_admitted(prepared, self.compiler.binding, &mut self.driver,
            &mut self.admission, &mut self.run, context, control)
    }
}

struct RunState { remaining: Int8Work, last_sequence: u64, failed: bool }
impl RunState {
    fn new(remaining: Int8Work) -> Self { Self { remaining, last_sequence: 0, failed: false } }
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn begin(&mut self, work: Int8Work, context: BatchRequestContext) -> Result<(), BatchItemFailure> {
        self.ready()?; self.failed = true;
        if context.request_seq <= self.last_sequence || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let remaining = subtract(self.remaining, work).map_err(|e| BatchItemFailure::fatal(e.fault))?;
        // Commit all five counters together, before fallible host/native calls.
        self.remaining = remaining; self.last_sequence = context.request_seq;
        Ok(())
    }
}

// Only private lifecycle tests can substitute a driver. Public construction
// requires the actual StrictInt8Engine and its actual model/identity checks.
trait Driver {
    fn clean(&self) -> bool;
    fn capacity(&self) -> usize;
    fn execute<C: DecodeStepControl>(&mut self, plan: &PreparedInt8Sentiment,
        identity: &ExecutionIdentity, budget: Int8ScoringBudget, control: &mut C)
        -> Result<Int8SentimentRun, Int8SentimentError>;
}
struct NativeDriver<'e, 'w>(&'e mut StrictInt8Engine<'w>);
impl Driver for NativeDriver<'_, '_> {
    fn clean(&self) -> bool { !self.0.is_poisoned() && self.0.kv_cache().all_slots_have_len(0) }
    fn capacity(&self) -> usize { self.0.kv_cache().capacity_positions() }
    fn execute<C: DecodeStepControl>(&mut self, plan: &PreparedInt8Sentiment,
        identity: &ExecutionIdentity, budget: Int8ScoringBudget, control: &mut C)
        -> Result<Int8SentimentRun, Int8SentimentError> {
        plan.execute_with_control(identity, self.0, budget, control)
    }
}
fn preflight(plan: &PreparedInt8BatchSentiment, driver: &impl Driver) -> Result<u64, BatchItemFailure> {
    if !driver.clean() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
    let bytes = (driver.capacity() as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
        .ok_or_else(|| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    if plan.plan.required_context() > driver.capacity() || bytes == 0 || bytes > plan.plan.task_budget().max_kv_bytes {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    Ok(bytes)
}
#[allow(clippy::too_many_arguments)]
fn execute_admitted<D: Driver, A: Int8SentimentAdmission, C: DecodeStepControl>(
    prepared: PreparedInt8BatchSentiment, binding: Sha256Digest, driver: &mut D, admission: &mut A,
    run: &mut RunState, context: BatchRequestContext, control: &mut C,
) -> Result<GuardedOutput<Int8SentimentRun, A::Guard>, BatchItemFailure> {
    if prepared.binding != binding { run.failed = true; return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
    let work = prepared.model_work(); run.begin(work, context)?;
    let outcome = (|| {
        checkpoint(control)?;
        let kv_bytes = preflight(&prepared, driver)?;
        let (identity, guard) = admission.admit(Int8SentimentAdmissionRequest {
            identity: prepared.execution_identity(), model_work: work,
            kv_reservation_bytes: kv_bytes, max_result_bytes: prepared.plan.max_result_bytes(),
        })?;
        prepared.plan.verify_identity(&identity).map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        checkpoint(control)?;
        let result = driver.execute(&prepared.plan, &identity, Int8ScoringBudget {
            native: Int8RunBudget::exact(work), max_kv_bytes: kv_bytes,
        }, control).map_err(execution_failure)?;
        if result.schema_version != 1 || result.execution != INT8_SENTIMENT_EXECUTION
            || result.numerics_profile != STRICT_INT8_PROFILE || result.model_work != work
            || result.head_count != prepared.head_count() || result.result.task_spec != "sentiment-v1"
            || result.result.work.dimensions != result.head_count
            || result.result.work.projected_logits != work.projected_logits
            || !result.result.dimensions.iter().map(|d| d.axis).eq(prepared.axes.iter().copied()) {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        checkpoint(control)?;
        Ok(GuardedOutput::new(result, guard))
    })();
    match outcome {
        Ok(result) if driver.clean() => { run.failed = false; Ok(result) }
        Ok(_) => Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)),
        Err(mut error) => {
            if !driver.clean() { error.stop = true; }
            if !error.stop { run.failed = false; }
            Err(error)
        }
    }
}
fn check_args(args: &SentimentBatchArgs, ceiling: TaskBudget) -> Result<(), BatchFault> {
    args.budget.validate().map_err(|_| BatchCode::Planning)?;
    let b = args.budget;
    if args.axes.is_empty() || args.axes.len() > SentimentAxis::ALL.len()
        || args.axes.iter().enumerate().any(|(i, axis)| args.axes[..i].contains(axis))
        || b.max_input_tokens > ceiling.max_input_tokens || b.max_output_tokens > ceiling.max_output_tokens
        || b.max_output_bytes > ceiling.max_output_bytes || b.max_grammar_states > ceiling.max_grammar_states
        || b.max_kv_bytes > ceiling.max_kv_bytes {
        return Err(BatchCode::Planning.into());
    }
    Ok(())
}
pub(crate) fn validate_work(w: Int8Work) -> Result<(), BatchFault> {
    if w.forward_positions == 0 || w.projected_logits == 0 || w.attention_pairs == 0
        || w.projections.dot_products == 0 || w.projections.multiply_accumulates == 0 {
        Err(BatchCode::InvalidLimits.into())
    } else { Ok(()) }
}
fn fits(a: Int8Work, b: Int8Work) -> bool {
    a.forward_positions <= b.forward_positions && a.projected_logits <= b.projected_logits
        && a.attention_pairs <= b.attention_pairs && a.projections.fits(b.projections)
}
fn subtract(a: Int8Work, b: Int8Work) -> Result<Int8Work, BatchItemFailure> {
    let sub = |a: u64, b: u64| a.checked_sub(b).ok_or_else(|| BatchItemFailure::reject(BatchCode::WorkLimit));
    Ok(Int8Work { forward_positions: sub(a.forward_positions, b.forward_positions)?,
        projected_logits: sub(a.projected_logits, b.projected_logits)?, attention_pairs: sub(a.attention_pairs, b.attention_pairs)?,
        projections: ProjectionWork { dot_products: sub(a.projections.dot_products, b.projections.dot_products)?,
            multiply_accumulates: sub(a.projections.multiply_accumulates, b.projections.multiply_accumulates)? } })
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}
fn planning_failure(error: Int8SentimentError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int8SentimentError::Allocation | Int8SentimentError::Planning(SentimentPlanningError::AllocationRefused)
            | Int8SentimentError::Task(SentimentError::AllocationRefused)
            | Int8SentimentError::Native(StrictInt8Error::Allocation)
            | Int8SentimentError::Scoring(Int8ScoringError::Allocation)
            => BatchItemFailure::fatal(BatchCode::Allocation),
        Int8SentimentError::Identity | Int8SentimentError::Planning(SentimentPlanningError::Identity)
            => BatchItemFailure::fatal(BatchCode::Admission),
        Int8SentimentError::Accounting => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        Int8SentimentError::Serialization => BatchItemFailure::fatal(BatchCode::Serialization),
        _ => BatchItemFailure::reject(BatchCode::Planning),
    }
}
fn execution_failure(error: Int8SentimentError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int8SentimentError::Identity => BatchItemFailure::fatal(BatchCode::Admission),
        Int8SentimentError::Accounting => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        // Even a clean-cache finalization failure ends the native stream. Never
        // convert a failed axis into a partial bundle, abstention or retry.
        _ => BatchItemFailure::fatal(BatchCode::Execution),
    }
}

#[cfg(test)] mod tests;
