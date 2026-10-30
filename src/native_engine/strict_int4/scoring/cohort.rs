//! Independent finite-candidate heads sharing actual grouped-INT4 projections.
//!
//! Each head owns one causal KV branch and its exact work allowance. Prompt
//! lengths, languages and score spaces may differ; only compatible projections
//! share weights. The shared candidate cursor retains every semantic decision.
use super::*;
use crate::{execution_identity::ExecutionIdentity,
    native_engine::{constrained_int4, lmhead::scoring::CandidateScoreCursor,
        strict_int4::{Int4ModelIdentity, cohort::{Int4CohortEngine, Int4CohortToken,
            MAX_INT4_SEQUENCES, branching::Int4BranchSession}}}};
mod execution;
pub(crate) mod task;

pub const INT4_SCORING_COHORT_EXECUTION: &str = "grouped-int4-independent-candidate-cohort-v1";

/// The enclosing task binds exact-token plans to their complete admitted
/// identities. This leaf is not a raw-text task or resource-admission grant.
pub struct Int4CandidateCohortRequest<'a> {
    pub identity: &'a ExecutionIdentity,
    pub plan: &'a Int4CandidatePlan,
    pub budget: Int4ScoringBudget,
}
/// Each head retains its own limits in addition to these aggregate ceilings.
/// KV is the SUM of allocated capacities, not the longest live prefix.
#[derive(Clone, Copy, Debug)]
pub struct Int4ScoringCohortBudget {
    pub native: Int4RunBudget,
    pub max_kv_bytes: u64,
    pub max_output_bytes: u64,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int4CandidateCohortRun {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    pub group_width: usize,
    /// Stable input-slot order, independent of completion order.
    pub heads: Vec<Int4CandidateRun>,
    pub group_steps: u64,
    pub projection_groups: u64,
    pub model_work: Int4Work,
}
/// Physical grouping statistics are separate from the unchanged task output.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int4ScoredCohort<T> {
    pub schema_version: u32,
    pub execution: String,
    pub group_steps: u64,
    pub projection_groups: u64,
    pub output: T,
}

/// Conservative simultaneous prompt/cursor/result workspace, separate from
/// native rails/KV and retained compiled preparation. Hosts reserve it BEFORE
/// constructing context vectors or flattened prompts. This is not an RSS claim.
pub fn required_task_workspace_bytes(work: Int4Work, heads: usize, max_output_bytes: u64)
    -> Result<u64, Int4ScoringError> {
    check_count(heads)?; check_output_limit(max_output_bytes)?;
    add(add(mul(work.forward_positions, 64)?, mul(max_output_bytes, 8)?)?, mul(heads as u64, 65536)?)
}

pub(crate) struct CompiledRequest<'a> {
    pub(crate) identity: &'a ExecutionIdentity,
    pub(crate) prompt: &'a [u32],
    pub(crate) scorer: &'a CandidateScorer,
    pub(crate) mode: ScoringMode,
    pub(crate) schedule: CandidateSchedule,
    pub(crate) budget: Int4ScoringBudget,
    pub(crate) max_output_bytes: u64,
}
impl<'a> From<&Int4CandidateCohortRequest<'a>> for CompiledRequest<'a> {
    fn from(request: &Int4CandidateCohortRequest<'a>) -> Self {
        let plan = request.plan;
        Self { identity: request.identity, prompt: &plan.prompt, scorer: &plan.scorer,
            mode: plan.mode, schedule: plan.schedule, budget: request.budget, max_output_bytes: plan.max_output_bytes }
    }
}
pub fn score_int4_cohort<C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4CandidateCohortRequest<'_>], budget: Int4ScoringCohortBudget, control: &mut C)
    -> Result<Int4CandidateCohortRun, Int4ScoringError> {
    check_count(requests.len())?;
    let mut compiled = reserve(requests.len())?;
    for request in requests { compiled.push(CompiledRequest::from(request)); }
    execute_compiled_with(engine, &compiled, budget, control, |run, _| Ok(run))
}

/// Every semantic finalizer remains inside the exclusive native lifetime.
/// Failure or cancellation poisons all heads; session drop clears every cache.
pub(crate) fn execute_compiled_with<C, T, E, F>(engine: &mut Int4CohortEngine<'_>,
    requests: &[CompiledRequest<'_>], budget: Int4ScoringCohortBudget, control: &mut C, finalize: F) -> Result<T, E>
where C: DecodeStepControl, E: From<Int4ScoringError>, F: FnOnce(Int4CandidateCohortRun, &mut C) -> Result<T, E> {
    let preflight = (|| -> Result<Vec<Int4RunBudget>, Int4ScoringError> {
        checkpoint(control)?; check_count(requests.len())?; engine.check_idle()?;
        if requests.len() != engine.sequence_count() { return Err(Int4ScoringError::Input); }
        let mut capacities = reserve(requests.len())?; let mut budgets = reserve(requests.len())?;
        for (slot, request) in requests.iter().enumerate() {
            checkpoint(control)?;
            check_identity(request.identity, engine.model_identity())?;
            if request.schedule.group_width != engine.model_identity().group_width() {
                return Err(Int4ScoringError::Identity);
            }
            if let Some(first) = requests.first() { same_model(first.identity, request.identity)?; }
            capacities.push(engine.capacity(slot)?);
            budgets.push(Int4RunBudget::exact(request.schedule.model));
        }
        check_geometry(requests, &capacities, budget)?;
        Ok(budgets)
    })();
    let budgets = preflight.map_err(E::from)?;
    let mut session = engine.branch_session(&budgets, control).map_err(Int4ScoringError::from).map_err(E::from)?;
    execution::drive_with(requests, budget, &mut session, finalize)
}
fn check_count(count: usize) -> Result<(), Int4ScoringError> {
    if count == 0 || count > MAX_INT4_SEQUENCES { return Err(Int4ScoringError::Input); } Ok(())
}
fn check_geometry(requests: &[CompiledRequest<'_>], capacities: &[usize], budget: Int4ScoringCohortBudget)
    -> Result<Int4Work, Int4ScoringError> {
    check_count(requests.len())?; check_output_limit(budget.max_output_bytes)?;
    if requests.len() != capacities.len() { return Err(Int4ScoringError::Input); }
    let width = requests[0].schedule.group_width; check_group(width)?;
    let mut work = Int4Work::default(); let mut kv = 0_u64;
    for (request, &capacity) in requests.iter().zip(capacities) {
        if request.schedule.group_width != width { return Err(Int4ScoringError::Identity); }
        if capacity == 0 || capacity > DEFAULT_ADMITTED_CONTEXT_CAP { return Err(StrictInt4Error::Context.into()); }
        check_prompt(request.prompt)?; check_output_limit(request.max_output_bytes)?;
        check_capacity(request.schedule, capacity, request.budget)?;
        work = work.checked_add(request.schedule.model)?;
        kv = add(kv, mul(capacity as u64, KV_BYTES_PER_TOKEN as u64)?)?;
    }
    if kv > budget.max_kv_bytes { return Err(StrictInt4Error::Memory.into()); }
    if work.forward_positions > budget.native.max_forward_positions || work.attention_pairs > budget.native.max_attention_pairs
        || !work.projections.fits(budget.native.max_projection_work) { return Err(StrictInt4Error::Work.into()); }
    Ok(work)
}
fn check_identity(identity: &ExecutionIdentity, source: &Int4ModelIdentity) -> Result<(), Int4ScoringError> {
    constrained_int4::check_model(identity, source).map_err(|_| Int4ScoringError::Identity)
}
fn same_model(a: &ExecutionIdentity, b: &ExecutionIdentity) -> Result<(), Int4ScoringError> {
    // Normalize only task-owned fields, preserving every model/host field
    // (including future additions) through full structural equality.
    let mut common = a.clone();
    common.template_digest = b.template_digest; common.task_spec = b.task_spec.clone();
    common.taskir_digest = b.taskir_digest; common.prompt_digest = b.prompt_digest;
    common.grammar_compiler_version = b.grammar_compiler_version.clone(); common.schema_digest = b.schema_digest;
    common.sampler_version = b.sampler_version.clone(); common.calibration_digest = b.calibration_digest;
    common.decision_policy_digest = b.decision_policy_digest;
    if &common != b { return Err(Int4ScoringError::Identity); } Ok(())
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), Int4ScoringError> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(StrictInt4Error::Cancelled(cause).into()), None => Ok(()) }
}
pub(crate) fn check_output<T: Serialize>(value: &T, cap: u64) -> Result<(), Int4ScoringError> {
    let size = canonjson::canonical_bytes(value).map_err(|_| Int4ScoringError::Serialization)?.len() as u64;
    if size > cap { return Err(Int4ScoringError::OutputBudget); } Ok(())
}

// Private differential fixture seam. Public execution requires the actual
// immutable model, exclusive native owner, and checked task/model identities.
trait GroupDriver {
    type Control: DecodeStepControl;
    fn control(&mut self) -> &mut Self::Control;
    fn position(&self, slot: usize) -> Result<usize, Int4ScoringError>;
    fn rewind(&mut self, slot: usize, retain: usize) -> Result<(), Int4ScoringError>;
    fn append_group(&mut self, steps: &[Int4CohortToken]) -> Result<(), Int4ScoringError>;
    fn logits_group(&mut self, slots: &[usize], rows: LinearRows<'_>) -> Result<Vec<f32>, Int4ScoringError>;
    fn work(&self, slot: usize) -> Result<Int4Work, Int4ScoringError>;
    fn rewound_positions(&self, slot: usize) -> Result<u64, Int4ScoringError>;
    fn abort(&mut self);
}
impl<C: DecodeStepControl> GroupDriver for Int4BranchSession<'_, '_, C> {
    type Control = C;
    fn control(&mut self) -> &mut C { Int4BranchSession::control(self) }
    fn position(&self, slot: usize) -> Result<usize, Int4ScoringError> { Ok(Int4BranchSession::position(self, slot)?) }
    fn rewind(&mut self, slot: usize, retain: usize) -> Result<(), Int4ScoringError> { Ok(Int4BranchSession::rewind(self, slot, retain)?) }
    fn append_group(&mut self, steps: &[Int4CohortToken]) -> Result<(), Int4ScoringError> { Ok(Int4BranchSession::append_group(self, steps)?) }
    fn logits_group(&mut self, slots: &[usize], rows: LinearRows<'_>) -> Result<Vec<f32>, Int4ScoringError> {
        Ok(Int4BranchSession::logits_group(self, slots, rows)?)
    }
    fn work(&self, slot: usize) -> Result<Int4Work, Int4ScoringError> { Ok(Int4BranchSession::work(self, slot)?) }
    fn rewound_positions(&self, slot: usize) -> Result<u64, Int4ScoringError> { Ok(Int4BranchSession::rewound_positions(self, slot)?) }
    fn abort(&mut self) { Int4BranchSession::abort(self); }
}
#[cfg(test)] mod tests;
