//! Physically grouped finite-candidate heads with independent causal KV.
//!
//! Prompt lengths, candidate languages and probability modes may differ. The
//! shared CandidateScorer cursor remains the only numerical finalizer. This
//! leaf neither loads weights nor grants task/artifact/runtime admission.
use super::*;
use crate::{execution_identity::{ExecutionIdentity, Sha256Digest},
    native_engine::{artifact_bridge::ArtifactIdentity, constrained_int8,
        lmhead::scoring::CandidateScoreCursor, portable_int8::batch::MAX_BATCH_ROWS,
        strict_int8::cohort::{CohortToken, Int8CohortEngine, branching::Int8BranchSession}}};
mod execution;

pub const INT8_SCORING_COHORT_EXECUTION: &str = "portable-int8-independent-candidate-cohort-v1";

/// Exact token plans are not raw-text task identities. The embedding task must
/// bind each plan to its admitted identity before calling this low-level API.
pub struct Int8CandidateCohortRequest<'a> {
    pub identity: &'a ExecutionIdentity,
    pub plan: &'a Int8CandidatePlan,
    pub budget: Int8ScoringBudget,
}
/// Aggregate limits supplement each head's limits. KV prices the SUM of full
/// resident capacities, never their maximum or just their populated prefixes.
#[derive(Clone, Copy, Debug)]
pub struct Int8ScoringCohortBudget {
    pub native: Int8RunBudget,
    pub max_kv_bytes: u64,
    pub max_output_bytes: u64,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int8CandidateCohortRun {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    /// Stable input-slot order, never the order in which heads complete.
    pub heads: Vec<Int8CandidateRun>,
    pub group_steps: u64,
    pub projection_groups: u64,
    pub model_work: Int8Work,
}
/// Physical execution statistics alongside an unchanged semantic task result.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int8ScoredCohort<T> {
    pub schema_version: u32,
    pub execution: String,
    pub group_steps: u64,
    pub projection_groups: u64,
    pub output: T,
}

pub(crate) struct CompiledRequest<'a> {
    pub(crate) identity: &'a ExecutionIdentity,
    pub(crate) prompt: &'a [u32],
    pub(crate) scorer: &'a CandidateScorer,
    pub(crate) mode: ScoringMode,
    pub(crate) schedule: CandidateSchedule,
    pub(crate) budget: Int8ScoringBudget,
    pub(crate) max_output_bytes: u64,
}
impl<'a> From<&Int8CandidateCohortRequest<'a>> for CompiledRequest<'a> {
    fn from(request: &Int8CandidateCohortRequest<'a>) -> Self {
        let plan = request.plan;
        Self { identity: request.identity, prompt: &plan.prompt, scorer: &plan.scorer,
            mode: plan.mode, schedule: plan.schedule, budget: request.budget, max_output_bytes: plan.max_output_bytes }
    }
}
pub fn score_int8_cohort<C: DecodeStepControl>(engine: &mut Int8CohortEngine<'_>,
    requests: &[Int8CandidateCohortRequest<'_>], budget: Int8ScoringCohortBudget, control: &mut C)
    -> Result<Int8CandidateCohortRun, Int8ScoringError> {
    check_count(requests.len())?;
    let mut compiled = reserve(requests.len())?;
    for request in requests { compiled.push(CompiledRequest::from(request)); }
    execute_compiled_with(engine, &compiled, budget, control, |run, _| Ok(run))
}

/// Keep the entire semantic finalizer INSIDE the same exclusive native session.
/// A failed finalizer or late cancellation poisons every head, and RAII clears
/// all native contexts before process-owned reservations can be released.
pub(crate) fn execute_compiled_with<C, T, E, F>(engine: &mut Int8CohortEngine<'_>,
    requests: &[CompiledRequest<'_>], budget: Int8ScoringCohortBudget, control: &mut C, finalize: F) -> Result<T, E>
where C: DecodeStepControl, E: From<Int8ScoringError>, F: FnOnce(Int8CandidateCohortRun, &mut C) -> Result<T, E> {
    let preflight = (|| -> Result<Vec<Int8RunBudget>, Int8ScoringError> {
        checkpoint(control)?; engine.check_idle()?;
        if requests.len() != engine.sequence_count() { return Err(Int8ScoringError::Input); }
        let mut capacities = reserve(requests.len())?; let mut budgets = reserve(requests.len())?;
        for (slot, request) in requests.iter().enumerate() {
            checkpoint(control)?;
            check_identity(request.identity, engine.artifact_identity())?;
            if let Some(first) = requests.first() { same_model(first.identity, request.identity)?; }
            capacities.push(engine.capacity(slot)?);
            budgets.push(Int8RunBudget::exact(request.schedule.model));
        }
        check_geometry(requests, &capacities, budget)?;
        Ok(budgets)
    })();
    let budgets = preflight.map_err(E::from)?;
    let mut session = engine.branch_session(&budgets, control).map_err(Int8ScoringError::from).map_err(E::from)?;
    execution::drive_with(requests, budget, &mut session, finalize)
}
fn check_count(count: usize) -> Result<(), Int8ScoringError> {
    if count == 0 || count > MAX_BATCH_ROWS { return Err(Int8ScoringError::Input); } Ok(())
}
fn check_geometry(requests: &[CompiledRequest<'_>], capacities: &[usize], budget: Int8ScoringCohortBudget)
    -> Result<Int8Work, Int8ScoringError> {
    check_count(requests.len())?; check_output_limit(budget.max_output_bytes)?;
    if requests.len() != capacities.len() { return Err(Int8ScoringError::Input); }
    let mut work = Int8Work::default(); let mut kv = 0_u64;
    for (request, &capacity) in requests.iter().zip(capacities) {
        check_prompt(request.prompt)?; check_output_limit(request.max_output_bytes)?;
        check_capacity(request.schedule, capacity, request.budget)?;
        work = work.checked_add(request.schedule.model)?;
        kv = add(kv, mul(capacity as u64, KV_BYTES_PER_TOKEN as u64)?)?;
    }
    if kv > budget.max_kv_bytes { return Err(StrictInt8Error::Memory.into()); }
    if work.forward_positions > budget.native.max_forward_positions || work.attention_pairs > budget.native.max_attention_pairs
        || !work.projections.fits(budget.native.max_projection_work) { return Err(StrictInt8Error::Work.into()); }
    Ok(work)
}
fn check_identity(identity: &ExecutionIdentity, source: &ArtifactIdentity) -> Result<(), Int8ScoringError> {
    constrained_int8::check_profile(identity).map_err(|_| Int8ScoringError::Identity)?;
    if source.model_id != "Nanbeige4.2-3B" || source.revision != identity.source_revision || source.recipe_id != identity.quant_recipe
        || Sha256Digest::from_hex(&source.logical_model_sha256).ok() != Some(identity.logical_model_digest) {
        return Err(Int8ScoringError::Identity);
    }
    Ok(())
}
fn same_model(a: &ExecutionIdentity, b: &ExecutionIdentity) -> Result<(), Int8ScoringError> {
    // Normalize TASK-owned differences only; preserve every model/host field,
    // including fields added in the future, through full structural equality.
    let mut common = a.clone();
    common.template_digest = b.template_digest; common.task_spec = b.task_spec.clone();
    common.taskir_digest = b.taskir_digest; common.prompt_digest = b.prompt_digest;
    common.grammar_compiler_version = b.grammar_compiler_version.clone(); common.schema_digest = b.schema_digest;
    common.sampler_version = b.sampler_version.clone(); common.calibration_digest = b.calibration_digest;
    common.decision_policy_digest = b.decision_policy_digest;
    if &common != b { return Err(Int8ScoringError::Identity); } Ok(())
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), Int8ScoringError> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(StrictInt8Error::Cancelled(cause).into()), None => Ok(()) }
}
pub(crate) fn check_output<T: Serialize>(value: &T, cap: u64) -> Result<(), Int8ScoringError> {
    let size = canonjson::canonical_bytes(value).map_err(|_| Int8ScoringError::Serialization)?.len() as u64;
    if size > cap { return Err(Int8ScoringError::OutputBudget); } Ok(())
}

// Private differential-test seam. Public callers cannot supply mock logits or
// a claimed receipt in place of the actual immutable native model binding.
trait GroupDriver {
    type Control: DecodeStepControl;
    fn control(&mut self) -> &mut Self::Control;
    fn position(&self, slot: usize) -> Result<usize, Int8ScoringError>;
    fn rewind(&mut self, slot: usize, retain: usize) -> Result<(), Int8ScoringError>;
    fn append_group(&mut self, steps: &[CohortToken]) -> Result<(), Int8ScoringError>;
    fn logits_group(&mut self, slots: &[usize], rows: LinearRows<'_>) -> Result<Vec<f32>, Int8ScoringError>;
    fn work(&self, slot: usize) -> Result<Int8Work, Int8ScoringError>;
    fn rewound_positions(&self, slot: usize) -> Result<u64, Int8ScoringError>;
    fn abort(&mut self);
}
impl<C: DecodeStepControl> GroupDriver for Int8BranchSession<'_, '_, C> {
    type Control = C;
    fn control(&mut self) -> &mut C { Int8BranchSession::control(self) }
    fn position(&self, slot: usize) -> Result<usize, Int8ScoringError> { Ok(Int8BranchSession::position(self, slot)?) }
    fn rewind(&mut self, slot: usize, retain: usize) -> Result<(), Int8ScoringError> { Ok(Int8BranchSession::rewind(self, slot, retain)?) }
    fn append_group(&mut self, steps: &[CohortToken]) -> Result<(), Int8ScoringError> { Ok(Int8BranchSession::append_group(self, steps)?) }
    fn logits_group(&mut self, slots: &[usize], rows: LinearRows<'_>) -> Result<Vec<f32>, Int8ScoringError> {
        Ok(Int8BranchSession::logits_group(self, slots, rows)?)
    }
    fn work(&self, slot: usize) -> Result<Int8Work, Int8ScoringError> { Ok(Int8BranchSession::work(self, slot)?) }
    fn rewound_positions(&self, slot: usize) -> Result<u64, Int8ScoringError> { Ok(Int8BranchSession::rewound_positions(self, slot)?) }
    fn abort(&mut self) { Int8BranchSession::abort(self); }
}
#[cfg(test)] mod tests;
