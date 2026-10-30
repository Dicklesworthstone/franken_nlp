//! Complete pairwise, rubric and faithfulness judgment on native grouped INT4.
//!
//! The shared compiler owns prompts, full-vocabulary languages and scored EOS.
//! Every head binds the actual model/group and completes before publication.
use std::{error::Error, fmt};
use super::*;
use super::super::common::Head;
use crate::native_engine::{
    generation::grouped_int4::bind_candidate_model,
    lmhead::scoring::ScoringMode,
    strict_int4::{Int4ModelIdentity, Int4RunBudget, Int4Work, StrictInt4Engine,
        StrictInt4Error, STRICT_INT4_EXECUTION, STRICT_INT4_PROFILE,
        prefill::Int4PrefillLimits,
        scoring::{self, CandidateSchedule, Int4CandidateRun, Int4ScoringBudget,
            Int4ScoringError, INT4_SCORING_EXECUTION}},
};
mod model;
pub use model::Int4JudgePlanner;

pub const INT4_JUDGE_EXECUTION: &str = "grouped-int4-complete-judge-prefix-heads-v1";

/// Typed causes remain available without putting private input in diagnostics.
pub enum Int4JudgeError {
    Task(JudgeError), Scoring(Int4ScoringError), Native(StrictInt4Error),
    Cancelled(DecodeCancellationKind), Identity, WorkBudget, Accounting,
}
impl fmt::Display for Int4JudgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Task(_) => "int4 judge planning or finalization refused",
            Self::Scoring(_) => "int4 judge complete candidate execution failed",
            Self::Native(_) => "int4 judge native engine refused",
            Self::Cancelled(_) => "int4 judge cancelled",
            Self::Identity => "int4 judge model, group or admitted identity differs",
            Self::WorkBudget => "int4 judge whole-bundle work allowance exceeded",
            Self::Accounting => "int4 judge complete-head accounting diverged",
        })
    }
}
impl fmt::Debug for Int4JudgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int4JudgeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Task(e) => Some(e), Self::Scoring(e) => Some(e),
            Self::Native(e) => Some(e), _ => None }
    }
}
impl From<JudgeError> for Int4JudgeError { fn from(e: JudgeError) -> Self { Self::Task(e) } }
impl From<Int4ScoringError> for Int4JudgeError { fn from(e: Int4ScoringError) -> Self { Self::Scoring(e) } }
impl From<StrictInt4Error> for Int4JudgeError { fn from(e: StrictInt4Error) -> Self { Self::Native(e) } }
impl Int4JudgeError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Cancelled(c) | Self::Native(StrictInt4Error::Cancelled(c)) => Some(*c),
            Self::Scoring(e) => e.cancellation(), _ => None }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int4JudgeRun {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    pub group_width: usize,
    pub result: JudgeResult,
    pub head_count: usize,
    pub model_work: Int4Work,
    pub rewound_positions: u64,
}

/// Sealed exact prompts, candidate schedules and model binding. No caller can
/// deserialize a plan, substitute a score provider or extract an eager plan.
pub struct PreparedInt4Judge {
    inner: PreparedJudge,
    schedules: Vec<CandidateSchedule>,
    group_width: usize,
    work: Int4Work,
    budget: TaskBudget,
}
impl JudgePlanner {
    pub fn plan_int4_with_control<C: DecodeStepControl>(&self, request: &JudgeRequest,
        context: &PlanContext<'_>, limits: JudgeLimits, model: &Int4ModelIdentity,
        control: &mut C) -> Result<PreparedInt4Judge, Int4JudgeError> {
        checkpoint(control)?;
        check_model(context.execution_identity(), model)?;
        self.compile_int4(request, context, limits, model.group_width(), control)
    }
    // Private fixtures may supply geometry. Every public planning path first
    // checks model identity from actual loaded source/snapshot weights above.
    fn compile_int4<C: DecodeStepControl>(&self, request: &JudgeRequest,
        context: &PlanContext<'_>, limits: JudgeLimits, group_width: usize,
        control: &mut C) -> Result<PreparedInt4Judge, Int4JudgeError> {
        checkpoint(control)?;
        check_profile(context.execution_identity())?;
        let inner = self.plan_with_profile(request, context, limits,
            NumericsProfile::StrictQuantized { version: 2 })?;
        checkpoint(control)?;
        PreparedInt4Judge::compile(inner, request.budget(), group_width, control)
    }
}
impl PreparedInt4Judge {
    fn compile<C: DecodeStepControl>(inner: PreparedJudge, budget: TaskBudget,
        group_width: usize, control: &mut C) -> Result<Self, Int4JudgeError> {
        let bundle = inner.executable.bundle();
        if bundle.heads.is_empty() { return Err(Int4JudgeError::Accounting); }
        let mut schedules = reserved(bundle.heads.len())?;
        let mut work = Int4Work::default();
        for head in &bundle.heads {
            checkpoint(control)?;
            let DecodeStrategy::PrefillOnly { candidates } = head.ir.decode_strategy() else {
                return Err(Int4JudgeError::Accounting);
            };
            let schedule = CandidateSchedule::new(group_width, head.prompt_len, candidates,
                ScoringMode::FullVocabulary)?;
            if schedule.scoring != head.work { return Err(Int4JudgeError::Accounting); }
            work = work.checked_add(schedule.model)?;
            schedules.push(schedule);
        }
        // The shared compiler applies the projected-logit ceiling to the sum
        // over every order, criterion and evidence head, not just the largest.
        if work.projected_logits != bundle.work.projected_logits { return Err(Int4JudgeError::Accounting); }
        checkpoint(control)?;
        Ok(Self { inner, schedules, group_width, work, budget })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.inner.execution_identity() }
    pub fn planned_work(&self) -> Int4Work { self.work }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn head_count(&self) -> usize { self.schedules.len() }
    pub fn task_budget(&self) -> TaskBudget { self.budget }
    pub fn max_result_bytes(&self) -> u64 { self.inner.executable.bundle().max_output_bytes }
    pub fn required_context(&self) -> usize { self.schedules.iter().map(|s| s.context).max().unwrap_or(0) }
    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), Int4JudgeError> {
        check_profile(admitted)?;
        self.inner.verify_identity(admitted).map_err(Into::into)
    }
    /// Admit every compiled head and all aggregate native work before the
    /// first forward. Each native preflight prices the full allocated KV.
    pub fn preflight(&self, admitted: &ExecutionIdentity, engine: &StrictInt4Engine<'_>,
        budget: Int4ScoringBudget) -> Result<(), Int4JudgeError> {
        self.verify_identity(admitted)?;
        check_model(admitted, engine.model_identity())?;
        if engine.model_identity().group_width() != self.group_width
            || self.schedules.len() != self.inner.executable.bundle().heads.len() || self.schedules.is_empty() {
            return Err(Int4JudgeError::Identity);
        }
        check_work(self.work, budget.native)?;
        for (head, &schedule) in self.inner.executable.bundle().heads.iter().zip(&self.schedules) {
            schedule.preflight(engine, Int4ScoringBudget { native: Int4RunBudget::exact(schedule.model),
                max_kv_bytes: budget.max_kv_bytes.min(head.ir.budget().max_kv_bytes) })?;
        }
        Ok(())
    }
    pub fn execute_with_control<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt4Engine<'_>, budget: Int4ScoringBudget, control: &mut C)
        -> Result<Int4JudgeRun, Int4JudgeError> {
        self.execute_with_prefill(admitted, engine, budget, None, control)
    }
    /// Group the initial prompt of every order, criterion and evidence head.
    /// All heads and total work preflight before the first forward; one head
    /// failure still discards the complete judgment. Extra scratch requires a
    /// separate caller-owned reservation retained through physical drain.
    pub fn execute_layer_major_with_control<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt4Engine<'_>, budget: Int4ScoringBudget,
        prefill: Int4PrefillLimits, control: &mut C) -> Result<Int4JudgeRun, Int4JudgeError> {
        self.execute_with_prefill(admitted, engine, budget, Some(prefill), control)
    }
    fn execute_with_prefill<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt4Engine<'_>, budget: Int4ScoringBudget,
        prefill: Option<Int4PrefillLimits>, control: &mut C) -> Result<Int4JudgeRun, Int4JudgeError> {
        if let Some(limits) = prefill { limits.validate()?; }
        checkpoint(control)?;
        self.preflight(admitted, engine, budget)?;
        self.evaluate_heads(control, |_, head, schedule, control| {
            let mut prompt = reserved(head.prompt_len)?;
            prompt.extend(head.ir.prompt_segments().iter().flat_map(|s| s.token_ids().iter().copied()));
            if prompt.len() != head.prompt_len { return Err(Int4JudgeError::Accounting); }
            let run = scoring::prefill::execute_compiled_with_prefill(&prompt, &head.scorer, ScoringMode::FullVocabulary,
                schedule, self.max_result_bytes(), engine,
                Int4ScoringBudget { native: Int4RunBudget::exact(schedule.model),
                    max_kv_bytes: budget.max_kv_bytes.min(head.ir.budget().max_kv_bytes) }, prefill, control)?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                return Err(Int4JudgeError::Accounting);
            }
            Ok(run)
        })
    }
    // Only fixtures substitute this numerical boundary. They retain the real
    // complete-score validator and all three shared semantic finalizers.
    fn evaluate_heads<C: DecodeStepControl, F>(&self, control: &mut C, mut evaluate: F)
        -> Result<Int4JudgeRun, Int4JudgeError>
    where F: FnMut(usize, &Head, CandidateSchedule, &mut C) -> Result<Int4CandidateRun, Int4JudgeError> {
        let bundle = self.inner.executable.bundle();
        let mut count = 0_usize;
        let mut work = Int4Work::default();
        let mut rewound = 0_u64;
        let scores = bundle.score_with::<Int4JudgeError, _>(|index, head| {
            checkpoint(control)?;
            let schedule = *self.schedules.get(index).ok_or(Int4JudgeError::Accounting)?;
            let run = evaluate(index, head, schedule, control)?;
            if run.schema_version != 1 || run.execution != INT4_SCORING_EXECUTION
                || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != self.group_width
                || run.model_work != schedule.model || run.scores.work != schedule.scoring {
                return Err(Int4JudgeError::Accounting);
            }
            work = work.checked_add(run.model_work)?;
            rewound = rewound.checked_add(run.rewound_positions).ok_or(Int4JudgeError::Accounting)?;
            count += 1;
            checkpoint(control)?;
            Ok(run.scores)
        })?;
        if count != self.schedules.len() || work != self.work { return Err(Int4JudgeError::Accounting); }
        checkpoint(control)?;
        let result = self.inner.executable.finish(scores)?;
        let run = Int4JudgeRun { schema_version: 1, execution: INT4_JUDGE_EXECUTION.to_owned(),
            numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: self.group_width,
            result, head_count: count, model_work: work, rewound_positions: rewound };
        bundle.check_output(&run)?;
        checkpoint(control)?;
        Ok(run)
    }
}
pub(crate) fn check_model(identity: &ExecutionIdentity, model: &Int4ModelIdentity) -> Result<(), Int4JudgeError> {
    check_profile(identity)?;
    let bound = bind_candidate_model(identity.clone(), model).map_err(|_| Int4JudgeError::Identity)?;
    if &bound != identity { return Err(Int4JudgeError::Identity); }
    Ok(())
}
fn check_profile(id: &ExecutionIdentity) -> Result<(), Int4JudgeError> {
    if id.task_spec != "judge-v1" || id.numerics_profile != (NumericsProfile::StrictQuantized { version: 2 })
        || id.backend_semantic_version != STRICT_INT4_EXECUTION || id.kv_dtype != "bf16"
        || id.thinking_mode != ThinkingMode::Disabled || id.tool_mode != ToolMode::None {
        return Err(Int4JudgeError::Identity);
    }
    Ok(())
}
fn check_work(work: Int4Work, budget: Int4RunBudget) -> Result<(), Int4JudgeError> {
    if work.forward_positions > budget.max_forward_positions || work.attention_pairs > budget.max_attention_pairs
        || !work.projections.fits(budget.max_projection_work) { return Err(Int4JudgeError::WorkBudget); }
    Ok(())
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), Int4JudgeError> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(Int4JudgeError::Cancelled(cause)), None => Ok(()) }
}

#[cfg(test)] mod tests;
