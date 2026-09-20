//! Native INT8 pairwise, rubric and full-source/evidence judgment.
//! The existing segmented planner, candidate language and semantic finalizers
//! remain the authority. No BF16 plan is relabeled, converted or exposed.

use std::{error::Error, fmt};
use super::*;
use super::super::common::Head;
use crate::native_engine::{
    artifact_bridge::ArtifactIdentity,
    lmhead::scoring::ScoringMode,
    strict_int8::{Int8RunBudget, Int8Work, StrictInt8Engine, StrictInt8Error,
        STRICT_INT8_EXECUTION, STRICT_INT8_PROFILE,
        scoring::{self, CandidateSchedule, Int8CandidateRun, Int8ScoringBudget,
            Int8ScoringError, INT8_SCORING_EXECUTION}},
};

pub const INT8_JUDGE_EXECUTION: &str = "portable-int8-complete-judge-prefix-heads-v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Int8JudgeError {
    Task(JudgeError),
    Scoring(Int8ScoringError),
    Native(StrictInt8Error),
    Cancelled(DecodeCancellationKind),
    Identity,
    WorkBudget,
    Accounting,
}
impl fmt::Display for Int8JudgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Task(_) => "int8 judge planning or finalization refused",
            Self::Scoring(_) => "int8 judge candidate execution failed",
            Self::Native(_) => "int8 judge native engine refused",
            Self::Cancelled(_) => "int8 judge cancelled",
            Self::Identity => "int8 judge model or execution identity differs",
            Self::WorkBudget => "int8 judge whole-bundle work allowance exceeded",
            Self::Accounting => "int8 judge complete-head accounting diverged",
        })
    }
}
impl Error for Int8JudgeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Task(e) => Some(e), Self::Scoring(e) => Some(e), Self::Native(e) => Some(e), _ => None }
    }
}
impl From<JudgeError> for Int8JudgeError { fn from(e: JudgeError) -> Self { Self::Task(e) } }
impl From<Int8ScoringError> for Int8JudgeError { fn from(e: Int8ScoringError) -> Self { Self::Scoring(e) } }
impl From<StrictInt8Error> for Int8JudgeError { fn from(e: StrictInt8Error) -> Self { Self::Native(e) } }
impl Int8JudgeError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Cancelled(c) | Self::Native(StrictInt8Error::Cancelled(c)) => Some(*c),
            Self::Scoring(e) => e.cancellation(), _ => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int8JudgeRun {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    pub result: JudgeResult,
    pub head_count: usize,
    pub model_work: Int8Work,
    pub rewound_positions: u64,
}

/// Sealed raw-text plan. Small schedule records do not duplicate prompts,
/// scorers, source documents, model weights or KV. No public inner conversion.
pub struct PreparedInt8Judge {
    inner: PreparedJudge,
    schedules: Vec<CandidateSchedule>,
    work: Int8Work,
    budget: TaskBudget,
}
impl JudgePlanner {
    /// Compile with the STRICT profile supplied by the caller, never by
    /// rewriting an eager context. Legacy tokenization is checked before and
    /// after the bounded planning operation; it is not preemptible mid-encode.
    pub fn plan_int8_with_control<C: DecodeStepControl>(&self, request: &JudgeRequest,
        context: &PlanContext<'_>, limits: JudgeLimits, control: &mut C)
        -> Result<PreparedInt8Judge, Int8JudgeError> {
        checkpoint(control)?;
        check_profile(context.execution_identity())?;
        let inner = self.plan_with_profile(request, context, limits, NumericsProfile::StrictQuantized { version: 1 })?;
        checkpoint(control)?;
        let mut schedules = reserved(inner.executable.bundle().heads.len())?;
        let mut work = Int8Work::default();
        for head in &inner.executable.bundle().heads {
            checkpoint(control)?;
            let DecodeStrategy::PrefillOnly { candidates } = head.ir.decode_strategy() else {
                return Err(Int8JudgeError::Accounting);
            };
            let schedule = CandidateSchedule::new(head.prompt_len, candidates, ScoringMode::FullVocabulary)?;
            if schedule.scoring != head.work { return Err(Int8JudgeError::Accounting); }
            work = work.checked_add(schedule.model)?;
            schedules.push(schedule);
        }
        checkpoint(control)?;
        Ok(PreparedInt8Judge { inner, schedules, work, budget: request.budget() })
    }
}
impl PreparedInt8Judge {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.inner.execution_identity() }
    pub fn planned_work(&self) -> Int8Work { self.work }
    pub fn head_count(&self) -> usize { self.schedules.len() }
    pub fn required_context(&self) -> usize { self.schedules.iter().map(|s| s.context).max().unwrap_or(0) }
    pub fn task_budget(&self) -> TaskBudget { self.budget }
    pub fn max_result_bytes(&self) -> u64 { self.inner.executable.bundle().max_output_bytes }
    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), Int8JudgeError> {
        check_profile(admitted)?;
        self.inner.verify_identity(admitted).map_err(Into::into)
    }
    /// Check EVERY head and the complete native allowance before any forward.
    /// The file bridge's identity facts are checked but never promoted to
    /// publisher authentication or a ratified artifact-format certificate.
    pub fn preflight(&self, admitted: &ExecutionIdentity, engine: &StrictInt8Engine<'_>,
        budget: Int8ScoringBudget) -> Result<(), Int8JudgeError> {
        self.verify_identity(admitted)?;
        check_model(admitted, engine.artifact_identity())?;
        check_work(self.work, budget.native)?;
        for (head, &schedule) in self.inner.executable.bundle().heads.iter().zip(&self.schedules) {
            schedule.preflight(engine, Int8ScoringBudget { native: Int8RunBudget::exact(schedule.model),
                max_kv_bytes: budget.max_kv_bytes.min(head.ir.budget().max_kv_bytes) })?;
        }
        if self.schedules.len() != self.inner.executable.bundle().heads.len() || self.schedules.is_empty() {
            return Err(Int8JudgeError::Accounting);
        }
        Ok(())
    }
    pub fn execute_with_control<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt8Engine<'_>, budget: Int8ScoringBudget, control: &mut C)
        -> Result<Int8JudgeRun, Int8JudgeError> {
        checkpoint(control)?;
        self.preflight(admitted, engine, budget)?;
        self.evaluate_heads(control, |_, head, schedule, control| {
            let mut prompt = reserved(head.prompt_len)?;
            prompt.extend(head.ir.prompt_segments().iter().flat_map(|s| s.token_ids().iter().copied()));
            if prompt.len() != head.prompt_len { return Err(Int8JudgeError::Accounting); }
            let run = scoring::execute_compiled(&prompt, &head.scorer, ScoringMode::FullVocabulary,
                schedule, self.max_result_bytes(), engine,
                Int8ScoringBudget { native: Int8RunBudget::exact(schedule.model),
                    max_kv_bytes: budget.max_kv_bytes.min(head.ir.budget().max_kv_bytes) }, control)?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                return Err(Int8JudgeError::Accounting);
            }
            Ok(run)
        })
    }
    // Only tests substitute this private numerical boundary. They still run
    // the real complete-score validator and all semantic judge finalizers.
    fn evaluate_heads<C: DecodeStepControl, F>(&self, control: &mut C, mut evaluate: F)
        -> Result<Int8JudgeRun, Int8JudgeError>
    where F: FnMut(usize, &Head, CandidateSchedule, &mut C) -> Result<Int8CandidateRun, Int8JudgeError> {
        let bundle = self.inner.executable.bundle();
        let mut count = 0_usize;
        let mut work = Int8Work::default();
        let mut rewound = 0_u64;
        let scores = bundle.score_with::<Int8JudgeError, _>(|index, head| {
            checkpoint(control)?;
            let schedule = *self.schedules.get(index).ok_or(Int8JudgeError::Accounting)?;
            let run = evaluate(index, head, schedule, control)?;
            if run.schema_version != 1 || run.execution != INT8_SCORING_EXECUTION
                || run.numerics_profile != STRICT_INT8_PROFILE || run.model_work != schedule.model
                || run.scores.work != schedule.scoring { return Err(Int8JudgeError::Accounting); }
            work = work.checked_add(run.model_work)?;
            rewound = rewound.checked_add(run.rewound_positions).ok_or(Int8JudgeError::Accounting)?;
            count += 1;
            Ok(run.scores)
        })?;
        if count != self.schedules.len() || work != self.work { return Err(Int8JudgeError::Accounting); }
        checkpoint(control)?;
        let result = self.inner.executable.finish(scores)?;
        let run = Int8JudgeRun { schema_version: 1, execution: INT8_JUDGE_EXECUTION.to_owned(),
            numerics_profile: STRICT_INT8_PROFILE.to_owned(), result, head_count: count,
            model_work: work, rewound_positions: rewound };
        bundle.check_output(&run)?;
        checkpoint(control)?;
        Ok(run)
    }
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), Int8JudgeError> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(Int8JudgeError::Cancelled(cause)), None => Ok(()) }
}
fn check_profile(id: &ExecutionIdentity) -> Result<(), Int8JudgeError> {
    if id.task_spec != "judge-v1" || id.numerics_profile != (NumericsProfile::StrictQuantized { version: 1 })
        || id.backend_semantic_version != STRICT_INT8_EXECUTION || id.kv_dtype != "bf16"
        || id.thinking_mode != ThinkingMode::Disabled || id.tool_mode != ToolMode::None {
        return Err(Int8JudgeError::Identity);
    }
    Ok(())
}
fn check_model(id: &ExecutionIdentity, source: &ArtifactIdentity) -> Result<(), Int8JudgeError> {
    if source.model_id != "Nanbeige4.2-3B" || source.revision != id.source_revision || source.recipe_id != id.quant_recipe
        || Sha256Digest::from_hex(&source.logical_model_sha256).ok() != Some(id.logical_model_digest) {
        return Err(Int8JudgeError::Identity);
    }
    Ok(())
}
fn check_work(work: Int8Work, budget: Int8RunBudget) -> Result<(), Int8JudgeError> {
    if work.forward_positions > budget.max_forward_positions || work.attention_pairs > budget.max_attention_pairs
        || !work.projections.fits(budget.max_projection_work) { return Err(Int8JudgeError::WorkBudget); }
    Ok(())
}

#[cfg(test)] mod tests;
