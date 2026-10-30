//! Independent affect dimensions on the native grouped-INT4 candidate scorer.
//!
//! The shared compiler owns exact prompts, five-bin languages and EOS. Each
//! axis starts a clean native session and retains its own score denominator;
//! the complete bundle binds the loaded model, group partition and work cap.
use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{
    canonjson,
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest},
    native_engine::{
        decode::{DecodeCancellationKind, DecodeStepControl},
        generation::grouped_int4::bind_candidate_model,
        lmhead::scoring::{ScoringMode, ScoringWork},
        strict_int4::{Int4ModelIdentity, Int4RunBudget, Int4Work, StrictInt4Engine,
            StrictInt4Error, STRICT_INT4_PROFILE,
            prefill::Int4PrefillLimits,
            scoring::{self, CandidateSchedule, Int4CandidateRun, Int4ScoringBudget,
                Int4ScoringError, INT4_SCORING_EXECUTION}},
    },
    tasks::ir::{DecodeStrategy, PlanContext, TaskBudget},
};
use super::{SentimentError, SentimentLimits, SentimentOptions, SentimentPlan, SentimentPlanner,
    SentimentPlanningError, SentimentRequest, SentimentResult, distribution::HeadPlan};
mod model;
pub use model::Int4SentimentPlanner;

pub const INT4_SENTIMENT_EXECUTION: &str = "grouped-int4-independent-affect-heads-v1";
const SAMPLER_VERSION: &str = "sentiment-complete-candidate-eos-v1";

pub enum Int4SentimentError {
    Planning(SentimentPlanningError), Task(SentimentError), Scoring(Int4ScoringError),
    Native(StrictInt4Error), Identity, WorkBudget, Accounting, Allocation,
    OutputBudget, Serialization, Cancelled(DecodeCancellationKind),
}
impl fmt::Display for Int4SentimentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Planning(_) => "int4 sentiment planning refused",
            Self::Task(_) => "int4 sentiment independent finalization refused",
            Self::Scoring(_) => "int4 sentiment finite-candidate execution failed",
            Self::Native(_) => "int4 sentiment native engine refused",
            Self::Identity => "int4 sentiment model or admitted identity differs",
            Self::WorkBudget => "int4 sentiment whole-bundle work budget exceeded",
            Self::Accounting => "int4 sentiment complete head/work contract diverged",
            Self::Allocation => "int4 sentiment bounded allocation refused",
            Self::OutputBudget => "complete int4 sentiment result exceeds its byte budget",
            Self::Serialization => "int4 sentiment serialization refused",
            Self::Cancelled(_) => "int4 sentiment cancelled",
        })
    }
}
impl fmt::Debug for Int4SentimentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int4SentimentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Planning(e) => Some(e), Self::Task(e) => Some(e),
            Self::Scoring(e) => Some(e), Self::Native(e) => Some(e), _ => None }
    }
}
impl From<SentimentPlanningError> for Int4SentimentError { fn from(e: SentimentPlanningError) -> Self { Self::Planning(e) } }
impl From<SentimentError> for Int4SentimentError { fn from(e: SentimentError) -> Self { Self::Task(e) } }
impl From<Int4ScoringError> for Int4SentimentError { fn from(e: Int4ScoringError) -> Self { Self::Scoring(e) } }
impl From<StrictInt4Error> for Int4SentimentError { fn from(e: StrictInt4Error) -> Self { Self::Native(e) } }
impl Int4SentimentError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Cancelled(c) | Self::Native(StrictInt4Error::Cancelled(c)) => Some(*c),
            Self::Scoring(e) => e.cancellation(), _ => None }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int4SentimentRun {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    pub group_width: usize,
    pub result: SentimentResult,
    pub head_count: usize,
    pub model_work: Int4Work,
    pub rewound_positions: u64,
}

/// Sealed exact input, native schedule and model binding. There is no wire
/// deserialization, unguarded inner-plan conversion or private-prompt Debug.
pub struct PreparedInt4Sentiment {
    inner: SentimentPlan,
    identity: ExecutionIdentity,
    schedules: Vec<CandidateSchedule>,
    group_width: usize,
    work: Int4Work,
    budget: TaskBudget,
}
impl SentimentPlanner {
    pub fn plan_int4_with_control<C: DecodeStepControl>(&self, request: &SentimentRequest,
        context: &PlanContext<'_>, limits: SentimentLimits, model: &Int4ModelIdentity,
        control: &mut C) -> Result<PreparedInt4Sentiment, Int4SentimentError> {
        checkpoint(control)?;
        check_model(context.execution_identity(), model)?;
        self.compile_int4(request, context, limits, model.group_width(), control)
    }
    // Only fixtures can supply a group without an actual loaded model. Public
    // compilation first checks the sealed model identity above.
    fn compile_int4<C: DecodeStepControl>(&self, request: &SentimentRequest,
        context: &PlanContext<'_>, limits: SentimentLimits, group_width: usize,
        control: &mut C) -> Result<PreparedInt4Sentiment, Int4SentimentError> {
        checkpoint(control)?;
        let inner = self.plan_for_profile(request, context, limits, NumericsProfile::StrictQuantized { version: 2 })?;
        checkpoint(control)?;
        PreparedInt4Sentiment::compile(inner, context.execution_identity().clone(), request.budget, group_width, control)
    }
}
impl PreparedInt4Sentiment {
    fn compile<C: DecodeStepControl>(inner: SentimentPlan, mut identity: ExecutionIdentity,
        budget: TaskBudget, group_width: usize, control: &mut C) -> Result<Self, Int4SentimentError> {
        let mut schedules = reserved(inner.heads.len())?;
        let mut work = Int4Work::default();
        for head in &inner.heads {
            checkpoint(control)?;
            let DecodeStrategy::PrefillOnly { candidates } = head.ir.decode_strategy() else {
                return Err(Int4SentimentError::Accounting);
            };
            let schedule = CandidateSchedule::new(group_width, head.prompt_len, candidates, inner.options.mode)?;
            if schedule.scoring != (ScoringWork { prefix_evaluations: head.work.prefix_evaluations,
                scored_edges: head.work.scored_edges, projected_logits: head.work.projected_logits }) {
                return Err(Int4SentimentError::Accounting);
            }
            work = work.checked_add(schedule.model)?;
            schedules.push(schedule);
        }
        // The shared compiler bounds projected logits across all axes. Native
        // admission separately checks positions, attention, group dots and MACs.
        if work.projected_logits != inner.planned_work().projected_logits {
            return Err(Int4SentimentError::Accounting);
        }
        identity.taskir_digest = *inner.binding_digest();
        let mut prompts = reserved(inner.heads.len())?;
        let mut language = reserved(inner.heads.len())?;
        for head in &inner.heads {
            prompts.push((head.axis, head.ir.prompt_segments()));
            language.push((head.axis, head.ir.decode_strategy(), &head.anchors));
        }
        identity.prompt_digest = digest(&prompts)?;
        identity.schema_digest = digest(&language)?;
        identity.grammar_compiler_version = "none".to_owned();
        identity.sampler_version = SAMPLER_VERSION.to_owned();
        identity.decision_policy_digest = digest(&(INT4_SENTIMENT_EXECUTION, inner.binding_digest(), group_width))?;
        identity.validate().map_err(|_| Int4SentimentError::Identity)?;
        checkpoint(control)?;
        Ok(Self { inner, identity, schedules, group_width, work, budget })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn planned_work(&self) -> Int4Work { self.work }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn head_count(&self) -> usize { self.schedules.len() }
    pub fn task_budget(&self) -> TaskBudget { self.budget }
    pub fn max_result_bytes(&self) -> u64 { self.inner.max_output_bytes }
    pub fn required_context(&self) -> usize { self.schedules.iter().map(|s| s.context).max().unwrap_or(0) }
    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), Int4SentimentError> {
        admitted.validate().map_err(|_| Int4SentimentError::Identity)?;
        if admitted != &self.identity { return Err(Int4SentimentError::Identity); }
        Ok(())
    }
    pub fn preflight(&self, admitted: &ExecutionIdentity, engine: &StrictInt4Engine<'_>, budget: Int4ScoringBudget)
        -> Result<(), Int4SentimentError> {
        self.verify_identity(admitted)?;
        check_model(admitted, engine.model_identity())?;
        if engine.model_identity().group_width() != self.group_width { return Err(Int4SentimentError::Identity); }
        check_work(self.work, budget.native)?;
        for &schedule in &self.schedules {
            schedule.preflight(engine, Int4ScoringBudget {
                max_kv_bytes: budget.max_kv_bytes.min(self.budget.max_kv_bytes), ..budget })?;
        }
        Ok(())
    }
    /// Preflight every head before the first forward. Each native session gets
    /// exactly its own work slice and releases logical KV on every exit path.
    pub fn execute_with_control<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt4Engine<'_>, budget: Int4ScoringBudget, control: &mut C)
        -> Result<Int4SentimentRun, Int4SentimentError> {
        self.execute_with_prefill(admitted, engine, budget, None, control)
    }
    /// Execute bounded prompt morsels for each independent axis. Candidate
    /// denominators, frozen run policy and aggregate work remain unchanged.
    /// The caller must admit extra scratch through all heads and drain.
    pub fn execute_layer_major_with_control<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt4Engine<'_>, budget: Int4ScoringBudget,
        prefill: Int4PrefillLimits, control: &mut C) -> Result<Int4SentimentRun, Int4SentimentError> {
        self.execute_with_prefill(admitted, engine, budget, Some(prefill), control)
    }
    fn execute_with_prefill<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt4Engine<'_>, budget: Int4ScoringBudget,
        prefill: Option<Int4PrefillLimits>, control: &mut C) -> Result<Int4SentimentRun, Int4SentimentError> {
        if let Some(limits) = prefill { limits.validate()?; }
        checkpoint(control)?;
        self.preflight(admitted, engine, budget)?;
        self.execute_heads(control, |head, mode, schedule, control| {
            let mut prompt = reserved(head.prompt_len)?;
            prompt.extend(head.ir.prompt_segments().iter().flat_map(|s| s.token_ids().iter().copied()));
            if prompt.len() != head.prompt_len { return Err(Int4SentimentError::Accounting); }
            let run = scoring::prefill::execute_compiled_with_prefill(&prompt, &head.scorer, mode, schedule,
                self.max_result_bytes(), engine, Int4ScoringBudget {
                    native: Int4RunBudget::exact(schedule.model),
                    max_kv_bytes: budget.max_kv_bytes.min(self.budget.max_kv_bytes),
                }, prefill, control)?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                return Err(Int4SentimentError::Accounting);
            }
            Ok(run)
        })
    }
    // Injection remains private: public callers cannot supply scores as native
    // results or turn a deserialized work report into execution authority.
    fn execute_heads<C: DecodeStepControl, F>(&self, control: &mut C, mut evaluate: F)
        -> Result<Int4SentimentRun, Int4SentimentError>
    where F: FnMut(&HeadPlan, ScoringMode, CandidateSchedule, &mut C) -> Result<Int4CandidateRun, Int4SentimentError> {
        let mut next = 0;
        let mut work = Int4Work::default();
        let mut rewound = 0_u64;
        let result = self.inner.execute_heads::<Int4SentimentError, _>(|head, mode| {
            checkpoint(control)?;
            let schedule = *self.schedules.get(next).ok_or(Int4SentimentError::Accounting)?;
            let run = evaluate(head, mode, schedule, control)?;
            if run.schema_version != 1 || run.execution != INT4_SCORING_EXECUTION
                || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != self.group_width
                || run.model_work != schedule.model || run.scores.work != schedule.scoring {
                return Err(Int4SentimentError::Accounting);
            }
            work = work.checked_add(run.model_work)?;
            rewound = rewound.checked_add(run.rewound_positions).ok_or(Int4SentimentError::Accounting)?;
            next += 1;
            checkpoint(control)?;
            Ok(run.scores)
        })?;
        if next != self.head_count() || work != self.work { return Err(Int4SentimentError::Accounting); }
        let run = Int4SentimentRun { schema_version: 1, execution: INT4_SENTIMENT_EXECUTION.to_owned(),
            numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: self.group_width,
            result, head_count: next, model_work: work, rewound_positions: rewound };
        let bytes = canonjson::canonical_bytes(&run).map_err(|_| Int4SentimentError::Serialization)?;
        if bytes.len() as u64 > self.max_result_bytes() { return Err(Int4SentimentError::OutputBudget); }
        checkpoint(control)?;
        Ok(run)
    }
}
pub(crate) fn check_model(identity: &ExecutionIdentity, model: &Int4ModelIdentity) -> Result<(), Int4SentimentError> {
    let bound = bind_candidate_model(identity.clone(), model).map_err(|_| Int4SentimentError::Identity)?;
    if &bound != identity { return Err(Int4SentimentError::Identity); }
    Ok(())
}
fn check_work(work: Int4Work, budget: Int4RunBudget) -> Result<(), Int4SentimentError> {
    if work.forward_positions > budget.max_forward_positions || work.attention_pairs > budget.max_attention_pairs
        || !work.projections.fits(budget.max_projection_work) { return Err(Int4SentimentError::WorkBudget); }
    Ok(())
}
fn digest<T: Serialize + ?Sized>(value: &T) -> Result<Sha256Digest, Int4SentimentError> {
    canonjson::canonical_bytes(value).map(|b| Sha256Digest::of_bytes(&b)).map_err(|_| Int4SentimentError::Serialization)
}
fn reserved<T>(count: usize) -> Result<Vec<T>, Int4SentimentError> {
    let mut out = Vec::new();
    out.try_reserve_exact(count).map_err(|_| Int4SentimentError::Allocation)?;
    Ok(out)
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), Int4SentimentError> {
    match control.prefill_checkpoint(0) { Some(c) => Err(Int4SentimentError::Cancelled(c)), None => Ok(()) }
}

#[cfg(test)] mod tests;
