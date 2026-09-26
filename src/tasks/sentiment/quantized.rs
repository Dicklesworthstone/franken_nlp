//! Complete, identity-bound independent sentiment axes on the native INT8 path.
//!
//! Reuses the pinned prompt compiler, candidate scorer and dimensional
//! finalizer. There is no conversion to an eager plan, invented model result,
//! second runtime, sampled label, or cross-axis probability normalization.
use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{
    canonjson,
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest},
    native_engine::{
        artifact_bridge::ArtifactIdentity,
        constrained_int8,
        decode::{DecodeCancellationKind, DecodeStepControl},
        lmhead::scoring::{ScoringMode, ScoringWork},
        strict_int8::{Int8RunBudget, Int8Work, StrictInt8Engine, StrictInt8Error, STRICT_INT8_PROFILE,
            scoring::{self, CandidateSchedule, Int8CandidateRun, Int8ScoringBudget,
                Int8ScoringError, INT8_SCORING_EXECUTION}},
    },
    tasks::ir::{DecodeStrategy, PlanContext, TaskBudget},
};
use super::{SentimentLimits, SentimentPlan, SentimentPlanner, SentimentPlanningError,
    SentimentRequest, SentimentResult, SentimentError, distribution::HeadPlan};

pub const INT8_SENTIMENT_EXECUTION: &str = "portable-int8-independent-affect-heads-v1";
const SAMPLER_VERSION: &str = "sentiment-complete-candidate-eos-v1";

#[derive(Debug)]
pub enum Int8SentimentError {
    Planning(SentimentPlanningError), Task(SentimentError), Scoring(Int8ScoringError),
    Native(StrictInt8Error), Identity, WorkBudget, Accounting, Allocation,
    OutputBudget, Serialization, Cancelled(DecodeCancellationKind),
}
impl fmt::Display for Int8SentimentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Planning(_) => "int8 sentiment planning refused",
            Self::Task(_) => "int8 sentiment independent finalization refused",
            Self::Scoring(_) => "int8 sentiment finite-candidate execution failed",
            Self::Native(_) => "int8 sentiment native engine refused",
            Self::Identity => "int8 sentiment model or admitted identity differs",
            Self::WorkBudget => "int8 sentiment whole-bundle work budget exceeded",
            Self::Accounting => "int8 sentiment complete head/work contract diverged",
            Self::Allocation => "int8 sentiment bounded allocation refused",
            Self::OutputBudget => "complete int8 sentiment result exceeds its byte budget",
            Self::Serialization => "int8 sentiment serialization refused",
            Self::Cancelled(_) => "int8 sentiment cancelled",
        })
    }
}
impl Error for Int8SentimentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Planning(e) => Some(e), Self::Task(e) => Some(e),
            Self::Scoring(e) => Some(e), Self::Native(e) => Some(e), _ => None }
    }
}
impl From<SentimentPlanningError> for Int8SentimentError { fn from(e: SentimentPlanningError) -> Self { Self::Planning(e) } }
impl From<SentimentError> for Int8SentimentError { fn from(e: SentimentError) -> Self { Self::Task(e) } }
impl From<Int8ScoringError> for Int8SentimentError { fn from(e: Int8ScoringError) -> Self { Self::Scoring(e) } }
impl From<StrictInt8Error> for Int8SentimentError { fn from(e: StrictInt8Error) -> Self { Self::Native(e) } }
impl Int8SentimentError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Cancelled(c) | Self::Native(StrictInt8Error::Cancelled(c)) => Some(*c),
            Self::Scoring(e) => e.cancellation(), _ => None }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int8SentimentRun {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    pub result: SentimentResult,
    pub head_count: usize,
    pub model_work: Int8Work,
    pub rewound_positions: u64,
}

/// Private exact input and identity. No Deserialize, Deref, unguarded inner
/// plan conversion or Debug can turn wire metadata into execution authority.
pub struct PreparedInt8Sentiment {
    inner: SentimentPlan,
    identity: ExecutionIdentity,
    schedules: Vec<CandidateSchedule>,
    work: Int8Work,
    budget: TaskBudget,
}
impl SentimentPlanner {
    /// Tokenizer/grammar construction is bounded but not internally preempted.
    /// Poll the same caller control before/after compilation and per schedule.
    pub fn plan_int8_with_control<C: DecodeStepControl>(&self, request: &SentimentRequest,
        context: &PlanContext<'_>, limits: SentimentLimits, control: &mut C)
        -> Result<PreparedInt8Sentiment, Int8SentimentError> {
        checkpoint(control)?;
        let inner = self.plan_for_profile(request, context, limits, NumericsProfile::StrictQuantized { version: 1 })?;
        checkpoint(control)?;
        PreparedInt8Sentiment::compile(inner, context.execution_identity().clone(), request.budget, control)
    }
}
impl PreparedInt8Sentiment {
    fn compile<C: DecodeStepControl>(inner: SentimentPlan, mut identity: ExecutionIdentity,
        budget: TaskBudget, control: &mut C) -> Result<Self, Int8SentimentError> {
        constrained_int8::check_profile(&identity).map_err(|_| Int8SentimentError::Identity)?;
        let mut schedules = Vec::new();
        schedules.try_reserve_exact(inner.heads.len()).map_err(|_| Int8SentimentError::Allocation)?;
        let mut work = Int8Work::default();
        for head in &inner.heads {
            checkpoint(control)?;
            let DecodeStrategy::PrefillOnly { candidates } = head.ir.decode_strategy() else {
                return Err(Int8SentimentError::Accounting);
            };
            let schedule = CandidateSchedule::new(head.prompt_len, candidates, inner.options.mode)?;
            if schedule.scoring != (ScoringWork { prefix_evaluations: head.work.prefix_evaluations,
                scored_edges: head.work.scored_edges, projected_logits: head.work.projected_logits }) {
                return Err(Int8SentimentError::Accounting);
            }
            work = work.checked_add(schedule.model)?;
            schedules.push(schedule);
        }
        // All task-owned identity fields are sealed before admission. The
        // shared binding includes axes, exact IRs/candidates, anchors, policy,
        // scoring mode, EOS and the complete result cap; never export it.
        identity.taskir_digest = *inner.binding_digest();
        let prompts: Vec<_> = inner.heads.iter().map(|h| (h.axis, h.ir.prompt_segments())).collect();
        identity.prompt_digest = digest(&prompts)?;
        let language: Vec<_> = inner.heads.iter().map(|h| (h.axis, h.ir.decode_strategy(), &h.anchors)).collect();
        identity.schema_digest = digest(&language)?;
        identity.grammar_compiler_version = "none".to_owned();
        identity.sampler_version = SAMPLER_VERSION.to_owned();
        identity.decision_policy_digest = digest(&(INT8_SENTIMENT_EXECUTION, inner.binding_digest()))?;
        identity.validate().map_err(|_| Int8SentimentError::Identity)?;
        checkpoint(control)?;
        Ok(Self { inner, identity, schedules, work, budget })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn planned_work(&self) -> Int8Work { self.work }
    pub fn head_count(&self) -> usize { self.schedules.len() }
    pub fn task_budget(&self) -> TaskBudget { self.budget }
    pub fn max_result_bytes(&self) -> u64 { self.inner.max_output_bytes }
    pub fn required_context(&self) -> usize { self.schedules.iter().map(|s| s.context).max().unwrap_or(0) }
    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), Int8SentimentError> {
        admitted.validate().map_err(|_| Int8SentimentError::Identity)?;
        if admitted != &self.identity { return Err(Int8SentimentError::Identity); }
        Ok(())
    }
    pub fn preflight(&self, admitted: &ExecutionIdentity, engine: &StrictInt8Engine<'_>, budget: Int8ScoringBudget)
        -> Result<(), Int8SentimentError> {
        self.verify_identity(admitted)?;
        check_model(admitted, engine.artifact_identity())?;
        check_work(self.work, budget.native)?;
        for &schedule in &self.schedules {
            schedule.preflight(engine, Int8ScoringBudget {
                max_kv_bytes: budget.max_kv_bytes.min(self.budget.max_kv_bytes), ..budget })?;
        }
        Ok(())
    }
    /// Every independent axis is preflighted before the first forward. Each
    /// native session gets only its exact slice of aggregate work. RAII clears
    /// all logical KV on success, failure, cancellation and unwind; the model
    /// and process/output reservations remain owned by the caller.
    pub fn execute_with_control<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt8Engine<'_>, budget: Int8ScoringBudget, control: &mut C)
        -> Result<Int8SentimentRun, Int8SentimentError> {
        checkpoint(control)?;
        self.preflight(admitted, engine, budget)?;
        self.execute_heads(control, |head, mode, schedule, control| {
            let mut prompt = Vec::new();
            prompt.try_reserve_exact(head.prompt_len).map_err(|_| Int8SentimentError::Allocation)?;
            prompt.extend(head.ir.prompt_segments().iter().flat_map(|s| s.token_ids().iter().copied()));
            if prompt.len() != head.prompt_len { return Err(Int8SentimentError::Accounting); }
            let run = scoring::execute_compiled(&prompt, &head.scorer, mode, schedule,
                self.max_result_bytes(), engine, Int8ScoringBudget {
                    native: Int8RunBudget::exact(schedule.model),
                    max_kv_bytes: budget.max_kv_bytes.min(self.budget.max_kv_bytes),
                }, control)?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                return Err(Int8SentimentError::Accounting);
            }
            Ok(run)
        })
    }
    // Fault injection is private. Public callers cannot supply synthetic
    // heads or deserialize a successful execution into an admitted result.
    fn execute_heads<C: DecodeStepControl, F>(&self, control: &mut C, mut evaluate: F)
        -> Result<Int8SentimentRun, Int8SentimentError>
    where F: FnMut(&HeadPlan, ScoringMode, CandidateSchedule, &mut C) -> Result<Int8CandidateRun, Int8SentimentError> {
        let mut next = 0; let mut work = Int8Work::default(); let mut rewound = 0_u64;
        let result = self.inner.execute_heads::<Int8SentimentError, _>(|head, mode| {
            checkpoint(control)?;
            let schedule = *self.schedules.get(next).ok_or(Int8SentimentError::Accounting)?;
            let run = evaluate(head, mode, schedule, control)?;
            if run.schema_version != 1 || run.execution != INT8_SCORING_EXECUTION
                || run.numerics_profile != STRICT_INT8_PROFILE || run.model_work != schedule.model
                || run.scores.work != schedule.scoring { return Err(Int8SentimentError::Accounting); }
            work = work.checked_add(run.model_work)?;
            rewound = rewound.checked_add(run.rewound_positions).ok_or(Int8SentimentError::Accounting)?;
            next += 1;
            checkpoint(control)?;
            Ok(run.scores)
        })?;
        if next != self.head_count() || work != self.work { return Err(Int8SentimentError::Accounting); }
        let run = Int8SentimentRun { schema_version: 1, execution: INT8_SENTIMENT_EXECUTION.to_owned(),
            numerics_profile: STRICT_INT8_PROFILE.to_owned(), result, head_count: next,
            model_work: work, rewound_positions: rewound };
        let bytes = canonjson::canonical_bytes(&run).map_err(|_| Int8SentimentError::Serialization)?;
        if bytes.len() as u64 > self.max_result_bytes() { return Err(Int8SentimentError::OutputBudget); }
        checkpoint(control)?;
        Ok(run)
    }
}
fn digest<T: Serialize + ?Sized>(value: &T) -> Result<Sha256Digest, Int8SentimentError> {
    canonjson::canonical_bytes(value).map(|b| Sha256Digest::of_bytes(&b)).map_err(|_| Int8SentimentError::Serialization)
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), Int8SentimentError> {
    match control.prefill_checkpoint(0) { Some(c) => Err(Int8SentimentError::Cancelled(c)), None => Ok(()) }
}
fn check_model(identity: &ExecutionIdentity, model: &ArtifactIdentity) -> Result<(), Int8SentimentError> {
    constrained_int8::check_profile(identity).map_err(|_| Int8SentimentError::Identity)?;
    if model.model_id != "Nanbeige4.2-3B" || model.revision != identity.source_revision
        || model.recipe_id != identity.quant_recipe
        || Sha256Digest::from_hex(&model.logical_model_sha256).ok() != Some(identity.logical_model_digest) {
        return Err(Int8SentimentError::Identity);
    }
    Ok(())
}
fn check_work(work: Int8Work, budget: Int8RunBudget) -> Result<(), Int8SentimentError> {
    if work.forward_positions > budget.max_forward_positions || work.attention_pairs > budget.max_attention_pairs
        || !work.projections.fits(budget.max_projection_work) { return Err(Int8SentimentError::WorkBudget); }
    Ok(())
}

#[cfg(test)] mod tests;
