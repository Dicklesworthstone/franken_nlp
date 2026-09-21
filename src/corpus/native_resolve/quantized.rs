//! Source-anchored entity resolution on the real strict-INT8 candidate scorer.
//!
//! Shared lexical blocking, prompt construction, exact A/B/C-plus-EOS language
//! and complete-link finalization remain the semantic authorities. Only the
//! backend-specific schedules/execution change. All pairs are admitted before
//! any forward; neither a pair nor a presentation order renews the whole budget.

use super::*;
use crate::native_engine::{
    artifact_bridge::ArtifactIdentity,
    constrained_int8,
    decode::DecodeCancellationKind,
    strict_int8::{Int8RunBudget, Int8Work, StrictInt8Engine, StrictInt8Error, STRICT_INT8_PROFILE,
        scoring::{self, CandidateSchedule, Int8CandidateRun, Int8ScoringBudget,
            Int8ScoringError, INT8_SCORING_EXECUTION}},
};

pub const INT8_RESOLUTION_EXECUTION: &str = "portable-int8-resolution-two-order-complete-link-v1";

/// The existing finite graph/prompt limits plus all five native work ceilings.
/// No default grants an effectively unlimited neural-work allowance. The host
/// must separately retain real source/plan/graph/output memory reservations.
#[derive(Clone, Copy, Debug)]
pub struct Int8ResolveLimits {
    pub planning: NativeResolveLimits,
    pub max_model_work: Int8Work,
}

pub enum Int8ResolveError {
    Planning(NativeResolveError), Resolution(ResolveError), Scoring(Int8ScoringError),
    Native(StrictInt8Error), Identity, WorkBudget, Accounting,
}
impl fmt::Display for Int8ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Planning(_) => "int8 resolution prompt planning refused",
            Self::Resolution(_) => "int8 resolution source or complete-link finalization refused",
            Self::Scoring(_) => "int8 resolution candidate scoring failed",
            Self::Native(_) => "int8 resolution native engine refused",
            Self::Identity => "int8 resolution model or complete admission identity differs",
            Self::WorkBudget => "int8 resolution whole-snapshot work budget exceeded",
            Self::Accounting => "int8 resolution complete two-order work contract diverged",
        })
    }
}
impl fmt::Debug for Int8ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int8ResolveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Planning(e) => Some(e), Self::Resolution(e) => Some(e),
            Self::Scoring(e) => Some(e), Self::Native(e) => Some(e), _ => None }
    }
}
impl From<NativeResolveError> for Int8ResolveError { fn from(e: NativeResolveError) -> Self { Self::Planning(e) } }
impl From<ResolveError> for Int8ResolveError { fn from(e: ResolveError) -> Self { Self::Resolution(e) } }
impl From<Int8ScoringError> for Int8ResolveError { fn from(e: Int8ScoringError) -> Self { Self::Scoring(e) } }
impl From<StrictInt8Error> for Int8ResolveError { fn from(e: StrictInt8Error) -> Self { Self::Native(e) } }
impl Int8ResolveError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Resolution(ResolveError::Cancelled(c))
            | Self::Planning(NativeResolveError::Resolution(ResolveError::Cancelled(c)))
            | Self::Native(StrictInt8Error::Cancelled(c)) => Some(*c),
            Self::Scoring(e) => e.cancellation(), _ => None,
        }
    }
}

/// This private shared core cannot escape into the eager executor. No Deref,
/// deserialization, profile rewrite or conversion to PreparedNativeResolution.
/// Execution consumes the complete snapshot commitment, not individual pairs.
pub struct PreparedInt8Resolution<'a, 'p, 's> {
    inner: PreparedNativeResolution<'a, 'p, 's>,
    schedules: Vec<[CandidateSchedule; 2]>,
    identity: ExecutionIdentity,
    work: Int8Work,
}

#[derive(Serialize)]
pub struct Int8ResolutionRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub scoring_execution: &'static str,
    /// False for a graph with no candidate pairs: no neural result is invented.
    pub model_evaluated: bool,
    pub head_count: usize,
    pub planned_model_work: Int8Work,
    pub model_work: Int8Work,
    pub rewound_positions: u64,
    pub result: ResolutionResult,
}

impl ResolutionPlanner {
    /// Compile BOTH orders of the ENTIRE source-validated candidate graph.
    /// The input identity must already name strict INT8; no eager executable is
    /// relabeled. Context/token/graph limits are the existing bounded compiler's.
    pub fn prepare_int8<'a, 'p, 's, C: DecodeStepControl>(&'a self, plan: &'p ResolutionPlan<'s>,
        identity: &ExecutionIdentity, limits: Int8ResolveLimits, control: &mut C)
        -> Result<PreparedInt8Resolution<'a, 'p, 's>, Int8ResolveError> {
        resolve::checkpoint(control)?;
        constrained_int8::check_profile(identity).map_err(|_| Int8ResolveError::Identity)?;
        let mut inner = self.prepare_profile(plan, identity, limits.planning, control,
            NumericsProfile::StrictQuantized { version: 1 })?;
        let mut schedules = resolve::reserved(inner.pair_count())?;
        let mut work = Int8Work::default();
        for pair in &mut inner.pairs {
            resolve::checkpoint(control)?;
            let heads = [
                CandidateSchedule::new(pair.prompts[0].len(), &self.candidates, ScoringMode::FullVocabulary)?,
                CandidateSchedule::new(pair.prompts[1].len(), &self.candidates, ScoringMode::FullVocabulary)?,
            ];
            for (head, expected) in heads.iter().zip(pair.head_work) {
                if head.model.forward_positions != expected.forward_positions
                    || head.model.projected_logits != expected.projected_logits
                    || head.scoring.prefix_evaluations != 4 || head.scoring.scored_edges != 6 {
                    return Err(Int8ResolveError::Accounting);
                }
                work = work.checked_add(head.model)?;
                if !within(work, limits.max_model_work) { return Err(Int8ResolveError::WorkBudget); }
            }
            pair.identity.decision_policy_digest = digest(&(INT8_RESOLUTION_EXECUTION,
                INT8_SCORING_EXECUTION, pair.identity.decision_policy_digest))?;
            schedules.push(heads);
        }
        if work.forward_positions != inner.work.forward_positions || work.projected_logits != inner.work.projected_logits {
            return Err(Int8ResolveError::Accounting);
        }
        resolve::checkpoint(control)?;
        Ok(PreparedInt8Resolution { inner, schedules, identity: identity.clone(), work })
    }
}
impl PreparedInt8Resolution<'_, '_, '_> {
    pub fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> {
        self.inner.pairs.iter().map(|pair| &pair.identity)
    }
    pub fn planned_work(&self) -> Int8Work { self.work }
    pub fn pair_count(&self) -> usize { self.inner.pair_count() }
    pub fn required_context_tokens(&self) -> usize { self.inner.required_context_tokens() }
    pub fn task_budget(&self) -> TaskBudget { self.inner.limits.per_head }
    pub fn max_result_bytes(&self) -> u64 {
        (self.inner.limits.max_result_bytes as u64).min(self.inner.limits.per_head.max_output_bytes)
    }
    fn verify_admitted<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity], control: &mut C)
        -> Result<(), Int8ResolveError> {
        if admitted.len() != self.pair_count() || self.schedules.len() != self.pair_count() {
            return Err(Int8ResolveError::Identity);
        }
        for (expected, actual) in self.execution_identities().zip(admitted) {
            resolve::checkpoint(control)?;
            verify_identity(expected, actual).map_err(|_| Int8ResolveError::Identity)?;
        }
        Ok(())
    }
    /// Validate the last pair as well as the first, BEFORE model work. The
    /// complete resident KV capacity is priced, not just the live prompt.
    pub fn preflight<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity], engine: &StrictInt8Engine<'_>,
        budget: Int8ScoringBudget, control: &mut C) -> Result<(), Int8ResolveError> {
        resolve::checkpoint(control)?;
        self.verify_admitted(admitted, control)?;
        check_model(&self.identity, engine.artifact_identity())?;
        check_budget(self.work, budget.native)?;
        if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
            return Err(Int8ResolveError::Accounting);
        }
        let kv_cap = budget.max_kv_bytes.min(self.task_budget().max_kv_bytes);
        if (engine.kv_cache().capacity_positions() as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
            .is_none_or(|bytes| bytes > kv_cap) { return Err(Int8ResolveError::WorkBudget); }
        for (pair, schedules) in self.inner.pairs.iter().zip(&self.schedules) {
            resolve::checkpoint(control)?;
            check_model(&pair.identity, engine.artifact_identity())?;
            for &schedule in schedules {
                schedule.preflight(engine, Int8ScoringBudget { max_kv_bytes: kv_cap, ..budget })?;
            }
        }
        Ok(())
    }
    /// The caller keeps process/preparation/output reservations through actual
    /// delivery. This borrows one native engine and discards every partial
    /// assessment on failure. No loader, runtime, retries or pair-local budget
    /// renewal exists. A host should discard a failed native invocation.
    pub fn execute_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt8Engine<'_>, budget: Int8ScoringBudget, control: &mut C)
        -> Result<Int8ResolutionRun, Int8ResolveError> {
        self.preflight(admitted, engine, budget, control)?;
        let planner = self.inner.planner;
        let kv_cap = budget.max_kv_bytes.min(self.task_budget().max_kv_bytes);
        let head_output = self.task_budget().max_output_bytes;
        self.execute_heads(admitted, control, |_, _, prompt, schedule, control| {
            let run = scoring::execute_compiled(prompt, &planner.scorer, ScoringMode::FullVocabulary,
                schedule, head_output, engine,
                Int8ScoringBudget { native: Int8RunBudget::exact(schedule.model), max_kv_bytes: kv_cap }, control)?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                return Err(Int8ResolveError::Accounting);
            }
            Ok(run)
        })
    }
    // Private fault-injection seam. Production always calls the concrete
    // strict-int8 scorer above; callers cannot manufacture a native receipt.
    fn execute_heads<C: DecodeStepControl, F>(self, admitted: &[ExecutionIdentity], control: &mut C, mut evaluate: F)
        -> Result<Int8ResolutionRun, Int8ResolveError>
    where F: FnMut(usize, usize, &[u32], CandidateSchedule, &mut C) -> Result<Int8CandidateRun, Int8ResolveError> {
        resolve::checkpoint(control)?;
        self.verify_admitted(admitted, control)?;
        let max_result_bytes = self.max_result_bytes() as usize;
        let expected_heads = self.pair_count().checked_mul(2).ok_or(Int8ResolveError::Accounting)?;
        let mut scores = resolve::reserved(self.pair_count())?;
        let mut work = Int8Work::default(); let mut rewound = 0_u64; let mut head_count = 0_usize;
        for (index, (pair, schedules)) in self.inner.pairs.into_iter().zip(self.schedules).enumerate() {
            let mut results = [PairLogProbabilities { same: 0.0, different: 0.0, uncertain: 0.0 }; 2];
            for (order, schedule) in schedules.into_iter().enumerate() {
                resolve::checkpoint(control)?;
                let run = evaluate(index, order, &pair.prompts[order], schedule, control)?;
                resolve::checkpoint(control)?;
                if run.schema_version != 1 || run.execution != INT8_SCORING_EXECUTION
                    || run.numerics_profile != STRICT_INT8_PROFILE || run.model_work != schedule.model
                    || run.scores.work != schedule.scoring || run.rewound_positions != 2 {
                    return Err(Int8ResolveError::Accounting);
                }
                results[order] = validate_scores(&run.scores, self.inner.planner.eos)?;
                work = work.checked_add(run.model_work)?;
                rewound = rewound.checked_add(run.rewound_positions).ok_or(Int8ResolveError::Accounting)?;
                if !within(work, self.work) { return Err(Int8ResolveError::Accounting); }
                head_count += 1;
            }
            scores.push(pair.ticket.finish(BidirectionalScores { forward: results[0], reverse: results[1] }));
        }
        if head_count != expected_heads || work != self.work { return Err(Int8ResolveError::Accounting); }
        resolve::checkpoint(control)?;
        let result = self.inner.plan.finalize(scores, control)?;
        let output = Int8ResolutionRun { schema_version: 1, execution: INT8_RESOLUTION_EXECUTION,
            numerics_profile: STRICT_INT8_PROFILE, scoring_execution: INT8_SCORING_EXECUTION,
            model_evaluated: head_count != 0, head_count, planned_model_work: self.work,
            model_work: work, rewound_positions: rewound, result };
        resolve::check_output(&output, max_result_bytes)?;
        resolve::checkpoint(control)?;
        Ok(output)
    }
}
fn check_model(identity: &ExecutionIdentity, model: &ArtifactIdentity) -> Result<(), Int8ResolveError> {
    constrained_int8::check_profile(identity).map_err(|_| Int8ResolveError::Identity)?;
    if model.model_id != "Nanbeige4.2-3B" || model.revision != identity.source_revision
        || model.recipe_id != identity.quant_recipe
        || Sha256Digest::from_hex(&model.logical_model_sha256).ok() != Some(identity.logical_model_digest) {
        return Err(Int8ResolveError::Identity);
    }
    Ok(())
}
fn within(work: Int8Work, cap: Int8Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn check_budget(work: Int8Work, cap: Int8RunBudget) -> Result<(), Int8ResolveError> {
    if work.forward_positions > cap.max_forward_positions || work.attention_pairs > cap.max_attention_pairs
        || !work.projections.fits(cap.max_projection_work) { return Err(Int8ResolveError::WorkBudget); }
    Ok(())
}

#[cfg(test)] mod tests;
