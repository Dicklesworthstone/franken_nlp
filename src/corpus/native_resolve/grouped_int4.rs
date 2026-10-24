//! Source-anchored entity resolution on the concrete grouped-INT4 scorer.
//!
//! The shared compiler owns both presentation orders, exact A/B/C-plus-EOS
//! language and source-validated complete-link finalization. This module binds
//! the actual model/group, accounts for every native head, and refuses partial
//! snapshot results. It adds no runtime, loader, alternate kernel or retry.

use super::*;
use crate::native_engine::{
    decode::DecodeCancellationKind,
    generation::grouped_int4::bind_candidate_model,
    strict_int4::{Int4ModelIdentity, Int4RunBudget, Int4Work, StrictInt4Engine,
        StrictInt4Error, STRICT_INT4_EXECUTION, STRICT_INT4_PROFILE,
        scoring::{self, CandidateSchedule, Int4CandidateRun, Int4ScoringBudget,
            Int4ScoringError, INT4_SCORING_EXECUTION}},
};

pub const INT4_RESOLUTION_EXECUTION: &str = "grouped-int4-resolution-two-order-complete-link-v1";

/// Complete graph/prompt limits and one nonrenewable native-work allowance.
/// Group width comes from actual loaded source/snapshot weights, not a caller
/// certificate. The host separately retains source/graph/preparation/output
/// memory admission through physical execution and delivery.
#[derive(Clone, Copy, Debug)]
pub struct Int4ResolveLimits {
    pub planning: NativeResolveLimits,
    pub max_model_work: Int4Work,
}
impl Int4ResolveLimits {
    pub fn validate(self) -> Result<(), Int4ResolveError> {
        self.planning.validate()?;
        // A zero work allowance is valid for a graph with no candidate pairs.
        // The complete compiled schedule checks all five counters below.
        Ok(())
    }
}

pub enum Int4ResolveError {
    Planning(NativeResolveError), Resolution(ResolveError), Scoring(Int4ScoringError),
    Native(StrictInt4Error), Identity, WorkBudget, Accounting,
}
impl fmt::Display for Int4ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Planning(_) => "int4 resolution prompt planning refused",
            Self::Resolution(_) => "int4 resolution source or complete-link finalization refused",
            Self::Scoring(_) => "int4 resolution complete candidate scoring failed",
            Self::Native(_) => "int4 resolution native engine refused",
            Self::Identity => "int4 resolution actual model, group or admitted identity differs",
            Self::WorkBudget => "int4 resolution whole-snapshot work budget exceeded",
            Self::Accounting => "int4 resolution complete two-order work contract diverged",
        })
    }
}
impl fmt::Debug for Int4ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int4ResolveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Planning(e) => Some(e), Self::Resolution(e) => Some(e),
            Self::Scoring(e) => Some(e), Self::Native(e) => Some(e), _ => None }
    }
}
impl From<NativeResolveError> for Int4ResolveError { fn from(e: NativeResolveError) -> Self { Self::Planning(e) } }
impl From<ResolveError> for Int4ResolveError { fn from(e: ResolveError) -> Self { Self::Resolution(e) } }
impl From<Int4ScoringError> for Int4ResolveError { fn from(e: Int4ScoringError) -> Self { Self::Scoring(e) } }
impl From<StrictInt4Error> for Int4ResolveError { fn from(e: StrictInt4Error) -> Self { Self::Native(e) } }
impl Int4ResolveError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Resolution(ResolveError::Cancelled(c))
            | Self::Planning(NativeResolveError::Resolution(ResolveError::Cancelled(c)))
            | Self::Native(StrictInt4Error::Cancelled(c)) => Some(*c),
            Self::Scoring(e) => e.cancellation(), _ => None,
        }
    }
}

/// A sealed borrowed snapshot. The shared eager plan cannot escape or execute
/// through an INT8/BF16 route. No public score-provider or deserialization seam
/// can manufacture a native receipt, including for an empty candidate graph.
pub struct PreparedInt4Resolution<'a, 'p, 's> {
    inner: PreparedNativeResolution<'a, 'p, 's>,
    schedules: Vec<[CandidateSchedule; 2]>,
    identity: ExecutionIdentity,
    group_width: usize,
    work: Int4Work,
}

#[derive(Serialize)]
pub struct Int4ResolutionRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub scoring_execution: &'static str,
    pub group_width: usize,
    /// False means the validated graph had no pairs and no native forward ran.
    pub model_evaluated: bool,
    pub head_count: usize,
    pub planned_model_work: Int4Work,
    pub model_work: Int4Work,
    pub rewound_positions: u64,
    pub result: ResolutionResult,
}

impl ResolutionPlanner {
    /// Bind the actual model before compiling BOTH orders of EVERY pair.
    /// Caller-supplied model fields must already match; they are not repaired.
    /// Even a zero-pair plan receives this model authentication before it can
    /// reach the no-native-engine finalizer.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_int4<'a, 'p, 's, C: DecodeStepControl>(&'a self, plan: &'p ResolutionPlan<'s>,
        identity: &ExecutionIdentity, limits: Int4ResolveLimits, model: &Int4ModelIdentity,
        control: &mut C) -> Result<PreparedInt4Resolution<'a, 'p, 's>, Int4ResolveError> {
        resolve::checkpoint(control)?;
        check_model(identity, model)?;
        self.compile_int4(plan, identity, limits, model.group_width(), control)
    }

    // Private fixtures can exercise pinned planning without allocating model
    // weights. Production can enter only after the actual model check above.
    fn compile_int4<'a, 'p, 's, C: DecodeStepControl>(&'a self, plan: &'p ResolutionPlan<'s>,
        identity: &ExecutionIdentity, limits: Int4ResolveLimits, group_width: usize,
        control: &mut C) -> Result<PreparedInt4Resolution<'a, 'p, 's>, Int4ResolveError> {
        resolve::checkpoint(control)?;
        check_profile(identity)?;
        limits.validate()?;
        // Validate geometry even if there are no candidate heads to schedule.
        Int4Work::for_sequence(group_width, 0, 0, 0)?;
        let mut inner = self.prepare_profile(plan, identity, limits.planning, control,
            NumericsProfile::StrictQuantized { version: 2 })?;
        let mut schedules = resolve::reserved(inner.pair_count())?;
        let mut work = Int4Work::default();
        for pair in &mut inner.pairs {
            resolve::checkpoint(control)?;
            let heads = [
                CandidateSchedule::new(group_width, pair.prompts[0].len(), &self.candidates, ScoringMode::FullVocabulary)?,
                CandidateSchedule::new(group_width, pair.prompts[1].len(), &self.candidates, ScoringMode::FullVocabulary)?,
            ];
            for (head, expected) in heads.iter().zip(pair.head_work) {
                if head.model.forward_positions != expected.forward_positions
                    || head.model.projected_logits != expected.projected_logits
                    || head.scoring.prefix_evaluations != 4 || head.scoring.scored_edges != 6 {
                    return Err(Int4ResolveError::Accounting);
                }
                work = work.checked_add(head.model)?;
                if !within(work, limits.max_model_work) { return Err(Int4ResolveError::WorkBudget); }
            }
            pair.identity.decision_policy_digest = digest(&(INT4_RESOLUTION_EXECUTION,
                INT4_SCORING_EXECUTION, group_width, pair.identity.decision_policy_digest))?;
            schedules.push(heads);
        }
        if work.forward_positions != inner.work.forward_positions || work.projected_logits != inner.work.projected_logits {
            return Err(Int4ResolveError::Accounting);
        }
        resolve::checkpoint(control)?;
        Ok(PreparedInt4Resolution { inner, schedules, identity: identity.clone(), group_width, work })
    }
}

impl PreparedInt4Resolution<'_, '_, '_> {
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> {
        self.inner.pairs.iter().map(|pair| &pair.identity)
    }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn planned_work(&self) -> Int4Work { self.work }
    pub fn pair_count(&self) -> usize { self.inner.pair_count() }
    /// Planning bounds pairs to 65,536, so both presentation orders fit usize.
    pub fn head_count(&self) -> usize { self.pair_count() * 2 }
    pub fn required_context_tokens(&self) -> usize { self.inner.required_context_tokens() }
    pub fn task_budget(&self) -> TaskBudget { self.inner.limits.per_head }
    pub fn max_result_bytes(&self) -> u64 {
        (self.inner.limits.max_result_bytes as u64).min(self.task_budget().max_output_bytes)
    }
    fn verify_admitted<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity], control: &mut C)
        -> Result<(), Int4ResolveError> {
        if admitted.len() != self.pair_count() || self.schedules.len() != self.pair_count() {
            return Err(Int4ResolveError::Identity);
        }
        for (expected, actual) in self.execution_identities().zip(admitted) {
            resolve::checkpoint(control)?;
            verify_identity(expected, actual).map_err(|_| Int4ResolveError::Identity)?;
        }
        Ok(())
    }
    /// Check the complete admitted set, actual model, nonrenewable budget and
    /// every native head before the first forward. Price the whole allocated
    /// resident context, even when every pair uses a shorter prompt.
    pub fn preflight<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity], engine: &StrictInt4Engine<'_>,
        budget: Int4ScoringBudget, control: &mut C) -> Result<(), Int4ResolveError> {
        resolve::checkpoint(control)?;
        self.verify_admitted(admitted, control)?;
        check_model(&self.identity, engine.model_identity())?;
        if engine.model_identity().group_width() != self.group_width {
            return Err(Int4ResolveError::Identity);
        }
        check_budget(self.work, budget.native)?;
        if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
            return Err(Int4ResolveError::Accounting);
        }
        let kv_cap = budget.max_kv_bytes.min(self.task_budget().max_kv_bytes);
        check_kv_capacity(engine.kv_cache().capacity_positions(), self.required_context_tokens(), kv_cap)?;
        for (pair, schedules) in self.inner.pairs.iter().zip(&self.schedules) {
            resolve::checkpoint(control)?;
            check_model(&pair.identity, engine.model_identity())?;
            for &schedule in schedules {
                schedule.preflight(engine, Int4ScoringBudget {
                    native: Int4RunBudget::exact(schedule.model), max_kv_bytes: kv_cap,
                })?;
            }
        }
        Ok(())
    }

    /// This skips only the native engine, never required pair comparisons.
    /// The original model-bound preparation and source validation still apply.
    pub fn finalize_without_model<C: DecodeStepControl>(self, control: &mut C)
        -> Result<Int4ResolutionRun, Int4ResolveError> {
        resolve::checkpoint(control)?;
        if self.pair_count() != 0 || self.work != Int4Work::default() {
            return Err(Int4ResolveError::Accounting);
        }
        self.execute_heads(&[], control, |_, _, _, _, _| Err(Int4ResolveError::Accounting))
    }

    /// One exclusive native engine, no pair-local retry or whole-budget
    /// renewal. Failure discards every provisional score. The caller retains
    /// all process-memory guards until physical completion and output delivery.
    pub fn execute_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt4Engine<'_>, budget: Int4ScoringBudget, control: &mut C)
        -> Result<Int4ResolutionRun, Int4ResolveError> {
        self.preflight(admitted, engine, budget, control)?;
        let planner = self.inner.planner;
        let kv_cap = budget.max_kv_bytes.min(self.task_budget().max_kv_bytes);
        let head_output = self.task_budget().max_output_bytes;
        self.execute_heads(admitted, control, |_, _, prompt, schedule, control| {
            let run = scoring::execute_compiled(prompt, &planner.scorer, ScoringMode::FullVocabulary,
                schedule, head_output, engine, Int4ScoringBudget {
                    native: Int4RunBudget::exact(schedule.model), max_kv_bytes: kv_cap,
                }, control)?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                return Err(Int4ResolveError::Accounting);
            }
            Ok(run)
        })
    }

    // Only private fixtures replace this numerical boundary; they still use
    // pinned planning, complete-score validation and the real graph finalizer.
    fn execute_heads<C: DecodeStepControl, F>(self, admitted: &[ExecutionIdentity], control: &mut C, mut evaluate: F)
        -> Result<Int4ResolutionRun, Int4ResolveError>
    where F: FnMut(usize, usize, &[u32], CandidateSchedule, &mut C) -> Result<Int4CandidateRun, Int4ResolveError> {
        resolve::checkpoint(control)?;
        self.verify_admitted(admitted, control)?;
        let max_result_bytes = self.max_result_bytes() as usize;
        let expected_heads = self.head_count();
        let mut scores = resolve::reserved(self.pair_count())?;
        let mut work = Int4Work::default();
        let mut rewound = 0_u64;
        let mut head_count = 0_usize;
        for (index, (pair, schedules)) in self.inner.pairs.into_iter().zip(self.schedules).enumerate() {
            let mut results = [PairLogProbabilities { same: 0.0, different: 0.0, uncertain: 0.0 }; 2];
            for (order, schedule) in schedules.into_iter().enumerate() {
                resolve::checkpoint(control)?;
                let run = evaluate(index, order, &pair.prompts[order], schedule, control)?;
                resolve::checkpoint(control)?;
                if run.schema_version != 1 || run.execution != INT4_SCORING_EXECUTION
                    || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != self.group_width
                    || run.model_work != schedule.model || run.scores.work != schedule.scoring
                    || run.rewound_positions != 2 {
                    return Err(Int4ResolveError::Accounting);
                }
                results[order] = validate_scores(&run.scores, self.inner.planner.eos)?;
                work = work.checked_add(run.model_work)?;
                rewound = rewound.checked_add(run.rewound_positions).ok_or(Int4ResolveError::Accounting)?;
                if !within(work, self.work) { return Err(Int4ResolveError::Accounting); }
                head_count = head_count.checked_add(1).ok_or(Int4ResolveError::Accounting)?;
            }
            scores.push(pair.ticket.finish(BidirectionalScores { forward: results[0], reverse: results[1] }));
        }
        if head_count != expected_heads || work != self.work { return Err(Int4ResolveError::Accounting); }
        resolve::checkpoint(control)?;
        let result = self.inner.plan.finalize(scores, control)?;
        let output = Int4ResolutionRun { schema_version: 1, execution: INT4_RESOLUTION_EXECUTION,
            numerics_profile: STRICT_INT4_PROFILE, scoring_execution: INT4_SCORING_EXECUTION,
            group_width: self.group_width, model_evaluated: head_count != 0, head_count,
            planned_model_work: self.work, model_work: work, rewound_positions: rewound, result };
        resolve::check_output(&output, max_result_bytes)?;
        resolve::checkpoint(control)?;
        Ok(output)
    }
}

pub(crate) fn check_model(identity: &ExecutionIdentity, model: &Int4ModelIdentity) -> Result<(), Int4ResolveError> {
    check_profile(identity)?;
    let bound = bind_candidate_model(identity.clone(), model).map_err(|_| Int4ResolveError::Identity)?;
    if &bound != identity { return Err(Int4ResolveError::Identity); }
    Ok(())
}
fn check_profile(identity: &ExecutionIdentity) -> Result<(), Int4ResolveError> {
    identity.validate().map_err(|_| Int4ResolveError::Identity)?;
    if identity.task_spec != RESOLVE_VERSION || identity.numerics_profile != (NumericsProfile::StrictQuantized { version: 2 })
        || identity.backend_semantic_version != STRICT_INT4_EXECUTION || identity.kv_dtype != "bf16"
        || identity.thinking_mode != ThinkingMode::Disabled || identity.tool_mode != ToolMode::None {
        return Err(Int4ResolveError::Identity);
    }
    Ok(())
}
fn within(work: Int4Work, cap: Int4Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn check_budget(work: Int4Work, cap: Int4RunBudget) -> Result<(), Int4ResolveError> {
    // Int4RunBudget exposes native forward/attention/integer axes; the fifth,
    // projected logits, is sealed by the aggregate planning cap and receipts.
    if work.forward_positions > cap.max_forward_positions || work.attention_pairs > cap.max_attention_pairs
        || !work.projections.fits(cap.max_projection_work) { return Err(Int4ResolveError::WorkBudget); }
    Ok(())
}
fn check_kv_capacity(capacity: usize, required: usize, cap: u64) -> Result<(), Int4ResolveError> {
    if capacity < required || (capacity as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
        .is_none_or(|bytes| bytes > cap) { return Err(Int4ResolveError::WorkBudget); }
    Ok(())
}

#[cfg(test)]
mod tests;
