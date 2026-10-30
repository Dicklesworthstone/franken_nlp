//! Model-bound exclusive and independent multi-label classification on INT4.
//!
//! The existing raw-text compiler and semantic finalizer remain authoritative
//! for prompts, opaque labels, scored EOS, score space and abstention. This
//! wrapper adds real INT4 execution, not a sampled-label or INT8 fallback.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{canonjson, execution_identity::{ExecutionIdentity, NumericsProfile},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        generation::grouped_int4::bind_candidate_model,
        lmhead::scoring::ScoringMode,
        strict_int4::{Int4ModelIdentity, Int4RunBudget, Int4Work, StrictInt4Engine, StrictInt4Error, STRICT_INT4_PROFILE,
            prefill::Int4PrefillLimits,
            scoring::{self, CandidateSchedule, Int4CandidateRun, Int4ScoringBudget, Int4ScoringError, INT4_SCORING_EXECUTION}}},
    tasks::ir::{DecodeStrategy, PlanContext, TaskBudget}};
use super::{ClassificationLimits, ClassificationPlanner, ClassificationPlanningError, ClassificationRequest,
    ClassificationTaskResult, PreparedClassification, planning::{self, ClassificationHead}};

pub const INT4_CLASSIFICATION_EXECUTION: &str = "grouped-int4-independent-classification-heads-v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Int4ClassificationError {
    Planning(ClassificationPlanningError), Scoring(Int4ScoringError), Native(StrictInt4Error),
    Identity, WorkBudget, Accounting,
}
impl fmt::Display for Int4ClassificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Planning(_) => "INT4 classification planning or finalization failed",
            Self::Scoring(_) => "INT4 classification candidate execution failed",
            Self::Native(_) => "INT4 classification native engine refused",
            Self::Identity => "INT4 classification model, partition or execution identity differs",
            Self::WorkBudget => "INT4 classification whole-bundle allowance exceeded",
            Self::Accounting => "INT4 classification complete head/work contract diverged",
        })
    }
}
impl Error for Int4ClassificationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Planning(e) => Some(e), Self::Scoring(e) => Some(e), Self::Native(e) => Some(e), _ => None }
    }
}
impl From<ClassificationPlanningError> for Int4ClassificationError { fn from(e: ClassificationPlanningError) -> Self { Self::Planning(e) } }
impl From<Int4ScoringError> for Int4ClassificationError { fn from(e: Int4ScoringError) -> Self { Self::Scoring(e) } }
impl From<StrictInt4Error> for Int4ClassificationError { fn from(e: StrictInt4Error) -> Self { Self::Native(e) } }
impl Int4ClassificationError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Planning(ClassificationPlanningError::Cancelled(c)) | Self::Native(StrictInt4Error::Cancelled(c)) => Some(*c),
            Self::Scoring(e) => e.cancellation(), _ => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int4ClassificationRun {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    pub group_width: usize,
    pub result: ClassificationTaskResult,
    pub head_count: usize,
    pub model_work: Int4Work,
    pub rewound_positions: u64,
}

/// Private complete task bundle. Schedules store geometry, not extra prompts,
/// scorers or weights. Heads reuse one native context sequentially; their full
/// work is summed and admitted before any head executes. No eager/INT8 coercion.
pub struct PreparedInt4Classification {
    inner: PreparedClassification,
    schedules: Vec<CandidateSchedule>,
    group_width: usize,
    work: Int4Work,
}
impl ClassificationPlanner {
    /// The supplied context must ALREADY opt into INT4 and match the actual
    /// model. Binding is checked before prompt compilation, never repaired.
    pub fn plan_int4_with_control<C: DecodeStepControl>(&self, request: &ClassificationRequest,
        context: &PlanContext<'_>, limits: ClassificationLimits, model: &Int4ModelIdentity,
        control: &mut C) -> Result<PreparedInt4Classification, Int4ClassificationError> {
        check_model(context.execution_identity(), model)?;
        self.compile_int4(request, context, limits, model.group_width(), control)
    }
    fn compile_int4<C: DecodeStepControl>(&self, request: &ClassificationRequest,
        context: &PlanContext<'_>, limits: ClassificationLimits, group_width: usize, control: &mut C)
        -> Result<PreparedInt4Classification, Int4ClassificationError> {
        let inner = self.plan_with_profile(request, context, limits, control, NumericsProfile::StrictQuantized { version: 2 })?;
        let mut schedules = planning::reserved(inner.head_count())?;
        let mut work = Int4Work::default();
        for head in &inner.heads {
            planning::checkpoint(control)?;
            let DecodeStrategy::PrefillOnly { candidates } = head.task.ir().decode_strategy() else {
                return Err(Int4ClassificationError::Accounting);
            };
            let schedule = CandidateSchedule::new(group_width, head.prompt_len, candidates, ScoringMode::FullVocabulary)?;
            if schedule.scoring != head.expected { return Err(Int4ClassificationError::Accounting); }
            work = work.checked_add(schedule.model)?;
            schedules.push(schedule);
        }
        if work.forward_positions != inner.work.forward_positions || work.projected_logits != inner.work.projected_logits {
            return Err(Int4ClassificationError::Accounting);
        }
        planning::checkpoint(control)?;
        Ok(PreparedInt4Classification { inner, schedules, group_width, work })
    }
}
impl PreparedInt4Classification {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.inner.execution_identity() }
    pub fn planned_work(&self) -> Int4Work { self.work }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn head_count(&self) -> usize { self.schedules.len() }
    pub fn task_budget(&self) -> TaskBudget { self.inner.budget }
    pub fn required_context(&self) -> usize { self.schedules.iter().map(|s| s.context).max().unwrap_or(0) }
    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), Int4ClassificationError> {
        self.inner.verify_identity(admitted).map_err(Into::into)
    }
    pub fn preflight(&self, admitted: &ExecutionIdentity, engine: &StrictInt4Engine<'_>, budget: Int4ScoringBudget)
        -> Result<(), Int4ClassificationError> {
        self.verify_identity(admitted)?;
        check_model(admitted, engine.model_identity())?;
        if self.group_width != engine.model_identity().group_width() { return Err(Int4ClassificationError::Identity); }
        check_work_ceiling(self.work, budget.native)?;
        let budget = Int4ScoringBudget { max_kv_bytes: budget.max_kv_bytes.min(self.inner.budget.max_kv_bytes), ..budget };
        for &schedule in &self.schedules { schedule.preflight(engine, budget)?; }
        Ok(())
    }
    /// Every head is preflighted before the first forward. A later error drops
    /// all earlier scores, not a partial classification or successful abstention.
    pub fn execute_with_control<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt4Engine<'_>, budget: Int4ScoringBudget, control: &mut C)
        -> Result<Int4ClassificationRun, Int4ClassificationError> {
        self.execute_with_prefill(admitted, engine, budget, None, control)
    }
    /// Group each independent head's prompt while preserving its complete
    /// language and full-vocabulary denominator. The host reserves the extra
    /// scratch once and retains it through every head and physical drain.
    pub fn execute_layer_major_with_control<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt4Engine<'_>, budget: Int4ScoringBudget,
        prefill: Int4PrefillLimits, control: &mut C) -> Result<Int4ClassificationRun, Int4ClassificationError> {
        self.execute_with_prefill(admitted, engine, budget, Some(prefill), control)
    }
    fn execute_with_prefill<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt4Engine<'_>, budget: Int4ScoringBudget,
        prefill: Option<Int4PrefillLimits>, control: &mut C) -> Result<Int4ClassificationRun, Int4ClassificationError> {
        if let Some(limits) = prefill { limits.validate()?; }
        planning::checkpoint(control)?;
        self.preflight(admitted, engine, budget)?;
        self.execute_heads(control, |_, head, schedule, control| {
            let mut prompt = planning::reserved(head.prompt_len)?;
            prompt.extend(head.task.ir().prompt_segments().iter().flat_map(|s| s.token_ids().iter().copied()));
            if prompt.len() != head.prompt_len { return Err(Int4ClassificationError::Accounting); }
            let run = scoring::prefill::execute_compiled_with_prefill(&prompt, &head.classifier.scorer, ScoringMode::FullVocabulary,
                schedule, self.inner.budget.max_output_bytes, engine,
                Int4ScoringBudget { native: Int4RunBudget::exact(schedule.model),
                    max_kv_bytes: budget.max_kv_bytes.min(self.inner.budget.max_kv_bytes) }, prefill, control)?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                return Err(Int4ClassificationError::Accounting);
            }
            Ok(run)
        })
    }
    // Test-only synthetic execution can exercise the actual semantic finalizer;
    // there is no public mechanism for injecting or deserializing native scores.
    fn execute_heads<C: DecodeStepControl, F>(&self, control: &mut C, mut evaluate: F)
        -> Result<Int4ClassificationRun, Int4ClassificationError>
    where F: FnMut(usize, &ClassificationHead, CandidateSchedule, &mut C) -> Result<Int4CandidateRun, Int4ClassificationError> {
        let mut next = 0; let mut work = Int4Work::default(); let mut rewound = 0_u64;
        let result = self.inner.execute_heads::<Int4ClassificationError, _>(|head| {
            planning::checkpoint(control)?;
            let schedule = *self.schedules.get(next).ok_or(Int4ClassificationError::Accounting)?;
            let run = evaluate(next, head, schedule, control)?;
            if run.schema_version != 1 || run.execution != INT4_SCORING_EXECUTION || run.numerics_profile != STRICT_INT4_PROFILE
                || run.group_width != self.group_width || run.model_work != schedule.model || run.scores.work != schedule.scoring {
                return Err(Int4ClassificationError::Accounting);
            }
            work = work.checked_add(run.model_work)?;
            rewound = rewound.checked_add(run.rewound_positions).ok_or(Int4ClassificationError::Accounting)?;
            next += 1; Ok(run.scores)
        })?;
        if next != self.schedules.len() || work != self.work { return Err(Int4ClassificationError::Accounting); }
        let run = Int4ClassificationRun { schema_version: 1, execution: INT4_CLASSIFICATION_EXECUTION.to_owned(),
            numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: self.group_width,
            result, head_count: next, model_work: work, rewound_positions: rewound };
        planning::check_output(&run, self.inner.budget.max_output_bytes)?;
        if canonjson::canonical_bytes(&run).map_err(|_| ClassificationPlanningError::Serialization)?.len() as u64 > self.inner.budget.max_output_bytes {
            return Err(ClassificationPlanningError::OutputBudget.into());
        }
        planning::checkpoint(control)?; Ok(run)
    }
}
pub(crate) fn check_model(identity: &ExecutionIdentity, model: &Int4ModelIdentity) -> Result<(), Int4ClassificationError> {
    let bound = bind_candidate_model(identity.clone(), model).map_err(|_| Int4ClassificationError::Identity)?;
    if &bound != identity { return Err(Int4ClassificationError::Identity); }
    Ok(())
}
fn check_work_ceiling(work: Int4Work, budget: Int4RunBudget) -> Result<(), Int4ClassificationError> {
    if work.forward_positions > budget.max_forward_positions || work.attention_pairs > budget.max_attention_pairs
        || !work.projections.fits(budget.max_projection_work) { return Err(Int4ClassificationError::WorkBudget); }
    Ok(())
}

mod model;
pub use model::Int4ClassificationPlanner;
#[cfg(test)] mod tests;
