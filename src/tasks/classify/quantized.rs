//! Raw-text exclusive and independent multi-label classification on int8.
//!
//! Reuses the existing planner, exact candidate languages and finalizer. The
//! wrapper never exposes a quantized PreparedClassification to eager drivers,
//! rewrites a BF16 identity, or produces partial label successes. No second
//! tokenizer, template renderer, model loader, runtime or weight copy exists.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{canonjson, execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest},
    native_engine::{artifact_bridge::ArtifactIdentity, decode::{DecodeCancellationKind, DecodeStepControl},
        lmhead::scoring::ScoringMode,
        strict_int8::{Int8RunBudget, Int8Work, StrictInt8Engine, StrictInt8Error, STRICT_INT8_EXECUTION, STRICT_INT8_PROFILE,
            scoring::{self, CandidateSchedule, Int8CandidateRun, Int8ScoringBudget, Int8ScoringError, INT8_SCORING_EXECUTION}}},
    tasks::ir::{DecodeStrategy, PlanContext, TaskBudget}};
use super::{ClassificationLimits, ClassificationPlanner, ClassificationPlanningError, ClassificationRequest,
    ClassificationTaskResult, PreparedClassification, planning::{self, ClassificationHead}};

pub const INT8_CLASSIFICATION_EXECUTION: &str = "portable-int8-independent-classification-heads-v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Int8ClassificationError {
    Planning(ClassificationPlanningError), Scoring(Int8ScoringError), Native(StrictInt8Error),
    Identity, WorkBudget, Accounting,
}
impl fmt::Display for Int8ClassificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Planning(_) => "int8 classification planning or finalization failed",
            Self::Scoring(_) => "int8 classification candidate execution failed",
            Self::Native(_) => "int8 classification native engine refused",
            Self::Identity => "int8 classification model or execution identity differs",
            Self::WorkBudget => "int8 classification whole-bundle allowance exceeded",
            Self::Accounting => "int8 classification complete head/work contract diverged",
        })
    }
}
impl Error for Int8ClassificationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Planning(e) => Some(e), Self::Scoring(e) => Some(e), Self::Native(e) => Some(e), _ => None }
    }
}
impl From<ClassificationPlanningError> for Int8ClassificationError { fn from(e: ClassificationPlanningError) -> Self { Self::Planning(e) } }
impl From<Int8ScoringError> for Int8ClassificationError { fn from(e: Int8ScoringError) -> Self { Self::Scoring(e) } }
impl From<StrictInt8Error> for Int8ClassificationError { fn from(e: StrictInt8Error) -> Self { Self::Native(e) } }
impl Int8ClassificationError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Planning(ClassificationPlanningError::Cancelled(c)) | Self::Native(StrictInt8Error::Cancelled(c)) => Some(*c),
            Self::Scoring(e) => e.cancellation(), _ => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int8ClassificationRun {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    pub result: ClassificationTaskResult,
    pub head_count: usize,
    pub model_work: Int8Work,
    pub rewound_positions: u64,
}

/// Exact request-owned bundle. Schedules add small work records, not duplicate
/// prompt/scorer/weight storage. Only one flattened token vector is live at a
/// time during execution. No conversion or Deref exposes the private inner plan.
pub struct PreparedInt8Classification {
    inner: PreparedClassification,
    schedules: Vec<CandidateSchedule>,
    work: Int8Work,
}
impl ClassificationPlanner {
    pub fn plan_int8_with_control<C: DecodeStepControl>(&self, request: &ClassificationRequest,
        context: &PlanContext<'_>, limits: ClassificationLimits, control: &mut C)
        -> Result<PreparedInt8Classification, Int8ClassificationError> {
        let inner = self.plan_with_profile(request, context, limits, control, NumericsProfile::StrictQuantized { version: 1 })?;
        let mut schedules = planning::reserved(inner.head_count())?;
        let mut work = Int8Work::default();
        for head in &inner.heads {
            planning::checkpoint(control)?;
            let DecodeStrategy::PrefillOnly { candidates } = head.task.ir().decode_strategy() else {
                return Err(Int8ClassificationError::Accounting);
            };
            let schedule = CandidateSchedule::new(head.prompt_len, candidates, ScoringMode::FullVocabulary)?;
            if schedule.scoring != head.expected { return Err(Int8ClassificationError::Accounting); }
            work = work.checked_add(schedule.model)?;
            schedules.push(schedule);
        }
        if work.forward_positions != inner.work.forward_positions || work.projected_logits != inner.work.projected_logits {
            return Err(Int8ClassificationError::Accounting);
        }
        planning::checkpoint(control)?;
        Ok(PreparedInt8Classification { inner, schedules, work })
    }
}
impl PreparedInt8Classification {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.inner.execution_identity() }
    pub fn planned_work(&self) -> Int8Work { self.work }
    pub fn head_count(&self) -> usize { self.schedules.len() }
    pub fn task_budget(&self) -> TaskBudget { self.inner.budget }
    pub fn required_context(&self) -> usize { self.schedules.iter().map(|s| s.context).max().unwrap_or(0) }
    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), Int8ClassificationError> {
        self.inner.verify_identity(admitted).map_err(Into::into)
    }
    /// Validate actual materialized model facts as well as the complete host
    /// identity. The bridge has no publisher/packing authentication authority;
    /// this comparison does not manufacture one or weaken the production gate.
    pub fn preflight(&self, admitted: &ExecutionIdentity, engine: &StrictInt8Engine<'_>, budget: Int8ScoringBudget)
        -> Result<(), Int8ClassificationError> {
        self.verify_identity(admitted)?;
        check_model(admitted, engine.artifact_identity())?;
        check_work_ceiling(self.work, budget.native)?;
        let budget = Int8ScoringBudget { max_kv_bytes: budget.max_kv_bytes.min(self.inner.budget.max_kv_bytes), ..budget };
        for &schedule in &self.schedules { schedule.preflight(engine, budget)?; }
        Ok(())
    }
    /// Every head is preflighted BEFORE the first forward. Native per-head
    /// budgets are exact slices of the aggregate ceiling, not renewed copies.
    /// The embedding host retains its real admission/output guards throughout.
    pub fn execute_with_control<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt8Engine<'_>, budget: Int8ScoringBudget, control: &mut C)
        -> Result<Int8ClassificationRun, Int8ClassificationError> {
        planning::checkpoint(control)?;
        self.preflight(admitted, engine, budget)?;
        self.execute_heads(control, |_, head, schedule, control| {
            let mut prompt = planning::reserved(head.prompt_len)?;
            prompt.extend(head.task.ir().prompt_segments().iter().flat_map(|s| s.token_ids().iter().copied()));
            if prompt.len() != head.prompt_len { return Err(Int8ClassificationError::Accounting); }
            let run = scoring::execute_compiled(&prompt, &head.classifier.scorer, ScoringMode::FullVocabulary,
                schedule, self.inner.budget.max_output_bytes, engine,
                Int8ScoringBudget { native: Int8RunBudget::exact(schedule.model),
                    max_kv_bytes: budget.max_kv_bytes.min(self.inner.budget.max_kv_bytes) }, control)?;
            if !engine.kv_cache().all_slots_have_len(0) { return Err(Int8ClassificationError::Accounting); }
            Ok(run)
        })
    }
    // Private seam: synthetic tests exercise the REAL semantic finalizer and
    // complete-head contract without exposing a fake-native public producer.
    fn execute_heads<C: DecodeStepControl, F>(&self, control: &mut C, mut evaluate: F)
        -> Result<Int8ClassificationRun, Int8ClassificationError>
    where F: FnMut(usize, &ClassificationHead, CandidateSchedule, &mut C) -> Result<Int8CandidateRun, Int8ClassificationError> {
        let mut next = 0; let mut work = Int8Work::default(); let mut rewound = 0_u64;
        let result = self.inner.execute_heads::<Int8ClassificationError, _>(|head| {
            planning::checkpoint(control)?;
            let schedule = *self.schedules.get(next).ok_or(Int8ClassificationError::Accounting)?;
            let run = evaluate(next, head, schedule, control)?;
            if run.schema_version != 1 || run.execution != INT8_SCORING_EXECUTION || run.numerics_profile != STRICT_INT8_PROFILE
                || run.model_work != schedule.model || run.scores.work != schedule.scoring {
                return Err(Int8ClassificationError::Accounting);
            }
            work = work.checked_add(run.model_work)?;
            rewound = rewound.checked_add(run.rewound_positions).ok_or(Int8ClassificationError::Accounting)?;
            next += 1; Ok(run.scores)
        })?;
        if next != self.schedules.len() || work != self.work { return Err(Int8ClassificationError::Accounting); }
        let run = Int8ClassificationRun { schema_version: 1, execution: INT8_CLASSIFICATION_EXECUTION.to_owned(),
            numerics_profile: STRICT_INT8_PROFILE.to_owned(), result, head_count: next, model_work: work, rewound_positions: rewound };
        planning::check_output(&run, self.inner.budget.max_output_bytes)?;
        if canonjson::canonical_bytes(&run).map_err(|_| ClassificationPlanningError::Serialization)?.len() as u64 > self.inner.budget.max_output_bytes {
            return Err(ClassificationPlanningError::OutputBudget.into());
        }
        planning::checkpoint(control)?; Ok(run)
    }
}
fn check_model(id: &ExecutionIdentity, source: &ArtifactIdentity) -> Result<(), Int8ClassificationError> {
    if id.numerics_profile != (NumericsProfile::StrictQuantized { version: 1 })
        || id.backend_semantic_version != STRICT_INT8_EXECUTION || id.kv_dtype != "bf16"
        || source.model_id != "Nanbeige4.2-3B" || source.revision != id.source_revision || source.recipe_id != id.quant_recipe
        || Sha256Digest::from_hex(&source.logical_model_sha256).ok() != Some(id.logical_model_digest) {
        return Err(Int8ClassificationError::Identity);
    }
    Ok(())
}
pub(crate) fn check_work_ceiling(work: Int8Work, budget: Int8RunBudget) -> Result<(), Int8ClassificationError> {
    if work.forward_positions > budget.max_forward_positions || work.attention_pairs > budget.max_attention_pairs
        || !work.projections.fits(budget.max_projection_work) { return Err(Int8ClassificationError::WorkBudget); }
    Ok(())
}

#[cfg(test)] mod tests;
