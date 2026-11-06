//! Shared all-window admission before execution or external persistence.
use super::*;
use super::cohort::Window;

impl PreparedInt4ClassificationMap<'_> {
    /// Validate every admitted identity and every future native window without
    /// allocating an engine. Callers may use this before publishing a snapshot
    /// or another external side effect; execution repeats the same checks.
    pub fn preflight_cohorts_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        model: &Int4ModelIdentity, cohort_chunks: usize, budget: Int4ClassificationMapCohortBudget,
        control: &mut C) -> Result<(), Int4ClassificationMapError> {
        self.admitted_cohort_windows(admitted, model, cohort_chunks, budget, control).map(|_| ())
    }

    pub(super) fn admitted_cohort_windows<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        model: &Int4ModelIdentity, cohort_chunks: usize, budget: Int4ClassificationMapCohortBudget,
        control: &mut C) -> Result<Vec<Window>, Int4ClassificationMapError> {
        planning::checkpoint(control)?;
        if model.group_width() != self.group_width() || admitted.len() != self.plans.len() {
            return Err(Int4ClassificationMapError::Admission);
        }
        for identity in admitted {
            planning::checkpoint(control)?;
            check_model(identity, model)?;
        }
        self.cohort_windows(admitted, cohort_chunks, budget, control)
    }
}
