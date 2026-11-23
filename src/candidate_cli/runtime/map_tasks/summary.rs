//! Complete-summary delivery checks against the exact pre-load request.
use super::*;
use crate::tasks::source_planning::quantized::long::summary::{Int8CorpusSummaryRun, Int8SummaryPreflight};

pub(super) fn check_completed(expected: &Int8SummaryPreflight, result: &Int8CorpusSummaryRun,
    control: &mut impl DecodeStepControl)
    -> Result<(), CandidateError> {
    expected.verify_completed_with_control(result, control)
        .map_err(|error| error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution))
}

#[cfg(test)] mod tests;
