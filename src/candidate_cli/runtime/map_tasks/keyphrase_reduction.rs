//! Delivery checks against the exact pre-load document and ranking request.
use super::*;
use crate::tasks::source_planning::quantized::long::keyphrase_reduction::{Int8CorpusKeyphraseRun, Int8KeyphrasePreflight};

pub(super) fn check_completed(expected: &Int8KeyphrasePreflight, result: &Int8CorpusKeyphraseRun,
    control: &mut impl DecodeStepControl)
    -> Result<(), CandidateError> {
    expected.verify_completed_with_control(result, control)
        .map_err(|error| error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution))
}

#[cfg(test)] mod tests;
