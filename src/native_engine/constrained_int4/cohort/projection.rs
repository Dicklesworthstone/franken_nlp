//! One declared head policy per cohort; individual selected-row caps remain
//! attached to their original rows. Scheduling never converts either policy.
use super::*;
use super::super::sparse::{self, Int4JsonSparseLimits};

#[derive(Clone, Copy)]
pub(super) enum Projection<'a> {
    Full,
    Selected(&'a [Int4JsonSparseLimits]),
}
impl Projection<'_> {
    pub(super) fn validate(self, count: usize) -> Result<(), Int4JsonError> {
        if let Self::Selected(limits) = self {
            if limits.len() != count {
                return Err(JsonDecodeError::InvalidRequest("cohort selected-row limit count").into());
            }
            for limit in limits { limit.validate()?; }
        }
        Ok(())
    }
    pub(super) fn limits(self, slot: usize) -> Result<Option<Int4JsonSparseLimits>, Int4JsonError> {
        match self {
            Self::Full => Ok(None),
            Self::Selected(limits) => limits.get(slot).copied().map(Some)
                .ok_or_else(|| JsonDecodeError::InvalidRequest("cohort selected-row limit count").into()),
        }
    }
    pub(super) fn preflight<V: Vocabulary>(self, group_width: usize, slot: usize,
        request: &Int4JsonCohortRequest<'_>, vocabulary: &V) -> Result<Int4Work, Int4JsonError> {
        match self.limits(slot)? {
            None => super::super::preflight(group_width, request.prompt, vocabulary,
                request.options, request.budget),
            Some(limits) => sparse::preflight(group_width, request.prompt, vocabulary,
                request.options, request.budget, limits),
        }
    }
    pub(super) fn row_execution(self) -> &'static str {
        match self { Self::Full => INT4_JSON_EXECUTION, Self::Selected(_) => sparse::INT4_SPARSE_JSON_EXECUTION }
    }
    pub(super) fn execution(self, packed: bool) -> &'static str {
        match (self, packed) {
            (Self::Full, false) => INT4_JSON_COHORT_EXECUTION,
            (Self::Full, true) => INT4_JSON_PACKED_COHORT_EXECUTION,
            (Self::Selected(_), false) => INT4_JSON_SELECTED_COHORT_EXECUTION,
            (Self::Selected(_), true) => INT4_JSON_SELECTED_PACKED_COHORT_EXECUTION,
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn check_counts(self, prefill: Option<Int4PrefillLimits>, total_positions: u64,
        longest_positions: u64, total_selections: u64, longest_selection_sequence: u64,
        projected_logits: u64, group_steps: u64, projection_groups: u64) -> Result<(), Int4JsonError> {
        match self {
            Self::Full => check_group_counts(prefill, total_positions, longest_positions,
                total_selections, longest_selection_sequence, group_steps, projection_groups),
            Self::Selected(_) => check_selected_group_counts(prefill, total_positions,
                longest_positions, total_selections, longest_selection_sequence,
                projected_logits, group_steps, projection_groups),
        }
    }
}

/// Selected heads project one sequence at a time in complete, ascending chunks
/// of at most 32 legal IDs. Count actual calls, including terminal EOS; a head
/// invocation is not a sampled-token count or a shared decoder invocation.
/// Exact per-row projected work remains independently checked by each task.
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_selected_group_counts(prefill: Option<Int4PrefillLimits>, total_positions: u64,
    longest_positions: u64, total_selections: u64, longest_selection_sequence: u64, projected_logits: u64,
    group_steps: u64, projection_groups: u64) -> Result<(), Int4JsonError> {
    check_forward_groups(prefill, total_positions, longest_positions, group_steps)?;
    let rows = sparse::selection::ROWS as u64;
    let remainder = projected_logits.checked_sub(total_selections).ok_or(Int4JsonError::WorkMismatch)?;
    let maximum = total_selections.checked_add(remainder / rows).ok_or(Int4JsonError::WorkMismatch)?;
    if total_selections == 0 || total_selections > total_positions
        || longest_selection_sequence == 0 || longest_selection_sequence > total_selections
        || group_steps < longest_selection_sequence
        || projection_groups < total_selections.max(projected_logits.div_ceil(rows))
        || projection_groups > maximum {
        return Err(Int4JsonError::WorkMismatch);
    }
    Ok(())
}
