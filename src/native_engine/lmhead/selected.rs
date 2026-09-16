//! True row-selective eager lm_head projection. Selecting a few candidate
//! rows never materializes or scans the rest of the vocabulary. Full-vocabulary
//! probability requests still compute EVERY denominator row. Both routes use
//! the reference f32 reduction followed by bf16 narrowing and f32 export.

use std::{error::Error, fmt};
use crate::native_engine::{
    decode::{DecodeCancellationKind, DecodeStepControl},
    tensor::Bf16,
    weights::{Bf16Matrix, WeightShapeError},
};
use super::{NANBEIGE_VOCAB_SIZE, scoring::ProjectionRows};

const ROW_POLL_INTERVAL: usize = 32;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HeadProjectionError {
    Shape(WeightShapeError),
    InvalidRows,
    RowBudget,
    AllocationRefused,
    NonFinite,
    Cancelled(DecodeCancellationKind),
}
impl fmt::Display for HeadProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Shape(_) => "lm_head shape mismatch",
            Self::InvalidRows => "lm_head rows must be bounded, ordered and distinct",
            Self::RowBudget => "lm_head row budget exhausted",
            Self::AllocationRefused => "lm_head output allocation refused",
            Self::NonFinite => "lm_head projection is non-finite",
            Self::Cancelled(_) => "lm_head projection cancelled",
        })
    }
}
impl Error for HeadProjectionError {}
impl From<WeightShapeError> for HeadProjectionError {
    fn from(error: WeightShapeError) -> Self { Self::Shape(error) }
}

/// Validate the complete request before allocation or any matrix-row read.
/// Selected IDs are ascending and unique, as required by CandidateLogits.
pub fn checked_row_count(
    rows: ProjectionRows<'_>, vocabulary_size: usize, max_rows: usize,
) -> Result<usize, HeadProjectionError> {
    if vocabulary_size == 0 || vocabulary_size > NANBEIGE_VOCAB_SIZE {
        return Err(HeadProjectionError::InvalidRows);
    }
    let count = match rows {
        ProjectionRows::FullVocabulary { vocabulary_size: requested } => {
            if requested != vocabulary_size { return Err(HeadProjectionError::InvalidRows); }
            vocabulary_size
        }
        ProjectionRows::Selected(ids) => {
            if ids.is_empty() || ids.iter().any(|&id| id as usize >= vocabulary_size)
                || ids.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(HeadProjectionError::InvalidRows);
            }
            ids.len()
        }
    };
    if count > max_rows { return Err(HeadProjectionError::RowBudget); }
    Ok(count)
}

/// Project exactly the requested rows. The caller must reserve/charge `count`
/// rows BEFORE calling: cancellation or failure must not refund attempted work.
/// The callback is polled at bounded row intervals and after the last row, with
/// the caller's unchanged token-commit index (not an invented decoded-token
/// count). No partially projected vector escapes on an error.
pub fn project_rows_with_control<C: DecodeStepControl>(
    hidden: &[Bf16],
    lm_head: &Bf16Matrix,
    rows: ProjectionRows<'_>,
    max_rows: usize,
    control: &mut C,
    next_token_index: usize,
) -> Result<Vec<f32>, HeadProjectionError> {
    let count = checked_row_count(rows, lm_head.rows(), max_rows)?;
    if hidden.len() != lm_head.columns() || hidden.is_empty() {
        return Err(WeightShapeError::ProjectionInput {
            expected: lm_head.columns(), actual: hidden.len(),
        }.into());
    }
    if hidden.iter().any(|value| !value.to_f32().is_finite()) {
        return Err(HeadProjectionError::NonFinite);
    }
    if let Some(cause) = control.checkpoint(next_token_index) {
        return Err(HeadProjectionError::Cancelled(cause));
    }
    let mut logits = Vec::new();
    logits.try_reserve_exact(count).map_err(|_| HeadProjectionError::AllocationRefused)?;
    for output_row in 0..count {
        if output_row != 0 && output_row % ROW_POLL_INTERVAL == 0 {
            if let Some(cause) = control.checkpoint(next_token_index) {
                return Err(HeadProjectionError::Cancelled(cause));
            }
        }
        let token = match rows {
            ProjectionRows::FullVocabulary { .. } => output_row,
            ProjectionRows::Selected(ids) => ids[output_row] as usize,
        };
        // Identical iterator/reduction and named casts to the reference
        // Bf16Matrix::project_f32_accumulate_bf16_then_export implementation.
        // Never project an entire matrix and slice its result afterwards.
        let accumulator = lm_head.row(token)?.iter().zip(hidden)
            .map(|(weight, activation)| weight.to_f32() * activation.to_f32())
            .sum::<f32>();
        let logit = Bf16::from_f32(accumulator).to_f32();
        if !logit.is_finite() { return Err(HeadProjectionError::NonFinite); }
        logits.push(logit);
    }
    if let Some(cause) = control.checkpoint(next_token_index) {
        return Err(HeadProjectionError::Cancelled(cause));
    }
    Ok(logits)
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    }
    #[test]
    fn selected_and_full_rows_have_the_exact_reference_bits() {
        let values = [0.1, 0.3, -2.0, 0.5, -0.25, 1.125, 9.5, -0.75, 0.03125, -0.0, 0.0, 8.5];
        let matrix = Bf16Matrix::new(4, 3, values.into_iter().map(Bf16::from_f32).collect()).unwrap();
        let hidden: Vec<_> = [0.25, -1.125, 0.3].into_iter().map(Bf16::from_f32).collect();
        let reference = super::super::export_logits_f32(&hidden, &matrix).unwrap();
        let full = project_rows_with_control(&hidden, &matrix, ProjectionRows::FullVocabulary { vocabulary_size: 4 }, 4, &mut Continue, 0).unwrap();
        assert_eq!(full.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        let selected = project_rows_with_control(&hidden, &matrix, ProjectionRows::Selected(&[1, 3]), 2, &mut Continue, 0).unwrap();
        assert_eq!(selected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), [reference[1].to_bits(), reference[3].to_bits()]);
    }
    #[test]
    fn selected_projection_does_not_evaluate_unrequested_rows() {
        let matrix = Bf16Matrix::new(3, 1, [1.0, f32::NAN, 3.0].into_iter().map(Bf16::from_f32).collect()).unwrap();
        let hidden = [Bf16::from_f32(2.0)];
        assert_eq!(project_rows_with_control(&hidden, &matrix, ProjectionRows::Selected(&[0, 2]), 2, &mut Continue, 0).unwrap(), [2.0, 6.0]);
        assert_eq!(project_rows_with_control(&hidden, &matrix, ProjectionRows::FullVocabulary { vocabulary_size: 3 }, 3, &mut Continue, 0), Err(HeadProjectionError::NonFinite));
    }
    #[test]
    fn invalid_rows_and_full_denominator_budget_refuse_before_polling() {
        struct NoPoll;
        impl DecodeStepControl for NoPoll {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { panic!("invalid request must refuse first"); }
        }
        let matrix = Bf16Matrix::new(3, 1, vec![Bf16::from_f32(1.0); 3]).unwrap();
        for ids in [&[][..], &[2, 1][..], &[1, 1][..], &[3][..]] {
            assert!(project_rows_with_control(&[Bf16::from_f32(1.0)], &matrix, ProjectionRows::Selected(ids), 3, &mut NoPoll, 0).is_err());
        }
        assert_eq!(checked_row_count(ProjectionRows::FullVocabulary { vocabulary_size: 3 }, 3, 2), Err(HeadProjectionError::RowBudget));
        assert_eq!(checked_row_count(ProjectionRows::FullVocabulary { vocabulary_size: 2 }, 3, 3), Err(HeadProjectionError::InvalidRows));
    }
}
