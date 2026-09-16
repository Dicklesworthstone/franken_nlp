//! Output-row-major scalar reference projection, shared by all batch stages.
//! Each row's dot expression/order and bf16 boundary are identical to
//! Bf16Matrix's reference projection. Reordering independent rows never combines
//! their reductions. Weight-row locality is structural, not measured bandwidth.
use super::*;
use super::super::weights::Bf16Matrix;

pub(super) fn project<T, C: BatchControl>(matrix: &Bf16Matrix, inputs: &[&[Bf16]],
    control: &mut C, cast: impl Fn(f32) -> T) -> Result<Vec<Vec<T>>, BatchError> {
    if inputs.is_empty() || inputs.len() > MAX_BATCH_ROWS || matrix.rows() == 0 || matrix.columns() == 0
        || inputs.iter().any(|row| row.len() != matrix.columns()) {
        return Err(BatchError::Contract("batched projection shape"));
    }
    let mut output = reserve(inputs.len())?;
    for _ in inputs { output.push(reserve(matrix.rows())?); }
    for index in 0..matrix.rows() {
        checkpoint(control)?;
        let weights = matrix.row(index).map_err(HfBf16EagerError::from)?;
        for (input, destination) in inputs.iter().zip(&mut output) {
            let sum = weights.iter().zip(*input)
                .map(|(weight, activation)| weight.to_f32() * activation.to_f32()).sum::<f32>();
            if !sum.is_finite() { return Err(BatchError::InvalidNumerics); }
            destination.push(cast(sum));
        }
    }
    Ok(output)
}
pub(super) fn activation<C: BatchControl>(matrix: &Bf16Matrix, input: &[Vec<Bf16>], control: &mut C)
    -> Result<Vec<Vec<Bf16>>, BatchError> {
    let mut views = reserve(input.len())?; views.extend(input.iter().map(Vec::as_slice));
    let output = project(matrix, &views, control, Bf16::from_f32)?;
    if output.iter().flatten().any(|x| !x.to_f32().is_finite()) { return Err(BatchError::InvalidNumerics); }
    Ok(output)
}
