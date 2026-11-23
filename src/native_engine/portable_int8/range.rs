//! Exact output-row leaves for one externally admitted INT8 projection.
//! The coordinator charges the whole operation once. Workers receive unchanged
//! quantized activations, immutable weights and private output; no leaf owns a
//! renewable budget, requantizes inputs, allocates, or spawns work.
use super::*;
#[cfg(any(feature = "asupersync-runtime", test))]
use std::ops::Range;

impl QuantizedLinear<'_> {
    /// Validate and charge every requested row before any range is dispatched.
    /// Failure or cancellation after this point never refunds the full debit.
    pub(crate) fn reserve_projection(&self, input: &ActivationBuffer, rows: LinearRows<'_>,
        length: usize, ledger: &mut ProjectionLedger) -> Result<(), LinearError> {
        let values = input.values()?;
        input.scale()?;
        if values.len() != self.columns { return Err(LinearError::Shape); }
        let count = rows.checked_count(self.rows)?;
        if length != count { return Err(LinearError::OutputShape); }
        ledger.charge(ProjectionWork::for_shape(count, self.columns)?)
    }

    /// `range` indexes the ordered output positions in `rows`, not a new
    /// matrix. Any partial output after an error is unpublished staging.
    #[cfg(any(feature = "asupersync-runtime", test))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn project_f32_range_into(&self, input: &[i8], activation_scale: f32,
        rows: LinearRows<'_>, range: Range<usize>, output: &mut [f32],
        control: &mut dyn DecodeStepControl) -> Result<(), LinearError> {
        self.check_range(input, activation_scale, rows, &range, output.len())?;
        self.project_range(input, activation_scale, rows, range, control,
            |index, value| { output[index] = value; Ok(()) })
    }

    /// Preserve the serial BF16 cast after the same fixed f32 epilogue.
    #[cfg(any(feature = "asupersync-runtime", test))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn project_bf16_range_into(&self, input: &[i8], activation_scale: f32,
        rows: LinearRows<'_>, range: Range<usize>, output: &mut [Bf16],
        control: &mut dyn DecodeStepControl) -> Result<(), LinearError> {
        self.check_range(input, activation_scale, rows, &range, output.len())?;
        self.project_range(input, activation_scale, rows, range, control, |index, value| {
            let cast = Bf16::from_f32(value);
            if !cast.to_f32().is_finite() { return Err(LinearError::NonFiniteOutput); }
            output[index] = cast; Ok(())
        })
    }

    #[cfg(any(feature = "asupersync-runtime", test))]
    fn check_range(&self, input: &[i8], activation_scale: f32, rows: LinearRows<'_>,
        range: &Range<usize>, length: usize) -> Result<(), LinearError> {
        if input.len() != self.columns { return Err(LinearError::Shape); }
        if !activation_scale.is_finite() || activation_scale <= 0.0 { return Err(LinearError::Activation); }
        let count = rows.checked_count(self.rows)?;
        if range.end > count || range.end.checked_sub(range.start) != Some(length) {
            return Err(LinearError::OutputShape);
        }
        Ok(())
    }

    #[cfg(any(feature = "asupersync-runtime", test))]
    #[allow(clippy::too_many_arguments)]
    fn project_range<C: DecodeStepControl + ?Sized>(&self, input: &[i8], activation_scale: f32,
        rows: LinearRows<'_>, range: Range<usize>, control: &mut C,
        mut store: impl FnMut(usize, f32) -> Result<(), LinearError>) -> Result<(), LinearError> {
        if self.kernel.tile_columns() > 1 {
            use crate::native_engine::simd::tiles::MAX_TILE_COLUMNS;
            for local in (0..range.len()).step_by(MAX_TILE_COLUMNS) {
                if local % ROWS_PER_CHECKPOINT == 0 { poll(control)?; }
                let columns = (range.len() - local).min(MAX_TILE_COLUMNS);
                let mut weights: [&[i8]; MAX_TILE_COLUMNS] = [&[]; MAX_TILE_COLUMNS];
                let mut scales = [1.0_f32; MAX_TILE_COLUMNS];
                for column in 0..columns {
                    let row = rows.row(range.start + local + column);
                    weights[column] = &self.values[row * self.columns..(row + 1) * self.columns];
                    scales[column] = self.scales[row];
                }
                let mut sums = [0_i32; MAX_TILE_COLUMNS];
                self.kernel.dot_tile_into(&[input], &weights[..columns], &mut sums[..columns])
                    .map_err(LinearError::CpuKernel)?;
                for column in 0..columns {
                    let value = dequantize_i32_fixed(sums[column], EpilogueScales {
                        activation: activation_scale, row: scales[column], column: 1.0, group: 1.0,
                    })?;
                    if !value.is_finite() { return Err(LinearError::NonFiniteOutput); }
                    store(local + column, value)?;
                }
            }
        } else {
            for (local, index) in range.enumerate() {
                if local % ROWS_PER_CHECKPOINT == 0 { poll(control)?; }
                let row = rows.row(index);
                let weights = &self.values[row * self.columns..(row + 1) * self.columns];
                let integer = if self.kernel.kind() == Int8KernelKind::Scalar {
                    dot_s8s8(input, weights)?
                } else { self.kernel.dot(input, weights).map_err(LinearError::CpuKernel)? };
                let value = dequantize_i32_fixed(integer, EpilogueScales {
                    activation: activation_scale, row: self.scales[row], column: 1.0, group: 1.0,
                })?;
                if !value.is_finite() { return Err(LinearError::NonFiniteOutput); }
                store(local, value)?;
            }
        }
        poll(control)
    }
}

mod batch;

#[cfg(test)]
mod tests;
