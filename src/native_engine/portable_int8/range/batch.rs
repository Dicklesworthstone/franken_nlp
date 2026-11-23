//! Shared-weight batch range leaves for the admitted persistent INT8 team.
//! Every independent dot keeps the serial batch kernel and fixed BF16 cast.
//! Inputs are borrowed quantized rows, and output holes belong to unpublished
//! coordinator staging. No leaf allocates, spawns, requantizes or debits work.
use super::super::*;
use crate::native_engine::portable_int8::batch::MAX_BATCH_ROWS;
#[cfg(any(feature = "asupersync-runtime", test))]
use std::ops::Range;
#[cfg(any(feature = "asupersync-runtime", test))]
use crate::native_engine::simd::{MAX_COHORT_ROWS, tiles::{MAX_TILE_COLUMNS, MAX_TILE_VALUES}};

impl QuantizedLinear<'_> {
    /// Validate every activation and the complete output before the sole
    /// M*N*K debit. A later failed shard never renews or refunds this allowance.
    pub(crate) fn reserve_batch_projection(&self, inputs: &[ActivationBuffer],
        rows: LinearRows<'_>, length: usize, ledger: &mut ProjectionLedger) -> Result<(), LinearError> {
        if inputs.is_empty() || inputs.len() > MAX_BATCH_ROWS { return Err(LinearError::Workspace); }
        let count = rows.checked_count(self.rows)?;
        let total = inputs.len().checked_mul(count).ok_or(LinearError::OutputShape)?;
        if length != total { return Err(LinearError::OutputShape); }
        for input in inputs {
            if input.values()?.len() != self.columns { return Err(LinearError::Shape); }
            input.scale()?;
        }
        ledger.charge(ProjectionWork::for_shape(total, self.columns)?)
    }

    /// `range` addresses ordered output positions in `rows`. Each activation
    /// writes range.len() local columns at activation * output_stride. Workers
    /// use compact stride; the coordinator uses the full projection width.
    /// The slice has exactly (M-1)*stride + range.len() cells, or zero cells for
    /// an empty range. Cells between activation rows are never touched.
    #[cfg(any(feature = "asupersync-runtime", test))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn project_batch_bf16_range_into(&self, inputs: &[&[i8]], activation_scales: &[f32],
        rows: LinearRows<'_>, range: Range<usize>, output: &mut [Bf16], output_stride: usize,
        control: &mut dyn DecodeStepControl) -> Result<(), LinearError> {
        self.check_batch_range(inputs, activation_scales, rows, &range, output.len(), output_stride)?;
        if range.is_empty() { return poll(control); }
        if self.kernel.tile_columns() > 1 {
            return self.project_batch_paired_range(inputs, activation_scales, rows, range,
                output, output_stride, control);
        }
        let mut since_checkpoint = ROWS_PER_CHECKPOINT;
        for (column, index) in range.enumerate() {
            let row = rows.row(index);
            let weights = &self.values[row * self.columns..(row + 1) * self.columns];
            for (tile, cohort) in inputs.chunks(MAX_COHORT_ROWS).enumerate() {
                if since_checkpoint > ROWS_PER_CHECKPOINT - cohort.len() { poll(control)?; since_checkpoint = 0; }
                let mut sums = [0_i32; MAX_COHORT_ROWS];
                if self.kernel.kind() == Int8KernelKind::Scalar {
                    for (k, &weight) in weights.iter().enumerate() {
                        let weight = i32::from(weight);
                        for lane in 0..cohort.len() { sums[lane] += i32::from(cohort[lane][k]) * weight; }
                    }
                } else {
                    self.kernel.dot_rows_into(cohort, weights, &mut sums[..cohort.len()])
                        .map_err(LinearError::CpuKernel)?;
                }
                for (lane, &sum) in sums[..cohort.len()].iter().enumerate() {
                    let activation = tile * MAX_COHORT_ROWS + lane;
                    let value = dequantize_i32_fixed(sum, EpilogueScales {
                        activation: activation_scales[activation], row: self.scales[row], column: 1.0, group: 1.0,
                    })?;
                    output[activation * output_stride + column] = cast(value)?;
                }
                since_checkpoint += cohort.len();
            }
        }
        poll(control)
    }

    #[cfg(any(feature = "asupersync-runtime", test))]
    #[allow(clippy::too_many_arguments)]
    fn check_batch_range(&self, inputs: &[&[i8]], activation_scales: &[f32], rows: LinearRows<'_>,
        range: &Range<usize>, length: usize, output_stride: usize) -> Result<(), LinearError> {
        if inputs.is_empty() || inputs.len() > MAX_BATCH_ROWS { return Err(LinearError::Workspace); }
        if inputs.iter().any(|input| input.len() != self.columns) { return Err(LinearError::Shape); }
        if activation_scales.len() != inputs.len()
            || activation_scales.iter().any(|scale| !scale.is_finite() || *scale <= 0.0) {
            return Err(LinearError::Activation);
        }
        let count = rows.checked_count(self.rows)?;
        let width = range.end.checked_sub(range.start).ok_or(LinearError::OutputShape)?;
        if range.end > count || output_stride < width { return Err(LinearError::OutputShape); }
        let span = if width == 0 { 0 } else {
            (inputs.len() - 1).checked_mul(output_stride)
                .and_then(|offset| offset.checked_add(width)).ok_or(LinearError::OutputShape)?
        };
        if length != span { return Err(LinearError::OutputShape); }
        Ok(())
    }

    #[cfg(any(feature = "asupersync-runtime", test))]
    #[allow(clippy::too_many_arguments)]
    fn project_batch_paired_range(&self, inputs: &[&[i8]], activation_scales: &[f32],
        rows: LinearRows<'_>, range: Range<usize>, output: &mut [Bf16], output_stride: usize,
        control: &mut dyn DecodeStepControl) -> Result<(), LinearError> {
        let mut since_checkpoint = ROWS_PER_CHECKPOINT;
        for base in (0..range.len()).step_by(MAX_TILE_COLUMNS) {
            let columns = (range.len() - base).min(MAX_TILE_COLUMNS);
            let mut weights: [&[i8]; MAX_TILE_COLUMNS] = [&[]; MAX_TILE_COLUMNS];
            let mut scales = [1.0_f32; MAX_TILE_COLUMNS];
            for column in 0..columns {
                let row = rows.row(range.start + base + column);
                weights[column] = &self.values[row * self.columns..(row + 1) * self.columns];
                scales[column] = self.scales[row];
            }
            for (tile, cohort) in inputs.chunks(MAX_COHORT_ROWS).enumerate() {
                let work = cohort.len() * columns;
                if since_checkpoint > ROWS_PER_CHECKPOINT - work { poll(control)?; since_checkpoint = 0; }
                let mut sums = [0_i32; MAX_TILE_VALUES];
                self.kernel.dot_tile_into(cohort, &weights[..columns], &mut sums[..work])
                    .map_err(LinearError::CpuKernel)?;
                for lane in 0..cohort.len() {
                    let activation = tile * MAX_COHORT_ROWS + lane;
                    for column in 0..columns {
                        let value = dequantize_i32_fixed(sums[lane * columns + column], EpilogueScales {
                            activation: activation_scales[activation], row: scales[column], column: 1.0, group: 1.0,
                        })?;
                        output[activation * output_stride + base + column] = cast(value)?;
                    }
                }
                since_checkpoint += work;
            }
        }
        poll(control)
    }
}

#[cfg(any(feature = "asupersync-runtime", test))]
fn cast(value: f32) -> Result<Bf16, LinearError> {
    if !value.is_finite() { return Err(LinearError::NonFiniteOutput); }
    let value = Bf16::from_f32(value);
    if !value.to_f32().is_finite() { return Err(LinearError::NonFiniteOutput); }
    Ok(value)
}

#[cfg(test)]
mod tests;
