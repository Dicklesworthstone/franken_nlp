//! Pair selected output columns while preserving each independent epilogue.
//! The public projection entrypoints own validation and the nonrefundable work
//! charge. This leaf is called only after that admission; it does not recharge.
use super::*;
use crate::native_engine::simd::{MAX_COHORT_ROWS, tiles::{MAX_TILE_COLUMNS, MAX_TILE_VALUES}};

impl QuantizedLinear<'_> {
    /// Inputs and selection have already been validated, and the full M*N*K
    /// work charged. Only the explicit two-column CPU body selects this route.
    /// A failure invalidates the entire caller output, as for serial projection.
    pub(super) fn project_paired<C: DecodeStepControl>(&self, inputs: &[ActivationBuffer], rows: LinearRows<'_>,
        count: usize, control: &mut C, mut store: impl FnMut(usize, f32) -> Result<(), LinearError>)
        -> Result<(), LinearError> {
        let mut since_checkpoint = ROWS_PER_CHECKPOINT;
        for column_base in (0..count).step_by(MAX_TILE_COLUMNS) {
            let columns = (count - column_base).min(MAX_TILE_COLUMNS);
            let mut weights: [&[i8]; MAX_TILE_COLUMNS] = [&[]; MAX_TILE_COLUMNS];
            let mut weight_scales = [1.0_f32; MAX_TILE_COLUMNS];
            for column in 0..columns {
                // Select by the actual row ID, never the adjacent physical row.
                let row = rows.row(column_base + column);
                weights[column] = &self.values[row * self.columns..(row + 1) * self.columns];
                weight_scales[column] = self.scales[row];
            }
            for (tile, cohort) in inputs.chunks(MAX_COHORT_ROWS).enumerate() {
                let tile_work = cohort.len() * columns;
                // Check BEFORE a tile that would exceed 32 independent dots,
                // rather than allowing two-column execution to double the gap.
                if since_checkpoint > ROWS_PER_CHECKPOINT - tile_work {
                    poll(control)?;
                    since_checkpoint = 0;
                }
                let mut activations: [&[i8]; MAX_COHORT_ROWS] = [&[]; MAX_COHORT_ROWS];
                let mut activation_scales = [1.0_f32; MAX_COHORT_ROWS];
                for (lane, input) in cohort.iter().enumerate() {
                    activations[lane] = input.values()?;
                    activation_scales[lane] = input.scale()?;
                }
                let mut sums = [0_i32; MAX_TILE_VALUES];
                self.kernel.dot_tile_into(&activations[..cohort.len()], &weights[..columns],
                    &mut sums[..tile_work]).map_err(LinearError::CpuKernel)?;
                for lane in 0..cohort.len() {
                    for column in 0..columns {
                        let value = dequantize_i32_fixed(sums[lane * columns + column], EpilogueScales {
                            activation: activation_scales[lane], row: weight_scales[column], column: 1.0, group: 1.0,
                        })?;
                        if !value.is_finite() { return Err(LinearError::NonFiniteOutput); }
                        store((tile * MAX_COHORT_ROWS + lane) * count + column_base + column, value)?;
                    }
                }
                since_checkpoint += tile_work;
            }
        }
        poll(control)
    }
}
