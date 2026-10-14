//! Bounded prompt-row projections sharing packed group reductions.
//!
//! One aggregate ledger prices every row before any output write. Integer
//! leaves share weights across up to four lanes; each lane keeps the scalar
//! activation scale, increasing-group sum and single final BF16 cast. No
//! allocation, padding, dense dequantization, dispatch policy or thread spawn.
use super::*;
use crate::native_engine::simd::MAX_COHORT_ROWS;

pub const MAX_INT4_BATCH_ROWS: usize = 64;

impl GroupedInt4Linear<'_> {
    /// Row-major [prompt rows, selected output rows]. All input rows use the
    /// same checked output selection. Partial staging is invalid on any error.
    pub fn project_batch_f32_into<C: DecodeStepControl>(&self, inputs: &[ActivationBuffer],
        rows: LinearRows<'_>, output: &mut [f32], scratch: &mut [i8],
        ledger: &mut ProjectionLedger, control: &mut C) -> Result<(), GroupedInt4Error> {
        self.project_batch(inputs, rows, output.len(), scratch, ledger, control,
            |index, value| { output[index] = value; Ok(()) })
    }
    pub fn project_batch_bf16_into<C: DecodeStepControl>(&self, inputs: &[ActivationBuffer],
        rows: LinearRows<'_>, output: &mut [Bf16], scratch: &mut [i8],
        ledger: &mut ProjectionLedger, control: &mut C) -> Result<(), GroupedInt4Error> {
        self.project_batch(inputs, rows, output.len(), scratch, ledger, control,
            |index, value| { output[index] = checked_bf16(value)?; Ok(()) })
    }
    #[allow(clippy::too_many_arguments)]
    fn project_batch<C: DecodeStepControl>(&self, inputs: &[ActivationBuffer], rows: LinearRows<'_>,
        length: usize, scratch: &mut [i8], ledger: &mut ProjectionLedger, control: &mut C,
        mut store: impl FnMut(usize, f32) -> Result<(), LinearError>) -> Result<(), GroupedInt4Error> {
        if inputs.is_empty() || inputs.len() > MAX_INT4_BATCH_ROWS { return Err(LinearError::Workspace.into()); }
        let count = rows.checked_count(self.rows)?;
        if count.checked_mul(inputs.len()) != Some(length) { return Err(LinearError::OutputShape.into()); }
        if scratch.len() < self.scratch_len() { return Err(GroupedInt4Error::Scratch); }
        let mut values: [&[i8]; MAX_INT4_BATCH_ROWS] = [&[]; MAX_INT4_BATCH_ROWS];
        let mut scales = [1.0_f32; MAX_INT4_BATCH_ROWS];
        // Validate even the last activation BEFORE charging or polling.
        for (index, input) in inputs.iter().enumerate() {
            values[index] = input.values()?;
            scales[index] = input.scale()?;
            if values[index].len() != self.columns { return Err(LinearError::Shape.into()); }
        }
        let one = self.planned_work(rows)?;
        let copies = inputs.len() as u64;
        ledger.charge(ProjectionWork {
            dot_products: one.dot_products.checked_mul(copies).ok_or(LinearError::WorkBudget)?,
            multiply_accumulates: one.multiply_accumulates.checked_mul(copies).ok_or(LinearError::WorkBudget)?,
        })?;
        let mut since_poll = ROWS_PER_CHECKPOINT;
        for index in 0..count {
            let row = rows.row(index);
            for base in (0..inputs.len()).step_by(MAX_COHORT_ROWS) {
                let lanes = (inputs.len() - base).min(MAX_COHORT_ROWS);
                let mut totals = [0.0_f32; MAX_COHORT_ROWS];
                for group in 0..self.groups {
                    if since_poll + lanes > ROWS_PER_CHECKPOINT { poll(control)?; since_poll = 0; }
                    let start = group * self.group_width;
                    let width = self.group_width.min(self.columns - start);
                    let mut tile: [&[i8]; MAX_COHORT_ROWS] = [&[]; MAX_COHORT_ROWS];
                    let mut dots = [0_i32; MAX_COHORT_ROWS];
                    for lane in 0..lanes { tile[lane] = &values[base + lane][start..start + width]; }
                    self.kernel.dot_rows_range_into(&tile[..lanes], self.packed,
                        row * self.columns + start, width, &mut dots[..lanes], scratch).map_err(LinearError::CpuKernel)?;
                    let weight_scale = self.scales[row * self.groups + group];
                    for lane in 0..lanes {
                        accumulate_group(&mut totals[lane], dots[lane], scales[base + lane], weight_scale)?;
                    }
                    since_poll += lanes;
                }
                for lane in 0..lanes { store((base + lane) * count + index, totals[lane])?; }
            }
        }
        poll(control)?;
        Ok(())
    }
}

#[cfg(test)] mod tests;
