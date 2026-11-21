//! Bounded prompt-row projections sharing each unpacked group across the batch.
//!
//! One aggregate ledger prices every row before any output write. A packed
//! group is prepared ONCE, then borrowed by every four-row integer tile, with
//! checkpoints between tiles. Each row retains increasing-group f32 summation
//! and its own activation scale; BF16 is cast once after the complete row.
//! No allocation, padding, dense weight expansion, dispatch or thread spawn.
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
        store: impl FnMut(usize, f32) -> Result<(), LinearError>) -> Result<(), GroupedInt4Error> {
        if scratch.len() < self.scratch_len() { return Err(GroupedInt4Error::Scratch); }
        self.reserve_batch_projection(inputs, rows, length, ledger)?;
        let count = rows.checked_count(self.rows)?;
        let mut values: [&[i8]; MAX_INT4_BATCH_ROWS] = [&[]; MAX_INT4_BATCH_ROWS];
        let mut scales = [1.0_f32; MAX_INT4_BATCH_ROWS];
        for (index, input) in inputs.iter().enumerate() {
            values[index] = input.values()?;
            scales[index] = input.scale()?;
        }
        self.project_batch_range(&values[..inputs.len()], &scales[..inputs.len()], rows,
            0..count, count, scratch, control, store)
    }

    /// One complete debit before externally scheduled output-row ranges.
    /// Every activation, including the last one, is checked before spending
    /// work. Worker leaves never receive a fresh or independently reset ledger.
    pub(crate) fn reserve_batch_projection(&self, inputs: &[ActivationBuffer], rows: LinearRows<'_>,
        length: usize, ledger: &mut ProjectionLedger) -> Result<(), GroupedInt4Error> {
        if inputs.is_empty() || inputs.len() > MAX_INT4_BATCH_ROWS { return Err(LinearError::Workspace.into()); }
        let count = rows.checked_count(self.rows)?;
        if count.checked_mul(inputs.len()) != Some(length) { return Err(LinearError::OutputShape.into()); }
        for input in inputs {
            if input.values()?.len() != self.columns { return Err(LinearError::Shape.into()); }
            input.scale()?;
        }
        let one = self.planned_work(rows)?;
        let copies = inputs.len() as u64;
        ledger.charge(ProjectionWork {
            dot_products: one.dot_products.checked_mul(copies).ok_or(LinearError::WorkBudget)?,
            multiply_accumulates: one.multiply_accumulates.checked_mul(copies).ok_or(LinearError::WorkBudget)?,
        })?;
        Ok(())
    }

    /// Exact batch range over already quantized inputs. `range` addresses the
    /// selected output positions; each input row writes its local shard at
    /// `lane * output_stride`. A larger stride lets the coordinator write its
    /// disjoint first shard directly into the final private staging buffer.
    #[cfg(any(feature = "asupersync-runtime", test))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn project_batch_bf16_range_into(&self, inputs: &[&[i8]], scales: &[f32],
        rows: LinearRows<'_>, range: Range<usize>, output: &mut [Bf16], output_stride: usize,
        scratch: &mut [i8], control: &mut dyn DecodeStepControl) -> Result<(), GroupedInt4Error> {
        self.check_batch_range(inputs, scales, rows, &range, output.len(), output_stride, scratch.len())?;
        self.project_batch_range(inputs, scales, rows, range, output_stride, scratch, control,
            |index, value| { output[index] = checked_bf16(value)?; Ok(()) })
    }

    #[cfg(any(feature = "asupersync-runtime", test))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn project_batch_f32_range_into(&self, inputs: &[&[i8]], scales: &[f32],
        rows: LinearRows<'_>, range: Range<usize>, output: &mut [f32], output_stride: usize,
        scratch: &mut [i8], control: &mut dyn DecodeStepControl) -> Result<(), GroupedInt4Error> {
        self.check_batch_range(inputs, scales, rows, &range, output.len(), output_stride, scratch.len())?;
        self.project_batch_range(inputs, scales, rows, range, output_stride, scratch, control,
            |index, value| { output[index] = value; Ok(()) })
    }

    #[cfg(any(feature = "asupersync-runtime", test))]
    #[allow(clippy::too_many_arguments)]
    fn check_batch_range(&self, inputs: &[&[i8]], scales: &[f32], rows: LinearRows<'_>,
        range: &Range<usize>, length: usize, output_stride: usize, scratch_len: usize)
        -> Result<(), GroupedInt4Error> {
        if inputs.is_empty() || inputs.len() > MAX_INT4_BATCH_ROWS || inputs.len() != scales.len() {
            return Err(LinearError::Workspace.into());
        }
        if inputs.iter().any(|input| input.len() != self.columns) { return Err(LinearError::Shape.into()); }
        if scales.iter().any(|scale| !scale.is_finite() || *scale <= 0.0) { return Err(LinearError::Activation.into()); }
        let count = rows.checked_count(self.rows)?;
        let width = range.end.checked_sub(range.start).ok_or(LinearError::OutputShape)?;
        if range.end > count || output_stride < width || output_stride.checked_mul(inputs.len()) != Some(length) {
            return Err(LinearError::OutputShape.into());
        }
        if scratch_len < self.scratch_len() { return Err(GroupedInt4Error::Scratch); }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn project_batch_range<C: DecodeStepControl + ?Sized>(&self, values: &[&[i8]], scales: &[f32],
        rows: LinearRows<'_>, range: Range<usize>, output_stride: usize, scratch: &mut [i8], control: &mut C,
        mut store: impl FnMut(usize, f32) -> Result<(), LinearError>) -> Result<(), GroupedInt4Error> {
        let mut since_poll = ROWS_PER_CHECKPOINT;
        for (local, index) in range.enumerate() {
            let row = rows.row(index);
            // Only one output column's totals are live, not an M*N matrix.
            // Keep each row's f32 accumulator across its increasing groups.
            let mut totals = [0.0_f32; MAX_INT4_BATCH_ROWS];
            for group in 0..self.groups {
                let first_lanes = values.len().min(MAX_COHORT_ROWS);
                if since_poll + first_lanes > ROWS_PER_CHECKPOINT { poll(control)?; since_poll = 0; }
                let start = group * self.group_width;
                let width = self.group_width.min(self.columns - start);
                let prepared = self.kernel.prepare_range(self.packed, row * self.columns + start,
                    width, scratch).map_err(LinearError::CpuKernel)?;
                let weight_scale = self.scales[row * self.groups + group];
                for base in (0..values.len()).step_by(MAX_COHORT_ROWS) {
                    let lanes = (values.len() - base).min(MAX_COHORT_ROWS);
                    // A large batch never creates a 64-dot cancellation gap.
                    if since_poll + lanes > ROWS_PER_CHECKPOINT { poll(control)?; since_poll = 0; }
                    let mut tile: [&[i8]; MAX_COHORT_ROWS] = [&[]; MAX_COHORT_ROWS];
                    let mut dots = [0_i32; MAX_COHORT_ROWS];
                    for lane in 0..lanes { tile[lane] = &values[base + lane][start..start + width]; }
                    prepared.dot_rows_into(&tile[..lanes], &mut dots[..lanes]).map_err(LinearError::CpuKernel)?;
                    for lane in 0..lanes {
                        accumulate_group(&mut totals[base + lane], dots[lane], scales[base + lane], weight_scale)?;
                    }
                    since_poll += lanes;
                }
            }
            // No partially accumulated group sum is cast or written out.
            for lane in 0..values.len() { store(lane * output_stride + local, totals[lane])?; }
        }
        poll(control)?;
        Ok(())
    }
}

#[cfg(test)] mod tests;
