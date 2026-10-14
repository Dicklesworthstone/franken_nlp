//! Allocation-free shared-group projection for a finite independent cohort.
//! Requests may select different sorted vocabulary rows. A bounded merge visits
//! only their union, unpacks each group once for its active lanes, and retains
//! each lane's original group order, activation scale, output index and budget.

use super::*;

pub const MAX_INT4_COHORT_ROWS: usize = crate::native_engine::simd::MAX_COHORT_ROWS;

/// Output remains private staging until the entire epoch succeeds. A failed
/// epoch invalidates ALL outputs, including earlier completed cells. The host
/// supplies one meet-composed epoch control and retains request/engine guards;
/// these finite leaves do not create tasks, acquire permits or retry failures.
pub struct Int4ProjectionRow<'a, T> {
    pub input: &'a ActivationBuffer,
    pub selection: LinearRows<'a>,
    pub output: &'a mut [T],
    pub ledger: &'a mut ProjectionLedger,
}

impl GroupedInt4Linear<'_> {
    pub fn project_cohort_f32_into<C: DecodeStepControl>(&self,
        members: &mut [Int4ProjectionRow<'_, f32>], scratch: &mut [i8], control: &mut C)
        -> Result<(), GroupedInt4Error> {
        self.project_cohort(members, scratch, control, Ok)
    }
    pub fn project_cohort_bf16_into<C: DecodeStepControl>(&self,
        members: &mut [Int4ProjectionRow<'_, Bf16>], scratch: &mut [i8], control: &mut C)
        -> Result<(), GroupedInt4Error> {
        self.project_cohort(members, scratch, control, checked_bf16)
    }
    fn project_cohort<'a, T, C: DecodeStepControl>(&self,
        members: &mut [Int4ProjectionRow<'a, T>], scratch: &mut [i8], control: &mut C,
        mut cast: impl FnMut(f32) -> Result<T, LinearError>) -> Result<(), GroupedInt4Error> {
        if members.is_empty() || members.len() > MAX_INT4_COHORT_ROWS { return Err(LinearError::Workspace.into()); }
        if scratch.len() < self.scratch_len() { return Err(GroupedInt4Error::Scratch); }
        let mut inputs: [&'a [i8]; MAX_INT4_COHORT_ROWS] = [&[]; MAX_INT4_COHORT_ROWS];
        let mut activation_scales = [1.0_f32; MAX_INT4_COHORT_ROWS];
        let mut counts = [0_usize; MAX_INT4_COHORT_ROWS];
        let mut work = [ProjectionWork::default(); MAX_INT4_COHORT_ROWS];
        // Validate EVERY member, including the last one's admission, before
        // charging any ledger, polling, unpacking or touching any output.
        for (index, member) in members.iter().enumerate() {
            let input: &'a ActivationBuffer = member.input;
            inputs[index] = input.values()?;
            activation_scales[index] = input.scale()?;
            if inputs[index].len() != self.columns { return Err(LinearError::Shape.into()); }
            counts[index] = member.selection.checked_count(self.rows)?;
            if member.output.len() != counts[index] { return Err(LinearError::OutputShape.into()); }
            work[index] = self.planned_work(member.selection)?;
            member.ledger.preflight(work[index])?;
            member.ledger.reserved().checked_add(work[index])?;
        }
        for (index, member) in members.iter_mut().enumerate() { member.ledger.charge(work[index])?; }
        let mut positions = [0_usize; MAX_INT4_COHORT_ROWS];
        let mut since_poll = ROWS_PER_CHECKPOINT;
        loop {
            let mut next = self.rows; // Sentinel outside every checked selection.
            for (index, member) in members.iter().enumerate() {
                if positions[index] < counts[index] { next = next.min(member.selection.row(positions[index])); }
            }
            if next == self.rows { break; }
            let mut lanes = [0_usize; MAX_INT4_COHORT_ROWS];
            let mut lane_count = 0;
            for (index, member) in members.iter().enumerate() {
                if positions[index] < counts[index] && member.selection.row(positions[index]) == next {
                    lanes[lane_count] = index; lane_count += 1;
                }
            }
            let mut totals = [0.0_f32; MAX_INT4_COHORT_ROWS];
            for group in 0..self.groups {
                // Never execute more than this many actual group dots between
                // polls, even when the tile population changes with the mask.
                if since_poll + lane_count > ROWS_PER_CHECKPOINT { poll(control)?; since_poll = 0; }
                let start = group * self.group_width;
                let width = self.group_width.min(self.columns - start);
                let mut activations: [&[i8]; MAX_INT4_COHORT_ROWS] = [&[]; MAX_INT4_COHORT_ROWS];
                let mut dots = [0_i32; MAX_INT4_COHORT_ROWS];
                for lane in 0..lane_count {
                    activations[lane] = &inputs[lanes[lane]][start..start + width];
                }
                self.kernel.dot_rows_range_into(&activations[..lane_count], self.packed,
                    next * self.columns + start, width, &mut dots[..lane_count], scratch).map_err(LinearError::CpuKernel)?;
                let weight_scale = self.scales[next * self.groups + group];
                for lane in 0..lane_count {
                    accumulate_group(&mut totals[lane], dots[lane], activation_scales[lanes[lane]], weight_scale)?;
                }
                since_poll += lane_count;
            }
            for lane in 0..lane_count {
                let member = lanes[lane];
                members[member].output[positions[member]] = cast(totals[lane])?;
                positions[member] += 1;
            }
        }
        poll(control)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
