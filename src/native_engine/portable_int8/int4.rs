//! Group-scaled INT4 projections on the existing dynamic INT8 activation rail.
//!
//! This is an explicit numerical candidate, not an INT8 model/profile alias.
//! Each signed group dot is dequantized in the canonical scale order BEFORE
//! increasing-group f32 summation. BF16 is cast once, after the complete row.
//! No dense weight expansion, hidden allocation, model activation or spawning.

use super::*;
use crate::native_engine::int4::PackedInt4Kernel;

pub const PORTABLE_INT4_LINEAR_VERSION: &str = "portable-s4-group-f32-dynamic-s8-rne-v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GroupedInt4Error {
    Geometry, GroupWidth, Scale, Scratch, Linear(LinearError),
}
impl fmt::Display for GroupedInt4Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry => f.write_str("grouped INT4 packed matrix geometry refused"),
            Self::GroupWidth => f.write_str("grouped INT4 group width is outside the bounded domain"),
            Self::Scale => f.write_str("grouped INT4 requires a positive finite scale for every row/group"),
            Self::Scratch => f.write_str("grouped INT4 unpacking scratch is too small"),
            Self::Linear(error) => write!(f, "grouped INT4 projection refused: {error}"),
        }
    }
}
impl Error for GroupedInt4Error {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Linear(error) => Some(error), _ => None }
    }
}
impl From<LinearError> for GroupedInt4Error {
    fn from(error: LinearError) -> Self { Self::Linear(error) }
}

/// Borrowed low-nibble-first signed weights, with no row/group padding.
/// Scales are row-major [N, ceil(K/group_width)]. Groups restart at each row;
/// both rows and groups may start on a high nibble. Total N*K must be even,
/// exactly as in PackedInt4Kernel. Every scale is checked at construction.
/// No row sum sidecar is trusted: the selected integer leaf owns correction.
#[derive(Clone, Copy)]
pub struct GroupedInt4Linear<'a> {
    rows: usize, columns: usize, group_width: usize, groups: usize,
    packed: &'a [u8], scales: &'a [f32], kernel: PackedInt4Kernel,
}
impl<'a> GroupedInt4Linear<'a> {
    pub fn checked(rows: usize, columns: usize, group_width: usize,
        packed: &'a [u8], scales: &'a [f32]) -> Result<Self, GroupedInt4Error> {
        if group_width == 0 || group_width > MAX_MODEL_K { return Err(GroupedInt4Error::GroupWidth); }
        if rows == 0 || rows > NANBEIGE_VOCAB_SIZE || columns == 0 || columns > MAX_MODEL_K {
            return Err(GroupedInt4Error::Geometry);
        }
        let values = rows.checked_mul(columns).ok_or(GroupedInt4Error::Geometry)?;
        let groups = columns.div_ceil(group_width);
        if values % 2 != 0 || packed.len() != values / 2
            || rows.checked_mul(groups) != Some(scales.len()) { return Err(GroupedInt4Error::Geometry); }
        if scales.iter().any(|scale| !scale.is_finite() || *scale <= 0.0) { return Err(GroupedInt4Error::Scale); }
        Ok(Self { rows, columns, group_width, groups, packed, scales, kernel: PackedInt4Kernel::scalar() })
    }
    pub fn with_kernel(mut self, kernel: PackedInt4Kernel) -> Self { self.kernel = kernel; self }
    pub fn kernel_kind(self) -> Int8KernelKind { self.kernel.kind() }
    pub fn rows(self) -> usize { self.rows }
    pub fn columns(self) -> usize { self.columns }
    pub fn group_width(self) -> usize { self.group_width }
    /// i8 elements, independent of N and vocabulary selection size.
    pub fn scratch_len(self) -> usize { self.group_width.min(self.columns) }
    /// Count actual GROUP dots, not output rows. Tail MACs are not padded.
    /// The existing ledger therefore also bounds the scale/summation stages.
    pub fn planned_work(self, rows: LinearRows<'_>) -> Result<ProjectionWork, GroupedInt4Error> {
        let count = rows.checked_count(self.rows)?;
        let mut work = ProjectionWork::for_shape(count, self.columns)?;
        work.dot_products = work.dot_products.checked_mul(self.groups as u64).ok_or(LinearError::WorkBudget)?;
        Ok(work)
    }
    /// Output is caller-private staging. On error/cancellation it may contain
    /// earlier completed rows and MUST NOT be published as a successful result.
    /// The complete requested work stays charged even after cancellation.
    pub fn project_f32_into<C: DecodeStepControl>(&self, input: &ActivationBuffer, rows: LinearRows<'_>,
        output: &mut [f32], scratch: &mut [i8], ledger: &mut ProjectionLedger, control: &mut C)
        -> Result<(), GroupedInt4Error> {
        self.project(input, rows, output.len(), scratch, ledger, control,
            |index, value| { output[index] = value; Ok(()) })
    }
    pub fn project_bf16_into<C: DecodeStepControl>(&self, input: &ActivationBuffer, rows: LinearRows<'_>,
        output: &mut [Bf16], scratch: &mut [i8], ledger: &mut ProjectionLedger, control: &mut C)
        -> Result<(), GroupedInt4Error> {
        self.project(input, rows, output.len(), scratch, ledger, control, |index, value| {
            let cast = Bf16::from_f32(value);
            if !cast.to_f32().is_finite() { return Err(LinearError::NonFiniteOutput); }
            output[index] = cast; Ok(())
        })
    }
    fn project<C: DecodeStepControl>(&self, input: &ActivationBuffer, rows: LinearRows<'_>, length: usize,
        scratch: &mut [i8], ledger: &mut ProjectionLedger, control: &mut C,
        mut store: impl FnMut(usize, f32) -> Result<(), LinearError>) -> Result<(), GroupedInt4Error> {
        let input_values = input.values()?;
        let activation_scale = input.scale()?;
        if input_values.len() != self.columns { return Err(LinearError::Shape.into()); }
        let count = rows.checked_count(self.rows)?;
        if length != count { return Err(LinearError::OutputShape.into()); }
        if scratch.len() < self.scratch_len() { return Err(GroupedInt4Error::Scratch); }
        ledger.charge(self.planned_work(rows)?)?;
        let mut since_poll = 0;
        for index in 0..count {
            let row = rows.row(index);
            let mut total = 0.0_f32;
            for group in 0..self.groups {
                if since_poll == 0 { poll(control)?; }
                let start = group * self.group_width;
                let width = self.group_width.min(self.columns - start);
                let mut dot = [0_i32];
                self.kernel.dot_rows_range_into(&[&input_values[start..start + width]], self.packed,
                    row * self.columns + start, width, &mut dot, scratch).map_err(LinearError::CpuKernel)?;
                let partial = dequantize_i32_fixed(dot[0], EpilogueScales {
                    activation: activation_scale, row: 1.0, column: 1.0,
                    group: self.scales[row * self.groups + group],
                }).map_err(LinearError::from)?;
                // Do not sum the integer groups first, factor out unequal
                // scales, change reduction order, or cast each partial to BF16.
                total += partial;
                if !total.is_finite() { return Err(LinearError::NonFiniteOutput.into()); }
                since_poll = (since_poll + 1) % ROWS_PER_CHECKPOINT;
            }
            store(index, total)?;
        }
        poll(control)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
