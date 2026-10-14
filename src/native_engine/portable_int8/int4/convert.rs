//! Bounded, row-streaming BF16 -> grouped signed-INT4 matrix conversion.
//!
//! An explicitly named symmetric [-7,7], nearest-even numerical candidate.
//! This is not a model-file converter, artifact-format choice or qualification
//! receipt. The output is directly executable by GroupedInt4Linear. Source
//! ownership/authenticity and process admission remain the caller's duties.

use super::*;
use sha2::{Digest, Sha256};

pub const GROUPED_INT4_RECIPE: &str = "bf16-s4-symmetric7-group-f32-rne-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4ConversionRequirement {
    pub packed_bytes: u64,
    pub scale_bytes: u64,
    pub scratch_bytes: u64,
    pub source_values: u64,
    pub groups: u64,
}
impl Int4ConversionRequirement {
    pub fn for_shape(rows: usize, columns: usize, group_width: usize) -> Result<Self, GroupedInt4Error> {
        if group_width == 0 || group_width > MAX_MODEL_K { return Err(GroupedInt4Error::GroupWidth); }
        if rows == 0 || rows > NANBEIGE_VOCAB_SIZE || columns == 0 || columns > MAX_MODEL_K {
            return Err(GroupedInt4Error::Geometry);
        }
        let values = (rows as u64).checked_mul(columns as u64).ok_or(GroupedInt4Error::Geometry)?;
        if values % 2 != 0 { return Err(GroupedInt4Error::Geometry); }
        let groups = (rows as u64).checked_mul(columns.div_ceil(group_width) as u64).ok_or(GroupedInt4Error::Geometry)?;
        Ok(Self { packed_bytes: values / 2, scale_bytes: groups.checked_mul(4).ok_or(GroupedInt4Error::Geometry)?,
            scratch_bytes: (columns as u64) * 2, source_values: values, groups })
    }
    /// Simultaneous payload of the result and one BF16 source row. Excludes
    /// the caller's source, Vec headers, allocator slack, and OS/runtime memory.
    pub fn peak_payload_bytes(self) -> Option<u64> {
        self.packed_bytes.checked_add(self.scale_bytes)?.checked_add(self.scratch_bytes)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Int4ConversionBudget {
    pub max_payload_bytes: u64,
    pub max_source_values: u64,
    pub max_groups: u64,
}

#[derive(Debug)]
pub enum Int4ConversionError<E> {
    Contract(GroupedInt4Error), Budget, Allocation, NonFiniteSource, ScaleUnderflow,
    Source(E), Cancelled(DecodeCancellationKind),
}
impl<E: fmt::Debug> fmt::Display for Int4ConversionError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(error) => write!(f, "INT4 conversion contract refused: {error}"),
            Self::Budget => f.write_str("INT4 conversion payload or work budget exceeded"),
            Self::Allocation => f.write_str("INT4 conversion allocation refused"),
            Self::NonFiniteSource => f.write_str("INT4 conversion requires a complete finite BF16 row"),
            Self::ScaleUnderflow => f.write_str("INT4 conversion group scale is not positive and finite"),
            // Caller errors may contain file paths or source text. Do not echo.
            Self::Source(_) => f.write_str("INT4 conversion source row read failed"),
            Self::Cancelled(_) => f.write_str("INT4 conversion cancelled"),
        }
    }
}
impl<E: Error + 'static> Error for Int4ConversionError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Contract(error) => Some(error), Self::Source(error) => Some(error), _ => None }
    }
}

/// Owned compact weights and f32 group scales. Payloads remain private and
/// no Debug/Deserialize implementation can bypass the checked constructor.
/// Conversion errors/cancellation drop the partial result instead of exposing it.
pub struct GroupedInt4Matrix {
    rows: usize, columns: usize, group_width: usize,
    packed: Vec<u8>, scales: Vec<f32>,
}
impl GroupedInt4Matrix {
    /// Convert exact row-major BF16 values through the same streaming program.
    pub fn from_bf16<C: DecodeStepControl>(rows: usize, columns: usize, group_width: usize,
        source: &[Bf16], budget: Int4ConversionBudget, control: &mut C)
        -> Result<Self, Int4ConversionError<std::convert::Infallible>> {
        if rows.checked_mul(columns) != Some(source.len()) {
            return Err(Int4ConversionError::Contract(GroupedInt4Error::Geometry));
        }
        Self::from_bf16_rows(rows, columns, group_width, budget, control, |row, values| {
            values.copy_from_slice(&source[row * columns..(row + 1) * columns]); Ok(())
        })
    }
    /// Each callback fills exactly one K-wide BF16 row, in increasing order.
    /// The callback is invoked only AFTER all geometry/work/payload checks and
    /// result/scratch allocation. It is synchronous, not preemptible. Cancellation
    /// is checked before/after reads and within group processing. Poison-filling
    /// the row catches callbacks that return success without filling every value.
    pub fn from_bf16_rows<E, C: DecodeStepControl>(rows: usize, columns: usize, group_width: usize,
        budget: Int4ConversionBudget, control: &mut C,
        mut read_row: impl FnMut(usize, &mut [Bf16]) -> Result<(), E>) -> Result<Self, Int4ConversionError<E>> {
        let requirement = Int4ConversionRequirement::for_shape(rows, columns, group_width)
            .map_err(Int4ConversionError::Contract)?;
        let payload = requirement.peak_payload_bytes().ok_or(Int4ConversionError::Budget)?;
        if payload > budget.max_payload_bytes || requirement.source_values > budget.max_source_values
            || requirement.groups > budget.max_groups { return Err(Int4ConversionError::Budget); }
        let packed_len = usize::try_from(requirement.packed_bytes).map_err(|_| Int4ConversionError::Budget)?;
        let scale_len = usize::try_from(requirement.groups).map_err(|_| Int4ConversionError::Budget)?;
        checkpoint(control)?;
        let mut packed = Vec::new();
        packed.try_reserve_exact(packed_len).map_err(|_| Int4ConversionError::Allocation)?;
        packed.resize(packed_len, 0_u8);
        let mut scales = Vec::new();
        scales.try_reserve_exact(scale_len).map_err(|_| Int4ConversionError::Allocation)?;
        let mut scratch = Vec::new();
        scratch.try_reserve_exact(columns).map_err(|_| Int4ConversionError::Allocation)?;
        let poison = Bf16::from_f32(f32::NAN);
        scratch.resize(columns, poison);
        let groups = columns.div_ceil(group_width);
        let mut since_poll = 0;
        for row in 0..rows {
            checkpoint(control)?;
            scratch.fill(poison);
            read_row(row, &mut scratch).map_err(Int4ConversionError::Source)?;
            checkpoint(control)?;
            for group in 0..groups {
                if since_poll == 0 { checkpoint(control)?; }
                let start = group * group_width;
                let end = columns.min(start + group_width);
                let values = &scratch[start..end];
                let mut maximum = 0.0_f32;
                for value in values {
                    let value = value.to_f32();
                    if !value.is_finite() { return Err(Int4ConversionError::NonFiniteSource); }
                    maximum = maximum.max(value.abs());
                }
                let scale = if maximum == 0.0 { 1.0 } else { maximum / 7.0 };
                if !scale.is_finite() || scale <= 0.0 { return Err(Int4ConversionError::ScaleUnderflow); }
                scales.push(scale);
                for (index, value) in values.iter().enumerate() {
                    let q = (value.to_f32() / scale).clamp(-7.0, 7.0).round_ties_even() as i8;
                    let at = row * columns + start + index;
                    let shift = (at % 2) * 4;
                    let mask = 15_u8 << shift;
                    packed[at / 2] = (packed[at / 2] & !mask) | ((q as u8 & 15) << shift);
                }
                since_poll = (since_poll + 1) % ROWS_PER_CHECKPOINT;
            }
        }
        checkpoint(control)?;
        Ok(Self { rows, columns, group_width, packed, scales })
    }
    pub fn rows(&self) -> usize { self.rows }
    pub fn columns(&self) -> usize { self.columns }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn packed_bytes(&self) -> &[u8] { &self.packed }
    pub fn group_scales(&self) -> &[f32] { &self.scales }
    pub fn recipe(&self) -> &'static str { GROUPED_INT4_RECIPE }
    /// Borrow the already checked compact result; never expand or clone it.
    pub fn as_linear(&self) -> GroupedInt4Linear<'_> {
        GroupedInt4Linear { rows: self.rows, columns: self.columns, group_width: self.group_width,
            groups: self.columns.div_ceil(self.group_width), packed: &self.packed, scales: &self.scales,
            kernel: PackedInt4Kernel::scalar() }
    }
    /// Matrix-only content identity over fixed framing and little-endian f32
    /// scale bits. Not the logical MODEL digest or a source/authenticity proof.
    /// Shape and group partition are bound even when payload bytes happen to match.
    pub fn content_sha256(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(GROUPED_INT4_RECIPE.as_bytes()); hash.update([0_u8]);
        for value in [self.rows, self.columns, self.group_width, self.packed.len(), self.scales.len()] {
            hash.update((value as u64).to_le_bytes());
        }
        hash.update(&self.packed);
        for scale in &self.scales { hash.update(scale.to_bits().to_le_bytes()); }
        hash.finalize().into()
    }
}
fn checkpoint<E, C: DecodeStepControl>(control: &mut C) -> Result<(), Int4ConversionError<E>> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(Int4ConversionError::Cancelled(cause)), None => Ok(()) }
}

#[cfg(test)]
mod tests;

mod restore;
