//! Executable portable int8 linear algebra over borrowed Generic rows.
//!
//! This is a code-first scalar implementation, NOT artifact activation, a
//! dispatch award, or a BF16 fidelity claim. It never expands the weight matrix
//! to floats. The caller owns process admission and the cancellation controller.
//! Activation storage is provisioned once and reusable across Q/K/V or gate/up.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use super::{
    artifact_bridge::PortableQuantizedMatrix,
    decode::{DecodeCancellationKind, DecodeStepControl},
    int8::{Int8KernelError, dequantize_i32_fixed, dot_s8s8},
    lmhead::NANBEIGE_VOCAB_SIZE,
    quant_algebra::{EpilogueScales, MAX_MODEL_K},
    tensor::Bf16,
};

pub const PORTABLE_INT8_LINEAR_VERSION: &str = "portable-s8-dynamic-rne-fixed-epilogue-v1";
const ROWS_PER_CHECKPOINT: usize = 32;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LinearError {
    Shape, WeightScale, RowSum, Activation, ScaleUnderflow, Workspace,
    Selection, OutputShape, WorkBudget, Allocation, NonFiniteOutput,
    Kernel(Int8KernelError), Cancelled(DecodeCancellationKind),
}
impl fmt::Display for LinearError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Shape => "portable int8 matrix geometry refused",
            Self::WeightScale => "portable int8 requires positive finite row scales",
            Self::RowSum => "portable int8 row-sum sidecar mismatch",
            Self::Activation => "portable int8 requires a finite nonempty activation",
            Self::ScaleUnderflow => "portable int8 activation scale underflowed",
            Self::Workspace => "portable int8 activation workspace exceeded or invalid",
            Self::Selection => "portable int8 selected rows must be nonempty, ascending and unique",
            Self::OutputShape => "portable int8 output slice has the wrong length",
            Self::WorkBudget => "portable int8 projection work budget exceeded",
            Self::Allocation => "portable int8 workspace allocation refused",
            Self::NonFiniteOutput => "portable int8 projection is not finite in its output dtype",
            Self::Kernel(_) => "portable int8 canonical integer or scale stage failed",
            Self::Cancelled(_) => "portable int8 projection cancelled",
        })
    }
}
impl Error for LinearError {}
impl From<Int8KernelError> for LinearError {
    fn from(error: Int8KernelError) -> Self { Self::Kernel(error) }
}

/// Exact algorithmic projection ceilings, not measured instructions or traffic.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionWork {
    pub dot_products: u64,
    pub multiply_accumulates: u64,
}
impl ProjectionWork {
    pub fn for_shape(rows: usize, columns: usize) -> Result<Self, LinearError> {
        let rows = u64::try_from(rows).map_err(|_| LinearError::WorkBudget)?;
        let columns = u64::try_from(columns).map_err(|_| LinearError::WorkBudget)?;
        Ok(Self { dot_products: rows, multiply_accumulates: rows.checked_mul(columns).ok_or(LinearError::WorkBudget)? })
    }
    pub fn checked_add(self, other: Self) -> Result<Self, LinearError> {
        Ok(Self {
            dot_products: self.dot_products.checked_add(other.dot_products).ok_or(LinearError::WorkBudget)?,
            multiply_accumulates: self.multiply_accumulates.checked_add(other.multiply_accumulates).ok_or(LinearError::WorkBudget)?,
        })
    }
    pub fn fits(self, ceiling: Self) -> bool {
        self.dot_products <= ceiling.dot_products && self.multiply_accumulates <= ceiling.multiply_accumulates
    }
}

/// Charges a complete projection before its first checkpoint/MAC. A cancelled,
/// failed or partially evaluated projection is NEVER refunded. No reset API.
pub struct ProjectionLedger { remaining: ProjectionWork, reserved: ProjectionWork }
impl ProjectionLedger {
    pub fn new(budget: ProjectionWork) -> Self { Self { remaining: budget, reserved: ProjectionWork::default() } }
    pub fn remaining(&self) -> ProjectionWork { self.remaining }
    pub fn reserved(&self) -> ProjectionWork { self.reserved }
    pub fn preflight(&self, work: ProjectionWork) -> Result<(), LinearError> {
        if work.fits(self.remaining) { Ok(()) } else { Err(LinearError::WorkBudget) }
    }
    fn charge(&mut self, work: ProjectionWork) -> Result<(), LinearError> {
        self.preflight(work)?;
        let reserved = self.reserved.checked_add(work)?;
        let remaining = ProjectionWork {
            dot_products: self.remaining.dot_products - work.dot_products,
            multiply_accumulates: self.remaining.multiply_accumulates - work.multiply_accumulates,
        };
        self.remaining = remaining; self.reserved = reserved; Ok(())
    }
}

/// One explicitly admitted activation rail. No allocation occurs in encode.
/// A failed encoding invalidates the previous bytes, so stale activations can
/// never be projected accidentally after a rejected norm or nonfinite input.
pub struct ActivationBuffer { values: Vec<i8>, length: usize, scale: f32, valid: bool }
impl ActivationBuffer {
    pub fn try_new(max_columns: usize) -> Result<Self, LinearError> {
        if max_columns == 0 || max_columns > MAX_MODEL_K { return Err(LinearError::Workspace); }
        let mut values = Vec::new();
        values.try_reserve_exact(max_columns).map_err(|_| LinearError::Allocation)?;
        values.resize(max_columns, 0);
        Ok(Self { values, length: 0, scale: 1.0, valid: false })
    }
    pub fn encode_bf16(&mut self, input: &[Bf16]) -> Result<(), LinearError> {
        self.encode(input.len(), |index| input[index].to_f32())
    }
    pub fn encode_f32(&mut self, input: &[f32]) -> Result<(), LinearError> {
        self.encode(input.len(), |index| input[index])
    }
    fn encode(&mut self, length: usize, at: impl Fn(usize) -> f32) -> Result<(), LinearError> {
        self.valid = false;
        if length == 0 { return Err(LinearError::Activation); }
        if length > self.values.len() { return Err(LinearError::Workspace); }
        let mut maximum = 0.0_f32;
        for index in 0..length {
            let value = at(index);
            if !value.is_finite() { return Err(LinearError::Activation); }
            maximum = maximum.max(value.abs());
        }
        let scale = if maximum == 0.0 { 1.0 } else { maximum / 127.0 };
        if scale == 0.0 { return Err(LinearError::ScaleUnderflow); }
        if !scale.is_finite() || scale < 0.0 { return Err(LinearError::Activation); }
        for index in 0..length {
            self.values[index] = (at(index) / scale).clamp(-127.0, 127.0).round_ties_even() as i8;
        }
        self.length = length; self.scale = scale; self.valid = true; Ok(())
    }
    pub fn values(&self) -> Result<&[i8], LinearError> {
        if !self.valid { return Err(LinearError::Workspace); }
        Ok(&self.values[..self.length])
    }
    pub fn scale(&self) -> Result<f32, LinearError> {
        if self.valid { Ok(self.scale) } else { Err(LinearError::Workspace) }
    }
}

/// True row slicing: unselected weights are neither read nor dequantized.
#[derive(Clone, Copy)]
pub enum LinearRows<'a> { All, Selected(&'a [u32]) }
impl LinearRows<'_> {
    pub fn checked_count(self, rows: usize) -> Result<usize, LinearError> {
        match self {
            Self::All if rows > 0 => Ok(rows),
            Self::Selected(ids) if !ids.is_empty()
                && ids.iter().all(|&id| (id as usize) < rows)
                && ids.windows(2).all(|pair| pair[0] < pair[1]) => Ok(ids.len()),
            _ => Err(LinearError::Selection),
        }
    }
    fn row(self, index: usize) -> usize {
        match self { Self::All => index, Self::Selected(ids) => ids[index] as usize }
    }
}

/// Borrowed immutable row-major int8 weights. Construction recomputes every
/// semantic sum once, checks the entire scale sidecar, and bounds integer K.
/// A view does not authenticate a file, choose an envelope authority or clone
/// weights. Full i8 bit patterns, including -128, retain exact scalar semantics.
#[derive(Clone, Copy)]
pub struct QuantizedLinear<'a> { rows: usize, columns: usize, values: &'a [i8], scales: &'a [f32] }
impl<'a> QuantizedLinear<'a> {
    pub fn checked(rows: usize, columns: usize, values: &'a [i8], scales: &'a [f32], sums: &[i32])
        -> Result<Self, LinearError> {
        if rows == 0 || rows > NANBEIGE_VOCAB_SIZE || columns == 0 || columns > MAX_MODEL_K
            || rows.checked_mul(columns) != Some(values.len()) || scales.len() != rows || sums.len() != rows {
            return Err(LinearError::Shape);
        }
        for (row, (&scale, &sum)) in values.chunks_exact(columns).zip(scales.iter().zip(sums)) {
            if !scale.is_finite() || scale <= 0.0 { return Err(LinearError::WeightScale); }
            let actual: i64 = row.iter().map(|&value| i64::from(value)).sum();
            if actual != i64::from(sum) { return Err(LinearError::RowSum); }
        }
        Ok(Self { rows, columns, values, scales })
    }
    pub fn from_portable(matrix: &'a PortableQuantizedMatrix) -> Result<Self, LinearError> {
        Self::checked(matrix.rows(), matrix.columns(), matrix.values(), matrix.row_scales(), matrix.row_sums())
    }
    pub fn rows(self) -> usize { self.rows }
    pub fn columns(self) -> usize { self.columns }
    pub fn project_f32_into<C: DecodeStepControl>(&self, input: &ActivationBuffer, rows: LinearRows<'_>,
        output: &mut [f32], ledger: &mut ProjectionLedger, control: &mut C) -> Result<(), LinearError> {
        self.project(input, rows, output.len(), ledger, control, |index, value| { output[index] = value; Ok(()) })
    }
    pub fn project_bf16_into<C: DecodeStepControl>(&self, input: &ActivationBuffer, rows: LinearRows<'_>,
        output: &mut [Bf16], ledger: &mut ProjectionLedger, control: &mut C) -> Result<(), LinearError> {
        self.project(input, rows, output.len(), ledger, control, |index, value| {
            let cast = Bf16::from_f32(value);
            if !cast.to_f32().is_finite() { return Err(LinearError::NonFiniteOutput); }
            output[index] = cast; Ok(())
        })
    }
    fn project<C: DecodeStepControl>(&self, input: &ActivationBuffer, rows: LinearRows<'_>, length: usize,
        ledger: &mut ProjectionLedger, control: &mut C, mut store: impl FnMut(usize, f32) -> Result<(), LinearError>)
        -> Result<(), LinearError> {
        let values = input.values()?;
        if values.len() != self.columns { return Err(LinearError::Shape); }
        let count = rows.checked_count(self.rows)?;
        if count != length { return Err(LinearError::OutputShape); }
        ledger.charge(ProjectionWork::for_shape(count, self.columns)?)?;
        for index in 0..count {
            if index % ROWS_PER_CHECKPOINT == 0 { poll(control)?; }
            let row = rows.row(index);
            let start = row * self.columns;
            let accumulator = dot_s8s8(values, &self.values[start..start + self.columns])?;
            let result = dequantize_i32_fixed(accumulator, EpilogueScales {
                activation: input.scale()?, row: self.scales[row], column: 1.0, group: 1.0,
            })?;
            if !result.is_finite() { return Err(LinearError::NonFiniteOutput); }
            store(index, result)?;
        }
        poll(control)
    }
}
fn poll<C: DecodeStepControl>(control: &mut C) -> Result<(), LinearError> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(LinearError::Cancelled(cause)), None => Ok(()) }
}

#[cfg(test)]
mod tests;
