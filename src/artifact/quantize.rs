//! Immutable, generic weight-quantization stages.
//!
//! The converter's first int8 stage is deliberately a small serial primitive:
//! one logical output row at a time, with no host-native packing or scheduler
//! decision.  The resulting signed bytes, per-row f32 scales, and row sums are
//! the semantic Generic representation that later packing derivations consume.

use std::error::Error;
use std::fmt;

/// The largest signed magnitude emitted by the portable int8 recipe.
pub const PORTABLE_I8_MAX_MAGNITUDE: i8 = 127;

/// A bounded row-major result of per-output-channel symmetric int8 quantization.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedI8Rows {
    /// Number of logical output channels.
    pub rows: usize,
    /// Logical input width for each output channel.
    pub columns: usize,
    /// Row-major signed Generic bytes, with zero point zero.
    pub values: Vec<i8>,
    /// One non-negative f32 scale per output channel.
    pub scales: Vec<f32>,
    /// One exact signed-byte sum per output channel for offset-domain kernels.
    pub row_sums: Vec<i32>,
}

/// Refusals at the portable quantization boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QuantizeError {
    /// The caller's source slice is not the declared row-major matrix size.
    Shape {
        rows: usize,
        columns: usize,
        expected: usize,
        observed: usize,
    },
    /// A zero-sized row would leave the quantization scale unspecified.
    EmptyRowWidth { rows: usize },
    /// Input to the deterministic recipe must be finite before scale selection.
    NonFinite {
        row: usize,
        column: usize,
        bits: u32,
    },
    /// A generic caller requested a matrix too large for the i32 row-sum ABI.
    RowSumOverflow { row: usize },
}

impl fmt::Display for QuantizeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape {
                rows,
                columns,
                expected,
                observed,
            } => write!(
                formatter,
                "portable-int8-v1 shape rows={rows} columns={columns} requires {expected} f32 values, observed {observed}"
            ),
            Self::EmptyRowWidth { rows } => write!(
                formatter,
                "portable-int8-v1 requires a non-zero row width, observed rows={rows} columns=0"
            ),
            Self::NonFinite { row, column, bits } => write!(
                formatter,
                "portable-int8-v1 rejects non-finite source value at row={row} column={column} bits=0x{bits:08x}"
            ),
            Self::RowSumOverflow { row } => write!(
                formatter,
                "portable-int8-v1 row-sum does not fit i32 for row={row}"
            ),
        }
    }
}

impl Error for QuantizeError {}

/// Quantize a row-major f32 matrix with the portable per-output-channel recipe.
///
/// The scale is `max(abs(row))/127` for each nonzero row, otherwise positive
/// and negative source zero both produce the canonical `+0.0` scale and all
/// zero bytes.  Values are divided by that scale, clamped to `[-127, 127]`,
/// and rounded to nearest with ties-to-even.  The function performs no native
/// packing; its row-major output is the only canonical Generic ordering.
pub fn quantize_per_output_channel_i8(
    source: &[f32],
    rows: usize,
    columns: usize,
) -> Result<QuantizedI8Rows, QuantizeError> {
    let expected = rows.checked_mul(columns).ok_or(QuantizeError::Shape {
        rows,
        columns,
        expected: usize::MAX,
        observed: source.len(),
    })?;
    if source.len() != expected {
        return Err(QuantizeError::Shape {
            rows,
            columns,
            expected,
            observed: source.len(),
        });
    }
    if columns == 0 {
        return Err(QuantizeError::EmptyRowWidth { rows });
    }

    let mut values = Vec::with_capacity(expected);
    let mut scales = Vec::with_capacity(rows);
    let mut row_sums = Vec::with_capacity(rows);

    for (row_index, row) in source.chunks_exact(columns).enumerate() {
        let mut max_magnitude = 0.0_f32;
        for (column, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(QuantizeError::NonFinite {
                    row: row_index,
                    column,
                    bits: value.to_bits(),
                });
            }
            max_magnitude = max_magnitude.max(value.abs());
        }

        if max_magnitude == 0.0 {
            scales.push(0.0);
            values.extend(std::iter::repeat_n(0_i8, columns));
            row_sums.push(0);
            continue;
        }

        let scale = max_magnitude / f32::from(PORTABLE_I8_MAX_MAGNITUDE);
        let mut row_sum = 0_i32;
        for &value in row {
            let normalized = (value / scale).clamp(
                -f32::from(PORTABLE_I8_MAX_MAGNITUDE),
                f32::from(PORTABLE_I8_MAX_MAGNITUDE),
            );
            let quantized = round_nearest_ties_even(normalized);
            row_sum = row_sum
                .checked_add(i32::from(quantized))
                .ok_or(QuantizeError::RowSumOverflow { row: row_index })?;
            values.push(quantized);
        }
        scales.push(scale);
        row_sums.push(row_sum);
    }

    Ok(QuantizedI8Rows {
        rows,
        columns,
        values,
        scales,
        row_sums,
    })
}

fn round_nearest_ties_even(value: f32) -> i8 {
    debug_assert!((-127.0..=127.0).contains(&value));
    let truncated = value as i32;
    let rounded = if value >= 0.0 {
        let fractional = value - truncated as f32;
        round_from_lower(truncated, fractional)
    } else {
        let fractional = truncated as f32 - value;
        -round_from_lower(-truncated, fractional)
    };
    i8::try_from(rounded).expect("clamped portable int8 value fits i8")
}

fn round_from_lower(lower: i32, fractional: f32) -> i32 {
    if fractional > 0.5 || (fractional == 0.5 && lower % 2 != 0) {
        lower + 1
    } else {
        lower
    }
}

#[cfg(test)]
mod tests {
    use super::{
        quantize_per_output_channel_i8, round_nearest_ties_even, QuantizeError,
        PORTABLE_I8_MAX_MAGNITUDE,
    };

    #[test]
    fn ties_round_to_even_in_both_directions() {
        let cases = [
            (0.5, 0),
            (1.5, 2),
            (2.5, 2),
            (-0.5, 0),
            (-1.5, -2),
            (-2.5, -2),
        ];
        for (input, expected) in cases {
            assert_eq!(round_nearest_ties_even(input), expected);
        }
    }

    #[test]
    fn zero_row_has_canonical_positive_zero_scale() {
        let output = quantize_per_output_channel_i8(&[-0.0, 0.0, -0.0], 1, 3)
            .expect("finite zero row quantizes");
        assert_eq!(output.values, vec![0, 0, 0]);
        assert_eq!(output.scales, vec![0.0]);
        assert_eq!(output.scales[0].to_bits(), 0);
        assert_eq!(output.row_sums, vec![0]);
    }

    #[test]
    fn endpoint_row_keeps_signed_zero_point_contract() {
        let output =
            quantize_per_output_channel_i8(&[1.0, -1.0, 0.0], 1, 3).expect("finite row quantizes");
        assert_eq!(output.values, vec![127, -127, 0]);
        assert_eq!(output.scales, vec![1.0 / 127.0]);
        assert_eq!(output.row_sums, vec![0]);
        assert_eq!(PORTABLE_I8_MAX_MAGNITUDE, 127);
    }

    #[test]
    fn non_finite_value_is_a_named_refusal() {
        let error = quantize_per_output_channel_i8(&[0.0, f32::NAN], 1, 2)
            .expect_err("NaN cannot select a deterministic scale");
        assert_eq!(
            error,
            QuantizeError::NonFinite {
                row: 0,
                column: 1,
                bits: f32::NAN.to_bits(),
            }
        );
    }

    #[test]
    fn matrix_shape_and_empty_row_width_refuse_before_output() {
        let shape = quantize_per_output_channel_i8(&[1.0], 1, 2)
            .expect_err("declared matrix size must match source");
        assert!(matches!(shape, QuantizeError::Shape { .. }));

        let empty = quantize_per_output_channel_i8(&[], 1, 0)
            .expect_err("per-output scale is undefined for an empty row");
        assert_eq!(empty, QuantizeError::EmptyRowWidth { rows: 1 });
    }
}
