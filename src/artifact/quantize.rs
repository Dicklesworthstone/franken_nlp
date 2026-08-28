//! Immutable, generic weight-quantization stages.
//!
//! The converter's first int8 stage is deliberately a small serial primitive:
//! one logical output row at a time, with no host-native packing or scheduler
//! decision.  The resulting signed bytes, per-row f32 scales, and row sums are
//! the semantic Generic representation that later packing derivations consume.

use std::error::Error;
use std::fmt;

use super::converter::StorageStage;

/// The largest signed magnitude emitted by the portable int8 recipe.
pub const PORTABLE_I8_MAX_MAGNITUDE: i8 = 127;
/// Canonical finite scale attached to an all-zero quantized row.
///
/// A zero row reconstructs exactly from all-zero int8 values with any finite
/// positive scale. The artifact reader deliberately rejects non-positive
/// scales, so the conversion recipe selects this one exact value rather than
/// emitting an otherwise ambiguous `+0.0` sidecar.
pub const PORTABLE_I8_ZERO_ROW_SCALE: f32 = 1.0;

/// One canonical Generic section triplet produced for a bounded tensor panel.
///
/// The converter appends each byte vector to its separately precomputed
/// section range.  This type deliberately contains no file offset, section
/// name, native layout, or writer identity: the envelope plan remains the
/// sole range authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenericPanelBytes {
    /// Canonical logical data bytes: source BF16 bytes or signed int8 bytes.
    pub data: Vec<u8>,
    /// Little-endian IEEE f32 per-output-row scales for an int8 panel.
    pub scales: Vec<u8>,
    /// Little-endian i32 per-output-row sums for an int8 panel.
    pub row_sums: Vec<u8>,
}

/// A bounded row-major result of per-output-channel symmetric int8 quantization.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedI8Rows {
    /// Number of logical output channels.
    pub rows: usize,
    /// Logical input width for each output channel.
    pub columns: usize,
    /// Row-major signed Generic bytes, with zero point zero.
    pub values: Vec<i8>,
    /// One finite positive f32 scale per output channel.
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
    /// A verified source panel and its decoded work panel disagree with the
    /// declared row-major geometry before any output bytes are made.
    PanelShape {
        rows: usize,
        columns: usize,
        expected_bf16_bytes: usize,
        observed_bf16_bytes: usize,
        expected_f32_values: usize,
        observed_f32_values: usize,
    },
    /// The f32 work panel was not the exact BF16 widening of the verified
    /// source bytes, so conversion must not quantize a substituted panel.
    DecodedBf16Mismatch {
        element: usize,
        expected_bits: u32,
        observed_bits: u32,
    },
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
            Self::PanelShape {
                rows,
                columns,
                expected_bf16_bytes,
                observed_bf16_bytes,
                expected_f32_values,
                observed_f32_values,
            } => write!(
                formatter,
                "generic panel geometry rows={rows} columns={columns}: expected bf16-bytes={expected_bf16_bytes} observed={observed_bf16_bytes}; expected f32-values={expected_f32_values} observed={observed_f32_values}"
            ),
            Self::DecodedBf16Mismatch {
                element,
                expected_bits,
                observed_bits,
            } => write!(
                formatter,
                "generic panel BF16 decode mismatch at element={element}: expected-f32-bits=0x{expected_bits:08x} observed=0x{observed_bits:08x}"
            ),
        }
    }
}

impl Error for QuantizeError {}

/// Quantize a row-major f32 matrix with the portable per-output-channel recipe.
///
/// The scale is `max(abs(row))/127` for each nonzero row. A row containing
/// only positive and/or negative zero produces all zero bytes and the fixed
/// positive [`PORTABLE_I8_ZERO_ROW_SCALE`] sidecar. Values are divided by their
/// scale, clamped to `[-127, 127]`, and rounded to nearest with ties-to-even.
/// The function performs no native packing; its row-major output is the only
/// canonical Generic ordering.
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
            scales.push(PORTABLE_I8_ZERO_ROW_SCALE);
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

/// Encode one validated BF16 work panel according to its converter storage stage.
///
/// `Bf16Verbatim` copies the original little-endian BF16 source bytes, never a
/// widened/re-rounded f32 reconstruction.  The three int8 stages share the
/// same portable row algebra while retaining their routing distinction in the
/// caller's tensor declaration.  In both cases the result contains only the
/// Generic data/scale/row-sum payload bytes; the authoritative envelope plan
/// assigns their eventual section offsets.
pub fn encode_generic_panel(
    stage: StorageStage,
    source_bf16: &[u8],
    decoded_f32: &[f32],
    rows: usize,
    columns: usize,
) -> Result<GenericPanelBytes, QuantizeError> {
    let expected_f32_values = rows.checked_mul(columns).ok_or(QuantizeError::PanelShape {
        rows,
        columns,
        expected_bf16_bytes: usize::MAX,
        observed_bf16_bytes: source_bf16.len(),
        expected_f32_values: usize::MAX,
        observed_f32_values: decoded_f32.len(),
    })?;
    let expected_bf16_bytes =
        expected_f32_values
            .checked_mul(2)
            .ok_or(QuantizeError::PanelShape {
                rows,
                columns,
                expected_bf16_bytes: usize::MAX,
                observed_bf16_bytes: source_bf16.len(),
                expected_f32_values,
                observed_f32_values: decoded_f32.len(),
            })?;
    if source_bf16.len() != expected_bf16_bytes || decoded_f32.len() != expected_f32_values {
        return Err(QuantizeError::PanelShape {
            rows,
            columns,
            expected_bf16_bytes,
            observed_bf16_bytes: source_bf16.len(),
            expected_f32_values,
            observed_f32_values: decoded_f32.len(),
        });
    }
    for (element, (source, &decoded)) in source_bf16
        .chunks_exact(2)
        .zip(decoded_f32.iter())
        .enumerate()
    {
        let expected_bits = u32::from(u16::from_le_bytes([source[0], source[1]])) << 16;
        if decoded.to_bits() != expected_bits {
            return Err(QuantizeError::DecodedBf16Mismatch {
                element,
                expected_bits,
                observed_bits: decoded.to_bits(),
            });
        }
    }

    if stage == StorageStage::Bf16Verbatim {
        return Ok(GenericPanelBytes {
            data: source_bf16.to_vec(),
            scales: Vec::new(),
            row_sums: Vec::new(),
        });
    }

    let quantized = quantize_per_output_channel_i8(decoded_f32, rows, columns)?;
    let mut scales = Vec::with_capacity(quantized.scales.len().saturating_mul(4));
    for scale in quantized.scales {
        scales.extend_from_slice(&scale.to_bits().to_le_bytes());
    }
    let mut row_sums = Vec::with_capacity(quantized.row_sums.len().saturating_mul(4));
    for row_sum in quantized.row_sums {
        row_sums.extend_from_slice(&row_sum.to_le_bytes());
    }
    Ok(GenericPanelBytes {
        data: quantized
            .values
            .into_iter()
            .map(|value| value.to_ne_bytes()[0])
            .collect(),
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
        PORTABLE_I8_MAX_MAGNITUDE, PORTABLE_I8_ZERO_ROW_SCALE, QuantizeError, encode_generic_panel,
        quantize_per_output_channel_i8, round_nearest_ties_even,
    };
    use crate::artifact::converter::StorageStage;

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
    fn zero_row_has_canonical_positive_scale() {
        let output = quantize_per_output_channel_i8(&[-0.0, 0.0, -0.0], 1, 3)
            .expect("finite zero row quantizes");
        assert_eq!(output.values, vec![0, 0, 0]);
        assert_eq!(output.scales, vec![PORTABLE_I8_ZERO_ROW_SCALE]);
        assert_eq!(output.scales[0].to_bits(), 1.0_f32.to_bits());
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

    #[test]
    fn bf16_stage_keeps_source_bytes_and_int8_stage_emits_generic_metadata() {
        let source = [0x80, 0x3f, 0x80, 0xbf];
        let bf16 = encode_generic_panel(StorageStage::Bf16Verbatim, &source, &[1.0, -1.0], 1, 2)
            .expect("exact BF16 source panel is copied");
        assert_eq!(bf16.data, source);
        assert!(bf16.scales.is_empty());
        assert!(bf16.row_sums.is_empty());

        let int8 = encode_generic_panel(StorageStage::Int8Stage2A, &source, &[1.0, -1.0], 1, 2)
            .expect("finite panel quantizes");
        assert_eq!(int8.data, vec![127, 129]);
        assert_eq!(
            int8.scales,
            (1.0_f32 / 127.0).to_bits().to_le_bytes().to_vec()
        );
        assert_eq!(int8.row_sums, 0_i32.to_le_bytes().to_vec());
    }

    #[test]
    fn panel_geometry_refuses_before_stage_selection() {
        let error = encode_generic_panel(StorageStage::Int8Stage2B, &[0, 0], &[0.0, 1.0], 1, 2)
            .expect_err("BF16 source and decoded work must share the declared shape");
        assert!(matches!(error, QuantizeError::PanelShape { .. }));
    }

    #[test]
    fn panel_refuses_f32_not_exactly_widened_from_its_bf16_source() {
        let error = encode_generic_panel(
            StorageStage::Int8Stage2C,
            &[0x80, 0x3f, 0, 0],
            &[1.0, 1.0],
            1,
            2,
        )
        .expect_err("a substituted work panel must not enter quantization");
        assert!(matches!(
            error,
            QuantizeError::DecodedBf16Mismatch { element: 1, .. }
        ));
    }
}
