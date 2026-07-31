//! Portable scalar integer kernels: the exact floor for every SIMD tier.
//!
//! Weights use signed i8 with zero point zero. Accumulation order is always
//! increasing K, and every release-path i32 addition is checked. SIMD tiers
//! must match these results bit-for-bit.

use std::error::Error;
use std::fmt;

/// Model K dimensions that this one-model kernel family supports.
pub const MODEL_KS: [usize; 3] = [3072, 6144, 10752];

/// Model output widths for projection and lm-head rows.
pub const MODEL_NS: [usize; 5] = [6144, 1024, 3072, 10752, 166_144];

/// `10752 * abs(-128 * -128)`: full-domain signed i8 MAC bound.
pub const MAX_S8_S8_K_10752: i64 = 176_160_768;

/// `10752 * abs(255 * -128)`: raw offset-domain u8-by-i8 MAC bound.
pub const MAX_U8_S8_RAW_K_10752: i64 = 350_945_280;

/// `128 * abs(sum_k(-128))` at K=10752.
pub const MAX_U8_S8_CORRECTION_K_10752: i64 = 176_160_768;

/// Conservative sum of the raw u8 MAC and correction magnitudes.
pub const MAX_U8_S8_RAW_PLUS_CORRECTION_K_10752: i64 = 527_106_048;

/// Fail-closed scalar-kernel shape and arithmetic errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Int8KernelError {
    LengthMismatch {
        operand: &'static str,
        expected: usize,
        observed: usize,
    },
    UnsupportedModelShape { k: usize, n: usize },
    LengthOverflow { rows: usize, columns: usize },
    AccumulatorOverflow { operation: &'static str, k_index: usize },
    Int4Layout { expected_weights: usize, observed_weights: usize },
}

impl fmt::Display for Int8KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                operand,
                expected,
                observed,
            } => write!(
                formatter,
                "{operand} length mismatch: expected {expected}, observed {observed}"
            ),
            Self::UnsupportedModelShape { k, n } => {
                write!(formatter, "unsupported fixed int8 model shape K={k}, N={n}")
            }
            Self::LengthOverflow { rows, columns } => {
                write!(formatter, "matrix length overflows usize for rows={rows}, columns={columns}")
            }
            Self::AccumulatorOverflow { operation, k_index } => {
                write!(formatter, "i32 accumulator overflow in {operation} at K index {k_index}")
            }
            Self::Int4Layout {
                expected_weights,
                observed_weights,
            } => write!(
                formatter,
                "packed int4 layout mismatch: expected {expected_weights} unpacked weights, observed {observed_weights}"
            ),
        }
    }
}

impl Error for Int8KernelError {}

/// Return whether a K/N pair belongs to Nanbeige4.2-3B's fixed weight shapes.
#[must_use]
pub const fn is_model_shape(k: usize, n: usize) -> bool {
    contains(&MODEL_KS, k) && contains(&MODEL_NS, n)
}

/// Plain scalar signed-i8 dot product with canonical increasing-K order.
pub fn dot_s8s8(input: &[i8], weights: &[i8]) -> Result<i32, Int8KernelError> {
    if input.len() != weights.len() {
        return Err(Int8KernelError::LengthMismatch {
            operand: "dot weights",
            expected: input.len(),
            observed: weights.len(),
        });
    }
    let mut accumulator = 0_i32;
    for (k_index, (&activation, &weight)) in input.iter().zip(weights).enumerate() {
        let product = i32::from(activation) * i32::from(weight);
        accumulator = accumulator
            .checked_add(product)
            .ok_or(Int8KernelError::AccumulatorOverflow {
                operation: "s8*s8 dot",
                k_index,
            })?;
    }
    Ok(accumulator)
}

/// Plain scalar GEMV with weights laid out as `N` consecutive `K` rows.
pub fn gemv_s8s8(input: &[i8], weights: &[i8], n: usize) -> Result<Vec<i32>, Int8KernelError> {
    let k = input.len();
    let expected_weights = checked_len(n, k)?;
    if weights.len() != expected_weights {
        return Err(Int8KernelError::LengthMismatch {
            operand: "gemv weights",
            expected: expected_weights,
            observed: weights.len(),
        });
    }
    let mut output = Vec::with_capacity(n);
    for row in weights.chunks_exact(k) {
        output.push(dot_s8s8(input, row)?);
    }
    Ok(output)
}

/// Plain scalar GEMM for dynamic M and fixed or caller-selected N/K.
///
/// Activations are `M × K`, weights are `N × K`, and results are row-major
/// `M × N`. Each result uses the same increasing-K order as [`dot_s8s8`].
pub fn gemm_s8s8(
    activations: &[i8],
    m: usize,
    k: usize,
    weights: &[i8],
    n: usize,
) -> Result<Vec<i32>, Int8KernelError> {
    let expected_activations = checked_len(m, k)?;
    if activations.len() != expected_activations {
        return Err(Int8KernelError::LengthMismatch {
            operand: "gemm activations",
            expected: expected_activations,
            observed: activations.len(),
        });
    }
    let expected_weights = checked_len(n, k)?;
    if weights.len() != expected_weights {
        return Err(Int8KernelError::LengthMismatch {
            operand: "gemm weights",
            expected: expected_weights,
            observed: weights.len(),
        });
    }
    let mut output = Vec::with_capacity(checked_len(m, n)?);
    for activation_row in activations.chunks_exact(k) {
        output.extend(gemv_s8s8(activation_row, weights, n)?);
    }
    Ok(output)
}

/// GEMV constrained to the model's fixed K/N catalog.
pub fn model_gemv_s8s8(
    input: &[i8],
    weights: &[i8],
    n: usize,
) -> Result<Vec<i32>, Int8KernelError> {
    if !is_model_shape(input.len(), n) {
        return Err(Int8KernelError::UnsupportedModelShape { k: input.len(), n });
    }
    gemv_s8s8(input, weights, n)
}

/// GEMM constrained to the model's fixed K/N catalog and dynamic M.
pub fn model_gemm_s8s8(
    activations: &[i8],
    m: usize,
    k: usize,
    weights: &[i8],
    n: usize,
) -> Result<Vec<i32>, Int8KernelError> {
    if !is_model_shape(k, n) {
        return Err(Int8KernelError::UnsupportedModelShape { k, n });
    }
    gemm_s8s8(activations, m, k, weights, n)
}

/// Unpack signed two's-complement int4 values (low nibble first) into i8.
#[must_use]
pub fn unpack_int4_to_i8(packed: &[u8]) -> Vec<i8> {
    let mut unpacked = Vec::with_capacity(packed.len() * 2);
    for &byte in packed {
        unpacked.push(sign_extend_int4(byte & 0x0f));
        unpacked.push(sign_extend_int4(byte >> 4));
    }
    unpacked
}

/// Int4 weights are unpacked to i8 then use the exact scalar i8 GEMV path.
pub fn gemv_int4_s8(
    input: &[i8],
    packed_weights: &[u8],
    n: usize,
) -> Result<Vec<i32>, Int8KernelError> {
    let expected_weights = checked_len(n, input.len())?;
    let unpacked = unpack_int4_to_i8(packed_weights);
    if unpacked.len() != expected_weights {
        return Err(Int8KernelError::Int4Layout {
            expected_weights,
            observed_weights: unpacked.len(),
        });
    }
    gemv_s8s8(input, &unpacked, n)
}

/// Canonical XOR-0x80 offset correction: `(u8 - 128) * s8` in fixed order.
///
/// This makes raw u8-by-s8 accumulation and the `128 * sum(weights)` correction
/// explicit for the offset-domain proof while producing signed-domain semantics.
pub fn dot_u8s8_xor128(input: &[u8], weights: &[i8]) -> Result<i32, Int8KernelError> {
    if input.len() != weights.len() {
        return Err(Int8KernelError::LengthMismatch {
            operand: "offset dot weights",
            expected: input.len(),
            observed: weights.len(),
        });
    }
    let mut raw = 0_i32;
    let mut row_sum = 0_i32;
    for (k_index, (&activation, &weight)) in input.iter().zip(weights).enumerate() {
        raw = raw
            .checked_add(i32::from(activation) * i32::from(weight))
            .ok_or(Int8KernelError::AccumulatorOverflow {
                operation: "u8*s8 raw dot",
                k_index,
            })?;
        row_sum = row_sum
            .checked_add(i32::from(weight))
            .ok_or(Int8KernelError::AccumulatorOverflow {
                operation: "u8*s8 row sum",
                k_index,
            })?;
    }
    let correction = row_sum
        .checked_mul(128)
        .ok_or(Int8KernelError::AccumulatorOverflow {
            operation: "u8*s8 offset correction",
            k_index: input.len(),
        })?;
    raw.checked_sub(correction)
        .ok_or(Int8KernelError::AccumulatorOverflow {
            operation: "u8*s8 corrected dot",
            k_index: input.len(),
        })
}

const fn contains(values: &[usize], candidate: usize) -> bool {
    let mut index = 0;
    while index < values.len() {
        if values[index] == candidate {
            return true;
        }
        index += 1;
    }
    false
}

fn checked_len(rows: usize, columns: usize) -> Result<usize, Int8KernelError> {
    rows.checked_mul(columns)
        .ok_or(Int8KernelError::LengthOverflow { rows, columns })
}

const fn sign_extend_int4(nibble: u8) -> i8 {
    let nibble = nibble & 0x0f;
    if nibble & 0x08 == 0 {
        nibble as i8
    } else {
        (nibble as i8) - 16
    }
}
