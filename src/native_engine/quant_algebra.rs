//! Canonical cross-ISA quantization algebra.
//!
//! Every tier starts with logical signed bytes (`s8`, zero point zero). A
//! signed tier therefore accumulates `sum(qx * w)` directly. x86 U8-by-S8
//! instructions consume `u = qx XOR 0x80`; because that byte representation is
//! exactly `qx + 128` in `u8`, they instead accumulate `sum(u * w)` and subtract
//! `128 * sum(w)` for the output row. The integer identity is exact over all
//! stored `i8` bit patterns, including `-128`; it ends at the corrected i32
//! accumulator. Float dequantization is deliberately a separate, fixed-order
//! operation below.
//!
//! At the model maximum `K = 10752`, `abs(sum(qx * w)) <= 176,160,768`, raw
//! U8-by-S8 accumulation is at most `350,945,280`, the row correction is at
//! most `176,160,768`, and the conservative raw-plus-correction envelope is
//! `527,106,048 < i32::MAX`. That leaves more than four times headroom. These
//! are bounds on every integer stage, not permissions to use a different
//! algebra in a hardware tier.

use std::error::Error;
use std::fmt;

use sha2::{Digest, Sha256};

/// Canonical logical activation and weight zero point.
pub const S8_ZERO_POINT: i8 = 0;

/// x86's exact offset-domain activation transform.
pub const X86_ACTIVATION_XOR_OFFSET: u8 = 0x80;

/// The largest K in Nanbeige4.2-3B's int8 linear shapes.
pub const MAX_MODEL_K: usize = 10_752;

/// Full-domain `s8 * s8` accumulation bound at [`MAX_MODEL_K`].
pub const MAX_S8_S8_ACCUMULATOR_K_10752: i64 = signed_s8_s8_bound(MAX_MODEL_K);

/// Full-domain raw `u8 * s8` accumulation bound at [`MAX_MODEL_K`].
pub const MAX_U8_S8_RAW_ACCUMULATOR_K_10752: i64 = raw_u8_s8_bound(MAX_MODEL_K);

/// Full-domain `128 * sum(w)` correction bound at [`MAX_MODEL_K`].
pub const MAX_OFFSET_CORRECTION_K_10752: i64 = offset_correction_bound(MAX_MODEL_K);

/// Conservative raw-plus-correction magnitude at [`MAX_MODEL_K`].
pub const MAX_OFFSET_INTERMEDIATE_K_10752: i64 = offset_intermediate_bound(MAX_MODEL_K);

/// Fixed physical table header: magic, version, reserved, row count, row width, digest.
pub const ROW_SUM_TABLE_HEADER_BYTES: usize = 52;

const ROW_SUM_TABLE_MAGIC: [u8; 8] = *b"FNLPQRS1";
const ROW_SUM_TABLE_VERSION: u16 = 1;

/// The named sequence in which every tier applies float scales after integer correction.
pub const SCALE_APPLICATION_ORDER: [ScaleOperand; 4] = [
    ScaleOperand::Activation,
    ScaleOperand::Row,
    ScaleOperand::Column,
    ScaleOperand::Group,
];

/// Int4 group partials are accumulated in increasing logical group index.
pub const INT4_GROUP_SUM_ORDER: &str = "increasing-logical-group-index-v1";

/// Full-domain signed `s8 * s8` accumulation bound for a fixed K.
#[must_use]
pub const fn signed_s8_s8_bound(k: usize) -> i64 {
    (k as i64) * 16_384
}

/// Full-domain raw `u8 * s8` accumulation bound for a fixed K.
#[must_use]
pub const fn raw_u8_s8_bound(k: usize) -> i64 {
    (k as i64) * 32_640
}

/// Full-domain `128 * sum(w)` correction bound for a fixed K.
#[must_use]
pub const fn offset_correction_bound(k: usize) -> i64 {
    (k as i64) * 16_384
}

/// Conservative raw-plus-correction intermediate bound for a fixed K.
#[must_use]
pub const fn offset_intermediate_bound(k: usize) -> i64 {
    raw_u8_s8_bound(k) + offset_correction_bound(k)
}

/// A digest of one native packed section or whole packed file.
///
/// This proves integrity of that physical copy. It is intentionally not a
/// claim that two differently packed copies have independent semantic identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhysicalSectionDigest([u8; 32]);

impl PhysicalSectionDigest {
    /// Computes the SHA-256 commitment for physical packed bytes.
    #[must_use]
    pub fn sha256(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    /// Returns the encoded SHA-256 bytes.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Typed failures at the canonical algebra's pre-dispatch boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QuantAlgebraError {
    LengthMismatch {
        operand: &'static str,
        expected: usize,
        observed: usize,
    },
    DimensionOverflow {
        rows: usize,
        row_width: usize,
    },
    IntegerRange {
        stage: &'static str,
        value: i64,
    },
    SectionDigestMismatch {
        section_id: String,
        expected: PhysicalSectionDigest,
        observed: PhysicalSectionDigest,
    },
    RowSumMismatch {
        section_id: String,
        row: usize,
        expected: i32,
        observed: i32,
    },
    RowSumTableMagic,
    RowSumTableVersion {
        observed: u16,
    },
    RowSumTableLength {
        expected: usize,
        observed: usize,
    },
    RowSumTableDimensions {
        rows: usize,
        row_width: usize,
    },
    NonFiniteScale {
        operand: ScaleOperand,
    },
    NonFiniteEpilogue,
}

impl fmt::Display for QuantAlgebraError {
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
            Self::DimensionOverflow { rows, row_width } => {
                write!(
                    formatter,
                    "row geometry overflows usize: rows={rows}, width={row_width}"
                )
            }
            Self::IntegerRange { stage, value } => {
                write!(
                    formatter,
                    "{stage} does not fit the canonical i32 stage: {value}"
                )
            }
            Self::SectionDigestMismatch {
                section_id,
                expected,
                observed,
            } => write!(
                formatter,
                "packed section digest mismatch for {section_id}: expected {}, observed {}",
                digest_hex(*expected),
                digest_hex(*observed)
            ),
            Self::RowSumMismatch {
                section_id,
                row,
                expected,
                observed,
            } => write!(
                formatter,
                "row-sum mismatch before dispatch for {section_id} row {row}: expected {expected}, observed {observed}"
            ),
            Self::RowSumTableMagic => write!(formatter, "invalid row-sum table magic"),
            Self::RowSumTableVersion { observed } => {
                write!(formatter, "unsupported row-sum table version {observed}")
            }
            Self::RowSumTableLength { expected, observed } => write!(
                formatter,
                "row-sum table length mismatch: expected {expected}, observed {observed}"
            ),
            Self::RowSumTableDimensions { rows, row_width } => write!(
                formatter,
                "invalid row-sum table dimensions: rows={rows}, width={row_width}"
            ),
            Self::NonFiniteScale { operand } => {
                write!(
                    formatter,
                    "non-finite {operand} scale is not a valid quant epilogue"
                )
            }
            Self::NonFiniteEpilogue => write!(formatter, "fixed-order quant epilogue overflowed"),
        }
    }
}

impl Error for QuantAlgebraError {}

/// One named multiplicative operand in the fixed float epilogue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaleOperand {
    Activation,
    Row,
    Column,
    Group,
}

impl fmt::Display for ScaleOperand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Activation => "activation",
            Self::Row => "row",
            Self::Column => "column",
            Self::Group => "group",
        };
        formatter.write_str(label)
    }
}

/// All scales consumed by the fixed-order quantized epilogue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpilogueScales {
    pub activation: f32,
    pub row: f32,
    pub column: f32,
    pub group: f32,
}

/// Converts a logical signed byte to the exact U8-by-S8 x86 input byte.
///
/// `i8 as u8` preserves the two's-complement byte pattern, so XORing its sign
/// bit maps `-128..=127` bijectively to `0..=255`, exactly as `qx + 128`.
#[must_use]
pub const fn s8_to_x86_offset_u8(qx: i8) -> u8 {
    (qx as u8) ^ X86_ACTIVATION_XOR_OFFSET
}

/// Applies the canonical x86 offset transform to a logical activation vector.
#[must_use]
pub fn s8_slice_to_x86_offset_u8(activations: &[i8]) -> Vec<u8> {
    activations
        .iter()
        .copied()
        .map(s8_to_x86_offset_u8)
        .collect()
}

/// Exact scalar signed-domain dot, shadowed in i64 before narrowing to i32.
pub fn signed_dot_i32(activations: &[i8], weights: &[i8]) -> Result<i32, QuantAlgebraError> {
    require_matching_lengths("signed dot weights", activations, weights)?;
    let total = activations
        .iter()
        .zip(weights)
        .fold(0_i64, |sum, (&qx, &weight)| {
            sum + i64::from(qx) * i64::from(weight)
        });
    narrow_i32("signed s8*s8 accumulator", total)
}

/// Exact x86 offset-domain dot using a freshly recomputed offline row sum.
pub fn corrected_x86_offset_dot_i32(
    activations: &[i8],
    weights: &[i8],
) -> Result<i32, QuantAlgebraError> {
    require_matching_lengths("offset dot weights", activations, weights)?;
    let row_sum = sum_row_i32(weights)?;
    corrected_x86_offset_dot_with_sum_i32(activations, weights, row_sum)
}

/// Applies the exact U8-by-S8 correction after a verified row sum is available.
///
/// Callers do not receive a correction table from this function; the only
/// production path to a stored sum is [`VerifiedPackedRows`], which is created
/// only after digest and fresh-row-sum verification.
fn corrected_x86_offset_dot_with_sum_i32(
    activations: &[i8],
    weights: &[i8],
    row_sum: i32,
) -> Result<i32, QuantAlgebraError> {
    require_matching_lengths("offset dot weights", activations, weights)?;
    let raw = activations
        .iter()
        .zip(weights)
        .fold(0_i64, |sum, (&qx, &weight)| {
            sum + i64::from(s8_to_x86_offset_u8(qx)) * i64::from(weight)
        });
    let correction = 128_i64 * i64::from(row_sum);
    let corrected = raw - correction;
    narrow_i32("corrected x86 u8*s8 accumulator", corrected)
}

/// Offline canonical sums of semantic logical signed weights, one per output row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RowSumTable {
    row_width: usize,
    row_sums: Vec<i32>,
}

impl RowSumTable {
    /// Derives sums from semantic logical weights before a physical copy is packed.
    pub fn from_semantic_weights(
        semantic_weights: &[i8],
        rows: usize,
        row_width: usize,
    ) -> Result<Self, QuantAlgebraError> {
        let expected = checked_matrix_len(rows, row_width)?;
        if semantic_weights.len() != expected {
            return Err(QuantAlgebraError::LengthMismatch {
                operand: "semantic weight matrix",
                expected,
                observed: semantic_weights.len(),
            });
        }
        if rows == 0 || row_width == 0 {
            return Err(QuantAlgebraError::RowSumTableDimensions { rows, row_width });
        }
        let mut row_sums = Vec::with_capacity(rows);
        for row in semantic_weights.chunks_exact(row_width) {
            row_sums.push(sum_row_i32(row)?);
        }
        Ok(Self {
            row_width,
            row_sums,
        })
    }

    /// The number of output rows represented by this table.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.row_sums.len()
    }

    /// The semantic K width for every row.
    #[must_use]
    pub const fn row_width(&self) -> usize {
        self.row_width
    }

    /// Binds this canonical semantic table to one physical native packed copy.
    #[must_use]
    pub fn bind_to_physical_copy(
        self,
        section_id: impl Into<String>,
        physical_section: &[u8],
    ) -> DigestBoundRowSums {
        DigestBoundRowSums {
            section_id: section_id.into(),
            table: self,
            physical_digest: PhysicalSectionDigest::sha256(physical_section),
        }
    }
}

/// A row-sum table whose exact physical packed copy is integrity-bound.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DigestBoundRowSums {
    section_id: String,
    table: RowSumTable,
    physical_digest: PhysicalSectionDigest,
}

impl DigestBoundRowSums {
    /// The identifier reported by pre-dispatch failures.
    #[must_use]
    pub fn section_id(&self) -> &str {
        &self.section_id
    }

    /// The digest that must cover the selected physical native copy.
    #[must_use]
    pub const fn physical_digest(&self) -> PhysicalSectionDigest {
        self.physical_digest
    }

    /// Encodes the portable row-sum table format with the physical digest binding.
    pub fn encode(&self) -> Result<Vec<u8>, QuantAlgebraError> {
        let rows = u32::try_from(self.table.row_count()).map_err(|_| {
            QuantAlgebraError::RowSumTableDimensions {
                rows: self.table.row_count(),
                row_width: self.table.row_width,
            }
        })?;
        let row_width = u32::try_from(self.table.row_width).map_err(|_| {
            QuantAlgebraError::RowSumTableDimensions {
                rows: self.table.row_count(),
                row_width: self.table.row_width,
            }
        })?;
        let body_bytes = self
            .table
            .row_count()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or(QuantAlgebraError::DimensionOverflow {
                rows: self.table.row_count(),
                row_width: std::mem::size_of::<i32>(),
            })?;
        let mut encoded = Vec::with_capacity(ROW_SUM_TABLE_HEADER_BYTES + body_bytes);
        encoded.extend_from_slice(&ROW_SUM_TABLE_MAGIC);
        encoded.extend_from_slice(&ROW_SUM_TABLE_VERSION.to_le_bytes());
        encoded.extend_from_slice(&0_u16.to_le_bytes());
        encoded.extend_from_slice(&rows.to_le_bytes());
        encoded.extend_from_slice(&row_width.to_le_bytes());
        encoded.extend_from_slice(&self.physical_digest.as_bytes());
        for &row_sum in &self.table.row_sums {
            encoded.extend_from_slice(&row_sum.to_le_bytes());
        }
        Ok(encoded)
    }

    /// Decodes a row-sum table whose section identity is supplied by its artifact mapping.
    pub fn decode(
        section_id: impl Into<String>,
        encoded: &[u8],
    ) -> Result<Self, QuantAlgebraError> {
        if encoded.len() < ROW_SUM_TABLE_HEADER_BYTES {
            return Err(QuantAlgebraError::RowSumTableLength {
                expected: ROW_SUM_TABLE_HEADER_BYTES,
                observed: encoded.len(),
            });
        }
        if encoded[..8] != ROW_SUM_TABLE_MAGIC {
            return Err(QuantAlgebraError::RowSumTableMagic);
        }
        let version = read_u16(encoded, 8);
        if version != ROW_SUM_TABLE_VERSION {
            return Err(QuantAlgebraError::RowSumTableVersion { observed: version });
        }
        let rows = usize::try_from(read_u32(encoded, 12)).expect("u32 always fits usize");
        let row_width = usize::try_from(read_u32(encoded, 16)).expect("u32 always fits usize");
        if rows == 0 || row_width == 0 {
            return Err(QuantAlgebraError::RowSumTableDimensions { rows, row_width });
        }
        let expected = ROW_SUM_TABLE_HEADER_BYTES
            .checked_add(
                rows.checked_mul(4)
                    .ok_or(QuantAlgebraError::DimensionOverflow { rows, row_width: 4 })?,
            )
            .ok_or(QuantAlgebraError::DimensionOverflow { rows, row_width })?;
        if encoded.len() != expected {
            return Err(QuantAlgebraError::RowSumTableLength {
                expected,
                observed: encoded.len(),
            });
        }
        let mut digest = [0_u8; 32];
        digest.copy_from_slice(&encoded[20..52]);
        let mut row_sums = Vec::with_capacity(rows);
        for offset in (ROW_SUM_TABLE_HEADER_BYTES..expected).step_by(4) {
            row_sums.push(read_i32(encoded, offset));
        }
        Ok(Self {
            section_id: section_id.into(),
            table: RowSumTable {
                row_width,
                row_sums,
            },
            physical_digest: PhysicalSectionDigest(digest),
        })
    }

    /// Verifies digest and freshly recomputed packed row sums before dispatch.
    ///
    /// The supplied physical layout is `N` consecutive logical signed K rows.
    /// A different native layout must expose an equivalent logical row iterator
    /// and perform the same digest-first, recompute-before-dispatch check.
    pub fn verify_contiguous_s8_rows<'a>(
        &'a self,
        physical_section: &'a [u8],
    ) -> Result<VerifiedPackedRows<'a>, QuantAlgebraError> {
        let observed_digest = PhysicalSectionDigest::sha256(physical_section);
        if observed_digest != self.physical_digest {
            return Err(QuantAlgebraError::SectionDigestMismatch {
                section_id: self.section_id.clone(),
                expected: self.physical_digest,
                observed: observed_digest,
            });
        }
        let expected = checked_matrix_len(self.table.row_count(), self.table.row_width)?;
        if physical_section.len() != expected {
            return Err(QuantAlgebraError::LengthMismatch {
                operand: "packed signed weight section",
                expected,
                observed: physical_section.len(),
            });
        }
        for (row_index, packed_row) in physical_section
            .chunks_exact(self.table.row_width)
            .enumerate()
        {
            let observed = sum_packed_s8_row(packed_row)?;
            let expected = self.table.row_sums[row_index];
            if observed != expected {
                return Err(QuantAlgebraError::RowSumMismatch {
                    section_id: self.section_id.clone(),
                    row: row_index,
                    expected,
                    observed,
                });
            }
        }
        Ok(VerifiedPackedRows {
            binding: self,
            physical_section,
        })
    }
}

/// A proof token for one digest-verified, row-sum-verified packed copy.
pub struct VerifiedPackedRows<'a> {
    binding: &'a DigestBoundRowSums,
    physical_section: &'a [u8],
}

impl VerifiedPackedRows<'_> {
    /// Runs the exact x86 offset identity for one already verified output row.
    pub fn corrected_x86_offset_dot_i32(
        &self,
        row: usize,
        activations: &[i8],
    ) -> Result<i32, QuantAlgebraError> {
        let row_weights = self
            .physical_section
            .chunks_exact(self.binding.table.row_width)
            .nth(row)
            .ok_or(QuantAlgebraError::LengthMismatch {
                operand: "verified row index",
                expected: self.binding.table.row_count(),
                observed: row,
            })?;
        if activations.len() != self.binding.table.row_width {
            return Err(QuantAlgebraError::LengthMismatch {
                operand: "verified row activations",
                expected: self.binding.table.row_width,
                observed: activations.len(),
            });
        }
        let mut signed_weights = Vec::with_capacity(row_weights.len());
        signed_weights.extend(row_weights.iter().map(|&byte| i8::from_ne_bytes([byte])));
        corrected_x86_offset_dot_with_sum_i32(
            activations,
            &signed_weights,
            self.binding.table.row_sums[row],
        )
    }
}

/// Applies the only permitted floating epilogue multiplication sequence.
pub fn apply_fixed_scale_order(
    accumulator: i32,
    scales: EpilogueScales,
) -> Result<f32, QuantAlgebraError> {
    for (operand, scale) in [
        (ScaleOperand::Activation, scales.activation),
        (ScaleOperand::Row, scales.row),
        (ScaleOperand::Column, scales.column),
        (ScaleOperand::Group, scales.group),
    ] {
        if !scale.is_finite() {
            return Err(QuantAlgebraError::NonFiniteScale { operand });
        }
    }
    let after_activation = accumulator as f32 * scales.activation;
    let after_row = after_activation * scales.row;
    let after_column = after_row * scales.column;
    let after_group = after_column * scales.group;
    if !after_group.is_finite() {
        return Err(QuantAlgebraError::NonFiniteEpilogue);
    }
    Ok(after_group)
}

/// Sums already-corrected int4 group partials in the single fixed group order.
pub fn sum_int4_groups_in_fixed_order(group_partials: &[i32]) -> Result<i32, QuantAlgebraError> {
    let total = group_partials
        .iter()
        .fold(0_i64, |sum, &partial| sum + i64::from(partial));
    narrow_i32("int4 group accumulator", total)
}

fn require_matching_lengths(
    operand: &'static str,
    activations: &[i8],
    weights: &[i8],
) -> Result<(), QuantAlgebraError> {
    if activations.len() == weights.len() {
        Ok(())
    } else {
        Err(QuantAlgebraError::LengthMismatch {
            operand,
            expected: activations.len(),
            observed: weights.len(),
        })
    }
}

fn checked_matrix_len(rows: usize, row_width: usize) -> Result<usize, QuantAlgebraError> {
    rows.checked_mul(row_width)
        .ok_or(QuantAlgebraError::DimensionOverflow { rows, row_width })
}

fn sum_row_i32(weights: &[i8]) -> Result<i32, QuantAlgebraError> {
    let total = weights
        .iter()
        .fold(0_i64, |sum, &weight| sum + i64::from(weight));
    narrow_i32("offline signed row sum", total)
}

fn sum_packed_s8_row(weights: &[u8]) -> Result<i32, QuantAlgebraError> {
    let total = weights.iter().fold(0_i64, |sum, &weight| {
        sum + i64::from(i8::from_ne_bytes([weight]))
    });
    narrow_i32("packed signed row sum", total)
}

fn narrow_i32(stage: &'static str, value: i64) -> Result<i32, QuantAlgebraError> {
    i32::try_from(value).map_err(|_| QuantAlgebraError::IntegerRange { stage, value })
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_i32(bytes: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn digest_hex(digest: PhysicalSectionDigest) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in digest.as_bytes() {
        use std::fmt::Write as _;
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}
