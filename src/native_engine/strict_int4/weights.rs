//! Full 201-tensor conversion from the existing sealed prepared-source session.
//! No whole BF16 model, whole shard, output file, or dense dequantized copy.

use std::{error::Error, fmt};
use sha2::{Digest, Sha256};
use crate::artifact::{
    converter::{PreparedConversionInput, PINNED_REVISION, validate_nanbeige42_census},
    safetensors::{RowPanel, PINNED_SHARD_BYTES},
};
use crate::native_engine::{
    decode::{DecodeCancellationKind, DecodeStepControl},
    kv::PHYSICAL_LAYER_COUNT,
    layer::{NANBEIGE_HIDDEN_SIZE as H, NANBEIGE_INTERMEDIATE_SIZE as I,
        NANBEIGE_Q_PROJECTION_SIZE as Q, NANBEIGE_KV_PROJECTION_SIZE as K},
    lmhead::NANBEIGE_VOCAB_SIZE as V,
    portable_int8::int4::{GroupedInt4Error, convert::{GroupedInt4Matrix,
        Int4ConversionBudget, Int4ConversionError, Int4ConversionRequirement, GROUPED_INT4_RECIPE}},
    quant_algebra::MAX_MODEL_K,
    tensor::Bf16,
    weights::Bf16Matrix,
};
use super::STRICT_INT4_PROFILE;

pub(super) const PROJECTION_SHAPES: [(usize, usize); 7] =
    [(Q, H), (K, H), (K, H), (H, Q), (I, H), (I, H), (H, I)];

#[derive(Debug)]
pub enum Int4ModelError {
    Contract, Budget, Allocation, Source, NonFinite,
    Quantization(GroupedInt4Error), Cancelled(DecodeCancellationKind),
}
impl fmt::Display for Int4ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Contract => "grouped INT4 full-model source contract refused",
            Self::Budget => "grouped INT4 full-model load budget exceeded",
            Self::Allocation => "grouped INT4 full-model allocation refused",
            Self::Source => "grouped INT4 prepared source read or revalidation failed",
            Self::NonFinite => "grouped INT4 source must contain complete finite BF16 values",
            Self::Quantization(_) => "grouped INT4 matrix conversion refused",
            Self::Cancelled(_) => "grouped INT4 full-model conversion cancelled",
        })
    }
}
impl Error for Int4ModelError {}
impl From<GroupedInt4Error> for Int4ModelError {
    fn from(error: GroupedInt4Error) -> Self { Self::Quantization(error) }
}

/// Algorithmic tensor-heap bounds, not process RSS or an instruction count.
/// The host separately admits source-session metadata/hash workspace, fixed
/// stack storage, caller-owned buffers, source preparation and allocator slack.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4ModelLoadRequirement {
    pub weight_payload_bytes: u64,
    pub scratch_payload_bytes: u64,
    pub source_values: u64,
    pub groups: u64,
    pub shard_revalidation_bytes: u64,
}
impl Int4ModelLoadRequirement {
    pub fn for_group_width(width: usize) -> Result<Self, Int4ModelError> {
        let mut result = Self {
            weight_payload_bytes: ((V * H + (2 * PHYSICAL_LAYER_COUNT + 1) * H) as u64) * 2,
            // One raw byte row coexists with the converter's BF16 scratch row.
            scratch_payload_bytes: 4 * MAX_MODEL_K as u64,
            source_values: (V * H + (2 * PHYSICAL_LAYER_COUNT + 1) * H) as u64,
            groups: 0,
            // The sealed retained handles are checked before AND after loading.
            shard_revalidation_bytes: 2 * PINNED_SHARD_BYTES,
        };
        for (rows, columns, copies) in PROJECTION_SHAPES.into_iter()
            .map(|(rows, columns)| (rows, columns, PHYSICAL_LAYER_COUNT))
            .chain(std::iter::once((V, H, 1))) {
            let matrix = Int4ConversionRequirement::for_shape(rows, columns, width)?;
            let copies = copies as u64;
            result.weight_payload_bytes = result.weight_payload_bytes
                .checked_add((matrix.packed_bytes + matrix.scale_bytes) * copies).ok_or(Int4ModelError::Budget)?;
            result.source_values = result.source_values.checked_add(matrix.source_values * copies).ok_or(Int4ModelError::Budget)?;
            result.groups = result.groups.checked_add(matrix.groups * copies).ok_or(Int4ModelError::Budget)?;
        }
        Ok(result)
    }
    pub fn peak_payload_bytes(self) -> Result<u64, Int4ModelError> {
        self.weight_payload_bytes.checked_add(self.scratch_payload_bytes).ok_or(Int4ModelError::Budget)
    }
    fn check(self, budget: Int4ModelLoadBudget) -> Result<(), Int4ModelError> {
        if self.peak_payload_bytes()? > budget.max_payload_bytes || self.source_values > budget.max_source_values
            || self.groups > budget.max_groups || self.shard_revalidation_bytes > budget.max_shard_revalidation_bytes {
            return Err(Int4ModelError::Budget);
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Int4ModelLoadBudget {
    pub max_payload_bytes: u64,
    pub max_source_values: u64,
    pub max_groups: u64,
    pub max_shard_revalidation_bytes: u64,
}

/// Source-bound candidate content identity, NOT a canonical artifact/model
/// digest or task ExecutionIdentity. It cannot be deserialized or supplied by
/// a caller. Kernel choice does not change it; group partition and bytes do.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Int4ModelIdentity {
    source_root_sha256: String,
    content_sha256: [u8; 32],
    group_width: usize,
}
impl Int4ModelIdentity {
    pub fn source_root_sha256(&self) -> &str { &self.source_root_sha256 }
    pub fn content_sha256(&self) -> [u8; 32] { self.content_sha256 }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn revision(&self) -> &'static str { PINNED_REVISION }
    pub fn profile(&self) -> &'static str { STRICT_INT4_PROFILE }
    pub fn matrix_recipe(&self) -> &'static str { GROUPED_INT4_RECIPE }
}

pub(super) struct Int4Layer {
    pub norm1: Vec<Bf16>, pub norm2: Vec<Bf16>,
    pub q: GroupedInt4Matrix, pub k: GroupedInt4Matrix, pub v: GroupedInt4Matrix,
    pub o: GroupedInt4Matrix, pub gate: GroupedInt4Matrix, pub up: GroupedInt4Matrix, pub down: GroupedInt4Matrix,
}
/// One owned immutable model; all engines borrow it. Private fields prevent
/// a partial/mixed census or caller-supplied content identity entering execution.
pub struct Int4ModelWeights {
    pub(super) embeddings: Bf16Matrix,
    pub(super) final_norm: Vec<Bf16>,
    pub(super) layers: [Int4Layer; PHYSICAL_LAYER_COUNT],
    pub(super) head: GroupedInt4Matrix,
    identity: Int4ModelIdentity,
}
impl Int4ModelWeights {
    /// Source preparation remains separately admitted by the host. All load
    /// ceilings are checked before rehash, source reads or weight allocation.
    /// The prepared source must not be mutated concurrently. Retained-handle
    /// checks detect persistent changes, not an adversarial mutate/restore race.
    /// Shard rehash and each read are synchronous and not preemptible; polls
    /// occur around these calls and throughout row/group conversion.
    pub fn from_prepared<C: DecodeStepControl>(source: &PreparedConversionInput, group_width: usize,
        budget: Int4ModelLoadBudget, control: &mut C) -> Result<Self, Int4ModelError> {
        Int4ModelLoadRequirement::for_group_width(group_width)?.check(budget)?;
        let (census, _, _) = source.checked_plan_parts().map_err(|_| Int4ModelError::Contract)?;
        validate_nanbeige42_census(census).map_err(|_| Int4ModelError::Contract)?;
        checkpoint(control)?;
        source.revalidate_retained_shards().map_err(|_| Int4ModelError::Source)?;
        checkpoint(control)?;
        let embeddings = Bf16Matrix::new(V, H, read_bf16(source, "model.embed_tokens.weight", V, H, false, control)?)
            .map_err(|_| Int4ModelError::Contract)?;
        let final_norm = read_bf16(source, "model.norm.weight", 1, H, true, control)?;
        let head = read_linear(source, "lm_head.weight", V, H, group_width, control)?;
        let mut layers = Vec::new();
        layers.try_reserve_exact(PHYSICAL_LAYER_COUNT).map_err(|_| Int4ModelError::Allocation)?;
        for index in 0..PHYSICAL_LAYER_COUNT {
            let prefix = format!("model.layers.{index}");
            layers.push(Int4Layer {
                norm1: read_bf16(source, &format!("{prefix}.input_layernorm.weight"), 1, H, true, control)?,
                norm2: read_bf16(source, &format!("{prefix}.post_attention_layernorm.weight"), 1, H, true, control)?,
                q: read_linear(source, &format!("{prefix}.self_attn.q_proj.weight"), Q, H, group_width, control)?,
                k: read_linear(source, &format!("{prefix}.self_attn.k_proj.weight"), K, H, group_width, control)?,
                v: read_linear(source, &format!("{prefix}.self_attn.v_proj.weight"), K, H, group_width, control)?,
                o: read_linear(source, &format!("{prefix}.self_attn.o_proj.weight"), H, Q, group_width, control)?,
                gate: read_linear(source, &format!("{prefix}.mlp.gate_proj.weight"), I, H, group_width, control)?,
                up: read_linear(source, &format!("{prefix}.mlp.up_proj.weight"), I, H, group_width, control)?,
                down: read_linear(source, &format!("{prefix}.mlp.down_proj.weight"), H, I, group_width, control)?,
            });
        }
        let mut model = Self { embeddings, final_norm, head,
            layers: layers.try_into().map_err(|_| Int4ModelError::Contract)?,
            identity: Int4ModelIdentity { source_root_sha256: source.source_root_sha256().to_owned(),
                content_sha256: [0; 32], group_width } };
        model.identity.content_sha256 = model.digest(control)?;
        checkpoint(control)?;
        source.revalidate_retained_shards().map_err(|_| Int4ModelError::Source)?;
        checkpoint(control)?;
        Ok(model)
    }
    pub fn identity(&self) -> &Int4ModelIdentity { &self.identity }
    pub fn group_width(&self) -> usize { self.identity.group_width }

    fn digest<C: DecodeStepControl>(&self, control: &mut C) -> Result<[u8; 32], Int4ModelError> {
        let mut hash = identity_prefix(&self.identity.source_root_sha256, self.group_width());
        frame(&mut hash, b"embed");
        for row in 0..V {
            checkpoint(control)?;
            hash_bf16(&mut hash, self.embeddings.row(row).map_err(|_| Int4ModelError::Contract)?);
        }
        frame(&mut hash, b"final_norm"); hash_bf16(&mut hash, &self.final_norm);
        frame(&mut hash, b"lm_head"); hash_matrix(&mut hash, &self.head, control)?;
        for (index, layer) in self.layers.iter().enumerate() {
            checkpoint(control)?;
            frame(&mut hash, b"layer"); hash.update((index as u64).to_le_bytes());
            frame(&mut hash, b"norm1"); hash_bf16(&mut hash, &layer.norm1);
            frame(&mut hash, b"norm2"); hash_bf16(&mut hash, &layer.norm2);
            for (name, matrix) in [("q", &layer.q), ("k", &layer.k), ("v", &layer.v), ("o", &layer.o),
                ("gate", &layer.gate), ("up", &layer.up), ("down", &layer.down)] {
                frame(&mut hash, name.as_bytes()); hash_matrix(&mut hash, matrix, control)?;
            }
        }
        checkpoint(control)?;
        Ok(hash.finalize().into())
    }
}
fn identity_prefix(source_root: &str, width: usize) -> Sha256 {
    let mut hash = Sha256::new();
    for field in ["fnlp-candidate-int4-model-content-v1", "Nanbeige4.2-3B", PINNED_REVISION,
        STRICT_INT4_PROFILE, GROUPED_INT4_RECIPE, source_root] { frame(&mut hash, field.as_bytes()); }
    for value in [width, V, H, Q, K, I, PHYSICAL_LAYER_COUNT] { hash.update((value as u64).to_le_bytes()); }
    hash
}
fn frame(hash: &mut Sha256, bytes: &[u8]) { hash.update((bytes.len() as u64).to_le_bytes()); hash.update(bytes); }
fn hash_bf16(hash: &mut Sha256, values: &[Bf16]) {
    hash.update((values.len() as u64).to_le_bytes());
    for value in values { hash.update(value.to_bits().to_le_bytes()); }
}
fn hash_matrix<C: DecodeStepControl>(hash: &mut Sha256, matrix: &GroupedInt4Matrix,
    control: &mut C) -> Result<(), Int4ModelError> {
    // Bound cancellation latency during the content pass too. In particular,
    // the full lm-head must not be one uninterruptible hash operation.
    for value in [matrix.rows(), matrix.columns(), matrix.group_width(),
        matrix.packed_bytes().len(), matrix.group_scales().len()] {
        hash.update((value as u64).to_le_bytes());
    }
    for bytes in matrix.packed_bytes().chunks(64 * 1024) {
        checkpoint(control)?; hash.update(bytes);
    }
    for scales in matrix.group_scales().chunks(1024) {
        checkpoint(control)?;
        for scale in scales { hash.update(scale.to_bits().to_le_bytes()); }
    }
    Ok(())
}
fn read_linear<C: DecodeStepControl>(source: &PreparedConversionInput, name: &str, rows: usize,
    columns: usize, width: usize, control: &mut C) -> Result<GroupedInt4Matrix, Int4ModelError> {
    let need = Int4ConversionRequirement::for_shape(rows, columns, width)?;
    let budget = Int4ConversionBudget { max_payload_bytes: need.peak_payload_bytes().ok_or(Int4ModelError::Budget)?,
        max_source_values: need.source_values, max_groups: need.groups };
    GroupedInt4Matrix::from_bf16_rows(rows, columns, width, budget, control, |row, output| {
        let bytes = source.read_verified_panel(name, RowPanel::Rows { start_row: row as u64, row_count: 1 })
            .map_err(|_| Int4ModelError::Source)?;
        decode_row(&bytes, output)
    }).map_err(|error| match error {
        Int4ConversionError::Contract(error) => Int4ModelError::Quantization(error),
        Int4ConversionError::Budget => Int4ModelError::Budget,
        Int4ConversionError::Allocation => Int4ModelError::Allocation,
        Int4ConversionError::NonFiniteSource | Int4ConversionError::ScaleUnderflow => Int4ModelError::NonFinite,
        Int4ConversionError::Source(error) => error,
        Int4ConversionError::Cancelled(cause) => Int4ModelError::Cancelled(cause),
    })
}
fn read_bf16<C: DecodeStepControl>(source: &PreparedConversionInput, name: &str, rows: usize,
    columns: usize, vector: bool, control: &mut C) -> Result<Vec<Bf16>, Int4ModelError> {
    let length = rows.checked_mul(columns).ok_or(Int4ModelError::Contract)?;
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(|_| Int4ModelError::Allocation)?;
    values.resize(length, Bf16::from_bits(0));
    for row in 0..rows {
        checkpoint(control)?;
        let panel = if vector { RowPanel::WholeTensor }
            else { RowPanel::Rows { start_row: row as u64, row_count: 1 } };
        let bytes = source.read_verified_panel(name, panel).map_err(|_| Int4ModelError::Source)?;
        checkpoint(control)?;
        decode_row(&bytes, &mut values[row * columns..(row + 1) * columns])?;
    }
    Ok(values)
}
fn decode_row(bytes: &[u8], output: &mut [Bf16]) -> Result<(), Int4ModelError> {
    if output.len().checked_mul(2) != Some(bytes.len()) { return Err(Int4ModelError::Contract); }
    for (out, pair) in output.iter_mut().zip(bytes.chunks_exact(2)) {
        let value = Bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]]));
        if !value.to_f32().is_finite() { return Err(Int4ModelError::NonFinite); }
        *out = value;
    }
    Ok(())
}
pub(super) fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), Int4ModelError> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(Int4ModelError::Cancelled(cause)), None => Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_census_work_and_memory_are_admitted_together() {
        let need = Int4ModelLoadRequirement::for_group_width(128).unwrap();
        assert_eq!(need.source_values * 2, crate::artifact::converter::PINNED_LOGICAL_PAYLOAD_BYTES);
        assert_eq!(need.scratch_payload_bytes, 43_008);
        let budget = Int4ModelLoadBudget { max_payload_bytes: need.peak_payload_bytes().unwrap(),
            max_source_values: need.source_values, max_groups: need.groups,
            max_shard_revalidation_bytes: need.shard_revalidation_bytes };
        assert!(need.check(budget).is_ok());
        for axis in 0..4 {
            let mut bad = budget;
            match axis { 0 => bad.max_payload_bytes -= 1, 1 => bad.max_source_values -= 1,
                2 => bad.max_groups -= 1, _ => bad.max_shard_revalidation_bytes -= 1 }
            assert!(matches!(need.check(bad), Err(Int4ModelError::Budget)));
        }
        for width in [0, MAX_MODEL_K + 1, usize::MAX] { assert!(Int4ModelLoadRequirement::for_group_width(width).is_err()); }
    }
    #[test]
    fn group_tail_accounting_covers_every_supported_partition() {
        let source_values = Int4ModelLoadRequirement::for_group_width(1).unwrap().source_values;
        for width in 1..=MAX_MODEL_K {
            let need = Int4ModelLoadRequirement::for_group_width(width).unwrap();
            let groups: u64 = PROJECTION_SHAPES.iter().map(|&(n, k)| (n * k.div_ceil(width)) as u64).sum::<u64>()
                * PHYSICAL_LAYER_COUNT as u64 + (V * H.div_ceil(width)) as u64;
            assert_eq!(need.groups, groups); assert_eq!(need.source_values, source_values);
        }
    }
    #[test]
    fn row_decode_rejects_truncation_extra_bytes_and_every_nonfinite_encoding() {
        for bits in 0_u32..=u16::MAX as u32 {
            let mut output = [Bf16::from_bits(0)];
            let result = decode_row(&(bits as u16).to_le_bytes(), &mut output);
            assert_eq!(result.is_ok(), f32::from_bits(bits << 16).is_finite());
            if result.is_ok() { assert_eq!(output[0].to_bits(), bits as u16); }
        }
        for bytes in [&[][..], &[1][..], &[1, 2, 3][..], &[1, 2, 3, 4][..]] {
            assert!(decode_row(bytes, &mut [Bf16::from_bits(0)]).is_err());
        }
    }
    #[test]
    fn identity_binds_source_partition_roles_and_exact_bf16_bits() {
        let digest = |root: &str, width, role: &[u8], bits| {
            let mut hash = identity_prefix(root, width); frame(&mut hash, role);
            hash_bf16(&mut hash, &[Bf16::from_bits(bits)]); <[u8; 32]>::from(hash.finalize())
        };
        let base = digest("source-a", 128, b"q", 0);
        assert_ne!(base, digest("source-b", 128, b"q", 0));
        assert_ne!(base, digest("source-a", 127, b"q", 0));
        assert_ne!(base, digest("source-a", 128, b"k", 0));
        assert_ne!(base, digest("source-a", 128, b"q", 0x8000));
    }
}

/// Digest-pinned local candidate snapshots; never canonical FNLPQ activation.
pub mod snapshot;
