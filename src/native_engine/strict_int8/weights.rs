//! Zero-copy binding of the existing materialized bridge's complete census.
//! The bridge is still synthetic/non-authoritative. This module adds no source
//! reader, envelope interpretation, activation capability or dequantized copy.

use super::*;
use crate::native_engine::{artifact_bridge::{ArtifactBf16Tensor, ArtifactIdentity, ArtifactWeightSet}, weights::Bf16Matrix};

pub(super) struct Int8Layer<'a> {
    pub norm1: &'a [Bf16], pub norm2: &'a [Bf16],
    pub q: QuantizedLinear<'a>, pub k: QuantizedLinear<'a>, pub v: QuantizedLinear<'a>,
    pub o: QuantizedLinear<'a>, pub gate: QuantizedLinear<'a>, pub up: QuantizedLinear<'a>, pub down: QuantizedLinear<'a>,
}

/// A shape-checked, borrowed full-model view. Not Clone, Deserialize or Debug:
/// a materialized tensor set is not an artifact-admission certificate.
pub struct Int8WeightView<'a> {
    pub(super) identity: &'a ArtifactIdentity,
    pub(super) embeddings: &'a Bf16Matrix,
    pub(super) final_norm: &'a [Bf16],
    pub(super) layers: [Int8Layer<'a>; PHYSICAL_LAYER_COUNT],
    pub(super) head: QuantizedLinear<'a>,
}
impl<'a> Int8WeightView<'a> {
    /// Bind all 201 existing bridge tensors without cloning their payloads.
    /// Missing, extra, wrongly typed/shaped tensors and invalid int8 sidecars
    /// refuse. The source still has ONLY its original synthetic evidence grade.
    pub fn from_materialized(source: &'a ArtifactWeightSet) -> Result<Self, StrictInt8Error> {
        if source.identity().model_id != "Nanbeige4.2-3B"
            || source.bf16().len() != 2 * PHYSICAL_LAYER_COUNT + 2
            || source.quantized().len() != 7 * PHYSICAL_LAYER_COUNT + 1 {
            return Err(StrictInt8Error::Weights);
        }
        let Some(ArtifactBf16Tensor::Matrix(embeddings)) = source.bf16().get("embed") else {
            return Err(StrictInt8Error::Weights);
        };
        geometry(embeddings.rows(), embeddings.columns(), V, H)?;
        let final_norm = norm(source, "final_norm")?;
        let head = linear(source, "lm_head", V, H)?;
        let mut layers = Vec::new();
        layers.try_reserve_exact(PHYSICAL_LAYER_COUNT).map_err(|_| StrictInt8Error::Allocation)?;
        for index in 0..PHYSICAL_LAYER_COUNT {
            let base = format!("layer.{index}");
            layers.push(Int8Layer {
                norm1: norm(source, &format!("{base}.norm1"))?, norm2: norm(source, &format!("{base}.norm2"))?,
                q: linear(source, &format!("{base}.attn.q"), Q, H)?,
                k: linear(source, &format!("{base}.attn.k"), K, H)?,
                v: linear(source, &format!("{base}.attn.v"), K, H)?,
                o: linear(source, &format!("{base}.attn.o"), H, Q)?,
                gate: linear(source, &format!("{base}.mlp.gate"), I, H)?,
                up: linear(source, &format!("{base}.mlp.up"), I, H)?,
                down: linear(source, &format!("{base}.mlp.down"), H, I)?,
            });
        }
        Ok(Self { identity: source.identity(), embeddings, final_norm, head,
            layers: layers.try_into().map_err(|_| StrictInt8Error::Weights)? })
    }
    pub fn artifact_identity(&self) -> &ArtifactIdentity { self.identity }
}
fn norm<'a>(source: &'a ArtifactWeightSet, name: &str) -> Result<&'a [Bf16], StrictInt8Error> {
    let Some(ArtifactBf16Tensor::Vector(values)) = source.bf16().get(name) else {
        return Err(StrictInt8Error::Weights);
    };
    if values.len() != H || values.iter().any(|value| !value.to_f32().is_finite()) {
        return Err(StrictInt8Error::Weights);
    }
    Ok(values)
}
fn linear<'a>(source: &'a ArtifactWeightSet, name: &str, rows: usize, columns: usize)
    -> Result<QuantizedLinear<'a>, StrictInt8Error> {
    let matrix = source.quantized().get(name).ok_or(StrictInt8Error::Weights)?;
    geometry(matrix.rows(), matrix.columns(), rows, columns)?;
    QuantizedLinear::from_portable(matrix).map_err(StrictInt8Error::from)
}
pub(super) fn geometry(rows: usize, columns: usize, expected_rows: usize, expected_columns: usize)
    -> Result<(), StrictInt8Error> {
    if rows == expected_rows && columns == expected_columns { Ok(()) } else { Err(StrictInt8Error::Weights) }
}
