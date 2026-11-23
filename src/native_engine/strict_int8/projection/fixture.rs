//! Tiny synthetic projection/protocol ownership, never a decoder model or
//! pinned-source oracle. Only tests bypass the real 201-tensor census here.
use super::*;
use crate::native_engine::{strict_int8::{weights::Int8Layer, PHYSICAL_LAYER_COUNT}, weights::Bf16Matrix};

struct Matrix { rows: usize, values: Vec<i8>, scales: Vec<f32>, sums: Vec<i32> }
pub(crate) struct ProjectionFixture {
    identity: ArtifactIdentity, embeddings: Bf16Matrix, norm: Vec<Bf16>, matrices: Vec<Matrix>,
}
impl ProjectionFixture {
    pub(crate) fn new(seed: usize) -> Self {
        let matrices = (0..=PHYSICAL_LAYER_COUNT * 7).map(|index| {
            let rows = if index == PHYSICAL_LAYER_COUNT * 7 { 10 } else { 6 };
            let values: Vec<_> = (0..rows * 6).map(|i| ((i * 53 + index * 7 + seed) % 256) as u8 as i8).collect();
            let scales = (0..rows).map(|row| (row + index % 7 + 1) as f32 / 64.0).collect();
            let sums = values.chunks_exact(6).map(|row| row.iter().map(|&v| i32::from(v)).sum()).collect();
            Matrix { rows, values, scales, sums }
        }).collect();
        Self { identity: ArtifactIdentity { model_id: "synthetic-int8-projection-fixture".to_owned(),
            revision: "not-a-decoder-model".to_owned(), recipe_id: "test-only".to_owned(),
            source_root_sha256: format!("{seed:064x}"), logical_model_sha256: format!("{seed:064x}") },
            embeddings: Bf16Matrix::new(10, 6, vec![Bf16::from_f32(1.0); 60]).unwrap(),
            norm: vec![Bf16::from_f32(1.0); 6], matrices }
    }
    pub(crate) fn weights(&self) -> Int8WeightView<'_> {
        let matrix = |index: usize| {
            let m = &self.matrices[index];
            QuantizedLinear::checked(m.rows, 6, &m.values, &m.scales, &m.sums).unwrap()
        };
        Int8WeightView { identity: &self.identity, embeddings: &self.embeddings, final_norm: &self.norm,
            layers: std::array::from_fn(|layer| Int8Layer { norm1: &self.norm, norm2: &self.norm,
                q: matrix(layer * 7), k: matrix(layer * 7 + 1), v: matrix(layer * 7 + 2),
                o: matrix(layer * 7 + 3), gate: matrix(layer * 7 + 4), up: matrix(layer * 7 + 5),
                down: matrix(layer * 7 + 6) }), head: matrix(PHYSICAL_LAYER_COUNT * 7) }
    }
    pub(crate) fn input(&self) -> ActivationBuffer {
        let mut input = ActivationBuffer::try_new(6).unwrap();
        input.encode_f32(&[-3.0, 127.0, -12.0, 0.0, 2.5, -55.0]).unwrap(); input
    }
}
