//! Split-half rotary-position embedding for `hf-bf16-eager`.

use super::tensor::Bf16;

/// Nanbeige's explicit head width. It is never derived from hidden/query heads.
pub const NANBEIGE_HEAD_DIM: usize = 128;
/// Nanbeige's pinned rotary base.
pub const NANBEIGE_ROPE_THETA: f32 = 70_000_000.0;

/// A shape or position error from the f32 RoPE table/application path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RopeError {
    /// Split-half rotation requires an even head dimension.
    OddHeadDimension { head_dim: usize },
    /// The requested position was not precomputed into the f32 table.
    PositionOutOfRange {
        position: usize,
        table_positions: usize,
    },
    /// Q or K did not have the table's exact head dimension.
    ActivationLength { expected: usize, actual: usize },
}

/// f32 cosine/sine tables, narrowed only when applied to bf16 Q/K vectors.
#[derive(Clone, Debug)]
pub struct RopeTablesF32 {
    head_dim: usize,
    positions: usize,
    cosine: Vec<f32>,
    sine: Vec<f32>,
}

impl RopeTablesF32 {
    /// Builds f32 tables for an explicit even head dimension and θ.
    pub fn new(positions: usize, head_dim: usize, theta: f32) -> Result<Self, RopeError> {
        if head_dim % 2 != 0 {
            return Err(RopeError::OddHeadDimension { head_dim });
        }
        let half_dim = head_dim / 2;
        let mut cosine = Vec::with_capacity(positions * half_dim);
        let mut sine = Vec::with_capacity(positions * half_dim);
        for position in 0..positions {
            for pair in 0..half_dim {
                let inverse_frequency = theta.powf(-(2.0 * pair as f32) / head_dim as f32);
                let phase = position as f32 * inverse_frequency;
                cosine.push(phase.cos());
                sine.push(phase.sin());
            }
        }
        Ok(Self {
            head_dim,
            positions,
            cosine,
            sine,
        })
    }

    /// Builds the model's 128-dimensional θ=7e7 f32 table.
    pub fn nanbeige(positions: usize) -> Result<Self, RopeError> {
        Self::new(positions, NANBEIGE_HEAD_DIM, NANBEIGE_ROPE_THETA)
    }

    /// Applies split-half RoPE in f32 then casts the Q/K activation back to bf16.
    pub fn apply_split_half(
        &self,
        position: usize,
        activation: &mut [Bf16],
    ) -> Result<(), RopeError> {
        if activation.len() != self.head_dim {
            return Err(RopeError::ActivationLength {
                expected: self.head_dim,
                actual: activation.len(),
            });
        }
        if position >= self.positions {
            return Err(RopeError::PositionOutOfRange {
                position,
                table_positions: self.positions,
            });
        }
        let half_dim = self.head_dim / 2;
        let offset = position * half_dim;
        for pair in 0..half_dim {
            let left = activation[pair].to_f32();
            let right = activation[pair + half_dim].to_f32();
            let cosine = self.cosine[offset + pair];
            let sine = self.sine[offset + pair];
            activation[pair] = Bf16::from_f32(left * cosine - right * sine);
            activation[pair + half_dim] = Bf16::from_f32(right * cosine + left * sine);
        }
        Ok(())
    }
}
