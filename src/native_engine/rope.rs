//! Admitted-cap, split-half rotary-position embedding.
//!
//! Nanbeige rotates the first and second halves of each 128-wide head as
//! `(x_j, x_{j + 64})`.  This is deliberately not adjacent-pair RoPE.  Tables
//! are f32 authority values; the active numerics profile chooses the explicit
//! application cast (`f32` stays f32, `hf-bf16-eager` narrows each result).

use super::tensor::Bf16;

/// Nanbeige's explicit head width. It is never derived from hidden/query heads.
pub const NANBEIGE_HEAD_DIM: usize = 128;
/// Nanbeige's pinned rotary base.
pub const NANBEIGE_ROPE_THETA: f32 = 70_000_000.0;
/// The default per-sequence context admission cap. This is not the observed
/// 262,144-position model limit and is the only default table allocation.
pub const DEFAULT_ADMITTED_CONTEXT_CAP: usize = 8_192;

/// The RoPE projection boundary selected for one measured dispatch row.
///
/// Fused epilogues are present for proof and benchmarking only until a
/// profile-scoped ledger measurement explicitly selects one. The default is
/// therefore the materialized, unfused route.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RopeProjectionVariant {
    /// Materialize projected Q/K, then rotate them before KV append/scoring.
    #[default]
    Unfused,
    /// Rotate Q/K at the projection epilogue before the caller writes KV.
    FusedEpilogue,
}

/// A shape, configuration, allocation, or position error from RoPE.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RopeError {
    /// There is no valid position row for an empty admitted context.
    ZeroAdmittedContext,
    /// Split-half rotation requires an even, nonzero head dimension.
    OddHeadDimension { head_dim: usize },
    /// θ must be finite and strictly positive; bits keep this error `Eq`.
    InvalidTheta { theta_bits: u32 },
    /// The requested f32 table would overflow a `usize` element count.
    TableSizeOverflow { positions: usize, half_dim: usize },
    /// Capacity reservation for a table or inverse-frequency vector failed.
    TableAllocationRefused { positions: usize, head_dim: usize },
    /// The requested position was not precomputed into the admitted-cap table.
    PositionOutOfRange {
        position: usize,
        table_positions: usize,
    },
    /// A frequency lane was not present in the half-width table.
    FrequencyLaneOutOfRange { lane: usize, half_dim: usize },
    /// Q or K did not have the table's exact one-head dimension.
    ActivationLength { expected: usize, actual: usize },
    /// A packed multi-head Q/K projection was not a whole number of heads.
    HeadAlignedActivationLength { head_dim: usize, actual: usize },
}

/// f32 inverse-frequency/cosine/sine tables bounded exactly by one admitted
/// context cap. Re-admission to a larger cap constructs a replacement table;
/// there is no hot-path growth API.
#[derive(Clone, Debug)]
pub struct RopeTablesF32 {
    head_dim: usize,
    positions: usize,
    inverse_frequencies: Vec<f32>,
    cosine: Vec<f32>,
    sine: Vec<f32>,
}

impl RopeTablesF32 {
    /// Builds f32 tables for one explicit admitted context cap, head dimension,
    /// and θ. Allocation and multiplication are checked before reserving.
    pub fn new(positions: usize, head_dim: usize, theta: f32) -> Result<Self, RopeError> {
        if positions == 0 {
            return Err(RopeError::ZeroAdmittedContext);
        }
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(RopeError::OddHeadDimension { head_dim });
        }
        if !theta.is_finite() || theta <= 0.0 {
            return Err(RopeError::InvalidTheta {
                theta_bits: theta.to_bits(),
            });
        }

        let half_dim = head_dim / 2;
        let table_elements =
            positions
                .checked_mul(half_dim)
                .ok_or(RopeError::TableSizeOverflow {
                    positions,
                    half_dim,
                })?;
        let mut inverse_frequencies = Vec::new();
        let mut cosine = Vec::new();
        let mut sine = Vec::new();
        inverse_frequencies
            .try_reserve_exact(half_dim)
            .and_then(|_| cosine.try_reserve_exact(table_elements))
            .and_then(|_| sine.try_reserve_exact(table_elements))
            .map_err(|_| RopeError::TableAllocationRefused {
                positions,
                head_dim,
            })?;

        for lane in 0..half_dim {
            inverse_frequencies.push(theta.powf(-(2.0 * lane as f32) / head_dim as f32));
        }
        for position in 0..positions {
            for &inverse_frequency in &inverse_frequencies {
                let phase = position as f32 * inverse_frequency;
                cosine.push(phase.cos());
                sine.push(phase.sin());
            }
        }
        Ok(Self {
            head_dim,
            positions,
            inverse_frequencies,
            cosine,
            sine,
        })
    }

    /// Names the admission boundary at the construction site.
    pub fn for_admitted_context(
        admitted_context_cap: usize,
        head_dim: usize,
        theta: f32,
    ) -> Result<Self, RopeError> {
        Self::new(admitted_context_cap, head_dim, theta)
    }

    /// Builds Nanbeige's 128-dimensional θ=7e7 table for the supplied cap.
    pub fn nanbeige(admitted_context_cap: usize) -> Result<Self, RopeError> {
        Self::for_admitted_context(admitted_context_cap, NANBEIGE_HEAD_DIM, NANBEIGE_ROPE_THETA)
    }

    /// Builds the default 8192-position admitted table, not a 262K table.
    pub fn nanbeige_default_admission() -> Result<Self, RopeError> {
        Self::nanbeige(DEFAULT_ADMITTED_CONTEXT_CAP)
    }

    /// The explicit per-head width bound into this table.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// The exact admitted position count represented by this allocation.
    #[must_use]
    pub const fn position_count(&self) -> usize {
        self.positions
    }

    /// Returns the precomputed θ^(-2j/head_dim) value for a split-half lane.
    pub fn inverse_frequency(&self, lane: usize) -> Result<f32, RopeError> {
        self.inverse_frequencies
            .get(lane)
            .copied()
            .ok_or(RopeError::FrequencyLaneOutOfRange {
                lane,
                half_dim: self.half_dim(),
            })
    }

    /// Returns one f32 `(cos, sin)` table lane for direct fixture comparison.
    pub fn table_value(&self, position: usize, lane: usize) -> Result<(f32, f32), RopeError> {
        let offset = self.table_offset(position, lane)?;
        Ok((self.cosine[offset], self.sine[offset]))
    }

    /// Applies one f32 table row to a single f32 Q or K head.
    pub fn apply_split_half_f32(
        &self,
        position: usize,
        activation: &mut [f32],
    ) -> Result<(), RopeError> {
        self.validate_one_head(activation.len())?;
        let offset = self.table_offset(position, 0)?;
        rotate_split_half_f32(
            activation,
            &self.cosine[offset..offset + self.half_dim()],
            &self.sine[offset..offset + self.half_dim()],
        );
        Ok(())
    }

    /// Applies a table row to every contiguous f32 head in a projected Q or K
    /// tensor.  This allocation-free path is used after projection and before
    /// KV append/attention scoring.
    pub fn apply_split_half_f32_all_heads(
        &self,
        position: usize,
        activation: &mut [f32],
    ) -> Result<(), RopeError> {
        self.validate_all_heads(activation.len())?;
        let offset = self.table_offset(position, 0)?;
        let cosine = &self.cosine[offset..offset + self.half_dim()];
        let sine = &self.sine[offset..offset + self.half_dim()];
        for head in activation.chunks_exact_mut(self.head_dim) {
            rotate_split_half_f32(head, cosine, sine);
        }
        Ok(())
    }

    /// Applies one f32 table row to a single bf16 Q or K head with the
    /// `hf-bf16-eager` RoPE cast graph.
    ///
    /// The remote eager path first casts the f32 table row to the Q/K bf16
    /// dtype, rounds each bf16 product, then rounds their sum.  Keeping either
    /// the table or intermediate products widened would be a different
    /// numeric profile even when its final output often agrees.
    pub fn apply_split_half(
        &self,
        position: usize,
        activation: &mut [Bf16],
    ) -> Result<(), RopeError> {
        self.validate_one_head(activation.len())?;
        let offset = self.table_offset(position, 0)?;
        let half_dim = self.half_dim();
        for lane in 0..half_dim {
            let left = activation[lane].to_f32();
            let right = activation[lane + half_dim].to_f32();
            let cosine = Bf16::from_f32(self.cosine[offset + lane]).to_f32();
            let sine = Bf16::from_f32(self.sine[offset + lane]).to_f32();
            let left_cosine = Bf16::from_f32(left * cosine).to_f32();
            let negated_right_sine = Bf16::from_f32((-right) * sine).to_f32();
            let right_cosine = Bf16::from_f32(right * cosine).to_f32();
            let left_sine = Bf16::from_f32(left * sine).to_f32();
            activation[lane] = Bf16::from_f32(left_cosine + negated_right_sine);
            activation[lane + half_dim] = Bf16::from_f32(right_cosine + left_sine);
        }
        Ok(())
    }

    /// The unfused projection boundary: projected Q and K are materialized,
    /// then rotated before the caller writes K/V or computes attention.
    pub fn apply_projected_qk_unfused(
        &self,
        position: usize,
        query: &mut [f32],
        key: &mut [f32],
    ) -> Result<(), RopeError> {
        self.apply_split_half_f32_all_heads(position, query)?;
        self.apply_split_half_f32_all_heads(position, key)
    }

    /// Applies the selected projection boundary. Dispatch rows must pass an
    /// explicit variant; callers using [`RopeProjectionVariant::default`] keep
    /// the fused candidate disabled.
    pub fn apply_projected_qk(
        &self,
        variant: RopeProjectionVariant,
        position: usize,
        query: &mut [f32],
        key: &mut [f32],
    ) -> Result<(), RopeError> {
        match variant {
            RopeProjectionVariant::Unfused => self.apply_projected_qk_unfused(position, query, key),
            RopeProjectionVariant::FusedEpilogue => {
                self.apply_projected_qk_fused_epilogue(position, query, key)
            }
        }
    }

    /// The fusion candidate's projection-epilogue boundary. Callers invoke it
    /// immediately after each Q/K GEMV writes its destination registers and
    /// before any KV-store write. It performs no allocation and preserves the
    /// same lane and operation order as [`Self::apply_projected_qk_unfused`].
    /// Selection remains default-off until a measured ledger row promotes it.
    pub fn apply_projected_qk_fused_epilogue(
        &self,
        position: usize,
        query: &mut [f32],
        key: &mut [f32],
    ) -> Result<(), RopeError> {
        self.validate_all_heads(query.len())?;
        self.validate_all_heads(key.len())?;
        let offset = self.table_offset(position, 0)?;
        let cosine = &self.cosine[offset..offset + self.half_dim()];
        let sine = &self.sine[offset..offset + self.half_dim()];
        for head in query.chunks_exact_mut(self.head_dim) {
            rotate_split_half_f32(head, cosine, sine);
        }
        for head in key.chunks_exact_mut(self.head_dim) {
            rotate_split_half_f32(head, cosine, sine);
        }
        Ok(())
    }

    fn half_dim(&self) -> usize {
        self.head_dim / 2
    }

    fn table_offset(&self, position: usize, lane: usize) -> Result<usize, RopeError> {
        if position >= self.positions {
            return Err(RopeError::PositionOutOfRange {
                position,
                table_positions: self.positions,
            });
        }
        if lane >= self.half_dim() {
            return Err(RopeError::FrequencyLaneOutOfRange {
                lane,
                half_dim: self.half_dim(),
            });
        }
        Ok(position * self.half_dim() + lane)
    }

    fn validate_one_head(&self, actual: usize) -> Result<(), RopeError> {
        if actual != self.head_dim {
            return Err(RopeError::ActivationLength {
                expected: self.head_dim,
                actual,
            });
        }
        Ok(())
    }

    fn validate_all_heads(&self, actual: usize) -> Result<(), RopeError> {
        if actual == 0 || actual % self.head_dim != 0 {
            return Err(RopeError::HeadAlignedActivationLength {
                head_dim: self.head_dim,
                actual,
            });
        }
        Ok(())
    }
}

fn rotate_split_half_f32(activation: &mut [f32], cosine: &[f32], sine: &[f32]) {
    let half_dim = activation.len() / 2;
    debug_assert_eq!(cosine.len(), half_dim);
    debug_assert_eq!(sine.len(), half_dim);
    for lane in 0..half_dim {
        let left = activation[lane];
        let right = activation[lane + half_dim];
        activation[lane] = left * cosine[lane] - right * sine[lane];
        activation[lane + half_dim] = right * cosine[lane] + left * sine[lane];
    }
}
