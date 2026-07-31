//! Native tensor metadata and storage.

/// A raw IEEE-754 bfloat16 value.
///
/// FrankenNLP keeps this representation local rather than adding `half`: the
/// `hf-bf16-eager` profile must make each widening and narrowing site explicit.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct Bf16(u16);

impl Bf16 {
    /// Creates a value from its stored IEEE-754 bfloat16 bits.
    #[must_use]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Returns the stored IEEE-754 bfloat16 bits.
    #[must_use]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Widens a bfloat16 value exactly into f32.
    #[must_use]
    pub const fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    /// Rounds f32 to bfloat16 using round-to-nearest, ties-to-even.
    #[must_use]
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let round_bias = 0x7fff + ((bits >> 16) & 1);
        Self((bits.wrapping_add(round_bias) >> 16) as u16)
    }
}

impl From<f32> for Bf16 {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl From<Bf16> for f32 {
    fn from(value: Bf16) -> Self {
        value.to_f32()
    }
}

/// Explicitly widens an activation/vector at an audited cast site.
#[must_use]
pub fn widen_bf16_to_f32(values: &[Bf16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

/// Explicitly narrows an activation/vector at an audited cast site.
#[must_use]
pub fn cast_f32_to_bf16(values: &[f32]) -> Vec<Bf16> {
    values.iter().copied().map(Bf16::from_f32).collect()
}
