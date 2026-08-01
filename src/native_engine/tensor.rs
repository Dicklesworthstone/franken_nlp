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
    ///
    /// NaNs use the canonical quiet bfloat16 NaN captured from the pinned
    /// PyTorch CPU eager environment (`0x7fc0`).  Rounding an f32 signaling
    /// NaN before this check can otherwise produce bfloat16 infinity when the
    /// payload lives only in the discarded low 16 bits.
    #[must_use]
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let exponent = bits & 0x7f80_0000;
        let mantissa = bits & 0x007f_ffff;
        if exponent == 0x7f80_0000 && mantissa != 0 {
            return Self::from_bits(0x7fc0);
        }
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

#[cfg(test)]
mod tests {
    use super::Bf16;

    #[test]
    fn from_f32_preserves_infinities() {
        assert_eq!(Bf16::from_f32(f32::INFINITY).to_bits(), 0x7f80);
        assert_eq!(Bf16::from_f32(f32::NEG_INFINITY).to_bits(), 0xff80);
    }

    #[test]
    fn from_f32_canonicalizes_signaling_and_quiet_nan_payloads() {
        // Code-first regression vectors captured with the pinned PyTorch 2.6
        // CPU eager environment.  They cover positive/negative signaling NaNs
        // and quiet NaNs whose payload would otherwise round to infinity.
        let cases = [
            0x7f80_0001, // positive signaling NaN
            0xff80_0001, // negative signaling NaN
            0x7fc1_2345, // positive quiet NaN payload
            0xffc1_2345, // negative quiet NaN payload
        ];

        for input in cases {
            let rounded = Bf16::from_f32(f32::from_bits(input));
            assert_eq!(rounded.to_bits(), 0x7fc0, "input=0x{input:08x}");
            assert!(rounded.to_f32().is_nan(), "input=0x{input:08x}");
        }
    }
}
