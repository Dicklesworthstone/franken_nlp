//! Reconstitute a producer-recipe matrix without cloning its compact payload.
use super::*;

impl GroupedInt4Matrix {
    /// The caller admits and owns both vectors before entering. Validation is
    /// bounded/checkpointed and does not allocate a dense expansion. This is
    /// crate-only: model restoration additionally checks the external content pin.
    pub(crate) fn restore_packed<C: DecodeStepControl>(rows: usize, columns: usize, group_width: usize,
        packed: Vec<u8>, scales: Vec<f32>, control: &mut C)
        -> Result<Self, Int4ConversionError<std::convert::Infallible>> {
        let need = Int4ConversionRequirement::for_shape(rows, columns, group_width)
            .map_err(Int4ConversionError::Contract)?;
        if packed.len() as u64 != need.packed_bytes || scales.len() as u64 != need.groups {
            return Err(Int4ConversionError::Contract(GroupedInt4Error::Geometry));
        }
        // The numerical leaf accepts -8, but this owned type names the
        // symmetric7 producer. Never restore a different recipe under its name.
        for chunk in packed.chunks(64 * 1024) {
            checkpoint(control)?;
            if chunk.iter().any(|&byte| byte & 15 == 8 || byte >> 4 == 8) {
                return Err(Int4ConversionError::Contract(GroupedInt4Error::Geometry));
            }
        }
        for chunk in scales.chunks(1024) {
            checkpoint(control)?;
            if chunk.iter().any(|&scale| !scale.is_finite() || scale <= 0.0) {
                return Err(Int4ConversionError::Contract(GroupedInt4Error::Scale));
            }
        }
        checkpoint(control)?;
        Ok(Self { rows, columns, group_width, packed, scales })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    }
    #[test]
    fn restoration_accepts_exactly_the_producer_nibble_domain() {
        for byte in 0..=u8::MAX {
            let result = GroupedInt4Matrix::restore_packed(1, 2, 2, vec![byte], vec![1.0], &mut Continue);
            assert_eq!(result.is_ok(), byte & 15 != 8 && byte >> 4 != 8);
        }
    }
    #[test]
    fn restored_odd_rows_keep_payload_scale_bits_and_matrix_identity() {
        let source = [-3.0, 7.0, 2.0, -0.0, 1.0, -4.0].map(Bf16::from_f32);
        for width in [1, 2, 3, 128] {
            let need = Int4ConversionRequirement::for_shape(2, 3, width).unwrap();
            let a = GroupedInt4Matrix::from_bf16(2, 3, width, &source, Int4ConversionBudget {
                max_payload_bytes: need.peak_payload_bytes().unwrap(), max_source_values: 6, max_groups: need.groups,
            }, &mut Continue).unwrap();
            let b = GroupedInt4Matrix::restore_packed(2, 3, width,
                a.packed_bytes().to_vec(), a.group_scales().to_vec(), &mut Continue).unwrap();
            assert_eq!(a.content_sha256(), b.content_sha256());
            assert_eq!(a.packed_bytes(), b.packed_bytes());
        }
    }
    #[test]
    fn restoration_refuses_shapes_scales_and_cancellation() {
        for scale in [0.0, -0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            assert!(GroupedInt4Matrix::restore_packed(1, 2, 2, vec![0x11], vec![scale], &mut Continue).is_err());
        }
        assert!(GroupedInt4Matrix::restore_packed(1, 2, 1, vec![0x11], vec![1.0], &mut Continue).is_err());
        assert!(GroupedInt4Matrix::restore_packed(1, 3, 1, vec![0x11], vec![1.0; 3], &mut Continue).is_err());
        struct Stop;
        impl DecodeStepControl for Stop {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
        }
        assert!(matches!(GroupedInt4Matrix::restore_packed(1, 2, 2, vec![0x11], vec![1.0], &mut Stop),
            Err(Int4ConversionError::Cancelled(DecodeCancellationKind::Deadline))));
    }
}
