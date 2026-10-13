//! Shared packed-INT4 routing regressions. Table timings are synthetic fixtures,
//! not measurements; an unsupported native target is never reported as passed.
use franken_nlp::native_engine::{
    dispatch::{DetectedFeatures, DispatchError, DispatchKey, DispatchRegime, DispatchRow,
        DispatchTable, Dispatcher, KernelOperation, KernelShape, KernelTier,
        MeasurementProvenance, SelectionProvenance, TileGeometry},
    int8,
    simd::{CpuKernelError, Int8Kernel, Int8KernelKind},
};

fn int8_tile(tier: KernelTier) -> TileGeometry {
    match tier {
        KernelTier::Scalar => TileGeometry::scalar(),
        KernelTier::X3aAvx2Low7HighBit | KernelTier::X3bAvx2WidenedI16 => TileGeometry { k: 32, m: 1, n: 1 },
        KernelTier::X2AvxVnni | KernelTier::X1bAvx512VnniYmm => TileGeometry { k: 32, m: 4, n: 1 },
        KernelTier::X1aAvx512VnniZmm => TileGeometry { k: 64, m: 4, n: 1 },
        KernelTier::A1Smmla => TileGeometry { k: 8, m: 4, n: 2 },
        KernelTier::A2Dotprod | KernelTier::A3Autovec => TileGeometry { k: 16, m: 4, n: 1 },
    }
}
fn fixture(tier: KernelTier, key: DispatchKey) -> DispatchRow {
    let native = int8_tile(tier);
    let tile = if key.operation == KernelOperation::Int4Gemv {
        TileGeometry { k: native.k, m: 1, n: 1 }
    } else { native };
    DispatchRow { key, selected_tier: tier, tile, selected_median_ns: 1,
        wider_tier_losses: Vec::new(), provenance: MeasurementProvenance {
            benchmark_id: "packed-int4-route-NOT-a-benchmark".to_owned(),
            host_class: "synthetic-test".to_owned(), recorded_on: "2026-10-13".to_owned() } }
}

pub fn verify(tiers: &[(KernelTier, Int8KernelKind)], features: DetectedFeatures) {
    for &(tier, kind) in tiers {
        assert_eq!(tier.supports_operation(KernelOperation::Int4Gemv), tier.is_implemented());
        let key = DispatchKey::new(KernelOperation::Int4Gemv, DispatchRegime::DecodeGemv,
            KernelShape { k: 4, m: 1, n: 1 });
        let row = fixture(tier, key);
        let expected_tile = row.tile;
        let dispatcher = Dispatcher::new(features, DispatchTable::from_rows(vec![row.clone()]).unwrap());
        let forced = dispatcher.int4_gemv(&[1, 2, 3, 4], &[0x78, 0xf0], 1,
            DispatchRegime::DecodeGemv, Some(tier));
        let selected = dispatcher.int4_gemv(&[1, 2, 3, 4], &[0x78, 0xf0], 1,
            DispatchRegime::DecodeGemv, None);
        match Int8Kernel::detect(kind) {
            Ok(_) => {
                for (measured, result) in [(false, forced), (true, selected)] {
                    let result = result.unwrap();
                    assert_eq!(result.output, [2]);
                    assert_eq!(result.selection.tier, tier);
                    assert_eq!(result.selection.tile, expected_tile);
                    assert_eq!(result.selection.kernel_id, tier.kernel_id(KernelOperation::Int4Gemv));
                    assert!(result.selection.kernel_id.ends_with("_int4_unpack_gemv_v1"));
                    assert_eq!(matches!(&result.selection.provenance, SelectionProvenance::Measured { .. }), measured);
                }
                // Exercise the actual packed executor across vector tails,
                // model reductions and rows that start in a high nibble.
                for k in [0_usize, 1, 7, 8, 31, 32, 33, 63, 64, 65, 3072, 6144, 10752] {
                    let n = if k % 2 == 0 { 3 } else { 2 };
                    let input: Vec<i8> = (0..k).map(|i| i.wrapping_mul(97) as u8 as i8).collect();
                    let packed: Vec<u8> = (0..n * k / 2).map(|i| i.wrapping_mul(173) as u8).collect();
                    let result = dispatcher.int4_gemv(&input, &packed, n,
                        DispatchRegime::DecodeGemv, Some(tier)).unwrap();
                    assert_eq!(result.output, int8::gemv_int4_s8(&input, &packed, n).unwrap());
                    assert_eq!(result.selection.tier, tier);
                }
                for (input, packed, n) in [(&[1_i8, 2][..], &[0_u8][..], 2_usize),
                    (&[1][..], &[0][..], 1), (&[1, 2][..], &[][..], usize::MAX)] {
                    assert_eq!(dispatcher.int4_gemv(input, packed, n, DispatchRegime::DecodeGemv,
                        Some(tier)).unwrap_err(), DispatchError::CpuKernel(CpuKernelError::Shape));
                }
                assert_eq!(dispatcher.int4_gemv(&vec![0; 10753], &[], 0, DispatchRegime::DecodeGemv,
                    Some(tier)).unwrap_err(), DispatchError::CpuKernel(CpuKernelError::ReductionTooLong));
                assert_eq!(dispatcher.int4_gemv(&[], &[], usize::MAX, DispatchRegime::DecodeGemv,
                    Some(tier)).unwrap_err(), DispatchError::AllocationRefused);
            }
            Err(CpuKernelError::UnsupportedIsa) if tier.is_implemented() => {
                for result in [forced, selected] {
                    assert_eq!(result.unwrap_err(), DispatchError::CpuKernel(CpuKernelError::UnsupportedIsa));
                }
            }
            Err(CpuKernelError::UnsupportedIsa) => {
                assert_eq!(forced.unwrap_err(), DispatchError::ForcedTierUnimplemented { requested: tier });
                let result = selected.unwrap();
                assert_eq!(result.output, [2]); assert_eq!(result.selection.tier, KernelTier::Scalar);
                assert!(matches!(result.selection.provenance, SelectionProvenance::ConservativeDefault { .. }));
            }
            Err(error) => panic!("unexpected native admission error: {error}"),
        }
        let scalar = dispatcher.int4_gemv(&[1, 2, 3, 4], &[0x78, 0xf0], 1,
            DispatchRegime::DecodeGemv, Some(KernelTier::Scalar)).unwrap();
        assert_eq!(scalar.output, [2]); assert_eq!(scalar.selection.tier, KernelTier::Scalar);
        let empty = Dispatcher::new(features, DispatchTable::default());
        assert_eq!(empty.select(key, None).unwrap().tier, KernelTier::Scalar);

        // Same input shape but an INT8 measurement is NOT packed-INT4 evidence.
        let int8_key = DispatchKey::new(KernelOperation::Int8Gemv, key.regime, key.shape);
        let unrelated = Dispatcher::new(features, DispatchTable::from_rows(vec![fixture(tier, int8_key)]).unwrap());
        let result = unrelated.int4_gemv(&[1, 2, 3, 4], &[0x78, 0xf0], 1,
            DispatchRegime::DecodeGemv, None).unwrap();
        assert_eq!(result.selection.tier, KernelTier::Scalar);
        assert!(matches!(result.selection.provenance, SelectionProvenance::ConservativeDefault { .. }));
        let mut wrong = row; wrong.tile.n = 4;
        assert!(matches!(DispatchTable::from_rows(vec![wrong]), Err(DispatchError::InvalidMeasurement { .. })));
    }
}
