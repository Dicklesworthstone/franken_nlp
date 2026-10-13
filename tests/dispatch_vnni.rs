//! VNNI dispatch definitions: reported tier, actual body and capability boundary.
//! Every table here is a synthetic policy fixture, NOT performance evidence.
#[path = "support/packed_int4_dispatch.rs"]
mod packed_int4;

use franken_nlp::native_engine::{
    dispatch::{Architecture, DetectedFeatures, DispatchError, DispatchKey, DispatchRegime,
        DispatchRow, DispatchTable, Dispatcher, KernelOperation, KernelShape, KernelTier,
        MeasurementProvenance, SelectionProvenance, TileGeometry, host_backend_report},
    int8,
    simd::{CpuKernelError, Int8Kernel, Int8KernelKind},
};

const TIERS: [(KernelTier, Int8KernelKind); 3] = [
    (KernelTier::X2AvxVnni, Int8KernelKind::AvxVnni),
    (KernelTier::X1bAvx512VnniYmm, Int8KernelKind::Avx512VnniYmm),
    (KernelTier::X1aAvx512VnniZmm, Int8KernelKind::Avx512VnniZmm),
];
fn available() -> Vec<KernelTier> {
    TIERS.into_iter().filter_map(|(tier, kind)| match Int8Kernel::detect(kind) {
        Ok(_) => Some(tier),
        Err(CpuKernelError::UnsupportedIsa) => {
            eprintln!("VNNI_DISPATCH tier={tier} RESULT=SKIPPED_UNSUPPORTED_ISA"); None
        }
        Err(error) => panic!("unexpected real-host refusal: {error}"),
    }).collect()
}
fn tile(tier: KernelTier) -> TileGeometry {
    TileGeometry { m: 4, n: 1, k: if tier == KernelTier::X1aAvx512VnniZmm { 64 } else { 32 } }
}
fn features(mask: u8, architecture: Architecture) -> DetectedFeatures {
    DetectedFeatures { architecture, aarch64_dotprod: false, aarch64_i8mm: false,
        x86_avx2: mask & 1 != 0, x86_avx512f: mask & 2 != 0, x86_avx512vl: mask & 4 != 0,
        x86_avx512vnni: mask & 8 != 0, x86_avxvnni: mask & 16 != 0 }
}
fn row(tier: KernelTier, key: DispatchKey) -> DispatchRow {
    DispatchRow { key, selected_tier: tier, selected_median_ns: 1, tile: tile(tier),
        wider_tier_losses: Vec::new(), provenance: MeasurementProvenance {
            benchmark_id: "synthetic-route-NOT-a-benchmark".to_owned(),
            host_class: "synthetic-test".to_owned(), recorded_on: "2026-10-12".to_owned() } }
}
fn pattern(length: usize, salt: usize) -> Vec<i8> {
    (0..length).map(|index| ((index * 53 + salt * 71) % 256) as u8 as i8).collect()
}

#[test]
fn every_forced_vnni_width_executes_gemm_and_gemv_with_its_own_identity() {
    let dispatcher = Dispatcher::host(DispatchTable::default());
    for tier in available() {
        for regime in DispatchRegime::ALL {
            for m in [1_usize, 4, 5] {
                for n in [1_usize, 3] {
                    for k in [0, 1, 31, 32, 33, 63, 64, 65, 3072, 6144, 10752] {
                        let a = pattern(m * k, 1); let w = pattern(n * k, 3);
                        let result = dispatcher.int8_gemm(&a, m, k, &w, n, regime, Some(tier)).unwrap();
                        assert_eq!(result.output, int8::gemm_s8s8(&a, m, k, &w, n).unwrap());
                        assert_eq!(result.selection.tier, tier);
                        assert_eq!(result.selection.kernel_id, tier.kernel_id(KernelOperation::Int8Gemm));
                        assert!(!result.selection.kernel_id.contains("pending"));
                        assert_eq!(result.selection.tile, tile(tier));
                        assert!(matches!(&result.selection.provenance,
                            SelectionProvenance::ConservativeDefault { detail } if detail.contains("explicit forced")));
                        let vector = dispatcher.int8_gemv(&a[..k], &w, n, regime, Some(tier)).unwrap();
                        assert_eq!(vector.output, int8::gemv_s8s8(&a[..k], &w, n).unwrap());
                        assert_eq!(vector.selection.tier, tier);
                        assert_eq!(vector.selection.kernel_id, tier.kernel_id(KernelOperation::Int8Gemv));
                        assert_eq!(vector.selection.tile, tile(tier));
                    }
                }
            }
        }
    }
}

#[test]
fn supplied_table_routes_the_selected_body_but_never_overrides_forced_scalar() {
    for tier in available() {
        let key = DispatchKey::new(KernelOperation::Int8Gemm, DispatchRegime::PrefillGemm,
            KernelShape { k: 65, m: 5, n: 3 });
        let dispatcher = Dispatcher::host(DispatchTable::from_rows(vec![row(tier, key)]).unwrap());
        let a = pattern(325, 7); let w = pattern(195, 11);
        let native = dispatcher.int8_gemm(&a, 5, 65, &w, 3, DispatchRegime::PrefillGemm, None).unwrap();
        assert_eq!(native.selection.tier, tier); assert_eq!(native.selection.tile, tile(tier));
        assert!(matches!(&native.selection.provenance,
            SelectionProvenance::Measured { benchmark_id, .. } if benchmark_id.contains("NOT-a-benchmark")));
        let scalar = dispatcher.int8_gemm(&a, 5, 65, &w, 3,
            DispatchRegime::PrefillGemm, Some(KernelTier::Scalar)).unwrap();
        assert_eq!(scalar.selection.tier, KernelTier::Scalar); assert_eq!(native.output, scalar.output);
        // A different regime does not inherit this row or invent a measurement.
        let other = dispatcher.int8_gemm(&a, 5, 65, &w, 3, DispatchRegime::BatchedDecodeSkinnyM, None).unwrap();
        assert_eq!(other.selection.tier, KernelTier::Scalar);
    }
}

#[test]
fn forged_features_cannot_authorize_vnni_instructions_via_forcing_or_table_rows() {
    for (tier, kind) in TIERS {
        let key = DispatchKey::new(KernelOperation::Int8Gemv, DispatchRegime::DecodeGemv,
            KernelShape { k: 65, m: 1, n: 1 });
        let dispatcher = Dispatcher::new(features(31, Architecture::X86_64),
            DispatchTable::from_rows(vec![row(tier, key)]).unwrap());
        let forced = dispatcher.int8_gemv(&[-128; 65], &[-128; 65], 1, DispatchRegime::DecodeGemv, Some(tier));
        let selected = dispatcher.int8_gemv(&[-128; 65], &[-128; 65], 1, DispatchRegime::DecodeGemv, None);
        match Int8Kernel::detect(kind) {
            Ok(_) => {
                for result in [forced, selected] {
                    let result = result.unwrap(); assert_eq!(result.output, [65 * 16384]);
                    assert_eq!(result.selection.tier, tier);
                }
            }
            Err(CpuKernelError::UnsupportedIsa) if cfg!(target_arch = "x86_64") => {
                for result in [forced, selected] {
                    assert_eq!(result.unwrap_err(), DispatchError::CpuKernel(CpuKernelError::UnsupportedIsa));
                }
            }
            Err(CpuKernelError::UnsupportedIsa) => {
                assert_eq!(forced.unwrap_err(), DispatchError::ForcedTierUnimplemented { requested: tier });
                // A foreign compilation target may ignore an inapplicable table
                // row, but must report scalar, never a fabricated VNNI result.
                let result = selected.unwrap(); assert_eq!(result.selection.tier, KernelTier::Scalar);
                assert!(matches!(result.selection.provenance, SelectionProvenance::ConservativeDefault { .. }));
            }
            Err(error) => panic!("unexpected host admission: {error}"),
        }
    }
}

#[test]
fn all_feature_combinations_keep_vex_evex_width_and_architecture_independent() {
    let key = DispatchKey::new(KernelOperation::Int8Gemv, DispatchRegime::DecodeGemv,
        KernelShape { k: 65, m: 1, n: 1 });
    for architecture in [Architecture::X86_64, Architecture::Aarch64, Architecture::Other] {
        for mask in 0_u8..32 {
            let f = features(mask, architecture);
            let dispatcher = Dispatcher::new(f, DispatchTable::default());
            let x86 = architecture == Architecture::X86_64;
            for (tier, expected) in [(KernelTier::X2AvxVnni, x86 && f.x86_avx2 && f.x86_avxvnni),
                (KernelTier::X1bAvx512VnniYmm, x86 && f.x86_avx2 && f.x86_avx512f && f.x86_avx512vnni && f.x86_avx512vl),
                (KernelTier::X1aAvx512VnniZmm, x86 && f.x86_avx512f && f.x86_avx512vnni)] {
                assert_eq!(tier.is_detected(f), expected, "tier={tier} mask={mask}");
                let result = dispatcher.select(key, Some(tier));
                if !expected {
                    assert!(matches!(result, Err(DispatchError::ForcedTierUnavailable { requested, .. }) if requested == tier));
                } else if tier.is_implemented() {
                    let result = result.unwrap(); assert_eq!(result.tier, tier); assert_eq!(result.tile, tile(tier));
                } else { assert_eq!(result.unwrap_err(), DispatchError::ForcedTierUnimplemented { requested: tier }); }
            }
            assert_eq!(dispatcher.select(key, None).unwrap().tier, KernelTier::Scalar);
        }
    }
}

#[test]
fn packed_int4_dispatch_uses_real_vnni_leaves_and_operation_specific_rows() {
    packed_int4::verify(&TIERS, features(31, Architecture::X86_64));
}

#[test]
fn registered_widths_reject_false_tile_geometry_without_changing_default_reporting() {
    let report = host_backend_report();
    assert_eq!(report.registry.len(), KernelTier::ALL.len());
    assert_eq!(report.selections.len(), 3 * 3 * 3 * 5);
    assert!(report.selections.iter().all(|s| s.tier == KernelTier::Scalar));
    assert!(report.selections.iter().all(|s| matches!(&s.provenance, SelectionProvenance::ConservativeDefault { .. })));
    for (tier, _) in TIERS {
        let entry = report.registry.iter().find(|entry| entry.tier == tier).unwrap();
        assert_eq!(entry.implementation == "registered", cfg!(target_arch = "x86_64"));
        for operation in [KernelOperation::Int8Gemm, KernelOperation::Int8Gemv] {
            assert_eq!(tier.supports_operation(operation), cfg!(target_arch = "x86_64"));
            let key = DispatchKey::new(operation, DispatchRegime::DecodeGemv, KernelShape { k: 65, m: 1, n: 1 });
            let mut wrong = row(tier, key); wrong.tile.m = 1;
            assert!(matches!(DispatchTable::from_rows(vec![wrong]), Err(DispatchError::InvalidMeasurement { .. })));
            let mut wrong_width = row(tier, key); wrong_width.tile.k = 128;
            assert!(matches!(DispatchTable::from_rows(vec![wrong_width]), Err(DispatchError::InvalidMeasurement { .. })));
            assert!(DispatchTable::from_rows(vec![row(tier, key)]).is_ok());
        }
    }
}

#[test]
fn dispatch_refuses_bad_shapes_k_bounds_and_capacity_before_output_allocation() {
    let dispatcher = Dispatcher::host(DispatchTable::default());
    for tier in available() {
        assert_eq!(dispatcher.int8_gemm(&[], usize::MAX, 2, &[], 1,
            DispatchRegime::PrefillGemm, Some(tier)).unwrap_err(), DispatchError::CpuKernel(CpuKernelError::Shape));
        assert_eq!(dispatcher.int8_gemm(&[], 0, 10753, &[], 0,
            DispatchRegime::PrefillGemm, Some(tier)).unwrap_err(), DispatchError::CpuKernel(CpuKernelError::ReductionTooLong));
        assert_eq!(dispatcher.int8_gemm(&[], usize::MAX, 0, &[], 1,
            DispatchRegime::PrefillGemm, Some(tier)).unwrap_err(), DispatchError::AllocationRefused);
        assert!(dispatcher.int8_gemm(&[], usize::MAX, 0, &[], 0,
            DispatchRegime::PrefillGemm, Some(tier)).unwrap().output.is_empty());
        assert_eq!(dispatcher.int8_gemm(&[], 2, 0, &[], 3,
            DispatchRegime::PrefillGemm, Some(tier)).unwrap().output, [0; 6]);
    }
}
