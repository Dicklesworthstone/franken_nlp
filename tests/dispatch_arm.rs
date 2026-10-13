//! ARM dispatch executes the reported body; injected features are not ISA authority.
//! Measurement rows here are synthetic routing fixtures, never benchmark evidence.
#[path = "support/packed_int4_dispatch.rs"]
mod packed_int4;

use franken_nlp::native_engine::{
    dispatch::{Architecture, DetectedFeatures, DispatchError, DispatchKey, DispatchRegime,
        DispatchRow, DispatchTable, Dispatcher, KernelOperation, KernelShape, KernelTier,
        MeasurementProvenance, SelectionProvenance, TileGeometry, host_backend_report},
    int8,
    simd::{CpuKernelError, Int8Kernel, Int8KernelKind},
};

const ARM: [(KernelTier, Int8KernelKind); 3] = [
    (KernelTier::A1Smmla, Int8KernelKind::Aarch64Smmla),
    (KernelTier::A2Dotprod, Int8KernelKind::Aarch64Dotprod),
    (KernelTier::A3Autovec, Int8KernelKind::Aarch64Autovec),
];
fn available() -> Vec<KernelTier> {
    ARM.into_iter().filter_map(|(tier, kind)| match Int8Kernel::detect(kind) {
        Ok(_) => Some(tier),
        Err(CpuKernelError::UnsupportedIsa) => {
            eprintln!("ARM_DISPATCH tier={tier} RESULT=SKIPPED_UNSUPPORTED_ISA"); None
        }
        Err(error) => panic!("unexpected ARM handle refusal: {error}"),
    }).collect()
}
fn synthetic_arm(dotprod: bool, i8mm: bool) -> DetectedFeatures {
    DetectedFeatures { architecture: Architecture::Aarch64,
        aarch64_dotprod: dotprod, aarch64_i8mm: i8mm,
        x86_avx2: false, x86_avx512f: false, x86_avx512vl: false,
        x86_avx512vnni: false, x86_avxvnni: false }
}
fn tile(tier: KernelTier) -> TileGeometry {
    match tier {
        KernelTier::A1Smmla => TileGeometry { k: 8, m: 4, n: 2 },
        KernelTier::A2Dotprod | KernelTier::A3Autovec => TileGeometry { k: 16, m: 4, n: 1 },
        _ => panic!("not an ARM test tier"),
    }
}
fn fixture(tier: KernelTier, key: DispatchKey) -> DispatchRow {
    DispatchRow { key, selected_tier: tier, tile: tile(tier), selected_median_ns: 1,
        wider_tier_losses: Vec::new(), provenance: MeasurementProvenance {
            benchmark_id: "synthetic-route-test-NOT-a-benchmark".to_owned(),
            host_class: "synthetic-test".to_owned(), recorded_on: "2026-10-11".to_owned() } }
}
fn pattern(length: usize, salt: usize) -> Vec<i8> {
    (0..length).map(|i| ((i * 53 + salt * 29 + 128) % 256) as u8 as i8).collect()
}

#[test]
fn forced_arm_gemm_and_gemv_keep_the_executed_tier_across_all_tile_tails() {
    let dispatcher = Dispatcher::host(DispatchTable::default());
    for tier in available() {
        for regime in DispatchRegime::ALL {
            for m in [1_usize, 2, 4, 5] {
                for n in [1_usize, 2, 3] {
                    for k in [0, 1, 7, 8, 9, 15, 16, 17, 33, 3072, 6144, 10752] {
                        let a = pattern(m * k, 1); let w = pattern(n * k, 2);
                        let result = dispatcher.int8_gemm(&a, m, k, &w, n, regime, Some(tier)).unwrap();
                        assert_eq!(result.output, int8::gemm_s8s8(&a, m, k, &w, n).unwrap());
                        assert_eq!(result.selection.tier, tier);
                        assert_eq!(result.selection.kernel_id, tier.kernel_id(KernelOperation::Int8Gemm));
                        assert!(!result.selection.kernel_id.contains("pending"));
                        assert_eq!(result.selection.tile, tile(tier));
                        assert!(matches!(result.selection.provenance,
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
fn supplied_arm_table_executes_native_and_explicit_scalar_still_overrides_it() {
    for tier in available() {
        let key = DispatchKey::new(KernelOperation::Int8Gemm, DispatchRegime::PrefillGemm,
            KernelShape { k: 33, m: 5, n: 3 });
        let dispatcher = Dispatcher::host(DispatchTable::from_rows(vec![fixture(tier, key)]).unwrap());
        let native = dispatcher.int8_gemm(&[-128; 165], 5, 33, &[-128; 99], 3,
            DispatchRegime::PrefillGemm, None).unwrap();
        assert_eq!(native.output, vec![33 * 16384; 15]);
        assert_eq!(native.selection.tier, tier);
        assert!(matches!(&native.selection.provenance,
            SelectionProvenance::Measured { benchmark_id, .. } if benchmark_id.contains("NOT-a-benchmark")));
        let scalar = dispatcher.int8_gemm(&[-128; 165], 5, 33, &[-128; 99], 3,
            DispatchRegime::PrefillGemm, Some(KernelTier::Scalar)).unwrap();
        assert_eq!(scalar.selection.tier, KernelTier::Scalar);
        assert_eq!(scalar.output, native.output);
    }
}

#[test]
fn invented_arm_feature_reports_cannot_mint_a_foreign_or_unsupported_executable() {
    let dispatcher = Dispatcher::new(synthetic_arm(true, true), DispatchTable::default());
    for (tier, kind) in ARM {
        let result = dispatcher.int8_gemv(&[127; 32], &[-128; 32], 1, DispatchRegime::DecodeGemv, Some(tier));
        match Int8Kernel::detect(kind) {
            Ok(_) => {
                let result = result.unwrap(); assert_eq!(result.output, [-520192]); assert_eq!(result.selection.tier, tier);
            }
            Err(CpuKernelError::UnsupportedIsa) => assert!(matches!(result,
                Err(DispatchError::CpuKernel(CpuKernelError::UnsupportedIsa))
                | Err(DispatchError::ForcedTierUnimplemented { .. }))),
            Err(error) => panic!("unexpected actual-host refusal: {error}"),
        }
    }
}

#[test]
fn packed_int4_dispatch_uses_real_arm_leaves_and_operation_specific_rows() {
    packed_int4::verify(&ARM, synthetic_arm(true, true));
}

#[test]
fn arm_registry_is_target_specific_and_empty_tables_do_not_invent_winners() {
    let report = host_backend_report();
    assert_eq!(report.registry.len(), KernelTier::ALL.len());
    assert_eq!(report.selections.len(), 3 * 3 * 3 * 5);
    for (tier, _) in ARM {
        let compiled = if tier == KernelTier::A3Autovec { cfg!(target_arch = "aarch64") }
            else { cfg!(all(target_arch = "aarch64", target_endian = "little")) };
        let entry = report.registry.iter().find(|entry| entry.tier == tier).unwrap();
        assert_eq!(entry.implementation == "registered", compiled);
        for operation in [KernelOperation::Int8Gemm, KernelOperation::Int8Gemv] {
            assert_eq!(tier.supports_operation(operation), compiled);
            let key = DispatchKey::new(operation, DispatchRegime::PrefillGemm, KernelShape { k: 33, m: 1, n: 3 });
            let mut wrong = fixture(tier, key); wrong.tile = TileGeometry::scalar();
            assert!(matches!(DispatchTable::from_rows(vec![wrong]), Err(DispatchError::InvalidMeasurement { .. })));
        }
    }
    assert!(report.selections.iter().all(|selection| selection.tier == KernelTier::Scalar));
    assert!(report.selections.iter().all(|selection| matches!(&selection.provenance,
        SelectionProvenance::ConservativeDefault { .. })));
}

#[test]
fn dotprod_and_i8mm_are_independent_capabilities_not_marketing_inferences() {
    let key = DispatchKey::new(KernelOperation::Int8Gemv, DispatchRegime::DecodeGemv,
        KernelShape { k: 8, m: 1, n: 1 });
    for dotprod in [false, true] {
        for i8mm in [false, true] {
            let features = synthetic_arm(dotprod, i8mm);
            assert_eq!(KernelTier::A1Smmla.is_detected(features), i8mm);
            assert_eq!(KernelTier::A2Dotprod.is_detected(features), dotprod);
            assert!(KernelTier::A3Autovec.is_detected(features));
            let dispatcher = Dispatcher::new(features, DispatchTable::default());
            for (tier, detected) in [(KernelTier::A1Smmla, i8mm), (KernelTier::A2Dotprod, dotprod)] {
                if !detected {
                    assert!(matches!(dispatcher.select(key, Some(tier)),
                        Err(DispatchError::ForcedTierUnavailable { requested, .. }) if requested == tier));
                }
            }
            assert_eq!(dispatcher.select(key, None).unwrap().tier, KernelTier::Scalar);
        }
    }
}

#[test]
fn arm_dispatch_rejects_overflow_and_unbounded_k_before_output_allocation() {
    let dispatcher = Dispatcher::host(DispatchTable::default());
    for tier in available() {
        assert_eq!(dispatcher.int8_gemm(&[], usize::MAX, 2, &[], 1,
            DispatchRegime::PrefillGemm, Some(tier)).unwrap_err(), DispatchError::CpuKernel(CpuKernelError::Shape));
        assert_eq!(dispatcher.int8_gemm(&[], 1, 10753, &[], 1,
            DispatchRegime::PrefillGemm, Some(tier)).unwrap_err(), DispatchError::CpuKernel(CpuKernelError::ReductionTooLong));
        assert_eq!(dispatcher.int8_gemm(&[], usize::MAX, 0, &[], 1,
            DispatchRegime::PrefillGemm, Some(tier)).unwrap_err(), DispatchError::AllocationRefused);
        assert!(dispatcher.int8_gemm(&[], usize::MAX, 0, &[], 0,
            DispatchRegime::PrefillGemm, Some(tier)).unwrap().output.is_empty());
        assert_eq!(dispatcher.int8_gemm(&[], 2, 0, &[], 3,
            DispatchRegime::PrefillGemm, Some(tier)).unwrap().output, [0; 6]);
    }
}
