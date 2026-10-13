//! Forced and table-selected dispatch must execute the reported INT8 body.
//! Synthetic table values below are policy fixtures, never performance evidence.
#[path = "support/packed_int4_dispatch.rs"]
mod packed_int4;

use franken_nlp::native_engine::{
    dispatch::{Architecture, DetectedFeatures, DispatchError, DispatchKey, DispatchRegime,
        DispatchRow, DispatchTable, Dispatcher, KernelOperation, KernelShape, KernelTier,
        MeasurementProvenance, SelectionProvenance, TileGeometry, host_backend_report},
    int8,
    simd::{CpuKernelError, Int8Kernel, Int8KernelKind},
};
const TIERS: [(KernelTier, Int8KernelKind); 2] = [
    (KernelTier::X3aAvx2Low7HighBit, Int8KernelKind::Avx2Low7HighBit),
    (KernelTier::X3bAvx2WidenedI16, Int8KernelKind::Avx2WidenedI16),
];
fn available() -> Vec<KernelTier> {
    TIERS.into_iter().filter_map(|(tier, kind)| match Int8Kernel::detect(kind) {
        Ok(_) => Some(tier),
        Err(CpuKernelError::UnsupportedIsa) => {
            eprintln!("AVX2_DISPATCH tier={tier} RESULT=SKIPPED_UNSUPPORTED_ISA"); None
        }
        Err(e) => panic!("unexpected kernel refusal: {e}"),
    }).collect()
}
fn synthetic_avx2() -> DetectedFeatures {
    DetectedFeatures { architecture: Architecture::X86_64, aarch64_dotprod: false,
        aarch64_i8mm: false, x86_avx2: true, x86_avx512f: false, x86_avx512vl: false,
        x86_avx512vnni: false, x86_avxvnni: false }
}
fn row(tier: KernelTier, key: DispatchKey) -> DispatchRow {
    DispatchRow { key, selected_tier: tier, tile: TileGeometry { k: 32, m: 1, n: 1 },
        selected_median_ns: 1, wider_tier_losses: Vec::new(),
        provenance: MeasurementProvenance { benchmark_id: "synthetic-route-test-NOT-a-benchmark".to_owned(),
            host_class: "synthetic-test".to_owned(), recorded_on: "2026-10-11".to_owned() } }
}

#[test]
fn forced_gemm_and_gemv_report_the_actual_body_across_vector_and_matrix_tails() {
    let dispatcher = Dispatcher::host(DispatchTable::default());
    for tier in available() {
        for k in (0..66).chain([3072, 6144, 10752]) {
            for m in [1_usize, 2, 5] {
                let n = 3_usize;
                let a: Vec<i8> = (0..m * k).map(|i| ((i * 53 + 128) % 256) as u8 as i8).collect();
                let w: Vec<i8> = (0..n * k).map(|i| ((i * 29 + 127) % 256) as u8 as i8).collect();
                let result = dispatcher.int8_gemm(&a, m, k, &w, n,
                    DispatchRegime::BatchedDecodeSkinnyM, Some(tier)).unwrap();
                assert_eq!(result.output, int8::gemm_s8s8(&a, m, k, &w, n).unwrap());
                assert_eq!(result.selection.tier, tier);
                assert_eq!(result.selection.kernel_id, tier.kernel_id(KernelOperation::Int8Gemm));
                assert!(!result.selection.kernel_id.contains("pending"));
                assert_eq!(result.selection.tile, TileGeometry { k: 32, m: 1, n: 1 });
                assert!(matches!(result.selection.provenance,
                    SelectionProvenance::ConservativeDefault { detail } if detail.contains("explicit forced")));
                let vector = dispatcher.int8_gemv(&a[..k], &w, n, DispatchRegime::DecodeGemv, Some(tier)).unwrap();
                assert_eq!(vector.output, int8::gemv_s8s8(&a[..k], &w, n).unwrap());
                assert_eq!(vector.selection.tier, tier);
                assert_eq!(vector.selection.kernel_id, tier.kernel_id(KernelOperation::Int8Gemv));
            }
        }
    }
}

#[test]
fn a_supplied_table_routes_native_execution_but_explicit_scalar_still_wins() {
    for tier in available() {
        let key = DispatchKey::new(KernelOperation::Int8Gemm, DispatchRegime::PrefillGemm,
            KernelShape { k: 33, m: 2, n: 3 });
        let dispatcher = Dispatcher::host(DispatchTable::from_rows(vec![row(tier, key)]).unwrap());
        let native = dispatcher.int8_gemm(&[-128; 66], 2, 33, &[-128; 99], 3,
            DispatchRegime::PrefillGemm, None).unwrap();
        assert_eq!(native.output, vec![33 * 16384; 6]);
        assert_eq!(native.selection.tier, tier);
        assert!(matches!(native.selection.provenance,
            SelectionProvenance::Measured { benchmark_id, .. } if benchmark_id.contains("NOT-a-benchmark")));
        let forced = dispatcher.int8_gemm(&[-128; 66], 2, 33, &[-128; 99], 3,
            DispatchRegime::PrefillGemm, Some(KernelTier::Scalar)).unwrap();
        assert_eq!(forced.selection.tier, KernelTier::Scalar);
        assert_eq!(forced.output, native.output);
    }
}

#[test]
fn injected_feature_reports_cannot_authorize_unsupported_machine_instructions() {
    let dispatcher = Dispatcher::new(synthetic_avx2(), DispatchTable::default());
    for (tier, kind) in TIERS {
        let result = dispatcher.int8_gemv(&[1; 32], &[-128; 32], 1, DispatchRegime::DecodeGemv, Some(tier));
        match Int8Kernel::detect(kind) {
            Ok(_) => {
                let result = result.unwrap(); assert_eq!(result.output, [-4096]); assert_eq!(result.selection.tier, tier);
            }
            Err(CpuKernelError::UnsupportedIsa) => {
                assert!(matches!(result, Err(DispatchError::CpuKernel(CpuKernelError::UnsupportedIsa))
                    | Err(DispatchError::ForcedTierUnimplemented { .. })));
            }
            Err(e) => panic!("unexpected kernel refusal: {e}"),
        }
    }
}

#[test]
fn packed_int4_dispatch_uses_real_avx2_leaves_and_operation_specific_rows() {
    packed_int4::verify(&TIERS, synthetic_avx2());
}

#[test]
fn native_shape_overflow_reduction_bounds_and_empty_shapes_fail_or_terminate_before_allocation() {
    let dispatcher = Dispatcher::host(DispatchTable::default());
    for tier in available() {
        assert_eq!(dispatcher.int8_gemm(&[], usize::MAX, 2, &[], 1, DispatchRegime::PrefillGemm, Some(tier)).unwrap_err(),
            DispatchError::CpuKernel(CpuKernelError::Shape));
        assert_eq!(dispatcher.int8_gemm(&[], 1, 10753, &[], 1, DispatchRegime::PrefillGemm, Some(tier)).unwrap_err(),
            DispatchError::CpuKernel(CpuKernelError::ReductionTooLong));
        assert_eq!(dispatcher.int8_gemm(&[], usize::MAX, 0, &[], 1, DispatchRegime::PrefillGemm, Some(tier)).unwrap_err(),
            DispatchError::AllocationRefused);
        assert!(dispatcher.int8_gemm(&[], usize::MAX, 0, &[], 0, DispatchRegime::PrefillGemm, Some(tier)).unwrap().output.is_empty());
        assert_eq!(dispatcher.int8_gemm(&[], 2, 0, &[], 3, DispatchRegime::PrefillGemm, Some(tier)).unwrap().output, [0; 6]);
    }
}

#[test]
fn registered_tiers_do_not_invent_measurements_or_unimplemented_tile_geometry() {
    let report = host_backend_report();
    for (tier, _) in TIERS {
        let entry = report.registry.iter().find(|entry| entry.tier == tier).unwrap();
        let compiled = cfg!(target_arch = "x86_64");
        assert_eq!(entry.implementation == "registered", compiled);
        assert_eq!(tier.supports_operation(KernelOperation::Int8Gemm), compiled);
        assert_eq!(tier.supports_operation(KernelOperation::Int8Gemv), compiled);
        let key = DispatchKey::new(KernelOperation::Int8Gemv, DispatchRegime::DecodeGemv,
            KernelShape { k: 32, m: 1, n: 1 });
        let mut invalid = row(tier, key); invalid.tile = TileGeometry { k: 64, m: 4, n: 4 };
        assert!(matches!(DispatchTable::from_rows(vec![invalid]), Err(DispatchError::InvalidMeasurement { .. })));
    }
    assert!(report.selections.iter().all(|selection| selection.tier == KernelTier::Scalar));
    assert!(report.selections.iter().all(|selection| matches!(selection.provenance,
        SelectionProvenance::ConservativeDefault { .. })));
}
