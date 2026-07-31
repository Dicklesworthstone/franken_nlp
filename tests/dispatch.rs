use franken_nlp::native_engine::dispatch::{
    Architecture, DetectedFeatures, DispatchError, DispatchKey, DispatchRegime, DispatchRow,
    DispatchTable, Dispatcher, KernelOperation, KernelShape, KernelTier, MBucket,
    MeasuredCandidate, MeasurementProvenance, SelectionProvenance, TileGeometry,
    host_backend_report,
};

fn x86(features: (bool, bool, bool, bool, bool)) -> DetectedFeatures {
    let (avx2, avx512f, avx512vl, avx512vnni, avxvnni) = features;
    DetectedFeatures {
        aarch64_dotprod: false,
        aarch64_i8mm: false,
        architecture: Architecture::X86_64,
        x86_avx2: avx2,
        x86_avx512f: avx512f,
        x86_avx512vl: avx512vl,
        x86_avx512vnni: avx512vnni,
        x86_avxvnni: avxvnni,
    }
}

fn key() -> DispatchKey {
    DispatchKey::new(
        KernelOperation::Int8Gemm,
        DispatchRegime::PrefillGemm,
        KernelShape {
            k: 3_072,
            m: 8,
            n: 6_144,
        },
    )
}

fn measured_scalar_row() -> DispatchRow {
    DispatchRow {
        key: key(),
        provenance: MeasurementProvenance {
            benchmark_id: "dispatch-test-scalar".to_owned(),
            host_class: "synthetic-test".to_owned(),
            recorded_on: "2026-07-31".to_owned(),
        },
        selected_median_ns: 100,
        selected_tier: KernelTier::Scalar,
        tile: TileGeometry::scalar(),
        wider_tier_losses: vec![MeasuredCandidate {
            median_ns: 101,
            tier: KernelTier::X3aAvx2Low7HighBit,
        }],
    }
}

#[test]
fn lookup_uses_measured_row_and_retains_its_provenance() {
    let table = DispatchTable::from_rows(vec![measured_scalar_row()]).unwrap();
    let selection = Dispatcher::new(x86((true, false, false, false, false)), table)
        .select(key(), None)
        .unwrap();

    assert_eq!(selection.tier, KernelTier::Scalar);
    assert_eq!(selection.tile, TileGeometry::scalar());
    match selection.provenance {
        SelectionProvenance::Measured {
            benchmark_id,
            selected_median_ns,
            wider_tier_losses,
            ..
        } => {
            assert_eq!(benchmark_id, "dispatch-test-scalar");
            assert_eq!(selected_median_ns, 100);
            assert_eq!(wider_tier_losses[0].tier, KernelTier::X3aAvx2Low7HighBit);
        }
        other => panic!("expected measured provenance, got {other:?}"),
    }
}

#[test]
fn no_measurement_is_explicit_conservative_scalar_default() {
    let dispatcher = Dispatcher::new(
        x86((true, true, true, true, true)),
        DispatchTable::default(),
    );
    let selection = dispatcher.select(key(), None).unwrap();
    assert_eq!(selection.tier, KernelTier::Scalar);
    assert!(selection.candidates.contains(&KernelTier::X1aAvx512VnniZmm));
    assert_eq!(
        selection.provenance,
        SelectionProvenance::ConservativeDefault {
            detail: "no measurement — conservative default"
        }
    );
}

#[test]
fn forced_unsupported_tier_fails_before_selection() {
    let dispatcher = Dispatcher::new(
        x86((false, false, false, false, false)),
        DispatchTable::default(),
    );
    let error = dispatcher
        .select(key(), Some(KernelTier::X2AvxVnni))
        .unwrap_err();
    assert_eq!(
        error,
        DispatchError::ForcedTierUnavailable {
            requested: KernelTier::X2AvxVnni,
            detected: vec![KernelTier::Scalar],
        }
    );
}

#[test]
fn forced_detected_unimplemented_tier_fails_without_entering_a_kernel() {
    let dispatcher = Dispatcher::new(
        x86((true, false, false, false, false)),
        DispatchTable::default(),
    );
    assert_eq!(
        dispatcher
            .select(key(), Some(KernelTier::X3aAvx2Low7HighBit))
            .unwrap_err(),
        DispatchError::ForcedTierUnimplemented {
            requested: KernelTier::X3aAvx2Low7HighBit,
        }
    );
}

#[test]
fn arbitrary_feature_shape_and_regime_queries_are_total_and_safe() {
    for mask in 0_u8..32 {
        let dispatcher = Dispatcher::new(
            x86((
                mask & 1 != 0,
                mask & 2 != 0,
                mask & 4 != 0,
                mask & 8 != 0,
                mask & 16 != 0,
            )),
            DispatchTable::default(),
        );
        for m in [0, 1, 2, 4, 5, 16, 17, 128] {
            for k in [0, 1, 3_072, 6_144, 10_752] {
                for n in [0, 1, 1_024, 3_072, 6_144, 10_752, 166_144] {
                    for regime in DispatchRegime::ALL {
                        let query = DispatchKey::new(
                            KernelOperation::Int8Gemm,
                            regime,
                            KernelShape { k, m, n },
                        );
                        let selection = dispatcher.select(query, None).unwrap();
                        assert!(selection.candidates.contains(&KernelTier::Scalar));
                        assert_eq!(selection.tier, KernelTier::Scalar);
                    }
                }
            }
        }
    }
}

#[test]
fn malformed_duplicate_and_unknown_table_rows_reject_cleanly() {
    let duplicate_rows = vec![measured_scalar_row(), measured_scalar_row()];
    assert!(matches!(
        DispatchTable::from_rows(duplicate_rows),
        Err(DispatchError::DuplicateTableKey { .. })
    ));

    let unknown = r#"[{"key":{"m_bucket":"five_to_sixteen","operation":"int8_gemm","regime":"prefill_gemm","shape":{"k":3072,"m":8,"n":6144}},"provenance":{"benchmark_id":"x","host_class":"x","recorded_on":"2026-07-31"},"selected_median_ns":1,"selected_tier":"scalar","tile":{"k":1,"m":1,"n":1},"wider_tier_losses":[],"unexpected":true}]"#;
    assert!(matches!(
        DispatchTable::from_json(unknown),
        Err(DispatchError::CorruptTable { .. })
    ));

    let duplicate_field = r#"[{"key":{"m_bucket":"five_to_sixteen","operation":"int8_gemm","regime":"prefill_gemm","shape":{"k":3072,"m":8,"n":6144}},"provenance":{"benchmark_id":"x","host_class":"x","recorded_on":"2026-07-31"},"selected_median_ns":1,"selected_median_ns":2,"selected_tier":"scalar","tile":{"k":1,"m":1,"n":1},"wider_tier_losses":[]}]"#;
    assert!(matches!(
        DispatchTable::from_json(duplicate_field),
        Err(DispatchError::CorruptTable { .. })
    ));
}

#[test]
fn entry_points_execute_the_scalar_floor_and_keep_the_selection() {
    let dispatcher = Dispatcher::new(
        x86((false, false, false, false, false)),
        DispatchTable::default(),
    );
    let gemm = dispatcher
        .int8_gemm(
            &[1, 2, 3, 4],
            2,
            2,
            &[3, 4, -5, 6],
            2,
            DispatchRegime::PrefillGemm,
            None,
        )
        .unwrap();
    assert_eq!(gemm.output, vec![11, 7, 25, 9]);
    assert_eq!(gemm.selection.tier, KernelTier::Scalar);

    let gemv = dispatcher
        .int8_gemv(&[1, 2], &[3, 4, -5, 6], 2, DispatchRegime::DecodeGemv, None)
        .unwrap();
    assert_eq!(gemv.output, vec![11, 7]);

    let int4 = dispatcher
        .int4_gemv(
            &[1, 2, 3, 4],
            &[0x78, 0xf0],
            1,
            DispatchRegime::DecodeGemv,
            None,
        )
        .unwrap();
    assert_eq!(int4.output, vec![2]);
}

#[test]
fn host_report_covers_every_fixed_shape_regime_and_tier() {
    let report = host_backend_report();
    assert_eq!(report.registry.len(), KernelTier::ALL.len());
    assert_eq!(report.selections.len(), 3 * 3 * 3 * 5);
    assert!(
        report
            .selections
            .iter()
            .all(|selection| selection.tier == KernelTier::Scalar)
    );
    assert!(report.selections.iter().all(|selection| {
        matches!(
            &selection.provenance,
            SelectionProvenance::ConservativeDefault { .. }
        )
    }));
    assert_eq!(MBucket::for_m(4), MBucket::TwoToFour);
    eprintln!(
        "DISPATCH RESULT=PASS selections={} architecture={:?}",
        report.selections.len(),
        report.architecture
    );
}
