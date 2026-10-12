use super::*;

fn empty_tier(status: TierStatus) -> TierReport {
    TierReport { tier: KernelTier::Scalar, status, cases_run: 0, cases_passed: 0, cases: Vec::new() }
}
fn complete_tier() -> TierReport {
    let case = CaseReport { operation: Operation::Gemv, pattern: Pattern::Mixed,
        m: 1, k: 2, n: 1, requested_kernel: "s_scalar_i32_int8_gemv",
        observed_kernel: Some("s_scalar_i32_int8_gemv"), expected_sha256: output_digest(&[11]),
        actual_sha256: Some(output_digest(&[11])), failure: None, first_mismatch: None };
    TierReport { tier: KernelTier::Scalar, status: TierStatus::Pass,
        cases_run: CASES_PER_TIER, cases_passed: CASES_PER_TIER, cases: vec![case; CASES_PER_TIER] }
}

#[test]
fn independent_i64_reference_matches_named_small_and_extreme_goldens() {
    assert_eq!(oracle(&[1, 2, 3, 4], 2, 2, &[3, 4, -5, 6], 2).unwrap(), [11, 7, 25, 9]);
    assert_eq!(oracle(&[], 4, 0, &[], 3).unwrap(), [0; 12]);
    let a = fixture(5, 10752, Pattern::Extremes, false).unwrap();
    let w = fixture(3, 10752, Pattern::Extremes, true).unwrap();
    let expected = oracle(&a, 5, 10752, &w, 3).unwrap();
    assert_eq!(&expected[..3], &[176_160_768, -174_784_512, 176_160_768]);
    assert_eq!(&expected[9..12], &[0, 0, 0]);
}

#[test]
fn matching_numbers_from_the_wrong_body_are_not_a_pass() {
    assert_eq!(compare(&[11, 7], &[11, 7], 2, false), (Some(CaseFailure::WrongKernel), None));
    assert_eq!(compare(&[11, 7], &[11, 7], 2, true), (None, None));
}

#[test]
fn shape_refusal_and_first_bad_integer_have_distinct_diagnostics() {
    assert_eq!(compare(&[1, 2], &[1], 2, true), (Some(CaseFailure::OutputShape), None));
    assert_eq!(compare(&[11, 7, 25, 9], &[11, 7, 25, 10], 2, true),
        (Some(CaseFailure::IntegerMismatch), Some(Mismatch { row: 1, column: 1, expected: 9, actual: 10 })));
}

#[test]
fn output_hash_has_fixed_domain_count_order_and_little_endian_bytes() {
    assert_eq!(output_digest(&[11, 7, 25, 9]),
        "70689752315275bb6f59522d5ba9334995f6efaddc331df8f18417e8c3611894");
    assert_ne!(output_digest(&[1, 2]), output_digest(&[2, 1]));
    assert_ne!(output_digest(&[]), output_digest(&[0]));
    assert_ne!(output_digest(&[-1]), output_digest(&[255]));
}

#[test]
fn unexecuted_truncated_and_failed_tiers_cannot_produce_overall_pass() {
    assert_eq!(summarize(&[], false), Verdict::Fail);
    assert_eq!(summarize(&[empty_tier(TierStatus::Pass)], false), Verdict::Fail);
    assert_eq!(summarize(&[empty_tier(TierStatus::NotImplemented)], false), Verdict::Fail);
    assert_eq!(summarize(&[empty_tier(TierStatus::Fail)], false), Verdict::Fail);
    let mut truncated = complete_tier(); truncated.cases.pop();
    assert_eq!(summarize(&[truncated], false), Verdict::Fail);
    let mut corrupted = complete_tier(); corrupted.cases[CASES_PER_TIER - 1].failure = Some(CaseFailure::IntegerMismatch);
    assert_eq!(summarize(&[corrupted], false), Verdict::Fail);
    let mut uncounted = complete_tier(); uncounted.cases_passed -= 1;
    assert_eq!(summarize(&[uncounted], false), Verdict::Fail);
}

#[test]
fn forced_unavailable_is_not_a_successful_all_available_run() {
    let skipped = empty_tier(TierStatus::SkippedUnsupportedIsa);
    assert_eq!(summarize(&[skipped.clone()], true), Verdict::Unsupported);
    assert_eq!(summarize(&[skipped.clone()], false), Verdict::Unsupported);
    assert_eq!(summarize(&[complete_tier(), skipped.clone()], false), Verdict::Pass);
    assert_eq!(summarize(&[complete_tier(), skipped], true), Verdict::Unsupported);
}

#[test]
fn fixed_population_covers_real_vector_and_matrix_tails_without_user_dimensions() {
    assert_eq!(CASES_PER_TIER, 180);
    assert_eq!(KS.iter().copied().max(), Some(10752));
    for boundary in [8_usize, 32, 64, 128] {
        for k in [boundary - 1, boundary, boundary + 1] { assert!(KS.contains(&k)); }
    }
    let mut identities = std::collections::BTreeSet::new();
    for k in KS {
        for (operation, m, n) in SHAPES {
            assert!(m <= 5 && n <= 3);
            if operation == Operation::Cohort { assert!(m <= 4 && n == 1); }
            for pattern in [Pattern::Extremes, Pattern::Mixed] {
                assert!(identities.insert(format!("{operation:?}:{pattern:?}:{m}:{k}:{n}")));
            }
        }
    }
    assert_eq!(identities.len(), CASES_PER_TIER);
}

#[test]
fn scalar_selftest_really_runs_every_case_without_a_model() {
    let report = run(SelftestRequest { tier: Some(KernelTier::Scalar) }).unwrap();
    assert_eq!(report.verdict, Verdict::Pass);
    assert_eq!(report.tiers.len(), 1);
    assert_eq!(report.cases_run, CASES_PER_TIER);
    assert_eq!(report.cases_passed, CASES_PER_TIER);
    assert_eq!(report.tiers_skipped, 0);
    for case in &report.tiers[0].cases {
        assert_eq!(case.observed_kernel, Some(case.requested_kernel));
        assert_eq!(case.actual_sha256.as_ref(), Some(&case.expected_sha256));
        assert!(case.failure.is_none() && case.first_mismatch.is_none());
    }
}
