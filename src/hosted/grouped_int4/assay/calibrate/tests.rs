//! Operation admission and ownership regressions without model execution.
use super::*;
use super::super::tests::host;

#[test]
fn explicit_raw_mode_keeps_the_existing_complete_evaluation_and_paired_quotes() {
    let host = host(); let limits = metrics::Limits::default();
    for widths in [&[127][..], &[127, 128][..]] {
        assert_eq!(host.requirements_for_operation(widths, limits, Int4AssayOperation::Raw).unwrap(),
            host.requirements(widths, limits).unwrap());
    }
}

#[test]
fn fitting_charges_its_full_workspace_without_copying_native_state_or_the_delivery_guard() {
    let host = host(); let limits = metrics::Limits::default();
    let raw = host.requirements(&[127], limits).unwrap();
    let fit = host.requirements_for_operation(&[127], limits, Int4AssayOperation::Fit).unwrap();
    assert!(fit.input_bytes > raw.input_bytes);
    assert_eq!(fit.input_bytes - raw.input_bytes,
        calibration::required_calibration_bytes(limits, true, 0).unwrap());
    assert_eq!(fit.native, raw.native); assert_eq!(fit.workspace_bytes, raw.workspace_bytes);
    assert_eq!(fit.output_reservation_bytes, raw.output_reservation_bytes);
}

#[test]
fn independent_application_owners_are_both_charged_while_paired_engines_reuse_one_peak() {
    let host = host(); let limits = metrics::Limits::default();
    let raw = host.requirements(&[127, 128], limits).unwrap();
    let one = host.requirements_for_operation(&[127, 128], limits,
        Int4AssayOperation::Apply { artifacts: 1 }).unwrap();
    let two = host.requirements_for_operation(&[127, 128], limits,
        Int4AssayOperation::Apply { artifacts: 2 }).unwrap();
    assert!(two.input_bytes > one.input_bytes); assert!(one.input_bytes > raw.input_bytes);
    assert_eq!(one.input_bytes - raw.input_bytes,
        calibration::required_calibration_bytes(limits, false, 1).unwrap());
    assert_eq!(two.input_bytes - raw.input_bytes,
        calibration::required_calibration_bytes(limits, false, 2).unwrap());
    for quote in [one, two] {
        assert_eq!(quote.native, raw.native); assert_eq!(quote.workspace_bytes, raw.workspace_bytes);
        assert_eq!(quote.output_reservation_bytes, raw.output_reservation_bytes);
    }
    let single = host.requirements_for_operation(&[127], limits,
        Int4AssayOperation::Apply { artifacts: 1 }).unwrap();
    assert_eq!(single.input_bytes - host.requirements(&[127], limits).unwrap().input_bytes,
        one.input_bytes - raw.input_bytes);
}

#[test]
fn quotes_reject_impossible_fit_and_artifact_arm_shapes_before_any_allocation() {
    let host = host(); let limits = metrics::Limits::default();
    assert!(host.requirements_for_operation(&[127, 128], limits, Int4AssayOperation::Fit).is_err());
    for (widths, artifacts) in [(&[127][..], 0), (&[127][..], 2),
        (&[127, 128][..], 0), (&[127, 128][..], 3), (&[127, 128][..], usize::MAX)] {
        assert!(host.requirements_for_operation(widths, limits, Int4AssayOperation::Apply { artifacts }).is_err());
    }
    for widths in [&[][..], &[127, 128, 129][..]] {
        for operation in [Int4AssayOperation::Raw, Int4AssayOperation::Fit,
            Int4AssayOperation::Apply { artifacts: 1 }] {
            assert!(host.requirements_for_operation(widths, limits, operation).is_err());
        }
    }
}

#[test]
fn ownership_derives_the_quote_for_candidate_only_baseline_only_and_paired_artifacts() {
    assert_eq!(owned_operation(false, 2, [false, false]).unwrap(), Int4AssayOperation::Raw);
    for present in [[true, false], [false, true]] {
        assert_eq!(owned_operation(false, 2, present).unwrap(), Int4AssayOperation::Apply { artifacts: 1 });
    }
    assert_eq!(owned_operation(false, 2, [true, true]).unwrap(), Int4AssayOperation::Apply { artifacts: 2 });
    assert_eq!(owned_operation(false, 1, [true, false]).unwrap(), Int4AssayOperation::Apply { artifacts: 1 });
    assert_eq!(owned_operation(true, 1, [false, false]).unwrap(), Int4AssayOperation::Fit);
    for present in [[false, true], [true, true]] { assert!(owned_operation(false, 1, present).is_err()); }
    for present in [[true, false], [false, true], [true, true]] { assert!(owned_operation(true, 1, present).is_err()); }
    for arms in [0, 2, 3, usize::MAX] { assert!(owned_operation(true, arms, [false, false]).is_err()); }
    for arms in [0, 3, usize::MAX] { assert!(owned_operation(false, arms, [false, false]).is_err()); }
}

#[test]
fn additional_calibration_storage_cannot_wrap_an_otherwise_valid_raw_quote() {
    let mut host = host(); let limits = metrics::Limits::default();
    let original = host.requirements(&[127], limits).unwrap();
    host.preparation_reserve_bytes = u64::MAX - (original.input_bytes - host.preparation_reserve_bytes);
    assert_eq!(host.requirements(&[127], limits).unwrap().input_bytes, u64::MAX);
    for operation in [Int4AssayOperation::Fit, Int4AssayOperation::Apply { artifacts: 1 }] {
        assert!(matches!(host.requirements_for_operation(&[127], limits, operation), Err(HostedError::Limits(_))));
    }
}

#[test]
fn controlled_fit_allocation_preserves_the_resource_category_and_sanitized_diagnostic() {
    let allocation = HostedError::from(crate::calibration::CalibrationError::AllocationRefused);
    assert!(matches!(allocation, HostedError::Assay(AssayError::Allocation)));
    assert_eq!(allocation.to_string(), "hosted native evaluation or qualification refused");
    assert_eq!(format!("{allocation:?}"), allocation.to_string());
    let malformed = HostedError::from(crate::calibration::CalibrationError::InvalidThreshold);
    assert!(matches!(malformed, HostedError::Assay(AssayError::Calibration)));
}

#[test]
fn sealed_applications_and_all_calibration_outcomes_cross_the_existing_physical_boundary() {
    fn send<T: Send + 'static>() {}
    send::<PreparedApplication>(); send::<CalibrationInput>();
    send::<FitOutcome>(); send::<ApplicationOutcome>(); send::<QualificationOutcome>();
    let _fit = NlpEngine::calibrate_int4_classification;
    let _apply = NlpEngine::evaluate_int4_classification_calibrated;
    let _pair = NlpEngine::qualify_int4_classification_calibrated;
}
