use super::*;

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn ledger(rows: usize, columns: usize) -> ProjectionLedger {
    ProjectionLedger::new(ProjectionWork::for_shape(rows, columns).unwrap())
}
fn activation(values: &[f32]) -> ActivationBuffer {
    let mut buffer = ActivationBuffer::try_new(values.len()).unwrap();
    buffer.encode_f32(values).unwrap(); buffer
}

#[test]
fn zero_and_negative_zero_have_canonical_positive_scale() {
    let buffer = activation(&[0.0, -0.0]);
    assert_eq!(buffer.values().unwrap(), [0, 0]);
    assert_eq!(buffer.scale().unwrap().to_bits(), 1.0_f32.to_bits());
}
#[test]
fn dynamic_quantization_uses_nearest_even_not_away_from_zero() {
    let buffer = activation(&[-127.0, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 127.0]);
    assert_eq!(buffer.values().unwrap(), [-127, -4, -2, -2, 0, 0, 2, 2, 4, 127]);
}
#[test]
fn failed_encoding_cannot_reuse_a_previous_valid_activation() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut buffer = activation(&[1.0]);
        assert_eq!(buffer.encode_f32(&[bad]), Err(LinearError::Activation));
        assert!(buffer.values().is_err()); assert!(buffer.scale().is_err());
    }
}
#[test]
fn scale_underflow_is_refused_not_silently_saturated() {
    let mut buffer = ActivationBuffer::try_new(1).unwrap();
    assert_eq!(buffer.encode_f32(&[f32::from_bits(1)]), Err(LinearError::ScaleUnderflow));
    assert!(buffer.values().is_err());
}
#[test]
fn largest_finite_input_quantizes_without_intermediate_overflow() {
    let buffer = activation(&[-f32::MAX, 0.0, f32::MAX]);
    assert_eq!(buffer.values().unwrap(), [-127, 0, 127]);
    assert!(buffer.scale().unwrap().is_finite());
}
#[test]
fn bf16_encoding_equals_exact_widening_and_reuses_storage() {
    let input = [Bf16::from_f32(-3.25), Bf16::from_f32(0.125), Bf16::from_f32(4.0)];
    let mut a = ActivationBuffer::try_new(10).unwrap();
    let ptr = a.values.as_ptr();
    a.encode_bf16(&input).unwrap();
    let b = activation(&input.map(Bf16::to_f32));
    assert_eq!(a.values().unwrap(), b.values().unwrap());
    assert_eq!(a.scale().unwrap().to_bits(), b.scale().unwrap().to_bits());
    a.encode_f32(&[0.0]).unwrap(); assert_eq!(ptr, a.values.as_ptr());
}
#[test]
fn workspace_shape_is_checked_before_encoding() {
    assert!(ActivationBuffer::try_new(0).is_err());
    assert!(ActivationBuffer::try_new(MAX_MODEL_K + 1).is_err());
    let mut a = activation(&[1.0]);
    assert_eq!(a.encode_f32(&[1.0, 2.0]), Err(LinearError::Workspace));
    assert!(a.values().is_err());
    assert_eq!(a.encode_f32(&[]), Err(LinearError::Activation));
}
#[test]
fn matrix_checks_all_sidecars_before_any_projection() {
    for scale in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(matches!(QuantizedLinear::checked(1, 2, &[1, 2], &[scale], &[3]), Err(LinearError::WeightScale)));
    }
    assert!(matches!(QuantizedLinear::checked(1, 2, &[1, 2], &[1.0], &[2]), Err(LinearError::RowSum)));
    assert!(matches!(QuantizedLinear::checked(1, 2, &[1], &[1.0], &[1]), Err(LinearError::Shape)));
    assert!(matches!(QuantizedLinear::checked(0, 2, &[], &[], &[]), Err(LinearError::Shape)));
}
#[test]
fn full_i8_domain_and_offset_algebra_agree_for_each_model_width() {
    for width in [3072, 6144, 10752] {
        let weights: Vec<_> = (0..width).map(|i| (i % 256) as u8 as i8).collect();
        let sum: i32 = weights.iter().map(|&v| i32::from(v)).sum();
        let matrix = QuantizedLinear::checked(1, width, &weights, &[0.5], &[sum]).unwrap();
        let source: Vec<_> = (0..width).map(|i| ((i % 255) as i32 - 127) as f32).collect();
        let input = activation(&source);
        let mut output = [0.0];
        matrix.project_f32_into(&input, LinearRows::All, &mut output, &mut ledger(1, width), &mut Continue).unwrap();
        let signed = dot_s8s8(input.values().unwrap(), &weights).unwrap();
        let offset = super::super::quant_algebra::corrected_x86_offset_dot_i32(input.values().unwrap(), &weights).unwrap();
        assert_eq!(signed, offset); assert_eq!(output[0], (signed as f32) * 0.5);
    }
}
#[test]
fn sliced_projection_equals_full_projection_and_charges_only_selected_rows() {
    let matrix = QuantizedLinear::checked(3, 3, &[1, 2, 3, -128, 127, 0, 4, -5, 6], &[0.5, 0.25, 2.0], &[6, -1, 5]).unwrap();
    let input = activation(&[-127.0, 64.0, 2.0]);
    let mut full = [0.0; 3]; let mut selected = [0.0; 2];
    matrix.project_f32_into(&input, LinearRows::All, &mut full, &mut ledger(3, 3), &mut Continue).unwrap();
    let mut budget = ledger(2, 3);
    matrix.project_f32_into(&input, LinearRows::Selected(&[0, 2]), &mut selected, &mut budget, &mut Continue).unwrap();
    assert_eq!(selected.map(f32::to_bits), [full[0].to_bits(), full[2].to_bits()]);
    assert_eq!(budget.reserved(), ProjectionWork { dot_products: 2, multiply_accumulates: 6 });
}
#[test]
fn malformed_selection_and_output_refuse_before_charging() {
    let matrix = QuantizedLinear::checked(2, 1, &[1, 2], &[1.0, 1.0], &[1, 2]).unwrap();
    let input = activation(&[1.0]);
    let mut budget = ledger(4, 1);
    for ids in [&[][..], &[0, 0], &[1, 0], &[2]] {
        let mut output = vec![0.0; ids.len()];
        assert_eq!(matrix.project_f32_into(&input, LinearRows::Selected(ids), &mut output, &mut budget, &mut Continue), Err(LinearError::Selection));
    }
    assert_eq!(matrix.project_f32_into(&input, LinearRows::All, &mut [0.0], &mut budget, &mut Continue), Err(LinearError::OutputShape));
    assert_eq!(budget.reserved(), ProjectionWork::default());
}
#[test]
fn insufficient_either_work_axis_is_atomic_and_does_not_touch_output() {
    let matrix = QuantizedLinear::checked(2, 2, &[1, 2, 3, 4], &[1.0, 1.0], &[3, 7]).unwrap();
    for cap in [ProjectionWork { dot_products: 1, multiply_accumulates: 4 }, ProjectionWork { dot_products: 2, multiply_accumulates: 3 }] {
        let mut budget = ProjectionLedger::new(cap); let mut output = [42.0; 2];
        assert_eq!(matrix.project_f32_into(&activation(&[1.0, 2.0]), LinearRows::All, &mut output, &mut budget, &mut Continue), Err(LinearError::WorkBudget));
        assert_eq!(output, [42.0; 2]); assert_eq!(budget.remaining(), cap);
    }
}
#[test]
fn cancellation_keeps_whole_projection_charged_and_retains_cause() {
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
    }
    let matrix = QuantizedLinear::checked(1, 1, &[1], &[1.0], &[1]).unwrap();
    let mut budget = ledger(1, 1); let mut output = [99.0];
    assert_eq!(matrix.project_f32_into(&activation(&[1.0]), LinearRows::All, &mut output, &mut budget, &mut Stop), Err(LinearError::Cancelled(DecodeCancellationKind::Deadline)));
    assert_eq!(output, [99.0]); assert_eq!(budget.remaining(), ProjectionWork::default());
}
#[test]
fn nonfinite_dequantization_and_bf16_cast_fail_closed() {
    let matrix = QuantizedLinear::checked(1, 1, &[127], &[f32::MAX], &[127]).unwrap();
    assert!(matrix.project_f32_into(&activation(&[127.0]), LinearRows::All, &mut [0.0], &mut ledger(1, 1), &mut Continue).is_err());
    let matrix = QuantizedLinear::checked(1, 1, &[1], &[f32::MAX], &[1]).unwrap();
    assert_eq!(matrix.project_bf16_into(&activation(&[1.0]), LinearRows::All, &mut [Bf16::from_f32(0.0)], &mut ledger(1, 1), &mut Continue), Err(LinearError::NonFiniteOutput));
}
#[test]
fn projection_is_independent_of_prior_rows_and_activation_workspace_history() {
    let matrix = QuantizedLinear::checked(1, 2, &[3, -7], &[0.25], &[-4]).unwrap();
    let mut reused = ActivationBuffer::try_new(4).unwrap();
    reused.encode_f32(&[1.0, 2.0, 3.0, 4.0]).unwrap(); reused.encode_f32(&[0.75, -3.0]).unwrap();
    let fresh = activation(&[0.75, -3.0]); let mut a = [0.0]; let mut b = [0.0];
    matrix.project_f32_into(&reused, LinearRows::All, &mut a, &mut ledger(1, 2), &mut Continue).unwrap();
    matrix.project_f32_into(&fresh, LinearRows::All, &mut b, &mut ledger(1, 2), &mut Continue).unwrap();
    assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
}
