use super::*;

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn exact(rows: usize, columns: usize, width: usize) -> Int4ConversionBudget {
    let requirement = Int4ConversionRequirement::for_shape(rows, columns, width).unwrap();
    Int4ConversionBudget { max_payload_bytes: requirement.peak_payload_bytes().unwrap(),
        max_source_values: requirement.source_values, max_groups: requirement.groups }
}
fn bf16(values: &[f32]) -> Vec<Bf16> { values.iter().map(|&value| Bf16::from_f32(value)).collect() }

#[test]
fn converter_pins_symmetric7_nearest_even_nibble_order_and_positive_zero_scale() {
    let source = bf16(&[-7.0, -6.0, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 6.0, 7.0]);
    let matrix = GroupedInt4Matrix::from_bf16(1, 12, 12, &source, exact(1, 12, 12), &mut Continue).unwrap();
    assert_eq!(matrix.packed_bytes(), [0xa9, 0xec, 0x0e, 0x20, 0x42, 0x76]);
    assert_eq!(matrix.group_scales(), [1.0]); assert_eq!(matrix.recipe(), GROUPED_INT4_RECIPE);
    let zero = GroupedInt4Matrix::from_bf16(2, 3, 2, &bf16(&[-0.0; 6]), exact(2, 3, 2), &mut Continue).unwrap();
    assert_eq!(zero.packed_bytes(), [0; 3]);
    assert!(zero.group_scales().iter().all(|scale| scale.to_bits() == 1.0_f32.to_bits()));
}

#[test]
fn streamed_and_contiguous_conversion_have_identical_compact_executable_results() {
    for columns in [1, 3, 33, 3072] {
        for width in [1, 3, 32, 128, MAX_MODEL_K] {
            let rows = 2;
            let source: Vec<_> = (0..rows * columns)
                .map(|i| Bf16::from_f32(((i * 53 % 1001) as f32 - 500.0) / 64.0)).collect();
            let direct = GroupedInt4Matrix::from_bf16(rows, columns, width, &source,
                exact(rows, columns, width), &mut Continue).unwrap();
            let mut visited = Vec::new();
            let streamed = GroupedInt4Matrix::from_bf16_rows(rows, columns, width,
                exact(rows, columns, width), &mut Continue, |row, out| {
                    visited.push((row, out.len()));
                    out.copy_from_slice(&source[row * columns..(row + 1) * columns]); Ok::<(), ()>(())
                }).unwrap();
            assert_eq!(visited, [(0, columns), (1, columns)]);
            assert_eq!(direct.packed_bytes(), streamed.packed_bytes());
            assert_eq!(direct.group_scales(), streamed.group_scales());
            assert_eq!(direct.content_sha256(), streamed.content_sha256());
            let view = streamed.as_linear();
            let mut input = ActivationBuffer::try_new(columns).unwrap();
            input.encode_bf16(&source[..columns]).unwrap();
            let mut output = [0.0; 2];
            let mut scratch = vec![0; view.scratch_len()];
            view.project_f32_into(&input, LinearRows::All, &mut output, &mut scratch,
                &mut ProjectionLedger::new(view.planned_work(LinearRows::All).unwrap()), &mut Continue).unwrap();
            assert!(output.iter().all(|value| value.is_finite()));
        }
    }
}

#[test]
fn all_bf16_bit_patterns_are_either_finite_convertible_values_or_typed_refusals() {
    for bits in 0_u32..=u16::MAX as u32 {
        let value = f32::from_bits(bits << 16);
        let result = GroupedInt4Matrix::from_bf16(1, 2, 2, &bf16(&[value, 0.0]), exact(1, 2, 2), &mut Continue);
        if value.is_finite() {
            let matrix = result.unwrap();
            assert!(matrix.group_scales()[0].is_finite() && matrix.group_scales()[0] > 0.0);
            let low = matrix.packed_bytes()[0] & 15;
            assert_ne!(low, 8); // This producer reserves -8; imported kernels still accept it.
            assert_eq!(matrix.packed_bytes()[0] >> 4, 0);
        } else { assert!(matches!(result, Err(Int4ConversionError::NonFiniteSource))); }
    }
}

#[test]
fn geometry_and_each_budget_axis_are_checked_before_the_source_is_touched() {
    let full = exact(2, 3, 2);
    for axis in 0..3 {
        let mut budget = full;
        match axis { 0 => budget.max_payload_bytes -= 1, 1 => budget.max_source_values -= 1, _ => budget.max_groups -= 1 }
        let mut reads = 0;
        let result = GroupedInt4Matrix::from_bf16_rows(2, 3, 2, budget, &mut Continue,
            |_, _| { reads += 1; Ok::<(), ()>(()) });
        assert!(matches!(result, Err(Int4ConversionError::Budget))); assert_eq!(reads, 0);
    }
    for (rows, columns, width) in [(usize::MAX, 2, 2), (1, MAX_MODEL_K + 1, 2), (0, 2, 2),
        (1, 1, 1), (1, 2, 0), (1, 2, MAX_MODEL_K + 1)] {
        let mut reads = 0;
        assert!(GroupedInt4Matrix::from_bf16_rows(rows, columns, width, full, &mut Continue,
            |_, _| { reads += 1; Ok::<(), ()>(()) }).is_err());
        assert_eq!(reads, 0);
    }
    assert!(GroupedInt4Matrix::from_bf16(2, 3, 2, &bf16(&[1.0; 5]), full, &mut Continue).is_err());
}

#[test]
fn incomplete_rows_cannot_reuse_stale_source_values_and_source_errors_do_not_leak_in_display() {
    let mut reads = 0;
    let incomplete = GroupedInt4Matrix::from_bf16_rows(2, 2, 2, exact(2, 2, 2), &mut Continue,
        |row, out| { reads += 1; if row == 0 { out.fill(Bf16::from_f32(1.0)); }
            else { out[0] = Bf16::from_f32(2.0); } Ok::<(), ()>(()) });
    assert!(matches!(incomplete, Err(Int4ConversionError::NonFiniteSource))); assert_eq!(reads, 2);
    let failed = GroupedInt4Matrix::from_bf16_rows(2, 2, 2, exact(2, 2, 2), &mut Continue,
        |_, _| Err("private source text"));
    let Err(error) = failed else { panic!("source failure must not return a matrix") };
    assert!(!error.to_string().contains("private"));
}

#[test]
fn cancellation_before_and_after_reads_never_returns_a_partial_matrix() {
    struct Stop { calls: usize, at: usize }
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1; (self.calls == self.at).then_some(DecodeCancellationKind::Deadline)
        }
    }
    for (at, expected_reads) in [(1, 0), (3, 1)] {
        let mut reads = 0; let mut stop = Stop { calls: 0, at };
        let result = GroupedInt4Matrix::from_bf16_rows(2, 2, 2, exact(2, 2, 2), &mut stop,
            |_, out| { reads += 1; out.fill(Bf16::from_f32(1.0)); Ok::<(), ()>(()) });
        assert!(matches!(result, Err(Int4ConversionError::Cancelled(DecodeCancellationKind::Deadline))));
        assert_eq!(reads, expected_reads);
    }
}

#[test]
fn matrix_identity_binds_shape_partition_and_scale_bits_not_only_packed_bytes() {
    let a = GroupedInt4Matrix::from_bf16(1, 4, 4, &bf16(&[1.0; 4]), exact(1, 4, 4), &mut Continue).unwrap();
    let b = GroupedInt4Matrix::from_bf16(2, 2, 2, &bf16(&[1.0; 4]), exact(2, 2, 2), &mut Continue).unwrap();
    let c = GroupedInt4Matrix::from_bf16(1, 4, 4, &bf16(&[2.0; 4]), exact(1, 4, 4), &mut Continue).unwrap();
    assert_eq!(a.packed_bytes(), b.packed_bytes()); assert_eq!(a.packed_bytes(), c.packed_bytes());
    assert_ne!(a.content_sha256(), b.content_sha256()); assert_ne!(a.content_sha256(), c.content_sha256());
}
