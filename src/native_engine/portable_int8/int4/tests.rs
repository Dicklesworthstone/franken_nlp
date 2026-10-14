use super::*;

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn activation(values: &[f32]) -> ActivationBuffer {
    let mut buffer = ActivationBuffer::try_new(values.len()).unwrap();
    buffer.encode_f32(values).unwrap(); buffer
}
fn pack(values: &[i8]) -> Vec<u8> {
    assert_eq!(values.len() % 2, 0);
    values.chunks_exact(2).map(|pair| (pair[0] as u8 & 15) | ((pair[1] as u8 & 15) << 4)).collect()
}
fn run(matrix: GroupedInt4Linear<'_>, input: &ActivationBuffer, rows: LinearRows<'_>) -> Vec<f32> {
    let mut output = vec![0.0; rows.checked_count(matrix.rows()).unwrap()];
    let mut ledger = ProjectionLedger::new(matrix.planned_work(rows).unwrap());
    let mut scratch = vec![99; matrix.scratch_len()];
    matrix.project_f32_into(input, rows, &mut output, &mut scratch, &mut ledger, &mut Continue).unwrap();
    assert_eq!(ledger.remaining(), ProjectionWork::default()); output
}
fn reference(input: &ActivationBuffer, weights: &[i8], scales: &[f32], width: usize) -> f32 {
    let mut total = 0.0_f32;
    for (group, (a, w)) in input.values().unwrap().chunks(width).zip(weights.chunks(width)).enumerate() {
        let integer: i64 = a.iter().zip(w).map(|(&a, &w)| i64::from(a) * i64::from(w)).sum();
        let partial = (((integer as f32) * input.scale().unwrap()) * 1.0) * 1.0 * scales[group];
        total += partial;
    }
    total
}

#[test]
fn unequal_group_scales_are_applied_before_summing() {
    let matrix = GroupedInt4Linear::checked(1, 4, 2, &[0x11, 0x11], &[1.0, 10.0]).unwrap();
    let input = activation(&[127.0, 127.0, 127.0, 127.0]);
    assert_eq!(run(matrix, &input, LinearRows::All), [2794.0]);
    assert_eq!(matrix.planned_work(LinearRows::All).unwrap(), ProjectionWork { dot_products: 2, multiply_accumulates: 4 });
}

#[test]
fn odd_group_and_row_boundaries_and_model_widths_match_independent_i64_reference() {
    for k in [1, 2, 3, 7, 8, 15, 16, 17, 33, 65, 3072, 6144, 10752] {
        let n = 4;
        let weights: Vec<_> = (0..n * k).map(|i| ((i * 7 + 3) % 16) as i8 - 8).collect();
        let packed = pack(&weights);
        let values: Vec<_> = (0..k).map(|i| ((i * 37) % 255) as f32 - 127.0).collect();
        let input = activation(&values);
        for width in [1, 3, 16, 32, 128, MAX_MODEL_K] {
            let groups = k.div_ceil(width);
            let scales: Vec<_> = (0..n * groups).map(|i| (i % 7 + 1) as f32 / 32.0).collect();
            let matrix = GroupedInt4Linear::checked(n, k, width, &packed, &scales).unwrap();
            let actual = run(matrix, &input, LinearRows::All);
            for row in 0..n {
                let expected = reference(&input, &weights[row * k..(row + 1) * k],
                    &scales[row * groups..(row + 1) * groups], width);
                assert_eq!(actual[row].to_bits(), expected.to_bits(), "K={k} group={width} row={row}");
            }
            assert_eq!(run(matrix, &input, LinearRows::Selected(&[1, 3])), [actual[1], actual[3]]);
        }
    }
}

#[test]
fn every_group_scale_is_validated_and_unselected_rows_are_not_executed() {
    for bad in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(GroupedInt4Linear::checked(2, 2, 1, &[0x11, 0x11], &[1.0, 1.0, 1.0, bad]), Err(GroupedInt4Error::Scale)));
    }
    let matrix = GroupedInt4Linear::checked(2, 2, 2, &[0x11, 0x77], &[1.0, f32::MAX]).unwrap();
    let input = activation(&[127.0, 127.0]);
    assert_eq!(run(matrix, &input, LinearRows::Selected(&[0])), [254.0]);
    let mut budget = ProjectionLedger::new(matrix.planned_work(LinearRows::All).unwrap());
    assert!(matrix.project_f32_into(&input, LinearRows::All, &mut [0.0; 2], &mut [0; 2], &mut budget, &mut Continue).is_err());
}

#[test]
fn invalid_geometry_groups_selection_and_scratch_refuse_without_charging_or_writes() {
    for (n, k, width, packed, scales) in [
        (0, 2, 1, &[][..], &[][..]), (1, 1, 1, &[0x11][..], &[1.0][..]),
        (1, 2, 0, &[0x11][..], &[1.0][..]), (1, 2, MAX_MODEL_K + 1, &[0x11][..], &[1.0][..]),
        (1, 2, 1, &[0x11][..], &[1.0][..]), (1, 2, 2, &[][..], &[1.0][..]),
    ] { assert!(GroupedInt4Linear::checked(n, k, width, packed, scales).is_err()); }
    let matrix = GroupedInt4Linear::checked(2, 2, 2, &[0x11, 0x11], &[1.0, 1.0]).unwrap();
    let input = activation(&[127.0, 1.0]);
    for mode in 0..6 {
        let mut output = [91.0; 2]; let mut scratch = [99; 2];
        let mut budget = ProjectionLedger::new(ProjectionWork { dot_products: 10, multiply_accumulates: 100 });
        let selection = match mode { 0 => LinearRows::Selected(&[1, 0]), 1 => LinearRows::Selected(&[0, 0]),
            2 => LinearRows::Selected(&[0, 2]), _ => LinearRows::All };
        let length = if mode == 3 { 1 } else { 2 };
        let scratch_len = if mode == 4 { 1 } else { 2 };
        let invalid = ActivationBuffer::try_new(2).unwrap();
        assert!(matrix.project_f32_into(if mode == 5 { &invalid } else { &input }, selection,
            &mut output[..length], &mut scratch[..scratch_len], &mut budget, &mut Continue).is_err());
        assert_eq!(output, [91.0; 2]); assert_eq!(scratch, [99; 2]);
        assert_eq!(budget.reserved(), ProjectionWork::default());
    }
}

#[test]
fn group_dot_and_mac_ceilings_are_independent_and_precede_work() {
    let matrix = GroupedInt4Linear::checked(1, 4, 3, &[0x11, 0x11], &[1.0; 2]).unwrap();
    for cap in [ProjectionWork { dot_products: 1, multiply_accumulates: 4 },
        ProjectionWork { dot_products: 2, multiply_accumulates: 3 }] {
        let mut ledger = ProjectionLedger::new(cap); let mut output = [17.0]; let mut scratch = [42; 3];
        assert!(matches!(matrix.project_f32_into(&activation(&[1.0; 4]), LinearRows::All,
            &mut output, &mut scratch, &mut ledger, &mut Continue), Err(GroupedInt4Error::Linear(LinearError::WorkBudget))));
        assert_eq!(ledger.remaining(), cap); assert_eq!(output, [17.0]); assert_eq!(scratch, [42; 3]);
    }
}

#[test]
fn cancellation_charges_complete_work_and_polls_within_one_long_row() {
    struct Stop { calls: usize }
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1;
            (self.calls == 2).then_some(DecodeCancellationKind::Deadline)
        }
    }
    let matrix = GroupedInt4Linear::checked(1, 66, 1, &[0x11; 33], &[1.0; 66]).unwrap();
    let expected = matrix.planned_work(LinearRows::All).unwrap();
    let mut ledger = ProjectionLedger::new(expected); let mut output = [43.0]; let mut stop = Stop { calls: 0 };
    assert_eq!(matrix.project_f32_into(&activation(&[1.0; 66]), LinearRows::All,
        &mut output, &mut [0], &mut ledger, &mut stop),
        Err(GroupedInt4Error::Linear(LinearError::Cancelled(DecodeCancellationKind::Deadline))));
    assert_eq!(stop.calls, 2); assert_eq!(output, [43.0]); assert_eq!(ledger.reserved(), expected);
}

#[test]
fn bf16_cast_occurs_only_after_the_full_ordered_group_sum() {
    let packed = [0xf1]; let scales = [1.001, 1.0];
    let matrix = GroupedInt4Linear::checked(1, 2, 1, &packed, &scales).unwrap();
    let input = activation(&[127.0, 127.0]); let full = run(matrix, &input, LinearRows::All);
    let mut output = [Bf16::from_f32(0.0)];
    matrix.project_bf16_into(&input, LinearRows::All, &mut output, &mut [0],
        &mut ProjectionLedger::new(matrix.planned_work(LinearRows::All).unwrap()), &mut Continue).unwrap();
    assert_eq!(output[0], Bf16::from_f32(full[0]));
    assert_ne!(output[0], Bf16::from_f32(0.0)); // Casting each group first would cancel to zero.
    let overflow = GroupedInt4Linear::checked(1, 2, 2, &[0x01], &[f32::MAX]).unwrap();
    assert!(overflow.project_bf16_into(&activation(&[1.0, 0.0]), LinearRows::All, &mut output, &mut [0; 2],
        &mut ProjectionLedger::new(overflow.planned_work(LinearRows::All).unwrap()), &mut Continue).is_err());
}

#[test]
fn available_native_leaves_match_scalar_group_epilogues_bit_for_bit() {
    let weights: Vec<_> = (0..4 * 130).map(|i| ((i * 3 + 8) % 16) as i8 - 8).collect();
    let packed = pack(&weights); let scales = vec![0.0625; 4 * 130_usize.div_ceil(33)];
    let matrix = GroupedInt4Linear::checked(4, 130, 33, &packed, &scales).unwrap();
    let input = activation(&vec![-127.0; 130]); let expected = run(matrix, &input, LinearRows::All);
    for kind in Int8KernelKind::ALL {
        match PackedInt4Kernel::detect(kind) {
            Ok(kernel) => {
                let native = matrix.with_kernel(kernel); assert_eq!(native.kernel_kind(), kind);
                assert_eq!(run(native, &input, LinearRows::All), expected);
            }
            Err(CpuKernelError::UnsupportedIsa) => eprintln!("GROUPED_INT4 tier={kind} RESULT=SKIPPED_UNSUPPORTED_ISA"),
            Err(error) => panic!("unexpected kernel error: {error}"),
        }
    }
}

#[test]
fn raw_group_ranges_cover_both_nibbles_and_refuse_before_writes() {
    let kernel = PackedInt4Kernel::scalar();
    for byte in 0_u16..=255 {
        for value in -128_i16..=127 {
            for offset in [0, 1] {
                let mut output = [99]; let mut scratch = [88];
                kernel.dot_rows_range_into(&[&[value as i8]], &[byte as u8], offset, 1,
                    &mut output, &mut scratch).unwrap();
                let nibble = ((byte as u8) >> (4 * offset)) & 15;
                let signed = if nibble < 8 { i32::from(nibble) } else { i32::from(nibble) - 16 };
                assert_eq!(output, [i32::from(value) * signed]);
            }
        }
    }
    for (offset, width, actual) in [(usize::MAX, 1, 1), (2, 1, 1), (0, 3, 2), (0, 1, 0)] {
        let mut output = [99]; let mut scratch = [88; 2]; let values = [1; 2];
        assert!(kernel.dot_rows_range_into(&[&values[..actual]], &[0xff], offset, width,
            &mut output, &mut scratch).is_err());
        assert_eq!(output, [99]); assert_eq!(scratch, [88; 2]);
    }
    let mut output = [99];
    kernel.dot_rows_range_into(&[&[]], &[0xff], 2, 0, &mut output, &mut []).unwrap();
    assert_eq!(output, [0]);
}
