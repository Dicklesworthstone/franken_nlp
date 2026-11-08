use super::*;

#[derive(Default)]
struct Continue { calls: usize }
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { self.calls += 1; None }
}
fn activation(values: &[f32]) -> ActivationBuffer {
    let mut input = ActivationBuffer::try_new(values.len()).unwrap();
    input.encode_f32(values).unwrap(); input
}
fn pack(values: &[i8]) -> Vec<u8> {
    values.chunks_exact(2).map(|pair| (pair[0] as u8 & 15) | ((pair[1] as u8 & 15) << 4)).collect()
}

#[test]
fn every_static_partition_preserves_serial_bits_for_full_and_sparse_rows() {
    let n = 6;
    for k in [1_usize, 3, 7, 33, 130, 3072, 6144, 10752] {
        let weights: Vec<_> = (0..n * k).map(|i| ((i * 7 + 3) % 16) as i8 - 8).collect();
        let packed = pack(&weights);
        let values: Vec<_> = (0..k).map(|i| ((i * 37) % 255) as f32 - 127.0).collect();
        let input = activation(&values);
        for group_width in [1, 3, 32, 128, MAX_MODEL_K] {
            let groups = k.div_ceil(group_width);
            let scales: Vec<_> = (0..n * groups).map(|i| (i % 7 + 1) as f32 / 32.0).collect();
            let matrix = GroupedInt4Linear::checked(n, k, group_width, &packed, &scales).unwrap();
            for rows in [LinearRows::All, LinearRows::Selected(&[0, 2, 5])] {
                let count = rows.checked_count(n).unwrap();
                let mut expected = vec![0.0; count];
                matrix.project_f32_into(&input, rows, &mut expected, &mut vec![0; matrix.scratch_len()],
                    &mut ProjectionLedger::new(matrix.planned_work(rows).unwrap()), &mut Continue::default()).unwrap();
                for (index, &actual) in expected.iter().enumerate() {
                    let row = rows.row(index);
                    let mut total = 0.0_f32;
                    for group in 0..groups {
                        let start = group * group_width;
                        let end = (start + group_width).min(k);
                        let integer: i64 = (start..end).map(|column| i64::from(input.values().unwrap()[column])
                            * i64::from(weights[row * k + column])).sum();
                        total += (((integer as f32) * input.scale().unwrap()) * 1.0) * 1.0
                            * scales[row * groups + group];
                    }
                    assert_eq!(actual.to_bits(), total.to_bits());
                }
                for width in [1, 2, 3, 8] {
                    let mut actual = vec![0.0_f32; count];
                    let mut bf16 = vec![Bf16::from_bits(0); count];
                    for ordinal in 0..width {
                        let range = (count * ordinal / width)..(count * (ordinal + 1) / width);
                        matrix.project_f32_range_into(input.values().unwrap(), input.scale().unwrap(), rows,
                            range.clone(), &mut actual[range.clone()], &mut vec![0; matrix.scratch_len()],
                            &mut Continue::default()).unwrap();
                        matrix.project_bf16_range_into(input.values().unwrap(), input.scale().unwrap(), rows,
                            range.clone(), &mut bf16[range], &mut vec![0; matrix.scratch_len()],
                            &mut Continue::default()).unwrap();
                    }
                    assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                    assert_eq!(bf16, expected.iter().copied().map(Bf16::from_f32).collect::<Vec<_>>());
                }
            }
        }
    }
}

#[test]
fn malformed_ranges_and_raw_activations_refuse_before_checkpoint_or_writes() {
    let matrix = GroupedInt4Linear::checked(4, 3, 2, &[0x11; 6], &[1.0; 8]).unwrap();
    for mode in 0..12 {
        let input = [1_i8; 3];
        let input_len = if mode == 0 { 2 } else { 3 };
        let scale = match mode { 1 => 0.0, 2 => -1.0, 3 => f32::NAN, 4 => f32::INFINITY, _ => 1.0 };
        let rows = match mode { 5 => LinearRows::Selected(&[1, 1]), 6 => LinearRows::Selected(&[1, 4]),
            _ => LinearRows::All };
        let range = match mode { 7 => Range { start: 2, end: 1 }, 8 => 3..5, 9 => 0..usize::MAX, _ => 1..3 };
        let length = if mode == 10 { 1 } else { 2 };
        let scratch_len = if mode == 11 { 1 } else { 2 };
        let mut output = [91.0; 2]; let mut bf16 = [Bf16::from_f32(91.0); 2];
        let mut scratch = [88; 2]; let mut control = Continue::default();
        assert!(matrix.project_f32_range_into(&input[..input_len], scale, rows, range.clone(),
            &mut output[..length], &mut scratch[..scratch_len], &mut control).is_err());
        assert!(matrix.project_bf16_range_into(&input[..input_len], scale, rows, range,
            &mut bf16[..length], &mut scratch[..scratch_len], &mut control).is_err());
        assert_eq!(output, [91.0; 2]); assert_eq!(bf16, [Bf16::from_f32(91.0); 2]);
        assert_eq!(scratch, [88; 2]); assert_eq!(control.calls, 0);
    }
}

#[test]
fn a_cancelled_range_keeps_the_complete_projection_debit_and_unpublished_output() {
    struct Stop { calls: usize }
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1;
            (self.calls == 2).then_some(DecodeCancellationKind::Deadline)
        }
    }
    let matrix = GroupedInt4Linear::checked(2, 66, 1, &[0x11; 66], &[1.0; 132]).unwrap();
    let input = activation(&[1.0; 66]);
    let work = matrix.planned_work(LinearRows::All).unwrap();
    let mut ledger = ProjectionLedger::new(work);
    matrix.reserve_projection(&input, LinearRows::All, 2, &mut ledger).unwrap();
    let mut output = [43.0]; let mut control = Stop { calls: 0 };
    assert_eq!(matrix.project_f32_range_into(input.values().unwrap(), input.scale().unwrap(),
        LinearRows::All, 1..2, &mut output, &mut [0], &mut control),
        Err(GroupedInt4Error::Linear(LinearError::Cancelled(DecodeCancellationKind::Deadline))));
    assert_eq!(control.calls, 2); assert_eq!(output, [43.0]);
    assert_eq!(ledger.reserved(), work); assert_eq!(ledger.remaining(), ProjectionWork::default());
}

#[test]
fn range_cast_stays_after_the_complete_ordered_group_sum() {
    let matrix = GroupedInt4Linear::checked(2, 2, 1, &[0xf1; 2], &[1.001, 1.0, 1.001, 1.0]).unwrap();
    let input = activation(&[127.0; 2]);
    let mut f32_output = [0.0]; let mut bf16_output = [Bf16::from_bits(0)];
    matrix.project_f32_range_into(input.values().unwrap(), input.scale().unwrap(), LinearRows::All,
        1..2, &mut f32_output, &mut [0], &mut Continue::default()).unwrap();
    matrix.project_bf16_range_into(input.values().unwrap(), input.scale().unwrap(), LinearRows::All,
        1..2, &mut bf16_output, &mut [0], &mut Continue::default()).unwrap();
    assert_eq!(bf16_output[0], Bf16::from_f32(f32_output[0]));
    assert_ne!(bf16_output[0], Bf16::from_bits(0));
}

#[test]
fn available_simd_ranges_preserve_full_and_sparse_scalar_bits() {
    let values: Vec<_> = (0..4 * 130).map(|i| ((i * 3 + 8) % 16) as i8 - 8).collect();
    let packed = pack(&values); let scales = vec![0.0625; 4 * 130_usize.div_ceil(33)];
    let matrix = GroupedInt4Linear::checked(4, 130, 33, &packed, &scales).unwrap();
    let input = activation(&[-127.0; 130]);
    for rows in [LinearRows::All, LinearRows::Selected(&[1, 3])] {
        let count = rows.checked_count(4).unwrap(); let mut expected = vec![0.0; count];
        matrix.project_f32_into(&input, rows, &mut expected, &mut [0; 33],
            &mut ProjectionLedger::new(matrix.planned_work(rows).unwrap()), &mut Continue::default()).unwrap();
        for kind in Int8KernelKind::ALL {
            let kernel = match PackedInt4Kernel::detect(kind) {
                Ok(kernel) => kernel,
                Err(CpuKernelError::UnsupportedIsa) => continue,
                Err(error) => panic!("unexpected kernel error: {error}"),
            };
            let native = matrix.with_kernel(kernel); let mut output = vec![0.0_f32; count];
            for index in 0..count {
                native.project_f32_range_into(input.values().unwrap(), input.scale().unwrap(), rows,
                    index..index + 1, &mut output[index..index + 1], &mut [0; 33], &mut Continue::default()).unwrap();
            }
            assert_eq!(output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
    }
}
