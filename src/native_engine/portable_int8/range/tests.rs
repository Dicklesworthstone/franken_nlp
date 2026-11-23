//! Independent integer goldens and serial/range regression definitions.
//! These source fixtures are not an executed parity or performance receipt.
use super::*;

#[derive(Default)]
struct Continue { polls: usize }
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { self.polls += 1; None }
}
fn activation(k: usize) -> ActivationBuffer {
    let mut input = ActivationBuffer::try_new(k).unwrap();
    let values: Vec<_> = (0..k).map(|i| ((i * 37 + 128) % 255) as f32 - 127.0).collect();
    input.encode_f32(&values).unwrap(); input
}
fn sums(values: &[i8], k: usize) -> Vec<i32> {
    values.chunks_exact(k).map(|row| row.iter().map(|&v| i32::from(v)).sum()).collect()
}

#[test]
fn partitioned_full_and_selected_rows_match_i64_goldens_and_both_serial_rails() {
    let n = 7;
    for k in [1, 3, 31, 32, 33, 130, 3072, 6144, MAX_MODEL_K] {
        let values: Vec<i8> = (0..n * k).map(|i| ((i * 53 + 128) % 256) as u8 as i8).collect();
        let scales: Vec<_> = (0..n).map(|row| (row + 1) as f32 / 32.0).collect();
        let row_sums = sums(&values, k);
        let matrix = QuantizedLinear::checked(n, k, &values, &scales, &row_sums).unwrap();
        let input = activation(k);
        for rows in [LinearRows::All, LinearRows::Selected(&[0, 2, 6])] {
            let count = rows.checked_count(n).unwrap();
            let work = ProjectionWork::for_shape(count, k).unwrap();
            let mut expected = vec![0.0_f32; count];
            matrix.project_f32_into(&input, rows, &mut expected,
                &mut ProjectionLedger::new(work), &mut Continue::default()).unwrap();
            let mut expected_bf16 = vec![Bf16::default(); count];
            matrix.project_bf16_into(&input, rows, &mut expected_bf16,
                &mut ProjectionLedger::new(work), &mut Continue::default()).unwrap();
            for (index, &value) in expected.iter().enumerate() {
                let row = rows.row(index);
                let integer: i64 = input.values().unwrap().iter().zip(&values[row * k..(row + 1) * k])
                    .map(|(&a, &b)| i64::from(a) * i64::from(b)).sum();
                let golden = (((integer as f32) * input.scale().unwrap()) * scales[row]) * 1.0 * 1.0;
                assert_eq!(value.to_bits(), golden.to_bits());
            }
            for width in [1, 2, 3, 8] {
                let mut actual = vec![0.0_f32; count];
                let mut bf16 = vec![Bf16::default(); count];
                for ordinal in 0..width {
                    let range = count * ordinal / width..count * (ordinal + 1) / width;
                    matrix.project_f32_range_into(input.values().unwrap(), input.scale().unwrap(), rows,
                        range.clone(), &mut actual[range.clone()], &mut Continue::default()).unwrap();
                    matrix.project_bf16_range_into(input.values().unwrap(), input.scale().unwrap(), rows,
                        range.clone(), &mut bf16[range], &mut Continue::default()).unwrap();
                }
                assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                assert_eq!(bf16, expected_bf16);
            }
        }
    }
}

#[test]
fn available_checked_kernels_keep_serial_bits_at_odd_and_selected_shard_edges() {
    let k = 130; let n = 7;
    let values: Vec<i8> = (0..n * k).map(|i| ((i * 97 + 128) % 256) as u8 as i8).collect();
    let scales = [0.03125, 0.0625, 0.1, 0.25, 0.5, 0.75, 1.25]; let row_sums = sums(&values, k);
    let matrix = QuantizedLinear::checked(n, k, &values, &scales, &row_sums).unwrap();
    let input = activation(k);
    for rows in [LinearRows::All, LinearRows::Selected(&[0, 2, 6])] {
        let count = rows.checked_count(n).unwrap();
        let mut expected = vec![0.0_f32; count];
        matrix.project_f32_into(&input, rows, &mut expected,
            &mut ProjectionLedger::new(ProjectionWork::for_shape(count, k).unwrap()), &mut Continue::default()).unwrap();
        for kind in Int8KernelKind::ALL {
            let kernel = match Int8Kernel::detect(kind) {
                Ok(kernel) => kernel,
                Err(CpuKernelError::UnsupportedIsa) => continue,
                Err(error) => panic!("unexpected kernel error: {error}"),
            };
            let matrix = matrix.with_kernel(kernel);
            let mut actual = vec![0.0_f32; count];
            let mut bf16 = vec![Bf16::default(); count];
            for range in [0..1, 1..count] {
                matrix.project_f32_range_into(input.values().unwrap(), input.scale().unwrap(), rows,
                    range.clone(), &mut actual[range.clone()], &mut Continue::default()).unwrap();
                matrix.project_bf16_range_into(input.values().unwrap(), input.scale().unwrap(), rows,
                    range.clone(), &mut bf16[range], &mut Continue::default()).unwrap();
            }
            assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{kind:?}");
            assert_eq!(bf16, expected.iter().copied().map(Bf16::from_f32).collect::<Vec<_>>());
        }
    }
}

#[test]
fn malformed_raw_inputs_selections_and_ranges_refuse_before_polls_or_writes() {
    let matrix = QuantizedLinear::checked(4, 3, &[1; 12], &[1.0; 4], &[3; 4]).unwrap();
    for mode in 0..12 {
        let input = [1_i8; 3]; let length = if mode == 0 { 2 } else { 3 };
        let scale = match mode { 1 => 0.0, 2 => -1.0, 3 => f32::NAN, 4 => f32::INFINITY, _ => 1.0 };
        let rows = match mode { 5 => LinearRows::Selected(&[1, 1]), 6 => LinearRows::Selected(&[1, 4]),
            7 => LinearRows::Selected(&[]), _ => LinearRows::All };
        let range = match mode { 8 => Range { start: 2, end: 1 }, 9 => 3..5, 10 => 0..usize::MAX, _ => 1..3 };
        let output_len = if mode == 11 { 1 } else { 2 };
        let mut output = [91.0; 2]; let mut bf16 = [Bf16::from_f32(91.0); 2];
        let mut control = Continue::default();
        assert!(matrix.project_f32_range_into(&input[..length], scale, rows, range.clone(),
            &mut output[..output_len], &mut control).is_err());
        assert!(matrix.project_bf16_range_into(&input[..length], scale, rows, range,
            &mut bf16[..output_len], &mut control).is_err());
        assert_eq!(output, [91.0; 2]); assert_eq!(bf16, [Bf16::from_f32(91.0); 2]);
        assert_eq!(control.polls, 0);
    }
}

#[test]
fn cancellation_keeps_the_complete_debit_and_every_original_terminal_cause() {
    struct Stop { cause: DecodeCancellationKind, polls: usize }
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.polls += 1; (self.polls == 2).then_some(self.cause)
        }
    }
    let matrix = QuantizedLinear::checked(65, 1, &[1; 65], &[1.0; 65], &[1; 65]).unwrap();
    let mut input = ActivationBuffer::try_new(1).unwrap(); input.encode_f32(&[127.0]).unwrap();
    let work = ProjectionWork::for_shape(65, 1).unwrap();
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let mut ledger = ProjectionLedger::new(work);
        matrix.reserve_projection(&input, LinearRows::All, 65, &mut ledger).unwrap();
        let mut output = [43.0; 65]; let mut control = Stop { cause, polls: 0 };
        assert_eq!(matrix.project_f32_range_into(input.values().unwrap(), input.scale().unwrap(),
            LinearRows::All, 0..65, &mut output, &mut control), Err(LinearError::Cancelled(cause)));
        assert_eq!(control.polls, 2); assert_eq!(&output[..32], &[127.0; 32]);
        assert_eq!(&output[32..], &[43.0; 33]);
        assert_eq!(ledger.reserved(), work); assert_eq!(ledger.remaining(), ProjectionWork::default());
    }
}

#[test]
fn empty_shards_are_safe_and_the_bf16_overflow_boundary_is_unchanged() {
    let matrix = QuantizedLinear::checked(1, 1, &[1], &[f32::MAX], &[1]).unwrap();
    let mut control = Continue::default();
    matrix.project_f32_range_into(&[1], 1.0, LinearRows::All, 1..1, &mut [], &mut control).unwrap();
    assert_eq!(control.polls, 1);
    let mut float = [0.0];
    matrix.project_f32_range_into(&[1], 1.0, LinearRows::All, 0..1, &mut float, &mut control).unwrap();
    assert_eq!(float[0], f32::MAX);
    let mut bf16 = [Bf16::from_f32(9.0)];
    assert_eq!(matrix.project_bf16_range_into(&[1], 1.0, LinearRows::All, 0..1, &mut bf16, &mut control),
        Err(LinearError::NonFiniteOutput));
    assert_eq!(bf16, [Bf16::from_f32(9.0)]);
}
