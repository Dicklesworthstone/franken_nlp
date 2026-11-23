//! Definitions for byte identity, shard geometry and nonrenewable batch work.
//! These fixtures certify no throughput, hardware availability or model parity.
use super::*;
use crate::native_engine::simd::Int8Kernel;

#[derive(Default)]
struct Control { polls: usize, stop_at: Option<usize>, cause: Option<DecodeCancellationKind> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1;
        if self.stop_at == Some(self.polls) { self.cause } else { None }
    }
}
fn inputs(rows: usize, columns: usize) -> Vec<ActivationBuffer> {
    (0..rows).map(|row| {
        let mut input = ActivationBuffer::try_new(columns).unwrap();
        let values: Vec<_> = (0..columns).map(|column|
            if row % 7 == 0 { 0.0 } else { ((row * 19 + column * 13) % 255) as f32 - 127.0 }).collect();
        input.encode_f32(&values).unwrap(); input
    }).collect()
}
fn range(count: usize, width: usize, ordinal: usize) -> Range<usize> {
    let base = count / width; let remainder = count % width;
    let start = ordinal * base + ordinal.min(remainder);
    start..start + base + usize::from(ordinal < remainder)
}
fn compare(kernel: Int8Kernel, m: usize, k: usize, selected: bool) {
    let values: Vec<_> = (0..7 * k).map(|i| ((i * 53 + 128) % 256) as u8 as i8).collect();
    let scales = [0.03125, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25];
    let sums: Vec<i32> = values.chunks(k).map(|row| row.iter().map(|&v| i32::from(v)).sum()).collect();
    let matrix = QuantizedLinear::checked(7, k, &values, &scales, &sums).unwrap().with_kernel(kernel);
    let inputs = inputs(m, k);
    let raw: Vec<_> = inputs.iter().map(|input| input.values().unwrap()).collect();
    let activation_scales: Vec<_> = inputs.iter().map(|input| input.scale().unwrap()).collect();
    let rows = if selected { LinearRows::Selected(&[0, 3, 6]) } else { LinearRows::All };
    let n = rows.checked_count(7).unwrap();
    let work = ProjectionWork::for_shape(m * n, k).unwrap();
    let mut expected = vec![Bf16::default(); m * n];
    matrix.project_batch_bf16_into(&inputs, rows, &mut expected,
        &mut ProjectionLedger::new(work), &mut Control::default()).unwrap();
    for width in [1, 2, 3, 8] {
        let mut actual = vec![Bf16::from_f32(91.0); m * n];
        for ordinal in 0..width {
            let shard = range(n, width, ordinal);
            let span = if shard.is_empty() { 0 } else { (m - 1) * n + shard.len() };
            matrix.project_batch_bf16_range_into(&raw, &activation_scales, rows, shard.clone(),
                &mut actual[shard.start..shard.start + span], n, &mut Control::default()).unwrap();
            let mut compact = vec![Bf16::default(); m * shard.len()];
            matrix.project_batch_bf16_range_into(&raw, &activation_scales, rows, shard.clone(),
                &mut compact, shard.len(), &mut Control::default()).unwrap();
            for activation in 0..m {
                assert_eq!(&compact[activation * shard.len()..(activation + 1) * shard.len()],
                    &expected[activation * n + shard.start..activation * n + shard.end]);
            }
        }
        assert_eq!(actual, expected, "kernel {:?}, M {m}, K {k}, width {width}", kernel.kind());
    }
}
#[test]
fn all_available_kernels_keep_serial_batch_bits_across_cohort_column_and_k_tails() {
    for kind in Int8KernelKind::ALL {
        let Ok(kernel) = Int8Kernel::detect(kind) else { continue; };
        for m in [1, 3, 4, 5, MAX_BATCH_ROWS] {
            for k in [1, 3, 31, 32, 33] {
                for selected in [false, true] { compare(kernel, m, k, selected); }
            }
        }
        compare(kernel, 5, crate::native_engine::quant_algebra::MAX_MODEL_K, true);
    }
}
#[test]
fn full_i8_domain_matches_independent_i64_dots_without_saturation() {
    let k = crate::native_engine::quant_algebra::MAX_MODEL_K;
    let a: Vec<_> = (0..k).map(|i| [-128_i8, 127, -1, 0, 1][i % 5]).collect();
    let b: Vec<_> = (0..k).map(|i| [127_i8, -128, 1, -1, 0][i % 5]).collect();
    let weights: Vec<_> = (0..3 * k).map(|i| [-128_i8, -128, 127, 127, 1, 0, -1][i % 7]).collect();
    let scales = [0.03125, 0.125, 0.5];
    let sums: Vec<i32> = weights.chunks(k).map(|row| row.iter().map(|&v| i32::from(v)).sum()).collect();
    let raw = [&a[..], &b[..]]; let activation_scales = [0.25, 0.125];
    let expected: Vec<_> = raw.iter().zip(activation_scales).flat_map(|(input, activation)| {
        weights.chunks(k).zip(scales).map(move |(row, scale)| {
            let dot: i64 = input.iter().zip(row).map(|(&a, &b)| i64::from(a) * i64::from(b)).sum();
            Bf16::from_f32(dequantize_i32_fixed(i32::try_from(dot).unwrap(),
                EpilogueScales { activation, row: scale, column: 1.0, group: 1.0 }).unwrap())
        })
    }).collect();
    for kind in Int8KernelKind::ALL {
        let Ok(kernel) = Int8Kernel::detect(kind) else { continue; };
        let matrix = QuantizedLinear::checked(3, k, &weights, &scales, &sums).unwrap().with_kernel(kernel);
        let mut output = vec![Bf16::default(); 6];
        matrix.project_batch_bf16_range_into(&raw, &activation_scales, LinearRows::All, 0..3,
            &mut output, 3, &mut Control::default()).unwrap();
        assert_eq!(output, expected, "{kind:?}");
    }
}
#[test]
fn malformed_last_input_scale_selection_and_output_geometry_refuse_before_poll_or_write() {
    let matrix = QuantizedLinear::checked(2, 3, &[1; 6], &[1.0; 2], &[3; 2]).unwrap();
    let raw = [&[1_i8, 2, 3][..], &[3_i8, 2, 1][..]];
    for mode in 0..12 {
        let mut inputs = raw.to_vec(); let mut scales = vec![1.0, 1.0];
        let mut rows = LinearRows::All; let mut shard = 0..2; let mut stride = 2; let mut length = 4;
        match mode {
            0 => inputs.clear(),
            1 => inputs.resize(MAX_BATCH_ROWS + 1, raw[0]),
            2 => inputs[1] = &raw[1][..2],
            3 => scales[1] = f32::NAN,
            4 => scales[1] = 0.0,
            5 => { scales.pop(); },
            6 => rows = LinearRows::Selected(&[0, 0]),
            7 => rows = LinearRows::Selected(&[0, 2]),
            8 => shard = Range { start: 2, end: 1 },
            9 => shard = 0..3,
            10 => stride = 1,
            _ => length = 3,
        }
        let sentinel = Bf16::from_f32(91.0); let mut output = vec![sentinel; length]; let mut control = Control::default();
        assert!(matrix.project_batch_bf16_range_into(&inputs, &scales, rows, shard,
            &mut output, stride, &mut control).is_err(), "mode {mode}");
        assert_eq!(control.polls, 0); assert!(output.iter().all(|value| *value == sentinel));
    }
    let mut output = [Bf16::from_f32(91.0)]; let mut control = Control::default();
    assert_eq!(matrix.project_batch_bf16_range_into(&[raw[0]; 3], &[1.0; 3], LinearRows::All, 0..1,
        &mut output, usize::MAX, &mut control), Err(LinearError::OutputShape));
    assert_eq!(control.polls, 0);
}
#[test]
fn empty_shards_checkpoint_without_writing_and_strided_holes_stay_untouched() {
    let matrix = QuantizedLinear::checked(3, 1, &[1, 2, 3], &[1.0; 3], &[1, 2, 3]).unwrap();
    let raw = [&[2_i8][..], &[3_i8][..]];
    let sentinel = Bf16::from_f32(91.0); let mut output = [sentinel; 6];
    matrix.project_batch_bf16_range_into(&raw, &[1.0; 2], LinearRows::All, 1..3,
        &mut output, 4, &mut Control::default()).unwrap();
    assert_eq!(output, [Bf16::from_f32(4.0), Bf16::from_f32(6.0), sentinel, sentinel,
        Bf16::from_f32(6.0), Bf16::from_f32(9.0)]);
    let mut control = Control::default();
    matrix.project_batch_bf16_range_into(&raw, &[1.0; 2], LinearRows::All, 3..3,
        &mut [], usize::MAX, &mut control).unwrap();
    assert_eq!(control.polls, 1);
}
#[test]
fn complete_batch_is_reserved_once_and_every_rejected_input_spends_nothing() {
    let matrix = QuantizedLinear::checked(2, 3, &[1; 6], &[1.0; 2], &[3; 2]).unwrap();
    let work = ProjectionWork::for_shape(6, 3).unwrap();
    for mode in 0..5 {
        let mut input = inputs(3, 3); let mut cap = work; let mut length = 6;
        match mode {
            0 => { assert!(input[2].encode_f32(&[f32::NAN; 3]).is_err()); },
            1 => cap.dot_products -= 1,
            2 => cap.multiply_accumulates -= 1,
            3 => length -= 1,
            _ => input[2] = inputs(1, 2).pop().unwrap(),
        }
        let mut ledger = ProjectionLedger::new(cap);
        assert!(matrix.reserve_batch_projection(&input, LinearRows::All, length, &mut ledger).is_err());
        assert_eq!(ledger.reserved(), ProjectionWork::default()); assert_eq!(ledger.remaining(), cap);
    }
    let input = inputs(3, 3); let mut ledger = ProjectionLedger::new(work);
    matrix.reserve_batch_projection(&input, LinearRows::All, 6, &mut ledger).unwrap();
    assert_eq!(ledger.reserved(), work); assert_eq!(ledger.remaining(), ProjectionWork::default());
    assert_eq!(matrix.reserve_batch_projection(&input, LinearRows::All, 6, &mut ledger), Err(LinearError::WorkBudget));
}
#[test]
fn all_stop_causes_preserve_whole_batch_debit_and_partial_output_is_never_success() {
    let matrix = QuantizedLinear::checked(7, 3, &[1; 21], &[1.0; 7], &[3; 7]).unwrap();
    let inputs = inputs(MAX_BATCH_ROWS, 3);
    let raw: Vec<_> = inputs.iter().map(|input| input.values().unwrap()).collect();
    let scales: Vec<_> = inputs.iter().map(|input| input.scale().unwrap()).collect();
    let work = ProjectionWork::for_shape(MAX_BATCH_ROWS * 7, 3).unwrap();
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for stop_at in [1, 3] {
            let mut ledger = ProjectionLedger::new(work);
            let sentinel = Bf16::from_f32(91.0); let mut output = vec![sentinel; MAX_BATCH_ROWS * 7];
            matrix.reserve_batch_projection(&inputs, LinearRows::All, output.len(), &mut ledger).unwrap();
            let mut control = Control { polls: 0, stop_at: Some(stop_at), cause: Some(cause) };
            assert_eq!(matrix.project_batch_bf16_range_into(&raw, &scales, LinearRows::All, 0..7,
                &mut output, 7, &mut control), Err(LinearError::Cancelled(cause)));
            assert_eq!(control.polls, stop_at); assert_eq!(ledger.reserved(), work);
            if stop_at == 1 { assert!(output.iter().all(|value| *value == sentinel)); }
            else { assert!(output.iter().any(|value| *value != sentinel)); assert!(output.contains(&sentinel)); }
        }
    }
}
