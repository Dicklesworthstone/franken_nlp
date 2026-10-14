use super::*;

#[derive(Default)]
struct Control { polls: usize, cancel: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1;
        (self.cancel == Some(self.polls)).then_some(DecodeCancellationKind::User)
    }
}
fn activation(k: usize, lane: usize) -> ActivationBuffer {
    let mut input = ActivationBuffer::try_new(k).unwrap();
    let values: Vec<_> = (0..k).map(|i| Bf16::from_f32(
        (((i * 11 + lane * 3) % 29) as f32 - 14.0) * (lane + 1) as f32 / 16.0)).collect();
    input.encode_bf16(&values).unwrap(); input
}
fn packed(n: usize, k: usize) -> Vec<u8> {
    (0..n * k / 2).map(|i| {
        let lo = ((2 * i * 7) % 16) as u8;
        let hi = (((2 * i + 1) * 7) % 16) as u8;
        lo | (hi << 4)
    }).collect()
}
fn budget(matrix: GroupedInt4Linear<'_>, rows: LinearRows<'_>, copies: usize) -> ProjectionWork {
    let one = matrix.planned_work(rows).unwrap();
    ProjectionWork { dot_products: one.dot_products * copies as u64,
        multiply_accumulates: one.multiply_accumulates * copies as u64 }
}
fn parity(kernel: PackedInt4Kernel) {
    for k in [1, 3, 7, 32, 129] { for width in [1, 3, 8, 128] {
        let n = 6; let bytes = packed(n, k);
        let scales: Vec<_> = (0..n * k.div_ceil(width)).map(|i| (i % 9 + 1) as f32 / 32.0).collect();
        let matrix = GroupedInt4Linear::checked(n, k, width, &bytes, &scales).unwrap();
        let selected = [0, 2, 5];
        for rows in [LinearRows::All, LinearRows::Selected(&selected)] {
            let count = rows.checked_count(n).unwrap();
            for m in [1, 3, 4, 5, 64] {
                let inputs: Vec<_> = (0..m).map(|lane| activation(k, lane)).collect();
                let work = budget(matrix, rows, m);
                let mut reference = vec![0.0_f32; m * count];
                let mut scalar_ledger = ProjectionLedger::new(work);
                for (input, out) in inputs.iter().zip(reference.chunks_mut(count)) {
                    matrix.project_f32_into(input, rows, out, &mut vec![0; width],
                        &mut scalar_ledger, &mut Control::default()).unwrap();
                }
                let mut actual = vec![0.0_f32; m * count];
                let mut ledger = ProjectionLedger::new(work);
                matrix.with_kernel(kernel).project_batch_f32_into(&inputs, rows, &mut actual,
                    &mut vec![0; width], &mut ledger, &mut Control::default()).unwrap();
                assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                assert_eq!(ledger.reserved(), scalar_ledger.reserved());
                let mut bf16 = vec![Bf16::from_bits(0); m * count];
                matrix.with_kernel(kernel).project_batch_bf16_into(&inputs, rows, &mut bf16,
                    &mut vec![0; width], &mut ProjectionLedger::new(work), &mut Control::default()).unwrap();
                assert_eq!(bf16.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    reference.iter().map(|&v| Bf16::from_f32(v).to_bits()).collect::<Vec<_>>());
            }
        }
    } }
}
#[test]
fn scalar_batch_matches_single_rows_bits_scales_tails_and_work() { parity(PackedInt4Kernel::scalar()); }
#[test]
fn every_available_checked_kernel_matches_scalar_batch() {
    for kind in Int8KernelKind::ALL {
        if kind == Int8KernelKind::Scalar { continue; }
        if let Ok(kernel) = PackedInt4Kernel::detect(kind) { parity(kernel); }
    }
}
#[test]
fn late_bad_activation_shape_output_scratch_and_work_refuse_without_side_effects() {
    let bytes = packed(2, 3); let scales = vec![0.25; 2];
    let matrix = GroupedInt4Linear::checked(2, 3, 3, &bytes, &scales).unwrap();
    for axis in 0..6 {
        let mut inputs = vec![activation(3, 0), activation(3, 1)];
        if axis == 0 { inputs[1] = ActivationBuffer::try_new(3).unwrap(); }
        if axis == 1 { inputs[1] = activation(2, 0); }
        let mut output = vec![123.0; if axis == 2 { 3 } else { 4 }];
        let mut scratch = vec![99_i8; if axis == 3 { 2 } else { 3 }];
        let mut work = budget(matrix, LinearRows::All, inputs.len());
        if axis == 4 { work.dot_products -= 1; }
        if axis == 5 { work.multiply_accumulates -= 1; }
        let mut ledger = ProjectionLedger::new(work); let mut control = Control::default();
        assert!(matrix.project_batch_f32_into(&inputs, LinearRows::All, &mut output,
            &mut scratch, &mut ledger, &mut control).is_err());
        assert_eq!(ledger.reserved(), ProjectionWork::default()); assert_eq!(control.polls, 0);
        assert!(output.iter().all(|&x| x == 123.0)); assert!(scratch.iter().all(|&x| x == 99));
    }
}
#[test]
fn cancellation_retains_full_group_dot_charge_and_never_retries() {
    let bytes = packed(2, 7); let scales = vec![0.125; 6];
    let matrix = GroupedInt4Linear::checked(2, 7, 3, &bytes, &scales).unwrap();
    let inputs: Vec<_> = (0..5).map(|lane| activation(7, lane)).collect();
    let work = budget(matrix, LinearRows::All, 5); let mut ledger = ProjectionLedger::new(work);
    let mut output = vec![123.0; 10]; let mut scratch = [99; 3];
    assert!(matches!(matrix.project_batch_f32_into(&inputs, LinearRows::All, &mut output,
        &mut scratch, &mut ledger, &mut Control { polls: 0, cancel: Some(1) }),
        Err(GroupedInt4Error::Linear(LinearError::Cancelled(DecodeCancellationKind::User)))));
    assert_eq!(ledger.reserved(), work); assert_eq!(output, vec![123.0; 10]); assert_eq!(scratch, [99; 3]);
}
#[test]
fn empty_oversized_and_invalid_selection_fail_before_charge() {
    let bytes = packed(2, 3); let matrix = GroupedInt4Linear::checked(2, 3, 3, &bytes, &[1.0; 2]).unwrap();
    for m in [0, 65] {
        let inputs: Vec<_> = (0..m).map(|i| activation(3, i)).collect();
        let mut ledger = ProjectionLedger::new(ProjectionWork::default());
        assert!(matrix.project_batch_f32_into(&inputs, LinearRows::All, &mut vec![0.0; m * 2],
            &mut [0; 3], &mut ledger, &mut Control::default()).is_err());
        assert_eq!(ledger.reserved(), ProjectionWork::default());
    }
    for selected in [vec![1, 0], vec![1, 1], vec![2]] {
        let mut ledger = ProjectionLedger::new(ProjectionWork::default());
        assert!(matrix.project_batch_f32_into(&[activation(3, 0)], LinearRows::Selected(&selected),
            &mut vec![0.0; selected.len()], &mut [0; 3], &mut ledger, &mut Control::default()).is_err());
        assert_eq!(ledger.reserved(), ProjectionWork::default());
    }
}
