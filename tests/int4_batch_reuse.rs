//! Group-major batch execution against an independent scalar floating program.
use franken_nlp::native_engine::{
    decode::{DecodeCancellationKind, DecodeStepControl},
    int4::PackedInt4Kernel,
    portable_int8::{ActivationBuffer, LinearRows, LinearError, ProjectionWork, ProjectionLedger,
        int4::{GroupedInt4Linear, GroupedInt4Error}},
    simd::{Int8KernelKind, CpuKernelError}, tensor::Bf16,
};
#[derive(Default)]
struct Control { polls: usize, cancel: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1;
        (self.cancel == Some(self.polls)).then_some(DecodeCancellationKind::User)
    }
}
fn kernels() -> Vec<PackedInt4Kernel> {
    Int8KernelKind::ALL.into_iter().filter_map(|kind| match PackedInt4Kernel::detect(kind) {
        Ok(kernel) => Some(kernel),
        Err(CpuKernelError::UnsupportedIsa) => {
            eprintln!("INT4_BATCH_REUSE tier={kind} RESULT=SKIPPED_UNSUPPORTED_ISA"); None
        }
        Err(error) => panic!("kernel construction: {error}"),
    }).collect()
}
fn input(k: usize, row: usize) -> ActivationBuffer {
    let mut result = ActivationBuffer::try_new(k).unwrap();
    let values: Vec<f32> = (0..k).map(|i| if row % 7 == 0 { 0.0 }
        else { ((i * 53 + row * 29) % 255) as f32 * (row + 1) as f32 / 37.0 - 127.0 }).collect();
    result.encode_f32(&values).unwrap(); result
}
fn work(matrix: GroupedInt4Linear<'_>, rows: LinearRows<'_>, m: usize) -> ProjectionWork {
    let one = matrix.planned_work(rows).unwrap();
    ProjectionWork { dot_products: one.dot_products * m as u64,
        multiply_accumulates: one.multiply_accumulates * m as u64 }
}
fn reference(input: &ActivationBuffer, packed: &[u8], row: usize, k: usize, width: usize, scales: &[f32]) -> f32 {
    let groups = k.div_ceil(width); let mut total = 0.0_f32;
    for group in 0..groups {
        let start = group * width; let end = k.min(start + width); let mut sum = 0_i64;
        for i in start..end {
            let index = row * k + i;
            let nibble = i64::from((packed[index / 2] >> (4 * (index % 2))) & 15);
            let weight = if nibble >= 8 { nibble - 16 } else { nibble };
            sum += i64::from(input.values().unwrap()[i]) * weight;
        }
        // Explicit scalar conversion/scale boundaries, independent of the
        // packed leaf and accumulate_group implementation under test.
        let mut value = i32::try_from(sum).unwrap() as f32;
        value *= input.scale().unwrap(); value *= 1.0; value *= 1.0;
        value *= scales[row * groups + group]; total += value;
    }
    total
}
fn check_case(kernel: PackedInt4Kernel, m: usize, k: usize, width: usize) {
    let packed: Vec<u8> = (0..6 * k / 2).map(|i| (i * 73 + 0x87) as u8).collect();
    let scales: Vec<f32> = (0..6 * k.div_ceil(width)).map(|i| (i % 11 + 1) as f32 / 97.0).collect();
    let matrix = GroupedInt4Linear::checked(6, k, width, &packed, &scales).unwrap().with_kernel(kernel);
    let inputs: Vec<_> = (0..m).map(|row| input(k, row)).collect();
    for ids in [vec![0, 2, 5], (0..6).collect()] {
        let rows = LinearRows::Selected(&ids); let budget = work(matrix, rows, m);
        let mut output = vec![123.0; m * ids.len()]; let mut ledger = ProjectionLedger::new(budget);
        let mut scratch = vec![99; matrix.scratch_len() + 3];
        matrix.project_batch_f32_into(&inputs, rows, &mut output, &mut scratch, &mut ledger, &mut Control::default()).unwrap();
        assert_eq!(ledger.reserved(), budget); assert_eq!(&scratch[matrix.scratch_len()..], &[99; 3]);
        let mut expected = Vec::new();
        for activation in &inputs { for &id in &ids { expected.push(reference(activation, &packed, id as usize, k, width, &scales)); } }
        assert_eq!(output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        let mut bf16 = vec![Bf16::from_bits(0); output.len()];
        matrix.project_batch_bf16_into(&inputs, rows, &mut bf16, &mut scratch,
            &mut ProjectionLedger::new(budget), &mut Control::default()).unwrap();
        assert_eq!(bf16.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), expected.iter().map(|&x| Bf16::from_f32(x).to_bits()).collect::<Vec<_>>());
    }
}
#[test]
fn group_major_reuse_preserves_all_row_and_group_tails_and_sparse_scales() {
    for kernel in kernels() { for m in [1, 4, 5, 17, 33, 64] { for k in [1, 7, 17, 33, 65] {
        for width in [1, 3, 16, 32] { check_case(kernel, m, k, width); }
    } } }
}
#[test]
fn full_model_reduction_lengths_keep_increasing_group_f32_sums() {
    for kernel in kernels() { for k in [3072, 6144, 10752] { check_case(kernel, 5, k, 32); } }
}
#[test]
fn every_checkpoint_can_cancel_without_refund_or_successful_partial_group_output() {
    let packed = [0x77; 33]; let scales = [0.125; 6];
    let matrix = GroupedInt4Linear::checked(2, 33, 16, &packed, &scales).unwrap();
    let inputs: Vec<_> = (0..64).map(|row| input(33, row)).collect();
    let budget = work(matrix, LinearRows::All, 64);
    let mut control = Control::default();
    matrix.project_batch_f32_into(&inputs, LinearRows::All, &mut [0.0; 128], &mut [0; 16],
        &mut ProjectionLedger::new(budget), &mut control).unwrap();
    // 384 group dots, no more than 32 between polls, and a final checkpoint.
    assert_eq!(control.polls, 13);
    for cancel in 1..=control.polls {
        let mut output = [123.0; 128]; let mut scratch = [99; 16]; let mut ledger = ProjectionLedger::new(budget);
        let mut control = Control { polls: 0, cancel: Some(cancel) };
        assert_eq!(matrix.project_batch_f32_into(&inputs, LinearRows::All, &mut output, &mut scratch,
            &mut ledger, &mut control), Err(GroupedInt4Error::Linear(LinearError::Cancelled(DecodeCancellationKind::User))));
        assert_eq!(ledger.reserved(), budget); assert_eq!(control.polls, cancel);
        if cancel <= 6 { assert_eq!(output, [123.0; 128]); }
        if cancel == 1 { assert_eq!(scratch, [99; 16]); }
    }
}
#[test]
fn last_row_and_aggregate_budgets_are_checked_before_any_scratch_or_output_write() {
    let matrix = GroupedInt4Linear::checked(2, 3, 2, &[0x71; 3], &[0.25; 4]).unwrap();
    for axis in 0..4 {
        let mut inputs: Vec<_> = (0..64).map(|row| input(3, row)).collect();
        if axis == 0 { inputs[63] = ActivationBuffer::try_new(3).unwrap(); }
        if axis == 1 { inputs[63] = input(2, 63); }
        let mut budget = work(matrix, LinearRows::All, 64);
        if axis == 2 { budget.dot_products -= 1; }
        if axis == 3 { budget.multiply_accumulates -= 1; }
        let mut ledger = ProjectionLedger::new(budget); let mut control = Control::default();
        let mut output = [123.0; 128]; let mut scratch = [99; 2];
        assert!(matrix.project_batch_f32_into(&inputs, LinearRows::All, &mut output, &mut scratch, &mut ledger, &mut control).is_err());
        assert_eq!(ledger.reserved(), ProjectionWork::default()); assert_eq!(control.polls, 0);
        assert_eq!(output, [123.0; 128]); assert_eq!(scratch, [99; 2]);
    }
}
#[test]
fn batch_permutation_and_independent_single_row_calls_preserve_results() {
    let matrix = GroupedInt4Linear::checked(2, 7, 3, &[0x87; 7], &[0.13, 0.7, 1.1, 0.2, 0.3, 0.4]).unwrap();
    let mut inputs: Vec<_> = (0..17).map(|row| input(7, row)).collect();
    let budget = work(matrix, LinearRows::All, 17); let mut whole = [0.0; 34]; let mut singles = [0.0; 34];
    matrix.project_batch_f32_into(&inputs, LinearRows::All, &mut whole, &mut [0; 3],
        &mut ProjectionLedger::new(budget), &mut Control::default()).unwrap();
    for (activation, output) in inputs.iter().zip(singles.chunks_mut(2)) {
        matrix.project_f32_into(activation, LinearRows::All, output, &mut [0; 3],
            &mut ProjectionLedger::new(budget), &mut Control::default()).unwrap();
    }
    assert_eq!(whole.map(f32::to_bits), singles.map(f32::to_bits)); inputs.reverse();
    matrix.project_batch_f32_into(&inputs, LinearRows::All, &mut singles, &mut [0; 3],
        &mut ProjectionLedger::new(budget), &mut Control::default()).unwrap();
    for row in 0..17 { assert_eq!(&whole[row * 2..row * 2 + 2], &singles[(16-row) * 2..(17-row) * 2]); }
}
