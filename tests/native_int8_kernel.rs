//! Native projection integration: integer ISA changes cannot move epilogue,
//! selected-row, BF16 cast, cancellation or nonrenewable-work boundaries.
use franken_nlp::native_engine::{
    decode::{DecodeCancellationKind, DecodeStepControl},
    portable_int8::{ActivationBuffer, LinearError, LinearRows, ProjectionLedger, ProjectionWork, QuantizedLinear},
    simd::{CpuKernelError, Int8Kernel, Int8KernelKind},
    tensor::Bf16,
};
#[derive(Default)]
struct Control { polls: usize, cancel: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1;
        (self.cancel == Some(self.polls)).then_some(DecodeCancellationKind::User)
    }
}
fn kernels() -> Vec<Int8Kernel> {
    Int8KernelKind::ALL.into_iter().filter_map(|kind| match Int8Kernel::detect(kind) {
        Ok(k) => Some(k),
        Err(CpuKernelError::UnsupportedIsa) => { eprintln!("NATIVE_INT8 tier={kind} RESULT=SKIPPED_UNSUPPORTED_ISA"); None }
        Err(e) => panic!("kernel refused: {e}"),
    }).collect()
}
fn input(k: usize, row: usize) -> ActivationBuffer {
    let mut buffer = ActivationBuffer::try_new(k).unwrap();
    let values: Vec<_> = (0..k).map(|i| if row == 0 { 0.0 } else { ((i * 29 + row * 17) % 257) as f32 * 0.03125 - 4.0 }).collect();
    buffer.encode_f32(&values).unwrap(); buffer
}

#[test]
fn selected_and_full_f32_and_bf16_projections_preserve_exact_scalar_bits() {
    let kernels = kernels();
    for k in [1, 31, 32, 33, 3072, 6144, 10752] {
        let weights: Vec<_> = (0..7 * k).map(|i| ((i * 53 + 128) % 256) as u8 as i8).collect();
        let sums: Vec<i32> = weights.chunks(k).map(|r| r.iter().map(|&v| i32::from(v)).sum()).collect();
        let scales = [0.03125, 0.1, 0.3, 0.5, 0.75, 1.0, 1.25];
        let scalar = QuantizedLinear::checked(7, k, &weights, &scales, &sums).unwrap();
        assert_eq!(scalar.kernel_kind(), Int8KernelKind::Scalar);
        for rows in [LinearRows::All, LinearRows::Selected(&[0, 2, 6])] {
            let n = rows.checked_count(7).unwrap();
            let work = ProjectionWork::for_shape(n, k).unwrap();
            let activation = input(k, 1);
            let mut expected = vec![0.0; n];
            scalar.project_f32_into(&activation, rows, &mut expected, &mut ProjectionLedger::new(work), &mut Control::default()).unwrap();
            let mut expected_bf16 = vec![Bf16::from_bits(0); n];
            scalar.project_bf16_into(&activation, rows, &mut expected_bf16, &mut ProjectionLedger::new(work), &mut Control::default()).unwrap();
            for &kernel in &kernels {
                let matrix = scalar.with_kernel(kernel);
                assert_eq!(matrix.kernel_kind(), kernel.kind());
                let mut actual = vec![0.0; n]; let mut ledger = ProjectionLedger::new(work);
                matrix.project_f32_into(&activation, rows, &mut actual, &mut ledger, &mut Control::default()).unwrap();
                assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                assert_eq!(ledger.reserved(), work);
                let mut actual = vec![Bf16::from_bits(0); n];
                matrix.project_bf16_into(&activation, rows, &mut actual, &mut ProjectionLedger::new(work), &mut Control::default()).unwrap();
                assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected_bf16.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            }
        }
    }
}

#[test]
fn grouped_prefill_and_skinny_decode_use_the_selected_kernel_with_identical_epilogues() {
    for k in [31, 32, 33, 3072] {
        let weights: Vec<_> = (0..3 * k).map(|i| ((i * 29 + 127) % 256) as u8 as i8).collect();
        let sums: Vec<i32> = weights.chunks(k).map(|r| r.iter().map(|&v| i32::from(v)).sum()).collect();
        let scalar = QuantizedLinear::checked(3, k, &weights, &[0.5, 0.1, 0.25], &sums).unwrap();
        for m in [1, 2, 3, 4, 5, 9] {
            let inputs: Vec<_> = (0..m).map(|i| input(k, i)).collect();
            let rows = LinearRows::Selected(&[0, 2]);
            let work = ProjectionWork::for_shape(m * 2, k).unwrap();
            let mut expected = vec![Bf16::from_bits(0); m * 2];
            let mut reference_control = Control::default();
            scalar.project_batch_bf16_into(&inputs, rows, &mut expected,
                &mut ProjectionLedger::new(work), &mut reference_control).unwrap();
            for kernel in kernels() {
                let mut actual = vec![Bf16::from_bits(0); m * 2];
                let mut ledger = ProjectionLedger::new(work); let mut control = Control::default();
                scalar.with_kernel(kernel).project_batch_bf16_into(&inputs, rows, &mut actual, &mut ledger, &mut control).unwrap();
                assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                assert_eq!(ledger.reserved(), work); assert_eq!(control.polls, reference_control.polls);
            }
        }
    }
}

#[test]
fn selected_kernel_preserves_no_write_before_admission_and_nonrefundable_cancel() {
    for kernel in kernels() {
        let matrix = QuantizedLinear::checked(1, 33, &[-128; 33], &[0.5], &[-4224]).unwrap().with_kernel(kernel);
        let activation = input(33, 1); let work = ProjectionWork::for_shape(1, 33).unwrap();
        let mut output = [123.0]; let mut control = Control::default();
        assert_eq!(matrix.project_f32_into(&activation, LinearRows::All, &mut output,
            &mut ProjectionLedger::new(ProjectionWork::default()), &mut control), Err(LinearError::WorkBudget));
        assert_eq!(output, [123.0]); assert_eq!(control.polls, 0);
        let mut ledger = ProjectionLedger::new(work); let mut control = Control { polls: 0, cancel: Some(1) };
        assert_eq!(matrix.project_f32_into(&activation, LinearRows::All, &mut output, &mut ledger, &mut control),
            Err(LinearError::Cancelled(DecodeCancellationKind::User)));
        assert_eq!(output, [123.0]); assert_eq!(ledger.reserved(), work);
    }
}
