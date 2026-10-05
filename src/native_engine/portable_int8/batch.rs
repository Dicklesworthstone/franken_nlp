//! Shared-weight, activation-row-tiled INT8 projections.
//!
//! The kernel borrows the existing checked Generic matrix. Each weight value
//! feeds up to four independent accumulators before the next K coordinate is
//! loaded. Every accumulator traverses K in the scalar order and uses the
//! SAME fixed dequantization epilogue; no floating reduction crosses rows.
//! No weights are expanded, no workers spawn, and no hot allocation occurs.
//! Source implementation only: no measured throughput or parity award.
use super::*;

/// Finite activation cohort ceiling, independent of the selected output width.
pub const MAX_BATCH_ROWS: usize = 64;
const TILE_ROWS: usize = 4;
pub const BATCH_LINEAR_VERSION: &str = "portable-s8-shared-weight-m4-fixed-epilogue-v1";

impl QuantizedLinear<'_> {
    /// Output is activation-major: M contiguous rows of selected output width.
    /// Caller owns admission and storage. Any execution error invalidates the
    /// entire output, including cells written before cancellation/overflow.
    pub fn project_batch_f32_into<C: DecodeStepControl>(&self, inputs: &[ActivationBuffer],
        rows: LinearRows<'_>, output: &mut [f32], ledger: &mut ProjectionLedger, control: &mut C)
        -> Result<(), LinearError> {
        self.project_batch(inputs, rows, output.len(), ledger, control,
            |index, value| { output[index] = value; Ok(()) })
    }

    /// Same scalar BF16 cast boundary after each independent f32 epilogue.
    pub fn project_batch_bf16_into<C: DecodeStepControl>(&self, inputs: &[ActivationBuffer],
        rows: LinearRows<'_>, output: &mut [Bf16], ledger: &mut ProjectionLedger, control: &mut C)
        -> Result<(), LinearError> {
        self.project_batch(inputs, rows, output.len(), ledger, control, |index, value| {
            let cast = Bf16::from_f32(value);
            if !cast.to_f32().is_finite() { return Err(LinearError::NonFiniteOutput); }
            output[index] = cast; Ok(())
        })
    }

    fn project_batch<C: DecodeStepControl>(&self, inputs: &[ActivationBuffer], rows: LinearRows<'_>,
        length: usize, ledger: &mut ProjectionLedger, control: &mut C,
        mut store: impl FnMut(usize, f32) -> Result<(), LinearError>) -> Result<(), LinearError> {
        if inputs.is_empty() || inputs.len() > MAX_BATCH_ROWS { return Err(LinearError::Workspace); }
        let count = rows.checked_count(self.rows)?;
        let total = inputs.len().checked_mul(count).ok_or(LinearError::OutputShape)?;
        if length != total { return Err(LinearError::OutputShape); }
        // Validate EVERY row before charging, polling, or writing a single cell.
        for input in inputs {
            if input.values()?.len() != self.columns { return Err(LinearError::Shape); }
            input.scale()?;
        }
        ledger.charge(ProjectionWork::for_shape(total, self.columns)?)?;
        let mut since_checkpoint = ROWS_PER_CHECKPOINT;
        for output_column in 0..count {
            let row = rows.row(output_column);
            let weights = &self.values[row * self.columns..(row + 1) * self.columns];
            for (tile, cohort) in inputs.chunks(TILE_ROWS).enumerate() {
                if since_checkpoint >= ROWS_PER_CHECKPOINT { poll(control)?; since_checkpoint = 0; }
                let mut activations: [&[i8]; TILE_ROWS] = [&[]; TILE_ROWS];
                let mut scales = [1.0_f32; TILE_ROWS];
                let mut sums = [0_i32; TILE_ROWS];
                for (lane, input) in cohort.iter().enumerate() {
                    activations[lane] = input.values()?; scales[lane] = input.scale()?;
                }
                // Checked K <= MAX_MODEL_K. Even the full i8 domain has
                // |partial sum| <= K*16384 < i32::MAX; no saturating arithmetic.
                for (k, &weight) in weights.iter().enumerate() {
                    let weight = i32::from(weight);
                    for lane in 0..cohort.len() { sums[lane] += i32::from(activations[lane][k]) * weight; }
                }
                for lane in 0..cohort.len() {
                    let value = dequantize_i32_fixed(sums[lane], EpilogueScales {
                        activation: scales[lane], row: self.scales[row], column: 1.0, group: 1.0,
                    })?;
                    if !value.is_finite() { return Err(LinearError::NonFiniteOutput); }
                    store((tile * TILE_ROWS + lane) * count + output_column, value)?;
                }
                since_checkpoint += cohort.len();
            }
        }
        poll(control)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)] struct Control { polls: usize, cancel: Option<usize> }
    impl DecodeStepControl for Control {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.polls += 1;
            (self.cancel == Some(self.polls)).then_some(DecodeCancellationKind::User)
        }
    }
    fn inputs(m: usize, k: usize) -> Vec<ActivationBuffer> {
        (0..m).map(|row| {
            let mut input = ActivationBuffer::try_new(k).unwrap();
            let values: Vec<f32> = (0..k).map(|col| if row % 7 == 0 { 0.0 }
                else { ((row * 19 + col * 13) % 255) as f32 - 127.0 }).collect();
            input.encode_f32(&values).unwrap(); input
        }).collect()
    }
    fn compare(m: usize, k: usize, selected: bool) {
        let values: Vec<i8> = (0..7 * k).map(|i| ((i * 53 + 128) % 256) as u8 as i8).collect();
        let scales = [0.03125, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25];
        let sums: Vec<i32> = values.chunks(k).map(|r| r.iter().map(|&v| i32::from(v)).sum()).collect();
        let matrix = QuantizedLinear::checked(7, k, &values, &scales, &sums).unwrap();
        let inputs = inputs(m, k);
        let rows = if selected { LinearRows::Selected(&[0, 2, 6]) } else { LinearRows::All };
        let n = rows.checked_count(7).unwrap();
        let work = ProjectionWork::for_shape(m * n, k).unwrap();
        let mut serial = vec![0.0; m * n]; let mut grouped = vec![0.0; m * n];
        let mut a = ProjectionLedger::new(work); let mut b = ProjectionLedger::new(work);
        for (input, output) in inputs.iter().zip(serial.chunks_mut(n)) {
            matrix.project_f32_into(input, rows, output, &mut a, &mut Control::default()).unwrap();
        }
        matrix.project_batch_f32_into(&inputs, rows, &mut grouped, &mut b, &mut Control::default()).unwrap();
        assert_eq!(serial.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            grouped.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert_eq!(a.reserved(), b.reserved()); assert_eq!(b.reserved(), work);
        let mut serial = vec![Bf16::from_bits(0); m * n]; let mut grouped = serial.clone();
        for (input, output) in inputs.iter().zip(serial.chunks_mut(n)) {
            matrix.project_bf16_into(input, rows, output, &mut ProjectionLedger::new(work), &mut Control::default()).unwrap();
        }
        matrix.project_batch_bf16_into(&inputs, rows, &mut grouped,
            &mut ProjectionLedger::new(work), &mut Control::default()).unwrap();
        assert_eq!(serial.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            grouped.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }
    #[test] fn scalar_and_grouped_epilogues_agree_across_activation_and_k_tails() {
        for m in [1, 2, 3, 4, 5, 9, MAX_BATCH_ROWS] {
            for k in [1, 3, 31, 32, 33] { for selected in [false, true] { compare(m, k, selected); } }
        }
    }
    #[test] fn full_model_k_keeps_exact_integer_accumulation() {
        compare(5, MAX_MODEL_K, true);
        assert!((MAX_MODEL_K as u64) * 16384 < i32::MAX as u64);
    }
    #[test] fn row_reordering_never_changes_an_independent_result() {
        let matrix = QuantizedLinear::checked(2, 3, &[-128, 127, 11, 127, -128, -11], &[0.5, 0.25], &[10, -12]).unwrap();
        let mut inputs = inputs(5, 3); let work = ProjectionWork::for_shape(10, 3).unwrap();
        let mut before = vec![0.0; 10]; let mut after = vec![0.0; 10];
        matrix.project_batch_f32_into(&inputs, LinearRows::All, &mut before,
            &mut ProjectionLedger::new(work), &mut Control::default()).unwrap();
        inputs.reverse();
        matrix.project_batch_f32_into(&inputs, LinearRows::All, &mut after,
            &mut ProjectionLedger::new(work), &mut Control::default()).unwrap();
        for i in 0..5 { assert_eq!(&before[i * 2..i * 2 + 2], &after[(4 - i) * 2..(4 - i) * 2 + 2]); }
    }
    #[test] fn invalid_last_activation_is_refused_before_charge_or_output() {
        let matrix = QuantizedLinear::checked(1, 1, &[1], &[1.0], &[1]).unwrap();
        let mut inputs = inputs(2, 1);
        assert!(inputs[1].encode_f32(&[f32::NAN]).is_err());
        let mut ledger = ProjectionLedger::new(ProjectionWork::for_shape(2, 1).unwrap());
        let mut output = [42.0; 2]; let mut control = Control::default();
        assert_eq!(matrix.project_batch_f32_into(&inputs, LinearRows::All, &mut output, &mut ledger, &mut control), Err(LinearError::Workspace));
        assert_eq!(output, [42.0; 2]); assert_eq!(ledger.reserved(), ProjectionWork::default()); assert_eq!(control.polls, 0);
    }
    #[test] fn insufficient_total_work_is_not_treated_as_a_per_row_allowance() {
        let matrix = QuantizedLinear::checked(1, 1, &[1], &[1.0], &[1]).unwrap();
        let mut ledger = ProjectionLedger::new(ProjectionWork::for_shape(1, 1).unwrap());
        let mut output = [42.0; 2];
        assert_eq!(matrix.project_batch_f32_into(&inputs(2, 1), LinearRows::All, &mut output,
            &mut ledger, &mut Control::default()), Err(LinearError::WorkBudget));
        assert_eq!(output, [42.0; 2]); assert_eq!(ledger.reserved(), ProjectionWork::default());
    }
    #[test] fn cancellation_charges_the_complete_cohort_without_refund() {
        let matrix = QuantizedLinear::checked(1, 1, &[1], &[1.0], &[1]).unwrap();
        let work = ProjectionWork::for_shape(5, 1).unwrap(); let mut ledger = ProjectionLedger::new(work);
        let mut output = [42.0; 5]; let mut control = Control { polls: 0, cancel: Some(1) };
        assert_eq!(matrix.project_batch_f32_into(&inputs(5, 1), LinearRows::All, &mut output,
            &mut ledger, &mut control), Err(LinearError::Cancelled(DecodeCancellationKind::User)));
        assert_eq!(ledger.reserved(), work); assert_eq!(output, [42.0; 5]);
    }
    #[test] fn invalid_dimensions_and_selection_cannot_mutate_output() {
        let matrix = QuantizedLinear::checked(1, 1, &[1], &[1.0], &[1]).unwrap();
        for (m, k, rows, length) in [(0, 1, LinearRows::All, 0), (65, 1, LinearRows::All, 65),
            (2, 2, LinearRows::All, 2), (2, 1, LinearRows::All, 1),
            (2, 1, LinearRows::Selected(&[0, 0]), 4), (2, 1, LinearRows::Selected(&[1]), 2)] {
            let mut output = vec![42.0; length];
            let mut ledger = ProjectionLedger::new(ProjectionWork::for_shape(128, 2).unwrap());
            assert!(matrix.project_batch_f32_into(&inputs(m, k), rows, &mut output,
                &mut ledger, &mut Control::default()).is_err());
            assert!(output.iter().all(|&v| v == 42.0)); assert_eq!(ledger.reserved(), ProjectionWork::default());
        }
    }
}
