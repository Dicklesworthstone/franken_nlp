use super::*;

#[derive(Default)]
struct Control { polls: usize, cancel: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1;
        (self.cancel == Some(self.polls)).then_some(DecodeCancellationKind::User)
    }
}
fn inputs(m: usize, k: usize) -> Vec<ActivationBuffer> {
    (0..m).map(|row| {
        let mut buffer = ActivationBuffer::try_new(k).unwrap();
        let values: Vec<_> = (0..k).map(|i| if row == 0 { 0.0 } else {
            ((i * 13 + row * 37) % 255) as f32 * (row as f32 + 0.25) - 127.0 }).collect();
        buffer.encode_f32(&values).unwrap(); buffer
    }).collect()
}
fn packed(n: usize, k: usize) -> Vec<u8> {
    (0..n * k / 2).map(|i| {
        let a = ((2 * i * 7 + 8) % 16) as u8;
        let b = (((2 * i + 1) * 7 + 8) % 16) as u8;
        a | (b << 4)
    }).collect()
}
fn compare(m: usize, k: usize, width: usize, kernel: PackedInt4Kernel) {
    let n = 6; let packed = packed(n, k);
    let scales: Vec<_> = (0..n * k.div_ceil(width)).map(|i| (i % 11 + 1) as f32 / 64.0).collect();
    let view = GroupedInt4Linear::checked(n, k, width, &packed, &scales).unwrap().with_kernel(kernel);
    let inputs = inputs(m, k);
    let masks = [LinearRows::Selected(&[0, 2, 5]), LinearRows::All,
        LinearRows::Selected(&[1, 2]), LinearRows::Selected(&[0, 4, 5])];
    let mut expected = Vec::new();
    for row in 0..m {
        let mut out = vec![0.0; masks[row].checked_count(n).unwrap()];
        view.project_f32_into(&inputs[row], masks[row], &mut out, &mut vec![0; view.scratch_len()],
            &mut ProjectionLedger::new(view.planned_work(masks[row]).unwrap()), &mut Control::default()).unwrap();
        expected.push(out);
    }
    let mut output: Vec<_> = expected.iter().map(|row| vec![42.0; row.len()]).collect();
    let mut ledgers: Vec<_> = masks[..m].iter().map(|&mask| ProjectionLedger::new(view.planned_work(mask).unwrap())).collect();
    let mut members: Vec<_> = inputs.iter().zip(output.iter_mut()).zip(ledgers.iter_mut()).enumerate()
        .map(|(i, ((input, out), ledger))| Int4ProjectionRow { input, selection: masks[i], output: out, ledger }).collect();
    view.project_cohort_f32_into(&mut members, &mut vec![0; view.scratch_len()], &mut Control::default()).unwrap();
    drop(members);
    for i in 0..m {
        assert_eq!(output[i].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected[i].iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert_eq!(ledgers[i].reserved(), view.planned_work(masks[i]).unwrap());
        assert_eq!(ledgers[i].remaining(), ProjectionWork::default());
    }
    // Reverse physical lane order without changing each semantic row's mask.
    let mut bf16: Vec<_> = expected.iter().map(|row| vec![Bf16::from_bits(0); row.len()]).collect();
    let mut budgets: Vec<_> = masks[..m].iter().map(|&mask| ProjectionLedger::new(view.planned_work(mask).unwrap())).collect();
    let mut members: Vec<_> = inputs.iter().zip(bf16.iter_mut()).zip(budgets.iter_mut()).enumerate()
        .map(|(i, ((input, out), ledger))| Int4ProjectionRow { input, selection: masks[i], output: out, ledger }).collect();
    members.reverse();
    view.project_cohort_bf16_into(&mut members, &mut vec![0; view.scratch_len()], &mut Control::default()).unwrap();
    drop(members);
    for i in 0..m { assert_eq!(bf16[i], expected[i].iter().copied().map(Bf16::from_f32).collect::<Vec<_>>()); }
}

#[test]
fn ragged_masks_and_independent_scales_match_serial_across_group_and_model_tails() {
    for m in [1, 2, 3, MAX_INT4_COHORT_ROWS] {
        for k in [1, 3, 33, 130, 3072, MAX_MODEL_K] {
            for width in [1, 3, 32, 128, MAX_MODEL_K] { compare(m, k, width, PackedInt4Kernel::scalar()); }
        }
    }
}

#[test]
fn every_available_native_cohort_keeps_scalar_results_and_executed_identity() {
    for kind in Int8KernelKind::ALL {
        match PackedInt4Kernel::detect(kind) {
            Ok(kernel) => { assert_eq!(kernel.kind(), kind); compare(4, 130, 33, kernel); }
            Err(CpuKernelError::UnsupportedIsa) => eprintln!("INT4_COHORT tier={kind} RESULT=SKIPPED_UNSUPPORTED_ISA"),
            Err(error) => panic!("unexpected kernel refusal: {error}"),
        }
    }
}

#[test]
fn invalid_last_member_or_ledger_does_not_charge_or_write_any_member() {
    let view = GroupedInt4Linear::checked(2, 2, 1, &[0x11; 2], &[1.0; 4]).unwrap();
    for mode in 0..5 {
        let mut inputs = inputs(2, 2);
        if mode == 0 { assert!(inputs[1].encode_f32(&[f32::NAN; 2]).is_err()); }
        let mut outputs = [[42.0; 2]; 2];
        let work = view.planned_work(LinearRows::All).unwrap();
        let mut budgets = [ProjectionLedger::new(work), ProjectionLedger::new(work)];
        if mode == 1 { budgets[1] = ProjectionLedger::new(ProjectionWork { dot_products: work.dot_products - 1, ..work }); }
        let mut members: Vec<_> = inputs.iter().zip(outputs.iter_mut()).zip(budgets.iter_mut()).enumerate()
            .map(|(i, ((input, out), ledger))| Int4ProjectionRow { input,
                selection: if i == 1 && mode == 2 { LinearRows::Selected(&[1, 0]) } else { LinearRows::All },
                output: if i == 1 && mode == 3 { &mut out[..1] } else { &mut out[..] }, ledger }).collect();
        let mut scratch = [99]; let mut control = Control::default();
        assert!(view.project_cohort_f32_into(&mut members,
            if mode == 4 { &mut [] } else { &mut scratch }, &mut control).is_err());
        drop(members);
        assert_eq!(outputs, [[42.0; 2]; 2]); assert_eq!(scratch, [99]); assert_eq!(control.polls, 0);
        assert!(budgets.iter().all(|budget| budget.reserved() == ProjectionWork::default()));
    }
}

#[test]
fn cancellation_within_a_shared_long_row_retains_each_full_reservation() {
    let view = GroupedInt4Linear::checked(2, 66, 1, &[0x11; 66], &[1.0; 132]).unwrap();
    let inputs = inputs(4, 66); let work = view.planned_work(LinearRows::All).unwrap();
    let mut outputs = [[42.0; 2]; 4];
    let mut budgets: Vec<_> = (0..4).map(|_| ProjectionLedger::new(work)).collect();
    let mut members: Vec<_> = inputs.iter().zip(outputs.iter_mut()).zip(budgets.iter_mut())
        .map(|((input, output), ledger)| Int4ProjectionRow { input, selection: LinearRows::All, output, ledger }).collect();
    let mut control = Control { polls: 0, cancel: Some(2) };
    assert_eq!(view.project_cohort_f32_into(&mut members, &mut [0], &mut control),
        Err(GroupedInt4Error::Linear(LinearError::Cancelled(DecodeCancellationKind::User))));
    drop(members);
    assert_eq!(outputs, [[42.0; 2]; 4]); assert_eq!(control.polls, 2);
    assert!(budgets.iter().all(|budget| budget.reserved() == work));
}

#[test]
fn a_row_outside_the_mask_union_is_not_evaluated() {
    let view = GroupedInt4Linear::checked(4, 2, 2, &[0x11, 0x11, 0x11, 0x77], &[1.0, 2.0, 3.0, f32::MAX]).unwrap();
    let mut input = ActivationBuffer::try_new(2).unwrap(); input.encode_f32(&[127.0; 2]).unwrap();
    let masks = [LinearRows::Selected(&[0, 2]), LinearRows::Selected(&[1, 2])];
    let mut outputs = [[0.0; 2]; 2];
    let mut budgets = masks.map(|mask| ProjectionLedger::new(view.planned_work(mask).unwrap()));
    let mut members: Vec<_> = outputs.iter_mut().zip(budgets.iter_mut()).enumerate()
        .map(|(i, (output, ledger))| Int4ProjectionRow { input: &input, selection: masks[i], output, ledger }).collect();
    view.project_cohort_f32_into(&mut members, &mut [0; 2], &mut Control::default()).unwrap();
    drop(members);
    assert_eq!(outputs, [[254.0, 762.0], [508.0, 762.0]]);
}

#[test]
fn empty_and_oversized_cohorts_are_refused_before_any_effect() {
    let view = GroupedInt4Linear::checked(1, 2, 2, &[0x11], &[1.0]).unwrap();
    for count in [0, MAX_INT4_COHORT_ROWS + 1] {
        let inputs = inputs(count, 2); let mut output = vec![[42.0]; count];
        let mut budgets: Vec<_> = (0..count).map(|_| ProjectionLedger::new(view.planned_work(LinearRows::All).unwrap())).collect();
        let mut members: Vec<_> = inputs.iter().zip(output.iter_mut()).zip(budgets.iter_mut())
            .map(|((input, output), ledger)| Int4ProjectionRow { input, selection: LinearRows::All, output, ledger }).collect();
        let mut control = Control::default();
        assert!(view.project_cohort_f32_into(&mut members, &mut [0; 2], &mut control).is_err());
        drop(members);
        assert_eq!(control.polls, 0); assert!(output.iter().all(|row| row == &[42.0]));
        assert!(budgets.iter().all(|budget| budget.reserved() == ProjectionWork::default()));
    }
}

#[test]
fn nonfinite_group_sums_and_bf16_casts_refuse_the_whole_epoch() {
    let mut input = ActivationBuffer::try_new(2).unwrap(); input.encode_f32(&[1.0; 2]).unwrap();
    let view = GroupedInt4Linear::checked(1, 2, 1, &[0x11], &[f32::MAX; 2]).unwrap();
    let work = view.planned_work(LinearRows::All).unwrap();
    let mut output = [42.0]; let mut ledger = ProjectionLedger::new(work);
    let mut members = [Int4ProjectionRow { input: &input, selection: LinearRows::All, output: &mut output, ledger: &mut ledger }];
    assert!(view.project_cohort_f32_into(&mut members, &mut [0], &mut Control::default()).is_err());
    assert_eq!(output, [42.0]); assert_eq!(ledger.reserved(), work);
    let view = GroupedInt4Linear::checked(1, 2, 1, &[0x01], &[f32::MAX, 1.0]).unwrap();
    let mut output = [Bf16::from_f32(42.0)]; let mut ledger = ProjectionLedger::new(work);
    let mut members = [Int4ProjectionRow { input: &input, selection: LinearRows::All, output: &mut output, ledger: &mut ledger }];
    assert!(view.project_cohort_bf16_into(&mut members, &mut [0], &mut Control::default()).is_err());
    assert_eq!(output, [Bf16::from_f32(42.0)]); assert_eq!(ledger.reserved(), work);
}
