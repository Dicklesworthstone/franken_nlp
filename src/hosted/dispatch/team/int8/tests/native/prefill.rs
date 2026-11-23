//! Batched prompt projections and one-row decode use one actual fixed team.
use super::*;

fn batch_inputs(rows: usize) -> Vec<ActivationBuffer> {
    (0..rows).map(|row| {
        let mut input = ActivationBuffer::try_new(6).unwrap();
        let values: Vec<_> = (0..6).map(|column| if row % 7 == 0 { 0.0 }
            else { ((row * 19 + column * 31) % 255) as f32 - 127.0 }).collect();
        input.encode_f32(&values).unwrap();
        input
    }).collect()
}

#[test]
fn one_admitted_team_keeps_exact_batch_rows_scales_and_decode_across_available_kernels() {
    for kind in Int8KernelKind::ALL {
        let Ok(kernel) = Int8Kernel::detect(kind) else { continue; };
        for width in [2, 3, 7, 16] {
            let fixture = ProjectionFixture::new(41);
            let weights = fixture.weights().with_kernel(kernel);
            let mut workspace = Workspace::with_batch_rows(width, 64).unwrap();
            let dropped = track_children(&mut workspace);
            let shared = RunShared::new(limits(1_000_000), CancellationToken::default()).unwrap();
            let stats = run_scope(&Cx::for_testing(), &weights, &mut workspace, width, kernel, shared,
                |control, executor| {
                    assert_eq!(executor.max_batch_rows(), 64);
                    assert_eq!(executor.kernel_kind(), kind);
                    use Int8ProjectionKind::*;
                    let operations = [
                        Int8ProjectionOp::Layer { layer: 0, kind: Query },
                        Int8ProjectionOp::Layer { layer: 3, kind: Key },
                        Int8ProjectionOp::Layer { layer: 7, kind: Value },
                        Int8ProjectionOp::Layer { layer: 11, kind: AttentionOutput },
                        Int8ProjectionOp::Layer { layer: 15, kind: Gate },
                        Int8ProjectionOp::Layer { layer: 17, kind: Up },
                        Int8ProjectionOp::Layer { layer: 21, kind: Down },
                    ];
                    for batch_rows in [1, 3, 4, 5, 64] {
                        let inputs = batch_inputs(batch_rows);
                        for operation in operations {
                            for rows in [LinearRows::All, LinearRows::Selected(&[0, 3, 5])] {
                                // Independent token-at-a-time leaves are the
                                // oracle, not the new batch executor itself.
                                let reference: Vec<_> = inputs.iter().flat_map(|input|
                                    expected(&weights, operation, input, rows).1).collect();
                                let mut output = vec![Bf16::from_bits(0xffff); reference.len()];
                                executor.project_batch_bf16_into(operation, &inputs, rows, &mut output, control)
                                    .map_err(HostedError::Native)?;
                                assert_eq!(output, reference);
                            }
                        }
                        // Head and later decode keep the original capacity,
                        // lanes and worker buffers after each prompt morsel.
                        let input = fixture.input();
                        let (reference_float, reference_bf16) = expected(&weights, Int8ProjectionOp::Head,
                            &input, LinearRows::All);
                        let mut floats = [f32::NAN; 10];
                        let mut bf16 = [Bf16::from_bits(0xffff); 10];
                        executor.project_f32_into(Int8ProjectionOp::Head, &input, LinearRows::All,
                            &mut floats, control).map_err(HostedError::Native)?;
                        executor.project_bf16_into(Int8ProjectionOp::Head, &input, LinearRows::All,
                            &mut bf16, control).map_err(HostedError::Native)?;
                        assert_eq!(floats.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                            reference_float.iter().map(|value| value.to_bits()).collect::<Vec<_>>());
                        assert_eq!(bf16.as_slice(), reference_bf16.as_slice());
                        assert_eq!(dropped.load(Ordering::SeqCst), 0);
                    }
                    Ok(executor.stats())
                }).unwrap();
            assert_eq!(stats.completed_projections, 80);
            assert_eq!(stats.completed_rows, 4951);
            assert_eq!(stats.completed_worker_rows, match width {
                2 => 2206, 3 => 3294, 7 => 3853, 16 => 3863, _ => unreachable!(),
            });
            assert_eq!(dropped.load(Ordering::SeqCst), width - 1);
        }
    }
}

#[test]
fn unsupported_batch_geometry_refuses_before_output_and_poisoning_prevents_reuse() {
    for case in 0..4 {
        let fixture = ProjectionFixture::new(43); let weights = fixture.weights();
        let mut workspace = Workspace::with_batch_rows(3, 3).unwrap();
        let dropped = track_children(&mut workspace);
        let shared = RunShared::new(limits(10_000), CancellationToken::default()).unwrap();
        let error = run_scope(&Cx::for_testing(), &weights, &mut workspace, 3, Int8Kernel::scalar(),
            shared, |control, executor| {
                let mut inputs = batch_inputs(if case == 0 { 5 } else { 2 });
                let operation = if case == 1 { Int8ProjectionOp::Head }
                    else { Int8ProjectionOp::Layer { layer: 3, kind: Int8ProjectionKind::Gate } };
                if case == 2 { assert!(inputs[1].encode_f32(&[f32::NAN; 6]).is_err()); }
                let length = if case == 1 { 20 } else if case == 3 { 11 } else { inputs.len() * 6 };
                let sentinel = Bf16::from_f32(13.0);
                let mut output = vec![sentinel; length];
                let error = executor.project_batch_bf16_into(operation, &inputs, LinearRows::All,
                    &mut output, control).unwrap_err();
                assert!(output.iter().all(|value| *value == sentinel));
                assert_eq!(executor.stats().completed_projections, 0);
                assert_eq!(executor.stats().completed_rows, 0);
                assert_eq!(executor.project_f32_into(Int8ProjectionOp::Head, &fixture.input(),
                    LinearRows::All, &mut [0.0; 10], control), Err(StrictInt8Error::ProjectionExecutor));
                Err::<(), _>(HostedError::Native(error))
            }).unwrap_err();
        assert!(matches!(error, HostedError::Stopped { .. }));
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
    }
}

#[test]
fn batched_inflight_cancellation_keeps_each_original_cause_and_drains_every_buffer() {
    struct StopOnSecond<'a> { inner: &'a mut RunControl, token: CancellationToken,
        cause: DecodeCancellationKind, calls: usize }
    impl DecodeStepControl for StopOnSecond<'_> {
        fn checkpoint(&mut self, step: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1;
            if self.calls == 2 { self.token.cancel(self.cause); }
            self.inner.checkpoint(step)
        }
    }
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let fixture = ProjectionFixture::new(47); let weights = fixture.weights();
        let inputs = batch_inputs(5);
        let mut workspace = Workspace::with_batch_rows(4, 5).unwrap();
        let dropped = track_children(&mut workspace);
        let token = CancellationToken::default();
        let shared = RunShared::new(limits(10_000), token.clone()).unwrap();
        let error = run_scope(&Cx::for_testing(), &weights, &mut workspace, 4, Int8Kernel::scalar(),
            shared, |control, executor| {
                let error = executor.project_batch_bf16_into(
                    Int8ProjectionOp::Layer { layer: 21, kind: Int8ProjectionKind::Down },
                    &inputs, LinearRows::All, &mut [Bf16::default(); 30],
                    &mut StopOnSecond { inner: control, token, cause, calls: 0 }).unwrap_err();
                assert_eq!(executor.stats().completed_projections, 0);
                Err::<(), _>(HostedError::Native(error))
            }).unwrap_err();
        assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind, .. }, .. } if kind == cause));
        assert_eq!(dropped.load(Ordering::SeqCst), 3);
    }
}

#[test]
fn decode_after_prefill_cannot_reset_the_original_checkpoint_allowance() {
    let fixture = ProjectionFixture::new(53); let weights = fixture.weights();
    let inputs = batch_inputs(3);
    let mut workspace = Workspace::with_batch_rows(3, 3).unwrap();
    let dropped = track_children(&mut workspace);
    let shared = RunShared::new(limits(10_000), CancellationToken::default()).unwrap();
    let authority = Arc::clone(&shared);
    let error = run_scope(&Cx::for_testing(), &weights, &mut workspace, 3, Int8Kernel::scalar(),
        shared, |control, executor| {
            executor.project_batch_bf16_into(
                Int8ProjectionOp::Layer { layer: 0, kind: Int8ProjectionKind::Query },
                &inputs, LinearRows::All, &mut [Bf16::default(); 18], control).map_err(HostedError::Native)?;
            assert_eq!(executor.stats().completed_rows, 18);
            authority.state.lock().unwrap().remaining = 0;
            let error = executor.project_f32_into(Int8ProjectionOp::Head, &fixture.input(), LinearRows::All,
                &mut [0.0; 10], control).unwrap_err();
            assert_eq!(error, StrictInt8Error::Cancelled(DecodeCancellationKind::PollQuota));
            assert_eq!(executor.stats().completed_projections, 1);
            assert_eq!(executor.stats().completed_rows, 18);
            Err::<(), _>(HostedError::Native(error))
        }).unwrap_err();
    assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind: DecodeCancellationKind::PollQuota, .. }, .. }));
    assert_eq!(dropped.load(Ordering::SeqCst), 2);
}

#[test]
fn batch_worker_and_coordinator_panics_join_the_entire_fixed_team() {
    let fixture = ProjectionFixture::new(59); let weights = fixture.weights();
    let inputs = batch_inputs(5);
    let operation = Int8ProjectionOp::Layer { layer: 7, kind: Int8ProjectionKind::Value };
    let mut workspace = Workspace::with_batch_rows(4, 5).unwrap(); workspace.panic_worker = Some(1);
    let dropped = track_children(&mut workspace);
    let shared = RunShared::new(limits(10_000), CancellationToken::default()).unwrap();
    let error = run_scope(&Cx::for_testing(), &weights, &mut workspace, 4, Int8Kernel::scalar(),
        shared, |control, executor| executor.project_batch_bf16_into(operation, &inputs, LinearRows::All,
            &mut [Bf16::default(); 30], control).map_err(HostedError::Native)).unwrap_err();
    assert!(matches!(error, HostedError::Scope {
        source: asupersync::cx::ScopedCpuError::ChildPanicked { .. }, .. }));
    assert_eq!(dropped.load(Ordering::SeqCst), 3);

    struct PanicOnSecond<'a> { inner: &'a mut RunControl, calls: usize }
    impl DecodeStepControl for PanicOnSecond<'_> {
        fn checkpoint(&mut self, step: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1;
            assert_ne!(self.calls, 2, "injected batched coordinator panic after dispatch");
            self.inner.checkpoint(step)
        }
    }
    let mut workspace = Workspace::with_batch_rows(4, 5).unwrap();
    let dropped = track_children(&mut workspace);
    let shared = RunShared::new(limits(10_000), CancellationToken::default()).unwrap();
    let outcome = catch_unwind(AssertUnwindSafe(|| run_scope(&Cx::for_testing(), &weights,
        &mut workspace, 4, Int8Kernel::scalar(), shared, |control, executor|
            executor.project_batch_bf16_into(operation, &inputs, LinearRows::All,
                &mut [Bf16::default(); 30], &mut PanicOnSecond { inner: control, calls: 0 })
                .map_err(HostedError::Native))));
    assert!(outcome.is_err());
    assert_eq!(dropped.load(Ordering::SeqCst), 3);
}
