//! Real scoped-team protocol regressions over tiny projection-only fixtures.
use super::*;
use std::{panic::{catch_unwind, AssertUnwindSafe}, sync::atomic::{AtomicUsize, Ordering}, time::Duration};
use crate::{hosted::RunStop, native_engine::{portable_int8::{ActivationBuffer, LinearRows, ProjectionLedger, ProjectionWork},
    simd::Int8KernelKind, strict_int8::projection::{fixture::ProjectionFixture, Int8ProjectionKind}}};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn limits(checkpoints: u64) -> RunLimits {
    RunLimits { max_elapsed: Duration::from_secs(60), max_checkpoints: checkpoints, cleanup_reserve_bytes: 65536 }
}
fn expected(weights: &Int8WeightView<'_>, operation: Int8ProjectionOp, input: &ActivationBuffer,
    rows: LinearRows<'_>) -> (Vec<f32>, Vec<Bf16>) {
    let linear = weights.projection(operation).unwrap();
    let count = rows.checked_count(linear.rows()).unwrap();
    let mut floats = vec![0.0; count];
    let mut bf16 = vec![Bf16::default(); count];
    let work = ProjectionWork::for_shape(count, linear.columns()).unwrap();
    linear.project_f32_into(input, rows, &mut floats, &mut ProjectionLedger::new(work), &mut Continue).unwrap();
    linear.project_bf16_into(input, rows, &mut bf16, &mut ProjectionLedger::new(work), &mut Continue).unwrap();
    (floats, bf16)
}
fn track_children(workspace: &mut Workspace) -> Arc<AtomicUsize> {
    let dropped = Arc::new(AtomicUsize::new(0));
    for child in &mut workspace.children { child.observe_drop(Arc::clone(&dropped)); }
    dropped
}

#[test]
fn one_team_reuses_workers_for_every_projection_role_both_passes_and_both_output_rails() {
    for width in [2, 3, 7, 16] {
        let fixture = ProjectionFixture::new(23);
        let weights = fixture.weights(); let input = fixture.input();
        let mut workspace = Workspace::new(width).unwrap();
        let dropped = track_children(&mut workspace);
        let shared = RunShared::new(limits(100_000), CancellationToken::default()).unwrap();
        let stats = run_scope(&Cx::for_testing(), &weights, &mut workspace, width,
            Int8Kernel::scalar(), shared, |control, executor| {
            assert_eq!(executor.artifact_identity(), weights.artifact_identity());
            assert_eq!(executor.kernel_kind(), weights.kernel_kind());
            let selected = [0, 3, 5];
            use Int8ProjectionKind::*;
            let operations = [Int8ProjectionOp::Head,
                Int8ProjectionOp::Layer { layer: 0, kind: Query },
                Int8ProjectionOp::Layer { layer: 3, kind: Key },
                Int8ProjectionOp::Layer { layer: 7, kind: Value },
                Int8ProjectionOp::Layer { layer: 11, kind: AttentionOutput },
                Int8ProjectionOp::Layer { layer: 15, kind: Gate },
                Int8ProjectionOp::Layer { layer: 17, kind: Up },
                Int8ProjectionOp::Layer { layer: 21, kind: Down }];
            for _ in 0..2 {
                for operation in operations {
                    for rows in [LinearRows::All, LinearRows::Selected(&selected)] {
                        let (float, bf16) = expected(&weights, operation, &input, rows);
                        let mut actual_float = vec![f32::NAN; float.len()];
                        let mut actual_bf16 = vec![Bf16::from_bits(0xffff); bf16.len()];
                        executor.project_f32_into(operation, &input, rows, &mut actual_float, control)
                            .map_err(HostedError::Native)?;
                        executor.project_bf16_into(operation, &input, rows, &mut actual_bf16, control)
                            .map_err(HostedError::Native)?;
                        assert_eq!(actual_float.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            float.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                        assert_eq!(actual_bf16, bf16);
                    }
                }
            }
            assert_eq!(dropped.load(Ordering::SeqCst), 0, "fixed buffers survive every command");
            Ok(executor.stats())
        }).unwrap();
        assert_eq!(stats.team_width, width); assert_eq!(stats.completed_projections, 64);
        assert_eq!(stats.completed_rows, 304);
        assert_eq!(stats.completed_worker_rows, match width {
            2 => 136, 3 => 200, 7 => 236, 16 => 240, _ => unreachable!(),
        });
        assert_eq!(dropped.load(Ordering::SeqCst), width - 1);
    }
}

#[test]
fn every_available_kernel_keeps_selected_row_order_and_idle_workers_do_not_invent_work() {
    let fixture = ProjectionFixture::new(31); let input = fixture.input();
    for kind in Int8KernelKind::ALL {
        let Ok(kernel) = Int8Kernel::detect(kind) else { continue; };
        let weights = fixture.weights().with_kernel(kernel);
        let mut workspace = Workspace::new(7).unwrap();
        let shared = RunShared::new(limits(10_000), CancellationToken::default()).unwrap();
        let stats = run_scope(&Cx::for_testing(), &weights, &mut workspace, 7, kernel, shared,
            |control, executor| {
                assert_eq!(executor.kernel_kind(), kind);
                let rows = LinearRows::Selected(&[9]);
                let (expected, _) = expected(&weights, Int8ProjectionOp::Head, &input, rows);
                let mut output = [f32::NAN];
                executor.project_f32_into(Int8ProjectionOp::Head, &input, rows, &mut output, control)
                    .map_err(HostedError::Native)?;
                assert_eq!(output[0].to_bits(), expected[0].to_bits());
                Ok(executor.stats())
            }).unwrap();
        assert_eq!(stats.completed_projections, 1);
        assert_eq!(stats.completed_rows, 1); assert_eq!(stats.completed_worker_rows, 0);
    }
}

#[test]
fn every_precancelled_cause_survives_formation_drain_without_entering_task_work() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let fixture = ProjectionFixture::new(1); let weights = fixture.weights();
        let mut workspace = Workspace::new(3).unwrap(); let dropped = track_children(&mut workspace);
        let token = CancellationToken::default(); token.cancel(cause);
        let shared = RunShared::new(limits(1000), token).unwrap();
        let error = run_scope(&Cx::for_testing(), &weights, &mut workspace, 3, Int8Kernel::scalar(),
            shared, |_, _| -> Result<(), HostedError> { panic!("cancelled request entered task work") }).unwrap_err();
        assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind, .. }, .. } if kind == cause));
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
    }
}

#[test]
fn formation_and_inflight_cancellation_share_the_original_quota_and_cause() {
    for checkpoints in [2, 3] {
        let fixture = ProjectionFixture::new(2); let weights = fixture.weights();
        let mut workspace = Workspace::new(7).unwrap(); let dropped = track_children(&mut workspace);
        let shared = RunShared::new(limits(checkpoints), CancellationToken::default()).unwrap();
        let error = run_scope(&Cx::for_testing(), &weights, &mut workspace, 7, Int8Kernel::scalar(),
            shared, |_, _| -> Result<(), HostedError> { panic!("partial team entered task work") }).unwrap_err();
        assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind: DecodeCancellationKind::PollQuota, .. }, .. }));
        assert_eq!(dropped.load(Ordering::SeqCst), 6);
    }
    struct StopOnSecond<'a> { inner: &'a mut RunControl, token: CancellationToken, calls: usize }
    impl DecodeStepControl for StopOnSecond<'_> {
        fn checkpoint(&mut self, step: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1;
            if self.calls == 2 { self.token.cancel(DecodeCancellationKind::User); }
            self.inner.checkpoint(step)
        }
    }
    let fixture = ProjectionFixture::new(3); let weights = fixture.weights(); let input = fixture.input();
    let mut workspace = Workspace::new(4).unwrap(); let dropped = track_children(&mut workspace);
    let token = CancellationToken::default();
    let shared = RunShared::new(limits(10_000), token.clone()).unwrap();
    let error = run_scope(&Cx::for_testing(), &weights, &mut workspace, 4, Int8Kernel::scalar(), shared,
        |control, executor| {
            let error = executor.project_f32_into(Int8ProjectionOp::Head, &input, LinearRows::All,
                &mut [0.0; 10], &mut StopOnSecond { inner: control, token, calls: 0 }).unwrap_err();
            assert_eq!(executor.stats().completed_projections, 0);
            Err::<(), _>(HostedError::Native(error))
        }).unwrap_err();
    assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind: DecodeCancellationKind::User, .. }, .. }));
    assert_eq!(dropped.load(Ordering::SeqCst), 3);
}

#[test]
fn invalid_shape_poisoning_refuses_all_later_commands_without_completed_rows() {
    let fixture = ProjectionFixture::new(4); let weights = fixture.weights(); let input = fixture.input();
    let mut workspace = Workspace::new(3).unwrap(); let dropped = track_children(&mut workspace);
    let shared = RunShared::new(limits(10_000), CancellationToken::default()).unwrap();
    let result = run_scope(&Cx::for_testing(), &weights, &mut workspace, 3, Int8Kernel::scalar(), shared,
        |control, executor| {
            let mut output = [13.0; 9];
            let error = executor.project_f32_into(Int8ProjectionOp::Head, &input, LinearRows::All,
                &mut output, control).unwrap_err();
            assert_eq!(error, StrictInt8Error::ProjectionExecutor); assert_eq!(output, [13.0; 9]);
            assert_eq!(executor.project_f32_into(Int8ProjectionOp::Head, &input, LinearRows::All,
                &mut [0.0; 10], control), Err(StrictInt8Error::ProjectionExecutor));
            assert_eq!(executor.stats().completed_rows, 0);
            Err::<(), _>(HostedError::Native(error))
        });
    assert!(result.is_err()); assert_eq!(dropped.load(Ordering::SeqCst), 2);
}

#[test]
fn cancellation_during_physical_drain_prevents_a_successful_publication() {
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Deadline, DecodeCancellationKind::Shutdown] {
        let fixture = ProjectionFixture::new(6); let weights = fixture.weights();
        let token = CancellationToken::default();
        let mut workspace = Workspace::new(3).unwrap();
        workspace.cancel_on_exit = Some((token.clone(), cause));
        let dropped = track_children(&mut workspace);
        let shared = RunShared::new(limits(10_000), token).unwrap();
        let error = run_scope(&Cx::for_testing(), &weights, &mut workspace, 3, Int8Kernel::scalar(),
            shared, |_, _| Ok(())).unwrap_err();
        assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind, .. }, .. } if kind == cause));
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
    }
}

#[test]
fn worker_and_coordinator_panics_join_every_child_before_scope_completion() {
    let fixture = ProjectionFixture::new(5); let weights = fixture.weights(); let input = fixture.input();
    let mut workspace = Workspace::new(4).unwrap(); workspace.panic_worker = Some(1);
    let dropped = track_children(&mut workspace);
    let shared = RunShared::new(limits(10_000), CancellationToken::default()).unwrap();
    let error = run_scope(&Cx::for_testing(), &weights, &mut workspace, 4, Int8Kernel::scalar(), shared,
        |control, executor| executor.project_f32_into(Int8ProjectionOp::Head, &input, LinearRows::All,
            &mut [0.0; 10], control).map_err(HostedError::Native)).unwrap_err();
    assert!(matches!(error, HostedError::Scope {
        source: asupersync::cx::ScopedCpuError::ChildPanicked { .. }, .. }));
    assert_eq!(dropped.load(Ordering::SeqCst), 3);

    struct PanicOnSecond<'a> { inner: &'a mut RunControl, calls: usize }
    impl DecodeStepControl for PanicOnSecond<'_> {
        fn checkpoint(&mut self, step: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1;
            assert_ne!(self.calls, 2, "injected coordinator panic after dispatch");
            self.inner.checkpoint(step)
        }
    }
    let mut workspace = Workspace::new(4).unwrap(); let dropped = track_children(&mut workspace);
    let shared = RunShared::new(limits(10_000), CancellationToken::default()).unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| run_scope(&Cx::for_testing(), &weights,
        &mut workspace, 4, Int8Kernel::scalar(), shared, |control, executor| {
            executor.project_f32_into(Int8ProjectionOp::Head, &input, LinearRows::All,
                &mut [0.0; 10], &mut PanicOnSecond { inner: control, calls: 0 }).map_err(HostedError::Native)
        })));
    assert!(result.is_err()); assert_eq!(dropped.load(Ordering::SeqCst), 3);
}

mod prefill;

struct EmptyStream;
impl crate::batch::BatchProcessor for EmptyStream {
    type Args = (); type Prepared = (); type Output = ();
    fn prepare(&mut self, _: crate::batch::BatchDocument<()>) -> Result<(), crate::batch::BatchItemFailure> {
        panic!("empty fixture stream produced an item")
    }
    fn planned_work(&self, _: &()) -> crate::batch::BatchWork { crate::batch::BatchWork::default() }
    fn execute<C: DecodeStepControl>(&mut self, _: (), _: &mut C) -> Result<(), crate::batch::BatchItemFailure> {
        panic!("empty fixture stream executed an item")
    }
}
fn stage_empty_footer(writer: &mut Vec<u8>, control: &mut RunControl)
    -> Result<crate::batch::deferred::DeferredCompletion, HostedError> {
    crate::batch::deferred::run_ndjson_deferred(&mut std::io::Cursor::new(b""), writer,
        &mut EmptyStream, crate::batch::BatchLimits::default(), control).map_err(HostedError::Batch)
}

#[test]
fn scoring_footer_follows_actual_join_and_spends_the_original_control_allowance() {
    let fixture = ProjectionFixture::new(41); let weights = fixture.weights(); let input = fixture.input();
    let mut workspace = Workspace::with_batch_rows(3, 4).unwrap();
    let dropped = track_children(&mut workspace); let cx = Cx::for_testing();
    let shared = RunShared::new(limits(10000), CancellationToken::default()).unwrap();
    let original_deadline = shared.deadline; let mut output = Vec::new();
    let completion = run_scope(&cx, &weights, &mut workspace, 3, Int8Kernel::scalar(),
        Arc::clone(&shared), |control, executor| {
            executor.project_f32_into(Int8ProjectionOp::Head, &input, LinearRows::All,
                &mut [0.0; 10], control).map_err(HostedError::Native)?;
            assert_eq!(dropped.load(Ordering::SeqCst), 0);
            stage_empty_footer(&mut output, control)
        }).unwrap();
    assert_eq!(dropped.load(Ordering::SeqCst), 2);
    assert!(!std::str::from_utf8(&output).unwrap().contains("run_complete"));
    let remaining = shared.state.lock().unwrap().remaining;
    let summary = finalize_after_join(&cx, &shared, completion, |completion, control| {
        assert_eq!(dropped.load(Ordering::SeqCst), 2, "footer observed a live child");
        assert!(Arc::ptr_eq(&control.shared, &shared));
        assert_eq!(control.shared.deadline, original_deadline);
        completion.publish(&mut output, control).map_err(HostedError::Batch)
    }).unwrap();
    assert_eq!(summary.succeeded, 0); assert_eq!(summary.failed, 0);
    assert!(shared.state.lock().unwrap().remaining < remaining);
    assert_eq!(std::str::from_utf8(&output).unwrap().matches("run_complete").count(), 1);
}

#[test]
fn every_post_join_cancellation_cause_suppresses_the_terminal_stream_frame() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let fixture = ProjectionFixture::new(42); let weights = fixture.weights();
        let mut workspace = Workspace::new(3).unwrap(); let dropped = track_children(&mut workspace);
        let cx = Cx::for_testing(); let token = CancellationToken::default();
        let shared = RunShared::new(limits(10000), token.clone()).unwrap();
        let mut output = Vec::new();
        let completion = run_scope(&cx, &weights, &mut workspace, 3, Int8Kernel::scalar(),
            Arc::clone(&shared), |control, _| stage_empty_footer(&mut output, control)).unwrap();
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
        let prefix = output.clone(); token.cancel(cause);
        let error = finalize_after_join(&cx, &shared, completion, |_, _| -> Result<(), HostedError> {
            panic!("cancelled finalizer attempted footer IO")
        }).unwrap_err();
        assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind, .. }, .. } if kind == cause));
        assert_eq!(output, prefix);
        assert!(!std::str::from_utf8(&output).unwrap().contains("run_complete"));
    }
}

#[test]
fn exhausted_original_quota_after_join_cannot_be_replaced_for_footer_delivery() {
    let fixture = ProjectionFixture::new(43); let weights = fixture.weights();
    let mut workspace = Workspace::new(3).unwrap(); let dropped = track_children(&mut workspace);
    let cx = Cx::for_testing(); let shared = RunShared::new(limits(10000), CancellationToken::default()).unwrap();
    let mut output = Vec::new();
    let completion = run_scope(&cx, &weights, &mut workspace, 3, Int8Kernel::scalar(),
        Arc::clone(&shared), |control, _| stage_empty_footer(&mut output, control)).unwrap();
    assert_eq!(dropped.load(Ordering::SeqCst), 2);
    shared.state.lock().unwrap().remaining = 0;
    let error = finalize_after_join(&cx, &shared, completion, |_, _| -> Result<(), HostedError> {
        panic!("footer received a renewed checkpoint allowance")
    }).unwrap_err();
    assert!(matches!(error, HostedError::Stopped {
        stop: RunStop { kind: DecodeCancellationKind::PollQuota, .. }, .. }));
    assert!(!std::str::from_utf8(&output).unwrap().contains("run_complete"));
}

#[test]
fn cancellation_at_worker_exit_never_reaches_the_scoring_footer_finalizer() {
    let fixture = ProjectionFixture::new(44); let weights = fixture.weights();
    let cx = Cx::for_testing(); let token = CancellationToken::default();
    let shared = RunShared::new(limits(10000), token.clone()).unwrap();
    let mut workspace = Workspace::new(3).unwrap();
    workspace.cancel_on_exit = Some((token, DecodeCancellationKind::Shutdown));
    let dropped = track_children(&mut workspace); let mut output = Vec::new();
    let result = run_scope(&cx, &weights, &mut workspace, 3, Int8Kernel::scalar(),
        Arc::clone(&shared), |control, _| stage_empty_footer(&mut output, control))
        .and_then(|completion| finalize_after_join(&cx, &shared, completion,
            |_, _| -> Result<(), HostedError> { panic!("worker-exit stop published success") }));
    assert!(matches!(result, Err(HostedError::Stopped {
        stop: RunStop { kind: DecodeCancellationKind::Shutdown, .. }, .. })));
    assert_eq!(dropped.load(Ordering::SeqCst), 2);
    assert!(!std::str::from_utf8(&output).unwrap().contains("run_complete"));
}
