//! Actual pinned scoped workers and the actual persistent command loop.
//! Tiny projection fixtures are not whole-model quality/parity evidence.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::native_engine::{portable_int8::{ActivationBuffer, LinearRows, ProjectionLedger},
    strict_int4::{weights::projection_fixture, projection::Int4ProjectionKind}};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn limits(checkpoints: u64) -> RunLimits {
    RunLimits { max_elapsed: Duration::from_secs(60), max_checkpoints: checkpoints, cleanup_reserve_bytes: 65536 }
}
fn expected(weights: &Int4ModelWeights, operation: Int4ProjectionOp, input: &ActivationBuffer,
    rows: LinearRows<'_>) -> (Vec<f32>, Vec<Bf16>) {
    let linear = weights.projection(operation).unwrap();
    let count = rows.checked_count(linear.rows()).unwrap();
    let mut floats = vec![0.0; count];
    let mut bf16 = vec![Bf16::default(); count];
    let mut scratch = vec![0; linear.scratch_len()];
    let work = linear.planned_work(rows).unwrap();
    linear.project_f32_into(input, rows, &mut floats, &mut scratch,
        &mut ProjectionLedger::new(work), &mut Continue).unwrap();
    linear.project_bf16_into(input, rows, &mut bf16, &mut scratch,
        &mut ProjectionLedger::new(work), &mut Continue).unwrap();
    (floats, bf16)
}
fn track_children(workspace: &mut Workspace) -> Arc<AtomicUsize> {
    let dropped = Arc::new(AtomicUsize::new(0));
    for child in &mut workspace.children { child.observe_drop(Arc::clone(&dropped)); }
    dropped
}

#[test]
fn one_team_reuses_real_workers_for_full_selected_and_both_projection_rails() {
    for width in [2, 3, 7, 16] {
        let (weights, input) = projection_fixture(4);
        let mut workspace = Workspace::new(width, 4).unwrap();
        let dropped = track_children(&mut workspace);
        let shared = RunShared::new(limits(100_000), CancellationToken::default()).unwrap();
        let stats = run_scope(&Cx::for_testing(), &weights, &mut workspace, width,
            PackedInt4Kernel::scalar(), shared, |weights, control, executor| {
            let selected = [0, 3, 5];
            let operations = [Int4ProjectionOp::Head,
                Int4ProjectionOp::Layer { layer: 3, kind: Int4ProjectionKind::Gate },
                Int4ProjectionOp::Layer { layer: 17, kind: Int4ProjectionKind::Query }];
            for _ in 0..4 {
                for operation in operations {
                    for rows in [LinearRows::All, LinearRows::Selected(&selected)] {
                        let (expected_float, expected_bf16) = expected(weights, operation, &input, rows);
                        let mut actual_float = vec![f32::NAN; expected_float.len()];
                        let mut actual_bf16 = vec![Bf16::from_bits(0xffff); expected_bf16.len()];
                        executor.project_f32_into(operation, &input, rows, &mut actual_float, control).map_err(native_error)?;
                        executor.project_bf16_into(operation, &input, rows, &mut actual_bf16, control).map_err(native_error)?;
                        assert_eq!(actual_float.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            expected_float.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                        assert_eq!(actual_bf16, expected_bf16);
                    }
                }
            }
            assert_eq!(dropped.load(Ordering::SeqCst), 0, "one buffer set per persistent child must survive all operators");
            Ok(executor.stats())
        }).unwrap();
        assert_eq!(stats.team_width, width);
        assert_eq!(stats.completed_projections, 48);
        assert_eq!(stats.completed_rows, 248);
        assert_eq!(stats.completed_worker_rows, match width { 2 => 112, 3 => 160, 7 => 192, 16 => 200, _ => unreachable!() });
        assert_eq!(dropped.load(Ordering::SeqCst), width - 1, "every child buffer drains before the scope returns");
    }
}

#[test]
fn every_precancelled_cause_refuses_before_formation_and_survives_drain() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let (weights, _) = projection_fixture(4);
        let mut workspace = Workspace::new(3, 4).unwrap();
        let dropped = track_children(&mut workspace);
        let token = CancellationToken::default(); token.cancel(cause);
        let shared = RunShared::new(limits(1000), token).unwrap();
        let error = run_scope(&Cx::for_testing(), &weights, &mut workspace, 3,
            PackedInt4Kernel::scalar(), shared, |_, _, _| -> Result<(), HostedError> {
                panic!("pre-cancelled team entered task work")
            }).unwrap_err();
        match error {
            HostedError::Stopped { stop, .. } => assert_eq!(stop.kind, cause),
            other => panic!("wrong cancellation outcome: {other}"),
        }
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
    }
}

#[test]
fn worker_checkpoints_share_one_monotone_request_allowance_and_first_cause() {
    let cx = Cx::for_testing().restrict::<cap::None>();
    let token = CancellationToken::default();
    let shared = RunShared::new(limits(7), token.clone()).unwrap();
    let successful = AtomicUsize::new(0);
    cx.scoped_cpu(3, |scope| {
        for _ in 0..3 {
            let shared = Arc::clone(&shared);
            let leaf = cx.clone();
            let successful = &successful;
            scope.spawn(move |cpu| loop {
                match shared.checkpoint(&leaf, || cpu.checkpoint()) {
                    None => { successful.fetch_add(1, Ordering::SeqCst); }
                    Some(cause) => { assert_eq!(cause, DecodeCancellationKind::PollQuota); break; }
                }
            }).unwrap();
        }
    }).unwrap();
    assert_eq!(successful.load(Ordering::SeqCst), 7);
    token.cancel(DecodeCancellationKind::User);
    assert_eq!(shared.checkpoint(&cx, || cx.checkpoint()), Some(DecodeCancellationKind::PollQuota));
    let error = shared.finish::<()>(Some(Ok(()))).unwrap_err();
    assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind: DecodeCancellationKind::PollQuota, .. }, .. }));
}

#[test]
fn cancellation_during_formation_never_enters_the_task_or_leaves_gated_workers() {
    for checkpoints in [2, 3] {
        let (weights, _) = projection_fixture(4);
        let width = 7;
        let mut workspace = Workspace::new(width, 4).unwrap();
        let dropped = track_children(&mut workspace);
        let shared = RunShared::new(limits(checkpoints), CancellationToken::default()).unwrap();
        let error = run_scope(&Cx::for_testing(), &weights, &mut workspace, width,
            PackedInt4Kernel::scalar(), shared, |_, _, _| -> Result<(), HostedError> {
                panic!("formation cancellation entered task work")
            }).unwrap_err();
        assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind: DecodeCancellationKind::PollQuota, .. }, .. }));
        assert_eq!(dropped.load(Ordering::SeqCst), width - 1);
    }
}

#[test]
fn worker_panic_disconnects_its_reply_and_drains_all_inflight_siblings_before_completion() {
    let (weights, input) = projection_fixture(4);
    let width = 4;
    let mut workspace = Workspace::new(width, 4).unwrap();
    workspace.panic_worker = Some(1);
    let dropped = track_children(&mut workspace);
    let shared = RunShared::new(limits(10_000), CancellationToken::default()).unwrap();
    let error = run_scope(&Cx::for_testing(), &weights, &mut workspace, width,
        PackedInt4Kernel::scalar(), shared, |_, control, executor| {
            let mut output = vec![0.0; 10];
            executor.project_f32_into(Int4ProjectionOp::Head, &input, LinearRows::All,
                &mut output, control).map_err(native_error)?;
            Ok(())
        }).unwrap_err();
    assert!(matches!(error, HostedError::Scope {
        source: asupersync::cx::ScopedCpuError::ChildPanicked { .. }, .. }));
    assert_eq!(dropped.load(Ordering::SeqCst), width - 1);
    // The same completion handoff used by run_blocking only becomes visible
    // after the real scope returned its joined-child panic.
    let (sender, receiver) = mpsc::sync_channel(1);
    Completion { cleanup: None, tracking: None, sender: Some(sender) }.finish(Err::<(), _>(error));
    assert!(receiver.recv().unwrap().unwrap().is_err());
    assert_eq!(dropped.load(Ordering::SeqCst), width - 1);
}

#[test]
fn coordinator_panic_after_dispatch_disconnects_workers_before_implicit_join() {
    struct PanicOnSecond<'a> { inner: &'a mut RunControl, calls: usize }
    impl DecodeStepControl for PanicOnSecond<'_> {
        fn checkpoint(&mut self, step: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1;
            assert_ne!(self.calls, 2, "injected coordinator panic after command dispatch");
            self.inner.checkpoint(step)
        }
    }
    let (weights, input) = projection_fixture(4);
    let width = 4;
    let mut workspace = Workspace::new(width, 4).unwrap();
    let dropped = track_children(&mut workspace);
    let shared = RunShared::new(limits(10_000), CancellationToken::default()).unwrap();
    let outcome = catch_unwind(AssertUnwindSafe(|| run_scope(&Cx::for_testing(), &weights,
        &mut workspace, width, PackedInt4Kernel::scalar(), shared, |_, control, executor| {
            let mut output = vec![0.0; 10];
            executor.project_f32_into(Int4ProjectionOp::Head, &input, LinearRows::All,
                &mut output, &mut PanicOnSecond { inner: control, calls: 0 }).map_err(native_error)?;
            Ok(())
        })));
    assert!(outcome.is_err());
    assert_eq!(dropped.load(Ordering::SeqCst), width - 1);
}
