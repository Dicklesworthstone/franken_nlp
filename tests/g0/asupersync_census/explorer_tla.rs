//! OQ-35 census — bounded exploration and TLA+ export are separate evidence
//! surfaces.  This test records an actual, finite DPOR-style exploration
//! receipt and verifies that an exported TLA+ module is an input artifact,
//! never a claim that TLC executed it.

use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use asupersync::{
    Budget,
    lab::{DporExplorer, ExplorerConfig, LabConfig, LabRuntime},
    trace::TlaExporter,
};

/// A one-shot cooperative yield produces a small but real schedule trace.
struct YieldOnce(bool);

impl Future for YieldOnce {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.0 {
            Poll::Ready(())
        } else {
            self.0 = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

/// Populate a bounded, two-task workload and drive it to quiescence.  The
/// explorer owns the fresh LabRuntime and schedule seed for every invocation.
fn explore_two_task_workload(runtime: &mut LabRuntime) {
    let region = runtime.state.create_root_region(Budget::INFINITE);
    let mut tasks = Vec::new();

    for _ in 0..2 {
        let (task, _) = runtime
            .state
            .create_task(region, Budget::INFINITE, async move {
                YieldOnce(false).await;
            })
            .expect("the Lab creates the bounded exploration task");
        tasks.push(task);
    }

    for task in tasks {
        runtime.scheduler.lock().schedule(task, 0);
    }
    runtime.run_until_quiescent();
    assert!(runtime.is_quiescent(), "the bounded workload must drain");
}

#[test]
fn dpor_explorer_reports_finite_coverage_not_exhaustiveness() {
    const BASE_SEED: u64 = 0x035C_D90A;
    const MAX_RUNS: usize = 4;

    let config = ExplorerConfig::new(BASE_SEED, MAX_RUNS)
        .worker_count(2)
        .max_steps(10_000);
    let mut explorer = DporExplorer::new(config);
    let report = explorer.explore(explore_two_task_workload);
    let coverage = explorer.dpor_coverage();

    assert!(
        (1..=MAX_RUNS).contains(&report.total_runs),
        "exploration must perform a finite nonzero number of runs within the declared budget"
    );
    assert_eq!(
        report.total_runs, coverage.base.total_runs,
        "the DPOR receipt and generic coverage receipt must agree on run count"
    );
    assert_eq!(
        report.unique_classes, coverage.base.equivalence_classes,
        "the DPOR receipt and generic coverage receipt must agree on classes"
    );
    assert_eq!(
        coverage.estimated_class_trend.len(),
        report.total_runs,
        "every bounded run needs one recorded class estimate"
    );
    assert!(
        !report.has_violations(),
        "the intentionally quiescent census workload must have no invariant violation"
    );

    println!(
        "G0_CENSUS item=dpor-explorer case=two-task-yield base_seed={BASE_SEED:#x} max_runs={MAX_RUNS} runs={} classes={} races={} hb_races={} backtracks={} sleep_pruned={} saturated={} scope=bounded-guided-not-exhaustive",
        report.total_runs,
        report.unique_classes,
        coverage.total_races,
        coverage.total_hb_races,
        coverage.total_backtrack_points,
        coverage.sleep_pruned,
        coverage.base.saturation.saturated,
    );
    println!(
        "G0_CENSUS item=dpor-explorer RESULT=RATIFIED evidence=bounded-run-class-race-backtrack-saturation-receipt scope=not-exhaustive"
    );
}

#[test]
fn tla_export_is_a_checked_input_not_a_tlc_result() {
    const SEED: u64 = 0x035C_71A;

    let mut runtime = LabRuntime::new(
        LabConfig::new(SEED)
            .max_steps(10_000)
            .with_default_replay_recording(),
    );
    explore_two_task_workload(&mut runtime);
    let events = runtime.trace().snapshot();
    assert!(
        !events.is_empty(),
        "a TLA+ behavior must be sourced from a non-empty Lab trace"
    );

    let exporter = TlaExporter::from_trace(&events);
    let behavior = exporter.export_behavior("OQ35 Trace Export");
    let skeleton = TlaExporter::export_spec_skeleton("OQ35 Bounded Model");

    assert_eq!(
        exporter.snapshot_count(),
        events.len() + 1,
        "the export includes an initial state plus every trace event"
    );
    assert!(
        behavior
            .source
            .starts_with("---- MODULE OQ35_Trace_Export ----")
    );
    assert!(behavior.source.contains("NoObligationLeaks"));
    assert!(behavior.source.contains("QuiescenceOnClose"));
    assert!(behavior.source.contains("ObligationLinearity"));
    assert!(
        skeleton
            .source
            .starts_with("---- MODULE OQ35_Bounded_Model ----")
    );
    assert!(
        skeleton
            .source
            .contains("CONSTANTS MaxTasks, MaxRegions, MaxObligations")
    );

    println!(
        "G0_CENSUS item=tla-export case=lab-trace seed={SEED:#x} events={} snapshots={} behavior_module={} skeleton_module={} tlc=NOT_RUN boundary=export-is-input-not-model-check",
        events.len(),
        exporter.snapshot_count(),
        behavior.name,
        skeleton.name,
    );
    println!(
        "G0_CENSUS item=tla-export RESULT=RATIFIED evidence=trace-behavior+bounded-skeleton scope=export-only-no-tlc-claim"
    );
}
