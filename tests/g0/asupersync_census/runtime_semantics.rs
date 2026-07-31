//! OQ-35 census — runtime-semantics items observable without a live runtime.
//!
//! Verdict vocabulary (bead franken_nlp-idt): every census item is either
//! RATIFIED at the pin with the observation logged, or ABSENT_WITH_FALLBACK
//! naming the hand-rolled replacement, or FAIL. A verdict here is only ever
//! derived from what this harness observes against the pinned crate.
//!
//! Residual items for the runtime-backed continuation of this census (they
//! need a constructed runtime or Lab and are NOT emitted from this file):
//! cast-enqueue-only, try-cast overflow policies, preset worker/blocking
//! observation, and Lab determinism + crashpack replay.

use std::{
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll, Waker},
};

use asupersync::combinator::first_ok_outcomes;
use asupersync::cx::{Cx, cap};
use asupersync::plan::execute::capture;
use asupersync::{Budget, CancelKind, CancelReason, CapabilityBudget, Outcome, PanicPayload};

fn census(item: &str, result: &str, evidence: &str) {
    println!("G0_CENSUS item={item} RESULT={result} evidence={evidence}");
}

/// The pin's CancelKind carries exactly the eleven doctrine variants, each
/// serde round-trippable; the match below is exhaustive, so a twelfth variant
/// added upstream would fail this census at compile time.
#[test]
fn cancelkind_eleven_variants_present_and_round_trip() {
    let all = [
        CancelKind::User,
        CancelKind::Timeout,
        CancelKind::Deadline,
        CancelKind::PollQuota,
        CancelKind::CostBudget,
        CancelKind::FailFast,
        CancelKind::RaceLost,
        CancelKind::ParentCancelled,
        CancelKind::ResourceUnavailable,
        CancelKind::Shutdown,
        CancelKind::LinkedExit,
    ];
    assert_eq!(
        all.len(),
        11,
        "doctrine demands exactly eleven CancelKind variants"
    );
    for kind in all {
        let encoded = serde_json::to_string(&kind).expect("CancelKind serializes");
        let decoded: CancelKind = serde_json::from_str(&encoded).expect("CancelKind deserializes");
        assert_eq!(kind, decoded, "round-trip must preserve the variant");
        let name = match kind {
            CancelKind::User => "User",
            CancelKind::Timeout => "Timeout",
            CancelKind::Deadline => "Deadline",
            CancelKind::PollQuota => "PollQuota",
            CancelKind::CostBudget => "CostBudget",
            CancelKind::FailFast => "FailFast",
            CancelKind::RaceLost => "RaceLost",
            CancelKind::ParentCancelled => "ParentCancelled",
            CancelKind::ResourceUnavailable => "ResourceUnavailable",
            CancelKind::Shutdown => "Shutdown",
            CancelKind::LinkedExit => "LinkedExit",
        };
        println!("G0_CENSUS item=cancelkind-eleven variant={name} round_trip=ok");
    }
    census(
        "cancelkind-eleven",
        "RATIFIED",
        "11-variant-exhaustive-match+serde-round-trip",
    );
}

/// first_ok_outcomes classifies an ALREADY-COMPLETED outcome vector: selection
/// is first success in INPUT order (not best, not latest), earlier failures
/// are preserved in order, and nothing here drives or cancels children. The
/// sequential fallback for live first-success remains the explicit ordered
/// for/await loop documented in the bead.
#[test]
fn first_ok_outcomes_is_input_order_classification_only() {
    let outcomes: Vec<Outcome<&str, &str>> = vec![
        Outcome::Err("first-failure"),
        Outcome::Ok("first-success-in-order"),
        Outcome::Ok("later-success-must-lose"),
    ];
    let picked = first_ok_outcomes(outcomes);
    let success = picked.success.expect("a success exists");
    assert_eq!(
        success.index, 1,
        "selection is first success in input order"
    );
    assert_eq!(picked.total, 3);
    assert_eq!(
        picked.failures.len(),
        1,
        "failures before the winner are preserved"
    );
    assert_eq!(picked.failures[0].0, 0, "failure indices keep input order");
    println!("G0_CENSUS item=first-ok-sequential case=mixed picked_index=1 failures_before=1");

    // OBSERVED at the pin: Cancelled and Panicked are chain-stopping — the
    // classifier returns at the first severity-terminal outcome and every
    // later entry stays unclassified. Here the Panicked at index 2 is never
    // reached, so had_panic is false and only two failures are recorded.
    let all_fail: Vec<Outcome<&str, &str>> = vec![
        Outcome::Err("a"),
        Outcome::Cancelled(CancelReason::new(CancelKind::Timeout)),
        Outcome::Panicked(PanicPayload::new("census probe panic payload")),
    ];
    let none = first_ok_outcomes(all_fail);
    assert!(none.success.is_none());
    assert_eq!(
        none.failures.len(),
        2,
        "cancellation stops the chain; index 2 unclassified"
    );
    assert!(
        none.was_cancelled,
        "cancellation is surfaced, not swallowed"
    );
    assert!(
        !none.had_panic,
        "the panic after the chain-stopping cancel is never seen"
    );
    println!("G0_CENSUS item=first-ok-sequential case=cancel-stops-chain classified=2of3");

    let panic_stops: Vec<Outcome<&str, &str>> = vec![
        Outcome::Err("a"),
        Outcome::Panicked(PanicPayload::new("census probe panic payload")),
        Outcome::Err("never-classified"),
    ];
    let stopped = first_ok_outcomes(panic_stops);
    assert!(stopped.success.is_none());
    assert_eq!(
        stopped.failures.len(),
        2,
        "panic stops the chain; index 2 unclassified"
    );
    assert!(stopped.had_panic);
    assert!(!stopped.was_cancelled);
    println!("G0_CENSUS item=first-ok-sequential case=panic-stops-chain classified=2of3");

    let empty: Vec<Outcome<&str, &str>> = Vec::new();
    let nothing = first_ok_outcomes(empty);
    assert!(nothing.success.is_none());
    assert_eq!(nothing.total, 0);
    println!("G0_CENSUS item=first-ok-sequential case=empty total=0");

    census(
        "first-ok-sequential",
        "RATIFIED",
        "classification-only+input-order-selection+cancel-and-panic-are-chain-stopping",
    );
}

/// Typed Budget carries deadline (Option<Time>), poll_quota (u32), cost_quota
/// (Option<u64>) and priority (u8); CapabilityBudget exists as the resource
/// envelope type at the crate root. Freezing the project-unit conversion is
/// the residual executable probe; until then no prose treats milliseconds as
/// a complete budget.
#[test]
fn budget_and_capability_budget_shapes_match_doctrine() {
    let budget = Budget::new();
    assert!(budget.deadline.is_none(), "default budget has no deadline");
    let _poll_quota: u32 = budget.poll_quota;
    let _cost_quota: Option<u64> = budget.cost_quota;
    let _priority: u8 = budget.priority;
    println!(
        "G0_CENSUS item=budget-typed default_poll_quota={} default_cost_quota={:?} default_priority={}",
        budget.poll_quota, budget.cost_quota, budget.priority
    );

    let capability = core::any::type_name::<CapabilityBudget>();
    assert!(capability.ends_with("CapabilityBudget"));
    println!("G0_CENSUS item=budget-typed capability_budget_type={capability}");

    census(
        "budget-typed",
        "RATIFIED",
        "deadline-pollquota-costquota-priority-fields+capability-budget-present",
    );
}

/// Preset builder values observed on constructed runtimes, not read from
/// prose: current_thread pins one worker; the deterministic default is the
/// host-independent DEFAULT_WORKER_THREADS = 4; high_throughput doubles it
/// (workers 8, steal batch 32) — never blind-pair it with a physical-core-wide
/// scoped team; low_latency trades throughput for tail latency (steal batch 4,
/// poll budget 32).
#[test]
fn preset_builder_values_observed_on_constructed_runtimes() {
    use asupersync::runtime::{RuntimeBuilder, RuntimeConfig};

    assert_eq!(
        RuntimeConfig::DEFAULT_WORKER_THREADS,
        4,
        "deterministic host-independent default worker count"
    );

    let current = RuntimeBuilder::current_thread()
        .build()
        .expect("current_thread runtime builds");
    assert_eq!(current.config().worker_threads, 1);
    println!(
        "G0_CENSUS item=preset-values preset=current_thread workers={}",
        current.config().worker_threads
    );
    drop(current);

    let default_rt = RuntimeBuilder::multi_thread()
        .build()
        .expect("multi_thread runtime builds");
    assert_eq!(
        default_rt.config().worker_threads,
        RuntimeConfig::DEFAULT_WORKER_THREADS
    );
    println!(
        "G0_CENSUS item=preset-values preset=multi_thread workers={}",
        default_rt.config().worker_threads
    );
    drop(default_rt);

    let throughput = RuntimeBuilder::high_throughput()
        .build()
        .expect("high_throughput runtime builds");
    assert_eq!(
        throughput.config().worker_threads,
        RuntimeConfig::DEFAULT_WORKER_THREADS * 2,
        "high_throughput doubles the deterministic default, not the host core count"
    );
    assert_eq!(throughput.config().steal_batch_size, 32);
    println!(
        "G0_CENSUS item=preset-values preset=high_throughput workers={} steal_batch={}",
        throughput.config().worker_threads,
        throughput.config().steal_batch_size
    );
    drop(throughput);

    let latency = RuntimeBuilder::low_latency()
        .build()
        .expect("low_latency runtime builds");
    assert_eq!(latency.config().steal_batch_size, 4);
    assert_eq!(latency.config().poll_budget, 32);
    println!(
        "G0_CENSUS item=preset-values preset=low_latency steal_batch={} poll_budget={}",
        latency.config().steal_batch_size,
        latency.config().poll_budget
    );
    drop(latency);

    census(
        "preset-values",
        "RATIFIED",
        "observed-on-built-runtimes:current1+default4+throughput8x32+latency4x32",
    );
}

/// `Cx::current()` retains the default `Cx<cap::All>` static type even while a
/// restricted context changes the capability *metadata* it reports. That
/// metadata observation is not an authority boundary: at this pin the
/// underlying handles used by `now`, random, and spawn do not consult it.
/// Product code must therefore pass an explicit narrowed `Cx` to leaves and
/// must never use ambient lookup as least-authority enforcement.
#[test]
fn ambient_current_capability_snapshot_is_not_authority_enforcement() {
    let full = Cx::for_testing();
    let _outer = Cx::set_current(Some(full.clone()));

    let unrestricted: Cx = Cx::current().expect("outer full context is installed");
    let unrestricted_capabilities = unrestricted.capabilities();
    assert!(unrestricted_capabilities.spawn);
    assert!(unrestricted_capabilities.time);
    assert!(unrestricted_capabilities.entropy);
    assert!(unrestricted_capabilities.io);
    assert!(unrestricted_capabilities.remote);

    let leaf: Cx<cap::None> = full.restrict::<cap::None>();
    {
        let _restriction = leaf.set_current_restricted();
        let ambient: Cx = Cx::current().expect("restricted context remains installed");
        let _: Cx<cap::All> = ambient.clone();
        let capabilities = ambient.capabilities();

        // This proves only the present metadata observation. It must not be
        // promoted to an effects-enforcement claim: `ambient` is statically
        // all-capability and its privileged methods use their stored handles.
        assert!(!capabilities.spawn, "restricted metadata omits SPAWN");
        assert!(!capabilities.time, "restricted metadata omits TIME");
        assert!(!capabilities.entropy, "restricted metadata omits RANDOM");
        assert!(!capabilities.io, "restricted metadata omits IO");
        assert!(!capabilities.remote, "restricted metadata omits REMOTE");
    }

    let restored: Cx = Cx::current().expect("outer full context is restored");
    let restored_capabilities = restored.capabilities();
    assert!(restored_capabilities.spawn);
    assert!(restored_capabilities.time);
    assert!(restored_capabilities.entropy);
    assert!(restored_capabilities.io);
    assert!(restored_capabilities.remote);

    census(
        "ambient-current-authority",
        "ABSENT_WITH_FALLBACK",
        "current-returns-static-all;restricted-capabilities-are-metadata-only;fallback=explicit-narrowed-cx-parameter-no-ambient-leaf-lookup",
    );
}

/// `ExecPlan::first_ok` is not the sequential mirror fallback required by
/// `fnlp pull`: its pinned implementation drives every child to completion
/// before it inspects the input-ordered result vector.  This probe makes both
/// facts observable with a completed first success and a later child whose
/// side effect would be absent under short-circuiting.
#[test]
fn execplan_first_ok_drives_every_child_before_input_order_selection() {
    let child_runs = Arc::new(AtomicUsize::new(0));
    let first_runs = Arc::clone(&child_runs);
    let second_runs = Arc::clone(&child_runs);
    let third_runs = Arc::clone(&child_runs);
    let plan = capture(move |capture| {
        let first = capture.labeled_leaf("first-success", async move {
            first_runs.fetch_add(1, Ordering::SeqCst);
            10_u8
        });
        let second = capture.labeled_leaf("second-success", async move {
            second_runs.fetch_add(1, Ordering::SeqCst);
            20_u8
        });
        let third = capture.labeled_leaf("late-success", async move {
            third_runs.fetch_add(1, Ordering::SeqCst);
            30_u8
        });
        capture.first_ok([first, second, third], |_| true)
    })
    .expect("first_ok capture has a nonempty tree");

    let execution_cx = Cx::for_testing();
    let selected = poll_immediately_ready(plan.execute_scalar(&execution_cx))
        .expect("all immediate children complete and a success is selected");
    assert_eq!(
        selected, 10,
        "first_ok selects the first input-order success"
    );
    assert_eq!(
        child_runs.load(Ordering::SeqCst),
        3,
        "the late child completed despite an already-complete first success"
    );
    println!(
        "G0_CENSUS item=execplan-first-ok case=drive-all children_completed=3 selected_input_index=0 loser_cancelled=false"
    );
    census(
        "execplan-first-ok",
        "RATIFIED",
        "drive-all-concurrently+input-order-selection+no-loser-cancellation",
    );
}

fn poll_immediately_ready<F>(future: F) -> F::Output
where
    F: Future,
{
    let waker = Waker::noop();
    let mut task_cx = Context::from_waker(waker);
    let mut future = std::pin::pin!(future);
    match future.as_mut().poll(&mut task_cx) {
        Poll::Ready(output) => output,
        Poll::Pending => panic!("immediate census futures must resolve in one poll"),
    }
}
