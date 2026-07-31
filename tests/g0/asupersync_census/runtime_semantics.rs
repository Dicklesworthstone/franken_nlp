//! OQ-35 census — runtime-semantics items observable without a live runtime.
//!
//! Verdict vocabulary (bead franken_nlp-idt): every census item is either
//! RATIFIED at the pin with the observation logged, or ABSENT_WITH_FALLBACK
//! naming the hand-rolled replacement, or FAIL. A verdict here is only ever
//! derived from what this harness observes against the pinned crate.
//!
//! Residual items for the runtime-backed continuation of this census (they
//! need a constructed runtime or Lab and are NOT emitted from this file):
//! cast-enqueue-only, try-cast overflow policies, ExecPlan::first_ok
//! drive-all-no-short-circuit, preset worker/blocking observation, Lab
//! determinism + crashpack replay, capability compile-fail suite.

use asupersync::combinator::first_ok_outcomes;
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
    assert_eq!(all.len(), 11, "doctrine demands exactly eleven CancelKind variants");
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
    assert_eq!(success.index, 1, "selection is first success in input order");
    assert_eq!(picked.total, 3);
    assert_eq!(picked.failures.len(), 1, "failures before the winner are preserved");
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
    assert_eq!(none.failures.len(), 2, "cancellation stops the chain; index 2 unclassified");
    assert!(none.was_cancelled, "cancellation is surfaced, not swallowed");
    assert!(!none.had_panic, "the panic after the chain-stopping cancel is never seen");
    println!("G0_CENSUS item=first-ok-sequential case=cancel-stops-chain classified=2of3");

    let panic_stops: Vec<Outcome<&str, &str>> = vec![
        Outcome::Err("a"),
        Outcome::Panicked(PanicPayload::new("census probe panic payload")),
        Outcome::Err("never-classified"),
    ];
    let stopped = first_ok_outcomes(panic_stops);
    assert!(stopped.success.is_none());
    assert_eq!(stopped.failures.len(), 2, "panic stops the chain; index 2 unclassified");
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
    println!("G0_CENSUS summary items=3 ratified=3 absent_with_fallback=0 fail=0 residual=cast-enqueue-only,try-cast-policies,execplan-first-ok,preset-values,lab-determinism,compile-fail-suite");
}
