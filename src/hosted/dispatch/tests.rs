//! Ownership primitives; real runtime entry is covered by hosted_runtime.rs.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Probe(Arc<AtomicUsize>);
impl Drop for Probe { fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); } }

#[test]
fn discarded_unstarted_work_drops_captures_before_completion_is_observable() {
    let count = Arc::new(AtomicUsize::new(0));
    let probe = Probe(Arc::clone(&count));
    let (sender, receiver) = mpsc::sync_channel(1);
    let package = Package::<_, ()> { work: move || drop(probe),
        signal: Completion { cleanup: None, tracking: None, sender: Some(sender) } };
    drop(package);
    assert!(receiver.recv().unwrap().is_none());
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert!(receiver.recv().is_err(), "terminal handoff must occur exactly once");
}

#[test]
fn completed_output_storage_remains_owned_by_the_receiver() {
    let count = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::sync_channel(1);
    Completion { cleanup: None, tracking: None, sender: Some(sender) }.finish(Ok(Probe(Arc::clone(&count))));
    assert_eq!(count.load(Ordering::SeqCst), 0);
    let output = receiver.recv().unwrap().unwrap().unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 0);
    drop(output);
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert!(receiver.recv().is_err());
}

#[test]
fn disconnected_delivery_still_drops_the_owned_result() {
    let count = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::sync_channel(1);
    drop(receiver);
    Completion { cleanup: None, tracking: None, sender: Some(sender) }.finish(Ok(Probe(Arc::clone(&count))));
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn first_requested_cancellation_is_stable_across_clones() {
    let cancellation = CancellationToken::default();
    let clone = cancellation.clone();
    assert_eq!(cancellation.cause(), None);
    clone.cancel(DecodeCancellationKind::User);
    cancellation.cancel(DecodeCancellationKind::Timeout);
    assert_eq!(clone.cause(), Some(DecodeCancellationKind::User));
    assert_eq!(cancellation.cause(), clone.cause());
}

#[test]
fn all_pinned_runtime_causes_keep_their_distinct_native_classification() {
    for (runtime, native) in [
        (CancelKind::User, DecodeCancellationKind::User),
        (CancelKind::Timeout, DecodeCancellationKind::Timeout),
        (CancelKind::Deadline, DecodeCancellationKind::Deadline),
        (CancelKind::PollQuota, DecodeCancellationKind::PollQuota),
        (CancelKind::CostBudget, DecodeCancellationKind::CostBudget),
        (CancelKind::FailFast, DecodeCancellationKind::FailFast),
        (CancelKind::RaceLost, DecodeCancellationKind::RaceLost),
        (CancelKind::ParentCancelled, DecodeCancellationKind::ParentCancelled),
        (CancelKind::ResourceUnavailable, DecodeCancellationKind::ResourceUnavailable),
        (CancelKind::Shutdown, DecodeCancellationKind::Shutdown),
        (CancelKind::LinkedExit, DecodeCancellationKind::LinkedExit),
    ] { assert_eq!(cancellation_kind(runtime), native); }
}

#[test]
fn nested_pool_entry_refuses_without_clearing_the_outer_marker() {
    let guard = NativeEntry::enter().unwrap();
    assert!(matches!(NativeEntry::enter(), Err(HostedError::Reentrant)));
    assert!(INSIDE_NATIVE.with(Cell::get));
    drop(guard);
    assert!(!INSIDE_NATIVE.with(Cell::get));
    drop(NativeEntry::enter().unwrap());
}

#[test]
fn unwind_restores_native_entry_marker() {
    let result = catch_unwind(|| {
        let _guard = NativeEntry::enter().unwrap();
        panic!("private test panic");
    });
    assert!(result.is_err());
    assert!(!INSIDE_NATIVE.with(Cell::get));
}
