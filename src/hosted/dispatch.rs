//! One owned blocking crossing, with separate wrapper and physical completion.
use super::*;
use std::{cell::Cell, panic::{catch_unwind, AssertUnwindSafe}, sync::{mpsc, Mutex}, time::Instant};
use asupersync::{cx::{cap, Cx}, types::{CancelKind, CancelReason}};
use crate::{BlockingClosureGuard, native_engine::decode::{DecodeCancellationKind, DecodeStepControl}};

thread_local! { static INSIDE_NATIVE: Cell<bool> = const { Cell::new(false) }; }
struct NativeEntry;
impl NativeEntry {
    fn enter() -> Result<Self, HostedError> {
        INSIDE_NATIVE.with(|flag| if flag.replace(true) { Err(HostedError::Reentrant) } else { Ok(Self) })
    }
}
impl Drop for NativeEntry { fn drop(&mut self) { INSIDE_NATIVE.with(|flag| flag.set(false)); } }

/// Explicit cooperative cancellation shared by a caller and its owned run.
/// The first requested cause wins; it never controls an unrelated runtime task.
#[derive(Clone, Default)]
pub struct CancellationToken(Arc<Mutex<Option<DecodeCancellationKind>>>);
impl CancellationToken {
    pub fn cancel(&self, cause: DecodeCancellationKind) {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if state.is_none() { *state = Some(cause); }
    }
    pub fn cause(&self) -> Option<DecodeCancellationKind> {
        self.0.lock().map(|state| *state).unwrap_or(Some(DecodeCancellationKind::FailFast))
    }
}

pub struct RunStop {
    pub kind: DecodeCancellationKind,
    /// Actual foundation attribution, when the runtime initiated the stop.
    pub runtime_reason: Option<CancelReason>,
    pub runtime_checkpoint_error: Option<asupersync::error::Error>,
}
impl fmt::Debug for RunStop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunStop").field("kind", &self.kind).finish_non_exhaustive()
    }
}
/// The leaf can checkpoint but receives no spawning, network, remote, entropy
/// or timer capability. Native work counters stay in their existing ledgers.
pub struct RunControl {
    cx: Cx<cap::None>, cancellation: CancellationToken, deadline: Instant,
    remaining: u64, stopped: Option<RunStop>,
}
impl RunControl {
    fn stop(&mut self, kind: DecodeCancellationKind) -> Option<DecodeCancellationKind> {
        self.stopped = Some(RunStop { kind, runtime_reason: None, runtime_checkpoint_error: None }); Some(kind)
    }
}
impl DecodeStepControl for RunControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        if let Some(stop) = &self.stopped { return Some(stop.kind); }
        if let Some(cause) = self.cancellation.cause() { return self.stop(cause); }
        if Instant::now() >= self.deadline { return self.stop(DecodeCancellationKind::Deadline); }
        if self.remaining == 0 { return self.stop(DecodeCancellationKind::PollQuota); }
        self.remaining -= 1;
        if let Err(error) = self.cx.checkpoint() {
            let reason = self.cx.cancel_reason();
            let kind = reason.as_ref().map(|r| cancellation_kind(r.kind)).unwrap_or(DecodeCancellationKind::FailFast);
            self.stopped = Some(RunStop { kind, runtime_reason: reason, runtime_checkpoint_error: Some(error) });
            return Some(kind);
        }
        None
    }
}
fn cancellation_kind(kind: CancelKind) -> DecodeCancellationKind {
    match kind {
        CancelKind::User => DecodeCancellationKind::User, CancelKind::Timeout => DecodeCancellationKind::Timeout,
        CancelKind::Deadline => DecodeCancellationKind::Deadline, CancelKind::PollQuota => DecodeCancellationKind::PollQuota,
        CancelKind::CostBudget => DecodeCancellationKind::CostBudget, CancelKind::FailFast => DecodeCancellationKind::FailFast,
        CancelKind::RaceLost => DecodeCancellationKind::RaceLost, CancelKind::ParentCancelled => DecodeCancellationKind::ParentCancelled,
        CancelKind::ResourceUnavailable => DecodeCancellationKind::ResourceUnavailable, CancelKind::Shutdown => DecodeCancellationKind::Shutdown,
        CancelKind::LinkedExit => DecodeCancellationKind::LinkedExit,
    }
}

pub(super) fn preflight(engine: &NlpEngine, limits: RunLimits) -> Result<(), HostedError> {
    limits.validate()?;
    if INSIDE_NATIVE.with(Cell::get) { return Err(HostedError::Reentrant); }
    // The current INT8 kernels are serial. Use exactly the existing coordinator,
    // not a second thread pool or an idle team claimed as parallel inference.
    if engine.resources().config().max_blocking_coordinators != 1 {
        return Err(HostedError::SingleCoordinatorRequired);
    }
    if !engine.resources().has_real_blocking_pool() { return Err(HostedError::MissingRuntimeContext); }
    // Validate the existing runtime-worker/re-entry guard before reservations.
    drop(engine.enter_sync_call().map_err(|_| HostedError::Reentrant)?);
    Ok(())
}

// Field order matters on a queued task that is discarded WITHOUT invocation:
// captured buffers/reservations drop before its completion signal fires.
struct Package<F, T> { work: F, signal: Completion<T> }
struct Completion<T> {
    cleanup: Option<Pending>,
    tracking: Option<Arc<BlockingClosureGuard>>,
    sender: Option<mpsc::SyncSender<Option<Result<T, HostedError>>>>,
}
impl<T> Completion<T> {
    fn finish(mut self, result: Result<T, HostedError>) {
        drop(self.cleanup.take());
        drop(self.tracking.take());
        if let Some(sender) = self.sender.take() { let _ = sender.send(Some(result)); }
    }
}
impl<T> Drop for Completion<T> {
    fn drop(&mut self) {
        drop(self.cleanup.take());
        drop(self.tracking.take());
        if let Some(sender) = self.sender.take() { let _ = sender.send(None); }
    }
}

pub(super) fn run<T, F>(engine: &NlpEngine, limits: RunLimits, cancellation: CancellationToken, work: F)
    -> Result<T, HostedError>
where T: Send + 'static, F: FnOnce(&mut RunControl) -> Result<T, HostedError> + Send + 'static {
    preflight(engine, limits)?;
    let _sync_call = engine.enter_sync_call().map_err(|_| HostedError::Reentrant)?;
    let deadline = Instant::now().checked_add(limits.max_elapsed).ok_or(HostedError::Limits("deadline arithmetic"))?;
    let lease = engine.resources().acquire_lease();
    let cleanup = Pending::reserve(&lease, MemoryClass::AdmissionReserve, limits.cleanup_reserve_bytes)?;
    let completion = Arc::new(lease.register_blocking_closure());
    let (sender, receiver) = mpsc::sync_channel(1);
    let package = Package { work, signal: Completion { cleanup: Some(cleanup), tracking: Some(Arc::clone(&completion)), sender: Some(sender) } };
    let wrapper = catch_unwind(AssertUnwindSafe(|| engine.resources().runtime().block_on(async move {
        let request_cx = Cx::current().ok_or(HostedError::MissingRuntimeContext)?;
        let mut handle = request_cx.spawn_blocking(move |blocking_cx| {
            let Package { work, signal } = package;
            // The catch encloses all native work, scope exit and buffer drops.
            // The public error does not copy panic text into task output.
            let result = catch_unwind(AssertUnwindSafe(|| {
                let _entry = NativeEntry::enter()?;
                let mut outcome = None;
                let scope = blocking_cx.scoped_cpu(0, |_| {
                    let leaf = blocking_cx.restrict::<cap::None>();
                    let _ambient = leaf.clone().set_current_restricted();
                    let mut control = RunControl { cx: leaf, cancellation, deadline,
                        remaining: limits.max_checkpoints, stopped: None };
                    let value = if control.checkpoint(0).is_some() {
                        // Keep work captured until after the cancellation decision.
                        drop(work); None
                    } else { Some(work(&mut control)) };
                    // Cancellation during finalization must suppress success.
                    control.checkpoint(0);
                    outcome = Some(match control.stopped.take() {
                        Some(stop) => Err(HostedError::Stopped { stop,
                            task_error: value.and_then(Result::err).map(Box::new) }),
                        None => value.unwrap_or(Err(HostedError::CompletionMissing)),
                    });
                });
                match scope {
                    Ok(()) => outcome.unwrap_or(Err(HostedError::CompletionMissing)),
                    Err(source) => Err(HostedError::Scope { source,
                        physical_error: outcome.and_then(Result::err).map(Box::new) }),
                }
            })).unwrap_or(Err(HostedError::Panicked));
            signal.finish(result);
        }).map_err(HostedError::Spawn)?;
        handle.join(&request_cx).await.map_err(|source| HostedError::Join { source, physical_error: None })
    })));
    if matches!(&wrapper, Ok(Err(HostedError::Join { source: JoinError::Cancelled(_), .. }))) {
        completion.mark_wrapper_cancelled();
    }
    // The synchronous caller is not an executor worker. Even a cancelled async
    // wrapper MUST wait for this closure-owned channel to complete/disconnect.
    // No timeout here can turn cooperative cancellation into thread preemption.
    let physical = receiver.recv().ok().flatten();
    drop(completion);
    drop(lease);
    match wrapper {
        Ok(Ok(())) => physical.unwrap_or(Err(HostedError::CompletionMissing)),
        Ok(Err(HostedError::Join { source, .. })) => Err(HostedError::Join { source,
            physical_error: physical.and_then(Result::err).map(Box::new) }),
        Ok(Err(error)) => { drop(physical); Err(error) }
        Err(_) => { drop(physical); Err(HostedError::Panicked) }
    }
}

#[cfg(test)] mod tests;
