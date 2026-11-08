//! One owned blocking crossing, with separate wrapper and physical completion.
use super::*;
use std::{cell::Cell, panic::{catch_unwind, AssertUnwindSafe}, sync::{mpsc, Mutex}, time::Instant};
use asupersync::{cx::{cap, Cx}, types::{CancelKind, CancelReason}};
use crate::{BlockingClosureGuard, native_engine::decode::{DecodeCancellationKind, DecodeStepControl}};

mod team;
pub(super) use team::{int4_team_workspace_bytes, preflight_int4_team, run_int4_team,
    run_int4_team_with_finalize};

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
    cx: Cx<cap::None>, shared: Arc<RunShared>,
}
struct RunState { remaining: u64, stopped: Option<RunStop> }
/// Every coordinator/worker checkpoint spends the same monotone allowance.
/// Keeping attribution under the same lock also makes the first stop stable
/// across workers; only the post-join owner may take the terminal reason.
struct RunShared {
    cancellation: CancellationToken, deadline: Instant, state: Mutex<RunState>,
}
impl RunShared {
    fn new(limits: RunLimits, cancellation: CancellationToken) -> Result<Arc<Self>, HostedError> {
        let deadline = Instant::now().checked_add(limits.max_elapsed)
            .ok_or(HostedError::Limits("deadline arithmetic"))?;
        Ok(Arc::new(Self { cancellation, deadline,
            state: Mutex::new(RunState { remaining: limits.max_checkpoints, stopped: None }) }))
    }
    fn checkpoint(&self, cx: &Cx<cap::None>, runtime: impl FnOnce() -> Result<(), asupersync::error::Error>)
        -> Option<DecodeCancellationKind> {
        let mut state = self.state.lock().unwrap_or_else(|poisoned| {
            let mut state = poisoned.into_inner();
            state.stopped.get_or_insert(RunStop { kind: DecodeCancellationKind::FailFast,
                runtime_reason: None, runtime_checkpoint_error: None });
            state
        });
        if let Some(stop) = &state.stopped { return Some(stop.kind); }
        let stop = self.cancellation.cause()
            .or_else(|| (Instant::now() >= self.deadline).then_some(DecodeCancellationKind::Deadline))
            .or_else(|| (state.remaining == 0).then_some(DecodeCancellationKind::PollQuota));
        if let Some(kind) = stop {
            state.stopped = Some(RunStop { kind, runtime_reason: None, runtime_checkpoint_error: None });
            return Some(kind);
        }
        state.remaining -= 1;
        if let Err(error) = runtime() {
            let reason = cx.cancel_reason();
            let kind = reason.as_ref().map(|r| cancellation_kind(r.kind)).unwrap_or(DecodeCancellationKind::FailFast);
            state.stopped = Some(RunStop { kind, runtime_reason: reason, runtime_checkpoint_error: Some(error) });
            return Some(kind);
        }
        None
    }
    fn finish<T>(&self, value: Option<Result<T, HostedError>>) -> Result<T, HostedError> {
        let stop = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).stopped.take();
        match stop {
            Some(stop) => Err(HostedError::Stopped { stop, task_error: value.and_then(Result::err).map(Box::new) }),
            None => value.unwrap_or(Err(HostedError::CompletionMissing)),
        }
    }
    fn stop(&self, kind: DecodeCancellationKind) {
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        state.stopped.get_or_insert_with(|| RunStop {
            kind: self.cancellation.cause().unwrap_or(kind),
            runtime_reason: None, runtime_checkpoint_error: None,
        });
    }
}
impl DecodeStepControl for RunControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.shared.checkpoint(&self.cx, || self.cx.checkpoint())
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
    // Every hosted route shares the one installed blocking coordinator. The
    // explicit INT4 team route separately validates its fixed child envelope.
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
impl<F, T> Package<F, T> {
    // Consume the WHOLE package at invocation, forcing closure capture of the
    // aggregate rather than allowing disjoint field capture to reorder drops.
    fn into_parts(self) -> (F, Completion<T>) { (self.work, self.signal) }
}
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
    let shared = RunShared::new(limits, cancellation)?;
    run_blocking(engine, limits, move |blocking_cx| {
        let mut outcome = None;
        let scope = blocking_cx.scoped_cpu(0, |_| {
            let leaf = blocking_cx.restrict::<cap::None>();
            let _ambient = leaf.clone().set_current_restricted();
            let mut control = RunControl { cx: leaf, shared: Arc::clone(&shared) };
            let value = if control.checkpoint(0).is_some() {
                // Keep work captured until after the cancellation decision.
                drop(work); None
            } else { Some(work(&mut control)) };
            // Cancellation during finalization must suppress success.
            control.checkpoint(0);
            outcome = Some(shared.finish(value));
        });
        match scope {
            Ok(()) => outcome.unwrap_or(Err(HostedError::CompletionMissing)),
            Err(source) => Err(HostedError::Scope { source,
                physical_error: outcome.and_then(Result::err).map(Box::new) }),
        }
    })
}

/// The serial and persistent-team routes share the same physical-completion
/// handoff. The supplied closure must return only after its native scope joins.
fn run_blocking<T, F>(engine: &NlpEngine, limits: RunLimits, work: F) -> Result<T, HostedError>
where T: Send + 'static, F: FnOnce(Cx) -> Result<T, HostedError> + Send + 'static {
    preflight(engine, limits)?;
    let _sync_call = engine.enter_sync_call().map_err(|_| HostedError::Reentrant)?;
    let lease = engine.resources().acquire_lease();
    let cleanup = Pending::reserve(&lease, MemoryClass::AdmissionReserve, limits.cleanup_reserve_bytes)?;
    let completion = Arc::new(lease.register_blocking_closure());
    let (sender, receiver) = mpsc::sync_channel(1);
    let package = Package { work, signal: Completion { cleanup: Some(cleanup), tracking: Some(Arc::clone(&completion)), sender: Some(sender) } };
    let wrapper = catch_unwind(AssertUnwindSafe(|| engine.resources().runtime().block_on(async move {
        let request_cx = Cx::current().ok_or(HostedError::MissingRuntimeContext)?;
        let mut handle = request_cx.spawn_blocking(move |blocking_cx| {
            let (work, signal) = package.into_parts();
            // The catch encloses all native work, scope exit and buffer drops.
            // The public error does not copy panic text into task output.
            let result = catch_unwind(AssertUnwindSafe(|| {
                let _entry = NativeEntry::enter()?;
                work(blocking_cx)
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
