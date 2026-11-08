//! One fixed INT4 projection team for a whole admitted generation request.
//! Only row ranges cross the lanes. The coordinator retains sequence/KV,
//! sampling, native work accounting and deterministic output assembly.
use super::*;
use std::sync::Condvar;
use asupersync::cx::CpuCx;
use crate::orchestrator::{capacity_one_lane, CapacityOneReceiver, CapacityOneSender,
    SealedCpuTeam, SealedTeamWorker, TeamDrainReason};
use crate::native_engine::{int4::PackedInt4Kernel, lmhead::NANBEIGE_VOCAB_SIZE,
    quant_algebra::MAX_MODEL_K, strict_int4::{Int4ModelWeights, StrictInt4Error,
        projection::{self as native_projection, Int4ProjectionExecutor, Int4ProjectionOp,
            Int4ProjectionStats, MAX_INT4_TEAM_WIDTH}}, tensor::Bf16};

mod projection;
use projection::{Command, Peer, Reply, TeamExecutor, WorkerBuffers};

pub(in crate::hosted) fn int4_team_workspace_bytes(width: usize, group_width: usize)
    -> Result<u64, HostedError> {
    native_projection::int4_team_workspace_bytes(width, group_width)
        .map_err(|_| HostedError::Limits("INT4 team workspace arithmetic or width"))
}

pub(in crate::hosted) fn preflight_int4_team(engine: &NlpEngine, limits: RunLimits, width: usize)
    -> Result<(), HostedError> {
    preflight(engine, limits)?;
    let children = width.checked_sub(1).filter(|_| (2..=MAX_INT4_TEAM_WIDTH).contains(&width))
        .ok_or(HostedError::Limits("INT4 team width"))?;
    if children > engine.resources().config().scoped_cpu_children_per_coordinator {
        return Err(HostedError::Limits("INT4 team exceeds installed child envelope"));
    }
    Ok(())
}

struct Workspace {
    children: Vec<WorkerBuffers>, coordinator_scratch: Vec<i8>,
    #[cfg(test)] panic_worker: Option<usize>,
}
impl Workspace {
    fn new(width: usize, group_width: usize) -> Result<Self, HostedError> {
        // The pure quote validates width/group geometry before any allocation.
        int4_team_workspace_bytes(width, group_width)?;
        let mut children = Vec::new();
        children.try_reserve_exact(width - 1).map_err(|_| HostedError::Limits("INT4 team allocation"))?;
        for _ in 1..width {
            children.push(WorkerBuffers::new(NANBEIGE_VOCAB_SIZE.div_ceil(width), group_width)
                .map_err(native_error)?);
        }
        Ok(Self { children, coordinator_scratch: zeroed(group_width).map_err(native_error)?,
            #[cfg(test)] panic_worker: None })
    }
}

fn zeroed<T: Default + Clone>(length: usize) -> Result<Vec<T>, StrictInt4Error> {
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(|_| StrictInt4Error::Allocation)?;
    values.resize(length, T::default());
    Ok(values)
}
fn native_error(error: StrictInt4Error) -> HostedError {
    HostedError::Int4(Int4HostError::Native(error))
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum GateState { Waiting, Released, Aborted }
struct EntryGate { state: Mutex<GateState>, changed: Condvar }
impl EntryGate {
    fn new() -> Self { Self { state: Mutex::new(GateState::Waiting), changed: Condvar::new() } }
    fn wait(&self) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        while *state == GateState::Waiting {
            state = self.changed.wait(state).unwrap_or_else(|p| p.into_inner());
        }
        *state == GateState::Released
    }
    fn release(&self) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if *state == GateState::Waiting { *state = GateState::Released; self.changed.notify_all(); }
    }
    fn abort(&self) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if *state == GateState::Waiting { *state = GateState::Aborted; self.changed.notify_all(); }
    }
}

struct WorkerControl<'a> {
    cpu: &'a CpuCx<cap::None>, cx: Cx<cap::None>, shared: Arc<RunShared>,
    #[cfg(test)] panic_on_command: bool,
}
impl DecodeStepControl for WorkerControl<'_> {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.shared.checkpoint(&self.cx, || self.cpu.checkpoint())
    }
}
struct WorkerExit { team: Arc<SealedCpuTeam>, worker: SealedTeamWorker }
impl Drop for WorkerExit {
    fn drop(&mut self) {
        if std::thread::panicking() { let _ = self.team.worker_panicked(self.worker); }
        else { let _ = self.team.worker_exited(self.worker); }
    }
}

/// Buffers and their process-ledger charge are captured by the blocking work.
/// Every lane is disconnected and every worker has physically joined before
/// the shared dispatch completion signal can release them.
#[allow(clippy::too_many_arguments)]
pub(in crate::hosted) fn run_int4_team<T, F>(engine: &NlpEngine, limits: RunLimits,
    cancellation: CancellationToken, model: ResidentInt4, width: usize, kernel: PackedInt4Kernel, work: F)
    -> Result<T, HostedError>
where T: Send + 'static,
    F: FnOnce(&Int4ModelWeights, &mut RunControl, &mut dyn Int4ProjectionExecutor)
        -> Result<T, HostedError> + Send + 'static {
    // Existing callers retain their exact checkpoint schedule and return their
    // guarded result only after physical join, as before.
    run_int4_team_then(engine, limits, cancellation, model, width, kernel, work,
        |value, _, _| Ok(value))
}

/// Publish a terminal result after the team joins, inside the same tracked
/// blocking invocation and NativeEntry. The finalizer borrows the original
/// deadline/checkpoint state; it cannot create another execution allowance.
#[allow(clippy::too_many_arguments)]
pub(in crate::hosted) fn run_int4_team_with_finalize<T, U, F, G>(engine: &NlpEngine,
    limits: RunLimits, cancellation: CancellationToken, model: ResidentInt4, width: usize,
    kernel: PackedInt4Kernel, work: F, finalize: G) -> Result<U, HostedError>
where T: Send + 'static, U: Send + 'static,
    F: FnOnce(&Int4ModelWeights, &mut RunControl, &mut dyn Int4ProjectionExecutor)
        -> Result<T, HostedError> + Send + 'static,
    G: FnOnce(T, &mut RunControl) -> Result<U, HostedError> + Send + 'static {
    run_int4_team_then(engine, limits, cancellation, model, width, kernel, work,
        move |value, cx, shared| finalize_after_join(cx, shared, value, finalize))
}

#[allow(clippy::too_many_arguments)]
fn run_int4_team_then<T, U, F, G>(engine: &NlpEngine, limits: RunLimits,
    cancellation: CancellationToken, model: ResidentInt4, width: usize, kernel: PackedInt4Kernel,
    work: F, finalize: G) -> Result<U, HostedError>
where T: Send + 'static, U: Send + 'static,
    F: FnOnce(&Int4ModelWeights, &mut RunControl, &mut dyn Int4ProjectionExecutor)
        -> Result<T, HostedError> + Send + 'static,
    G: FnOnce(T, &Cx, &Arc<RunShared>) -> Result<U, HostedError> + Send + 'static {
    preflight_int4_team(engine, limits, width)?;
    if !Arc::ptr_eq(engine.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
    let shared = RunShared::new(limits, cancellation)?;
    let lease = engine.resources().acquire_lease();
    let bytes = int4_team_workspace_bytes(width, model.model_identity().group_width())?;
    let pending = Pending::reserve(&lease, MemoryClass::ActivationScratch, bytes)?;
    let workspace = allocate(pending, || Workspace::new(width, model.model_identity().group_width()))?;
    run_blocking(engine, limits, move |blocking_cx| {
        // Keep the charged aggregate whole: its fields drop storage before
        // reservation even when setup or the coordinator unwinds.
        let mut workspace = workspace;
        let weights = model.weights();
        let physical = run_scope(&blocking_cx, weights, &mut workspace.value, width, kernel,
            Arc::clone(&shared), work)
            .and_then(|value| finalize(value, &blocking_cx, &shared));
        drop(workspace);
        drop(model);
        drop(lease);
        physical
    })
}

fn finalize_after_join<T, U>(blocking_cx: &Cx, shared: &Arc<RunShared>, value: T,
    finalize: impl FnOnce(T, &mut RunControl) -> Result<U, HostedError>) -> Result<U, HostedError> {
    let leaf = blocking_cx.restrict::<cap::None>();
    let _ambient = leaf.clone().set_current_restricted();
    let mut control = RunControl { cx: leaf, shared: Arc::clone(shared) };
    let outcome = if control.checkpoint(0).is_some() {
        drop(value); drop(finalize); None
    } else { Some(finalize(value, &mut control)) };
    // Writer callbacks remain cooperative. A cancellation during final flush
    // still makes the invocation fail; a footer alone never replaces success.
    control.checkpoint(0);
    shared.finish(outcome)
}

fn run_scope<T, F>(blocking_cx: &Cx, weights: &Int4ModelWeights, workspace: &mut Workspace,
    width: usize, kernel: PackedInt4Kernel, shared: Arc<RunShared>, work: F) -> Result<T, HostedError>
where F: FnOnce(&Int4ModelWeights, &mut RunControl, &mut dyn Int4ProjectionExecutor) -> Result<T, HostedError> {
        let leaf = blocking_cx.restrict::<cap::None>();
        let _ambient = leaf.clone().set_current_restricted();
        let team = Arc::new(SealedCpuTeam::new(width - 1));
        let gate = Arc::new(EntryGate::new());
        let mut outcome = None;
        let mut formation_complete = false;
        #[cfg(test)] let panic_worker = workspace.panic_worker;
        let scope_result = leaf.scoped_cpu(width - 1, |scope| {
            let mut executor = TeamExecutor::new(weights, kernel, width,
                std::mem::take(&mut workspace.coordinator_scratch),
                Arc::clone(&team), Arc::clone(&gate), Arc::clone(&shared))?;
            let mut control = RunControl { cx: leaf.clone(), shared: Arc::clone(&shared) };
            // This is the sole formation loop. No operator path has a scope
            // handle or any method capable of creating another child.
            for buffers in std::mem::take(&mut workspace.children) {
                if let Some(cause) = control.checkpoint(0) { return Err(native_error(StrictInt4Error::Cancelled(cause))); }
                let worker = team.form_worker().map_err(|_| native_error(StrictInt4Error::ProjectionExecutor))?;
                let (command_sender, command_receiver) = capacity_one_lane();
                let (reply_sender, reply_receiver) = capacity_one_lane();
                let worker_team = Arc::clone(&team);
                let worker_gate = Arc::clone(&gate);
                let worker_shared = Arc::clone(&shared);
                let worker_leaf = leaf.clone();
                // Store the sender before spawn, so unwinding formation always
                // disconnects every lane already handed to a child.
                executor.peers.push(Peer { command: Some(command_sender), reply: reply_receiver,
                    buffers: Some(buffers) });
                scope.spawn(move |cpu| {
                    if !worker_gate.wait() { return; }
                    if worker_team.worker_started(worker).is_err() { return; }
                    let _exit = WorkerExit { team: worker_team, worker };
                    let _entry = match NativeEntry::enter() {
                        Ok(entry) => entry,
                        Err(_) => { let _ = reply_sender.try_send(Reply::Ready(Err(StrictInt4Error::ProjectionExecutor))); return; }
                    };
                    let _ambient = worker_leaf.clone().set_current_restricted();
                    let mut control = WorkerControl { cpu, cx: worker_leaf, shared: worker_shared,
                        #[cfg(test)] panic_on_command: panic_worker == Some(worker.ordinal()) };
                    let ready = control.checkpoint(0).map_or(Ok(()), |cause| Err(StrictInt4Error::Cancelled(cause)));
                    let stopped = ready.is_err();
                    if reply_sender.try_send(Reply::Ready(ready)).is_err() || stopped { return; }
                    worker_loop(weights, kernel, command_receiver, reply_sender, &mut control);
                }).map_err(|source| HostedError::Scope { source, physical_error: None })?;
            }
            team.seal().map_err(|_| native_error(StrictInt4Error::ProjectionExecutor))?;
            team.release_workers().map_err(|_| native_error(StrictInt4Error::ProjectionExecutor))?;
            gate.release();
            // Ready acknowledgements prevent a zero-work/early-stop request
            // from entering drain before a released worker records its start.
            for peer in &executor.peers {
                match peer.reply.recv() {
                    Ok(Reply::Ready(Ok(()))) => {}
                    Ok(Reply::Ready(Err(error))) => return Err(native_error(error)),
                    _ => return Err(native_error(StrictInt4Error::ProjectionExecutor)),
                }
            }
            formation_complete = true;
            outcome = if control.checkpoint(0).is_some() { drop(work); None }
                else { Some(work(weights, &mut control, &mut executor)) };
            // The final checkpoint is still covered by the same quota and
            // first-cause state as every worker's morsel checks.
            let stopped = control.checkpoint(0).is_some();
            executor.complete = matches!(&outcome, Some(Ok(_))) && !stopped;
            Ok(())
            // Executor drop disconnects lanes before scoped_cpu starts joining.
        });
        // The foundation has now joined all children, even on panic. Never
        // take the shared stop reason while a worker might still update it.
        match scope_result {
            Ok(Ok(())) => {
                if formation_complete && team.join().is_err() {
                    drop(outcome); Err(native_error(StrictInt4Error::ProjectionExecutor))
                } else { shared.finish(outcome) }
            }
            Ok(Err(error)) => shared.finish(Some(Err(error))),
            Err(source) => Err(HostedError::Scope { source,
                physical_error: shared.finish(outcome).err().map(Box::new) }),
        }
}

fn worker_loop(weights: &Int4ModelWeights, kernel: PackedInt4Kernel,
    commands: CapacityOneReceiver<Command>, replies: CapacityOneSender<Reply>, control: &mut WorkerControl<'_>) {
    while let Ok(command) = commands.recv() {
        #[cfg(test)]
        if control.panic_on_command { panic!("injected INT4 projection worker panic"); }
        let reply = projection::execute_command(weights, kernel, command, control);
        let stopped = reply.result.is_err();
        if stopped { control.shared.stop(DecodeCancellationKind::FailFast); }
        // A failed/full/disconnected reply is terminal. A worker never waits
        // for coordinator receive during panic drain.
        if replies.try_send(Reply::Projected(reply)).is_err() || stopped { return; }
    }
}

#[cfg(test)] mod tests;
