//! One persistent INT8 row team, owned by the existing blocking dispatcher.
//! The coordinator owns sequence order, KV, quantization and native work debit.
use super::{allocate, cap, capacity_one_lane, run_blocking, preflight, Arc, Bf16,
    CancellationToken, CapacityOneReceiver, CapacityOneSender, Cx, DecodeCancellationKind,
    DecodeStepControl, EntryGate, HostedError, MAX_MODEL_K, MemoryClass, NANBEIGE_VOCAB_SIZE,
    NativeEntry, NlpEngine, Pending, ResidentInt8, RunControl, RunLimits, RunShared,
    SealedCpuTeam, TeamDrainReason, WorkerControl, WorkerExit};
use crate::native_engine::{simd::Int8Kernel,
    strict_int8::{CurrentCandidateInt8Model, Int8WeightView, StrictInt8Error,
        projection::{self as native_projection, Int8ProjectionExecutor, Int8ProjectionOp,
            Int8ProjectionStats, MAX_INT8_TEAM_WIDTH}}};

mod projection;
use projection::{Command, Peer, Reply, TeamExecutor, WorkerBuffers};

pub(in crate::hosted) fn int8_team_workspace_bytes(width: usize) -> Result<u64, HostedError> {
    native_projection::int8_team_workspace_bytes(width)
        .map_err(|_| HostedError::Limits("INT8 team workspace arithmetic or width"))
}

pub(in crate::hosted) fn preflight_int8_team(engine: &NlpEngine, limits: RunLimits, width: usize)
    -> Result<(), HostedError> {
    preflight(engine, limits)?;
    let children = width.checked_sub(1).filter(|_| (2..=MAX_INT8_TEAM_WIDTH).contains(&width))
        .ok_or(HostedError::Limits("INT8 team width"))?;
    if children > engine.resources().config().scoped_cpu_children_per_coordinator {
        return Err(HostedError::Limits("INT8 team exceeds installed child envelope"));
    }
    Ok(())
}

struct Workspace {
    children: Vec<WorkerBuffers>,
    #[cfg(test)] panic_worker: Option<usize>,
    #[cfg(test)] cancel_on_exit: Option<(CancellationToken, DecodeCancellationKind)>,
}
impl Workspace {
    fn new(width: usize) -> Result<Self, HostedError> {
        int8_team_workspace_bytes(width)?;
        let mut children = Vec::new();
        children.try_reserve_exact(width - 1).map_err(|_| HostedError::Native(StrictInt8Error::Allocation))?;
        for _ in 1..width {
            children.push(WorkerBuffers::new(NANBEIGE_VOCAB_SIZE.div_ceil(width))
                .map_err(HostedError::Native)?);
        }
        Ok(Self { children, #[cfg(test)] panic_worker: None, #[cfg(test)] cancel_on_exit: None })
    }
}

fn zeroed<T: Default + Clone>(length: usize) -> Result<Vec<T>, StrictInt8Error> {
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(|_| StrictInt8Error::Allocation)?;
    values.resize(length, T::default());
    Ok(values)
}

/// Worker storage is charged before the blocking crossing and remains owned
/// until all fixed children physically join, including cancellation and panic.
#[allow(clippy::too_many_arguments)]
pub(in crate::hosted) fn run_int8_team<T, A, F>(engine: &NlpEngine, limits: RunLimits,
    cancellation: CancellationToken, model: ResidentInt8, width: usize, kernel: Int8Kernel, admission: A, work: F)
    -> Result<T, HostedError>
where T: Send + 'static, A: Send + 'static,
    F: FnOnce(&CurrentCandidateInt8Model, &mut A, &mut RunControl, &mut dyn Int8ProjectionExecutor)
        -> Result<T, HostedError> + Send + 'static {
    preflight_int8_team(engine, limits, width)?;
    if !Arc::ptr_eq(engine.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
    let shared = RunShared::new(limits, cancellation)?;
    let lease = engine.resources().acquire_lease();
    let pending = Pending::reserve(&lease, MemoryClass::ActivationScratch, int8_team_workspace_bytes(width)?)?;
    let workspace = allocate(pending, || Workspace::new(width))?;
    run_blocking(engine, limits, move |blocking_cx| {
        let mut workspace = workspace;
        // Native KV/scratch and output claims are owned outside the scoped
        // work callback. Its early return or unwind cannot release them while
        // any child is still live; scope drain always precedes this owner.
        let mut admission = admission;
        let native_model = &model.inner.loaded.value;
        let weights = native_model.weight_view().map_err(HostedError::Native)?.with_kernel(kernel);
        let physical = run_scope(&blocking_cx, &weights, &mut workspace.value, width, kernel,
            Arc::clone(&shared), |control, team| work(native_model, &mut admission, control, team));
        drop(workspace);
        drop(admission);
        drop(model);
        drop(lease);
        physical
    })
}

fn run_scope<T, F>(blocking_cx: &Cx, weights: &Int8WeightView<'_>, workspace: &mut Workspace,
    width: usize, kernel: Int8Kernel, shared: Arc<RunShared>, work: F) -> Result<T, HostedError>
where F: FnOnce(&mut RunControl, &mut dyn Int8ProjectionExecutor) -> Result<T, HostedError> {
    let leaf = blocking_cx.restrict::<cap::None>();
    let _ambient = leaf.clone().set_current_restricted();
    let team = Arc::new(SealedCpuTeam::new(width - 1));
    let gate = Arc::new(EntryGate::new());
    let mut outcome = None;
    let mut formation_complete = false;
    #[cfg(test)] let panic_worker = workspace.panic_worker;
    #[cfg(test)] let cancel_on_exit = workspace.cancel_on_exit.clone();
    let scope_result = leaf.scoped_cpu(width - 1, |scope| {
        let mut executor = TeamExecutor::new(weights, kernel, width,
            Arc::clone(&team), Arc::clone(&gate), Arc::clone(&shared))?;
        let mut control = RunControl { cx: leaf.clone(), shared: Arc::clone(&shared) };
        // This is the only formation loop. Projection commands cannot spawn.
        for buffers in std::mem::take(&mut workspace.children) {
            if let Some(cause) = control.checkpoint(0) {
                return Err(HostedError::Native(StrictInt8Error::Cancelled(cause)));
            }
            let worker = team.form_worker().map_err(|_| HostedError::Native(StrictInt8Error::ProjectionExecutor))?;
            let (command_sender, command_receiver) = capacity_one_lane();
            let (reply_sender, reply_receiver) = capacity_one_lane();
            let worker_team = Arc::clone(&team);
            let worker_gate = Arc::clone(&gate);
            let worker_shared = Arc::clone(&shared);
            let worker_leaf = leaf.clone();
            #[cfg(test)] let cancel_on_exit = cancel_on_exit.clone();
            executor.peers.push(Peer { command: Some(command_sender), reply: reply_receiver,
                buffers: Some(buffers) });
            scope.spawn(move |cpu| {
                if !worker_gate.wait() { return; }
                if worker_team.worker_started(worker).is_err() { return; }
                let _exit = WorkerExit { team: worker_team, worker };
                let _entry = match NativeEntry::enter() {
                    Ok(entry) => entry,
                    Err(_) => {
                        let _ = reply_sender.try_send(Reply::Ready(Err(StrictInt8Error::ProjectionExecutor)));
                        return;
                    }
                };
                let _ambient = worker_leaf.clone().set_current_restricted();
                let mut control = WorkerControl { cpu, cx: worker_leaf, shared: worker_shared,
                    #[cfg(test)] panic_on_command: panic_worker == Some(worker.ordinal()) };
                let ready = control.checkpoint(0).map_or(Ok(()), |cause| Err(StrictInt8Error::Cancelled(cause)));
                let stopped = ready.is_err();
                if reply_sender.try_send(Reply::Ready(ready)).is_err() || stopped { return; }
                worker_loop(weights, kernel, command_receiver, reply_sender, &mut control);
                #[cfg(test)]
                if let Some((token, cause)) = cancel_on_exit { token.cancel(cause); }
            }).map_err(|source| HostedError::Scope { source, physical_error: None })?;
        }
        team.seal().map_err(|_| HostedError::Native(StrictInt8Error::ProjectionExecutor))?;
        team.release_workers().map_err(|_| HostedError::Native(StrictInt8Error::ProjectionExecutor))?;
        gate.release();
        // Even zero-work requests acknowledge every child's physical start.
        for peer in &executor.peers {
            match peer.reply.recv() {
                Ok(Reply::Ready(Ok(()))) => {}
                Ok(Reply::Ready(Err(error))) => return Err(HostedError::Native(error)),
                _ => return Err(HostedError::Native(StrictInt8Error::ProjectionExecutor)),
            }
        }
        formation_complete = true;
        outcome = if control.checkpoint(0).is_some() { drop(work); None }
            else { Some(work(&mut control, &mut executor)) };
        let stopped = control.checkpoint(0).is_some();
        executor.complete = matches!(&outcome, Some(Ok(_))) && !stopped;
        Ok(())
        // Executor drop disconnects lanes before the scope joins its workers.
    });
    match scope_result {
        Ok(Ok(())) => {
            if formation_complete && team.join().is_err() {
                drop(outcome); Err(HostedError::Native(StrictInt8Error::ProjectionExecutor))
            } else {
                // A stop can arrive while the physical children drain. The
                // terminal observation spends the original allowance after
                // join, before any successful result reaches the caller.
                let mut control = RunControl { cx: leaf, shared: Arc::clone(&shared) };
                control.checkpoint(0);
                shared.finish(outcome)
            }
        }
        Ok(Err(error)) => shared.finish(Some(Err(error))),
        Err(source) => Err(HostedError::Scope { source,
            physical_error: shared.finish(outcome).err().map(Box::new) }),
    }
}

fn worker_loop(weights: &Int8WeightView<'_>, kernel: Int8Kernel,
    commands: CapacityOneReceiver<Command>, replies: CapacityOneSender<Reply>, control: &mut WorkerControl<'_>) {
    while let Ok(command) = commands.recv() {
        #[cfg(test)]
        if control.panic_on_command { panic!("injected INT8 projection worker panic"); }
        let reply = projection::execute_command(weights, kernel, command, control);
        let stopped = reply.result.is_err();
        if stopped { control.shared.stop(DecodeCancellationKind::FailFast); }
        if replies.try_send(Reply::Projected(reply)).is_err() || stopped { return; }
    }
}

#[cfg(test)] mod tests;
