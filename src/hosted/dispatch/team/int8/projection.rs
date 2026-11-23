//! Fixed capacity-one projection commands with deterministic ordinal assembly.
use super::*;
use std::ops::Range;
use crate::native_engine::{artifact_bridge::ArtifactIdentity,
    portable_int8::{ActivationBuffer, LinearRows}, simd::Int8KernelKind};

pub(super) struct WorkerBuffers {
    input: Vec<i8>, selected: Vec<u32>, floats: Vec<f32>, bf16: Vec<Bf16>,
    #[cfg(test)] drop_probe: Option<Arc<std::sync::atomic::AtomicUsize>>,
}
impl WorkerBuffers {
    pub(super) fn new(rows: usize) -> Result<Self, StrictInt8Error> {
        Ok(Self { input: zeroed(MAX_MODEL_K)?, selected: zeroed(rows)?, floats: zeroed(rows)?, bf16: zeroed(rows)?,
            #[cfg(test)] drop_probe: None })
    }
    #[cfg(test)]
    pub(super) fn payload_bytes(&self) -> usize {
        self.input.len() + self.selected.len() * std::mem::size_of::<u32>()
            + self.floats.len() * std::mem::size_of::<f32>() + self.bf16.len() * std::mem::size_of::<Bf16>()
    }
    #[cfg(all(test, feature = "asupersync-census"))]
    pub(super) fn observe_drop(&mut self, probe: Arc<std::sync::atomic::AtomicUsize>) { self.drop_probe = Some(probe); }
}
#[cfg(test)]
impl Drop for WorkerBuffers {
    fn drop(&mut self) {
        if let Some(probe) = &self.drop_probe { probe.fetch_add(1, std::sync::atomic::Ordering::SeqCst); }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputKind { Float, Bf16 }
pub(super) struct Command {
    epoch: u64, operation: Int8ProjectionOp, kind: OutputKind, range: Range<usize>,
    selected: bool, input_len: usize, input_scale: f32, buffers: WorkerBuffers,
}
pub(super) struct ProjectionReply {
    epoch: u64, operation: Int8ProjectionOp, kind: OutputKind, range: Range<usize>,
    buffers: WorkerBuffers, pub(super) result: Result<(), StrictInt8Error>,
}
pub(super) enum Reply { Ready(Result<(), StrictInt8Error>), Projected(ProjectionReply) }
pub(super) struct Peer {
    pub(super) command: Option<CapacityOneSender<Command>>,
    pub(super) reply: CapacityOneReceiver<Reply>,
    pub(super) buffers: Option<WorkerBuffers>,
}

/// Contiguous output positions retain the exact selected-row order. A row's
/// integer reduction and floating epilogue always run on a single participant.
pub(super) fn row_range(rows: usize, width: usize, ordinal: usize) -> Range<usize> {
    let base = rows / width;
    let remainder = rows % width;
    let start = ordinal * base + ordinal.min(remainder);
    start..start + base + usize::from(ordinal < remainder)
}

pub(super) fn execute_command(weights: &Int8WeightView<'_>, kernel: Int8Kernel,
    mut command: Command, control: &mut dyn DecodeStepControl) -> ProjectionReply {
    let result = (|| {
        if let Some(cause) = control.checkpoint(0) { return Err(StrictInt8Error::Cancelled(cause)); }
        let linear = weights.projection(command.operation)?.with_kernel(kernel);
        let length = command.range.end.checked_sub(command.range.start)
            .ok_or(StrictInt8Error::ProjectionExecutor)?;
        if command.input_len > command.buffers.input.len() || command.input_len != linear.columns()
            || length > command.buffers.floats.len() || length > command.buffers.bf16.len()
            || length > command.buffers.selected.len() {
            return Err(StrictInt8Error::ProjectionExecutor);
        }
        // Narrow selections can leave children idle; they acknowledge the
        // command without invoking a leaf that requires a nonempty selection.
        if length == 0 { return Ok(()); }
        let (rows, range) = if command.selected {
            (LinearRows::Selected(&command.buffers.selected[..length]), 0..length)
        } else { (LinearRows::All, command.range.clone()) };
        let input = &command.buffers.input[..command.input_len];
        match command.kind {
            OutputKind::Float => linear.project_f32_range_into(input, command.input_scale, rows,
                range, &mut command.buffers.floats[..length], control),
            OutputKind::Bf16 => linear.project_bf16_range_into(input, command.input_scale, rows,
                range, &mut command.buffers.bf16[..length], control),
        }.map_err(Into::into)
    })();
    ProjectionReply { epoch: command.epoch, operation: command.operation, kind: command.kind,
        range: command.range, buffers: command.buffers, result }
}

enum Output<'a> { Float(&'a mut [f32]), Bf16(&'a mut [Bf16]) }
impl Output<'_> {
    fn kind(&self) -> OutputKind { match self { Self::Float(_) => OutputKind::Float, Self::Bf16(_) => OutputKind::Bf16 } }
    fn len(&self) -> usize { match self { Self::Float(values) => values.len(), Self::Bf16(values) => values.len() } }
    fn assemble(&mut self, range: Range<usize>, buffers: &WorkerBuffers) {
        let length = range.len();
        match self {
            Self::Float(values) => values[range].copy_from_slice(&buffers.floats[..length]),
            Self::Bf16(values) => values[range].copy_from_slice(&buffers.bf16[..length]),
        }
    }
}

pub(super) struct TeamExecutor<'view, 'weights> {
    weights: &'view Int8WeightView<'weights>, kernel: Int8Kernel, width: usize,
    pub(super) peers: Vec<Peer>, stats: Int8ProjectionStats,
    team: Arc<SealedCpuTeam>, gate: Arc<EntryGate>, shared: Arc<RunShared>,
    poisoned: bool, pub(super) complete: bool,
}
impl<'view, 'weights> TeamExecutor<'view, 'weights> {
    pub(super) fn new(weights: &'view Int8WeightView<'weights>, kernel: Int8Kernel, width: usize,
        team: Arc<SealedCpuTeam>, gate: Arc<EntryGate>, shared: Arc<RunShared>) -> Result<Self, HostedError> {
        let mut peers = Vec::new();
        peers.try_reserve_exact(width - 1).map_err(|_| HostedError::Native(StrictInt8Error::Allocation))?;
        Ok(Self { weights, kernel, width, peers,
            stats: Int8ProjectionStats { team_width: width, ..Int8ProjectionStats::default() },
            team, gate, shared, poisoned: false, complete: false })
    }
    fn disconnect(&mut self) {
        self.gate.abort();
        for peer in &mut self.peers { drop(peer.command.take()); }
    }
    fn project(&mut self, operation: Int8ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, output: Output<'_>, control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt8Error> {
        if self.poisoned { return Err(StrictInt8Error::ProjectionExecutor); }
        let mut result = self.project_inner(operation, input, rows, output, control);
        if result.is_err() {
            self.poisoned = true;
            self.shared.stop(DecodeCancellationKind::FailFast);
            self.team.begin_drain(TeamDrainReason::Cancelled);
            self.disconnect();
            // Recover an original leaf error if a sibling only observed its
            // shared stop, and return every in-flight buffer before join.
            for peer in &mut self.peers {
                if peer.buffers.is_some() { continue; }
                if let Ok(Reply::Projected(reply)) = peer.reply.recv() {
                    if matches!(&result, Err(StrictInt8Error::Cancelled(_))) {
                        if let Err(error) = reply.result {
                            if !matches!(error, StrictInt8Error::Cancelled(_)) { result = Err(error); }
                        }
                    }
                    peer.buffers = Some(reply.buffers);
                }
            }
        }
        result
    }
    fn project_inner(&mut self, operation: Int8ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, mut output: Output<'_>, control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt8Error> {
        if let Some(cause) = control.checkpoint(0) { return Err(StrictInt8Error::Cancelled(cause)); }
        let linear = self.weights.projection(operation)?.with_kernel(self.kernel);
        let values = input.values()?;
        let scale = input.scale()?;
        let count = rows.checked_count(linear.rows())?;
        if values.len() != linear.columns() || output.len() != count || self.peers.len() != self.width - 1 {
            return Err(StrictInt8Error::ProjectionExecutor);
        }
        let epoch = self.stats.completed_projections.checked_add(1).ok_or(StrictInt8Error::Work)?;
        let coordinator = row_range(count, self.width, 0);
        let completed_rows = self.stats.completed_rows.checked_add(count as u64).ok_or(StrictInt8Error::Work)?;
        let completed_worker_rows = self.stats.completed_worker_rows
            .checked_add((count - coordinator.len()) as u64).ok_or(StrictInt8Error::Work)?;
        let kind = output.kind();
        for (ordinal, peer) in self.peers.iter_mut().enumerate() {
            let range = row_range(count, self.width, ordinal + 1);
            let mut buffers = peer.buffers.take().ok_or(StrictInt8Error::ProjectionExecutor)?;
            if values.len() > buffers.input.len() || range.len() > buffers.selected.len()
                || range.len() > buffers.floats.len() || range.len() > buffers.bf16.len() {
                return Err(StrictInt8Error::ProjectionExecutor);
            }
            buffers.input[..values.len()].copy_from_slice(values);
            let selected = match rows {
                LinearRows::All => false,
                LinearRows::Selected(ids) => {
                    buffers.selected[..range.len()].copy_from_slice(&ids[range.clone()]); true
                }
            };
            let command = Command { epoch, operation, kind, range, selected, input_len: values.len(),
                input_scale: scale, buffers };
            peer.command.as_ref().ok_or(StrictInt8Error::ProjectionExecutor)?
                .try_send(command).map_err(|_| StrictInt8Error::ProjectionExecutor)?;
        }
        // The coordinator computes its own share; no row reduction is split.
        match &mut output {
            Output::Float(output) => linear.project_f32_range_into(values, scale, rows,
                coordinator.clone(), &mut output[coordinator], control),
            Output::Bf16(output) => linear.project_bf16_range_into(values, scale, rows,
                coordinator.clone(), &mut output[coordinator], control),
        }?;
        for (ordinal, peer) in self.peers.iter_mut().enumerate() {
            let reply = match peer.reply.recv().map_err(|_| StrictInt8Error::ProjectionExecutor)? {
                Reply::Projected(reply) => reply,
                Reply::Ready(_) => return Err(StrictInt8Error::ProjectionExecutor),
            };
            let expected = row_range(count, self.width, ordinal + 1);
            let valid = reply.epoch == epoch && reply.operation == operation && reply.kind == kind && reply.range == expected;
            let result = reply.result;
            peer.buffers = Some(reply.buffers);
            if !valid { return Err(StrictInt8Error::ProjectionExecutor); }
            result?;
            output.assemble(expected, peer.buffers.as_ref().ok_or(StrictInt8Error::ProjectionExecutor)?);
        }
        if let Some(cause) = control.checkpoint(0) { return Err(StrictInt8Error::Cancelled(cause)); }
        self.stats.completed_projections = epoch;
        self.stats.completed_rows = completed_rows;
        self.stats.completed_worker_rows = completed_worker_rows;
        Ok(())
    }
}
impl Drop for TeamExecutor<'_, '_> {
    fn drop(&mut self) {
        let reason = if std::thread::panicking() { TeamDrainReason::CoordinatorPanicked }
            else if self.complete { TeamDrainReason::Completed } else { TeamDrainReason::Cancelled };
        if !self.complete { self.shared.stop(DecodeCancellationKind::FailFast); }
        self.team.begin_drain(reason);
        self.disconnect();
    }
}
impl Int8ProjectionExecutor for TeamExecutor<'_, '_> {
    fn artifact_identity(&self) -> &ArtifactIdentity { self.weights.artifact_identity() }
    fn kernel_kind(&self) -> Int8KernelKind { self.kernel.kind() }
    fn stats(&self) -> Int8ProjectionStats { self.stats }
    fn project_bf16_into(&mut self, operation: Int8ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, output: &mut [Bf16], control: &mut dyn DecodeStepControl) -> Result<(), StrictInt8Error> {
        self.project(operation, input, rows, Output::Bf16(output), control)
    }
    fn project_f32_into(&mut self, operation: Int8ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, output: &mut [f32], control: &mut dyn DecodeStepControl) -> Result<(), StrictInt8Error> {
        self.project(operation, input, rows, Output::Float(output), control)
    }
}
