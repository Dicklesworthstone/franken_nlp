//! Fixed capacity-one projection commands with deterministic ordinal assembly.
use super::*;
use std::ops::Range;
use crate::native_engine::{artifact_bridge::ArtifactIdentity,
    layer::NANBEIGE_INTERMEDIATE_SIZE,
    portable_int8::{ActivationBuffer, LinearRows, batch::MAX_BATCH_ROWS}, simd::Int8KernelKind};

pub(super) struct WorkerBuffers {
    input: Vec<i8>, selected: Vec<u32>, floats: Vec<f32>, bf16: Vec<Bf16>,
    #[cfg(test)] drop_probe: Option<Arc<std::sync::atomic::AtomicUsize>>,
}
impl WorkerBuffers {
    pub(super) fn new(width: usize, max_batch_rows: usize) -> Result<Self, StrictInt8Error> {
        native_projection::int8_team_workspace_bytes_with_prefill(width, max_batch_rows)?;
        let head_rows = NANBEIGE_VOCAB_SIZE.div_ceil(width);
        let batch_cells = NANBEIGE_INTERMEDIATE_SIZE.div_ceil(width).checked_mul(max_batch_rows)
            .ok_or(StrictInt8Error::Memory)?;
        let inputs = MAX_MODEL_K.checked_mul(max_batch_rows).ok_or(StrictInt8Error::Memory)?;
        Ok(Self { input: zeroed(inputs)?, selected: zeroed(head_rows)?, floats: zeroed(head_rows)?,
            bf16: zeroed(head_rows.max(batch_cells))?, #[cfg(test)] drop_probe: None })
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
    selected: bool, input_len: usize, batch_rows: usize,
    scales: [f32; MAX_BATCH_ROWS], buffers: WorkerBuffers,
}
pub(super) struct ProjectionReply {
    epoch: u64, operation: Int8ProjectionOp, kind: OutputKind, range: Range<usize>,
    batch_rows: usize,
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
        let output_length = length.checked_mul(command.batch_rows).ok_or(StrictInt8Error::ProjectionExecutor)?;
        if command.batch_rows == 0 || command.batch_rows > MAX_BATCH_ROWS
            || (command.batch_rows > 1 && (command.operation == Int8ProjectionOp::Head || command.kind == OutputKind::Float))
            || command.input_len.checked_mul(command.batch_rows).is_none_or(|length| length > command.buffers.input.len())
            || command.input_len != linear.columns()
            || output_length > command.buffers.bf16.len()
            || (command.kind == OutputKind::Float && length > command.buffers.floats.len())
            || length > command.buffers.selected.len() {
            return Err(StrictInt8Error::ProjectionExecutor);
        }
        // Narrow selections can leave children idle; they acknowledge the
        // command without invoking a leaf that requires a nonempty selection.
        if length == 0 { return Ok(()); }
        let mut inputs: [&[i8]; MAX_BATCH_ROWS] = [&[]; MAX_BATCH_ROWS];
        for (index, input) in inputs[..command.batch_rows].iter_mut().enumerate() {
            *input = &command.buffers.input[index * command.input_len..(index + 1) * command.input_len];
        }
        let (rows, range) = if command.selected {
            (LinearRows::Selected(&command.buffers.selected[..length]), 0..length)
        } else { (LinearRows::All, command.range.clone()) };
        match (command.kind, command.batch_rows) {
            (OutputKind::Float, 1) => linear.project_f32_range_into(inputs[0], command.scales[0], rows,
                range, &mut command.buffers.floats[..length], control),
            (OutputKind::Bf16, 1) => linear.project_bf16_range_into(inputs[0], command.scales[0], rows,
                range, &mut command.buffers.bf16[..length], control),
            (OutputKind::Bf16, batch) => linear.project_batch_bf16_range_into(&inputs[..batch],
                &command.scales[..batch], rows, range, &mut command.buffers.bf16[..output_length],
                length, control),
            _ => return Err(StrictInt8Error::ProjectionExecutor),
        }.map_err(Into::into)
    })();
    ProjectionReply { epoch: command.epoch, operation: command.operation, kind: command.kind,
        range: command.range, batch_rows: command.batch_rows, buffers: command.buffers, result }
}

enum Output<'a> { Float(&'a mut [f32]), Bf16(&'a mut [Bf16]) }
impl Output<'_> {
    fn kind(&self) -> OutputKind { match self { Self::Float(_) => OutputKind::Float, Self::Bf16(_) => OutputKind::Bf16 } }
    fn len(&self) -> usize { match self { Self::Float(values) => values.len(), Self::Bf16(values) => values.len() } }
    fn assemble(&mut self, range: Range<usize>, count: usize, batch_rows: usize, buffers: &WorkerBuffers) {
        let length = range.len();
        for lane in 0..batch_rows {
            let target = lane * count + range.start..lane * count + range.end;
            let source = lane * length..(lane + 1) * length;
            match self {
                Self::Float(values) => values[target].copy_from_slice(&buffers.floats[source]),
                Self::Bf16(values) => values[target].copy_from_slice(&buffers.bf16[source]),
            }
        }
    }
}

pub(super) struct TeamExecutor<'view, 'weights> {
    weights: &'view Int8WeightView<'weights>, kernel: Int8Kernel, width: usize, max_batch_rows: usize,
    pub(super) peers: Vec<Peer>, stats: Int8ProjectionStats,
    team: Arc<SealedCpuTeam>, gate: Arc<EntryGate>, shared: Arc<RunShared>,
    poisoned: bool, pub(super) complete: bool,
}
impl<'view, 'weights> TeamExecutor<'view, 'weights> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(weights: &'view Int8WeightView<'weights>, kernel: Int8Kernel, width: usize,
        max_batch_rows: usize, team: Arc<SealedCpuTeam>, gate: Arc<EntryGate>, shared: Arc<RunShared>)
        -> Result<Self, HostedError> {
        int8_prefill_team_workspace_bytes(width, max_batch_rows)?;
        let mut peers = Vec::new();
        peers.try_reserve_exact(width - 1).map_err(|_| HostedError::Native(StrictInt8Error::Allocation))?;
        Ok(Self { weights, kernel, width, max_batch_rows, peers,
            stats: Int8ProjectionStats { team_width: width, ..Int8ProjectionStats::default() },
            team, gate, shared, poisoned: false, complete: false })
    }
    fn disconnect(&mut self) {
        self.gate.abort();
        for peer in &mut self.peers { drop(peer.command.take()); }
    }
    fn project(&mut self, operation: Int8ProjectionOp, inputs: &[ActivationBuffer],
        rows: LinearRows<'_>, output: Output<'_>, control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt8Error> {
        if self.poisoned { return Err(StrictInt8Error::ProjectionExecutor); }
        let mut result = self.project_inner(operation, inputs, rows, output, control);
        if let Err(error) = &result {
            self.poisoned = true;
            self.shared.stop(match error {
                StrictInt8Error::Cancelled(cause) => *cause,
                _ => DecodeCancellationKind::FailFast,
            });
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
    fn project_inner(&mut self, operation: Int8ProjectionOp, inputs: &[ActivationBuffer],
        rows: LinearRows<'_>, mut output: Output<'_>, control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt8Error> {
        if let Some(cause) = control.checkpoint(0) { return Err(StrictInt8Error::Cancelled(cause)); }
        let linear = self.weights.projection(operation)?.with_kernel(self.kernel);
        let batch_rows = inputs.len();
        let kind = output.kind();
        if batch_rows == 0 || batch_rows > self.max_batch_rows || batch_rows > MAX_BATCH_ROWS
            || (batch_rows > 1 && (operation == Int8ProjectionOp::Head || kind == OutputKind::Float)) {
            return Err(StrictInt8Error::ProjectionExecutor);
        }
        let mut values: [&[i8]; MAX_BATCH_ROWS] = [&[]; MAX_BATCH_ROWS];
        let mut scales = [1.0_f32; MAX_BATCH_ROWS];
        for (index, input) in inputs.iter().enumerate() {
            values[index] = input.values()?; scales[index] = input.scale()?;
            if values[index].len() != linear.columns() { return Err(StrictInt8Error::ProjectionExecutor); }
        }
        let input_len = linear.columns();
        let input_cells = input_len.checked_mul(batch_rows).ok_or(StrictInt8Error::ProjectionExecutor)?;
        let count = rows.checked_count(linear.rows())?;
        let output_cells = count.checked_mul(batch_rows).ok_or(StrictInt8Error::ProjectionExecutor)?;
        if output.len() != output_cells || self.peers.len() != self.width - 1 {
            return Err(StrictInt8Error::ProjectionExecutor);
        }
        let epoch = self.stats.completed_projections.checked_add(1).ok_or(StrictInt8Error::Work)?;
        let coordinator = row_range(count, self.width, 0);
        let completed_rows = self.stats.completed_rows.checked_add(output_cells as u64).ok_or(StrictInt8Error::Work)?;
        let completed_worker_rows = self.stats.completed_worker_rows
            .checked_add(((count - coordinator.len()) * batch_rows) as u64).ok_or(StrictInt8Error::Work)?;
        for (ordinal, peer) in self.peers.iter_mut().enumerate() {
            let range = row_range(count, self.width, ordinal + 1);
            let mut buffers = peer.buffers.take().ok_or(StrictInt8Error::ProjectionExecutor)?;
            if input_cells > buffers.input.len() || range.len() > buffers.selected.len()
                || range.len() * batch_rows > buffers.bf16.len()
                || (kind == OutputKind::Float && range.len() > buffers.floats.len()) {
                return Err(StrictInt8Error::ProjectionExecutor);
            }
            for (lane, input) in values[..batch_rows].iter().enumerate() {
                buffers.input[lane * input_len..(lane + 1) * input_len].copy_from_slice(input);
            }
            let selected = match rows {
                LinearRows::All => false,
                LinearRows::Selected(ids) => {
                    buffers.selected[..range.len()].copy_from_slice(&ids[range.clone()]); true
                }
            };
            let command = Command { epoch, operation, kind, range, selected, input_len,
                batch_rows, scales, buffers };
            peer.command.as_ref().ok_or(StrictInt8Error::ProjectionExecutor)?
                .try_send(command).map_err(|_| StrictInt8Error::ProjectionExecutor)?;
        }
        // The coordinator writes only its column stripe into the full
        // activation-major result, without a second batch-sized scratch copy.
        let span = (batch_rows - 1) * count + coordinator.len();
        match (&mut output, batch_rows) {
            (Output::Float(output), 1) => linear.project_f32_range_into(values[0], scales[0], rows,
                coordinator.clone(), &mut output[coordinator.clone()], control),
            (Output::Bf16(output), 1) => linear.project_bf16_range_into(values[0], scales[0], rows,
                coordinator.clone(), &mut output[coordinator.clone()], control),
            (Output::Bf16(output), batch) => linear.project_batch_bf16_range_into(&values[..batch],
                &scales[..batch], rows, coordinator.clone(), &mut output[..span], count, control),
            _ => return Err(StrictInt8Error::ProjectionExecutor),
        }?;
        for (ordinal, peer) in self.peers.iter_mut().enumerate() {
            let reply = match peer.reply.recv().map_err(|_| StrictInt8Error::ProjectionExecutor)? {
                Reply::Projected(reply) => reply,
                Reply::Ready(_) => return Err(StrictInt8Error::ProjectionExecutor),
            };
            let expected = row_range(count, self.width, ordinal + 1);
            let valid = reply.epoch == epoch && reply.operation == operation && reply.kind == kind
                && reply.range == expected && reply.batch_rows == batch_rows;
            let result = reply.result;
            peer.buffers = Some(reply.buffers);
            if !valid { return Err(StrictInt8Error::ProjectionExecutor); }
            result?;
            output.assemble(expected, count, batch_rows, peer.buffers.as_ref().ok_or(StrictInt8Error::ProjectionExecutor)?);
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
    fn max_batch_rows(&self) -> usize { self.max_batch_rows }
    fn stats(&self) -> Int8ProjectionStats { self.stats }
    fn project_bf16_into(&mut self, operation: Int8ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, output: &mut [Bf16], control: &mut dyn DecodeStepControl) -> Result<(), StrictInt8Error> {
        self.project(operation, std::slice::from_ref(input), rows, Output::Bf16(output), control)
    }
    fn project_f32_into(&mut self, operation: Int8ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, output: &mut [f32], control: &mut dyn DecodeStepControl) -> Result<(), StrictInt8Error> {
        self.project(operation, std::slice::from_ref(input), rows, Output::Float(output), control)
    }
    fn project_batch_bf16_into(&mut self, operation: Int8ProjectionOp, inputs: &[ActivationBuffer],
        rows: LinearRows<'_>, output: &mut [Bf16], control: &mut dyn DecodeStepControl) -> Result<(), StrictInt8Error> {
        self.project(operation, inputs, rows, Output::Bf16(output), control)
    }
}

#[cfg(test)]
mod tests;
