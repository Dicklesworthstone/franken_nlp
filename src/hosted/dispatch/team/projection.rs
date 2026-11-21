//! Reusable capacity-one row commands; no spawning or allocation per operator.
use super::*;
use std::ops::Range;
use crate::native_engine::{portable_int8::{ActivationBuffer, LinearRows},
    simd::Int8KernelKind, strict_int4::Int4ModelIdentity};
use crate::native_engine::portable_int8::int4::batch::MAX_INT4_BATCH_ROWS;

pub(super) struct WorkerBuffers {
    input: Vec<i8>, selected: Vec<u32>, floats: Vec<f32>, bf16: Vec<Bf16>, unpack: Vec<i8>,
    #[cfg(test)] drop_probe: Option<Arc<std::sync::atomic::AtomicUsize>>,
}
impl WorkerBuffers {
    pub(super) fn new(rows: usize, group_width: usize, max_batch_rows: usize) -> Result<Self, StrictInt4Error> {
        if max_batch_rows == 0 || max_batch_rows > MAX_INT4_BATCH_ROWS { return Err(StrictInt4Error::Memory); }
        let output = rows.checked_mul(max_batch_rows).ok_or(StrictInt4Error::Memory)?;
        Ok(Self { input: zeroed(MAX_MODEL_K * max_batch_rows)?, selected: zeroed(rows)?,
            floats: zeroed(output)?, bf16: zeroed(output)?, unpack: zeroed(group_width)?,
            #[cfg(test)] drop_probe: None })
    }
    #[cfg(test)]
    pub(super) fn payload_bytes(&self) -> usize {
        self.input.len() + self.unpack.len() + self.selected.len() * std::mem::size_of::<u32>()
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
    epoch: u64, operation: Int4ProjectionOp, kind: OutputKind, range: Range<usize>,
    selected: bool, input_len: usize, batch_rows: usize,
    scales: [f32; MAX_INT4_BATCH_ROWS], buffers: WorkerBuffers,
}
pub(super) struct ProjectionReply {
    epoch: u64, operation: Int4ProjectionOp, kind: OutputKind, range: Range<usize>,
    batch_rows: usize,
    buffers: WorkerBuffers, pub(super) result: Result<(), StrictInt4Error>,
}
pub(super) enum Reply { Ready(Result<(), StrictInt4Error>), Projected(ProjectionReply) }
pub(super) struct Peer {
    pub(super) command: Option<CapacityOneSender<Command>>,
    pub(super) reply: CapacityOneReceiver<Reply>,
    pub(super) buffers: Option<WorkerBuffers>,
}

/// Balanced contiguous OUTPUT positions. The selected IDs themselves remain
/// sorted in the planner's exact order; no row's floating reduction is split.
fn row_range(rows: usize, width: usize, ordinal: usize) -> Range<usize> {
    let base = rows / width;
    let remainder = rows % width;
    let start = ordinal * base + ordinal.min(remainder);
    start..start + base + usize::from(ordinal < remainder)
}

pub(super) fn execute_command(weights: &Int4ModelWeights, kernel: PackedInt4Kernel,
    mut command: Command, control: &mut dyn DecodeStepControl) -> ProjectionReply {
    let result = (|| {
        if let Some(cause) = control.checkpoint(0) { return Err(StrictInt4Error::Cancelled(cause)); }
        let linear = weights.projection(command.operation)?.with_kernel(kernel);
        let length = command.range.end.checked_sub(command.range.start)
            .ok_or(StrictInt4Error::ProjectionExecutor)?;
        let output_length = length.checked_mul(command.batch_rows).ok_or(StrictInt4Error::ProjectionExecutor)?;
        if command.batch_rows == 0 || command.batch_rows > MAX_INT4_BATCH_ROWS
            || command.input_len.checked_mul(command.batch_rows).is_none_or(|length| length > command.buffers.input.len())
            || command.input_len != linear.columns()
            || output_length > command.buffers.floats.len() || output_length > command.buffers.bf16.len()
            || length > command.buffers.selected.len() {
            return Err(StrictInt4Error::ProjectionExecutor);
        }
        // A narrow matrix/selection can leave later ordinals idle. Such an
        // empty shard still acknowledges the command without inventing rows.
        if length == 0 { return Ok(()); }
        let mut inputs: [&[i8]; MAX_INT4_BATCH_ROWS] = [&[]; MAX_INT4_BATCH_ROWS];
        for (index, input) in inputs[..command.batch_rows].iter_mut().enumerate() {
            *input = &command.buffers.input[index * command.input_len..(index + 1) * command.input_len];
        }
        let (rows, range) = if command.selected {
            (LinearRows::Selected(&command.buffers.selected[..length]), 0..length)
        } else { (LinearRows::All, command.range.clone()) };
        match (command.kind, command.batch_rows) {
            (OutputKind::Float, 1) => linear.project_f32_range_into(inputs[0], command.scales[0], rows, range,
                &mut command.buffers.floats[..length], &mut command.buffers.unpack, control),
            (OutputKind::Bf16, 1) => linear.project_bf16_range_into(inputs[0], command.scales[0], rows, range,
                &mut command.buffers.bf16[..length], &mut command.buffers.unpack, control),
            (OutputKind::Float, batch) => linear.project_batch_f32_range_into(&inputs[..batch],
                &command.scales[..batch], rows, range, &mut command.buffers.floats[..output_length],
                length, &mut command.buffers.unpack, control),
            (OutputKind::Bf16, batch) => linear.project_batch_bf16_range_into(&inputs[..batch],
                &command.scales[..batch], rows, range, &mut command.buffers.bf16[..output_length],
                length, &mut command.buffers.unpack, control),
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

pub(super) struct TeamExecutor<'a> {
    weights: &'a Int4ModelWeights, kernel: PackedInt4Kernel, width: usize, max_batch_rows: usize,
    scratch: Vec<i8>, pub(super) peers: Vec<Peer>, stats: Int4ProjectionStats,
    team: Arc<SealedCpuTeam>, gate: Arc<EntryGate>, shared: Arc<RunShared>,
    poisoned: bool, pub(super) complete: bool,
}
impl<'a> TeamExecutor<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(weights: &'a Int4ModelWeights, kernel: PackedInt4Kernel, width: usize, max_batch_rows: usize,
        scratch: Vec<i8>, team: Arc<SealedCpuTeam>, gate: Arc<EntryGate>, shared: Arc<RunShared>)
        -> Result<Self, HostedError> {
        let mut peers = Vec::new();
        peers.try_reserve_exact(width - 1).map_err(|_| native_error(StrictInt4Error::Allocation))?;
        Ok(Self { weights, kernel, width, max_batch_rows, scratch, peers,
            stats: Int4ProjectionStats { team_width: width, ..Int4ProjectionStats::default() },
            team, gate, shared, poisoned: false, complete: false })
    }
    fn disconnect(&mut self) {
        self.gate.abort();
        for peer in &mut self.peers { drop(peer.command.take()); }
    }
    fn project(&mut self, operation: Int4ProjectionOp, inputs: &[ActivationBuffer],
        rows: LinearRows<'_>, output: Output<'_>, control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt4Error> {
        if self.poisoned { return Err(StrictInt4Error::ProjectionExecutor); }
        let mut result = self.project_inner(operation, inputs, rows, output, control);
        if result.is_err() {
            self.poisoned = true;
            self.shared.stop(DecodeCancellationKind::FailFast);
            self.team.begin_drain(TeamDrainReason::Cancelled);
            self.disconnect();
            // A worker may have reported a real leaf failure immediately
            // before the coordinator observed shared FailFast. Retain that
            // original error instead of replacing it with a sibling's stop.
            // Disconnection wakes idle children; an in-flight child has one
            // bounded reply or a disconnected lane, never a fixed barrier.
            for peer in &mut self.peers {
                if peer.buffers.is_some() { continue; }
                if let Ok(Reply::Projected(reply)) = peer.reply.recv() {
                    if matches!(&result, Err(StrictInt4Error::Cancelled(_))) {
                        if let Err(error) = reply.result {
                            if !matches!(error, StrictInt4Error::Cancelled(_)) { result = Err(error); }
                        }
                    }
                    peer.buffers = Some(reply.buffers);
                }
            }
        }
        result
    }
    fn project_inner(&mut self, operation: Int4ProjectionOp, inputs: &[ActivationBuffer],
        rows: LinearRows<'_>, mut output: Output<'_>, control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt4Error> {
        if let Some(cause) = control.checkpoint(0) { return Err(StrictInt4Error::Cancelled(cause)); }
        let linear = self.weights.projection(operation)?.with_kernel(self.kernel);
        let batch_rows = inputs.len();
        if batch_rows == 0 || batch_rows > self.max_batch_rows || batch_rows > MAX_INT4_BATCH_ROWS {
            return Err(StrictInt4Error::ProjectionExecutor);
        }
        let mut values: [&[i8]; MAX_INT4_BATCH_ROWS] = [&[]; MAX_INT4_BATCH_ROWS];
        let mut scales = [1.0_f32; MAX_INT4_BATCH_ROWS];
        for (index, input) in inputs.iter().enumerate() {
            values[index] = input.values()?; scales[index] = input.scale()?;
            if values[index].len() != linear.columns() { return Err(StrictInt4Error::ProjectionExecutor); }
        }
        let input_len = linear.columns();
        let count = rows.checked_count(linear.rows())?;
        if count.checked_mul(batch_rows) != Some(output.len())
            || self.peers.len() != self.width - 1 {
            return Err(StrictInt4Error::ProjectionExecutor);
        }
        let epoch = self.stats.completed_projections.checked_add(1).ok_or(StrictInt4Error::Work)?;
        let coordinator = row_range(count, self.width, 0);
        let completed_rows = self.stats.completed_rows.checked_add((count * batch_rows) as u64).ok_or(StrictInt4Error::Work)?;
        let completed_worker_rows = self.stats.completed_worker_rows
            .checked_add(((count - coordinator.len()) * batch_rows) as u64).ok_or(StrictInt4Error::Work)?;
        let kind = output.kind();
        for (ordinal, peer) in self.peers.iter_mut().enumerate() {
            let range = row_range(count, self.width, ordinal + 1);
            let mut buffers = peer.buffers.take().ok_or(StrictInt4Error::ProjectionExecutor)?;
            if input_len * batch_rows > buffers.input.len() || range.len() > buffers.selected.len()
                || range.len() * batch_rows > buffers.floats.len() || range.len() * batch_rows > buffers.bf16.len() {
                return Err(StrictInt4Error::ProjectionExecutor);
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
            let command = Command { epoch, operation, kind, range, selected,
                input_len, batch_rows, scales, buffers };
            peer.command.as_ref().ok_or(StrictInt4Error::ProjectionExecutor)?
                .try_send(command).map_err(|_| StrictInt4Error::ProjectionExecutor)?;
        }
        // The coordinator is a real participant, not a waiting extra thread.
        match (&mut output, batch_rows) {
            (Output::Float(output), 1) => linear.project_f32_range_into(values[0], scales[0], rows,
                coordinator.clone(), &mut output[coordinator.clone()], &mut self.scratch, control),
            (Output::Bf16(output), 1) => linear.project_bf16_range_into(values[0], scales[0], rows,
                coordinator.clone(), &mut output[coordinator.clone()], &mut self.scratch, control),
            (Output::Float(output), batch) => linear.project_batch_f32_range_into(&values[..batch],
                &scales[..batch], rows, coordinator.clone(), output, count, &mut self.scratch, control),
            (Output::Bf16(output), batch) => linear.project_batch_bf16_range_into(&values[..batch],
                &scales[..batch], rows, coordinator.clone(), output, count, &mut self.scratch, control),
        }?;
        // Deterministic ordinal assembly happens only after matching the
        // command identity, range and output rail on that worker's reply.
        for (ordinal, peer) in self.peers.iter_mut().enumerate() {
            let reply = match peer.reply.recv().map_err(|_| StrictInt4Error::ProjectionExecutor)? {
                Reply::Projected(reply) => reply,
                Reply::Ready(_) => return Err(StrictInt4Error::ProjectionExecutor),
            };
            let expected = row_range(count, self.width, ordinal + 1);
            let valid = reply.epoch == epoch && reply.operation == operation && reply.kind == kind
                && reply.range == expected && reply.batch_rows == batch_rows;
            let result = reply.result;
            peer.buffers = Some(reply.buffers);
            if !valid { return Err(StrictInt4Error::ProjectionExecutor); }
            result?;
            output.assemble(expected, count, batch_rows, peer.buffers.as_ref().ok_or(StrictInt4Error::ProjectionExecutor)?);
        }
        if let Some(cause) = control.checkpoint(0) { return Err(StrictInt4Error::Cancelled(cause)); }
        self.stats.completed_projections = epoch;
        self.stats.completed_rows = completed_rows;
        self.stats.completed_worker_rows = completed_worker_rows;
        Ok(())
    }
}
impl Drop for TeamExecutor<'_> {
    fn drop(&mut self) {
        let reason = if std::thread::panicking() { TeamDrainReason::CoordinatorPanicked }
            else if self.complete { TeamDrainReason::Completed } else { TeamDrainReason::Cancelled };
        if !self.complete { self.shared.stop(DecodeCancellationKind::FailFast); }
        self.team.begin_drain(reason);
        // Both gate abort and lane disconnect run on every unwind/error path.
        // No child depends on a fixed-count barrier or a blocking final reply.
        self.disconnect();
    }
}
impl Int4ProjectionExecutor for TeamExecutor<'_> {
    fn model_identity(&self) -> &Int4ModelIdentity { self.weights.identity() }
    fn kernel_kind(&self) -> Int8KernelKind { self.kernel.kind() }
    fn max_batch_rows(&self) -> usize { self.max_batch_rows }
    fn stats(&self) -> Int4ProjectionStats { self.stats }
    fn project_bf16_into(&mut self, operation: Int4ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, output: &mut [Bf16], control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt4Error> {
        self.project(operation, std::slice::from_ref(input), rows, Output::Bf16(output), control)
    }
    fn project_f32_into(&mut self, operation: Int4ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, output: &mut [f32], control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt4Error> {
        self.project(operation, std::slice::from_ref(input), rows, Output::Float(output), control)
    }
    fn project_batch_bf16_into(&mut self, operation: Int4ProjectionOp, inputs: &[ActivationBuffer],
        rows: LinearRows<'_>, output: &mut [Bf16], control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt4Error> {
        self.project(operation, inputs, rows, Output::Bf16(output), control)
    }
}
