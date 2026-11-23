//! Internal output-row scheduling for the exact single-sequence INT8 engine.
//! The native session owns its sole work ledger. A persistent executor may
//! partition independent rows, but never changes quantization or arithmetic.
use super::{ActivationBuffer, ArtifactIdentity, Bf16, DecodeStepControl,
    LinearRows, ProjectionLedger, QuantizedLinear, StrictInt8Error};
#[cfg(any(feature = "asupersync-runtime", test))]
use super::Int8WeightView;
use crate::native_engine::{layer::NANBEIGE_INTERMEDIATE_SIZE, lmhead::NANBEIGE_VOCAB_SIZE,
    portable_int8::batch::MAX_BATCH_ROWS, quant_algebra::MAX_MODEL_K, simd::Int8KernelKind};
#[cfg(any(feature = "asupersync-runtime", test))]
use serde::{Deserialize, Serialize};

pub(crate) const MAX_INT8_TEAM_WIDTH: usize = 64;

/// Additional fixed worker payload and conservative lane/formation bookkeeping.
/// Ordinary engine scratch and OS thread stacks remain separately admitted;
/// this quote is not a measured process RSS bound.
pub(crate) fn int8_team_workspace_bytes(width: usize) -> Result<u64, StrictInt8Error> {
    int8_team_workspace_bytes_with_prefill(width, 1)
}

/// Fixed payload for a persistent team that also admits finite prompt cohorts.
/// Only decoder BF16 projections batch; the final vocabulary head stays one
/// activation row. Child BF16 storage covers the larger of those two shapes.
/// The 4096-byte child allowance also includes its fixed 64 activation scales.
pub(crate) fn int8_team_workspace_bytes_with_prefill(width: usize, rows: usize) -> Result<u64, StrictInt8Error> {
    if !(2..=MAX_INT8_TEAM_WIDTH).contains(&width) || !(1..=MAX_BATCH_ROWS).contains(&rows) {
        return Err(StrictInt8Error::Memory);
    }
    let head = NANBEIGE_VOCAB_SIZE.div_ceil(width) as u64;
    let layer = NANBEIGE_INTERMEDIATE_SIZE.div_ceil(width) as u64;
    let bf16 = layer.checked_mul(rows as u64).ok_or(StrictInt8Error::Memory)?.max(head);
    let child = head.checked_mul((std::mem::size_of::<f32>() + std::mem::size_of::<u32>()) as u64)
        .and_then(|bytes| bytes.checked_add(bf16.checked_mul(std::mem::size_of::<Bf16>() as u64)?))
        .and_then(|bytes| bytes.checked_add((rows as u64).checked_mul(MAX_MODEL_K as u64)?))
        .and_then(|bytes| bytes.checked_add(4096)).ok_or(StrictInt8Error::Memory)?;
    child.checked_mul((width - 1) as u64)
        .and_then(|bytes| bytes.checked_add(65_536)).ok_or(StrictInt8Error::Memory)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Int8ProjectionKind { Query, Key, Value, AttentionOutput, Gate, Up, Down }

/// Both loop passes address the same immutable physical layer matrices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Int8ProjectionOp {
    Head,
    Layer { layer: usize, kind: Int8ProjectionKind },
}

/// Completed physical rows, not promised parallelism or a speedup claim.
/// Failed projections retain their native debit but report no successful rows.
#[cfg(any(feature = "asupersync-runtime", test))]
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Int8ProjectionStats {
    pub team_width: usize,
    pub completed_projections: u64,
    pub completed_rows: u64,
    pub completed_worker_rows: u64,
}

/// Installed only by one hosted, whole-request, sealed CPU team. Its workers
/// share the coordinator's original cancellation/checkpoint authority. A batch
/// capacity of zero means single-row projections only; prefill checks the
/// whole morsel capacity before any work, KV mutation or allocation.
pub(crate) trait Int8ProjectionExecutor: Send {
    fn artifact_identity(&self) -> &ArtifactIdentity;
    fn kernel_kind(&self) -> Int8KernelKind;
    fn max_batch_rows(&self) -> usize { 0 }
    #[cfg(any(feature = "asupersync-runtime", test))]
    fn stats(&self) -> Int8ProjectionStats;
    fn project_bf16_into(&mut self, op: Int8ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, output: &mut [Bf16], control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt8Error>;
    fn project_f32_into(&mut self, op: Int8ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, output: &mut [f32], control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt8Error>;
    /// Optional capability, never a serial fallback for an installed team.
    fn project_batch_bf16_into(&mut self, _op: Int8ProjectionOp, _inputs: &[ActivationBuffer],
        _rows: LinearRows<'_>, _output: &mut [Bf16], _control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt8Error> { Err(StrictInt8Error::ProjectionExecutor) }
}

#[cfg(any(feature = "asupersync-runtime", test))]
impl Int8WeightView<'_> {
    /// Constant-time lookup inside the already validated full-model view. No
    /// source name construction, payload scan or substitute weight can enter
    /// the per-command worker path.
    pub(crate) fn projection(&self, op: Int8ProjectionOp) -> Result<QuantizedLinear<'_>, StrictInt8Error> {
        Ok(match op {
            Int8ProjectionOp::Head => self.head,
            Int8ProjectionOp::Layer { layer, kind } => {
                let layer = self.layers.get(layer).ok_or(StrictInt8Error::ProjectionExecutor)?;
                match kind {
                    Int8ProjectionKind::Query => layer.q,
                    Int8ProjectionKind::Key => layer.k,
                    Int8ProjectionKind::Value => layer.v,
                    Int8ProjectionKind::AttentionOutput => layer.o,
                    Int8ProjectionKind::Gate => layer.gate,
                    Int8ProjectionKind::Up => layer.up,
                    Int8ProjectionKind::Down => layer.down,
                }
            }
        })
    }
}

pub(super) fn check_binding(identity: &ArtifactIdentity, kernel: Int8KernelKind,
    executor: &dyn Int8ProjectionExecutor) -> Result<(), StrictInt8Error> {
    if executor.artifact_identity() != identity || executor.kernel_kind() != kernel {
        return Err(StrictInt8Error::ProjectionExecutor);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn project_bf16<'executor, C: DecodeStepControl>(
    executor: &mut Option<&'executor mut (dyn Int8ProjectionExecutor + 'executor)>,
    op: Int8ProjectionOp, matrix: QuantizedLinear<'_>, input: &ActivationBuffer,
    rows: LinearRows<'_>, output: &mut [Bf16], ledger: &mut ProjectionLedger, control: &mut C)
    -> Result<(), StrictInt8Error> {
    match executor {
        Some(executor) => {
            matrix.reserve_projection(input, rows, output.len(), ledger)?;
            executor.project_bf16_into(op, input, rows, output, control)
        }
        None => matrix.project_bf16_into(input, rows, output, ledger, control).map_err(Into::into),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn project_f32<'executor, C: DecodeStepControl>(
    executor: &mut Option<&'executor mut (dyn Int8ProjectionExecutor + 'executor)>,
    op: Int8ProjectionOp, matrix: QuantizedLinear<'_>, input: &ActivationBuffer,
    rows: LinearRows<'_>, output: &mut [f32], ledger: &mut ProjectionLedger, control: &mut C)
    -> Result<(), StrictInt8Error> {
    match executor {
        Some(executor) => {
            matrix.reserve_projection(input, rows, output.len(), ledger)?;
            executor.project_f32_into(op, input, rows, output, control)
        }
        None => matrix.project_f32_into(input, rows, output, ledger, control).map_err(Into::into),
    }
}

/// One whole-cohort debit remains in the native session, never per child or
/// per activation. The single-row final lm-head uses the original f32 seam.
#[allow(clippy::too_many_arguments)]
pub(super) fn project_batch_bf16<'executor, C: DecodeStepControl>(
    executor: &mut Option<&'executor mut (dyn Int8ProjectionExecutor + 'executor)>,
    op: Int8ProjectionOp, matrix: QuantizedLinear<'_>, inputs: &[ActivationBuffer],
    rows: LinearRows<'_>, output: &mut [Bf16], ledger: &mut ProjectionLedger, control: &mut C)
    -> Result<(), StrictInt8Error> {
    match executor {
        Some(executor) => {
            if inputs.len() > executor.max_batch_rows()
                || (matches!(op, Int8ProjectionOp::Head) && inputs.len() > 1) {
                return Err(StrictInt8Error::ProjectionExecutor);
            }
            matrix.reserve_batch_projection(inputs, rows, output.len(), ledger)?;
            executor.project_batch_bf16_into(op, inputs, rows, output, control)
        }
        None => matrix.project_batch_bf16_into(inputs, rows, output, ledger, control).map_err(Into::into),
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod batch_tests;
#[cfg(test)]
pub(crate) mod fixture;
