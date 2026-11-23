//! Internal output-row scheduling for the exact single-sequence INT8 engine.
//! The native session owns its sole work ledger. A persistent executor may
//! partition independent rows, but never changes quantization or arithmetic.
use super::{ActivationBuffer, ArtifactIdentity, Bf16, DecodeStepControl,
    LinearRows, ProjectionLedger, QuantizedLinear, StrictInt8Error};
#[cfg(any(feature = "asupersync-runtime", test))]
use super::Int8WeightView;
use crate::native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, quant_algebra::MAX_MODEL_K,
    simd::Int8KernelKind};
#[cfg(any(feature = "asupersync-runtime", test))]
use serde::{Deserialize, Serialize};

pub(crate) const MAX_INT8_TEAM_WIDTH: usize = 64;

/// Additional fixed worker payload and conservative lane/formation bookkeeping.
/// Ordinary engine scratch and OS thread stacks remain separately admitted;
/// this quote is not a measured process RSS bound.
pub(crate) fn int8_team_workspace_bytes(width: usize) -> Result<u64, StrictInt8Error> {
    if !(2..=MAX_INT8_TEAM_WIDTH).contains(&width) { return Err(StrictInt8Error::Memory); }
    let rows = NANBEIGE_VOCAB_SIZE.div_ceil(width) as u64;
    let output = (std::mem::size_of::<f32>() + std::mem::size_of::<Bf16>()
        + std::mem::size_of::<u32>()) as u64;
    let child = rows.checked_mul(output)
        .and_then(|bytes| bytes.checked_add(MAX_MODEL_K as u64))
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
/// share the coordinator's original cancellation/checkpoint authority. This
/// initial interface supports one activation row; layer-major prefill refuses
/// an installed executor rather than silently bypassing it.
pub(crate) trait Int8ProjectionExecutor: Send {
    fn artifact_identity(&self) -> &ArtifactIdentity;
    fn kernel_kind(&self) -> Int8KernelKind;
    #[cfg(any(feature = "asupersync-runtime", test))]
    fn stats(&self) -> Int8ProjectionStats;
    fn project_bf16_into(&mut self, op: Int8ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, output: &mut [Bf16], control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt8Error>;
    fn project_f32_into(&mut self, op: Int8ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, output: &mut [f32], control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt8Error>;
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

#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) mod fixture;
