//! Internal scheduling seam for the exact single-sequence INT4 projections.
//!
//! The native engine retains the one work ledger and validates model/kernel
//! binding. An executor may partition output rows, but cannot replace the
//! increasing-group floating program, requantize inputs, or spawn per operator.

use super::{ActivationBuffer, Bf16, DecodeStepControl, GroupedInt4Matrix,
    Int4ModelIdentity, Int8KernelKind, LinearRows,
    PackedInt4Kernel, ProjectionLedger, StrictInt4Error};
#[cfg(any(feature = "asupersync-runtime", test))]
use crate::native_engine::portable_int8::int4::GroupedInt4Linear;
#[cfg(any(feature = "asupersync-runtime", test))]
use super::Int4ModelWeights;
use crate::native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, quant_algebra::MAX_MODEL_K};
#[cfg(any(feature = "asupersync-runtime", test))]
use serde::{Deserialize, Serialize};

pub(crate) const MAX_INT4_TEAM_WIDTH: usize = 64;

/// Extra team payload and conservative lane/formation bookkeeping, above the
/// existing native engine requirement. OS thread stacks, allocator slack and
/// process residency remain host reserves; this is not a process RSS bound.
/// Both the no-runtime CLI preflight and hosted allocation use this one quote.
pub(crate) fn int4_team_workspace_bytes(team_width: usize, group_width: usize)
    -> Result<u64, StrictInt4Error> {
    if team_width < 2 || team_width > MAX_INT4_TEAM_WIDTH
        || group_width == 0 || group_width > MAX_MODEL_K { return Err(StrictInt4Error::Memory); }
    let rows = NANBEIGE_VOCAB_SIZE.div_ceil(team_width) as u64;
    // Every child cycles one owned buffer through its capacity-one lanes:
    // prequantized input, group unpacking, selected row IDs, and both output
    // rails. The coordinator writes directly to its private output shard.
    let output_bytes = (std::mem::size_of::<u32>() + std::mem::size_of::<f32>()
        + std::mem::size_of::<Bf16>()) as u64;
    let per_child = rows.checked_mul(output_bytes)
        .and_then(|bytes| bytes.checked_add(MAX_MODEL_K as u64))
        .and_then(|bytes| bytes.checked_add(group_width as u64))
        .and_then(|bytes| bytes.checked_add(4096)).ok_or(StrictInt4Error::Memory)?;
    per_child.checked_mul((team_width - 1) as u64)
        .and_then(|bytes| bytes.checked_add(group_width as u64))
        .and_then(|bytes| bytes.checked_add(65536)).ok_or(StrictInt4Error::Memory)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Int4ProjectionKind {
    Query, Key, Value, AttentionOutput, Gate, Up, Down,
}

/// Physical layer identity is shared by both loop passes. Workers resolve it
/// against the same immutable full model retained by the admitted request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Int4ProjectionOp {
    Head,
    Layer { layer: usize, kind: Int4ProjectionKind },
}

/// Completed work, not promised parallelism or a measured speedup. A failed
/// projection contributes no successful rows; its native work debit remains.
#[cfg(any(feature = "asupersync-runtime", test))]
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Int4ProjectionStats {
    pub team_width: usize,
    pub completed_projections: u64,
    pub completed_rows: u64,
    pub completed_worker_rows: u64,
}

/// The hosted implementation forms one sealed team for the complete request.
/// The engine invokes these methods only after validating and charging the
/// complete projection. Output is private staging until every shard succeeds.
/// The coordinator control and all workers share one request allowance; the
/// executor must never copy a remaining budget into independent worker limits.
pub(crate) trait Int4ProjectionExecutor: Send {
    fn model_identity(&self) -> &Int4ModelIdentity;
    fn kernel_kind(&self) -> Int8KernelKind;
    #[cfg(any(feature = "asupersync-runtime", test))]
    fn stats(&self) -> Int4ProjectionStats;

    fn project_bf16_into(&mut self, op: Int4ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, output: &mut [Bf16], control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt4Error>;
    fn project_f32_into(&mut self, op: Int4ProjectionOp, input: &ActivationBuffer,
        rows: LinearRows<'_>, output: &mut [f32], control: &mut dyn DecodeStepControl)
        -> Result<(), StrictInt4Error>;
}

#[cfg(any(feature = "asupersync-runtime", test))]
impl Int4ModelWeights {
    /// Resolves only an immutable matrix already covered by the complete model
    /// identity. No external weights, reconstructed names or partial model can
    /// enter the worker command path.
    pub(crate) fn projection(&self, op: Int4ProjectionOp) -> Result<GroupedInt4Linear<'_>, StrictInt4Error> {
        let matrix = match op {
            Int4ProjectionOp::Head => &self.head,
            Int4ProjectionOp::Layer { layer, kind } => {
                let layer = self.layers.get(layer).ok_or(StrictInt4Error::ProjectionExecutor)?;
                match kind {
                    Int4ProjectionKind::Query => &layer.q,
                    Int4ProjectionKind::Key => &layer.k,
                    Int4ProjectionKind::Value => &layer.v,
                    Int4ProjectionKind::AttentionOutput => &layer.o,
                    Int4ProjectionKind::Gate => &layer.gate,
                    Int4ProjectionKind::Up => &layer.up,
                    Int4ProjectionKind::Down => &layer.down,
                }
            }
        };
        Ok(matrix.as_linear())
    }
}

pub(super) fn check_binding(model: &Int4ModelIdentity, kernel: Int8KernelKind,
    executor: &dyn Int4ProjectionExecutor) -> Result<(), StrictInt4Error> {
    if executor.model_identity() != model || executor.kernel_kind() != kernel {
        return Err(StrictInt4Error::ProjectionExecutor);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn project_bf16<'executor, C: DecodeStepControl>(
    executor: &mut Option<&'executor mut (dyn Int4ProjectionExecutor + 'executor)>,
    op: Int4ProjectionOp, matrix: &GroupedInt4Matrix, input: &ActivationBuffer,
    rows: LinearRows<'_>, output: &mut [Bf16], scratch: &mut [i8], kernel: PackedInt4Kernel,
    ledger: &mut ProjectionLedger, control: &mut C) -> Result<(), StrictInt4Error> {
    match executor {
        Some(executor) => {
            matrix.as_linear().reserve_projection(input, rows, output.len(), ledger)?;
            executor.project_bf16_into(op, input, rows, output, control)
        }
        None => matrix.as_linear().with_kernel(kernel)
            .project_bf16_into(input, rows, output, scratch, ledger, control).map_err(Into::into),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn project_f32<'executor, C: DecodeStepControl>(
    executor: &mut Option<&'executor mut (dyn Int4ProjectionExecutor + 'executor)>,
    op: Int4ProjectionOp, matrix: &GroupedInt4Matrix, input: &ActivationBuffer,
    rows: LinearRows<'_>, output: &mut [f32], scratch: &mut [i8], kernel: PackedInt4Kernel,
    ledger: &mut ProjectionLedger, control: &mut C) -> Result<(), StrictInt4Error> {
    match executor {
        Some(executor) => {
            matrix.as_linear().reserve_projection(input, rows, output.len(), ledger)?;
            executor.project_f32_into(op, input, rows, output, control)
        }
        None => matrix.as_linear().with_kernel(kernel)
            .project_f32_into(input, rows, output, scratch, ledger, control).map_err(Into::into),
    }
}

#[cfg(test)]
mod tests;
