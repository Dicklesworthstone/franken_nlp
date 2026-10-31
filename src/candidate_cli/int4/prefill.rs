//! Physical prompt scheduling for explicitly supported tasks. No semantic-policy change.
use super::*;

/// Reveal scheduling only at a root command wired to a hosted single task.
/// Shared source/extraction definitions stay strict for their composed routes.
pub(super) fn definition(command: clap::Command) -> clap::Command {
    command.mut_arg("prefill_rows", |arg| arg.hide(false)
        .help("Explicit INT4 layer-major prompt execution, 1..=64 rows; extra workspace is process-admitted"))
}

impl Int4Args {
    pub(super) fn prefill(&self) -> Result<Option<Int4PrefillLimits>, CandidateError> {
        // Reuse only the shared parser's row count, never its INT8 workspace
        // or executable type. The INT4 implementation owns its own admission.
        self.policy.prefill()?.map(|parsed| {
            let rows = parsed.max_batch_rows;
            let bytes = Int4PrefillLimits::required_extra_scratch_bytes(rows)
                .map_err(|_| CandidateError::Arguments)?;
            Ok(Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: bytes })
        }).transpose()
    }
}

pub(super) fn matches(args: &Int4Args, limits: Int4Limits) -> Result<bool, CandidateError> {
    let facts = |p: Int4PrefillLimits| (p.max_batch_rows, p.max_extra_scratch_bytes);
    Ok(args.prefill()?.map(facts) == limits.prefill.map(facts))
}

#[cfg(test)] mod tests;
