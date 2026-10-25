//! Explicit bounded quote-compression levels; never a single-pass fallback.
use super::*;
use crate::tasks::source_planning::grouped_int4::long::summary::synthesis::hierarchy::SummaryHierarchyLimits;

#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct HierarchyArgs {
    /// Enable lossy, context-sized quote-compression levels before the final
    /// summary. Quotes not selected by intermediate summaries may be lost.
    #[arg(long, requires = "synthesize_summary")]
    hierarchical_summary: bool,
    /// Maximum levels INCLUDING the final single-group pass; default 8.
    #[arg(long, requires = "hierarchical_summary")]
    summary_max_levels: Option<usize>,
    /// ALL additional native passes across ALL levels, including final; default 16.
    /// Every allowed pass is reserved at maximum context before discovery starts.
    #[arg(long, requires = "hierarchical_summary")]
    summary_max_passes: Option<usize>,
    /// Shared additional exact-token grouping calls; default 8192.
    #[arg(long, requires = "hierarchical_summary")]
    summary_max_tokenizer_calls: Option<usize>,
    /// Shared additional grouping input bytes; default 67108864.
    #[arg(long, requires = "hierarchical_summary")]
    summary_max_tokenizer_bytes: Option<u64>,
}
impl HierarchyArgs {
    pub(super) fn limits(self, synthesize: bool) -> Result<Option<SummaryHierarchyLimits>, CandidateError> {
        let supplied = self.summary_max_levels.is_some() || self.summary_max_passes.is_some()
            || self.summary_max_tokenizer_calls.is_some() || self.summary_max_tokenizer_bytes.is_some();
        if !self.hierarchical_summary {
            return if supplied { Err(CandidateError::Arguments) } else { Ok(None) };
        }
        if !synthesize { return Err(CandidateError::Arguments); }
        let defaults = SummaryHierarchyLimits::default();
        let limits = SummaryHierarchyLimits {
            max_levels: self.summary_max_levels.unwrap_or(defaults.max_levels),
            max_passes: self.summary_max_passes.unwrap_or(defaults.max_passes),
            max_tokenizer_calls: self.summary_max_tokenizer_calls.unwrap_or(defaults.max_tokenizer_calls),
            max_tokenizer_bytes: self.summary_max_tokenizer_bytes.unwrap_or(defaults.max_tokenizer_bytes),
        };
        limits.validate().map_err(|_| CandidateError::Arguments)?;
        Ok(Some(limits))
    }
}

pub(in super::super) fn workspace_bytes(budget: TaskBudget, mapping: Int4SourceMapLimits,
    chunks: usize, reserve: u64, synthesis: SummarySynthesisLimits, hierarchy: SummaryHierarchyLimits)
    -> Result<u64, CandidateError> {
    hierarchy.validate().map_err(|_| CandidateError::Arguments)?;
    let per_pass = sum(&[times(budget.max_output_bytes, 4)?, times(u64::from(budget.max_output_tokens), 8)?])?;
    sum(&[super::workspace_bytes(budget, mapping, chunks, reserve, synthesis)?,
        times(per_pass, (hierarchy.max_passes - 1) as u64)?,
        times(hierarchy.max_passes as u64, 8192 + 256)?, times(hierarchy.max_levels as u64, 128)?])
}

#[cfg(test)]
mod tests;
