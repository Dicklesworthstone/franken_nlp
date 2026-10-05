//! Explicit multi-level lossy quote selection, never an automatic retry.
use super::*;
use crate::tasks::source_planning::quantized::long::summary::synthesis::hierarchy::SummaryHierarchyLimits;

#[derive(Args)]
pub(in crate::candidate_cli) struct HierarchyArgs {
    /// Allow bounded context-sized quote-compression levels before the final
    /// summary. Quotes not selected by intermediate summaries may be lost.
    #[arg(long, requires = "synthesize_summary")]
    hierarchical_summary: bool,
    /// Maximum levels INCLUDING the final synthesis, default 8.
    #[arg(long, requires = "hierarchical_summary")]
    summary_max_levels: Option<usize>,
    /// ALL additional native passes across ALL levels, default 16.
    /// Every allowed pass is reserved at maximum context before weights load.
    #[arg(long, requires = "hierarchical_summary")]
    summary_max_passes: Option<usize>,
    /// Shared additional exact-token grouping calls, default 8192.
    #[arg(long, requires = "hierarchical_summary")]
    summary_max_tokenizer_calls: Option<usize>,
    /// Shared additional grouping input bytes, default 67108864.
    #[arg(long, requires = "hierarchical_summary")]
    summary_max_tokenizer_bytes: Option<u64>,
}
impl HierarchyArgs {
    pub(in crate::candidate_cli) fn limits(&self, synthesize: bool) -> Result<Option<SummaryHierarchyLimits>, CandidateError> {
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
#[cfg(test)] mod tests;
