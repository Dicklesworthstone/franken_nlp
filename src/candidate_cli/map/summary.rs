//! Explicit complete-summary policy, separate from native per-chunk options.
use super::*;
use crate::{corpus::summarize::CorpusSummaryLimits,
    tasks::source_planning::quantized::long::summary::Int8SummaryLimits};

#[derive(Args)]
pub(in crate::candidate_cli) struct SummaryArgs {
    /// Union exact summary bullets and evidence, then rank the complete document.
    /// Requires --task summarize; this is not an additional neural synthesis pass.
    #[arg(long)]
    pub reduce_summary: bool,
    /// Final document-wide bullet cap (default 16), NOT a per-chunk truncation.
    #[arg(long, requires = "reduce_summary")]
    summary_bullets: Option<usize>,
    /// Entire pre-selection bullet union (default 4096). Overflow fails the run.
    #[arg(long, requires = "reduce_summary")]
    max_unique_summary_bullets: Option<usize>,
    /// All retained chunk/bullet citations, including unselected bullets (default 16384).
    #[arg(long, requires = "reduce_summary")]
    max_summary_citations: Option<usize>,
    /// Complete union of original-source evidence spans (default 65536).
    #[arg(long, requires = "reduce_summary")]
    max_summary_evidence_spans: Option<usize>,
    /// Shared independent citation-verification scan allowance (default 536870912).
    #[arg(long, requires = "reduce_summary")]
    max_summary_scan_steps: Option<u64>,
}
impl SummaryArgs {
    pub(super) fn validate(&self, task: &str, bytes: usize) -> Result<(), CandidateError> {
        if self.reduce_summary && task != "summarize" { return Err(CandidateError::Arguments); }
        self.limits(bytes).map(|_| ())
    }
    pub(in crate::candidate_cli) fn limits(&self, bytes: usize) -> Result<Option<Int8SummaryLimits>, CandidateError> {
        if !self.reduce_summary {
            if self.summary_bullets.is_some() || self.max_unique_summary_bullets.is_some()
                || self.max_summary_citations.is_some() || self.max_summary_evidence_spans.is_some()
                || self.max_summary_scan_steps.is_some() { return Err(CandidateError::Arguments); }
            return Ok(None);
        }
        let defaults = CorpusSummaryLimits::default();
        let aggregation = CorpusSummaryLimits {
            max_unique_bullets: self.max_unique_summary_bullets.unwrap_or(defaults.max_unique_bullets),
            max_citations: self.max_summary_citations.unwrap_or(defaults.max_citations),
            max_evidence_spans: self.max_summary_evidence_spans.unwrap_or(defaults.max_evidence_spans),
            max_value_bytes: bytes,
            max_scan_steps: self.max_summary_scan_steps.unwrap_or(defaults.max_scan_steps),
        };
        aggregation.validate().map_err(|_| CandidateError::Arguments)?;
        let max_bullets = self.summary_bullets.unwrap_or(16);
        if !(1..=1024).contains(&max_bullets) || aggregation.max_scan_steps > 1_000_000_000_000 {
            return Err(CandidateError::Arguments);
        }
        // The existing map/result, live-value and cumulative-value budgets also
        // apply. A small final top-k cannot hide an oversized complete union.
        Ok(Some(Int8SummaryLimits { aggregation, max_bullets, max_result_bytes: bytes }))
    }
}

#[cfg(test)] mod tests;
