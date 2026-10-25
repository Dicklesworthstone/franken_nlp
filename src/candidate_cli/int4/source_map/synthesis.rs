//! Explicit all-evidence neural synthesis, separate from the exact bullet union.
use super::*;
use crate::tasks::{summarize::SummaryOptions, source_planning::{SourcePlanningLimits,
    grouped_int4::long::summary::synthesis::{SourceSummarySynthesis, SummarySynthesisLimits}}};
pub(super) mod hierarchy;

#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SynthesisArgs {
    /// Add one neural summary pass over every collected, independently verified
    /// source quote. Generated map bullets are never source evidence.
    #[arg(long)]
    pub synthesize_summary: bool,
    #[command(flatten)]
    hierarchy: hierarchy::HierarchyArgs,
    /// Neural bullet cap for the final and any explicitly enabled hierarchy
    /// passes; omitted preserves the per-chunk options cap.
    #[arg(long, requires = "synthesize_summary")]
    synthesis_bullets: Option<usize>,
    /// Maximum distinct (chunk, quote) segments; default 256, with no truncation.
    #[arg(long, requires = "synthesize_summary")]
    max_synthesis_evidence_segments: Option<usize>,
    /// Complete evidence bytes including separators; default min(input cap, 65536).
    /// Exact prompt, token and context capacity are independently enforced.
    #[arg(long, requires = "synthesize_summary")]
    max_synthesis_evidence_bytes: Option<usize>,
    /// Whole-discovery and synthesis field checks; default 4096.
    #[arg(long, requires = "synthesize_summary")]
    max_synthesis_fields: Option<usize>,
    /// Whole-discovery and synthesis occurrence checks; default 16384.
    #[arg(long, requires = "synthesize_summary")]
    max_synthesis_matches: Option<usize>,
    /// Nonrenewable discovery, collection and citation-lift work; default 67108864.
    #[arg(long, requires = "synthesize_summary")]
    max_synthesis_scan_steps: Option<u64>,
}
impl SynthesisArgs {
    pub(super) fn hierarchy(self)
        -> Result<Option<crate::tasks::source_planning::grouped_int4::long::summary::synthesis::hierarchy::SummaryHierarchyLimits>, CandidateError> {
        self.hierarchy.limits(self.synthesize_summary)
    }
    pub(super) fn limits(self, planning: SourcePlanningLimits)
        -> Result<Option<SummarySynthesisLimits>, CandidateError> {
        self.hierarchy()?;
        let supplied = self.synthesis_bullets.is_some() || self.max_synthesis_evidence_segments.is_some()
            || self.max_synthesis_evidence_bytes.is_some() || self.max_synthesis_fields.is_some()
            || self.max_synthesis_matches.is_some() || self.max_synthesis_scan_steps.is_some();
        if !self.synthesize_summary {
            return if supplied { Err(CandidateError::Arguments) } else { Ok(None) };
        }
        if self.synthesis_bullets.is_some_and(|n| !(1..=1024).contains(&n)) {
            return Err(CandidateError::Arguments);
        }
        let defaults = GroundingBudget::default();
        let limits = SummarySynthesisLimits {
            max_evidence_segments: self.max_synthesis_evidence_segments.unwrap_or(256),
            max_evidence_bytes: self.max_synthesis_evidence_bytes.unwrap_or(planning.max_input_bytes.min(65_536)),
            verification: GroundingBudget {
                max_fields: self.max_synthesis_fields.unwrap_or(defaults.max_fields),
                max_matches: self.max_synthesis_matches.unwrap_or(defaults.max_matches),
                max_scan_steps: self.max_synthesis_scan_steps.unwrap_or(defaults.max_scan_steps),
            },
        };
        limits.validate(planning).map_err(|_| CandidateError::Arguments)?;
        Ok(Some(limits))
    }
    pub(super) fn request(self, map_options: SummaryOptions, planning: SourcePlanningLimits)
        -> Result<SourceSummarySynthesis, CandidateError> {
        let limits = self.limits(planning)?.ok_or(CandidateError::Arguments)?;
        let mut synthesis_options = map_options;
        if let Some(bullets) = self.synthesis_bullets { synthesis_options.max_bullets = bullets; }
        let request = SourceSummarySynthesis { map_options, synthesis_options, limits };
        request.validate(planning).map_err(|_| CandidateError::Arguments)?;
        Ok(request)
    }
}

/// Matches the host's charged map-plus-synthesis workspace quote. These finite
/// payload bounds are admission inputs, never an observed RSS claim.
pub(super) fn workspace_bytes(budget: TaskBudget, mapping: Int4SourceMapLimits,
    chunks: usize, reserve: u64, limits: SummarySynthesisLimits) -> Result<u64, CandidateError> {
    sum(&[super::workspace_bytes(budget, mapping, chunks, reserve)?,
        times(budget.max_output_bytes, 4)?, times(u64::from(budget.max_output_tokens), 8)?,
        times(limits.max_evidence_bytes as u64, 64)?, times(limits.max_evidence_segments as u64, 128)?,
        times(limits.verification.max_matches as u64, 128)?,
        times(limits.verification.max_fields as u64, 128)?])
}

#[cfg(test)]
mod tests;
