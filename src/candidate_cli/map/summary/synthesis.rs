//! Explicit neural synthesis; never an implicit replacement for exact ranking.
use super::*;
use crate::{tasks::{summarize::SummaryOptions, source_planning::SourcePlanningLimits,
    source_planning::quantized::long::summary::synthesis::{SourceSummarySynthesis, SummarySynthesisLimits}},
    validation::grounded_fields::GroundingBudget};

#[derive(Args)]
pub(in crate::candidate_cli) struct SynthesisArgs {
    /// Run an additional neural summary over verified source quotes from ALL
    /// chunk summaries. No generated map bullet is used as a source fact.
    /// All evidence must fit: overflow fails rather than selecting a subset.
    #[arg(long, conflicts_with = "reduce_summary")]
    pub synthesize_summary: bool,
    /// Final neural bullet cap; omitted uses the per-chunk SummaryOptions cap.
    #[arg(long, requires = "synthesize_summary")]
    synthesis_bullets: Option<usize>,
    /// Distinct (chunk, quote) segments, default 256; duplicates are reverified.
    #[arg(long, requires = "synthesize_summary")]
    max_synthesis_evidence_segments: Option<usize>,
    /// Complete evidence bytes INCLUDING separators, default min(input cap, 65536).
    /// Actual prompt/token/context fit is checked independently before synthesis.
    #[arg(long, requires = "synthesize_summary")]
    max_synthesis_evidence_bytes: Option<usize>,
    /// Independent collection + final-lift field checks, default 4096.
    #[arg(long, requires = "synthesize_summary")]
    max_synthesis_fields: Option<usize>,
    /// Aggregate scan matches AND original-coordinate fanout, default 16384.
    #[arg(long, requires = "synthesize_summary")]
    max_synthesis_matches: Option<usize>,
    /// Shared independent verification work, default 67108864.
    #[arg(long, requires = "synthesize_summary")]
    max_synthesis_scan_steps: Option<u64>,
}
impl SynthesisArgs {
    pub(in crate::candidate_cli) fn validate(&self, task: &str, reduce: bool) -> Result<(), CandidateError> {
        let supplied = self.synthesis_bullets.is_some() || self.max_synthesis_evidence_segments.is_some()
            || self.max_synthesis_evidence_bytes.is_some() || self.max_synthesis_fields.is_some()
            || self.max_synthesis_matches.is_some() || self.max_synthesis_scan_steps.is_some();
        if !self.synthesize_summary {
            return if supplied { Err(CandidateError::Arguments) } else { Ok(()) };
        }
        if task != "summarize" || reduce || self.synthesis_bullets.is_some_and(|n| !(1..=1024).contains(&n)) {
            return Err(CandidateError::Arguments);
        }
        self.limits(SourcePlanningLimits::default()).map(|_| ())
    }
    pub(in crate::candidate_cli) fn check_planning(&self, planning: SourcePlanningLimits) -> Result<(), CandidateError> {
        if !self.synthesize_summary { return Err(CandidateError::Arguments); }
        self.limits(planning).map(|_| ())
    }
    fn limits(&self, planning: SourcePlanningLimits) -> Result<SummarySynthesisLimits, CandidateError> {
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
        Ok(limits)
    }
    pub(in crate::candidate_cli) fn request(&self, map_options: SummaryOptions, planning: SourcePlanningLimits)
        -> Result<SourceSummarySynthesis, CandidateError> {
        self.validate("summarize", false)?;
        if !self.synthesize_summary { return Err(CandidateError::Arguments); }
        let mut synthesis_options = map_options;
        if let Some(n) = self.synthesis_bullets { synthesis_options.max_bullets = n; }
        let request = SourceSummarySynthesis { map_options, synthesis_options, limits: self.limits(planning)? };
        request.validate(planning).map_err(|_| CandidateError::Arguments)?;
        Ok(request)
    }
}
#[cfg(test)] mod tests;
