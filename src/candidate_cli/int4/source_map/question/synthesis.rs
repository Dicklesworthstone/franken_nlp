//! Optional complete-evidence answer pass; independent answers remain the default.
use super::*;
use crate::tasks::source_planning::{SourcePlanningLimits,
    grouped_int4::long::question::synthesis::{QuestionSynthesisLimits, SourceQuestionSynthesis}};

pub(in super::super) mod hierarchy;

#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SynthesisArgs {
    /// Add one native answer pass over ALL collected, verified source quotes.
    /// Generated chunk answers remain output data and are never source facts.
    #[arg(long)]
    pub synthesize_answer: bool,
    /// Complete evidence passage cap; default 32, at most 1024, never top-k truncation.
    #[arg(long, requires = "synthesize_answer")]
    max_synthesis_passages: Option<usize>,
    /// Logical evidence bytes; default min(input cap, 16384). Actual question,
    /// manifest, token and context fit must also pass independently.
    #[arg(long, requires = "synthesize_answer")]
    max_synthesis_evidence_bytes: Option<usize>,
    #[command(flatten)]
    hierarchy: hierarchy::HierarchyArgs,
}
impl SynthesisArgs {
    pub(super) fn validate(self) -> Result<(), CandidateError> {
        let supplied = self.max_synthesis_passages.is_some() || self.max_synthesis_evidence_bytes.is_some();
        if (!self.synthesize_answer && supplied)
            || self.max_synthesis_passages.is_some_and(|n| !(1..=1024).contains(&n))
            || self.max_synthesis_evidence_bytes.is_some_and(|n| !(1..=MAX_INPUT_BYTES).contains(&n)) {
            return Err(CandidateError::Arguments);
        }
        self.hierarchy()?;
        Ok(())
    }
    pub(super) fn hierarchy(self)
        -> Result<Option<crate::tasks::source_planning::grouped_int4::long::question::synthesis::hierarchy::QuestionHierarchyLimits>, CandidateError> {
        self.hierarchy.limits(self.synthesize_answer)
    }
    pub(super) fn planning(self, mut planning: SourcePlanningLimits) -> SourcePlanningLimits {
        if self.synthesize_answer {
            // Only the explicit collected-evidence request expands this cap.
            // Ordinary passage QA and independent document maps keep theirs.
            planning.max_passages = self.max_synthesis_passages.unwrap_or(shared::MAX_PASSAGES);
        }
        planning
    }
    pub(super) fn limits(self, planning: SourcePlanningLimits)
        -> Result<Option<QuestionSynthesisLimits>, CandidateError> {
        self.validate()?;
        if !self.synthesize_answer { return Ok(None); }
        let limits = QuestionSynthesisLimits {
            max_evidence_passages: self.max_synthesis_passages.unwrap_or(shared::MAX_PASSAGES),
            max_evidence_bytes: self.max_synthesis_evidence_bytes.unwrap_or(planning.max_input_bytes.min(16384)),
        };
        if limits.max_evidence_passages > planning.max_passages || limits.max_evidence_bytes == 0
            || limits.max_evidence_bytes > planning.max_input_bytes {
            return Err(CandidateError::Arguments);
        }
        Ok(Some(limits))
    }
    pub(super) fn request(self, question: SourceQuestion, planning: SourcePlanningLimits)
        -> Result<SourceQuestionSynthesis, CandidateError> {
        let limits = self.limits(planning)?.ok_or(CandidateError::Arguments)?;
        question.validate().map_err(|_| CandidateError::Input)?;
        Ok(SourceQuestionSynthesis { question, limits })
    }
}

pub(in super::super) fn workspace_bytes(budget: TaskBudget, mapping: Int4SourceMapLimits,
    chunks: usize, reserve: u64, verification: GroundingBudget, limits: QuestionSynthesisLimits)
    -> Result<u64, CandidateError> {
    sum(&[super::workspace_bytes(budget, mapping, chunks, reserve, verification)?,
        times(limits.max_evidence_bytes as u64, 64)?, times(limits.max_evidence_passages as u64, 1024)?,
        times(verification.max_matches as u64, 128)?, times(verification.max_fields as u64, 128)?,
        times(budget.max_output_bytes, 4)?, times(u64::from(budget.max_output_tokens), 8)?])
}

#[cfg(test)]
mod tests;
