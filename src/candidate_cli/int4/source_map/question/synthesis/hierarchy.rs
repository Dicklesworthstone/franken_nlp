//! Explicit lossy answer-evidence reduction with complete question-aware grouping.
use super::*;
use crate::tasks::source_planning::grouped_int4::long::question::synthesis::hierarchy::QuestionHierarchyLimits;

#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct HierarchyArgs {
    /// Enable bounded quote-selection levels before the final answer. Evidence
    /// not selected by intermediate answers may be lost.
    #[arg(long, requires = "synthesize_answer")]
    hierarchical_answer: bool,
    /// Maximum levels INCLUDING the final single-group pass; default 8.
    #[arg(long, requires = "hierarchical_answer")]
    answer_max_levels: Option<usize>,
    /// ALL additional native passes across ALL levels, including final; default 16.
    /// Every permitted pass is reserved at maximum context before discovery.
    #[arg(long, requires = "hierarchical_answer")]
    answer_max_passes: Option<usize>,
    /// Shared exact grouping attempts over question, manifest and quotes; default 8192.
    #[arg(long, requires = "hierarchical_answer")]
    answer_max_tokenizer_calls: Option<usize>,
    /// Shared grouping bytes including question and canonical manifests; default 67108864.
    #[arg(long, requires = "hierarchical_answer")]
    answer_max_tokenizer_bytes: Option<u64>,
}
impl HierarchyArgs {
    pub(super) fn limits(self, synthesize: bool) -> Result<Option<QuestionHierarchyLimits>, CandidateError> {
        let supplied = self.answer_max_levels.is_some() || self.answer_max_passes.is_some()
            || self.answer_max_tokenizer_calls.is_some() || self.answer_max_tokenizer_bytes.is_some();
        if !self.hierarchical_answer {
            return if supplied { Err(CandidateError::Arguments) } else { Ok(None) };
        }
        if !synthesize { return Err(CandidateError::Arguments); }
        let defaults = QuestionHierarchyLimits::default();
        let limits = QuestionHierarchyLimits {
            max_levels: self.answer_max_levels.unwrap_or(defaults.max_levels),
            max_passes: self.answer_max_passes.unwrap_or(defaults.max_passes),
            max_tokenizer_calls: self.answer_max_tokenizer_calls.unwrap_or(defaults.max_tokenizer_calls),
            max_tokenizer_bytes: self.answer_max_tokenizer_bytes.unwrap_or(defaults.max_tokenizer_bytes),
        };
        limits.validate().map_err(|_| CandidateError::Arguments)?;
        Ok(Some(limits))
    }
}

pub(in super::super::super) fn workspace_bytes(budget: TaskBudget, mapping: Int4SourceMapLimits,
    chunks: usize, reserve: u64, verification: GroundingBudget, synthesis: QuestionSynthesisLimits,
    hierarchy: QuestionHierarchyLimits) -> Result<u64, CandidateError> {
    hierarchy.validate().map_err(|_| CandidateError::Arguments)?;
    let per_pass = sum(&[times(budget.max_output_bytes, 4)?, times(u64::from(budget.max_output_tokens), 8)?])?;
    sum(&[super::workspace_bytes(budget, mapping, chunks, reserve, verification, synthesis)?,
        times(per_pass, (hierarchy.max_passes - 1) as u64)?,
        times(hierarchy.max_passes as u64, 8192 + 256)?, times(hierarchy.max_levels as u64, 128)?])
}

#[cfg(test)]
mod tests;
