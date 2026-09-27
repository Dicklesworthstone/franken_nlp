//! Explicit question file and whole-document QA verification limits.
use super::*;
use crate::{tasks::{answer::AnswerOptions, source_planning::quantized::long::question::SourceQuestion},
    validation::grounded_fields::GroundingBudget};

pub(in crate::candidate_cli) const QUESTION_BYTES: usize = source::OPTIONS_BYTES;

#[derive(Args)]
pub(in crate::candidate_cli) struct QuestionArgs {
    /// Exact UTF-8 question file for --task answer. '-' is refused: stdin is
    /// reserved for the original document. This is private data, not a template.
    #[arg(long, value_name = "FILE")]
    question: Option<PathBuf>,
    /// Independent citation-verification fields across ALL passages (default 4096).
    #[arg(long, requires = "question")]
    max_qa_citations: Option<usize>,
    /// Exact occurrence spans across ALL passage citations (default 16384).
    #[arg(long, requires = "question")]
    max_qa_evidence_spans: Option<usize>,
    /// Nonrenewable independent verification scan work (default 67108864).
    #[arg(long, requires = "question")]
    max_qa_scan_steps: Option<u64>,
}
impl QuestionArgs {
    pub(super) fn validate(&self, task: &str) -> Result<(), CandidateError> {
        let has_limits = self.max_qa_citations.is_some() || self.max_qa_evidence_spans.is_some()
            || self.max_qa_scan_steps.is_some();
        if task != "answer" {
            if self.question.is_some() || has_limits { return Err(CandidateError::Arguments); }
            return Ok(());
        }
        let path = self.question.as_ref().ok_or(CandidateError::Arguments)?;
        if path.as_os_str().is_empty() || path.as_os_str() == "-" { return Err(CandidateError::Arguments); }
        self.verification()?;
        Ok(())
    }
    pub(in crate::candidate_cli) fn path(&self) -> Result<&std::path::Path, CandidateError> {
        self.question.as_deref().ok_or(CandidateError::Arguments)
    }
    fn verification(&self) -> Result<GroundingBudget, CandidateError> {
        let defaults = GroundingBudget::default();
        let budget = GroundingBudget { max_fields: self.max_qa_citations.unwrap_or(defaults.max_fields),
            max_matches: self.max_qa_evidence_spans.unwrap_or(defaults.max_matches),
            max_scan_steps: self.max_qa_scan_steps.unwrap_or(defaults.max_scan_steps) };
        if !(1..=1_000_000).contains(&budget.max_fields) || !(1..=1_000_000).contains(&budget.max_matches)
            || !(1..=1_000_000_000_000).contains(&budget.max_scan_steps) { return Err(CandidateError::Arguments); }
        Ok(budget)
    }
    pub(in crate::candidate_cli) fn task(&self, question: String, raw: Option<&str>) -> Result<SourceQuestion, CandidateError> {
        self.validate("answer")?;
        if question.len() > QUESTION_BYTES { return Err(CandidateError::Input); }
        let options = options::<AnswerOptions>(raw)?;
        options.schema_source().map_err(|_| CandidateError::Planning)?;
        let task = SourceQuestion { question, options, verification: self.verification()? };
        task.validate().map_err(|_| CandidateError::Input)?;
        Ok(task)
    }
}
impl MapCommand {
    /// Requested ceilings only. The question-aware planner tightens these using
    /// its real repeated question, manifest and trusted scaffold before mapping.
    pub(in crate::candidate_cli) fn question_mapping(&self) -> Result<Int8SourceMapLimits, CandidateError> {
        if self.kind()? != BuiltInTask::Answer { return Err(CandidateError::Arguments); }
        self.question.validate(&self.task)?;
        let chunks = ChunkLimits { max_input_bytes: self.host.max_input_bytes,
            max_chunk_bytes: self.max_chunk_bytes.unwrap_or(self.host.context_tokens).min(self.host.max_input_bytes),
            max_chunk_tokens: self.host.context_tokens, context_tokens: self.host.context_tokens,
            reserved_tokens: self.host.max_new_tokens, max_chunks: self.max_chunks,
            max_tokenizer_calls: self.max_tokenizer_calls };
        chunks.effective_token_limit().map_err(|_| CandidateError::Arguments)?;
        Ok(self.mapping_limits(chunks))
    }
}

#[cfg(test)] mod tests;
