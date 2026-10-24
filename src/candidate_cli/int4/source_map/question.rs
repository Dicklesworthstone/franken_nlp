//! One private question repeated over the complete original document.
use super::*;
use crate::tasks::{answer::AnswerOptions,
    source_planning::grouped_int4::long::question::SourceQuestion};

pub(super) const QUESTION_BYTES: usize = shared::OPTIONS_BYTES;

#[derive(Args, Clone, Eq, PartialEq)]
pub(super) struct QuestionArgs {
    /// Exact UTF-8 question file; stdin is reserved for the complete document.
    #[arg(long, value_name = "FILE")]
    question: PathBuf,
    /// Independent citation fields across ALL answered chunks.
    #[arg(long, default_value_t = 4096)]
    max_qa_citations: usize,
    /// All compatible source occurrences, without resetting for another chunk.
    #[arg(long, default_value_t = 16384)]
    max_qa_evidence_spans: usize,
    /// Nonrenewable citation verification scan work across the complete document.
    #[arg(long, default_value_t = 67_108_864)]
    max_qa_scan_steps: u64,
}
impl std::fmt::Debug for QuestionArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("QuestionArgs") }
}
pub(super) fn definition(command: clap::Command) -> clap::Command {
    QuestionArgs::augment_args(command.long_about("Ask one exact question from --question FILE of every nonblank chunk of one complete UTF-8 document. The question, options, source manifest and trusted template determine the actual context capacity. All chunks and combined native work preflight before the first forward. Keep every per-chunk answer, explicit uncalibrated abstention, and whitespace-only chunk in original order, without majority voting or a global answer. Citations retain every compatible original-document byte and Unicode-scalar occurrence. A failed chunk, cancellation or exceeded budget returns no document result; failure never becomes abstention. --options is the complete AnswerOptions JSON object, independent of the complete-document --max-qa-* verification ceilings. Questions and options are private local files; '-' is reserved for document stdin. Source conversion and independently pinned snapshot loading share the same path. No retrieval, answer synthesis, semantic support guarantee, tools, or thinking mode is implied."))
}
impl QuestionArgs {
    pub(super) fn validate(&self) -> Result<(), CandidateError> {
        if self.question.as_os_str().is_empty() || self.question.as_os_str() == "-" {
            return Err(CandidateError::Arguments);
        }
        self.verification()?;
        Ok(())
    }
    pub(super) fn path(&self) -> &std::path::Path { &self.question }
    pub(super) fn verification(&self) -> Result<GroundingBudget, CandidateError> {
        let verification = GroundingBudget { max_fields: self.max_qa_citations,
            max_matches: self.max_qa_evidence_spans, max_scan_steps: self.max_qa_scan_steps };
        if !(1..=1_000_000).contains(&verification.max_fields)
            || !(1..=1_000_000).contains(&verification.max_matches)
            || !(1..=1_000_000_000_000).contains(&verification.max_scan_steps) {
            return Err(CandidateError::Arguments);
        }
        Ok(verification)
    }
    pub(super) fn task(&self, question: String, raw: Option<&str>) -> Result<SourceQuestion, CandidateError> {
        self.validate()?;
        if question.len() > QUESTION_BYTES { return Err(CandidateError::Input); }
        let options = match raw {
            None => AnswerOptions::default(),
            Some(raw) => {
                if raw.len() > shared::OPTIONS_BYTES { return Err(CandidateError::Input); }
                let value = canonjson::parse_str_with_limits(raw, canonjson::ParseLimits {
                    max_depth: 8, max_string_bytes: shared::OPTIONS_BYTES,
                }).map_err(|_| CandidateError::Input)?;
                serde_json::from_value::<AnswerOptions>(value).map_err(|_| CandidateError::Input)?
            }
        };
        options.schema_source().map_err(|_| CandidateError::Planning)?;
        let task = SourceQuestion { question, options, verification: self.verification()? };
        task.validate().map_err(|_| CandidateError::Input)?;
        Ok(task)
    }
}

pub(super) fn workspace_bytes(budget: TaskBudget, mapping: Int4SourceMapLimits,
    chunks: usize, reserve: u64, verification: GroundingBudget) -> Result<u64, CandidateError> {
    use crate::tasks::{answer::{PassageCitation, PassageOccurrence}, summarize::SourceCitation};
    let fields = times(verification.max_fields as u64, sum(&[
        std::mem::size_of::<PassageCitation>() as u64,
        std::mem::size_of::<SourceCitation>() as u64])?)?;
    let spans = times(verification.max_matches as u64, sum(&[
        std::mem::size_of::<PassageOccurrence>() as u64,
        times(std::mem::size_of::<VerifiedSourceSpan>() as u64, 2)?])?)?;
    sum(&[super::workspace_bytes(budget, mapping, chunks, reserve)?, fields, spans])
}

#[cfg(test)]
mod tests;
