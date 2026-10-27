//! One private question repeated over the complete original document.
use super::*;
use crate::tasks::{answer::AnswerOptions,
    source_planning::grouped_int4::long::question::SourceQuestion};

pub(super) mod synthesis;

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
    #[command(flatten)]
    synthesis: synthesis::SynthesisArgs,
}
impl std::fmt::Debug for QuestionArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("QuestionArgs") }
}
pub(super) fn definition(command: clap::Command) -> clap::Command {
    QuestionArgs::augment_args(command.long_about("Ask one exact question from --question FILE of every nonblank chunk of one complete UTF-8 document. The default preserves independent chunk answers without selecting a global answer. --synthesize-answer explicitly adds one native answer pass over ALL independently verified verbatim quotes collected from those answers. Adding --hierarchical-answer explicitly enables bounded, lossy quote-selection levels before the final answer: each level consumes its entire ordered evidence frontier in groups that fit the original question, canonical passage manifest and exact context. Only verified original-source quotes selected by intermediate answers enter the next level; generated answer prose never becomes source evidence. Every allowed pass is reserved before discovery against one nonrenewable five-axis native work ceiling and mask allowance. The --max-qa-* verification counters span discovery, every level and original-coordinate citation lifting. Exhausted levels, passes, grouping or verification limits, an oversized atomic quote, nonprogress or cancellation returns no complete result. Evidence exhausted by intermediate abstentions remains distinct from a final model abstention or no initially collected evidence. All discovery answers and hierarchy pass receipts remain available. Final citations lift to every evidence-reachable original-document byte and Unicode-scalar occurrence. --options is the complete AnswerOptions JSON object. Questions and options are private local files; '-' is reserved for document stdin. Source conversion and independently pinned snapshot loading share the same path. Optional snapshot save follows complete-map and reserved-synthesis preflight and survives later task failure. No retrieval, calibrated answerability, semantic support guarantee, full-document equivalence, recall, contradiction resolution, tools, or thinking mode is implied."))
}
impl QuestionArgs {
    pub(super) fn validate(&self) -> Result<(), CandidateError> {
        if self.question.as_os_str().is_empty() || self.question.as_os_str() == "-" {
            return Err(CandidateError::Arguments);
        }
        self.verification()?;
        self.synthesis.validate()?;
        Ok(())
    }
    pub(super) fn path(&self) -> &std::path::Path { &self.question }
    pub(super) fn planning(&self, planning: crate::tasks::source_planning::SourcePlanningLimits)
        -> crate::tasks::source_planning::SourcePlanningLimits {
        self.synthesis.planning(planning)
    }
    pub(super) fn synthesis_limits(&self, planning: crate::tasks::source_planning::SourcePlanningLimits)
        -> Result<Option<crate::tasks::source_planning::grouped_int4::long::question::synthesis::QuestionSynthesisLimits>, CandidateError> {
        self.synthesis.limits(planning)
    }
    pub(super) fn hierarchy(&self)
        -> Result<Option<crate::tasks::source_planning::grouped_int4::long::question::synthesis::hierarchy::QuestionHierarchyLimits>, CandidateError> {
        self.synthesis.hierarchy()
    }
    #[cfg(feature = "asupersync-runtime")]
    pub(super) fn synthesis_request(&self, question: SourceQuestion,
        planning: crate::tasks::source_planning::SourcePlanningLimits)
        -> Result<crate::tasks::source_planning::grouped_int4::long::question::synthesis::SourceQuestionSynthesis, CandidateError> {
        self.synthesis.request(question, planning)
    }
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
