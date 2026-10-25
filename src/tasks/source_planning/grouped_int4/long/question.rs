//! One question over a complete document on the actual grouped-INT4 profile.
//!
//! Every nonblank chunk has its own native answer and independently verified
//! original-document citations. All answers survive; exact string agreement
//! does not establish semantic agreement, entailment or calibrated confidence.
use super::*;
use crate::tasks::{
    answer::{AnswerCalibration, AnswerResult, AnswerStatus, PassageCitation,
        ANSWER_PASSAGE_LAYOUT_VERSION},
    summarize::{CitationGuarantee, SourceCitation, SummarySemanticSupport},
};

mod planning;
mod execution;
/// Opt-in native answer synthesis using verified verbatim passage evidence.
pub mod synthesis;

pub const INT4_QUESTION_EXECUTION: &str = "grouped-int4-question-independent-passages-v1";
const QUESTION_SEMANTICS: &str = "independent-passage-answers-no-global-selection-v1";
// Model-provided passage ids never supply original-document authority.
const PASSAGE_ID: &str = "document_chunk";

/// Private source context, deliberately neither Debug nor Serialize. Options
/// govern each native answer; verification is a single complete-document budget.
pub struct SourceQuestion {
    pub question: String,
    pub options: AnswerOptions,
    pub verification: GroundingBudget,
}
impl SourceQuestion {
    pub fn validate(&self) -> Result<(), Int4SourceMapError> {
        self.options.validate().map_err(Int4SourceError::from)?;
        if self.question.len() > 64 * 1024 * 1024 || !has_text(&self.question)
            || !(1..=1_000_000).contains(&self.verification.max_fields)
            || !(1..=1_000_000).contains(&self.verification.max_matches)
            || self.verification.max_scan_steps == 0 {
            return Err(Int4SourceMapError::InvalidLimits);
        }
        Ok(())
    }
}
fn has_text(text: &str) -> bool { text.chars().any(|c| !c.is_whitespace()) }

/// Pinned source/question geometry and exact work, not a native result. Only
/// the planner creates this receipt; private fields prevent invented authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4QuestionPreflight {
    source_span: VerifiedSourceSpan,
    chunks: usize,
    native_chunks: usize,
    group_width: usize,
    work: Int4Work,
    masks: u64,
    verification: GroundingBudget,
}
impl Int4QuestionPreflight {
    pub fn source_span(&self) -> VerifiedSourceSpan { self.source_span }
    pub fn chunk_count(&self) -> usize { self.chunks }
    pub fn native_chunks(&self) -> usize { self.native_chunks }
    pub fn whitespace_chunks(&self) -> usize { self.chunks - self.native_chunks }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn planned_work(&self) -> Int4Work { self.work }
    pub fn reserved_mask_visits(&self) -> u64 { self.masks }
    pub fn verify_completed(&self, run: &Int4QuestionRun) -> Result<(), Int4SourceMapError> {
        let root = run.mapped.root();
        let stats = execution::statistics(root.value())?;
        if run.schema_version != 1 || run.execution != INT4_QUESTION_EXECUTION
            || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != self.group_width
            || run.semantics != QUESTION_SEMANTICS
            || run.calibration != AnswerCalibration::Uncalibrated
            || run.citation_guarantee != CitationGuarantee::StructuralSourceMembership
            || run.semantic_support != SummarySemanticSupport::NotAssessed
            || run.untrusted_fields != ["mapped.root.value"]
            || run.answered_chunks != stats.answered || run.abstained_chunks != stats.abstained
            || run.whitespace_chunks != stats.blank || run.distinct_answer_texts != stats.distinct
            || run.outcome != stats.outcome || run.model_work != stats.work
            || run.mask_node_visit_charge != stats.masks
            || run.verification_used.fields != stats.citations || run.verification_used.matches != stats.spans
            || !run.verification_used.fits(self.verification)
            || root.source_span() != self.source_span || root.chunk_range() != (0..self.chunks)
            || root.value().chunks.len() != self.chunks || run.native_chunks != self.native_chunks
            || run.whitespace_chunks != self.whitespace_chunks()
            || run.answered_chunks.checked_add(run.abstained_chunks) != Some(self.native_chunks)
            || run.planned_model_work != self.work || run.reserved_mask_node_visits != self.masks
            || !fits(run.model_work, self.work) || run.mask_node_visit_charge > self.masks {
            return Err(Int4SourceError::InvalidResult.into());
        }
        let mut byte_end = 0; let mut scalar_end = 0;
        for (index, chunk) in root.value().chunks().enumerate() {
            let span = chunk.source_span;
            if chunk.chunk_id != index || span.byte_start != byte_end || span.scalar_start != scalar_end
                || span.byte_end <= span.byte_start || span.scalar_end <= span.scalar_start {
                return Err(Int4SourceError::InvalidResult.into());
            }
            byte_end = span.byte_end; scalar_end = span.scalar_end;
        }
        if byte_end != self.source_span.byte_end || scalar_end != self.source_span.scalar_end {
            return Err(Int4SourceError::InvalidResult.into());
        }
        Ok(())
    }
}

/// Exact answer-text counts only; no voting or contradiction assessment.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionOutcome { NoAnswerProposed, OneAnswerText, MultipleAnswerTexts }
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionChunkStatus { Answered, Abstained, WhitespaceOnly }

#[derive(Serialize)]
pub struct QuestionChunk {
    pub chunk_id: usize,
    pub source_span: VerifiedSourceSpan,
    pub status: QuestionChunkStatus,
    pub answer: Option<String>,
    /// All exact occurrences within this chunk, in ORIGINAL-document coordinates.
    pub citations: Vec<SourceCitation>,
    pub model_work: Int4Work,
    pub mask_node_visit_charge: u64,
}
pub struct QuestionValue { chunks: Vec<Arc<QuestionChunk>> }
impl QuestionValue {
    pub fn chunks(&self) -> impl ExactSizeIterator<Item = &QuestionChunk> { self.chunks.iter().map(Arc::as_ref) }
}
impl Serialize for QuestionValue {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.chunks())
    }
}

#[derive(Serialize)]
pub struct Int4QuestionRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub group_width: usize,
    pub semantics: &'static str,
    pub outcome: QuestionOutcome,
    pub distinct_answer_texts: usize,
    pub native_chunks: usize,
    pub whitespace_chunks: usize,
    pub answered_chunks: usize,
    pub abstained_chunks: usize,
    pub planned_model_work: Int4Work,
    pub model_work: Int4Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub verification_used: SourceMapVerification,
    pub calibration: AnswerCalibration,
    pub citation_guarantee: CitationGuarantee,
    pub semantic_support: SummarySemanticSupport,
    pub untrusted_fields: [&'static str; 1],
    pub mapped: MapReduceResult<QuestionValue>,
}

/// Consumed complete-document commitment. Whitespace chunks have no fabricated
/// native plan, abstention, generated tokens or renewed work allowance.
pub struct PreparedInt4Question<'s> {
    chunks: ChunkPlan<'s>,
    plans: Vec<PreparedInt4SourceTask>,
    plan_indices: Vec<Option<usize>>,
    mapping: Int4SourceMapLimits,
    options: AnswerOptions,
    verification: GroundingBudget,
    expected: Int4QuestionPreflight,
    kv_ceiling: u64,
}
impl PreparedInt4Question<'_> {
    pub fn preflight_metadata(&self) -> Int4QuestionPreflight { self.expected }
    pub fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> {
        self.plans.iter().map(PreparedInt4SourceTask::execution_identity)
    }
    pub fn max_result_bytes(&self) -> u64 { self.mapping.reduction.max_result_bytes as u64 }
    pub fn max_kv_bytes(&self) -> u64 { self.kv_ceiling }
    pub fn group_width(&self) -> usize { self.expected.group_width }
}

#[cfg(test)] mod tests;
