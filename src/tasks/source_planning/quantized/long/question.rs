//! Passage-scoped QA over a complete, losslessly partitioned document.
//!
//! Every nonblank chunk sees the SAME question and its own exact passage.
//! Answers are not votes: different texts survive, and no reducer invents a
//! global answer or treats model abstention as calibrated absence of evidence.

use super::*;
use crate::{
    tasks::{answer::{AnswerCalibration, AnswerResult, AnswerStatus, PassageCitation,
            ANSWER_PASSAGE_LAYOUT_VERSION, ANSWER_TASK_VERSION},
        ir::ScoreSpace, summarize::{CitationGuarantee, SourceCitation, SummarySemanticSupport}},
    validation::grounded_fields::{GroundingBudget, SourceOccurrence, scan_occurrences},
};

mod planning;
mod execution;

pub const INT8_QUESTION_EXECUTION: &str = "portable-int8-question-independent-passages-v1";
// Stable LOCAL passage id; the coordinator, not a model-supplied id, owns the
// original-document chunk id and coordinates. It never appears in trusted prose.
const PASSAGE_ID: &str = "document_chunk";

/// Private question text: deliberately not Debug or Serialize. Options govern
/// EACH native answer; verification is one nonrenewable whole-document budget.
pub struct SourceQuestion {
    pub question: String,
    pub options: AnswerOptions,
    pub verification: GroundingBudget,
}
impl SourceQuestion {
    pub fn validate(&self) -> Result<(), Int8SourceMapError> {
        self.options.validate().map_err(Int8SourceError::from)?;
        if self.question.len() > 64 * 1024 * 1024 || !has_text(&self.question) || !(1..=1_000_000).contains(&self.verification.max_fields)
            || !(1..=1_000_000).contains(&self.verification.max_matches)
            || self.verification.max_scan_steps == 0 {
            return Err(Int8SourceMapError::InvalidLimits);
        }
        Ok(())
    }
}
fn has_text(text: &str) -> bool { text.chars().any(|c| !c.is_whitespace()) }

/// Geometry and exact planned work, not a native receipt or source fingerprint.
/// Compiled from the pinned question, manifest and source encoders without
/// weights. Private fields prevent caller-invented preflight authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8QuestionPreflight {
    source_span: VerifiedSourceSpan,
    chunks: usize,
    native_chunks: usize,
    work: Int8Work,
    masks: u64,
    verification: GroundingBudget,
}
impl Int8QuestionPreflight {
    pub fn source_span(&self) -> VerifiedSourceSpan { self.source_span }
    pub fn chunk_count(&self) -> usize { self.chunks }
    pub fn native_chunks(&self) -> usize { self.native_chunks }
    pub fn whitespace_chunks(&self) -> usize { self.chunks - self.native_chunks }
    pub fn planned_work(&self) -> Int8Work { self.work }
    pub fn reserved_mask_visits(&self) -> u64 { self.masks }
    pub fn verify_completed(&self, run: &Int8QuestionRun) -> Result<(), Int8SourceMapError> {
        let root = run.mapped.root();
        let stats = execution::statistics(root.value())?;
        if run.schema_version != 1 || run.execution != INT8_QUESTION_EXECUTION
            || run.numerics_profile != STRICT_INT8_PROFILE
            || run.semantics != "independent-passage-answers-no-global-selection-v1"
            || run.calibration != AnswerCalibration::Uncalibrated
            || run.citation_guarantee != CitationGuarantee::StructuralSourceMembership
            || run.semantic_support != SummarySemanticSupport::NotAssessed
            || run.untrusted_fields != ["mapped.root.value"]
            || run.answered_chunks != stats.answered || run.abstained_chunks != stats.abstained
            || run.whitespace_chunks != stats.blank || run.distinct_answer_texts != stats.distinct
            || run.outcome != stats.outcome || run.model_work != stats.work || run.mask_node_visit_charge != stats.masks
            || run.verification_scan_steps > self.verification.max_scan_steps
            || stats.citations > self.verification.max_fields || stats.spans > self.verification.max_matches
            || root.source_span() != self.source_span || root.chunk_range() != (0..self.chunks)
            || root.value().chunks.len() != self.chunks || run.native_chunks != self.native_chunks
            || run.whitespace_chunks != self.whitespace_chunks()
            || run.answered_chunks.checked_add(run.abstained_chunks) != Some(self.native_chunks)
            || run.planned_model_work != self.work || run.reserved_mask_node_visits != self.masks
            || !within(run.model_work, self.work) || run.mask_node_visit_charge > self.masks {
            return Err(Int8SourceError::InvalidResult.into());
        }
        Ok(())
    }
}

/// Distinctness is EXACT text, not semantic agreement or contradiction analysis.
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
    /// All exact occurrences IN THIS CHUNK, in ORIGINAL-document coordinates.
    pub citations: Vec<SourceCitation>,
    pub model_work: Int8Work,
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
pub struct Int8QuestionRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub semantics: &'static str,
    pub outcome: QuestionOutcome,
    pub distinct_answer_texts: usize,
    pub native_chunks: usize,
    pub whitespace_chunks: usize,
    pub answered_chunks: usize,
    pub abstained_chunks: usize,
    pub planned_model_work: Int8Work,
    pub model_work: Int8Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub verification_scan_steps: u64,
    pub calibration: AnswerCalibration,
    pub citation_guarantee: CitationGuarantee,
    pub semantic_support: SummarySemanticSupport,
    pub untrusted_fields: [&'static str; 1],
    pub mapped: MapReduceResult<QuestionValue>,
}

/// Consumed whole-document commitment. Blank chunks retain source coverage but
/// have NO native plan, fake answer, generated tokens or renewed work allowance.
pub struct PreparedInt8Question<'s> {
    chunks: ChunkPlan<'s>,
    plans: Vec<PreparedInt8SourceTask>,
    plan_indices: Vec<Option<usize>>,
    mapping: Int8SourceMapLimits,
    options: AnswerOptions,
    verification: GroundingBudget,
    expected: Int8QuestionPreflight,
}
impl PreparedInt8Question<'_> {
    pub fn preflight_metadata(&self) -> Int8QuestionPreflight { self.expected }
    pub fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> {
        self.plans.iter().map(PreparedInt8SourceTask::execution_identity)
    }
    pub fn max_result_bytes(&self) -> u64 { self.mapping.reduction.max_result_bytes as u64 }
}

#[cfg(test)] mod tests;
