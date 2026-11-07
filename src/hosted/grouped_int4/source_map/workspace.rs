//! Identity-free complete-frontier accounting shared by one-off, live corpus
//! and durable source documents. These quotes do not grant native admission.
use super::*;
use crate::batch::source::grouped_int4::document::Int4SourceDocumentTask;
use crate::tasks::{
    answer::{PassageCitation, PassageOccurrence},
    summarize::SourceCitation,
    source_planning::grouped_int4::long::{
        keyphrases::Int4KeyphraseLimits,
        question::{SourceQuestion, synthesis::{SourceQuestionSynthesis,
            hierarchy::QuestionHierarchyLimits}},
        summary::synthesis::{SourceSummarySynthesis, hierarchy::SummaryHierarchyLimits},
    },
};

#[derive(Clone, Copy)]
pub(super) struct Workspace {
    pub(super) budget: TaskBudget,
    pub(super) mapping: Int4SourceMapLimits,
    pub(super) reserve_bytes: u64,
}
impl Workspace {
    pub(super) fn map(self, chunks: usize) -> Result<u64, HostedError> {
        if chunks == 0 || chunks > self.mapping.chunks.max_chunks || chunks > 256
            || self.reserve_bytes == 0 {
            return Err(HostedError::Limits("INT4 source map partition or reduction reserve"));
        }
        let native = multiply(self.budget.max_output_bytes, chunks as u64)?;
        let staged = multiply(sum(&[native, self.mapping.reduction.max_live_value_bytes as u64])?, 4)?;
        let tokens = multiply(multiply(u64::from(self.budget.max_output_tokens), chunks as u64)?, 8)?;
        let fields = multiply(self.mapping.verification.max_fields as u64,
            std::mem::size_of::<SourceFieldEvidence>() as u64)?;
        let spans = multiply(multiply(self.mapping.verification.max_matches as u64,
            std::mem::size_of::<VerifiedSourceSpan>() as u64)?, 2)?;
        sum(&[staged, tokens, fields, spans, self.reserve_bytes])
    }

    pub(super) fn keyphrases(self, chunks: usize, limits: Int4KeyphraseLimits)
        -> Result<u64, HostedError> {
        limits.validate(self.mapping).map_err(|e| HostedError::Int4(Int4HostError::SourceKeyphrases(e)))?;
        let values = multiply(limits.aggregation.max_value_bytes as u64, 8)?;
        let candidates = multiply(limits.aggregation.max_unique_phrases as u64, 512)?;
        let evidence = multiply(multiply(limits.aggregation.max_evidence_spans as u64,
            std::mem::size_of::<VerifiedSourceSpan>() as u64)?, 2)?;
        sum(&[self.map(chunks)?, values, candidates, evidence])
    }

    pub(super) fn question(self, chunks: usize, task: &SourceQuestion) -> Result<u64, HostedError> {
        let citations = multiply(task.verification.max_fields as u64,
            (std::mem::size_of::<PassageCitation>() + std::mem::size_of::<SourceCitation>()) as u64)?;
        let spans = multiply(task.verification.max_matches as u64,
            (std::mem::size_of::<PassageOccurrence>() + 2 * std::mem::size_of::<VerifiedSourceSpan>()) as u64)?;
        sum(&[self.map(chunks)?, citations, spans])
    }

    pub(super) fn summary_synthesis(self, chunks: usize, task: &SourceSummarySynthesis)
        -> Result<u64, HostedError> {
        let limits = task.limits;
        sum(&[self.map(chunks)?, multiply(self.budget.max_output_bytes, 4)?,
            multiply(u64::from(self.budget.max_output_tokens), 8)?,
            multiply(limits.max_evidence_bytes as u64, 64)?,
            multiply(limits.max_evidence_segments as u64, 128)?,
            multiply(limits.verification.max_matches as u64, 128)?,
            multiply(limits.verification.max_fields as u64, 128)?])
    }

    pub(super) fn question_synthesis(self, chunks: usize, task: &SourceQuestionSynthesis)
        -> Result<u64, HostedError> {
        let verification = task.question.verification;
        let limits = task.limits;
        let fields = multiply(verification.max_fields as u64,
            (std::mem::size_of::<PassageCitation>() + std::mem::size_of::<SourceCitation>()) as u64)?;
        let spans = multiply(verification.max_matches as u64,
            (std::mem::size_of::<PassageOccurrence>() + 2 * std::mem::size_of::<VerifiedSourceSpan>()) as u64)?;
        sum(&[self.map(chunks)?, fields, spans,
            multiply(limits.max_evidence_bytes as u64, 64)?,
            multiply(limits.max_evidence_passages as u64, 1024)?,
            multiply(verification.max_matches as u64, 128)?,
            multiply(verification.max_fields as u64, 128)?,
            multiply(self.budget.max_output_bytes, 4)?,
            multiply(u64::from(self.budget.max_output_tokens), 8)?])
    }

    pub(super) fn summary_hierarchy(self, chunks: usize, task: &SourceSummarySynthesis,
        hierarchy: SummaryHierarchyLimits) -> Result<u64, HostedError> {
        hierarchy.validate().map_err(map_error)?;
        sum(&[self.summary_synthesis(chunks, task)?,
            self.hierarchy(hierarchy.max_passes, hierarchy.max_levels)?])
    }

    pub(super) fn question_hierarchy(self, chunks: usize, task: &SourceQuestionSynthesis,
        hierarchy: QuestionHierarchyLimits) -> Result<u64, HostedError> {
        hierarchy.validate().map_err(map_error)?;
        sum(&[self.question_synthesis(chunks, task)?,
            self.hierarchy(hierarchy.max_passes, hierarchy.max_levels)?])
    }

    fn hierarchy(self, passes: usize, levels: usize) -> Result<u64, HostedError> {
        let per_pass = sum(&[multiply(self.budget.max_output_bytes, 4)?,
            multiply(u64::from(self.budget.max_output_tokens), 8)?])?;
        // The flat synthesis quote already owns one pass. Every permitted
        // transcript and complete-level witness remains live until reduction.
        sum(&[multiply(per_pass, (passes - 1) as u64)?,
            multiply(passes as u64, 8192 + 256)?, multiply(levels as u64, 128)?])
    }
}

fn multiply(left: u64, right: u64) -> Result<u64, HostedError> {
    left.checked_mul(right).ok_or(HostedError::Limits("INT4 source frontier arithmetic"))
}

pub(in crate::hosted::grouped_int4) fn document_bytes(task: &Int4SourceDocumentTask,
    budget: TaskBudget, mapping: Int4SourceMapLimits, reserve_bytes: u64) -> Result<u64, HostedError> {
    let workspace = Workspace { budget, mapping, reserve_bytes };
    let chunks = mapping.chunks.max_chunks;
    match task {
        Int4SourceDocumentTask::Map(_) | Int4SourceDocumentTask::Summary { .. } => workspace.map(chunks),
        Int4SourceDocumentTask::Keyphrases { limits, .. } => workspace.keyphrases(chunks, *limits),
        Int4SourceDocumentTask::Question(question) => workspace.question(chunks, question),
        Int4SourceDocumentTask::SummarySynthesis(request) => workspace.summary_synthesis(chunks, request),
        Int4SourceDocumentTask::SummaryHierarchy { request, limits } =>
            workspace.summary_hierarchy(chunks, request, *limits),
        Int4SourceDocumentTask::QuestionSynthesis(request) => workspace.question_synthesis(chunks, request),
        Int4SourceDocumentTask::QuestionHierarchy { request, limits } =>
            workspace.question_hierarchy(chunks, request, *limits),
    }
}
