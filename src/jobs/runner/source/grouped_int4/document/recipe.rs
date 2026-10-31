//! Exhaustive private projections for exact keyed resume commitments.
use super::*;
use crate::corpus::summarize::{CorpusSummaryLimits, SUMMARY_REDUCTION_POLICY};
use crate::tasks::corpus_keyphrases::{CorpusKeyphraseLimits, CORPUS_KEYPHRASE_POLICY};

#[derive(Serialize)]
pub struct Int4SourceDocumentJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    execution: &'static str,
    chunk_profile: &'static str,
    reservation_policy: &'static str,
    source: Int4SourceJobRecipe,
    mapping: MappingRecipe,
    #[serde(serialize_with = "serialize_task")]
    pub(super) task: Int4SourceDocumentTask,
    pub(super) lifetime: Int4SourceDocumentLifetime,
    max_attempts: u64,
    // Omitted entirely for the original serial profile: existing private
    // recipe bytes and authenticated resume commitments stay unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    chunk_cohorts: Option<&'static str>,
}
impl Int4SourceDocumentJobRecipe {
    pub(super) fn new(source: Int4SourceJobRecipe, task: Int4SourceDocumentTask,
        mapping: Int4SourceMapLimits, lifetime: Int4SourceDocumentLifetime, max_attempts: u64)
        -> Result<Self, BatchFault> {
        let Int4SourceMapLimits { chunks, reduction, max_model_work, mask_limits,
            mask_visits_per_chunk, max_mask_visits, verification } = mapping;
        let MaskWorkLimits { max_trie_node_visits, checkpoint_interval_nodes } = mask_limits;
        let recipe = Self { version: 1, dependency_scope: "item-local-complete-document",
            execution: task.execution(), chunk_profile: CHUNK_PROFILE,
            reservation_policy: "exact-discovery-and-complete-synthesis-reserve-per-durable-attempt-v1",
            source, mapping: MappingRecipe { chunks, reduction, max_model_work,
                max_trie_node_visits, checkpoint_interval_nodes, mask_visits_per_chunk,
                max_mask_visits, verification }, task, lifetime, max_attempts, chunk_cohorts: None };
        // Question bytes, options and every nested limit stay in this private
        // projection. Only JobRunner's per-job keyed commitment is persisted.
        bounded_json(&recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(recipe)
    }
    pub(super) fn enable_chunk_cohorts(&mut self) -> Result<(), BatchFault> {
        self.chunk_cohorts = Some("native-document-chunk-cohorts-v1");
        bounded_json(self, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(())
    }
    pub(super) fn uses_chunk_cohorts(&self) -> bool { self.chunk_cohorts.is_some() }
}
#[derive(Serialize)]
struct MappingRecipe {
    chunks: ChunkLimits,
    reduction: ExecutionLimits,
    max_model_work: Int4Work,
    max_trie_node_visits: usize,
    checkpoint_interval_nodes: usize,
    mask_visits_per_chunk: u64,
    max_mask_visits: u64,
    verification: GroundingBudget,
}
#[derive(Serialize)]
struct SummaryLimitsRecipe {
    policy: &'static str,
    aggregation: CorpusSummaryLimits,
    max_bullets: usize,
    max_result_bytes: usize,
}
#[derive(Serialize)]
struct KeyphraseLimitsRecipe {
    policy: &'static str,
    max_unique_phrases: usize,
    max_evidence_spans: usize,
    max_value_bytes: usize,
    max_scan_work: u64,
    max_phrases: usize,
    max_result_bytes: usize,
}
impl From<Int4KeyphraseLimits> for KeyphraseLimitsRecipe {
    fn from(value: Int4KeyphraseLimits) -> Self {
        let Int4KeyphraseLimits { aggregation, max_phrases, max_result_bytes } = value;
        let CorpusKeyphraseLimits { max_unique_phrases, max_evidence_spans, max_value_bytes, max_scan_work } = aggregation;
        Self { policy: CORPUS_KEYPHRASE_POLICY, max_unique_phrases, max_evidence_spans,
            max_value_bytes, max_scan_work, max_phrases, max_result_bytes }
    }
}
impl From<Int4SummaryLimits> for SummaryLimitsRecipe {
    fn from(value: Int4SummaryLimits) -> Self {
        let Int4SummaryLimits { aggregation, max_bullets, max_result_bytes } = value;
        Self { policy: SUMMARY_REDUCTION_POLICY, aggregation, max_bullets, max_result_bytes }
    }
}
#[derive(Serialize)]
struct SummarySynthesisRecipe {
    map_options: SummaryOptions,
    synthesis_options: SummaryOptions,
    max_evidence_segments: usize,
    max_evidence_bytes: usize,
    verification: GroundingBudget,
}
impl From<SourceSummarySynthesis> for SummarySynthesisRecipe {
    fn from(request: SourceSummarySynthesis) -> Self {
        let SourceSummarySynthesis { map_options, synthesis_options, limits } = request;
        let SummarySynthesisLimits { max_evidence_segments, max_evidence_bytes, verification } = limits;
        Self { map_options, synthesis_options, max_evidence_segments, max_evidence_bytes, verification }
    }
}
#[derive(Serialize)]
struct QuestionRecipe<'a> { question: &'a str, options: AnswerOptions, verification: GroundingBudget }
impl<'a> From<&'a SourceQuestion> for QuestionRecipe<'a> {
    fn from(request: &'a SourceQuestion) -> Self {
        let SourceQuestion { question, options, verification } = request;
        Self { question, options: *options, verification: *verification }
    }
}
#[derive(Serialize)]
struct QuestionSynthesisRecipe<'a> {
    question: QuestionRecipe<'a>, max_evidence_passages: usize, max_evidence_bytes: usize,
}
impl<'a> From<&'a SourceQuestionSynthesis> for QuestionSynthesisRecipe<'a> {
    fn from(request: &'a SourceQuestionSynthesis) -> Self {
        let SourceQuestionSynthesis { question, limits } = request;
        let QuestionSynthesisLimits { max_evidence_passages, max_evidence_bytes } = *limits;
        Self { question: question.into(), max_evidence_passages, max_evidence_bytes }
    }
}
#[derive(Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
enum TaskRecipe<'a> {
    Map { task: &'a SourceMapTask },
    Keyphrases { options: KeyphraseOptions, limits: KeyphraseLimitsRecipe, max_lifetime_scan_work: u64 },
    Summary { options: SummaryOptions, limits: SummaryLimitsRecipe },
    Question { request: QuestionRecipe<'a> },
    SummarySynthesis { request: SummarySynthesisRecipe },
    SummaryHierarchy { request: SummarySynthesisRecipe, limits: SummaryHierarchyLimits },
    QuestionSynthesis { request: QuestionSynthesisRecipe<'a> },
    QuestionHierarchy { request: QuestionSynthesisRecipe<'a>, limits: QuestionHierarchyLimits },
}
fn serialize_task<S: serde::Serializer>(task: &Int4SourceDocumentTask, serializer: S) -> Result<S::Ok, S::Error> {
    let recipe = match task {
        Int4SourceDocumentTask::Map(task) => TaskRecipe::Map { task },
        Int4SourceDocumentTask::Keyphrases { options, limits, max_lifetime_scan_work } => TaskRecipe::Keyphrases {
            options: *options, limits: (*limits).into(), max_lifetime_scan_work: *max_lifetime_scan_work },
        Int4SourceDocumentTask::Summary { options, limits } => TaskRecipe::Summary {
            options: *options, limits: (*limits).into() },
        Int4SourceDocumentTask::Question(question) => TaskRecipe::Question { request: question.into() },
        Int4SourceDocumentTask::SummarySynthesis(request) => TaskRecipe::SummarySynthesis { request: (*request).into() },
        Int4SourceDocumentTask::SummaryHierarchy { request, limits } => TaskRecipe::SummaryHierarchy {
            request: (*request).into(), limits: *limits },
        Int4SourceDocumentTask::QuestionSynthesis(request) => TaskRecipe::QuestionSynthesis { request: request.into() },
        Int4SourceDocumentTask::QuestionHierarchy { request, limits } => TaskRecipe::QuestionHierarchy {
            request: request.into(), limits: *limits },
    };
    recipe.serialize(serializer)
}
