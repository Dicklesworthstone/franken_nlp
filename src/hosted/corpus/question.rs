//! Complete documents and private per-record questions on one INT8 stream.
//! Original source/question owners and native reservations survive worker join.
use super::*;
use crate::{
    batch::source::quantized::question::{Int8QuestionDocumentBatchConfig,
        Int8QuestionDocumentBatchPlanner, NativeInt8QuestionDocumentBatch,
        Int8QuestionDocumentAdmission, Int8QuestionDocumentBatchAdmission,
        memory::{Int8QuestionDocumentMemoryLimits, Int8QuestionDocumentMemoryRequirement}},
    native_engine::{constrained_int8, simd::Int8Kernel,
        strict_int8::{StrictInt8Engine, prefill::Int8PrefillLimits}},
    tasks::{answer::ANSWER_TASK_VERSION, extract::ExtractionVocabulary},
};

/// Simultaneous complete-document ownership, including optional prompt and
/// worker storage. The model weights and caller's CLI preparation are separate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8QuestionDocumentCorpusRequirements {
    pub document: Int8QuestionDocumentMemoryRequirement,
    pub team_workspace_bytes: u64,
}
impl Int8QuestionDocumentCorpusRequirements {
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        let d = self.document;
        sum(&[d.native.kv_bytes, d.workspace_bytes, d.input_bytes,
            d.document_workspace_bytes, d.output_reservation_bytes, self.team_workspace_bytes])
    }
}

/// Pure quotation before reading options, population records or model bytes.
/// Preparation and IO declarations include the selected compiler, encoder and
/// owned streams; source/question copies and complete frontier are additional.
pub fn int8_question_document_corpus_requirements(config: Int8QuestionDocumentBatchConfig,
    limits: CorpusLimits, reduction_reserve_bytes: u64, prefill: Option<Int8PrefillLimits>,
    compute_threads: Option<usize>) -> Result<Int8QuestionDocumentCorpusRequirements, HostedError> {
    limits.reservation_bytes()?;
    let mut document = Int8QuestionDocumentMemoryLimits {
        context_tokens: limits.native.context_tokens,
        allocator_reserve_bytes: limits.native.allocator_reserve_bytes,
        preparation_reserve_bytes: limits.preparation_reserve_bytes,
        io_reserve_bytes: limits.io_reserve_bytes, reduction_reserve_bytes,
    }.requirements(config, limits.transport).map_err(HostedError::SourceMap)?;
    document.workspace_bytes = crate::hosted::scored::scoring_scratch(document.workspace_bytes, prefill)?;
    let team_workspace_bytes = compute_threads.map(|width|
        dispatch::int8_prefill_team_workspace_bytes(width, prefill.map_or(1, |p| p.max_batch_rows)))
        .transpose()?.unwrap_or(0);
    let required = Int8QuestionDocumentCorpusRequirements { document, team_workspace_bytes };
    required.total_bytes()?;
    Ok(required)
}

impl NlpEngine {
    /// Stream strict {id,text,task_args:{question}} records. Each document keeps
    /// every independent passage answer, abstention and whitespace range with
    /// original source coordinates. No failed chunk produces partial success.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_question_documents<R, W>(&self, model: &ResidentInt8,
        planner: Int8QuestionDocumentBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        limits: CorpusLimits, reduction_reserve_bytes: u64, reader: R, writer: W,
        cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.question_documents_scheduled(model, planner, vocabulary, limits,
            reduction_reserve_bytes, None, None, reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_question_documents_layer_major<R, W>(&self, model: &ResidentInt8,
        planner: Int8QuestionDocumentBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        limits: CorpusLimits, reduction_reserve_bytes: u64, prefill: Int8PrefillLimits,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.question_documents_scheduled(model, planner, vocabulary, limits,
            reduction_reserve_bytes, Some(prefill), None, reader, writer, cancellation)
    }

    /// One fixed projection team serves all records and source chunks. The
    /// terminal stream frame is published only after every worker has joined.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_question_documents_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Int8QuestionDocumentBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        limits: CorpusLimits, reduction_reserve_bytes: u64, kernel: Int8Kernel, compute_threads: usize,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.question_documents_scheduled(model, planner, vocabulary, limits,
            reduction_reserve_bytes, None, Some((kernel, compute_threads)), reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_question_documents_layer_major_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Int8QuestionDocumentBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        limits: CorpusLimits, reduction_reserve_bytes: u64, kernel: Int8Kernel, compute_threads: usize,
        prefill: Int8PrefillLimits, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.question_documents_scheduled(model, planner, vocabulary, limits,
            reduction_reserve_bytes, Some(prefill), Some((kernel, compute_threads)), reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn question_documents_scheduled<R, W>(&self, model: &ResidentInt8,
        planner: Int8QuestionDocumentBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        limits: CorpusLimits, reduction_reserve_bytes: u64, prefill: Option<Int8PrefillLimits>,
        team: Option<(Int8Kernel, usize)>, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, QuestionStream { planner: Arc::new(planner), vocabulary,
            reduction_reserve_bytes }, limits, prefill, team, reader, writer, cancellation)
    }
}

struct QuestionStream {
    planner: Arc<Int8QuestionDocumentBatchPlanner>,
    vocabulary: Arc<ExtractionVocabulary>,
    reduction_reserve_bytes: u64,
}
impl scored::CorpusTask for QuestionStream {
    fn identity(&self) -> &ExecutionIdentity { self.planner.base_execution_identity() }
    fn retain_for_join(&self) -> Option<Box<dyn Send>> {
        Some(Box::new((Arc::clone(&self.planner), Arc::clone(&self.vocabulary))))
    }
    fn validate(&self, model: &ArtifactIdentity, limits: CorpusLimits,
        native: Int8MemoryRequirement) -> Result<u64, HostedError> {
        check_model_identity(model, self.identity())?;
        let config = *self.planner.config();
        let required = int8_question_document_corpus_requirements(config, limits,
            self.reduction_reserve_bytes, None, None)?.document;
        if self.identity().task_spec != ANSWER_TASK_VERSION || required.native != native
            || self.planner.max_result_bytes() != required.max_output_bytes
            || self.planner.max_output_tokens() != required.max_output_tokens
            || self.planner.encoding_workspace_bytes().map_err(HostedError::BatchSetup)?
                > limits.preparation_reserve_bytes {
            return Err(HostedError::Limits("INT8 question corpus complete admission"));
        }
        // scored::run already retains the ordinary transport/preparation/IO
        // envelope. Add only the remaining source/question/identity owners and
        // complete native/reduction frontier, then quote that same peak in CLI.
        let extra = required.input_bytes.checked_sub(limits.reservation_bytes()?)
            .ok_or(HostedError::Limits("INT8 question corpus preparation accounting"))?;
        sum(&[extra, required.document_workspace_bytes])
    }
    fn execute<R: BufRead, W: Write>(self, engine: &mut StrictInt8Engine<'_>,
        admission: CorpusAdmission<'_>, prefill: Option<Int8PrefillLimits>,
        reader: &mut R, writer: &mut W, limits: BatchLimits, defer: bool,
        control: &mut RunControl) -> Result<scored::Completion, HostedError> {
        let config = *self.planner.config();
        let capacity = QuestionCapacity { config, kv_bytes: admission.kv_bytes,
            max_result_bytes: self.planner.max_result_bytes(), max_output_tokens: self.planner.max_output_tokens() };
        let admission = QuestionAdmission::new(admission.lease, admission.model,
            self.planner.base_execution_identity(), capacity);
        let mut processor = match prefill {
            Some(prefill) => NativeInt8QuestionDocumentBatch::new_layer_major(&self.planner,
                engine, &self.vocabulary, admission, prefill),
            None => NativeInt8QuestionDocumentBatch::new(&self.planner, engine, &self.vocabulary, admission),
        }.map_err(HostedError::BatchSetup)?;
        scored::execute_stream(reader, writer, &mut processor, limits, defer, control)
    }
}

#[derive(Clone, Copy)]
pub(in crate::hosted::corpus) struct QuestionCapacity {
    pub config: Int8QuestionDocumentBatchConfig,
    pub kv_bytes: u64,
    pub max_result_bytes: u64,
    pub max_output_tokens: u64,
}
pub(in crate::hosted::corpus) struct QuestionAdmission<'a> {
    lease: &'a EngineLease,
    model: &'a ArtifactIdentity,
    base: &'a ExecutionIdentity,
    capacity: QuestionCapacity,
}
impl<'a> QuestionAdmission<'a> {
    pub(in crate::hosted::corpus) fn new(lease: &'a EngineLease, model: &'a ArtifactIdentity,
        base: &'a ExecutionIdentity, capacity: QuestionCapacity) -> Self {
        Self { lease, model, base, capacity }
    }
}
impl Int8QuestionDocumentBatchAdmission for QuestionAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int8QuestionDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure> {
        check_question_request(self.base, self.capacity, &request)?;
        let mut identities = Vec::new();
        identities.try_reserve_exact(request.identities.len())
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
        for identity in request.identities {
            check_model_identity(self.model, identity)
                .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
            identities.push(identity.clone());
        }
        let guard = output_claim(self.lease, request.max_result_bytes, request.max_output_tokens)
            .map_err(admission_failure)?;
        Ok((identities, guard))
    }
}

pub(in crate::hosted::corpus) fn check_question_request(base: &ExecutionIdentity,
    capacity: QuestionCapacity, request: &Int8QuestionDocumentAdmission<'_>) -> Result<(), BatchItemFailure> {
    let config = capacity.config; let mapping = config.mapping;
    let admission = || BatchItemFailure::fatal(BatchCode::Admission);
    constrained_int8::check_profile(base).map_err(|_| admission())?;
    if base.task_spec != ANSWER_TASK_VERSION || request.chunk_count == 0
        || request.chunk_count > mapping.chunks.max_chunks || request.chunk_count > 256
        || request.native_chunks == 0 || request.native_chunks > request.chunk_count
        || request.identities.len() != request.native_chunks || capacity.kv_bytes == 0
        || request.kv_reservation_bytes != capacity.kv_bytes || capacity.kv_bytes != config.budget.max_kv_bytes {
        return Err(admission());
    }
    for identity in request.identities {
        constrained_int8::check_profile(identity).map_err(|_| admission())?;
        if identity.task_spec != ANSWER_TASK_VERSION || identity.tokenizer_digest != base.tokenizer_digest
            || identity.template_digest != base.template_digest {
            return Err(admission());
        }
    }
    let work = request.model_work;
    let tokens = (request.native_chunks as u64).checked_mul(u64::from(config.budget.max_output_tokens));
    if mapping.mask_visits_per_chunk.checked_mul(request.native_chunks as u64) != Some(request.mask_node_visits)
        || request.mask_node_visits == 0 || request.mask_node_visits > mapping.max_mask_visits
        || request.mask_node_visits > config.max_mask_visits
        || request.mask_limits.max_trie_node_visits != mapping.mask_limits.max_trie_node_visits
        || request.mask_limits.checkpoint_interval_nodes != mapping.mask_limits.checkpoint_interval_nodes
        || tokens.and_then(|n| n.checked_mul(NANBEIGE_VOCAB_SIZE as u64)) != Some(work.projected_logits)
        || work.forward_positions == 0 || work.attention_pairs == 0
        || work.projections.dot_products == 0 || work.projections.multiply_accumulates == 0
        || !work_fits(work, mapping.max_model_work) || !work_fits(work, config.max_model_work) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    if capacity.max_result_bytes == 0 || capacity.max_result_bytes != mapping.reduction.max_result_bytes as u64
        || request.max_result_bytes != capacity.max_result_bytes
        || capacity.max_output_tokens != 0 || request.max_output_tokens != 0 {
        return Err(BatchItemFailure::fatal(BatchCode::OutputLineLimit));
    }
    Ok(())
}

fn work_fits(work: Int8Work, maximum: Int8Work) -> bool {
    work.forward_positions <= maximum.forward_positions && work.projected_logits <= maximum.projected_logits
        && work.attention_pairs <= maximum.attention_pairs && work.projections.fits(maximum.projections)
}

#[cfg(test)] mod tests;
