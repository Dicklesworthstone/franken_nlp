//! Complete-document INT8 sentiment through the shared owned corpus host.
use super::*;
use crate::{
    batch::sentiment::quantized::{
        INT8_SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES,
        Int8SentimentDocumentAdmission, Int8SentimentDocumentBatchAdmission,
        Int8SentimentDocumentBatchPlanner, NativeInt8SentimentDocumentBatch,
    },
    native_engine::{constrained_int8, kv::KV_BYTES_PER_TOKEN, simd::Int8Kernel,
        strict_int8::prefill::Int8PrefillLimits},
    tasks::{ir::TaskBudget, sentiment::{SentimentLimits, SentimentPlanner,
        quantized::long::{Int8SentimentMapLimits,
            memory::{Int8SentimentMapMemoryLimits, Int8SentimentMapMemoryRequirement}}}},
    tokenizer::untrusted::UntrustedEncoding,
};

pub use crate::batch::sentiment::quantized::Int8SentimentDocumentBatchConfig
    as SentimentDocumentCorpusConfig;

/// Checked peak ownership before private input, tokenizer or model IO. Four
/// axes price every valid per-record replacement, independently of defaults.
/// Native storage is reused; complete plans, retained chunk evidence, frozen
/// request identities and transport buffers coexist. This is a modeled
/// reservation, not measured RSS. Defaults and their physical-join keepalive
/// must fit the explicit preparation envelope. Prefill and worker scratch are additional.
pub fn int8_sentiment_document_corpus_requirements(task: TaskBudget, planning: SentimentLimits,
    mapping: Int8SentimentMapLimits, encoding: &UntrustedEncoding, limits: CorpusLimits)
    -> Result<Int8SentimentMapMemoryRequirement, HostedError> {
    let transport = limits.reservation_bytes()?;
    if mapping.max_result_bytes > limits.transport.max_output_line_bytes as u64 {
        return Err(HostedError::Limits("complete sentiment document output reservation"));
    }
    let mut required = Int8SentimentMapMemoryLimits {
        context_tokens: limits.native.context_tokens,
        preparation_reserve_bytes: limits.preparation_reserve_bytes,
        allocator_reserve_bytes: limits.native.allocator_reserve_bytes,
        max_axes: 4,
    }.requirements(task, planning, mapping)
        .and_then(|required| required.with_input_encoding(encoding))
        .map_err(HostedError::SentimentMap)?;
    let metadata = (mapping.chunks.max_chunks as u64)
        .checked_mul(INT8_SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES)
        .ok_or(HostedError::Limits("sentiment document metadata arithmetic"))?;
    required.input_bytes = sum(&[required.input_bytes, transport, metadata])?;
    sum(&[required.native.kv_bytes, required.workspace_bytes, required.input_bytes,
        required.output_reservation_bytes])?;
    Ok(required)
}

impl NlpEngine {
    /// Process complete original documents with one resident native engine.
    /// One deadline/checkpoint domain and five-axis work ledger cover the
    /// stream, including failed attempts. Compact preparation never renews its
    /// invocation allowance at a record, chunk, reconstruction or flush.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_sentiment_documents<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SentimentPlanner>, config: SentimentDocumentCorpusConfig,
        limits: CorpusLimits, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, SentimentDocumentStream { planner, config }, limits,
            None, None, reader, writer, cancellation)
    }

    /// Reuse one bounded prompt workspace for every independent affect head.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_sentiment_documents_layer_major<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SentimentPlanner>, config: SentimentDocumentCorpusConfig,
        limits: CorpusLimits, prefill: Int8PrefillLimits,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, SentimentDocumentStream { planner, config }, limits,
            Some(prefill), None, reader, writer, cancellation)
    }

    /// One fixed numerical team spans the entire population. Complete result
    /// guards survive each write/flush; the sole terminal footer follows the
    /// physical worker join under the original invocation's control.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_sentiment_documents_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SentimentPlanner>, config: SentimentDocumentCorpusConfig,
        limits: CorpusLimits, kernel: Int8Kernel, compute_threads: usize,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, SentimentDocumentStream { planner, config }, limits,
            None, Some((kernel, compute_threads)), reader, writer, cancellation)
    }

    /// Compose bounded prompt morsels with the same persistent worker team.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_sentiment_documents_layer_major_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SentimentPlanner>, config: SentimentDocumentCorpusConfig,
        limits: CorpusLimits, kernel: Int8Kernel, compute_threads: usize, prefill: Int8PrefillLimits,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, SentimentDocumentStream { planner, config }, limits,
            Some(prefill), Some((kernel, compute_threads)), reader, writer, cancellation)
    }
}

struct SentimentDocumentStream {
    planner: Arc<SentimentPlanner>,
    config: SentimentDocumentCorpusConfig,
}
impl scored::CorpusTask for SentimentDocumentStream {
    fn identity(&self) -> &ExecutionIdentity { &self.config.identity }
    fn retain_for_join(&self) -> Option<Box<dyn Send>> {
        Some(Box::new((Arc::clone(&self.planner), self.config.defaults.clone())))
    }
    fn validate(&self, _model: &ArtifactIdentity, limits: CorpusLimits, native: Int8MemoryRequirement)
        -> Result<u64, HostedError> {
        self.config.validate(&self.planner).map_err(HostedError::BatchSetup)?;
        let required = int8_sentiment_document_corpus_requirements(self.config.task_ceiling,
            self.config.planning, self.config.mapping, self.planner.document_encoding(), limits)?;
        if required.native != native {
            return Err(HostedError::Limits("sentiment document native allocation differs"));
        }
        // Shared CorpusTask adds transport exactly once; the returned extra
        // contains the complete map frontier and frozen identity metadata.
        let task_bytes = required.input_bytes.checked_sub(limits.reservation_bytes()?)
            .ok_or(HostedError::Limits("sentiment document input arithmetic"))?;
        if self.config.retained_defaults_bytes().map_err(HostedError::BatchSetup)?
            > limits.preparation_reserve_bytes {
            return Err(HostedError::Limits("sentiment defaults preparation envelope"));
        }
        Ok(task_bytes)
    }
    #[allow(clippy::too_many_arguments)]
    fn execute<R: BufRead, W: Write>(self, engine: &mut crate::native_engine::strict_int8::StrictInt8Engine<'_>,
        admission: CorpusAdmission<'_>, prefill: Option<Int8PrefillLimits>,
        reader: &mut R, writer: &mut W, limits: BatchLimits, defer: bool,
        control: &mut RunControl) -> Result<scored::Completion, HostedError> {
        let Self { planner, config } = self;
        let compiler = Int8SentimentDocumentBatchPlanner::new(&planner, config)
            .map_err(HostedError::BatchSetup)?;
        let mut processor = match prefill {
            None => NativeInt8SentimentDocumentBatch::new(compiler, engine, admission),
            Some(prefill) => NativeInt8SentimentDocumentBatch::new_layer_major(compiler, engine, admission, prefill),
        }.map_err(HostedError::BatchSetup)?;
        scored::execute_stream(reader, writer, &mut processor, limits, defer, control)
    }
}

impl Int8SentimentDocumentBatchAdmission for CorpusAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int8SentimentDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Pending), BatchItemFailure> {
        validate_admission(self.model, self.kv_bytes, self.output_bytes, &request)?;
        let mut identities = Vec::new();
        identities.try_reserve_exact(request.identities.len())
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
        identities.extend(request.identities.iter().cloned());
        // Scores may use any explicitly declared finite-language probability
        // space. Projected rows are never mistaken for generated token counts.
        let guard = output_claim(self.lease, request.max_result_bytes, 0).map_err(admission_failure)?;
        Ok((identities, guard))
    }
}
fn validate_admission(model: &ArtifactIdentity, kv_bytes: u64, output_bytes: u64,
    request: &Int8SentimentDocumentAdmission<'_>) -> Result<(), BatchItemFailure> {
    if !(1..=256).contains(&request.identities.len()) || !(1..=4).contains(&request.head_count)
        || request.required_context == 0 || request.kv_reservation_bytes == 0
        || request.kv_reservation_bytes != kv_bytes
        || (request.required_context as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
            .is_none_or(|required| required > kv_bytes)
        || kv_bytes.checked_mul(request.head_count as u64)
            .is_none_or(|required| required > request.task_budget.max_kv_bytes) {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    request.task_budget.validate().map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    validate_work(request.model_work).map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    if request.max_result_bytes == 0 || request.max_result_bytes > output_bytes {
        return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit));
    }
    for identity in request.identities {
        if identity.task_spec != "sentiment-v1" {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        constrained_int8::check_profile(identity)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        check_model_identity(model, identity)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
