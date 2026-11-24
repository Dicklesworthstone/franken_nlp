//! Process-owned complete-document schema extraction across a bounded corpus.
use super::*;
use crate::{
    batch::extract::quantized::long::{
        object::{Int8ObjectExtractionPlanner, Int8ObjectExtractionMapLimits,
            Int8ObjectExtractionMapError},
        stream::{ExtractionDocumentBatchConfig, ExtractionDocumentAdmission,
            ExtractionDocumentBatchAdmission, NativeInt8DocumentExtractionBatch},
    },
    tasks::extract::object_reconcile::{RootObjectLimits, RootObjectSchema},
};

/// CorpusLimits prices ALL retained per-document schema/source plans and owned
/// IO. This additional headroom covers reduction/allocator metadata beyond the
/// shared explicit map payload calculation, not an observed RSS assertion.
pub struct ExtractionDocumentCorpusConfig {
    pub batch: ExtractionDocumentBatchConfig,
    pub reduction_reserve_bytes: u64,
}
impl ExtractionDocumentCorpusConfig {
    /// Exact task workspace in addition to the corpus's owned IO, caller
    /// preparation, native buffers and output guard. Compact input adds one
    /// transient encoder and its cumulative ledger beside the complete map
    /// frontier; it never discounts original source or schema storage.
    pub fn preparation_workspace_bytes(&self, planner: &Int8ExtractionBatchPlanner)
        -> Result<u64, HostedError> {
        sum(&[task_workspace(self)?, planner.map_document_encoding_workspace_bytes()
            .map_err(HostedError::BatchSetup)?])
    }
}
/// Explicit partial-root-object corpus policy. Every input document uses the
/// unchanged original schema in mapping.batch.request, with one independently
/// validated final object and the complete native chunk/evidence transcript.
pub struct Int8ObjectExtractionDocumentCorpusConfig {
    pub mapping: ExtractionDocumentCorpusConfig,
    pub reconciliation: RootObjectLimits,
}
impl ExtractionDocumentCorpusConfig {
    pub fn with_root_object_reconciliation(self, reconciliation: RootObjectLimits)
        -> Int8ObjectExtractionDocumentCorpusConfig {
        Int8ObjectExtractionDocumentCorpusConfig { mapping: self, reconciliation }
    }
}
impl Int8ObjectExtractionDocumentCorpusConfig {
    pub fn limits(&self) -> Int8ObjectExtractionMapLimits {
        Int8ObjectExtractionMapLimits { mapping: self.mapping.batch.limits,
            reconciliation: self.reconciliation }
    }
    /// Price the complete map, both original/partial declarations and exact
    /// reconciliation together. Compact encoding retains one invocation ledger.
    pub fn preparation_workspace_bytes(&self, planner: &Int8ObjectExtractionPlanner)
        -> Result<u64, HostedError> {
        sum(&[self.mapping.preparation_workspace_bytes(planner.base_planner())?,
            object_workspace(&self.mapping, self.reconciliation, planner.compiler_limits())?])
    }
}
fn object_workspace(config: &ExtractionDocumentCorpusConfig, reconciliation: RootObjectLimits,
    compiler: crate::grammar::CompileLimits) -> Result<u64, HostedError> {
    sum(&[Int8ObjectExtractionMapLimits { mapping: config.batch.limits, reconciliation }
        .reconciliation_workspace_bytes().map_err(HostedError::Int8ObjectExtractionMap)?,
        RootObjectSchema::workspace_bytes(compiler).map_err(|error|
            HostedError::Int8ObjectExtractionMap(Int8ObjectExtractionMapError::from(error)))?])
}

impl NlpEngine {
    /// Stream original documents larger than one context using one native
    /// engine and one invocation. Successful rows contain every chunk value
    /// and its exact original-coordinate source evidence.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_extract_document<R, W>(&self, model: &ResidentInt8,
        planner: Arc<Int8ExtractionBatchPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: ExtractionDocumentCorpusConfig, limits: CorpusLimits, reader: R, writer: W,
        cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, ExtractionDocumentStream { planner: DocumentPlanner::Map(planner), vocabulary, config }, limits,
            None, None, reader, writer, cancellation)
    }

    /// Reuse bounded prompt storage across all chunks of every document,
    /// preserving source reconstruction, identity binding and exact finalizers.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_extract_document_layer_major<R, W>(&self, model: &ResidentInt8,
        planner: Arc<Int8ExtractionBatchPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: ExtractionDocumentCorpusConfig, limits: CorpusLimits,
        prefill: crate::native_engine::strict_int8::prefill::Int8PrefillLimits,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, ExtractionDocumentStream { planner: DocumentPlanner::Map(planner), vocabulary, config }, limits,
            Some(prefill), None, reader, writer, cancellation)
    }

    /// One persistent team spans every immutable chunk and complete result.
    /// The terminal frame follows physical join with native/IO claims retained.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_extract_document_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Arc<Int8ExtractionBatchPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: ExtractionDocumentCorpusConfig, limits: CorpusLimits,
        kernel: crate::native_engine::simd::Int8Kernel, compute_threads: usize,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, ExtractionDocumentStream { planner: DocumentPlanner::Map(planner), vocabulary, config }, limits,
            None, Some((kernel, compute_threads)), reader, writer, cancellation)
    }

    /// Compose prefill and workers without renewing model, mask or source
    /// verification allowances at any chunk, document or flush boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_extract_document_layer_major_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Arc<Int8ExtractionBatchPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: ExtractionDocumentCorpusConfig, limits: CorpusLimits,
        kernel: crate::native_engine::simd::Int8Kernel, compute_threads: usize,
        prefill: crate::native_engine::strict_int8::prefill::Int8PrefillLimits,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, ExtractionDocumentStream { planner: DocumentPlanner::Map(planner), vocabulary, config }, limits,
            Some(prefill), Some((kernel, compute_threads)), reader, writer, cancellation)
    }
}

impl NlpEngine {
    /// Reconcile complementary root properties across every complete document
    /// in one resident stream. Conflicts or missing required fields fail that
    /// document; equal exact values retain all original chunk origins.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_extract_object_document<R, W>(&self, model: &ResidentInt8,
        planner: Arc<Int8ObjectExtractionPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int8ObjectExtractionDocumentCorpusConfig, limits: CorpusLimits, reader: R, writer: W,
        cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, ExtractionDocumentStream::object(planner, vocabulary, config), limits,
            None, None, reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_extract_object_document_layer_major<R, W>(&self, model: &ResidentInt8,
        planner: Arc<Int8ObjectExtractionPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int8ObjectExtractionDocumentCorpusConfig, limits: CorpusLimits,
        prefill: crate::native_engine::strict_int8::prefill::Int8PrefillLimits,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, ExtractionDocumentStream::object(planner, vocabulary, config), limits,
            Some(prefill), None, reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_extract_object_document_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Arc<Int8ObjectExtractionPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int8ObjectExtractionDocumentCorpusConfig, limits: CorpusLimits,
        kernel: crate::native_engine::simd::Int8Kernel, compute_threads: usize,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, ExtractionDocumentStream::object(planner, vocabulary, config), limits,
            None, Some((kernel, compute_threads)), reader, writer, cancellation)
    }

    /// One persistent team and bounded prompt workspace cover the full corpus.
    /// Model, mask, source-verification and tokenizer work never renew at a
    /// chunk, document, reconciliation or flush boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_extract_object_document_layer_major_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Arc<Int8ObjectExtractionPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int8ObjectExtractionDocumentCorpusConfig, limits: CorpusLimits,
        kernel: crate::native_engine::simd::Int8Kernel, compute_threads: usize,
        prefill: crate::native_engine::strict_int8::prefill::Int8PrefillLimits,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, ExtractionDocumentStream::object(planner, vocabulary, config), limits,
            Some(prefill), Some((kernel, compute_threads)), reader, writer, cancellation)
    }
}

enum DocumentPlanner {
    Map(Arc<Int8ExtractionBatchPlanner>),
    Object { planner: Arc<Int8ObjectExtractionPlanner>, reconciliation: RootObjectLimits },
}
impl DocumentPlanner {
    fn base_planner(&self) -> &Int8ExtractionBatchPlanner { match self {
        Self::Map(planner) => planner,
        Self::Object { planner, .. } => planner.base_planner(),
    } }
}
struct ExtractionDocumentStream {
    planner: DocumentPlanner, vocabulary: Arc<ExtractionVocabulary>,
    config: ExtractionDocumentCorpusConfig,
}
impl ExtractionDocumentStream {
    fn object(planner: Arc<Int8ObjectExtractionPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int8ObjectExtractionDocumentCorpusConfig) -> Self {
        Self { planner: DocumentPlanner::Object { planner, reconciliation: config.reconciliation },
            vocabulary, config: config.mapping }
    }
}
impl scored::CorpusTask for ExtractionDocumentStream {
    fn identity(&self) -> &ExecutionIdentity { self.planner.base_planner().base_execution_identity() }
    fn validate(&self, _model: &ArtifactIdentity, limits: CorpusLimits,
        native: Int8MemoryRequirement) -> Result<u64, HostedError> {
        self.config.batch.validate().map_err(HostedError::BatchSetup)?;
        validate(&self.config, limits, native.kv_bytes)?;
        let mapping = self.config.preparation_workspace_bytes(self.planner.base_planner())?;
        match &self.planner {
            DocumentPlanner::Map(_) => Ok(mapping),
            DocumentPlanner::Object { planner, reconciliation } => sum(&[
                mapping, object_workspace(&self.config, *reconciliation, planner.compiler_limits())?]),
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn execute<R: BufRead, W: Write>(self, engine: &mut crate::native_engine::strict_int8::StrictInt8Engine<'_>,
        admission: CorpusAdmission<'_>, prefill: Option<crate::native_engine::strict_int8::prefill::Int8PrefillLimits>,
        reader: &mut R, writer: &mut W, limits: BatchLimits, defer: bool,
        control: &mut RunControl) -> Result<scored::Completion, HostedError> {
        let Self { planner, vocabulary, config } = self;
        let mut processor = match &planner {
            DocumentPlanner::Map(planner) => match prefill {
                Some(prefill) => NativeInt8DocumentExtractionBatch::new_layer_major(planner, config.batch,
                    engine, &vocabulary, admission, prefill),
                None => NativeInt8DocumentExtractionBatch::new(planner, config.batch,
                    engine, &vocabulary, admission),
            },
            DocumentPlanner::Object { planner, reconciliation } =>
                NativeInt8DocumentExtractionBatch::new_object_scheduled(planner, *reconciliation,
                    config.batch, engine, &vocabulary, admission, prefill),
        }.map_err(HostedError::BatchSetup)?;
        scored::execute_stream(reader, writer, &mut processor, limits, defer, control)
    }
}
fn task_workspace(config: &ExtractionDocumentCorpusConfig) -> Result<u64, HostedError> {
    sum(&[config.batch.request.schema.capacity() as u64,
        super::super::extraction_map::document_workspace_bytes(config.batch.request.budget,
            config.batch.limits, config.batch.limits.mapping.chunks.max_chunks, config.reduction_reserve_bytes)?])
}

fn validate(config: &ExtractionDocumentCorpusConfig, limits: CorpusLimits, kv: u64) -> Result<(), HostedError> {
    let b = &config.batch;
    if config.reduction_reserve_bytes == 0 || kv == 0 || kv > b.request.budget.max_kv_bytes
        || b.limits.mapping.chunks.context_tokens != limits.native.context_tokens
        || b.limits.mapping.reduction.max_result_bytes == 0
        || b.limits.mapping.reduction.max_result_bytes > limits.transport.max_output_line_bytes {
        return Err(HostedError::Limits("document extraction corpus context, output or reduction reservation"));
    }
    Ok(())
}

impl ExtractionDocumentBatchAdmission for CorpusAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: ExtractionDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Pending), BatchItemFailure> {
        check_document_request(self.model, self.kv_bytes, self.output_bytes, &request)?;
        let mut identities = Vec::new();
        identities.try_reserve_exact(request.identities.len()).map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
        identities.extend(request.identities.iter().cloned());
        let tokens = request.model_work.projected_logits / NANBEIGE_VOCAB_SIZE as u64;
        // ONE output charge for the whole document, retained through the NDJSON
        // runner's canonical staging, write and flush. Never charge just chunk 0.
        let guard = output_claim(self.lease, request.max_result_bytes, tokens).map_err(admission_failure)?;
        Ok((identities, guard))
    }
}
fn check_document_request(model: &ArtifactIdentity, kv: u64, output: u64,
    request: &ExtractionDocumentAdmission<'_>) -> Result<(), BatchItemFailure> {
    if !(1..=256).contains(&request.identities.len()) || request.mask_node_visits == 0
        || request.mask_limits.max_trie_node_visits == 0 || request.mask_limits.checkpoint_interval_nodes == 0 {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    for identity in request.identities {
        identity.validate().map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        if identity.task_spec != "extract-v1" { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        check_request(model, kv, 0, output, identity, request.model_work,
            request.kv_reservation_bytes, 0, request.max_result_bytes)?;
    }
    Ok(())
}

#[cfg(test)] mod tests;
