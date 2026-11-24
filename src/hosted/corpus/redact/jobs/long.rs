//! Retain one complete long-document edit, never individual NER chunks.
use super::*;
use crate::tasks::redact::corpus::{self as document_batch, LongRedactionBatchConfig, NativeInt8DocumentRedactionBatch};

impl<A: Int8RedactionBatchAdmission> Native for NativeInt8DocumentRedactionBatch<'_, '_, '_, '_, '_, '_, A> {
    fn poisoned(&self) -> bool { self.is_poisoned() }
    fn check_prepared(&self, plan: &Self::Prepared) -> Result<(), BatchItemFailure> {
        NativeInt8DocumentRedactionBatch::check_prepared(self, plan)
    }
}
impl NlpEngine {
    /// Durable whole-document edits preserve exact source, partition, key and
    /// compact-input authority before any repair or model execution.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_redact_document<R>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig<LongRedactionBatchConfig>, pseudonyms: Option<RedactionPseudonyms>,
        request: SourceJobRequest, limits: JobHostLimits, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        team::run(self, model, planner, vocabulary, config, pseudonyms, request,
            limits, None, None, reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_redact_document_layer_major<R>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig<LongRedactionBatchConfig>, pseudonyms: Option<RedactionPseudonyms>,
        request: SourceJobRequest, limits: JobHostLimits, prefill: Int8PrefillLimits,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        team::run(self, model, planner, vocabulary, config, pseudonyms, request,
            limits, Some(prefill), None, reader, cancellation)
    }

    /// One fixed team covers all pending documents and both native maps.
    /// Changing the physical schedule on resume never changes saved policy.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_redact_document_with_threads<R>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig<LongRedactionBatchConfig>, pseudonyms: Option<RedactionPseudonyms>,
        request: SourceJobRequest, limits: JobHostLimits, kernel: Int8Kernel,
        compute_threads: usize, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        team::run(self, model, planner, vocabulary, config, pseudonyms, request,
            limits, None, Some((kernel, compute_threads)), reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_redact_document_layer_major_with_threads<R>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig<LongRedactionBatchConfig>, pseudonyms: Option<RedactionPseudonyms>,
        request: SourceJobRequest, limits: JobHostLimits, kernel: Int8Kernel,
        compute_threads: usize, prefill: Int8PrefillLimits, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        team::run(self, model, planner, vocabulary, config, pseudonyms, request,
            limits, Some(prefill), Some((kernel, compute_threads)), reader, cancellation)
    }
}

impl team::Policy for RedactionCorpusConfig<LongRedactionBatchConfig> {
    fn identity(&self) -> &ExecutionIdentity { &self.batch.ner_identity }
    fn request(&self) -> &RedactionRequest { &self.batch.request }
    fn validate(&self, planner: &SourceTaskPlanner, request: &SourceJobRequest,
        limits: JobHostLimits, native: Int8MemoryRequirement) -> Result<u64, HostedError> {
        recipe::bounded(&self.batch.request)?; recipe::bounded(&self.batch.detector.ner)?;
        crate::hosted::redact::check_encoding_preparation(planner, limits.preparation_reserve_bytes)?;
        document_batch::check_configuration(planner, &self.batch).map_err(HostedError::BatchSetup)?;
        let d = &self.batch.detector;
        capacity(d.planning.max_context_tokens.max(d.mapping.chunks.context_tokens), d.per_chunk.max_kv_bytes,
            d.max_result_bytes, self.edit_reserve_bytes, limits.native.context_tokens,
            native.kv_bytes, request.limits.max_result_bytes)?;
        super::super::long::temporary_bytes(d, self.edit_reserve_bytes)
    }
    fn recipe(&self, planner: &SourceTaskPlanner, context: Option<&Pseudonyms<'_>>)
        -> Result<RedactionJobRecipe, HostedError> {
        RedactionJobRecipe::long(&self.batch, context, planner)
    }
    #[allow(clippy::too_many_arguments)]
    fn execute<C: DecodeStepControl>(self, planner: &SourceTaskPlanner,
        vocabulary: &ExtractionVocabulary, context: Option<&Pseudonyms<'_>>,
        engine: &mut StrictInt8Engine<'_>, admission: CorpusAdmission<'_>, prefill: Option<Int8PrefillLimits>,
        request: SourceJobRequest, population: &crate::hosted::corpus::jobs::Population<'_>,
        recipe: RedactionJobRecipe, control: &mut C) -> Result<JobProgress, JobRunError> {
        let identity = self.batch.ner_identity.clone();
        let native = match prefill {
            None => NativeInt8DocumentRedactionBatch::new(planner, self.batch, engine, vocabulary, context, admission),
            Some(prefill) => NativeInt8DocumentRedactionBatch::new_layer_major(
                planner, self.batch, engine, vocabulary, context, admission, prefill),
        }.map_err(JobRunError::Processor)?;
        run_population(request, population, RedactionJob { native, identity, recipe }, control)
    }
}
