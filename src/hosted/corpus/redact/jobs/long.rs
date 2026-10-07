//! Retain one complete long-document edit, never individual NER chunks.
use super::*;
use crate::tasks::redact::corpus::{self as document_batch, LongRedactionBatchConfig, NativeInt8DocumentRedactionBatch};

impl<A: Int8RedactionBatchAdmission> Native for NativeInt8DocumentRedactionBatch<'_, '_, '_, '_, '_, '_, A> {
    fn poisoned(&self) -> bool { self.is_poisoned() }
}
impl NlpEngine {
    /// Durable long-document redaction. Rules still scan whole text, native NER
    /// uses original-source-aligned chunks, and verification re-chunks the ACTUAL
    /// edited text. Resume authenticates the complete partition/reduction recipe,
    /// detector policy and key scope before any repair or model forward.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_redact_document<R>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig<LongRedactionBatchConfig>, pseudonyms: Option<RedactionPseudonyms>,
        request: SourceJobRequest, limits: JobHostLimits, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.batch.ner_identity)?;
        recipe::bounded(&config.batch.request)?; recipe::bounded(&config.batch.detector.ner)?;
        document_batch::check_configuration(&planner, &config.batch).map_err(HostedError::BatchSetup)?;
        let required = requirements(limits.native)?;
        let d = &config.batch.detector;
        capacity(d.planning.max_context_tokens.max(d.mapping.chunks.context_tokens), d.per_chunk.max_kv_bytes,
            d.max_result_bytes, config.edit_reserve_bytes, limits.native.context_tokens,
            required.kv_bytes, request.limits.max_result_bytes)?;
        let buffers = sum(&[limits.required_buffer_bytes(request.limits)?,
            super::super::long::temporary_bytes(d, config.edit_reserve_bytes)?,
            secret_bytes(pseudonyms.as_ref())?, 4 * RECIPE_BYTES as u64])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, buffers)?,
            || Ok(StreamInput { planner: Some((planner, vocabulary, config, pseudonyms, request)), reader, writer: () }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        let completed = dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            let (planner, vocabulary, config, secret, request) = input.value.planner.take()
                .ok_or(HostedError::CompletionMissing)?;
            let context = pseudonym_context(secret.as_ref(), &config.batch.request)?;
            let identity = config.batch.ner_identity.clone();
            let recipe = RedactionJobRecipe::long(&config.batch, context.as_ref())?;
            let population = match JobPopulation::read_ndjson(&mut input.value.reader, &request.key,
                request.job_id, request.limits, limits.transport, control) {
                Ok(p) => p, Err(e) => return Ok(Err(JobRunError::Storage(e))),
            };
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let admission = CorpusAdmission { lease: &lease, model: model.artifact_identity(),
                kv_bytes: required.kv_bytes, sampler_bytes: 0, output_bytes: request.limits.max_result_bytes as u64 };
            let result = (|| {
                let native = NativeInt8DocumentRedactionBatch::new(&planner, config.batch,
                    &mut engine.value, &vocabulary, context.as_ref(), admission).map_err(JobRunError::Processor)?;
                run_population(request, &population, RedactionJob { native, identity, recipe }, control)
            })();
            drop(engine); drop(population); drop(context); drop(secret);
            drop(vocabulary); drop(planner); drop(input); drop(lease);
            Ok(result)
        })?;
        completed.map_err(HostedJobError::Job)
    }
}
