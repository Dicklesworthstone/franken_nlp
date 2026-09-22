//! One process-owned invocation for an entire INT8 redaction stream.
//! Reuse corpus IO ownership, resident weights, native scratch and admission.
use super::*;
use crate::tasks::{
    source_planning::SourceTaskPlanner,
    redact::{RedactError, RedactionRequest, pseudonym::Pseudonyms,
        batch::{self as redaction_batch, Int8RedactionBatchConfig, NativeInt8RedactionBatch}},
};

/// Explicit item policy plus whole-stream work ceilings. Preparation and IO
/// storage are priced by CorpusLimits; rule/union/edit allocation headroom is
/// separate. No field is a measured RSS guarantee or a deserialized permit.
pub struct RedactionCorpusConfig {
    pub batch: Int8RedactionBatchConfig,
    pub edit_reserve_bytes: u64,
}

impl NlpEngine {
    /// Stream original UTF-8 documents through native NER, selected rules,
    /// source-occurrence verification, transactional edits and optional fresh
    /// NER/rule verification. One immutable policy and pseudonym scope survives
    /// every document and flush. Input task_args cannot change that policy.
    ///
    /// The host owns one runtime invocation and one native allocation for the
    /// whole stream. No per-document host call, model reload, new scheduler,
    /// rules-only fallback, durable resume or public CLI activation is implied.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_redact<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig, pseudonyms: Option<RedactionPseudonyms>,
        limits: CorpusLimits, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.batch.ner_identity)?;
        redaction_batch::check_configuration(&planner, &config.batch).map_err(HostedError::BatchSetup)?;
        let required = requirements(limits.native)?;
        validate_limits(&config, limits, required.kv_bytes)?;
        let buffers = sum(&[limits.reservation_bytes()?, temporary_bytes(&config)?, secret_bytes(pseudonyms.as_ref())?])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, buffers)?,
            || Ok(StreamInput { planner: Some((planner, vocabulary, config, pseudonyms)), reader, writer }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            // Capture the complete storage-before-charge aggregate, including
            // the key handle and owned IO, even if queued work is discarded.
            let mut input = input;
            let (planner, vocabulary, config, secret) = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            // Construct ONE full256 scope and validate saved commitments before
            // native allocation or reading/writing any corpus record.
            let context = pseudonym_context(secret.as_ref(), &config.batch.request)?;
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let admission = CorpusAdmission { lease: &lease, model: model.artifact_identity(),
                kv_bytes: required.kv_bytes, sampler_bytes: 0,
                output_bytes: limits.transport.max_output_line_bytes as u64 };
            let result = {
                let mut processor = NativeInt8RedactionBatch::new(&planner, config.batch,
                    &mut engine.value, &vocabulary, context.as_ref(), admission).map_err(HostedError::BatchSetup)?;
                batch::run_ndjson(&mut input.value.reader, &mut input.value.writer, &mut processor,
                    limits.transport, control).map_err(HostedError::Batch)
            };
            drop(engine);
            drop(context);
            drop(secret);
            drop(vocabulary);
            drop(planner);
            drop(input); // Reader/writer and preparation before the memory charge.
            drop(lease);
            result
        })
    }
}
fn validate_limits(config: &RedactionCorpusConfig, limits: CorpusLimits, kv: u64) -> Result<(), HostedError> {
    if config.edit_reserve_bytes == 0 || kv == 0 || kv > config.batch.detector.per_pass.max_kv_bytes
        || config.batch.detector.planning.max_context_tokens > limits.native.context_tokens
        || config.batch.detector.max_result_bytes == 0
        || config.batch.detector.max_result_bytes > limits.transport.max_output_line_bytes as u64 {
        return Err(HostedError::Limits("redaction corpus context, output or edit reservation"));
    }
    Ok(())
}
fn temporary_bytes(config: &RedactionCorpusConfig) -> Result<u64, HostedError> {
    // The first edited document stays live while fresh NER on that text is
    // checked. Per-item output admission prices that document; this additional
    // commitment prices NER result/canonical/token staging and rule/edit data.
    let output = config.batch.detector.per_pass.max_output_bytes.checked_mul(4)
        .ok_or(HostedError::Limits("redaction corpus intermediate output arithmetic"))?;
    let tokens = u64::from(config.batch.detector.per_pass.max_output_tokens).checked_mul(8)
        .ok_or(HostedError::Limits("redaction corpus intermediate token arithmetic"))?;
    sum(&[output, tokens, config.edit_reserve_bytes])
}
fn secret_bytes(secret: Option<&RedactionPseudonyms>) -> Result<u64, HostedError> {
    match secret {
        None => Ok(0),
        Some(secret) => {
            if secret.namespace.is_empty() || secret.namespace.len() > 256 {
                return Err(HostedError::Redaction(RedactError::InvalidOptions.into()));
            }
            // Price retained capacity, not just length. PseudonymKey bounds its
            // private key block/id; full256 has no retained value dictionary.
            sum(&[secret.namespace.capacity() as u64, 2048])
        }
    }
}
fn pseudonym_context<'a>(secret: Option<&'a RedactionPseudonyms>, request: &RedactionRequest)
    -> Result<Option<Pseudonyms<'a>>, HostedError> {
    let context = secret.map(|s| Pseudonyms::full256(&s.key, &s.namespace,
        request.actions.expected_key_commitment.as_deref())).transpose()
        .map_err(|e| HostedError::Redaction(e.into()))?;
    request.actions.check_key(context.as_ref()).map_err(|e| HostedError::Redaction(e.into()))?;
    Ok(context)
}

#[cfg(test)] mod tests;
