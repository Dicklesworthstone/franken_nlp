//! One process-owned invocation for an entire INT8 redaction stream.
//! Reuse corpus IO ownership, resident weights, native scratch and admission.
use super::*;
mod long;
mod team;
use crate::native_engine::{simd::Int8Kernel, strict_int8::{prefill::Int8PrefillLimits, StrictInt8Engine}};
#[cfg(all(feature = "metadata-store", target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
mod jobs;
use crate::tasks::{
    source_planning::SourceTaskPlanner,
    redact::{RedactError, RedactionRequest, pseudonym::Pseudonyms,
        batch::{self as redaction_batch, Int8RedactionBatchConfig, NativeInt8RedactionBatch}},
};

/// Explicit item policy plus whole-stream work ceilings. Preparation and IO
/// storage are priced by CorpusLimits; rule/union/edit allocation headroom is
/// separate. The default keeps short documents; LongRedactionBatchConfig selects
/// the concrete chunked-document entry point. Neither is a deserialized permit.
pub struct RedactionCorpusConfig<B = Int8RedactionBatchConfig> {
    pub batch: B,
    pub edit_reserve_bytes: u64,
}

impl NlpEngine {
    /// Stream complete edits under one fixed detector/key policy, native
    /// engine and nonrenewable whole-stream ledger. Retention remains explicit.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_redact<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig, pseudonyms: Option<RedactionPseudonyms>,
        limits: CorpusLimits, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        team::run(self, model, planner, vocabulary, config, pseudonyms, limits,
            None, None, reader, writer, cancellation)
    }

    /// Bound prompt morsels without renewing per-item or whole-stream work.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_redact_layer_major<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig, pseudonyms: Option<RedactionPseudonyms>,
        limits: CorpusLimits, prefill: Int8PrefillLimits, reader: R, writer: W,
        cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        team::run(self, model, planner, vocabulary, config, pseudonyms, limits,
            Some(prefill), None, reader, writer, cancellation)
    }

    /// One fixed team spans every original, verification pass and flush. The
    /// terminal frame follows physical join under the original run authority.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_redact_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig, pseudonyms: Option<RedactionPseudonyms>,
        limits: CorpusLimits, kernel: Int8Kernel, compute_threads: usize,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        team::run(self, model, planner, vocabulary, config, pseudonyms, limits,
            None, Some((kernel, compute_threads)), reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_redact_layer_major_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig, pseudonyms: Option<RedactionPseudonyms>,
        limits: CorpusLimits, kernel: Int8Kernel, compute_threads: usize, prefill: Int8PrefillLimits,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        team::run(self, model, planner, vocabulary, config, pseudonyms, limits,
            Some(prefill), Some((kernel, compute_threads)), reader, writer, cancellation)
    }
}

impl team::Policy for RedactionCorpusConfig {
    fn identity(&self) -> &ExecutionIdentity { &self.batch.ner_identity }
    fn request(&self) -> &RedactionRequest { &self.batch.request }
    fn validate(&self, planner: &SourceTaskPlanner, limits: CorpusLimits,
        native: Int8MemoryRequirement) -> Result<u64, HostedError> {
        redaction_batch::check_configuration(planner, &self.batch).map_err(HostedError::BatchSetup)?;
        validate_limits(self, limits, native.kv_bytes)?;
        temporary_bytes(self)
    }
    #[allow(clippy::too_many_arguments)]
    fn execute<R: BufRead, W: Write>(self, planner: &SourceTaskPlanner,
        vocabulary: &ExtractionVocabulary, pseudonyms: Option<&Pseudonyms<'_>>,
        engine: &mut StrictInt8Engine<'_>, admission: CorpusAdmission<'_>, prefill: Option<Int8PrefillLimits>,
        reader: &mut R, writer: &mut W, limits: BatchLimits, defer: bool, control: &mut RunControl)
        -> Result<super::scored::Completion, HostedError> {
        let mut processor = match prefill {
            None => NativeInt8RedactionBatch::new(planner, self.batch, engine, vocabulary, pseudonyms, admission),
            Some(prefill) => NativeInt8RedactionBatch::new_layer_major(
                planner, self.batch, engine, vocabulary, pseudonyms, admission, prefill),
        }.map_err(HostedError::BatchSetup)?;
        super::scored::execute_stream(reader, writer, &mut processor, limits, defer, control)
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
