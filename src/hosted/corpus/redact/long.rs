//! Long-document redaction streams retain one runtime, native engine and key scope.
use super::*;
use crate::tasks::redact::{
    corpus::{self as document_batch, LongRedactionBatchConfig, NativeInt8DocumentRedactionBatch},
    long::LongRedactionConfig,
};

impl NlpEngine {
    /// Each original retains one compact operation through preflight, execution
    /// and fresh transformed-source verification; flushes never renew work.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_redact_document<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig<LongRedactionBatchConfig>, pseudonyms: Option<RedactionPseudonyms>,
        limits: CorpusLimits, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        team::run(self, model, planner, vocabulary, config, pseudonyms, limits,
            None, None, reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_redact_document_layer_major<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig<LongRedactionBatchConfig>, pseudonyms: Option<RedactionPseudonyms>,
        limits: CorpusLimits, prefill: Int8PrefillLimits, reader: R, writer: W,
        cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        team::run(self, model, planner, vocabulary, config, pseudonyms, limits,
            Some(prefill), None, reader, writer, cancellation)
    }

    /// The same team spans every chunk of every original and edited document.
    /// Native buffers, key, planner and IO ownership survive physical join.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_redact_document_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig<LongRedactionBatchConfig>, pseudonyms: Option<RedactionPseudonyms>,
        limits: CorpusLimits, kernel: Int8Kernel, compute_threads: usize,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        team::run(self, model, planner, vocabulary, config, pseudonyms, limits,
            None, Some((kernel, compute_threads)), reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_redact_document_layer_major_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig<LongRedactionBatchConfig>, pseudonyms: Option<RedactionPseudonyms>,
        limits: CorpusLimits, kernel: Int8Kernel, compute_threads: usize, prefill: Int8PrefillLimits,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        team::run(self, model, planner, vocabulary, config, pseudonyms, limits,
            Some(prefill), Some((kernel, compute_threads)), reader, writer, cancellation)
    }
}

impl team::Policy for RedactionCorpusConfig<LongRedactionBatchConfig> {
    fn identity(&self) -> &ExecutionIdentity { &self.batch.ner_identity }
    fn request(&self) -> &RedactionRequest { &self.batch.request }
    fn validate(&self, planner: &SourceTaskPlanner, limits: CorpusLimits,
        native: Int8MemoryRequirement) -> Result<u64, HostedError> {
        crate::hosted::redact::check_encoding_preparation(planner, limits.preparation_reserve_bytes)?;
        document_batch::check_configuration(planner, &self.batch).map_err(HostedError::BatchSetup)?;
        validate(self, limits, native.kv_bytes)?;
        temporary_bytes(&self.batch.detector, self.edit_reserve_bytes)
    }
    #[allow(clippy::too_many_arguments)]
    fn execute<R: BufRead, W: Write>(self, planner: &SourceTaskPlanner,
        vocabulary: &ExtractionVocabulary, pseudonyms: Option<&Pseudonyms<'_>>,
        engine: &mut StrictInt8Engine<'_>, admission: CorpusAdmission<'_>, prefill: Option<Int8PrefillLimits>,
        reader: &mut R, writer: &mut W, limits: BatchLimits, defer: bool, control: &mut RunControl)
        -> Result<super::super::scored::Completion, HostedError> {
        let mut processor = match prefill {
            None => NativeInt8DocumentRedactionBatch::new(planner, self.batch, engine, vocabulary, pseudonyms, admission),
            Some(prefill) => NativeInt8DocumentRedactionBatch::new_layer_major(
                planner, self.batch, engine, vocabulary, pseudonyms, admission, prefill),
        }.map_err(HostedError::BatchSetup)?;
        super::super::scored::execute_stream(reader, writer, &mut processor, limits, defer, control)
    }
}
fn validate(config: &RedactionCorpusConfig<LongRedactionBatchConfig>, limits: CorpusLimits, kv: u64)
    -> Result<(), HostedError> {
    let d = &config.batch.detector;
    if config.edit_reserve_bytes == 0 || kv == 0 || kv > d.per_chunk.max_kv_bytes
        || d.planning.max_context_tokens > limits.native.context_tokens
        || d.mapping.chunks.context_tokens > limits.native.context_tokens
        || d.max_result_bytes == 0 || d.max_result_bytes > limits.transport.max_output_line_bytes as u64 {
        return Err(HostedError::Limits("document redaction corpus context, output or editing reservation"));
    }
    Ok(())
}
pub(super) fn temporary_bytes(detector: &LongRedactionConfig, edit: u64) -> Result<u64, HostedError> {
    let count = detector.mapping.chunks.max_chunks;
    if edit == 0 || !(1..=256).contains(&count) || detector.mapping.reduction.max_live_value_bytes == 0 {
        return Err(HostedError::Limits("document redaction corpus finite map frontier"));
    }
    // One map at peak, not one result and not two simultaneous maps. All chunk
    // outputs/coordinate lifts and reduction staging survive until the stage
    // has been independently verified. The edited document has its own guard.
    let native = detector.per_chunk.max_output_bytes.checked_mul(count as u64)
        .ok_or(HostedError::Limits("document redaction corpus native intermediates"))?;
    let staged = sum(&[native, detector.mapping.reduction.max_live_value_bytes as u64])?
        .checked_mul(4).ok_or(HostedError::Limits("document redaction corpus map staging"))?;
    let tokens = u64::from(detector.per_chunk.max_output_tokens).checked_mul(count as u64)
        .and_then(|n| n.checked_mul(8)).ok_or(HostedError::Limits("document redaction corpus token storage"))?;
    sum(&[staged, tokens, edit])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{grammar::mask::MaskWorkLimits, tasks::{ir::TaskBudget, ner::NerOptions,
        mapreduce::{ChunkLimits, ExecutionLimits}, source_planning::{SourcePlanningLimits,
            quantized::long::Int8SourceMapLimits}}};
    fn detector() -> LongRedactionConfig {
        LongRedactionConfig { ner: NerOptions::default(), per_chunk: TaskBudget {
            max_input_tokens: 8192, max_output_tokens: 16, max_output_bytes: 1024,
            max_grammar_states: 4096, max_kv_bytes: 1 << 31 }, planning: SourcePlanningLimits::default(),
            mapping: Int8SourceMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 16,
                max_chunk_tokens: 16, context_tokens: 8192, reserved_tokens: 16, max_chunks: 8, max_tokenizer_calls: 1024 },
                reduction: ExecutionLimits::default(), max_model_work: Int8Work::default(),
                mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000, max_mask_visits: 16000 },
            max_result_bytes: 4096 }
    }
    #[test]
    fn reservation_prices_all_chunk_outputs_and_the_live_map_frontier() {
        let d = detector(); let edit = 4096;
        let expected = (8 * 1024 + d.mapping.reduction.max_live_value_bytes as u64) * 4 + 8 * 16 * 8 + edit;
        assert_eq!(temporary_bytes(&d, edit).unwrap(), expected);
        assert!(expected > d.max_result_bytes);
    }
    #[test]
    fn map_and_token_arithmetic_never_saturates_or_wraps() {
        let mut d = detector(); d.per_chunk.max_output_bytes = u64::MAX;
        assert!(temporary_bytes(&d, 1).is_err());
        let mut d = detector(); d.mapping.reduction.max_live_value_bytes = usize::MAX;
        assert!(temporary_bytes(&d, u64::MAX).is_err());
    }
    #[test]
    fn compact_factory_workspace_is_owned_before_stream_admission_without_spending_document_work() {
        use crate::tokenizer::{embedded::EmbeddedTokenizer, pinned_controls,
            untrusted::{SafeBpeLimits, SafeBpeWork, SafeBpeWorkLimits, UntrustedEncoding}};
        let tokenizer = EmbeddedTokenizer::pinned().unwrap();
        let controls = pinned_controls::pinned().unwrap();
        let factory = || SourceTaskPlanner::pinned(controls.template_controls(), tokenizer.eos_token_id().unwrap()).unwrap();
        let policy = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 128, max_output_tokens: 128,
            max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 });
        let unbounded = factory().with_document_encoding(policy.clone()).unwrap();
        assert!(crate::hosted::redact::check_encoding_preparation(&unbounded, u64::MAX).is_err());
        let planner = factory().with_map_document_encoding(policy, SafeBpeWorkLimits {
            max_input_bytes: 16384, max_output_tokens: 16384, max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 }).unwrap();
        let workspace = planner.map_document_encoding_workspace_bytes().unwrap();
        assert!(workspace > 1024);
        assert!(crate::hosted::redact::check_encoding_preparation(&planner, workspace - 1).is_err());
        crate::hosted::redact::check_encoding_preparation(&planner, workspace).unwrap();
        let mut limits = CorpusLimits { native: NativeLimits { context_tokens: 8192, allocator_reserve_bytes: 4096,
            run: RunLimits { max_elapsed: std::time::Duration::from_secs(60), max_checkpoints: 100000,
                cleanup_reserve_bytes: 4096 } },
            transport: BatchLimits::default(), preparation_reserve_bytes: 1, io_reserve_bytes: 4096 };
        let before = limits.reservation_bytes().unwrap();
        limits.preparation_reserve_bytes = workspace;
        assert_eq!(limits.reservation_bytes().unwrap() - before, workspace - 1);
        assert_eq!(planner.map_document_encoding_work().unwrap(), Some(SafeBpeWork::default()));
        let _entry = NlpEngine::batch_int8_redact_document::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    }
    #[test]
    fn unbounded_or_zero_frontiers_are_refused() {
        let mut d = detector(); assert!(temporary_bytes(&d, 0).is_err());
        d.mapping.chunks.max_chunks = 257; assert!(temporary_bytes(&d, 1).is_err());
        d.mapping.chunks.max_chunks = 0; assert!(temporary_bytes(&d, 1).is_err());
        d.mapping.chunks.max_chunks = 8; d.mapping.reduction.max_live_value_bytes = 0;
        assert!(temporary_bytes(&d, 1).is_err());
    }
}
