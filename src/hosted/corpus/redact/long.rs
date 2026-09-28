//! Long-document redaction streams retain one runtime, native engine and key scope.
use super::*;
use crate::tasks::redact::{
    corpus::{self as document_batch, LongRedactionBatchConfig, NativeInt8DocumentRedactionBatch},
    long::LongRedactionConfig,
};

impl NlpEngine {
    /// Process bounded NDJSON records whose text need not fit one NER context.
    /// Every item uses whole-source rules and independently verified native NER
    /// maps, then optionally repartitions the actual transformed source. The
    /// complete per-item work ceiling is charged to the corpus before inference.
    /// Flush and failed documents never renew native or mask authority.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_redact_document<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig<LongRedactionBatchConfig>, pseudonyms: Option<RedactionPseudonyms>,
        limits: CorpusLimits, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.batch.ner_identity)?;
        document_batch::check_configuration(&planner, &config.batch).map_err(HostedError::BatchSetup)?;
        let required = requirements(limits.native)?;
        validate(&config, limits, required.kv_bytes)?;
        let buffers = sum(&[limits.reservation_bytes()?,
            temporary_bytes(&config.batch.detector, config.edit_reserve_bytes)?, secret_bytes(pseudonyms.as_ref())?])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, buffers)?,
            || Ok(StreamInput { planner: Some((planner, vocabulary, config, pseudonyms)), reader, writer }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            let (planner, vocabulary, config, secret) = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            // Check one immutable full256 HMAC scope before native allocation
            // or the first read/write. Never re-read a secret at a flush boundary.
            let context = pseudonym_context(secret.as_ref(), &config.batch.request)?;
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let admission = CorpusAdmission { lease: &lease, model: model.artifact_identity(),
                kv_bytes: required.kv_bytes, sampler_bytes: 0,
                output_bytes: limits.transport.max_output_line_bytes as u64 };
            let result = {
                let mut processor = NativeInt8DocumentRedactionBatch::new(&planner, config.batch,
                    &mut engine.value, &vocabulary, context.as_ref(), admission).map_err(HostedError::BatchSetup)?;
                batch::run_ndjson(&mut input.value.reader, &mut input.value.writer, &mut processor,
                    limits.transport, control).map_err(HostedError::Batch)
            };
            drop(engine);
            drop(context);
            drop(secret);
            drop(vocabulary);
            drop(planner);
            drop(input); // Owned IO and compiler storage before the charge/physical handoff.
            drop(lease);
            result
        })
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
fn temporary_bytes(detector: &LongRedactionConfig, edit: u64) -> Result<u64, HostedError> {
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
    fn unbounded_or_zero_frontiers_are_refused() {
        let mut d = detector(); assert!(temporary_bytes(&d, 0).is_err());
        d.mapping.chunks.max_chunks = 257; assert!(temporary_bytes(&d, 1).is_err());
        d.mapping.chunks.max_chunks = 0; assert!(temporary_bytes(&d, 1).is_err());
        d.mapping.chunks.max_chunks = 8; d.mapping.reduction.max_live_value_bytes = 0;
        assert!(temporary_bytes(&d, 1).is_err());
    }
}
