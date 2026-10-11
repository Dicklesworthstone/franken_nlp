//! Process-owned complete-document schema extraction across a bounded corpus.
use super::*;
use crate::batch::extract::quantized::long::stream::{
    ExtractionDocumentBatchConfig, ExtractionDocumentAdmission,
    ExtractionDocumentBatchAdmission, NativeInt8DocumentExtractionBatch,
};

/// CorpusLimits prices ALL retained per-document schema/source plans and owned
/// IO. This additional headroom covers reduction/allocator metadata beyond the
/// shared explicit map payload calculation, not an observed RSS assertion.
pub struct ExtractionDocumentCorpusConfig {
    pub batch: ExtractionDocumentBatchConfig,
    pub reduction_reserve_bytes: u64,
}
impl NlpEngine {
    /// Stream original documents larger than one context using one native
    /// engine and one process invocation. Every successful row contains ALL
    /// chunk JSON values and their original-coordinate source evidence.
    /// Schema/grounding are fixed; flush never renews corpus work allowances.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_extract_document<R, W>(&self, model: &ResidentInt8,
        planner: Arc<Int8ExtractionBatchPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: ExtractionDocumentCorpusConfig, limits: CorpusLimits, reader: R, writer: W,
        cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), planner.base_execution_identity())?;
        config.batch.validate().map_err(HostedError::BatchSetup)?;
        let required = requirements(limits.native)?;
        validate(&config, limits, required.kv_bytes)?;
        let buffers = sum(&[limits.reservation_bytes()?, config.batch.request.schema.capacity() as u64,
            super::super::extraction_map::document_workspace_bytes(config.batch.request.budget,
                config.batch.limits, config.batch.limits.mapping.chunks.max_chunks,
                config.reduction_reserve_bytes)?])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, buffers)?,
            || Ok(StreamInput { planner: Some((planner, vocabulary, config)), reader, writer }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input; // Retain the whole charged owner while queued and running.
            let (planner, vocabulary, config) = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let admission = CorpusAdmission { lease: &lease, model: model.artifact_identity(),
                kv_bytes: required.kv_bytes, sampler_bytes: 0,
                output_bytes: limits.transport.max_output_line_bytes as u64 };
            let result = {
                let mut processor = NativeInt8DocumentExtractionBatch::new(&planner, config.batch,
                    &mut engine.value, &vocabulary, admission).map_err(HostedError::BatchSetup)?;
                batch::run_ndjson(&mut input.value.reader, &mut input.value.writer, &mut processor,
                    limits.transport, control).map_err(HostedError::Batch)
            };
            drop(engine); drop(vocabulary); drop(planner);
            drop(input); // Reader/writer, schema and compiler storage before their charge.
            drop(lease);
            result
        })
    }
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
