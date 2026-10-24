//! Complete-document redaction streams on one charged resident INT4 engine.
use super::*;
use std::io::{BufRead, Write};
use crate::{
    batch::{self as stream, BatchCode, BatchItemFailure, BatchSummary},
    grammar::mask::MaskWorkLimits,
    hosted::corpus::{CorpusLimits, RedactionCorpusConfig},
    native_engine::lmhead::NANBEIGE_VOCAB_SIZE,
    tasks::redact::grouped_int4::corpus::{self as document_batch,
        Int4DocumentRedactionBatchConfig, Int4RedactionAdmission, Int4RedactionBatchAdmission,
        NativeInt4DocumentRedactionBatch},
};

struct Preparation {
    planner: Arc<SourceTaskPlanner>,
    vocabulary: Arc<ExtractionVocabulary>,
    config: RedactionCorpusConfig<Int4DocumentRedactionBatchConfig>,
    secret: Option<RedactionPseudonyms>,
}
struct CorpusInput<R, W> {
    preparation: Option<Preparation>,
    reader: R,
    writer: W,
}

impl NlpEngine {
    /// Reuse one model, native allocation and immutable full256 key scope for
    /// the complete ordered stream. Every item may span multiple NER contexts
    /// and receives its own transactional edit and fresh transformed-text scan.
    /// Whole-stream work authority never renews at documents or flush epochs.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_redact_documents<R, W>(&self, model: &ResidentInt4,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig<Int4DocumentRedactionBatchConfig>,
        pseudonyms: Option<RedactionPseudonyms>, limits: CorpusLimits, kernel: PackedInt4Kernel,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        redact_task::check_model(&config.batch.ner_identity, model.model_identity()).map_err(redaction_error)?;
        document_batch::check_configuration(&planner, &config.batch, model.model_identity())
            .map_err(HostedError::BatchSetup)?;
        let width = model.model_identity().group_width();
        let required = Int4MemoryRequirement::for_context(limits.native.context_tokens, width).map_err(native_error)?;
        validate_limits(&config, limits, required.kv_bytes)?;
        let buffers = sum(&[limits.reservation_bytes()?,
            long::document_workspace_bytes(&config.batch.detector, config.edit_reserve_bytes)?,
            secret_bytes(pseudonyms.as_ref())?])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, buffers)?,
            || Ok(CorpusInput { preparation: Some(Preparation { planner, vocabulary, config, secret: pseudonyms }),
                reader, writer }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            checkpoint(control)?;
            let Preparation { planner, vocabulary, config, secret } =
                input.value.preparation.take().ok_or(HostedError::CompletionMissing)?;
            // One scope, checked before native allocation or the first IO.
            // No secret reread, key generation or policy change at a flush.
            let context = pseudonym_context(secret.as_ref(), &config.batch.request)?;
            let mut engine = allocate_native(kv, workspace, || {
                StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens, Int4MemoryBudget {
                    max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                    max_scratch_payload_bytes: required.scratch_payload_bound,
                }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let result = {
                let admission = DocumentAdmission {
                    lease: &lease, model: model.model_identity(),
                    fixed: FixedItem::new(&config.batch, width, required.kv_bytes,
                        limits.transport.max_output_line_bytes as u64),
                };
                let mut processor = NativeInt4DocumentRedactionBatch::new(&planner, config.batch,
                    &mut engine.value, &vocabulary, context.as_ref(), admission).map_err(HostedError::BatchSetup)?;
                stream::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                    &mut processor, limits.transport, control).map_err(HostedError::Batch)
            };
            drop(engine);
            drop(context);
            drop(secret);
            drop(vocabulary);
            drop(planner);
            drop(input); // Owned reader/writer and preparation before their charge.
            drop(lease);
            result
        })
    }
}

fn validate_limits(config: &RedactionCorpusConfig<Int4DocumentRedactionBatchConfig>,
    limits: CorpusLimits, kv_bytes: u64) -> Result<(), HostedError> {
    let detector = &config.batch.detector;
    if config.edit_reserve_bytes == 0 || limits.native.allocator_reserve_bytes == 0
        || kv_bytes == 0 || kv_bytes > detector.per_chunk.max_kv_bytes
        || detector.planning.max_context_tokens > limits.native.context_tokens
        || detector.mapping.chunks.context_tokens > limits.native.context_tokens
        || detector.max_result_bytes == 0
        || detector.max_result_bytes > limits.transport.max_output_line_bytes as u64 {
        return Err(HostedError::Limits("INT4 redaction corpus context, output or editing reservation"));
    }
    Ok(())
}
fn secret_bytes(secret: Option<&RedactionPseudonyms>) -> Result<u64, HostedError> {
    match secret {
        None => Ok(0),
        Some(secret) => {
            if secret.namespace.is_empty() || secret.namespace.len() > 256 {
                return Err(detector_error(RedactError::InvalidOptions));
            }
            sum(&[secret.namespace.capacity() as u64, 2048])
        }
    }
}

/// A complete per-document allowance, comprising independent chunk contexts.
/// It is deliberately NOT interpreted as one concatenated native sequence.
struct FixedItem {
    identity: ExecutionIdentity,
    group_width: usize,
    model_work: Int4Work,
    mask_visits: u64,
    mask_limits: MaskWorkLimits,
    kv_bytes: u64,
    result_bytes: u64,
    output_line_bytes: u64,
}
impl FixedItem {
    fn new(config: &Int4DocumentRedactionBatchConfig, width: usize, kv_bytes: u64, output_line_bytes: u64) -> Self {
        Self { identity: config.ner_identity.clone(), group_width: width, model_work: config.item_model_work(),
            mask_visits: config.detector.mapping.max_mask_visits, mask_limits: config.detector.mapping.mask_limits,
            kv_bytes, result_bytes: config.detector.max_result_bytes, output_line_bytes }
    }
}
struct DocumentAdmission<'a> {
    lease: &'a EngineLease,
    model: &'a Int4ModelIdentity,
    fixed: FixedItem,
}
impl Int4RedactionBatchAdmission for DocumentAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int4RedactionAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        constrained_int4::check_model(request.identity, self.model)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        check_admission(&self.fixed, &request)?;
        // Native chunk tokens/maps/evidence already have the complete workspace
        // reservation. This guard prices the completed redaction envelope and
        // stays with it through serialization, delivery, and writer flushing.
        let guard = output_claim(self.lease, request.max_result_bytes, 0).map_err(admission_failure)?;
        Ok((request.identity.clone(), guard))
    }
}
fn check_admission(fixed: &FixedItem, request: &Int4RedactionAdmission<'_>) -> Result<(), BatchItemFailure> {
    constrained_int4::check_profile(request.identity, fixed.group_width)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    let canonical = |identity: &ExecutionIdentity| crate::canonjson::canonical_bytes(identity)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Serialization));
    if request.identity.task_spec != NER_TASK_VERSION
        || canonical(request.identity)? != canonical(&fixed.identity)?
        || fixed.kv_bytes == 0 || request.kv_reservation_bytes != fixed.kv_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let work = request.model_work;
    if work != fixed.model_work || work.forward_positions == 0 || work.projected_logits == 0
        || work.projected_logits % NANBEIGE_VOCAB_SIZE as u64 != 0 || work.attention_pairs == 0
        || work.projections.dot_products == 0 || work.projections.multiply_accumulates == 0
        || request.mask_node_visits == 0 || request.mask_node_visits != fixed.mask_visits
        || request.mask_limits.max_trie_node_visits != fixed.mask_limits.max_trie_node_visits
        || request.mask_limits.checkpoint_interval_nodes != fixed.mask_limits.checkpoint_interval_nodes
        || fixed.mask_limits.max_trie_node_visits == 0 || fixed.mask_limits.checkpoint_interval_nodes == 0 {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    if request.max_result_bytes == 0 || request.max_result_bytes > fixed.output_line_bytes {
        return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit));
    }
    if request.max_result_bytes != fixed.result_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    Ok(())
}
fn admission_failure(error: HostedError) -> BatchItemFailure {
    match error {
        HostedError::Reservation(_) => BatchItemFailure::fatal(BatchCode::Admission),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    }
}

#[cfg(test)]
mod tests;
