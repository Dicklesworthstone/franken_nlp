//! Long-document discovery and complete resolution share the existing process host.
use super::*;
use crate::corpus::entities_int8::long::{PreparedInt8DocumentEntityCorpus, Int8DocumentEntityRun};

struct DocumentInput { prepared: Option<PreparedInt8DocumentEntityCorpus>, vocabulary: Arc<ExtractionVocabulary> }
impl NlpEngine {
    /// Execute one consumed long-document snapshot. All exact chunk plans were
    /// preflighted; one grammar is rebuilt at a time and checked against its
    /// private witness. The same resident engine then scores the original-source
    /// graph. Output authority survives serialization and actual delivery.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_document_entities(&self, model: &ResidentInt8, prepared: PreparedInt8DocumentEntityCorpus,
        vocabulary: Arc<ExtractionVocabulary>, native: NativeLimits, preparation_reserve_bytes: u64,
        graph_reserve_bytes: u64, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8DocumentEntityRun>, HostedError> {
        dispatch::preflight(self, native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.source_identity())?;
        check_model_identity(model.artifact_identity(), prepared.resolution_identity())?;
        let required = requirements(native)?;
        let config = &prepared.config().entities;
        validate(config, prepared.required_ner_context_tokens(), native,
            preparation_reserve_bytes, graph_reserve_bytes, required.kv_bytes)?;
        let bytes = sum(&[prepared.retained_input_bytes().map_err(execution_error)?, preparation_reserve_bytes])?;
        // Same peak as one short NER task, plus the complete expanded graph.
        // No retained array of chunk grammars, native transcripts or map values.
        let temporary_bytes = temporary_bytes(config, graph_reserve_bytes)?;
        let output_bytes = config.max_result_bytes as u64;
        let empty = prepared.document_count() == 0;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(DocumentInput { prepared: Some(prepared), vocabulary }))?;
        let model = model.clone();
        dispatch::run(self, native.run, cancellation, move |control| {
            let mut input = input; // Whole storage-before-charge package, even on cancellation.
            let temporary = Pending::reserve(&lease, MemoryClass::JobBuffers, temporary_bytes)?;
            let output = output_claim(&lease, output_bytes, 0)?;
            let result = if empty {
                let prepared = input.value.prepared.take().ok_or(HostedError::CompletionMissing)?;
                allocate(output, || prepared.finalize_without_model(control).map_err(execution_error))?
            } else {
                let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
                let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                    sum(&[required.rope_bytes, required.scratch_payload_bound, native.allocator_reserve_bytes])?)?;
                let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                    .engine(native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
                let prepared = input.value.prepared.take().ok_or(HostedError::CompletionMissing)?;
                let result = allocate(output, || prepared.execute_with_control(&mut engine.value,
                    &input.value.vocabulary, control).map_err(execution_error))?;
                drop(engine);
                result
            };
            drop(temporary);
            drop(input);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}
