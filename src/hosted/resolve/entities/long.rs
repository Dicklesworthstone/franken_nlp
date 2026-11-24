//! Long-document discovery and complete resolution share the existing process host.
use super::*;
use crate::corpus::entities_int8::long::{PreparedInt8DocumentEntityCorpus, Int8DocumentEntityRun};

struct DocumentInput { prepared: Option<PreparedInt8DocumentEntityCorpus>, vocabulary: Arc<ExtractionVocabulary> }
impl NlpEngine {
    /// Execute one consumed long-document snapshot. All exact chunk plans were
    /// preflighted; one grammar is rebuilt at a time and checked against its
    /// private witness. The same resident engine then scores the original-source
    /// graph. Output authority survives serialization and actual delivery.
    /// Grouped prompt workspace is derived from the prepared plan and remains
    /// charged through all NER chunks, pair heads and complete finalization.
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
        let scratch_bytes = crate::hosted::scored::scoring_scratch(sum(&[required.rope_bytes,
            required.scratch_payload_bound, native.allocator_reserve_bytes])?, prepared.prefill_limits())?;
        let bytes = sum(&[prepared.retained_input_bytes().map_err(execution_error)?, preparation_reserve_bytes,
            prepared.resolution_encoding_workspace_bytes().map_err(execution_error)?])?;
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
                let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
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

impl NlpEngine {
    /// Execute every original-document NER chunk and the complete resolution
    /// graph on one persistent numerical team. The consumed plan keeps exact
    /// partition witnesses, source offsets and the original cumulative budgets.
    /// Its optional layer-major prompt schedule is shared by both stages.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_document_entities_with_threads(&self, model: &ResidentInt8,
        prepared: PreparedInt8DocumentEntityCorpus, vocabulary: Arc<ExtractionVocabulary>,
        native: NativeLimits, preparation_reserve_bytes: u64, graph_reserve_bytes: u64,
        kernel: crate::native_engine::simd::Int8Kernel, compute_threads: usize,
        cancellation: CancellationToken) -> Result<HostedInt8TeamResult<Int8DocumentEntityRun>, HostedError> {
        dispatch::preflight_int8_team(self, native.run, compute_threads)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.source_identity())?;
        check_model_identity(model.artifact_identity(), prepared.resolution_identity())?;
        let required = requirements(native)?;
        let config = &prepared.config().entities;
        validate(config, prepared.required_ner_context_tokens(), native,
            preparation_reserve_bytes, graph_reserve_bytes, required.kv_bytes)?;
        let (scratch_bytes, rows) = team_scratch(native, required, compute_threads, prepared.prefill_limits())?;
        let bytes = sum(&[prepared.retained_input_bytes().map_err(execution_error)?, preparation_reserve_bytes,
            prepared.resolution_encoding_workspace_bytes().map_err(execution_error)?])?;
        let temporary_bytes = temporary_bytes(config, graph_reserve_bytes)?;
        let output_bytes = config.max_result_bytes as u64;
        let empty = prepared.document_count() == 0;
        let lease = self.resources().acquire_lease();
        let temporary = Pending::reserve(&lease, MemoryClass::JobBuffers, temporary_bytes)?;
        let output = output_claim(&lease, output_bytes, 0)?;
        let native_claims = if empty { TeamNativeClaims::default() }
            else { TeamNativeClaims::reserve(&lease, required, scratch_bytes)? };
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(DocumentInput { prepared: Some(prepared), vocabulary }))?;
        let admission = TeamAdmission { input, native: native_claims, output: Some(output),
            _temporary: temporary, _lease: lease };
        dispatch::run_int8_prefill_team(self, native.run, cancellation, model.clone(), compute_threads,
            kernel, rows, admission, move |model, admission, control, team| {
                let result = if empty {
                    let prepared = admission.input.value.prepared.take().ok_or(HostedError::CompletionMissing)?;
                    prepared.finalize_without_model(control).map_err(execution_error)?
                } else {
                    let mut engine = model.engine(native.context_tokens, memory_budget(required))
                        .map_err(HostedError::Model)?.with_kernel(kernel)
                        .with_projection_executor(team).map_err(HostedError::Native)?;
                    admission.native.commit()?;
                    let prepared = admission.input.value.prepared.take().ok_or(HostedError::CompletionMissing)?;
                    let result = prepared.execute_with_control(&mut engine,
                        &admission.input.value.vocabulary, control).map_err(execution_error)?;
                    if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                        return Err(execution_error(Int8EntityError::Accounting));
                    }
                    drop(engine);
                    result
                };
                let compute = team.stats().into();
                admission.finish(result, compute)
            })
    }
}
