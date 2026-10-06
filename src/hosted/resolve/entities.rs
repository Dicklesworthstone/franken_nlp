//! One process-owned automatic NER + resolution invocation, with all physical
//! source/graph/native storage charged until completion and output delivery.
use super::super::*;
use crate::{corpus::{entities_int8::{Int8EntityConfig, Int8EntityError, Int8EntityRun, PreparedInt8EntityCorpus},
    native_resolve::quantized::Int8ResolveError}, tasks::source_planning::quantized::Int8SourceError};
mod long;

struct Input { prepared: Option<PreparedInt8EntityCorpus>, vocabulary: Arc<ExtractionVocabulary> }
impl NlpEngine {
    /// Execute a fully preflighted raw-document snapshot using one resident
    /// model, native engine, blocking invocation and cancellation budget.
    /// Preparation includes retained pinned planners/configuration/allocator
    /// headroom. Graph includes expanded mentions, all pair plans and clustering.
    /// Those explicit commitments are modeled reservations, not measured RSS.
    /// A plan's explicit grouped-prompt choice is priced before native allocation
    /// and applies to both source NER and the complete pair-scoring stage.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_entities(&self, model: &ResidentInt8, prepared: PreparedInt8EntityCorpus,
        vocabulary: Arc<ExtractionVocabulary>, native: NativeLimits, preparation_reserve_bytes: u64,
        graph_reserve_bytes: u64, cancellation: CancellationToken) -> Result<HostedOutput<Int8EntityRun>, HostedError> {
        dispatch::preflight(self, native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.source_identity())?;
        check_model_identity(model.artifact_identity(), prepared.resolution_identity())?;
        let required = requirements(native)?;
        validate(prepared.config(), prepared.required_ner_context_tokens(), native,
            preparation_reserve_bytes, graph_reserve_bytes, required.kv_bytes)?;
        let scratch_bytes = crate::hosted::scored::scoring_scratch(sum(&[required.rope_bytes,
            required.scratch_payload_bound, native.allocator_reserve_bytes])?, prepared.prefill_limits())?;
        let bytes = sum(&[prepared.retained_input_bytes().map_err(execution_error)?, preparation_reserve_bytes])?;
        let temporary_bytes = temporary_bytes(prepared.config(), graph_reserve_bytes)?;
        let output_bytes = prepared.config().max_result_bytes as u64;
        let empty = prepared.document_count() == 0;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(Input { prepared: Some(prepared), vocabulary }))?;
        let model = model.clone();
        dispatch::run(self, native.run, cancellation, move |control| {
            // Capture the WHOLE charged package. Option::take consumes the
            // plan without releasing its charge while source ownership moves
            // through NER, occurrence recovery and the complete pair graph.
            let mut input = input;
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
fn validate(config: &Int8EntityConfig, ner_context: usize, native: NativeLimits, preparation: u64,
    graph: u64, kv_bytes: u64) -> Result<(), HostedError> {
    native.run.validate()?;
    if preparation == 0 || graph == 0 || ner_context > native.context_tokens
        || config.source_planning.max_context_tokens > native.context_tokens
        || config.scoring.planning.max_context_tokens > native.context_tokens
        || kv_bytes > config.ner_budget.max_kv_bytes || kv_bytes > config.scoring.planning.per_head.max_kv_bytes
        || !(1..=64 * 1024 * 1024).contains(&config.max_result_bytes) {
        return Err(HostedError::Limits("entity preparation, graph, output or full resident context"));
    }
    Ok(())
}
fn temporary_bytes(config: &Int8EntityConfig, graph: u64) -> Result<u64, HostedError> {
    let result = config.ner_budget.max_output_bytes.checked_mul(4)
        .ok_or(HostedError::Limits("entity NER intermediate arithmetic"))?;
    let tokens = u64::from(config.ner_budget.max_output_tokens).checked_mul(8)
        .ok_or(HostedError::Limits("entity token arithmetic"))?;
    sum(&[result, tokens, graph])
}
// Preserve stage-specific typed errors, especially cooperative cancellation,
// through the established hosted dispatcher rather than inventing a new host.
fn execution_error(error: Int8EntityError) -> HostedError {
    match error {
        Int8EntityError::Source(error) => HostedError::Source(error),
        Int8EntityError::Resolution(error) => HostedError::Resolution(error),
        Int8EntityError::Graph(error) => HostedError::Resolution(error.into()),
        Int8EntityError::Native(error) => HostedError::Native(error),
        Int8EntityError::Identity => HostedError::ModelIdentity,
        Int8EntityError::WorkBudget => HostedError::Resolution(Int8ResolveError::WorkBudget),
        Int8EntityError::Accounting => HostedError::Source(Int8SourceError::InvalidResult),
        Int8EntityError::InvalidInput => HostedError::Limits("entity original input"),
        Int8EntityError::InvalidLimits => HostedError::Limits("entity configured limits"),
        Int8EntityError::Allocation => HostedError::Limits("entity allocation refused"),
    }
}

#[cfg(test)] mod tests;
