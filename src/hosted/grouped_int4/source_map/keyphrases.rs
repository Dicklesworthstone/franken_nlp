//! Complete deterministic keyphrase ranking with all native maps still owned.
use super::*;
use crate::tasks::source_planning::grouped_int4::long::keyphrases::{
    Int4CorpusKeyphraseError, Int4CorpusKeyphraseRun, Int4KeyphraseLimits,
};

impl<T> Int4SourceMapConfig<T> {
    /// Complete native frontier plus the independent exact-text candidate and
    /// occurrence union. The final top-k cannot reduce the admitted frontier.
    /// These are conservative owner allowances, not an observed RSS promise.
    pub fn keyphrase_workspace_bytes(&self, chunks: usize, limits: Int4KeyphraseLimits)
        -> Result<u64, HostedError> {
        limits.validate(self.mapping).map_err(keyphrase_error)?;
        let multiply = |value: u64, count: u64| value.checked_mul(count)
            .ok_or(HostedError::Limits("INT4 keyphrase frontier arithmetic"));
        let values = multiply(limits.aggregation.max_value_bytes as u64, 8)?;
        let candidates = multiply(limits.aggregation.max_unique_phrases as u64, 512)?;
        let evidence = multiply(multiply(limits.aggregation.max_evidence_spans as u64,
            std::mem::size_of::<VerifiedSourceSpan>() as u64)?, 2)?;
        // Aggregate construction, canonical serialization and exact matching
        // coexist with the original native/source-verification frontier.
        sum(&[self.workspace_bytes(chunks)?, values, candidates, evidence])
    }
}

impl NlpEngine {
    /// Extract from every original chunk, independently verify every occurrence
    /// and return one exact-text ranking. All candidates survive intermediate
    /// reductions; final ordering uses supporting-chunk count, local rank sum,
    /// first source position and exact text. Repeated occurrences within one
    /// chunk contribute one support vote. This is not neural reranking.
    #[allow(clippy::too_many_arguments)]
    pub fn reduce_int4_keyphrases_document(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig, keyphrases: Int4KeyphraseLimits,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4CorpusKeyphraseRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(&config.identity, model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let width = model.model_identity().group_width();
        let required = keyphrase_requirements(&config, source.len(), width, &planner, keyphrases)?;
        let bytes = sum(&[source.capacity() as u64, config.preparation_reserve_bytes])?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            let input = input;
            let config = &input.value.config;
            let context = PlanContext::new(&config.identity, config.budget)
                .map_err(|_| HostedError::Limits("INT4 keyphrase map task context"))?;
            let prepared = input.value.planner.plan_int4_map_with_control(&input.value.source,
                &config.task, config.budget, &context, config.planning, config.mapping, width, control)
                .map_err(map_error)?;
            prepared.check_keyphrases(keyphrases).map_err(keyphrase_error)?;
            let expected = prepared.preflight_metadata();
            let count = prepared.chunk_count();
            if prepared.group_width() != width || expected.group_width() != width
                || prepared.execution_identities().len() != count || required.kv_bytes > prepared.max_kv_bytes() {
                return Err(HostedError::ModelIdentity);
            }
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers,
                config.keyphrase_workspace_bytes(count, keyphrases)?)?;
            // The final ranking retains no generated-token transcripts. All
            // temporary native tokens remain charged in the complete frontier.
            let output = output_claim(&lease, keyphrases.max_result_bytes as u64, 0)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(count).map_err(|_| map_error(Int4SourceMapError::Allocation))?;
            for identity in prepared.execution_identities() {
                checkpoint(control)?;
                constrained_int4::check_model(identity, model.model_identity())
                    .map_err(|_| HostedError::ModelIdentity)?;
                admitted.push(identity.clone());
            }
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.rope_bytes, required.scratch_payload_bound, config.native.allocator_reserve_bytes])?)?;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, config.native.context_tokens, Int4MemoryBudget {
                    max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                    max_scratch_payload_bytes: required.scratch_payload_bound,
                }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let result = allocate(output, || {
                let result = prepared.execute_keyphrases_with_control(&admitted,
                    &mut engine.value, &input.value.vocabulary, keyphrases, control).map_err(keyphrase_error)?;
                expected.verify_keyphrases_completed(&result).map_err(keyphrase_error)?;
                check_clean(&engine.value)?;
                checkpoint(control)?;
                Ok(result)
            })?;
            drop(engine);
            drop(admitted);
            drop(input);
            drop(reduction);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

fn keyphrase_requirements(config: &Int4SourceMapConfig, source_bytes: usize, width: usize,
    planner: &SourceTaskPlanner, limits: Int4KeyphraseLimits) -> Result<Int4MemoryRequirement, HostedError> {
    let required = map_requirements(config, source_bytes, width, planner)?;
    if !matches!(&config.task, SourceMapTask::Keyphrases(_)) {
        return Err(HostedError::Limits("INT4 complete keyphrase ranking requires keyphrase maps"));
    }
    // Quote the maximum frontier before input capture; every actual partition
    // is also re-quoted before native execution. No top-k-based capacity relief.
    config.keyphrase_workspace_bytes(config.mapping.chunks.max_chunks, limits)?;
    Ok(required)
}
fn keyphrase_error(error: Int4CorpusKeyphraseError) -> HostedError {
    HostedError::Int4(Int4HostError::SourceKeyphrases(error))
}

#[cfg(test)]
mod tests;
