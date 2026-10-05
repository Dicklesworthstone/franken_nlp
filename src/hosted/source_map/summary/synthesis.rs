//! One charged hosted invocation for discovery plus evidence-only synthesis.
use super::*;
use crate::tasks::{summarize::SUMMARIZE_TASK_VERSION,
    source_planning::quantized::long::summary::synthesis::{SourceSummarySynthesis, Int8SummarySynthesisRun}};

impl NlpEngine {
    /// Synthesize a new cited summary from all collected verbatim map evidence.
    /// Generated map text is never fed back as source. This explicit API neither
    /// replaces exact-union summaries nor claims full-document recall/equivalence.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int8_summary(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceSummarySynthesis>, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8SummarySynthesisRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.identity)?;
        let required = requirements(config.native)?;
        validate_synthesis(&config, source.len(), required.kv_bytes)?;
        check_assets(&config.identity, planner.tokenizer_digest(), *planner.template_digest())?;
        let bytes = sum(&[source.capacity() as u64, config.preparation_reserve_bytes])?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // Retain the WHOLE owner through native drain. Preparation includes
            // all map programs and the subsequent final evidence program.
            let input = input;
            let config = &input.value.config;
            let context = PlanContext::new(&config.identity, config.budget)
                .map_err(|_| HostedError::Limits("summary synthesis task context"))?;
            let prepared = input.value.planner.plan_int8_summary_synthesis_with_control(&input.value.source,
                config.task, config.budget, &context, config.planning, config.mapping, control)
                .map_err(HostedError::SourceMap)?;
            let expected = prepared.preflight_metadata();
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers,
                synthesis_bytes(config, expected.chunk_count())?)?;
            let tokens = (expected.chunk_count() as u64).checked_add(1)
                .and_then(|n| n.checked_mul(u64::from(config.budget.max_output_tokens)))
                .ok_or(HostedError::Limits("summary synthesis token arithmetic"))?;
            // Unlike ranked-union summaries, native map AND final transcripts
            // survive in this output and remain charged after the engine drains.
            let output = output_claim(&lease, prepared.max_result_bytes(), tokens)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(expected.chunk_count())
                .map_err(|_| HostedError::SourceMap(Int8SourceMapError::Allocation))?;
            for identity in prepared.execution_identities() {
                if let Some(cause) = control.prefill_checkpoint(0) {
                    return Err(HostedError::Source(Int8SourceError::Cancelled(cause)));
                }
                check_model_identity(model.artifact_identity(), identity)?;
                constrained_int8::check_profile(identity).map_err(|_| HostedError::ModelIdentity)?;
                admitted.push(identity.clone());
            }
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.rope_bytes, required.scratch_payload_bound, config.native.allocator_reserve_bytes])?)?;
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(config.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let result = allocate(output, || {
                let result = prepared.execute_with_control(&admitted, &input.value.planner,
                    &mut engine.value, &input.value.vocabulary, control).map_err(HostedError::SourceMap)?;
                expected.verify_completed(&result).map_err(HostedError::SourceMap)?;
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
fn validate_synthesis(config: &SourceMapConfig<SourceSummarySynthesis>, source_bytes: usize, kv_bytes: u64)
    -> Result<(), HostedError> {
    validate_common(config, source_bytes, kv_bytes)?;
    if config.identity.task_spec != SUMMARIZE_TASK_VERSION { return Err(HostedError::ModelIdentity); }
    config.task.validate(config.planning).map_err(HostedError::SourceMap)
}
fn synthesis_bytes(config: &SourceMapConfig<SourceSummarySynthesis>, chunks: usize) -> Result<u64, HostedError> {
    let base = reduction_bytes(config, chunks)?;
    let l = config.task.limits;
    // Existing map frontier + complete final native payload + bounded evidence
    // copies, KMP prefix storage, origin tables and overlapping occurrence fanout.
    // These are modeled reservation inputs, not allocator-enforced RSS bounds.
    let mul = |a: u64, b: u64| a.checked_mul(b).ok_or(HostedError::Limits("summary evidence arithmetic"));
    sum(&[base, mul(config.budget.max_output_bytes, 4)?,
        mul(u64::from(config.budget.max_output_tokens), 8)?,
        mul(l.max_evidence_bytes as u64, 64)?, mul(l.max_evidence_segments as u64, 128)?,
        mul(l.verification.max_matches as u64, 128)?, mul(l.verification.max_fields as u64, 128)?])
}
#[cfg(test)] mod tests;
