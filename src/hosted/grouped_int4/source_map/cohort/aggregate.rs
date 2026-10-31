//! Native chunk cohorts feeding the existing complete-document reducers.
//! The host retains every map/frontier allowance until final ranking, without
//! reducing admission to the small requested final top-k.
use super::*;
use crate::tasks::source_planning::grouped_int4::long::{
    PreparedInt4SourceMap, Int4SourceMapPreflight,
    summary::{Int4SummaryLimits, Int4CorpusSummaryError, Int4CorpusSummaryCohortRun},
    keyphrases::{Int4KeyphraseLimits, Int4CorpusKeyphraseError, Int4CorpusKeyphraseCohortRun},
};

impl NlpEngine {
    /// Summarize every original chunk with native cohorts, then perform the
    /// complete exact-text bullet/evidence union and final deterministic top-k.
    /// Citation membership retains its existing guarantee; it does not assess
    /// entailment, reconcile contradictions or perform neural synthesis.
    #[allow(clippy::too_many_arguments)]
    pub fn summarize_int4_source_cohorts(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig, summary: Int4SummaryLimits,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4CorpusSummaryCohortRun>, HostedError> {
        execute(self, model, source, planner, vocabulary, config, summary, kernel, cancellation)
    }

    /// Extract all original-source keyphrase candidates with native cohorts,
    /// independently verify their complete occurrences and rank the exact
    /// document-wide union. Only the final root applies the requested top-k.
    #[allow(clippy::too_many_arguments)]
    pub fn reduce_int4_keyphrases_document_cohorts(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig, keyphrases: Int4KeyphraseLimits,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4CorpusKeyphraseCohortRun>, HostedError> {
        execute(self, model, source, planner, vocabulary, config, keyphrases, kernel, cancellation)
    }
}

/// Closed static composition for the two shipped deterministic reductions.
/// Callers cannot inject a result or replace the validation/finalization path.
trait DocumentAggregation: Copy + Send + 'static {
    type Output: Send + 'static;
    fn validate(self, config: &Int4SourceMapConfig) -> Result<(), HostedError>;
    fn check_plan(self, prepared: &PreparedInt4SourceMap<'_>) -> Result<(), HostedError>;
    fn workspace_bytes(self, config: &Int4SourceMapConfig, chunks: usize) -> Result<u64, HostedError>;
    fn output_bytes(self) -> Result<u64, HostedError>;
    #[allow(clippy::too_many_arguments)]
    fn run<C: DecodeStepControl>(self, prepared: PreparedInt4SourceMap<'_>, admitted: &[ExecutionIdentity],
        weights: &Int4ModelWeights, vocabulary: &ExtractionVocabulary, budget: Int4SourceMapCohortBudget,
        kernel: PackedInt4Kernel, control: &mut C) -> Result<Self::Output, HostedError>;
    fn verify(self, expected: Int4SourceMapPreflight, result: &Self::Output) -> Result<(), HostedError>;
}

impl DocumentAggregation for Int4SummaryLimits {
    type Output = Int4CorpusSummaryCohortRun;
    fn validate(self, config: &Int4SourceMapConfig) -> Result<(), HostedError> {
        if !matches!(&config.task, SourceMapTask::Summarize(_)) {
            return Err(HostedError::Limits("INT4 summary cohorts require summary maps"));
        }
        self.validate(config.mapping).map_err(summary_error)
    }
    fn check_plan(self, prepared: &PreparedInt4SourceMap<'_>) -> Result<(), HostedError> {
        prepared.check_summary(self).map_err(summary_error)
    }
    fn workspace_bytes(self, config: &Int4SourceMapConfig, chunks: usize) -> Result<u64, HostedError> {
        config.workspace_bytes(chunks)
    }
    fn output_bytes(self) -> Result<u64, HostedError> { sum(&[self.max_result_bytes as u64, 4096]) }
    fn run<C: DecodeStepControl>(self, prepared: PreparedInt4SourceMap<'_>, admitted: &[ExecutionIdentity],
        weights: &Int4ModelWeights, vocabulary: &ExtractionVocabulary, budget: Int4SourceMapCohortBudget,
        kernel: PackedInt4Kernel, control: &mut C) -> Result<Self::Output, HostedError> {
        prepared.execute_summary_cohorts_with_control(admitted, weights, vocabulary, self, budget, kernel, control)
            .map_err(summary_error)
    }
    fn verify(self, expected: Int4SourceMapPreflight, result: &Self::Output) -> Result<(), HostedError> {
        expected.verify_summary_completed(&result.document).map_err(summary_error)
    }
}

impl DocumentAggregation for Int4KeyphraseLimits {
    type Output = Int4CorpusKeyphraseCohortRun;
    fn validate(self, config: &Int4SourceMapConfig) -> Result<(), HostedError> {
        if !matches!(&config.task, SourceMapTask::Keyphrases(_)) {
            return Err(HostedError::Limits("INT4 keyphrase cohorts require keyphrase maps"));
        }
        self.validate(config.mapping).map_err(keyphrase_error)
    }
    fn check_plan(self, prepared: &PreparedInt4SourceMap<'_>) -> Result<(), HostedError> {
        prepared.check_keyphrases(self).map_err(keyphrase_error)
    }
    fn workspace_bytes(self, config: &Int4SourceMapConfig, chunks: usize) -> Result<u64, HostedError> {
        config.keyphrase_workspace_bytes(chunks, self)
    }
    fn output_bytes(self) -> Result<u64, HostedError> { sum(&[self.max_result_bytes as u64, 4096]) }
    fn run<C: DecodeStepControl>(self, prepared: PreparedInt4SourceMap<'_>, admitted: &[ExecutionIdentity],
        weights: &Int4ModelWeights, vocabulary: &ExtractionVocabulary, budget: Int4SourceMapCohortBudget,
        kernel: PackedInt4Kernel, control: &mut C) -> Result<Self::Output, HostedError> {
        prepared.execute_keyphrases_cohorts_with_control(admitted, weights, vocabulary, self, budget, kernel, control)
            .map_err(keyphrase_error)
    }
    fn verify(self, expected: Int4SourceMapPreflight, result: &Self::Output) -> Result<(), HostedError> {
        expected.verify_keyphrases_completed(&result.document).map_err(keyphrase_error)
    }
}

#[allow(clippy::too_many_arguments)]
fn execute<A: DocumentAggregation>(host: &NlpEngine, model: &ResidentInt4, source: String,
    planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>, config: Int4SourceMapConfig,
    aggregation: A, kernel: PackedInt4Kernel, cancellation: CancellationToken)
    -> Result<HostedOutput<A::Output>, HostedError> {
    dispatch::preflight(host, config.native.run)?;
    if !Arc::ptr_eq(host.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
    constrained_int4::check_model(&config.identity, model.model_identity()).map_err(|_| HostedError::ModelIdentity)?;
    let width = model.model_identity().group_width();
    let row = map_requirements(&config, source.len(), width, &planner)?;
    aggregation.validate(&config)?;
    let ceiling = config.cohort_requirements(width)?;
    // Bound the full declared frontier before capture. The actual partition
    // is quoted again before inference; top-k cannot shrink either allowance.
    aggregation.workspace_bytes(&config, config.mapping.chunks.max_chunks)?;
    let output_bytes = aggregation.output_bytes()?;
    let bytes = sum(&[source.capacity() as u64, config.preparation_reserve_bytes])?;
    let run = config.native.run;
    let lease = host.resources().acquire_lease();
    let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
        || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
    let model = model.clone();
    dispatch::run(host, run, cancellation, move |control| {
        let input = input;
        let config = &input.value.config;
        checkpoint(control)?;
        let context = PlanContext::new(&config.identity, config.budget)
            .map_err(|_| HostedError::Limits("INT4 aggregate cohort task context"))?;
        let prepared = input.value.planner.plan_int4_map_with_control(&input.value.source,
            &config.task, config.budget, &context, config.planning, config.mapping, width, control).map_err(map_error)?;
        aggregation.check_plan(&prepared)?;
        let expected = prepared.preflight_metadata();
        let count = prepared.chunk_count();
        if prepared.group_width() != width || expected.group_width() != width
            || prepared.execution_identities().len() != count || row.kv_bytes > prepared.max_kv_bytes() {
            return Err(HostedError::ModelIdentity);
        }
        let required = prepared.cohort_requirements(config.native.context_tokens).map_err(map_error)?;
        check_peak(required, ceiling)?;
        let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers,
            aggregation.workspace_bytes(config, count)?)?;
        // Aggregates omit native token transcripts; all transient row results
        // and tokens remain separately funded by the complete map frontier.
        let output = output_claim(&lease, output_bytes, 0)?;
        let mut admitted = Vec::new();
        admitted.try_reserve_exact(count).map_err(|_| map_error(Int4SourceMapError::Allocation))?;
        for identity in prepared.execution_identities() {
            checkpoint(control)?;
            constrained_int4::check_model(identity, model.model_identity()).map_err(|_| HostedError::ModelIdentity)?;
            if identity.task_spec != config.identity.task_spec
                || identity.tokenizer_digest != config.identity.tokenizer_digest
                || identity.template_digest != config.identity.template_digest {
                return Err(HostedError::ModelIdentity);
            }
            admitted.push(identity.clone());
        }
        checkpoint(control)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.native.rope_bytes, required.native.scratch_payload_bound,
                config.native.allocator_reserve_bytes])?)?;
        let workspace = Pending::reserve(&lease, MemoryClass::JobBuffers, required.workspace_bytes)?;
        let budget = Int4SourceMapCohortBudget { context_limit: config.native.context_tokens,
            native_memory: Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                max_rope_bytes: required.native.rope_bytes,
                max_scratch_payload_bytes: required.native.scratch_payload_bound },
            max_workspace_bytes: required.workspace_bytes, max_result_bytes: required.max_result_bytes };
        let result = allocate(output, || {
            let result = aggregation.run(prepared, &admitted, &model.inner.loaded.value,
                &input.value.vocabulary, budget, kernel, control)?;
            aggregation.verify(expected, &result)?;
            checkpoint(control)?;
            Ok(result)
        })?;
        drop(admitted);
        drop(input);
        drop(workspace);
        drop(scratch);
        drop(kv);
        drop(reduction);
        drop(lease);
        Ok(GuardedOutput::new(result.value, result._memory))
    })
}

fn summary_error(error: Int4CorpusSummaryError) -> HostedError {
    HostedError::Int4(Int4HostError::SourceSummary(error))
}
fn keyphrase_error(error: Int4CorpusKeyphraseError) -> HostedError {
    HostedError::Int4(Int4HostError::SourceKeyphrases(error))
}

#[cfg(test)] mod tests;
