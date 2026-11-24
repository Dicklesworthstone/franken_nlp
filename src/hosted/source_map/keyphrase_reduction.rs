//! Complete keyphrase ranking on the same process-owned native map host.
//! The whole candidate/evidence frontier is charged, not merely the final top-k.
use super::*;
use crate::{
    native_engine::decode::DecodeStepControl,
    tasks::{mapreduce::ExecutionError, source_planning::quantized::{Int8SourceError,
        long::keyphrase_reduction::{Int8CorpusKeyphraseError, Int8CorpusKeyphraseRun, Int8KeyphraseLimits}}},
};

impl NlpEngine {
    /// Run every original chunk with the resident INT8 model, then return one
    /// exact-text, source-backed document-wide ranking. Native per-chunk options
    /// remain fixed by config.task; final selection happens only after complete
    /// evidence validation and all reductions. No second model or runtime.
    #[allow(clippy::too_many_arguments)]
    pub fn keyphrases_int8_source(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig, keyphrases: Int8KeyphraseLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8CorpusKeyphraseRun>, HostedError> {
        self.keyphrases_int8_source_scheduled(model, source, planner, vocabulary, config, keyphrases, None, cancellation)
    }

    /// Exact phrase/evidence ranking with one bounded prompt workspace.
    #[allow(clippy::too_many_arguments)]
    pub fn keyphrases_int8_source_layer_major(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>, config: SourceMapConfig,
        keyphrases: Int8KeyphraseLimits, prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8CorpusKeyphraseRun>, HostedError> {
        self.keyphrases_int8_source_scheduled(model, source, planner, vocabulary, config, keyphrases, Some(prefill), cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn keyphrases_int8_source_scheduled(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>, config: SourceMapConfig,
        keyphrases: Int8KeyphraseLimits, prefill: Option<Int8PrefillLimits>, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8CorpusKeyphraseRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.identity)?;
        let required = requirements(config.native)?;
        let scratch_bytes = team::scratch_bytes(config.native, required, prefill)?;
        validate(&config, source.len(), required.kv_bytes)?;
        validate_keyphrases(&config.task, config.mapping, keyphrases)?;
        check_assets(&config.identity, planner.tokenizer_digest(), *planner.template_digest())?;
        let bytes = input_bytes(&config, source.capacity(), &planner)?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // Keep the entire charged input alive through all source borrowers,
            // including failure/unwind paths and temporary native map outputs.
            let input = input;
            let config = &input.value.config;
            let context = PlanContext::new(&config.identity, config.budget)
                .map_err(|_| HostedError::Limits("keyphrase map task context"))?;
            let prepared = input.value.planner.plan_int8_map_with_control(&input.value.source,
                &config.task, config.budget, &context, config.planning, config.mapping, control)
                .map_err(HostedError::SourceMap)?;
            prepared.check_keyphrases(keyphrases).map_err(execution_error)?;
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers,
                reduction_bytes(config, prepared.chunk_count())?)?;
            // Final ranking has no retained generated-token transcripts. Native
            // temporary tokens still have the complete reduction reservation.
            let output = output_claim(&lease, keyphrases.max_result_bytes as u64, 0)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(prepared.chunk_count())
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
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(config.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let result = allocate(output, || match prefill {
                None => prepared.execute_keyphrases_with_control(&admitted, &mut engine.value,
                    &input.value.vocabulary, keyphrases, control),
                Some(prefill) => prepared.execute_keyphrases_layer_major_with_control(&admitted,
                    &mut engine.value, &input.value.vocabulary, keyphrases, prefill, control),
            }.map_err(execution_error))?;
            drop(engine);
            drop(admitted);
            drop(input);
            drop(reduction);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
    /// Every native map and exact reduction shares one persistent row team.
    #[allow(clippy::too_many_arguments)]
    pub fn keyphrases_int8_source_with_threads(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>, config: SourceMapConfig,
        keyphrases: Int8KeyphraseLimits, kernel: Int8Kernel, compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8CorpusKeyphraseRun>, HostedError> {
        self.keyphrases_int8_source_team(model, source, planner, vocabulary, config,
            keyphrases, kernel, compute_threads, None, cancellation)
    }

    /// Bounded prompt morsels use the same workers and whole-document ledger.
    #[allow(clippy::too_many_arguments)]
    pub fn keyphrases_int8_source_layer_major_with_threads(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>, config: SourceMapConfig,
        keyphrases: Int8KeyphraseLimits, kernel: Int8Kernel, compute_threads: usize,
        prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8CorpusKeyphraseRun>, HostedError> {
        self.keyphrases_int8_source_team(model, source, planner, vocabulary, config,
            keyphrases, kernel, compute_threads, Some(prefill), cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn keyphrases_int8_source_team(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>, config: SourceMapConfig,
        keyphrases: Int8KeyphraseLimits, kernel: Int8Kernel, compute_threads: usize,
        prefill: Option<Int8PrefillLimits>, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8CorpusKeyphraseRun>, HostedError> {
        validate_keyphrases(&config.task, config.mapping, keyphrases)?;
        team::run(self, model, source, planner, vocabulary, config, kernel, compute_threads,
            prefill, keyphrases.max_result_bytes as u64, false, cancellation,
            move |prepared, admitted, engine, vocabulary, prefill, control| {
                match prefill {
                    None => prepared.execute_keyphrases_with_control(admitted, engine, vocabulary, keyphrases, control),
                    Some(prefill) => prepared.execute_keyphrases_layer_major_with_control(
                        admitted, engine, vocabulary, keyphrases, prefill, control),
                }.map_err(execution_error)
            })
    }
}
fn validate_keyphrases(task: &SourceMapTask, mapping: Int8SourceMapLimits, keyphrases: Int8KeyphraseLimits)
    -> Result<(), HostedError> {
    if !matches!(task, SourceMapTask::Keyphrases(_)) {
        return Err(HostedError::Limits("complete keyphrase ranking requires native keyphrase maps"));
    }
    keyphrases.validate(mapping).map_err(execution_error)
}
fn execution_error(error: Int8CorpusKeyphraseError) -> HostedError {
    match error {
        Int8CorpusKeyphraseError::Map(e) => HostedError::SourceMap(e),
        Int8CorpusKeyphraseError::Source(e) => HostedError::Source(e),
        Int8CorpusKeyphraseError::Keyphrases(e)
            | Int8CorpusKeyphraseError::Execution(ExecutionError::Task { source: e, .. }) => HostedError::Source(e.into()),
        Int8CorpusKeyphraseError::Execution(ExecutionError::Checkpoint(e)) => HostedError::SourceMap(Int8SourceMapError::Chunk(e)),
        Int8CorpusKeyphraseError::Execution(ExecutionError::AllocationRefused) => HostedError::SourceMap(Int8SourceMapError::Allocation),
        Int8CorpusKeyphraseError::Execution(ExecutionError::Serialization | ExecutionError::Invariant
            | ExecutionError::InvalidBatch | ExecutionError::InvalidPolicy)
            | Int8CorpusKeyphraseError::Accounting => HostedError::Source(Int8SourceError::InvalidResult),
        Int8CorpusKeyphraseError::Execution(_) | Int8CorpusKeyphraseError::InvalidLimits
            => HostedError::Limits("complete keyphrase reduction limits"),
    }
}

#[cfg(test)] mod tests;
