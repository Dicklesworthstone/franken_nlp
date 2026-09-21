//! Process-hosted source corpora with the existing ledger-backed admission.
//! One charged resident model, one native engine, one owned stream invocation.
use super::*;
use crate::{
    batch::source::{SourceBatchArgs, quantized::{self as source_batch,
        Int8SourceBatchLimits, Int8SourceBatchPlanner, NativeInt8SourceBatch}},
    tasks::{ir::TaskBudget, source_planning::{SourceTaskPlanner, SourcePlanningLimits}},
};

/// Fixed task/model/planner contract. Not deserializable execution authority;
/// defaults may contain private evidence passages, so no Debug implementation.
pub struct SourceCorpusConfig {
    pub identity: ExecutionIdentity,
    pub task_ceiling: TaskBudget,
    pub planning: SourcePlanningLimits,
    pub defaults: Option<SourceBatchArgs>,
    /// Complete native plus grammar-mask work across the whole stream. Neither
    /// document failures, early finishes nor flushes renew these reservations.
    pub native_work: Int8SourceBatchLimits,
}
impl SourceCorpusConfig {
    fn validate(&self, planner: &SourceTaskPlanner, model: &ArtifactIdentity, allocated_kv: u64)
        -> Result<(), HostedError> {
        check_model_identity(model, &self.identity)?;
        source_batch::check_configuration(planner, &self.identity, self.task_ceiling,
            self.planning, self.defaults.as_ref()).map_err(HostedError::BatchSetup)?;
        source_batch::validate_limits(self.native_work).map_err(HostedError::BatchSetup)?;
        if allocated_kv == 0 || allocated_kv > self.task_ceiling.max_kv_bytes {
            return Err(HostedError::Limits("source corpus complete KV reservation"));
        }
        Ok(())
    }
}

impl NlpEngine {
    /// Execute NER/keyphrases/cited summaries/passage QA over bounded NDJSON.
    /// `text` is source text except for QA, where it is the question; the
    /// separately typed passages remain the only source of citation evidence.
    /// The host supplies concrete admission and retains all captures until
    /// actual physical completion, including cancellation and queued disposal.
    /// No model reload, per-document runtime, fallback or inference network.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_source<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceCorpusConfig, limits: CorpusLimits, reader: R, writer: W,
        cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        let required = requirements(limits.native)?;
        config.validate(&planner, model.artifact_identity(), required.kv_bytes)?;
        let lease = self.resources().acquire_lease();
        // Preparation prices the pinned planner/vocabulary, defaults, source
        // graphs and transient task copies. IO prices the actual supplied
        // reader/writer buffers. These modeled charges are not an RSS bound.
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, limits.reservation_bytes()?)?,
            || Ok(StreamInput { planner: Some((planner, vocabulary, config)), reader, writer }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            // Capture the entire storage-before-charge aggregate. Partial
            // field captures would lose that ownership order on queued drop.
            let mut input = input;
            let (planner, vocabulary, config) = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let compiler = Int8SourceBatchPlanner::new(&planner, config.identity, config.task_ceiling,
                config.planning, config.defaults).map_err(HostedError::BatchSetup)?;
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            // Source admission is the same contract as extraction: reuse the
            // real process-ledger provider and never fabricate a no-op guard.
            let admission = CorpusAdmission { lease: &lease, model: model.artifact_identity(),
                kv_bytes: required.kv_bytes, sampler_bytes: 0,
                output_bytes: limits.transport.max_output_line_bytes as u64 };
            let result = {
                let mut processor = NativeInt8SourceBatch::new(compiler, &mut engine.value,
                    &vocabulary, admission, config.native_work).map_err(HostedError::BatchSetup)?;
                batch::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                    &mut processor, limits.transport, control).map_err(HostedError::Batch)
            };
            drop(engine);
            drop(vocabulary);
            drop(planner);
            drop(input); // owned IO storage before its charge and physical handoff
            drop(lease);
            result
        })
    }
}

#[cfg(test)] mod tests;
