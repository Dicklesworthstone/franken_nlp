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
    /// `text` is source text except for QA, where it is the question; separately
    /// typed passages remain the only source of citation evidence.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_source<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceCorpusConfig, limits: CorpusLimits, reader: R, writer: W,
        cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, SourceStream { planner, vocabulary, config }, limits,
            None, None, reader, writer, cancellation)
    }

    /// Bound each prompt's layer-major morsels while retaining exact schema,
    /// source finalizers and all native/mask allowances across the stream.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_source_layer_major<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceCorpusConfig, limits: CorpusLimits,
        prefill: crate::native_engine::strict_int8::prefill::Int8PrefillLimits,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, SourceStream { planner, vocabulary, config }, limits,
            Some(prefill), None, reader, writer, cancellation)
    }

    /// One fixed projection team serves every ordered record. Terminal success
    /// follows physical join under the original control and retained IO claims.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_source_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceCorpusConfig, limits: CorpusLimits,
        kernel: crate::native_engine::simd::Int8Kernel, compute_threads: usize,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, SourceStream { planner, vocabulary, config }, limits,
            None, Some((kernel, compute_threads)), reader, writer, cancellation)
    }

    /// Compose bounded prompt storage and a persistent numerical team. Source
    /// coordinates, full-document results and whole-run work remain unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_source_layer_major_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceCorpusConfig, limits: CorpusLimits,
        kernel: crate::native_engine::simd::Int8Kernel, compute_threads: usize,
        prefill: crate::native_engine::strict_int8::prefill::Int8PrefillLimits,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, SourceStream { planner, vocabulary, config }, limits,
            Some(prefill), Some((kernel, compute_threads)), reader, writer, cancellation)
    }
}

struct SourceStream {
    planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>, config: SourceCorpusConfig,
}
impl scored::CorpusTask for SourceStream {
    fn identity(&self) -> &ExecutionIdentity { &self.config.identity }
    fn validate(&self, model: &ArtifactIdentity, _limits: CorpusLimits,
        native: Int8MemoryRequirement) -> Result<u64, HostedError> {
        self.config.validate(&self.planner, model, native.kv_bytes)?;
        // A compact tokenizer workspace coexists with both caller preparation
        // and transferred planner/vocabulary/transport state.
        super::classify::encoding_workspace_bytes(self.planner.document_encoding())
    }
    #[allow(clippy::too_many_arguments)]
    fn execute<R: BufRead, W: Write>(self, engine: &mut crate::native_engine::strict_int8::StrictInt8Engine<'_>,
        admission: CorpusAdmission<'_>, prefill: Option<crate::native_engine::strict_int8::prefill::Int8PrefillLimits>,
        reader: &mut R, writer: &mut W, limits: BatchLimits, defer: bool,
        control: &mut RunControl) -> Result<scored::Completion, HostedError> {
        let Self { planner, vocabulary, config } = self;
        let compiler = Int8SourceBatchPlanner::new(&planner, config.identity, config.task_ceiling,
            config.planning, config.defaults).map_err(HostedError::BatchSetup)?;
        let mut processor = match prefill {
            Some(prefill) => NativeInt8SourceBatch::new_layer_major(compiler, engine,
                &vocabulary, admission, config.native_work, prefill),
            None => NativeInt8SourceBatch::new(compiler, engine, &vocabulary, admission, config.native_work),
        }.map_err(HostedError::BatchSetup)?;
        scored::execute_stream(reader, writer, &mut processor, limits, defer, control)
    }
}

#[cfg(test)] mod tests;
