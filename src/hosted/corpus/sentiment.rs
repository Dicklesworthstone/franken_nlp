//! One process-owned invocation for a resident independent-axis sentiment stream.
use super::*;
use crate::tasks::sentiment::{SentimentPlanner, batch::{Int8SentimentBatchPlanner,
    NativeInt8SentimentBatch, Int8SentimentAdmission, Int8SentimentAdmissionRequest}};
pub use crate::tasks::sentiment::batch::SentimentBatchConfig as SentimentCorpusConfig;

impl NlpEngine {
    /// Run complete independent-axis bundles through one process-owned native
    /// invocation. Policy and score space remain fixed by the pinned planner.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_sentiment<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SentimentPlanner>, config: SentimentCorpusConfig,
        limits: CorpusLimits, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, SentimentStream { planner, config }, limits,
            None, None, reader, writer, cancellation)
    }

    /// Evaluate bounded prompt morsels for every independent axis while
    /// retaining exact candidate scoring and the whole-stream native ledger.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_sentiment_layer_major<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SentimentPlanner>, config: SentimentCorpusConfig,
        limits: CorpusLimits, prefill: crate::native_engine::strict_int8::prefill::Int8PrefillLimits,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, SentimentStream { planner, config }, limits,
            Some(prefill), None, reader, writer, cancellation)
    }

    /// Keep one persistent numerical team across every document and axis.
    /// Per-record outputs remain complete; terminal success is deferred until
    /// every worker joins, with reader/writer reservations still owned.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_sentiment_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SentimentPlanner>, config: SentimentCorpusConfig,
        limits: CorpusLimits, kernel: crate::native_engine::simd::Int8Kernel,
        compute_threads: usize, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, SentimentStream { planner, config }, limits,
            None, Some((kernel, compute_threads)), reader, writer, cancellation)
    }

    /// Compose packed prompts and persistent workers without renewing work,
    /// tokenizer, transport or deadline limits at document/axis boundaries.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_sentiment_layer_major_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SentimentPlanner>, config: SentimentCorpusConfig,
        limits: CorpusLimits, kernel: crate::native_engine::simd::Int8Kernel,
        compute_threads: usize, prefill: crate::native_engine::strict_int8::prefill::Int8PrefillLimits,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, SentimentStream { planner, config }, limits,
            Some(prefill), Some((kernel, compute_threads)), reader, writer, cancellation)
    }
}

struct SentimentStream { planner: Arc<SentimentPlanner>, config: SentimentCorpusConfig }
impl scored::CorpusTask for SentimentStream {
    fn identity(&self) -> &ExecutionIdentity { &self.config.identity }
    fn validate(&self, _model: &ArtifactIdentity, limits: CorpusLimits, native: Int8MemoryRequirement) -> Result<u64, HostedError> {
        self.config.validate_with_document_encoding(&self.planner).map_err(HostedError::BatchSetup)?;
        validate_capacity(self.config.task_ceiling, native.kv_bytes, limits.transport.max_output_line_bytes)?;
        super::classify::encoding_workspace_bytes(self.planner.document_encoding())
    }
    #[allow(clippy::too_many_arguments)]
    fn execute<R: BufRead, W: Write>(self, engine: &mut crate::native_engine::strict_int8::StrictInt8Engine<'_>,
        admission: CorpusAdmission<'_>, prefill: Option<crate::native_engine::strict_int8::prefill::Int8PrefillLimits>,
        reader: &mut R, writer: &mut W, limits: BatchLimits, defer: bool,
        control: &mut RunControl) -> Result<scored::Completion, HostedError> {
        let Self { planner, config } = self;
        let compiler = Int8SentimentBatchPlanner::new_with_document_encoding(&planner, config)
            .map_err(HostedError::BatchSetup)?;
        let mut processor = match prefill {
            Some(prefill) => NativeInt8SentimentBatch::new_layer_major(compiler, engine, admission, prefill),
            None => NativeInt8SentimentBatch::new(compiler, engine, admission),
        }.map_err(HostedError::BatchSetup)?;
        scored::execute_stream(reader, writer, &mut processor, limits, defer, control)
    }
}

impl Int8SentimentAdmission for CorpusAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int8SentimentAdmissionRequest<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        if request.identity.task_spec != "sentiment-v1" || request.kv_reservation_bytes != self.kv_bytes {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        check_model_identity(self.model, request.identity).map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        crate::native_engine::constrained_int8::check_profile(request.identity)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        validate_work(request.model_work).map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
        if request.max_result_bytes == 0 || request.max_result_bytes > self.output_bytes {
            return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit));
        }
        // No generated-token buffer exists here. The entire bounded result
        // already includes candidate strings/scores/distributions. Do not infer
        // a token count from projected rows: the library supports three named
        // score spaces with different row counts. The CLI selects full vocab.
        let guard = output_claim(self.lease, request.max_result_bytes, 0).map_err(admission_failure)?;
        Ok((request.identity.clone(), guard))
    }
}
fn validate_capacity(task: crate::tasks::ir::TaskBudget, kv_bytes: u64, line_bytes: usize)
    -> Result<(), HostedError> {
    task.validate().map_err(|_| HostedError::Limits("sentiment corpus task ceiling"))?;
    if kv_bytes == 0 || kv_bytes > task.max_kv_bytes || task.max_output_bytes > line_bytes as u64 {
        return Err(HostedError::Limits("sentiment corpus complete KV/output envelope"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::ir::TaskBudget;
    fn budget() -> TaskBudget {
        TaskBudget { max_input_tokens: 4096, max_output_tokens: 16,
            max_output_bytes: 8192, max_grammar_states: 4096, max_kv_bytes: 65536 }
    }
    #[test]
    fn full_native_kv_and_complete_result_must_fit_even_for_short_documents() {
        validate_capacity(budget(), 65536, 8192).unwrap();
        assert!(validate_capacity(budget(), 65537, 8192).is_err());
        assert!(validate_capacity(budget(), 65536, 8191).is_err());
        assert!(validate_capacity(budget(), 0, 8192).is_err());
    }
    #[test]
    fn zero_task_axes_cannot_be_used_to_skip_admission() {
        let mut b = budget(); b.max_output_bytes = 0;
        assert!(validate_capacity(b, 65536, 8192).is_err());
        let mut b = budget(); b.max_output_tokens = 0;
        assert!(validate_capacity(b, 65536, 8192).is_err());
    }
    #[test]
    fn corpus_configuration_and_captures_cross_the_owned_runtime_boundary() {
        fn send<T: Send + 'static>() {}
        send::<SentimentCorpusConfig>();
        send::<StreamInput<(Arc<SentimentPlanner>, SentimentCorpusConfig),
            std::io::Cursor<Vec<u8>>, Vec<u8>>>();
    }
}
