//! Resident ordered classification with concrete, process-ledger admission.
mod document;
pub use document::{ClassificationDocumentCorpusConfig, int8_classification_document_corpus_requirements};
use super::*;
use crate::{
    batch::classify::{ClassificationBatchArgs, quantized::{Int8ClassificationAdmission,
        Int8ClassificationBatchPlanner, NativeInt8ClassificationBatch}},
    tasks::{classify::{ClassificationLimits, ClassificationPlanner}, ir::TaskBudget},
};

/// Immutable-by-transfer settings for a bounded classification stream. Request
/// budgets may narrow `task_ceiling`, never enlarge it. The whole output ceiling
/// is conservatively reserved for each item because the existing admission
/// interface exposes identity/work, not the item's individual byte budget.
/// No Debug: defaults may contain private labels/descriptions.
pub struct ClassificationCorpusConfig {
    pub identity: ExecutionIdentity,
    pub task_ceiling: TaskBudget,
    pub planning: ClassificationLimits,
    pub defaults: Option<ClassificationBatchArgs>,
    /// Five nonrefundable native counters across all documents and flushes.
    pub max_model_work: Int8Work,
}

impl NlpEngine {
    /// Run exclusive or multi-label NDJSON requests on one resident INT8 engine.
    /// Full candidate/EOS scoring and exact finalization retain their original
    /// semantics. Successful return does not imply zero rejected records.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_classify<R, W>(&self, model: &ResidentInt8,
        planner: Arc<ClassificationPlanner>, config: ClassificationCorpusConfig,
        limits: CorpusLimits, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, ClassificationStream { planner, config }, limits,
            None, None, reader, writer, cancellation)
    }

    /// Bound each head's prompt morsels while preserving all five whole-run
    /// work counters and the same ordered transport through every flush.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_classify_layer_major<R, W>(&self, model: &ResidentInt8,
        planner: Arc<ClassificationPlanner>, config: ClassificationCorpusConfig,
        limits: CorpusLimits, prefill: crate::native_engine::strict_int8::prefill::Int8PrefillLimits,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, ClassificationStream { planner, config }, limits,
            Some(prefill), None, reader, writer, cancellation)
    }

    /// Reuse one admitted output-row team throughout the scoring stream.
    /// Complete item records remain ordered; run_complete follows physical
    /// worker join under the original invocation's cancellation and deadline.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_classify_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Arc<ClassificationPlanner>, config: ClassificationCorpusConfig,
        limits: CorpusLimits, kernel: crate::native_engine::simd::Int8Kernel,
        compute_threads: usize, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, ClassificationStream { planner, config }, limits,
            None, Some((kernel, compute_threads)), reader, writer, cancellation)
    }

    /// One fixed worker batch capacity covers packed prompts and candidate
    /// continuations. Coordinator and child buffers are separately admitted;
    /// changing the physical schedule grants no extra work or scoring authority.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_classify_layer_major_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Arc<ClassificationPlanner>, config: ClassificationCorpusConfig,
        limits: CorpusLimits, kernel: crate::native_engine::simd::Int8Kernel,
        compute_threads: usize, prefill: crate::native_engine::strict_int8::prefill::Int8PrefillLimits,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        scored::run(self, model, ClassificationStream { planner, config }, limits,
            Some(prefill), Some((kernel, compute_threads)), reader, writer, cancellation)
    }
}

struct ClassificationStream {
    planner: Arc<ClassificationPlanner>,
    config: ClassificationCorpusConfig,
}
impl scored::CorpusTask for ClassificationStream {
    fn identity(&self) -> &ExecutionIdentity { &self.config.identity }
    fn validate(&self, _model: &ArtifactIdentity, limits: CorpusLimits, native: Int8MemoryRequirement) -> Result<u64, HostedError> {
        validate_work(self.config.max_model_work)?;
        output_ceiling(self.config.task_ceiling, limits.transport.max_output_line_bytes)?;
        if native.kv_bytes > self.config.task_ceiling.max_kv_bytes {
            return Err(HostedError::Limits("classification corpus whole KV allocation"));
        }
        encoding_workspace_bytes(self.planner.document_encoding())
    }
    #[allow(clippy::too_many_arguments)]
    fn execute<R: BufRead, W: Write>(self, engine: &mut crate::native_engine::strict_int8::StrictInt8Engine<'_>,
        admission: CorpusAdmission<'_>, prefill: Option<crate::native_engine::strict_int8::prefill::Int8PrefillLimits>,
        reader: &mut R, writer: &mut W, limits: BatchLimits, defer: bool,
        control: &mut RunControl) -> Result<scored::Completion, HostedError> {
        let Self { planner, config } = self;
        let output_bytes = output_ceiling(config.task_ceiling, limits.max_output_line_bytes)?;
        let compiler = Int8ClassificationBatchPlanner::new_with_document_encoding(&planner, config.identity,
            config.task_ceiling, config.planning, config.defaults).map_err(HostedError::BatchSetup)?;
        let admission = ClassificationAdmission { output_bytes, inner: admission };
        let mut processor = match prefill {
            Some(prefill) => NativeInt8ClassificationBatch::new_layer_major(compiler, engine,
                admission, config.max_model_work, prefill),
            None => NativeInt8ClassificationBatch::new(compiler, engine, admission, config.max_model_work),
        }.map_err(HostedError::BatchSetup)?;
        scored::execute_stream(reader, writer, &mut processor, limits, defer, control)
    }
}

/// Transient tokenization is charged in addition to the caller's retained
/// prompt, transport and output buffers; a compact planner cannot discount it.
pub(super) fn encoding_workspace_bytes(encoding: &crate::tokenizer::untrusted::UntrustedEncoding)
    -> Result<u64, HostedError> {
    use crate::tokenizer::untrusted::UntrustedEncoding;
    encoding.validate().map_err(|_| HostedError::Limits("scored corpus input encoding"))?;
    match encoding {
        UntrustedEncoding::ByteFallback => Ok(0),
        UntrustedEncoding::SafeBpeV1(limits) => limits.requirements()
            .map(|required| required.workspace_bytes)
            .map_err(|_| HostedError::Limits("scored corpus tokenizer workspace")),
    }
}

struct ClassificationAdmission<'a> { inner: CorpusAdmission<'a>, output_bytes: u64 }
impl Int8ClassificationAdmission for ClassificationAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, identity: &ExecutionIdentity, work: Int8Work)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        if identity.task_spec != "classify-v1" { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        self.inner.admit_output(identity, work, self.inner.kv_bytes, 0, self.output_bytes)
    }
}
fn output_ceiling(task: TaskBudget, line_bytes: usize) -> Result<u64, HostedError> {
    task.validate().map_err(|_| HostedError::Limits("classification corpus task ceiling"))?;
    if task.max_output_bytes == 0 || task.max_output_bytes > line_bytes as u64 {
        return Err(HostedError::Limits("classification result exceeds transport line envelope"));
    }
    Ok(task.max_output_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn budget() -> TaskBudget {
        TaskBudget { max_input_tokens: 4096, max_output_tokens: 64,
            max_output_bytes: 8192, max_grammar_states: 4096, max_kv_bytes: 65536 }
    }
    #[test]
    fn admission_prices_task_ceiling_not_caller_ids_or_work_estimates() {
        assert_eq!(output_ceiling(budget(), 16384).unwrap(), 8192);
        assert_eq!(output_ceiling(budget(), 8192).unwrap(), 8192);
    }
    #[test]
    fn oversized_result_ceiling_refuses_before_streaming() {
        assert!(output_ceiling(budget(), 8191).is_err());
        assert!(output_ceiling(budget(), 0).is_err());
        let mut task = budget(); task.max_output_bytes = u64::MAX;
        assert!(output_ceiling(task, 8192).is_err());
    }
    #[test]
    fn zero_result_budget_cannot_bypass_reservation() {
        let mut task = budget(); task.max_output_bytes = 0;
        assert!(output_ceiling(task, 8192).is_err());
    }
    #[test]
    fn corpus_configuration_is_owned_send_data() {
        fn send<T: Send + 'static>() {}
        send::<ClassificationCorpusConfig>();
        send::<StreamInput<(Arc<ClassificationPlanner>, ClassificationCorpusConfig),
            std::io::Cursor<Vec<u8>>, Vec<u8>>>();
    }
}
