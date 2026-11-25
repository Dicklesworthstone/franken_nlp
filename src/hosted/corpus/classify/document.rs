//! Complete-document classification composed with the resident INT8 host.
//! The shared stream owner retains the compiler, native claims and IO until
//! physical worker join; terminal publication uses the original run control.
use super::*;
use crate::{
    batch::classify::quantized::long::{
        CLASSIFICATION_DOCUMENT_CHUNK_METADATA_BYTES, Int8ClassificationDocumentAdmission,
        Int8ClassificationDocumentBatchAdmission, Int8ClassificationDocumentBatchPlanner,
        NativeInt8ClassificationDocumentBatch,
    },
    hosted::{Int8ClassificationMapHostConfig, Int8ClassificationMapMemoryRequirement},
    native_engine::{simd::Int8Kernel, strict_int8::prefill::Int8PrefillLimits},
    tasks::classify::quantized::long::Int8ClassificationMapLimits,
    tokenizer::untrusted::{SafeBpeWorkLimits, UntrustedEncoding},
};

/// Immutable source, label, chunk and aggregate limits for one resident run.
/// Per-record arguments may replace the defaults and narrow task budgets.
/// Whole-run native and tokenizer ceilings never renew at flush boundaries.
pub struct ClassificationDocumentCorpusConfig {
    pub identity: ExecutionIdentity,
    pub task_ceiling: TaskBudget,
    pub planning: ClassificationLimits,
    pub mapping: Int8ClassificationMapLimits,
    pub defaults: Option<ClassificationBatchArgs>,
    pub max_model_work: Int8Work,
    /// Required for compact input; absent for byte fallback. Actual failed
    /// preparation/reconstruction work is charged to this invocation allowance.
    pub max_tokenizer_work: Option<SafeBpeWorkLimits>,
}

impl ClassificationDocumentCorpusConfig {
    /// Check fixed labels, limits, exact compiler identity and both invocation
    /// work ceilings before private population or native weight IO.
    pub fn validate(&self, planner: &ClassificationPlanner) -> Result<(), HostedError> {
        Int8ClassificationDocumentBatchPlanner::validate_configuration(planner, &self.identity,
            self.task_ceiling, self.planning, self.mapping, self.defaults.as_ref(),
            self.max_model_work, self.max_tokenizer_work).map_err(HostedError::BatchSetup)
    }
}

/// Complete simultaneous reservation, excluding loaded weights, cleanup and
/// an optional physical prefill/team schedule. Map preparation and transport
/// preparation have distinct owners and are deliberately additive.
///
/// The shared host adds transport to the returned task-only charge internally;
/// callers using this complete quote must not add transport a second time.
/// This models memory ownership rather than promising an OS RSS bound.
pub fn int8_classification_document_corpus_requirements(task: TaskBudget,
    planning: ClassificationLimits, mapping: Int8ClassificationMapLimits,
    encoding: &UntrustedEncoding, limits: CorpusLimits)
    -> Result<Int8ClassificationMapMemoryRequirement, HostedError> {
    let config = Int8ClassificationMapHostConfig { native: limits.native, planning, mapping,
        preparation_reserve_bytes: limits.preparation_reserve_bytes };
    let mut required = config.requirements_with_encoding(task, encoding)?;
    if required.max_output_bytes > limits.transport.max_output_line_bytes as u64 {
        return Err(HostedError::Limits("complete classification result exceeds corpus output line"));
    }
    let metadata = (mapping.chunks.max_chunks as u64)
        .checked_mul(CLASSIFICATION_DOCUMENT_CHUNK_METADATA_BYTES)
        .ok_or(HostedError::Limits("classification corpus identity storage"))?;
    required.input_bytes = sum(&[required.input_bytes, limits.reservation_bytes()?, metadata])?;
    sum(&[required.input_bytes, required.native.kv_bytes,
        required.workspace_bytes, required.output_reservation_bytes])?;
    Ok(required)
}

impl NlpEngine {
    /// Stream complete original documents through the exact native classifier
    /// and source-weighted reducer. One model/cache serves the full run.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_classification_documents<R, W>(&self, model: &ResidentInt8,
        planner: Arc<ClassificationPlanner>, config: ClassificationDocumentCorpusConfig,
        limits: CorpusLimits, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        let task = ClassificationDocumentStream::new(planner, config)?;
        scored::run(self, model, task, limits, None, None, reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_classification_documents_layer_major<R, W>(&self, model: &ResidentInt8,
        planner: Arc<ClassificationPlanner>, config: ClassificationDocumentCorpusConfig,
        limits: CorpusLimits, prefill: Int8PrefillLimits,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        let task = ClassificationDocumentStream::new(planner, config)?;
        scored::run(self, model, task, limits, Some(prefill), None, reader, writer, cancellation)
    }

    /// All chunks and label heads share one persistent projection team.
    /// The terminal frame is withheld until every worker physically joins.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_classification_documents_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Arc<ClassificationPlanner>, config: ClassificationDocumentCorpusConfig,
        limits: CorpusLimits, kernel: Int8Kernel, compute_threads: usize,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        let task = ClassificationDocumentStream::new(planner, config)?;
        scored::run(self, model, task, limits, None, Some((kernel, compute_threads)),
            reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_classification_documents_layer_major_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Arc<ClassificationPlanner>, config: ClassificationDocumentCorpusConfig,
        limits: CorpusLimits, kernel: Int8Kernel, compute_threads: usize, prefill: Int8PrefillLimits,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        let task = ClassificationDocumentStream::new(planner, config)?;
        scored::run(self, model, task, limits, Some(prefill), Some((kernel, compute_threads)),
            reader, writer, cancellation)
    }
}

struct ClassificationDocumentStream { compiler: Arc<Int8ClassificationDocumentBatchPlanner> }
impl ClassificationDocumentStream {
    fn new(planner: Arc<ClassificationPlanner>, config: ClassificationDocumentCorpusConfig)
        -> Result<Self, HostedError> {
        let compiler = Int8ClassificationDocumentBatchPlanner::new(planner, config.identity,
            config.task_ceiling, config.planning, config.mapping, config.defaults,
            config.max_model_work, config.max_tokenizer_work).map_err(HostedError::BatchSetup)?;
        Ok(Self { compiler: Arc::new(compiler) })
    }
}
impl scored::CorpusTask for ClassificationDocumentStream {
    fn identity(&self) -> &ExecutionIdentity { self.compiler.base_execution_identity() }
    fn retain_for_join(&self) -> Option<Box<dyn Send>> {
        // Retain the actual compiler AND original defaults allocations, not
        // newly cloned strings whose capacities differ from admitted owners.
        Some(Box::new(Arc::clone(&self.compiler)))
    }
    fn validate(&self, _model: &ArtifactIdentity, limits: CorpusLimits,
        native: Int8MemoryRequirement) -> Result<u64, HostedError> {
        let required = int8_classification_document_corpus_requirements(self.compiler.task_budget(),
            self.compiler.planning_limits(), self.compiler.mapping_limits(),
            self.compiler.document_encoding(), limits)?;
        if native != required.native
            || self.compiler.retained_defaults_bytes().map_err(HostedError::BatchSetup)?
                > limits.preparation_reserve_bytes {
            return Err(HostedError::Limits("classification corpus transferred owners"));
        }
        required.input_bytes.checked_sub(limits.reservation_bytes()?)
            .ok_or(HostedError::Limits("classification corpus preparation arithmetic"))
    }
    #[allow(clippy::too_many_arguments)]
    fn execute<R: BufRead, W: Write>(self, engine: &mut crate::native_engine::strict_int8::StrictInt8Engine<'_>,
        admission: CorpusAdmission<'_>, prefill: Option<Int8PrefillLimits>,
        reader: &mut R, writer: &mut W, limits: BatchLimits, defer: bool,
        control: &mut RunControl) -> Result<scored::Completion, HostedError> {
        let mapping = self.compiler.mapping_limits();
        let output_bytes = document_output_reservation(mapping.max_result_bytes)?;
        let admission = DocumentAdmission { inner: admission, mapping,
            ceiling: self.compiler.task_budget(), max_heads: self.compiler.planning_limits().max_labels,
            output_bytes };
        let mut processor = match prefill {
            Some(prefill) => NativeInt8ClassificationDocumentBatch::new_layer_major(self.compiler,
                engine, admission, prefill),
            None => NativeInt8ClassificationDocumentBatch::new(self.compiler, engine, admission),
        }.map_err(HostedError::BatchSetup)?;
        scored::execute_stream(reader, writer, &mut processor, limits, defer, control)
    }
}

fn document_output_reservation(bytes: u64) -> Result<u64, HostedError> {
    bytes.checked_mul(8).and_then(|bytes| bytes.checked_add(4096))
        .ok_or(HostedError::Limits("classification corpus complete output arithmetic"))
}

struct DocumentAdmission<'a> {
    inner: CorpusAdmission<'a>,
    mapping: Int8ClassificationMapLimits,
    ceiling: TaskBudget,
    max_heads: usize,
    output_bytes: u64,
}
impl Int8ClassificationDocumentBatchAdmission for DocumentAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int8ClassificationDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Pending), BatchItemFailure> {
        check_document_capacity(self.mapping, self.ceiling, self.max_heads,
            self.inner.kv_bytes, &request)?;
        for identity in request.identities {
            if identity.task_spec != "classify-v1" {
                return Err(BatchItemFailure::fatal(BatchCode::Admission));
            }
            check_request(self.inner.model, self.inner.kv_bytes, 0, self.inner.output_bytes,
                identity, request.model_work, request.kv_reservation_bytes, 0, request.max_result_bytes)?;
        }
        let guard = Pending::reserve(self.inner.lease, MemoryClass::JobBuffers, self.output_bytes)
            .map_err(admission_failure)?;
        let mut admitted = Vec::new();
        admitted.try_reserve_exact(request.identities.len())
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
        admitted.extend(request.identities.iter().cloned());
        Ok((admitted, guard))
    }
}
fn check_document_capacity(mapping: Int8ClassificationMapLimits, ceiling: TaskBudget, max_heads: usize,
    kv: u64, request: &Int8ClassificationDocumentAdmission<'_>) -> Result<(), BatchItemFailure> {
    let task = request.task_budget;
    if request.identities.is_empty() || request.identities.len() > mapping.chunks.max_chunks
        || request.required_context == 0 || request.required_context > mapping.chunks.context_tokens
        || request.head_count == 0 || request.head_count > max_heads
        || request.kv_reservation_bytes != kv || kv > task.max_kv_bytes
        || request.max_result_bytes != mapping.max_result_bytes
        || task.max_input_tokens > ceiling.max_input_tokens
        || task.max_output_tokens > ceiling.max_output_tokens || task.max_output_bytes > ceiling.max_output_bytes
        || task.max_grammar_states > ceiling.max_grammar_states || task.max_kv_bytes > ceiling.max_kv_bytes
        || !work_fits(request.model_work, mapping.max_model_work) {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    Ok(())
}
fn work_fits(work: Int8Work, cap: Int8Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}

#[cfg(test)] mod tests;
