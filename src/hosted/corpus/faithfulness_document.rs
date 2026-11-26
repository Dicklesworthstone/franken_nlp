//! Complete-source faithfulness on the shared owned INT8 corpus host.
//! The exact source, claim, full evidence frontier and output remain admitted
//! through physical completion, including failed records and terminal drain.
use super::*;
use crate::{
    batch::judge::quantized::{FAITHFULNESS_DOCUMENT_CHUNK_METADATA_BYTES,
        Int8FaithfulnessDocumentAdmission, Int8FaithfulnessDocumentBatchAdmission,
        Int8FaithfulnessDocumentBatchPlanner, NativeInt8FaithfulnessDocumentBatch},
    hosted::{Int8FaithfulnessMapHostConfig, Int8FaithfulnessMapMemoryRequirement},
    native_engine::{simd::Int8Kernel, strict_int8::prefill::Int8PrefillLimits},
    tasks::{ir::TaskBudget, judge::{FaithfulnessPolicy, JudgeLimits, JudgePlanner,
        quantized::faithfulness_map::{Int8FaithfulnessMapLimits, MAX_FAITHFULNESS_MAP_HEADS}}},
    tokenizer::untrusted::{SafeBpeWorkLimits, UntrustedEncoding},
};

/// One fixed semantic policy for an ordered population of exact source/claim
/// pairs. Record and flush boundaries never renew either work allowance.
pub struct FaithfulnessDocumentCorpusConfig {
    pub identity: ExecutionIdentity,
    pub task_ceiling: TaskBudget,
    pub planning: JudgeLimits,
    pub mapping: Int8FaithfulnessMapLimits,
    pub policy: FaithfulnessPolicy,
    pub max_model_work: Int8Work,
    /// Required for compact input. Includes actual failed preparation work
    /// and rebuilding, in addition to the per-document shared allowance.
    pub max_tokenizer_work: Option<SafeBpeWorkLimits>,
}
impl FaithfulnessDocumentCorpusConfig {
    pub fn validate(&self, planner: &JudgePlanner) -> Result<(), HostedError> {
        Int8FaithfulnessDocumentBatchPlanner::validate_configuration(planner, &self.identity,
            self.task_ceiling, self.planning, self.mapping, self.policy,
            self.max_model_work, self.max_tokenizer_work).map_err(HostedError::BatchSetup)
    }
}

/// Complete simultaneous owner quote, including corpus transport and every
/// retained chunk identity. Loaded weights, cleanup, prompt morsels and CPU
/// workers are separate charges; this is not an operating-system RSS cap.
pub fn int8_faithfulness_document_corpus_requirements(task: TaskBudget, planning: JudgeLimits,
    mapping: Int8FaithfulnessMapLimits, encoding: &UntrustedEncoding, limits: CorpusLimits)
    -> Result<Int8FaithfulnessMapMemoryRequirement, HostedError> {
    let config = Int8FaithfulnessMapHostConfig { native: limits.native, planning, mapping,
        preparation_reserve_bytes: limits.preparation_reserve_bytes };
    let mut required = config.requirements_with_encoding(task, encoding)?;
    if required.max_output_bytes > limits.transport.max_output_line_bytes as u64 {
        return Err(HostedError::Limits("complete faithfulness result exceeds corpus output line"));
    }
    let metadata = (mapping.chunks.max_chunks as u64)
        .checked_mul(FAITHFULNESS_DOCUMENT_CHUNK_METADATA_BYTES)
        .ok_or(HostedError::Limits("faithfulness corpus identity storage"))?;
    required.input_bytes = sum(&[required.input_bytes, limits.reservation_bytes()?, metadata])?;
    sum(&[required.input_bytes, required.native.kv_bytes,
        required.workspace_bytes, required.output_reservation_bytes])?;
    Ok(required)
}

impl NlpEngine {
    /// Every source is losslessly partitioned and every chunk sees its exact
    /// original claim. All evidence and the complete aggregate form one item;
    /// failed chunks produce no successful prefix. One resident model/cache
    /// and optional persistent team serve the entire ordered stream.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_faithfulness_documents_scheduled<R, W>(&self, model: &ResidentInt8,
        planner: Arc<JudgePlanner>, config: FaithfulnessDocumentCorpusConfig,
        limits: CorpusLimits, prefill: Option<Int8PrefillLimits>, team: Option<(Int8Kernel, usize)>,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        let compiler = Int8FaithfulnessDocumentBatchPlanner::new(planner, config.identity,
            config.task_ceiling, config.planning, config.mapping, config.policy,
            config.max_model_work, config.max_tokenizer_work).map_err(HostedError::BatchSetup)?;
        let task = FaithfulnessDocumentStream { compiler: Arc::new(compiler) };
        scored::run(self, model, task, limits, prefill, team, reader, writer, cancellation)
    }
}

struct FaithfulnessDocumentStream { compiler: Arc<Int8FaithfulnessDocumentBatchPlanner> }
impl scored::CorpusTask for FaithfulnessDocumentStream {
    fn identity(&self) -> &ExecutionIdentity { self.compiler.base_execution_identity() }
    fn retain_for_join(&self) -> Option<Box<dyn Send>> {
        // Retain the actual planner and configuration owner outside the
        // consuming numerical callback until every scoped child has joined.
        Some(Box::new(Arc::clone(&self.compiler)))
    }
    fn validate(&self, _model: &ArtifactIdentity, limits: CorpusLimits,
        native: Int8MemoryRequirement) -> Result<u64, HostedError> {
        let required = int8_faithfulness_document_corpus_requirements(self.compiler.task_budget(),
            self.compiler.planning_limits(), self.compiler.mapping_limits(),
            self.compiler.document_encoding(), limits)?;
        if native != required.native {
            return Err(HostedError::Limits("faithfulness corpus native geometry differs"));
        }
        required.input_bytes.checked_sub(limits.reservation_bytes()?)
            .ok_or(HostedError::Limits("faithfulness corpus preparation arithmetic"))
    }
    #[allow(clippy::too_many_arguments)]
    fn execute<R: BufRead, W: Write>(self, engine: &mut crate::native_engine::strict_int8::StrictInt8Engine<'_>,
        admission: CorpusAdmission<'_>, prefill: Option<Int8PrefillLimits>, reader: &mut R, writer: &mut W,
        limits: BatchLimits, defer: bool, control: &mut RunControl) -> Result<scored::Completion, HostedError> {
        let mapping = self.compiler.mapping_limits();
        let output_bytes = mapping.max_result_bytes.checked_mul(8).and_then(|n| n.checked_add(4096))
            .ok_or(HostedError::Limits("faithfulness corpus output arithmetic"))?;
        let admission = DocumentAdmission { inner: admission, mapping,
            ceiling: self.compiler.task_budget(), output_bytes };
        let mut processor = match prefill {
            Some(prefill) => NativeInt8FaithfulnessDocumentBatch::new_layer_major(self.compiler,
                engine, admission, prefill),
            None => NativeInt8FaithfulnessDocumentBatch::new(self.compiler, engine, admission),
        }.map_err(HostedError::BatchSetup)?;
        scored::execute_stream(reader, writer, &mut processor, limits, defer, control)
    }
}

struct DocumentAdmission<'a> {
    inner: CorpusAdmission<'a>,
    mapping: Int8FaithfulnessMapLimits,
    ceiling: TaskBudget,
    output_bytes: u64,
}
impl Int8FaithfulnessDocumentBatchAdmission for DocumentAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int8FaithfulnessDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Pending), BatchItemFailure> {
        check_capacity(self.mapping, self.ceiling, self.inner.kv_bytes, &request)?;
        for identity in request.identities {
            if identity.task_spec != "judge-v1" { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
            check_request(self.inner.model, self.inner.kv_bytes, 0, self.inner.output_bytes, identity,
                request.model_work, request.kv_reservation_bytes, 0, request.max_result_bytes)?;
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
fn check_capacity(mapping: Int8FaithfulnessMapLimits, ceiling: TaskBudget, kv: u64,
    request: &Int8FaithfulnessDocumentAdmission<'_>) -> Result<(), BatchItemFailure> {
    let work = request.model_work;
    let cap = mapping.max_model_work;
    if request.identities.is_empty() || request.identities.len() > mapping.chunks.max_chunks
        || request.required_context == 0 || request.required_context > mapping.chunks.context_tokens
        || request.max_head_count == 0 || request.max_head_count > MAX_FAITHFULNESS_MAP_HEADS
        || request.task_budget != ceiling || request.kv_reservation_bytes != kv || kv > ceiling.max_kv_bytes
        || request.max_result_bytes != mapping.max_result_bytes
        || work.forward_positions > cap.forward_positions || work.projected_logits > cap.projected_logits
        || work.attention_pairs > cap.attention_pairs || !work.projections.fits(cap.projections) {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    Ok(())
}

#[cfg(test)] mod tests;
