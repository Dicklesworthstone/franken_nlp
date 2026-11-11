//! Whole-document source tasks on one charged grouped-INT4 engine.
//! Every chunk is planned and admitted before the first native forward.
use super::*;
use crate::{
    native_engine::{constrained_int4, lmhead::NANBEIGE_VOCAB_SIZE,
        strict_int4::projection::Int4ProjectionExecutor},
    tasks::{ir::{PlanContext, TaskBudget},
        source_planning::{SourcePlanningLimits, SourceTaskPlanner,
            grouped_int4::long::{Int4SourceMapError, Int4SourceMapLimits, Int4SourceMapRun,
                PreparedInt4SourceMap, SourceMapTask}}},
    validation::grounded_fields::{SourceFieldEvidence, VerifiedSourceSpan},
};
mod summary;
mod question;
mod keyphrases;
mod cohort;
mod corpus;
mod team;
mod synthesis;
pub use corpus::int4_question_document_corpus_requirements;
#[cfg(all(feature = "metadata-store", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use corpus::job::int4_question_document_job_requirements;
pub(super) mod workspace;
pub(super) mod documents;
pub use documents::{Int4SourceDocumentCorpusRequirement, int4_source_document_corpus_requirements};

/// One original document, fixed task/model and nonrenewable work envelope.
/// Preparation prices all retained prompts/grammars, tokenizer/vocabulary
/// owners and configuration. Reduction reserve prices metadata/allocator
/// overhead beyond the explicit result/frontier/evidence payload floor.
pub struct Int4SourceMapConfig<T = SourceMapTask> {
    pub identity: ExecutionIdentity,
    pub task: T,
    pub budget: TaskBudget,
    pub planning: SourcePlanningLimits,
    pub mapping: Int4SourceMapLimits,
    pub native: NativeLimits,
    pub preparation_reserve_bytes: u64,
    pub reduction_reserve_bytes: u64,
}
impl<T> Int4SourceMapConfig<T> {
    /// Checked complete-frontier quote shared with CLI preflight. This is not
    /// a memory permit or an observed RSS limit; the host reserves it below.
    pub fn workspace_bytes(&self, chunks: usize) -> Result<u64, HostedError> {
        self.workspace().map(chunks)
    }
    fn workspace(&self) -> workspace::Workspace {
        workspace::Workspace { budget: self.budget, mapping: self.mapping,
            reserve_bytes: self.reduction_reserve_bytes }
    }
}
struct SourceMapInput<T = SourceMapTask> {
    source: String,
    planner: Arc<SourceTaskPlanner>,
    vocabulary: Arc<ExtractionVocabulary>,
    config: Int4SourceMapConfig<T>,
}
// Queued cancellation drops the complete input owner before its lease.
struct DocumentAdmission {
    input: Charged<SourceMapInput>,
    required: Int4MemoryRequirement,
    lease: EngineLease,
}

/// Typed final reducers share one admitted document and native execution path.
/// Their output remains distinct: selecting a physical team cannot change the
/// exact reducer, its ceilings or the final result validator.
trait DocumentOperation: Send + 'static {
    type Output: Send + 'static;
    fn validate(&self, config: &Int4SourceMapConfig) -> Result<(), HostedError>;
    fn allocation(&self, config: &Int4SourceMapConfig, prepared: &PreparedInt4SourceMap<'_>)
        -> Result<DocumentAllocation, HostedError>;
    fn execute(self, prepared: PreparedInt4SourceMap<'_>, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary,
        control: &mut RunControl) -> Result<Self::Output, HostedError>;
}
struct DocumentAllocation { workspace_bytes: u64, output_bytes: u64, output_tokens: u64 }
struct MapOperation;
impl DocumentOperation for MapOperation {
    type Output = Int4SourceMapRun;
    fn validate(&self, _: &Int4SourceMapConfig) -> Result<(), HostedError> { Ok(()) }
    fn allocation(&self, config: &Int4SourceMapConfig, prepared: &PreparedInt4SourceMap<'_>)
        -> Result<DocumentAllocation, HostedError> {
        let count = prepared.chunk_count();
        let output_tokens = (count as u64).checked_mul(u64::from(config.budget.max_output_tokens))
            .ok_or(HostedError::Limits("INT4 source map output tokens"))?;
        Ok(DocumentAllocation { workspace_bytes: config.workspace_bytes(count)?,
            output_bytes: prepared.max_result_bytes(), output_tokens })
    }
    fn execute(self, prepared: PreparedInt4SourceMap<'_>, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary,
        control: &mut RunControl) -> Result<Self::Output, HostedError> {
        let expected = prepared.preflight_metadata();
        let result = prepared.execute_with_control(admitted, engine, vocabulary, control).map_err(map_error)?;
        expected.verify_completed(&result).map_err(map_error)?;
        Ok(result)
    }
}

impl NlpEngine {
    /// Execute lossless document-wide NER, keyphrase or cited-summary maps.
    /// Output keeps independent chunk results plus original-document evidence;
    /// no global ranking or neural reconciliation is implied by this method.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int4_source(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4SourceMapRun>, HostedError> {
        let run = config.native.run;
        let admitted = admit_document(self, model, source, planner, vocabulary, config, &MapOperation)?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            execute_document(admitted, model.weights(), kernel, control, None, MapOperation)
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn admit_document<O: DocumentOperation>(host: &NlpEngine, model: &ResidentInt4, source: String,
    planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>, config: Int4SourceMapConfig,
    operation: &O) -> Result<DocumentAdmission, HostedError> {
    dispatch::preflight(host, config.native.run)?;
    if !Arc::ptr_eq(host.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
    constrained_int4::check_model(&config.identity, model.model_identity())
        .map_err(|_| HostedError::ModelIdentity)?;
    let required = map_requirements(&config, source.len(), model.model_identity().group_width(), &planner)?;
    operation.validate(&config)?;
    let bytes = sum(&[source.capacity() as u64, config.preparation_reserve_bytes])?;
    let lease = host.resources().acquire_lease();
    let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
        || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
    Ok(DocumentAdmission { input, required, lease })
}

/// Both schedules use identical planning, source seals, work ledgers and final
/// reducer validation. Only projection scheduling changes, on the one native
/// engine that survives every chunk. The complete input/frontier/output owners
/// remain charged through errors and physical team drain.
fn execute_document<'weights, O: DocumentOperation>(admission: DocumentAdmission,
    weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel, control: &mut RunControl,
    team: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>, operation: O)
    -> Result<HostedOutput<O::Output>, HostedError> {
    let DocumentAdmission { input, required, lease } = admission;
    let config = &input.value.config;
    let width = weights.group_width();
    checkpoint(control)?;
    let context = PlanContext::new(&config.identity, config.budget)
        .map_err(|_| HostedError::Limits("INT4 source map task context"))?;
    let prepared = input.value.planner.plan_int4_map_with_control(&input.value.source,
        &config.task, config.budget, &context, config.planning, config.mapping, width, control)
        .map_err(map_error)?;
    let expected = prepared.preflight_metadata();
    let count = prepared.chunk_count();
    if prepared.group_width() != width || expected.group_width() != width
        || prepared.execution_identities().len() != count || required.kv_bytes > prepared.max_kv_bytes() {
        return Err(HostedError::ModelIdentity);
    }
    let allocation = operation.allocation(config, &prepared)?;
    let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers, allocation.workspace_bytes)?;
    let output = output_claim(&lease, allocation.output_bytes, allocation.output_tokens)?;
    let mut admitted = Vec::new();
    admitted.try_reserve_exact(count).map_err(|_| map_error(Int4SourceMapError::Allocation))?;
    for identity in prepared.execution_identities() {
        checkpoint(control)?;
        constrained_int4::check_model(identity, weights.identity()).map_err(|_| HostedError::ModelIdentity)?;
        admitted.push(identity.clone());
    }
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
    let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
        sum(&[required.rope_bytes, required.scratch_payload_bound, config.native.allocator_reserve_bytes])?)?;
    let mut engine = allocate_native(kv, scratch, || {
        let engine = StrictInt4Engine::new(weights, config.native.context_tokens, Int4MemoryBudget {
            max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
            max_scratch_payload_bytes: required.scratch_payload_bound,
        }).map_err(native_error)?.with_kernel(kernel);
        match team {
            Some(team) => engine.with_projection_executor(team).map_err(native_error),
            None => Ok(engine),
        }
    })?;
    let result = allocate(output, || {
        let result = operation.execute(prepared, &admitted, &mut engine.value, &input.value.vocabulary, control)?;
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
}

fn map_requirements(config: &Int4SourceMapConfig, source_bytes: usize, width: usize,
    planner: &SourceTaskPlanner) -> Result<Int4MemoryRequirement, HostedError> {
    let required = common_requirements(config, source_bytes, width, planner)?;
    let task = match &config.task {
        SourceMapTask::Ner(_) => "ner-v1", SourceMapTask::Keyphrases(_) => "keyphrases-v1",
        SourceMapTask::Summarize(_) => "summarize-v1",
    };
    if config.identity.task_spec != task {
        return Err(HostedError::ModelIdentity);
    }
    Ok(required)
}
fn common_requirements<T>(config: &Int4SourceMapConfig<T>, source_bytes: usize, width: usize,
    planner: &SourceTaskPlanner) -> Result<Int4MemoryRequirement, HostedError> {
    config.native.run.validate()?;
    config.budget.validate().map_err(|_| HostedError::Limits("INT4 source map task budget"))?;
    config.identity.validate().map_err(|_| HostedError::ModelIdentity)?;
    constrained_int4::check_profile(&config.identity, width).map_err(|_| HostedError::ModelIdentity)?;
    if config.identity.tokenizer_digest != planner.tokenizer_digest()
        || config.identity.template_digest != *planner.template_digest() {
        return Err(HostedError::ModelIdentity);
    }
    let required = Int4MemoryRequirement::for_context(config.native.context_tokens, width).map_err(native_error)?;
    let encoding_workspace = planner.map_document_encoding_workspace_bytes()
        .map_err(|_| HostedError::Limits("INT4 source map encoding admission"))?;
    if source_bytes == 0 || source_bytes > config.mapping.chunks.max_input_bytes
        || config.preparation_reserve_bytes == 0 || config.reduction_reserve_bytes == 0
        || config.preparation_reserve_bytes < encoding_workspace
        || config.native.allocator_reserve_bytes == 0
        || config.planning.max_context_tokens > config.native.context_tokens
        || required.kv_bytes > config.budget.max_kv_bytes
        || config.mapping.max_model_work.projected_logits < NANBEIGE_VOCAB_SIZE as u64 {
        return Err(HostedError::Limits("INT4 source map input, preparation or resident capacity"));
    }
    Ok(required)
}
fn map_error(error: Int4SourceMapError) -> HostedError { HostedError::Int4(Int4HostError::SourceMap(error)) }
fn check_clean(engine: &StrictInt4Engine<'_>) -> Result<(), HostedError> {
    if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
        return Err(HostedError::Int4(Int4HostError::SourceTask(
            crate::tasks::source_planning::grouped_int4::Int4SourceError::InvalidResult)));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
