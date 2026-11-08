//! A complete ordered source corpus on one charged grouped-INT4 model.

use super::*;
use crate::{
    batch::source::{SourceBatchArgs, grouped_int4::{self as source_batch,
        Int4SourceAdmission, Int4SourceBatchAdmission, Int4SourceBatchLimits,
        Int4SourceBatchPlanner, NativeInt4SourceBatch, Int4SourceCohortBatchLimits,
        Int4SourceCohortBatchStats, NativeInt4SourceCohortBatch,
        memory::Int4SourceCohortMemoryLimits}},
    native_engine::strict_int4::prefill::Int4PrefillLimits,
    tasks::{ir::TaskBudget, source_planning::{SourcePlanningLimits, SourceTaskPlanner}},
};

/// One closed task/model contract for the whole stream. Per-line task arguments
/// may tighten its ceilings, but cannot switch task, model, backend or groups.
/// Defaults may contain private passages and are intentionally not Debug.
pub struct Int4SourceCorpusConfig {
    pub identity: ExecutionIdentity,
    pub task_ceiling: TaskBudget,
    pub planning: SourcePlanningLimits,
    pub defaults: Option<SourceBatchArgs>,
    pub native_work: Int4SourceBatchLimits,
    /// Fixed all-legal-row ceiling for the serial or cohort stream. The compiler
    /// seals it into each actual source task before admission; rows cannot
    /// override it. None retains the full-vocabulary execution contract.
    pub selected_rows: Option<Int4JsonSparseLimits>,
}

/// Semantic results retain their existing source-task NDJSON envelopes.
/// Physical grouping counters describe this complete invocation separately.
pub struct Int4SourceCohortBatchResult {
    pub summary: BatchSummary,
    pub cohorts: Int4SourceCohortBatchStats,
}
impl Int4SourceCorpusConfig {
    fn validate(&self, planner: &SourceTaskPlanner, model: &Int4ModelIdentity,
        allocated_kv: u64) -> Result<(), HostedError> {
        constrained_int4::check_model(&self.identity, model).map_err(|_| HostedError::ModelIdentity)?;
        self.validate_planning(planner, model.group_width(), allocated_kv)
    }

    fn validate_planning(&self, planner: &SourceTaskPlanner, group_width: usize,
        allocated_kv: u64) -> Result<(), HostedError> {
        source_batch::check_configuration_with_document_encoding(planner, &self.identity, self.task_ceiling,
            self.planning, self.defaults.as_ref(), group_width).map_err(HostedError::BatchSetup)?;
        source_batch::validate_limits(self.native_work).map_err(HostedError::BatchSetup)?;
        if let Some(limits) = self.selected_rows {
            limits.validate().map_err(|_| HostedError::Limits("INT4 source corpus selected-row ceiling"))?;
        }
        if allocated_kv == 0 || allocated_kv > self.task_ceiling.max_kv_bytes {
            return Err(HostedError::Limits("INT4 source corpus complete KV reservation"));
        }
        Ok(())
    }
}

struct SourceInput<R, W> {
    preparation: Option<(Arc<SourceTaskPlanner>, Arc<ExtractionVocabulary>, Int4SourceCorpusConfig)>,
    reader: R,
    writer: W,
}

impl NlpEngine {
    /// Run bounded NER, keyphrases, cited summaries or passage QA as ordered
    /// NDJSON. For QA each line's `text` is the question and its typed passages
    /// supply the evidence. One model, engine, vocabulary and work reservation
    /// serve the entire stream, including all transport epochs.
    ///
    /// Preparation/IO reservations price the owned captures. The source
    /// planner's fixed encoder workspace is added automatically. Cancellation
    /// retains their charge through physical cleanup and writer disposal.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_source<R, W>(&self, model: &ResidentInt4,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceCorpusConfig, limits: CorpusLimits, kernel: PackedInt4Kernel,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_source_batch(model, planner, vocabulary, config, limits, kernel,
            None, reader, writer, cancellation)
    }

    /// Serve the same fixed NER, keyphrase, cited-summary or passage-QA stream
    /// with one persistent team of 2..=64 compute threads, including the
    /// coordinator. Complete source/passages, grammar legality, selected-row
    /// policy and independent task finalizers remain unchanged.
    ///
    /// One native engine and nonrenewable work/mask/checkpoint allowances span
    /// every document and flush. Worker scratch is charged separately. Owned
    /// IO survives physical worker join and final run_complete publication
    /// under the original run control; callbacks cannot reenter the host.
    /// This route does not combine with native cohorts or packed prefill.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_source_with_threads<R, W>(&self, model: &ResidentInt4,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceCorpusConfig, limits: CorpusLimits, kernel: PackedInt4Kernel,
        compute_threads: usize, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_source_batch(model, planner, vocabulary, config, limits, kernel,
            Some(compute_threads), reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_source_batch<R, W>(&self, model: &ResidentInt4,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceCorpusConfig, limits: CorpusLimits, kernel: PackedInt4Kernel,
        compute_threads: Option<usize>, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        match compute_threads {
            Some(width) => dispatch::preflight_int4_team(self, limits.native.run, width)?,
            None => dispatch::preflight(self, limits.native.run)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let width = model.model_identity().group_width();
        let required = requirements_with_document_encoding(width, config.native_work, limits,
            planner.document_encoding())?;
        config.validate(&planner, model.model_identity(), required.native.kv_bytes)?;
        required.check_team_workspace(compute_threads, width)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(SourceInput { preparation: Some((planner, vocabulary, config)), reader, writer }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let model = model.clone();
        match compute_threads {
        None => dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            checkpoint(control)?;
            let (planner, vocabulary, config) = input.value.preparation.take().ok_or(HostedError::CompletionMissing)?;
            let task_spec = config.identity.task_spec.clone();
            let compiler = Int4SourceBatchPlanner::new_with_document_encoding(&planner, config.identity, config.task_ceiling,
                config.planning, config.defaults, width).map_err(HostedError::BatchSetup)?;
            let compiler = match config.selected_rows {
                Some(limits) => compiler.with_selected_rows(limits).map_err(HostedError::BatchSetup)?,
                None => compiler,
            };
            let mut engine = allocate_native(kv, workspace, || {
                StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens, Int4MemoryBudget {
                    max_kv_bytes: required.native.kv_bytes, max_rope_bytes: required.native.rope_bytes,
                    max_scratch_payload_bytes: required.native.scratch_payload_bound,
                }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let admission = SourceAdmission { inner: CorpusAdmission {
                lease: &lease, model: model.model_identity(), kv_bytes: required.native.kv_bytes,
                masks: config.native_work.masks, output_bytes: limits.transport.max_output_line_bytes as u64,
                selected_rows: config.selected_rows,
            }, task_spec: &task_spec };
            let result = {
                let mut processor = NativeInt4SourceBatch::new_with_document_encoding(compiler, &mut engine.value,
                    &vocabulary, admission, config.native_work).map_err(HostedError::BatchSetup)?;
                stream::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                    &mut processor, limits.transport, control).map_err(HostedError::Batch)
            };
            drop(engine);
            drop(vocabulary);
            drop(planner);
            drop(input);
            drop(lease);
            result
        }),
        Some(threads) => dispatch::run_int4_team_with_finalize(self, limits.native.run, cancellation,
            model, threads, kernel, move |weights, control, team| {
                let mut input = input;
                checkpoint(control)?;
                let (planner, vocabulary, config) = input.value.preparation.take().ok_or(HostedError::CompletionMissing)?;
                let task_spec = config.identity.task_spec.clone();
                let compiler = Int4SourceBatchPlanner::new_with_document_encoding(&planner, config.identity, config.task_ceiling,
                    config.planning, config.defaults, width).map_err(HostedError::BatchSetup)?;
                let compiler = match config.selected_rows {
                    Some(limits) => compiler.with_selected_rows(limits).map_err(HostedError::BatchSetup)?,
                    None => compiler,
                };
                let mut engine = allocate_native(kv, workspace, || {
                    StrictInt4Engine::new(weights, limits.native.context_tokens, Int4MemoryBudget {
                        max_kv_bytes: required.native.kv_bytes, max_rope_bytes: required.native.rope_bytes,
                        max_scratch_payload_bytes: required.native.scratch_payload_bound,
                    }).map_err(native_error)?.with_kernel(kernel)
                        .with_projection_executor(team).map_err(native_error)
                })?;
                let admission = SourceAdmission { inner: CorpusAdmission {
                    lease: &lease, model: weights.identity(), kv_bytes: required.native.kv_bytes,
                    masks: config.native_work.masks, output_bytes: limits.transport.max_output_line_bytes as u64,
                    selected_rows: config.selected_rows,
                }, task_spec: &task_spec };
                let completion = {
                    let mut processor = NativeInt4SourceBatch::new_with_document_encoding(compiler, &mut engine.value,
                        &vocabulary, admission, config.native_work).map_err(HostedError::BatchSetup)?;
                    stream::deferred::run_ndjson_deferred(&mut input.value.reader, &mut input.value.writer,
                        &mut processor, limits.transport, control).map_err(HostedError::Batch)?
                };
                drop(engine);
                drop(vocabulary);
                drop(planner);
                drop(lease);
                Ok(DeferredStream { completion, input })
            }, |pending, control| pending.publish(control, |input| &mut input.writer)),
        }
    }

    /// Run NER, keyphrases, cited summaries or passage QA in bounded native
    /// cohorts. Task kind stays fixed across the corpus, while source bytes,
    /// passages, options and tighter per-row budgets remain independent.
    ///
    /// The whole window plus one lookahead is priced before input IO. Native
    /// engines use exact row contexts within the declared maximum reservation;
    /// neither a document nor an explicit flush renews invocation work limits.
    /// All source finalizers complete before any successful group is delivered.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_source_cohorts<R, W>(&self, model: &ResidentInt4,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceCorpusConfig, limits: CorpusLimits, max_documents: usize,
        kernel: PackedInt4Kernel, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<Int4SourceCohortBatchResult, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.batch_int4_source_cohorts_for_schedule(model, planner, vocabulary, config, limits,
            max_documents, kernel, None, reader, writer, cancellation)
    }

    /// Explicit packed prompt/feedback execution for the same source-task
    /// corpus. Per-document grammars, evidence, finalizers and work are fixed;
    /// the extra packed workspace is charged once for the entire live window.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_source_cohorts_packed<R, W>(&self, model: &ResidentInt4,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceCorpusConfig, limits: CorpusLimits, max_documents: usize,
        kernel: PackedInt4Kernel, prefill: Int4PrefillLimits,
        reader: R, writer: W, cancellation: CancellationToken)
        -> Result<Int4SourceCohortBatchResult, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.batch_int4_source_cohorts_for_schedule(model, planner, vocabulary, config, limits,
            max_documents, kernel, Some(prefill), reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_int4_source_cohorts_for_schedule<R, W>(&self, model: &ResidentInt4,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceCorpusConfig, limits: CorpusLimits, max_documents: usize,
        kernel: PackedInt4Kernel, prefill: Option<Int4PrefillLimits>,
        reader: R, writer: W, cancellation: CancellationToken)
        -> Result<Int4SourceCohortBatchResult, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let width = model.model_identity().group_width();
        let memory = Int4SourceCohortMemoryLimits {
            max_documents, context_tokens: limits.native.context_tokens,
            preparation_reserve_bytes: limits.preparation_reserve_bytes,
            io_reserve_bytes: limits.io_reserve_bytes,
            allocator_reserve_bytes: limits.native.allocator_reserve_bytes,
        };
        let required = memory.requirements_for_execution_with_document_encoding(width, config.native_work, limits.transport,
            config.selected_rows, prefill, planner.document_encoding()).map_err(HostedError::BatchSetup)?;
        // TaskBudget belongs to one document. The aggregate native allocation
        // must never be mistaken for authority to widen that document's KV.
        let row_kv = Int4MemoryRequirement::for_context(limits.native.context_tokens, width)
            .map_err(native_error)?.kv_bytes;
        config.validate(&planner, model.model_identity(), row_kv)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(SourceInput { preparation: Some((planner, vocabulary, config)), reader, writer }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let native = Int4SourceCohortBatchLimits {
            max_documents, context_limit: limits.native.context_tokens,
            native_memory: Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                max_rope_bytes: required.native.rope_bytes,
                max_scratch_payload_bytes: required.native.scratch_payload_bound },
            max_workspace_bytes: required.decode_workspace_bytes,
            max_item_result_bytes: limits.transport.max_output_line_bytes as u64,
            max_result_bytes: required.max_result_bytes,
        };
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            let native_guards = (kv, workspace);
            checkpoint(control)?;
            let (planner, vocabulary, config) = input.value.preparation.take().ok_or(HostedError::CompletionMissing)?;
            let task_spec = config.identity.task_spec.clone();
            let compiler = Int4SourceBatchPlanner::new_with_document_encoding(&planner, config.identity, config.task_ceiling,
                config.planning, config.defaults, width).map_err(HostedError::BatchSetup)?;
            let compiler = match config.selected_rows {
                Some(limits) => compiler.with_selected_rows(limits).map_err(HostedError::BatchSetup)?,
                None => compiler,
            };
            let admission = SourceCohortAdmission { lease: &lease, model: model.model_identity(),
                context_limit: limits.native.context_tokens, masks: config.native_work.masks,
                output_bytes: limits.transport.max_output_line_bytes as u64, task_spec: &task_spec,
                selected_rows: config.selected_rows };
            let mut processor = match prefill {
                Some(prefill) => NativeInt4SourceCohortBatch::new_packed_with_document_encoding(compiler, &model.inner.loaded.value,
                    &vocabulary, admission, config.native_work, native, kernel, prefill),
                None => NativeInt4SourceCohortBatch::new_with_document_encoding(compiler, &model.inner.loaded.value,
                    &vocabulary, admission, config.native_work, native, kernel),
            }.map_err(HostedError::BatchSetup)?;
            let result = stream::grouped::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                &mut processor, limits.transport, stream::grouped::GroupLimits { max_records: max_documents }, control)
                .map_err(HostedError::Batch);
            let cohorts = processor.cohort_stats();
            drop(processor); drop(vocabulary); drop(planner); drop(input); drop(native_guards); drop(lease);
            result.map(|summary| Int4SourceCohortBatchResult { summary, cohorts })
        })
    }
}

struct SourceAdmission<'a> { inner: CorpusAdmission<'a>, task_spec: &'a str }
impl Int4SourceBatchAdmission for SourceAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int4SourceAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        self.inner.admit_for_task(request, self.task_spec)
    }
}

struct SourceCohortAdmission<'a> {
    lease: &'a EngineLease,
    model: &'a Int4ModelIdentity,
    context_limit: usize,
    masks: ExtractionMaskBudget,
    output_bytes: u64,
    task_spec: &'a str,
    selected_rows: Option<Int4JsonSparseLimits>,
}
impl Int4SourceBatchAdmission for SourceCohortAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int4SourceAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        constrained_int4::check_model(request.identity, self.model)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        let tokens = check_source_cohort_request(self.model.group_width(), self.context_limit,
            self.output_bytes, self.masks, &request, self.task_spec, self.selected_rows)?;
        let guard = output_claim(self.lease, request.max_result_bytes, tokens).map_err(admission_failure)?;
        Ok((request.identity.clone(), guard))
    }
}
fn check_source_cohort_request(width: usize, context_limit: usize, output_bytes: u64,
    masks: ExtractionMaskBudget, request: &Int4SourceAdmission<'_>, task_spec: &str,
    selected_rows: Option<Int4JsonSparseLimits>) -> Result<u64, BatchItemFailure> {
    let exact_kv = request.model_work.forward_positions.checked_mul(KV_BYTES_PER_TOKEN as u64)
        .ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))?;
    if !matches!(task_spec, "ner-v1" | "keyphrases-v1" | "summarize-v1" | "answer-v1")
        || request.model_work.forward_positions > context_limit as u64 || request.kv_reservation_bytes != exact_kv {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    check_request_for_projection(width, exact_kv, output_bytes, masks, request, task_spec, selected_rows)
}

#[cfg(test)] mod tests;
