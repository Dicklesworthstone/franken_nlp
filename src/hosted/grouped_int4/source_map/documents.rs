//! Live ordered complete-source documents under one physical resident owner.
use super::*;
use std::io::{BufRead, Write};
use crate::{
    batch::{self as stream, BatchCode, BatchItemFailure, BatchSummary,
        source::grouped_int4::document::{self as document,
            SOURCE_DOCUMENT_CHUNK_METADATA_BYTES,
            Int4SourceDocumentAdmission, Int4SourceDocumentBatchAdmission,
            Int4SourceDocumentBatchLimits, Int4SourceDocumentBatchPlanner,
            Int4SourceDocumentTask, NativeInt4SourceDocumentBatch}},
    hosted::corpus::CorpusLimits,
    tasks::source_planning::grouped_int4::long::cohort::{
        Int4SourceMapCohortBudget, Int4SourceMapCohortMemoryLimits, Int4SourceMapCohortRequirement,
        hierarchy::Int4HierarchyCohortSchedule},
};
pub(in crate::hosted::grouped_int4) mod admission;
use admission::{check_encoding_preparation, DocumentAdmission};

/// Simultaneous invocation reservations, excluding resident model weights,
/// global runtime/preparation and RunLimits.cleanup_reserve_bytes. A pure quote
/// is not a permit or a measured RSS claim; the host acquires every owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SourceDocumentCorpusRequirement {
    pub native: Int4MemoryRequirement,
    /// RoPE, native activation scratch and the native allocator reserve.
    pub native_workspace_bytes: u64,
    /// Stream framing/JSON/ID buffers, caller IO and preparation, complete
    /// source storage, fixed task allocation capacities, frozen metadata and
    /// any additional simultaneous hierarchy preparation owners.
    pub input_bytes: u64,
    /// Complete map/reducer/evidence frontier and every permitted synthesis
    /// transcript/witness. This stays charged between documents and epochs.
    pub document_workspace_bytes: u64,
    /// Additional simultaneous cohort decoder workspace; its native fields
    /// already appear in `native` and are not charged a second time.
    pub cohort: Option<Int4SourceMapCohortRequirement>,
    /// Complete semantic result, canonical staging and retained token vectors;
    /// the output guard survives serialization, writer delivery and flush.
    pub output_reservation_bytes: u64,
    /// Checked sum of the preceding simultaneous owners and native KV bytes.
    pub total_bytes: u64,
}

/// Quote the full live corpus before model or source IO. No execution identity,
/// tokenizer, runtime or native allocation is needed. The task's exact retained
/// private allocation capacities are included, so this can be repeated after
/// bounded options/question loading without exposing their contents.
#[allow(clippy::too_many_arguments)]
pub fn int4_source_document_corpus_requirements(width: usize, task: &Int4SourceDocumentTask,
    budget: TaskBudget, planning: SourcePlanningLimits, mapping: Int4SourceMapLimits,
    run: Int4SourceDocumentBatchLimits, limits: CorpusLimits, reduction_reserve_bytes: u64,
    chunk_cohorts: bool, hierarchy: Option<Int4HierarchyCohortSchedule>)
    -> Result<Int4SourceDocumentCorpusRequirement, HostedError> {
    document::validate_limits(task, budget, planning, mapping, run).map_err(HostedError::BatchSetup)?;
    let buffers = limits.reservation_bytes()?;
    let scalar = Int4MemoryRequirement::for_context(limits.native.context_tokens, width)
        .map_err(native_error)?;
    let result_bytes = task.max_result_bytes(mapping);
    if limits.native.allocator_reserve_bytes == 0
        || planning.max_context_tokens != limits.native.context_tokens
        || scalar.kv_bytes != budget.max_kv_bytes
        || result_bytes == 0 || result_bytes > limits.transport.max_output_line_bytes as u64 {
        return Err(HostedError::Limits("INT4 source corpus complete resident and output envelope"));
    }
    let cohort = cohort_requirements(width, task, budget, planning, mapping, chunk_cohorts, hierarchy)?;
    let native = cohort.map_or(scalar, |quote| quote.native);
    if native.kv_bytes < scalar.kv_bytes || native.rope_bytes < scalar.rope_bytes
        || native.scratch_payload_bound < scalar.scratch_payload_bound {
        return Err(HostedError::Limits("INT4 source corpus incomplete phase peak"));
    }
    let metadata = (mapping.chunks.max_chunks as u64).checked_mul(SOURCE_DOCUMENT_CHUNK_METADATA_BYTES)
        .ok_or(HostedError::Limits("INT4 source corpus metadata arithmetic"))?;
    let hierarchy_preparation = hierarchy.map(|schedule|
        schedule.additional_preparation_bytes(limits.preparation_reserve_bytes).map_err(map_error))
        .transpose()?.unwrap_or(0);
    let input_bytes = sum(&[buffers, mapping.chunks.max_input_bytes as u64, metadata,
        task.retained_bytes().map_err(HostedError::BatchSetup)?, hierarchy_preparation])?;
    let native_workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
        limits.native.allocator_reserve_bytes])?;
    let document_workspace_bytes = workspace::document_bytes(task, budget, mapping, reduction_reserve_bytes)?;
    let output_tokens = task.output_token_ceiling(mapping, budget).map_err(HostedError::BatchSetup)?;
    let output_reservation_bytes = output_bytes(result_bytes, output_tokens)?;
    let total_bytes = sum(&[native.kv_bytes, native_workspace_bytes, input_bytes,
        document_workspace_bytes, cohort.map_or(0, |quote| quote.workspace_bytes), output_reservation_bytes])?;
    Ok(Int4SourceDocumentCorpusRequirement { native, native_workspace_bytes, input_bytes,
        document_workspace_bytes, cohort, output_reservation_bytes, total_bytes })
}

#[allow(clippy::too_many_arguments)]
fn cohort_requirements(width: usize, task: &Int4SourceDocumentTask, budget: TaskBudget,
    planning: SourcePlanningLimits, mapping: Int4SourceMapLimits, chunk_cohorts: bool,
    hierarchy: Option<Int4HierarchyCohortSchedule>)
    -> Result<Option<Int4SourceMapCohortRequirement>, HostedError> {
    if let Some(schedule) = hierarchy {
        let passes = match task {
            Int4SourceDocumentTask::SummaryHierarchy { limits, .. } => limits.max_passes,
            Int4SourceDocumentTask::QuestionHierarchy { limits, .. } => limits.max_passes,
            _ => return Err(HostedError::Limits("INT4 source corpus hierarchy task")),
        };
        schedule.validate(passes).map_err(map_error)?;
        if schedule.discovery_cohorts != chunk_cohorts {
            return Err(HostedError::Limits("INT4 source corpus immutable discovery schedule"));
        }
    }
    let discovery = if chunk_cohorts {
        if mapping.reduction.map_batch_chunks > mapping.chunks.max_chunks {
            return Err(HostedError::Limits("INT4 source corpus native chunk ceiling"));
        }
        Some(Int4SourceMapCohortMemoryLimits {
            max_chunks: mapping.reduction.map_batch_chunks, context_tokens: planning.max_context_tokens,
            max_output_tokens_per_chunk: u64::from(budget.max_output_tokens),
            max_result_bytes_per_chunk: budget.max_output_bytes,
        }.requirements(width).map_err(map_error)?)
    } else { None };
    if let Some(schedule) = hierarchy {
        return schedule.requirements(discovery, planning.max_context_tokens, width, budget)
            .map(Some).map_err(map_error);
    }
    let Some(quote) = discovery else { return Ok(None); };
    if matches!(task, Int4SourceDocumentTask::SummarySynthesis(_)
        | Int4SourceDocumentTask::SummaryHierarchy { .. }
        | Int4SourceDocumentTask::QuestionSynthesis(_)
        | Int4SourceDocumentTask::QuestionHierarchy { .. }) {
        quote.with_synthesis_context(planning.max_context_tokens, width, budget.max_kv_bytes)
            .map(Some).map_err(map_error)
    } else { Ok(Some(quote)) }
}

fn output_bytes(bytes: u64, tokens: u64) -> Result<u64, HostedError> {
    sum(&[bytes.checked_mul(4).ok_or(HostedError::Limits("INT4 source corpus output arithmetic"))?,
        tokens.checked_mul(8).ok_or(HostedError::Limits("INT4 source corpus token arithmetic"))?, 4096])
}

struct StreamInput<R, W> {
    preparation: Option<(Int4SourceDocumentBatchPlanner, Arc<ExtractionVocabulary>)>,
    reader: R,
    writer: W,
}

// The complete footer and private stream owners survive physical worker join.
// A team panic or terminal cancellation cannot publish run_complete early.
struct DeferredDocumentStream<R, W> {
    completion: stream::deferred::DeferredCompletion,
    input: Charged<StreamInput<R, W>>,
    frontier: Pending,
}

impl Int4SourceDocumentBatchPlanner {
    /// Complete task-specific frontier, shared with one-off and durable hosts.
    pub fn workspace_bytes(&self, reduction_reserve_bytes: u64) -> Result<u64, HostedError> {
        workspace::document_bytes(self.task(), self.task_budget(), self.document_limits(), reduction_reserve_bytes)
    }
}

impl NlpEngine {
    /// Process complete original documents without a retained job or metadata
    /// store. Fixed task settings, model, vocabulary, reader and writer belong
    /// to one blocking invocation. Every document preflights all discovery and
    /// reserved dependent passes; incomplete maps/hierarchies publish no result.
    ///
    /// All five native work axes, masks, verification and reducer allowances
    /// remain nonrenewable across failed attempts and flush epochs. A semantic
    /// result guard survives actual stream delivery. Cancellation returns only
    /// after native state and every captured private owner physically drain.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_source_documents<R, W>(&self, model: &ResidentInt4,
        planner: Int4SourceDocumentBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        limits: CorpusLimits, reduction_reserve_bytes: u64, kernel: PackedInt4Kernel,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_source_documents(model, planner, vocabulary, limits,
            reduction_reserve_bytes, kernel, None, reader, writer, cancellation)
    }

    /// Keep one sealed numerical team across every document, chunk, evidence
    /// continuation and flush. The serial semantic pipeline and its whole-run
    /// budgets are unchanged. Worker scratch is charged once for the complete
    /// invocation; run_complete is delivered only after every worker joins.
    /// Native chunk and hierarchy-group cohorts are separate schedules.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_source_documents_with_threads<R, W>(&self, model: &ResidentInt4,
        planner: Int4SourceDocumentBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        limits: CorpusLimits, reduction_reserve_bytes: u64, kernel: PackedInt4Kernel,
        compute_threads: usize, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_source_documents(model, planner, vocabulary, limits,
            reduction_reserve_bytes, kernel, Some(compute_threads), reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_source_documents<R, W>(&self, model: &ResidentInt4,
        planner: Int4SourceDocumentBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        limits: CorpusLimits, reduction_reserve_bytes: u64, kernel: PackedInt4Kernel,
        compute_threads: Option<usize>, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        match compute_threads {
            Some(width) => dispatch::preflight_int4_team(self, limits.native.run, width)?,
            None => dispatch::preflight(self, limits.native.run)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        if planner.group_width() != model.model_identity().group_width() { return Err(HostedError::ModelIdentity); }
        constrained_int4::check_model(planner.execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        check_encoding_preparation(planner.encoding_workspace_bytes().map_err(HostedError::BatchSetup)?,
            limits.preparation_reserve_bytes)?;
        let mapping = planner.document_limits();
        if compute_threads.is_some() && (planner.uses_native_cohorts()
            || mapping.reduction.map_batch_chunks != 1) {
            return Err(HostedError::Limits("INT4 source corpus teams require serial chunk scheduling"));
        }
        let required = int4_source_document_corpus_requirements(planner.group_width(), planner.task(),
            planner.task_budget(), planner.planning_limits(), mapping, planner.run_limits(), limits,
            reduction_reserve_bytes, planner.uses_chunk_cohorts(), planner.hierarchy_cohort_schedule())?;
        let actual_cohort = if planner.uses_native_cohorts() {
            Some(planner.cohort_requirements().map_err(HostedError::BatchSetup)?)
        } else { None };
        if actual_cohort != required.cohort {
            return Err(HostedError::Limits("INT4 source corpus complete native declaration"));
        }
        let team_scratch = compute_threads.map(|width|
            int4_task_team_workspace_bytes(width, planner.group_width())).transpose()?.unwrap_or(0);
        // The dispatcher owns this additional reservation; do not add it to
        // the scalar engine scratch guard and charge the same bytes twice.
        sum(&[required.total_bytes, team_scratch])?;
        let result_bytes = planner.max_result_bytes();
        let output_tokens = planner.output_token_ceiling().map_err(HostedError::BatchSetup)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(StreamInput { preparation: Some((planner, vocabulary)), reader, writer }))?;
        let frontier = Pending::reserve(&lease, MemoryClass::JobBuffers, required.document_workspace_bytes)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.native_workspace_bytes)?;
        let decoder = required.cohort.map(|quote| Pending::reserve(&lease,
            MemoryClass::JobBuffers, quote.workspace_bytes)).transpose()?;
        let model = model.clone();
        match compute_threads {
        None => dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            let frontier = frontier;
            checkpoint(control)?;
            let (planner, vocabulary) = input.value.preparation.take().ok_or(HostedError::CompletionMissing)?;
            let admission = DocumentAdmission { lease: &lease, model: model.model_identity(),
                base: planner.execution_identity(),
                kv_bytes: required.native.kv_bytes, mapping, result_bytes, output_tokens };
            let result = if let Some(quote) = required.cohort {
                let decoder = decoder.ok_or(HostedError::CompletionMissing)?;
                let budget = Int4SourceMapCohortBudget {
                    context_limit: limits.native.context_tokens,
                    native_memory: Int4MemoryBudget { max_kv_bytes: quote.native.kv_bytes,
                        max_rope_bytes: quote.native.rope_bytes,
                        max_scratch_payload_bytes: quote.native.scratch_payload_bound },
                    max_workspace_bytes: quote.workspace_bytes, max_result_bytes: quote.max_result_bytes,
                };
                let result = {
                    let mut processor = NativeInt4SourceDocumentBatch::new_cohorts(&planner,
                        &model.inner.loaded.value, &vocabulary, budget, kernel, admission)
                        .map_err(HostedError::BatchSetup)?;
                    stream::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                        &mut processor, limits.transport, control).map_err(HostedError::Batch)
                };
                // Exact per-window allocations are already destroyed. Keep
                // the whole componentwise peak charged until that point.
                drop(decoder); drop(scratch); drop(kv);
                result
            } else {
                let mut engine = allocate_native(kv, scratch, || {
                    StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens,
                        Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                            max_rope_bytes: required.native.rope_bytes,
                            max_scratch_payload_bytes: required.native.scratch_payload_bound })
                        .map(|engine| engine.with_kernel(kernel)).map_err(native_error)
                })?;
                let result = {
                    let mut processor = NativeInt4SourceDocumentBatch::new(&planner,
                        &mut engine.value, &vocabulary, admission).map_err(HostedError::BatchSetup)?;
                    stream::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                        &mut processor, limits.transport, control).map_err(HostedError::Batch)
                };
                drop(engine);
                result
            };
            drop(vocabulary); drop(planner);
            drop(input); // Actual reader/writer destruction precedes completion.
            drop(frontier); drop(lease);
            result
        }),
        Some(width) => dispatch::run_int4_team_with_finalize(self, limits.native.run, cancellation,
            model, width, kernel, move |weights, control, team| {
                let mut input = input;
                checkpoint(control)?;
                let (planner, vocabulary) = input.value.preparation.take().ok_or(HostedError::CompletionMissing)?;
                let mut engine = allocate_native(kv, scratch, || {
                    StrictInt4Engine::new(weights, limits.native.context_tokens,
                        Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                            max_rope_bytes: required.native.rope_bytes,
                            max_scratch_payload_bytes: required.native.scratch_payload_bound })
                        .map_err(native_error)?.with_kernel(kernel)
                        .with_projection_executor(team).map_err(native_error)
                })?;
                let admission = DocumentAdmission { lease: &lease, model: weights.identity(),
                    base: planner.execution_identity(), kv_bytes: required.native.kv_bytes,
                    mapping, result_bytes, output_tokens };
                let completion = {
                    let mut processor = NativeInt4SourceDocumentBatch::new(&planner,
                        &mut engine.value, &vocabulary, admission).map_err(HostedError::BatchSetup)?;
                    stream::deferred::run_ndjson_deferred(&mut input.value.reader, &mut input.value.writer,
                        &mut processor, limits.transport, control).map_err(HostedError::Batch)?
                };
                drop(engine); drop(vocabulary); drop(planner); drop(lease);
                Ok(DeferredDocumentStream { completion, input, frontier })
            }, |pending, control| {
                let DeferredDocumentStream { completion, mut input, frontier } = pending;
                let result = completion.publish(&mut input.value.writer, control).map_err(HostedError::Batch);
                drop(input); drop(frontier);
                result
            }),
        }
    }
}

#[cfg(test)] mod tests;
