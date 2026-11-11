//! Complete-document classification on the owned durable job boundary.
//! One input document owns all chunk plans, scores and the final aggregate.
use super::*;
use crate::native_engine::strict_int4::projection::Int4ProjectionExecutor;
use crate::{
    batch::{BatchCode, BatchItemFailure, classify::ClassificationBatchArgs},
    jobs::{JobWork, runner::scored::grouped_int4::document::{
        Int4ClassificationDocumentAdmission, Int4ClassificationDocumentBatchAdmission,
        Int4ClassificationDocumentJobPlanner, Int4ClassificationDocumentJobProcessor}},
    tasks::classify::{ClassificationLabel, ClassificationLimits, ClassificationMode,
        grouped_int4::long::{Int4ClassificationMapCohortBudget, Int4ClassificationMapError,
            Int4ClassificationMapLimits, MAX_CLASSIFICATION_MAP_CHUNKS,
            memory::{Int4ClassificationMapCohortMemoryRequirement,
                Int4ClassificationMapMemoryLimits, Int4ClassificationMapMemoryRequirement}}},
};

struct DocumentInput<R> {
    preparation: Option<(Int4ClassificationDocumentJobPlanner, SourceJobRequest)>,
    reader: R,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DocumentRequirements {
    memory: Int4ClassificationMapMemoryRequirement,
    /// Complete original population, private factory and one map frontier.
    input_bytes: u64,
}

struct DocumentExecution<R> {
    input: Charged<DocumentInput<R>>,
    kv: Pending,
    scratch: Pending,
    limits: JobHostLimits,
    required: DocumentRequirements,
    cohort: Option<(usize, Int4ClassificationMapCohortMemoryRequirement)>,
    lease: EngineLease,
}

/// Cohort processors construct each exact window themselves. Their peak
/// native declarations remain owned here; no unused scalar cache is built.
enum DocumentNative<'w> {
    Serial(ChargedNative<StrictInt4Engine<'w>>),
    Cohorts { budget: Int4ClassificationMapCohortBudget, _kv: Pending, _scratch: Pending },
}

impl NlpEngine {
    /// Start or resume classification of complete original documents with
    /// fixed private labels, mode and decision policy. Each durable result
    /// contains every chunk and its separately named aggregate estimator.
    ///
    /// The immutable factory and original population authenticate before
    /// execution or tail repair. The existing runner persists all five native
    /// work counters before each complete attempt and holds the result claim
    /// through spool sync and journal acknowledgement. Explicit cohort mode
    /// shares projections without changing the complete-document commit unit.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_classification_document<R>(&self, model: &ResidentInt4,
        planner: Int4ClassificationDocumentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_classification_document_job(model, planner, request, limits, kernel, None,
            reader, cancellation)
    }

    /// Keep one persistent numerical team across every complete-document
    /// attempt. Worker width is operational: authenticated recipes, result
    /// identities and all lifetime work debits remain unchanged on resume.
    /// Worker scratch is reserved before population IO and every worker joins
    /// before this method returns. Native chunk cohorts cannot be combined.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_classification_document_with_threads<R>(&self, model: &ResidentInt4,
        planner: Int4ClassificationDocumentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, compute_threads: usize, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_classification_document_job(model, planner, request, limits, kernel, Some(compute_threads),
            reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_classification_document_job<R>(&self, model: &ResidentInt4,
        planner: Int4ClassificationDocumentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, compute_threads: Option<usize>, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        match compute_threads {
            Some(width) => dispatch::preflight_int4_team(self, limits.native.run, width)?,
            None => dispatch::preflight(self, limits.native.run)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) {
            return Err(HostedError::ResourceDomain.into());
        }
        constrained_int4::check_model(planner.execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        if planner.group_width() != model.model_identity().group_width()
            || planner.execution_identity().task_spec != "classify-v1" {
            return Err(HostedError::ModelIdentity.into());
        }
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        check_team_schedule(compute_threads, planner.cohort_chunks())?;
        let cohort = planner.cohort_chunks().map(|chunks|
            planner.cohort_requirements().map(|quote| (chunks, quote)))
            .transpose().map_err(HostedError::BatchSetup)?;
        let mut required = requirements(planner.group_width(), planner.document_budget(),
            planner.planning_limits(), planner.document_limits(), planner.request(),
            request.limits, limits, cohort, request.mode.input_memory())?;
        let base_input_bytes = required.memory.input_bytes;
        required.memory = required.memory.with_input_encoding(planner.document_encoding()).map_err(map_error)?;
        required.input_bytes = sum(&[required.input_bytes, required.memory.input_bytes.checked_sub(base_input_bytes)
            .ok_or(HostedError::Limits("classification document encoding memory arithmetic"))?])?;
        let team_scratch = compute_threads.map(|width|
            int4_task_team_workspace_bytes(width, planner.group_width())).transpose()?.unwrap_or(0);
        sum(&[required.input_bytes, required.memory.native.kv_bytes, required.memory.workspace_bytes,
            required.memory.output_reservation_bytes, team_scratch])?;
        if planner.max_result_bytes() != required.memory.max_output_bytes {
            return Err(HostedError::Limits("INT4 classification document result envelope").into());
        }
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(DocumentInput { preparation: Some((planner, request)), reader }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.memory.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.memory.workspace_bytes)?;
        let execution = DocumentExecution { input, kv, scratch, limits, required, cohort, lease };
        let model = model.clone();
        let completed = match compute_threads {
            None => dispatch::run(self, limits.native.run, cancellation, move |control|
                execution.run(&model.inner.loaded.value, kernel, None, control)),
            Some(width) => dispatch::run_int4_team(self, limits.native.run, cancellation,
                model, width, kernel, move |weights, control, team|
                    execution.run(weights, kernel, Some(team), control)),
        }?;
        // Physical drain and terminal cancellation take precedence even if a
        // durable result committed; authenticated resume reconciles progress.
        completed.map_err(HostedJobError::Job)
    }
}

fn check_team_schedule(threads: Option<usize>, cohort_chunks: Option<usize>) -> Result<(), HostedError> {
    if threads.is_some() && cohort_chunks.is_some() {
        return Err(HostedError::Limits("INT4 classification document job team cannot combine with native chunk cohorts"));
    }
    Ok(())
}

impl<R: BufRead> DocumentExecution<R> {
    fn run<'weights>(self, weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel,
        projection: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        // Capture the complete transferred owner so queued cancellation drops
        // caller buffers, fixed labels and key before their charged storage.
        let Self { mut input, kv, scratch, limits, required, cohort, lease } = self;
        checkpoint(control)?;
        let (planner, request) = input.value.preparation.take()
            .ok_or(HostedError::CompletionMissing)?;
        let population = match read_population(&mut input.value.reader,
            &request, limits.transport, control) {
            Ok(population) => population,
            Err(error) => return Ok(Err(JobRunError::Storage(error))),
        };
        checkpoint(control)?;
        let memory = required.memory;
        let native_budget = Int4MemoryBudget { max_kv_bytes: memory.native.kv_bytes,
            max_rope_bytes: memory.native.rope_bytes,
            max_scratch_payload_bytes: memory.native.scratch_payload_bound };
        let mut native = if let Some((_, quote)) = cohort {
            DocumentNative::Cohorts { budget: Int4ClassificationMapCohortBudget {
                native: Int4RunBudget::exact(planner.document_limits().max_model_work),
                memory: native_budget, max_task_workspace_bytes: quote.task_workspace_bytes,
                max_output_bytes: quote.max_cohort_output_bytes,
            }, _kv: kv, _scratch: scratch }
        } else {
            DocumentNative::Serial(allocate_native(kv, scratch, || {
                let engine = StrictInt4Engine::new(weights, limits.native.context_tokens,
                    native_budget).map_err(native_error)?.with_kernel(kernel);
                match projection {
                    Some(team) => engine.with_projection_executor(team).map_err(native_error),
                    None => Ok(engine),
                }
            })?)
        };
        let admission = DocumentAdmission { lease: &lease, model: weights.identity(),
            base: planner.execution_identity(), document: planner.document_limits(),
            kv_bytes: memory.native.kv_bytes, head_count: head_count(planner.request()),
            result_bytes: memory.max_output_bytes, output_bytes: memory.output_reservation_bytes };
        let result = (|| {
            match &mut native {
                DocumentNative::Serial(engine) => {
                    let processor = Int4ClassificationDocumentJobProcessor::new(
                        &planner, &mut engine.value, admission).map_err(JobRunError::Processor)?;
                    run_population(request, &population, processor, control)
                }
                DocumentNative::Cohorts { budget, .. } => {
                    let processor = Int4ClassificationDocumentJobProcessor::new_cohorts(
                        &planner, weights, *budget, kernel, admission)
                        .map_err(JobRunError::Processor)?;
                    run_population(request, &population, processor, control)
                }
            }
        })();
        // Runner/processor and per-result guards have physically dropped
        // before their population, native state and complete-map charge.
        drop(native); drop(population); drop(planner); drop(input); drop(lease);
        Ok(result)
    }
}

fn memory_limits(host: JobHostLimits) -> Int4ClassificationMapMemoryLimits {
    Int4ClassificationMapMemoryLimits { context_tokens: host.native.context_tokens,
        preparation_reserve_bytes: host.preparation_reserve_bytes,
        allocator_reserve_bytes: host.native.allocator_reserve_bytes }
}

#[allow(clippy::too_many_arguments)]
fn requirements(width: usize, task: TaskBudget, planning: ClassificationLimits,
    document: Int4ClassificationMapLimits, settings: &ClassificationBatchArgs,
    job: JobLimits, host: JobHostLimits,
    cohort: Option<(usize, Int4ClassificationMapCohortMemoryRequirement)>, input: JobInputMemory)
    -> Result<DocumentRequirements, HostedJobError> {
    let population_bytes = host.required_buffer_bytes_for_input(job, input)?;
    if settings.budget != task || settings.labels.is_empty()
        || settings.labels.len() > planning.max_labels
        || host.native.context_tokens != document.chunks.context_tokens
        || job.max_work.mask_node_visits != 0
        || !JobWork::from_int4(document.max_model_work, 0).fits(job.max_work) {
        return Err(HostedError::Limits("INT4 classification document job envelope").into());
    }
    let quote = memory_limits(host);
    let memory = if let Some((chunks, frozen)) = cohort {
        let required = quote.cohort_requirements(width, task, planning, document,
            head_count(settings), chunks).map_err(map_error)?;
        // Resume permits changed host preparation/allocator allowances. The
        // current complete quote owns those bytes; the frozen physical native
        // geometry, task workspace and result envelopes must still match.
        if frozen.memory.native != required.memory.native
            || frozen.memory.max_output_bytes != required.memory.max_output_bytes
            || frozen.memory.output_reservation_bytes != required.memory.output_reservation_bytes
            || frozen.task_workspace_bytes != required.task_workspace_bytes
            || frozen.max_cohort_output_bytes != required.max_cohort_output_bytes {
            return Err(HostedError::Limits("INT4 classification document cohort memory binding").into());
        }
        required.memory
    } else { quote.requirements(width, task, planning, document).map_err(map_error)? };
    if memory.max_output_bytes > job.max_result_bytes as u64 {
        return Err(HostedError::Limits("INT4 classification document complete result capacity").into());
    }
    // Both quotes include the same private planner/defaults preparation
    // reserve. Keep it once while charging the population and live map owners
    // together. The cohort task workspace is already inside memory.input_bytes.
    let map_frontier = memory.input_bytes.checked_sub(host.preparation_reserve_bytes)
        .ok_or(HostedError::Limits("INT4 classification document preparation arithmetic"))?;
    let input_bytes = sum(&[population_bytes, map_frontier, spare_label_bytes(settings, planning)?])?;
    sum(&[input_bytes, memory.native.kv_bytes, memory.workspace_bytes, memory.output_reservation_bytes])?;
    Ok(DocumentRequirements { memory, input_bytes })
}

fn head_count(settings: &ClassificationBatchArgs) -> usize {
    if settings.mode == ClassificationMode::Exclusive { 1 } else { settings.labels.len() }
}

fn spare_label_bytes(settings: &ClassificationBatchArgs, planning: ClassificationLimits) -> Result<u64, HostedError> {
    let labels = (settings.labels.capacity().saturating_sub(planning.max_labels) as u64)
        .checked_mul(std::mem::size_of::<ClassificationLabel>() as u64)
        .ok_or(HostedError::Limits("INT4 classification document label capacity arithmetic"))?;
    let text = settings.labels.iter().try_fold(0_u64, |total, label|
        sum(&[total, label.id.capacity() as u64, label.description.capacity() as u64]))?;
    sum(&[labels, text.saturating_sub(planning.max_total_label_bytes as u64)])
}

struct DocumentAdmission<'a> {
    lease: &'a EngineLease,
    model: &'a Int4ModelIdentity,
    base: &'a ExecutionIdentity,
    document: Int4ClassificationMapLimits,
    kv_bytes: u64,
    head_count: usize,
    result_bytes: u64,
    output_bytes: u64,
}
impl Int4ClassificationDocumentBatchAdmission for DocumentAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int4ClassificationDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure> {
        check_request(self.model.group_width(), self.base, self.document, self.kv_bytes,
            self.head_count, self.result_bytes, &request)?;
        let mut identities = Vec::new();
        identities.try_reserve_exact(request.identities.len())
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        for identity in request.identities {
            constrained_int4::check_model(identity, self.model)
                .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
            identities.push(identity.clone());
        }
        let guard = Pending::reserve(self.lease, MemoryClass::JobBuffers, self.output_bytes)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        Ok((identities, guard))
    }
}

#[allow(clippy::too_many_arguments)]
fn check_request(width: usize, base: &ExecutionIdentity, document: Int4ClassificationMapLimits,
    kv_bytes: u64, head_count: usize, result_bytes: u64, request: &Int4ClassificationDocumentAdmission<'_>)
    -> Result<(), BatchItemFailure> {
    if base.task_spec != "classify-v1" || request.chunk_count == 0
        || request.chunk_count > document.chunks.max_chunks
        || request.chunk_count > MAX_CLASSIFICATION_MAP_CHUNKS
        || request.identities.len() != request.chunk_count
        || request.head_count == 0 || request.head_count != head_count
        || request.kv_reservation_bytes == 0 || request.kv_reservation_bytes != kv_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    for identity in request.identities {
        constrained_int4::check_profile(identity, width)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        if identity.task_spec != "classify-v1" || identity.tokenizer_digest != base.tokenizer_digest
            || identity.template_digest != base.template_digest {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
    }
    let work = request.model_work;
    if work.forward_positions == 0 || work.projected_logits == 0 || work.attention_pairs == 0
        || work.projections.dot_products == 0 || work.projections.multiply_accumulates == 0
        || !JobWork::from_int4(work, 0).fits(JobWork::from_int4(document.max_model_work, 0)) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    if request.max_result_bytes == 0 || request.max_result_bytes != result_bytes {
        return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit));
    }
    Ok(())
}

fn map_error(error: Int4ClassificationMapError) -> HostedError {
    HostedError::Int4(Int4HostError::ClassificationMap(error))
}

#[cfg(test)] mod tests;
