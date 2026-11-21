//! Durable populations of reconciled objects with complete-source review.
//! Each authenticated attempt owns every extraction chunk, grounding pass and
//! field review. No stage can acknowledge a partial result or renew its work.

use super::*;
use std::io::BufRead;
use crate::{
    batch::{BatchCode, BatchItemFailure, extract::semantic::document::{
        Int4SemanticObjectDocumentAdmission, Int4SemanticObjectDocumentBatchAdmission,
        Int4SemanticObjectDocumentBatchLimits, Int4SemanticObjectDocumentBatchPlanner}},
    hosted::corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest,
        Population, read_population, run_population},
    jobs::{JobInputMemory, JobLimits, JobProgress, JobWork,
        population::{JobPopulation, PopulationReadLimits}, runner::{JobRunError,
            extract::grouped_int4::semantic::document::{Int4SemanticObjectDocumentJobConfig,
                Int4SemanticObjectDocumentJobPlanner, Int4SemanticObjectDocumentJobProcessor,
                SEMANTIC_OBJECT_JOB_RECIPE_BYTES, check_job_limits as check_durable_limits}}},
};

/// Fixed whole-document geometry plus independent durable-storage owners.
/// The reader's retained allocation, journal cache and serializer overhead
/// remain explicit even when original input records are retained on disk.
#[derive(Clone, Copy, Debug)]
pub struct Int4SemanticObjectDocumentJobHostLimits {
    pub document: Int4SemanticObjectDocumentHostLimits,
    pub transport: PopulationReadLimits,
    pub io_reserve_bytes: u64,
    pub journal_reserve_bytes: u64,
    pub serialization_reserve_bytes: u64,
}
impl Int4SemanticObjectDocumentJobHostLimits {
    fn storage(self) -> JobHostLimits {
        JobHostLimits { native: self.document.native, transport: self.transport,
            preparation_reserve_bytes: self.document.preparation_reserve_bytes,
            io_reserve_bytes: self.io_reserve_bytes,
            journal_reserve_bytes: self.journal_reserve_bytes,
            serialization_reserve_bytes: self.serialization_reserve_bytes }
    }
}

/// Complete simultaneous job payload, excluding resident weights, caller
/// preparation and runtime cleanup. This quote is not an RSS measurement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SemanticObjectDocumentJobRequirements {
    pub native: Int4MemoryRequirement,
    pub workspace_bytes: u64,
    /// Whole-document preparation, retained fixed settings and recipe buffers;
    /// this component is already included in input_bytes.
    pub preparation_bytes: u64,
    /// Preparation plus population/index/parser, IO, journal and serialization.
    pub input_bytes: u64,
    /// Separate complete-result claim kept through durable acknowledgement.
    pub output_reservation_bytes: u64,
    /// One persistent projection team, separately reserved by dispatch.
    pub team_workspace_bytes: u64,
}
impl Int4SemanticObjectDocumentJobRequirements {
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        sum(&[self.native.kv_bytes, self.workspace_bytes, self.input_bytes,
            self.output_reservation_bytes, self.team_workspace_bytes])
    }
}

/// Pure admission before original schema, population, key or model IO.
/// Grounding is not a persisted JobWork axis, so its full allowance is funded
/// for every permitted attempt, including retries and interrupted attempts.
/// Input retention changes ownership, never the authenticated lifetime caps.
#[allow(clippy::too_many_arguments)]
pub fn int4_semantic_object_job_requirements(width: usize, extraction: TaskBudget,
    options: &Int4SemanticOptions, encoding: &UntrustedEncoding,
    work: Int4SemanticObjectDocumentBatchLimits, limits: Int4SemanticObjectDocumentJobHostLimits,
    job: JobLimits, input: JobInputMemory, compute_threads: Option<usize>)
    -> Result<Int4SemanticObjectDocumentJobRequirements, HostedJobError> {
    options.validate().map_err(semantic_error)?;
    let mapping = limits.document.extraction.mapping;
    work.validate_item(options.max_model_work, mapping.mapping.max_mask_visits, mapping.verification)
        .map_err(HostedError::BatchSetup)?;
    let job_buffers = limits.storage().required_buffer_bytes_for_input(job, input)?;
    if job.max_work != JobWork::from_int4(work.max_model_work, work.max_mask_node_visits) {
        return Err(HostedError::Limits("semantic object job lifetime native and mask work").into());
    }
    if options.limits.max_output_bytes > job.max_result_bytes as u64 {
        return Err(HostedError::Limits("semantic object job complete stored-result envelope").into());
    }
    let every_attempt = |n: u64| n.checked_mul(job.max_attempts)
        .ok_or(HostedError::Limits("semantic object job lifetime grounding arithmetic"));
    if every_attempt(mapping.verification.max_fields as u64)? > work.max_verification.max_fields as u64
        || every_attempt(mapping.verification.max_matches as u64)? > work.max_verification.max_matches as u64
        || every_attempt(mapping.verification.max_scan_steps)? > work.max_verification.max_scan_steps {
        return Err(HostedError::Limits("semantic object job grounding for every durable attempt").into());
    }
    let document = int4_semantic_object_document_requirements(width, extraction,
        options, encoding, limits.document, compute_threads)?;
    let staging = job_buffers.checked_sub(limits.document.preparation_reserve_bytes)
        .ok_or(HostedError::Limits("semantic object job preparation arithmetic"))?;
    let preparation_bytes = preparation_with_recipe(document.preparation_bytes,
        limits.document.compiler, options)?;
    let required = Int4SemanticObjectDocumentJobRequirements {
        native: document.native, workspace_bytes: document.workspace_bytes,
        preparation_bytes, input_bytes: sum(&[staging, preparation_bytes])?,
        output_reservation_bytes: document.output_reservation_bytes,
        team_workspace_bytes: document.team_workspace_bytes,
    };
    required.total_bytes()?;
    Ok(required)
}

fn preparation_with_recipe(document_bytes: u64, compiler: CompileLimits,
    options: &Int4SemanticOptions) -> Result<u64, HostedError> {
    // The input configuration, private recipe and immutable batch factory
    // coexist with the prepared document. Add their copies independently of
    // the original single-document peak and both canonical recipe buffers.
    // One sentinel byte covers bounded schema readers that retain cap + 1.
    sum(&[document_bytes, product(compiler.max_schema_bytes as u64, 2)?,
        product(options_storage_bytes(options)?, 2)?,
        product(SEMANTIC_OBJECT_JOB_RECIPE_BYTES as u64, 2)?, 1])
}

fn check_config(config: &Int4SemanticObjectDocumentJobConfig,
    limits: Int4SemanticObjectDocumentJobHostLimits, job: JobLimits) -> Result<(), HostedError> {
    if config.extraction.schema.is_empty()
        || config.extraction.schema.len() > config.compiler_limits.max_schema_bytes
        || config.compiler_limits != limits.document.compiler
        || config.source_limits != limits.document.source
        || !super::corpus::same_extraction_limits(config.extraction_limits, limits.document.extraction)
        || !super::corpus::same_review_limits(config.review_limits, limits.document.review)
        || !super::super::corpus::same_planning(config.judge_planning, limits.document.judge_planning) {
        return Err(HostedError::Limits("semantic object job fixed schema and host declarations"));
    }
    check_durable_limits(config, job).map_err(HostedError::BatchSetup)
}

struct JobInput<R> {
    config: Int4SemanticObjectDocumentJobConfig,
    request: Option<SourceJobRequest>,
    vocabulary: Option<Arc<ExtractionVocabulary>>,
    compiler: Option<Arc<Int4SemanticObjectDocumentBatchPlanner>>,
    population: Option<JobPopulation>,
    reader: R,
}
/// Shared with the result guard until serialization, spool synchronization and
/// journal acknowledgement finish. The outer invocation keeps every claim
/// until physical join, including discarded callbacks and native unwinding.
struct JobOwners {
    _input: CommittedMemory,
    _kv: CommittedMemory,
    _scratch: CommittedMemory,
    _output: CommittedMemory,
    _lease: EngineLease,
}
struct JobExecution<R> {
    input: JobInput<R>,
    limits: Int4SemanticObjectDocumentJobHostLimits,
    required: Int4SemanticObjectDocumentJobRequirements,
    owners: Arc<JobOwners>,
}
struct DeferredDocumentJob<R> {
    result: Result<Result<JobProgress, JobRunError>, HostedError>,
    input: JobInput<R>,
    owners: Arc<JobOwners>,
}
impl<R> DeferredDocumentJob<R> {
    fn finish(self, control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        checkpoint(control)?;
        let Self { result, input, owners } = self;
        drop(input); drop(owners);
        result
    }
}
fn take_job<R>(queued: &Mutex<Option<JobExecution<R>>>) -> Result<JobExecution<R>, HostedError> {
    queued.lock().map_err(|_| HostedError::Panicked)?.take().ok_or(HostedError::CompletionMissing)
}

impl NlpEngine {
    /// Create or explicitly resume a fixed-schema whole-document semantic job.
    /// The original schema, complete input envelopes, reconciliation, review
    /// policy, model and work declarations authenticate before stored replay.
    /// All extraction chunks and every field review form one atomic result.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_semantic_object_documents<R>(&self, model: &ResidentInt4,
        config: Int4SemanticObjectDocumentJobConfig, request: SourceJobRequest,
        limits: Int4SemanticObjectDocumentJobHostLimits, kernel: PackedInt4Kernel,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_semantic_object_job(model, config, request, limits, kernel, None, reader, cancellation)
    }

    /// One fixed projection team spans every extraction chunk, field review
    /// and durable acknowledgement. Its physical width may change on resume
    /// without changing the keyed semantic recipe or refunding any attempt.
    /// Completion waits for every child and all native ownership to drain.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_semantic_object_documents_with_threads<R>(&self, model: &ResidentInt4,
        config: Int4SemanticObjectDocumentJobConfig, request: SourceJobRequest,
        limits: Int4SemanticObjectDocumentJobHostLimits, kernel: PackedInt4Kernel,
        compute_threads: usize, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_semantic_object_job(model, config, request, limits, kernel,
            Some(compute_threads), reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_semantic_object_job<R>(&self, model: &ResidentInt4,
        config: Int4SemanticObjectDocumentJobConfig, request: SourceJobRequest,
        limits: Int4SemanticObjectDocumentJobHostLimits, kernel: PackedInt4Kernel,
        threads: Option<usize>, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        let native = limits.document.native;
        match threads {
            None => dispatch::preflight(self, native.run)?,
            Some(width) => dispatch::preflight_int4_team(self, native.run, width)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) {
            return Err(HostedError::ResourceDomain.into());
        }
        check_config(&config, limits, request.limits)?;
        let mut required = int4_semantic_object_job_requirements(model.model_identity().group_width(),
            config.extraction.budget, &config.options, &config.encoding, config.work,
            limits, request.limits, request.mode.input_memory(), threads)?;
        // A caller may move a short schema with a large retained allocation.
        let excess = config.extraction.schema.capacity()
            .saturating_sub(config.compiler_limits.max_schema_bytes).saturating_sub(1) as u64;
        required.preparation_bytes = sum(&[required.preparation_bytes, excess])?;
        required.input_bytes = sum(&[required.input_bytes, excess])?;
        required.total_bytes()?;
        let lease = self.resources().acquire_lease();
        let input = Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_reservation_bytes)?;
        let owners = Arc::new(JobOwners { _input: input.commit()?, _kv: kv.commit()?,
            _scratch: scratch.commit()?, _output: output.commit()?, _lease: lease });
        let execution = JobExecution { input: JobInput { config, request: Some(request),
            vocabulary: None, compiler: None, population: None, reader },
            limits, required, owners: Arc::clone(&owners) };
        // Formation failure may discard the callback. Keep its actual reader,
        // configuration and storage request until dispatch physically joins.
        let queued_owner = Arc::new(Mutex::new(Some(execution)));
        let queued = Arc::clone(&queued_owner);
        let model = model.clone();
        let completed = match threads {
            None => dispatch::run(self, native.run, cancellation, move |control| {
                take_job(&queued)?.run(&model.inner.loaded.value, kernel, None, control).finish(control)
            }),
            Some(width) => dispatch::run_int4_team_with_finalize(self, native.run, cancellation,
                model, width, kernel, move |weights, control, team|
                    Ok(take_job(&queued)?.run(weights, kernel, Some(team), control)),
                |pending, control| pending.finish(control)),
        };
        drop(queued_owner); drop(owners);
        completed?.map_err(HostedJobError::Job)
    }
}

impl<R: BufRead> JobExecution<R> {
    fn run<'weights>(self, weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel,
        projection: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> DeferredDocumentJob<R> {
        let Self { mut input, limits, required, owners } = self;
        let result = catch_unwind(AssertUnwindSafe(|| {
            checkpoint(control)?;
            let planner = Int4SemanticObjectDocumentJobPlanner::for_model(weights.identity(), input.config.clone())
                .map_err(HostedError::BatchSetup)?;
            // Keep actual immutable compiler/schema/review assets outside the
            // runner and unwind boundary, not only their reservation counter.
            input.compiler = Some(planner.retain_compiler());
            let request = input.request.take().ok_or(HostedError::CompletionMissing)?;
            planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
            for identity in [planner.execution_identity(), planner.judge_identity()] {
                constrained_int4::check_model(identity, weights.identity()).map_err(|_| HostedError::ModelIdentity)?;
            }
            let controls = crate::tokenizer::pinned_controls::pinned().map_err(|_| HostedError::ModelIdentity)?;
            let vocabulary = Arc::new(ExtractionVocabulary::pinned(controls.template_controls())
                .map_err(|error| semantic_error(error.into()))?);
            input.vocabulary = Some(Arc::clone(&vocabulary));
            checkpoint(control)?;
            let population = match read_population(&mut input.reader, &request, limits.transport, control) {
                Ok(Population::Original(original)) =>
                    Population::BorrowedOriginal(input.population.insert(original)),
                Ok(population) => population,
                Err(error) => return Ok(Err(JobRunError::Storage(error))),
            };
            checkpoint(control)?;
            let mut engine = StrictInt4Engine::new(weights, limits.document.native.context_tokens,
                Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                    max_rope_bytes: required.native.rope_bytes,
                    max_scratch_payload_bytes: required.native.scratch_payload_bound })
                .map_err(native_error)?.with_kernel(kernel);
            if let Some(team) = projection {
                engine = engine.with_projection_executor(team).map_err(native_error)?;
            }
            let admission = DocumentJobAdmission { model: weights.identity(), config: &input.config,
                limits, job: request.limits, required, owners: &owners,
                extraction_identity: planner.execution_identity().clone(),
                judge_identity: planner.judge_identity().clone() };
            let result = (|| {
                let processor = Int4SemanticObjectDocumentJobProcessor::new(planner, &mut engine,
                    &vocabulary, admission).map_err(JobRunError::Processor)?;
                run_population(request, &population, processor, control)
            })();
            drop(engine); drop(population);
            Ok(result)
        })).unwrap_or(Err(HostedError::Panicked));
        DeferredDocumentJob { result, input, owners }
    }
}

struct DocumentJobAdmission<'a> {
    model: &'a Int4ModelIdentity,
    config: &'a Int4SemanticObjectDocumentJobConfig,
    limits: Int4SemanticObjectDocumentJobHostLimits,
    job: JobLimits,
    required: Int4SemanticObjectDocumentJobRequirements,
    extraction_identity: ExecutionIdentity,
    judge_identity: ExecutionIdentity,
    owners: &'a Arc<JobOwners>,
}
impl Int4SemanticObjectDocumentBatchAdmission for DocumentJobAdmission<'_> {
    type Guard = Arc<JobOwners>;
    fn admit(&mut self, request: Int4SemanticObjectDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure> {
        let fail = || BatchItemFailure::fatal(BatchCode::Admission);
        for identity in request.identities.iter().chain(std::iter::once(request.judge_identity)) {
            constrained_int4::check_model(identity, self.model).map_err(|_| fail())?;
        }
        if request.judge_identity != &self.judge_identity
            || request.identities.iter().any(|identity|
                identity.tokenizer_digest != self.extraction_identity.tokenizer_digest
                    || identity.template_digest != self.extraction_identity.template_digest) {
            return Err(fail());
        }
        check_request(self.model.group_width(), self.config, self.limits, self.job, self.required, &request)?;
        let mut identities = Vec::new();
        identities.try_reserve_exact(request.identities.len()).map_err(|_| fail())?;
        identities.extend(request.identities.iter().cloned());
        Ok((identities, Arc::clone(self.owners)))
    }
}

fn check_request(width: usize, config: &Int4SemanticObjectDocumentJobConfig,
    limits: Int4SemanticObjectDocumentJobHostLimits, job: JobLimits,
    required: Int4SemanticObjectDocumentJobRequirements, request: &Int4SemanticObjectDocumentAdmission<'_>)
    -> Result<(), BatchItemFailure> {
    let fail = || BatchItemFailure::fatal(BatchCode::Admission);
    check_config(config, limits, job).map_err(|_| fail())?;
    let mapping = config.extraction_limits.mapping.mapping;
    for identity in request.identities.iter().chain(std::iter::once(request.judge_identity)) {
        constrained_int4::check_profile(identity, width).map_err(|_| fail())?;
    }
    let tokens = (request.chunk_count as u64).checked_mul(u64::from(config.extraction.budget.max_output_tokens))
        .ok_or_else(fail)?;
    if request.identities.len() != request.chunk_count || request.chunk_count == 0
        || request.chunk_count > mapping.chunks.max_chunks
        || request.identities.iter().any(|identity| identity.task_spec != "extract-v1")
        || request.judge_identity.task_spec != "judge-v1" || request.group_width != width
        || request.model_work != config.options.max_model_work
        || !fits(request.model_work, config.work.max_model_work)
        || !positive_work(request.extraction_reserved_work)
        || !fits(request.extraction_reserved_work, request.model_work)
        || !fits(request.extraction_reserved_work, mapping.max_model_work)
        || request.mask_node_visits != mapping.max_mask_visits
        || request.mask_limits != mapping.mask_limits
        || request.verification_budget != config.extraction_limits.mapping.verification
        || !super::corpus::same_extraction_limits(request.extraction_limits, config.extraction_limits)
        || !super::corpus::same_review_limits(request.review_limits, config.review_limits)
        || request.kv_reservation_bytes != required.native.kv_bytes
        || request.max_output_tokens != tokens || request.extraction_budget != config.extraction.budget
        || request.judge_budget != config.options.specification.judge_budget
        || request.semantic_limits != config.options.limits
        || !super::super::corpus::same_planning(request.judge_planning_limits, config.judge_planning)
        || request.source_bytes == 0 || request.source_bytes > mapping.chunks.max_input_bytes
        || request.source_bytes > job.max_input_bytes_per_item
        || request.input_encoding != &config.encoding {
        return Err(fail());
    }
    if request.max_result_bytes != config.options.limits.max_output_bytes
        || request.max_result_bytes > job.max_result_bytes as u64 {
        return Err(BatchItemFailure::fatal(BatchCode::OutputLineLimit));
    }
    let actual = int4_semantic_object_document_requirements(width, request.extraction_budget,
        &config.options, request.input_encoding, limits.document, None).map_err(|_| fail())?;
    if actual.native != required.native || actual.workspace_bytes != required.workspace_bytes
        || preparation_with_recipe(actual.preparation_bytes, config.compiler_limits, &config.options)
            .map_err(|_| fail())? > required.preparation_bytes
        || actual.output_reservation_bytes > required.output_reservation_bytes {
        return Err(fail());
    }
    Ok(())
}

#[cfg(test)] mod tests;
