//! Authenticated retained extraction plus complete correlated field review.
//! One declared attempt covers both stages; JobRunner owns the durable debit,
//! original-population authentication and spool-first acknowledgement.

use super::*;
use std::{io::BufRead, panic::{catch_unwind, AssertUnwindSafe}, sync::Mutex};
use crate::{
    batch::{BatchCode, BatchItemFailure, extract::semantic::{Int4SemanticExtractionAdmission,
        Int4SemanticExtractionBatchAdmission, Int4SemanticExtractionBatchLimits}},
    hosted::corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest, read_population, run_population},
    jobs::{JobInputMemory, JobLimits, JobProgress, JobWork, runner::{JobRunError,
        extract::grouped_int4::semantic::{Int4SemanticExtractionJobConfig,
            Int4SemanticExtractionJobPlanner, Int4SemanticExtractionJobProcessor}}},
    native_engine::strict_int4::projection::Int4ProjectionExecutor,
    tasks::extract::semantic::grouped_int4::Int4SemanticOptions,
};

/// Complete simultaneous job payload, excluding resident weights, caller
/// preparation and runtime cleanup. This quote is not an RSS measurement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SemanticJobRequirements {
    pub native: Int4MemoryRequirement,
    pub workspace_bytes: u64,
    /// Full semantic preparation, already included in input_bytes.
    pub preparation_bytes: u64,
    /// Semantic preparation, population/index/parser, IO, journal and canonical
    /// result staging. Retained-input modes keep one original record at a time.
    pub input_bytes: u64,
    /// Separate combined-result owner retained through durable acknowledgement.
    pub output_reservation_bytes: u64,
    /// One persistent projection team, separately reserved by dispatch.
    pub team_workspace_bytes: u64,
}
impl Int4SemanticJobRequirements {
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        sum(&[self.native.kv_bytes, self.workspace_bytes, self.input_bytes,
            self.output_reservation_bytes, self.team_workspace_bytes])
    }
}

/// Pure admission before schema, population, key or model IO. One decoded
/// source cannot exceed its complete original JSON envelope, so the immutable
/// item-byte ceiling also bounds every simultaneous full-source review copy.
/// All dynamic field judges coexist; their preparation is never priced as one
/// recycled head. Input mode changes payload ownership, never lifetime work.
#[allow(clippy::too_many_arguments)]
pub fn int4_semantic_job_requirements(width: usize, extraction: TaskBudget,
    options: &Int4SemanticOptions, judge_planning: JudgeLimits, encoding: &UntrustedEncoding,
    work: Int4SemanticExtractionBatchLimits, limits: JobHostLimits, job: JobLimits,
    input: JobInputMemory, compute_threads: Option<usize>)
    -> Result<Int4SemanticJobRequirements, HostedJobError> {
    options.validate().map_err(semantic_error)?;
    work.validate_item(options.max_model_work).map_err(HostedError::BatchSetup)?;
    super::corpus::validate_judge_planning(judge_planning)?;
    let job_buffers = limits.required_buffer_bytes_for_input(job, input)?;
    if job.max_work != JobWork::from_int4(work.max_model_work, work.masks.max_visits_per_run) {
        return Err(HostedError::Limits("semantic job lifetime native and mask work").into());
    }
    if options.limits.max_output_bytes > job.max_result_bytes as u64 {
        return Err(HostedError::Limits("semantic job complete stored-result envelope").into());
    }
    let semantic = semantic_host(limits, work).requirements_with_encoding(width, extraction,
        options.specification.judge_budget, options.limits, judge_planning,
        job.max_input_bytes_per_item, encoding)?;
    // Replace the shared base preparation charge, retaining every independent
    // population, journal, IO and serialization owner from the ordinary job.
    let staging = job_buffers.checked_sub(limits.preparation_reserve_bytes)
        .ok_or(HostedError::Limits("semantic job preparation arithmetic"))?;
    let required = Int4SemanticJobRequirements {
        native: semantic.native, workspace_bytes: semantic.native_workspace_bytes,
        preparation_bytes: semantic.preparation_bytes,
        input_bytes: sum(&[staging, semantic.preparation_bytes])?,
        output_reservation_bytes: semantic.output_bytes,
        team_workspace_bytes: compute_threads.map(|threads|
            int4_task_team_workspace_bytes(threads, width)).transpose()?.unwrap_or(0),
    };
    required.total_bytes()?;
    Ok(required)
}

fn semantic_host(limits: JobHostLimits, work: Int4SemanticExtractionBatchLimits) -> Int4SemanticHostLimits {
    Int4SemanticHostLimits { native: limits.native, masks: work.masks.per_mask,
        max_mask_node_visits: work.masks.max_visits_per_item,
        preparation_reserve_bytes: limits.preparation_reserve_bytes }
}

struct JobInput<R> {
    config: Int4SemanticExtractionJobConfig,
    request: Option<SourceJobRequest>,
    vocabulary: Option<Arc<ExtractionVocabulary>>,
    reader: R,
}
/// Output guards clone this owner through serialization, spool sync and
/// journal acknowledgement. The call also retains it through physical join,
/// including queued cancellation, setup failure and unwinding native work.
struct JobOwners {
    _input: CommittedMemory,
    _kv: CommittedMemory,
    _scratch: CommittedMemory,
    _output: CommittedMemory,
    _lease: EngineLease,
}
struct JobExecution<R> {
    input: JobInput<R>,
    limits: JobHostLimits,
    required: Int4SemanticJobRequirements,
    owners: Arc<JobOwners>,
}
struct DeferredSemanticJob<R> {
    result: Result<Result<JobProgress, JobRunError>, HostedError>,
    input: JobInput<R>,
    owners: Arc<JobOwners>,
}
impl<R> DeferredSemanticJob<R> {
    fn finish(self, control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        checkpoint(control)?;
        let Self { result, input, owners } = self;
        drop(input); drop(owners);
        result
    }
}
fn take_execution<R>(queued: &Mutex<Option<JobExecution<R>>>) -> Result<JobExecution<R>, HostedError> {
    queued.lock().map_err(|_| HostedError::Panicked)?.take().ok_or(HostedError::CompletionMissing)
}

impl NlpEngine {
    /// Start or resume a fixed-schema extraction job whose complete retained
    /// result also contains every configured field review. The review remains
    /// an explicitly requested, correlated diagnostic: it never repairs or
    /// accepts extraction. Failure in either stage commits no partial result.
    ///
    /// Exact input, schema, review policy, model and all work limits authenticate
    /// before replay or tail repair. Results use the existing protected spool;
    /// input retention is separately selected by SourceJobRequest. Native work
    /// and mask allowances are debited for the full attempt before execution.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_semantic_extract<R>(&self, model: &ResidentInt4,
        config: Int4SemanticExtractionJobConfig, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_semantic_job(model, config, request, limits, kernel, None, reader, cancellation)
    }

    /// One admitted output-row team spans extraction, every field judgment,
    /// durable acknowledgement and materialization. Its physical width may
    /// change on authenticated resume without changing the private recipe or
    /// refunding attempts/native/mask work. Return waits for physical join.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_semantic_extract_with_threads<R>(&self, model: &ResidentInt4,
        config: Int4SemanticExtractionJobConfig, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, compute_threads: usize,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_semantic_job(model, config, request, limits, kernel,
            Some(compute_threads), reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_semantic_job<R>(&self, model: &ResidentInt4, config: Int4SemanticExtractionJobConfig,
        request: SourceJobRequest, limits: JobHostLimits, kernel: PackedInt4Kernel,
        threads: Option<usize>, reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        match threads {
            None => dispatch::preflight(self, limits.native.run)?,
            Some(width) => dispatch::preflight_int4_team(self, limits.native.run, width)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain.into()); }
        if config.extraction.schema.is_empty()
            || config.extraction.schema.len() > config.compiler_limits.max_schema_bytes {
            return Err(HostedError::Limits("semantic job fixed schema size").into());
        }
        let required = int4_semantic_job_requirements(model.model_identity().group_width(),
            config.extraction.budget, &config.options, config.judge_planning, &config.encoding,
            config.work, limits, request.limits, request.mode.input_memory(), threads)?;
        let lease = self.resources().acquire_lease();
        let input = Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_reservation_bytes)?;
        let owners = Arc::new(JobOwners { _input: input.commit()?, _kv: kv.commit()?,
            _scratch: scratch.commit()?, _output: output.commit()?, _lease: lease });
        let execution = JobExecution { input: JobInput { config, request: Some(request), vocabulary: None, reader },
            limits, required, owners: Arc::clone(&owners) };
        // A formation failure can discard the callback. The outer owner keeps
        // the actual reader/configuration and all claims until dispatch joins.
        let queued_owner = Arc::new(Mutex::new(Some(execution)));
        let queued = Arc::clone(&queued_owner);
        let model = model.clone();
        let completed = match threads {
            None => dispatch::run(self, limits.native.run, cancellation, move |control| {
                take_execution(&queued)?.run(&model.inner.loaded.value, kernel, None, control).finish(control)
            }),
            Some(width) => dispatch::run_int4_team_with_finalize(self, limits.native.run, cancellation,
                model, width, kernel, move |weights, control, team|
                    Ok(take_execution(&queued)?.run(weights, kernel, Some(team), control)),
                |pending, control| pending.finish(control)),
        };
        drop(queued_owner);
        drop(owners);
        completed?.map_err(HostedJobError::Job)
    }
}

impl<R: BufRead> JobExecution<R> {
    fn run<'weights>(self, weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel,
        projection: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> DeferredSemanticJob<R> {
        let Self { mut input, limits, required, owners } = self;
        // Borrow the private input owner across all fallible setup and native
        // work, retaining it outside the unwind boundary for post-join disposal.
        let result = catch_unwind(AssertUnwindSafe(|| {
            checkpoint(control)?;
            let planner = Int4SemanticExtractionJobPlanner::for_model(weights.identity(), input.config.clone())
                .map_err(HostedError::BatchSetup)?;
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
                Ok(population) => population,
                Err(error) => return Ok(Err(JobRunError::Storage(error))),
            };
            checkpoint(control)?;
            let mut engine = StrictInt4Engine::new(weights, limits.native.context_tokens,
                Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                    max_rope_bytes: required.native.rope_bytes,
                    max_scratch_payload_bytes: required.native.scratch_payload_bound })
                .map_err(native_error)?.with_kernel(kernel);
            if let Some(team) = projection {
                engine = engine.with_projection_executor(team).map_err(native_error)?;
            }
            let admission = SemanticJobAdmission { model: weights.identity(), config: &input.config,
                limits, job: request.limits, required, owners: &owners };
            let result = (|| {
                let processor = Int4SemanticExtractionJobProcessor::new(planner, &mut engine,
                    &vocabulary, admission).map_err(JobRunError::Processor)?;
                run_population(request, &population, processor, control)
            })();
            drop(engine); drop(population);
            Ok(result)
        })).unwrap_or(Err(HostedError::Panicked));
        DeferredSemanticJob { result, input, owners }
    }
}

struct SemanticJobAdmission<'a> {
    model: &'a Int4ModelIdentity,
    config: &'a Int4SemanticExtractionJobConfig,
    limits: JobHostLimits,
    job: JobLimits,
    required: Int4SemanticJobRequirements,
    owners: &'a Arc<JobOwners>,
}
impl Int4SemanticExtractionBatchAdmission for SemanticJobAdmission<'_> {
    type Guard = Arc<JobOwners>;
    fn admit(&mut self, request: Int4SemanticExtractionAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        for identity in [request.identity, request.judge_identity] {
            constrained_int4::check_model(identity, self.model)
                .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        }
        check_request(self.model.group_width(), self.config, self.limits, self.job, self.required, &request)?;
        Ok((request.identity.clone(), Arc::clone(self.owners)))
    }
}
fn check_request(width: usize, config: &Int4SemanticExtractionJobConfig,
    limits: JobHostLimits, job: JobLimits, required: Int4SemanticJobRequirements,
    request: &Int4SemanticExtractionAdmission<'_>) -> Result<(), BatchItemFailure> {
    let fail = || BatchItemFailure::fatal(BatchCode::Admission);
    for identity in [request.identity, request.judge_identity] {
        constrained_int4::check_profile(identity, width).map_err(|_| fail())?;
    }
    if request.identity.task_spec != "extract-v1" || request.judge_identity.task_spec != "judge-v1"
        || request.group_width != width || request.model_work != config.options.max_model_work
        || !fits(request.model_work, config.work.max_model_work)
        || request.extraction_reserved_work.forward_positions == 0
        || request.extraction_reserved_work.projected_logits == 0
        || request.extraction_reserved_work.attention_pairs == 0
        || request.extraction_reserved_work.projections.dot_products == 0
        || request.extraction_reserved_work.projections.multiply_accumulates == 0
        || !fits(request.extraction_reserved_work, request.model_work)
        || request.extraction_reserved_work.forward_positions > limits.native.context_tokens as u64
        || request.mask_node_visits != config.work.masks.max_visits_per_item
        || request.mask_limits != config.work.masks.per_mask
        || request.kv_reservation_bytes != required.native.kv_bytes
        || request.extraction_budget != config.extraction.budget
        || request.judge_budget != config.options.specification.judge_budget
        || request.semantic_limits != config.options.limits
        || !super::corpus::same_planning(request.judge_planning_limits, config.judge_planning)
        || request.source_bytes == 0 || request.source_bytes > job.max_input_bytes_per_item
        || request.input_encoding != &config.encoding {
        return Err(fail());
    }
    if request.max_result_bytes != config.options.limits.max_output_bytes
        || request.max_result_bytes > job.max_result_bytes as u64 {
        return Err(BatchItemFailure::fatal(BatchCode::OutputLineLimit));
    }
    let actual = semantic_host(limits, config.work).requirements_with_encoding(width,
        request.extraction_budget, request.judge_budget, request.semantic_limits,
        request.judge_planning_limits, request.source_bytes, request.input_encoding).map_err(|_| fail())?;
    if actual.native != required.native || actual.native_workspace_bytes != required.workspace_bytes
        || actual.preparation_bytes > required.preparation_bytes
        || actual.output_bytes > required.output_reservation_bytes {
        return Err(fail());
    }
    Ok(())
}

#[cfg(test)] mod tests;
