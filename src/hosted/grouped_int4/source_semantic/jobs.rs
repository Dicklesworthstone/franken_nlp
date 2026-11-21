//! Authenticated retained summaries and passage answers with complete-source review.
//! One declared attempt covers both stages; JobRunner owns the durable debit,
//! original-population authentication and spool-first acknowledgement.

use super::*;
use std::{io::BufRead, panic::{catch_unwind, AssertUnwindSafe}, sync::Mutex};
use crate::{
    batch::{BatchCode, BatchItemFailure, source::semantic::{Int4SemanticSourceAdmission,
        Int4SemanticSourceBatchAdmission, Int4SemanticSourceBatchLimits,
        Int4SemanticSourceBatchPlanner}},
    hosted::corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest,
        Population, read_population, run_population},
    jobs::{JobInputMemory, JobLimits, JobProgress, JobWork, population::JobPopulation, runner::{JobRunError,
        source::grouped_int4::semantic::{Int4SemanticSourceJobConfig,
            Int4SemanticSourceJobPlanner, Int4SemanticSourceJobProcessor,
            SEMANTIC_SOURCE_JOB_RECIPE_BYTES}}},
    native_engine::strict_int4::projection::Int4ProjectionExecutor,
    tasks::source_planning::{SourcePlanningLimits, grouped_int4::semantic::Int4SourceSemanticOptions},
};

/// Complete simultaneous job payload, excluding resident weights, caller
/// preparation and runtime cleanup. This quote is not an RSS measurement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SourceSemanticJobRequirements {
    pub native: Int4MemoryRequirement,
    pub workspace_bytes: u64,
    /// Full semantic preparation and bounded recipe buffers, already included in input_bytes.
    pub preparation_bytes: u64,
    /// Semantic preparation, population/index/parser, IO, journal and canonical
    /// result staging. Retained-input modes keep one original record at a time.
    pub input_bytes: u64,
    /// Separate combined-result owner retained through durable acknowledgement.
    pub output_reservation_bytes: u64,
    /// One persistent projection team, separately reserved by dispatch.
    pub team_workspace_bytes: u64,
}
impl Int4SourceSemanticJobRequirements {
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        sum(&[self.native.kv_bytes, self.workspace_bytes, self.input_bytes,
            self.output_reservation_bytes, self.team_workspace_bytes])
    }
}

/// Pure admission before private population, key or model IO. Input facts
/// declare the complete source, question and passage maxima for ONE original
/// record, bounded by its immutable JSON envelope. All dynamic statement
/// judges coexist; preparation is never priced as one recycled review head.
/// Input retention changes payload ownership, never lifetime work authority.
#[allow(clippy::too_many_arguments)]
pub fn int4_source_semantic_job_requirements(width: usize, source: TaskBudget,
    options: &Int4SourceSemanticOptions, judge_planning: JudgeLimits, input: SourceSemanticInputFacts,
    encoding: &UntrustedEncoding, work: Int4SemanticSourceBatchLimits, limits: JobHostLimits,
    job: JobLimits, population: JobInputMemory, compute_threads: Option<usize>)
    -> Result<Int4SourceSemanticJobRequirements, HostedJobError> {
    options.validate().map_err(task_error)?;
    work.validate_item(options.max_model_work).map_err(HostedError::BatchSetup)?;
    let job_buffers = limits.required_buffer_bytes_for_input(job, population)?;
    if job.max_work != JobWork::from_int4(work.max_model_work, work.masks.max_visits_per_run) {
        return Err(HostedError::Limits("semantic source job lifetime native and mask work").into());
    }
    if options.limits.max_output_bytes > job.max_result_bytes as u64 {
        return Err(HostedError::Limits("semantic source job complete stored-result envelope").into());
    }
    if job.max_id_bytes > 256 || input.source_bytes > job.max_input_bytes_per_item
        || input.question_bytes > job.max_input_bytes_per_item {
        return Err(HostedError::Limits("semantic source job complete record input ceiling").into());
    }
    let semantic = semantic_host(limits, work).requirements_with_encoding(width, source,
        options.judge_budget, options.limits, judge_planning, input, encoding)?;
    // Replace only the base preparation charge. Population/index/parser, IO,
    // journal and result serialization all retain their independent owners.
    let staging = job_buffers.checked_sub(limits.preparation_reserve_bytes)
        .ok_or(HostedError::Limits("semantic source job preparation arithmetic"))?;
    let preparation_bytes = preparation_with_recipe(semantic.preparation_bytes)?;
    let required = Int4SourceSemanticJobRequirements {
        native: semantic.native, workspace_bytes: semantic.native_workspace_bytes,
        preparation_bytes, input_bytes: sum(&[staging, preparation_bytes])?,
        output_reservation_bytes: semantic.output_bytes,
        team_workspace_bytes: compute_threads.map(|threads|
            int4_task_team_workspace_bytes(threads, width)).transpose()?.unwrap_or(0),
    };
    required.total_bytes()?;
    Ok(required)
}

fn input_ceiling(config: &Int4SemanticSourceJobConfig, limits: JobHostLimits, job: JobLimits)
    -> Result<SourceSemanticInputFacts, HostedError> {
    config.task.validate_semantic(&config.options).map_err(HostedError::BatchSetup)?;
    let SourcePlanningLimits { max_input_bytes, max_context_tokens, max_passages, .. } = config.planning;
    if !(1..=64 * 1024 * 1024).contains(&max_input_bytes)
        || max_input_bytes > job.max_input_bytes_per_item
        || !(1..=262_144).contains(&max_context_tokens) || max_context_tokens > limits.native.context_tokens
        || !(1..=1024).contains(&max_passages) {
        return Err(HostedError::Limits("semantic source job complete input planning"));
    }
    let answer = matches!(config.task, crate::batch::source::semantic::Int4SemanticSourceBatchTask::Answer { .. });
    Ok(SourceSemanticInputFacts {
        source_bytes: max_input_bytes.min(config.options.limits.max_source_bytes),
        question_bytes: if answer { max_input_bytes.min(config.options.limits.max_claim_bytes) } else { 0 },
        passage_count: if answer { max_passages } else { 0 },
    })
}

// Private recipe binding may hold bounded serialized and canonical buffers
// together. Neither buffer is covered by the caller's compiler-capacity reserve.
fn preparation_with_recipe(semantic_bytes: u64) -> Result<u64, HostedError> {
    let recipe_bytes = SEMANTIC_SOURCE_JOB_RECIPE_BYTES as u64;
    sum(&[semantic_bytes, recipe_bytes, recipe_bytes])
}

fn semantic_host(limits: JobHostLimits, work: Int4SemanticSourceBatchLimits) -> Int4SourceSemanticHostLimits {
    Int4SourceSemanticHostLimits { native: limits.native, masks: work.masks.per_mask,
        max_mask_node_visits: work.masks.max_visits_per_item,
        preparation_reserve_bytes: limits.preparation_reserve_bytes }
}

struct JobInput<R> {
    config: Int4SemanticSourceJobConfig,
    request: Option<SourceJobRequest>,
    vocabulary: Option<Arc<ExtractionVocabulary>>,
    compiler: Option<Arc<Int4SemanticSourceBatchPlanner<'static>>>,
    population: Option<JobPopulation>,
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
    required: Int4SourceSemanticJobRequirements,
    owners: Arc<JobOwners>,
}
struct DeferredSourceSemanticJob<R> {
    result: Result<Result<JobProgress, JobRunError>, HostedError>,
    input: JobInput<R>,
    owners: Arc<JobOwners>,
}
impl<R> DeferredSourceSemanticJob<R> {
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
    /// Start or resume cited summaries or passage answers with complete
    /// same-source semantic review. Preserve every original task result and
    /// citation; the correlated review remains a separately requested,
    /// uncalibrated diagnostic. Failure in either stage commits no partial item.
    ///
    /// Exact input, task, review policy, both compilers and work limits authenticate
    /// before replay or tail repair. Results use the existing protected spool;
    /// input retention is separately selected by SourceJobRequest. Native work
    /// and mask allowances are debited for the full attempt before execution.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_semantic_source<R>(&self, model: &ResidentInt4,
        config: Int4SemanticSourceJobConfig, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_source_semantic_job(model, config, request, limits, kernel, None, reader, cancellation)
    }

    /// One admitted output-row team spans source generation, every statement review,
    /// durable acknowledgement and materialization. Its physical width may
    /// change on authenticated resume without changing the private recipe or
    /// refunding attempts/native/mask work. Return waits for physical join.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_semantic_source_with_threads<R>(&self, model: &ResidentInt4,
        config: Int4SemanticSourceJobConfig, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, compute_threads: usize,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_source_semantic_job(model, config, request, limits, kernel,
            Some(compute_threads), reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_source_semantic_job<R>(&self, model: &ResidentInt4, config: Int4SemanticSourceJobConfig,
        request: SourceJobRequest, limits: JobHostLimits, kernel: PackedInt4Kernel,
        threads: Option<usize>, reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        match threads {
            None => dispatch::preflight(self, limits.native.run)?,
            Some(width) => dispatch::preflight_int4_team(self, limits.native.run, width)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain.into()); }
        let facts = input_ceiling(&config, limits, request.limits)?;
        let required = int4_source_semantic_job_requirements(model.model_identity().group_width(),
            config.task.budget(), &config.options, config.judge_planning, facts, &config.encoding,
            config.work, limits, request.limits, request.mode.input_memory(), threads)?;
        let lease = self.resources().acquire_lease();
        let input = Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_reservation_bytes)?;
        let owners = Arc::new(JobOwners { _input: input.commit()?, _kv: kv.commit()?,
            _scratch: scratch.commit()?, _output: output.commit()?, _lease: lease });
        let execution = JobExecution { input: JobInput { config, request: Some(request), vocabulary: None,
            compiler: None, population: None, reader },
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
        control: &mut RunControl) -> DeferredSourceSemanticJob<R> {
        let Self { mut input, limits, required, owners } = self;
        // Borrow the private input owner across all fallible setup and native
        // work, retaining it outside the unwind boundary for post-join disposal.
        let result = catch_unwind(AssertUnwindSafe(|| {
            checkpoint(control)?;
            let planner = Int4SemanticSourceJobPlanner::for_model(weights.identity(), input.config.clone())
                .map_err(HostedError::BatchSetup)?;
            // Keep the actual shared source/compiler/judge assets, not only
            // their reservation, outside the fallible worker and runner scopes.
            input.compiler = Some(planner.retain_compiler());
            let request = input.request.take().ok_or(HostedError::CompletionMissing)?;
            planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
            for identity in [planner.execution_identity(), planner.judge_identity()] {
                constrained_int4::check_model(identity, weights.identity()).map_err(|_| HostedError::ModelIdentity)?;
            }
            let controls = crate::tokenizer::pinned_controls::pinned().map_err(|_| HostedError::ModelIdentity)?;
            let vocabulary = Arc::new(ExtractionVocabulary::pinned(controls.template_controls())
                .map_err(|error| task_error(error.into()))?);
            input.vocabulary = Some(Arc::clone(&vocabulary));
            checkpoint(control)?;
            let population = match read_population(&mut input.reader, &request, limits.transport, control) {
                Ok(Population::Original(original)) => {
                    // The runner borrows this same authenticated population.
                    // Its plaintext bytes remain owned until physical join,
                    // including processor failure and native unwinding.
                    Population::BorrowedOriginal(input.population.insert(original))
                }
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
            let admission = SourceSemanticJobAdmission { model: weights.identity(), config: &input.config,
                limits, job: request.limits, required, owners: &owners };
            let result = (|| {
                let processor = Int4SemanticSourceJobProcessor::new(planner, &mut engine,
                    &vocabulary, admission).map_err(JobRunError::Processor)?;
                run_population(request, &population, processor, control)
            })();
            drop(engine); drop(population);
            Ok(result)
        })).unwrap_or(Err(HostedError::Panicked));
        DeferredSourceSemanticJob { result, input, owners }
    }
}

struct SourceSemanticJobAdmission<'a> {
    model: &'a Int4ModelIdentity,
    config: &'a Int4SemanticSourceJobConfig,
    limits: JobHostLimits,
    job: JobLimits,
    required: Int4SourceSemanticJobRequirements,
    owners: &'a Arc<JobOwners>,
}
impl Int4SemanticSourceBatchAdmission for SourceSemanticJobAdmission<'_> {
    type Guard = Arc<JobOwners>;
    fn admit(&mut self, request: Int4SemanticSourceAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        for identity in [request.identity, request.judge_identity] {
            constrained_int4::check_model(identity, self.model)
                .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        }
        check_request(self.model.group_width(), self.config, self.limits, self.job, self.required, &request)?;
        Ok((request.identity.clone(), Arc::clone(self.owners)))
    }
}
fn check_request(width: usize, config: &Int4SemanticSourceJobConfig,
    limits: JobHostLimits, job: JobLimits, required: Int4SourceSemanticJobRequirements,
    request: &Int4SemanticSourceAdmission<'_>) -> Result<(), BatchItemFailure> {
    let fail = || BatchItemFailure::fatal(BatchCode::Admission);
    for identity in [request.identity, request.judge_identity] {
        constrained_int4::check_profile(identity, width).map_err(|_| fail())?;
    }
    let ceiling = input_ceiling(config, limits, job).map_err(|_| fail())?;
    if request.identity.task_spec != config.task.built_in().spec().identity() || request.judge_identity.task_spec != "judge-v1"
        || request.group_width != width || request.model_work != config.options.max_model_work
        || !fits(request.model_work, config.work.max_model_work)
        || request.source_reserved_work.forward_positions == 0
        || request.source_reserved_work.projected_logits == 0
        || request.source_reserved_work.attention_pairs == 0
        || request.source_reserved_work.projections.dot_products == 0
        || request.source_reserved_work.projections.multiply_accumulates == 0
        || !fits(request.source_reserved_work, request.model_work)
        || request.source_reserved_work.forward_positions > limits.native.context_tokens as u64
        || request.mask_node_visits != config.work.masks.max_visits_per_item
        || request.mask_limits != config.work.masks.per_mask
        || request.kv_reservation_bytes != required.native.kv_bytes
        || request.source_budget != config.task.budget()
        || request.judge_budget != config.options.judge_budget
        || request.semantic_limits != config.options.limits
        || !same_planning(request.judge_planning_limits, config.judge_planning)
        || request.input_facts.source_bytes == 0 || request.input_facts.source_bytes > ceiling.source_bytes
        || request.input_facts.question_bytes > ceiling.question_bytes
        || request.input_facts.passage_count > ceiling.passage_count
        || (request.input_facts.passage_count == 0) != (ceiling.passage_count == 0)
        || request.input_encoding != &config.encoding {
        return Err(fail());
    }
    if request.max_result_bytes != config.options.limits.max_output_bytes
        || request.max_result_bytes > job.max_result_bytes as u64 {
        return Err(BatchItemFailure::fatal(BatchCode::OutputLineLimit));
    }
    let actual = semantic_host(limits, config.work).requirements_with_encoding(width,
        request.source_budget, request.judge_budget, request.semantic_limits,
        request.judge_planning_limits, request.input_facts, request.input_encoding).map_err(|_| fail())?;
    if actual.native != required.native || actual.native_workspace_bytes != required.workspace_bytes
        || preparation_with_recipe(actual.preparation_bytes).map_err(|_| fail())? > required.preparation_bytes
        || actual.output_bytes > required.output_reservation_bytes {
        return Err(fail());
    }
    Ok(())
}

fn same_planning(a: JudgeLimits, b: JudgeLimits) -> bool {
    a.max_total_prompt_tokens == b.max_total_prompt_tokens
        && a.max_total_candidate_tokens == b.max_total_candidate_tokens
        && a.max_total_projected_logits == b.max_total_projected_logits
        && a.max_output_bytes == b.max_output_bytes
        && a.per_head.max_candidates == b.per_head.max_candidates
        && a.per_head.max_total_tokens == b.per_head.max_total_tokens
        && a.per_head.max_nodes == b.per_head.max_nodes
        && a.per_head.max_depth == b.per_head.max_depth
        && a.per_head.max_candidate_id_bytes == b.per_head.max_candidate_id_bytes
        && a.per_head.max_projected_logits == b.per_head.max_projected_logits
}

#[cfg(test)] mod tests;
