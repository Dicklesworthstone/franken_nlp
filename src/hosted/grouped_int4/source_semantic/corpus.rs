//! One resident summary/answer compiler and correlated reader for NDJSON.
//! Exact source results and complete original evidence survive every review;
//! document and run work ceilings never renew at a flush.

use super::*;
use std::{fmt, io::{BufRead, Write}, panic::{catch_unwind, AssertUnwindSafe}, sync::Mutex};
use crate::{
    batch::{self as stream, BatchCode, BatchItemFailure, BatchSummary,
        source::semantic::{Int4SemanticSourceAdmission, Int4SemanticSourceBatchAdmission,
            Int4SemanticSourceBatchLimits, Int4SemanticSourceBatchPlanner,
            Int4SemanticSourceBatchTask, NativeInt4SemanticSourceBatch}},
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    hosted::corpus::CorpusLimits,
    native_engine::strict_int4::{STRICT_INT4_EXECUTION, projection::Int4ProjectionExecutor},
    tasks::{judge::grouped_int4::Int4JudgePlanner,
        source_planning::{SourcePlanningLimits, SourceTaskPlanner,
            grouped_int4::semantic::Int4SourceSemanticOptions}},
};

/// One fixed summary/answer policy and complete-document review budget.
/// Summary records carry their source in text. Answer records carry their
/// question in text and separate passages in the closed task_args shape.
/// The source planning input ceiling bounds the COMPLETE QA aggregate.
///
/// Preparation must price supplied options' capacities, pinned compilers and
/// vocabulary. The quote adds every retained source, review plan and result.
/// Caller-owned IO storage must fit the explicit IO reservation.
#[derive(Clone)]
pub struct Int4SourceSemanticCorpusConfig {
    pub task: Int4SemanticSourceBatchTask,
    pub planning: SourcePlanningLimits,
    pub options: Int4SourceSemanticOptions,
    pub judge_planning: JudgeLimits,
    pub encoding: UntrustedEncoding,
    pub work: Int4SemanticSourceBatchLimits,
    pub limits: CorpusLimits,
}
impl fmt::Debug for Int4SourceSemanticCorpusConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Int4SourceSemanticCorpusConfig")
            .field("task", &self.task.built_in().spec().identity())
            .field("source_budget", &self.task.budget())
            .field("semantic_limits", &self.options.limits)
            .field("document_work", &self.options.max_model_work)
            .field("run_limits", &self.limits)
            .finish_non_exhaustive()
    }
}

/// Simultaneous admitted payload excluding weights, caller preparation and
/// dispatch cleanup. Team scratch is separately reserved by the dispatcher.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SourceSemanticCorpusRequirements {
    pub native: Int4MemoryRequirement,
    pub workspace_bytes: u64,
    /// Complete source/compiler/review preparation, already in input_bytes.
    pub preparation_bytes: u64,
    pub input_bytes: u64,
    pub output_reservation_bytes: u64,
    pub team_workspace_bytes: u64,
}
impl Int4SourceSemanticCorpusRequirements {
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        sum(&[self.native.kv_bytes, self.workspace_bytes, self.input_bytes,
            self.output_reservation_bytes, self.team_workspace_bytes])
    }
}
impl Int4SourceSemanticCorpusConfig {
    pub fn requirements(&self, width: usize) -> Result<Int4SourceSemanticCorpusRequirements, HostedError> {
        self.requirements_for_schedule(width, None)
    }
    pub fn requirements_with_threads(&self, width: usize, compute_threads: usize)
        -> Result<Int4SourceSemanticCorpusRequirements, HostedError> {
        self.requirements_for_schedule(width, Some(compute_threads))
    }
    fn input_ceiling(&self) -> Result<SourceSemanticInputFacts, HostedError> {
        self.task.validate_semantic(&self.options).map_err(HostedError::BatchSetup)?;
        if self.planning.max_input_bytes != self.limits.transport.max_document_bytes
            || !(1..=64 * 1024 * 1024).contains(&self.planning.max_input_bytes)
            || !(1..=262_144).contains(&self.planning.max_context_tokens)
            || self.planning.max_context_tokens > self.limits.native.context_tokens
            || !(1..=1024).contains(&self.planning.max_passages) {
            return Err(HostedError::Limits("semantic source corpus complete input planning"));
        }
        let answer = matches!(self.task, Int4SemanticSourceBatchTask::Answer { .. });
        Ok(SourceSemanticInputFacts {
            source_bytes: self.planning.max_input_bytes.min(self.options.limits.max_source_bytes),
            question_bytes: if answer {
                self.planning.max_input_bytes.min(self.options.limits.max_claim_bytes)
            } else { 0 },
            passage_count: if answer { self.planning.max_passages } else { 0 },
        })
    }
    fn requirements_for_schedule(&self, width: usize, threads: Option<usize>)
        -> Result<Int4SourceSemanticCorpusRequirements, HostedError> {
        int4_source_semantic_corpus_requirements(width, self.task.budget(), &self.options,
            self.judge_planning, self.input_ceiling()?, &self.encoding, self.work, self.limits, threads)
    }
}

/// Pure peak quote before task options, private population or model IO. Input
/// facts are DECLARED maxima for one complete source/QA request, not stream
/// totals. Quote both question and all passages; the source compiler separately
/// enforces its one aggregate byte/token ceiling before encoding a record.
#[allow(clippy::too_many_arguments)]
pub fn int4_source_semantic_corpus_requirements(width: usize, source: TaskBudget,
    options: &Int4SourceSemanticOptions, judge_planning: JudgeLimits, input: SourceSemanticInputFacts,
    encoding: &UntrustedEncoding, work: Int4SemanticSourceBatchLimits,
    limits: CorpusLimits, compute_threads: Option<usize>)
    -> Result<Int4SourceSemanticCorpusRequirements, HostedError> {
    work.validate_item(options.max_model_work).map_err(HostedError::BatchSetup)?;
    options.validate().map_err(task_error)?;
    if options.limits.max_output_bytes > limits.transport.max_output_line_bytes as u64
        || input.source_bytes > limits.transport.max_document_bytes
        || input.question_bytes > limits.transport.max_document_bytes {
        return Err(HostedError::Limits("semantic source corpus document and run ceilings"));
    }
    let transport = limits.reservation_bytes()?;
    let required = semantic_host(work, limits).requirements_with_encoding(width, source,
        options.judge_budget, options.limits, judge_planning, input, encoding)?;
    let staging = transport.checked_sub(limits.preparation_reserve_bytes)
        .ok_or(HostedError::Limits("semantic source corpus shared preparation arithmetic"))?;
    let team_workspace_bytes = compute_threads.map(|threads|
        int4_task_team_workspace_bytes(threads, width)).transpose()?.unwrap_or(0);
    let quote = Int4SourceSemanticCorpusRequirements {
        native: required.native, workspace_bytes: required.native_workspace_bytes,
        preparation_bytes: required.preparation_bytes,
        input_bytes: sum(&[required.preparation_bytes, staging])?,
        output_reservation_bytes: required.output_bytes, team_workspace_bytes,
    };
    quote.total_bytes()?;
    Ok(quote)
}
fn semantic_host(work: Int4SemanticSourceBatchLimits, limits: CorpusLimits) -> Int4SourceSemanticHostLimits {
    Int4SourceSemanticHostLimits { native: limits.native, masks: work.masks.per_mask,
        max_mask_node_visits: work.masks.max_visits_per_item,
        preparation_reserve_bytes: limits.preparation_reserve_bytes }
}

struct ResidentAssets {
    _source: Arc<SourceTaskPlanner>,
    _judge: Arc<Int4JudgePlanner>,
    _vocabulary: Arc<ExtractionVocabulary>,
}
struct StreamInput<R, W> {
    config: Int4SourceSemanticCorpusConfig,
    resident: Option<ResidentAssets>,
    reader: R,
    writer: W,
}
// One output reservation spans the invocation. Per-result clones retain it
// through serialization/flush; panic cannot free the charge before team join.
struct StreamOwners {
    _input: CommittedMemory,
    _kv: CommittedMemory,
    _scratch: CommittedMemory,
    _output: CommittedMemory,
    _lease: EngineLease,
}
struct StreamExecution<R, W> {
    input: StreamInput<R, W>,
    required: Int4SourceSemanticCorpusRequirements,
    owners: Arc<StreamOwners>,
}
struct DeferredSourceSemanticStream<R, W> {
    completion: Result<stream::deferred::DeferredCompletion, HostedError>,
    input: StreamInput<R, W>,
    owners: Arc<StreamOwners>,
}
fn take_execution<R, W>(queued: &Mutex<Option<StreamExecution<R, W>>>)
    -> Result<StreamExecution<R, W>, HostedError> {
    queued.lock().map_err(|_| HostedError::Panicked)?.take().ok_or(HostedError::CompletionMissing)
}

impl NlpEngine {
    /// Run cited summaries or passage answers and explicitly selected complete
    /// source review using one resident model, compiler pair and vocabulary.
    /// Original task results remain unchanged; review is a same-model,
    /// uncalibrated diagnostic. One failed stage emits no combined item result.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_semantic_source<R, W>(&self, model: &ResidentInt4,
        config: Int4SourceSemanticCorpusConfig, kernel: PackedInt4Kernel,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_source_semantic_corpus(model, config, kernel, None, reader, writer, cancellation)
    }

    /// One fixed numerical team spans source generation and every complete
    /// review in every record. Physical join precedes the terminal footer,
    /// with the original cancellation, deadline, work and checkpoint authority.
    /// Blocking IO is cooperative and must return before cleanup completes.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_semantic_source_with_threads<R, W>(&self, model: &ResidentInt4,
        config: Int4SourceSemanticCorpusConfig, kernel: PackedInt4Kernel, compute_threads: usize,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_source_semantic_corpus(model, config, kernel, Some(compute_threads), reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_source_semantic_corpus<R, W>(&self, model: &ResidentInt4,
        config: Int4SourceSemanticCorpusConfig, kernel: PackedInt4Kernel, threads: Option<usize>,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        let run = config.limits.native.run;
        match threads {
            None => dispatch::preflight(self, run)?,
            Some(width) => dispatch::preflight_int4_team(self, run, width)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let required = config.requirements_for_schedule(model.model_identity().group_width(), threads)?;
        let lease = self.resources().acquire_lease();
        let input = Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_reservation_bytes)?;
        let owners = Arc::new(StreamOwners { _input: input.commit()?, _kv: kv.commit()?,
            _scratch: scratch.commit()?, _output: output.commit()?, _lease: lease });
        let execution = StreamExecution {
            input: StreamInput { config, resident: None, reader, writer }, required, owners: Arc::clone(&owners),
        };
        // Formation may discard its callback before joining children. Keep the
        // actual input alive until the dispatcher's physical-completion handoff.
        let queued_owner = Arc::new(Mutex::new(Some(execution)));
        let queued = Arc::clone(&queued_owner);
        let model = model.clone();
        let completed = match threads {
            None => dispatch::run(self, run, cancellation, move |control| {
                let pending = take_execution(&queued)?.run(&model.inner.loaded.value, kernel, None, control);
                let result = pending.publish(control);
                drop(model); drop(owners);
                result
            }),
            Some(width) => dispatch::run_int4_team_with_finalize(self, run, cancellation,
                model, width, kernel, move |weights, control, team| {
                    Ok(take_execution(&queued)?.run(weights, kernel, Some(team), control))
                }, move |pending, control| {
                    let result = pending.publish(control);
                    drop(owners);
                    result
                }),
        };
        drop(queued_owner);
        completed
    }
}

impl<R: BufRead, W: Write> StreamExecution<R, W> {
    fn run<'weights>(self, weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel,
        projection: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> DeferredSourceSemanticStream<R, W> {
        let Self { mut input, required, owners } = self;
        // Actual IO/configuration and resident assets stay outside unwinding.
        // Their owners return for disposal after the numerical workers join.
        let completion = catch_unwind(AssertUnwindSafe(|| {
            checkpoint(control)?;
            let config = &input.config;
            let controls = crate::tokenizer::pinned_controls::pinned().map_err(|_| HostedError::ModelIdentity)?;
            let mut ends = controls.template_controls().entries().iter()
                .filter(|entry| entry.special && entry.surface == crate::template::IM_END);
            let eos = ends.next().ok_or(HostedError::ModelIdentity)?.id;
            if ends.next().is_some() { return Err(HostedError::ModelIdentity); }
            let source = Arc::new(SourceTaskPlanner::pinned(controls.template_controls(), eos)
                .and_then(|planner| planner.with_document_encoding(config.encoding.clone()))
                .map_err(|error| task_error(Int4SourceSemanticError::Source(error.into())))?);
            let identity = source_identity(weights.identity(), &source, &config.task)?;
            let judge = Arc::new(Int4JudgePlanner::for_model(weights.identity(),
                config.options.judge_budget, config.judge_planning)
                .and_then(|planner| planner.with_document_encoding(config.encoding.clone()))
                .map_err(|error| task_error(error.into()))?);
            let vocabulary = Arc::new(ExtractionVocabulary::pinned(controls.template_controls())
                .map_err(|_| HostedError::Limits("semantic source corpus pinned vocabulary"))?);
            input.resident = Some(ResidentAssets { _source: Arc::clone(&source),
                _judge: Arc::clone(&judge), _vocabulary: Arc::clone(&vocabulary) });
            let planner = Int4SemanticSourceBatchPlanner::new(&source, identity, config.task.budget(),
                config.planning, config.task, judge, config.options.clone(), config.work)
                .map_err(HostedError::BatchSetup)?;
            checkpoint(control)?;
            let mut engine = StrictInt4Engine::new(weights, config.limits.native.context_tokens,
                Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                    max_rope_bytes: required.native.rope_bytes,
                    max_scratch_payload_bytes: required.native.scratch_payload_bound })
                .map_err(native_error)?.with_kernel(kernel);
            if let Some(team) = projection {
                engine = engine.with_projection_executor(team).map_err(native_error)?;
            }
            let admission = SemanticAdmission { model: weights.identity(), config, required, owners: &owners };
            let result = {
                let mut processor = NativeInt4SemanticSourceBatch::new(planner, &mut engine,
                    &vocabulary, admission).map_err(HostedError::BatchSetup)?;
                stream::deferred::run_ndjson_deferred(&mut input.reader, &mut input.writer,
                    &mut processor, config.limits.transport, control).map_err(HostedError::Batch)
            };
            drop(engine);
            result
        })).unwrap_or(Err(HostedError::Panicked));
        DeferredSourceSemanticStream { completion, input, owners }
    }
}
impl<R, W: Write> DeferredSourceSemanticStream<R, W> {
    fn publish(self, control: &mut RunControl) -> Result<BatchSummary, HostedError> {
        let Self { completion, mut input, owners } = self;
        let result = completion.and_then(|completion|
            completion.publish(&mut input.writer, control).map_err(HostedError::Batch));
        drop(input);
        drop(owners);
        result
    }
}

fn source_identity(model: &Int4ModelIdentity, planner: &SourceTaskPlanner,
    task: &Int4SemanticSourceBatchTask) -> Result<ExecutionIdentity, HostedError> {
    let absent = Sha256Digest::of_bytes(b"null");
    // Only the loaded model supplies artifact/packing authority. Placeholder
    // fields remain local until binding; the real source plan binds its bytes.
    bind_candidate_model(ExecutionIdentity {
        schema_version: 1, source_revision: "unbound".to_owned(), logical_model_digest: absent,
        artifact_format: "unbound".to_owned(), quant_recipe: "unbound".to_owned(), packing_set_digest: absent,
        tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
        task_spec: task.built_in().spec().identity(), taskir_digest: absent, prompt_digest: absent,
        grammar_compiler_version: "uncompiled".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None,
    }, model).map_err(|_| HostedError::ModelIdentity)
}

struct SemanticAdmission<'a> {
    model: &'a Int4ModelIdentity,
    config: &'a Int4SourceSemanticCorpusConfig,
    required: Int4SourceSemanticCorpusRequirements,
    owners: &'a Arc<StreamOwners>,
}
impl Int4SemanticSourceBatchAdmission for SemanticAdmission<'_> {
    type Guard = Arc<StreamOwners>;
    fn admit(&mut self, request: Int4SemanticSourceAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        for identity in [request.identity, request.judge_identity] {
            constrained_int4::check_model(identity, self.model)
                .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        }
        check_request(self.model.group_width(), self.config, self.required, &request)?;
        Ok((request.identity.clone(), Arc::clone(self.owners)))
    }
}
fn check_request(width: usize, config: &Int4SourceSemanticCorpusConfig,
    required: Int4SourceSemanticCorpusRequirements, request: &Int4SemanticSourceAdmission<'_>)
    -> Result<(), BatchItemFailure> {
    let fail = || BatchItemFailure::fatal(BatchCode::Admission);
    for identity in [request.identity, request.judge_identity] {
        constrained_int4::check_profile(identity, width).map_err(|_| fail())?;
    }
    let ceiling = config.input_ceiling().map_err(|_| fail())?;
    if request.identity.task_spec != config.task.built_in().spec().identity()
        || request.judge_identity.task_spec != "judge-v1"
        || request.group_width != width || request.model_work != config.options.max_model_work
        || !fits(request.model_work, config.work.max_model_work)
        || !nonzero(request.source_reserved_work)
        || !fits(request.source_reserved_work, request.model_work)
        || request.source_reserved_work.forward_positions > config.limits.native.context_tokens as u64
        || request.mask_node_visits != config.work.masks.max_visits_per_item
        || request.mask_limits != config.work.masks.per_mask
        || request.kv_reservation_bytes != required.native.kv_bytes
        || request.source_budget != config.task.budget()
        || request.judge_budget != config.options.judge_budget
        || request.semantic_limits != config.options.limits
        || !same_planning(request.judge_planning_limits, config.judge_planning)
        || request.input_facts.source_bytes == 0
        || request.input_facts.source_bytes > ceiling.source_bytes
        || request.input_facts.question_bytes > ceiling.question_bytes
        || request.input_facts.passage_count > ceiling.passage_count
        || (request.input_facts.passage_count == 0) != (ceiling.passage_count == 0)
        || request.input_encoding != &config.encoding {
        return Err(fail());
    }
    if request.max_result_bytes != config.options.limits.max_output_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::OutputLineLimit));
    }
    let actual = semantic_host(config.work, config.limits).requirements_with_encoding(width,
        request.source_budget, request.judge_budget, request.semantic_limits,
        request.judge_planning_limits, request.input_facts, request.input_encoding).map_err(|_| fail())?;
    if actual.native != required.native || actual.native_workspace_bytes != required.workspace_bytes
        || actual.preparation_bytes > required.preparation_bytes
        || actual.output_bytes > required.output_reservation_bytes {
        return Err(fail());
    }
    Ok(())
}
fn nonzero(work: Int4Work) -> bool {
    work.forward_positions != 0 && work.projected_logits != 0 && work.attention_pairs != 0
        && work.projections.dot_products != 0 && work.projections.multiply_accumulates != 0
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
