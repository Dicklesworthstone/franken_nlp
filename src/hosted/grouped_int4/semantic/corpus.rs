//! One resident extraction compiler and correlated reader for ordered NDJSON.
//! Every record retains its original schema-valid extraction and complete
//! full-source review. Document and run work ceilings never renew at a flush.

use super::*;
use std::{fmt, io::{BufRead, Write}, panic::{catch_unwind, AssertUnwindSafe}, sync::Mutex};
use crate::{
    batch::{self as stream, BatchCode, BatchItemFailure, BatchSummary,
        extract::{ExtractionBatchArgs, grouped_int4::Int4ExtractionBatchPlanner,
            semantic::{Int4SemanticExtractionAdmission, Int4SemanticExtractionBatchAdmission,
                Int4SemanticExtractionBatchLimits, Int4SemanticExtractionBatchPlanner,
                NativeInt4SemanticExtractionBatch}}},
    grammar::{CompileLimits, runtime::SourceRuntimeLimits},
    hosted::corpus::CorpusLimits,
    native_engine::strict_int4::projection::Int4ProjectionExecutor,
    tasks::{extract::semantic::grouped_int4::Int4SemanticOptions,
        judge::grouped_int4::Int4JudgePlanner},
};

/// Fixed raw schema, claim rules and document budgets for the complete run.
/// Record task_args overrides are refused before encoding. Preparation must
/// price the supplied schema/spec capacities, pinned compilers and vocabulary;
/// the quote independently adds every retained source, field plan and result.
#[derive(Clone)]
pub struct Int4SemanticCorpusConfig {
    pub extraction: ExtractionBatchArgs,
    pub compiler_limits: CompileLimits,
    pub source_limits: SourceRuntimeLimits,
    pub options: Int4SemanticOptions,
    pub judge_planning: JudgeLimits,
    pub encoding: UntrustedEncoding,
    pub work: Int4SemanticExtractionBatchLimits,
    pub limits: CorpusLimits,
}
impl fmt::Debug for Int4SemanticCorpusConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Int4SemanticCorpusConfig")
            .field("schema_bytes", &self.extraction.schema.len())
            .field("extraction_budget", &self.extraction.budget)
            .field("semantic_limits", &self.options.limits)
            .field("document_work", &self.options.max_model_work)
            .field("run_limits", &self.limits)
            .finish_non_exhaustive()
    }
}

/// Simultaneous admitted payload excluding weights, caller-owned preparation
/// and dispatch cleanup. Team scratch is separately reserved by the dispatcher.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SemanticCorpusRequirements {
    pub native: Int4MemoryRequirement,
    pub workspace_bytes: u64,
    /// Complete semantic compiler/source/review preparation, a component of
    /// input_bytes rather than an additional simultaneous charge.
    pub preparation_bytes: u64,
    /// Semantic preparation plus bounded transport and supplied IO storage.
    pub input_bytes: u64,
    pub output_reservation_bytes: u64,
    pub team_workspace_bytes: u64,
}
impl Int4SemanticCorpusRequirements {
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        sum(&[self.native.kv_bytes, self.workspace_bytes, self.input_bytes,
            self.output_reservation_bytes, self.team_workspace_bytes])
    }
}
impl Int4SemanticCorpusConfig {
    pub fn requirements(&self, width: usize) -> Result<Int4SemanticCorpusRequirements, HostedError> {
        self.requirements_for_schedule(width, None)
    }
    pub fn requirements_with_threads(&self, width: usize, compute_threads: usize)
        -> Result<Int4SemanticCorpusRequirements, HostedError> {
        self.requirements_for_schedule(width, Some(compute_threads))
    }
    fn requirements_for_schedule(&self, width: usize, threads: Option<usize>)
        -> Result<Int4SemanticCorpusRequirements, HostedError> {
        if self.extraction.schema.is_empty()
            || self.extraction.schema.len() > self.compiler_limits.max_schema_bytes {
            return Err(HostedError::Limits("semantic corpus fixed schema size"));
        }
        int4_semantic_corpus_requirements(width, self.extraction.budget, &self.options,
            self.judge_planning, &self.encoding, self.work, self.limits, threads)
    }
}

/// Pure peak quote before schema, private population or model IO. The source
/// bound is one decoded document, never the whole input stream. A longer run
/// changes work authority without multiplying the resident memory geometry.
#[allow(clippy::too_many_arguments)]
pub fn int4_semantic_corpus_requirements(width: usize, extraction: TaskBudget,
    options: &Int4SemanticOptions, judge_planning: JudgeLimits, encoding: &UntrustedEncoding,
    work: Int4SemanticExtractionBatchLimits, limits: CorpusLimits, compute_threads: Option<usize>)
    -> Result<Int4SemanticCorpusRequirements, HostedError> {
    work.validate().map_err(HostedError::BatchSetup)?;
    options.validate().map_err(semantic_error)?;
    validate_judge_planning(judge_planning)?;
    if !fits(options.max_model_work, work.max_model_work)
        || options.limits.max_output_bytes > limits.transport.max_output_line_bytes as u64 {
        return Err(HostedError::Limits("semantic corpus document and run ceilings"));
    }
    let transport = limits.reservation_bytes()?;
    let required = semantic_host(work, limits).requirements_with_encoding(width, extraction,
        options.specification.judge_budget, options.limits, judge_planning,
        limits.transport.max_document_bytes, encoding)?;
    let staging = transport.checked_sub(limits.preparation_reserve_bytes)
        .ok_or(HostedError::Limits("semantic corpus shared preparation arithmetic"))?;
    let team_workspace_bytes = compute_threads.map(|threads|
        int4_task_team_workspace_bytes(threads, width)).transpose()?.unwrap_or(0);
    let quote = Int4SemanticCorpusRequirements {
        native: required.native, workspace_bytes: required.native_workspace_bytes,
        preparation_bytes: required.preparation_bytes,
        input_bytes: sum(&[required.preparation_bytes, staging])?,
        output_reservation_bytes: required.output_bytes, team_workspace_bytes,
    };
    quote.total_bytes()?;
    Ok(quote)
}

fn semantic_host(work: Int4SemanticExtractionBatchLimits, limits: CorpusLimits) -> Int4SemanticHostLimits {
    Int4SemanticHostLimits { native: limits.native, masks: work.masks.per_mask,
        max_mask_node_visits: work.masks.max_visits_per_item,
        preparation_reserve_bytes: limits.preparation_reserve_bytes }
}
pub(super) fn validate_judge_planning(limits: JudgeLimits) -> Result<(), HostedError> {
    if limits.max_total_prompt_tokens == 0 || limits.max_total_candidate_tokens == 0
        || limits.max_total_projected_logits == 0 || limits.max_output_bytes == 0
        || limits.per_head.max_candidates == 0 || limits.per_head.max_total_tokens == 0
        || limits.per_head.max_nodes == 0 || limits.per_head.max_depth == 0
        || limits.per_head.max_candidate_id_bytes == 0 || limits.per_head.max_projected_logits == 0 {
        return Err(HostedError::Limits("semantic corpus complete judge planning"));
    }
    Ok(())
}

struct ResidentAssets {
    _judge: Arc<Int4JudgePlanner>,
    _vocabulary: Arc<ExtractionVocabulary>,
}
struct StreamInput<R, W> {
    config: Int4SemanticCorpusConfig,
    resident: Option<ResidentAssets>,
    reader: R,
    writer: W,
}
// Reserve the complete output maximum once for the invocation. Each successful
// result carries a cloned owner through serialization/flush, while a writer
// panic cannot release its charge ahead of the physical worker join.
struct StreamOwners {
    _input: CommittedMemory,
    _kv: CommittedMemory,
    _scratch: CommittedMemory,
    _output: CommittedMemory,
    _lease: EngineLease,
}
struct StreamExecution<R, W> {
    input: StreamInput<R, W>,
    required: Int4SemanticCorpusRequirements,
    owners: Arc<StreamOwners>,
}
struct DeferredSemanticStream<R, W> {
    completion: Result<stream::deferred::DeferredCompletion, HostedError>,
    input: StreamInput<R, W>,
    owners: Arc<StreamOwners>,
}
fn take_execution<R, W>(queued: &Mutex<Option<StreamExecution<R, W>>>)
    -> Result<StreamExecution<R, W>, HostedError> {
    queued.lock().map_err(|_| HostedError::Panicked)?.take().ok_or(HostedError::CompletionMissing)
}

impl NlpEngine {
    /// Extract and explicitly review bounded source records using one resident
    /// model, compiler pair and vocabulary. The exact original extraction is
    /// retained; the correlated diagnostic never repairs or accepts a field.
    /// One failed stage publishes no combined document result.
    ///
    /// Supplied IO storage must fit the explicit reservation. Blocking IO is
    /// cooperative; cancellation waits for physical completion and cleanup.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_semantic_extract<R, W>(&self, model: &ResidentInt4,
        config: Int4SemanticCorpusConfig, kernel: PackedInt4Kernel,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_semantic_corpus(model, config, kernel, None, reader, writer, cancellation)
    }

    /// One fixed numerical team spans extraction and every field review in
    /// every record. It joins before the terminal footer. Native work, masks,
    /// input encoding, deadlines and checkpoints retain their original scope.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_semantic_extract_with_threads<R, W>(&self, model: &ResidentInt4,
        config: Int4SemanticCorpusConfig, kernel: PackedInt4Kernel, compute_threads: usize,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_semantic_corpus(model, config, kernel, Some(compute_threads), reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_semantic_corpus<R, W>(&self, model: &ResidentInt4,
        config: Int4SemanticCorpusConfig, kernel: PackedInt4Kernel, threads: Option<usize>,
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
        // Formation can discard its callback before joining children. Retain
        // actual input owners until the dispatcher's physical-completion handoff.
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
        control: &mut RunControl) -> DeferredSemanticStream<R, W> {
        let Self { mut input, required, owners } = self;
        // Configuration, IO, shared judge/vocabulary and all charges remain
        // outside the unwind boundary and return for post-join disposal.
        let completion = catch_unwind(AssertUnwindSafe(|| {
            checkpoint(control)?;
            let config = &input.config;
            let extraction = Int4ExtractionBatchPlanner::for_model(weights.identity(),
                config.extraction.budget, config.compiler_limits, config.source_limits,
                Some(config.extraction.clone()))
                .and_then(|planner| planner.with_document_encoding(config.encoding.clone()))
                .map_err(HostedError::BatchSetup)?;
            let judge = Arc::new(Int4JudgePlanner::for_model(weights.identity(),
                config.options.specification.judge_budget, config.judge_planning)
                .and_then(|planner| planner.with_document_encoding(config.encoding.clone()))
                .map_err(|error| semantic_error(error.into()))?);
            let controls = crate::tokenizer::pinned_controls::pinned()
                .map_err(|_| HostedError::ModelIdentity)?;
            let vocabulary = Arc::new(ExtractionVocabulary::pinned(controls.template_controls())
                .map_err(|error| semantic_error(error.into()))?);
            input.resident = Some(ResidentAssets { _judge: Arc::clone(&judge), _vocabulary: Arc::clone(&vocabulary) });
            let planner = Int4SemanticExtractionBatchPlanner::new(extraction, judge,
                config.options.clone(), config.work).map_err(HostedError::BatchSetup)?;
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
                let mut processor = NativeInt4SemanticExtractionBatch::new(planner, &mut engine,
                    &vocabulary, admission).map_err(HostedError::BatchSetup)?;
                stream::deferred::run_ndjson_deferred(&mut input.reader, &mut input.writer,
                    &mut processor, config.limits.transport, control).map_err(HostedError::Batch)
            };
            drop(engine);
            result
        })).unwrap_or(Err(HostedError::Panicked));
        DeferredSemanticStream { completion, input, owners }
    }
}
impl<R, W: Write> DeferredSemanticStream<R, W> {
    fn publish(self, control: &mut RunControl) -> Result<BatchSummary, HostedError> {
        let Self { completion, mut input, owners } = self;
        let result = completion.and_then(|completion|
            completion.publish(&mut input.writer, control).map_err(HostedError::Batch));
        drop(input);
        drop(owners);
        result
    }
}

struct SemanticAdmission<'a> {
    model: &'a Int4ModelIdentity,
    config: &'a Int4SemanticCorpusConfig,
    required: Int4SemanticCorpusRequirements,
    owners: &'a Arc<StreamOwners>,
}
impl Int4SemanticExtractionBatchAdmission for SemanticAdmission<'_> {
    type Guard = Arc<StreamOwners>;
    fn admit(&mut self, request: Int4SemanticExtractionAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        for identity in [request.identity, request.judge_identity] {
            constrained_int4::check_model(identity, self.model)
                .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        }
        check_request(self.model.group_width(), self.config, self.required, &request)?;
        Ok((request.identity.clone(), Arc::clone(self.owners)))
    }
}
fn check_request(width: usize, config: &Int4SemanticCorpusConfig,
    required: Int4SemanticCorpusRequirements, request: &Int4SemanticExtractionAdmission<'_>)
    -> Result<(), BatchItemFailure> {
    let fail = || BatchItemFailure::fatal(BatchCode::Admission);
    for identity in [request.identity, request.judge_identity] {
        constrained_int4::check_profile(identity, width).map_err(|_| fail())?;
    }
    if request.identity.task_spec != "extract-v1" || request.judge_identity.task_spec != "judge-v1"
        || request.group_width != width || request.model_work != config.options.max_model_work
        || !fits(request.model_work, config.work.max_model_work)
        || !nonzero(request.extraction_reserved_work)
        || !fits(request.extraction_reserved_work, request.model_work)
        || request.extraction_reserved_work.forward_positions > config.limits.native.context_tokens as u64
        || request.mask_node_visits != config.work.masks.max_visits_per_item
        || request.mask_limits != config.work.masks.per_mask
        || request.kv_reservation_bytes != required.native.kv_bytes
        || request.extraction_budget != config.extraction.budget
        || request.judge_budget != config.options.specification.judge_budget
        || request.semantic_limits != config.options.limits
        || !same_planning(request.judge_planning_limits, config.judge_planning)
        || request.source_bytes == 0 || request.source_bytes > config.limits.transport.max_document_bytes
        || request.input_encoding != &config.encoding {
        return Err(fail());
    }
    if request.max_result_bytes != config.options.limits.max_output_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::OutputLineLimit));
    }
    let actual = semantic_host(config.work, config.limits).requirements_with_encoding(width,
        request.extraction_budget, request.judge_budget, request.semantic_limits,
        request.judge_planning_limits, request.source_bytes, request.input_encoding).map_err(|_| fail())?;
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
pub(super) fn same_planning(a: JudgeLimits, b: JudgeLimits) -> bool {
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
