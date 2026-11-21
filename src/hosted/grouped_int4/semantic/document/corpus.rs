//! Complete-document object extraction and full-source review on one resident
//! invocation. The compiler, native cache and optional row team span the
//! stream; each original document receives one atomic combined result.

use super::*;
use std::{fmt, io::{BufRead, Write}};
use crate::{
    batch::{self as stream, BatchCode, BatchItemFailure, BatchLimits, BatchSummary,
        extract::semantic::document::{Int4SemanticObjectDocumentAdmission,
            Int4SemanticObjectDocumentBatchAdmission, Int4SemanticObjectDocumentBatchLimits,
            Int4SemanticObjectDocumentBatchPlanner, NativeInt4SemanticObjectDocumentBatch}},
    hosted::corpus::CorpusLimits,
};

/// One fixed original schema, reconciliation policy and correlated reviewer.
/// All transport and native work ceilings cover the whole invocation; an
/// original document is never a caller-supplied chunk or partial success.
#[derive(Clone)]
pub struct Int4SemanticObjectDocumentCorpusConfig {
    pub extraction: ExtractionBatchArgs,
    pub options: Int4SemanticOptions,
    pub encoding: UntrustedEncoding,
    pub host: Int4SemanticObjectDocumentHostLimits,
    pub work: Int4SemanticObjectDocumentBatchLimits,
    pub transport: BatchLimits,
    /// Storage retained by the supplied reader and writer, including buffered
    /// sinks or a whole-corpus cursor. The population is otherwise streamed.
    pub io_reserve_bytes: u64,
}
impl fmt::Debug for Int4SemanticObjectDocumentCorpusConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Int4SemanticObjectDocumentCorpusConfig")
            .field("schema_bytes", &self.extraction.schema.len())
            .field("document_work", &self.options.max_model_work)
            .field("transport", &self.transport).finish_non_exhaustive()
    }
}

/// The existing whole-document peak plus bounded stream/IO staging. Weights
/// and caller-owned configuration are additional. Preparation is a component
/// of input_bytes, not an additional simultaneous charge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SemanticObjectDocumentCorpusRequirements {
    pub native: Int4MemoryRequirement,
    pub workspace_bytes: u64,
    pub preparation_bytes: u64,
    pub input_bytes: u64,
    pub output_reservation_bytes: u64,
    pub team_workspace_bytes: u64,
}
impl Int4SemanticObjectDocumentCorpusRequirements {
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        sum(&[self.native.kv_bytes, self.workspace_bytes, self.input_bytes,
            self.output_reservation_bytes, self.team_workspace_bytes])
    }
}
impl Int4SemanticObjectDocumentCorpusConfig {
    pub fn requirements(&self, width: usize)
        -> Result<Int4SemanticObjectDocumentCorpusRequirements, HostedError> {
        self.requirements_for_schedule(width, None)
    }
    pub fn requirements_with_threads(&self, width: usize, threads: usize)
        -> Result<Int4SemanticObjectDocumentCorpusRequirements, HostedError> {
        self.requirements_for_schedule(width, Some(threads))
    }
    fn requirements_for_schedule(&self, width: usize, threads: Option<usize>)
        -> Result<Int4SemanticObjectDocumentCorpusRequirements, HostedError> {
        if self.extraction.schema.is_empty()
            || self.extraction.schema.len() > self.host.compiler.max_schema_bytes {
            return Err(HostedError::Limits("semantic object corpus original schema size"));
        }
        let mut quote = int4_semantic_object_document_corpus_requirements(width,
            self.extraction.budget, &self.options, &self.encoding, self.host,
            self.work, self.transport, self.io_reserve_bytes, threads)?;
        // Public callers may move a short string with a much larger capacity.
        // The original allocation stays alive beside the compiler's copy.
        let excess = self.extraction.schema.capacity()
            .saturating_sub(self.host.compiler.max_schema_bytes).saturating_sub(1) as u64;
        quote.preparation_bytes = sum(&[quote.preparation_bytes, excess])?;
        quote.input_bytes = sum(&[quote.input_bytes, excess])?;
        quote.total_bytes()?;
        Ok(quote)
    }
}

/// Pure declaration-based admission before original schema, population or
/// model IO. A longer stream changes work authority, not resident geometry.
#[allow(clippy::too_many_arguments)]
pub fn int4_semantic_object_document_corpus_requirements(width: usize, extraction: TaskBudget,
    options: &Int4SemanticOptions, encoding: &UntrustedEncoding,
    host: Int4SemanticObjectDocumentHostLimits, work: Int4SemanticObjectDocumentBatchLimits,
    transport: BatchLimits, io_reserve_bytes: u64, compute_threads: Option<usize>)
    -> Result<Int4SemanticObjectDocumentCorpusRequirements, HostedError> {
    work.validate().map_err(HostedError::BatchSetup)?;
    let mapping = host.extraction.mapping.mapping;
    let verification = host.extraction.mapping.verification;
    if !fits(options.max_model_work, work.max_model_work)
        || work.max_mask_node_visits < mapping.max_mask_visits
        || work.max_verification.max_fields < verification.max_fields
        || work.max_verification.max_matches < verification.max_matches
        || work.max_verification.max_scan_steps < verification.max_scan_steps
        || transport.max_document_bytes != mapping.chunks.max_input_bytes
        || transport.max_work.forward_positions != work.max_model_work.forward_positions
        || transport.max_work.projected_logits != work.max_model_work.projected_logits
        || options.limits.max_output_bytes > transport.max_output_line_bytes as u64 {
        return Err(HostedError::Limits("semantic object corpus document and run ceilings"));
    }
    let staging = CorpusLimits { native: host.native, transport,
        preparation_reserve_bytes: host.preparation_reserve_bytes, io_reserve_bytes }
        .reservation_bytes()?.checked_sub(host.preparation_reserve_bytes)
        .ok_or(HostedError::Limits("semantic object corpus shared preparation arithmetic"))?;
    let document = int4_semantic_object_document_requirements(width, extraction,
        options, encoding, host, compute_threads)?;
    // The run keeps the original raw request and an immutable factory copy.
    // Include their overlap independently of output/result limits. The
    // original schema slot also covers one bounded-reader sentinel byte.
    let preparation = sum(&[document.preparation_bytes,
        host.compiler.max_schema_bytes as u64, options_storage_bytes(options)?, 1])?;
    let quote = Int4SemanticObjectDocumentCorpusRequirements {
        native: document.native, workspace_bytes: document.workspace_bytes,
        preparation_bytes: preparation, input_bytes: sum(&[preparation, staging])?,
        output_reservation_bytes: document.output_reservation_bytes,
        team_workspace_bytes: document.team_workspace_bytes,
    };
    quote.total_bytes()?;
    Ok(quote)
}

struct ResidentAssets {
    _planner: Arc<Int4SemanticObjectDocumentBatchPlanner>,
    _vocabulary: Arc<ExtractionVocabulary>,
}
struct StreamInput<R, W> {
    config: Int4SemanticObjectDocumentCorpusConfig,
    resident: Option<ResidentAssets>,
    reader: R,
    writer: W,
}
struct StreamOwners {
    _input: CommittedMemory,
    _kv: CommittedMemory,
    _scratch: CommittedMemory,
    _output: CommittedMemory,
    _lease: EngineLease,
}
struct StreamExecution<R, W> {
    input: StreamInput<R, W>,
    required: Int4SemanticObjectDocumentCorpusRequirements,
    owners: Arc<StreamOwners>,
}
struct DeferredStream<R, W> {
    completion: Result<stream::deferred::DeferredCompletion, HostedError>,
    input: StreamInput<R, W>,
    owners: Arc<StreamOwners>,
}
fn take_stream<R, W>(queued: &Mutex<Option<StreamExecution<R, W>>>)
    -> Result<StreamExecution<R, W>, HostedError> {
    queued.lock().map_err(|_| HostedError::Panicked)?.take().ok_or(HostedError::CompletionMissing)
}

impl NlpEngine {
    /// Reconcile and review complete original documents in ordered NDJSON with
    /// one resident model and fixed compiler pair. Failed admitted documents
    /// retain their full debit; flush controls never renew any work ceiling.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_semantic_object_documents<R, W>(&self, model: &ResidentInt4,
        config: Int4SemanticObjectDocumentCorpusConfig, kernel: PackedInt4Kernel,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_semantic_object_corpus(model, config, kernel, None, reader, writer, cancellation)
    }

    /// One sealed numerical team spans every chunk, field review and document.
    /// Complete results retain their output owner through serialization/flush;
    /// terminal completion is deferred until all physical workers have joined.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_semantic_object_documents_with_threads<R, W>(&self, model: &ResidentInt4,
        config: Int4SemanticObjectDocumentCorpusConfig, kernel: PackedInt4Kernel, threads: usize,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_semantic_object_corpus(model, config, kernel, Some(threads), reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_semantic_object_corpus<R, W>(&self, model: &ResidentInt4,
        config: Int4SemanticObjectDocumentCorpusConfig, kernel: PackedInt4Kernel, threads: Option<usize>,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        let run = config.host.native.run;
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
        // Retain the actual IO/configuration if team formation discards its
        // callback before joining workers, independently of its memory charge.
        let queued_owner = Arc::new(Mutex::new(Some(execution)));
        let queued = Arc::clone(&queued_owner);
        let model = model.clone();
        let result = match threads {
            None => dispatch::run(self, run, cancellation, move |control| {
                take_stream(&queued)?.run(model.weights(), kernel, None, control).publish(control)
            }),
            Some(width) => dispatch::run_int4_team_with_finalize(self, run, cancellation,
                model, width, kernel, move |weights, control, team| {
                    Ok(take_stream(&queued)?.run(weights, kernel, Some(team), control))
                }, |pending, control| pending.publish(control)),
        };
        drop(queued_owner); drop(owners);
        result
    }
}

impl<R: BufRead, W: Write> StreamExecution<R, W> {
    fn run<'weights>(self, weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel,
        projection: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> DeferredStream<R, W> {
        let Self { mut input, required, owners } = self;
        let completion = catch_unwind(AssertUnwindSafe(|| {
            checkpoint(control)?;
            let config = &input.config;
            let planner = Int4ExtractionBatchPlanner::for_model(weights.identity(), config.extraction.budget,
                config.host.compiler, config.host.source, None).map_err(HostedError::BatchSetup)?;
            let planner = Int4ObjectExtractionPlanner::new(planner).map_err(object_error)?
                .with_document_encoding(config.encoding.clone()).map_err(HostedError::BatchSetup)?;
            let judge = Arc::new(Int4JudgePlanner::for_model(weights.identity(),
                config.options.specification.judge_budget, config.host.judge_planning)
                .map_err(judge_error)?.with_document_encoding(config.encoding.clone()).map_err(judge_error)?);
            let planner = Arc::new(Int4SemanticObjectDocumentBatchPlanner::new(planner,
                config.extraction.clone(), judge, config.options.clone(), config.host.extraction,
                config.host.review, config.work).map_err(HostedError::BatchSetup)?);
            let controls = crate::tokenizer::pinned_controls::pinned().map_err(|_| HostedError::ModelIdentity)?;
            let vocabulary = Arc::new(ExtractionVocabulary::pinned(controls.template_controls())
                .map_err(|error| semantic_error(error.into()))?);
            // Keep the ACTUAL complete factory (both compilers and original
            // schema/review policy) outside the fallible processor lifetime.
            input.resident = Some(ResidentAssets { _planner: Arc::clone(&planner),
                _vocabulary: Arc::clone(&vocabulary) });
            checkpoint(control)?;
            let mut native = StrictInt4Engine::new(weights, config.host.native.context_tokens,
                Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                    max_rope_bytes: required.native.rope_bytes,
                    max_scratch_payload_bytes: required.native.scratch_payload_bound })
                .map_err(native_error)?.with_kernel(kernel);
            if let Some(team) = projection {
                native = native.with_projection_executor(team).map_err(native_error)?;
            }
            let admission = DocumentAdmission { model: weights.identity(), config, required,
                extraction_identity: planner.base_execution_identity().clone(),
                judge_identity: planner.judge_identity().clone(), owners: &owners };
            let result = {
                let mut processor = NativeInt4SemanticObjectDocumentBatch::new_shared(planner,
                    &mut native, &vocabulary, admission).map_err(HostedError::BatchSetup)?;
                stream::deferred::run_ndjson_deferred(&mut input.reader, &mut input.writer,
                    &mut processor, config.transport, control).map_err(HostedError::Batch)
            };
            drop(native);
            result
        })).unwrap_or(Err(HostedError::Panicked));
        DeferredStream { completion, input, owners }
    }
}
impl<R, W: Write> DeferredStream<R, W> {
    fn publish(self, control: &mut RunControl) -> Result<BatchSummary, HostedError> {
        let Self { completion, mut input, owners } = self;
        let result = completion.and_then(|completion|
            completion.publish(&mut input.writer, control).map_err(HostedError::Batch));
        drop(input); drop(owners);
        result
    }
}

struct DocumentAdmission<'a> {
    model: &'a Int4ModelIdentity,
    config: &'a Int4SemanticObjectDocumentCorpusConfig,
    required: Int4SemanticObjectDocumentCorpusRequirements,
    extraction_identity: ExecutionIdentity,
    judge_identity: ExecutionIdentity,
    owners: &'a Arc<StreamOwners>,
}
impl Int4SemanticObjectDocumentBatchAdmission for DocumentAdmission<'_> {
    type Guard = Arc<StreamOwners>;
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
        check_request(self.model.group_width(), self.config, self.required, &request)?;
        let mut admitted = Vec::new();
        admitted.try_reserve_exact(request.identities.len()).map_err(|_| fail())?;
        admitted.extend(request.identities.iter().cloned());
        Ok((admitted, Arc::clone(self.owners)))
    }
}
pub(super) fn check_request(width: usize, config: &Int4SemanticObjectDocumentCorpusConfig,
    required: Int4SemanticObjectDocumentCorpusRequirements, request: &Int4SemanticObjectDocumentAdmission<'_>)
    -> Result<(), BatchItemFailure> {
    let fail = || BatchItemFailure::fatal(BatchCode::Admission);
    let mapping = config.host.extraction.mapping.mapping;
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
        || request.extraction_reserved_work.forward_positions == 0
        || !fits(request.extraction_reserved_work, request.model_work)
        || request.mask_node_visits != mapping.max_mask_visits
        || request.mask_limits != mapping.mask_limits
        || request.verification_budget != config.host.extraction.mapping.verification
        || !same_extraction_limits(request.extraction_limits, config.host.extraction)
        || !same_review_limits(request.review_limits, config.host.review)
        || request.kv_reservation_bytes != required.native.kv_bytes
        || request.max_output_tokens != tokens || request.extraction_budget != config.extraction.budget
        || request.judge_budget != config.options.specification.judge_budget
        || request.semantic_limits != config.options.limits
        || !super::super::corpus::same_planning(request.judge_planning_limits, config.host.judge_planning)
        || request.source_bytes == 0 || request.source_bytes > config.transport.max_document_bytes
        || request.input_encoding != &config.encoding {
        return Err(fail());
    }
    if request.max_result_bytes != config.options.limits.max_output_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::OutputLineLimit));
    }
    let actual = int4_semantic_object_document_requirements(width, request.extraction_budget,
        &config.options, request.input_encoding, config.host, None).map_err(|_| fail())?;
    if actual.native != required.native || actual.workspace_bytes != required.workspace_bytes
        || actual.preparation_bytes > required.preparation_bytes
        || actual.output_reservation_bytes > required.output_reservation_bytes {
        return Err(fail());
    }
    Ok(())
}

pub(super) fn same_extraction_limits(a: Int4ObjectExtractionMapLimits, b: Int4ObjectExtractionMapLimits) -> bool {
    let x = a.mapping.mapping;
    let y = b.mapping.mapping;
    a.reconciliation == b.reconciliation && a.mapping.verification == b.mapping.verification
        && a.mapping.root_array_union.is_none() && b.mapping.root_array_union.is_none()
        && x.chunks == y.chunks && x.reduction == y.reduction && x.max_model_work == y.max_model_work
        && x.mask_limits == y.mask_limits && x.mask_visits_per_chunk == y.mask_visits_per_chunk
        && x.max_mask_visits == y.max_mask_visits
}
pub(super) fn same_review_limits(a: Int4FaithfulnessMapLimits, b: Int4FaithfulnessMapLimits) -> bool {
    a.chunks == b.chunks && a.max_model_work == b.max_model_work
        && a.max_retained_result_bytes == b.max_retained_result_bytes && a.max_result_bytes == b.max_result_bytes
        && a.aggregation.minimum_classified_coverage_ppm == b.aggregation.minimum_classified_coverage_ppm
}

#[cfg(test)] mod tests;
