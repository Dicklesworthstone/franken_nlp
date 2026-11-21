//! One admitted complete-document object extraction and full-source review.
//! The original object, every partial-map result and all field review plans
//! coexist. One native cache, encoder workspace and optional numerical team
//! serve the complete invocation without renewing any task allowance.

use super::*;
mod corpus;
#[cfg(all(feature = "metadata-store", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
mod jobs;
#[cfg(all(feature = "metadata-store", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use jobs::{Int4SemanticObjectDocumentJobHostLimits, Int4SemanticObjectDocumentJobRequirements,
    int4_semantic_object_job_requirements};
pub use corpus::{Int4SemanticObjectDocumentCorpusConfig, Int4SemanticObjectDocumentCorpusRequirements,
    int4_semantic_object_document_corpus_requirements};
use std::{panic::{catch_unwind, AssertUnwindSafe}, sync::Mutex};
use crate::{
    batch::extract::{ExtractionBatchArgs, grouped_int4::Int4ExtractionBatchPlanner,
        grouped_int4::long::object::{Int4ObjectExtractionMapError, Int4ObjectExtractionMapLimits,
            Int4ObjectExtractionPlanner}},
    grammar::{CompileLimits, runtime::SourceRuntimeLimits},
    native_engine::strict_int4::projection::Int4ProjectionExecutor,
    tasks::{extract::{object_reconcile::RootObjectSchema, semantic::{ClaimPathStep,
        grouped_int4::{Int4SemanticOptions, document::{Int4SemanticObjectDocumentError,
            Int4SemanticObjectDocumentRun, PreparedInt4SemanticObjectDocument}}}},
        judge::grouped_int4::{Int4JudgePlanner, faithfulness_map::{Int4FaithfulnessMapLimits,
            memory::{Int4FaithfulnessMapMemoryLimits, MIN_FAITHFULNESS_MAP_PREPARATION_BYTES}}}},
};

/// Fixed declarations usable before reading a schema, source or model. The
/// source compiler owns each retained chunk program; the review compiler is
/// shared, while all complete field maps remain independently owned.
#[derive(Clone, Copy, Debug)]
pub struct Int4SemanticObjectDocumentHostLimits {
    pub native: NativeLimits,
    pub compiler: CompileLimits,
    pub source: SourceRuntimeLimits,
    pub extraction: Int4ObjectExtractionMapLimits,
    pub judge_planning: JudgeLimits,
    pub review: Int4FaithfulnessMapLimits,
    pub preparation_reserve_bytes: u64,
    pub reduction_reserve_bytes: u64,
}

/// Complete simultaneous modeled payload. Resident weights, caller-owned
/// preparation and runtime cleanup are additional; this is not measured RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SemanticObjectDocumentRequirements {
    pub native: Int4MemoryRequirement,
    /// One native RoPE/scratch owner and allocator reserve; excludes the team.
    pub workspace_bytes: u64,
    /// Original inputs/configuration, schemas, retained extraction programs,
    /// reconciliation, claims and every complete field map/result owner.
    pub preparation_bytes: u64,
    /// Complete combined result plus canonical serialization and token storage.
    pub output_reservation_bytes: u64,
    /// Separately reserved by the existing fixed-team dispatcher.
    pub team_workspace_bytes: u64,
}
impl Int4SemanticObjectDocumentRequirements {
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        sum(&[self.native.kv_bytes, self.workspace_bytes, self.preparation_bytes,
            self.output_reservation_bytes, self.team_workspace_bytes])
    }
}

/// Price the full declared peak before private schema/source/model IO. Every
/// field retains a complete faithfulness map, including construction of the
/// next chunk before a work check can refuse it. Only the immutable compiler,
/// native cache and transient encoder workspace are shared between fields.
#[allow(clippy::too_many_arguments)]
pub fn int4_semantic_object_document_requirements(width: usize, extraction_budget: TaskBudget,
    options: &Int4SemanticOptions, encoding: &UntrustedEncoding,
    limits: Int4SemanticObjectDocumentHostLimits, compute_threads: Option<usize>)
    -> Result<Int4SemanticObjectDocumentRequirements, HostedError> {
    options.validate().map_err(semantic_error)?;
    validate_limits(extraction_budget, options, limits)?;
    let m = limits.extraction.mapping.mapping;
    let fields = options.limits.max_fields as u64;
    let chunks = m.chunks.max_chunks as u64;
    let native = Int4MemoryRequirement::for_context(limits.native.context_tokens, width).map_err(native_error)?;
    if native.kv_bytes > extraction_budget.max_kv_bytes
        || native.kv_bytes > options.specification.judge_budget.max_kv_bytes {
        return Err(HostedError::Limits("semantic object document full resident KV for both stages"));
    }
    let encoding_workspace = crate::batch::extract::grouped_int4::memory::encoding_workspace_bytes(encoding)
        .map_err(HostedError::BatchSetup)?;
    let schemas = RootObjectSchema::workspace_bytes(limits.compiler)
        .map_err(|error| object_error(error.into()))?;
    let reconciliation = limits.extraction.reconciliation_workspace_bytes().map_err(object_error)?;
    let extraction_results = super::super::extraction_map::document_workspace_bytes(extraction_budget,
        limits.extraction.mapping, m.chunks.max_chunks, limits.reduction_reserve_bytes)?;
    // The compiled partial schema and source grammar belong to EACH native
    // chunk, unlike the shared pinned tokenizer/vocabulary. The declaration
    // quote covers lexical/path copies and node construction overlap; native
    // automata, masks, source indexes, prompts and identities add separately.
    // A structural-only schema may use less, but private schema inspection is
    // deliberately unnecessary for this conservative pre-IO declaration.
    let chunk_program = sum(&[schemas, product(limits.compiler.max_transitions as u64, 64)?,
        limits.compiler.max_mask_bytes as u64, limits.source.index.max_index_bytes as u64,
        product(u64::from(extraction_budget.max_input_tokens), 64)?, 65_536])?;
    let extraction_plans = product(chunks, chunk_program)?;
    let review = Int4FaithfulnessMapMemoryLimits {
        context_tokens: limits.native.context_tokens,
        preparation_reserve_bytes: limits.preparation_reserve_bytes,
        allocator_reserve_bytes: limits.native.allocator_reserve_bytes,
    }.requirements_with_max_heads(width, options.specification.judge_budget, limits.judge_planning,
        limits.review, options.specification.judge_policy.max_evidence_windows as usize + 1)
        .map_err(faithfulness_error)?;
    if review.native != native {
        return Err(HostedError::Limits("semantic object document shared native quote"));
    }
    // Exactly one pinned judge compiler is built and shared by every field.
    // Keep every per-field source/claim, plan, construction-overlap, retained
    // result, lifted evidence and map-serialization term from its own quote.
    let review_variable = review.input_bytes.checked_sub(limits.preparation_reserve_bytes)
        .ok_or(HostedError::Limits("semantic object document shared preparation arithmetic"))?;
    let reviews = product(fields, sum(&[review_variable, review.output_reservation_bytes])?)?;
    let semantic = options.limits;
    let claims = sum(&[
        product(semantic.max_input_json_bytes as u64, 128)?,
        product(semantic.max_walk_nodes as u64, 128)?,
        product(product(fields, semantic.max_input_json_bytes as u64)?, 2)?,
        product(semantic.max_total_claim_bytes as u64, 4)?,
        product(semantic.max_template_bytes as u64, 4)?,
        product(options_storage_bytes(options)?, 2)?,
    ])?;
    let tokens = product(chunks, u64::from(extraction_budget.max_output_tokens))?;
    let required = Int4SemanticObjectDocumentRequirements {
        native, workspace_bytes: review.workspace_bytes,
        preparation_bytes: sum(&[limits.preparation_reserve_bytes, encoding_workspace,
            product(m.chunks.max_input_bytes as u64, 4)?, limits.compiler.max_schema_bytes as u64,
            schemas, reconciliation, extraction_plans, extraction_results, reviews, claims])?,
        output_reservation_bytes: sum(&[product(semantic.max_output_bytes, 8)?, product(tokens, 8)?, 4096])?,
        team_workspace_bytes: compute_threads.map(|threads|
            int4_task_team_workspace_bytes(threads, width)).transpose()?.unwrap_or(0),
    };
    required.total_bytes()?;
    Ok(required)
}

fn validate_limits(task: TaskBudget, options: &Int4SemanticOptions,
    limits: Int4SemanticObjectDocumentHostLimits) -> Result<(), HostedError> {
    limits.native.run.validate()?;
    task.validate().map_err(|_| HostedError::Limits("semantic object document extraction task"))?;
    super::corpus::validate_judge_planning(limits.judge_planning)?;
    let m = limits.extraction.mapping.mapping;
    let r = m.reduction;
    let v = limits.extraction.mapping.verification;
    let c = limits.compiler;
    let s = limits.source;
    m.chunks.effective_token_limit().map_err(|_| HostedError::Limits("semantic object document chunk geometry"))?;
    limits.review.validate(options.specification.judge_budget).map_err(faithfulness_error)?;
    limits.extraction.reconciliation_workspace_bytes().map_err(object_error)?;
    if limits.preparation_reserve_bytes < MIN_FAITHFULNESS_MAP_PREPARATION_BYTES
        || limits.reduction_reserve_bytes == 0 || limits.native.allocator_reserve_bytes == 0
        || m.chunks.context_tokens != limits.native.context_tokens
        || limits.review.chunks.context_tokens > limits.native.context_tokens
        || m.chunks.max_input_bytes == 0 || !(1..=256).contains(&m.chunks.max_chunks)
        || r.map_batch_chunks != 1 || !(2..=1024).contains(&r.reduce_fan_in)
        || r.max_reduction_levels > 64 || !(1..=1_000_000).contains(&r.max_task_calls)
        || [r.max_value_bytes, r.max_live_value_bytes, r.max_total_value_bytes, r.max_result_bytes]
            .iter().any(|n| !(1..=64 * 1024 * 1024).contains(n))
        || r.max_value_bytes > r.max_live_value_bytes
        || r.max_result_bytes as u64 > options.limits.max_output_bytes
        || limits.review.max_result_bytes > options.limits.max_output_bytes
        || limits.extraction.reconciliation.max_json_bytes > options.limits.max_input_json_bytes
        || m.mask_limits.max_trie_node_visits == 0 || m.mask_limits.checkpoint_interval_nodes == 0
        || m.mask_visits_per_chunk == 0 || m.max_mask_visits < m.mask_visits_per_chunk
        || !positive_work(m.max_model_work)
        || !(1..=1_000_000).contains(&v.max_fields) || !(1..=1_000_000).contains(&v.max_matches)
        || v.max_scan_steps == 0
        || [c.max_schema_bytes, c.max_string_bytes, c.max_array_items, c.max_output_bytes,
            c.max_states, c.max_transitions, c.max_mask_bytes].contains(&0)
        || s.index.max_source_bytes == 0 || s.index.max_index_bytes == 0
        || s.index.max_build_steps == 0 || s.index.max_matches == 0
        || s.verification.max_fields == 0 || s.verification.max_matches == 0 || s.verification.max_scan_steps == 0 {
        return Err(HostedError::Limits("semantic object document complete preparation and work declarations"));
    }
    let reviewed_input = m.chunks.max_input_bytes.checked_add(options.limits.max_claim_bytes)
        .ok_or(HostedError::Limits("semantic object document source and claim arithmetic"))?;
    if reviewed_input > limits.review.chunks.max_input_bytes {
        return Err(HostedError::Limits("semantic object document review must cover the entire source and claim"));
    }
    Ok(())
}
fn positive_work(work: Int4Work) -> bool {
    work.forward_positions != 0 && work.projected_logits != 0 && work.attention_pairs != 0
        && work.projections.dot_products != 0 && work.projections.multiply_accumulates != 0
}
fn options_storage_bytes(options: &Int4SemanticOptions) -> Result<u64, HostedError> {
    let spec = &options.specification;
    let mut bytes = sum(&[std::mem::size_of::<Int4SemanticOptions>() as u64, spec.revision.capacity() as u64,
        product(spec.claims.capacity() as u64, std::mem::size_of::<crate::tasks::extract::semantic::ClaimRule>() as u64)?])?;
    for rule in &spec.claims {
        bytes = sum(&[bytes, rule.id.capacity() as u64, rule.prefix.capacity() as u64,
            rule.suffix.capacity() as u64,
            product(rule.path.capacity() as u64, std::mem::size_of::<ClaimPathStep>() as u64)?])?;
        for step in &rule.path {
            if let ClaimPathStep::Property(name) = step { bytes = sum(&[bytes, name.capacity() as u64])?; }
        }
    }
    Ok(bytes)
}

struct DocumentInput {
    source: String,
    request: ExtractionBatchArgs,
    options: Int4SemanticOptions,
    encoding: UntrustedEncoding,
    planner: Option<Arc<Int4ObjectExtractionPlanner>>,
    judge: Option<Arc<Int4JudgePlanner>>,
    vocabulary: Option<Arc<ExtractionVocabulary>>,
}
/// These owners stay outside every fallible callback and native unwind. The
/// complete input separately survives a discarded team-formation callback.
struct DocumentOwners {
    _preparation: CommittedMemory,
    _kv: CommittedMemory,
    _scratch: CommittedMemory,
    output: Mutex<Option<CommittedMemory>>,
    _lease: EngineLease,
}
struct DocumentExecution {
    input: DocumentInput,
    limits: Int4SemanticObjectDocumentHostLimits,
    required: Int4SemanticObjectDocumentRequirements,
    owners: Arc<DocumentOwners>,
}
struct DeferredDocument {
    result: Result<Int4SemanticObjectDocumentRun, HostedError>,
    compute: Option<Int4ComputeReceipt>,
    input: DocumentInput,
    owners: Arc<DocumentOwners>,
}
impl DeferredDocument {
    fn finish(self, control: &mut RunControl)
        -> Result<(HostedOutput<Int4SemanticObjectDocumentRun>, Option<Int4ComputeReceipt>), HostedError> {
        checkpoint(control)?;
        let Self { result, compute, input, owners } = self;
        let result = result?;
        let memory = owners.output.lock().map_err(|_| HostedError::Panicked)?.take()
            .ok_or(HostedError::CompletionMissing)?;
        drop(input); drop(owners);
        Ok((GuardedOutput::new(result, memory), compute))
    }
}
fn take_execution(queued: &Mutex<Option<DocumentExecution>>) -> Result<DocumentExecution, HostedError> {
    queued.lock().map_err(|_| HostedError::Panicked)?.take().ok_or(HostedError::CompletionMissing)
}

impl NlpEngine {
    /// Extract every source chunk, reconcile one original-schema root object,
    /// then review each configured normalized scalar against every original
    /// source chunk. The original object stays unchanged; review is an explicit
    /// correlated, uncalibrated diagnostic with no repair or acceptance power.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_int4_semantic_object_document(&self, model: &ResidentInt4, source: String,
        request: ExtractionBatchArgs, options: Int4SemanticOptions, encoding: UntrustedEncoding,
        limits: Int4SemanticObjectDocumentHostLimits, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4SemanticObjectDocumentRun>, HostedError> {
        self.run_int4_semantic_object_document(model, source, request, options, encoding, limits,
            kernel, None, cancellation).map(|(result, _)| result)
    }

    /// One admitted numerical row team serves the complete extraction and all
    /// field maps. The full input/plan/vocabulary owners and every memory claim
    /// survive physical join, including failure, cancellation and unwinding.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_int4_semantic_object_document_with_threads(&self, model: &ResidentInt4, source: String,
        request: ExtractionBatchArgs, options: Int4SemanticOptions, encoding: UntrustedEncoding,
        limits: Int4SemanticObjectDocumentHostLimits, kernel: PackedInt4Kernel, compute_threads: usize,
        cancellation: CancellationToken) -> Result<HostedInt4TeamResult<Int4SemanticObjectDocumentRun>, HostedError> {
        let (output, compute) = self.run_int4_semantic_object_document(model, source, request, options,
            encoding, limits, kernel, Some(compute_threads), cancellation)?;
        Ok(HostedInt4TeamResult::new(output, compute.ok_or(HostedError::CompletionMissing)?))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_semantic_object_document(&self, model: &ResidentInt4, source: String,
        request: ExtractionBatchArgs, options: Int4SemanticOptions, encoding: UntrustedEncoding,
        limits: Int4SemanticObjectDocumentHostLimits, kernel: PackedInt4Kernel, threads: Option<usize>,
        cancellation: CancellationToken)
        -> Result<(HostedOutput<Int4SemanticObjectDocumentRun>, Option<Int4ComputeReceipt>), HostedError> {
        match threads {
            None => dispatch::preflight(self, limits.native.run)?,
            Some(width) => dispatch::preflight_int4_team(self, limits.native.run, width)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let max_source = limits.extraction.mapping.mapping.chunks.max_input_bytes;
        if source.is_empty() || source.len() > max_source || request.schema.is_empty()
            || request.schema.len() > limits.compiler.max_schema_bytes {
            return Err(HostedError::Limits("semantic object document original schema and source bounds"));
        }
        let mut required = int4_semantic_object_document_requirements(model.model_identity().group_width(),
            request.budget, &options, &encoding, limits, threads)?;
        // Moved strings may retain excess capacity even when their visible
        // content is tiny. Price those actual allocations without copying them.
        required.preparation_bytes = sum(&[required.preparation_bytes,
            source.capacity().saturating_sub(max_source) as u64,
            request.schema.capacity().saturating_sub(limits.compiler.max_schema_bytes) as u64])?;
        required.total_bytes()?;
        let lease = self.resources().acquire_lease();
        let preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, required.preparation_bytes)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_reservation_bytes)?;
        let owners = Arc::new(DocumentOwners { _preparation: preparation.commit()?, _kv: kv.commit()?,
            _scratch: scratch.commit()?, output: Mutex::new(Some(output.commit()?)), _lease: lease });
        let execution = DocumentExecution {
            input: DocumentInput { source, request, options, encoding, planner: None, judge: None, vocabulary: None },
            limits, required, owners: Arc::clone(&owners),
        };
        let queued_owner = Arc::new(Mutex::new(Some(execution)));
        let queued = Arc::clone(&queued_owner);
        let model = model.clone();
        let result = match threads {
            None => dispatch::run(self, limits.native.run, cancellation, move |control|
                take_execution(&queued)?.run(model.weights(), kernel, None, control).finish(control)),
            Some(width) => dispatch::run_int4_team_with_finalize(self, limits.native.run, cancellation,
                model, width, kernel, move |weights, control, team| {
                    let mut pending = take_execution(&queued)?.run(weights, kernel, Some(&mut *team), control);
                    // The semantic engine has dropped its exclusive executor
                    // borrow; the dispatcher still owns and joins this team.
                    pending.compute = Some(team.stats().into());
                    Ok(pending)
                },
                |pending, control| pending.finish(control)),
        };
        drop(queued_owner); drop(owners);
        result
    }
}

impl DocumentExecution {
    fn run<'weights>(self, weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel,
        projection: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> DeferredDocument {
        let Self { mut input, limits, required, owners } = self;
        let result = catch_unwind(AssertUnwindSafe(|| {
            checkpoint(control)?;
            let planner = Int4ExtractionBatchPlanner::for_model(weights.identity(), input.request.budget,
                limits.compiler, limits.source, None).map_err(HostedError::BatchSetup)?;
            let planner = Arc::new(Int4ObjectExtractionPlanner::new(planner).map_err(object_error)?
                .with_document_encoding(input.encoding.clone()).map_err(HostedError::BatchSetup)?);
            input.planner = Some(Arc::clone(&planner));
            checkpoint(control)?;
            let judge = Arc::new(Int4JudgePlanner::for_model(weights.identity(),
                input.options.specification.judge_budget, limits.judge_planning)
                .map_err(judge_error)?.with_document_encoding(input.encoding.clone()).map_err(judge_error)?);
            input.judge = Some(Arc::clone(&judge));
            for identity in [planner.base_execution_identity(), judge.execution_identity()] {
                constrained_int4::check_model(identity, weights.identity()).map_err(|_| HostedError::ModelIdentity)?;
                constrained_int4::check_profile(identity, weights.group_width()).map_err(|_| HostedError::ModelIdentity)?;
            }
            let controls = crate::tokenizer::pinned_controls::pinned().map_err(|_| HostedError::ModelIdentity)?;
            let vocabulary = Arc::new(ExtractionVocabulary::pinned(controls.template_controls())
                .map_err(|error| semantic_error(error.into()))?);
            input.vocabulary = Some(Arc::clone(&vocabulary));
            checkpoint(control)?;
            let extraction = planner.plan_document_with_control(&input.source, &input.request,
                limits.extraction, control).map_err(object_error)?;
            if extraction.max_kv_bytes() < required.native.kv_bytes {
                return Err(HostedError::Limits("semantic object document extraction full KV capacity"));
            }
            let prepared = PreparedInt4SemanticObjectDocument::new(extraction, judge,
                input.options.clone(), limits.review, control).map_err(document_error)?;
            if prepared.group_width() != weights.group_width() || prepared.source_bytes() != input.source.len()
                || prepared.extraction_budget() != input.request.budget
                || prepared.judge_budget() != input.options.specification.judge_budget
                || prepared.input_encoding() != &input.encoding
                || prepared.max_result_bytes() != input.options.limits.max_output_bytes {
                return Err(HostedError::ModelIdentity);
            }
            let expected = prepared.preflight_metadata();
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(prepared.execution_identities().len())
                .map_err(|_| HostedError::Limits("semantic object document identity storage"))?;
            for identity in prepared.execution_identities() {
                checkpoint(control)?;
                constrained_int4::check_model(identity, weights.identity()).map_err(|_| HostedError::ModelIdentity)?;
                constrained_int4::check_profile(identity, weights.group_width()).map_err(|_| HostedError::ModelIdentity)?;
                if identity.task_spec != "extract-v1"
                    || identity.tokenizer_digest != planner.base_execution_identity().tokenizer_digest
                    || identity.template_digest != planner.base_execution_identity().template_digest {
                    return Err(HostedError::ModelIdentity);
                }
                admitted.push(identity.clone());
            }
            checkpoint(control)?;
            let mut native = StrictInt4Engine::new(weights, limits.native.context_tokens,
                Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                    max_rope_bytes: required.native.rope_bytes,
                    max_scratch_payload_bytes: required.native.scratch_payload_bound })
                .map_err(native_error)?.with_kernel(kernel);
            if let Some(team) = projection {
                native = native.with_projection_executor(team).map_err(native_error)?;
            }
            let result = prepared.execute_with_control(&admitted, &mut native, &vocabulary, control)
                .map_err(document_error)?;
            expected.verify_completed_with_control(&result, control).map_err(document_error)?;
            if native.is_poisoned() || !native.kv_cache().all_slots_have_len(0) {
                return Err(document_error(Int4SemanticObjectDocumentError::Accounting));
            }
            checkpoint(control)?;
            drop(native);
            Ok(result)
        })).unwrap_or(Err(HostedError::Panicked));
        DeferredDocument { result, compute: None, input, owners }
    }
}

fn object_error(error: Int4ObjectExtractionMapError) -> HostedError {
    HostedError::Int4(Int4HostError::ObjectExtractionMap(error))
}
fn judge_error(error: crate::tasks::judge::grouped_int4::Int4JudgeError) -> HostedError {
    HostedError::Int4(Int4HostError::Judge(error))
}
fn faithfulness_error(error: crate::tasks::judge::grouped_int4::faithfulness_map::Int4FaithfulnessMapError)
    -> HostedError {
    HostedError::Int4(Int4HostError::FaithfulnessMap(error))
}
fn document_error(error: Int4SemanticObjectDocumentError) -> HostedError {
    HostedError::Int4(Int4HostError::SemanticObjectDocument(error))
}

#[cfg(test)] mod tests;
