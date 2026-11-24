//! Resident INT8 extraction and complete correlated field review for NDJSON.
//! Fixed schemas and review policies share one compiler, engine and work ledger.
use super::*;
use crate::{
    batch::extract::semantic::int8::{Int8SemanticExtractionAdmission,
        Int8SemanticExtractionBatchAdmission, Int8SemanticExtractionBatchLimits,
        Int8SemanticExtractionBatchPlanner, NativeInt8SemanticExtractionBatch},
    native_engine::{constrained_int8, simd::Int8Kernel,
        strict_int8::{StrictInt8Engine, prefill::Int8PrefillLimits}},
    tasks::{ir::TaskBudget, judge::JudgeLimits,
        extract::semantic::int8::Int8SemanticOptions},
    tokenizer::untrusted::{SafeBpeWorkLimits, UntrustedEncoding},
};

/// Complete simultaneous payload, excluding resident weights, caller-owned
/// preparation and dispatch cleanup. Preparation is already in input_bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8SemanticCorpusRequirements {
    pub native: Int8MemoryRequirement,
    pub native_workspace_bytes: u64,
    pub preparation_bytes: u64,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub team_workspace_bytes: u64,
}
impl Int8SemanticCorpusRequirements {
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        sum(&[self.native.kv_bytes, self.native_workspace_bytes, self.input_bytes,
            self.output_bytes, self.team_workspace_bytes])
    }
}

/// Pure admission before schema, population or model IO. A run's native,
/// mask and tokenizer totals are separate from each complete document's caps;
/// none are renewed by a flush. All actual field plans coexist before review.
#[allow(clippy::too_many_arguments)]
pub fn int8_semantic_corpus_requirements(extraction: TaskBudget,
    options: &Int8SemanticOptions, judge_planning: JudgeLimits,
    encoding: &UntrustedEncoding, encoding_work_limits: Option<SafeBpeWorkLimits>,
    work: Int8SemanticExtractionBatchLimits, limits: CorpusLimits,
    prefill: Option<Int8PrefillLimits>, compute_threads: Option<usize>)
    -> Result<Int8SemanticCorpusRequirements, HostedError> {
    options.validate().map_err(HostedError::SemanticExtraction)?;
    work.validate_item(options.max_model_work).map_err(HostedError::BatchSetup)?;
    validate_judge_planning(judge_planning)?;
    let ledger = encoding_ledger_bytes(encoding, encoding_work_limits)?;
    if options.limits.max_output_bytes > limits.transport.max_output_line_bytes as u64 {
        return Err(HostedError::Limits("INT8 semantic corpus complete result envelope"));
    }
    let transport = limits.reservation_bytes()?;
    let required = semantic_host(limits.native, limits.preparation_reserve_bytes, work)
        .requirements_with_schedule(extraction, options.specification.judge_budget,
            options.limits, judge_planning, limits.transport.max_document_bytes, encoding, prefill)?;
    let staging = transport.checked_sub(limits.preparation_reserve_bytes)
        .ok_or(HostedError::Limits("INT8 semantic corpus preparation arithmetic"))?;
    let preparation_bytes = sum(&[required.preparation_bytes, ledger])?;
    let quote = Int8SemanticCorpusRequirements {
        native: required.native, native_workspace_bytes: required.native_workspace_bytes,
        preparation_bytes, input_bytes: sum(&[preparation_bytes, staging])?,
        output_bytes: required.output_bytes,
        team_workspace_bytes: team_bytes(prefill, compute_threads)?,
    };
    quote.total_bytes()?;
    Ok(quote)
}

pub(super) fn semantic_host(native: NativeLimits, preparation_reserve_bytes: u64,
    work: Int8SemanticExtractionBatchLimits) -> Int8SemanticHostLimits {
    Int8SemanticHostLimits { native, masks: work.masks.per_mask,
        max_mask_node_visits: work.masks.max_visits_per_item, preparation_reserve_bytes }
}
pub(super) fn encoding_ledger_bytes(encoding: &UntrustedEncoding,
    limits: Option<SafeBpeWorkLimits>) -> Result<u64, HostedError> {
    encoding.validate().map_err(|_| HostedError::Limits("INT8 semantic encoding policy"))?;
    match (encoding, limits) {
        (UntrustedEncoding::ByteFallback, None) => Ok(0),
        (UntrustedEncoding::SafeBpeV1(_), Some(limits)) => {
            limits.validate().map_err(|_| HostedError::Limits("INT8 semantic invocation encoding work"))?;
            Ok(1024)
        }
        _ => Err(HostedError::Limits("INT8 semantic encoding work policy mismatch")),
    }
}
pub(super) fn team_bytes(prefill: Option<Int8PrefillLimits>, threads: Option<usize>)
    -> Result<u64, HostedError> {
    threads.map(|width| dispatch::int8_prefill_team_workspace_bytes(width,
        prefill.map_or(1, |limits| limits.max_batch_rows))).transpose().map(|bytes| bytes.unwrap_or(0))
}
pub(super) fn validate_judge_planning(limits: JudgeLimits) -> Result<(), HostedError> {
    if limits.max_total_prompt_tokens == 0 || limits.max_total_candidate_tokens == 0
        || limits.max_total_projected_logits == 0 || limits.max_output_bytes == 0
        || limits.per_head.max_candidates == 0 || limits.per_head.max_total_tokens == 0
        || limits.per_head.max_nodes == 0 || limits.per_head.max_depth == 0
        || limits.per_head.max_candidate_id_bytes == 0 || limits.per_head.max_projected_logits == 0 {
        return Err(HostedError::Limits("INT8 semantic complete judge planning"));
    }
    Ok(())
}

/// Actual compiler/vocabulary owners accompany the output reservation through
/// serialization, flush and physical team join. Pending explicitly aborts only
/// after the final owner drops; no allocation completion is fabricated.
pub(super) struct SemanticAssets {
    pub(super) planner: Arc<Int8SemanticExtractionBatchPlanner>,
    pub(super) vocabulary: Arc<ExtractionVocabulary>,
    _output: Pending,
    _lease: EngineLease,
}
impl SemanticAssets {
    pub(super) fn admit(host: &NlpEngine, planner: Arc<Int8SemanticExtractionBatchPlanner>,
        vocabulary: Arc<ExtractionVocabulary>, output_bytes: u64) -> Result<Arc<Self>, HostedError> {
        let lease = host.resources().acquire_lease();
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, output_bytes)?;
        Ok(Arc::new(Self { planner, vocabulary, _output: output, _lease: lease }))
    }
}
struct SemanticStream {
    assets: Arc<SemanticAssets>,
    limits: CorpusLimits,
    required: Int8SemanticCorpusRequirements,
    prefill: Option<Int8PrefillLimits>,
}
impl NlpEngine {
    /// Extract and review every bounded record with one resident INT8 engine.
    /// The original extraction stays exact; review is an explicit same-model,
    /// uncalibrated diagnostic and never repairs or automatically accepts it.
    #[allow(clippy::too_many_arguments)]
    pub fn corpus_int8_semantic_extract<R, W>(&self, model: &ResidentInt8,
        planner: Arc<Int8SemanticExtractionBatchPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        limits: CorpusLimits, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.corpus_int8_semantic_extract_scheduled(model, planner, vocabulary, limits,
            None, None, reader, writer, cancellation)
    }

    /// One optional persistent team and reusable prompt workspace cover both
    /// stages of every record. The terminal frame is published after physical
    /// join. Any failed stage returns no successful combined document result.
    #[allow(clippy::too_many_arguments)]
    pub fn corpus_int8_semantic_extract_scheduled<R, W>(&self, model: &ResidentInt8,
        planner: Arc<Int8SemanticExtractionBatchPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        limits: CorpusLimits, prefill: Option<Int8PrefillLimits>, team: Option<(Int8Kernel, usize)>,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        match team {
            None => dispatch::preflight(self, limits.native.run)?,
            Some((_, width)) => dispatch::preflight_int8_team(self, limits.native.run, width)?,
        }
        self.check_resident_domain(model)?;
        for identity in [planner.base_execution_identity(), planner.judge_identity()] {
            check_model_identity(model.artifact_identity(), identity)?;
        }
        let required = planner_requirements(&planner, limits, prefill, team.map(|(_, width)| width))?;
        let assets = SemanticAssets::admit(self, planner, vocabulary, required.output_bytes)?;
        scored::run(self, model, SemanticStream { assets, limits, required, prefill }, limits,
            prefill, team, reader, writer, cancellation)
    }
}
fn planner_requirements(planner: &Int8SemanticExtractionBatchPlanner, limits: CorpusLimits,
    prefill: Option<Int8PrefillLimits>, threads: Option<usize>)
    -> Result<Int8SemanticCorpusRequirements, HostedError> {
    int8_semantic_corpus_requirements(planner.extraction_budget(), planner.options(),
        planner.judge_planning_limits(), planner.document_encoding(), planner.encoding_work_limits(),
        planner.limits(), limits, prefill, threads)
}
impl scored::CorpusTask for SemanticStream {
    fn identity(&self) -> &ExecutionIdentity { self.assets.planner.base_execution_identity() }
    fn retain_for_join(&self) -> Option<Box<dyn Send>> { Some(Box::new(Arc::clone(&self.assets))) }
    fn validate(&self, model: &ArtifactIdentity, limits: CorpusLimits,
        native: Int8MemoryRequirement) -> Result<u64, HostedError> {
        let planner = &self.assets.planner;
        check_model_identity(model, planner.judge_identity())?;
        let actual = planner_requirements(planner, limits, self.prefill, None)?;
        if actual.native != native || actual.native != self.required.native
            || actual.input_bytes != self.required.input_bytes
            || actual.output_bytes != self.required.output_bytes
            || actual.native_workspace_bytes != self.required.native_workspace_bytes {
            return Err(HostedError::Limits("INT8 semantic corpus admitted geometry"));
        }
        actual.preparation_bytes.checked_sub(limits.preparation_reserve_bytes)
            .ok_or(HostedError::Limits("INT8 semantic corpus extra preparation"))
    }
    #[allow(clippy::too_many_arguments)]
    fn execute<R: BufRead, W: Write>(self, engine: &mut StrictInt8Engine<'_>,
        admission: CorpusAdmission<'_>, prefill: Option<Int8PrefillLimits>,
        reader: &mut R, writer: &mut W, limits: BatchLimits, defer: bool,
        control: &mut RunControl) -> Result<scored::Completion, HostedError> {
        let assets = self.assets;
        let admission = SemanticAdmission { inner: admission, assets: Arc::clone(&assets),
            host: semantic_host(self.limits.native, self.limits.preparation_reserve_bytes, assets.planner.limits()),
            max_source_bytes: self.limits.transport.max_document_bytes,
            preparation_bytes: self.required.preparation_bytes, output_bytes: self.required.output_bytes,
            prefill };
        let mut processor = match prefill {
            None => NativeInt8SemanticExtractionBatch::new_shared(Arc::clone(&assets.planner),
                engine, &assets.vocabulary, admission),
            Some(prefill) => NativeInt8SemanticExtractionBatch::new_shared_layer_major(
                Arc::clone(&assets.planner), engine, &assets.vocabulary, admission, prefill),
        }.map_err(HostedError::BatchSetup)?;
        scored::execute_stream(reader, writer, &mut processor, limits, defer, control)
    }
}

pub(super) struct SemanticAdmission<'a> {
    pub(super) inner: CorpusAdmission<'a>,
    pub(super) assets: Arc<SemanticAssets>,
    pub(super) host: Int8SemanticHostLimits,
    pub(super) max_source_bytes: usize,
    pub(super) preparation_bytes: u64,
    pub(super) output_bytes: u64,
    pub(super) prefill: Option<Int8PrefillLimits>,
}
impl Int8SemanticExtractionBatchAdmission for SemanticAdmission<'_> {
    type Guard = Arc<SemanticAssets>;
    fn admit(&mut self, request: Int8SemanticExtractionAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        for identity in [request.identity, request.judge_identity] {
            check_model_identity(self.inner.model, identity)
                .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        }
        check_request(&self.assets.planner, self.host, self.prefill,
            self.inner.kv_bytes, self.inner.output_bytes, self.max_source_bytes,
            self.preparation_bytes, self.output_bytes, &request)?;
        Ok((request.identity.clone(), Arc::clone(&self.assets)))
    }
}

#[allow(clippy::too_many_arguments)]
fn check_request(planner: &Int8SemanticExtractionBatchPlanner, host: Int8SemanticHostLimits,
    prefill: Option<Int8PrefillLimits>, kv_bytes: u64, envelope_bytes: u64,
    max_source_bytes: usize, preparation_bytes: u64, output_bytes: u64,
    request: &Int8SemanticExtractionAdmission<'_>) -> Result<(), BatchItemFailure> {
    let fail = || BatchItemFailure::fatal(BatchCode::Admission);
    for identity in [request.identity, request.judge_identity] {
        constrained_int8::check_profile(identity).map_err(|_| fail())?;
    }
    let work = planner.limits();
    if request.identity.task_spec != "extract-v1" || request.judge_identity.task_spec != "judge-v1"
        || request.model_work != planner.options().max_model_work
        || !fits(request.model_work, work.max_model_work)
        || !nonzero(request.extraction_reserved_work)
        || !fits(request.extraction_reserved_work, request.model_work)
        || request.extraction_reserved_work.forward_positions > host.native.context_tokens as u64
        || request.mask_node_visits != work.masks.max_visits_per_item
        || request.mask_limits != work.masks.per_mask
        || request.kv_reservation_bytes != kv_bytes
        || request.extraction_budget != planner.extraction_budget()
        || request.judge_budget != planner.judge_budget()
        || request.semantic_limits != planner.semantic_limits()
        || !same_planning(request.judge_planning_limits, planner.judge_planning_limits())
        || request.source_bytes == 0 || request.source_bytes > max_source_bytes
        || request.input_encoding != planner.document_encoding()
        || request.encoding_work_limits != planner.encoding_work_limits() {
        return Err(fail());
    }
    if request.max_result_bytes != planner.max_result_bytes()
        || request.max_result_bytes > envelope_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::OutputLineLimit));
    }
    let actual = host.requirements_with_schedule(request.extraction_budget, request.judge_budget,
        request.semantic_limits, request.judge_planning_limits, request.source_bytes,
        request.input_encoding, prefill).map_err(|_| fail())?;
    let ledger = encoding_ledger_bytes(request.input_encoding, request.encoding_work_limits).map_err(|_| fail())?;
    if actual.native.kv_bytes != kv_bytes
        || sum(&[actual.preparation_bytes, ledger]).map_err(|_| fail())? > preparation_bytes
        || actual.output_bytes > output_bytes {
        return Err(fail());
    }
    Ok(())
}
fn fits(work: Int8Work, cap: Int8Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn nonzero(work: Int8Work) -> bool {
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
