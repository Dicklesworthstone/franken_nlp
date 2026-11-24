//! Complete-document exact-schema extraction on the process-owned INT8 host.
//! No global object merge, implicit model loading, or per-chunk budget renewal.
use super::*;
use crate::{
    batch::extract::{ExtractionBatchArgs, quantized::{Int8ExtractionBatchPlanner,
        long::{Int8ExtractionMapLimits, Int8ExtractionMapRun, PreparedInt8ExtractionMap}}},
    native_engine::{constrained_int8, decode::DecodeStepControl, simd::Int8Kernel,
        strict_int8::{StrictInt8Engine, prefill::Int8PrefillLimits}},
    tasks::{ir::TaskBudget, source_planning::quantized::{Int8SourceError, long::Int8SourceMapError}},
    validation::grounded_fields::{SourceFieldEvidence, VerifiedSourceSpan},
};

/// Explicit whole-document admission. The compiler's immutable base identity
/// is authoritative; callers cannot substitute a second identity in this config.
/// Preparation prices ALL retained schema/source plans and immutable compiler
/// assets. Reduction headroom is additional to the payload calculation below.
/// These are modeled ledger reservations, not an allocator or an RSS guarantee.
pub struct ExtractionMapConfig {
    pub request: ExtractionBatchArgs,
    pub limits: Int8ExtractionMapLimits,
    pub native: NativeLimits,
    pub preparation_reserve_bytes: u64,
    pub reduction_reserve_bytes: u64,
}
struct ExtractionMapInput {
    source: String,
    planner: Arc<Int8ExtractionBatchPlanner>,
    vocabulary: Arc<ExtractionVocabulary>,
    config: ExtractionMapConfig,
}

/// All source, output and native charges outlive the scoped callback. Worker
/// shutdown on cancellation or unwind happens before this aggregate drops.
struct ExtractionMapAdmission {
    input: Charged<ExtractionMapInput>,
    kv: Option<Pending>,
    scratch: Option<Pending>,
    output: Option<Pending>,
    reduction: Option<Pending>,
    _kv: Option<CommittedMemory>,
    _scratch: Option<CommittedMemory>,
    lease: EngineLease,
}

impl NlpEngine {
    /// Extract independent, exact-schema JSON values from every source chunk.
    /// One physical invocation owns all preparation, native work, cancellation
    /// and reduction. The output retains its memory charge after native drain.
    /// Source membership is optional and is never advertised as factual truth.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_int8_document(&self, model: &ResidentInt8, source: String,
        planner: Arc<Int8ExtractionBatchPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: ExtractionMapConfig, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8ExtractionMapRun>, HostedError> {
        self.extract_int8_document_scheduled(model, source, planner, vocabulary, config, None, cancellation)
    }

    /// Reuse one charged prompt workspace across every exact-schema chunk.
    /// This changes only scheduling; complete-document binding, independent
    /// JSON values and source-coordinate verification retain their authority.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_int8_document_layer_major(&self, model: &ResidentInt8, source: String,
        planner: Arc<Int8ExtractionBatchPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: ExtractionMapConfig, prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8ExtractionMapRun>, HostedError> {
        self.extract_int8_document_scheduled(model, source, planner, vocabulary, config, Some(prefill), cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn extract_int8_document_scheduled(&self, model: &ResidentInt8, source: String,
        planner: Arc<Int8ExtractionBatchPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: ExtractionMapConfig, prefill: Option<Int8PrefillLimits>, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8ExtractionMapRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        self.check_resident_domain(model)?;
        planner.map_document_encoding_workspace_bytes().map_err(HostedError::BatchSetup)?;
        let identity = planner.base_execution_identity();
        check_model_identity(model.artifact_identity(), identity)?;
        let required = requirements(config.native)?;
        let scratch_bytes = scratch_bytes(config.native, required, prefill)?;
        validate(&config, identity, source.len(), required.kv_bytes)?;
        let bytes = input_bytes(source.capacity(), &config, &planner)?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(ExtractionMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // Keep the WHOLE owner: disjoint-field capture must never drop the
            // charge while queued source/schema/compiler storage is still live.
            let input = input;
            let config = &input.value.config;
            let (prepared, admitted) = prepare(&input.value, model.artifact_identity(), control)?;
            let count = prepared.preflight_metadata().chunk_count();
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers,
                reduction_bytes(config, count)?)?;
            let tokens = (count as u64).checked_mul(u64::from(config.request.budget.max_output_tokens))
                .ok_or(HostedError::Limits("extraction map token arithmetic"))?;
            let output = output_claim(&lease, prepared.max_result_bytes(), tokens)?;
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(config.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let result = allocate(output, || {
                execute(prepared, &admitted, &mut engine.value, &input.value.vocabulary, prefill, control)
            })?;
            drop(engine);
            drop(admitted);
            drop(input);
            drop(reduction);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }

    /// Keep one fixed numerical team across all chunks, both model passes and
    /// constrained decode. Complete results are returned only after worker join.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_int8_document_with_threads(&self, model: &ResidentInt8, source: String,
        planner: Arc<Int8ExtractionBatchPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: ExtractionMapConfig, kernel: Int8Kernel, compute_threads: usize,
        cancellation: CancellationToken) -> Result<HostedInt8TeamResult<Int8ExtractionMapRun>, HostedError> {
        self.extract_int8_document_team(model, source, planner, vocabulary, config,
            kernel, compute_threads, None, cancellation)
    }

    /// Compose bounded layer-major prompts with the same persistent team used
    /// for decoding. Coordinator and worker prompt buffers are separate charges.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_int8_document_layer_major_with_threads(&self, model: &ResidentInt8, source: String,
        planner: Arc<Int8ExtractionBatchPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: ExtractionMapConfig, kernel: Int8Kernel, compute_threads: usize,
        prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8ExtractionMapRun>, HostedError> {
        self.extract_int8_document_team(model, source, planner, vocabulary, config,
            kernel, compute_threads, Some(prefill), cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn extract_int8_document_team(&self, model: &ResidentInt8, source: String,
        planner: Arc<Int8ExtractionBatchPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: ExtractionMapConfig, kernel: Int8Kernel, compute_threads: usize,
        prefill: Option<Int8PrefillLimits>, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8ExtractionMapRun>, HostedError> {
        dispatch::preflight_int8_team(self, config.native.run, compute_threads)?;
        self.check_resident_domain(model)?;
        planner.map_document_encoding_workspace_bytes().map_err(HostedError::BatchSetup)?;
        check_model_identity(model.artifact_identity(), planner.base_execution_identity())?;
        let required = requirements(config.native)?;
        validate(&config, planner.base_execution_identity(), source.len(), required.kv_bytes)?;
        let scratch_bytes = scratch_bytes(config.native, required, prefill)?;
        let max_batch_rows = prefill.map_or(1, |limits| limits.max_batch_rows);
        sum(&[scratch_bytes, dispatch::int8_prefill_team_workspace_bytes(compute_threads, max_batch_rows)?])?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let bytes = input_bytes(source.capacity(), &config, &planner)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(ExtractionMapInput { source, planner, vocabulary, config }))?;
        let admission = ExtractionMapAdmission { input, kv: Some(kv), scratch: Some(scratch),
            output: None, reduction: None, _kv: None, _scratch: None, lease };
        let artifact = model.artifact_identity().clone();
        dispatch::run_int8_prefill_team(self, run, cancellation, model.clone(), compute_threads,
            kernel, max_batch_rows, admission, move |model, admission, control, team| {
                let input = &admission.input.value;
                let config = &input.config;
                let (prepared, admitted) = prepare(input, &artifact, control)?;
                let count = prepared.preflight_metadata().chunk_count();
                admission.reduction = Some(Pending::reserve(&admission.lease, MemoryClass::JobBuffers,
                    reduction_bytes(config, count)?)?);
                let tokens = (count as u64).checked_mul(u64::from(config.request.budget.max_output_tokens))
                    .ok_or(HostedError::Limits("extraction map token arithmetic"))?;
                admission.output = Some(output_claim(&admission.lease, prepared.max_result_bytes(), tokens)?);
                let mut engine = model.engine(config.native.context_tokens, memory_budget(required))
                    .map_err(HostedError::Model)?.with_kernel(kernel)
                    .with_projection_executor(team).map_err(HostedError::Native)?;
                admission._kv = Some(admission.kv.take()
                    .ok_or(HostedError::Limits("extraction map KV admission consumed"))?.commit()?);
                admission._scratch = Some(admission.scratch.take()
                    .ok_or(HostedError::Limits("extraction map scratch admission consumed"))?.commit()?);
                let result = execute(prepared, &admitted, &mut engine, &input.vocabulary, prefill, control)?;
                drop(engine);
                let compute = team.stats().into();
                let output = admission.output.take()
                    .ok_or(HostedError::Limits("extraction map output admission consumed"))?;
                let memory = match output.commit() {
                    Ok(memory) => memory,
                    Err(error) => { drop(result); return Err(error); }
                };
                Ok(HostedInt8TeamResult::from_guarded(GuardedOutput::new(result, memory), compute))
            })
    }
}

fn scratch_bytes(limits: NativeLimits, native: Int8MemoryRequirement, prefill: Option<Int8PrefillLimits>)
    -> Result<u64, HostedError> {
    let extra = prefill.map(Int8PrefillLimits::validate).transpose().map_err(HostedError::Native)?.unwrap_or(0);
    sum(&[native.rope_bytes, native.scratch_payload_bound, limits.allocator_reserve_bytes, extra])
}

fn prepare<'s>(input: &'s ExtractionMapInput, artifact: &ArtifactIdentity, control: &mut RunControl)
    -> Result<(PreparedInt8ExtractionMap<'s>, Vec<ExecutionIdentity>), HostedError> {
    let prepared = input.planner.plan_document_with_control(&input.source,
        &input.config.request, input.config.limits, control).map_err(HostedError::SourceMap)?;
    let mut admitted = Vec::new();
    admitted.try_reserve_exact(prepared.preflight_metadata().chunk_count())
        .map_err(|_| HostedError::SourceMap(Int8SourceMapError::Allocation))?;
    for identity in prepared.execution_identities() {
        poll(control)?;
        check_model_identity(artifact, identity)?;
        check_identity(identity)?;
        let base = input.planner.base_execution_identity();
        if identity.tokenizer_digest != base.tokenizer_digest || identity.template_digest != base.template_digest {
            return Err(HostedError::ModelIdentity);
        }
        admitted.push(identity.clone());
    }
    poll(control)?;
    Ok((prepared, admitted))
}

fn execute(prepared: PreparedInt8ExtractionMap<'_>, admitted: &[ExecutionIdentity],
    engine: &mut StrictInt8Engine<'_>, vocabulary: &ExtractionVocabulary,
    prefill: Option<Int8PrefillLimits>, control: &mut RunControl) -> Result<Int8ExtractionMapRun, HostedError> {
    let expected = prepared.preflight_metadata();
    let result = match prefill {
        None => prepared.execute_with_control(admitted, engine, vocabulary, control),
        Some(prefill) => prepared.execute_layer_major_with_control(admitted, engine, vocabulary, prefill, control),
    }.map_err(HostedError::SourceMap)?;
    expected.verify_completed_with_control(&result, control).map_err(HostedError::SourceMap)?;
    poll(control)?;
    Ok(result)
}
fn check_identity(identity: &ExecutionIdentity) -> Result<(), HostedError> {
    identity.validate().map_err(|_| HostedError::ModelIdentity)?;
    constrained_int8::check_profile(identity).map_err(|_| HostedError::ModelIdentity)?;
    if identity.task_spec != "extract-v1" { return Err(HostedError::ModelIdentity); }
    Ok(())
}
fn validate(config: &ExtractionMapConfig, identity: &ExecutionIdentity, source_bytes: usize,
    kv_bytes: u64) -> Result<(), HostedError> {
    check_identity(identity)?;
    config.native.run.validate()?;
    config.request.budget.validate().map_err(|_| HostedError::Limits("extraction map task budget"))?;
    let m = config.limits.mapping;
    let v = config.limits.verification;
    let w = m.max_model_work;
    if source_bytes == 0 || source_bytes > m.chunks.max_input_bytes || config.request.schema.is_empty()
        || config.preparation_reserve_bytes == 0 || config.reduction_reserve_bytes == 0
        || m.chunks.context_tokens != config.native.context_tokens
        || kv_bytes > config.request.budget.max_kv_bytes
        || !(1..=256).contains(&m.chunks.max_chunks)
        || m.mask_visits_per_chunk == 0 || m.max_mask_visits < m.mask_visits_per_chunk
        || m.mask_limits.max_trie_node_visits == 0 || m.mask_limits.checkpoint_interval_nodes == 0
        || w.forward_positions == 0 || w.projected_logits == 0 || w.attention_pairs == 0
        || w.projections.dot_products == 0 || w.projections.multiply_accumulates == 0
        || m.reduction.max_result_bytes == 0 || m.reduction.max_live_value_bytes == 0
        || !(1..=1_000_000).contains(&v.max_fields) || !(1..=1_000_000).contains(&v.max_matches)
        || v.max_scan_steps == 0 {
        return Err(HostedError::Limits("extraction map source, capacity or whole-document limits"));
    }
    m.chunks.effective_token_limit().map_err(|_| HostedError::Limits("extraction map chunk geometry"))?;
    Ok(())
}
fn input_bytes(source_capacity: usize, config: &ExtractionMapConfig, planner: &Int8ExtractionBatchPlanner)
    -> Result<u64, HostedError> {
    let encoding = planner.map_document_encoding_workspace_bytes().map_err(HostedError::BatchSetup)?;
    sum(&[source_capacity as u64, config.request.schema.capacity() as u64,
        config.preparation_reserve_bytes, encoding])
}
fn reduction_bytes(config: &ExtractionMapConfig, chunks: usize) -> Result<u64, HostedError> {
    document_workspace_bytes(config.request.budget, config.limits, chunks, config.reduction_reserve_bytes)
}
/// Shared by single-document and corpus hosts. A stream reserves its maximum
/// possible document frontier ONCE, not fresh per-row independent allowances.
pub(super) fn document_workspace_bytes(budget: TaskBudget, limits: Int8ExtractionMapLimits,
    chunks: usize, reserve: u64) -> Result<u64, HostedError> {
    if chunks == 0 || chunks > limits.mapping.chunks.max_chunks || chunks > 256 || reserve == 0 {
        return Err(HostedError::Limits("extraction map partition cardinality or reserve"));
    }
    let native = budget.max_output_bytes.checked_mul(chunks as u64)
        .ok_or(HostedError::Limits("extraction map native result arithmetic"))?;
    let staged = sum(&[native, limits.mapping.reduction.max_live_value_bytes as u64])?
        .checked_mul(4).ok_or(HostedError::Limits("extraction map staging arithmetic"))?;
    let tokens = u64::from(budget.max_output_tokens).checked_mul(chunks as u64)
        .and_then(|n| n.checked_mul(8)).ok_or(HostedError::Limits("extraction map token arithmetic"))?;
    // Coordinate vectors contain FOUR usize coordinates per match, not the
    // small Anchored/Ambiguous enum tag. Price both the temporary scan vector
    // and retained lifted spans before a serialized-size check can reject them.
    let v = limits.verification;
    let fields = (v.max_fields as u64).checked_mul(std::mem::size_of::<SourceFieldEvidence>() as u64)
        .ok_or(HostedError::Limits("extraction map field arithmetic"))?;
    let spans = (v.max_matches as u64).checked_mul(std::mem::size_of::<VerifiedSourceSpan>() as u64)
        .and_then(|n| n.checked_mul(2)).ok_or(HostedError::Limits("extraction map occurrence arithmetic"))?;
    sum(&[staged, tokens, fields, spans, reserve])
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), HostedError> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(HostedError::SourceMap(Int8SourceMapError::Source(Int8SourceError::Cancelled(cause)))),
        None => Ok(()),
    }
}

#[cfg(test)] mod tests;
