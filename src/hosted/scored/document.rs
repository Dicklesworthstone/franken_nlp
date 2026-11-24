//! Complete-document INT8 classification on one charged native invocation.
//! Every chunk and independent label head is admitted before model execution.
use super::*;
use crate::{
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{decode::DecodeStepControl, simd::Int8Kernel,
        strict_int8::{StrictInt8Engine, STRICT_INT8_EXECUTION}},
    tasks::{classify::{ClassificationLabel, ClassificationLimits, ClassificationMode,
        ClassificationPlanner, ClassificationPlanningError, ClassificationRequest,
        quantized::long::{Int8ClassificationMapError, Int8ClassificationMapLimits,
            Int8ClassificationMapRun, PreparedInt8ClassificationMap}},
        ir::PlanContext},
    tokenizer::{pinned_controls, untrusted::UntrustedEncoding},
};

const MIN_PREPARATION_BYTES: u64 = 128 * 1024 * 1024;
const BASE_IDENTITY_BYTES: u64 = 16_384;

/// Explicit whole-document limits. The planning input ceiling covers the
/// original source; native context is reused sequentially by all chunk heads.
#[derive(Clone, Copy, Debug)]
pub struct Int8ClassificationMapHostConfig {
    pub native: NativeLimits,
    pub planning: ClassificationLimits,
    pub mapping: Int8ClassificationMapLimits,
    pub preparation_reserve_bytes: u64,
}

/// Pure simultaneous ownership quote, not an observed or enforced RSS bound.
/// Loaded model memory and the host's cleanup reserve are separate charges.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8ClassificationMapMemoryRequirement {
    pub native: Int8MemoryRequirement,
    /// RoPE, coordinator scratch and allocator reserve; excludes KV and
    /// persistent worker storage. Explicit prefill adds its derived payload.
    pub workspace_bytes: u64,
    /// Source, labels, compiler, retained plans and complete result frontier.
    pub input_bytes: u64,
    /// Complete output and serialization remain charged after native drain.
    pub output_reservation_bytes: u64,
    pub max_output_bytes: u64,
}

impl Int8ClassificationMapHostConfig {
    /// Quote the maximum declared label/head geometry before private input or
    /// model IO. A single KV allocation serves every sequential native head.
    pub fn requirements(self, task: TaskBudget)
        -> Result<Int8ClassificationMapMemoryRequirement, HostedError> {
        self.native.run.validate()?;
        self.mapping.validate(task).map_err(map_error)?;
        self.planning.validate().map_err(planning_error)?;
        let scoring = self.planning.scoring;
        if self.preparation_reserve_bytes < MIN_PREPARATION_BYTES
            || self.native.allocator_reserve_bytes == 0
            || self.mapping.chunks.context_tokens > self.native.context_tokens
            || self.mapping.chunks.max_input_bytes > self.planning.max_input_bytes
            || self.native.context_tokens > self.planning.max_context_tokens
            || scoring.max_candidates == 0 || scoring.max_total_tokens == 0
            || scoring.max_nodes == 0 || scoring.max_depth == 0
            || scoring.max_candidate_id_bytes == 0 || scoring.max_projected_logits == 0 {
            return Err(map_error(Int8ClassificationMapError::Limits));
        }
        let native = requirements(self.native)?;
        if native.kv_bytes > task.max_kv_bytes {
            return Err(map_error(Int8ClassificationMapError::Admission));
        }
        let chunks = self.mapping.chunks.max_chunks as u64;
        let labels = self.planning.max_labels as u64;
        // Multi-label repeats the source in independent binary heads. The
        // largest legal plan can exist before the aggregate work check rejects
        // it, alongside every previously accepted chunk plan.
        let per_chunk_prompt = mul(u64::from(task.max_input_tokens), labels)?
            .min(self.planning.max_total_prompt_tokens as u64);
        let declared_tokens = mul(per_chunk_prompt, chunks)?;
        let plan_tokens = declared_tokens.min(sum(&[
            self.mapping.max_model_work.forward_positions.min(declared_tokens),
            per_chunk_prompt,
        ])?);
        let heads = mul(labels, chunks)?;
        let nodes = (scoring.max_nodes as u64).max(mul(labels, 8)?);
        let node_bytes = sum(&[192, mul(scoring.max_depth.min(task.max_output_tokens as usize) as u64, 8)?])?;
        let plans = sum(&[
            mul(plan_tokens, 64)?, mul(heads, 16_384)?,
            mul(mul(nodes, node_bytes)?, chunks)?,
            mul(mul(self.planning.max_total_label_bytes as u64, 4)?, chunks)?,
        ])?;
        let per_chunk_result = task.max_output_bytes;
        let retained = mul(sum(&[per_chunk_result, 4096])?, chunks)?
            .min(self.mapping.max_retained_result_bytes);
        // The next complete chunk and its serialization exist before the
        // retained-result ceiling can refuse them.
        let results = mul(sum(&[retained, per_chunk_result, 4096])?, 8)?;
        let input_bytes = sum(&[
            self.preparation_reserve_bytes, BASE_IDENTITY_BYTES,
            mul(self.mapping.chunks.max_input_bytes as u64, 4)?,
            mul(self.planning.max_total_label_bytes as u64, 8)?, plans, results,
        ])?;
        let workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
            self.native.allocator_reserve_bytes])?;
        let max_output_bytes = self.mapping.max_result_bytes;
        let output_reservation_bytes = sum(&[mul(max_output_bytes, 8)?, 4096])?;
        sum(&[native.kv_bytes, workspace_bytes, input_bytes, output_reservation_bytes])?;
        Ok(Int8ClassificationMapMemoryRequirement { native, workspace_bytes, input_bytes,
            output_reservation_bytes, max_output_bytes })
    }

    /// Price one transient compact encoder workspace for the entire document.
    /// Failed-size probes, labels and chunk plans share its cumulative work
    /// allowance; neither memory nor work renews at a chunk/head boundary.
    pub fn requirements_with_encoding(self, task: TaskBudget, encoding: &UntrustedEncoding)
        -> Result<Int8ClassificationMapMemoryRequirement, HostedError> {
        self.requirements(task)?.with_input_encoding(encoding)
    }
}

impl Int8ClassificationMapMemoryRequirement {
    pub fn with_input_encoding(mut self, encoding: &UntrustedEncoding) -> Result<Self, HostedError> {
        let workspace = match encoding {
            UntrustedEncoding::ByteFallback => 0,
            UntrustedEncoding::SafeBpeV1(limits) => limits.requirements()
                .map_err(|_| map_error(Int8ClassificationMapError::Limits))?.workspace_bytes,
        };
        self.input_bytes = sum(&[self.input_bytes, workspace])?;
        sum(&[self.native.kv_bytes, self.workspace_bytes, self.input_bytes, self.output_reservation_bytes])?;
        Ok(self)
    }

    /// Add one reusable coordinator prompt workspace. Persistent children
    /// have a separate geometry-derived quote and reservation; neither this
    /// payload nor the resident KV allocation is multiplied by chunk count.
    pub fn with_prefill(mut self, prefill: Int8PrefillLimits) -> Result<Self, HostedError> {
        self.workspace_bytes = scoring_scratch(self.workspace_bytes, Some(prefill))?;
        sum(&[self.native.kv_bytes, self.workspace_bytes, self.input_bytes, self.output_reservation_bytes])?;
        Ok(self)
    }
}

struct MapInput {
    request: ClassificationRequest,
    identity: ExecutionIdentity,
    config: Int8ClassificationMapHostConfig,
    encoding: UntrustedEncoding,
}

/// Native claims are a separate field so preparation can borrow the exact
/// admitted request while allocation commits its own reservations. The whole
/// owner remains outside the scoped callback until every worker has joined.
struct MapNativeClaims {
    kv: Option<Pending>, scratch: Option<Pending>,
    _kv: Option<CommittedMemory>, _scratch: Option<CommittedMemory>,
}
impl MapNativeClaims {
    fn commit(&mut self) -> Result<(), HostedError> {
        self._kv = Some(self.kv.take().ok_or(HostedError::Limits("INT8 classification map KV consumed"))?.commit()?);
        self._scratch = Some(self.scratch.take().ok_or(HostedError::Limits("INT8 classification map scratch consumed"))?.commit()?);
        Ok(())
    }
}
struct MapTeamAdmission {
    input: Charged<MapInput>,
    native: MapNativeClaims,
    output: Option<Pending>,
    _lease: EngineLease,
}
impl MapTeamAdmission {
    fn finish(&mut self, result: Int8ClassificationMapRun, compute: Int8ComputeReceipt)
        -> Result<HostedInt8TeamResult<Int8ClassificationMapRun>, HostedError> {
        let pending = self.output.take().ok_or(HostedError::Limits("INT8 classification map output consumed"))?;
        match pending.commit() {
            Ok(memory) => Ok(HostedInt8TeamResult::from_guarded(GuardedOutput::new(result, memory), compute)),
            Err(error) => { drop(result); Err(error) }
        }
    }
}

impl NlpEngine {
    /// Classify the entire original document, preserving every chunk result
    /// and the separately named scalar-weighted aggregate. No prefix-only
    /// success, independent per-chunk budget renewal or implicit model load.
    pub fn map_int8_classify(&self, model: &ResidentInt8, request: ClassificationRequest,
        base_identity: ExecutionIdentity, config: Int8ClassificationMapHostConfig,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8ClassificationMapRun>, HostedError> {
        self.map_int8_classify_with_encoding(model, request, base_identity, config,
            UntrustedEncoding::ByteFallback, cancellation)
    }

    /// Execute an explicit byte-fallback or compact-input document policy on
    /// the same resident model and one process-owned blocking coordinator.
    /// The caller's base identity must already name the exact classifier
    /// template/tokenizer policy; this method never repairs mismatched claims.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int8_classify_with_encoding(&self, model: &ResidentInt8, request: ClassificationRequest,
        base_identity: ExecutionIdentity, config: Int8ClassificationMapHostConfig,
        encoding: UntrustedEncoding, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8ClassificationMapRun>, HostedError> {
        self.map_int8_classify_scheduled(model, request, base_identity, config, encoding, None, cancellation)
    }

    /// Reuse bounded layer-major prompt storage across every independent
    /// chunk/head, with the original document budget and completion seal.
    /// This preserves the sequential map API's one-context memory contract.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int8_classify_layer_major_with_encoding(&self, model: &ResidentInt8,
        request: ClassificationRequest, base_identity: ExecutionIdentity,
        config: Int8ClassificationMapHostConfig, encoding: UntrustedEncoding,
        prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8ClassificationMapRun>, HostedError> {
        self.map_int8_classify_scheduled(model, request, base_identity, config, encoding, Some(prefill), cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn map_int8_classify_scheduled(&self, model: &ResidentInt8, request: ClassificationRequest,
        base_identity: ExecutionIdentity, config: Int8ClassificationMapHostConfig,
        encoding: UntrustedEncoding, prefill: Option<Int8PrefillLimits>, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8ClassificationMapRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        self.check_resident_domain(model)?;
        validate_base_identity(&base_identity)?;
        check_model_identity(model.artifact_identity(), &base_identity)?;
        let required = scheduled_requirements(config, request.budget, &encoding, prefill, None)?;
        validate_request(&request, config)?;
        let bytes = owned_input_bytes(&request, &base_identity, config, required)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(MapInput { request, identity: base_identity, config, encoding }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_reservation_bytes)?;
        let model = model.clone();
        dispatch::run(self, config.native.run, cancellation, move |control| {
            // Preserve the whole aggregate: its charge cannot drop while a
            // queued closure still owns source, identity or preparation state.
            let input = input;
            let (prepared, admitted) = prepare_map(&input.value, model.artifact_identity(), control)?;
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(config.native.context_tokens, memory_budget(required.native)).map_err(HostedError::Model))?;
            let result = allocate(output, || execute_prepared(&prepared, &admitted,
                &mut engine.value, required.native.kv_bytes, prefill, control))?;
            drop(engine); drop(admitted); drop(prepared); drop(input); drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }

    /// One persistent numerical team spans every chunk and all of its label
    /// heads. Chunk order, candidate language, full-vocabulary denominators
    /// and scalar-weighted aggregation retain their existing exact contracts.
    /// Width includes the coordinator and must fit the installed process host.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int8_classify_with_threads(&self, model: &ResidentInt8, request: ClassificationRequest,
        base_identity: ExecutionIdentity, config: Int8ClassificationMapHostConfig,
        encoding: UntrustedEncoding, kernel: Int8Kernel, compute_threads: usize,
        cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8ClassificationMapRun>, HostedError> {
        self.map_int8_classify_team(model, request, base_identity, config, encoding,
            kernel, compute_threads, None, cancellation)
    }

    /// Compose bounded prompt morsels with that same fixed team for the
    /// entire document. Coordinator scratch and fixed child batch storage
    /// are separately admitted before work and retained through physical join.
    /// No semantic identity, work allowance or result seal includes the width.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int8_classify_layer_major_with_threads(&self, model: &ResidentInt8,
        request: ClassificationRequest, base_identity: ExecutionIdentity,
        config: Int8ClassificationMapHostConfig, encoding: UntrustedEncoding,
        kernel: Int8Kernel, compute_threads: usize, prefill: Int8PrefillLimits,
        cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8ClassificationMapRun>, HostedError> {
        self.map_int8_classify_team(model, request, base_identity, config, encoding,
            kernel, compute_threads, Some(prefill), cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn map_int8_classify_team(&self, model: &ResidentInt8, request: ClassificationRequest,
        base_identity: ExecutionIdentity, config: Int8ClassificationMapHostConfig,
        encoding: UntrustedEncoding, kernel: Int8Kernel, compute_threads: usize,
        prefill: Option<Int8PrefillLimits>, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8ClassificationMapRun>, HostedError> {
        dispatch::preflight_int8_team(self, config.native.run, compute_threads)?;
        self.check_resident_domain(model)?;
        validate_base_identity(&base_identity)?;
        check_model_identity(model.artifact_identity(), &base_identity)?;
        let required = scheduled_requirements(config, request.budget, &encoding, prefill, Some(compute_threads))?;
        validate_request(&request, config)?;
        let bytes = owned_input_bytes(&request, &base_identity, config, required)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(MapInput { request, identity: base_identity, config, encoding }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_reservation_bytes)?;
        let admission = MapTeamAdmission { input, native: MapNativeClaims {
            kv: Some(kv), scratch: Some(scratch), _kv: None, _scratch: None },
            output: Some(output), _lease: lease };
        let max_batch_rows = prefill.map_or(1, |limits| limits.max_batch_rows);
        dispatch::run_int8_prefill_team(self, config.native.run, cancellation, model.clone(),
            compute_threads, kernel, max_batch_rows, admission, move |model, admission, control, team| {
                let (prepared, admitted) = prepare_map(&admission.input.value, model.artifact_identity(), control)?;
                let mut engine = model.engine(config.native.context_tokens, memory_budget(required.native))
                    .map_err(HostedError::Model)?.with_kernel(kernel)
                    .with_projection_executor(team).map_err(HostedError::Native)?;
                admission.native.commit()?;
                let result = execute_prepared(&prepared, &admitted, &mut engine,
                    required.native.kv_bytes, prefill, control)?;
                drop(engine); drop(admitted); drop(prepared);
                let compute = team.stats().into();
                admission.finish(result, compute)
            })
    }
}

fn scheduled_requirements(config: Int8ClassificationMapHostConfig, task: TaskBudget,
    encoding: &UntrustedEncoding, prefill: Option<Int8PrefillLimits>, width: Option<usize>)
    -> Result<Int8ClassificationMapMemoryRequirement, HostedError> {
    let mut required = config.requirements_with_encoding(task, encoding)?;
    if let Some(prefill) = prefill { required = required.with_prefill(prefill)?; }
    if let Some(width) = width {
        let workers = dispatch::int8_prefill_team_workspace_bytes(width,
            prefill.map_or(1, |limits| limits.max_batch_rows))?;
        sum(&[required.native.kv_bytes, required.workspace_bytes, required.input_bytes,
            required.output_reservation_bytes, workers])?;
    }
    Ok(required)
}

fn prepare_map<'a>(input: &'a MapInput, model: &ArtifactIdentity, control: &mut impl DecodeStepControl)
    -> Result<(PreparedInt8ClassificationMap<'a>, Vec<ExecutionIdentity>), HostedError> {
    checkpoint(control)?;
    let registry = pinned_controls::pinned()
        .map_err(|_| planning_error(ClassificationPlanningError::ControlRegistry))?;
    let controls = registry.template_controls();
    let eos = controls.entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END)
        .map(|entry| entry.id)
        .ok_or_else(|| planning_error(ClassificationPlanningError::ControlRegistry))?;
    let planner = ClassificationPlanner::pinned(controls, eos)
        .and_then(|planner| planner.with_document_encoding(input.encoding.clone()))
        .map_err(planning_error)?;
    check_planner_binding(&planner, &input.identity, &input.encoding)?;
    let context = PlanContext::new(&input.identity, input.request.budget)
        .map_err(|_| HostedError::ModelIdentity)?;
    let prepared = planner.plan_int8_document_with_control(&input.request, &context,
        input.config.planning, input.config.mapping, control).map_err(map_error)?;
    let admitted = admit_prepared(&prepared, input, model, control)?;
    Ok((prepared, admitted))
}

fn execute_prepared(prepared: &PreparedInt8ClassificationMap<'_>, admitted: &[ExecutionIdentity],
    engine: &mut StrictInt8Engine<'_>, kv_bytes: u64, prefill: Option<Int8PrefillLimits>,
    control: &mut impl DecodeStepControl) -> Result<Int8ClassificationMapRun, HostedError> {
    let expected = prepared.preflight_metadata();
    let budget = Int8ScoringBudget { native: Int8RunBudget::exact(prepared.planned_work()), max_kv_bytes: kv_bytes };
    let result = match prefill {
        Some(prefill) => prepared.execute_layer_major_with_control(admitted, engine, budget, prefill, control),
        None => prepared.execute_with_control(admitted, engine, budget, control),
    }.map_err(map_error)?;
    expected.verify_completed_with_control(&result, control).map_err(map_error)?;
    if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
        return Err(map_error(Int8ClassificationMapError::Admission));
    }
    checkpoint(control)?;
    Ok(result)
}

fn validate_base_identity(identity: &ExecutionIdentity) -> Result<(), HostedError> {
    identity.validate().map_err(|_| HostedError::ModelIdentity)?;
    if identity.task_spec != "classify-v1"
        || identity.numerics_profile != (NumericsProfile::StrictQuantized { version: 1 })
        || identity.backend_semantic_version != STRICT_INT8_EXECUTION
        || identity.kv_dtype != "bf16" || identity.thinking_mode != ThinkingMode::Disabled
        || identity.tool_mode != ToolMode::None
        || identity_strings(identity).iter().any(|value| value.len() > 256) {
        return Err(HostedError::ModelIdentity);
    }
    Ok(())
}

fn check_planner_binding(planner: &ClassificationPlanner, identity: &ExecutionIdentity,
    encoding: &UntrustedEncoding) -> Result<(), HostedError> {
    validate_base_identity(identity)?;
    if planner.document_encoding() != encoding
        || identity.template_digest != *planner.template_digest()
        || identity.tokenizer_digest != planner.tokenizer_digest() {
        return Err(HostedError::ModelIdentity);
    }
    Ok(())
}

fn admit_prepared(prepared: &PreparedInt8ClassificationMap<'_>, input: &MapInput,
    model: &ArtifactIdentity, control: &mut impl DecodeStepControl) -> Result<Vec<ExecutionIdentity>, HostedError> {
    let expected_heads = if input.request.mode == ClassificationMode::Exclusive { 1 } else { input.request.labels.len() };
    if prepared.head_count() != expected_heads || prepared.chunk_count() == 0
        || prepared.chunk_count() > input.config.mapping.chunks.max_chunks
        || prepared.task_budget() != input.request.budget
        || prepared.required_context() > input.config.native.context_tokens
        || prepared.max_result_bytes() != input.config.mapping.max_result_bytes {
        return Err(map_error(Int8ClassificationMapError::Admission));
    }
    let admitted = prepared.identities().map_err(map_error)?;
    if admitted.len() != prepared.chunk_count() {
        return Err(map_error(Int8ClassificationMapError::Admission));
    }
    for identity in &admitted {
        checkpoint(control)?;
        validate_base_identity(identity)?;
        check_model_identity(model, identity)?;
        if identity.tokenizer_digest != input.identity.tokenizer_digest
            || identity.template_digest != input.identity.template_digest {
            return Err(HostedError::ModelIdentity);
        }
    }
    checkpoint(control)?;
    Ok(admitted)
}

fn validate_request(request: &ClassificationRequest, config: Int8ClassificationMapHostConfig) -> Result<(), HostedError> {
    let limits = config.planning;
    if request.document.is_empty() { return Err(map_error(Int8ClassificationMapError::EmptySource)); }
    if request.document.len() > config.mapping.chunks.max_input_bytes || request.document.len() > limits.max_input_bytes
        || request.labels.is_empty() || request.labels.len() > limits.max_labels
        || (request.mode == ClassificationMode::Exclusive && request.labels.len() < 2)
        || request.policy.minimum_candidate_weight_ppm > 1_000_000 || request.policy.minimum_margin_ppm > 1_000_000 {
        return Err(map_error(Int8ClassificationMapError::Limits));
    }
    let mut bytes = 0_usize;
    for (index, label) in request.labels.iter().enumerate() {
        bytes = bytes.checked_add(label.id.len()).and_then(|n| n.checked_add(label.description.len()))
            .ok_or_else(|| map_error(Int8ClassificationMapError::Limits))?;
        if label.id.len() > limits.max_label_id_bytes || label.description.len() > limits.max_label_description_bytes
            || bytes > limits.max_total_label_bytes || label.id.trim().is_empty() || label.id.chars().any(char::is_control)
            || request.labels[..index].iter().any(|previous| previous.id == label.id) {
            return Err(map_error(Int8ClassificationMapError::Limits));
        }
    }
    Ok(())
}

fn owned_input_bytes(request: &ClassificationRequest, identity: &ExecutionIdentity,
    config: Int8ClassificationMapHostConfig, required: Int8ClassificationMapMemoryRequirement) -> Result<u64, HostedError> {
    // Transferred caller capacities can exceed logical ceilings. Retain the
    // original owners and charge their excess rather than cloning/shrinking.
    let source = request.document.capacity().saturating_sub(config.mapping.chunks.max_input_bytes) as u64;
    let labels = mul(request.labels.capacity().saturating_sub(config.planning.max_labels) as u64,
        std::mem::size_of::<ClassificationLabel>() as u64)?;
    let text = request.labels.iter().try_fold(0_u64, |total, label|
        sum(&[total, label.id.capacity() as u64, label.description.capacity() as u64]))?;
    let identity_bytes = identity_strings(identity).iter().try_fold(std::mem::size_of::<ExecutionIdentity>() as u64,
        |total, value| sum(&[total, value.capacity() as u64]))?;
    sum(&[required.input_bytes, source, labels,
        text.saturating_sub(config.planning.max_total_label_bytes as u64),
        identity_bytes.saturating_sub(BASE_IDENTITY_BYTES)])
}

fn identity_strings(identity: &ExecutionIdentity) -> [&String; 8] {
    [&identity.source_revision, &identity.artifact_format, &identity.quant_recipe,
        &identity.task_spec, &identity.grammar_compiler_version, &identity.kv_dtype,
        &identity.sampler_version, &identity.backend_semantic_version]
}
fn mul(a: u64, b: u64) -> Result<u64, HostedError> {
    a.checked_mul(b).ok_or_else(|| map_error(Int8ClassificationMapError::Limits))
}
fn map_error(error: Int8ClassificationMapError) -> HostedError { HostedError::ClassificationMap(error) }
fn planning_error(error: ClassificationPlanningError) -> HostedError {
    map_error(Int8ClassificationMapError::Classification(Int8ClassificationError::Planning(error)))
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), HostedError> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(planning_error(ClassificationPlanningError::Cancelled(cause))),
        None => Ok(()),
    }
}

#[cfg(test)] mod tests;
