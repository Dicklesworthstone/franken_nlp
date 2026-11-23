//! Complete-source INT8 faithfulness under one process-owned invocation.
//! All lossless chunks and every local evidence head are admitted together.
use super::*;
use crate::{
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{decode::DecodeStepControl,
        strict_int8::{Int8Work, STRICT_INT8_EXECUTION}},
    tasks::{ir::PlanContext,
        judge::{FaithfulnessPolicy, JudgeError, JudgeLimits, JudgePlanner, JudgeRequest,
            quantized::faithfulness_map::{Int8FaithfulnessMapError, Int8FaithfulnessMapLimits,
                Int8FaithfulnessMapRun, PreparedInt8FaithfulnessMap, MAX_FAITHFULNESS_MAP_HEADS}}},
    tokenizer::{pinned_controls, untrusted::UntrustedEncoding},
};

const MIN_PREPARATION_BYTES: u64 = 128 * 1024 * 1024;
const BASE_IDENTITY_BYTES: u64 = 16_384;

/// The source and unchanged claim share one input ceiling. All chunk plans
/// and evidence results remain owned while one serial native cache is reused.
#[derive(Clone, Copy, Debug)]
pub struct Int8FaithfulnessMapHostConfig {
    pub native: NativeLimits,
    pub planning: JudgeLimits,
    pub mapping: Int8FaithfulnessMapLimits,
    pub preparation_reserve_bytes: u64,
}

/// Pure simultaneous ownership quote, not an observed or enforced RSS bound.
/// Loaded model memory and the host's cleanup reserve are separate charges.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8FaithfulnessMapMemoryRequirement {
    pub native: Int8MemoryRequirement,
    /// RoPE, native scratch and allocator reserve; excludes KV.
    pub workspace_bytes: u64,
    /// Source, claim, compiler, every retained head and complete result frontier.
    pub input_bytes: u64,
    /// Complete output and serialization remain charged after native drain.
    pub output_reservation_bytes: u64,
    pub max_output_bytes: u64,
}

impl Int8FaithfulnessMapHostConfig {
    /// Price the maximum 32 local heads for every declared source chunk before
    /// private input or model IO. The fixed E/C/U language cannot grow with
    /// generic scorer ceilings. Sequential heads share one physical KV row.
    pub fn requirements(self, task: TaskBudget)
        -> Result<Int8FaithfulnessMapMemoryRequirement, HostedError> {
        self.native.run.validate()?;
        self.mapping.validate(task).map_err(map_error)?;
        let scoring = self.planning.per_head;
        if self.preparation_reserve_bytes < MIN_PREPARATION_BYTES
            || self.native.allocator_reserve_bytes == 0
            || self.mapping.chunks.context_tokens > self.native.context_tokens
            || self.planning.max_total_prompt_tokens == 0
            || self.planning.max_total_candidate_tokens == 0
            || self.planning.max_total_projected_logits == 0 || self.planning.max_output_bytes == 0
            || scoring.max_candidates == 0 || scoring.max_total_tokens == 0
            || scoring.max_nodes == 0 || scoring.max_depth == 0
            || scoring.max_candidate_id_bytes == 0 || scoring.max_projected_logits == 0 {
            return Err(map_error(Int8FaithfulnessMapError::Limits));
        }
        let native = requirements(self.native)?;
        if native.kv_bytes > task.max_kv_bytes {
            return Err(map_error(Int8FaithfulnessMapError::Admission));
        }
        let chunks = self.mapping.chunks.max_chunks as u64;
        let max_heads = MAX_FAITHFULNESS_MAP_HEADS as u64;
        let per_chunk_prompt = mul(u64::from(task.max_input_tokens), max_heads)?
            .min(self.planning.max_total_prompt_tokens as u64);
        let declared_tokens = mul(per_chunk_prompt, chunks)?;
        // The next full chunk can exist before its aggregate work check
        // refuses it, alongside all earlier accepted source/claim plans.
        let plan_tokens = declared_tokens.min(sum(&[
            self.mapping.max_model_work.forward_positions.min(declared_tokens), per_chunk_prompt,
        ])?);
        let heads = mul(max_heads, chunks)?;
        let plans = sum(&[
            mul(plan_tokens, 64)?, mul(heads, 16_384)?,
            mul(mul(8, 256)?, heads)?,
            mul(self.mapping.chunks.max_input_bytes as u64, 4)?,
        ])?;
        let per_chunk_result = task.max_output_bytes.min(self.planning.max_output_bytes);
        let retained = mul(sum(&[per_chunk_result, 4096])?, chunks)?
            .min(self.mapping.max_retained_result_bytes);
        // The next complete result, serialization and lifted aggregate
        // witnesses coexist with the retained frontier; no evidence is pruned.
        let results = mul(sum(&[retained, per_chunk_result, 4096])?, 8)?;
        let input_bytes = sum(&[
            self.preparation_reserve_bytes, BASE_IDENTITY_BYTES,
            mul(self.mapping.chunks.max_input_bytes as u64, 4)?, plans, results,
        ])?;
        let workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
            self.native.allocator_reserve_bytes])?;
        let max_output_bytes = self.mapping.max_result_bytes;
        let output_reservation_bytes = sum(&[mul(max_output_bytes, 8)?, 4096])?;
        sum(&[native.kv_bytes, workspace_bytes, input_bytes, output_reservation_bytes])?;
        Ok(Int8FaithfulnessMapMemoryRequirement { native, workspace_bytes, input_bytes,
            output_reservation_bytes, max_output_bytes })
    }

    /// Source probes, repeated claims and evidence heads share one transient
    /// encoder workspace and one cumulative tokenizer allowance.
    pub fn requirements_with_encoding(self, task: TaskBudget, encoding: &UntrustedEncoding)
        -> Result<Int8FaithfulnessMapMemoryRequirement, HostedError> {
        self.requirements(task)?.with_input_encoding(encoding)
    }
}
impl Int8FaithfulnessMapMemoryRequirement {
    pub fn with_input_encoding(mut self, encoding: &UntrustedEncoding) -> Result<Self, HostedError> {
        let workspace = match encoding {
            UntrustedEncoding::ByteFallback => 0,
            UntrustedEncoding::SafeBpeV1(limits) => limits.requirements()
                .map_err(|_| map_error(Int8FaithfulnessMapError::Limits))?.workspace_bytes,
        };
        self.input_bytes = sum(&[self.input_bytes, workspace])?;
        sum(&[self.native.kv_bytes, self.workspace_bytes, self.input_bytes, self.output_reservation_bytes])?;
        Ok(self)
    }
}

struct MapInput {
    request: JudgeRequest,
    identity: ExecutionIdentity,
    config: Int8FaithfulnessMapHostConfig,
    encoding: UntrustedEncoding,
}

impl NlpEngine {
    /// Judge the exact claim against every lossless source chunk, preserving
    /// all original-source evidence and explicit coverage/conflict abstention.
    /// Independent chunk judgments do not imply inference across contexts.
    /// No failed or cancelled chunk can publish a successful document prefix.
    pub fn map_int8_faithfulness(&self, model: &ResidentInt8, request: JudgeRequest,
        base_identity: ExecutionIdentity, config: Int8FaithfulnessMapHostConfig,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8FaithfulnessMapRun>, HostedError> {
        self.map_int8_faithfulness_with_encoding(model, request, base_identity, config,
            UntrustedEncoding::ByteFallback, cancellation)
    }

    /// Bind one explicit byte-preserving policy to every claim, source probe
    /// and evidence head. The base identity must already name the resident
    /// model and exact judge tokenizer/template; mismatches are never repaired.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int8_faithfulness_with_encoding(&self, model: &ResidentInt8, request: JudgeRequest,
        base_identity: ExecutionIdentity, config: Int8FaithfulnessMapHostConfig,
        encoding: UntrustedEncoding, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8FaithfulnessMapRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        self.check_resident_domain(model)?;
        validate_base_identity(&base_identity)?;
        check_model_identity(model.artifact_identity(), &base_identity)?;
        let task = validate_request(&request, config)?;
        let required = config.requirements_with_encoding(task, &encoding)?;
        let bytes = owned_input_bytes(&request, &base_identity, config, required)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(MapInput { request, identity: base_identity, config, encoding }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_reservation_bytes)?;
        let model = model.clone();
        dispatch::run(self, config.native.run, cancellation, move |control| {
            // The whole owner and its charge survive queued work, preparation,
            // execution, cancellation and result verification in this closure.
            let input = input;
            checkpoint(control)?;
            let registry = pinned_controls::pinned()
                .map_err(|_| planning_error(JudgeError::Contract("pinned judge control registry")))?;
            let controls = registry.template_controls();
            let eos = controls.entries().iter()
                .find(|entry| entry.special && entry.surface == crate::template::IM_END)
                .map(|entry| entry.id)
                .ok_or_else(|| planning_error(JudgeError::Contract("pinned judge EOS")))?;
            let planner = JudgePlanner::pinned(controls, eos)
                .and_then(|planner| planner.with_document_encoding(input.value.encoding.clone()))
                .map_err(planning_error)?;
            let (prepared, admitted) = prepare(&planner, &input.value, control)?;
            for identity in &admitted {
                checkpoint(control)?;
                check_model_identity(model.artifact_identity(), identity)?;
            }
            let expected = prepared.preflight_metadata();
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(config.native.context_tokens, memory_budget(required.native)).map_err(HostedError::Model))?;
            let result = allocate(output, || {
                let result = prepared.execute_with_control(&admitted, &mut engine.value,
                    Int8ScoringBudget { native: Int8RunBudget::exact(prepared.planned_work()),
                        max_kv_bytes: required.native.kv_bytes }, control).map_err(map_error)?;
                expected.verify_completed_with_control(&result, control).map_err(map_error)?;
                if engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(map_error(Int8FaithfulnessMapError::Admission));
                }
                checkpoint(control)?;
                Ok(result)
            })?;
            drop(engine); drop(admitted); drop(prepared); drop(planner); drop(registry); drop(input); drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

fn validate_request(request: &JudgeRequest, config: Int8FaithfulnessMapHostConfig)
    -> Result<TaskBudget, HostedError> {
    let JudgeRequest::Faithfulness { source, claim, policy, budget } = request else {
        return Err(map_error(Int8FaithfulnessMapError::Limits));
    };
    if source.is_empty() || claim.is_empty() { return Err(map_error(Int8FaithfulnessMapError::EmptySource)); }
    if source.len().checked_add(claim.len()).is_none_or(|bytes| bytes > config.mapping.chunks.max_input_bytes) {
        return Err(map_error(Int8FaithfulnessMapError::Limits));
    }
    policy.validate().map_err(planning_error)?;
    budget.validate().map_err(|_| map_error(Int8FaithfulnessMapError::Limits))?;
    // The entire source need not fit one evidence partition. Its lossless
    // chunks each receive a complete local partition during sealed planning.
    Ok(*budget)
}

fn validate_base_identity(identity: &ExecutionIdentity) -> Result<(), HostedError> {
    identity.validate().map_err(|_| HostedError::ModelIdentity)?;
    if identity.task_spec != "judge-v1"
        || identity.numerics_profile != (NumericsProfile::StrictQuantized { version: 1 })
        || identity.backend_semantic_version != STRICT_INT8_EXECUTION
        || identity.kv_dtype != "bf16" || identity.thinking_mode != ThinkingMode::Disabled
        || identity.tool_mode != ToolMode::None
        || identity_strings(identity).iter().any(|value| value.len() > 256) {
        return Err(HostedError::ModelIdentity);
    }
    Ok(())
}
fn check_planner_binding(planner: &JudgePlanner, identity: &ExecutionIdentity,
    encoding: &UntrustedEncoding) -> Result<(), HostedError> {
    validate_base_identity(identity)?;
    if planner.document_encoding() != encoding
        || identity.template_digest != *planner.template_digest()
        || identity.tokenizer_digest != planner.tokenizer_digest() {
        return Err(HostedError::ModelIdentity);
    }
    Ok(())
}

fn prepare<'a>(planner: &JudgePlanner, input: &'a MapInput, control: &mut impl DecodeStepControl)
    -> Result<(PreparedInt8FaithfulnessMap<'a>, Vec<ExecutionIdentity>), HostedError> {
    let config = input.config;
    let task = validate_request(&input.request, config)?;
    check_planner_binding(planner, &input.identity, &input.encoding)?;
    let context = PlanContext::new(&input.identity, task).map_err(|_| HostedError::ModelIdentity)?;
    let prepared = planner.plan_int8_faithfulness_document_with_control(&input.request, &context,
        config.planning, config.mapping, control).map_err(map_error)?;
    let JudgeRequest::Faithfulness { policy, .. } = &input.request else {
        return Err(map_error(Int8FaithfulnessMapError::Admission));
    };
    let max_heads = policy_head_limit(*policy);
    if prepared.chunk_count() == 0 || prepared.chunk_count() > config.mapping.chunks.max_chunks
        || prepared.head_count() < prepared.chunk_count()
        || prepared.head_count() > prepared.chunk_count().checked_mul(max_heads)
            .ok_or_else(|| map_error(Int8FaithfulnessMapError::Limits))?
        || !(1..=max_heads).contains(&prepared.max_head_count())
        || prepared.task_budget() != task || prepared.required_context() > config.native.context_tokens
        || prepared.max_result_bytes() != config.mapping.max_result_bytes {
        return Err(map_error(Int8FaithfulnessMapError::Admission));
    }
    check_work(prepared.planned_work(), config.mapping.max_model_work)?;
    let admitted = prepared.identities().map_err(map_error)?;
    if admitted.len() != prepared.chunk_count() {
        return Err(map_error(Int8FaithfulnessMapError::Admission));
    }
    for identity in &admitted {
        checkpoint(control)?;
        validate_base_identity(identity)?;
        if identity.tokenizer_digest != input.identity.tokenizer_digest
            || identity.template_digest != input.identity.template_digest
            || identity.source_revision != input.identity.source_revision
            || identity.logical_model_digest != input.identity.logical_model_digest
            || identity.quant_recipe != input.identity.quant_recipe
            || identity.artifact_format != input.identity.artifact_format
            || identity.packing_set_digest != input.identity.packing_set_digest {
            return Err(HostedError::ModelIdentity);
        }
    }
    checkpoint(control)?;
    Ok((prepared, admitted))
}
fn policy_head_limit(policy: FaithfulnessPolicy) -> usize {
    if policy.max_evidence_windows == 1 { 1 } else { policy.max_evidence_windows as usize + 1 }
}
fn check_work(work: Int8Work, cap: Int8Work) -> Result<(), HostedError> {
    if work.forward_positions > cap.forward_positions || work.projected_logits > cap.projected_logits
        || work.attention_pairs > cap.attention_pairs || !work.projections.fits(cap.projections) {
        return Err(map_error(Int8FaithfulnessMapError::Work));
    }
    Ok(())
}
fn owned_input_bytes(request: &JudgeRequest, identity: &ExecutionIdentity,
    config: Int8FaithfulnessMapHostConfig, required: Int8FaithfulnessMapMemoryRequirement) -> Result<u64, HostedError> {
    let JudgeRequest::Faithfulness { source, claim, .. } = request else {
        return Err(map_error(Int8FaithfulnessMapError::Limits));
    };
    // Moved caller capacities can exceed logical ceilings. Keep and price
    // their actual owners without copying or shrinking private source/claim.
    let capacity = sum(&[source.capacity() as u64, claim.capacity() as u64])?;
    let identity_bytes = identity_strings(identity).iter().try_fold(std::mem::size_of::<ExecutionIdentity>() as u64,
        |total, value| sum(&[total, value.capacity() as u64]))?;
    sum(&[required.input_bytes, capacity.saturating_sub(config.mapping.chunks.max_input_bytes as u64),
        identity_bytes.saturating_sub(BASE_IDENTITY_BYTES)])
}
fn identity_strings(identity: &ExecutionIdentity) -> [&String; 8] {
    [&identity.source_revision, &identity.artifact_format, &identity.quant_recipe,
        &identity.task_spec, &identity.grammar_compiler_version, &identity.kv_dtype,
        &identity.sampler_version, &identity.backend_semantic_version]
}
fn mul(a: u64, b: u64) -> Result<u64, HostedError> {
    a.checked_mul(b).ok_or_else(|| map_error(Int8FaithfulnessMapError::Limits))
}
fn map_error(error: Int8FaithfulnessMapError) -> HostedError { HostedError::FaithfulnessMap(error) }
fn planning_error(error: JudgeError) -> HostedError { map_error(error.into()) }
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), HostedError> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(planning_error(JudgeError::Cancelled(cause))),
        None => Ok(()),
    }
}

#[cfg(test)] mod tests;
