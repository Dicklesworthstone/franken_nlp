//! Complete-source faithfulness under one owned native invocation.
use super::*;
use crate::{native_engine::lmhead::NANBEIGE_VOCAB_SIZE,
    tasks::judge::{JudgeLimits, JudgeRequest,
    grouped_int4::{Int4JudgePlanner,
        faithfulness_map::{Int4FaithfulnessMapError, Int4FaithfulnessMapLimits,
            Int4FaithfulnessMapRun, Int4FaithfulnessMapCohortBudget, Int4FaithfulnessMapCohortRun,
            PreparedInt4FaithfulnessMap, INT4_FAITHFULNESS_MAP_COHORT_EXECUTION,
            memory::{Int4FaithfulnessMapMemoryLimits, Int4FaithfulnessMapMemoryRequirement,
                Int4FaithfulnessMapCohortMemoryRequirement}}}}};

/// One claim, the entire original source, and fixed per-chunk judgment
/// semantics. Work and output ceilings cover the complete document.
#[derive(Clone, Copy, Debug)]
pub struct Int4FaithfulnessMapHostConfig {
    pub native: NativeLimits,
    pub planning: JudgeLimits,
    pub mapping: Int4FaithfulnessMapLimits,
    pub preparation_reserve_bytes: u64,
}
impl Int4FaithfulnessMapHostConfig {
    /// Pure quote for every source/claim byte, all retained plans and results,
    /// one serial native cache, and complete output. Usable before input IO.
    pub fn requirements(self, width: usize, task: TaskBudget)
        -> Result<Int4FaithfulnessMapMemoryRequirement, HostedError> {
        self.native.run.validate()?;
        self.memory_limits().requirements(width, task, self.planning, self.mapping).map_err(map_error)
    }
    /// Price the largest admissible ragged window: at most 32 heads per
    /// chunk and 64 active heads. Exact planned windows still must fit; this
    /// allocation bound never changes the requested chunk schedule.
    pub fn cohort_requirements(self, width: usize, task: TaskBudget, cohort_chunks: usize)
        -> Result<Int4FaithfulnessMapCohortMemoryRequirement, HostedError> {
        self.native.run.validate()?;
        self.memory_limits().cohort_requirements(width, task, self.planning, self.mapping,
            32, cohort_chunks).map_err(map_error)
    }
    fn memory_limits(self) -> Int4FaithfulnessMapMemoryLimits {
        Int4FaithfulnessMapMemoryLimits { context_tokens: self.native.context_tokens,
            preparation_reserve_bytes: self.preparation_reserve_bytes,
            allocator_reserve_bytes: self.native.allocator_reserve_bytes }
    }
}

struct MapInput { request: JudgeRequest, config: Int4FaithfulnessMapHostConfig }

impl NlpEngine {
    /// Judge every lossless source chunk against the same exact claim. The
    /// returned evidence uses original-source coordinates and retains its
    /// memory guard through caller publication. A failed chunk yields no map.
    pub fn map_int4_faithfulness(&self, model: &ResidentInt4, request: JudgeRequest,
        config: Int4FaithfulnessMapHostConfig, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4FaithfulnessMapRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let task = validate_request(&request, config)?;
        let required = config.requirements(model.model_identity().group_width(), task)?;
        let input_bytes = owned_input_bytes(&request, config, required)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?,
            || Ok(MapInput { request, config }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_reservation_bytes)?;
        let model = model.clone();
        dispatch::run(self, config.native.run, cancellation, move |control| {
            let input = input;
            checkpoint(control)?;
            let planner = Int4JudgePlanner::for_model(model.model_identity(), task, config.planning)
                .map_err(judge_error)?;
            let (prepared, admitted) = prepare(&planner, &model, &input.value, control)?;
            let expected = prepared.preflight_metadata();
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, config.native.context_tokens,
                    native_budget(required.native)).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let result = allocate(output, || {
                let result = prepared.execute_with_control(&admitted, &mut engine.value,
                    Int4ScoringBudget { native: Int4RunBudget::exact(prepared.planned_work()),
                        max_kv_bytes: required.native.kv_bytes }, control).map_err(map_error)?;
                expected.verify_completed(&result).map_err(map_error)?;
                if engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(map_error(Int4FaithfulnessMapError::Admission));
                }
                checkpoint(control)?;
                Ok(result)
            })?;
            drop(engine); drop(admitted); drop(prepared); drop(planner); drop(input); drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }

    /// Share native decoder projections across complete chunks and their
    /// independent evidence heads. Every future ragged window is admitted
    /// before the first forward, with no fallback or partial publication.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int4_faithfulness_cohorts(&self, model: &ResidentInt4, request: JudgeRequest,
        config: Int4FaithfulnessMapHostConfig, cohort_chunks: usize, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4FaithfulnessMapCohortRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let task = validate_request(&request, config)?;
        let required = config.cohort_requirements(model.model_identity().group_width(), task, cohort_chunks)?;
        let memory = required.memory;
        let input_bytes = owned_input_bytes(&request, config, memory)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?,
            || Ok(MapInput { request, config }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, memory.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, memory.workspace_bytes)?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, memory.output_reservation_bytes)?;
        let model = model.clone();
        dispatch::run(self, config.native.run, cancellation, move |control| {
            let input = input;
            checkpoint(control)?;
            let planner = Int4JudgePlanner::for_model(model.model_identity(), task, config.planning)
                .map_err(judge_error)?;
            let (prepared, admitted) = prepare(&planner, &model, &input.value, control)?;
            prepared.validate_cohort_chunks(cohort_chunks).map_err(map_error)?;
            let expected = prepared.preflight_metadata();
            // These peak charges survive every exact-sized native window,
            // cancellation/drain and final evidence reduction.
            let kv = kv.commit()?;
            let scratch = scratch.commit()?;
            let result = allocate(output, || {
                let result = prepared.execute_cohorts_with_control(&admitted, &model.inner.loaded.value,
                    kernel, cohort_chunks, Int4FaithfulnessMapCohortBudget {
                        native: Int4RunBudget::exact(prepared.planned_work()),
                        memory: native_budget(memory.native),
                        max_task_workspace_bytes: required.task_workspace_bytes,
                        max_output_bytes: required.max_cohort_output_bytes,
                    }, control).map_err(map_error)?;
                if result.schema_version != 1 || result.execution != INT4_FAITHFULNESS_MAP_COHORT_EXECUTION
                    || result.chunk_cohorts != prepared.chunk_count().div_ceil(cohort_chunks)
                    || !valid_physical_counts(result.output.model_work, result.group_steps, result.projection_groups) {
                    return Err(map_error(Int4FaithfulnessMapError::Admission));
                }
                expected.verify_completed(&result.output).map_err(map_error)?;
                checkpoint(control)?;
                Ok(result)
            })?;
            drop(admitted); drop(prepared); drop(planner); drop(input); drop(scratch); drop(kv); drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

fn validate_request(request: &JudgeRequest, config: Int4FaithfulnessMapHostConfig)
    -> Result<TaskBudget, HostedError> {
    let JudgeRequest::Faithfulness { source, claim, policy, budget } = request else {
        return Err(map_error(Int4FaithfulnessMapError::Limits));
    };
    if source.is_empty() || claim.is_empty()
        || source.len().checked_add(claim.len()).is_none_or(|bytes| bytes > config.mapping.chunks.max_input_bytes) {
        return Err(map_error(Int4FaithfulnessMapError::Limits));
    }
    policy.validate().map_err(|error| judge_error(error.into()))?;
    Ok(*budget)
}
fn owned_input_bytes(request: &JudgeRequest, config: Int4FaithfulnessMapHostConfig,
    required: Int4FaithfulnessMapMemoryRequirement) -> Result<u64, HostedError> {
    let JudgeRequest::Faithfulness { source, claim, .. } = request else {
        return Err(map_error(Int4FaithfulnessMapError::Limits));
    };
    // Price moved caller capacities, including excess retained by a tiny
    // source or claim, without reallocating or copying either private string.
    let capacity = sum(&[source.capacity() as u64, claim.capacity() as u64])?;
    sum(&[required.input_bytes, capacity.saturating_sub(config.mapping.chunks.max_input_bytes as u64)])
}
fn prepare<'a>(planner: &Int4JudgePlanner, model: &ResidentInt4, input: &'a MapInput,
    control: &mut impl DecodeStepControl)
    -> Result<(PreparedInt4FaithfulnessMap<'a>, Vec<ExecutionIdentity>), HostedError> {
    let config = input.config;
    let task = validate_request(&input.request, config)?;
    let prepared = planner.plan_faithfulness_document_with_control(&input.request, config.mapping, control)
        .map_err(map_error)?;
    if prepared.group_width() != model.model_identity().group_width()
        || prepared.chunk_count() == 0 || prepared.chunk_count() > config.mapping.chunks.max_chunks
        || prepared.head_count() < prepared.chunk_count() || !(1..=32).contains(&prepared.max_head_count())
        || prepared.task_budget() != task || prepared.required_context() > config.native.context_tokens
        || prepared.max_result_bytes() != config.mapping.max_result_bytes {
        return Err(map_error(Int4FaithfulnessMapError::Admission));
    }
    check_work(prepared.planned_work(), config.mapping.max_model_work)?;
    let admitted = prepared.identities().map_err(map_error)?;
    if admitted.len() != prepared.chunk_count() { return Err(map_error(Int4FaithfulnessMapError::Admission)); }
    for identity in &admitted {
        checkpoint(control)?;
        judge_task::check_model(identity, model.model_identity()).map_err(judge_error)?;
        if identity.task_spec != planner.execution_identity().task_spec
            || identity.tokenizer_digest != planner.execution_identity().tokenizer_digest
            || identity.template_digest != planner.execution_identity().template_digest {
            return Err(map_error(Int4FaithfulnessMapError::Admission));
        }
    }
    checkpoint(control)?;
    Ok((prepared, admitted))
}
fn check_work(work: Int4Work, cap: Int4Work) -> Result<(), HostedError> {
    if work.forward_positions > cap.forward_positions || work.projected_logits > cap.projected_logits
        || work.attention_pairs > cap.attention_pairs || !work.projections.fits(cap.projections) {
        return Err(map_error(Int4FaithfulnessMapError::Work));
    }
    Ok(())
}
fn native_budget(required: Int4MemoryRequirement) -> Int4MemoryBudget {
    Int4MemoryBudget { max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
        max_scratch_payload_bytes: required.scratch_payload_bound }
}
fn valid_physical_counts(work: Int4Work, groups: u64, projections: u64) -> bool {
    // A projection group is one or more full-vocabulary prefix evaluations,
    // never one group per projected logit. Retain the exact unit conversion.
    let vocabulary = NANBEIGE_VOCAB_SIZE as u64;
    groups > 0 && groups <= work.forward_positions && work.projected_logits % vocabulary == 0
        && projections > 0 && projections <= work.projected_logits / vocabulary
}
fn map_error(error: Int4FaithfulnessMapError) -> HostedError {
    HostedError::Int4(Int4HostError::FaithfulnessMap(error))
}

#[cfg(test)] mod tests;
