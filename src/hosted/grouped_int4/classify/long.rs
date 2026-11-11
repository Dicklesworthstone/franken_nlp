//! One charged invocation from complete document to complete mixture.
use super::*;
use crate::tokenizer::untrusted::UntrustedEncoding;
use crate::tasks::classify::{ClassificationLabel, ClassificationLimits, ClassificationMode, ClassificationRequest,
    grouped_int4::{Int4ClassificationPlanner,
        long::{Int4ClassificationMapCohortBudget, Int4ClassificationMapCohortRun,
            Int4ClassificationMapError, Int4ClassificationMapLimits, Int4ClassificationMapRun,
            PreparedInt4ClassificationMap,
            memory::{Int4ClassificationMapMemoryLimits, Int4ClassificationMapMemoryRequirement,
                Int4ClassificationMapCohortMemoryRequirement}}}};

#[derive(Clone, Copy, Debug)]
pub struct Int4ClassificationMapHostConfig {
    pub native: NativeLimits,
    /// `max_input_bytes` bounds the entire original source, not one chunk.
    pub planning: ClassificationLimits,
    pub mapping: Int4ClassificationMapLimits,
    pub preparation_reserve_bytes: u64,
}
impl Int4ClassificationMapHostConfig {
    /// Pure checked quote, usable before any source or model IO. It admits
    /// the declared maximum label count in either classification mode.
    pub fn requirements(self, width: usize, task: TaskBudget)
        -> Result<Int4ClassificationMapMemoryRequirement, HostedError> {
        self.native.run.validate()?;
        Int4ClassificationMapMemoryLimits { context_tokens: self.native.context_tokens,
            preparation_reserve_bytes: self.preparation_reserve_bytes,
            allocator_reserve_bytes: self.native.allocator_reserve_bytes }
            .requirements(width, task, self.planning, self.mapping).map_err(map_error)
    }
    pub fn requirements_with_encoding(self, width: usize, task: TaskBudget, encoding: &UntrustedEncoding)
        -> Result<Int4ClassificationMapMemoryRequirement, HostedError> {
        self.requirements(width, task)?.with_input_encoding(encoding).map_err(map_error)
    }
    /// Complete checked quote for explicit native chunk cohorts. `max_heads`
    /// is one for exclusive classification or the declared multilabel count.
    /// Its task workspace is already charged inside `memory.input_bytes`.
    pub fn cohort_requirements(self, width: usize, task: TaskBudget, max_heads: usize, cohort_chunks: usize)
        -> Result<Int4ClassificationMapCohortMemoryRequirement, HostedError> {
        self.native.run.validate()?;
        Int4ClassificationMapMemoryLimits { context_tokens: self.native.context_tokens,
            preparation_reserve_bytes: self.preparation_reserve_bytes,
            allocator_reserve_bytes: self.native.allocator_reserve_bytes }
            .cohort_requirements(width, task, self.planning, self.mapping, max_heads, cohort_chunks).map_err(map_error)
    }
    pub fn cohort_requirements_with_encoding(self, width: usize, task: TaskBudget, max_heads: usize,
        cohort_chunks: usize, encoding: &UntrustedEncoding)
        -> Result<Int4ClassificationMapCohortMemoryRequirement, HostedError> {
        let mut required = self.cohort_requirements(width, task, max_heads, cohort_chunks)?;
        required.memory = required.memory.with_input_encoding(encoding).map_err(map_error)?;
        Ok(required)
    }
}
struct MapInput { request: ClassificationRequest, config: Int4ClassificationMapHostConfig, encoding: UntrustedEncoding }

impl NlpEngine {
    /// Admit every original source chunk and its complete independent heads,
    /// then execute serially on one native cache. All five native work axes
    /// have one nonrenewable document ceiling. The returned guarded output
    /// retains complete chunks and the separately named aggregate estimator.
    pub fn map_int4_classify(&self, model: &ResidentInt4, request: ClassificationRequest,
        config: Int4ClassificationMapHostConfig, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4ClassificationMapRun>, HostedError> {
        self.map_int4_classify_with_encoding(model, request, config, UntrustedEncoding::ByteFallback, kernel, cancellation)
    }
    /// Explicit compact input uses one document-wide tokenizer allowance,
    /// including all failed-size probes and every actual chunk plan. Model
    /// execution still starts only after the entire source has been planned.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int4_classify_with_encoding(&self, model: &ResidentInt4, request: ClassificationRequest,
        config: Int4ClassificationMapHostConfig, encoding: UntrustedEncoding, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4ClassificationMapRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let required = config.requirements_with_encoding(model.model_identity().group_width(), request.budget, &encoding)?;
        validate_request(&request, config)?;
        let input_bytes = owned_input_bytes(&request, config, required)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?,
            || Ok(MapInput { request, config, encoding }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_reservation_bytes)?;
        let model = model.clone();
        dispatch::run(self, config.native.run, cancellation, move |control| {
            let input = input;
            checkpoint(control)?;
            let planner = Int4ClassificationPlanner::for_model(model.model_identity(), input.value.request.budget,
                input.value.config.planning).and_then(|planner| planner.with_document_encoding(input.value.encoding.clone()))
                .map_err(classification_error)?;
            let (prepared, admitted) = prepare(&planner, &model, &input.value, control)?;
            let expected = prepared.preflight_metadata();
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, config.native.context_tokens, Int4MemoryBudget {
                    max_kv_bytes: required.native.kv_bytes, max_rope_bytes: required.native.rope_bytes,
                    max_scratch_payload_bytes: required.native.scratch_payload_bound,
                }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let result = allocate(output, || {
                let result = prepared.execute_with_control(&admitted, &mut engine.value,
                    Int4ScoringBudget { native: Int4RunBudget::exact(prepared.planned_work()),
                        max_kv_bytes: required.native.kv_bytes }, control).map_err(map_error)?;
                expected.verify_completed(&result).map_err(map_error)?;
                if engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(map_error(Int4ClassificationMapError::Admission));
                }
                checkpoint(control)?; Ok(result)
            })?;
            drop(engine); drop(admitted); drop(prepared); drop(planner); drop(input); drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }

    /// Keep one sealed output-row team across every chunk and every label
    /// head. Planning, the complete work ceiling and deterministic reduction
    /// are identical to the serial document route. Only physical projection
    /// scheduling changes; independent chunk cohorts are a separate mode.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int4_classify_with_threads(&self, model: &ResidentInt4, request: ClassificationRequest,
        config: Int4ClassificationMapHostConfig, kernel: PackedInt4Kernel, compute_threads: usize,
        cancellation: CancellationToken) -> Result<HostedInt4TeamResult<Int4ClassificationMapRun>, HostedError> {
        self.map_int4_classify_with_encoding_and_threads(model, request, config,
            UntrustedEncoding::ByteFallback, kernel, compute_threads, cancellation)
    }

    /// Combine an explicit complete-document input policy with one sealed
    /// numerical team. Tokenizer work belongs to the whole plan and is never
    /// renewed by a chunk, label head or worker. Each workspace is priced once.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int4_classify_with_encoding_and_threads(&self, model: &ResidentInt4, request: ClassificationRequest,
        config: Int4ClassificationMapHostConfig, encoding: UntrustedEncoding, kernel: PackedInt4Kernel,
        compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt4TeamResult<Int4ClassificationMapRun>, HostedError> {
        dispatch::preflight_int4_team(self, config.native.run, compute_threads)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let required = config.requirements_with_encoding(model.model_identity().group_width(), request.budget, &encoding)?;
        validate_request(&request, config)?;
        let input_bytes = owned_input_bytes(&request, config, required)?;
        // The dispatcher owns this additional charge until its workers have
        // physically joined, including cancelled and failed document calls.
        sum(&[required.workspace_bytes,
            int4_task_team_workspace_bytes(compute_threads, model.model_identity().group_width())?])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?,
            || Ok(MapInput { request, config, encoding }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_reservation_bytes)?;
        let model = model.clone();
        dispatch::run_int4_team(self, config.native.run, cancellation, model.clone(), compute_threads,
            kernel, move |weights, control, team| {
                let input = input;
                checkpoint(control)?;
                let planner = Int4ClassificationPlanner::for_model(model.model_identity(), input.value.request.budget,
                    input.value.config.planning).and_then(|planner| planner.with_document_encoding(input.value.encoding.clone()))
                    .map_err(classification_error)?;
                let (prepared, admitted) = prepare(&planner, &model, &input.value, control)?;
                let expected = prepared.preflight_metadata();
                let mut engine = allocate_native(kv, scratch, || {
                    StrictInt4Engine::new(weights, config.native.context_tokens, Int4MemoryBudget {
                        max_kv_bytes: required.native.kv_bytes, max_rope_bytes: required.native.rope_bytes,
                        max_scratch_payload_bytes: required.native.scratch_payload_bound,
                    }).map_err(native_error)?.with_kernel(kernel)
                        .with_projection_executor(team).map_err(native_error)
                })?;
                let result = allocate(output, || {
                    let result = prepared.execute_with_control(&admitted, &mut engine.value,
                        Int4ScoringBudget { native: Int4RunBudget::exact(prepared.planned_work()),
                            max_kv_bytes: required.native.kv_bytes }, control).map_err(map_error)?;
                    expected.verify_completed(&result).map_err(map_error)?;
                    if engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                        return Err(map_error(Int4ClassificationMapError::Admission));
                    }
                    checkpoint(control)?; Ok(result)
                })?;
                drop(engine); drop(admitted); drop(prepared); drop(planner); drop(input); drop(lease);
                Ok(HostedInt4TeamResult::new(GuardedOutput::new(result.value, result._memory), team.stats().into()))
            })
    }

    /// Share native projections across complete independent chunk/head
    /// contexts. One deadline, cancellation token and document work budget
    /// cover every window, reduction and complete guarded publication.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int4_classify_cohorts(&self, model: &ResidentInt4, request: ClassificationRequest,
        config: Int4ClassificationMapHostConfig, cohort_chunks: usize, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4ClassificationMapCohortRun>, HostedError> {
        self.map_int4_classify_cohorts_with_encoding(model, request, config, cohort_chunks,
            UntrustedEncoding::ByteFallback, kernel, cancellation)
    }
    /// Compact document maps retain independent contexts in every native
    /// window. The encoding allowance belongs to the complete source, not a
    /// window, and the same sealed result metadata verifies either schedule.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int4_classify_cohorts_with_encoding(&self, model: &ResidentInt4, request: ClassificationRequest,
        config: Int4ClassificationMapHostConfig, cohort_chunks: usize, encoding: UntrustedEncoding,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4ClassificationMapCohortRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        validate_cohort_request(&request, config, cohort_chunks)?;
        let required = config.cohort_requirements_with_encoding(model.model_identity().group_width(), request.budget,
            head_count(&request), cohort_chunks, &encoding)?;
        let memory = required.memory;
        let input_bytes = owned_input_bytes(&request, config, memory)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?,
            || Ok(MapInput { request, config, encoding }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, memory.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, memory.workspace_bytes)?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, memory.output_reservation_bytes)?;
        let model = model.clone();
        dispatch::run(self, config.native.run, cancellation, move |control| {
            let input = input;
            checkpoint(control)?;
            let planner = Int4ClassificationPlanner::for_model(model.model_identity(), input.value.request.budget,
                input.value.config.planning).and_then(|planner| planner.with_document_encoding(input.value.encoding.clone()))
                .map_err(classification_error)?;
            let (prepared, admitted) = prepare(&planner, &model, &input.value, control)?;
            let expected = prepared.preflight_metadata();
            // These charges remain owned until every exact-sized native
            // window has physically dropped, including failure/cancellation.
            let kv = kv.commit()?;
            let scratch = scratch.commit()?;
            let result = allocate(output, || {
                let result = prepared.execute_cohorts_with_control(&admitted, &model.inner.loaded.value,
                    kernel, cohort_chunks, Int4ClassificationMapCohortBudget {
                        native: Int4RunBudget::exact(prepared.planned_work()),
                        memory: Int4MemoryBudget { max_kv_bytes: memory.native.kv_bytes,
                            max_rope_bytes: memory.native.rope_bytes,
                            max_scratch_payload_bytes: memory.native.scratch_payload_bound },
                        max_task_workspace_bytes: required.task_workspace_bytes,
                        max_output_bytes: required.max_cohort_output_bytes,
                    }, control).map_err(map_error)?;
                expected.verify_completed_cohorts(&result, cohort_chunks).map_err(map_error)?;
                checkpoint(control)?; Ok(result)
            })?;
            drop(admitted); drop(prepared); drop(planner); drop(input);
            drop(scratch); drop(kv); drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

fn head_count(request: &ClassificationRequest) -> usize {
    if request.mode == ClassificationMode::Exclusive { 1 } else { request.labels.len() }
}
fn prepare<'a>(planner: &Int4ClassificationPlanner, model: &ResidentInt4, input: &'a MapInput,
    control: &mut impl DecodeStepControl)
    -> Result<(PreparedInt4ClassificationMap<'a>, Vec<ExecutionIdentity>), HostedError> {
    if planner.document_encoding() != &input.encoding { return Err(HostedError::ModelIdentity); }
    let prepared = planner.plan_document_with_control(&input.request, input.config.mapping, control).map_err(map_error)?;
    let expected_heads = head_count(&input.request);
    if prepared.group_width() != model.model_identity().group_width() || prepared.head_count() != expected_heads
        || prepared.chunk_count() == 0 || prepared.chunk_count() > input.config.mapping.chunks.max_chunks
        || prepared.task_budget() != input.request.budget
        || prepared.required_context() > input.config.native.context_tokens
        || prepared.max_result_bytes() != input.config.mapping.max_result_bytes {
        return Err(map_error(Int4ClassificationMapError::Admission));
    }
    let admitted = prepared.identities().map_err(map_error)?;
    if admitted.len() != prepared.chunk_count() { return Err(map_error(Int4ClassificationMapError::Admission)); }
    for identity in &admitted {
        checkpoint(control)?;
        grouped_int4::check_model(identity, model.model_identity()).map_err(classification_error)?;
        if identity.task_spec != planner.execution_identity().task_spec
            || identity.tokenizer_digest != planner.execution_identity().tokenizer_digest
            || identity.template_digest != planner.execution_identity().template_digest {
            return Err(map_error(Int4ClassificationMapError::Admission));
        }
    }
    checkpoint(control)?;
    Ok((prepared, admitted))
}

fn validate_cohort_request(request: &ClassificationRequest, config: Int4ClassificationMapHostConfig,
    cohort_chunks: usize) -> Result<(), HostedError> {
    validate_request(request, config)?;
    let heads = head_count(request);
    if cohort_chunks == 0 || cohort_chunks > config.mapping.chunks.max_chunks
        || cohort_chunks.checked_mul(heads).is_none_or(|rows|
            rows > crate::native_engine::strict_int4::cohort::MAX_INT4_SEQUENCES) {
        return Err(map_error(Int4ClassificationMapError::Limits));
    }
    Ok(())
}
fn validate_request(request: &ClassificationRequest, config: Int4ClassificationMapHostConfig) -> Result<(), HostedError> {
    let limits = config.planning;
    if request.document.is_empty() { return Err(map_error(Int4ClassificationMapError::EmptySource)); }
    if request.document.len() > config.mapping.chunks.max_input_bytes || request.document.len() > limits.max_input_bytes
        || request.labels.is_empty() || request.labels.len() > limits.max_labels
        || (request.mode == ClassificationMode::Exclusive && request.labels.len() < 2)
        || request.policy.minimum_candidate_weight_ppm > 1_000_000 || request.policy.minimum_margin_ppm > 1_000_000 {
        return Err(map_error(Int4ClassificationMapError::Limits));
    }
    let mut bytes = 0_usize;
    for (index, label) in request.labels.iter().enumerate() {
        bytes = bytes.checked_add(label.id.len()).and_then(|n| n.checked_add(label.description.len()))
            .ok_or_else(|| map_error(Int4ClassificationMapError::Limits))?;
        if label.id.trim().is_empty() || label.id.chars().any(char::is_control)
            || label.id.len() > limits.max_label_id_bytes || label.description.len() > limits.max_label_description_bytes
            || bytes > limits.max_total_label_bytes || request.labels[..index].iter().any(|previous| previous.id == label.id) {
            return Err(map_error(Int4ClassificationMapError::Limits));
        }
    }
    Ok(())
}
fn owned_input_bytes(request: &ClassificationRequest, config: Int4ClassificationMapHostConfig,
    required: Int4ClassificationMapMemoryRequirement) -> Result<u64, HostedError> {
    // Retained caller capacities can exceed their declared logical maxima.
    // Charge the excess without cloning or shrinking the transferred owner.
    let source = request.document.capacity().saturating_sub(config.mapping.chunks.max_input_bytes) as u64;
    let labels = (request.labels.capacity().saturating_sub(config.planning.max_labels) as u64)
        .checked_mul(std::mem::size_of::<ClassificationLabel>() as u64)
        .ok_or_else(|| map_error(Int4ClassificationMapError::Limits))?;
    let text = request.labels.iter().try_fold(0_u64, |total, label|
        sum(&[total, label.id.capacity() as u64, label.description.capacity() as u64]))?;
    sum(&[required.input_bytes, source, labels, text.saturating_sub(config.planning.max_total_label_bytes as u64)])
}
fn map_error(error: Int4ClassificationMapError) -> HostedError { HostedError::Int4(Int4HostError::ClassificationMap(error)) }

#[cfg(test)] mod tests;
