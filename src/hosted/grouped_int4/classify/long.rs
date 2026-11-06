//! One charged serial invocation from complete document to complete mixture.
use super::*;
use crate::tasks::classify::{ClassificationLabel, ClassificationLimits, ClassificationMode, ClassificationRequest,
    grouped_int4::{Int4ClassificationPlanner,
        long::{Int4ClassificationMapError, Int4ClassificationMapLimits, Int4ClassificationMapRun,
            memory::{Int4ClassificationMapMemoryLimits, Int4ClassificationMapMemoryRequirement}}}};

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
}
struct MapInput { request: ClassificationRequest, config: Int4ClassificationMapHostConfig }

impl NlpEngine {
    /// Admit every original source chunk and its complete independent heads,
    /// then execute serially on one native cache. All five native work axes
    /// have one nonrenewable document ceiling. The returned guarded output
    /// retains complete chunks and the separately named aggregate estimator.
    pub fn map_int4_classify(&self, model: &ResidentInt4, request: ClassificationRequest,
        config: Int4ClassificationMapHostConfig, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4ClassificationMapRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let required = config.requirements(model.model_identity().group_width(), request.budget)?;
        validate_request(&request, config)?;
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
            let planner = Int4ClassificationPlanner::for_model(model.model_identity(), input.value.request.budget,
                input.value.config.planning).map_err(classification_error)?;
            let prepared = planner.plan_document_with_control(&input.value.request, input.value.config.mapping, control)
                .map_err(map_error)?;
            let expected = prepared.preflight_metadata();
            let expected_heads = if input.value.request.mode == ClassificationMode::Exclusive { 1 }
                else { input.value.request.labels.len() };
            if prepared.group_width() != model.model_identity().group_width() || prepared.head_count() != expected_heads
                || prepared.chunk_count() == 0 || prepared.chunk_count() > config.mapping.chunks.max_chunks
                || prepared.task_budget() != input.value.request.budget
                || prepared.required_context() > config.native.context_tokens
                || prepared.max_result_bytes() != config.mapping.max_result_bytes {
                return Err(map_error(Int4ClassificationMapError::Admission));
            }
            let admitted = prepared.identities().map_err(map_error)?;
            for identity in &admitted {
                checkpoint(control)?;
                grouped_int4::check_model(identity, model.model_identity()).map_err(classification_error)?;
                if identity.task_spec != planner.execution_identity().task_spec
                    || identity.tokenizer_digest != planner.execution_identity().tokenizer_digest
                    || identity.template_digest != planner.execution_identity().template_digest {
                    return Err(map_error(Int4ClassificationMapError::Admission));
                }
            }
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
