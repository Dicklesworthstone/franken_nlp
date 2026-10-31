//! One owned, fully admitted invocation for complete-document sentiment.
use super::*;
use crate::{native_engine::lmhead::NANBEIGE_VOCAB_SIZE,
    tasks::sentiment::{SentimentAxis, SentimentLimits, SentimentOptions, SentimentRequest,
        grouped_int4::{Int4SentimentPlanner,
            long::{Int4SentimentMapCohortBudget, Int4SentimentMapCohortRun,
                Int4SentimentMapError, Int4SentimentMapLimits, Int4SentimentMapPreflight,
                Int4SentimentMapRun, PreparedInt4SentimentMap,
                INT4_SENTIMENT_MAP_COHORT_EXECUTION,
                memory::{Int4SentimentMapMemoryLimits, Int4SentimentMapMemoryRequirement}}}}};

/// Fixed per-chunk semantics and complete-document work/result allowances.
/// The source request retains the same task budget in serial and cohort modes.
#[derive(Clone, Copy, Debug)]
pub struct Int4SentimentMapHostConfig {
    pub native: NativeLimits,
    pub planning: SentimentLimits,
    pub mapping: Int4SentimentMapLimits,
    pub options: SentimentOptions,
    pub preparation_reserve_bytes: u64,
}
impl Int4SentimentMapHostConfig {
    /// Pure quote usable before source/model IO. `max_axes = 4` admits the
    /// largest valid unknown request; actual cohort geometry is still checked.
    pub fn requirements(self, group_width: usize, task: TaskBudget, max_axes: usize,
        cohort_chunks: Option<usize>) -> Result<Int4SentimentMapMemoryRequirement, HostedError> {
        self.native.run.validate()?;
        if self.options.eos_token_id as usize >= NANBEIGE_VOCAB_SIZE
            || self.options.policy.minimum_peak_weight_ppm > 1_000_000
            || self.options.policy.maximum_normalized_entropy_ppm > 1_000_000 {
            return Err(map_error(Int4SentimentMapError::Limits));
        }
        Int4SentimentMapMemoryLimits { context_tokens: self.native.context_tokens,
            preparation_reserve_bytes: self.preparation_reserve_bytes,
            allocator_reserve_bytes: self.native.allocator_reserve_bytes,
            max_axes, cohort_chunks }
            .requirements(group_width, task, self.planning, self.mapping).map_err(map_error)
    }
}

struct MapInput { request: SentimentRequest, config: Int4SentimentMapHostConfig }

impl NlpEngine {
    /// Plan every chunk, admit the whole five-axis work sum, then execute on
    /// one reusable native cache. Complete chunk evidence and the original-
    /// scalar-weighted aggregate share one guarded output and one deadline.
    pub fn map_int4_sentiment(&self, model: &ResidentInt4, request: SentimentRequest,
        config: Int4SentimentMapHostConfig, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4SentimentMapRun>, HostedError> {
        let (required, input_bytes) = admit(self, model, &request, config, None)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?,
            || Ok(MapInput { request, config }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_reservation_bytes)?;
        // All declarations are admitted before cloning the resident handle or
        // constructing the tokenizer, source chunks or retained task plans.
        let model = model.clone();
        dispatch::run(self, config.native.run, cancellation, move |control| {
            let input = input;
            checkpoint(control)?;
            let planner = planner(&model, &input.value)?;
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
                finish(&expected, &result, control)?;
                if engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(map_error(Int4SentimentMapError::Admission));
                }
                Ok(result)
            })?;
            drop(engine);
            drop(admitted);
            drop(prepared);
            drop(planner);
            drop(input);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }

    /// Share real decoder layers across finite cohorts of independent chunk
    /// axes. All exact ragged windows are admitted before the first forward;
    /// partial chunks, documents or aggregate outputs are never published.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int4_sentiment_cohorts(&self, model: &ResidentInt4, request: SentimentRequest,
        config: Int4SentimentMapHostConfig, cohort_chunks: usize, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4SentimentMapCohortRun>, HostedError> {
        let (required, input_bytes) = admit(self, model, &request, config, Some(cohort_chunks))?;
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
            let planner = planner(&model, &input.value)?;
            let (prepared, admitted) = prepare(&planner, &model, &input.value, control)?;
            let expected = prepared.preflight_metadata();
            // Commit the active peak before native allocation, then retain
            // both charges while exact-sized engines for every ragged window
            // are constructed, used, cleared and physically destroyed.
            let kv = kv.commit()?;
            let scratch = scratch.commit()?;
            let result = allocate(output, || {
                let result = prepared.execute_cohorts_with_control(&admitted, &model.inner.loaded.value,
                    kernel, cohort_chunks, Int4SentimentMapCohortBudget {
                        native: Int4RunBudget::exact(prepared.planned_work()),
                        memory: native_budget(required.native),
                        max_task_workspace_bytes: required.task_workspace_bytes,
                        max_output_bytes: required.max_cohort_output_bytes,
                    }, control).map_err(map_error)?;
                if result.schema_version != 1 || result.execution != INT4_SENTIMENT_MAP_COHORT_EXECUTION
                    || result.chunk_cohorts != prepared.chunk_count().div_ceil(cohort_chunks)
                    || result.group_steps == 0 || result.group_steps > result.output.model_work.forward_positions
                    || result.projection_groups == 0 || result.projection_groups > result.output.model_work.projected_logits {
                    return Err(map_error(Int4SentimentMapError::Admission));
                }
                finish(&expected, &result.output, control)?;
                Ok(result)
            })?;
            drop(admitted);
            drop(prepared);
            drop(planner);
            drop(input);
            drop(scratch);
            drop(kv);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

fn admit(engine: &NlpEngine, model: &ResidentInt4, request: &SentimentRequest,
    config: Int4SentimentMapHostConfig, cohort_chunks: Option<usize>)
    -> Result<(Int4SentimentMapMemoryRequirement, u64), HostedError> {
    dispatch::preflight(engine, config.native.run)?;
    if !Arc::ptr_eq(engine.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
    validate_request(request, config, cohort_chunks)?;
    let required = config.requirements(model.model_identity().group_width(), request.budget,
        request.axes.len(), cohort_chunks)?;
    Ok((required, owned_input_bytes(request, config, required)?))
}
fn owned_input_bytes(request: &SentimentRequest, config: Int4SentimentMapHostConfig,
    required: Int4SentimentMapMemoryRequirement) -> Result<u64, HostedError> {
    // A caller-created String/Vec can retain more capacity than its logical
    // length. Admit that excess too, without cloning or shrinking the owner.
    let excess_source = request.document.capacity().saturating_sub(config.mapping.chunks.max_input_bytes) as u64;
    let excess_axes = (request.axes.capacity().saturating_sub(4) as u64)
        .checked_mul(std::mem::size_of::<SentimentAxis>() as u64)
        .ok_or_else(|| map_error(Int4SentimentMapError::Limits))?;
    sum(&[required.input_bytes, excess_source, excess_axes])
}
fn validate_request(request: &SentimentRequest, config: Int4SentimentMapHostConfig,
    cohort_chunks: Option<usize>) -> Result<(), HostedError> {
    if request.document.is_empty() { return Err(map_error(Int4SentimentMapError::EmptySource)); }
    if request.document.len() > config.mapping.chunks.max_input_bytes
        || !(1..=4).contains(&request.axes.len())
        || request.axes.iter().enumerate().any(|(index, axis)| request.axes[..index].contains(axis))
        || cohort_chunks.is_some_and(|chunks| chunks == 0
            || chunks.checked_mul(request.axes.len()).is_none_or(|heads| heads > 64)) {
        return Err(map_error(Int4SentimentMapError::Limits));
    }
    Ok(())
}
fn planner(model: &ResidentInt4, input: &MapInput) -> Result<Int4SentimentPlanner, HostedError> {
    Int4SentimentPlanner::for_model(model.model_identity(), input.config.options,
        input.request.budget, input.config.planning)
        .map_err(|error| map_error(error.into()))
}
fn prepare<'a>(planner: &Int4SentimentPlanner, model: &ResidentInt4, input: &'a MapInput,
    control: &mut impl DecodeStepControl)
    -> Result<(PreparedInt4SentimentMap<'a>, Vec<ExecutionIdentity>), HostedError> {
    let prepared = planner.plan_document_with_control(&input.request, input.config.mapping, control)
        .map_err(map_error)?;
    if prepared.group_width() != model.model_identity().group_width()
        || prepared.head_count() != input.request.axes.len() || prepared.chunk_count() == 0
        || prepared.chunk_count() > input.config.mapping.chunks.max_chunks
        || prepared.task_budget() != input.request.budget
        || prepared.required_context() > input.config.native.context_tokens
        || prepared.max_result_bytes() != input.config.mapping.max_result_bytes {
        return Err(map_error(Int4SentimentMapError::Admission));
    }
    check_work(prepared.planned_work(), input.config.mapping.max_model_work)?;
    let admitted = prepared.identities().map_err(map_error)?;
    if admitted.len() != prepared.chunk_count() { return Err(map_error(Int4SentimentMapError::Admission)); }
    for identity in &admitted {
        checkpoint(control)?;
        grouped_int4::check_model(identity, model.model_identity()).map_err(|error| map_error(error.into()))?;
        if identity.task_spec != planner.execution_identity().task_spec
            || identity.tokenizer_digest != planner.execution_identity().tokenizer_digest
            || identity.template_digest != planner.execution_identity().template_digest {
            return Err(map_error(Int4SentimentMapError::Admission));
        }
    }
    checkpoint(control)?;
    Ok((prepared, admitted))
}
fn check_work(work: Int4Work, cap: Int4Work) -> Result<(), HostedError> {
    if work.forward_positions > cap.forward_positions || work.projected_logits > cap.projected_logits
        || work.attention_pairs > cap.attention_pairs || !work.projections.fits(cap.projections) {
        return Err(map_error(Int4SentimentMapError::Work));
    }
    Ok(())
}
fn finish(expected: &Int4SentimentMapPreflight, document: &Int4SentimentMapRun,
    control: &mut impl DecodeStepControl) -> Result<(), HostedError> {
    // This verifies complete canonical document bytes as well as every chunk
    // and work counter. The sealed cohort executor separately checks its
    // fixed-size physical wrapper; do not allocate a redundant second tree.
    expected.verify_completed(document).map_err(map_error)?;
    checkpoint(control)
}
fn native_budget(required: Int4MemoryRequirement) -> Int4MemoryBudget {
    Int4MemoryBudget { max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
        max_scratch_payload_bytes: required.scratch_payload_bound }
}
fn map_error(error: Int4SentimentMapError) -> HostedError {
    HostedError::Int4(Int4HostError::SentimentMap(error))
}

#[cfg(test)] mod tests;
