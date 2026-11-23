//! One admitted invocation for every chunk of a complete INT8 document.
use super::*;
use crate::{
    native_engine::{
        constrained_int8,
        decode::DecodeStepControl,
    },
    tasks::{
        ir::PlanContext,
        sentiment::{
            SentimentAxis, SentimentLimits, SentimentPlanner, SentimentRequest,
            quantized::long::{
                Int8SentimentMapError, Int8SentimentMapLimits, Int8SentimentMapRun,
                memory::{Int8SentimentMapMemoryLimits, Int8SentimentMapMemoryRequirement},
            },
        },
    },
    tokenizer::untrusted::UntrustedEncoding,
};

/// One immutable task context, per-chunk planning ceilings and nonrenewable
/// whole-document work/output allowances. The supplied planner fixes the
/// scoring policy and byte-preserving input encoder.
#[derive(Clone, Debug)]
pub struct Int8SentimentMapHostConfig {
    pub identity: ExecutionIdentity,
    pub native: NativeLimits,
    pub planning: SentimentLimits,
    pub mapping: Int8SentimentMapLimits,
    pub preparation_reserve_bytes: u64,
}
impl Int8SentimentMapHostConfig {
    /// Pure checked memory quote, usable before input, tokenizer or model IO.
    /// Four axes prices any supported request before its private JSON is read.
    pub fn requirements(
        &self,
        task: TaskBudget,
        max_axes: usize,
        encoding: &UntrustedEncoding,
    ) -> Result<Int8SentimentMapMemoryRequirement, HostedError> {
        self.native.run.validate()?;
        Int8SentimentMapMemoryLimits {
            context_tokens: self.native.context_tokens,
            preparation_reserve_bytes: self.preparation_reserve_bytes,
            allocator_reserve_bytes: self.native.allocator_reserve_bytes,
            max_axes,
        }
        .requirements(task, self.planning, self.mapping)
        .and_then(|need| need.with_input_encoding(encoding))
        .map_err(HostedError::SentimentMap)
    }
}

// Drop order keeps all retained source/planner allocations ahead of the
// Charged wrapper's memory guard on success, cancellation and unwinding.
struct MapInput {
    request: SentimentRequest,
    planner: Arc<SentimentPlanner>,
    config: Int8SentimentMapHostConfig,
}

impl NlpEngine {
    /// Plan every original source chunk before the first native forward, then
    /// reuse one KV allocation for all chunks and independent dimensions. The
    /// caller must admit request/planner storage before transferring ownership;
    /// this method prices the complete retained ownership before dispatch.
    pub fn map_int8_sentiment(
        &self,
        model: &ResidentInt8,
        request: SentimentRequest,
        planner: Arc<SentimentPlanner>,
        config: Int8SentimentMapHostConfig,
        cancellation: CancellationToken,
    ) -> Result<HostedOutput<Int8SentimentMapRun>, HostedError> {
        self.map_int8_sentiment_scheduled(model, request, planner, config, None, cancellation)
    }

    /// Explicit packed prompt scheduling changes only physical execution.
    /// Its reusable scratch is charged once across the complete document.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int8_sentiment_layer_major(
        &self,
        model: &ResidentInt8,
        request: SentimentRequest,
        planner: Arc<SentimentPlanner>,
        config: Int8SentimentMapHostConfig,
        prefill: Int8PrefillLimits,
        cancellation: CancellationToken,
    ) -> Result<HostedOutput<Int8SentimentMapRun>, HostedError> {
        self.map_int8_sentiment_scheduled(
            model,
            request,
            planner,
            config,
            Some(prefill),
            cancellation,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn map_int8_sentiment_scheduled(
        &self,
        model: &ResidentInt8,
        request: SentimentRequest,
        planner: Arc<SentimentPlanner>,
        config: Int8SentimentMapHostConfig,
        prefill: Option<Int8PrefillLimits>,
        cancellation: CancellationToken,
    ) -> Result<HostedOutput<Int8SentimentMapRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.identity)?;
        validate_request(&request, &config)?;
        constrained_int8::check_profile(&config.identity)
            .map_err(|_| HostedError::ModelIdentity)?;
        if config.identity.task_spec != "sentiment-v1"
            || config.identity.tokenizer_digest != planner.tokenizer_digest()
            || config.identity.template_digest != *planner.template_digest()
        {
            return Err(HostedError::ModelIdentity);
        }
        let required = config.requirements(
            request.budget,
            request.axes.len(),
            planner.document_encoding(),
        )?;
        let input_bytes = owned_input_bytes(&request, &config, required)?;
        let scratch_bytes =
            super::super::scored::scoring_scratch(required.workspace_bytes, prefill)?;
        let native = config.native;
        let lease = self.resources().acquire_lease();
        let input = allocate(
            Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?,
            || {
                Ok(MapInput {
                    request,
                    planner,
                    config,
                })
            },
        )?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
        let output = Pending::reserve(
            &lease,
            MemoryClass::JobBuffers,
            required.output_reservation_bytes,
        )?;
        let model = model.clone();
        dispatch::run(self, native.run, cancellation, move |control| {
            let input = input;
            checkpoint(control)?;
            let config = &input.value.config;
            let context = PlanContext::new(&config.identity, input.value.request.budget)
                .map_err(|_| HostedError::Limits("sentiment document context"))?;
            let prepared = input
                .value
                .planner
                .plan_int8_document_with_control(
                    &input.value.request,
                    &context,
                    config.planning,
                    config.mapping,
                    control,
                )
                .map_err(HostedError::SentimentMap)?;
            if prepared.chunk_count() == 0
                || prepared.chunk_count() > config.mapping.chunks.max_chunks
                || prepared.head_count() != input.value.request.axes.len()
                || prepared.task_budget() != input.value.request.budget
                || prepared.required_context() > native.context_tokens
                || prepared.max_result_bytes() != config.mapping.max_result_bytes
            {
                return Err(HostedError::SentimentMap(Int8SentimentMapError::Admission));
            }
            let expected = prepared.preflight_metadata();
            let admitted = prepared.identities().map_err(HostedError::SentimentMap)?;
            if admitted.len() != prepared.chunk_count() {
                return Err(HostedError::SentimentMap(Int8SentimentMapError::Admission));
            }
            for identity in &admitted {
                checkpoint(control)?;
                check_model_identity(model.artifact_identity(), identity)?;
            }
            let mut engine = allocate_native(kv, scratch, || {
                model
                    .inner
                    .loaded
                    .value
                    .engine(native.context_tokens, memory_budget(required.native))
                    .map_err(HostedError::Model)
            })?;
            let result = allocate(output, || {
                let budget = Int8ScoringBudget {
                    native: Int8RunBudget::exact(prepared.planned_work()),
                    max_kv_bytes: required.native.kv_bytes,
                };
                let run = match prefill {
                    None => {
                        prepared.execute_with_control(&admitted, &mut engine.value, budget, control)
                    }
                    Some(prefill) => prepared.execute_layer_major_with_control(
                        &admitted,
                        &mut engine.value,
                        budget,
                        prefill,
                        control,
                    ),
                }
                .map_err(HostedError::SentimentMap)?;
                expected
                    .verify_completed_with_control(&run, control)
                    .map_err(HostedError::SentimentMap)?;
                if engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(HostedError::SentimentMap(Int8SentimentMapError::Admission));
                }
                checkpoint(control)?;
                Ok(run)
            })?;
            drop(engine);
            drop(admitted);
            drop(prepared);
            drop(input);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

fn validate_request(
    request: &SentimentRequest,
    config: &Int8SentimentMapHostConfig,
) -> Result<(), HostedError> {
    if request.document.is_empty() {
        return Err(HostedError::SentimentMap(
            Int8SentimentMapError::EmptySource,
        ));
    }
    if request.document.len() > config.mapping.chunks.max_input_bytes
        || !(1..=4).contains(&request.axes.len())
        || request
            .axes
            .iter()
            .enumerate()
            .any(|(index, axis)| request.axes[..index].contains(axis))
    {
        return Err(HostedError::SentimentMap(Int8SentimentMapError::Limits));
    }
    Ok(())
}
fn owned_input_bytes(
    request: &SentimentRequest,
    config: &Int8SentimentMapHostConfig,
    required: Int8SentimentMapMemoryRequirement,
) -> Result<u64, HostedError> {
    let source = request
        .document
        .capacity()
        .saturating_sub(config.mapping.chunks.max_input_bytes) as u64;
    let axes = (request.axes.capacity().saturating_sub(4) as u64)
        .checked_mul(std::mem::size_of::<SentimentAxis>() as u64)
        .ok_or(HostedError::Limits(
            "sentiment document allocation arithmetic",
        ))?;
    sum(&[required.input_bytes, source, axes])
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), HostedError> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(HostedError::SentimentMap(Int8SentimentMapError::Sentiment(
            Int8SentimentError::Cancelled(cause),
        ))),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests;
