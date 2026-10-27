//! Complete original text populations on one charged durable INT4 invocation.
use super::*;
use crate::{
    jobs::{JobWork, runner::generation::grouped_int4::{Int4GenerationAdmission,
        Int4GenerationBatchAdmission, Int4GenerationJobPlanner, Int4GenerationJobProcessor}},
    native_engine::generation::grouped_int4::Int4GenerationBudget,
};

struct GenerationInput<R> {
    preparation: Option<(Int4GenerationJobPlanner, SourceJobRequest)>,
    reader: R,
}

#[derive(Clone, Copy)]
struct GenerationEnvelope {
    context_tokens: usize,
    kv_bytes: u64,
    native: Int4GenerationBudget,
    lifetime: JobWork,
    result_bytes: u64,
    output_tokens: u64,
}

impl NlpEngine {
    /// Start or resume generation/chat over the COMPLETE original NDJSON
    /// population. The model-bound factory fixes task, seed, sampling options,
    /// context and lifetime budgets before input or retained job access.
    ///
    /// The shared runner authenticates that recipe and the original population
    /// before tail repair or native forwards. Original item IDs/sample indices
    /// retain their addressed RNG; original ordinals retain delivery sequence.
    /// Failed attempts consume lifetime work. Only complete validated native
    /// results enter the protected spool, and their guards survive sync and
    /// journal acknowledgement. This route uses the existing no-thinking,
    /// no-tools template and does not expose provisional token events.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_generation<R>(&self, model: &ResidentInt4, planner: Int4GenerationJobPlanner,
        request: SourceJobRequest, limits: JobHostLimits, kernel: PackedInt4Kernel,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) {
            return Err(HostedError::ResourceDomain.into());
        }
        constrained_int4::check_model(planner.execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        if planner.group_width() != model.model_identity().group_width()
            || !matches!(planner.execution_identity().task_spec.as_str(), "generate-v1" | "chat-v1") {
            return Err(HostedError::ModelIdentity.into());
        }
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let envelope = GenerationEnvelope {
            context_tokens: planner.context_tokens(), kv_bytes: planner.native_limits().max_kv_bytes,
            native: planner.native_limits(), lifetime: request.limits.max_work,
            result_bytes: planner.max_result_bytes(), output_tokens: planner.output_token_ceiling(),
        };
        let required = requirements(planner.group_width(), planner.task_budget(), envelope,
            request.limits, limits)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(GenerationInput { preparation: Some((planner, request)), reader }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let model = model.clone();
        let completed = dispatch::run(self, limits.native.run, cancellation, move |control| {
            // Capture the whole owner even while queued. Its storage drops
            // before the charge, including cancellation before the first poll.
            let mut input = input;
            checkpoint(control)?;
            let (planner, request) = input.value.preparation.take()
                .ok_or(HostedError::CompletionMissing)?;
            let population = match JobPopulation::read_ndjson(&mut input.value.reader,
                &request.key, request.job_id, request.limits, limits.transport, control) {
                Ok(population) => population,
                Err(error) => return Ok(Err(JobRunError::Storage(error))),
            };
            checkpoint(control)?;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, envelope.context_tokens,
                    Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                        max_rope_bytes: required.native.rope_bytes,
                        max_scratch_payload_bytes: required.native.scratch_payload_bound })
                    .map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let admission = GenerationAdmission { lease: &lease, model: model.model_identity(),
                base: planner.execution_identity(), envelope };
            let result = (|| {
                let processor = Int4GenerationJobProcessor::new(&planner, &mut engine.value, admission)
                    .map_err(JobRunError::Processor)?;
                run_population(request, &population, processor, control)
            })();
            drop(engine);
            drop(population);
            drop(planner);
            drop(input);
            drop(lease);
            Ok(result)
        })?;
        // Runtime cancellation/drain failure takes precedence over a job return;
        // authenticated resume reconciles any lost durable acknowledgement.
        completed.map_err(HostedJobError::Job)
    }
}

fn requirements(width: usize, task: TaskBudget, envelope: GenerationEnvelope,
    job: JobLimits, limits: JobHostLimits) -> Result<Requirements, HostedJobError> {
    task.validate().map_err(|_| HostedError::Limits("INT4 generation job task budget"))?;
    let input_bytes = limits.required_buffer_bytes(job)?;
    let native = Int4MemoryRequirement::for_context(limits.native.context_tokens, width).map_err(native_error)?;
    let run = envelope.native.native;
    if envelope.context_tokens != limits.native.context_tokens
        || envelope.kv_bytes != native.kv_bytes || envelope.native.max_kv_bytes != native.kv_bytes
        || task.max_kv_bytes != native.kv_bytes || limits.native.allocator_reserve_bytes == 0
        || envelope.native.max_sampler_bytes == 0 || envelope.result_bytes == 0
        || envelope.result_bytes != task.max_output_bytes || envelope.result_bytes > job.max_result_bytes as u64
        || envelope.output_tokens == 0 || envelope.output_tokens > u64::from(task.max_output_tokens)
        || envelope.lifetime != job.max_work || job.max_work.model.projected_logits == 0
        || run.max_forward_positions == 0 || run.max_attention_pairs == 0
        || run.max_projection_work.dot_products == 0 || run.max_projection_work.multiply_accumulates == 0
        || run.max_forward_positions > job.max_work.model.forward_positions
        || run.max_attention_pairs > job.max_work.model.attention_pairs
        || run.max_projection_work.dot_products > job.max_work.model.projections.dot_products
        || run.max_projection_work.multiply_accumulates > job.max_work.model.projections.multiply_accumulates {
        return Err(HostedError::Limits("INT4 generation job complete resident and lifetime envelope").into());
    }
    // Sampler vectors coexist with the native workspace; explicit preparation
    // owns the tokenizer/options and per-item compilation graph separately.
    let workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
        limits.native.allocator_reserve_bytes, envelope.native.max_sampler_bytes])?;
    Ok(Requirements { native, workspace_bytes, input_bytes })
}

struct GenerationAdmission<'a> {
    lease: &'a EngineLease,
    model: &'a Int4ModelIdentity,
    base: &'a ExecutionIdentity,
    envelope: GenerationEnvelope,
}
impl Int4GenerationBatchAdmission for GenerationAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int4GenerationAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        constrained_int4::check_model(request.identity, self.model)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        check_request(self.model.group_width(), self.base, self.envelope, &request)?;
        let guard = output_claim(self.lease, request.max_result_bytes, request.max_output_tokens)
            .map_err(admission_failure)?;
        Ok((request.identity.clone(), guard))
    }
}

fn check_request(width: usize, base: &ExecutionIdentity, envelope: GenerationEnvelope,
    request: &Int4GenerationAdmission<'_>) -> Result<(), BatchItemFailure> {
    constrained_int4::check_profile(request.identity, width)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    if !matches!(base.task_spec.as_str(), "generate-v1" | "chat-v1")
        || request.identity.task_spec != base.task_spec
        || request.identity.tokenizer_digest != base.tokenizer_digest
        || request.identity.template_digest != base.template_digest
        || request.identity.sampler_version != base.sampler_version
        || request.identity.thinking_mode != base.thinking_mode || request.identity.tool_mode != base.tool_mode
        || request.kv_reservation_bytes != envelope.kv_bytes
        || request.sampler_bytes == 0 || request.sampler_bytes > envelope.native.max_sampler_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let work = request.model_work;
    let run = envelope.native.native;
    if work.forward_positions == 0 || work.forward_positions > envelope.context_tokens as u64
        || work.projected_logits == 0 || work.projected_logits % NANBEIGE_VOCAB_SIZE as u64 != 0
        || work.projected_logits / NANBEIGE_VOCAB_SIZE as u64 > work.forward_positions
        || work.forward_positions > run.max_forward_positions || work.attention_pairs > run.max_attention_pairs
        || work.projections.dot_products > run.max_projection_work.dot_products
        || work.projections.multiply_accumulates > run.max_projection_work.multiply_accumulates
        || !JobWork::from_int4(work, 0).fits(envelope.lifetime) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    let positions = usize::try_from(work.forward_positions)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    let rows = usize::try_from(work.projected_logits)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    // Each item is one empty-cache sequence. Recompute with its actual INT4
    // partition, never the historical INT8 formula behind the job wire fields.
    if Int4Work::for_sequence(width, 0, positions, rows).ok() != Some(work) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    if request.max_result_bytes == 0 || request.max_result_bytes != envelope.result_bytes
        || request.max_output_tokens == 0 || request.max_output_tokens != envelope.output_tokens {
        return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
