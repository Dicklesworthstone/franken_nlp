//! Complete sampled-sentiment documents under one physical native lifetime.
//! All chunk plans and cold work/mask bounds are admitted before any forward.
//! Source evidence, paired document samples and final output stay charged.
use super::*;
use crate::{
    native_engine::strict_int4::projection::Int4ProjectionExecutor,
    tasks::sentiment::{SentimentAxis, SentimentRequest,
        sampled::{SampledSentimentLimits, SampledSentimentOptions, SampledSentimentPlanner,
            long::{PreparedSampledSentimentMap, SampledSentimentMapError,
                SampledSentimentMapLimits, SampledSentimentMapRun,
                memory::{SampledSentimentMapMemoryLimits, SampledSentimentMapMemoryRequirement}}}},
    tokenizer::untrusted::UntrustedEncoding,
};

/// Per-chunk sampling policy and whole-document work, mask and output bounds.
/// The seed is private; the existing sampling options redact it from Debug.
/// Optional prefix reuse is isolated to one chunk and axis. Prefill rows and
/// numerical team width never multiply the document's work or mask authority.
#[derive(Clone, Debug)]
pub struct SampledSentimentMapHostConfig {
    pub native: NativeLimits,
    pub planning: SampledSentimentLimits,
    pub mapping: SampledSentimentMapLimits,
    pub options: SampledSentimentOptions,
    pub preparation_reserve_bytes: u64,
    pub max_sampler_bytes: u64,
    pub mask_limits: MaskWorkLimits,
    pub max_mask_node_visits: u64,
    pub prefill: Option<Int4PrefillLimits>,
    pub reuse_prefill: bool,
}
impl SampledSentimentMapHostConfig {
    /// Pure byte-input quote, before private source/model IO or tokenization.
    /// Four axes admits the largest valid request before its axes are known.
    pub fn requirements(&self, group_width: usize, task: TaskBudget, max_axes: usize)
        -> Result<SampledSentimentMapMemoryRequirement, HostedError> {
        self.requirements_with_encoding(group_width, task, max_axes, &UntrustedEncoding::ByteFallback)
    }
    /// One complete-document encoding allowance and one optional prefill
    /// workspace coexist with all retained plans, chunk evidence and output.
    /// Numerical team scratch is separately charged by the team dispatcher.
    pub fn requirements_with_encoding(&self, group_width: usize, task: TaskBudget,
        max_axes: usize, encoding: &UntrustedEncoding)
        -> Result<SampledSentimentMapMemoryRequirement, HostedError> {
        self.native.run.validate()?;
        if self.mask_limits.max_trie_node_visits == 0
            || self.mask_limits.checkpoint_interval_nodes == 0 || self.max_mask_node_visits == 0 {
            return Err(HostedError::Limits("sampled sentiment document mask allowance"));
        }
        let mut required = SampledSentimentMapMemoryLimits {
            context_tokens: self.native.context_tokens,
            preparation_reserve_bytes: self.preparation_reserve_bytes,
            allocator_reserve_bytes: self.native.allocator_reserve_bytes,
            max_axes, max_sampler_bytes: self.max_sampler_bytes,
        }.requirements(group_width, task, self.planning, self.mapping, &self.options, encoding)
            .map_err(map_error)?;
        required.workspace_bytes = super::super::prefill::with_extra_workspace(
            required.workspace_bytes, self.prefill)?;
        sum(&[required.input_bytes, required.native.kv_bytes,
            required.workspace_bytes, required.output_reservation_bytes])?;
        Ok(required)
    }
}

struct MapInput {
    request: SentimentRequest,
    config: SampledSentimentMapHostConfig,
    encoding: UntrustedEncoding,
}
struct MapExecution {
    input: Charged<MapInput>,
    kv: Pending,
    scratch: Pending,
    output: Pending,
    required: SampledSentimentMapMemoryRequirement,
    lease: EngineLease,
}
// The native state itself cannot cross the borrowed-team boundary. Its charges
// and the complete input owner survive until dispatch confirms physical join.
struct MapCompletion {
    result: Charged<SampledSentimentMapRun>,
    input: Charged<MapInput>,
    kv: CommittedMemory,
    scratch: CommittedMemory,
    lease: EngineLease,
}
impl MapCompletion {
    fn finish(self) -> HostedOutput<SampledSentimentMapRun> {
        let Self { result, input, kv, scratch, lease } = self;
        drop(input); drop(kv); drop(scratch); drop(lease);
        GuardedOutput::new(result.value, result._memory)
    }
}

impl NlpEngine {
    /// Partition the exact source, plan every independent chunk bundle, then
    /// execute and reduce a complete document under one deadline and ledger.
    /// Every chunk keeps its scores, explanations and original source spans;
    /// the document estimator never claims single-context equivalence.
    pub fn map_int4_sampled_sentiment(&self, model: &ResidentInt4, request: SentimentRequest,
        config: SampledSentimentMapHostConfig, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<SampledSentimentMapRun>, HostedError> {
        self.map_int4_sampled_sentiment_with_encoding(model, request, config,
            UntrustedEncoding::ByteFallback, kernel, cancellation)
    }

    /// Share one bounded input-encoding allowance across every chunk probe
    /// and actual plan. Compact input is explicit and may change model output.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int4_sampled_sentiment_with_encoding(&self, model: &ResidentInt4,
        request: SentimentRequest, config: SampledSentimentMapHostConfig,
        encoding: UntrustedEncoding, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<SampledSentimentMapRun>, HostedError> {
        self.run_int4_sampled_sentiment_map(model, request, config, encoding, kernel, None, cancellation)
            .map(|(output, _)| output)
    }

    /// Keep one fixed row-projection team across every chunk, axis and sample.
    /// All workers join before the guarded complete document is returned.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int4_sampled_sentiment_with_threads(&self, model: &ResidentInt4,
        request: SentimentRequest, config: SampledSentimentMapHostConfig,
        kernel: PackedInt4Kernel, compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt4TeamResult<SampledSentimentMapRun>, HostedError> {
        self.map_int4_sampled_sentiment_with_encoding_and_threads(model, request, config,
            UntrustedEncoding::ByteFallback, kernel, compute_threads, cancellation)
    }

    /// Complete-document encoding and same-axis prefix reuse retain their
    /// ordinary semantics on one persistent numerical team. Layer-major
    /// prompt morsels are a separate schedule and cannot combine with a team.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int4_sampled_sentiment_with_encoding_and_threads(&self, model: &ResidentInt4,
        request: SentimentRequest, config: SampledSentimentMapHostConfig,
        encoding: UntrustedEncoding, kernel: PackedInt4Kernel, compute_threads: usize,
        cancellation: CancellationToken) -> Result<HostedInt4TeamResult<SampledSentimentMapRun>, HostedError> {
        let (output, compute) = self.run_int4_sampled_sentiment_map(model, request, config,
            encoding, kernel, Some(compute_threads), cancellation)?;
        Ok(HostedInt4TeamResult::new(output, compute.ok_or(HostedError::CompletionMissing)?))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_sampled_sentiment_map(&self, model: &ResidentInt4, request: SentimentRequest,
        config: SampledSentimentMapHostConfig, encoding: UntrustedEncoding,
        kernel: PackedInt4Kernel, compute_threads: Option<usize>, cancellation: CancellationToken)
        -> Result<(HostedOutput<SampledSentimentMapRun>, Option<Int4ComputeReceipt>), HostedError> {
        if compute_threads.is_some() && config.prefill.is_some() {
            return Err(HostedError::Limits("sampled sentiment document team/prefill schedule"));
        }
        match compute_threads {
            None => dispatch::preflight(self, config.native.run)?,
            Some(width) => dispatch::preflight_int4_team(self, config.native.run, width)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        validate_request(&request, &config)?;
        let required = config.requirements_with_encoding(model.model_identity().group_width(),
            request.budget, request.axes.len(), &encoding)?;
        let input_bytes = owned_input_bytes(&request, &config, required.input_bytes)?;
        let team = compute_threads.map(|width|
            int4_task_team_workspace_bytes(width, model.model_identity().group_width())).transpose()?.unwrap_or(0);
        sum(&[input_bytes, required.native.kv_bytes, required.workspace_bytes,
            required.output_reservation_bytes, team])?;
        let native = config.native;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?,
            || Ok(MapInput { request, config, encoding }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_reservation_bytes)?;
        let execution = MapExecution { input, kv, scratch, output, required, lease };
        let model = model.clone();
        let (completion, compute) = match compute_threads {
            None => dispatch::run(self, native.run, cancellation, move |control|
                execution.run(&model.inner.loaded.value, kernel, None, control).map(|done| (done, None))),
            Some(width) => dispatch::run_int4_team(self, native.run, cancellation, model, width,
                kernel, move |weights, control, team| {
                    let done = execution.run(weights, kernel, Some(&mut *team), control)?;
                    Ok((done, Some(team.stats().into())))
                }),
        }?;
        Ok((completion.finish(), compute))
    }
}

impl MapExecution {
    fn run<'weights>(self, weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel,
        projection: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> Result<MapCompletion, HostedError> {
        let Self { input, kv, scratch, output, required, lease } = self;
        checkpoint(control)?;
        let config = &input.value.config;
        let planner = SampledSentimentPlanner::for_model_with_encoding(weights.identity(),
            config.options.clone(), input.value.request.budget, config.planning, input.value.encoding.clone())
            .map_err(task_error)?;
        let vocabulary = SampledSentimentVocabulary::pinned().map_err(task_error)?;
        let prepared = planner.plan_document_with_control(&input.value.request, config.mapping, control)
            .map_err(map_error)?;
        validate_prepared(&prepared, &input.value, required, weights.identity())?;
        let budget = document_budget(&prepared, config, required.native.kv_bytes);
        let mask_charge = prepared.complete_mask_charge(&vocabulary, budget, config.max_sampler_bytes)
            .map_err(map_error)?;
        if mask_charge == 0 || mask_charge > config.max_mask_node_visits {
            return Err(HostedError::Limits("sampled sentiment complete-document mask charge"));
        }
        let expected = prepared.preflight_metadata();
        checkpoint(control)?;
        let mut engine = allocate_native(kv, scratch, || {
            let engine = StrictInt4Engine::new(weights, config.native.context_tokens, Int4MemoryBudget {
                max_kv_bytes: required.native.kv_bytes, max_rope_bytes: required.native.rope_bytes,
                max_scratch_payload_bytes: required.native.scratch_payload_bound })
                .map_err(native_error)?.with_kernel(kernel);
            match projection { Some(team) => engine.with_projection_executor(team).map_err(native_error),
                None => Ok(engine) }
        })?;
        let result = allocate(output, || {
            let result = prepared.execute_scheduled_with_control(&mut engine.value, &vocabulary,
                budget, config.max_sampler_bytes, config.prefill, config.reuse_prefill, control)
                .map_err(map_error)?;
            expected.verify_completed_with_control(&result, config.reuse_prefill, control).map_err(map_error)?;
            if engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                return Err(HostedError::Limits("sampled sentiment document native completion"));
            }
            checkpoint(control)?;
            Ok(result)
        })?;
        let ChargedNative { value, _kv, _scratch } = engine;
        drop(value); drop(prepared); drop(vocabulary); drop(planner);
        Ok(MapCompletion { result, input, kv: _kv, scratch: _scratch, lease })
    }
}

fn validate_request(request: &SentimentRequest, config: &SampledSentimentMapHostConfig)
    -> Result<(), HostedError> {
    if request.document.is_empty() || request.document.len() > config.mapping.chunks.max_input_bytes
        || !(1..=SentimentAxis::ALL.len()).contains(&request.axes.len())
        || request.axes.iter().enumerate().any(|(index, axis)| request.axes[..index].contains(axis)) {
        return Err(HostedError::Limits("sampled sentiment complete-document input"));
    }
    Ok(())
}
fn owned_input_bytes(request: &SentimentRequest, config: &SampledSentimentMapHostConfig, declared: u64)
    -> Result<u64, HostedError> {
    let source = request.document.capacity().saturating_sub(config.mapping.chunks.max_input_bytes) as u64;
    let axes = (request.axes.capacity().saturating_sub(SentimentAxis::ALL.len()) as u64)
        .checked_mul(std::mem::size_of::<SentimentAxis>() as u64)
        .ok_or(HostedError::Limits("sampled sentiment document input capacity"))?;
    sum(&[declared, source, axes])
}
fn validate_prepared(prepared: &PreparedSampledSentimentMap, input: &MapInput,
    required: SampledSentimentMapMemoryRequirement, model: &Int4ModelIdentity) -> Result<(), HostedError> {
    crate::native_engine::constrained_int4::check_model(prepared.execution_identity(), model)
        .map_err(SampledSentimentError::from).map_err(task_error)?;
    let config = &input.config;
    let samples = prepared.chunk_count().checked_mul(input.request.axes.len())
        .and_then(|count| count.checked_mul(config.options.samples))
        .ok_or(HostedError::Limits("sampled sentiment document sample count"))?;
    if prepared.group_width() != model.group_width() || prepared.chunk_count() == 0
        || prepared.chunk_count() > config.mapping.chunks.max_chunks || prepared.sample_count() != samples
        || prepared.task_budget() != input.request.budget || prepared.required_context() == 0
        || prepared.required_context() > config.native.context_tokens
        || prepared.max_result_bytes() != config.mapping.max_result_bytes
        || prepared.max_result_bytes() > required.max_output_bytes
        || prepared.max_output_tokens() > required.max_output_tokens
        || prepared.preparation_requirement_bytes() == 0
        || prepared.preparation_requirement_bytes() > required.input_bytes
        || !fits(prepared.planned_work(), config.mapping.max_model_work) {
        return Err(HostedError::Limits("sampled sentiment complete-document preparation"));
    }
    Ok(())
}
fn document_budget(prepared: &PreparedSampledSentimentMap, config: &SampledSentimentMapHostConfig,
    kv_bytes: u64) -> Int4JsonBudget {
    let work = prepared.planned_work();
    Int4JsonBudget { native: Int4RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: kv_bytes, max_total_mask_node_visits: config.max_mask_node_visits,
        mask_limits: config.mask_limits } }
}
fn map_error(error: SampledSentimentMapError) -> HostedError {
    HostedError::Int4(Int4HostError::SampledSentimentMap(error))
}

#[cfg(test)] mod tests;
