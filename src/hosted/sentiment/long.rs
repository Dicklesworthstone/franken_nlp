//! One admitted invocation for every chunk of a complete INT8 document.
use super::*;
use crate::{
    native_engine::{
        constrained_int8,
        decode::DecodeStepControl,
        simd::Int8Kernel,
        strict_int8::StrictInt8Engine,
    },
    tasks::{
        ir::PlanContext,
        sentiment::{
            SentimentAxis, SentimentLimits, SentimentPlanner, SentimentRequest,
            quantized::long::{
                Int8SentimentMapError, Int8SentimentMapLimits, Int8SentimentMapRun,
                PreparedInt8SentimentMap,
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

// The dispatcher owns every reservation outside the scoped callback until
// all workers have physically joined, including on cancellation and unwind.
// Keep native claims disjoint from the input borrowed by the prepared map.
struct MapNativeClaims {
    kv: Option<Pending>,
    scratch: Option<Pending>,
    _kv: Option<CommittedMemory>,
    _scratch: Option<CommittedMemory>,
}
impl MapNativeClaims {
    fn commit(&mut self) -> Result<(), HostedError> {
        self._kv = Some(self.kv.take()
            .ok_or(HostedError::Limits("INT8 sentiment map KV consumed"))?.commit()?);
        self._scratch = Some(self.scratch.take()
            .ok_or(HostedError::Limits("INT8 sentiment map scratch consumed"))?.commit()?);
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
    fn finish(&mut self, result: Int8SentimentMapRun, compute: Int8ComputeReceipt)
        -> Result<HostedInt8TeamResult<Int8SentimentMapRun>, HostedError> {
        let pending = self.output.take()
            .ok_or(HostedError::Limits("INT8 sentiment map output consumed"))?;
        match pending.commit() {
            Ok(memory) => Ok(HostedInt8TeamResult::from_guarded(
                GuardedOutput::new(result, memory), compute)),
            Err(error) => { drop(result); Err(error) }
        }
    }
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
        self.map_int8_sentiment_scheduled(model, request, planner, config, Some(prefill), cancellation)
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
        validate_input(&request, &planner, &config, model.artifact_identity())?;
        let required = scheduled_requirements(&config, request.budget, request.axes.len(),
            planner.document_encoding(), prefill, None)?;
        let input_bytes = owned_input_bytes(&request, &config, required)?;
        let native = config.native;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?,
            || Ok(MapInput { request, planner, config }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_reservation_bytes)?;
        let model = model.clone();
        dispatch::run(self, native.run, cancellation, move |control| {
            let input = input;
            let (prepared, admitted) = prepare_map(&input.value, model.artifact_identity(), control)?;
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(native.context_tokens, memory_budget(required.native)).map_err(HostedError::Model))?;
            let result = allocate(output, || execute_prepared(&prepared, &admitted, &mut engine.value,
                required.native.kv_bytes, prefill, control))?;
            drop(engine);
            drop(admitted);
            drop(prepared);
            drop(input);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }

    /// One fixed numerical team spans every source chunk, independent axis
    /// and candidate continuation. The complete source/result bindings and
    /// whole-document allowances remain identical to serial execution.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int8_sentiment_with_threads(&self, model: &ResidentInt8,
        request: SentimentRequest, planner: Arc<SentimentPlanner>,
        config: Int8SentimentMapHostConfig, kernel: Int8Kernel, compute_threads: usize,
        cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8SentimentMapRun>, HostedError> {
        self.map_int8_sentiment_team(model, request, planner, config, kernel,
            compute_threads, None, cancellation)
    }

    /// Reuse the same persistent team for bounded layer-major prompt morsels
    /// and scalar candidate continuations. Coordinator and worker buffers are
    /// independently priced before dispatch and retained through physical join.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int8_sentiment_layer_major_with_threads(&self, model: &ResidentInt8,
        request: SentimentRequest, planner: Arc<SentimentPlanner>,
        config: Int8SentimentMapHostConfig, kernel: Int8Kernel, compute_threads: usize,
        prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8SentimentMapRun>, HostedError> {
        self.map_int8_sentiment_team(model, request, planner, config, kernel,
            compute_threads, Some(prefill), cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn map_int8_sentiment_team(&self, model: &ResidentInt8, request: SentimentRequest,
        planner: Arc<SentimentPlanner>, config: Int8SentimentMapHostConfig,
        kernel: Int8Kernel, compute_threads: usize, prefill: Option<Int8PrefillLimits>,
        cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8SentimentMapRun>, HostedError> {
        dispatch::preflight_int8_team(self, config.native.run, compute_threads)?;
        self.check_resident_domain(model)?;
        validate_input(&request, &planner, &config, model.artifact_identity())?;
        let required = scheduled_requirements(&config, request.budget, request.axes.len(),
            planner.document_encoding(), prefill, Some(compute_threads))?;
        let input_bytes = owned_input_bytes(&request, &config, required)?;
        let native = config.native;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?,
            || Ok(MapInput { request, planner, config }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_reservation_bytes)?;
        let admission = MapTeamAdmission { input, native: MapNativeClaims {
            kv: Some(kv), scratch: Some(scratch), _kv: None, _scratch: None },
            output: Some(output), _lease: lease };
        let max_batch_rows = prefill.map_or(1, |limits| limits.max_batch_rows);
        dispatch::run_int8_prefill_team(self, native.run, cancellation, model.clone(),
            compute_threads, kernel, max_batch_rows, admission, move |model, admission, control, team| {
                let (prepared, admitted) = prepare_map(&admission.input.value, model.artifact_identity(), control)?;
                let mut engine = model.engine(native.context_tokens, memory_budget(required.native))
                    .map_err(HostedError::Model)?.with_kernel(kernel)
                    .with_projection_executor(team).map_err(HostedError::Native)?;
                admission.native.commit()?;
                let result = execute_prepared(&prepared, &admitted, &mut engine,
                    required.native.kv_bytes, prefill, control)?;
                drop(engine);
                drop(admitted);
                drop(prepared);
                let compute = team.stats().into();
                admission.finish(result, compute)
            })
    }
}

fn scheduled_requirements(config: &Int8SentimentMapHostConfig, task: TaskBudget,
    axes: usize, encoding: &UntrustedEncoding, prefill: Option<Int8PrefillLimits>, width: Option<usize>)
    -> Result<Int8SentimentMapMemoryRequirement, HostedError> {
    let mut required = config.requirements(task, axes, encoding)?;
    required.workspace_bytes = super::super::scored::scoring_scratch(required.workspace_bytes, prefill)?;
    let worker_bytes = width.map(|width| dispatch::int8_prefill_team_workspace_bytes(
        width, prefill.map_or(1, |limits| limits.max_batch_rows))).transpose()?.unwrap_or(0);
    sum(&[required.native.kv_bytes, required.workspace_bytes, required.input_bytes,
        required.output_reservation_bytes, worker_bytes])?;
    Ok(required)
}

fn validate_input(request: &SentimentRequest, planner: &SentimentPlanner,
    config: &Int8SentimentMapHostConfig, model: &ArtifactIdentity) -> Result<(), HostedError> {
    check_model_identity(model, &config.identity)?;
    validate_request(request, config)?;
    constrained_int8::check_profile(&config.identity).map_err(|_| HostedError::ModelIdentity)?;
    if config.identity.task_spec != "sentiment-v1"
        || config.identity.tokenizer_digest != planner.tokenizer_digest()
        || config.identity.template_digest != *planner.template_digest() {
        return Err(HostedError::ModelIdentity);
    }
    Ok(())
}

fn prepare_map<'a>(input: &'a MapInput, model: &ArtifactIdentity,
    control: &mut impl DecodeStepControl)
    -> Result<(PreparedInt8SentimentMap<'a>, Vec<ExecutionIdentity>), HostedError> {
    checkpoint(control)?;
    let config = &input.config;
    let context = PlanContext::new(&config.identity, input.request.budget)
        .map_err(|_| HostedError::Limits("sentiment document context"))?;
    let prepared = input.planner.plan_int8_document_with_control(&input.request, &context,
        config.planning, config.mapping, control).map_err(HostedError::SentimentMap)?;
    if prepared.chunk_count() == 0 || prepared.chunk_count() > config.mapping.chunks.max_chunks
        || prepared.head_count() != input.request.axes.len() || prepared.task_budget() != input.request.budget
        || prepared.required_context() > config.native.context_tokens
        || prepared.max_result_bytes() != config.mapping.max_result_bytes {
        return Err(HostedError::SentimentMap(Int8SentimentMapError::Admission));
    }
    let admitted = prepared.identities().map_err(HostedError::SentimentMap)?;
    if admitted.len() != prepared.chunk_count() {
        return Err(HostedError::SentimentMap(Int8SentimentMapError::Admission));
    }
    for identity in &admitted {
        checkpoint(control)?;
        check_model_identity(model, identity)?;
    }
    checkpoint(control)?;
    Ok((prepared, admitted))
}

fn execute_prepared(prepared: &PreparedInt8SentimentMap<'_>, admitted: &[ExecutionIdentity],
    engine: &mut StrictInt8Engine<'_>, kv_bytes: u64, prefill: Option<Int8PrefillLimits>,
    control: &mut impl DecodeStepControl) -> Result<Int8SentimentMapRun, HostedError> {
    let expected = prepared.preflight_metadata();
    let budget = Int8ScoringBudget { native: Int8RunBudget::exact(prepared.planned_work()), max_kv_bytes: kv_bytes };
    let run = match prefill {
        None => prepared.execute_with_control(admitted, engine, budget, control),
        Some(prefill) => prepared.execute_layer_major_with_control(admitted, engine, budget, prefill, control),
    }.map_err(HostedError::SentimentMap)?;
    expected.verify_completed_with_control(&run, control).map_err(HostedError::SentimentMap)?;
    if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
        return Err(HostedError::SentimentMap(Int8SentimentMapError::Admission));
    }
    checkpoint(control)?;
    Ok(run)
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
