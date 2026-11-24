//! One physical INT8 team owner shared by source maps and exact reducers.
use super::*;
use crate::{
    native_engine::{decode::DecodeStepControl, strict_int8::StrictInt8Engine},
    tasks::source_planning::quantized::{Int8SourceError, long::PreparedInt8SourceMap},
};

/// This whole aggregate belongs to the dispatcher, outside its scoped callback.
/// Every source/plan borrow ends before charges can drop, including unwind and
/// cancellation. Native allocations are local to the callback and drop first.
struct Admission {
    input: Charged<SourceMapInput>,
    kv: Option<Pending>, scratch: Option<Pending>,
    output: Option<Pending>, reduction: Option<Pending>,
    _kv: Option<CommittedMemory>, _scratch: Option<CommittedMemory>,
    lease: EngineLease,
}

pub(super) fn scratch_bytes(limits: NativeLimits, native: Int8MemoryRequirement,
    prefill: Option<Int8PrefillLimits>) -> Result<u64, HostedError> {
    let extra = prefill.map(Int8PrefillLimits::validate).transpose()
        .map_err(HostedError::Native)?.unwrap_or(0);
    sum(&[native.rope_bytes, native.scratch_payload_bound, limits.allocator_reserve_bytes, extra])
}

fn workspace(limits: NativeLimits, native: Int8MemoryRequirement, width: usize,
    prefill: Option<Int8PrefillLimits>) -> Result<(u64, usize), HostedError> {
    let scratch = scratch_bytes(limits, native, prefill)?;
    let rows = prefill.map_or(1, |limits| limits.max_batch_rows);
    // Fixed child buffers coexist with all coordinator prompt/native storage.
    // The dispatcher separately reserves this exact worker geometry.
    sum(&[scratch, dispatch::int8_prefill_team_workspace_bytes(width, rows)?])?;
    Ok((scratch, rows))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run<T, F>(host: &NlpEngine, model: &ResidentInt8, source: String,
    planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>, config: SourceMapConfig,
    kernel: Int8Kernel, width: usize, prefill: Option<Int8PrefillLimits>,
    max_result_bytes: u64, retain_tokens: bool, cancellation: CancellationToken, execute: F)
    -> Result<HostedInt8TeamResult<T>, HostedError>
where T: Send + 'static,
    F: FnOnce(PreparedInt8SourceMap<'_>, &[ExecutionIdentity], &mut StrictInt8Engine<'_>,
        &ExtractionVocabulary, Option<Int8PrefillLimits>, &mut RunControl) -> Result<T, HostedError>
        + Send + 'static {
    dispatch::preflight_int8_team(host, config.native.run, width)?;
    host.check_resident_domain(model)?;
    check_model_identity(model.artifact_identity(), &config.identity)?;
    let required = requirements(config.native)?;
    validate(&config, source.len(), required.kv_bytes)?;
    check_assets(&config.identity, planner.tokenizer_digest(), *planner.template_digest())?;
    let (scratch_bytes, rows) = workspace(config.native, required, width, prefill)?;
    if max_result_bytes == 0 { return Err(HostedError::Limits("source map complete output")); }
    let run = config.native.run;
    let lease = host.resources().acquire_lease();
    let bytes = input_bytes(&config, source.capacity(), &planner)?;
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
    let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
    let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
        || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
    let admission = Admission { input, kv: Some(kv), scratch: Some(scratch), output: None,
        reduction: None, _kv: None, _scratch: None, lease };
    let artifact = model.artifact_identity().clone();
    dispatch::run_int8_prefill_team(host, run, cancellation, model.clone(), width, kernel, rows,
        admission, move |model, admission, control, team| {
            let input = &admission.input.value;
            let config = &input.config;
            let (prepared, admitted) = prepare(input, &artifact, control)?;
            let count = prepared.chunk_count();
            admission.reduction = Some(Pending::reserve(&admission.lease, MemoryClass::JobBuffers,
                reduction_bytes(config, count)?)?);
            let tokens = if retain_tokens {
                (count as u64).checked_mul(u64::from(config.budget.max_output_tokens))
                    .ok_or(HostedError::Limits("source map complete token allowance"))?
            } else { 0 };
            admission.output = Some(output_claim(&admission.lease, max_result_bytes, tokens)?);
            let mut engine = model.engine(config.native.context_tokens, memory_budget(required))
                .map_err(HostedError::Model)?.with_kernel(kernel)
                .with_projection_executor(team).map_err(HostedError::Native)?;
            admission._kv = Some(admission.kv.take()
                .ok_or(HostedError::Limits("source map KV admission consumed"))?.commit()?);
            admission._scratch = Some(admission.scratch.take()
                .ok_or(HostedError::Limits("source map scratch admission consumed"))?.commit()?);
            let result = execute(prepared, &admitted, &mut engine, &input.vocabulary, prefill, control)?;
            poll(control)?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                return Err(HostedError::Source(Int8SourceError::InvalidResult));
            }
            drop(engine);
            drop(admitted);
            let compute = team.stats().into();
            let pending = admission.output.take()
                .ok_or(HostedError::Limits("source map output admission consumed"))?;
            let memory = match pending.commit() {
                Ok(memory) => memory,
                Err(error) => { drop(result); return Err(error); }
            };
            Ok(HostedInt8TeamResult::from_guarded(GuardedOutput::new(result, memory), compute))
        })
}

fn prepare<'s>(input: &'s SourceMapInput, artifact: &ArtifactIdentity, control: &mut RunControl)
    -> Result<(PreparedInt8SourceMap<'s>, Vec<ExecutionIdentity>), HostedError> {
    poll(control)?;
    let config = &input.config;
    let context = PlanContext::new(&config.identity, config.budget)
        .map_err(|_| HostedError::Limits("source map task context"))?;
    let prepared = input.planner.plan_int8_map_with_control(&input.source, &config.task,
        config.budget, &context, config.planning, config.mapping, control).map_err(HostedError::SourceMap)?;
    let mut admitted = Vec::new();
    admitted.try_reserve_exact(prepared.chunk_count())
        .map_err(|_| HostedError::SourceMap(Int8SourceMapError::Allocation))?;
    for identity in prepared.execution_identities() {
        poll(control)?;
        check_model_identity(artifact, identity)?;
        constrained_int8::check_profile(identity).map_err(|_| HostedError::ModelIdentity)?;
        check_assets(identity, input.planner.tokenizer_digest(), *input.planner.template_digest())?;
        admitted.push(identity.clone());
    }
    poll(control)?;
    Ok((prepared, admitted))
}

fn poll(control: &mut impl DecodeStepControl) -> Result<(), HostedError> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(HostedError::Source(Int8SourceError::Cancelled(cause))),
        None => Ok(()),
    }
}

#[cfg(test)] mod tests;
