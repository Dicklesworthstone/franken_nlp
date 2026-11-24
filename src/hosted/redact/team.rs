//! Physical owner for one complete INT8 redaction and both detector stages.
use super::*;
use crate::native_engine::{decode::DecodeStepControl, strict_int8::StrictInt8Error};

/// The dispatcher retains this aggregate outside the scoped callback. Neither
/// cancellation, early failure nor unwinding releases private input or native
/// claims while a projection worker can still be physically alive.
struct Admission<D> {
    input: Charged<RedactInput<D>>,
    _temporary: Pending,
    kv: Option<Pending>,
    scratch: Option<Pending>,
    output: Option<Pending>,
    _kv: Option<CommittedMemory>,
    _scratch: Option<CommittedMemory>,
    _lease: EngineLease,
}

pub(super) fn scratch_bytes(native: NativeLimits, required: Int8MemoryRequirement,
    prefill: Option<Int8PrefillLimits>) -> Result<u64, HostedError> {
    let prompt = prefill.map(Int8PrefillLimits::validate).transpose()
        .map_err(HostedError::Native)?.unwrap_or(0);
    sum(&[required.rope_bytes, required.scratch_payload_bound, native.allocator_reserve_bytes, prompt])
}

fn workspace(native: NativeLimits, required: Int8MemoryRequirement, width: usize,
    prefill: Option<Int8PrefillLimits>) -> Result<(u64, usize), HostedError> {
    let scratch = scratch_bytes(native, required, prefill)?;
    let rows = prefill.map_or(1, |limits| limits.max_batch_rows);
    // The coordinator prompt workspace and fixed child shards coexist. The
    // dispatcher charges the latter separately for this exact geometry.
    sum(&[scratch, dispatch::int8_prefill_team_workspace_bytes(width, rows)?])?;
    Ok((scratch, rows))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run<D, T, F>(host: &NlpEngine, model: &ResidentInt8, source: String,
    planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
    config: RedactConfig<D>, pseudonyms: Option<RedactionPseudonyms>, kernel: Int8Kernel,
    width: usize, prefill: Option<Int8PrefillLimits>, max_result_bytes: u64, temporary_bytes: u64,
    cancellation: CancellationToken,
    validate: fn(&RedactConfig<D>, usize, Sha256Digest, Sha256Digest, u64) -> Result<(), HostedError>,
    execute: F) -> Result<HostedInt8TeamResult<T>, HostedError>
where D: Send + 'static, T: Send + 'static,
    F: FnOnce(&RedactInput<D>, &mut StrictInt8Engine<'_>, Option<Int8PrefillLimits>,
        &mut RunControl) -> Result<T, HostedError> + Send + 'static {
    dispatch::preflight_int8_team(host, config.native.run, width)?;
    host.check_resident_domain(model)?;
    check_model_identity(model.artifact_identity(), &config.ner_identity)?;
    let required = requirements(config.native)?;
    validate(&config, source.len(), planner.tokenizer_digest(), *planner.template_digest(), required.kv_bytes)?;
    check_encoding_preparation(&planner, config.preparation_reserve_bytes)?;
    let (scratch_bytes, rows) = workspace(config.native, required, width, prefill)?;
    if max_result_bytes == 0 || temporary_bytes == 0 {
        return Err(HostedError::Limits("redaction complete output or editing reservation"));
    }
    let run = config.native.run;
    let lease = host.resources().acquire_lease();
    let bytes = input_bytes(&source, &config, pseudonyms.as_ref())?;
    let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
        || Ok(RedactInput { source, planner, vocabulary, config, pseudonyms }))?;
    let temporary = Pending::reserve(&lease, MemoryClass::JobBuffers, temporary_bytes)?;
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
    let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
    let output = output_claim(&lease, max_result_bytes, 0)?;
    let admission = Admission { input, _temporary: temporary, kv: Some(kv), scratch: Some(scratch),
        output: Some(output), _kv: None, _scratch: None, _lease: lease };
    dispatch::run_int8_prefill_team(host, run, cancellation, model.clone(), width, kernel, rows,
        admission, move |model, admission, control, team| {
            let input = &admission.input.value;
            let mut engine = model.engine(input.config.native.context_tokens, memory_budget(required))
                .map_err(HostedError::Model)?.with_kernel(kernel)
                .with_projection_executor(team).map_err(HostedError::Native)?;
            admission._kv = Some(admission.kv.take()
                .ok_or(HostedError::Limits("redaction KV admission consumed"))?.commit()?);
            admission._scratch = Some(admission.scratch.take()
                .ok_or(HostedError::Limits("redaction scratch admission consumed"))?.commit()?);
            let result = execute(input, &mut engine, prefill, control)?;
            poll(control)?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                return Err(HostedError::Redaction(Int8RedactionError::InvalidResult));
            }
            drop(engine);
            let compute = team.stats().into();
            let pending = admission.output.take()
                .ok_or(HostedError::Limits("redaction output admission consumed"))?;
            let memory = match pending.commit() {
                Ok(memory) => memory,
                Err(error) => { drop(result); return Err(error); }
            };
            Ok(HostedInt8TeamResult::from_guarded(GuardedOutput::new(result, memory), compute))
        })
}

fn poll(control: &mut impl DecodeStepControl) -> Result<(), HostedError> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(HostedError::Native(StrictInt8Error::Cancelled(cause))),
        None => Ok(()),
    }
}

#[cfg(test)] mod tests;
