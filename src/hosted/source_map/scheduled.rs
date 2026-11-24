//! Shared owned execution for complete QA, synthesis and quotation hierarchies.
//! This private path accepts only the concrete task wrappers in this module.
use super::*;
use crate::{native_engine::{decode::DecodeStepControl, strict_int8::{
    StrictInt8Engine, projection::Int8ProjectionExecutor}},
    tasks::source_planning::quantized::Int8SourceError};

pub(super) enum Completion<T> {
    Serial(HostedOutput<T>),
    Team(HostedInt8TeamResult<T>),
}
impl<T> Completion<T> {
    pub(super) fn serial(self) -> Result<HostedOutput<T>, HostedError> {
        match self { Self::Serial(output) => Ok(output), Self::Team(_) => Err(HostedError::CompletionMissing) }
    }
    pub(super) fn team(self) -> Result<HostedInt8TeamResult<T>, HostedError> {
        match self { Self::Team(output) => Ok(output), Self::Serial(_) => Err(HostedError::CompletionMissing) }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Requirements {
    native: Int8MemoryRequirement,
    scratch_bytes: u64,
    worker_bytes: u64,
    rows: usize,
}
fn quote(limits: NativeLimits, prefill: Option<Int8PrefillLimits>, width: Option<usize>)
    -> Result<Requirements, HostedError> {
    let native = requirements(limits)?;
    let scratch_bytes = team::scratch_bytes(limits, native, prefill)?;
    let rows = prefill.map_or(1, |prefill| prefill.max_batch_rows);
    let worker_bytes = width.map(|width| dispatch::int8_prefill_team_workspace_bytes(width, rows))
        .transpose()?.unwrap_or(0);
    sum(&[native.kv_bytes, scratch_bytes, worker_bytes])?;
    Ok(Requirements { native, scratch_bytes, worker_bytes, rows })
}

/// The dispatch owner retains every charge outside the scoped callback, even
/// if a dynamic pass, cancellation or unwind ends that callback early.
struct Claims {
    kv: Option<Pending>,
    scratch: Option<Pending>,
    output: Option<Pending>,
    reduction: Option<Pending>,
    _kv: Option<CommittedMemory>,
    _scratch: Option<CommittedMemory>,
}
struct Owner<T> {
    input: Charged<SourceMapInput<T>>,
    claims: Claims,
    lease: EngineLease,
}

/// A single-use borrowed view. Plans borrow the genuine owned input; the
/// projection executor cannot escape into an output or a later invocation.
pub(super) struct Invocation<'a, 'weights> {
    model: &'weights CurrentCandidateInt8Model,
    native: NativeLimits,
    required: Requirements,
    kernel: Option<Int8Kernel>,
    projection: Option<&'weights mut (dyn Int8ProjectionExecutor + 'weights)>,
    prefill: Option<Int8PrefillLimits>,
    claims: &'a mut Claims,
    lease: &'a EngineLease,
}
impl Invocation<'_, '_> {
    pub(super) fn artifact(&self) -> &ArtifactIdentity { self.model.artifact_identity() }
    pub(super) fn prefill(&self) -> Option<Int8PrefillLimits> { self.prefill }

    /// Admit the actual planned frontier and retained transcript allowance
    /// before creating native storage. No pass may renew these reservations.
    pub(super) fn execute<T, F>(self, reduction_bytes: u64, output_bytes: u64,
        output_tokens: u64, control: &mut RunControl, execute: F) -> Result<HostedOutput<T>, HostedError>
    where T: Send + 'static,
        F: FnOnce(&mut StrictInt8Engine<'_>, &mut RunControl) -> Result<T, HostedError> {
        self.claims.reduction = Some(Pending::reserve(self.lease, MemoryClass::JobBuffers, reduction_bytes)?);
        self.claims.output = Some(output_claim(self.lease, output_bytes, output_tokens)?);
        let engine = self.model.engine(self.native.context_tokens, memory_budget(self.required.native))
            .map_err(HostedError::Model)?;
        let engine = match self.kernel { Some(kernel) => engine.with_kernel(kernel), None => engine };
        let mut engine = match self.projection {
            Some(projection) => engine.with_projection_executor(projection).map_err(HostedError::Native)?,
            None => engine,
        };
        self.claims._kv = Some(self.claims.kv.take()
            .ok_or(HostedError::Limits("document KV admission consumed"))?.commit()?);
        self.claims._scratch = Some(self.claims.scratch.take()
            .ok_or(HostedError::Limits("document scratch admission consumed"))?.commit()?);
        let result = execute(&mut engine, control)?;
        poll(control)?;
        if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
            return Err(HostedError::Source(Int8SourceError::InvalidResult));
        }
        drop(engine);
        let pending = self.claims.output.take()
            .ok_or(HostedError::Limits("document output admission consumed"))?;
        let memory = match pending.commit() {
            Ok(memory) => memory,
            Err(error) => { drop(result); return Err(error); }
        };
        Ok(GuardedOutput::new(result, memory))
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run<S, T, F>(host: &NlpEngine, model: &ResidentInt8, input: SourceMapInput<S>,
    input_bytes: u64, prefill: Option<Int8PrefillLimits>, team: Option<(Int8Kernel, usize)>,
    cancellation: CancellationToken, execute: F) -> Result<Completion<T>, HostedError>
where S: Send + 'static, T: Send + 'static,
    F: FnOnce(&SourceMapInput<S>, Invocation<'_, '_>, &mut RunControl)
        -> Result<HostedOutput<T>, HostedError> + Send + 'static {
    let run = input.config.native.run;
    match team {
        Some((_, width)) => dispatch::preflight_int8_team(host, run, width)?,
        None => dispatch::preflight(host, run)?,
    }
    host.check_resident_domain(model)?;
    check_model_identity(model.artifact_identity(), &input.config.identity)?;
    check_assets(&input.config.identity, input.planner.tokenizer_digest(), *input.planner.template_digest())?;
    let native = input.config.native;
    let required = quote(native, prefill, team.map(|(_, width)| width))?;
    sum(&[input_bytes, required.native.kv_bytes, required.scratch_bytes, required.worker_bytes])?;
    let lease = host.resources().acquire_lease();
    let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?, || Ok(input))?;
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
    let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.scratch_bytes)?;
    let owner = Owner { input, claims: Claims { kv: Some(kv), scratch: Some(scratch),
        output: None, reduction: None, _kv: None, _scratch: None }, lease };
    let model = model.clone();
    match team {
        None => dispatch::run(host, run, cancellation, move |control| {
            let mut owner = owner;
            execute(&owner.input.value, Invocation { model: &model.inner.loaded.value, native, required,
                kernel: None, projection: None, prefill, claims: &mut owner.claims, lease: &owner.lease }, control)
        }).map(Completion::Serial),
        Some((kernel, width)) => dispatch::run_int8_prefill_team(host, run, cancellation, model,
            width, kernel, required.rows, owner, move |model, owner, control, projection| {
                let output = execute(&owner.input.value, Invocation { model, native, required,
                    kernel: Some(kernel), projection: Some(&mut *projection), prefill,
                    claims: &mut owner.claims, lease: &owner.lease }, control)?;
                let compute = projection.stats().into();
                Ok(HostedInt8TeamResult::from_guarded(output, compute))
            }).map(Completion::Team),
    }
}

/// Complete identity admission remains before the first native forward.
pub(super) fn admit<'a>(identities: impl ExactSizeIterator<Item = &'a ExecutionIdentity>,
    artifact: &ArtifactIdentity, planner: &SourceTaskPlanner, control: &mut RunControl)
    -> Result<Vec<ExecutionIdentity>, HostedError> {
    let mut admitted = Vec::new();
    admitted.try_reserve_exact(identities.len())
        .map_err(|_| HostedError::SourceMap(Int8SourceMapError::Allocation))?;
    for identity in identities {
        poll(control)?;
        check_model_identity(artifact, identity)?;
        constrained_int8::check_profile(identity).map_err(|_| HostedError::ModelIdentity)?;
        check_assets(identity, planner.tokenizer_digest(), *planner.template_digest())?;
        admitted.push(identity.clone());
    }
    Ok(admitted)
}

fn poll(control: &mut impl DecodeStepControl) -> Result<(), HostedError> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(HostedError::Source(Int8SourceError::Cancelled(cause))),
        None => Ok(()),
    }
}

#[cfg(test)] mod tests;
