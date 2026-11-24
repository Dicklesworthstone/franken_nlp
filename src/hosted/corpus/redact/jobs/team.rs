//! Physical ownership for authenticated short and complete-document INT8 jobs.
use super::*;
use crate::{
    hosted::corpus::jobs::Population,
    jobs::{JobInputMemory, JobLimits, population::JobPopulation},
    native_engine::strict_int8::{CurrentCandidateInt8Model, projection::Int8ProjectionExecutor},
};

/// Sealed native policies preserve the existing exact saved recipes and work
/// debits. Physical scheduling is never added to semantic replay identity.
pub(super) trait Policy: Send + Sized + 'static {
    fn identity(&self) -> &ExecutionIdentity;
    fn request(&self) -> &RedactionRequest;
    fn validate(&self, planner: &SourceTaskPlanner, request: &SourceJobRequest,
        limits: JobHostLimits, native: Int8MemoryRequirement) -> Result<u64, HostedError>;
    fn recipe(&self, planner: &SourceTaskPlanner, context: Option<&Pseudonyms<'_>>)
        -> Result<RedactionJobRecipe, HostedError>;
    #[allow(clippy::too_many_arguments)]
    fn execute<C: DecodeStepControl>(self, planner: &SourceTaskPlanner,
        vocabulary: &ExtractionVocabulary, context: Option<&Pseudonyms<'_>>,
        engine: &mut StrictInt8Engine<'_>, admission: CorpusAdmission<'_>, prefill: Option<Int8PrefillLimits>,
        request: SourceJobRequest, population: &Population<'_>, recipe: RedactionJobRecipe,
        control: &mut C) -> Result<JobProgress, JobRunError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Requirements {
    native: Int8MemoryRequirement,
    input_bytes: u64,
    scratch_bytes: u64,
    worker_bytes: u64,
    max_batch_rows: usize,
}
impl Requirements {
    fn total_bytes(self) -> Result<u64, HostedError> {
        sum(&[self.input_bytes, self.native.kv_bytes, self.scratch_bytes, self.worker_bytes])
    }
}
#[allow(clippy::too_many_arguments)]
fn quote(limits: JobHostLimits, job: JobLimits, input: JobInputMemory, native: Int8MemoryRequirement,
    temporary_bytes: u64, secret_bytes: u64, prefill: Option<Int8PrefillLimits>, width: Option<usize>)
    -> Result<Requirements, HostedJobError> {
    let input_bytes = sum(&[limits.required_buffer_bytes_for_input(job, input)?,
        temporary_bytes, secret_bytes, 4 * RECIPE_BYTES as u64])?;
    let prompt = prefill.map(Int8PrefillLimits::validate).transpose()
        .map_err(HostedError::Native)?.unwrap_or(0);
    let scratch_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
        limits.native.allocator_reserve_bytes, prompt])?;
    let max_batch_rows = prefill.map_or(1, |limits| limits.max_batch_rows);
    let worker_bytes = width.map(|width|
        dispatch::int8_prefill_team_workspace_bytes(width, max_batch_rows))
        .transpose()?.unwrap_or(0);
    let required = Requirements { native, input_bytes, scratch_bytes, worker_bytes, max_batch_rows };
    required.total_bytes()?;
    Ok(required)
}

struct Input<P, R> {
    planner: Arc<SourceTaskPlanner>,
    vocabulary: Arc<ExtractionVocabulary>,
    pseudonyms: Option<RedactionPseudonyms>,
    policy: Option<P>,
    request: Option<SourceJobRequest>,
    reader: R,
}
struct Execution<P, R> {
    // Original population bytes die before Input's encompassing charge, and
    // only after every scoped worker has joined. Retained modes use Input's
    // reader or the authenticated journal's bounded record instead.
    original: Option<JobPopulation>,
    input: Charged<Input<P, R>>,
    kv: Option<Pending>,
    scratch: Option<Pending>,
    _kv: Option<CommittedMemory>,
    _scratch: Option<CommittedMemory>,
    limits: JobHostLimits,
    required: Requirements,
    prefill: Option<Int8PrefillLimits>,
    lease: EngineLease,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run<P, R>(host: &NlpEngine, model: &ResidentInt8,
    planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
    policy: P, pseudonyms: Option<RedactionPseudonyms>, request: SourceJobRequest,
    limits: JobHostLimits, prefill: Option<Int8PrefillLimits>, team: Option<(Int8Kernel, usize)>,
    reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
where P: Policy, R: BufRead + Send + 'static {
    match team {
        Some((_, width)) => dispatch::preflight_int8_team(host, limits.native.run, width)?,
        None => dispatch::preflight(host, limits.native.run)?,
    }
    host.check_resident_domain(model)?;
    check_model_identity(model.artifact_identity(), policy.identity())?;
    let native = requirements(limits.native)?;
    let temporary_bytes = policy.validate(&planner, &request, limits, native)?;
    let required = quote(limits, request.limits, request.mode.input_memory(), native,
        temporary_bytes, secret_bytes(pseudonyms.as_ref())?, prefill, team.map(|(_, width)| width))?;
    let lease = host.resources().acquire_lease();
    let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
        || Ok(Input { planner, vocabulary, pseudonyms, policy: Some(policy), request: Some(request), reader }))?;
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
    let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.scratch_bytes)?;
    let execution = Execution { original: None, input, kv: Some(kv), scratch: Some(scratch),
        _kv: None, _scratch: None, limits, required, prefill, lease };
    let model = model.clone();
    let result = match team {
        None => dispatch::run(host, limits.native.run, cancellation, move |control| {
            let mut execution = execution;
            execution.execute(&model.inner.loaded.value, None, None, control)
        }),
        Some((kernel, width)) => dispatch::run_int8_prefill_team(
            host, limits.native.run, cancellation, model, width, kernel, required.max_batch_rows,
            execution, move |model, execution, control, projection| {
                execution.execute(model, Some(kernel), Some(projection), control)
            }),
    }?;
    result.map_err(HostedJobError::Job)
}

impl<P: Policy, R: BufRead> Execution<P, R> {
    fn execute<'weights>(&mut self, model: &'weights CurrentCandidateInt8Model,
        kernel: Option<Int8Kernel>, projection: Option<&'weights mut (dyn Int8ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        let input = &mut self.input.value;
        let policy = input.policy.take().ok_or(HostedError::CompletionMissing)?;
        let request = input.request.take().ok_or(HostedError::CompletionMissing)?;
        // The actual pseudonym key is checked before private population IO.
        // This exact scope stays borrowed from the outer physical owner.
        let context = pseudonym_context(input.pseudonyms.as_ref(), policy.request())?;
        let recipe = policy.recipe(&input.planner, context.as_ref())?;
        let population = match read_population(&mut input.reader, &request, self.limits.transport, control) {
            Ok(Population::Original(original)) => {
                self.original = Some(original);
                Population::BorrowedOriginal(self.original.as_ref().ok_or(HostedError::CompletionMissing)?)
            }
            Ok(other) => other,
            Err(error) => return Ok(Err(JobRunError::Storage(error))),
        };
        let engine = model.engine(self.limits.native.context_tokens, memory_budget(self.required.native))
            .map_err(HostedError::Model)?;
        let engine = match kernel { Some(kernel) => engine.with_kernel(kernel), None => engine };
        let mut engine = match projection {
            Some(projection) => engine.with_projection_executor(projection).map_err(HostedError::Native)?,
            None => engine,
        };
        self._kv = Some(self.kv.take()
            .ok_or(HostedError::Limits("redaction job KV admission consumed"))?.commit()?);
        self._scratch = Some(self.scratch.take()
            .ok_or(HostedError::Limits("redaction job scratch admission consumed"))?.commit()?);
        let admission = CorpusAdmission { lease: &self.lease, model: model.artifact_identity(),
            kv_bytes: self.required.native.kv_bytes, sampler_bytes: 0,
            output_bytes: request.limits.max_result_bytes as u64 };
        let result = policy.execute(&input.planner, &input.vocabulary, context.as_ref(),
            &mut engine, admission, self.prefill, request, &population, recipe, control);
        // The existing runner consumes the storage secret, closes its journal
        // and releases item guards only after acknowledgement. Original inputs,
        // pseudonym assets and all native/storage claims outlive physical join.
        drop(engine);
        Ok(result)
    }
}

#[cfg(test)] mod tests;
