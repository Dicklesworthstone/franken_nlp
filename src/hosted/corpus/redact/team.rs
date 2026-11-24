//! One physical owner for the two sealed native redaction stream policies.
use super::*;
use super::super::scored::Completion;
use crate::native_engine::strict_int8::{
    CurrentCandidateInt8Model, projection::Int8ProjectionExecutor,
};

/// Only the short and complete-document native adapters implement this private
/// boundary. The schedule grants no additional item or whole-stream authority.
pub(super) trait Policy: Send + Sized + 'static {
    fn identity(&self) -> &ExecutionIdentity;
    fn request(&self) -> &RedactionRequest;
    /// Validate complete native/encoding admission and return map/edit storage.
    fn validate(&self, planner: &SourceTaskPlanner, limits: CorpusLimits,
        native: Int8MemoryRequirement) -> Result<u64, HostedError>;
    #[allow(clippy::too_many_arguments)]
    fn execute<R: BufRead, W: Write>(self, planner: &SourceTaskPlanner,
        vocabulary: &ExtractionVocabulary, pseudonyms: Option<&Pseudonyms<'_>>,
        engine: &mut StrictInt8Engine<'_>, admission: CorpusAdmission<'_>, prefill: Option<Int8PrefillLimits>,
        reader: &mut R, writer: &mut W, limits: BatchLimits, defer: bool, control: &mut RunControl)
        -> Result<Completion, HostedError>;
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
fn quote(limits: CorpusLimits, native: Int8MemoryRequirement, temporary_bytes: u64,
    secret_bytes: u64, prefill: Option<Int8PrefillLimits>, width: Option<usize>)
    -> Result<Requirements, HostedError> {
    let input_bytes = sum(&[limits.reservation_bytes()?, temporary_bytes, secret_bytes])?;
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

/// These assets stay in the outer owner even after the per-stream processor
/// finishes. Keys/planner/vocabulary cannot drop at scoped-callback return.
struct Input<P, R, W> {
    planner: Arc<SourceTaskPlanner>,
    vocabulary: Arc<ExtractionVocabulary>,
    pseudonyms: Option<RedactionPseudonyms>,
    policy: Option<P>,
    reader: R,
    writer: W,
}
struct Execution<P, R, W> {
    input: Charged<Input<P, R, W>>,
    kv: Option<Pending>,
    scratch: Option<Pending>,
    _kv: Option<CommittedMemory>,
    _scratch: Option<CommittedMemory>,
    limits: CorpusLimits,
    required: Requirements,
    prefill: Option<Int8PrefillLimits>,
    lease: EngineLease,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run<P, R, W>(host: &NlpEngine, model: &ResidentInt8,
    planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
    policy: P, pseudonyms: Option<RedactionPseudonyms>, limits: CorpusLimits,
    prefill: Option<Int8PrefillLimits>, team: Option<(Int8Kernel, usize)>,
    reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
where P: Policy, R: BufRead + Send + 'static, W: Write + Send + 'static {
    match team {
        Some((_, width)) => dispatch::preflight_int8_team(host, limits.native.run, width)?,
        None => dispatch::preflight(host, limits.native.run)?,
    }
    host.check_resident_domain(model)?;
    check_model_identity(model.artifact_identity(), policy.identity())?;
    let native = requirements(limits.native)?;
    let temporary_bytes = policy.validate(&planner, limits, native)?;
    let required = quote(limits, native, temporary_bytes, secret_bytes(pseudonyms.as_ref())?,
        prefill, team.map(|(_, width)| width))?;
    let lease = host.resources().acquire_lease();
    let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
        || Ok(Input { planner, vocabulary, pseudonyms, policy: Some(policy), reader, writer }))?;
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
    let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.scratch_bytes)?;
    let execution = Execution { input, kv: Some(kv), scratch: Some(scratch),
        _kv: None, _scratch: None, limits, required, prefill, lease };
    let model = model.clone();
    match team {
        None => dispatch::run(host, limits.native.run, cancellation, move |control| {
            let mut execution = execution;
            match execution.execute(&model.inner.loaded.value, None, None, false, control)? {
                Completion::Published(summary) => Ok(summary),
                Completion::Deferred(_) => Err(HostedError::CompletionMissing),
            }
        }),
        Some((kernel, width)) => dispatch::run_int8_prefill_team_with_finalize(
            host, limits.native.run, cancellation, model, width, kernel, required.max_batch_rows,
            execution, move |model, execution, control, projection| {
                execution.execute(model, Some(kernel), Some(projection), true, control)
            },
            |completion, execution, control| match completion {
                Completion::Deferred(completion) => completion
                    .publish(&mut execution.input.value.writer, control).map_err(HostedError::Batch),
                Completion::Published(_) => Err(HostedError::CompletionMissing),
            }),
    }
}

impl<P: Policy, R: BufRead, W: Write> Execution<P, R, W> {
    fn execute<'weights>(&mut self, model: &'weights CurrentCandidateInt8Model,
        kernel: Option<Int8Kernel>, projection: Option<&'weights mut (dyn Int8ProjectionExecutor + 'weights)>,
        defer: bool, control: &mut RunControl) -> Result<Completion, HostedError> {
        let input = &mut self.input.value;
        let policy = input.policy.take().ok_or(HostedError::CompletionMissing)?;
        // Validate one immutable full256 scope before native allocation or
        // reading a private record. The underlying key remains in Input.
        let context = pseudonym_context(input.pseudonyms.as_ref(), policy.request())?;
        let engine = model.engine(self.limits.native.context_tokens, memory_budget(self.required.native))
            .map_err(HostedError::Model)?;
        let engine = match kernel { Some(kernel) => engine.with_kernel(kernel), None => engine };
        let mut engine = match projection {
            Some(projection) => engine.with_projection_executor(projection).map_err(HostedError::Native)?,
            None => engine,
        };
        self._kv = Some(self.kv.take()
            .ok_or(HostedError::Limits("redaction stream KV admission consumed"))?.commit()?);
        self._scratch = Some(self.scratch.take()
            .ok_or(HostedError::Limits("redaction stream scratch admission consumed"))?.commit()?);
        let admission = CorpusAdmission { lease: &self.lease, model: model.artifact_identity(),
            kv_bytes: self.required.native.kv_bytes, sampler_bytes: 0,
            output_bytes: self.limits.transport.max_output_line_bytes as u64 };
        let result = policy.execute(&input.planner, &input.vocabulary, context.as_ref(),
            &mut engine, admission, self.prefill, &mut input.reader, &mut input.writer,
            self.limits.transport, defer, control);
        drop(engine);
        // Input, transient map/edit/IO reservation, native claims and lease
        // outlive every child and the post-join terminal publication.
        result
    }
}

#[cfg(test)] mod tests;
