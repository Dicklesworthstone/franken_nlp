//! Shared physical lifetime for sealed native INT8 structured and scored streams.
//! Task compilers retain their own semantics, validation and work ledgers.
//! No callback or user-provided processor can enter this private host path.
use super::*;
use crate::{
    batch::{BatchProcessor, deferred::{DeferredCompletion, run_ndjson_deferred}},
    native_engine::{decode::DecodeStepControl, simd::Int8Kernel, strict_int8::{
        StrictInt8Engine, prefill::Int8PrefillLimits, projection::Int8ProjectionExecutor,
    }},
};

pub(super) enum Completion {
    Published(BatchSummary),
    Deferred(DeferredCompletion),
}

pub(super) trait CorpusTask: Send + Sized + 'static {
    fn identity(&self) -> &ExecutionIdentity;
    /// Keep the actual shared compiler/vocabulary and any combined-output
    /// owner outside the consuming callback through physical worker join.
    fn retain_for_join(&self) -> Option<Box<dyn Send>> { None }
    /// Validate the exact task, complete native allocation and output envelope.
    /// Return additional retained/temporary task storage beyond the common
    /// transport and caller preparation: tokenizer workspaces or complete
    /// document schema/chunk/reduction owners are simultaneously live.
    fn validate(&self, model: &ArtifactIdentity, limits: CorpusLimits,
        native: Int8MemoryRequirement) -> Result<u64, HostedError>;
    #[allow(clippy::too_many_arguments)]
    fn execute<R: BufRead, W: Write>(self, engine: &mut StrictInt8Engine<'_>,
        admission: CorpusAdmission<'_>, prefill: Option<Int8PrefillLimits>,
        reader: &mut R, writer: &mut W, limits: BatchLimits, defer: bool,
        control: &mut RunControl) -> Result<Completion, HostedError>;
}

/// Reuse the existing bounded NDJSON transport, including item guards, failed
/// records, flush epochs, backpressure and nonrefundable processor work.
pub(super) fn execute_stream<R: BufRead, W: Write, P: BatchProcessor, C: DecodeStepControl>(
    reader: &mut R, writer: &mut W, processor: &mut P, limits: BatchLimits,
    defer: bool, control: &mut C) -> Result<Completion, HostedError> {
    if defer {
        run_ndjson_deferred(reader, writer, processor, limits, control)
            .map(Completion::Deferred).map_err(HostedError::Batch)
    } else {
        batch::run_ndjson(reader, writer, processor, limits, control)
            .map(Completion::Published).map_err(HostedError::Batch)
    }
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
fn quote(limits: CorpusLimits, native: Int8MemoryRequirement, task_bytes: u64,
    prefill: Option<Int8PrefillLimits>, width: Option<usize>) -> Result<Requirements, HostedError> {
    let input_bytes = sum(&[limits.reservation_bytes()?, task_bytes])?;
    let scratch_bytes = crate::hosted::scored::scoring_scratch(
        sum(&[native.rope_bytes, native.scratch_payload_bound,
            limits.native.allocator_reserve_bytes])?, prefill)?;
    let max_batch_rows = prefill.map_or(1, |limits| limits.max_batch_rows);
    let worker_bytes = width.map(|width|
        dispatch::int8_prefill_team_workspace_bytes(width, max_batch_rows))
        .transpose()?.unwrap_or(0);
    // Worker capacity belongs to the dispatcher and never discounts the
    // coordinator's packed prompt workspace or either input owner.
    let required = Requirements { native, input_bytes, scratch_bytes, worker_bytes, max_batch_rows };
    required.total_bytes()?;
    Ok(required)
}

/// This aggregate stays OUTSIDE scoped_cpu's callback. A failed record,
/// cancellation, panic, or early callback return cannot release its reader,
/// writer, native claims or lease while any numerical child remains live.
struct RetainedInput<P, R, W> {
    _retained: Option<Box<dyn Send>>,
    stream: StreamInput<P, R, W>,
}
struct Execution<P, R, W> {
    input: Charged<RetainedInput<P, R, W>>,
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
pub(super) fn run<P, R, W>(host: &NlpEngine, model: &ResidentInt8, task: P,
    limits: CorpusLimits, prefill: Option<Int8PrefillLimits>, team: Option<(Int8Kernel, usize)>,
    reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
where P: CorpusTask, R: BufRead + Send + 'static, W: Write + Send + 'static {
    match team {
        Some((_, width)) => dispatch::preflight_int8_team(host, limits.native.run, width)?,
        None => dispatch::preflight(host, limits.native.run)?,
    }
    host.check_resident_domain(model)?;
    check_model_identity(model.artifact_identity(), task.identity())?;
    let native = requirements(limits.native)?;
    let task_bytes = task.validate(model.artifact_identity(), limits, native)?;
    let required = quote(limits, native, task_bytes, prefill, team.map(|(_, width)| width))?;
    let lease = host.resources().acquire_lease();
    let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
        || {
            let retained = task.retain_for_join();
            Ok(RetainedInput { _retained: retained,
                stream: StreamInput { planner: Some(task), reader, writer } })
        })?;
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
    let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.scratch_bytes)?;
    let execution = Execution { input, kv: Some(kv), scratch: Some(scratch),
        _kv: None, _scratch: None, limits, required, prefill, lease };
    let model = model.clone();
    match team {
        None => dispatch::run(host, limits.native.run, cancellation, move |control| {
            let mut execution = execution;
            match execution.run(&model.inner.loaded.value, None, None, false, control)? {
                Completion::Published(summary) => Ok(summary),
                Completion::Deferred(_) => Err(HostedError::CompletionMissing),
            }
        }),
        Some((kernel, width)) => dispatch::run_int8_prefill_team_with_finalize(
            host, limits.native.run, cancellation, model, width, kernel,
            required.max_batch_rows, execution,
            move |model, execution, control, projection| {
                execution.run(model, Some(kernel), Some(projection), true, control)
            },
            |completion, execution, control| match completion {
                Completion::Deferred(completion) => completion
                    .publish(&mut execution.input.value.stream.writer, control).map_err(HostedError::Batch),
                Completion::Published(_) => Err(HostedError::CompletionMissing),
            }),
    }
}

impl<P: CorpusTask, R: BufRead, W: Write> Execution<P, R, W> {
    fn commit_native(&mut self) -> Result<(), HostedError> {
        self._kv = Some(self.kv.take()
            .ok_or(HostedError::Limits("corpus KV admission consumed"))?.commit()?);
        self._scratch = Some(self.scratch.take()
            .ok_or(HostedError::Limits("corpus scratch admission consumed"))?.commit()?);
        Ok(())
    }

    fn run<'weights>(&mut self, model: &'weights CurrentCandidateInt8Model,
        kernel: Option<Int8Kernel>, projection: Option<&'weights mut (dyn Int8ProjectionExecutor + 'weights)>,
        defer: bool, control: &mut RunControl) -> Result<Completion, HostedError> {
        let task = self.input.value.stream.planner.take().ok_or(HostedError::CompletionMissing)?;
        let engine = model.engine(self.limits.native.context_tokens, memory_budget(self.required.native))
            .map_err(HostedError::Model)?;
        let engine = match kernel { Some(kernel) => engine.with_kernel(kernel), None => engine };
        let mut engine = match projection {
            Some(projection) => engine.with_projection_executor(projection).map_err(HostedError::Native)?,
            None => engine,
        };
        self.commit_native()?;
        let admission = CorpusAdmission { lease: &self.lease, model: model.artifact_identity(),
            kv_bytes: self.required.native.kv_bytes, sampler_bytes: 0,
            output_bytes: self.limits.transport.max_output_line_bytes as u64 };
        let result = task.execute(&mut engine, admission, self.prefill,
            &mut self.input.value.stream.reader, &mut self.input.value.stream.writer,
            self.limits.transport, defer, control);
        // Native buffers end before scope exit; their outer claims survive
        // every physical child and, on success, terminal footer publication.
        drop(engine);
        result
    }
}

#[cfg(test)]
mod tests;
