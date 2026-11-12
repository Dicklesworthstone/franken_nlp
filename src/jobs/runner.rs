//! Durable, serial execution over the existing item-local batch boundary.
//!
//! This profile freezes exact JSON request envelopes, with identity
//! normalization only. It does not turn the live batch epoch into a snapshot.
//! No task failures are cached as successful results and no retry is implicit.
use super::{FrozenManifest, InputRetentionLimits, JobContract, JobError, JobId, JobInput, JobLimits, JobProgress,
    JobSecret, JobWork, OwnedJob, TailPolicy, checkpoint,
    manifest::ManifestBuilder, population::PopulationReadLimits, retained::RetainedInputCursor};
use crate::{batch::{BatchDocument, BatchFault, BatchItemFailure, BatchProcessor, BatchRequestContext},
    canonjson, execution_identity::ExecutionIdentity, native_engine::decode::DecodeStepControl};
use serde::Serialize;
use std::{error::Error, fmt, fs::File, io::BufRead, path::Path};

pub mod source;
pub mod extract;
pub mod scored;
pub mod generation;
pub mod redact;

pub const OWNED_BATCH_PROTOCOL: &str = "fnlp-owned-batch-v1";

/// Trusted embedding interface, like BatchProcessor, NOT model authority.
/// Implementations must keep identity/recipe immutable; bind every default,
/// effective seed, sampler address, dependency scope and execution limit; and
/// return/enforce all native work axes, not just BatchWork's two counters.
/// Preparation performs no model work. Execution returns only after native
/// completion; any admission guard must travel inside the serializable output.
/// The built-in source adapter constructs its native processor from these same
/// private settings and still uses the host's real admission checks.
pub trait DurableBatchProcessor: BatchProcessor {
    type Recipe: Serialize + ?Sized;
    fn execution_identity(&self) -> &ExecutionIdentity;
    fn job_recipe(&self) -> &Self::Recipe;
    /// Validate relationships between this immutable recipe and the immutable
    /// journal limits before create/resume touches storage. For example, a
    /// per-attempt verification allowance must fit across ALL durable attempts,
    /// including failed attempts from previous process invocations. This hook
    /// grants no authority and never replaces OwnedJob's persisted work checks.
    fn validate_job_limits(&self, _limits: JobLimits) -> Result<(), BatchFault> { Ok(()) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure>;
    fn max_result_bytes(&self, prepared: &Self::Prepared) -> u64;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobRunError {
    Storage(JobError), Processor(BatchFault), InvalidEnvelope, WorkContract, Stopped,
}
impl From<JobError> for JobRunError {
    fn from(error: JobError) -> Self { Self::Storage(error) }
}
impl From<BatchItemFailure> for JobRunError {
    fn from(error: BatchItemFailure) -> Self { Self::Processor(error.fault) }
}
impl fmt::Display for JobRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "owned batch refused: {self:?}") }
}
impl Error for JobRunError {}

#[derive(Serialize)]
struct Recipe<'a, R: Serialize + ?Sized> {
    protocol: &'static str,
    input_profile: &'static str,
    processor: &'a R,
}

/// Every execution input is authenticated against the complete frozen population.
/// No raw job/processor mutation or retry/reset handle is exposed. The host
/// accounts for input, planning, journal, canonicalization and frame staging
/// memory in addition to the native admission guard. Work debits are durable;
/// memory admission is deliberately NOT inferred from JobLimits.
pub struct JobRunner<'items, 'bytes, P: DurableBatchProcessor> {
    // Declaration order is the cleanup contract: the input file and its one
    // record buffer close BEFORE OwnedJob releases its exclusive storage lock.
    inputs: Inputs<'items, 'bytes>,
    job: OwnedJob,
    processor: P,
    limits: JobLimits,
    stopped: bool,
}
enum Inputs<'items, 'bytes> {
    Borrowed(&'items [JobInput<'bytes>]),
    Retained(RetainedInputCursor<File>),
}
impl Inputs<'_, '_> {
    fn get<'input, C: DecodeStepControl>(&'input mut self, ordinal: usize,
        job: &mut OwnedJob, control: &mut C) -> Result<JobInput<'input>, JobError> {
        match self {
            Self::Borrowed(inputs) => inputs.get(ordinal).map(|input| JobInput {
                id: input.id, original: input.original, normalized: input.normalized }).ok_or(JobError::Corrupt),
            Self::Retained(cursor) => job.read_retained_input(cursor, control),
        }
    }
}
impl<'items, 'bytes, P: DurableBatchProcessor> JobRunner<'items, 'bytes, P> {
    /// Creates only in an existing protected directory. The host generates and
    /// protects the random key/job ID first; supplying this API is explicit
    /// result-retention consent. All envelopes are checked before opening it.
    pub fn create<C: DecodeStepControl>(root: &Path, key: JobSecret, job_id: JobId,
        limits: JobLimits, inputs: &'items [JobInput<'bytes>], processor: P, control: &mut C)
        -> Result<Self, JobRunError> {
        let manifest = freeze(&key, job_id, limits, inputs, &processor, control)?;
        let job = OwnedJob::create(root, key, manifest, control)?;
        Ok(Self { job, processor, inputs: Inputs::Borrowed(inputs), limits, stopped: false })
    }

    /// Explicitly retain the complete exact envelope population under its own
    /// disk cap. The input file and directory are synced before the first
    /// authenticated journal receipt can acknowledge the retained snapshot.
    pub fn create_with_retained_inputs<C: DecodeStepControl>(root: &Path, key: JobSecret,
        job_id: JobId, limits: JobLimits, retention: InputRetentionLimits,
        inputs: &'items [JobInput<'bytes>], processor: P, control: &mut C) -> Result<Self, JobRunError> {
        retention.validate()?;
        let manifest = freeze(&key, job_id, limits, inputs, &processor, control)?;
        let job = OwnedJob::create_with_retained_inputs(root, key, manifest, retention, inputs, control)?;
        Ok(Self { job, processor, inputs: Inputs::Borrowed(inputs), limits, stopped: false })
    }

    /// Explicitly retain an exact NDJSON population without keeping its raw
    /// records resident. Each original envelope is bounded, typed and frozen
    /// before the complete retained snapshot is authenticated and acknowledged.
    /// Preparation and execution begin only after that complete validation.
    /// The runner owns one authenticated replay record plus the bounded keyed
    /// manifest index; hosts must admit both before entering this constructor.
    /// An interruption before the initial journal receipt may leave orphan
    /// input bytes; these are never adopted or silently overwritten.
    #[allow(clippy::too_many_arguments)]
    pub fn create_with_retained_ndjson<R: BufRead, C: DecodeStepControl>(root: &Path,
        key: JobSecret, job_id: JobId, limits: JobLimits, retention: InputRetentionLimits,
        reader: R, transport: PopulationReadLimits, processor: P, control: &mut C)
        -> Result<Self, JobRunError> {
        checkpoint(control)?;
        retention.validate()?;
        transport.validate()?;
        let fresh = manifest_builder(&key, job_id, limits, &processor, control)?;
        let (job, cursor) = OwnedJob::create_with_retained_ndjson(root, key, limits, retention,
            reader, transport, fresh, control, |input, _| -> Result<(), JobRunError> {
                drop(parse::<P>(input, limits)?);
                Ok(())
            })?;
        Ok(Self { job, processor, inputs: Inputs::Retained(cursor), limits, stopped: false })
    }

    /// Requires the entire original population and a newly constructed clean
    /// processor. Frozen recipe/identity/limits mismatches refuse before spool
    /// repair. Persisted failed-attempt debits never reset with the processor.
    pub fn resume<C: DecodeStepControl>(root: &Path, key: JobSecret, job_id: JobId,
        limits: JobLimits, inputs: &'items [JobInput<'bytes>], processor: P,
        tail: TailPolicy, control: &mut C) -> Result<Self, JobRunError> {
        let manifest = freeze(&key, job_id, limits, inputs, &processor, control)?;
        let job = OwnedJob::resume(root, key, manifest, tail, control)?;
        Ok(Self { job, processor, inputs: Inputs::Borrowed(inputs), limits, stopped: false })
    }
    /// Restore only an explicitly retained, authenticated complete snapshot.
    /// No original reader is needed. The runner first authenticates and
    /// refreezes EVERY input with the clean processor's typed arguments and
    /// recipe, then replays pending inputs through one reusable record buffer.
    /// Hosts admit that record plus the bounded keyed index before entering.
    /// Result-tail repair and new debits follow the complete initial scan.
    pub fn resume_with_retained_inputs<C: DecodeStepControl>(root: &Path, key: JobSecret,
        job_id: JobId, limits: JobLimits, retention: InputRetentionLimits, processor: P,
        tail: TailPolicy, control: &mut C) -> Result<Self, JobRunError> {
        checkpoint(control)?;
        retention.validate()?;
        let fresh = manifest_builder(&key, job_id, limits, &processor, control)?;
        let (job, cursor) = OwnedJob::resume_with_retained_inputs(root, key, job_id, limits,
            retention, tail, fresh, control, |input, _| -> Result<(), JobRunError> {
                drop(parse::<P>(input, limits)?);
                Ok(())
            })?;
        Ok(Self { job, processor, inputs: Inputs::Retained(cursor), limits, stopped: false })
    }
    pub fn progress(&self) -> JobProgress { self.job.progress() }
    pub fn is_stopped(&self) -> bool { self.stopped || self.job.is_poisoned() }

    /// Execute at most one uncommitted ordinal. The two-axis batch estimate
    /// must agree with the full durable estimate. Admission/running debits are
    /// synced before execute_with_context; the guarded result is borrowed all
    /// the way through the spool sync and journal acknowledgement.
    ///
    /// Every failure stops this runner, including ordinarily recoverable batch
    /// document errors. Drop/reopen explicitly; never turn an inference error
    /// into a successful QA abstention or a committed error-shaped result.
    pub fn step<C: DecodeStepControl>(&mut self, control: &mut C)
        -> Result<Option<JobProgress>, JobRunError> {
        if self.is_stopped() { return Err(JobRunError::Stopped); }
        let progress = self.job.progress();
        if progress.committed == progress.items { return Ok(None); }
        // Set BEFORE any fallible planning/host call, including an unwind.
        self.stopped = true;
        checkpoint(control)?;
        let ordinal = usize::try_from(progress.committed).map_err(|_| JobError::Limit)?;
        let input = self.inputs.get(ordinal, &mut self.job, control)?;
        let document = parse::<P>(&input, self.limits)?;
        let prepared = self.processor.prepare_with_control(document, control)?;
        checkpoint(control)?;
        let work = self.processor.durable_work(&prepared)?;
        let batch = self.processor.planned_work(&prepared);
        if batch.forward_positions != work.model.forward_positions
            || batch.projected_logits != work.model.projected_logits {
            return Err(JobRunError::WorkContract);
        }
        if self.processor.max_result_bytes(&prepared) > self.limits.max_result_bytes as u64 {
            return Err(JobError::Limit.into());
        }
        let sequence = progress.committed.checked_add(1).ok_or(JobError::Limit)?;
        // Stable delivery coordinates, never an attempt-derived sampling seed.
        let context = BatchRequestContext { request_seq: sequence, epoch: 1,
            input_line: sequence, byte_offset: 0 };
        let attempt = self.job.begin(&input, work, control)?;
        let output = self.processor.execute_with_context(prepared, context, control)?;
        let committed = attempt.commit(&output, control)?;
        drop(output);
        self.stopped = false;
        Ok(Some(committed))
    }
    pub fn run<C: DecodeStepControl>(&mut self, control: &mut C) -> Result<JobProgress, JobRunError> {
        while self.step(control)?.is_some() {}
        Ok(self.progress())
    }
    /// Explicitly reads retained private output; callers own buffer admission.
    pub fn read_committed<C: DecodeStepControl>(&mut self, ordinal: u64, control: &mut C)
        -> Result<Vec<u8>, JobRunError> {
        if self.is_stopped() { return Err(JobRunError::Stopped); }
        self.stopped = true;
        let result = self.job.read_committed(ordinal, control)?;
        self.stopped = false; Ok(result)
    }
    /// Publish the complete ordered output without replacing an existing file.
    pub fn materialize_ordered<C: DecodeStepControl>(&mut self, control: &mut C)
        -> Result<JobProgress, JobRunError> {
        if self.is_stopped() { return Err(JobRunError::Stopped); }
        self.stopped = true;
        let result = self.job.materialize_ordered(control)?;
        self.stopped = false; Ok(result)
    }
    pub fn verify<C: DecodeStepControl>(&mut self, control: &mut C)
        -> Result<JobProgress, JobRunError> {
        if self.is_stopped() { return Err(JobRunError::Stopped); }
        self.stopped = true;
        let result = self.job.verify(control)?;
        self.stopped = false; Ok(result)
    }
}

fn freeze<P: DurableBatchProcessor, C: DecodeStepControl>(key: &JobSecret, job_id: JobId,
    limits: JobLimits, inputs: &[JobInput<'_>], processor: &P, control: &mut C)
    -> Result<FrozenManifest, JobRunError> {
    // Freeze first bounds the TOTAL population before envelope parsing. It
    // retains only keyed metadata and performs no filesystem/native work.
    let mut fresh = manifest_builder(key, job_id, limits, processor, control)?;
    for input in inputs { fresh.push(key, input, control)?; }
    let manifest = fresh.finish(key)?;
    for input in inputs {
        checkpoint(control)?;
        drop(parse::<P>(input, limits)?);
    }
    Ok(manifest)
}
fn manifest_builder<P: DurableBatchProcessor, C: DecodeStepControl>(key: &JobSecret, job_id: JobId,
    limits: JobLimits, processor: &P, control: &mut C) -> Result<ManifestBuilder, JobRunError> {
    limits.validate()?;
    processor.validate_job_limits(limits).map_err(JobRunError::Processor)?;
    let recipe = Recipe { protocol: OWNED_BATCH_PROTOCOL,
        input_profile: "identity-json-envelope-v1", processor: processor.job_recipe() };
    Ok(ManifestBuilder::new(key, JobContract { job_id, execution: processor.execution_identity(),
        recipe: &recipe, limits }, control)?)
}
fn parse<P: BatchProcessor>(input: &JobInput<'_>, limits: JobLimits)
    -> Result<BatchDocument<P::Args>, JobRunError> {
    if input.original != input.normalized || input.original.len() > limits.max_input_bytes_per_item {
        return Err(JobRunError::InvalidEnvelope);
    }
    let text = std::str::from_utf8(input.original).map_err(|_| JobRunError::InvalidEnvelope)?;
    let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits {
        max_depth: super::memory::INPUT_JSON_DEPTH, max_string_bytes: limits.max_input_bytes_per_item,
    }).map_err(|_| JobRunError::InvalidEnvelope)?;
    let document: BatchDocument<P::Args> = serde_json::from_value(value).map_err(|_| JobRunError::InvalidEnvelope)?;
    if document.id != input.id { return Err(JobRunError::InvalidEnvelope); }
    Ok(document)
}

#[cfg(test)] mod tests;
