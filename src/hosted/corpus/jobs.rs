//! Durable INT8 source jobs on the existing process-owned blocking seam.
//! The host owns admission; callers cannot substitute an uncharged processor.
//! Input ingestion, journal lifetime, native work and optional publication all
//! finish inside one physical invocation before its completion is observable.
use super::*;
mod manage;
mod extract;
mod scored;
mod generation;
mod source_document;
pub use manage::{JobManagementLimits, StoredJobOperation, StoredJobRequest};
use crate::{
    jobs::{InputRetentionLimits, JobError, JobId, JobInputMemory, JobLimits, JobProgress, JobSecret, TailPolicy,
        population::{JobPopulation, PopulationReadLimits},
        runner::{DurableBatchProcessor, JobRunError, JobRunner, source::Int8SourceJobProcessor}},
    batch::source::quantized as source_batch,
    tasks::source_planning::SourceTaskPlanner,
    native_engine::decode::DecodeStepControl,
};
use std::cell::Cell;

/// Retry is an explicit new invocation; no automatic reopen or work refund.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobOpenMode {
    Create,
    Resume(TailPolicy),
    /// Stream exact input envelopes into the protected job directory, then
    /// authenticate the complete snapshot before executing its first item.
    /// This separately grants private-source retention under its own disk cap.
    CreateRetainingInputs(InputRetentionLimits),
    /// Resume from an authenticated retained population without reading the
    /// supplied reader. The current read ceiling may tighten the original cap.
    ResumeRetained { tail: TailPolicy, retention: InputRetentionLimits },
}
impl JobOpenMode {
    /// Payload ownership implied by the actual job source. This changes only
    /// the memory quote, never the stored contract or lifetime work limits.
    pub fn input_memory(self) -> JobInputMemory {
        match self {
            Self::CreateRetainingInputs(_) | Self::ResumeRetained { .. } => JobInputMemory::RetainedRecord,
            Self::Create | Self::Resume(_) => JobInputMemory::CompletePopulation,
        }
    }
}

/// Explicit owner-only result-retention request. Input retention additionally
/// requires `CreateRetainingInputs`; ordinary create/resume never save inputs.
/// The host generates/protects
/// the random secret and ID BEFORE this call; never put the key in argv or a
/// public receipt. No Debug/Clone/Deserialize or implicit secret generation.
pub struct SourceJobRequest {
    pub root: PathBuf,
    pub key: JobSecret,
    pub job_id: JobId,
    pub limits: JobLimits,
    pub mode: JobOpenMode,
    /// Publish fixed `materialized.ndjson` only after all results commit.
    pub materialize: bool,
}

/// Per-invocation host accounting, distinct from immutable lifetime JobLimits.
/// The supplied reader and preparation objects may already exist: their entire
/// retained allocation, not just a read window, must be priced here. Like the
/// other hosted APIs this is an admission model, not an RSS/allocator monitor.
#[derive(Clone, Copy, Debug)]
pub struct JobHostLimits {
    pub native: NativeLimits,
    pub transport: PopulationReadLimits,
    /// Planner/vocabulary/defaults, source/grammar/TaskIR graphs and slack.
    pub preparation_reserve_bytes: u64,
    /// Supplied reader buffers, retained backing input, path/key host overhead.
    pub io_reserve_bytes: u64,
    /// fsqlite connection/page/cache/transaction allocations. A database file
    /// byte cap does NOT prove a corresponding database-engine RAM bound.
    pub journal_reserve_bytes: u64,
    /// Canonical JSON trees and serializer/allocator overhead in addition to
    /// the byte-buffer floor below. Results keep their separate output guard.
    pub serialization_reserve_bytes: u64,
}
impl JobHostLimits {
    /// Checked complete-population commitment, excluding native KV,
    /// workspace and per-result claims. Entrypoints may reject impossible
    /// admission before loading weights; this neither reserves nor grants RAM.
    pub fn required_buffer_bytes(self, job: JobLimits) -> Result<u64, HostedJobError> {
        self.reservation_bytes(job)
    }
    fn reservation_bytes(self, job: JobLimits) -> Result<u64, HostedJobError> {
        self.required_buffer_bytes_for_input(job, JobInputMemory::CompletePopulation)
    }
    /// Price the actual input owner. Explicit retained creation and replay
    /// keep the keyed manifest index and one record; ordinary original-input
    /// create/resume keep the full population. Entrypoints use `request.mode`.
    pub fn required_buffer_bytes_for_input(self, job: JobLimits, input: JobInputMemory)
        -> Result<u64, HostedJobError> {
        self.native.run.validate()?; self.transport.validate()?; job.validate()?;
        if self.preparation_reserve_bytes == 0 || self.io_reserve_bytes == 0
            || self.journal_reserve_bytes == 0 || self.serialization_reserve_bytes == 0 {
            return Err(HostedError::Limits("explicit job preparation/IO/journal/serialization reservations required").into());
        }
        Ok(sum(&[
            input.buffer_bytes(job)?,
            self.preparation_reserve_bytes, self.io_reserve_bytes,
            self.journal_reserve_bytes, self.serialization_reserve_bytes,
        ])?)
    }
}

/// Runtime cancellation/drain errors take precedence over a job return value.
/// An interrupted acknowledgement is reconciled by authenticated resume, never
/// by trusting an incomplete return or rerunning an apparently missing result.
/// Nested diagnostics remain opt-in; default formatting exposes no content.
pub enum HostedJobError { Host(HostedError), Job(JobRunError) }
impl From<HostedError> for HostedJobError { fn from(e: HostedError) -> Self { Self::Host(e) } }
impl From<JobError> for HostedJobError { fn from(e: JobError) -> Self { Self::Job(e.into()) } }
impl From<JobRunError> for HostedJobError { fn from(e: JobRunError) -> Self { Self::Job(e) } }
impl fmt::Display for HostedJobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Host(_) => f.write_str("hosted owned-job invocation failed"),
            Self::Job(e) => fmt::Display::fmt(e, f) }
    }
}
impl fmt::Debug for HostedJobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for HostedJobError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Host(e) => Some(e), Self::Job(e) => Some(e) }
    }
}

impl NlpEngine {
    /// Create or explicitly resume a complete NER/keyphrase/cited-summary/QA
    /// job from original NDJSON or explicitly retained input envelopes. The
    /// same resident model is used for every
    /// item. The caller supplies no admission implementation or native driver.
    ///
    /// Complete input validation precedes native preparation and execution.
    /// Explicit retained creation stages one original record at a time under
    /// the admitted native/input owners before acknowledging the snapshot.
    /// Per-item identity/admission checks and all lifetime work debits remain
    /// mandatory throughout the existing process-owned invocation.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_source<R>(&self, model: &ResidentInt8, planner: Arc<SourceTaskPlanner>,
        vocabulary: Arc<ExtractionVocabulary>, config: SourceCorpusConfig,
        request: SourceJobRequest, limits: JobHostLimits, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        let bytes = limits.required_buffer_bytes_for_input(request.limits, request.mode.input_memory())?;
        let required = requirements(limits.native)?;
        check_model_identity(model.artifact_identity(), &config.identity)?;
        source_batch::check_configuration(&planner, &config.identity, config.task_ceiling,
            config.planning, config.defaults.as_ref()).map_err(HostedError::BatchSetup)?;
        source_batch::validate_limits(config.native_work).map_err(HostedError::BatchSetup)?;
        if required.kv_bytes == 0 || required.kv_bytes > config.task_ceiling.max_kv_bytes {
            return Err(HostedError::Limits("job complete KV reservation").into());
        }
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(StreamInput { planner: Some((planner, vocabulary, config, request)), reader, writer: () }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        let completed = dispatch::run(self, limits.native.run, cancellation, move |control| {
            // Force whole-aggregate capture: queued cancellation must drop
            // owned reader/configuration storage BEFORE its ledger charge.
            let mut input = input;
            let (planner, vocabulary, config, request) = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let population = match read_population(&mut input.value.reader, &request,
                limits.transport, control) {
                Ok(population) => population,
                Err(error) => return Ok(Err(JobRunError::Storage(error))),
            };
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let admission = CorpusAdmission { lease: &lease, model: model.artifact_identity(),
                kv_bytes: required.kv_bytes, sampler_bytes: 0, output_bytes: request.limits.max_result_bytes as u64 };
            let result = (|| {
                let processor = Int8SourceJobProcessor::new(&planner, config.identity, config.task_ceiling,
                    config.planning, config.defaults, &mut engine.value, &vocabulary, admission, config.native_work)
                    .map_err(JobRunError::Processor)?;
                run_population(request, &population, processor, control)
            })();
            // run_population drops its runner (journal, spool, lock and native
            // processor borrow) first. Output guards already survived each
            // spool sync/journal acknowledgement. No job object escapes.
            drop(engine);
            drop(population);
            drop(vocabulary);
            drop(planner);
            drop(input);
            drop(lease);
            Ok(result)
        })?;
        completed.map_err(HostedJobError::Job)
    }
}

/// Original-input jobs retain the complete population. Explicit retained
/// creation transfers a one-shot reader to the runner, where the processor's
/// typed argument contract can validate each original record before freezing.
/// This owner remains inside the blocking invocation and drops before the
/// charged input that supplies its reader.
pub(in crate::hosted) enum Population<'reader> {
    Original(JobPopulation),
    Retaining { reader: Cell<Option<&'reader mut dyn BufRead>>, transport: PopulationReadLimits },
    Retained,
}

/// Retained modes defer to the runner so complete input authentication, recipe
/// validation and storage initialization/recovery share the exclusive job lock.
/// No supplied stream can substitute for journal-authorized replay bytes.
pub(in crate::hosted) fn read_population<'reader, R: BufRead, C: DecodeStepControl>(reader: &'reader mut R,
    request: &SourceJobRequest, transport: PopulationReadLimits, control: &mut C)
    -> Result<Population<'reader>, JobError> {
    match request.mode {
        JobOpenMode::ResumeRetained { retention, .. } => {
            crate::jobs::checkpoint(control)?;
            retention.validate()?;
            return Ok(Population::Retained);
        }
        JobOpenMode::CreateRetainingInputs(retention) => {
            crate::jobs::checkpoint(control)?;
            retention.validate()?; request.limits.validate()?; transport.validate()?;
            return Ok(Population::Retaining { reader: Cell::new(Some(reader)), transport });
        }
        JobOpenMode::Create | JobOpenMode::Resume(_) => (),
    }
    JobPopulation::read_ndjson(reader, &request.key, request.job_id, request.limits,
        transport, control).map(Population::Original)
}

// Shared ordered job lifecycle, not a public fake-native injection surface.
// Production uses sealed source/extraction/scored processors + process admission.
pub(in crate::hosted) fn run_population<P: DurableBatchProcessor, C: DecodeStepControl>(request: SourceJobRequest,
    population: &Population<'_>, processor: P, control: &mut C) -> Result<JobProgress, JobRunError> {
    let SourceJobRequest { root, key, job_id, limits, mode, materialize } = request;
    let inputs = match population {
        Population::Original(original) => Some(original.borrowed_inputs()?),
        Population::Retaining { .. } | Population::Retained => None,
    };
    let supplied = || inputs.as_deref().ok_or(JobError::InvalidLimits);
    let mut runner = match (mode, population) {
        (JobOpenMode::Create, Population::Original(_)) =>
            JobRunner::create(&root, key, job_id, limits, supplied()?, processor, control)?,
        (JobOpenMode::Resume(tail), Population::Original(_)) =>
            JobRunner::resume(&root, key, job_id, limits, supplied()?, processor, tail, control)?,
        (JobOpenMode::CreateRetainingInputs(retention), Population::Retaining { reader, transport }) => {
            let reader = reader.take().ok_or(JobError::InvalidLimits)?;
            JobRunner::create_with_retained_ndjson(&root, key, job_id, limits, retention,
                reader, *transport, processor, control)?
        }
        (JobOpenMode::ResumeRetained { tail, retention }, Population::Retained) => {
            JobRunner::resume_with_retained_inputs(&root, key, job_id, limits, retention,
                processor, tail, control)?
        }
        _ => return Err(JobError::InvalidLimits.into()),
    };
    let progress = runner.run(control)?;
    if materialize { runner.materialize_ordered(control) } else { Ok(progress) }
}

#[cfg(test)] mod tests;
