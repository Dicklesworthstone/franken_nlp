//! Explicit candidate job execution; stored data never becomes stdout content.
use super::*;
use std::{io::{BufRead, BufReader}, sync::Arc};
use super::source_tasks::{Session, source_identity};
use crate::{
    candidate_cli::{jobs::{self as command, Defaults, Failure, JobArgs, RunMode}, extract as schema_cli},
    batch::{BatchCode, extract::quantized::Int8ExtractionBatchLimits, source::quantized as source_batch},
    hosted::{HostedError, corpus::{SourceCorpusConfig, jobs::{HostedJobError, JobHostLimits,
        JobOpenMode, SourceJobRequest}}},
    jobs::{JobError, JobId, JobLimits, JobProgress, JobSecret, JobWork, TailPolicy,
        population::PopulationReadLimits, runner::{JobRunError, extract::Int8ExtractionJobPlanner}},
    native_engine::decode::{DecodeCancellationKind, DecodeStepControl},
    tasks::{extract::ExtractionVocabulary, source_planning::SourceTaskPlanner},
};

const IO_BYTES: usize = 64 * 1024;
// This is an owned factory, never a result/recipe deserialized from input.
enum PreparedJob {
    Source { planner: Arc<SourceTaskPlanner>, config: SourceCorpusConfig },
    Extract(Int8ExtractionJobPlanner),
}
fn prepare(args: &JobArgs, facts: &ArtifactIdentity, ceiling: TaskBudget, defaults: Defaults,
    native: Int8ExtractionBatchLimits, control: &mut impl DecodeStepControl)
    -> Result<(PreparedJob, Arc<ExtractionVocabulary>), Failure> {
    checkpoint(control)?;
    let registry = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END)
        .map(|e| e.id).ok_or(CandidateError::Identity)?;
    let prepared = match defaults {
        Defaults::Source(defaults) => {
            if args.task == "extract" { return Err(Failure::usage("job_task_defaults")); }
            let planner = SourceTaskPlanner::pinned(controls, eos).map_err(|_| CandidateError::Planning)?;
            let identity = source_identity(facts, &planner, args.task_kind()?)?;
            source_batch::check_configuration(&planner, &identity, ceiling, args.host.planning(), defaults.as_ref())
                .map_err(|_| CandidateError::Planning)?;
            PreparedJob::Source { planner: Arc::new(planner), config: SourceCorpusConfig {
                identity, task_ceiling: ceiling, planning: args.host.planning(), defaults, native_work: native } }
        }
        Defaults::Extract(defaults) => {
            if args.task != "extract" { return Err(Failure::usage("job_task_defaults")); }
            let mut identity = candidate_identity(facts)?; identity.task_spec = "extract-v1".to_owned();
            PreparedJob::Extract(Int8ExtractionJobPlanner::pinned(controls, eos, identity, ceiling,
                schema_cli::compiler(&args.host), args.host.planning().source, defaults, native)
                .map_err(|_| CandidateError::Planning)?)
        }
    };
    checkpoint(control)?;
    let vocabulary = Arc::new(ExtractionVocabulary::pinned(controls).map_err(|_| CandidateError::Planning)?);
    checkpoint(control)?;
    Ok((prepared, vocabulary))
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), Failure> {
    match control.prefill_checkpoint(0) { Some(c) => Err(cancelled(c)), None => Ok(()) }
}
fn read_config(session: &Session, path: &std::path::Path, cap: usize) -> Result<String, Failure> {
    session.remaining()?;
    // Existing bounded regular-file primitive, not a second stdin reader.
    let mut file = crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?;
    let text = read_input(&mut file, cap)?;
    session.remaining()?; Ok(text)
}
fn host_limits(args: &JobArgs, common: &CandidateArgs, limits: Limits, session: &Session)
    -> Result<JobHostLimits, Failure> {
    Ok(JobHostLimits { native: session.native(common)?,
        transport: PopulationReadLimits { max_stream_bytes: args.max_stream_mib.checked_mul(MIB)
            .ok_or_else(|| Failure::usage("job_transport"))?, max_lines: args.max_input_lines },
        preparation_reserve_bytes: limits.preparation_bytes, io_reserve_bytes: (IO_BYTES * 2) as u64,
        journal_reserve_bytes: args.journal_memory_mib.checked_mul(MIB).ok_or_else(|| Failure::usage("job_memory"))?,
        serialization_reserve_bytes: args.serialization_memory_mib.checked_mul(MIB).ok_or_else(|| Failure::usage("job_memory"))? })
}
fn preflight_memory(host: JobHostLimits, job: JobLimits, limits: Limits) -> Result<(), Failure> {
    let buffers = host.required_buffer_bytes(job).map_err(host_failure)?;
    // Necessary bound only; the real host additionally admits KV/workspace,
    // resident metadata and guarded output. Do not call this a memory permit.
    let minimum = buffers.checked_add(limits.preparation_bytes).and_then(|n| n.checked_add(limits.weight_bytes))
        .ok_or_else(|| Failure::usage("job_memory_arithmetic"))?;
    if minimum > limits.memory_bytes {
        return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "job_population_memory" });
    }
    Ok(())
}
fn open_mode(mode: RunMode) -> JobOpenMode {
    match mode {
        RunMode::Start => JobOpenMode::Create,
        RunMode::Resume { discard_uncommitted: false } => JobOpenMode::Resume(TailPolicy::Refuse),
        RunMode::Resume { discard_uncommitted: true } => JobOpenMode::Resume(TailPolicy::DiscardUncommitted),
    }
}

pub(in crate::candidate_cli) fn execute<R: Read + Send + 'static>(mode: RunMode, args: JobArgs,
    common: CandidateArgs, limits: Limits, input: R, output: &mut impl Write) -> Result<(), Failure> {
    // Declared first: preparation/input/factory/result storage drops BEFORE its
    // charge. No key/config/model IO occurs before this preparation admission.
    let session = Session::new(&common, limits)?;
    let lifetime = command::parse_limits(&read_config(&session, &args.limits_file, command::LIMIT_BYTES)?)?;
    let native = command::native_limits(&args, lifetime)?;
    preflight_memory(host_limits(&args, &common, limits, &session)?, lifetime, limits)?;
    let key = JobSecret::read(&args.key_file).map_err(job_failure)?;
    session.remaining()?;
    let defaults = args.defaults.as_ref().map(|path| read_config(&session, path, command::DEFAULT_BYTES)).transpose()?;
    let schema = args.schema.as_ref().map(|path| read_config(&session, path, schema_cli::SCHEMA_BYTES)).transpose()?;
    let ceiling = args.host.task_budget(limits);
    let defaults = command::load_defaults(&args, ceiling, defaults.as_deref(), schema)?;
    let facts = session.facts(&common)?;
    let (prepared, vocabulary) = prepare(&args, &facts, ceiling, defaults, native, &mut session.control())?;
    // Only owned IO crosses the worker boundary; root CLI must hold no stdio
    // locks. This does NOT persist the original population for future resume.
    let reader: Box<dyn BufRead + Send> = if args.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BYTES,
            crate::local_io::open_document(&args.input).map_err(|_| CandidateError::Input)?))
    };
    let cancellation = CancellationToken::default();
    let model = session.load(&common, limits, &facts, cancellation.clone())?;
    let host = host_limits(&args, &common, limits, &session)?;
    let job_id = args.job_id; let materialize = args.materialize;
    let request = SourceJobRequest { root: args.job_dir, key, job_id, limits: lifetime,
        mode: open_mode(mode), materialize };
    // Ingestion and complete typed-population authentication happen inside the
    // existing host BEFORE job-file access/native forwards, but AFTER weights
    // are resident. Committed results are skipped by the unchanged JobRunner.
    let progress = match prepared {
        PreparedJob::Source { planner, config } => session.engine.job_int8_source(&model, planner,
            vocabulary, config, request, host, reader, cancellation),
        PreparedJob::Extract(planner) => session.engine.job_int8_extract(&model, planner,
            vocabulary, request, host, reader, cancellation),
    }.map_err(host_failure)?;
    session.remaining()?;
    let report = progress_report(progress, job_id, lifetime, materialize, mode, &args.task)?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: &report };
    // Metadata only. All persisted private results stay in their explicit
    // protected spool/materialization. Failed delivery never implies rollback.
    publish(&response, command::REPORT_BYTES, output).map_err(Failure::from)
}

#[derive(Serialize)]
struct ProgressReport<'a> {
    operation: &'static str, task: &'a str, status: &'static str, job_id: String,
    items: u64, committed: u64, attempts: u64, reserved_work: JobWork,
    committed_spool_bytes: u64, materialized: bool,
}
fn progress_report(p: JobProgress, expected: JobId, limits: JobLimits, materialize: bool,
    mode: RunMode, task: &str) -> Result<ProgressReport<'_>, Failure> {
    if p.job_id != expected || p.items == 0 || p.items > limits.max_items || p.committed != p.items
        || p.attempts < p.committed || p.attempts > limits.max_attempts
        || !p.reserved_work.fits(limits.max_work) || p.spool_bytes > limits.max_spool_bytes
        || materialize && !p.materialized {
        return Err(Failure { exit: ErrorCode::Generic, code: "job_completion_contract" });
    }
    Ok(ProgressReport { operation: mode.name(), task, status: "complete", job_id: command::job_id_hex(p.job_id),
        items: p.items, committed: p.committed, attempts: p.attempts, reserved_work: p.reserved_work,
        committed_spool_bytes: p.spool_bytes, materialized: p.materialized })
}
fn cancelled(cause: DecodeCancellationKind) -> Failure {
    let budget = matches!(cause, DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout
        | DecodeCancellationKind::PollQuota | DecodeCancellationKind::CostBudget);
    Failure { exit: if budget { ErrorCode::BudgetOrTimeout } else { ErrorCode::Cancelled },
        code: if budget { "job_execution_budget" } else { "job_cancelled" } }
}
fn job_failure(error: JobError) -> Failure {
    let (exit, code) = match error {
        JobError::Cancelled(c) => return cancelled(c),
        JobError::Authentication | JobError::Corrupt => (ErrorCode::ArtifactIntegrityOrFormatOrVersion, "job_integrity"),
        JobError::Mismatch(_) => (ErrorCode::ArtifactIntegrityOrFormatOrVersion, "job_contract_mismatch"),
        JobError::UnsafeStorage => (ErrorCode::AdmissionOrResourceLimit, "unsafe_job_storage"),
        JobError::Busy => (ErrorCode::AdmissionOrResourceLimit, "job_busy"),
        JobError::AlreadyExists => (ErrorCode::AdmissionOrResourceLimit, "job_already_exists"),
        JobError::InvalidLimits => (ErrorCode::Usage, "job_limits"),
        JobError::InvalidInput | JobError::DuplicateId | JobError::InvalidIdentity => (ErrorCode::InputDecodeOrParse, "job_population"),
        JobError::WorkLimit => (ErrorCode::BudgetOrTimeout, "job_lifetime_work_or_attempts"),
        JobError::Limit | JobError::Allocation | JobError::Platform => (ErrorCode::AdmissionOrResourceLimit, "job_resource_limit"),
        JobError::UncommittedTail => (ErrorCode::StructuredTaskNoResult, "job_uncommitted_tail_or_stage"),
        JobError::PublicationUncertain => (ErrorCode::Generic, "job_publication_uncertain"),
        _ => (ErrorCode::Generic, "job_storage_or_state"),
    };
    Failure { exit, code }
}
fn host_failure(error: HostedJobError) -> Failure {
    match error {
        HostedJobError::Job(JobRunError::Storage(e)) => job_failure(e),
        HostedJobError::Job(JobRunError::InvalidEnvelope) => Failure { exit: ErrorCode::InputDecodeOrParse, code: "job_envelope" },
        HostedJobError::Job(JobRunError::Processor(fault)) => {
            if let Some(c) = fault.cancellation { return cancelled(c); }
            let exit = match fault.code {
                BatchCode::WorkLimit => ErrorCode::BudgetOrTimeout,
                BatchCode::Admission | BatchCode::Allocation => ErrorCode::AdmissionOrResourceLimit,
                _ => ErrorCode::StructuredTaskNoResult,
            };
            Failure { exit, code: "job_task_failed" }
        }
        HostedJobError::Host(HostedError::Stopped { stop, .. }) => cancelled(stop.kind),
        HostedJobError::Host(HostedError::Reservation(_) | HostedError::Limits(_) | HostedError::Reentrant
            | HostedError::MissingRuntimeContext | HostedError::SingleCoordinatorRequired) =>
            Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "job_host_admission" },
        _ => Failure { exit: ErrorCode::Generic, code: "job_runtime_or_completion" },
    }
}

#[cfg(test)] mod tests;
