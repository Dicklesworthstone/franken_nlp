//! Complete sampled documents through the existing authenticated INT8 job host.
//! The limits file is the only IO before complete population-memory admission.
use super::*;
use std::cell::Cell;
use crate::{
    batch::{BatchCode, BatchFault},
    candidate_cli::{batch::IO_BUFFER_BYTES, jobs::{self as job_cli, RunMode},
        sampled_sentiment::map::job::{JobArgs, JobExecutionLimits, storage_failure},
        runtime::owned_jobs::{self as owned, open_mode, progress_report}},
    hosted::{int8_sampled_sentiment_document_job_requirements,
        corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest}},
    jobs::{JobError, JobLimits, JobSecret,
        runner::{JobRunError, sampled_sentiment::quantized::long::Int8SampledSentimentDocumentJobPlanner}},
    native_engine::strict_int8::prefill::{Int8PrefillLimits, LAYER_MAJOR_PREFILL_VERSION},
};

pub(in crate::candidate_cli) fn execute<R: Read + Send + 'static>(mode: RunMode, job: JobArgs,
    _admitted: MapLimits, input: R, output: &mut impl Write) -> Result<(), Failure> {
    // Revalidate the actual configuration even if an internal caller supplied
    // a stale quote. No key, artifact, population or output has been touched.
    let admitted = job.validate()?;
    job.storage.input_retention.validate(mode, &admitted.common.input).map_err(storage_failure)?;
    // Declared first: every compiler, vocabulary, population, held model file
    // and report allocation drops before this caller preparation reservation.
    let session = Session::new_with_prefill(&admitted.common, admitted.shared, admitted.prefill)
        .map_err(setup_failure)?;
    let check = || session.remaining().map(|_| ());
    let lifetime = checked_stage(read_config(&job.storage.limits_file, &check)
        .and_then(|text| job_cli::parse_limits(&text).map_err(storage_failure)), &check, |error| error)?;
    let limits = checked_stage(job.lifetime_limits(admitted, lifetime), &check, |error| error)?;
    let document = &limits.document;
    checked_stage(check_host_quote(&job, &limits, lifetime,
        session.remaining().map_err(setup_failure)?), &check, |error| error)?;
    let options = checked_stage(job.document.sampled.options(), &check, setup_failure)?;
    let cancellation = CancellationToken::default();
    let key = checked_stage(JobSecret::read(&job.storage.key_file), &check, storage_error)?;
    let facts = checked_stage(session.facts_with_error_detail(&document.common), &check, session_failure)?;
    let compiler = checked_stage(Int8SampledSentimentPlanner::for_model_with_encoding(&facts,
        options, document.task, limits.planning, document.encoding.clone()), &check, task_failure)?;
    let compiler = checked_stage(job.document.sampled.configure_planner(compiler), &check, setup_failure)?;
    let vocabulary = Arc::new(checked_stage(SampledSentimentVocabulary::pinned(), &check,
        |error| task_failure(error.into()))?);
    let planner = checked_stage(Int8SampledSentimentDocumentJobPlanner::new(compiler, vocabulary,
        document.mapping, limits.native, lifetime), &check, batch_failure)?;
    let planner = checked_stage(match document.prefill {
        Some(prefill) => planner.with_layer_major_prefill(prefill),
        None => Ok(planner),
    }, &check, batch_failure)?;
    checked_stage(planner.check_job_limits(lifetime), &check, batch_failure)?;
    checked_stage(if planner.preparation_requirement_bytes() != limits.preparation_reserve_bytes
        || planner.max_result_bytes() != document.memory.max_output_bytes
        || planner.document_encoding() != &document.encoding
        || planner.selected_rows() != checked_stage(job.document.sampled.selected_rows(), &check, setup_failure)?
        || planner.task_budget() != document.task
        || planner.prefill_limits().map(|p| (p.max_batch_rows, p.max_extra_scratch_bytes))
            != document.prefill.map(|p| (p.max_batch_rows, p.max_extra_scratch_bytes)) {
        Err(Failure::output("int8_sampled_document_job_planner_quote"))
    } else { Ok(()) }, &check, |error| error)?;
    let identity = planner.base_execution_identity().clone();
    let model = checked_stage(session.load_with_error_detail(&document.common, document.shared, &facts, cancellation.clone()),
        &check, |error| cancellation.cause().map(cancelled).unwrap_or_else(|| session_failure(error)))?;
    let reader = checked_stage(job_cli::population_reader(mode, job.storage.input_retention,
        &document.common.input, input, IO_BUFFER_BYTES), &check, storage_failure)?;
    let job_id = job.storage.job_id;
    let materialize = job.storage.materialize;
    let open = checked_stage(open_mode(mode, job.storage.input_retention), &check, storage_failure)?;
    let request = SourceJobRequest { root: job.storage.job_dir, key, job_id,
        limits: lifetime, mode: open, materialize };
    let host = host_limits(&limits, session.remaining().map_err(setup_failure)?);
    let progress = checked_stage(if document.shared.compute.threads == 1 {
        session.engine.job_int8_sampled_sentiment_documents(&model, planner, request,
            host, reader, cancellation.clone())
    } else {
        session.engine.job_int8_sampled_sentiment_documents_with_threads(&model, planner, request,
            host, crate::native_engine::simd::config::process_kernel(),
            document.shared.compute.threads, reader, cancellation.clone())
    }, &check, job_failure)?;
    if let Some(cause) = cancellation.cause() { return Err(cancelled(cause)); }
    let report = checked_stage(progress_report(progress, job_id, lifetime, materialize, mode,
        "sentiment-sampled-map"), &check, storage_failure)?;
    let response = Response {
        protocol: "fnlp-candidate-int8-sentiment-sampled-document-job-v1",
        schema_version: 1, scope: "real-artifact-current-candidate", evidence: "non_authoritative",
        task: "sentiment-sampled-map", durable_unit: "complete_sampled_document",
        model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        candidate_format: &identity.artifact_format, quant_recipe: &facts.recipe_id,
        input_encoding: input_encoding::reported_policy(&document.encoding),
        compute_threads: (document.shared.compute.threads > 1).then_some(document.shared.compute.threads),
        prefill_schedule: document.prefill.map(PrefillSchedule::new).transpose().map_err(setup_failure)?,
        max_selected_rows: job.document.sampled.max_selected_rows,
        output: &report,
    };
    // The host has joined every numerical worker and durable item commit.
    // Delivery can still fail: neither error nor timeout implies rollback.
    publish_report(&response, output, &|| session.remaining().map_err(setup_failure), &cancellation)
}

fn check_host_quote(job: &JobArgs, limits: &JobExecutionLimits, lifetime: JobLimits, remaining: Duration)
    -> Result<(), Failure> {
    let document = &limits.document;
    let quote = int8_sampled_sentiment_document_job_requirements(document.task, limits.planning,
        document.mapping, &job.document.sampled.options().map_err(setup_failure)?,
        &document.encoding, limits.native, lifetime, host_limits(limits, remaining),
        job.storage.input_retention.input_memory().map_err(storage_failure)?, document.prefill,
        (document.shared.compute.threads > 1).then_some(document.shared.compute.threads)).map_err(job_failure)?;
    if quote.document != document.memory || quote.document.input_bytes != limits.buffer_bytes
        || quote.team_workspace_bytes != document.shared.compute.workspace_bytes
        || quote.total_bytes().map_err(map_host_failure)? != limits.execution_bytes
        || document.minimum_memory_bytes > document.shared.memory_bytes {
        return Err(Failure::usage("int8_sampled_document_job_host_quote"));
    }
    Ok(())
}
fn host_limits(limits: &JobExecutionLimits, remaining: Duration) -> JobHostLimits {
    JobHostLimits { native: NativeLimits {
        context_tokens: limits.document.common.context_tokens, allocator_reserve_bytes: 64 * MIB,
        run: RunLimits { max_elapsed: remaining,
            max_checkpoints: limits.document.common.max_checkpoints, cleanup_reserve_bytes: 65_536 } },
        transport: limits.transport, preparation_reserve_bytes: limits.preparation_reserve_bytes,
        io_reserve_bytes: limits.io_reserve_bytes, journal_reserve_bytes: limits.journal_reserve_bytes,
        serialization_reserve_bytes: limits.serialization_reserve_bytes }
}
fn read_config(path: &std::path::Path, check: &impl Fn() -> Result<(), CandidateError>)
    -> Result<String, Failure> {
    check().map_err(setup_failure)?;
    checked_stage((|| {
        let mut file = crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?;
        read_input(&mut file, job_cli::LIMIT_BYTES)
    })(), check, setup_failure)
}
fn publish_report<T: Serialize>(response: &T, output: &mut impl Write,
    remaining: &impl Fn() -> Result<Duration, Failure>, cancellation: &CancellationToken)
    -> Result<(), Failure> {
    let first_failure = Cell::new(None);
    let result = publish_checked(response, job_cli::REPORT_BYTES, output, &|| {
        if first_failure.get().is_some() { return Err(CandidateError::Execution); }
        let failure = cancellation.cause().map(cancelled).or_else(|| remaining().err());
        match failure {
            Some(failure) => {
                first_failure.set(Some(failure));
                Err(CandidateError::Execution)
            }
            None => Ok(()),
        }
    });
    if let Some(failure) = first_failure.get() { return Err(failure); }
    // A failed write or flush still owes the invocation its terminal poll.
    // Keep an observed cancellation ahead of a later incidental IO/deadline.
    if let Some(cause) = cancellation.cause() { return Err(cancelled(cause)); }
    remaining()?;
    result.map_err(setup_failure)
}
fn storage_error(error: JobError) -> Failure {
    if let JobError::Cancelled(cause) = error { return cancelled(cause); }
    storage_failure(owned::job_failure(error))
}
fn batch_failure(fault: BatchFault) -> Failure {
    if let Some(cause) = fault.cancellation { return cancelled(cause); }
    use BatchCode::*;
    let (exit, code) = match fault.code {
        InvalidLimits => (ErrorCode::Usage, "int8_sampled_document_job_limits"),
        InvalidUtf8 | InvalidJson | InvalidEnvelope | InvalidId | DuplicateId | Planning | InputIo =>
            (ErrorCode::InputDecodeOrParse, "int8_sampled_document_job_input"),
        LineLimit | DocumentLimit | EpochIdLimit | InputLimit | RequestLimit
            | OutputLineLimit | OutputLimit | WorkLimit =>
            (ErrorCode::BudgetOrTimeout, "int8_sampled_document_job_budget"),
        Admission | Allocation => (ErrorCode::AdmissionOrResourceLimit, "int8_sampled_document_job_admission"),
        Execution => (ErrorCode::StructuredTaskNoResult, "int8_sampled_document_job_no_result"),
        Cancelled => (ErrorCode::Generic, "int8_sampled_document_job_missing_cancellation"),
        SequenceOverflow | InvalidExecution | Serialization | OutputIo =>
            (ErrorCode::Generic, "int8_sampled_document_job_runtime_or_delivery"),
    };
    Failure { exit, code, cancellation: None }
}
fn job_failure(error: HostedJobError) -> Failure {
    match error {
        HostedJobError::Host(HostedError::BatchSetup(fault)) => batch_failure(fault),
        HostedJobError::Host(error) => map_host_failure(error),
        HostedJobError::Job(JobRunError::Storage(error)) => storage_error(error),
        HostedJobError::Job(JobRunError::Processor(fault)) => batch_failure(fault),
        HostedJobError::Job(JobRunError::InvalidEnvelope) => Failure {
            exit: ErrorCode::InputDecodeOrParse, code: "int8_sampled_document_job_envelope", cancellation: None },
        HostedJobError::Job(JobRunError::WorkContract | JobRunError::Stopped) =>
            Failure::output("int8_sampled_document_job_completion_invariant"),
    }
}

#[derive(Serialize)]
struct PrefillSchedule { execution: &'static str, max_batch_rows: usize, admitted_workspace_bytes: u64 }
impl PrefillSchedule {
    fn new(limits: Int8PrefillLimits) -> Result<Self, CandidateError> {
        Ok(Self { execution: LAYER_MAJOR_PREFILL_VERSION, max_batch_rows: limits.max_batch_rows,
            admitted_workspace_bytes: limits.validate().map_err(|_| CandidateError::Arguments)? })
    }
}
#[derive(Serialize)]
struct Response<'a, T: Serialize> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str,
    task: &'static str, durable_unit: &'static str, model_id: &'a str,
    source_revision: &'a str, source_root_sha256: &'a str, logical_model_sha256: &'a str,
    candidate_format: &'a str, quant_recipe: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_threads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_schedule: Option<PrefillSchedule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_selected_rows: Option<usize>,
    output: &'a T,
}

#[cfg(test)] mod tests;
