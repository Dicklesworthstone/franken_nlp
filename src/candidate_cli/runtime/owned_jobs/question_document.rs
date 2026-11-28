//! Complete original source/question jobs through the resident INT8 owner.
//! Only bounded limits/options are read before the independent whole-job quote.
use std::{cell::Cell, io::{Read, Write}, sync::Arc, time::Duration};
use serde::Serialize;
use crate::{
    batch::source::quantized::question::Int8QuestionDocumentBatchPlanner,
    candidate_cli::{CandidateError, MIB, read_input, publish_checked,
        batch::IO_BUFFER_BYTES, input_encoding, source,
        jobs::{self as job_cli, RunMode, question_document::{JobArgs, JobExecutionLimits, storage_failure}},
        question_corpus::{self as question, QuestionDocumentLimits},
        runtime::{owned_jobs::{self as owned, open_mode, progress_report},
            question_corpus::{setup_failure, session_failure, host_failure, batch_failure,
                checked_stage, cancelled}, source_tasks::{Session, source_identity}}, stream::Failure},
    error::ErrorCode,
    hosted::{CancellationToken, HostedError, NativeLimits, RunLimits,
        int8_question_document_job_requirements,
        corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest}},
    jobs::{JobError, JobLimits, JobSecret,
        runner::{JobRunError, source::question_document::Int8QuestionDocumentJobPlanner}},
    native_engine::strict_int8::prefill::{Int8PrefillLimits, LAYER_MAJOR_PREFILL_VERSION},
    tasks::{BuiltInTask, extract::ExtractionVocabulary, source_planning::{SourceTaskPlanner, SourcePlanningError},
        source_planning::quantized::long::question::INT8_QUESTION_EXECUTION},
    tokenizer::{pinned_controls, untrusted::{SafeBpeWorkLimits, UntrustedEncoding}},
};

pub(in crate::candidate_cli) fn execute<R: Read + Send + 'static>(mode: RunMode, job: JobArgs,
    _admitted: QuestionDocumentLimits, input: R, output: &mut impl Write) -> Result<(), Failure> {
    let admitted = job.validate()?;
    job.storage.input_retention.validate(mode, &job.input).map_err(storage_failure)?;
    // This first owner outlives bounded configuration, compiler, vocabulary,
    // authenticated population, model and final report delivery on every exit.
    let session = Session::new_with_prefill(&admitted.common, admitted.shared, admitted.prefill)
        .map_err(setup_failure)?;
    let check = || session.remaining().map(|_| ());
    let lifetime = checked_stage(read_config(&job.storage.limits_file, job_cli::LIMIT_BYTES, &check)
        .and_then(|text| job_cli::parse_limits(&text).map_err(storage_failure)), &check, |error| error)?;
    let options = match &job.document.options {
        Some(path) => {
            let text = read_config(path, source::OPTIONS_BYTES, &check)?;
            checked_stage(question::parse_options(Some(&text)), &check, setup_failure)?
        }
        None => checked_stage(question::parse_options(None), &check, setup_failure)?,
    };
    let limits = checked_stage(job.lifetime_limits(admitted, lifetime, options), &check, |error| error)?;
    checked_stage(check_host_quote(&job, &limits, lifetime,
        session.remaining().map_err(setup_failure)?), &check, |error| error)?;
    let document = &limits.document;
    let cancellation = CancellationToken::default();
    let key = checked_stage(JobSecret::read(&job.storage.key_file), &check, storage_error)?;
    let facts = checked_stage(session.facts_with_error_detail(&document.common), &check, session_failure)?;
    let registry = checked_stage(pinned_controls::pinned(), &check,
        |_| setup_failure(CandidateError::Identity))?;
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|entry| entry.special && entry.surface == crate::template::IM_END)
        .map(|entry| entry.id).ok_or_else(|| setup_failure(CandidateError::Identity))?;
    let compiler = checked_stage(SourceTaskPlanner::pinned(controls, eos), &check, planning_failure)?;
    let identity = checked_stage(source_identity(&facts, &compiler, BuiltInTask::Answer), &check, setup_failure)?;
    let batch = checked_stage(Int8QuestionDocumentBatchPlanner::for_model(&facts,
        Arc::new(compiler), identity, limits.config), &check, batch_failure)?;
    let planner = checked_stage(Int8QuestionDocumentJobPlanner::new(batch, lifetime), &check, batch_failure)?;
    let planner = checked_stage(match document.encoding_work {
        Some(total) => planner.with_map_document_encoding(document.encoding.clone(), total),
        None => Ok(planner),
    }, &check, batch_failure)?;
    let planner = checked_stage(match document.prefill {
        Some(prefill) => planner.with_layer_major_prefill(prefill),
        None => Ok(planner),
    }, &check, batch_failure)?;
    checked_stage(check_planner_quote(&planner, &limits, lifetime), &check, |error| error)?;
    let vocabulary = Arc::new(checked_stage(ExtractionVocabulary::pinned(controls), &check,
        |_| setup_failure(CandidateError::Planning))?);
    let identity = planner.execution_identity().clone();
    let model = checked_stage(session.load_with_error_detail(&document.common, document.shared,
        &facts, cancellation.clone()), &check,
        |error| cancellation.cause().map(cancelled).unwrap_or_else(|| session_failure(error)))?;
    let reader = checked_stage(job_cli::population_reader(mode, job.storage.input_retention,
        &job.input, input, IO_BUFFER_BYTES), &check, storage_failure)?;
    let job_id = job.storage.job_id;
    let materialize = job.storage.materialize;
    let open = checked_stage(open_mode(mode, job.storage.input_retention), &check, storage_failure)?;
    let request = SourceJobRequest { root: job.storage.job_dir, key, job_id,
        limits: lifetime, mode: open, materialize };
    let host = host_limits(&limits, session.remaining().map_err(setup_failure)?);
    let progress = checked_stage(if document.shared.compute.threads == 1 {
        session.engine.job_int8_question_documents(&model, planner, vocabulary, request,
            host, document.reduction_reserve_bytes, reader, cancellation.clone())
    } else {
        session.engine.job_int8_question_documents_with_threads(&model, planner, vocabulary, request,
            host, document.reduction_reserve_bytes, crate::native_engine::simd::config::process_kernel(),
            document.shared.compute.threads, reader, cancellation.clone())
    }, &check, job_failure)?;
    if let Some(cause) = cancellation.cause() { return Err(cancelled(cause)); }
    let report = checked_stage(progress_report(progress, job_id, lifetime, materialize, mode, "answer"),
        &check, storage_failure)?;
    let response = Response {
        protocol: "fnlp-candidate-int8-question-document-job-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative", task: "answer",
        execution: INT8_QUESTION_EXECUTION, durable_unit: "complete_original_source_question",
        model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        candidate_format: &identity.artifact_format, quant_recipe: &facts.recipe_id,
        input_encoding: input_encoding::reported_policy(&document.encoding),
        document_encoding_work_limits: document.encoding_work,
        document_encoding_work_scope: document.encoding_work.map(|_|
            "one-original-source-question-prepare-and-execute-reconstruction-v1"),
        compute_threads: (document.shared.compute.threads > 1).then_some(document.shared.compute.threads),
        prefill_schedule: document.prefill.map(PrefillSchedule::new).transpose().map_err(setup_failure)?,
        output: &report,
    };
    // Durable progress may already exist when external report delivery fails.
    // The original invocation still owns its terminal cancellation/deadline.
    publish_report(&response, output, &|| session.remaining().map_err(setup_failure), &cancellation)
}

fn check_host_quote(job: &JobArgs, limits: &JobExecutionLimits, lifetime: JobLimits, remaining: Duration)
    -> Result<(), Failure> {
    let document = &limits.document;
    let quote = int8_question_document_job_requirements(limits.config, lifetime,
        host_limits(limits, remaining), job.storage.input_retention.input_memory().map_err(storage_failure)?,
        document.reduction_reserve_bytes, document.prefill,
        (document.shared.compute.threads > 1).then_some(document.shared.compute.threads)).map_err(job_failure)?;
    if quote.document != limits.memory.document || quote.population_buffer_bytes != limits.memory.population_buffer_bytes
        || quote.team_workspace_bytes != document.shared.compute.workspace_bytes
        || quote.total_bytes().map_err(host_failure)? != limits.execution_bytes
        || document.shared.compute != document.common.compute.admit_prefill(document.prefill).map_err(setup_failure)?
        || limits.minimum_memory_bytes > document.shared.memory_bytes {
        return Err(Failure::usage("int8_question_document_job_host_quote"));
    }
    Ok(())
}

fn check_planner_quote(planner: &Int8QuestionDocumentJobPlanner, limits: &JobExecutionLimits,
    lifetime: JobLimits) -> Result<(), Failure> {
    let document = &limits.document;
    planner.check_job_limits(lifetime).map_err(batch_failure)?;
    let workspace = match &document.encoding {
        UntrustedEncoding::ByteFallback => 0,
        UntrustedEncoding::SafeBpeV1(call) => call.requirements().map_err(|_| Failure::usage("question_job_encoding"))?
            .workspace_bytes.checked_add(1024).ok_or_else(|| Failure::usage("question_job_encoding"))?,
    };
    if planner.document_encoding() != &document.encoding
        || planner.map_document_encoding_limits() != document.encoding_work
        || planner.encoding_workspace_bytes().map_err(batch_failure)? != workspace
        || workspace > document.shared.preparation_bytes
        || planner.task_budget() != document.task
        || planner.max_result_bytes() != limits.memory.document.max_output_bytes
        || planner.max_output_tokens() != limits.memory.document.max_output_tokens
        || planner.prefill_limits().map(|p| (p.max_batch_rows, p.max_extra_scratch_bytes))
            != document.prefill.map(|p| (p.max_batch_rows, p.max_extra_scratch_bytes)) {
        return Err(Failure::usage("int8_question_document_job_planner_quote"));
    }
    Ok(())
}

fn host_limits(limits: &JobExecutionLimits, remaining: Duration) -> JobHostLimits {
    JobHostLimits { native: NativeLimits { context_tokens: limits.document.common.context_tokens,
        allocator_reserve_bytes: 64 * MIB, run: RunLimits { max_elapsed: remaining,
            max_checkpoints: limits.document.common.max_checkpoints, cleanup_reserve_bytes: 65_536 } },
        transport: limits.transport, preparation_reserve_bytes: limits.document.shared.preparation_bytes,
        io_reserve_bytes: limits.io_reserve_bytes, journal_reserve_bytes: limits.journal_reserve_bytes,
        serialization_reserve_bytes: limits.serialization_reserve_bytes }
}
fn read_config(path: &std::path::Path, cap: usize, check: &impl Fn() -> Result<(), CandidateError>)
    -> Result<String, Failure> {
    check().map_err(setup_failure)?;
    checked_stage((|| {
        let mut file = crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?;
        read_input(&mut file, cap)
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
            Some(failure) => { first_failure.set(Some(failure)); Err(CandidateError::Execution) }
            None => Ok(()),
        }
    });
    if let Some(failure) = first_failure.get() { return Err(failure); }
    if let Some(cause) = cancellation.cause() { return Err(cancelled(cause)); }
    remaining()?;
    result.map_err(setup_failure)
}
fn planning_failure(error: SourcePlanningError) -> Failure {
    error.cancellation().map(cancelled).unwrap_or_else(|| setup_failure(CandidateError::Planning))
}
fn storage_error(error: JobError) -> Failure {
    if let JobError::Cancelled(cause) = error { return cancelled(cause); }
    storage_failure(owned::job_failure(error))
}
fn job_failure(error: HostedJobError) -> Failure {
    match error {
        HostedJobError::Host(HostedError::BatchSetup(fault)) => batch_failure(fault),
        HostedJobError::Host(error) => host_failure(error),
        HostedJobError::Job(JobRunError::Storage(error)) => storage_error(error),
        HostedJobError::Job(JobRunError::Processor(fault)) => batch_failure(fault),
        HostedJobError::Job(JobRunError::InvalidEnvelope) => Failure {
            exit: ErrorCode::InputDecodeOrParse, code: "int8_question_document_job_envelope", cancellation: None },
        HostedJobError::Job(JobRunError::WorkContract | JobRunError::Stopped) =>
            Failure::output("int8_question_document_job_completion_invariant"),
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
    task: &'static str, execution: &'static str, durable_unit: &'static str, model_id: &'a str,
    source_revision: &'a str, source_root_sha256: &'a str, logical_model_sha256: &'a str,
    candidate_format: &'a str, quant_recipe: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    document_encoding_work_limits: Option<SafeBpeWorkLimits>,
    #[serde(skip_serializing_if = "Option::is_none")]
    document_encoding_work_scope: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_threads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_schedule: Option<PrefillSchedule>,
    output: &'a T,
}

#[cfg(test)] mod tests;
