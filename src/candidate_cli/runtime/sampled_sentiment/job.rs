//! Authenticated INT8 sampled jobs use the existing private storage lifecycle.
use super::*;
use std::cell::Cell;
use super::super::{owned_jobs, source_tasks::SessionStageError};
use crate::{
    candidate_cli::{jobs::{self as job_cli, RunMode},
        sampled_sentiment::job::{Envelope, JobArgs, storage_failure}},
    hosted::corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest,
        Int8SampledSentimentJobRequirements, int8_sampled_sentiment_job_requirements},
    jobs::{JobError, JobLimits, JobSecret, population::PopulationReadLimits,
        runner::{JobRunError, sampled_sentiment::quantized::{
            Int8SampledSentimentJobLimits, Int8SampledSentimentJobPlanner}}},
    native_engine::strict_int8::prefill::Int8PrefillLimits,
    tokenizer::untrusted::SafeBpeWorkLimits,
};

fn native_limits(envelope: Envelope, job: JobLimits) -> Int8SampledSentimentJobLimits {
    Int8SampledSentimentJobLimits { max_item_work: envelope.max_item_work,
        max_model_work: job.max_work.model, masks: envelope.masks,
        max_sampler_bytes: envelope.sampler_bytes }
}
fn host_limits(args: &JobArgs, common: &CandidateArgs, limits: Limits, session: &Session)
    -> Result<JobHostLimits, Failure> {
    let storage = &args.storage;
    let bytes = |mib: u64| mib.checked_mul(MIB)
        .ok_or_else(|| Failure::usage("int8_sampled_sentiment_job_memory"));
    Ok(JobHostLimits { native: session.native(common).map_err(setup_failure)?,
        transport: PopulationReadLimits { max_stream_bytes: bytes(storage.max_stream_mib)?,
            max_lines: storage.max_input_lines },
        preparation_reserve_bytes: limits.preparation_bytes,
        io_reserve_bytes: (2 * crate::candidate_cli::batch::IO_BUFFER_BYTES) as u64,
        journal_reserve_bytes: bytes(storage.journal_memory_mib)?,
        serialization_reserve_bytes: bytes(storage.serialization_memory_mib)? })
}
fn quote(args: &JobArgs, envelope: Envelope, job: JobLimits, limits: Limits, host: JobHostLimits)
    -> Result<Int8SampledSentimentJobRequirements, Failure> {
    let prefill = args.sampled.prefill().map_err(setup_failure)?;
    let required = int8_sampled_sentiment_job_requirements(envelope.task, native_limits(envelope, job),
        envelope.preparation_bytes, job, host,
        args.storage.input_retention.input_memory().map_err(storage_failure)?, prefill,
        (limits.compute.threads > 1).then_some(limits.compute.threads)).map_err(failure)?;
    if required.native.kv_bytes != limits.kv_bytes || required.team_workspace_bytes != limits.compute.workspace_bytes {
        return Err(Failure::usage("int8_sampled_sentiment_job_quote_mismatch"));
    }
    Ok(required)
}
fn minimum_memory(limits: Limits, required: Int8SampledSentimentJobRequirements) -> Result<u64, Failure> {
    // The pure host quote includes transferred preparation and every native,
    // sampler, prompt, worker, population, journal and output owner exactly once.
    // Resident weights/metadata, caller preparation and runtime cleanup coexist.
    [limits.weight_bytes, 128 * MIB, 64 * MIB, limits.preparation_bytes,
        required.total_bytes().map_err(host_failure)?, 65_536, 64 * 1024]
        .into_iter().try_fold(0_u64, |total, bytes| total.checked_add(bytes)
            .ok_or_else(|| Failure::usage("int8_sampled_sentiment_job_memory_arithmetic")))
}
fn check_memory(limits: Limits, required: Int8SampledSentimentJobRequirements) -> Result<(), Failure> {
    if minimum_memory(limits, required)? > limits.memory_bytes {
        return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit,
            code: "int8_sampled_sentiment_job_population_memory", cancellation: None });
    }
    Ok(())
}
fn prepare(args: &JobArgs, common: &CandidateArgs, limits: Limits, job: JobLimits, envelope: Envelope,
    facts: &ArtifactIdentity, vocabulary: Arc<SampledSentimentVocabulary>)
    -> Result<Int8SampledSentimentJobPlanner, Failure> {
    let encoding = common.input_encoding_policy(limits).map_err(setup_failure)?;
    let totals = args.encoding_work_limits()?;
    let prefill = args.sampled.prefill().map_err(setup_failure)?;
    let compiler = Int8SampledSentimentPlanner::for_model_with_encoding(facts,
        args.sampled.options().map_err(setup_failure)?, envelope.task, envelope.planning, encoding.clone())
        .map_err(task_failure)?;
    let compiler = args.sampled.configure_planner(compiler).map_err(setup_failure)?;
    let factory = Int8SampledSentimentJobPlanner::new_with_document_encoding(
        compiler, vocabulary, native_limits(envelope, job), job, totals).map_err(corpus::failure)?;
    let factory = match prefill {
        Some(prefill) => factory.with_layer_major_prefill(prefill).map_err(corpus::failure)?,
        None => factory,
    };
    factory.check_job_limits(job).map_err(corpus::failure)?;
    let schedule = |value: Option<Int8PrefillLimits>| value.map(|p| (p.max_batch_rows, p.max_extra_scratch_bytes));
    if factory.document_encoding() != &encoding || factory.encoding_work_limits() != totals
        || factory.selected_rows() != args.sampled.selected_rows().map_err(setup_failure)?
        || factory.preparation_requirement_bytes() != envelope.preparation_bytes
        || factory.task_budget() != envelope.task
        || factory.max_result_bytes() != envelope.task.max_output_bytes.min(envelope.planning.max_result_bytes)
        || schedule(factory.prefill_limits()) != schedule(prefill) {
        return Err(Failure::usage("int8_sampled_sentiment_job_factory_mismatch"));
    }
    Ok(factory)
}
fn read_limits(session: &Session, path: &std::path::Path) -> Result<String, CandidateError> {
    let mut file = crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?;
    let text = read_input(&mut file, job_cli::LIMIT_BYTES)?;
    session.remaining()?;
    Ok(text)
}

pub(in crate::candidate_cli) fn execute<R: Read + Send + 'static>(mode: RunMode, args: JobArgs,
    common: CandidateArgs, limits: Limits, input: R, output: &mut impl Write) -> Result<(), Failure> {
    args.storage.input_retention.validate(mode, &common.input).map_err(storage_failure)?;
    let (verified, expected) = args.validate()?;
    if expected != limits || verified.compute != common.compute || verified.input_encoding != common.input_encoding
        || verified.input != common.input || verified.model != common.model
        || verified.context_tokens != common.context_tokens || verified.timeout_seconds != common.timeout_seconds
        || verified.max_checkpoints != common.max_checkpoints || verified.max_input_bytes != common.max_input_bytes {
        return Err(Failure::usage("int8_sampled_sentiment_job_admission_mismatch"));
    }
    let prefill = args.sampled.prefill().map_err(setup_failure)?;
    let encoding = common.input_encoding_policy(limits).map_err(setup_failure)?;
    let totals = args.encoding_work_limits()?;
    // Declared before all private configuration/planner/report owners so the
    // independently charged CLI preparation remains live through final delivery.
    let session = Session::new_with_prefill(&common, limits, prefill).map_err(setup_failure)?;
    let check = || session.remaining().map(|_| ());
    let raw = checked_stage(read_limits(&session, &args.storage.limits_file), &check, setup_failure)?;
    let lifetime = checked_stage(job_cli::parse_limits(&raw), &check, storage_failure)?;
    let envelope = checked_stage(args.lifetime_limits(limits, lifetime), &check, |error| error)?;
    let host = checked_stage(host_limits(&args, &common, limits, &session), &check, |error| error)?;
    let required = checked_stage(quote(&args, envelope, lifetime, limits, host), &check, |error| error)?;
    checked_stage(check_memory(limits, required), &check, |error| error)?;
    // Only the bounded limits file was read before the complete population,
    // native and storage quote passed. No key, model or population fallback.
    let key = checked_stage(JobSecret::read(&args.storage.key_file), &check, storage_error)?;
    let facts = checked_stage(session.facts_with_error_detail(&common), &check, stage_failure)?;
    let vocabulary = Arc::new(checked_stage(SampledSentimentVocabulary::pinned(), &check,
        |error| task_failure(error.into()))?);
    let factory = Arc::new(checked_stage(prepare(&args, &common, limits, lifetime, envelope,
        &facts, vocabulary), &check, |error| error)?);
    let reader = checked_stage(job_cli::population_reader(mode, args.storage.input_retention,
        &common.input, input, crate::candidate_cli::batch::IO_BUFFER_BYTES), &check, storage_failure)?;
    let cancellation = CancellationToken::default();
    let model = checked_stage(session.load_with_error_detail(&common, limits, &facts, cancellation.clone()), &check,
        |error| {
            let failure = stage_failure(error);
            if failure.cancellation.is_some() { failure }
            else { cancellation.cause().map(cancelled).unwrap_or(failure) }
        })?;
    let host = checked_stage(host_limits(&args, &common, limits, &session), &check, |error| error)?;
    let job_id = args.storage.job_id;
    let materialize = args.storage.materialize;
    let request = SourceJobRequest { root: args.storage.job_dir, key, job_id, limits: lifetime,
        mode: owned_jobs::open_mode(mode, args.storage.input_retention).map_err(storage_failure)?, materialize };
    let progress = checked_stage(if limits.compute.threads == 1 {
        session.engine.job_int8_sampled_sentiment(&model, factory, request, host, reader, cancellation.clone())
    } else {
        session.engine.job_int8_sampled_sentiment_with_threads(&model, factory, request, host,
            crate::native_engine::simd::config::process_kernel(), limits.compute.threads, reader, cancellation.clone())
    }, &check, failure)?;
    if let Some(cause) = cancellation.cause() { return Err(cancelled(cause)); }
    let report = checked_stage(owned_jobs::progress_report(progress, job_id, lifetime, materialize,
        mode, "sentiment-sampled"), &check, storage_failure)?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: &report };
    let response = Response { response: &response, protocol: "fnlp-candidate-int8-sentiment-sampled-job-v1",
        durable_unit: "complete_independent_sample_bundle",
        input_encoding: input_encoding::reported_policy(&encoding), invocation_encoding_work_limits: totals,
        compute_threads: (limits.compute.threads > 1).then_some(limits.compute.threads),
        prefill_rows: prefill.map(|p| p.max_batch_rows), max_selected_rows: args.sampled.max_selected_rows };
    // Never serialize the private recipe, seed, population or generated scores.
    publish_report(&response, output, &|| session.remaining(), &cancellation)
}
fn stage_failure(error: SessionStageError) -> Failure {
    match error {
        SessionStageError::Setup(error) => setup_failure(error),
        SessionStageError::Artifact(ArtifactBridgeError::Cancelled(cause)) => cancelled(cause),
        SessionStageError::Artifact(_) => setup_failure(CandidateError::Model),
        SessionStageError::Host(error) => host_failure(error),
    }
}
fn storage_error(error: JobError) -> Failure {
    match error {
        JobError::Cancelled(cause) => cancelled(cause),
        error => storage_failure(owned_jobs::job_failure(error)),
    }
}
fn failure(error: HostedJobError) -> Failure {
    match error {
        HostedJobError::Host(error) => host_failure(error),
        HostedJobError::Job(JobRunError::Storage(error)) => storage_error(error),
        HostedJobError::Job(JobRunError::Processor(fault)) => corpus::failure(fault),
        HostedJobError::Job(JobRunError::InvalidEnvelope) => Failure {
            exit: ErrorCode::InputDecodeOrParse, code: "int8_sampled_sentiment_job_envelope", cancellation: None },
        HostedJobError::Job(JobRunError::WorkContract | JobRunError::Stopped) =>
            Failure::output("int8_sampled_sentiment_job_completion_invariant"),
    }
}
fn publish_report<T: Serialize>(response: &T, output: &mut impl Write,
    remaining: &impl Fn() -> Result<Duration, CandidateError>, cancellation: &CancellationToken)
    -> Result<(), Failure> {
    let first_failure = Cell::new(None);
    publish_checked(response, job_cli::REPORT_BYTES, output, &|| {
        if first_failure.get().is_some() { return Err(CandidateError::Execution); }
        let failure = cancellation.cause().map(cancelled)
            .or_else(|| remaining().err().map(setup_failure));
        match failure {
            Some(failure) => { first_failure.set(Some(failure)); Err(CandidateError::Execution) }
            None => Ok(()),
        }
    }).map_err(|error| first_failure.get().unwrap_or_else(|| setup_failure(error)))
}
#[derive(Serialize)]
struct Response<'a, T: Serialize> {
    #[serde(flatten)] response: &'a T,
    protocol: &'static str,
    durable_unit: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    invocation_encoding_work_limits: Option<SafeBpeWorkLimits>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_threads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_rows: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_selected_rows: Option<usize>,
}

#[cfg(test)] mod tests;
