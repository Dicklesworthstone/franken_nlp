//! Native scored jobs reuse the authenticated lifecycle and scorer configuration.
use super::*;
use crate::candidate_cli::{jobs::scored::ScoreJobArgs,
    runtime::scored_tasks::batch::{self as scored_batch, Corpus}};

fn host_limits(args: &ScoreJobArgs, common: &CandidateArgs, limits: Limits, session: &Session)
    -> Result<JobHostLimits, Failure> {
    Ok(JobHostLimits { native: session.native(common)?,
        transport: PopulationReadLimits { max_stream_bytes: args.max_stream_mib.checked_mul(MIB)
            .ok_or_else(|| Failure::usage("score_job_transport"))?, max_lines: args.max_input_lines },
        preparation_reserve_bytes: limits.preparation_bytes, io_reserve_bytes: (IO_BYTES * 2) as u64,
        journal_reserve_bytes: args.journal_memory_mib.checked_mul(MIB).ok_or_else(|| Failure::usage("score_job_memory"))?,
        serialization_reserve_bytes: args.serialization_memory_mib.checked_mul(MIB)
            .ok_or_else(|| Failure::usage("score_job_memory"))? })
}
fn result_ceiling(args: &ScoreJobArgs, lifetime: JobLimits) -> Result<(), Failure> {
    if args.host.max_result_bytes > lifetime.max_result_bytes {
        return Err(Failure::usage("score_job_result_ceiling"));
    }
    Ok(())
}

pub(in crate::candidate_cli) fn execute<R: Read + Send + 'static>(mode: RunMode, args: ScoreJobArgs,
    common: CandidateArgs, limits: Limits, input: R, output: &mut impl Write) -> Result<(), Failure> {
    args.input_retention.validate(mode, &args.host.input)?;
    // First local owner: settings, planners, reader and report drop before the
    // CLI preparation charge. The host separately owns the transferred state.
    let prefill = args.prefill_limits()?;
    if common.compute != args.compute {
        return Err(Failure::usage("score_job_schedule_admission"));
    }
    let native_bytes = args.execution_memory_bytes(limits)?;
    if native_bytes > limits.memory_bytes {
        return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "score_job_native_memory" });
    }
    let session = Session::new_with_prefill(&common, limits, prefill)?;
    let lifetime = command::parse_limits(&read_config(&session, &args.limits_file, command::LIMIT_BYTES)?)?;
    result_ceiling(&args, lifetime)?;
    let host = host_limits(&args, &common, limits, &session)?;
    check_population_memory(&args, host, lifetime, limits)?;
    let key = JobSecret::read(&args.key_file).map_err(job_failure)?;
    session.remaining()?;
    let defaults = args.defaults.as_ref().map(|p| read_config(&session, p, command::DEFAULT_BYTES)).transpose()?;
    let defaults = args.parse_defaults(defaults.as_deref(), args.host.budget(limits))?;
    let facts = session.facts(&common)?;
    // Exact same pinned planners/full-vocabulary policy as score-batch; the
    // job adds retention, not a divergent scoring implementation or defaults.
    let corpus = scored_batch::configure_scored(args.kind()?, &args.host, &facts, limits, defaults)?;
    session.remaining()?;
    let reader = command::population_reader(mode, args.input_retention, &args.host.input, input, IO_BYTES)?;
    let cancellation = CancellationToken::default();
    let model = session.load(&common, limits, &facts, cancellation.clone())?;
    let host = host_limits(&args, &common, limits, &session)?;
    let job_id = args.job_id; let materialize = args.materialize;
    let request = SourceJobRequest { root: args.job_dir, key, job_id,
        limits: lifetime, mode: open_mode(mode, args.input_retention)?, materialize };
    let kernel = crate::native_engine::simd::config::process_kernel();
    let progress = match (corpus, limits.compute.threads, prefill) {
        (Corpus::Classify(planner, config), 1, None) => session.engine.job_int8_classify(&model,
            planner, config, request, host, reader, cancellation.clone()),
        (Corpus::Classify(planner, config), 1, Some(prefill)) => session.engine.job_int8_classify_layer_major(&model,
            planner, config, request, host, prefill, reader, cancellation.clone()),
        (Corpus::Classify(planner, config), width, None) => session.engine.job_int8_classify_with_threads(&model,
            planner, config, request, host, kernel, width, reader, cancellation.clone()),
        (Corpus::Classify(planner, config), width, Some(prefill)) => session.engine.job_int8_classify_layer_major_with_threads(
            &model, planner, config, request, host, kernel, width, prefill, reader, cancellation.clone()),
        (Corpus::Sentiment(planner, config), 1, None) => session.engine.job_int8_sentiment(&model,
            planner, config, request, host, reader, cancellation.clone()),
        (Corpus::Sentiment(planner, config), 1, Some(prefill)) => session.engine.job_int8_sentiment_layer_major(&model,
            planner, config, request, host, prefill, reader, cancellation.clone()),
        (Corpus::Sentiment(planner, config), width, None) => session.engine.job_int8_sentiment_with_threads(&model,
            planner, config, request, host, kernel, width, reader, cancellation.clone()),
        (Corpus::Sentiment(planner, config), width, Some(prefill)) => session.engine.job_int8_sentiment_layer_major_with_threads(
            &model, planner, config, request, host, kernel, width, prefill, reader, cancellation.clone()),
        (Corpus::Judge(planner, config), 1, None) => session.engine.job_int8_judge(&model,
            planner, config, request, host, reader, cancellation.clone()),
        (Corpus::Judge(planner, config), 1, Some(prefill)) => session.engine.job_int8_judge_layer_major(&model,
            planner, config, request, host, prefill, reader, cancellation.clone()),
        (Corpus::Judge(planner, config), width, None) => session.engine.job_int8_judge_with_threads(&model,
            planner, config, request, host, kernel, width, reader, cancellation.clone()),
        (Corpus::Judge(planner, config), width, Some(prefill)) => session.engine.job_int8_judge_layer_major_with_threads(
            &model, planner, config, request, host, kernel, width, prefill, reader, cancellation.clone()),
    }.map_err(host_failure)?;
    if let Some(cause) = cancellation.cause() { return Err(job_failure(JobError::Cancelled(cause))); }
    session.remaining()?;
    let report = progress_report(progress, job_id, lifetime, materialize, mode, &args.task)?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: &report };
    // No item ids, labels, texts or native scores enter this metadata report.
    // Delivery failure can follow durable progress; never imply rollback.
    let response = ScheduledResponse { response: &response,
        compute_threads: (limits.compute.threads > 1).then_some(limits.compute.threads),
        prefill_rows: prefill.map(|limits| limits.max_batch_rows) };
    publish_report(&response, output, &|| session.remaining(), &cancellation)
}

/// Include the actual input mode: retained replay keeps an index plus one
/// record, while ordinary original-input jobs retain the complete population.
fn population_memory_bytes(args: &ScoreJobArgs, host: JobHostLimits, job: JobLimits, limits: Limits)
    -> Result<u64, Failure> {
    let buffers = host.required_buffer_bytes_for_input(job, args.input_retention.input_memory()?)
        .map_err(host_failure)?;
    args.execution_memory_bytes(limits)?.checked_add(buffers)
        .ok_or_else(|| Failure::usage("score_job_memory_arithmetic"))
}
fn check_population_memory(args: &ScoreJobArgs, host: JobHostLimits, job: JobLimits, limits: Limits)
    -> Result<(), Failure> {
    if population_memory_bytes(args, host, job, limits)? > limits.memory_bytes {
        return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "score_job_population_memory" });
    }
    Ok(())
}

fn publish_report<T: Serialize>(response: &T, output: &mut impl Write,
    remaining: &impl Fn() -> Result<Duration, CandidateError>, cancellation: &CancellationToken)
    -> Result<(), Failure> {
    publish_checked(response, command::REPORT_BYTES, output, &|| {
        if cancellation.cause().is_some() { return Err(CandidateError::Execution); }
        remaining().map(|_| ())
    }).map_err(|error| match (error, cancellation.cause()) {
        (CandidateError::Execution, Some(cause)) => job_failure(JobError::Cancelled(cause)),
        (error, _) => Failure::from(error),
    })
}

/// Selected physical schedule, reported only after hosted worker drain.
/// Omitted default fields preserve the existing serial metadata bytes.
#[derive(Serialize)]
struct ScheduledResponse<'a, T: Serialize> {
    #[serde(flatten)] response: &'a T,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_threads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_rows: Option<usize>,
}

#[cfg(test)] mod tests;
