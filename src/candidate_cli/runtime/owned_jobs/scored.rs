//! Native scored jobs reuse the authenticated lifecycle and scorer configuration.
use super::*;
use crate::candidate_cli::{jobs::scored::ScoreJobArgs,
    runtime::scored_tasks::batch::{self as scored_batch, Corpus}};
use crate::jobs::runner::scored::{Int8ClassificationJobPlanner, Int8SentimentJobPlanner};
mod document;

fn configure(args: &ScoreJobArgs, facts: &ArtifactIdentity, limits: Limits,
    defaults: crate::candidate_cli::scored_batch::Defaults) -> Result<Corpus, Failure> {
    let encoding = args.encoding_policy()?;
    let corpus = if encoding == UntrustedEncoding::ByteFallback {
        scored_batch::configure_scored(args.kind()?, &args.host, facts, limits, defaults)?
    } else {
        scored_batch::configure_scored_with_encoding(args.kind()?, &args.host, facts, limits, defaults, &encoding)?
    };
    check_encoding_quote(&corpus, &encoding, args.encoding_work_limits()?, args.encoding_workspace_bytes()?)?;
    Ok(corpus)
}

/// Bind admitted memory and immutable resume caps to the actual factory before
/// loading weights. Equal workspace geometry is not equal encoding authority.
fn check_encoding_quote(corpus: &Corpus, policy: &UntrustedEncoding,
    total: Option<SafeBpeWorkLimits>, expected: u64) -> Result<(), Failure> {
    let (actual, actual_total, workspace) = match corpus {
        Corpus::Classify(planner, config) => {
            let factory = Int8ClassificationJobPlanner::new_with_document_encoding(planner,
                config.identity.clone(), config.task_ceiling, config.planning, config.defaults.clone(),
                config.max_model_work, total).map_err(|_| CandidateError::Planning)?;
            (factory.document_encoding().clone(), factory.encoding_work_limits(),
                factory.encoding_workspace_bytes().map_err(|_| CandidateError::Planning)?)
        }
        Corpus::Sentiment(planner, config) => {
            let config = crate::hosted::corpus::SentimentCorpusConfig {
                identity: config.identity.clone(), task_ceiling: config.task_ceiling,
                planning: config.planning, defaults: config.defaults.clone(),
                max_item_work: config.max_item_work, max_model_work: config.max_model_work,
            };
            let factory = Int8SentimentJobPlanner::new_with_document_encoding(planner, config, total)
                .map_err(|_| CandidateError::Planning)?;
            (factory.document_encoding().clone(), factory.encoding_work_limits(),
                factory.encoding_workspace_bytes().map_err(|_| CandidateError::Planning)?)
        }
        Corpus::Judge(planner, config) => config.encoding_requirements(planner, total)
            .map_err(|_| CandidateError::Planning)?,
    };
    if actual != *policy || actual_total != total || workspace != expected {
        return Err(Failure::usage("score_job_encoding_quote_mismatch"));
    }
    Ok(())
}

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
    if args.document_config.is_some() {
        return document::execute(mode, args, common, limits, input, output);
    }
    args.input_retention.validate(mode, &args.host.input)?;
    // First local owner: settings, planners, reader and report drop before the
    // CLI preparation charge. The host separately owns the transferred state.
    let prefill = args.prefill_limits()?;
    let encoding = args.encoding_policy()?;
    let encoding_work_limits = args.encoding_work_limits()?;
    if common.input_encoding != args.encoding {
        return Err(Failure::usage("score_job_encoding_admission"));
    }
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
    let corpus = configure(&args, &facts, limits, defaults)?;
    session.remaining()?;
    let reader = command::population_reader(mode, args.input_retention, &args.host.input, input, IO_BYTES)?;
    let cancellation = CancellationToken::default();
    let model = session.load(&common, limits, &facts, cancellation.clone())?;
    let host = host_limits(&args, &common, limits, &session)?;
    let job_id = args.job_id; let materialize = args.materialize;
    let request = SourceJobRequest { root: args.job_dir, key, job_id,
        limits: lifetime, mode: open_mode(mode, args.input_retention)?, materialize };
    let kernel = crate::native_engine::simd::config::process_kernel();
    let team = (limits.compute.threads > 1).then_some((kernel, limits.compute.threads));
    let progress = match corpus {
        Corpus::Classify(planner, config) => session.engine.job_int8_classify_scheduled(&model,
            planner, config, request, host, encoding_work_limits, prefill, team, reader, cancellation.clone()),
        Corpus::Sentiment(planner, config) => session.engine.job_int8_sentiment_scheduled(&model,
            planner, config, request, host, encoding_work_limits, prefill, team, reader, cancellation.clone()),
        Corpus::Judge(planner, config) => session.engine.job_int8_judge_scheduled(&model,
            planner, config, request, host, encoding_work_limits, prefill, team, reader, cancellation.clone()),
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
    let response = EncodingReport { response: &response,
        input_encoding: input_encoding::reported_policy(&encoding),
        invocation_encoding_work_limits: encoding_work_limits, root_object_reconciliation: None };
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

#[cfg(test)] mod tests;
