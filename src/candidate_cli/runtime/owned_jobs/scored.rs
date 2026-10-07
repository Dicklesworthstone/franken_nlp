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
    // First local owner: settings, planners, reader and report drop before the
    // CLI preparation charge. The host separately owns the transferred state.
    let prefill = args.prefill_limits()?;
    let session = Session::new(&common, limits)?;
    let lifetime = command::parse_limits(&read_config(&session, &args.limits_file, command::LIMIT_BYTES)?)?;
    result_ceiling(&args, lifetime)?;
    preflight_memory(host_limits(&args, &common, limits, &session)?, lifetime, limits)?;
    let key = JobSecret::read(&args.key_file).map_err(job_failure)?;
    session.remaining()?;
    let defaults = args.defaults.as_ref().map(|p| read_config(&session, p, command::DEFAULT_BYTES)).transpose()?;
    let defaults = args.parse_defaults(defaults.as_deref(), args.host.budget(limits))?;
    let facts = session.facts(&common)?;
    // Exact same pinned planners/full-vocabulary policy as score-batch; the
    // job adds retention, not a divergent scoring implementation or defaults.
    let corpus = scored_batch::configure_scored(args.kind()?, &args.host, &facts, limits, defaults)?;
    session.remaining()?;
    let reader: Box<dyn BufRead + Send> = if args.host.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BYTES,
            crate::local_io::open_document(&args.host.input).map_err(|_| CandidateError::Input)?))
    };
    let cancellation = CancellationToken::default();
    let model = session.load(&common, limits, &facts, cancellation.clone())?;
    let host = host_limits(&args, &common, limits, &session)?;
    let job_id = args.job_id; let materialize = args.materialize;
    let request = SourceJobRequest { root: args.job_dir, key, job_id,
        limits: lifetime, mode: open_mode(mode), materialize };
    let progress = match corpus {
        Corpus::Classify(planner, config) => session.engine.job_int8_classify(&model, planner,
            config, request, host, reader, cancellation),
        Corpus::Sentiment(planner, config) => session.engine.job_int8_sentiment(&model, planner,
            config, request, host, reader, cancellation),
        Corpus::Judge(planner, config) => match prefill {
            Some(prefill) => session.engine.job_int8_judge_layer_major(&model, planner,
                config, request, host, prefill, reader, cancellation),
            None => session.engine.job_int8_judge(&model, planner,
                config, request, host, reader, cancellation),
        },
    }.map_err(host_failure)?;
    session.remaining()?;
    let report = progress_report(progress, job_id, lifetime, materialize, mode, &args.task)?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: &report };
    // No item ids, labels, texts or native scores enter this metadata report.
    // Delivery failure can follow durable progress; never imply rollback.
    publish(&response, command::REPORT_BYTES, output).map_err(Failure::from)
}

#[cfg(test)] mod tests;
