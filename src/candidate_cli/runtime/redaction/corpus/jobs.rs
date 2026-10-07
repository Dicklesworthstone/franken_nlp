//! Retained edits use the same native corpus configuration as the live pipe.
//! Only completion metadata reaches stdout; sensitive edits remain protected.
use super::*;
use crate::{
    candidate_cli::jobs as job_cli,
    hosted::corpus::jobs::{JobHostLimits, JobOpenMode, SourceJobRequest},
    jobs::{JobId, JobLimits, JobProgress, JobSecret, JobWork, TailPolicy, population::PopulationReadLimits},
};

fn host_limits(command: &RedactCommand, common: &CandidateArgs, limits: Limits,
    envelope: CorpusEnvelope, session: &Session) -> Result<JobHostLimits, CandidateError> {
    Ok(JobHostLimits { native: session.native(common)?,
        transport: PopulationReadLimits { max_stream_bytes: envelope.transport.max_input_bytes,
            max_lines: command.retention.max_lines() },
        preparation_reserve_bytes: limits.preparation_bytes, io_reserve_bytes: 2 * IO_BUFFER_BYTES as u64,
        journal_reserve_bytes: command.retention.journal_bytes()?,
        serialization_reserve_bytes: command.retention.serialization_bytes()? })
}
fn read_config(session: &Session, path: &std::path::Path, cap: usize) -> Result<String, CandidateError> {
    session.remaining()?;
    let mut file = crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?;
    let text = read_input(&mut file, cap)?;
    session.remaining()?; Ok(text)
}

pub(super) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    command: RedactCommand, common: CandidateArgs, limits: Limits, envelope: CorpusEnvelope,
    mut input: R, mut output: W) -> Result<(), CandidateError> {
    let session = Session::new(&common, limits)?;
    let retention = &command.retention;
    let path = retention.job_limits.as_ref().ok_or(CandidateError::Arguments)?;
    let lifetime = job_cli::parse_limits(&read_config(&session, path, job_cli::LIMIT_BYTES)?)
        .map_err(|_| CandidateError::Input)?;
    retention.check_limits(&command, envelope, lifetime)?;
    let host = host_limits(&command, &common, limits, envelope, &session)?;
    let minimum = host.required_buffer_bytes(lifetime).map_err(|_| CandidateError::Memory)?
        .checked_add(limits.preparation_bytes).and_then(|n| n.checked_add(limits.weight_bytes))
        .ok_or(CandidateError::Memory)?;
    if minimum > limits.memory_bytes { return Err(CandidateError::Memory); }
    // Independent protected job secret. Never substitute it for the user's
    // pseudonym key or create either key implicitly on start/resume.
    let key = JobSecret::read(retention.job_key_file.as_ref().ok_or(CandidateError::Arguments)?)
        .map_err(|_| CandidateError::Input)?;
    let request = command.request();
    let secret = key_scope(&command, &mut input, &request)?;
    session.remaining()?;
    let options = command.ner_options.as_ref().map(|p| read_config(&session, p, source::OPTIONS_BYTES)).transpose()?;
    let ner = command::ner_options(options.as_deref())?;
    drop(detectors::detect("", &request.rules, request.rule_budget).map_err(|_| CandidateError::Planning)?);
    let facts = session.facts(&common)?;
    let (planner, vocabulary) = planner()?;
    let identity = source_identity(&facts, &planner, BuiltInTask::Ner)?;
    let prepared = prepare(&command, limits, envelope, &planner, identity, ner, request)?;
    session.remaining()?;
    let reader: Box<dyn BufRead + Send> = if command.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES,
            crate::local_io::open_document(&command.input).map_err(|_| CandidateError::Input)?))
    };
    let cancellation = CancellationToken::default();
    let model = session.load(&common, limits, &facts, cancellation.clone())?;
    // Refresh remaining time after preparation/loading; do not renew it.
    let host = host_limits(&command, &common, limits, envelope, &session)?;
    let job_id = retention.job_id.ok_or(CandidateError::Arguments)?;
    let mode = if retention.resume {
        JobOpenMode::Resume(if retention.discard_uncommitted_tail { TailPolicy::DiscardUncommitted } else { TailPolicy::Refuse })
    } else { JobOpenMode::Create };
    let job = SourceJobRequest { root: retention.job_dir.clone().ok_or(CandidateError::Arguments)?,
        key, job_id, limits: lifetime, mode, materialize: retention.materialize };
    let edit_reserve_bytes = command.edit_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?;
    let progress = match prepared {
        PreparedCorpus::Short(batch) => session.engine.job_int8_redact(&model, Arc::new(planner), Arc::new(vocabulary),
            RedactionCorpusConfig { batch, edit_reserve_bytes }, secret, job, host, reader, cancellation),
        PreparedCorpus::Document(batch) => session.engine.job_int8_redact_document(&model, Arc::new(planner), Arc::new(vocabulary),
            RedactionCorpusConfig { batch, edit_reserve_bytes }, secret, job, host, reader, cancellation),
    }.map_err(|_| CandidateError::Batch)?;
    session.remaining()?;
    let report = report(progress, job_id, lifetime, retention.resume, retention.materialize, command.long.chunked)?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate", evidence: "non_authoritative",
        model_id: &facts.model_id, source_revision: &facts.revision, source_root_sha256: &facts.source_root_sha256,
        logical_model_sha256: &facts.logical_model_sha256, quant_recipe: &facts.recipe_id, output: &report };
    // No CandidateWriter: no item events or private result prefix reaches
    // stdout. Report failure can follow durable progress; no rollback claim.
    publish(&response, job_cli::REPORT_BYTES, &mut output)
}

#[derive(Serialize)]
struct Report {
    operation: &'static str, task: &'static str, status: &'static str, redaction_mode: &'static str,
    job_id: String, items: u64, committed: u64, attempts: u64, reserved_work: JobWork,
    committed_spool_bytes: u64, materialized: bool, retained_output_is_sensitive: bool,
}
fn report(p: JobProgress, expected: JobId, limits: JobLimits, resume: bool, materialize: bool, chunked: bool)
    -> Result<Report, CandidateError> {
    if p.job_id != expected || p.items == 0 || p.items > limits.max_items || p.committed != p.items
        || p.attempts < p.committed || p.attempts > limits.max_attempts || !p.reserved_work.fits(limits.max_work)
        || p.spool_bytes > limits.max_spool_bytes || materialize && !p.materialized {
        return Err(CandidateError::Execution);
    }
    Ok(Report { operation: if resume { "resume" } else { "start" }, task: "redact", status: "complete",
        redaction_mode: if chunked { "chunked" } else { "single_context" }, job_id: job_cli::job_id_hex(p.job_id),
        items: p.items, committed: p.committed, attempts: p.attempts, reserved_work: p.reserved_work,
        committed_spool_bytes: p.spool_bytes, materialized: p.materialized, retained_output_is_sensitive: true })
}

#[cfg(test)] mod tests;
