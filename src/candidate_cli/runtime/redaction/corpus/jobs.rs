//! Retained edits use the same native corpus configuration as the live pipe.
//! Only completion metadata reaches stdout; sensitive edits remain protected.
use super::*;
use crate::{
    candidate_cli::{jobs as job_cli, runtime::owned_jobs::open_mode},
    hosted::corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest},
    jobs::{JobError, JobId, JobInputMemory, JobLimits, JobProgress, JobSecret, JobWork,
        population::PopulationReadLimits, runner::JobRunError},
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

// Quote the actual retained owner after bounded JobLimits parsing but before
// protected keys, options, original/retained population IO or model loading.
// Caller preparation and transferred preparation coexist; the host's buffer
// quote includes only the transferred owner.
fn retained_peak_bytes(command: &RedactCommand, common: &CandidateArgs, limits: Limits,
    host: JobHostLimits, lifetime: JobLimits, input_memory: JobInputMemory) -> Result<u64, CandidateError> {
    let native = Int8MemoryRequirement::for_context(common.context_tokens).map_err(|_| CandidateError::Memory)?;
    let buffers = host.required_buffer_bytes_for_input(lifetime, input_memory).map_err(|_| CandidateError::Memory)?;
    let edit = command.edit_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Memory)?;
    let (map_bytes, tokens, live_bytes, output_rows) = if command.long.chunked {
        let detector = command.long.config(&command.host, limits, NerOptions::default())?;
        let count = detector.mapping.chunks.max_chunks as u64;
        (detector.per_chunk.max_output_bytes.checked_mul(count).ok_or(CandidateError::Memory)?,
            u64::from(detector.per_chunk.max_output_tokens).checked_mul(count).ok_or(CandidateError::Memory)?,
            detector.mapping.reduction.max_live_value_bytes as u64,
            detector.mapping.max_model_work.projected_logits / crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE as u64)
    } else {
        let detector = short_detector(command, limits, NerOptions::default())?;
        (detector.per_pass.max_output_bytes, u64::from(detector.per_pass.max_output_tokens), 0,
            detector.max_model_work.projected_logits / crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE as u64)
    };
    // CorpusAdmission prices the full per-item projected rows in its output
    // guard, independently of intermediate map token staging.
    let base = crate::candidate_cli::compute::request_peak_bytes(limits, native,
        command.host.max_result_bytes as u64, output_rows, 0, 0)?;
    let staging = map_bytes.checked_add(live_bytes).and_then(|n| n.checked_mul(4)).ok_or(CandidateError::Memory)?;
    let tokens = tokens.checked_mul(8).ok_or(CandidateError::Memory)?;
    // Match the closed host's bounded private recipe copies and full256 key
    // metadata. These reservations do not permit retaining any source text.
    let prompt = command.prefill.limits()?.map(|limits| limits.validate())
        .transpose().map_err(|_| CandidateError::Memory)?.unwrap_or(0);
    [base, buffers, staging, tokens, edit, 4096, 4 * (MIB - 1024), prompt]
        .into_iter().try_fold(0_u64, |sum, bytes| sum.checked_add(bytes)).ok_or(CandidateError::Memory)
}
fn job_execution_error(error: HostedJobError) -> CandidateError {
    match error {
        HostedJobError::Host(error) => corpus_execution_error(error),
        HostedJobError::Job(JobRunError::Processor(error)) =>
            error.cancellation.map(batch_cancelled).unwrap_or(CandidateError::Batch),
        HostedJobError::Job(JobRunError::Storage(JobError::Cancelled(cause))) => batch_cancelled(cause),
        _ => CandidateError::Batch,
    }
}

pub(super) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    command: RedactCommand, _common: CandidateArgs, admitted: Limits, admitted_envelope: CorpusEnvelope,
    mut input: R, mut output: W) -> Result<(), CandidateError> {
    let (common, limits, envelope) = readmit(&command, admitted, admitted_envelope)?;
    let retention = &command.retention;
    let mode = retention.mode();
    retention.input_retention.validate(mode, &command.input).map_err(|_| CandidateError::Arguments)?;
    let prefill = command.prefill.limits()?;
    let session = Session::new_with_prefill(&common, limits, prefill)?;
    let path = retention.job_limits.as_ref().ok_or(CandidateError::Arguments)?;
    let lifetime = job_cli::parse_limits(&read_config(&session, path, job_cli::LIMIT_BYTES)?)
        .map_err(|_| CandidateError::Input)?;
    retention.check_limits(&command, envelope, lifetime)?;
    let host = host_limits(&command, &common, limits, envelope, &session)?;
    let input_memory = retention.input_retention.input_memory().map_err(|_| CandidateError::Arguments)?;
    let minimum = retained_peak_bytes(&command, &common, limits, host, lifetime, input_memory)?;
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
    let planner = configure_planner(&command, limits, planner)?;
    let input_encoding = population_encoding(&planner)?;
    let identity = source_identity(&facts, &planner, BuiltInTask::Ner)?;
    let prepared = prepare(&command, limits, envelope, &planner, identity, ner, request)?;
    session.remaining()?;
    let reader = job_cli::population_reader(mode, retention.input_retention, &command.input, input, IO_BUFFER_BYTES)
        .map_err(|_| CandidateError::Input)?;
    let cancellation = CancellationToken::default();
    let model = session.load(&common, limits, &facts, cancellation.clone())?;
    // Refresh remaining time after preparation/loading; do not renew it.
    let host = host_limits(&command, &common, limits, envelope, &session)?;
    let job_id = retention.job_id.ok_or(CandidateError::Arguments)?;
    let mode = open_mode(mode, retention.input_retention).map_err(|_| CandidateError::Arguments)?;
    let job = SourceJobRequest { root: retention.job_dir.clone().ok_or(CandidateError::Arguments)?,
        key, job_id, limits: lifetime, mode, materialize: retention.materialize };
    let edit_reserve_bytes = command.edit_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?;
    let planner = Arc::new(planner); let vocabulary = Arc::new(vocabulary);
    let width = limits.compute.threads;
    let kernel = crate::native_engine::simd::config::process_kernel();
    let progress = match prepared {
        PreparedCorpus::Short(batch) => {
            let config = RedactionCorpusConfig { batch, edit_reserve_bytes };
            if width == 1 { match prefill {
                None => session.engine.job_int8_redact(&model, planner, vocabulary,
                    config, secret, job, host, reader, cancellation),
                Some(prefill) => session.engine.job_int8_redact_layer_major(&model, planner, vocabulary,
                    config, secret, job, host, prefill, reader, cancellation),
            } } else { match prefill {
                None => session.engine.job_int8_redact_with_threads(&model, planner, vocabulary,
                    config, secret, job, host, kernel, width, reader, cancellation),
                Some(prefill) => session.engine.job_int8_redact_layer_major_with_threads(&model, planner, vocabulary,
                    config, secret, job, host, kernel, width, prefill, reader, cancellation),
            } }
        },
        PreparedCorpus::Document(batch) => {
            let config = RedactionCorpusConfig { batch, edit_reserve_bytes };
            if width == 1 { match prefill {
                None => session.engine.job_int8_redact_document(&model, planner, vocabulary,
                    config, secret, job, host, reader, cancellation),
                Some(prefill) => session.engine.job_int8_redact_document_layer_major(&model, planner, vocabulary,
                    config, secret, job, host, prefill, reader, cancellation),
            } } else { match prefill {
                None => session.engine.job_int8_redact_document_with_threads(&model, planner, vocabulary,
                    config, secret, job, host, kernel, width, reader, cancellation),
                Some(prefill) => session.engine.job_int8_redact_document_layer_major_with_threads(
                    &model, planner, vocabulary, config, secret, job, host,
                    kernel, width, prefill, reader, cancellation),
            } }
        },
    }.map_err(job_execution_error)?;
    session.remaining()?;
    let report = report(progress, job_id, lifetime, retention.resume, retention.materialize, command.long.chunked)?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate", evidence: "non_authoritative",
        model_id: &facts.model_id, source_revision: &facts.revision, source_root_sha256: &facts.source_root_sha256,
        logical_model_sha256: &facts.logical_model_sha256, quant_recipe: &facts.recipe_id, output: &report };
    // No CandidateWriter: no item events or private result prefix reaches
    // stdout. Report failure can follow durable progress; no rollback claim.
    let response = PopulationResponse { response: &response, input_encoding };
    publish_checked(&response, job_cli::REPORT_BYTES, &mut output,
        &|| session.remaining().map(|_| ()))
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
