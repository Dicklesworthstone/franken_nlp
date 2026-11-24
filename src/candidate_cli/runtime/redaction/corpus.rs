//! Redaction corpus dispatch: one owned stream, native allocation and secret scope.
use super::*;
use std::io::{BufRead, BufReader};
use crate::{
    candidate_cli::{batch::{self as transport, CandidateWriter, IO_BUFFER_BYTES},
        redact::corpus::{CorpusEnvelope, short_detector}},
    hosted::corpus::{CorpusLimits, RedactionCorpusConfig},
    tasks::redact::{batch::{self as short_batch, Int8RedactionBatchConfig},
        corpus::{self as long_batch, LongRedactionBatchConfig}},
};
#[cfg(all(feature = "metadata-store", target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
mod jobs;

#[derive(Serialize)]
struct Provenance<'a> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str,
    model_id: &'a str, source_revision: &'a str, source_root_sha256: &'a str,
    logical_model_sha256: &'a str, quant_recipe: &'a str, task: &'static str, redaction_mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<PopulationEncoding>,
}
enum PreparedCorpus { Short(Int8RedactionBatchConfig), Document(LongRedactionBatchConfig) }

/// Population factories have no observed document work. State the exact
/// allowance and scope without presenting an untouched factory as a receipt.
#[derive(Serialize)]
struct PopulationEncoding {
    policy: UntrustedEncoding,
    complete_document_limits: SafeBpeWorkLimits,
    budget_scope: &'static str,
}
fn population_encoding(planner: &SourceTaskPlanner) -> Result<Option<PopulationEncoding>, CandidateError> {
    match (planner.document_encoding(), planner.map_document_encoding_limits()) {
        (UntrustedEncoding::ByteFallback, None) => Ok(None),
        (policy @ UntrustedEncoding::SafeBpeV1(_), Some(complete_document_limits)) => Ok(Some(PopulationEncoding {
            policy: policy.clone(), complete_document_limits,
            budget_scope: "original_preflight_execution_and_edited_verification_per_document_attempt",
        })),
        _ => Err(CandidateError::Planning),
    }
}
#[cfg(any(test, all(feature = "metadata-store", target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[derive(Serialize)]
struct PopulationResponse<'a, T: Serialize> {
    #[serde(flatten)]
    response: &'a T,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<PopulationEncoding>,
}
fn configure_planner(command: &RedactCommand, limits: Limits, planner: SourceTaskPlanner)
    -> Result<SourceTaskPlanner, CandidateError> {
    let (policy, complete) = command.encoding_policy(limits)?;
    let planner = match (policy, complete) {
        (UntrustedEncoding::ByteFallback, None) => planner,
        (policy @ UntrustedEncoding::SafeBpeV1(_), Some(complete)) if command.long.chunked =>
            planner.with_map_document_encoding(policy, complete).map_err(|_| CandidateError::Planning)?,
        _ => return Err(CandidateError::Arguments),
    };
    if planner.map_document_encoding_workspace_bytes().map_err(|_| CandidateError::Planning)?
        > limits.preparation_bytes { return Err(CandidateError::Memory); }
    Ok(planner)
}
fn readmit(command: &RedactCommand, admitted: Limits, admitted_envelope: CorpusEnvelope)
    -> Result<(CandidateArgs, Limits, CorpusEnvelope), CandidateError> {
    if !command.corpus.ndjson { return Err(CandidateError::Arguments); }
    let (common, limits) = command.validate()?;
    let envelope = command.corpus.envelope(command, limits)?;
    if limits != admitted || envelope.transport != admitted_envelope.transport
        || envelope.max_model_work != admitted_envelope.max_model_work
        || envelope.max_mask_visits != admitted_envelope.max_mask_visits
        || envelope.output_bytes != admitted_envelope.output_bytes
        || envelope.io_bytes != admitted_envelope.io_bytes {
        return Err(CandidateError::Arguments);
    }
    Ok((common, limits, envelope))
}
fn batch_cancelled(cause: DecodeCancellationKind) -> CandidateError {
    match cause {
        DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout => CandidateError::Timeout,
        _ => CandidateError::Batch,
    }
}
fn corpus_execution_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => batch_cancelled(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => corpus_execution_error(*error),
        HostedError::Batch(error) => error.fault.cancellation.map(batch_cancelled).unwrap_or(CandidateError::Batch),
        HostedError::BatchSetup(error) => error.cancellation.map(batch_cancelled).unwrap_or(CandidateError::Batch),
        HostedError::SourceMap(error) => error.cancellation().map(batch_cancelled).unwrap_or(CandidateError::Batch),
        HostedError::Redaction(error) => error.cancellation().map(batch_cancelled).unwrap_or(CandidateError::Batch),
        HostedError::Native(crate::native_engine::strict_int8::StrictInt8Error::Cancelled(cause)) => batch_cancelled(cause),
        HostedError::Reservation(_) => CandidateError::Memory,
        HostedError::ModelIdentity | HostedError::ResourceDomain => CandidateError::Identity,
        _ => CandidateError::Batch,
    }
}

// Both live and retained paths compile the same fixed native policy. A job
// changes delivery/durability, not the detector or its source/verification scope.
#[allow(clippy::too_many_arguments)]
fn prepare(command: &RedactCommand, limits: Limits, envelope: CorpusEnvelope,
    planner: &SourceTaskPlanner, identity: ExecutionIdentity, ner: NerOptions, request: RedactionRequest)
    -> Result<PreparedCorpus, CandidateError> {
    if command.long.chunked {
        let detector = command.long.config(&command.host, limits, ner)?;
        let batch = LongRedactionBatchConfig { ner_identity: identity, detector, request,
            max_model_work: envelope.max_model_work, max_mask_visits: envelope.max_mask_visits };
        long_batch::check_configuration(planner, &batch).map_err(|_| CandidateError::Planning)?;
        Ok(PreparedCorpus::Document(batch))
    } else {
        let detector = short_detector(command, limits, ner)?;
        let batch = Int8RedactionBatchConfig { ner_identity: identity, detector, request,
            max_model_work: envelope.max_model_work, max_mask_visits: envelope.max_mask_visits };
        short_batch::check_configuration(planner, &batch).map_err(|_| CandidateError::Planning)?;
        Ok(PreparedCorpus::Short(batch))
    }
}
fn key_scope(command: &RedactCommand, input: &mut impl Read, request: &RedactionRequest)
    -> Result<Option<RedactionPseudonyms>, CandidateError> {
    let secret = command.key(input)?.map(|(key, namespace)| RedactionPseudonyms { key: Arc::new(key), namespace });
    {
        let context = secret.as_ref().map(|s| Pseudonyms::full256(&s.key, &s.namespace,
            request.actions.expected_key_commitment.as_deref())).transpose().map_err(|_| CandidateError::Identity)?;
        request.actions.check_key(context.as_ref()).map_err(|_| CandidateError::Arguments)?;
    }
    Ok(secret)
}
pub(in crate::candidate_cli) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    command: RedactCommand, _common: CandidateArgs, admitted: Limits, admitted_envelope: CorpusEnvelope,
    mut input: R, output: W,
) -> Result<(), CandidateError> {
    // Recompute every owner and work envelope before a Session or private IO.
    let (common, limits, envelope) = readmit(&command, admitted, admitted_envelope)?;
    if command.retention.store_results {
        #[cfg(all(feature = "metadata-store", target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
        { return jobs::execute(command, common, limits, envelope, input, output); }
        #[cfg(not(all(feature = "metadata-store", target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64"))))]
        { return Err(CandidateError::Unavailable); }
    }
    let prefill = command.prefill.limits()?;
    let session = Session::new_with_prefill(&common, limits, prefill)?;
    // A private binary key is read ONCE from the inherited input handle, only
    // with a separate corpus path. Commitment checks precede all corpus IO.
    let request = command.request();
    let secret = key_scope(&command, &mut input, &request)?;
    session.remaining()?;
    let options = command.ner_options.as_ref().map(|p| session.read(p, &mut input, source::OPTIONS_BYTES)).transpose()?;
    let ner = command::ner_options(options.as_deref())?;
    drop(detectors::detect("", &request.rules, request.rule_budget).map_err(|_| CandidateError::Planning)?);
    let facts = session.facts(&common)?;
    let (planner, vocabulary) = planner()?;
    let planner = configure_planner(&command, limits, planner)?;
    let identity = source_identity(&facts, &planner, BuiltInTask::Ner)?;
    let prepared = prepare(&command, limits, envelope, &planner, identity, ner, request)?;
    session.remaining()?;
    let provenance = Provenance { protocol: "fnlp-candidate-batch-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative",
        model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, task: "redact",
        redaction_mode: if command.long.chunked { "chunked" } else { "single_context" },
        input_encoding: population_encoding(&planner)? };
    let writer = CandidateWriter::new(output, &provenance, envelope.transport.max_output_line_bytes, envelope.output_bytes)?;
    // No caller-held stdio locks, input collection, pipe relay, detached worker
    // or per-document runtime. Buffers and handles move into the corpus host.
    let reader: Box<dyn BufRead + Send> = if command.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, File::open(&command.input).map_err(|_| CandidateError::Input)?))
    };
    let cancellation = CancellationToken::default();
    let model = session.load(&common, limits, &facts, cancellation.clone())?;
    let corpus = CorpusLimits { native: session.native(&common)?, transport: envelope.transport,
        preparation_reserve_bytes: limits.preparation_bytes, io_reserve_bytes: envelope.io_bytes };
    let edit_reserve_bytes = command.edit_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?;
    let planner = Arc::new(planner); let vocabulary = Arc::new(vocabulary);
    let width = limits.compute.threads;
    let kernel = crate::native_engine::simd::config::process_kernel();
    let summary = match prepared {
        PreparedCorpus::Short(batch) => {
            let config = RedactionCorpusConfig { batch, edit_reserve_bytes };
            if width == 1 { match prefill {
                None => session.engine.batch_int8_redact(&model, planner, vocabulary,
                    config, secret, corpus, reader, writer, cancellation),
                Some(prefill) => session.engine.batch_int8_redact_layer_major(&model, planner, vocabulary,
                    config, secret, corpus, prefill, reader, writer, cancellation),
            } } else { match prefill {
                None => session.engine.batch_int8_redact_with_threads(&model, planner, vocabulary,
                    config, secret, corpus, kernel, width, reader, writer, cancellation),
                Some(prefill) => session.engine.batch_int8_redact_layer_major_with_threads(&model, planner, vocabulary,
                    config, secret, corpus, kernel, width, prefill, reader, writer, cancellation),
            } }
        },
        PreparedCorpus::Document(batch) => {
            let config = RedactionCorpusConfig { batch, edit_reserve_bytes };
            if width == 1 { match prefill {
                None => session.engine.batch_int8_redact_document(&model, planner, vocabulary,
                    config, secret, corpus, reader, writer, cancellation),
                Some(prefill) => session.engine.batch_int8_redact_document_layer_major(&model, planner, vocabulary,
                    config, secret, corpus, prefill, reader, writer, cancellation),
            } } else { match prefill {
                None => session.engine.batch_int8_redact_document_with_threads(&model, planner, vocabulary,
                    config, secret, corpus, kernel, width, reader, writer, cancellation),
                Some(prefill) => session.engine.batch_int8_redact_document_layer_major_with_threads(
                    &model, planner, vocabulary, config, secret, corpus,
                    kernel, width, prefill, reader, writer, cancellation),
            } }
        },
    }.map_err(corpus_execution_error)?;
    // A terminal protocol frame is not proof all documents succeeded. Preserve
    // earlier completed frames on later failure; never append a second terminal.
    transport::completed(summary)
}

#[cfg(test)] mod tests;
