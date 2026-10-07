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
}
enum PreparedCorpus { Short(Int8RedactionBatchConfig), Document(LongRedactionBatchConfig) }

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
    command: RedactCommand, common: CandidateArgs, limits: Limits, envelope: CorpusEnvelope,
    mut input: R, output: W,
) -> Result<(), CandidateError> {
    if command.retention.store_results {
        #[cfg(all(feature = "metadata-store", target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
        { return jobs::execute(command, common, limits, envelope, input, output); }
        #[cfg(not(all(feature = "metadata-store", target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64"))))]
        { return Err(CandidateError::Unavailable); }
    }
    let session = Session::new(&common, limits)?;
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
    let identity = source_identity(&facts, &planner, BuiltInTask::Ner)?;
    let prepared = prepare(&command, limits, envelope, &planner, identity, ner, request)?;
    session.remaining()?;
    let provenance = Provenance { protocol: "fnlp-candidate-batch-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative",
        model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, task: "redact",
        redaction_mode: if command.long.chunked { "chunked" } else { "single_context" } };
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
    let summary = match prepared {
        PreparedCorpus::Short(batch) => session.engine.batch_int8_redact(&model, Arc::new(planner), Arc::new(vocabulary),
            RedactionCorpusConfig { batch, edit_reserve_bytes }, secret, corpus, reader, writer, cancellation),
        PreparedCorpus::Document(batch) => session.engine.batch_int8_redact_document(&model, Arc::new(planner), Arc::new(vocabulary),
            RedactionCorpusConfig { batch, edit_reserve_bytes }, secret, corpus, reader, writer, cancellation),
    }.map_err(|_| CandidateError::Batch)?;
    // A terminal protocol frame is not proof all documents succeeded. Preserve
    // earlier completed frames on later failure; never append a second terminal.
    transport::completed(summary)
}
