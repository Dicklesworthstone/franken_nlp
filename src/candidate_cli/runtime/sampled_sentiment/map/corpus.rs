//! Complete-document NDJSON on one admitted, checked-file INT8 host session.
use super::*;
use std::io::{BufRead, BufReader};
use crate::{
    candidate_cli::{
        batch::{CandidateWriter, IO_BUFFER_BYTES},
        sampled_sentiment::map::corpus::Envelope,
    },
    hosted::Int8SampledSentimentDocumentBatchHostConfig,
};

#[derive(Serialize)]
struct Provenance<'a> {
    protocol: &'static str,
    schema_version: u32,
    scope: &'static str,
    evidence: &'static str,
    task: &'static str,
    model_id: &'a str,
    source_revision: &'a str,
    source_root_sha256: &'a str,
    logical_model_sha256: &'a str,
    quant_recipe: &'a str,
    max_document_work: Int8Work,
    max_population_work: Int8Work,
    max_document_mask_node_visits: u64,
    max_population_mask_node_visits: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_rows: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_threads: Option<usize>,
}

fn configuration(command: &SampledSentimentMapCommand, limits: &MapLimits, envelope: &Envelope)
    -> Result<Int8SampledSentimentDocumentBatchHostConfig, CandidateError> {
    Ok(Int8SampledSentimentDocumentBatchHostConfig {
        document: super::config(command, limits)?,
        task: limits.task, encoding: limits.encoding.clone(),
        transport: envelope.transport, io_reserve_bytes: envelope.io_bytes,
        max_model_work: envelope.max_model_work,
        max_mask_node_visits: envelope.max_mask_node_visits,
    })
}

fn check_quote(limits: &MapLimits, envelope: &Envelope,
    config: &Int8SampledSentimentDocumentBatchHostConfig) -> Result<(), Failure> {
    let need = if limits.shared.compute.threads == 1 { config.requirements() }
        else { config.requirements_with_threads(limits.shared.compute.threads) }.map_err(map_host_failure)?;
    let prefill = limits.prefill.map(|p| p.validate()).transpose()
        .map_err(|_| setup_failure(CandidateError::Arguments))?.unwrap_or(0);
    let expected_input = limits.memory.input_bytes.checked_add(envelope.transport_bytes)
        .ok_or_else(|| setup_failure(CandidateError::Arguments))?;
    let expected_workspace = limits.memory.workspace_bytes.checked_add(prefill)
        .and_then(|n| n.checked_add(limits.shared.compute.workspace_bytes))
        .ok_or_else(|| setup_failure(CandidateError::Arguments))?;
    if need.native != limits.memory.native || need.input_bytes != expected_input
        || need.workspace_bytes != expected_workspace
        || need.output_reservation_bytes != limits.memory.output_reservation_bytes
        || need.max_output_tokens != limits.memory.max_output_tokens
        || need.max_output_bytes != limits.memory.max_output_bytes
        || limits.shared.compute != limits.common.compute.admit_prefill(limits.prefill).map_err(setup_failure)?
        || limits.minimum_memory_bytes > limits.shared.memory_bytes {
        return Err(Failure::usage("int8_sampled_sentiment_corpus_memory_quote"));
    }
    Ok(())
}

pub(in crate::candidate_cli) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    command: SampledSentimentMapCommand, limits: MapLimits, envelope: Envelope,
    input: R, output: W) -> Result<(), Failure> {
    let mut config = configuration(&command, &limits, &envelope).map_err(setup_failure)?;
    // Compare the independent pure CLI quote before private input or model IO.
    check_quote(&limits, &envelope, &config)?;
    let session = Session::new_with_prefill(&limits.common, limits.shared, limits.prefill)
        .map_err(setup_failure)?;
    let check = || session.remaining().map(|_| ());
    let facts = checked_stage(session.facts_with_error_detail(&limits.common), &check, session_failure)?;
    let provenance = Provenance {
        protocol: "fnlp-candidate-int8-sentiment-sampled-corpus-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative",
        task: "sentiment-sampled-document",
        model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id,
        max_document_work: config.document.mapping.max_model_work,
        max_population_work: envelope.max_model_work,
        max_document_mask_node_visits: config.document.max_mask_node_visits,
        max_population_mask_node_visits: envelope.max_mask_node_visits,
        input_encoding: input_encoding::reported_policy(&limits.encoding),
        prefill_rows: limits.prefill.map(|p| p.max_batch_rows),
        compute_threads: (limits.shared.compute.threads > 1).then_some(limits.shared.compute.threads),
    };
    let writer = checked_stage(CandidateWriter::new(output, &provenance,
        envelope.transport.max_output_line_bytes, envelope.output_bytes), &check, setup_failure)?;
    let reader: Box<dyn BufRead + Send> = if limits.common.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        let file = checked_stage(crate::local_io::open_document(&limits.common.input)
            .map_err(|_| CandidateError::Input), &check, setup_failure)?;
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, file))
    };
    let cancellation = CancellationToken::default();
    let model = checked_stage(session.load_with_error_detail(&limits.common, limits.shared, &facts, cancellation.clone()),
        &check, session_failure)?;
    config.document.native.run.max_elapsed = checked_stage(session.remaining(), &check, setup_failure)?;
    let summary = checked_stage(if limits.shared.compute.threads == 1 {
        session.engine.batch_int8_sampled_sentiment_documents(&model, config, reader, writer, cancellation)
    } else {
        session.engine.batch_int8_sampled_sentiment_documents_with_threads(&model, config,
            crate::native_engine::simd::config::process_kernel(), limits.shared.compute.threads,
            reader, writer, cancellation)
    }, &check, map_host_failure)?;
    if summary.failed != 0 {
        return Err(Failure::output("int8_sampled_sentiment_corpus_items_failed"));
    }
    Ok(())
}

#[cfg(test)] mod tests;
