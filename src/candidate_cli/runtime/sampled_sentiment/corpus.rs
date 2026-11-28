//! Transfer owned IO into one checked-model session with a single deadline.
use super::*;
use std::io::{BufRead, BufReader};
use crate::{
    batch::{BatchCode, BatchFault},
    candidate_cli::{batch::{CandidateWriter, IO_BUFFER_BYTES}, sampled_sentiment::corpus::Envelope},
    hosted::corpus::{CorpusLimits, SampledSentimentCorpusConfig},
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
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_rows: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_threads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_reuse_execution: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_selected_rows: Option<usize>,
}

pub(in crate::candidate_cli) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    command: SampledSentimentCommand, common: CandidateArgs, limits: Limits, envelope: Envelope,
    input: R, output: W) -> Result<(), Failure> {
    let prefill = command.prefill().map_err(setup_failure)?;
    let session = Session::new_with_prefill(&common, limits, prefill).map_err(setup_failure)?;
    let check = || session.remaining().map(|_| ());
    let options = checked_stage(command.options(), &check, setup_failure)?;
    let budget = checked_stage(command.budget(limits, SentimentAxis::ALL.len()), &check, setup_failure)?;
    let encoding = checked_stage(common.input_encoding_policy(limits), &check, setup_failure)?;
    let facts = checked_stage(session.facts_with_error_detail(&common), &check, session_failure)?;
    let planner = checked_stage(Int8SampledSentimentPlanner::for_model_with_encoding(&facts,
        options, budget, command.task_limits(), encoding.clone()), &check, task_failure)?;
    let planner = checked_stage(command.configure_planner(planner), &check, setup_failure)?;
    if planner.selected_rows() != checked_stage(command.selected_rows(), &check, setup_failure)? {
        return Err(setup_failure(CandidateError::Planning));
    }
    let vocabulary = Arc::new(checked_stage(SampledSentimentVocabulary::pinned(), &check,
        |error| task_failure(error.into()))?);
    let provenance = Provenance { protocol: "fnlp-candidate-int8-sentiment-sampled-batch-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative", task: "sentiment-sampled",
        model_id: &facts.model_id, source_revision: &facts.revision, source_root_sha256: &facts.source_root_sha256,
        logical_model_sha256: &facts.logical_model_sha256, quant_recipe: &facts.recipe_id,
        input_encoding: input_encoding::reported_policy(&encoding), prefill_rows: prefill.map(|p| p.max_batch_rows),
        compute_threads: (limits.compute.threads > 1).then_some(limits.compute.threads),
        prefill_reuse_execution: command.prefill_reuse_execution(),
        max_selected_rows: command.max_selected_rows };
    let writer = checked_stage(CandidateWriter::new(output, &provenance,
        envelope.transport.max_output_line_bytes, envelope.output_bytes), &check, setup_failure)?;
    let reader: Box<dyn BufRead + Send> = if common.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        let file = checked_stage(crate::local_io::open_document(&common.input).map_err(|_| CandidateError::Input),
            &check, setup_failure)?;
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, file))
    };
    let cancellation = CancellationToken::default();
    let model = checked_stage(session.load_with_error_detail(&common, limits, &facts, cancellation.clone()), &check, session_failure)?;
    let config = SampledSentimentCorpusConfig { max_model_work: command.work_ceiling(),
        mask_limits: command.masks(), max_mask_node_visits: command.max_mask_node_visits };
    let corpus = CorpusLimits { native: checked_stage(session.native(&common), &check, setup_failure)?,
        transport: envelope.transport, preparation_reserve_bytes: limits.preparation_bytes,
        io_reserve_bytes: envelope.io_bytes };
    let summary = checked_stage(if limits.compute.threads == 1 {
        if command.reuse_prefill {
            session.engine.batch_int8_sampled_sentiment_reusing_prefill(&model, planner, vocabulary,
                config, corpus, prefill, reader, writer, cancellation)
        } else {
            session.engine.batch_int8_sampled_sentiment(&model, planner, vocabulary,
                config, corpus, prefill, reader, writer, cancellation)
        }
    } else {
        if command.reuse_prefill {
            session.engine.batch_int8_sampled_sentiment_reusing_prefill_with_threads(&model, planner, vocabulary,
                config, corpus, prefill, crate::native_engine::simd::config::process_kernel(),
                limits.compute.threads, reader, writer, cancellation)
        } else {
            session.engine.batch_int8_sampled_sentiment_with_threads(&model, planner, vocabulary,
                config, corpus, prefill, crate::native_engine::simd::config::process_kernel(),
                limits.compute.threads, reader, writer, cancellation)
        }
    }, &check, host_failure)?;
    if summary.failed > 0 { return Err(Failure::output("int8_sampled_sentiment_batch_items_failed")); }
    Ok(())
}

pub(super) fn failure(fault: BatchFault) -> Failure {
    if let Some(cause) = fault.cancellation { return cancelled(cause); }
    use BatchCode::*;
    let (exit, code) = match fault.code {
        InvalidLimits => (ErrorCode::Usage, "int8_sampled_sentiment_batch_limits"),
        InvalidUtf8 | InvalidJson | InvalidEnvelope | InvalidId | DuplicateId | Planning | InputIo =>
            (ErrorCode::InputDecodeOrParse, "int8_sampled_sentiment_batch_input"),
        LineLimit | DocumentLimit | EpochIdLimit | InputLimit | RequestLimit | OutputLineLimit | OutputLimit | WorkLimit =>
            (ErrorCode::BudgetOrTimeout, "int8_sampled_sentiment_batch_budget"),
        Admission | Allocation => (ErrorCode::AdmissionOrResourceLimit, "int8_sampled_sentiment_batch_admission"),
        Execution => (ErrorCode::StructuredTaskNoResult, "int8_sampled_sentiment_batch_execution"),
        _ => (ErrorCode::Generic, "int8_sampled_sentiment_batch_delivery_or_invariant"),
    };
    Failure { exit, code, cancellation: None }
}
