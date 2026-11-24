//! One resident model, one native engine and one owned schema-extraction stream.
use super::*;
use std::sync::Arc;
use super::super::extraction as planning;
use crate::{candidate_cli::extract::SCHEMA_BYTES,
    batch::extract::quantized::Int8ExtractionBatchPlanner,
    hosted::corpus::ExtractionDocumentCorpusConfig,
    tokenizer::untrusted::SafeBpeWorkLimits};

#[derive(Serialize)]
struct EncodingProvenance<'a, T: Serialize> {
    #[serde(flatten)]
    response: input_encoding::Response<'a, T>,
    /// Declared invocation caps, not an observation of tokenizer work spent.
    #[serde(skip_serializing_if = "Option::is_none")]
    invocation_encoding_work_limits: Option<SafeBpeWorkLimits>,
}

fn check_encoding_quote(command: &BatchCommand, limits: Limits,
    compiler: &Int8ExtractionBatchPlanner, document: Option<&ExtractionDocumentCorpusConfig>,
    expected_workspace: u64) -> Result<(), CandidateError> {
    let actual_workspace = if command.document.chunked {
        compiler.map_document_encoding_workspace_bytes()
    } else { compiler.encoding_workspace_bytes() }.map_err(|_| CandidateError::Planning)?;
    if compiler.document_encoding() != &command.encoding_policy(limits)?
        || compiler.map_document_encoding_limits() != command.document_encoding_work_limits()?
        || actual_workspace != expected_workspace || document.is_some() != command.document.chunked {
        return Err(CandidateError::Arguments);
    }
    if let Some(document) = document {
        // Pre-IO admission reserves the schema maximum. Bind the real complete
        // frontier quote to that bound before transferring this same planner.
        let expected = command.schedule_document_reserve()?.0.checked_add(expected_workspace)
            .ok_or(CandidateError::Arguments)?;
        if document.preparation_workspace_bytes(compiler).map_err(|_| CandidateError::Planning)? > expected {
            return Err(CandidateError::Arguments);
        }
    }
    Ok(())
}

pub(super) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    command: BatchCommand, args: CandidateArgs, limits: Limits, envelope: CorpusEnvelope,
    mut input: R, output: W,
) -> Result<(), CandidateError> {
    let encoding = command.encoding_policy(limits)?;
    let encoding_work_limits = command.document_encoding_work_limits()?;
    let encoding_workspace_bytes = command.encoding_workspace_bytes(limits)?;
    let (session, prefill) = scheduled_session(&command, &args, limits, envelope)?;
    let schema = command.schema.as_ref().map(|path| session.read(path, &mut input, SCHEMA_BYTES)).transpose()?;
    let raw_defaults = command.defaults.as_ref()
        .map(|path| session.read(path, &mut input, MAX_SOURCE_ARGUMENT_BYTES)).transpose()?;
    let defaults = command.load_extract_defaults(schema, raw_defaults.as_deref(), command.host.task_budget(limits))?;
    // Move the exact shared schema into the fixed document policy rather than
    // retaining an unused second default-schema copy inside the compiler.
    let (defaults, document) = if command.document.chunked {
        let request = defaults.ok_or(CandidateError::Arguments)?;
        (None, Some(ExtractionDocumentCorpusConfig {
            batch: command.document_configuration(request, envelope)?,
            reduction_reserve_bytes: command.document_reduction_bytes()?,
        }))
    } else { (defaults, None) };
    session.remaining()?;
    let facts = session.facts(&args)?;
    let compiler = planning::planner(&facts, &command.host, limits, defaults)?;
    let compiler = match encoding_work_limits {
        Some(total) => compiler.with_map_document_encoding(encoding.clone(), total),
        None => compiler.with_document_encoding(encoding.clone()),
    }.map_err(|_| CandidateError::Planning)?;
    check_encoding_quote(&command, limits, &compiler, document.as_ref(), encoding_workspace_bytes)?;
    session.remaining()?;
    let vocabulary = planning::vocabulary()?;
    session.remaining()?;
    let provenance = Provenance { protocol: "fnlp-candidate-batch-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative",
        model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, task: &command.task,
        compute_threads: (limits.compute.threads > 1).then_some(limits.compute.threads),
        prefill_rows: prefill.map(|limits| limits.max_batch_rows) };
    let provenance = EncodingProvenance { response: input_encoding::Response { response: &provenance,
        input_encoding: input_encoding::reported_policy(&encoding) },
        invocation_encoding_work_limits: encoding_work_limits };
    let writer = CandidateWriter::new(output, &provenance, envelope.transport.max_output_line_bytes,
        envelope.output_bytes)?;
    let reader: Box<dyn BufRead + Send> = if command.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES,
            File::open(&command.input).map_err(|_| CandidateError::Input)?))
    };
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let corpus = CorpusLimits { native: session.native(&args)?, transport: envelope.transport,
        preparation_reserve_bytes: limits.preparation_bytes, io_reserve_bytes: envelope.io_bytes };
    // Both routes cross the process host ONCE and retain owned IO through
    // physical completion. Chunked documents do not loop over host invocations
    // or replenish native/mask/verification allowances at document/flush boundaries.
    let width = limits.compute.threads;
    let summary = if width == 1 {
        match document {
            Some(config) => match prefill {
                None => session.engine.batch_int8_extract_document(&model, Arc::new(compiler),
                    vocabulary, config, corpus, reader, writer, cancellation),
                Some(prefill) => session.engine.batch_int8_extract_document_layer_major(&model,
                    Arc::new(compiler), vocabulary, config, corpus, prefill, reader, writer, cancellation),
            },
            None => match prefill {
                None => session.engine.batch_int8_extract(&model, compiler, vocabulary, envelope.native,
                    corpus, reader, writer, cancellation),
                Some(prefill) => session.engine.batch_int8_extract_layer_major(&model, compiler,
                    vocabulary, envelope.native, corpus, prefill, reader, writer, cancellation),
            },
        }
    } else {
        let kernel = crate::native_engine::simd::config::process_kernel();
        match document {
            Some(config) => match prefill {
                None => session.engine.batch_int8_extract_document_with_threads(&model, Arc::new(compiler),
                    vocabulary, config, corpus, kernel, width, reader, writer, cancellation),
                Some(prefill) => session.engine.batch_int8_extract_document_layer_major_with_threads(&model,
                    Arc::new(compiler), vocabulary, config, corpus, kernel, width, prefill, reader, writer, cancellation),
            },
            None => match prefill {
                None => session.engine.batch_int8_extract_with_threads(&model, compiler, vocabulary,
                    envelope.native, corpus, kernel, width, reader, writer, cancellation),
                Some(prefill) => session.engine.batch_int8_extract_layer_major_with_threads(&model,
                    compiler, vocabulary, envelope.native, corpus, kernel, width, prefill, reader, writer, cancellation),
            },
        }
    }.map_err(|_| CandidateError::Batch)?;
    // A successfully drained stream can still contain failed documents. Keep
    // its nonzero exit and never append a second stdout completion/error frame.
    command::completed(summary)
}

#[cfg(test)] mod tests;
