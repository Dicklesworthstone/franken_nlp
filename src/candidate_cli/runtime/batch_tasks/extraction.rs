//! One resident model, one native engine and one owned schema-extraction stream.
use super::*;
use super::super::extraction as planning;
use crate::candidate_cli::extract::SCHEMA_BYTES;

pub(super) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    command: BatchCommand, args: CandidateArgs, limits: Limits, envelope: CorpusEnvelope,
    mut input: R, output: W,
) -> Result<(), CandidateError> {
    let session = Session::new(&args, limits)?;
    let schema = command.schema.as_ref().map(|path| session.read(path, &mut input, SCHEMA_BYTES)).transpose()?;
    let raw_defaults = command.defaults.as_ref()
        .map(|path| session.read(path, &mut input, MAX_SOURCE_ARGUMENT_BYTES)).transpose()?;
    let defaults = command.load_extract_defaults(schema, raw_defaults.as_deref(), command.host.task_budget(limits))?;
    session.remaining()?;
    let facts = session.facts(&args)?;
    let compiler = planning::planner(&facts, &command.host, limits, defaults)?;
    session.remaining()?;
    let vocabulary = planning::vocabulary()?;
    session.remaining()?;
    let provenance = Provenance { protocol: "fnlp-candidate-batch-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative",
        model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, task: &command.task };
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
    // Int8SourceBatchLimits is the existing alias of Int8ExtractionBatchLimits:
    // schema extraction pays the same complete native/mask and transport axes.
    // The hosted adapter supplies real admission and retains all owned IO until
    // physical completion. Schema/source plans share its one run control.
    let summary = session.engine.batch_int8_extract(&model, compiler, vocabulary, envelope.native,
        CorpusLimits { native: session.native(&args)?, transport: envelope.transport,
            preparation_reserve_bytes: limits.preparation_bytes, io_reserve_bytes: envelope.io_bytes },
        reader, writer, cancellation).map_err(|_| CandidateError::Batch)?;
    // A successfully drained stream can still contain failed documents. Keep
    // its nonzero exit and never append a second stdout completion/error frame.
    command::completed(summary)
}

#[cfg(test)] mod tests;
