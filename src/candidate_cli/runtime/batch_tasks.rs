//! One process-owned blocking invocation for a bounded source-task corpus.
use super::*;
use std::{io::{BufRead, BufReader}, sync::Arc};
use crate::{
    batch::source::quantized::{check_configuration, MAX_SOURCE_ARGUMENT_BYTES},
    hosted::corpus::{CorpusLimits, SourceCorpusConfig},
};
use crate::candidate_cli::batch::{self as command, BatchCommand, CandidateWriter, CorpusEnvelope, IO_BUFFER_BYTES};
use super::source_tasks::{Session, planner, source_identity};
mod extraction;

#[derive(Serialize)]
struct Provenance<'a> {
    protocol: &'static str,
    schema_version: u32,
    scope: &'static str,
    evidence: &'static str,
    model_id: &'a str,
    source_revision: &'a str,
    source_root_sha256: &'a str,
    logical_model_sha256: &'a str,
    quant_recipe: &'a str,
    task: &'a str,
}

pub(in crate::candidate_cli) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    command: BatchCommand, args: CandidateArgs, limits: Limits, envelope: CorpusEnvelope,
    mut input: R, output: W,
) -> Result<(), CandidateError> {
    if command.task == "extract" {
        return extraction::execute(command, args, limits, envelope, input, output);
    }
    let session = Session::new(&args, limits)?;
    let raw_defaults = command.defaults.as_ref()
        .map(|path| session.read(path, &mut input, MAX_SOURCE_ARGUMENT_BYTES)).transpose()?;
    let ceiling = command.host.task_budget(limits);
    let defaults = command.load_defaults(raw_defaults.as_deref(), ceiling)?;
    let facts = session.facts(&args)?;
    let (planner, vocabulary) = planner()?;
    session.remaining()?;
    let identity = source_identity(&facts, &planner, command.task()?)?;
    // Validate the fixed backend/task/template/default contract before weights.
    // Per-document source/prompt compilation remains bounded inside the one
    // hosted corpus invocation using its SAME nonrenewable run control.
    check_configuration(&planner, &identity, ceiling, command.host.planning(), defaults.as_ref())
        .map_err(|_| CandidateError::Planning)?;
    let config = SourceCorpusConfig { identity, task_ceiling: ceiling,
        planning: command.host.planning(), defaults, native_work: envelope.native };
    let provenance = Provenance { protocol: "fnlp-candidate-batch-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative",
        model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, task: &command.task };
    let writer = CandidateWriter::new(output, &provenance, envelope.transport.max_output_line_bytes,
        envelope.output_bytes)?;
    // These OWNED handles cross the blocking boundary. In particular, the CLI
    // thread must not hold stdin/stdout locks while the worker uses those same
    // streams. No ad-hoc pipe thread, whole-input buffer or detached writer.
    let reader: Box<dyn BufRead + Send> = if command.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES,
            File::open(&command.input).map_err(|_| CandidateError::Input)?))
    };
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let summary = session.engine.batch_int8_source(&model, Arc::new(planner), Arc::new(vocabulary),
        config, CorpusLimits { native: session.native(&args)?, transport: envelope.transport,
            preparation_reserve_bytes: limits.preparation_bytes, io_reserve_bytes: envelope.io_bytes },
        reader, writer, cancellation).map_err(|_| CandidateError::Batch)?;
    // run_complete acknowledges the protocol, NOT all documents succeeding.
    // Native failure is never turned into successful QA abstention. No second
    // stdout terminal frame is appended after a poisoned or partial transport.
    command::completed(summary)
}
