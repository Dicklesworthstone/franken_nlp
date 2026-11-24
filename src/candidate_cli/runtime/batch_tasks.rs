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
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_threads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_rows: Option<usize>,
}

fn scheduled_session(command: &BatchCommand, args: &CandidateArgs, limits: Limits, envelope: CorpusEnvelope)
    -> Result<(Session, Option<crate::native_engine::strict_int8::prefill::Int8PrefillLimits>), CandidateError> {
    let prefill = command.prefill.limits()?;
    command.check_scheduling_memory(limits, envelope, prefill)?;
    if args.compute != command.compute { return Err(CandidateError::Arguments); }
    Ok((Session::new_with_prefill(args, limits, prefill)?, prefill))
}

pub(in crate::candidate_cli) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    command: BatchCommand, args: CandidateArgs, limits: Limits, envelope: CorpusEnvelope,
    mut input: R, output: W,
) -> Result<(), CandidateError> {
    if command.task == "extract" {
        return extraction::execute(command, args, limits, envelope, input, output);
    }
    let encoding = args.input_encoding_policy(limits)?;
    let (session, prefill) = scheduled_session(&command, &args, limits, envelope)?;
    let raw_defaults = command.defaults.as_ref()
        .map(|path| session.read(path, &mut input, MAX_SOURCE_ARGUMENT_BYTES)).transpose()?;
    let ceiling = command.host.task_budget(limits);
    let defaults = command.load_defaults(raw_defaults.as_deref(), ceiling)?;
    let facts = session.facts(&args)?;
    let (planner, vocabulary) = planner()?;
    let planner = planner.with_document_encoding(encoding.clone()).map_err(|_| CandidateError::Planning)?;
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
        quant_recipe: &facts.recipe_id, task: &command.task,
        compute_threads: (limits.compute.threads > 1).then_some(limits.compute.threads),
        prefill_rows: prefill.map(|limits| limits.max_batch_rows) };
    let provenance = input_encoding::Response { response: &provenance,
        input_encoding: input_encoding::reported_policy(&encoding) };
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
    let corpus = CorpusLimits { native: session.native(&args)?, transport: envelope.transport,
        preparation_reserve_bytes: limits.preparation_bytes, io_reserve_bytes: envelope.io_bytes };
    let planner = Arc::new(planner);
    let vocabulary = Arc::new(vocabulary);
    let width = limits.compute.threads;
    let summary = if width == 1 {
        match prefill {
            None => session.engine.batch_int8_source(&model, planner, vocabulary, config,
                corpus, reader, writer, cancellation),
            Some(prefill) => session.engine.batch_int8_source_layer_major(&model, planner, vocabulary,
                config, corpus, prefill, reader, writer, cancellation),
        }
    } else {
        let kernel = crate::native_engine::simd::config::process_kernel();
        match prefill {
            None => session.engine.batch_int8_source_with_threads(&model, planner, vocabulary, config,
                corpus, kernel, width, reader, writer, cancellation),
            Some(prefill) => session.engine.batch_int8_source_layer_major_with_threads(&model, planner,
                vocabulary, config, corpus, kernel, width, prefill, reader, writer, cancellation),
        }
    }.map_err(|_| CandidateError::Batch)?;
    // run_complete acknowledges the protocol, NOT all documents succeeding.
    // Native failure is never turned into successful QA abstention. No second
    // stdout terminal frame is appended after a poisoned or partial transport.
    command::completed(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn command(task: &str) -> BatchCommand {
        let matches = crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "batch", "--task", task, "--model", "must-not-open.fnlpq",
            "--memory-mib", "16384", "--compute-threads", "4", "--prefill-rows", "7",
        ]).unwrap();
        let CandidateCommand::Batch(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong route") };
        command
    }
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private source read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("partial output written") }
        fn flush(&mut self) -> io::Result<()> { panic!("output flushed") }
    }
    #[test]
    fn transferred_schedule_and_memory_are_revalidated_before_any_owned_io() {
        for task in ["ner", "keyphrases", "summarize", "answer", "extract"] {
            for corruption in 0..3 {
                let cmd = command(task);
                let (mut args, mut limits, envelope) = cmd.validate().unwrap();
                match corruption {
                    0 => limits.compute.workspace_bytes -= 1,
                    1 => args.compute = compute::Args::default(),
                    _ => limits.memory_bytes = cmd.schedule_peak_bytes(limits, envelope,
                        cmd.prefill.limits().unwrap()).unwrap() - 1,
                }
                assert_eq!(execute(cmd, args, limits, envelope, NoIo, NoIo), Err(CandidateError::Arguments));
            }
        }
    }
    #[test]
    fn scheduling_provenance_is_optional_and_reports_only_the_selected_physical_schedule() {
        let mut p = Provenance { protocol: "fnlp-candidate-batch-v1", schema_version: 1,
            scope: "real-artifact-current-candidate", evidence: "non_authoritative",
            model_id: "model", source_revision: "revision", source_root_sha256: "source",
            logical_model_sha256: "logical", quant_recipe: "recipe", task: "ner",
            compute_threads: None, prefill_rows: None };
        let plain = serde_json::to_value(&p).unwrap();
        assert!(plain.get("compute_threads").is_none()); assert!(plain.get("prefill_rows").is_none());
        p.compute_threads = Some(4); p.prefill_rows = Some(7);
        let selected = serde_json::to_value(&p).unwrap();
        assert_eq!(selected["compute_threads"], 4); assert_eq!(selected["prefill_rows"], 7);
        for field in ["protocol", "schema_version", "scope", "evidence", "model_id", "source_revision",
            "source_root_sha256", "logical_model_sha256", "quant_recipe", "task"] {
            assert_eq!(selected[field], plain[field]);
        }
    }
}
