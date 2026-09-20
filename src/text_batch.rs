//! Model-free, bounded NDJSON text processing over the existing batch protocol.
//!
//! One input record is completed and flushed before the next is admitted. This
//! is not model inference, a parallel scheduler, a durable job, or an extension
//! of `robot schema`. Text and coordinate semantics are the ordinary textutil
//! semantics, including exact original-source coordinates and no Unicode repair.

use std::{io::{BufRead, Write}, path::PathBuf, process::ExitCode};

use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};

use crate::{
    batch::{self, BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchLimits,
        BatchProcessor, BatchWork},
    error::ErrorCode,
    native_engine::decode::{DecodeCancellationKind, DecodeStepControl},
    textutil::{self, NormalizeOptions, SplitOptions, TextBudget, TextError},
};

const MAX_BYTES: usize = 64 * 1024 * 1024;
const TASK_ARGUMENT_BYTES: usize = 64 * 1024;
// Room for caller-id escaping, delivery coordinates, work and protocol fields.
const ENVELOPE_RESERVE: usize = 2048;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum TextBatchTask { Normalize, Split }

fn default_chunk_bytes() -> usize { 4096 }

/// A complete per-item override, not a merge with the command defaults. The
/// discriminant must match the selected task. Unknown fields are refused.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TextBatchOptions {
    Normalize {
        #[serde(default)] trim_ascii_horizontal: bool,
        #[serde(default)] collapse_ascii_horizontal: bool,
    },
    Split {
        #[serde(default = "default_chunk_bytes")] max_chunk_bytes: usize,
    },
}
impl TextBatchOptions {
    pub fn for_task(task: TextBatchTask) -> Self {
        match task {
            TextBatchTask::Normalize => Self::Normalize {
                trim_ascii_horizontal: false, collapse_ascii_horizontal: false,
            },
            TextBatchTask::Split => Self::Split { max_chunk_bytes: default_chunk_bytes() },
        }
    }
    pub fn task(self) -> TextBatchTask {
        match self { Self::Normalize { .. } => TextBatchTask::Normalize, Self::Split { .. } => TextBatchTask::Split }
    }
    fn validate(self) -> Result<(), BatchFault> {
        if let Self::Split { max_chunk_bytes } = self {
            if !(4..=MAX_BYTES).contains(&max_chunk_bytes) { return Err(BatchCode::Planning.into()); }
        }
        Ok(())
    }
}

/// Request-owned prepared content is intentionally neither Debug nor serde.
pub struct PreparedText { text: String, options: TextBatchOptions }

#[derive(Serialize)]
pub struct TextBatchOutput {
    pub schema_version: u32,
    pub task: &'static str,
    /// The same versioned data as the corresponding single-document utility.
    /// Normalization retains coordinate edits; splitting retains every byte.
    pub result: serde_json::Value,
}

pub struct TextBatchProcessor { defaults: TextBatchOptions, budget: TextBudget }
impl TextBatchProcessor {
    pub fn new(defaults: TextBatchOptions, budget: TextBudget) -> Result<Self, BatchFault> {
        defaults.validate()?;
        budget.validate().map_err(|_| BatchCode::InvalidLimits)?;
        Ok(Self { defaults, budget })
    }
}
impl BatchProcessor for TextBatchProcessor {
    type Args = TextBatchOptions;
    type Prepared = PreparedText;
    type Output = TextBatchOutput;

    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        let options = document.task_args.unwrap_or(self.defaults);
        options.validate().map_err(|fault| BatchItemFailure { fault, stop: false })?;
        if options.task() != self.defaults.task() { return Err(BatchItemFailure::reject(BatchCode::Planning)); }
        if document.text.len() > self.budget.max_input_bytes {
            return Err(BatchItemFailure::reject(BatchCode::DocumentLimit));
        }
        Ok(PreparedText { text: document.text, options })
    }
    fn planned_work(&self, _: &Self::Prepared) -> BatchWork { BatchWork::default() }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        checkpoint(control)?;
        let output = match prepared.options {
            TextBatchOptions::Normalize { trim_ascii_horizontal, collapse_ascii_horizontal } => {
                let value = textutil::normalize(&prepared.text,
                    NormalizeOptions { trim_ascii_horizontal, collapse_ascii_horizontal }, self.budget).map_err(text_failure)?;
                owned_result("normalize", &value, self.budget.max_output_bytes)?
            }
            TextBatchOptions::Split { max_chunk_bytes } => {
                let value = textutil::split(&prepared.text, SplitOptions { max_chunk_bytes }, self.budget).map_err(text_failure)?;
                owned_result("split", &value, self.budget.max_output_bytes)?
            }
        };
        checkpoint(control)?;
        Ok(output)
    }
}

fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), BatchItemFailure> {
    match control.checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))),
        None => Ok(()),
    }
}
fn text_failure(error: TextError) -> BatchItemFailure {
    let code = match error {
        TextError::InvalidOptions => BatchCode::Planning,
        TextError::InputBudget => BatchCode::DocumentLimit,
        TextError::OutputBudget | TextError::ItemBudget => BatchCode::OutputLineLimit,
        TextError::AllocationRefused => BatchCode::Allocation,
        TextError::InputRead => BatchCode::InputIo,
        TextError::InvalidUtf8 => BatchCode::InvalidUtf8,
        TextError::Serialization => BatchCode::Serialization,
    };
    BatchItemFailure::reject(code)
}
fn owned_result(task: &'static str, value: &impl Serialize, cap: usize) -> Result<TextBatchOutput, BatchItemFailure> {
    // Textutil has already checked the original result's size. This one-record
    // owned representation does not retain a reference into the input buffer.
    let result = serde_json::to_value(value).map_err(|_| BatchItemFailure::reject(BatchCode::Serialization))?;
    let output = TextBatchOutput { schema_version: 1, task, result };
    let mut counter = SizeCounter { remaining: cap, exceeded: false };
    if serde_json::to_writer(&mut counter, &output).is_err() {
        return Err(BatchItemFailure::reject(if counter.exceeded { BatchCode::OutputLineLimit } else { BatchCode::Serialization }));
    }
    Ok(output)
}
struct SizeCounter { remaining: usize, exceeded: bool }
impl Write for SizeCounter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.remaining {
            self.exceeded = true;
            return Err(std::io::Error::other("text batch output bound"));
        }
        self.remaining -= bytes.len(); Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

#[derive(Args)]
pub(crate) struct BatchCommand {
    /// Currently available model-free tasks; unavailable model tasks are refused.
    #[arg(long, value_enum)] task: TextBatchTask,
    /// Bounded JSON defaults, including the task's kind. Never executable code.
    #[arg(long)] task_args: Option<PathBuf>,
    #[arg(long, default_value_t = 1024 * 1024)] max_line_bytes: usize,
    #[arg(long, default_value_t = 512 * 1024)] max_document_bytes: usize,
    #[arg(long, default_value_t = 1024 * 1024 * 1024)] max_input_bytes: u64,
    #[arg(long, default_value_t = 100_000)] max_requests: u64,
    #[arg(long, default_value_t = 4 * 1024 * 1024)] max_output_line_bytes: usize,
    #[arg(long, default_value_t = 1024 * 1024 * 1024)] max_output_bytes: u64,
    /// Maximum distinct caller IDs between flush controls.
    #[arg(long, default_value_t = 4096)] max_epoch_ids: usize,
    /// Maximum retained caller-ID bytes between flush controls.
    #[arg(long, default_value_t = 256 * 1024)] max_epoch_id_bytes: usize,
    /// Maximum normalization edits or split chunks in one document.
    #[arg(long, default_value_t = 16_384)] max_items: usize,
}

pub(crate) fn definition() -> clap::Command {
    BatchCommand::augment_args(clap::Command::new("batch"))
        .about("Process bounded NDJSON documents with model-free tasks")
        .after_help("Input: {\"id\":\"doc-1\",\"text\":\"original UTF-8\",\"task_args\":null}. Optional task_args is a complete override with kind=normalize or split. {\"flush\":true} acknowledges the epoch and permits ID reuse. Output uses fnlp-item-local-batch-v1, not robot schema. One document is live; each event is flushed before further input. Empty LF/CRLF records are ignored; malformed records get typed errors and processing continues. Any rejected document makes the process exit nonzero, even when run_complete is emitted. No weights, network, background pool, file output or implicit persistence. A complete run requires its terminal event; a terminated process may have delivered only a prefix.")
}
impl BatchCommand {
    fn configure(&self) -> Result<(TextBatchProcessor, BatchLimits), ErrorCode> {
        if self.max_line_bytes > MAX_BYTES || self.max_output_line_bytes > MAX_BYTES
            || self.max_output_line_bytes < 2 * ENVELOPE_RESERVE || self.max_epoch_ids > 1_000_000
            || self.max_epoch_id_bytes > MAX_BYTES { return Err(ErrorCode::Usage); }
        let limits = BatchLimits {
            max_line_bytes: self.max_line_bytes, max_document_bytes: self.max_document_bytes,
            max_input_bytes: self.max_input_bytes, max_requests: self.max_requests,
            max_output_line_bytes: self.max_output_line_bytes, max_output_bytes: self.max_output_bytes,
            max_epoch_ids: self.max_epoch_ids, max_epoch_id_bytes: self.max_epoch_id_bytes,
            // Non-model processors never claim or consume native forward work.
            max_work: BatchWork::default(), ..BatchLimits::default()
        };
        limits.validate().map_err(|_| ErrorCode::Usage)?;
        let defaults = match &self.task_args {
            None => TextBatchOptions::for_task(self.task),
            Some(path) => {
                if path.as_os_str() == "-" { return Err(ErrorCode::Usage); }
                let mut file = crate::local_io::open_document(path).map_err(|_| ErrorCode::InputDecodeOrParse)?;
                let json = textutil::read_utf8(&mut file, TASK_ARGUMENT_BYTES).map_err(|_| ErrorCode::InputDecodeOrParse)?;
                // Reject duplicate keys before serde could keep the last value.
                crate::validation::parse_json_with_limits(&json, crate::validation::JsonLimits {
                    max_input_bytes: TASK_ARGUMENT_BYTES, max_string_lexeme_bytes: TASK_ARGUMENT_BYTES,
                    max_depth: 4, max_container_entries: 16, ..Default::default()
                }).map_err(|_| ErrorCode::InputDecodeOrParse)?;
                serde_json::from_str(&json).map_err(|_| ErrorCode::InputDecodeOrParse)?
            }
        };
        if defaults.task() != self.task { return Err(ErrorCode::Usage); }
        let processor = TextBatchProcessor::new(defaults, TextBudget {
            max_input_bytes: self.max_document_bytes,
            max_output_bytes: self.max_output_line_bytes - ENVELOPE_RESERVE,
            max_items: self.max_items,
        }).map_err(|_| ErrorCode::Usage)?;
        Ok((processor, limits))
    }
    pub(crate) fn run(self, input: &mut impl BufRead, output: &mut impl Write, error: &mut impl Write) -> ExitCode {
        let (mut processor, limits) = match self.configure() {
            Ok(value) => value,
            Err(code) => {
                let _ = writeln!(error, "fnlp: invalid or unreadable batch configuration").and_then(|()| error.flush());
                return code.as_process_exit();
            }
        };
        match batch::run_ndjson(input, output, &mut processor, limits, &mut CliControl) {
            Ok(summary) if summary.failed == 0 => ExitCode::SUCCESS,
            Ok(_) => ErrorCode::InputDecodeOrParse.as_process_exit(),
            Err(failure) => {
                // The runner owns structured errors, terminal reserve and output
                // poisoning. Never append a second stdout event after I/O fails.
                let _ = writeln!(error, "fnlp: batch did not complete").and_then(|()| error.flush());
                match failure.fault.code {
                    BatchCode::InputIo | BatchCode::InvalidUtf8 | BatchCode::InvalidJson | BatchCode::InvalidEnvelope =>
                        ErrorCode::InputDecodeOrParse.as_process_exit(),
                    BatchCode::OutputIo | BatchCode::Serialization => ErrorCode::Generic.as_process_exit(),
                    _ => ErrorCode::AdmissionOrResourceLimit.as_process_exit(),
                }
            }
        }
    }
}
struct CliControl;
impl DecodeStepControl for CliControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::FromArgMatches;
    use serde_json::{Value, json};
    fn command(args: &[&str]) -> BatchCommand {
        BatchCommand::from_arg_matches(&definition().try_get_matches_from(args).unwrap()).unwrap()
    }
    fn run(args: &[&str], source: &[u8]) -> (ExitCode, Vec<Value>, Vec<u8>) {
        let (mut output, mut error) = (Vec::new(), Vec::new());
        let code = command(args).run(&mut &*source, &mut output, &mut error);
        let rows = output.split(|b| *b == b'\n').filter(|s| !s.is_empty())
            .map(|s| serde_json::from_slice(s).unwrap()).collect();
        (code, rows, error)
    }
    fn documents(rows: &[Value]) -> Vec<&Value> { rows.iter().filter_map(|row| row.get("result")).collect() }
    #[test]
    fn normalize_stream_preserves_edits_and_reports_zero_model_work() {
        let input = b"{\"id\":\"a\",\"text\":\"x\\r\\ny\"}\n{\"id\":\"b\",\"text\":\"z\"}";
        let (code, rows, errors) = run(&["batch", "--task", "normalize"], input);
        assert_eq!(code, ExitCode::SUCCESS); assert!(errors.is_empty());
        let outputs = documents(&rows); assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0]["result"]["text"], "x\ny");
        assert_eq!(outputs[0]["result"]["edits"].as_array().unwrap().len(), 1);
        let last = rows.last().unwrap(); assert_eq!(last["event"], "run_complete");
        assert_eq!(last["summary"]["succeeded"], 2);
        assert_eq!(last["summary"]["reserved_work"], json!({"forward_positions":0,"projected_logits":0}));
        assert!(rows.iter().all(|r| r["protocol"] == batch::BATCH_PROTOCOL));
    }
    #[test]
    fn split_stream_roundtrips_unicode_crlf_and_original_coordinates() {
        let text = "é AB\r\n😀 e\u{301}\t尾";
        let input = json!({"id":"s","text":text,"task_args":{"kind":"split","max_chunk_bytes":5}}).to_string();
        let (code, rows, _) = run(&["batch", "--task", "split"], input.as_bytes());
        assert_eq!(code, ExitCode::SUCCESS);
        let result = documents(&rows); let chunks = result[0]["result"]["chunks"].as_array().unwrap();
        let recovered: String = chunks.iter().map(|c| c["text"].as_str().unwrap()).collect();
        assert_eq!(recovered, text);
        for chunk in chunks {
            let start = chunk["span"]["byte_start"].as_u64().unwrap() as usize;
            let end = chunk["span"]["byte_end"].as_u64().unwrap() as usize;
            assert_eq!(&text[start..end], chunk["text"].as_str().unwrap());
            assert_eq!(text[..start].chars().count(), chunk["span"]["scalar_start"].as_u64().unwrap() as usize);
            assert!(end - start <= 5);
        }
    }
    #[test]
    fn per_item_override_is_complete_and_does_not_mutate_later_defaults() {
        let input = concat!("{\"id\":\"a\",\"text\":\"  x  \",\"task_args\":{\"kind\":\"normalize\",\"trim_ascii_horizontal\":true}}\n",
            "{\"id\":\"b\",\"text\":\"  x  \"}\n");
        let (code, rows, _) = run(&["batch", "--task", "normalize"], input.as_bytes());
        assert_eq!(code, ExitCode::SUCCESS); let output = documents(&rows);
        assert_eq!(output[0]["result"]["text"], "x"); assert_eq!(output[1]["result"]["text"], "  x  ");
    }
    #[test]
    fn malformed_duplicate_and_wrong_task_records_do_not_hide_later_success() {
        let input = concat!("not json\n", "{\"id\":\"x\",\"text\":\"a\"}\n",
            "{\"id\":\"x\",\"text\":\"b\"}\n", "{\"id\":\"y\",\"text\":\"a\",\"task_args\":{\"kind\":\"split\"}}\n",
            "{\"id\":\"z\",\"text\":\"ok\"}\n");
        let (code, rows, _) = run(&["batch", "--task", "normalize"], input.as_bytes());
        assert_ne!(code, ExitCode::SUCCESS); assert_eq!(documents(&rows).len(), 2);
        assert_eq!(rows.last().unwrap()["summary"]["failed"], 3);
        assert_eq!(rows.last().unwrap()["event"], "run_complete");
    }
    #[test]
    fn explicit_flush_permits_id_reuse_but_never_reuses_delivery_sequence() {
        let input = b"{\"id\":\"x\",\"text\":\"a\"}\n{\"flush\":true}\n{\"id\":\"x\",\"text\":\"b\"}\n";
        let (code, rows, _) = run(&["batch", "--task", "normalize"], input);
        assert_eq!(code, ExitCode::SUCCESS);
        let results: Vec<_> = rows.iter().filter(|r| r.get("result").is_some()).collect();
        assert_eq!(results.len(), 2); assert_ne!(results[0]["epoch"], results[1]["epoch"]);
        assert_ne!(results[0]["request_seq"], results[1]["request_seq"]);
    }
    #[test]
    fn private_malformed_content_never_appears_in_error_records() {
        let secret = "PRIVATE_DOCUMENT_MARKER_2949";
        let (code, rows, errors) = run(&["batch", "--task", "normalize"], secret.as_bytes());
        assert_ne!(code, ExitCode::SUCCESS);
        assert!(!serde_json::to_string(&rows).unwrap().contains(secret));
        assert!(!String::from_utf8(errors).unwrap().contains(secret));
    }
    #[test]
    fn invalid_options_are_rejected_before_reading_input() {
        struct NoRead;
        impl std::io::Read for NoRead { fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> { panic!("input consumed") } }
        impl BufRead for NoRead {
            fn fill_buf(&mut self) -> std::io::Result<&[u8]> { panic!("input consumed") }
            fn consume(&mut self, _: usize) { panic!("input consumed") }
        }
        let mut out = Vec::new(); let mut err = Vec::new();
        let code = command(&["batch", "--task", "split", "--max-output-line-bytes", "1"])
            .run(&mut NoRead, &mut out, &mut err);
        assert_ne!(code, ExitCode::SUCCESS); assert!(out.is_empty());
        assert!(definition().try_get_matches_from(["batch", "--task", "generate"]).is_err());
    }
    #[test]
    fn cancelled_library_execution_is_fatal_and_never_returns_transformed_text() {
        struct Cancel;
        impl DecodeStepControl for Cancel {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
        }
        let mut processor = TextBatchProcessor::new(TextBatchOptions::for_task(TextBatchTask::Normalize), TextBudget::default()).unwrap();
        let prepared = processor.prepare(BatchDocument { id: "x".to_owned(), text: "private".to_owned(), task_args: None }).unwrap();
        let failure = match processor.execute(prepared, &mut Cancel) { Err(failure) => failure, Ok(_) => panic!("cancelled output") };
        assert!(failure.stop); assert_eq!(failure.fault.code, BatchCode::Cancelled);
    }
    #[test]
    fn malformed_task_arguments_reject_unknown_fields_and_duplicate_keys() {
        for args in [r#"{"kind":"normalize","shell":"anything"}"#, r#"{"kind":"normalize","kind":"split"}"#] {
            let input = format!("{{\"id\":\"x\",\"text\":\"a\",\"task_args\":{args}}}\n");
            let (code, rows, _) = run(&["batch", "--task", "normalize"], input.as_bytes());
            assert_ne!(code, ExitCode::SUCCESS); assert!(documents(&rows).is_empty());
        }
    }
    #[test]
    fn output_failure_stops_before_another_document_is_admitted() {
        struct FailingWriter;
        impl Write for FailingWriter {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> { Err(std::io::Error::other("private adapter error")) }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        let mut input = std::io::Cursor::new(b"{\"id\":\"x\",\"text\":\"a\"}\n");
        let mut err = Vec::new();
        let code = command(&["batch", "--task", "normalize"]).run(&mut input, &mut FailingWriter, &mut err);
        assert_ne!(code, ExitCode::SUCCESS); assert_eq!(input.position(), 0);
        assert!(!String::from_utf8(err).unwrap().contains("private adapter error"));
    }
    #[test]
    fn empty_stream_completes_and_blank_lines_do_not_become_requests() {
        for input in [b"".as_slice(), b"\n\r\n".as_slice()] {
            let (code, rows, _) = run(&["batch", "--task", "normalize"], input);
            assert_eq!(code, ExitCode::SUCCESS); assert_eq!(rows.last().unwrap()["summary"]["requests"], 0);
        }
    }
}
