//! Bounded, fail-fast text generation corpus protocol, not token streaming.
use super::*;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, io::BufRead};

pub(super) const PROTOCOL: &str = "fnlp-candidate-text-batch-v1";
pub(super) const FOOTER_BYTES: usize = 4096;

#[derive(Clone, Copy, clap::ValueEnum)]
pub(super) enum TextTask { Generate, Chat }

#[derive(Args)]
pub(crate) struct TextBatchCommand {
    /// Input is NDJSON with unique id, optional sample_index, and prompt/messages.
    #[arg(long, value_enum, default_value = "generate")]
    pub(super) task: TextTask,
    /// Shared per-record generation and host options; input is an NDJSON file.
    #[command(flatten)]
    pub(super) common: CandidateArgs,
    /// Explicit cross-document INT8 groups, 1..=64. Omitted keeps serial records.
    /// A cohort shares one checkpoint budget; cannot combine with prefill-rows.
    #[arg(long, value_name = "ROWS")]
    pub(super) cohort_rows: Option<usize>,
    /// Whole-corpus record ceiling. IDs are retained only for duplicate refusal.
    #[arg(long, default_value_t = 1000)]
    pub(super) max_records: u64,
    /// Entire consumed NDJSON input, including record framing and newlines.
    #[arg(long, default_value_t = 67_108_864)]
    pub(super) max_total_input_bytes: u64,
    /// Entire output transport, including the mandatory batch_complete frame.
    #[arg(long, default_value_t = 67_108_864)]
    pub(super) max_total_output_bytes: u64,
    /// Sum of admitted worst-case native forward positions, without refunds.
    #[arg(long, default_value_t = 1_000_000)]
    pub(super) max_total_forward_positions: u64,
    /// Sum of admitted worst-case projected logits, without refunds.
    #[arg(long, default_value_t = 100_000_000_000)]
    pub(super) max_total_projected_logits: u64,
}

pub(super) fn definition() -> clap::Command {
    TextBatchCommand::augment_args(clap::Command::new("text-batch")
        .about("Generate or chat over bounded NDJSON with one resident local INT8 model")
        .long_about("Explicit non-certified local candidate only. Each generate record is {\"id\":\"unique\",\"prompt\":\"...\"}; chat uses a messages array instead. Optional sample_index is a u64. IDs, not physical row positions, address seeded draws. Generation options apply to every record. max-input-bytes includes each record's JSON framing; max-checkpoints is per native invocation (one record by default, one whole cohort with --cohort-rows); timeout-seconds covers the entire invocation. One model is loaded lazily and reused, with fresh request state. --cohort-rows explicitly groups 1..=64 documents with independent KV and shared-weight computation; it cannot combine with --prefill-rows. Preparation, simultaneous sampler/KV and all retained outputs must fit the memory ledger. Output is ordered completed-result NDJSON, never provisional tokens. The batch_complete frame plus successful process exit is required for whole-corpus success. Invalid input, duplicate IDs, budgets, native failure or a broken output stream stop the corpus without retries; earlier completed records remain valid. A cohort must fully execute and validate before its first result is published. Empty input completes without opening the model. Blocking IO is cooperative, not preemptible. No network or tool execution."))
}

impl TextBatchCommand {
    pub(super) fn validate(&self) -> Result<Limits, CandidateError> {
        let limits = self.common.validate()?;
        if let Some(rows) = self.cohort_rows {
            if rows == 0 || rows > crate::native_engine::portable_int8::batch::MAX_BATCH_ROWS
                || self.common.policy.prefill()?.is_some() {
                return Err(CandidateError::Arguments);
            }
        }
        if self.max_records == 0 || self.max_records > 1_000_000
            || self.max_total_input_bytes == 0 || self.max_total_input_bytes > 16 * 1024 * MIB
            || self.max_total_output_bytes < FOOTER_BYTES as u64
            || self.max_total_output_bytes > 16 * 1024 * MIB
            || self.max_total_forward_positions == 0 || self.max_total_projected_logits == 0 {
            return Err(CandidateError::Arguments);
        }
        // Shared tokenizer and whole-corpus ID set remain single charges;
        // all simultaneously retained document plans and staging scale by M.
        // Conservative declared payload model, not allocator/RSS certification.
        let rows = self.cohort_rows.unwrap_or(1) as u64;
        let inputs = (self.common.max_input_bytes as u64).checked_mul(32)
            .and_then(|bytes| bytes.checked_mul(rows)).ok_or(CandidateError::Arguments)?;
        let staging = (limits.result_bytes as u64).checked_add(8192)
            .and_then(|bytes| bytes.checked_mul(2)).and_then(|bytes| bytes.checked_mul(rows))
            .ok_or(CandidateError::Arguments)?;
        let floor = self.max_records.checked_mul(1024)
            .and_then(|n| n.checked_add(128 * MIB))
            .and_then(|n| n.checked_add(inputs))
            .and_then(|n| n.checked_add(staging))
            .and_then(|n| n.checked_add(65_536)).ok_or(CandidateError::Arguments)?;
        if floor > limits.preparation_bytes { return Err(CandidateError::Arguments); }
        Ok(limits)
    }
    pub(super) fn execute(self, input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
        let limits = self.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::text_batch::execute(self, limits, input, output).map_err(|_| CandidateError::Batch) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, limits, input, output); Err(CandidateError::Unavailable) }
    }
}

pub(super) enum Content { Generate(String), Chat(Vec<ChatMessage>) }
pub(super) struct Record { pub id: String, pub sample_index: u64, pub content: Content }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GenerateLine { id: String, #[serde(default)] sample_index: u64, prompt: String }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatLine { id: String, #[serde(default)] sample_index: u64, messages: Vec<ChatMessage> }

pub(super) fn parse_record(task: TextTask, line: &str, cap: usize) -> Result<Record, CandidateError> {
    if line.len() > cap { return Err(CandidateError::Input); }
    let value = canonjson::parse_str_with_limits(line, canonjson::ParseLimits {
        max_depth: 6, max_string_bytes: cap,
    }).map_err(|_| CandidateError::Input)?;
    let record = match task {
        TextTask::Generate => {
            let row: GenerateLine = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
            Record { id: row.id, sample_index: row.sample_index, content: Content::Generate(row.prompt) }
        }
        TextTask::Chat => {
            let row: ChatLine = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
            if row.messages.is_empty() || row.messages.len() > 128
                || row.messages.last().is_none_or(|m| m.role != ChatRole::User) {
                return Err(CandidateError::Input);
            }
            let mut expected = ChatRole::User;
            for (index, message) in row.messages.iter().enumerate() {
                if index == 0 && message.role == ChatRole::System { continue; }
                if message.role != expected { return Err(CandidateError::Input); }
                expected = if expected == ChatRole::User { ChatRole::Assistant } else { ChatRole::User };
            }
            Record { id: row.id, sample_index: row.sample_index, content: Content::Chat(row.messages) }
        }
    };
    if record.id.trim().is_empty() || record.id.len() > 256 || record.id.chars().any(char::is_control) {
        return Err(CandidateError::Input);
    }
    Ok(record)
}

/// Does not drain an oversized record. Caps include CR/LF framing; a final
/// nonempty record without LF is accepted, while a blank record is not skipped.
pub(super) fn read_line(input: &mut impl BufRead, cap: usize, total: &mut u64, total_cap: u64)
    -> Result<Option<String>, CandidateError> {
    let mut bytes = Vec::new();
    loop {
        let available = input.fill_buf().map_err(|_| CandidateError::Input)?;
        if available.is_empty() { break; }
        let end = available.iter().position(|&b| b == b'\n').map(|i| i + 1);
        let take = end.unwrap_or(available.len());
        let length = bytes.len().checked_add(take).filter(|&n| n <= cap).ok_or(CandidateError::Input)?;
        let next = total.checked_add(take as u64).filter(|&n| n <= total_cap).ok_or(CandidateError::Input)?;
        bytes.try_reserve_exact(length - bytes.len()).map_err(|_| CandidateError::Memory)?;
        bytes.extend_from_slice(&available[..take]);
        input.consume(take);
        *total = next;
        if end.is_some() { break; }
    }
    if bytes.is_empty() { return Ok(None); }
    String::from_utf8(bytes).map(Some).map_err(|_| CandidateError::Input)
}

#[derive(Clone, Copy, Default, Serialize)]
pub(super) struct ReservedWork { pub forward_positions: u64, pub projected_logits: u64 }
#[derive(Default)]
pub(super) struct Ledger { ids: BTreeSet<String>, pub records: u64, pub work: ReservedWork }
impl Ledger {
    pub(super) fn admit(&mut self, id: &str, work: ReservedWork, command: &TextBatchCommand) -> Result<(), CandidateError> {
        if self.records >= command.max_records || self.ids.contains(id) { return Err(CandidateError::Batch); }
        let forward_positions = self.work.forward_positions.checked_add(work.forward_positions)
            .filter(|&n| n <= command.max_total_forward_positions).ok_or(CandidateError::Batch)?;
        let projected_logits = self.work.projected_logits.checked_add(work.projected_logits)
            .filter(|&n| n <= command.max_total_projected_logits).ok_or(CandidateError::Batch)?;
        self.ids.insert(id.to_owned());
        self.records += 1;
        self.work = ReservedWork { forward_positions, projected_logits };
        Ok(())
    }
}

/// Counts actual transport bytes. A partial write or failed flush is fatal;
/// nothing retries a record, and no completion frame follows an error.
pub(super) struct Output<'a, W: Write> { inner: &'a mut W, cap: u64, pub written: u64 }
impl<'a, W: Write> Output<'a, W> {
    pub(super) fn new(inner: &'a mut W, cap: u64) -> Self { Self { inner, cap, written: 0 } }
    pub(super) fn admit_frame(&self, frame_cap: usize) -> Result<(), CandidateError> {
        let needed = (frame_cap as u64).checked_add(FOOTER_BYTES as u64).ok_or(CandidateError::Output)?;
        if needed > self.cap - self.written { return Err(CandidateError::Output); }
        Ok(())
    }
}
impl<W: Write> Write for Output<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() as u64 > self.cap - self.written { return Err(io::Error::other("text batch output budget")); }
        let count = self.inner.write(bytes)?;
        if count > bytes.len() { return Err(io::Error::other("invalid writer count")); }
        self.written += count as u64;
        Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> { self.inner.flush() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, BufReader};

    fn command(extra: &[&str]) -> TextBatchCommand {
        let mut argv = vec!["text-batch", "--model", "local.fnlpq", "--memory-mib", "8192"];
        argv.extend_from_slice(extra);
        TextBatchCommand::from_arg_matches(&definition().try_get_matches_from(argv).unwrap()).unwrap()
    }
    #[test]
    fn cli_has_shared_generation_controls_and_bounded_corpus_defaults() {
        let command = command(&["--task", "chat", "--stop", "END", "--min-new-tokens", "3"]);
        assert!(command.validate().is_ok());
        assert!(matches!(command.task, TextTask::Chat));
        assert_eq!(command.common.options(166_101).unwrap().min_new_tokens, 3);
        let matches = super::super::definition().try_get_matches_from(["candidate", "text-batch",
            "--model", "local.fnlpq", "--memory-mib", "8192"]).unwrap();
        assert!(matches!(CandidateCommand::from_matches(&matches).unwrap(), CandidateCommand::TextBatch(_)));
    }
    #[test]
    fn source_bytes_and_semantic_sample_addresses_are_preserved() {
        let row = parse_record(TextTask::Generate, r#"{"id":"doc-1","sample_index":7,"prompt":" é\n上海 "}"#, 4096).unwrap();
        assert_eq!(row.id, "doc-1"); assert_eq!(row.sample_index, 7);
        let Content::Generate(text) = row.content else { panic!("wrong task") };
        assert_eq!(text, " é\n上海 ");
    }
    #[test]
    fn chat_requires_a_complete_alternating_transcript() {
        assert!(parse_record(TextTask::Chat, r#"{"id":"c","messages":[{"role":"system","content":"s"},{"role":"user","content":"u"},{"role":"assistant","content":"a"},{"role":"user","content":"v"}]}"#, 4096).is_ok());
        for messages in ["[]", r#"[{"role":"assistant","content":"a"}]"#,
            r#"[{"role":"user","content":"u"},{"role":"user","content":"v"}]"#] {
            assert!(parse_record(TextTask::Chat, &format!(r#"{{"id":"c","messages":{messages}}}"#), 4096).is_err());
        }
    }
    #[test]
    fn ambiguous_records_and_policy_overrides_are_refused() {
        for row in [r#"{"id":"a","id":"b","prompt":"x"}"#, r#"{"id":" ","prompt":"x"}"#,
            r#"{"id":"a\n","prompt":"x"}"#, r#"{"id":"a","prompt":"x","seed":"1"}"#,
            r#"{"id":"a","prompt":"x","sample_index":-1}"#, r#"{"id":"a","messages":[]}"#] {
            assert!(parse_record(TextTask::Generate, row, 4096).is_err());
        }
    }
    #[test]
    fn record_and_total_byte_limits_include_line_endings() {
        let mut reader = BufReader::with_capacity(2, Cursor::new(b"a\r\nb\n")); let mut total = 0;
        assert_eq!(read_line(&mut reader, 3, &mut total, 4).unwrap().as_deref(), Some("a\r\n"));
        assert_eq!(total, 3);
        assert!(read_line(&mut reader, 3, &mut total, 4).is_err());
        let mut total = 0; let mut reader = Cursor::new(b"abcd\nmore\n");
        assert!(read_line(&mut reader, 4, &mut total, 100).is_err());
        assert_eq!(reader.position(), 0);
    }
    #[test]
    fn final_record_without_newline_and_empty_corpus_are_distinct_from_blank_rows() {
        let mut reader = Cursor::new("é"); let mut total = 0;
        assert_eq!(read_line(&mut reader, 2, &mut total, 2).unwrap().as_deref(), Some("é"));
        assert!(read_line(&mut reader, 2, &mut total, 2).unwrap().is_none());
        assert!(parse_record(TextTask::Generate, "\n", 10).is_err());
        let mut reader = Cursor::new([0xff]); let mut total = 0;
        assert!(read_line(&mut reader, 2, &mut total, 2).is_err());
    }
    #[test]
    fn duplicate_ids_and_whole_corpus_work_limits_do_not_reset_per_record() {
        let command = command(&["--max-total-forward-positions", "10", "--max-total-projected-logits", "20"]);
        let mut ledger = Ledger::default();
        assert!(ledger.admit("a", ReservedWork { forward_positions: 6, projected_logits: 10 }, &command).is_ok());
        assert!(ledger.admit("a", ReservedWork::default(), &command).is_err());
        assert!(ledger.admit("b", ReservedWork { forward_positions: 5, projected_logits: 1 }, &command).is_err());
        assert_eq!(ledger.records, 1); assert_eq!(ledger.work.forward_positions, 6);
        assert!(ledger.admit("b", ReservedWork { forward_positions: 4, projected_logits: 10 }, &command).is_ok());
        assert_eq!(ledger.records, 2);
    }
    #[test]
    fn record_limit_and_id_memory_are_admitted_before_io() {
        let command = command(&["--max-records", "1"]); let mut ledger = Ledger::default();
        ledger.admit("a", ReservedWork::default(), &command).unwrap();
        assert!(ledger.admit("b", ReservedWork::default(), &command).is_err());
        assert!(super::tests::command(&["--max-records", "1000000"]).validate().is_err());
        assert!(super::tests::command(&["--max-total-output-bytes", "1"]).validate().is_err());
    }
    #[test]
    fn output_budget_reserves_completion_and_counts_short_writes() {
        struct Short(Vec<u8>);
        impl Write for Short {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { let n = bytes.len().min(2); self.0.extend_from_slice(&bytes[..n]); Ok(n) }
            fn flush(&mut self) -> io::Result<()> { Ok(()) }
        }
        let mut inner = Short(Vec::new()); let mut output = Output::new(&mut inner, FOOTER_BYTES as u64 + 4);
        assert!(output.admit_frame(4).is_ok());
        output.write_all(b"data").unwrap(); assert_eq!(output.written, 4);
        assert!(output.admit_frame(1).is_err());
        output.write_all(b"{}\n").unwrap(); assert_eq!(output.written, 7);
    }
}
#[cfg(test)] mod cohort_tests;
