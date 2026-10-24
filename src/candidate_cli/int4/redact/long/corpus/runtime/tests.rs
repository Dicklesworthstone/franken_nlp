//! Publication and admission fixtures; none fabricates successful neural inference.
use super::*;
use super::super::tests::arguments;
use std::io::Cursor;
use crate::{batch::{BatchFault, BatchRunError, BatchSummary},
    tasks::redact::{grouped_int4::Int4RedactionError, redact_rules}};

struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refusal consumed private input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refusal published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refusal flushed output"); }
}
fn provenance() -> Provenance<'static> {
    Provenance { protocol: "fnlp-candidate-int4-redact-corpus-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "redact",
        redaction_mode: "complete_document", model_id: "Nanbeige4.2-3B", source_revision: "fixture",
        source_root_sha256: "fixture-source", model_content_sha256: "fixture-model",
        candidate_format: "fixture-format", quant_recipe: "fixture:group=128", group_width: 128,
        integer_kernel: "s_scalar_i32", snapshot: Some(SnapshotRecord::new("loaded",
            Int4SnapshotRequirements::for_group_width(128).unwrap())) }
}

#[test]
fn stale_stream_admission_cannot_bypass_revalidation_before_any_owned_io() {
    let (mut command, args) = arguments(&[]);
    let quote = command.validate(&args).unwrap();
    command.stream.max_total_output_bytes = 1;
    assert_eq!(execute(args, command, quote, NoIo, NoIo), Err(CandidateError::Arguments));
    let (command, mut args) = arguments(&[]);
    let quote = command.validate(&args).unwrap(); args.seed = Some("ab".repeat(32));
    assert_eq!(execute(args, command, quote, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn one_full256_key_scope_checks_commitment_before_corpus_io_and_never_renews_per_row() {
    let (command, args) = arguments(&["missing-corpus.ndjson", "--action", "pseudonymize",
        "--key-stdin", "--key-id", "rotation", "--namespace", "private-scope"]);
    let limits = command.validate(&args).unwrap();
    let request = command.document.request(&args, limits.document);
    let secret = key_scope(&command, &mut Cursor::new([0x80_u8; 32]), &request).unwrap().unwrap();
    let expected = secret.key.commitment();
    let context = Pseudonyms::full256(&secret.key, &secret.namespace, Some(&expected)).unwrap();
    assert_eq!(context.identity().encoding, crate::tasks::redact::pseudonym::PseudonymEncoding::Full256);
    let first = context.pseudonym(crate::tasks::redact::PiiKind::Email, "a@example.org").unwrap();
    assert_eq!(first, context.pseudonym(crate::tasks::redact::PiiKind::Email, "a@example.org").unwrap());
    let (pinned, args) = arguments(&["missing-corpus.ndjson", "--action", "pseudonymize",
        "--key-stdin", "--key-id", "rotation", "--namespace", "private-scope", "--expected-key-commitment", &expected]);
    let request = pinned.document.request(&args, pinned.validate(&args).unwrap().document);
    assert!(matches!(key_scope(&pinned, &mut Cursor::new([1_u8; 32]), &request), Err(CandidateError::Identity)));
    assert!(key_scope(&pinned, &mut Cursor::new([0x80_u8; 32]), &request).is_ok());
    let (plain, args) = arguments(&[]);
    let request = plain.document.request(&args, plain.validate(&args).unwrap().document);
    assert!(key_scope(&plain, &mut NoIo, &request).unwrap().is_none());
}

#[test]
fn corpus_errors_preserve_deadlines_and_failed_summaries_never_report_success() {
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout] {
        assert_eq!(execution_error(HostedError::Batch(BatchRunError {
            fault: BatchFault::cancelled(cause), summary: BatchSummary::default() })), CandidateError::Timeout);
        assert_eq!(execution_error(HostedError::BatchSetup(BatchFault::cancelled(cause))), CandidateError::Timeout);
        assert_eq!(execution_error(HostedError::Int4(Int4HostError::Redaction(Int4RedactionError::Cancelled(cause)))),
            CandidateError::Timeout);
    }
    assert_eq!(execution_error(HostedError::BatchSetup(BatchFault::cancelled(DecodeCancellationKind::PollQuota))),
        CandidateError::Batch);
    assert_eq!(execution_error(HostedError::ModelIdentity), CandidateError::Identity);
    assert_eq!(execution_error(HostedError::ResourceDomain), CandidateError::Identity);
    assert_eq!(completed(BatchSummary { succeeded: 1, failed: 1, ..BatchSummary::default() }), Err(CandidateError::Batch));
    assert_eq!(completed(BatchSummary { succeeded: 2, ..BatchSummary::default() }), Ok(()));
}

#[test]
fn complete_event_wrapping_preserves_exact_numbers_maps_and_private_provenance_boundaries() {
    let (command, args) = arguments(&["--rules", "email", "--action", "placeholder", "--include-map"]);
    let limits = command.validate(&args).unwrap();
    let request = command.document.request(&args, limits.document);
    let result = redact_rules("é a@example.org 上海", &request, None).unwrap();
    let event = serde_json::json!({"event":"rules_only_fixture","result":result});
    let mut inner = crate::canonjson::canonical_bytes(&event).unwrap(); inner.push(b'\n');
    let mut output = Vec::new();
    {
        let mut writer = CandidateWriter::new(&mut output, &provenance(), 8192, 32768).unwrap();
        writer.write_all(&inner).unwrap(); writer.flush().unwrap();
        writer.write_all(b"{\"event\":\"integer_fixture\",\"exact\":12345678901234567890123456789012345678}\n").unwrap();
        writer.flush().unwrap();
    }
    let text = std::str::from_utf8(&output).unwrap();
    let first: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    assert_eq!(first["protocol"], "fnlp-candidate-int4-redact-corpus-v1");
    assert_eq!(first["record"]["result"]["text"], "é [redacted:email] 上海");
    assert_eq!(first["record"]["result"]["edits"][0]["original"]["span"]["byte_start"], 3);
    assert_eq!(first["snapshot"]["operation"], "loaded");
    assert!(text.contains("\"exact\":12345678901234567890123456789012345678"));
    assert!(!text.contains("a@example.org"));
    for absent in ["source", "document", "prompt", "prompt_digest", "namespace", "key",
        "source_dir", "source_manifest", "snapshot_path"] { assert!(first.get(absent).is_none()); }
}

#[test]
fn partial_or_oversized_inner_events_never_publish_and_drop_never_flushes_them() {
    let mut output = Vec::new();
    {
        let mut writer = CandidateWriter::new(&mut output, &provenance(), 4096, 8192).unwrap();
        writer.write_all(b"{\"event\":\"unfinished").unwrap();
    }
    assert!(output.is_empty());
    {
        let mut writer = CandidateWriter::new(&mut output, &provenance(), 16, 8192).unwrap();
        assert!(writer.write_all(b"{\"event\":\"oversized_complete_fixture\"}\n").is_err());
        assert!(writer.flush().is_err());
    }
    assert!(output.is_empty());
}

#[test]
fn failed_writer_flush_poisoning_prevents_a_second_terminal_or_retry() {
    struct FlushFailure(Vec<u8>);
    impl Write for FlushFailure {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.0.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("fixture flush failure")) }
    }
    let mut output = FlushFailure(Vec::new());
    {
        let mut writer = CandidateWriter::new(&mut output, &provenance(), 4096, 8192).unwrap();
        writer.write_all(b"{\"event\":\"fixture\"}\n").unwrap();
        assert!(writer.flush().is_err());
        assert!(writer.write_all(b"{\"event\":\"run_complete\"}\n").is_err());
        assert!(writer.flush().is_err());
    }
    assert_eq!(output.0.iter().filter(|&&b| b == b'\n').count(), 1);
    assert!(!std::str::from_utf8(&output.0).unwrap().contains("run_complete"));
}
