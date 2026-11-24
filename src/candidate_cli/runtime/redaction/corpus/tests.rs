//! Admission and pinned-policy regression definitions, not model execution.
use super::*;
use std::io;
use crate::candidate_cli::redact::tests::command;

fn compact_command() -> RedactCommand {
    let mut command = command(&["--ndjson", "--chunked", "--input-encoding", "safe-bpe-v1",
        "--max-ner-chunk-bytes", "128", "--max-new-tokens", "64"]);
    command.host.memory_mib = 16_384;
    command
}
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".into(), revision: "f56ec5a9650268aa098496734743c25ea778bd2d".into(),
        recipe_id: "redaction-population-planning-fixture".into(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
struct NoIo;
impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private input was read") } }
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output was written") }
    fn flush(&mut self) -> io::Result<()> { panic!("output was flushed") }
}

#[test]
fn stale_population_admission_refuses_before_host_creation_or_any_private_io() {
    for axis in 0..10 {
        let mut command = compact_command();
        let (common, mut limits) = command.validate().unwrap();
        let mut envelope = command.corpus.envelope(&command, limits).unwrap();
        match axis {
            0 => limits.preparation_bytes -= 1,
            1 => envelope.transport.max_work.forward_positions -= 1,
            2 => envelope.max_model_work.projections.multiply_accumulates -= 1,
            3 => envelope.max_mask_visits -= 1,
            4 => envelope.output_bytes -= 1,
            5 => envelope.io_bytes -= 1,
            6 => envelope.transport.max_document_bytes += 1,
            7 => envelope.transport.max_input_bytes += 1,
            8 => command.long.chunked = false,
            _ => command.corpus.ndjson = false,
        }
        assert_eq!(execute(command, common, limits, envelope, NoIo, NoIo), Err(CandidateError::Arguments), "{axis}");
    }
}

#[test]
fn compact_factory_binds_identity_without_expanding_original_input_or_native_corpus_authority() {
    let compact = compact_command();
    let mut byte = command(&["--ndjson", "--chunked", "--max-ner-chunk-bytes", "128", "--max-new-tokens", "64"]);
    byte.host.memory_mib = 16_384;
    let (_, compact_limits) = compact.validate().unwrap();
    let (_, byte_limits) = byte.validate().unwrap();
    let compact_envelope = compact.corpus.envelope(&compact, compact_limits).unwrap();
    let byte_envelope = byte.corpus.envelope(&byte, byte_limits).unwrap();
    assert_eq!(compact_envelope.transport, byte_envelope.transport);
    assert_eq!(compact_envelope.max_model_work, byte_envelope.max_model_work);
    assert_eq!(compact_envelope.max_mask_visits, byte_envelope.max_mask_visits);
    let (compact_factory, _) = planner().unwrap();
    let compact_factory = configure_planner(&compact, compact_limits, compact_factory).unwrap();
    let (byte_factory, _) = planner().unwrap();
    let byte_factory = configure_planner(&byte, byte_limits, byte_factory).unwrap();
    let compact_identity = source_identity(&facts(), &compact_factory, BuiltInTask::Ner).unwrap();
    let byte_identity = source_identity(&facts(), &byte_factory, BuiltInTask::Ner).unwrap();
    assert_ne!(compact_identity.template_digest, byte_identity.template_digest);
    assert_eq!(compact_identity.tokenizer_digest, byte_identity.tokenizer_digest);
    let PreparedCorpus::Document(compact_batch) = prepare(&compact, compact_limits, compact_envelope,
        &compact_factory, compact_identity.clone(), NerOptions::default(), compact.request()).unwrap()
        else { panic!("complete-document compact mode"); };
    let PreparedCorpus::Document(byte_batch) = prepare(&byte, byte_limits, byte_envelope,
        &byte_factory, byte_identity, NerOptions::default(), byte.request()).unwrap()
        else { panic!("complete-document byte mode"); };
    assert_eq!(compact_batch.ner_identity, compact_identity);
    assert_eq!(compact_batch.item_model_work(), byte_batch.item_model_work());
    assert_eq!(compact_batch.detector.mapping.chunks, byte_batch.detector.mapping.chunks);
    assert_eq!(compact_batch.request.rule_budget.max_input_bytes,
        compact.host.max_input_bytes.max(compact.host.max_result_bytes));
    assert_eq!(compact_batch.detector.mapping.chunks.max_input_bytes,
        compact.host.max_input_bytes.max(compact.host.max_result_bytes));
    assert_eq!(compact_envelope.transport.max_document_bytes, compact.host.max_input_bytes);
    assert_eq!(compact_factory.map_document_encoding_work().unwrap(), Some(SafeBpeWork::default()),
        "configuration is not a document attempt");
    assert_eq!(compact_factory.map_document_encoding_limits(), compact.encoding_policy(compact_limits).unwrap().1);
}

fn provenance(input_encoding: Option<PopulationEncoding>) -> Provenance<'static> {
    Provenance { protocol: "fnlp-candidate-batch-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative", model_id: "fixture-model",
        source_revision: "fixture-revision", source_root_sha256: "fixture-root",
        logical_model_sha256: "fixture-logical", quant_recipe: "fixture-recipe",
        task: "redact", redaction_mode: "chunked", input_encoding }
}

#[test]
fn population_metadata_preserves_byte_wire_and_never_reports_factory_work_as_document_work() {
    let (byte, _) = planner().unwrap();
    let byte = provenance(population_encoding(&byte).unwrap());
    let expected = br#"{"protocol":"fnlp-candidate-batch-v1","schema_version":1,"scope":"real-artifact-current-candidate","evidence":"non_authoritative","model_id":"fixture-model","source_revision":"fixture-revision","source_root_sha256":"fixture-root","logical_model_sha256":"fixture-logical","quant_recipe":"fixture-recipe","task":"redact","redaction_mode":"chunked"}"#;
    assert_eq!(serde_json::to_vec(&byte).unwrap().as_slice(), expected);
    let command = compact_command(); let (_, limits) = command.validate().unwrap();
    let (factory, _) = planner().unwrap();
    let factory = configure_planner(&command, limits, factory).unwrap();
    let before = serde_json::to_vec(&provenance(population_encoding(&factory).unwrap())).unwrap();
    let operation = factory.fork_map_document_operation().unwrap();
    operation.source_encoder().encode_with_control("private Alice 上海 <think>", 128, 128, &mut || None).unwrap();
    assert!(operation.map_document_encoding_work().unwrap().unwrap().input_bytes() > 0);
    assert_eq!(factory.map_document_encoding_work().unwrap(), Some(SafeBpeWork::default()));
    let after = serde_json::to_vec(&provenance(population_encoding(&factory).unwrap())).unwrap();
    assert_eq!(before, after);
    let value: serde_json::Value = serde_json::from_slice(&after).unwrap();
    assert_eq!(value["input_encoding"]["policy"], serde_json::to_value(factory.document_encoding()).unwrap());
    assert_eq!(value["input_encoding"]["complete_document_limits"],
        serde_json::to_value(factory.map_document_encoding_limits().unwrap()).unwrap());
    assert_eq!(value["input_encoding"]["budget_scope"],
        "original_preflight_execution_and_edited_verification_per_document_attempt");
    assert!(value["input_encoding"].get("work").is_none());
    let text = String::from_utf8(after).unwrap();
    for private in ["private Alice", "上海", "<think>", "prompt_digest", "source_digest", "key_commitment", "namespace"] {
        assert!(!text.contains(private));
    }
}

#[test]
fn population_source_and_batch_cancellation_preserve_timeout_classification() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = match cause { Deadline | Timeout => CandidateError::Timeout, _ => CandidateError::Batch };
        assert_eq!(corpus_execution_error(HostedError::BatchSetup(crate::batch::BatchFault::cancelled(cause))), expected);
        assert_eq!(corpus_execution_error(HostedError::SourceMap(
            crate::tasks::source_planning::quantized::long::Int8SourceMapError::Source(Int8SourceError::Cancelled(cause)))), expected);
    }
}
