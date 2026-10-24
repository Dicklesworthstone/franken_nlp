//! Owned-stream parsing/resource fixtures; no synthetic neural success.
use super::*;
use crate::tasks::redact::batch::RedactionBatchArgs;

fn parsed(flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", "redact-corpus", "--memory-mib", "8192"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("wrong INT4 corpus dispatcher"); };
    Ok(command)
}
fn command(extra: &[&str]) -> Int4Command {
    let mut flags = vec!["--source-dir", "explicit-source", "--source-manifest", "explicit-manifest"];
    flags.extend_from_slice(extra); parsed(&flags).unwrap()
}
pub(super) fn arguments(extra: &[&str]) -> (CorpusArgs, Int4Args) {
    let command = command(extra);
    let Int4Task::RedactCorpus(options) = command.task else { panic!("redaction corpus required"); };
    (options, command.args)
}
struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refusal consumed corpus or secret"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refusal published corpus output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refusal flushed corpus output"); }
}

#[test]
fn corpus_requires_owned_io_and_supports_the_same_two_actual_model_routes() {
    assert_eq!(command(&[]).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    let (options, args) = arguments(&[]);
    let source = options.validate(&args).unwrap();
    assert_eq!(source.transport.max_requests, 1000);
    assert_eq!(source.transport.max_document_bytes, 1_048_576);
    assert_eq!(source.transport.max_line_bytes, 4_194_304);
    assert_eq!(format!("{:?}", options), "RedactionCorpusArgs");
    let pin = "ab".repeat(32);
    let reload = parsed(&["--snapshot", "s", "--expected-model-sha256", &pin]).unwrap();
    let Int4Task::RedactCorpus(selected) = reload.task else { panic!("corpus task required"); };
    let snapshot = selected.validate(&reload.args).unwrap();
    assert_eq!(source.transport, snapshot.transport);
    assert_eq!(source.model_work, snapshot.model_work); assert_eq!(source.masks, snapshot.masks);
    assert_eq!(snapshot.common.snapshot_pin.unwrap().model_content_sha256, [0xab; 32]);
}

#[test]
fn fixed_document_policy_survives_the_stream_envelope_and_records_cannot_override_it() {
    let (options, args) = arguments(&["--rules", "email,date", "--action", "placeholder",
        "--include-map", "--no-verify", "--ner-options", "types.json", "--max-input-bytes", "32"]);
    let limits = options.validate(&args).unwrap();
    let request = options.document.request(&args, limits.document);
    assert_eq!(request.actions.default_action, RedactionAction::Placeholder);
    assert!(request.actions.include_map); assert!(!request.verify);
    assert_eq!(limits.transport.max_document_bytes, 32);
    assert_eq!(limits.document.mapping.chunks.max_input_bytes, limits.common.result_bytes);
    assert_eq!(request.rule_budget.max_input_bytes, limits.common.result_bytes);
    assert_eq!(options.document.redaction.source.options_path().unwrap(), std::path::Path::new("types.json"));
    assert!(serde_json::from_str::<RedactionBatchArgs>("{}").is_ok());
    for text in [r#"{"verify":false}"#, r#"{"action":"mask"}"#, r#"{"namespace":"changed"}"#,
        r#"{"key":"private"}"#, r#"{"types":["person"]}"#, r#"{"max_chunks":1}"#] {
        assert!(serde_json::from_str::<RedactionBatchArgs>(text).is_err());
    }
}

#[test]
fn all_five_work_axes_and_masks_are_scaled_from_one_complete_item_and_can_only_tighten() {
    let (options, args) = arguments(&["--max-requests", "3"]);
    let limits = options.validate(&args).unwrap();
    let mut item = limits.document.mapping.max_model_work;
    item.projected_logits -= item.projected_logits % NANBEIGE_VOCAB_SIZE as u64;
    assert_eq!(limits.model_work.forward_positions, item.forward_positions * 3);
    assert_eq!(limits.model_work.projected_logits, item.projected_logits * 3);
    assert_eq!(limits.model_work.attention_pairs, item.attention_pairs * 3);
    assert_eq!(limits.model_work.projections.dot_products, item.projections.dot_products * 3);
    assert_eq!(limits.model_work.projections.multiply_accumulates, item.projections.multiply_accumulates * 3);
    assert_eq!(limits.masks, limits.document.mapping.max_mask_visits * 3);
    assert_eq!(limits.transport.max_work.forward_positions, limits.model_work.forward_positions);
    assert_eq!(limits.transport.max_work.projected_logits, limits.model_work.projected_logits);
    let mut tightened = options.clone();
    tightened.stream.max_corpus_forward_positions = Some(item.forward_positions * 2);
    tightened.stream.max_corpus_projected_logits = Some(item.projected_logits * 2);
    tightened.stream.max_corpus_attention_pairs = Some(item.attention_pairs * 2);
    tightened.stream.max_corpus_dot_products = Some(item.projections.dot_products * 2);
    tightened.stream.max_corpus_multiply_accumulates = Some(item.projections.multiply_accumulates * 2);
    tightened.stream.max_corpus_mask_node_visits = Some(limits.document.mapping.max_mask_visits * 2);
    let narrow = tightened.validate(&args).unwrap();
    assert_eq!(narrow.model_work.forward_positions * 3, limits.model_work.forward_positions * 2);
    assert_eq!(narrow.model_work.projected_logits * 3, limits.model_work.projected_logits * 2);
    assert_eq!(narrow.model_work.attention_pairs * 3, limits.model_work.attention_pairs * 2);
    assert_eq!(narrow.model_work.projections.dot_products * 3, limits.model_work.projections.dot_products * 2);
    assert_eq!(narrow.model_work.projections.multiply_accumulates * 3, limits.model_work.projections.multiply_accumulates * 2);
    assert_eq!(narrow.masks * 3, limits.masks * 2);
    assert_eq!(narrow.common.native, limits.common.native);
    assert_eq!(narrow.common.minimum_memory_bytes, limits.common.minimum_memory_bytes);
}

#[test]
fn explicit_finite_caps_bound_overflowing_populations_without_creating_implicit_saturation() {
    assert_eq!(scaled_ceiling(10, 3, None), Ok(30));
    assert_eq!(scaled_ceiling(10, 3, Some(20)), Ok(20));
    assert_eq!(scaled_ceiling(10, 3, Some(100)), Ok(30));
    assert_eq!(scaled_ceiling(10, 3, Some(9)), Err(CandidateError::Arguments));
    assert_eq!(scaled_ceiling(u64::MAX, 2, None), Err(CandidateError::Arguments));
    assert_eq!(scaled_ceiling(u64::MAX, 2, Some(u64::MAX)), Ok(u64::MAX));
    assert_eq!(scaled_ceiling(u64::MAX, 2, Some(u64::MAX - 1)), Err(CandidateError::Arguments));
    assert_eq!(scaled_ceiling(0, 2, None), Err(CandidateError::Arguments));
    assert_eq!(scaled_ceiling(1, 0, None), Err(CandidateError::Arguments));
}

#[test]
fn corpus_quote_replaces_standalone_capture_with_owned_transport_without_renewing_preparation() {
    let (options, mut args) = arguments(&["--max-requests", "8"]);
    let document = options.document.validate(&args).unwrap();
    let limits = options.validate(&args).unwrap();
    let t = limits.transport;
    let corpus_buffers = t.max_line_bytes as u64 * 8 + t.max_output_line_bytes as u64 * 4
        + t.max_epoch_ids as u64 * 128 + t.max_epoch_id_bytes as u64
        + limits.common.preparation_bytes + limits.io_bytes;
    assert_eq!(limits.common.minimum_memory_bytes,
        document.common.minimum_memory_bytes - args.max_input_bytes as u64 - 1
            - document.common.preparation_bytes + corpus_buffers);
    assert_eq!(t.max_output_bytes + (8 + 4) * FRAME_ALLOWANCE, limits.output_bytes);
    assert_eq!(limits.document.workspace_bytes, document.workspace_bytes);
    assert_eq!(limits.common.native.kv_bytes, document.common.native.kv_bytes);
    args.memory_mib = limits.common.minimum_memory_bytes.saturating_sub(1) / MIB;
    assert!(options.validate(&args).is_err());
    args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
    assert!(options.validate(&args).is_ok());
}

#[test]
fn invalid_stream_limits_and_private_key_scope_refuse_before_owned_io() {
    for flags in [vec!["--max-requests", "0"], vec!["--max-requests", "100001"],
        vec!["--max-total-input-bytes", "0"], vec!["--max-total-output-bytes", "1"],
        vec!["--max-line-bytes", "1048575"], vec!["--max-line-bytes", "4194305"],
        vec!["--max-corpus-forward-positions", "1"], vec!["--max-corpus-projected-logits", "1"],
        vec!["--max-corpus-attention-pairs", "1"], vec!["--max-corpus-dot-products", "1"],
        vec!["--max-corpus-multiply-accumulates", "1"], vec!["--max-corpus-mask-node-visits", "1"],
        vec!["--key-stdin"], vec!["--action", "pseudonymize", "--key-stdin", "--key-id", "k", "--namespace", "n"],
        vec!["--preparation-mib", "1"], vec!["--max-ner-chunks", "0"]] {
        let (options, args) = arguments(&flags);
        assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    }
    for flag in ["--store-results", "--resume", "--job-dir", "--key", "--key-file",
        "--rules-only", "--ndjson", "--chunked", "--seed", "--logprobs", "--prefill-rows"] {
        assert!(parsed(&["--source-dir", "d", "--source-manifest", "m", flag, "private"]).is_err());
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_runtime_does_not_consume_corpus_or_binary_key() {
    for flags in [vec![], vec!["corpus.ndjson", "--action", "pseudonymize",
        "--key-stdin", "--key-id", "k", "--namespace", "n"]] {
        let (options, args) = arguments(&flags);
        assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Unavailable));
    }
}

#[cfg(feature = "asupersync-runtime")]
#[test]
#[ignore = "requires controller-admitted pinned sources, a fresh snapshot and an available explicit SIMD kernel"]
fn source_scalar_and_pinned_snapshot_simd_match_across_document_and_flush_boundaries() {
    use std::{io::Cursor, sync::{Arc, Mutex}};
    struct Captured(Arc<Mutex<Vec<u8>>>);
    impl Write for Captured {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.0.lock().unwrap().extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    fn collect(options: CorpusArgs, args: Int4Args, bytes: Vec<u8>) -> Vec<serde_json::Value> {
        let output = Arc::new(Mutex::new(Vec::new()));
        options.execute_owned(args, Cursor::new(bytes), Captured(output.clone())).unwrap();
        assert_eq!(Arc::strong_count(&output), 1, "host must release owned writer before returning");
        let text = String::from_utf8(output.lock().unwrap().clone()).unwrap();
        text.lines().map(|line| serde_json::from_str(line).unwrap()).collect()
    }
    let source = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit pinned sources required");
    let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit source manifest required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required").parse().unwrap();
    let snapshot = std::env::var("FNLP_INT4_TEST_REDACT_CORPUS_SNAPSHOT").expect("explicit fresh snapshot required");
    assert!(!std::path::Path::new(&snapshot).exists(), "fixture deletes nothing");
    let kernel: Int8KernelKind = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit SIMD kernel required").parse().unwrap();
    assert_ne!(kernel, Int8KernelKind::Scalar); PackedInt4Kernel::detect(kernel).unwrap();
    let input = "{\"id\":\"a\",\"text\":\"é Contact a@example.org near 上海.\"}\n{\"flush\":true}\n{\"id\":\"b\",\"text\":\"Another b@example.org appears.\"}\n".as_bytes().to_vec();
    let (options, mut args) = arguments(&["--rules", "email", "--max-requests", "4",
        "--max-ner-chunk-bytes", "16", "--save-snapshot", &snapshot]);
    args.source_dir = Some(PathBuf::from(source)); args.source_manifest = Some(PathBuf::from(manifest)); args.memory_mib = memory;
    let scalar = collect(options, args, input.clone());
    let results: Vec<_> = scalar.iter().filter(|row| row["record"].get("result").is_some()).collect();
    assert_eq!(results.len(), 2); assert_eq!(results[0]["record"]["caller_id"], "a"); assert_eq!(results[1]["record"]["caller_id"], "b");
    for row in results {
        assert_eq!(row["record"]["result"]["result"]["verification"], "clean_declared_union");
        assert!(!row["record"]["result"]["result"]["text"].as_str().unwrap().contains("@example.org"));
    }
    assert_eq!(scalar.last().unwrap()["record"]["event"], "run_complete");
    assert_eq!(scalar.last().unwrap()["record"]["summary"]["failed"], 0);
    let pin = scalar[0]["model_content_sha256"].as_str().unwrap();
    let reload = parsed(&["--rules", "email", "--max-requests", "4", "--max-ner-chunk-bytes", "16",
        "--snapshot", &snapshot, "--expected-model-sha256", pin, "--kernel", kernel.id()]).unwrap();
    let Int4Task::RedactCorpus(options) = reload.task else { panic!("corpus task required"); };
    let mut args = reload.args; args.memory_mib = memory;
    let selected = collect(options, args, input);
    assert_eq!(selected.iter().map(|row| &row["record"]).collect::<Vec<_>>(),
        scalar.iter().map(|row| &row["record"]).collect::<Vec<_>>());
    assert!(selected.iter().all(|row| row["integer_kernel"] == kernel.id()));
}
