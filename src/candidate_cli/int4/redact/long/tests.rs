//! CLI admission fixtures; source/snapshot execution remains an explicit gate.
use super::*;
use std::io::Cursor;

fn parsed(flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", "redact-document", "--memory-mib", "8192"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("wrong INT4 document redaction dispatcher"); };
    Ok(command)
}
pub(super) fn command(extra: &[&str]) -> Int4Command {
    let mut flags = vec!["--source-dir", "explicit-source", "--source-manifest", "explicit-manifest"];
    flags.extend_from_slice(extra); parsed(&flags).unwrap()
}
pub(super) fn options(command: &Int4Command) -> &DocumentArgs {
    let Int4Task::RedactDocument(options) = &command.task else { panic!("document redaction required"); };
    options
}
struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refusal consumed private input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refusal published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refusal flushed output"); }
}

#[test]
fn default_whole_document_route_preserves_shared_policy_and_both_model_inputs() {
    let command = command(&[]); let selected = options(&command);
    let limits = selected.validate(&command.args).unwrap();
    let request = selected.request(&command.args, limits);
    assert_eq!(command.args.max_input_bytes, 1_048_576);
    assert_eq!(command.args.preparation_mib, 1024);
    assert_eq!(limits.mapping.chunks.max_chunks, 64);
    assert_eq!(limits.mapping.chunks.max_chunk_bytes, command.args.context_tokens);
    assert_eq!(limits.mapping.chunks.reserved_tokens, command.args.max_new_tokens);
    assert_eq!(limits.mapping.reduction.max_result_bytes, 16 * 1024 * 1024);
    assert_eq!(request.actions.default_action, RedactionAction::Mask);
    assert!(request.verify); assert!(!request.actions.include_map);
    assert_eq!(request.grounding_budget, limits.mapping.verification);
    assert_eq!(format!("{:?}", selected), "DocumentRedactArgs");
    let pin = "ab".repeat(32);
    let reload = parsed(&["--snapshot", "explicit.snapshot", "--expected-model-sha256", &pin]).unwrap();
    let reloaded = options(&reload).validate(&reload.args).unwrap();
    assert_eq!(reloaded.common.snapshot_pin.unwrap().model_content_sha256, [0xab; 32]);
    assert_eq!(reloaded.common.native.kv_bytes, limits.common.native.kv_bytes);
    for invalid in [vec![], vec!["--snapshot", "s"], vec!["--model", "int8.fnlpq"],
        vec!["--snapshot", "s", "--expected-model-sha256", &pin, "--save-snapshot", "s2"]] {
        assert!(parsed(&invalid).is_err());
    }
}

#[test]
fn all_native_axes_masks_and_independent_verification_pools_are_explicit() {
    let command = command(&["--max-ner-chunks", "8", "--max-ner-chunk-bytes", "128",
        "--max-ner-tokenizer-calls", "256", "--max-ner-map-bytes", "2097152",
        "--max-total-mask-node-visits", "8000000000", "--max-forward-positions", "10000",
        "--max-projected-logits", "2000000", "--max-attention-pairs", "300000000",
        "--max-dot-products", "4000000000000", "--max-multiply-accumulates", "5000000000000",
        "--max-source-fields", "123", "--max-source-evidence-spans", "456", "--max-source-scan-steps", "789",
        "--no-verify", "--include-map", "--action", "placeholder"]);
    let selected = options(&command); let limits = selected.validate(&command.args).unwrap();
    let work = limits.mapping.max_model_work;
    assert_eq!((work.forward_positions, work.projected_logits, work.attention_pairs), (10000, 2000000, 300000000));
    assert_eq!((work.projections.dot_products, work.projections.multiply_accumulates), (4000000000000, 5000000000000));
    assert_eq!(limits.mapping.max_mask_visits, 8000000000);
    assert_eq!(limits.mapping.chunks.max_chunks, 8);
    assert_eq!(limits.mapping.chunks.max_chunk_bytes, 128);
    assert_eq!(limits.mapping.chunks.max_tokenizer_calls, 256);
    assert_eq!(limits.mapping.reduction.max_live_value_bytes, 4 * 1024 * 1024);
    assert_eq!(limits.mapping.reduction.max_total_value_bytes, 8 * 1024 * 1024);
    let request = selected.request(&command.args, limits);
    assert_eq!(request.grounding_budget, GroundingBudget { max_fields: 123, max_matches: 456, max_scan_steps: 789 });
    assert_eq!(request.grounding_budget, limits.mapping.verification);
    assert!(!request.verify); assert!(request.actions.include_map);
    assert_eq!(request.actions.default_action, RedactionAction::Placeholder);
}

#[test]
fn invalid_document_bounds_and_generation_or_corpus_controls_refuse_before_io() {
    for flags in [vec!["--max-ner-chunks", "0"], vec!["--max-ner-chunks", "257"],
        vec!["--max-ner-chunk-bytes", "3"], vec!["--max-ner-chunk-bytes", "1048577"],
        vec!["--max-ner-tokenizer-calls", "0"], vec!["--max-ner-tokenizer-calls", "1000001"],
        vec!["--max-ner-map-bytes", "1"], vec!["--max-ner-map-bytes", "16777217"],
        vec!["--max-total-mask-node-visits", "1"], vec!["--max-total-mask-node-visits", "1000000000000001"],
        vec!["--max-forward-positions", "0"], vec!["--max-projected-logits", "166143"],
        vec!["--max-attention-pairs", "0"], vec!["--max-dot-products", "0"], vec!["--max-multiply-accumulates", "0"],
        vec!["--max-source-fields", "0"], vec!["--max-source-evidence-spans", "1000001"],
        vec!["--max-source-scan-steps", "0"], vec!["--max-input-bytes", "3"],
        vec!["--edit-reserve-mib", "1"], vec!["--context-tokens", "512"]] {
        assert_eq!(command(&flags).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
    for flag in ["--ndjson", "--store-results", "--resume", "--job-dir", "--chunked",
        "--key", "--key-file", "--seed", "--logprobs", "--prefill-rows"] {
        assert!(parsed(&["--source-dir", "d", "--source-manifest", "m", flag, "private"]).is_err());
    }
}

#[test]
fn document_key_contract_is_unchanged_and_refuses_stdin_conflicts_before_io() {
    for flags in [vec!["--key-stdin"], vec!["--action", "pseudonymize"],
        vec!["--action", "pseudonymize", "--key-stdin", "--key-id", "k", "--namespace", "n"],
        vec!["document", "--action", "pseudonymize", "--key-stdin", "--key-id", "bad id", "--namespace", "n"]] {
        assert_eq!(command(&flags).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
    let selected = command(&["document", "--action", "pseudonymize", "--key-stdin", "--key-id", "k", "--namespace", "n"]);
    let (key, namespace) = options(&selected).redaction.key(&mut Cursor::new([0x80_u8; 32])).unwrap().unwrap();
    let short = super::super::tests::command(&["document", "--action", "pseudonymize",
        "--key-stdin", "--key-id", "k", "--namespace", "n"]);
    let (same, same_namespace) = super::super::tests::options(&short).key(&mut Cursor::new([0x80_u8; 32])).unwrap().unwrap();
    assert_eq!(key.commitment(), same.commitment()); assert_eq!(namespace, same_namespace);
    let commitment = key.commitment();
    let pinned = command(&["missing-document", "--action", "pseudonymize", "--key-stdin",
        "--key-id", "k", "--namespace", "n", "--expected-key-commitment", &commitment]);
    assert!(matches!(options(&pinned).redaction.key(&mut Cursor::new([1_u8; 32])), Err(CandidateError::Identity)));
}

#[test]
fn original_input_bound_stays_separate_from_expanded_transformed_text_envelope() {
    let command = command(&["--max-input-bytes", "32", "--max-ner-chunk-bytes", "16", "--ner-options", "types.json"]);
    let selected = options(&command); let limits = selected.validate(&command.args).unwrap();
    let request = selected.request(&command.args, limits);
    let detector = selected.detector(&command.args, limits, NerOptions::default());
    assert_eq!(command.args.max_input_bytes, 32);
    assert_eq!(detector.planning.max_input_bytes, limits.common.result_bytes);
    assert_eq!(detector.mapping.chunks.max_input_bytes, limits.common.result_bytes);
    assert_eq!(request.rule_budget.max_input_bytes, limits.common.result_bytes);
    assert_eq!(selected.redaction.source.options_path().unwrap(), std::path::Path::new("types.json"));
    assert_eq!(detector.per_chunk.max_kv_bytes, limits.common.native.kv_bytes);
    assert_eq!(detector.mapping.chunks.max_chunk_bytes, 16);
}

#[test]
fn memory_admission_retains_all_chunk_outputs_full_kv_and_two_preparations() {
    for width in [3, 128, 10752] {
        let mut command = command(&[]);
        command.args.group_width = width; command.args.max_load_mib = 65536; command.args.memory_mib = 65536;
        let selected = options(&command);
        let short = selected.redaction.validate(&command.args).unwrap();
        let limits = selected.validate(&command.args).unwrap();
        let budget = selected.redaction.source.budget(&command.args, short.common);
        let expected = 4 * (budget.max_output_bytes * 64 + 32 * MIB)
            + 8 * u64::from(budget.max_output_tokens) * 64
            + 4096 * std::mem::size_of::<SourceFieldEvidence>() as u64
            + 2 * 16384 * std::mem::size_of::<VerifiedSourceSpan>() as u64 + short.edit_bytes;
        assert_eq!(limits.workspace_bytes, expected);
        assert_eq!(limits.common.minimum_memory_bytes,
            short.common.minimum_memory_bytes - (budget.max_output_bytes * 4
                + u64::from(budget.max_output_tokens) * 8 + short.edit_bytes) + expected);
        assert_eq!(budget.max_kv_bytes, Int4MemoryRequirement::for_context(command.args.context_tokens, width).unwrap().kv_bytes);
        command.args.memory_mib = limits.common.minimum_memory_bytes.saturating_sub(1) / MIB;
        assert!(options(&command).validate(&command.args).is_err());
        command.args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
        assert!(options(&command).validate(&command.args).is_ok());
    }
}

#[test]
fn a_single_plan_preparation_reserve_cannot_admit_the_complete_chunk_population() {
    let mut command = command(&[]);
    command.args.preparation_mib = 384;
    let selected = options(&command);
    assert!(selected.redaction.validate(&command.args).is_ok());
    assert!(selected.validate(&command.args).is_err());
    let mut budget = selected.redaction.source.budget(&command.args, selected.redaction.validate(&command.args).unwrap().common);
    let mapping = selected.mapping_limits(&command.args, 1_048_576).unwrap();
    budget.max_output_bytes = u64::MAX;
    assert!(workspace_bytes(budget, mapping, 1).is_err());
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_does_not_consume_the_document_or_binary_key() {
    for command in [command(&[]), command(&["document", "--action", "pseudonymize",
        "--key-stdin", "--key-id", "k", "--namespace", "n"])] {
        assert_eq!(command.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Unavailable));
    }
}

#[cfg(feature = "asupersync-runtime")]
#[test]
#[ignore = "requires controller-admitted pinned sources, a fresh snapshot and an available explicit SIMD kernel"]
fn source_scalar_and_pinned_snapshot_simd_match_after_complete_repartitioned_verification() {
    let source = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit pinned source directory required");
    let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit source manifest required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required")
        .parse().expect("memory ceiling must be MiB");
    let snapshot = std::env::var("FNLP_INT4_TEST_REDACT_DOCUMENT_SNAPSHOT").expect("explicit new snapshot required");
    assert!(!std::path::Path::new(&snapshot).exists(), "fixture requires a fresh path and deletes nothing");
    let kernel: Int8KernelKind = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit SIMD kernel required")
        .parse().expect("canonical kernel ID required");
    assert_ne!(kernel, Int8KernelKind::Scalar);
    PackedInt4Kernel::detect(kernel).expect("requested kernel must be available");
    let document = "é Contact a@example.org near 上海. Another a@example.org appears.";
    let mut convert = command(&["--rules", "email", "--max-ner-chunk-bytes", "16", "--save-snapshot", &snapshot]);
    convert.args.source_dir = Some(PathBuf::from(source)); convert.args.source_manifest = Some(PathBuf::from(manifest));
    convert.args.memory_mib = memory;
    let mut bytes = Vec::new(); convert.execute(&mut document.as_bytes(), &mut bytes).unwrap();
    let scalar: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(scalar["output"]["original"]["preflight"]["chunks"].as_u64().unwrap() > 1);
    assert_eq!(scalar["output"]["result"]["verification"], "clean_declared_union");
    assert!(!scalar["output"]["result"]["text"].as_str().unwrap().contains("a@example.org"));
    assert!(scalar["output"]["verification"].is_object());
    let pin = scalar["model_content_sha256"].as_str().unwrap();
    let mut reload = parsed(&["--rules", "email", "--max-ner-chunk-bytes", "16", "--snapshot", &snapshot,
        "--expected-model-sha256", pin, "--kernel", kernel.id()]).unwrap();
    reload.args.memory_mib = memory;
    let mut bytes = Vec::new(); reload.execute(&mut document.as_bytes(), &mut bytes).unwrap();
    let selected: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(selected["output"], scalar["output"]);
    assert_eq!(selected["integer_kernel"], kernel.id()); assert_eq!(selected["snapshot"]["operation"], "loaded");
}
