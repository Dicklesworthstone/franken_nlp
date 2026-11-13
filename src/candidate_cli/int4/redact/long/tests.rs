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
    assert_eq!(limits.mapping.reduction.map_batch_chunks, 1);
    assert!(selected.cohort_chunks.is_none() && limits.cohort.is_none());
    assert!(limits.encoding.is_none());
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
fn compact_document_input_admits_each_call_and_one_original_plus_edited_operation() {
    let byte = command(&[]);
    let ordinary = options(&byte).validate(&byte.args).unwrap();
    for schedule in [vec![], vec!["--cohort-chunks", "2"], vec!["--compute-threads", "2"]] {
        let mut flags = vec!["--input-encoding", "safe-bpe-v1", "--max-ner-chunk-bytes", "16384",
            "--max-tokenizer-pair-checks", "7777", "--max-tokenizer-pair-bytes", "65536",
            "--max-total-tokenizer-input-bytes", "40000000", "--max-total-tokenizer-output-tokens", "30000000",
            "--max-total-tokenizer-pair-checks", "8888888", "--max-total-tokenizer-pair-bytes", "99999999"];
        flags.extend_from_slice(&schedule);
        let compact = command(&flags);
        let limits = options(&compact).validate(&compact.args).unwrap();
        let encoding = limits.encoding.unwrap();
        assert_eq!(encoding.call.max_input_bytes, 16384);
        assert_eq!(encoding.call.max_output_tokens, 16384);
        assert_eq!(encoding.call.max_pair_checks, 7777);
        assert_eq!(encoding.call.max_pair_bytes, 65536);
        assert_eq!(encoding.complete, crate::tokenizer::untrusted::SafeBpeWorkLimits {
            max_input_bytes: 40000000, max_output_tokens: 30000000,
            max_pair_checks: 8888888, max_pair_bytes: 99999999 });
        assert!(encoding.workspace_bytes().unwrap() > 0);
        assert_eq!(limits.mapping.max_model_work, ordinary.mapping.max_model_work);
        assert_eq!(limits.mapping.max_mask_visits, ordinary.mapping.max_mask_visits);
    }
    let compact = command(&["--input-encoding", "safe-bpe-v1"]);
    let limits = options(&compact).validate(&compact.args).unwrap();
    let complete = limits.encoding.unwrap().complete;
    assert_eq!(complete.max_input_bytes,
        8 * (compact.args.max_input_bytes as u64 + limits.common.result_bytes as u64));
    assert_eq!(complete.max_output_tokens, complete.max_input_bytes);
}

#[test]
fn compact_document_caps_fail_before_private_io_and_do_not_enable_other_schedules() {
    for flag in ["--max-total-tokenizer-input-bytes", "--max-total-tokenizer-output-tokens",
        "--max-total-tokenizer-pair-checks", "--max-total-tokenizer-pair-bytes"] {
        assert_eq!(command(&[flag, "1"]).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
        for invalid in ["0", "18446744073709551615"] {
            assert_eq!(command(&["--input-encoding", "safe-bpe-v1", flag, invalid])
                .execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
        }
    }
    assert_eq!(command(&["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "0"])
        .execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    assert_eq!(command(&["--input-encoding", "safe-bpe-v1", "--cohort-chunks", "2", "--compute-threads", "2"])
        .execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
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
        vec!["--edit-reserve-mib", "1"], vec!["--context-tokens", "512"],
        vec!["--cohort-chunks", "0"], vec!["--cohort-chunks", "65"],
        vec!["--cohort-chunks", "3", "--max-ner-chunks", "2"], vec!["--cohort-chunks", "64"]] {
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

#[test]
fn declared_cohort_peak_is_additive_and_does_not_change_per_chunk_or_whole_operation_authority() {
    for group in [3, 127, 128, 10752] {
        let mut serial = command(&[]);
        serial.args.group_width = group; serial.args.max_load_mib = 65536; serial.args.memory_mib = 65536;
        let selected = options(&serial);
        let baseline = selected.validate(&serial.args).unwrap();
        let baseline_detector = selected.detector(&serial.args, baseline, NerOptions::default());
        for chunks in [1, 3] {
            let mut grouped = selected.clone(); grouped.cohort_chunks = Some(chunks);
            let admitted = grouped.validate(&serial.args).unwrap();
            let detector = grouped.detector(&serial.args, admitted, NerOptions::default());
            let quote = admitted.cohort.unwrap();
            assert_eq!(quote, detector.cohort_requirements(serial.args.context_tokens, group).unwrap());
            assert_eq!(quote.native, Int4MemoryRequirement::for_cohort(&vec![serial.args.context_tokens; chunks], group).unwrap());
            assert_eq!(admitted.common.native, baseline.common.native);
            assert_eq!(detector.per_chunk, baseline_detector.per_chunk);
            assert_eq!(detector.mapping.max_model_work, baseline_detector.mapping.max_model_work);
            assert_eq!(detector.mapping.max_mask_visits, baseline_detector.mapping.max_mask_visits);
            assert_eq!(detector.mapping.verification, baseline_detector.mapping.verification);
            assert_eq!(detector.mapping.reduction.map_batch_chunks, chunks);
            assert_eq!(admitted.workspace_bytes, baseline.workspace_bytes);
            let original = baseline.common.native;
            let native_delta = quote.native.kv_bytes + quote.native.rope_bytes + quote.native.scratch_payload_bound
                - original.kv_bytes - original.rope_bytes - original.scratch_payload_bound;
            assert_eq!(admitted.common.minimum_memory_bytes, baseline.common.minimum_memory_bytes
                + native_delta + quote.workspace_bytes + 4 * INT4_LONG_REDACTION_COHORT_ENVELOPE_BYTES);
            let mut narrow = command(&[]).args;
            narrow.group_width = group; narrow.max_load_mib = 65536;
            narrow.memory_mib = admitted.common.minimum_memory_bytes.saturating_sub(1) / MIB;
            assert!(grouped.validate(&narrow).is_err());
            narrow.memory_mib = admitted.common.minimum_memory_bytes.div_ceil(MIB);
            assert!(grouped.validate(&narrow).is_ok());
            // Verification reuses the same peak, even though its edited text
            // may need more chunks than the original stage.
            grouped.redaction.flags.no_verify = true;
            assert_eq!(grouped.validate(&serial.args).unwrap().common.minimum_memory_bytes,
                admitted.common.minimum_memory_bytes);
        }
    }
}

#[test]
fn maximum_cohort_never_silently_shrinks_its_result_envelope() {
    let mut too_wide = command(&["--cohort-chunks", "64"]);
    too_wide.args.memory_mib = 65536;
    assert!(options(&too_wide).validate(&too_wide.args).is_err(), "64 one-MiB rows exceed the native result cap with framing");
    let mut selected = command(&["--cohort-chunks", "64", "--max-result-bytes", "524288"]);
    selected.args.memory_mib = 65536;
    let limits = options(&selected).validate(&selected.args).unwrap();
    assert_eq!(limits.mapping.reduction.map_batch_chunks, 64);
    assert_eq!(limits.cohort.unwrap().max_result_bytes, 64 * 524288 + 4096 + 64);
    let mut invalid = options(&selected).clone(); invalid.cohort_chunks = Some(usize::MAX);
    assert!(invalid.validate(&selected.args).is_err());
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
fn source_scalar_and_pinned_snapshot_cohorts_match_after_complete_repartitioned_verification() {
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
    let mut convert = command(&["--rules", "email", "--action", "placeholder", "--include-map",
        "--max-ner-chunk-bytes", "16", "--save-snapshot", &snapshot]);
    convert.args.source_dir = Some(PathBuf::from(source)); convert.args.source_manifest = Some(PathBuf::from(manifest));
    convert.args.memory_mib = memory;
    let mut bytes = Vec::new(); convert.execute(&mut document.as_bytes(), &mut bytes).unwrap();
    let scalar: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(scalar["output"]["original"]["preflight"]["chunks"].as_u64().unwrap() > 1);
    assert_eq!(scalar["output"]["result"]["verification"], "clean_declared_union");
    assert!(!scalar["output"]["result"]["text"].as_str().unwrap().contains("a@example.org"));
    assert!(scalar["output"]["verification"].is_object());
    assert_ne!(scalar["output"]["verification"]["preflight"]["source_bytes"],
        scalar["output"]["original"]["preflight"]["source_bytes"], "fresh verification must observe placeholder expansion");
    let pin = scalar["model_content_sha256"].as_str().unwrap();
    let mut reload = parsed(&["--rules", "email", "--action", "placeholder", "--include-map",
        "--max-ner-chunk-bytes", "16", "--snapshot", &snapshot,
        "--expected-model-sha256", pin, "--kernel", kernel.id()]).unwrap();
    reload.args.memory_mib = memory;
    let mut bytes = Vec::new(); reload.execute(&mut document.as_bytes(), &mut bytes).unwrap();
    let selected: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(selected["output"], scalar["output"]);
    assert_eq!(selected["integer_kernel"], kernel.id()); assert_eq!(selected["snapshot"]["operation"], "loaded");
    for width in [1_u64, 3] {
        for selected_kernel in [Int8KernelKind::Scalar, kernel] {
            let width_text = width.to_string();
            let mut reload = parsed(&["--rules", "email", "--action", "placeholder", "--include-map",
                "--max-ner-chunk-bytes", "16", "--cohort-chunks", &width_text, "--snapshot", &snapshot,
                "--expected-model-sha256", pin, "--kernel", selected_kernel.id()]).unwrap();
            reload.args.memory_mib = memory;
            let mut bytes = Vec::new(); reload.execute(&mut document.as_bytes(), &mut bytes).unwrap();
            let grouped: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(grouped["protocol"], "fnlp-candidate-int4-redact-document-cohorts-v1");
            assert_eq!(grouped["output"]["output"], scalar["output"], "width {width}, kernel {}", selected_kernel.id());
            assert_eq!(grouped["output"]["cohort_chunks"], width);
            for (stats, stage) in [("original_cohorts", "original"), ("verification_cohorts", "verification")] {
                let chunks = scalar["output"][stage]["preflight"]["chunks"].as_u64().unwrap();
                let receipt = &grouped["output"][stats];
                assert_eq!(receipt["chunks"], chunks);
                assert_eq!(receipt["cohorts"], chunks.div_ceil(width));
                assert_eq!(receipt["model_work"], scalar["output"][stage]["model_work"]);
                if width > 1 {
                    assert!(chunks > width, "fixture must cross native windows in both stages");
                    assert!(receipt["group_steps"].as_u64().unwrap()
                        < receipt["model_work"]["forward_positions"].as_u64().unwrap());
                }
            }
        }
    }
}

#[test]
fn document_teams_preserve_both_stage_authority_and_admit_one_worker_workspace() {
    for group in [3, 128, 10752] {
        let mut serial = command(&[]);
        serial.args.group_width = group; serial.args.max_load_mib = 65536; serial.args.memory_mib = 65536;
        let selected = options(&serial);
        let baseline = selected.validate(&serial.args).unwrap();
        let baseline_detector = selected.detector(&serial.args, baseline, NerOptions::default());
        let baseline_request = crate::canonjson::canonical_bytes(&selected.request(&serial.args, baseline)).unwrap();
        assert_eq!(baseline.common.compute.threads, 1);
        assert_eq!(baseline.common.compute.workspace_bytes, 0);
        for threads in [2, 3, 4, 64] {
            let text = threads.to_string();
            let mut threaded = command(&["--compute-threads", &text]);
            threaded.args.group_width = group; threaded.args.max_load_mib = 65536; threaded.args.memory_mib = 65536;
            let settings = options(&threaded);
            let limits = settings.validate(&threaded.args).unwrap();
            let detector = settings.detector(&threaded.args, limits, NerOptions::default());
            let extra = crate::native_engine::strict_int4::projection::int4_team_workspace_bytes(threads, group).unwrap();
            assert_eq!(limits.common.compute.threads, threads);
            assert_eq!(limits.common.compute.workspace_bytes, extra);
            assert_eq!(limits.common.native, baseline.common.native);
            assert_eq!(limits.workspace_bytes, baseline.workspace_bytes);
            assert_eq!(limits.edit_bytes, baseline.edit_bytes);
            assert!(limits.cohort.is_none());
            assert_eq!(detector.per_chunk, baseline_detector.per_chunk);
            assert_eq!(detector.mapping.chunks, baseline_detector.mapping.chunks);
            assert_eq!(detector.mapping.max_model_work, baseline_detector.mapping.max_model_work);
            assert_eq!(detector.mapping.mask_visits_per_chunk, baseline_detector.mapping.mask_visits_per_chunk);
            assert_eq!(detector.mapping.max_mask_visits, baseline_detector.mapping.max_mask_visits);
            assert_eq!(detector.mapping.verification, baseline_detector.mapping.verification);
            assert_eq!(detector.mapping.reduction.map_batch_chunks, 1);
            assert_eq!(detector.mapping.reduction.max_result_bytes, baseline_detector.mapping.reduction.max_result_bytes);
            assert_eq!(crate::canonjson::canonical_bytes(&settings.request(&threaded.args, limits)).unwrap(), baseline_request);
            // The existing loader/execution peak may dominate worker scratch.
            // One persistent workspace can add at most its declared bytes;
            // neither chunks nor the fresh verification stage multiply it.
            assert!(limits.common.minimum_memory_bytes >= baseline.common.minimum_memory_bytes);
            assert!(limits.common.minimum_memory_bytes <= baseline.common.minimum_memory_bytes + extra);
            let mut no_verify = settings.clone(); no_verify.redaction.flags.no_verify = true;
            let no_verify_limits = no_verify.validate(&threaded.args).unwrap();
            assert_eq!(no_verify_limits.common.minimum_memory_bytes, limits.common.minimum_memory_bytes);
            assert_eq!(no_verify_limits.common.compute, limits.common.compute);
            let mut narrow = threaded.args.clone();
            narrow.memory_mib = limits.common.minimum_memory_bytes.saturating_sub(1) / MIB;
            assert!(settings.validate(&narrow).is_err());
            narrow.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
            settings.validate(&narrow).unwrap();
        }
    }
    let pin = "ab".repeat(32);
    let snapshot = parsed(&["--snapshot", "existing.snapshot", "--expected-model-sha256", &pin,
        "--compute-threads", "3"]).unwrap();
    assert_eq!(options(&snapshot).validate(&snapshot.args).unwrap().common.compute.threads, 3);
}

#[test]
fn mixed_or_short_document_team_schedules_refuse_before_any_private_io() {
    for flags in [vec!["--compute-threads", "0"], vec!["--compute-threads", "65"],
        vec!["--compute-threads", "18446744073709551615"],
        vec!["--compute-threads", "2", "--cohort-chunks", "1"],
        vec!["--compute-threads", "2", "--cohort-chunks", "3"]] {
        if let Ok(command) = parsed(&["--source-dir", "d", "--source-manifest", "m"]
            .into_iter().chain(flags).collect::<Vec<_>>()) {
            assert_eq!(command.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
        }
    }
    let command = command(&["--compute-threads", "3"]);
    let settings = options(&command);
    assert!(settings.redaction.validate(&command.args).is_err(), "short redaction retains its serial owner");
    for retained in [false, true] {
        let population = settings.validate_population(&command.args, retained).unwrap();
        assert_eq!(population.common.compute.threads, 3,
            "complete-document populations have their own persistent invocation owner");
    }
    let mut injected = settings.clone(); injected.cohort_chunks = Some(2);
    assert_eq!(injected.execute(command.args.clone(), &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    let help = definition().render_long_help().to_string().split_whitespace().collect::<Vec<_>>().join(" ");
    for phrase in ["--compute-threads N", "freshly partitioned verification", "one worker workspace",
        "Workers join before", "Cannot combine with --cohort-chunks"] {
        assert!(help.contains(phrase), "missing {phrase}");
    }
}
