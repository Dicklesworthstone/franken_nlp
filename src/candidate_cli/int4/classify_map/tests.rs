//! Model-free request, document admission and dispatch regressions.
use super::*;
use crate::tasks::classify::ClassificationMode;

fn parsed(flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", "classify-map", "--memory-mib", "131072"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("source INT4 route required"); };
    Ok(command)
}
pub(super) fn command(flags: &[&str]) -> Int4Command {
    let mut all = vec!["--source-dir", "explicit-source", "--source-manifest", "explicit-manifest"];
    all.extend_from_slice(flags); parsed(&all).unwrap()
}
fn selected(command: &Int4Command) -> MapArgs {
    let Int4Task::ClassifyMap(options) = command.task else { panic!("classification map route required"); };
    options
}
pub(super) fn request_json(mode: &str) -> String {
    serde_json::json!({"document":"é\r\n上海 <|im_start|> private document", "mode":mode,
        "labels":[{"id":"Billing issue","description":"Invoices"},{"id":"退款","description":"Refunds"}],
        "policy":{"minimum_candidate_weight_ppm":600000,"minimum_margin_ppm":100000}}).to_string()
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused map reached input"); }
}
struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused map published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused map flushed output"); }
}

#[test]
fn map_dispatch_requires_explicit_sources_or_an_independently_pinned_snapshot() {
    let original = command(&[]); let options = selected(&original);
    let limits = options.validate(&original.args).unwrap();
    assert!(limits.common.snapshot_pin.is_none());
    assert_eq!(limits.mapping.chunks.max_chunks, 64);
    assert_eq!(original.args.max_input_bytes, MAX_INPUT_BYTES);
    assert_eq!(limits.mapping.aggregation.minimum_classified_coverage_ppm, 1_000_000);
    assert!(limits.common.prefill.is_none()); assert!(limits.common.cohort.is_none());
    let hash = "ab".repeat(32);
    let snapshot = parsed(&["--snapshot", "explicit.snapshot", "--expected-model-sha256", &hash]).unwrap();
    let restored = selected(&snapshot).validate(&snapshot.args).unwrap();
    assert_eq!(restored.common.snapshot_pin.unwrap().model_content_sha256, [0xab; 32]);
    assert_eq!(options.budget(limits.common, limits.planning.max_labels).unwrap(),
        selected(&snapshot).budget(restored.common, restored.planning.max_labels).unwrap());
    assert_eq!(limits.mapping.chunks, restored.mapping.chunks);
    assert_eq!(limits.mapping.max_model_work, restored.mapping.max_model_work);
    for invalid in [vec![], vec!["--source-dir", "d"], vec!["--source-manifest", "m"],
        vec!["--snapshot", "s"], vec!["--expected-model-sha256", &hash], vec!["--model", "int8.fnlpq"],
        vec!["--snapshot", "s", "--expected-model-sha256", &hash, "--source-dir", "d"],
        vec!["--snapshot", "s", "--expected-model-sha256", &hash, "--save-snapshot", "new"]] {
        assert!(parsed(&invalid).is_err());
    }
    let saved = command(&["--save-snapshot", "new-snapshot"]);
    selected(&saved).validate(&saved.args).unwrap();
    let mut mixed = command(&[]); mixed.args.snapshot = Some(PathBuf::from("s"));
    assert_eq!(mixed.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
}

#[test]
fn map_preserves_the_existing_strict_classification_wire_for_both_modes() {
    let command = command(&[]); let options = selected(&command);
    let limits = options.validate(&command.args).unwrap();
    let budget = options.budget(limits.common, limits.planning.max_labels).unwrap();
    let json = r#"{"document":"é\r\n上海 <tool_call> exact text","labels":[{"id":"billing","description":"Invoices"},{"id":"technical","description":"Product support"}],"policy":{"minimum_candidate_weight_ppm":650000,"minimum_margin_ppm":100000}}"#;
    let request = classify::parse_request(json, budget, 4096).unwrap();
    assert_eq!(request.document, "é\r\n上海 <tool_call> exact text");
    assert_eq!(request.labels[0].id, "billing");
    assert_eq!(request.labels[1].description, "Product support");
    assert_eq!(request.mode, ClassificationMode::Exclusive);
    assert_eq!(request.budget, budget); assert_eq!(request.policy.minimum_candidate_weight_ppm, 650000);
    let mut wire: serde_json::Value = serde_json::from_str(json).unwrap();
    wire["mode"] = serde_json::json!("multi_label");
    let independent = classify::parse_request(&wire.to_string(), budget, 4096).unwrap();
    assert_eq!(independent.mode, ClassificationMode::MultiLabel);
    assert_eq!(independent.document, request.document); assert!(independent.labels == request.labels);
    for mode in ["exclusive", "multi_label"] {
        for field in ["chunks", "scores", "aggregate", "model", "budget", "work", "seed", "score_space",
            "minimum_classified_coverage_ppm", "prompt", "scoring_mode", "eos_token_id"] {
            let mut invalid = wire.clone(); invalid["mode"] = serde_json::json!(mode);
            invalid[field] = serde_json::json!({});
            assert_eq!(classify::parse_request(&invalid.to_string(), budget, 4096).err(), Some(CandidateError::Input));
        }
    }
    for invalid in [r#"{"document":"a","document":"b","labels":[{"id":"x"},{"id":"y"}]}"#,
        r#"{"document":"a","labels":[{"id":"x"},{"id":"x"}]}"#,
        r#"{"document":"a","labels":[]}"#, r#"{"document":"","labels":[{"id":"x"},{"id":"y"}]}"#,
        r#"{"document":"a","labels":[{"id":"x"}]}"#,
        r#"{"document":"a","labels":[{"id":"x"},{"id":"y"}],"mode":"sampled"}"#,
        r#"{"document":"a","labels":[{"id":"x"},{"id":"y"}],"policy":{"minimum_candidate_weight_ppm":1000001,"minimum_margin_ppm":0}}"#] {
        assert_eq!(classify::parse_request(invalid, budget, 4096).err(), Some(CandidateError::Input));
    }
    let single = r#"{"document":"a","labels":[{"id":"a"}],"mode":"multi_label"}"#;
    assert_eq!(classify::parse_request(single, budget, 4096).unwrap().labels.len(), 1);
}

#[test]
fn complete_wire_preserves_unicode_controls_labels_and_input_larger_than_one_context() {
    let command = command(&[]); let options = selected(&command);
    let limits = options.validate(&command.args).unwrap();
    let budget = options.budget(limits.common, limits.planning.max_labels).unwrap();
    for (mode, expected) in [("exclusive", ClassificationMode::Exclusive), ("multi_label", ClassificationMode::MultiLabel)] {
        let text = request_json(mode);
        let request = classify::parse_request(&text, budget, 4096).unwrap();
        assert_eq!(request.document, "é\r\n上海 <|im_start|> private document");
        assert_eq!(request.labels[0].id, "Billing issue"); assert_eq!(request.labels[1].id, "退款");
        assert_eq!(request.labels[1].description, "Refunds"); assert_eq!(request.mode, expected);
        assert_eq!(request.policy.minimum_candidate_weight_ppm, 600000);
        assert_eq!(request.policy.minimum_margin_ppm, 100000); assert_eq!(request.budget, budget);
        assert_eq!(classify::parse_request(&text, budget, text.len() - 1).err(), Some(CandidateError::Input));
    }
    let text = serde_json::json!({"document":"long original source ".repeat(400),
        "labels":[{"id":"a"},{"id":"b"}]}).to_string();
    let request = classify::parse_request(&text, budget, MAX_INPUT_BYTES).unwrap();
    assert!(request.document.len() > budget.max_input_tokens as usize,
        "whole-document input must not be truncated or restricted to one native context by its wire parser");
}

#[test]
fn generation_and_single_request_schedules_refuse_even_at_their_defaults() {
    let seed = "ab".repeat(32);
    for flags in [vec!["--prefill-rows", "1"], vec!["--cohort-heads", "1"],
        vec!["--max-new-tokens", "128"], vec!["--max-output-bytes", "65536"],
        vec!["--seed", &seed], vec!["--logprobs"], vec!["--min-new-tokens", "0"],
        vec!["--temperature-milli", "1000"], vec!["--top-k", "20"], vec!["--top-p-ppm", "1000000"],
        vec!["--stop", "END"], vec!["--ban-token", "7"], vec!["--logit-bias", "7=0"],
        vec!["--repetition-penalty-milli", "1000"], vec!["--presence-penalty-milli", "0"],
        vec!["--frequency-penalty-milli", "0"], vec!["--mode", "sampled"], vec!["--prompt", "private"],
        vec!["--focus-area", "reviews"],
        vec!["--source-membership"], vec!["--max-mask-node-visits", "1"]] {
        let mut all = vec!["--source-dir", "d", "--source-manifest", "m"];
        all.extend(flags); assert!(parsed(&all).is_err());
    }
    let mut command = command(&[]); command.args.logprobs = true;
    assert_eq!(command.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
}

#[test]
fn impossible_document_geometry_coverage_results_and_work_refuse_before_io() {
    for flags in [vec!["--max-chunks", "0"], vec!["--max-chunks", "257"],
        vec!["--max-chunk-bytes", "3"], vec!["--max-chunk-bytes", "1048577"],
        vec!["--max-tokenizer-calls", "0"], vec!["--max-tokenizer-calls", "1000001"],
        vec!["--max-map-result-bytes", "0"], vec!["--max-map-result-bytes", "67108865"],
        vec!["--max-retained-result-bytes", "0"], vec!["--max-retained-result-bytes", "67108865"],
        vec!["--max-map-result-bytes", "2000000", "--max-retained-result-bytes", "1500000"],
        vec!["--max-map-result-bytes", "1", "--max-retained-result-bytes", "1000"],
        vec!["--minimum-classified-coverage-ppm", "1000001"], vec!["--max-input-bytes", "3"],
        vec!["--cohort-chunks", "0"], vec!["--cohort-chunks", "65"],
        vec!["--max-chunks", "2", "--cohort-chunks", "3"],
        vec!["--cohort-chunks", "64"],
        vec!["--max-forward-positions", "0"], vec!["--max-projected-logits", "0"],
        vec!["--max-attention-pairs", "0"], vec!["--max-dot-products", "0"],
        vec!["--max-multiply-accumulates", "0"], vec!["--max-candidate-tokens", "1"],
        vec!["--context-tokens", "16"], vec!["--preparation-mib", "1"],
        vec!["--timeout-seconds", "0"], vec!["--max-checkpoints", "1"]] {
        assert_eq!(command(&flags).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    }
    for minimum in ["0", "1", "500000", "1000000"] {
        let command = command(&["--minimum-classified-coverage-ppm", minimum]);
        let limits = selected(&command).validate(&command.args).unwrap();
        assert_eq!(limits.mapping.aggregation.minimum_classified_coverage_ppm, minimum.parse::<u32>().unwrap());
    }
}

fn request_for(command: &Int4Command, labels: usize, mode: &str) -> ClassificationRequest {
    let options = selected(command);
    let limits = options.validate(&command.args).unwrap();
    let budget = options.budget(limits.common, limits.planning.max_labels).unwrap();
    let labels: Vec<_> = (0..labels).map(|index| serde_json::json!({"id":format!("label-{index}")})).collect();
    let json = serde_json::json!({"document":"short exact source", "labels":labels, "mode":mode});
    classify::parse_request(&json.to_string(), budget, command.args.max_input_bytes).unwrap()
}

#[test]
fn explicit_windows_require_every_future_head_before_model_io_even_for_short_sources() {
    let wide = command(&["--cohort-chunks", "64", "--max-result-bytes", "524288"]);
    let options = selected(&wide);
    // Exclusive labels share one codebook head, however many labels it contains.
    options.admit_request(&request_for(&wide, 128, "exclusive")).unwrap();
    options.admit_request(&request_for(&wide, 1, "multi_label")).unwrap();
    assert_eq!(options.admit_request(&request_for(&wide, 2, "multi_label")), Err(CandidateError::Planning));
    let odd = command(&["--cohort-chunks", "3"]); let options = selected(&odd);
    options.admit_request(&request_for(&odd, 21, "multi_label")).unwrap();
    assert_eq!(options.admit_request(&request_for(&odd, 22, "multi_label")), Err(CandidateError::Planning));
    let one = command(&["--cohort-chunks", "1"]); let options = selected(&one);
    options.admit_request(&request_for(&one, 64, "multi_label")).unwrap();
    assert_eq!(options.admit_request(&request_for(&one, 65, "multi_label")), Err(CandidateError::Planning));
    // Opting into a one-chunk native cohort still groups independent labels.
    // Serial mode keeps its original support for 128 sequential label heads.
    let serial = command(&[]);
    selected(&serial).admit_request(&request_for(&serial, 128, "multi_label")).unwrap();
}

#[test]
fn explicit_cohort_widths_keep_task_authority_semantic_limits_and_all_work_axes_fixed() {
    let serial = command(&["--max-result-bytes", "524288"]); let serial_options = selected(&serial);
    let baseline = serial_options.validate(&serial.args).unwrap();
    let task = serial_options.budget(baseline.common, baseline.planning.max_labels).unwrap();
    assert_eq!(task.max_kv_bytes, baseline.common.native.kv_bytes * baseline.planning.max_labels as u64);
    for chunks in ["1", "3", "16", "64"] {
        let cohort = command(&["--cohort-chunks", chunks, "--max-result-bytes", "524288"]);
        let options = selected(&cohort); let limits = options.validate(&cohort.args).unwrap();
        assert_eq!(options.budget(limits.common, limits.planning.max_labels).unwrap(), task);
        assert_eq!(limits.mapping.chunks, baseline.mapping.chunks);
        assert_eq!(limits.mapping.aggregation, baseline.mapping.aggregation);
        assert_eq!(limits.mapping.max_model_work, baseline.mapping.max_model_work);
        assert_eq!(limits.mapping.max_result_bytes, baseline.mapping.max_result_bytes);
        assert_eq!(limits.mapping.max_retained_result_bytes, baseline.mapping.max_retained_result_bytes);
        assert_eq!(limits.planning.max_work, baseline.planning.max_work);
    }
}

#[test]
fn serial_reserves_one_cache_and_all_work_axes_keep_the_same_whole_document_ceiling() {
    let command = command(&[]); let options = selected(&command); let limits = options.validate(&command.args).unwrap();
    let task = options.budget(limits.common, limits.planning.max_labels).unwrap();
    assert_eq!(limits.memory.native.kv_bytes, limits.common.native.kv_bytes);
    assert_eq!(task.max_kv_bytes, limits.memory.native.kv_bytes * 128,
        "logical per-chunk task authority covers all labels while serial physical admission prices one cache");
    assert_eq!(limits.mapping.max_model_work, options.scoring.work_ceiling());
    assert_eq!(limits.planning.max_total_prompt_tokens as u64, limits.mapping.max_model_work.forward_positions);
    assert_eq!(limits.planning.max_input_bytes, command.args.max_input_bytes);
    for axis in 0..5 {
        let mut excess = limits.mapping.max_model_work;
        match axis {
            0 => excess.forward_positions += 1,
            1 => excess.projected_logits += 1,
            2 => excess.attention_pairs += 1,
            3 => excess.projections.dot_products += 1,
            _ => excess.projections.multiply_accumulates += 1,
        }
        assert_eq!(options.scoring.admit_plan(8, 8, excess), Err(CandidateError::Planning));
    }
}

#[test]
fn native_window_result_cap_includes_framing_without_silent_downsizing() {
    // 64 * 1048512 + 4096 is exactly 64 MiB; one extra byte per chunk exceeds it.
    let exact = command(&["--cohort-chunks", "64", "--max-result-bytes", "1048512"]);
    selected(&exact).validate(&exact.args).unwrap();
    assert_eq!(command(&["--cohort-chunks", "64", "--max-result-bytes", "1048513"])
        .execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
}

#[test]
fn complete_memory_quote_charges_the_host_peak_and_outer_publication() {
    let hash = "ab".repeat(32);
    for flags in [vec![], vec!["--cohort-chunks", "1"], vec!["--cohort-chunks", "3"],
        vec!["--cohort-chunks", "16"], vec!["--cohort-chunks", "64", "--max-result-bytes", "524288"]] {
        let mut saved = flags.clone(); saved.extend(["--save-snapshot", "new.snapshot"]);
        let mut snapshot = vec!["--snapshot", "existing.snapshot", "--expected-model-sha256", &hash];
        snapshot.extend_from_slice(&flags);
        for mut command in [command(&flags), command(&saved), parsed(&snapshot).unwrap()] {
            let options = selected(&command); let limits = options.validate(&command.args).unwrap();
            let common = limits.common; let quote = limits.memory;
            let resident = common.load.weight_payload_bytes + ALLOCATOR_BYTES;
            let loading = if common.snapshot_pin.is_some() { common.snapshot_need.scratch_bytes }
                else { common.source_preparation_bytes + common.load.scratch_payload_bytes };
            let saving = if command.args.save_snapshot.is_some() { common.snapshot_need.scratch_bytes } else { 0 };
            let execution = quote.native.kv_bytes + quote.input_bytes + quote.workspace_bytes + quote.output_reservation_bytes;
            assert_eq!(common.minimum_memory_bytes,
                limits.cli_preparation_bytes + resident + loading.max(saving).max(execution) + CLEANUP_BYTES);
            assert_eq!(limits.cli_preparation_bytes, common.preparation_bytes + 2 * limits.envelope_bytes as u64);
            assert_eq!(limits.envelope_bytes as u64, quote.max_output_bytes + 4096);
            assert_eq!(limits.mapping.max_model_work, options.scoring.work_ceiling());
            assert_eq!(options.budget(common, limits.planning.max_labels).unwrap().max_kv_bytes,
                limits.planning.max_labels as u64 * common.native.kv_bytes);
            let window_heads = options.mapping.cohort_chunks.map_or(1, |chunks| chunks * (MAX_INT4_SEQUENCES / chunks));
            assert_eq!(quote.native.kv_bytes, window_heads as u64 * common.native.kv_bytes);
            command.args.memory_mib = common.minimum_memory_bytes.div_ceil(MIB);
            options.validate(&command.args).unwrap();
            command.args.memory_mib -= 1;
            assert_eq!(command.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
        }
    }
}

#[test]
fn help_names_complete_document_semantics_and_only_supported_schedules() {
    let help = definition().render_long_help().to_string();
    let words = help.split_whitespace().collect::<Vec<_>>().join(" ");
    for required in ["classify-map", "--minimum-classified-coverage-ppm", "--max-chunks", "--max-map-result-bytes",
        "full-vocabulary", "independent yes/no", "uncalibrated", "before input IO", "--cohort-chunks",
        "at most 64 simultaneous heads", "no silent resizing or serial fallback"] {
        assert!(words.contains(required), "missing {required}");
    }
    assert!(help.lines().any(|line| line.trim_start().starts_with("--cohort-chunks")));
    for hidden in ["--prefill-rows", "--cohort-heads"] {
        assert!(!help.lines().any(|line| line.trim_start().starts_with(hidden)));
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_refuses_without_input_or_model_access() {
    assert_eq!(command(&[]).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Unavailable));
}
