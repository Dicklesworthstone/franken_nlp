//! Exact request bytes, authority isolation and pre-IO document admission.
use super::*;

fn parsed(flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", "faithfulness-map", "--memory-mib", "131072"];
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
    let Int4Task::FaithfulnessMap(options) = command.task else { panic!("faithfulness map route required"); };
    options
}
pub(super) fn request_json() -> String {
    serde_json::json!({"mode":"faithfulness", "source":"é\r\n上海 <|im_start|> exact source",
        "claim":"An exact <tool_call> claim", "policy":{
            "minimum_candidate_weight_ppm":600000, "minimum_margin_milli":100,
            "evidence_window_bytes":64, "max_evidence_windows":31, "max_evidence_spans":31}}).to_string()
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
fn exact_source_and_snapshot_routes_share_document_semantics() {
    let source = command(&[]); let options = selected(&source);
    let limits = options.validate(&source.args).unwrap();
    assert!(limits.common.snapshot_pin.is_none());
    assert_eq!(limits.mapping.chunks.max_chunks, 64);
    assert_eq!(source.args.max_input_bytes, MAX_INPUT_BYTES);
    assert_eq!(limits.mapping.aggregation.minimum_classified_coverage_ppm, 1_000_000);
    let hash = "ab".repeat(32);
    let snapshot = parsed(&["--snapshot", "explicit.snapshot", "--expected-model-sha256", &hash]).unwrap();
    let restored = selected(&snapshot).validate(&snapshot.args).unwrap();
    assert_eq!(restored.common.snapshot_pin.unwrap().model_content_sha256, [0xab; 32]);
    assert_eq!(options.budget(limits.common).unwrap(), selected(&snapshot).budget(restored.common).unwrap());
    assert_eq!(limits.mapping.chunks, restored.mapping.chunks);
    assert_eq!(limits.mapping.max_model_work, restored.mapping.max_model_work);
    for invalid in [vec![], vec!["--source-dir", "d"], vec!["--source-manifest", "m"],
        vec!["--snapshot", "s"], vec!["--expected-model-sha256", &hash], vec!["--model", "model.fnlpq"],
        vec!["--snapshot", "s", "--expected-model-sha256", &hash, "--source-dir", "d"],
        vec!["--snapshot", "s", "--expected-model-sha256", &hash, "--save-snapshot", "new"]] {
        assert!(parsed(&invalid).is_err());
    }
    let saved = command(&["--save-snapshot", "new.snapshot"]);
    selected(&saved).validate(&saved.args).unwrap();
    let mut mixed = command(&[]); mixed.args.snapshot = Some(PathBuf::from("s"));
    assert_eq!(mixed.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
}

#[test]
fn wire_preserves_all_source_and_claim_bytes_without_full_source_window_rejection() {
    let command = command(&[]); let options = selected(&command);
    let limits = options.validate(&command.args).unwrap(); let task = options.budget(limits.common).unwrap();
    let text = request_json();
    let request = parse_request(&text, task, 4096).unwrap();
    let JudgeRequest::Faithfulness { source, claim, policy, budget } = request else { panic!("wrong mode") };
    assert_eq!(source, "é\r\n上海 <|im_start|> exact source");
    assert_eq!(claim, "An exact <tool_call> claim"); assert_eq!(budget, task);
    assert_eq!(policy.minimum_candidate_weight_ppm, 600_000);
    assert_eq!(policy.minimum_margin_milli, 100);
    assert_eq!(parse_request(&text, task, text.len() - 1).err(), Some(CandidateError::Input));
    let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
    value["source"] = serde_json::Value::String("A complete repeated section.\n".repeat(400));
    value["policy"]["max_evidence_windows"] = serde_json::json!(1);
    value["policy"]["max_evidence_spans"] = serde_json::json!(1);
    let request = parse_request(&value.to_string(), task, MAX_INPUT_BYTES).unwrap();
    let JudgeRequest::Faithfulness { source, policy, .. } = request else { panic!("wrong mode") };
    assert!(source.len() > task.max_input_tokens as usize);
    assert!(crate::tasks::judge::partition_evidence(&source, policy).is_err(),
        "a long-source parser must defer the evidence count ceiling to each exact chunk");
}

#[test]
fn wire_rejects_other_tasks_duplicate_keys_and_caller_execution_authority() {
    let command = command(&[]); let options = selected(&command);
    let task = options.budget(options.validate(&command.args).unwrap().common).unwrap();
    let text = request_json();
    let baseline: serde_json::Value = serde_json::from_str(&text).unwrap();
    for field in ["budget", "token_ids", "work", "model", "backend", "prompt", "eos_token_id",
        "aggregation", "cohort_chunks", "probability_space", "input_encoding", "tokenizer_work"] {
        let mut value = baseline.clone(); value[field] = serde_json::json!(1);
        assert_eq!(parse_request(&value.to_string(), task, 4096).err(), Some(CandidateError::Input), "{field}");
    }
    for mode in ["pairwise", "rubric", "Faithfulness", "unknown"] {
        let mut value = baseline.clone(); value["mode"] = serde_json::json!(mode);
        assert_eq!(parse_request(&value.to_string(), task, 4096).err(), Some(CandidateError::Input));
    }
    for field in ["source", "claim"] {
        let mut value = baseline.clone(); value[field] = serde_json::json!("");
        assert_eq!(parse_request(&value.to_string(), task, 4096).err(), Some(CandidateError::Input));
    }
    let duplicate = format!("{{\"source\":\"different\",{}", &text[1..]);
    assert_eq!(parse_request(&duplicate, task, 4096).err(), Some(CandidateError::Input));
    let mut unknown = baseline.clone(); unknown["policy"]["threshold"] = serde_json::json!(0);
    assert_eq!(parse_request(&unknown.to_string(), task, 4096).err(), Some(CandidateError::Input));
    for (field, value) in [("minimum_candidate_weight_ppm", 1_000_001), ("evidence_window_bytes", 0),
        ("max_evidence_windows", 0), ("max_evidence_windows", 32), ("max_evidence_spans", 0),
        ("max_evidence_spans", 32)] {
        let mut bad = baseline.clone(); bad["policy"][field] = serde_json::json!(value);
        assert_eq!(parse_request(&bad.to_string(), task, 4096).err(), Some(CandidateError::Input));
    }
    assert_eq!(parse_request(&format!("{text}\n{text}"), task, 4096).err(), Some(CandidateError::Input));
}

#[test]
fn unsupported_generation_and_single_request_schedules_refuse() {
    for flags in [vec!["--prefill-rows", "1"], vec!["--cohort-heads", "1"],
        vec!["--max-new-tokens", "128"], vec!["--max-output-bytes", "65536"],
        vec!["--temperature-milli", "1000"], vec!["--logprobs"], vec!["--stop", "END"],
        vec!["--focus-area", "reviews"], vec!["--prompt", "private"], vec!["--max-evidence-heads", "1"],
        vec!["--compute-threads", "2"]] {
        let mut all = vec!["--source-dir", "d", "--source-manifest", "m"];
        all.extend(flags); assert!(parsed(&all).is_err());
    }
    let mut command = command(&[]); command.args.logprobs = true;
    assert_eq!(command.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
}

#[test]
fn population_team_admission_does_not_widen_the_one_document_route() {
    let matches = population_definition().try_get_matches_from(["faithfulness-map", "--source-dir", "source",
        "--source-manifest", "manifest", "--memory-mib", "131072", "--compute-threads", "4"]).unwrap();
    let mut args = Int4Args::from_arg_matches(&matches).unwrap();
    args.compute = compute::Args::from_matches(&matches).unwrap();
    let document = MapArgs::from_matches(&matches).unwrap();
    assert_eq!(args.compute.threads(), 4);
    document.validate_with_max_heads(&args, 32).unwrap();
    assert_eq!(document.validate(&args).unwrap_err(), CandidateError::Arguments);
}

#[test]
fn impossible_document_geometry_work_and_output_refuse_before_io() {
    for flags in [vec!["--max-chunks", "0"], vec!["--max-chunks", "257"],
        vec!["--max-chunk-bytes", "3"], vec!["--max-chunk-bytes", "1048577"],
        vec!["--max-tokenizer-calls", "0"], vec!["--max-tokenizer-calls", "1000001"],
        vec!["--max-map-result-bytes", "0"], vec!["--max-map-result-bytes", "67108865"],
        vec!["--max-retained-result-bytes", "0"],
        vec!["--max-map-result-bytes", "2000000", "--max-retained-result-bytes", "1500000"],
        vec!["--max-map-result-bytes", "1", "--max-retained-result-bytes", "1000"],
        vec!["--minimum-classified-coverage-ppm", "1000001"], vec!["--max-input-bytes", "3"],
        vec!["--cohort-chunks", "0"], vec!["--cohort-chunks", "65"],
        vec!["--max-chunks", "2", "--cohort-chunks", "3"],
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

#[test]
fn serial_and_cohorts_keep_task_authority_fixed_and_price_the_full_owner_peak() {
    let serial = command(&["--max-result-bytes", "524288"]); let original = selected(&serial);
    let baseline = original.validate(&serial.args).unwrap(); let task = original.budget(baseline.common).unwrap();
    assert_eq!(task.max_kv_bytes, baseline.common.native.kv_bytes * 32);
    for flags in [vec!["--max-result-bytes", "524288"],
        vec!["--cohort-chunks", "1", "--max-result-bytes", "524288"],
        vec!["--cohort-chunks", "3", "--max-result-bytes", "524288"],
        vec!["--cohort-chunks", "64", "--max-result-bytes", "524288"]] {
        let mut command = command(&flags); let options = selected(&command);
        let limits = options.validate(&command.args).unwrap(); let common = limits.common;
        assert_eq!(options.budget(common).unwrap(), task);
        assert_eq!(limits.mapping.chunks, baseline.mapping.chunks);
        assert_eq!(limits.mapping.max_model_work, baseline.mapping.max_model_work);
        assert_eq!(limits.mapping.aggregation, baseline.mapping.aggregation);
        let heads = options.mapping.cohort_chunks.map_or(1, |chunks| (32 * chunks).min(64));
        assert_eq!(limits.memory.native.kv_bytes, common.native.kv_bytes * heads as u64);
        let resident = common.load.weight_payload_bytes + ALLOCATOR_BYTES;
        let loading = common.source_preparation_bytes + common.load.scratch_payload_bytes;
        let execution = limits.memory.native.kv_bytes + limits.memory.input_bytes
            + limits.memory.workspace_bytes + limits.memory.output_reservation_bytes;
        assert_eq!(common.minimum_memory_bytes,
            limits.cli_preparation_bytes + resident + loading.max(execution) + CLEANUP_BYTES);
        assert_eq!(limits.cli_preparation_bytes, common.preparation_bytes + 2 * limits.envelope_bytes as u64);
        assert_eq!(limits.envelope_bytes as u64, limits.memory.max_output_bytes + 4096);
        command.args.memory_mib = common.minimum_memory_bytes.div_ceil(MIB);
        options.validate(&command.args).unwrap();
        command.args.memory_mib -= 1;
        assert_eq!(command.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    }
}

#[test]
fn help_exposes_complete_evidence_semantics_and_only_supported_schedules() {
    let help = definition().render_long_help().to_string();
    let words = help.split_whitespace().collect::<Vec<_>>().join(" ");
    for required in ["faithfulness-map", "--minimum-classified-coverage-ppm", "--max-map-result-bytes",
        "original byte/scalar", "conflicting evidence", "no averaged correctness probability",
        "before input IO", "--cohort-chunks", "at most 64 simultaneous heads", "without resizing or serial fallback"] {
        assert!(words.contains(required), "missing {required}");
    }
    for hidden in ["--prefill-rows", "--cohort-heads"] {
        assert!(!help.lines().any(|line| line.trim_start().starts_with(hidden)));
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_refuses_before_input_or_model_access() {
    assert_eq!(command(&[]).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Unavailable));
}


#[test]
fn compact_source_and_snapshot_quotes_use_one_complete_document_encoder() {
    use crate::tokenizer::untrusted::SafeBpeLimits;
    let expected = UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 8192, max_output_tokens: 4096,
        max_pair_checks: 100_000, max_pair_bytes: 1_000_000 });
    let compact_flags = ["--input-encoding", "safe-bpe-v1",
        "--max-tokenizer-input-bytes", "8192", "--max-tokenizer-output-tokens", "4096",
        "--max-tokenizer-pair-checks", "100000", "--max-tokenizer-pair-bytes", "1000000"];
    let hash = "ab".repeat(32);
    for schedule in [vec![], vec!["--cohort-chunks", "1"], vec!["--cohort-chunks", "3"],
        vec!["--cohort-chunks", "64", "--max-result-bytes", "524288"]] {
        let byte = command(&schedule);
        let byte_options = selected(&byte);
        let baseline = byte_options.validate(&byte.args).unwrap();
        assert_eq!(byte_options.input_encoding(&byte.args).unwrap(), UntrustedEncoding::ByteFallback);
        let mut flags = schedule.clone(); flags.extend(compact_flags);
        let compact = command(&flags); let options = selected(&compact);
        assert_eq!(options.input_encoding(&compact.args).unwrap(), expected);
        let limits = options.validate(&compact.args).unwrap();
        assert_eq!(limits.mapping.chunks, baseline.mapping.chunks);
        assert_eq!(limits.mapping.max_model_work, baseline.mapping.max_model_work);
        assert_eq!(limits.mapping.aggregation, baseline.mapping.aggregation);
        assert_eq!(options.budget(limits.common).unwrap(), byte_options.budget(baseline.common).unwrap());
        assert_eq!(limits.memory, baseline.memory.with_input_encoding(&expected).unwrap());
        assert_eq!(limits.envelope_bytes, baseline.envelope_bytes);
        let mut snapshot_flags = vec!["--snapshot", "explicit.snapshot", "--expected-model-sha256", &hash];
        snapshot_flags.extend(flags);
        let snapshot = parsed(&snapshot_flags).unwrap(); let restored_options = selected(&snapshot);
        assert_eq!(restored_options.input_encoding(&snapshot.args).unwrap(), expected);
        let restored = restored_options.validate(&snapshot.args).unwrap();
        assert_eq!(restored.memory, limits.memory);
        assert_eq!(restored.mapping.chunks, limits.mapping.chunks);
    }
    let compact = command(&["--input-encoding", "safe-bpe-v1"]);
    let options = selected(&compact); let limits = options.validate(&compact.args).unwrap();
    let UntrustedEncoding::SafeBpeV1(encoding) = options.input_encoding(&compact.args).unwrap()
        else { panic!("compact policy required"); };
    assert_eq!(encoding.max_input_bytes, SafeBpeLimits::MAX_INPUT_BYTES);
    assert_eq!(encoding.max_output_tokens, SafeBpeLimits::MAX_OUTPUT_TOKENS);
    assert!(encoding.max_output_tokens > limits.common.max_prompt_tokens,
        "cumulative complete-document work is independent of a per-head context");
}

#[test]
fn invalid_or_unused_tokenizer_limits_and_compact_snapshot_saving_refuse_before_io() {
    for flag in ["--max-tokenizer-input-bytes", "--max-tokenizer-output-tokens",
        "--max-tokenizer-pair-checks", "--max-tokenizer-pair-bytes"] {
        let source = ["--source-dir", "d", "--source-manifest", "m", flag, "1"];
        if let Ok(command) = parsed(&source) {
            assert_eq!(command.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
        }
        for value in ["0", "18446744073709551615"] {
            let flags = ["--source-dir", "d", "--source-manifest", "m",
                "--input-encoding", "safe-bpe-v1", flag, value];
            if let Ok(command) = parsed(&flags) {
                assert_eq!(command.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
            }
        }
    }
    let compact = command(&["--input-encoding", "safe-bpe-v1", "--save-snapshot", "must-not-be-created"]);
    assert_eq!(compact.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    assert!(selected(&command(&["--save-snapshot", "byte.snapshot"]))
        .validate(&command(&["--save-snapshot", "byte.snapshot"]).args).is_ok());
}

#[test]
fn compact_map_help_explains_actual_tokens_and_the_complete_work_allowance() {
    let help = one_off_definition().render_long_help().to_string();
    let words = help.split_whitespace().collect::<Vec<_>>().join(" ");
    for required in ["--input-encoding", "--max-tokenizer-input-bytes", "--max-tokenizer-output-tokens",
        "--max-tokenizer-pair-checks", "--max-tokenizer-pair-bytes", "actual compact token counts",
        "accepted or rejected source-token probe", "All four tokenizer limits are cumulative",
        "One encoder workspace", "--save-snapshot is refused in compact mode"] {
        assert!(words.contains(required), "missing {required}");
    }
}
