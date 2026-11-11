//! Complete-document CLI admission and strict request regressions.
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
    let Int4Task::ClassifyMap(options) = command.task else { panic!("classification map required"); };
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
fn document_team_preserves_task_limits_and_admits_one_additional_worker_workspace() {
    let serial = command(&[]);
    let serial_options = selected(&serial);
    let base = serial_options.validate_with_threads(&serial.args).unwrap();
    for threads in [2, 3, 8, 64] {
        let width = threads.to_string();
        let command = command(&["--compute-threads", &width]);
        let options = selected(&command);
        let limits = options.validate_with_threads(&command.args).unwrap();
        assert_eq!(limits.common.compute.threads, threads);
        assert_eq!(limits.common.compute.workspace_bytes,
            crate::native_engine::strict_int4::projection::int4_team_workspace_bytes(threads, command.args.group_width).unwrap());
        assert_eq!(limits.memory, base.memory);
        assert_eq!(limits.mapping.chunks, base.mapping.chunks);
        assert_eq!(limits.mapping.max_model_work, base.mapping.max_model_work);
        assert_eq!(options.budget(limits.common, limits.planning.max_labels).unwrap(),
            serial_options.budget(base.common, base.planning.max_labels).unwrap());
        let execution = limits.memory.native.kv_bytes + limits.memory.input_bytes
            + limits.memory.workspace_bytes + limits.memory.output_reservation_bytes
            + limits.common.compute.workspace_bytes;
        assert!(limits.common.minimum_memory_bytes >= limits.cli_preparation_bytes
            + limits.common.load.weight_payload_bytes + ALLOCATOR_BYTES + execution + CLEANUP_BYTES);
        assert!(options.validate(&command.args).is_err(), "durable/shared callers remain explicit");
    }
}

#[test]
fn compact_document_teams_preserve_one_tokenizer_allowance_and_price_both_workspaces() {
    let flags = ["--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "262144",
        "--max-tokenizer-output-tokens", "131072", "--max-tokenizer-pair-checks", "33554432",
        "--max-tokenizer-pair-bytes", "268435456"];
    let serial = command(&flags); let serial_options = selected(&serial);
    let base = serial_options.validate(&serial.args).unwrap();
    let encoding = serial_options.input_encoding(&serial.args).unwrap();
    for threads in [2, 4] {
        let width = threads.to_string();
        let mut flags = flags.to_vec(); flags.extend(["--compute-threads", &width]);
        let parallel = command(&flags); let options = selected(&parallel);
        let limits = options.validate_with_threads(&parallel.args).unwrap();
        assert_eq!(options.input_encoding(&parallel.args).unwrap(), encoding);
        assert_eq!(limits.memory, base.memory);
        assert_eq!(limits.mapping.chunks, base.mapping.chunks);
        assert_eq!(limits.mapping.max_model_work, base.mapping.max_model_work);
        assert_eq!(options.budget(limits.common, limits.planning.max_labels).unwrap(),
            serial_options.budget(base.common, base.planning.max_labels).unwrap());
        let extra = crate::native_engine::strict_int4::projection::int4_team_workspace_bytes(
            threads, parallel.args.group_width).unwrap();
        assert_eq!(limits.common.compute.workspace_bytes, extra);
        assert_eq!(limits.common.minimum_memory_bytes, base.common.minimum_memory_bytes + extra);
        assert!(options.validate_with_max_labels(&parallel.args, 4).is_err());
    }
    for extra in [vec!["--cohort-chunks", "1"], vec!["--save-snapshot", "must-not-create.snapshot"]] {
        let mut flags = flags.to_vec(); flags.extend(["--compute-threads", "2"]); flags.extend(extra);
        assert_eq!(command(&flags).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    }
}

#[test]
fn invalid_document_team_and_cohort_composition_refuse_before_input_or_model_io() {
    for flags in [vec!["--compute-threads", "0"], vec!["--compute-threads", "65"],
        vec!["--compute-threads", "2", "--cohort-chunks", "1"]] {
        assert_eq!(command(&flags).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    }
    let mut short = command(&["--compute-threads", "2"]); short.args.memory_mib = 1;
    assert_eq!(short.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    let explicit_serial = command(&["--compute-threads", "1", "--cohort-chunks", "2"]);
    assert!(selected(&explicit_serial).validate_with_threads(&explicit_serial.args).is_ok());
    assert!(!corpus::definition().get_arguments().any(|arg| arg.get_id() == "compute_threads"));
}

#[test]
fn source_and_independently_pinned_snapshot_select_the_complete_document_route() {
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
        vec!["--snapshot", "s"], vec!["--expected-model-sha256", &hash],
        vec!["--model", "int8.fnlpq"],
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
fn complete_wire_preserves_unicode_controls_labels_policy_and_independent_decision_modes() {
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
fn wire_cannot_supply_chunk_scores_or_override_execution_and_aggregation_authority() {
    let command = command(&[]); let options = selected(&command);
    let limits = options.validate(&command.args).unwrap();
    let budget = options.budget(limits.common, limits.planning.max_labels).unwrap();
    let valid: serde_json::Value = serde_json::from_str(&request_json("exclusive")).unwrap();
    for mode in ["exclusive", "multi_label"] {
        for field in ["chunks", "scores", "aggregate", "model", "budget", "work", "seed", "score_space",
            "minimum_classified_coverage_ppm", "prompt", "scoring_mode", "eos_token_id"] {
            let mut value = valid.clone(); value["mode"] = serde_json::json!(mode);
            value[field] = serde_json::json!({});
            assert_eq!(classify::parse_request(&value.to_string(), budget, 4096).err(), Some(CandidateError::Input));
        }
    }
    for invalid in [r#"{"document":"a","document":"b","labels":[{"id":"a"},{"id":"b"}]}"#,
        r#"{"document":"a","labels":[{"id":"a"},{"id":"a"}]}"#,
        r#"{"document":"a","labels":[{"id":"a"}]}"#,
        r#"{"document":"a","labels":[]}"#, r#"{"document":"","labels":[{"id":"a"},{"id":"b"}]}"#,
        r#"{"document":"a","labels":[{"id":"a"},{"id":"b"}],"mode":"sampled"}"#,
        r#"{"document":"a","labels":[{"id":"a"},{"id":"b"}],"policy":{"minimum_candidate_weight_ppm":1000001,"minimum_margin_ppm":0}}"#] {
        assert_eq!(classify::parse_request(invalid, budget, 4096).err(), Some(CandidateError::Input));
    }
    let single = r#"{"document":"a","labels":[{"id":"a"}],"mode":"multi_label"}"#;
    assert_eq!(classify::parse_request(single, budget, 4096).unwrap().labels.len(), 1);
}

#[test]
fn generation_and_unwired_scheduling_flags_refuse_even_at_their_default_values() {
    let seed = "ab".repeat(32);
    for flags in [vec!["--prefill-rows", "1"], vec!["--cohort-heads", "1"],
        vec!["--max-new-tokens", "128"], vec!["--max-output-bytes", "65536"],
        vec!["--seed", &seed], vec!["--logprobs"], vec!["--min-new-tokens", "0"],
        vec!["--temperature-milli", "1000"], vec!["--top-k", "20"], vec!["--top-p-ppm", "1000000"],
        vec!["--stop", "END"], vec!["--ban-token", "7"], vec!["--logit-bias", "7=0"],
        vec!["--repetition-penalty-milli", "1000"], vec!["--presence-penalty-milli", "0"],
        vec!["--frequency-penalty-milli", "0"], vec!["--mode", "sampled"], vec!["--prompt", "private"],
        vec!["--focus-area", "reviews"], vec!["--source-membership"], vec!["--max-mask-node-visits", "1"]] {
        let mut all = vec!["--source-dir", "d", "--source-manifest", "m"];
        all.extend(flags); assert!(parsed(&all).is_err());
    }
    let mut command = command(&[]); command.args.logprobs = true;
    assert_eq!(command.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
}

#[test]
fn invalid_complete_document_geometry_and_every_native_work_limit_refuse_before_io() {
    for flags in [vec!["--max-chunks", "0"], vec!["--max-chunks", "257"],
        vec!["--max-chunk-bytes", "3"], vec!["--max-chunk-bytes", "1048577"],
        vec!["--max-input-bytes", "3"], vec!["--max-tokenizer-calls", "0"],
        vec!["--max-tokenizer-calls", "1000001"], vec!["--max-map-result-bytes", "0"],
        vec!["--max-map-result-bytes", "67108865"], vec!["--max-retained-result-bytes", "0"],
        vec!["--max-retained-result-bytes", "67108865"],
        vec!["--max-map-result-bytes", "2000000", "--max-retained-result-bytes", "1500000"],
        vec!["--max-map-result-bytes", "1", "--max-retained-result-bytes", "1000"],
        vec!["--minimum-classified-coverage-ppm", "1000001"],
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

#[test]
fn one_cache_is_admitted_for_every_chunk_and_all_work_axes_keep_the_document_ceiling() {
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
fn actual_mode_head_geometry_is_strict_and_never_silently_reduces_a_cohort() {
    let wire = |mode, labels: usize| serde_json::json!({"document":"short source", "mode":mode,
        "labels":(0..labels).map(|index| serde_json::json!({"id":format!("label-{index}")})).collect::<Vec<_>>()});
    for chunks in [1, 2, 3, 16, 32, 64] {
        let value = chunks.to_string();
        let command = command(&["--cohort-chunks", &value, "--max-result-bytes", "1048512"]);
        let options = selected(&command); let limits = options.validate(&command.args).unwrap();
        let budget = options.budget(limits.common, limits.planning.max_labels).unwrap();
        // An exclusive codebook has one native head, even with 128 labels.
        let exclusive = classify::parse_request(&wire("exclusive", 128).to_string(), budget, MAX_INPUT_BYTES).unwrap();
        options.admit_request(&exclusive).unwrap();
        let accepted = classify::parse_request(&wire("multi_label", 64 / chunks).to_string(), budget, MAX_INPUT_BYTES).unwrap();
        options.admit_request(&accepted).unwrap();
        let refused = classify::parse_request(&wire("multi_label", 64 / chunks + 1).to_string(), budget, MAX_INPUT_BYTES).unwrap();
        assert_eq!(options.admit_request(&refused), Err(CandidateError::Planning));
        assert_eq!(options.mapping.cohort_chunks, Some(chunks));
    }
    let serial = command(&[]); let options = selected(&serial); let limits = options.validate(&serial.args).unwrap();
    let budget = options.budget(limits.common, limits.planning.max_labels).unwrap();
    let request = classify::parse_request(&wire("multi_label", 128).to_string(), budget, MAX_INPUT_BYTES).unwrap();
    options.admit_request(&request).unwrap();
    let mut malformed = request.clone(); malformed.labels.clear();
    assert_eq!(options.admit_request(&malformed), Err(CandidateError::Planning));
    let mut malformed = request.clone(); malformed.labels.push(request.labels[0].clone());
    for mode in [ClassificationMode::Exclusive, ClassificationMode::MultiLabel] {
        malformed.mode = mode;
        assert_eq!(options.admit_request(&malformed), Err(CandidateError::Planning));
    }
    malformed.labels.truncate(1); malformed.mode = ClassificationMode::Exclusive;
    assert_eq!(options.admit_request(&malformed), Err(CandidateError::Planning));
}

#[test]
fn serial_and_cohort_tasks_share_semantic_authority_with_separate_physical_quotes() {
    let serial = command(&["--max-result-bytes", "1048512"]);
    let options = selected(&serial); let a = options.validate(&serial.args).unwrap();
    let task = options.budget(a.common, a.planning.max_labels).unwrap();
    for chunks in [1, 3, 16, 64] {
        let value = chunks.to_string();
        let cohort = command(&["--cohort-chunks", &value, "--max-result-bytes", "1048512"]);
        let options = selected(&cohort); let b = options.validate(&cohort.args).unwrap();
        assert_eq!(task, options.budget(b.common, b.planning.max_labels).unwrap());
        assert_eq!(a.common.native, b.common.native, "the logical per-head basis must not grow with the physical schedule");
        assert_eq!(a.mapping.chunks, b.mapping.chunks);
        assert_eq!(a.mapping.aggregation, b.mapping.aggregation);
        assert_eq!(a.mapping.max_model_work, b.mapping.max_model_work);
        assert_eq!(a.mapping.max_result_bytes, b.mapping.max_result_bytes);
        assert_eq!(a.mapping.max_retained_result_bytes, b.mapping.max_retained_result_bytes);
        assert_eq!(a.planning.max_work, b.planning.max_work);
        let per_chunk_heads = b.planning.max_labels.min(options.cohort_head_limit().unwrap());
        let quote = Int4ClassificationMapMemoryLimits { context_tokens: cohort.args.context_tokens,
            preparation_reserve_bytes: b.common.preparation_bytes, allocator_reserve_bytes: ALLOCATOR_BYTES }
            .cohort_requirements(cohort.args.group_width, task, b.planning, b.mapping, per_chunk_heads, chunks).unwrap();
        let contexts = [cohort.args.context_tokens; MAX_INT4_SEQUENCES];
        assert_eq!(quote.memory, b.memory);
        assert_eq!(b.memory.native, Int4MemoryRequirement::for_cohort(
            &contexts[..per_chunk_heads * chunks], cohort.args.group_width).unwrap());
        assert!(b.memory.native.kv_bytes > b.common.native.kv_bytes);
        assert!(b.memory.input_bytes >= a.memory.input_bytes + quote.task_workspace_bytes);
        assert_eq!(b.envelope_bytes as u64, b.memory.max_output_bytes + 4096);
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
fn memory_floor_covers_complete_plans_results_and_outer_publication_on_each_model_route() {
    let hash = "ab".repeat(32);
    for flags in [vec![], vec!["--cohort-chunks", "1"], vec!["--cohort-chunks", "3"],
        vec!["--cohort-chunks", "16"], vec!["--cohort-chunks", "64", "--max-result-bytes", "1048512"]] {
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
            let heads = options.mapping.cohort_chunks.map_or(1, |chunks|
                chunks * limits.planning.max_labels.min(MAX_INT4_SEQUENCES / chunks));
            assert_eq!(quote.native.kv_bytes, heads as u64 * common.native.kv_bytes);
            command.args.memory_mib = common.minimum_memory_bytes.div_ceil(MIB);
            options.validate(&command.args).unwrap();
            command.args.memory_mib -= 1;
            assert_eq!(command.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
        }
    }
}

#[test]
fn help_names_complete_document_semantics_without_advertising_unwired_schedules() {
    let help = definition().render_long_help().to_string();
    let words = help.split_whitespace().collect::<Vec<_>>().join(" ");
    for required in ["classify-map", "--minimum-classified-coverage-ppm", "--max-chunks", "--max-map-result-bytes",
        "full-vocabulary", "independent yes/no", "uncalibrated", "before input IO", "--cohort-chunks",
        "before model IO", "before the first native allocation", "optional snapshot save"] {
        assert!(words.contains(required), "missing {required}");
    }
    assert!(help.lines().any(|line| line.trim_start().starts_with("--cohort-chunks")));
    for hidden in ["--prefill-rows", "--cohort-heads"] {
        assert!(!help.lines().any(|line| line.trim_start().starts_with(hidden)));
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_refuses_without_touching_input_model_or_output() {
    assert_eq!(command(&[]).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Unavailable));
    assert_eq!(command(&["--input-encoding", "safe-bpe-v1"])
        .execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Unavailable));
}

#[test]
fn compact_map_policy_is_explicit_bounded_and_charged_beside_every_chunk_plan() {
    for schedule in [vec![], vec!["--cohort-chunks", "2"]] {
        let byte = command(&schedule); let baseline = selected(&byte).validate(&byte.args).unwrap();
        let mut flags = schedule;
        flags.extend(["--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "262144",
            "--max-tokenizer-output-tokens", "131072", "--max-tokenizer-pair-checks", "33554432",
            "--max-tokenizer-pair-bytes", "268435456"]);
        let compact = command(&flags); let options = selected(&compact);
        let limits = options.validate(&compact.args).unwrap();
        let expected = SafeBpeLimits { max_input_bytes: 262144, max_output_tokens: 131072,
            max_pair_checks: 33554432, max_pair_bytes: 268435456 };
        assert_eq!(options.input_encoding(&compact.args).unwrap(), UntrustedEncoding::SafeBpeV1(expected));
        assert_eq!(limits.memory.input_bytes - baseline.memory.input_bytes, expected.requirements().unwrap().workspace_bytes);
        assert_eq!(limits.memory.native, baseline.memory.native);
        assert_eq!(limits.memory.workspace_bytes, baseline.memory.workspace_bytes);
        assert_eq!(limits.memory.output_reservation_bytes, baseline.memory.output_reservation_bytes);
        assert_eq!(limits.mapping.chunks, baseline.mapping.chunks);
        assert_eq!(limits.mapping.max_model_work, baseline.mapping.max_model_work);
        assert_eq!(options.budget(limits.common, limits.planning.max_labels).unwrap(),
            selected(&byte).budget(baseline.common, baseline.planning.max_labels).unwrap());
    }
    let hash = "ab".repeat(32);
    let snapshot = parsed(&["--snapshot", "explicit.snapshot", "--expected-model-sha256", &hash,
        "--input-encoding", "safe-bpe-v1"]).unwrap();
    let options = selected(&snapshot); let limits = options.validate(&snapshot.args).unwrap();
    assert!(limits.common.snapshot_pin.is_some());
    let UntrustedEncoding::SafeBpeV1(policy) = options.input_encoding(&snapshot.args).unwrap() else {
        panic!("explicit compact policy required");
    };
    assert_eq!(policy.max_input_bytes, SafeBpeLimits::MAX_INPUT_BYTES);
    assert_eq!(policy.max_output_tokens, SafeBpeLimits::MAX_OUTPUT_TOKENS);
}

#[test]
fn compact_limits_and_repeated_presave_planning_refuse_before_private_io() {
    for flags in [vec!["--max-tokenizer-input-bytes", "100"],
        vec!["--input-encoding", "byte-fallback", "--max-tokenizer-output-tokens", "100"],
        vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "0"],
        vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "1048577"],
        vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-output-tokens", "0"],
        vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-output-tokens", "1048577"],
        vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "0"],
        vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "67108865"],
        vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-bytes", "0"],
        vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-bytes", "536870913"],
        vec!["--input-encoding", "safe-bpe-v1", "--save-snapshot", "must-not-create.snapshot"]] {
        assert_eq!(command(&flags).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    }
    for flag in ["--max-tokenizer-pair-checks", "--max-tokenizer-pair-bytes"] {
        assert!(parsed(&["--source-dir", "d", "--source-manifest", "m", flag, "1"]).is_err());
    }
}

#[test]
fn corpus_and_retained_jobs_do_not_inherit_unpriced_compact_map_options() {
    for mut argv in [vec!["candidate", "source-int4", "classify-corpus"],
        vec!["candidate", "source-int4", "classify-document-job", "start"],
        vec!["candidate", "source-int4", "classify-document-job", "resume"]] {
        argv.extend(["--input-encoding", "safe-bpe-v1"]);
        assert!(crate::candidate_cli::definition().try_get_matches_from(argv).is_err());
    }
    let compact = command(&["--input-encoding", "safe-bpe-v1"]);
    assert_eq!(selected(&compact).validate_with_max_labels(&compact.args, 4).err(), Some(CandidateError::Arguments));
    let help = one_off_definition().render_long_help().to_string();
    let words = help.split_whitespace().collect::<Vec<_>>().join(" ");
    for required in ["--input-encoding", "--max-tokenizer-input-bytes", "--max-tokenizer-output-tokens",
        "oversized probes", "one shared tokenizer allowance", "refuse --save-snapshot"] {
        assert!(words.contains(required), "missing {required}");
    }
}
