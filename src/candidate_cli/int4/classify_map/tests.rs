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
    for field in ["chunks", "scores", "aggregate", "model", "budget", "work", "seed", "score_space",
        "minimum_classified_coverage_ppm", "prompt", "scoring_mode", "eos_token_id"] {
        let mut value = valid.clone(); value[field] = serde_json::json!({});
        assert_eq!(classify::parse_request(&value.to_string(), budget, 4096).err(), Some(CandidateError::Input));
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
    for flags in [vec!["--prefill-rows", "1"], vec!["--cohort-heads", "1"], vec!["--cohort-chunks", "1"],
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
fn memory_floor_covers_complete_plans_results_and_outer_publication_on_each_model_route() {
    let hash = "ab".repeat(32);
    let commands = [command(&[]), command(&["--save-snapshot", "new.snapshot"]),
        parsed(&["--snapshot", "existing.snapshot", "--expected-model-sha256", &hash]).unwrap()];
    for mut command in commands {
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
        command.args.memory_mib = common.minimum_memory_bytes.div_ceil(MIB);
        options.validate(&command.args).unwrap();
        command.args.memory_mib -= 1;
        assert_eq!(command.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    }
}

#[test]
fn help_names_complete_document_semantics_without_advertising_unwired_schedules() {
    let help = definition().render_long_help().to_string();
    let words = help.split_whitespace().collect::<Vec<_>>().join(" ");
    for required in ["classify-map", "--minimum-classified-coverage-ppm", "--max-chunks", "--max-map-result-bytes",
        "full-vocabulary", "independent yes/no", "uncalibrated", "before input IO"] {
        assert!(words.contains(required), "missing {required}");
    }
    assert!(!help.lines().any(|line| line.trim_start().starts_with("--cohort-chunks")));
    assert!(!help.lines().any(|line| line.trim_start().starts_with("--prefill-rows")));
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_refuses_without_touching_input_model_or_output() {
    assert_eq!(command(&[]).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Unavailable));
}
