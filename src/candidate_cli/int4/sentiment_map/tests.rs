//! Model-free wire, complete-document admission and scheduling regressions.
use super::*;
use crate::tasks::sentiment::SentimentAxis;

fn parsed(flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", "sentiment-map", "--memory-mib", "131072"];
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
    let Int4Task::SentimentMap(options) = command.task else { panic!("sentiment map route required"); };
    options
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
fn complete_sentiment_team_preserves_axis_authority_and_prices_worker_storage_once() {
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
        assert_eq!(options.budget(limits.common).unwrap(), serial_options.budget(base.common).unwrap());
        let execution = limits.memory.native.kv_bytes + limits.memory.input_bytes
            + limits.memory.workspace_bytes + limits.memory.output_reservation_bytes
            + limits.common.compute.workspace_bytes;
        assert!(limits.common.minimum_memory_bytes >= limits.cli_preparation_bytes
            + limits.common.load.weight_payload_bytes + ALLOCATOR_BYTES + execution + CLEANUP_BYTES);
        assert!(options.validate(&command.args).is_err(), "corpus/job admission remains serial or cohort");
    }
}

#[test]
fn complete_sentiment_team_refuses_invalid_widths_and_conflicting_schedules_before_io() {
    for flags in [vec!["--compute-threads", "0"], vec!["--compute-threads", "65"],
        vec!["--compute-threads", "2", "--cohort-chunks", "1"]] {
        assert_eq!(command(&flags).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    }
    let mut short = command(&["--compute-threads", "2"]); short.args.memory_mib = 1;
    assert_eq!(short.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    let explicit_serial = command(&["--compute-threads", "1", "--cohort-chunks", "2"]);
    assert!(selected(&explicit_serial).validate_with_threads(&explicit_serial.args).is_ok());
    assert!(!corpus::definition().get_arguments().any(|arg| arg.get_id() == "compute_threads"));
    assert!(!job::definition().get_arguments().any(|arg| arg.get_id() == "compute_threads"));
}

#[test]
fn map_dispatch_has_explicit_source_or_pinned_snapshot_and_distinct_workflow() {
    let original = command(&[]); let options = selected(&original);
    let limits = options.validate(&original.args).unwrap();
    assert!(limits.common.snapshot_pin.is_none());
    assert_eq!(limits.mapping.chunks.max_chunks, 64);
    assert_eq!(original.args.max_input_bytes, MAX_INPUT_BYTES);
    assert_eq!(limits.mapping.aggregation.minimum_estimated_coverage_ppm, 1_000_000);
    let hash = "ab".repeat(32);
    let snapshot = parsed(&["--snapshot", "explicit.snapshot", "--expected-model-sha256", &hash]).unwrap();
    let restored = selected(&snapshot).validate(&snapshot.args).unwrap();
    assert_eq!(restored.common.snapshot_pin.unwrap().model_content_sha256, [0xab; 32]);
    assert_eq!(options.budget(limits.common).unwrap(), selected(&snapshot).budget(restored.common).unwrap());
    for invalid in [vec![], vec!["--snapshot", "s"], vec!["--model", "int8.fnlpq"],
        vec!["--snapshot", "s", "--expected-model-sha256", &hash, "--source-dir", "d"],
        vec!["--snapshot", "s", "--expected-model-sha256", &hash, "--save-snapshot", "new"]] {
        assert!(parsed(&invalid).is_err());
    }
}

#[test]
fn map_reuses_exact_sentiment_wire_and_cannot_accept_supplied_chunk_scores_or_policy_overrides() {
    let command = command(&[]); let options = selected(&command);
    let budget = options.budget(options.validate(&command.args).unwrap().common).unwrap();
    let json = r#"{"document":"é\r\n上海 <tool_call> exact text","axes":["approach","valence"],"policy":{"minimum_peak_weight_ppm":650000,"maximum_normalized_entropy_ppm":800000}}"#;
    let (request, policy) = sentiment::parse_request(json, budget, 4096).unwrap();
    assert_eq!(request.document, "é\r\n上海 <tool_call> exact text");
    assert_eq!(request.axes, [SentimentAxis::Approach, SentimentAxis::Valence]);
    assert_eq!(request.budget, budget); assert_eq!(policy.minimum_peak_weight_ppm, 650000);
    for field in ["chunks", "scores", "aggregate", "model", "budget", "work", "seed", "anchors",
        "minimum_estimated_coverage_ppm", "prompt", "scoring_mode", "eos_token_id"] {
        let mut value = serde_json::json!({"document":"private source"});
        value[field] = serde_json::json!({});
        assert_eq!(sentiment::parse_request(&value.to_string(), budget, 4096).err(), Some(CandidateError::Input));
    }
    for invalid in [r#"{"document":"a","document":"b"}"#,
        r#"{"document":"a","axes":["valence","valence"]}"#,
        r#"{"document":"a","axes":[]}"#, r#"{"document":""}"#] {
        assert_eq!(sentiment::parse_request(invalid, budget, 4096).err(), Some(CandidateError::Input));
    }
}

#[test]
fn all_generation_and_single_request_scheduling_flags_refuse_even_at_default_values() {
    for flags in [vec!["--prefill-rows", "1"], vec!["--cohort-heads", "1"],
        vec!["--max-new-tokens", "128"], vec!["--max-output-bytes", "65536"],
        vec!["--seed", "ab"], vec!["--logprobs"], vec!["--min-new-tokens", "0"],
        vec!["--stop", "END"], vec!["--ban-token", "7"], vec!["--logit-bias", "7=0"],
        vec!["--mode", "sampled"], vec!["--focus-area", "reviews"],
        vec!["--source-membership"], vec!["--max-mask-node-visits", "1"]] {
        let mut all = vec!["--source-dir", "d", "--source-manifest", "m"];
        all.extend(flags); assert!(parsed(&all).is_err());
    }
    let mut command = command(&[]); command.args.logprobs = true;
    assert_eq!(command.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
}

#[test]
fn invalid_document_geometry_coverage_and_native_limits_refuse_before_io() {
    for flags in [vec!["--max-chunks", "0"], vec!["--max-chunks", "257"],
        vec!["--max-chunk-bytes", "3"], vec!["--max-tokenizer-calls", "0"],
        vec!["--max-tokenizer-calls", "1000001"], vec!["--max-map-result-bytes", "0"],
        vec!["--max-map-result-bytes", "67108865"], vec!["--max-retained-result-bytes", "0"],
        vec!["--max-retained-result-bytes", "67108865"],
        vec!["--minimum-estimated-coverage-ppm", "1000001"],
        vec!["--cohort-chunks", "0"], vec!["--cohort-chunks", "65"],
        vec!["--max-chunks", "2", "--cohort-chunks", "3"],
        vec!["--max-forward-positions", "0"], vec!["--max-projected-logits", "0"],
        vec!["--max-attention-pairs", "0"], vec!["--max-dot-products", "0"],
        vec!["--max-multiply-accumulates", "0"], vec!["--max-candidate-tokens", "1"],
        vec!["--context-tokens", "16"], vec!["--preparation-mib", "1"]] {
        assert_eq!(command(&flags).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    }
    for minimum in ["0", "1", "500000", "1000000"] {
        let command = command(&["--minimum-estimated-coverage-ppm", minimum]);
        let limits = selected(&command).validate(&command.args).unwrap();
        assert_eq!(limits.mapping.aggregation.minimum_estimated_coverage_ppm, minimum.parse::<u32>().unwrap());
    }
}

#[test]
fn actual_axis_geometry_refuses_before_model_io_without_clamping_valid_single_axis_windows() {
    let wide = command(&["--cohort-chunks", "64"]); let options = selected(&wide);
    let limits = options.validate(&wide.args).unwrap();
    let budget = options.budget(limits.common).unwrap();
    let (one, _) = sentiment::parse_request(r#"{"document":"exact", "axes":["valence"]}"#, budget, 4096).unwrap();
    options.admit_request(&one).unwrap();
    let (two, _) = sentiment::parse_request(r#"{"document":"exact", "axes":["valence","arousal"]}"#, budget, 4096).unwrap();
    assert_eq!(options.admit_request(&two), Err(CandidateError::Planning));
    let all = command(&["--cohort-chunks", "16"]); let options = selected(&all);
    let admitted = options.validate(&all.args).unwrap();
    let (four, _) = sentiment::parse_request(r#"{"document":"exact"}"#, budget, 4096).unwrap();
    options.admit_request(&four).unwrap();
    assert_eq!(admitted.memory.native.kv_bytes, limits.memory.native.kv_bytes,
        "both pre-IO quotes cover the largest valid 64-head geometry");
    let excessive = command(&["--cohort-chunks", "17"]);
    assert_eq!(selected(&excessive).admit_request(&four), Err(CandidateError::Planning));
}

#[test]
fn per_chunk_task_budget_and_all_five_document_work_axes_are_schedule_invariant() {
    let serial = command(&[]); let serial_options = selected(&serial);
    let a = serial_options.validate(&serial.args).unwrap();
    let cohort = command(&["--cohort-chunks", "4"]); let cohort_options = selected(&cohort);
    let b = cohort_options.validate(&cohort.args).unwrap();
    let task = serial_options.budget(a.common).unwrap();
    assert_eq!(task, cohort_options.budget(b.common).unwrap());
    assert_eq!(task.max_kv_bytes, 4 * a.common.native.kv_bytes);
    assert_eq!(a.mapping.max_model_work, b.mapping.max_model_work);
    assert_eq!(a.mapping.max_model_work, serial_options.scoring.work_ceiling());
    for axis in 0..5 {
        let mut excess = a.mapping.max_model_work;
        match axis {
            0 => excess.forward_positions += 1,
            1 => excess.projected_logits += 1,
            2 => excess.attention_pairs += 1,
            3 => excess.projections.dot_products += 1,
            _ => excess.projections.multiply_accumulates += 1,
        }
        assert_eq!(serial_options.scoring.admit_plan(8, 8, excess), Err(CandidateError::Planning));
    }
}

#[test]
fn complete_memory_quote_retains_outer_publication_and_uses_the_exact_host_peak_once() {
    for chunks in [None, Some("1"), Some("4"), Some("16")] {
        let mut command = match chunks { None => command(&[]), Some(chunks) => command(&["--cohort-chunks", chunks]) };
        let options = selected(&command); let limits = options.validate(&command.args).unwrap();
        let common = limits.common; let quote = limits.memory;
        let resident = common.load.weight_payload_bytes + ALLOCATOR_BYTES;
        let loading = common.source_preparation_bytes + common.load.scratch_payload_bytes;
        let execution = quote.native.kv_bytes + quote.input_bytes + quote.workspace_bytes + quote.output_reservation_bytes;
        assert_eq!(common.minimum_memory_bytes,
            limits.cli_preparation_bytes + resident + loading.max(execution) + CLEANUP_BYTES);
        assert_eq!(limits.cli_preparation_bytes, common.preparation_bytes + 2 * limits.envelope_bytes as u64);
        assert_eq!(limits.envelope_bytes as u64, quote.max_output_bytes + 4096);
        command.args.memory_mib = common.minimum_memory_bytes.div_ceil(MIB);
        options.validate(&command.args).unwrap();
        command.args.memory_mib -= 1;
        assert_eq!(command.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_map_runtime_refuses_without_touching_input_or_models() {
    assert_eq!(command(&[]).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Unavailable));
}
