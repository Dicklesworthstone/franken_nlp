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
    let hash = "ab".repeat(32);
    let snapshot = parsed(&["--snapshot", "explicit.snapshot", "--expected-model-sha256", &hash]).unwrap();
    let restored = selected(&snapshot).validate(&snapshot.args).unwrap();
    assert_eq!(restored.common.snapshot_pin.unwrap().model_content_sha256, [0xab; 32]);
    assert_eq!(options.budget(limits.common), selected(&snapshot).budget(restored.common));
    for invalid in [vec![], vec!["--snapshot", "s"], vec!["--model", "int8.fnlpq"],
        vec!["--snapshot", "s", "--expected-model-sha256", &hash, "--source-dir", "d"],
        vec!["--snapshot", "s", "--expected-model-sha256", &hash, "--save-snapshot", "new"]] {
        assert!(parsed(&invalid).is_err());
    }
}

#[test]
fn map_preserves_the_existing_strict_classification_wire_for_both_modes() {
    let command = command(&[]); let options = selected(&command);
    let budget = options.budget(options.validate(&command.args).unwrap().common);
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
    for field in ["chunks", "scores", "aggregate", "model", "budget", "work", "seed",
        "minimum_classified_coverage_ppm", "prompt", "scoring_mode", "eos_token_id"] {
        let mut invalid = wire.clone(); invalid[field] = serde_json::json!({});
        assert_eq!(classify::parse_request(&invalid.to_string(), budget, 4096).err(), Some(CandidateError::Input));
    }
    for invalid in [r#"{"document":"a","document":"b","labels":[{"id":"x"},{"id":"y"}]}"#,
        r#"{"document":"a","labels":[{"id":"x"},{"id":"x"}]}"#,
        r#"{"document":"a","labels":[]}"#, r#"{"document":"","labels":[{"id":"x"},{"id":"y"}]}"#,
        r#"{"document":"a","labels":[{"id":"x"}]}"#,
        r#"{"document":"a","labels":[{"id":"x"},{"id":"y"}],"policy":{"minimum_candidate_weight_ppm":1000001,"minimum_margin_ppm":0}}"#] {
        assert_eq!(classify::parse_request(invalid, budget, 4096).err(), Some(CandidateError::Input));
    }
}

#[test]
fn generation_and_single_request_schedules_refuse_even_at_their_defaults() {
    for flags in [vec!["--prefill-rows", "1"], vec!["--cohort-heads", "1"], vec!["--cohort-chunks", "1"],
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
fn impossible_document_geometry_coverage_results_and_work_refuse_before_io() {
    for flags in [vec!["--max-chunks", "0"], vec!["--max-chunks", "257"],
        vec!["--max-chunk-bytes", "3"], vec!["--max-chunk-bytes", "1048577"],
        vec!["--max-tokenizer-calls", "0"], vec!["--max-tokenizer-calls", "1000001"],
        vec!["--max-map-result-bytes", "0"], vec!["--max-map-result-bytes", "67108865"],
        vec!["--max-retained-result-bytes", "0"], vec!["--max-retained-result-bytes", "67108865"],
        vec!["--max-map-result-bytes", "2", "--max-retained-result-bytes", "1"],
        vec!["--minimum-classified-coverage-ppm", "1000001"], vec!["--max-input-bytes", "3"],
        vec!["--max-forward-positions", "0"], vec!["--max-projected-logits", "0"],
        vec!["--max-attention-pairs", "0"], vec!["--max-dot-products", "0"],
        vec!["--max-multiply-accumulates", "0"], vec!["--context-tokens", "16"],
        vec!["--preparation-mib", "1"]] {
        assert_eq!(command(&flags).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    }
    for minimum in ["0", "1", "500000", "1000000"] {
        let command = command(&["--minimum-classified-coverage-ppm", minimum]);
        let limits = selected(&command).validate(&command.args).unwrap();
        assert_eq!(limits.mapping.aggregation.minimum_classified_coverage_ppm, minimum.parse::<u32>().unwrap());
    }
}

#[test]
fn complete_memory_quote_charges_the_host_peak_and_outer_publication() {
    let mut command = command(&[]); let options = selected(&command);
    let limits = options.validate(&command.args).unwrap();
    let common = limits.common; let quote = limits.memory;
    let resident = common.load.weight_payload_bytes + ALLOCATOR_BYTES;
    let loading = common.source_preparation_bytes + common.load.scratch_payload_bytes;
    let execution = quote.native.kv_bytes + quote.input_bytes + quote.workspace_bytes + quote.output_reservation_bytes;
    assert_eq!(common.minimum_memory_bytes,
        limits.cli_preparation_bytes + resident + loading.max(execution) + CLEANUP_BYTES);
    assert_eq!(limits.cli_preparation_bytes, common.preparation_bytes + 2 * limits.envelope_bytes as u64);
    assert_eq!(limits.envelope_bytes as u64, quote.max_output_bytes + 4096);
    assert_eq!(limits.mapping.max_model_work, options.scoring.work_ceiling());
    assert_eq!(options.budget(common).max_kv_bytes, common.native.kv_bytes);
    command.args.memory_mib = common.minimum_memory_bytes.div_ceil(MIB);
    options.validate(&command.args).unwrap();
    command.args.memory_mib -= 1;
    assert_eq!(command.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_refuses_without_input_or_model_access() {
    assert_eq!(command(&[]).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Unavailable));
}
