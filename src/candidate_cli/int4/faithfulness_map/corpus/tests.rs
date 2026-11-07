//! Command-boundary regressions; these fixtures perform no model inference.
use super::*;

fn parsed(flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", "faithfulness-corpus", "--memory-mib", "131072"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("source INT4 route required"); };
    Ok(command)
}
fn command(flags: &[&str]) -> Int4Command {
    let mut all = vec!["--source-dir", "explicit-source", "--source-manifest", "explicit-manifest",
        "--policy", "private-policy.json"];
    all.extend_from_slice(flags);
    parsed(&all).unwrap()
}
pub(super) fn arguments(flags: &[&str]) -> (CorpusArgs, Int4Args) {
    let command = command(flags);
    let Int4Task::FaithfulnessCorpus(options) = command.task else { panic!("faithfulness corpus required"); };
    (options, command.args)
}
struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused corpus read input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused corpus wrote output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused corpus flushed output"); }
}

pub(super) const POLICY: &str = r#"{"minimum_candidate_weight_ppm":700000,"minimum_margin_milli":100,"evidence_window_bytes":256,"max_evidence_windows":31,"max_evidence_spans":31}"#;

#[test]
fn policy_is_explicit_strict_and_cannot_smuggle_a_claim_or_execution_authority() {
    let policy = parse_policy(POLICY).unwrap();
    assert_eq!(policy.minimum_candidate_weight_ppm, 700_000);
    assert_eq!(policy.evidence_window_bytes, 256);
    assert_eq!(policy.max_evidence_windows, 31);
    assert!(parsed(&["--source-dir", "source", "--source-manifest", "manifest"]).is_err());
    for extra in [r#", "claim":"private""#, r#", "source":"private""#,
        r#", "budget":{}"#, r#", "policy":{}"#, r#", "minimum_margin_milli":1"#] {
        let malformed = format!("{}{extra}}}", &POLICY[..POLICY.len() - 1]);
        assert_eq!(parse_policy(&malformed), Err(CandidateError::Input));
    }
    for invalid in ["null", "[]", "{}", r#"{"policy":{}}"#] {
        assert_eq!(parse_policy(invalid), Err(CandidateError::Input));
    }
    for (from, to) in [("700000", "1000001"), ("256", "0"), ("31", "32")] {
        assert_eq!(parse_policy(&POLICY.replace(from, to)), Err(CandidateError::Planning));
    }
    assert_eq!(parse_policy(&" ".repeat(POLICY_BYTES + 1)), Err(CandidateError::Input));
    let duplicate = POLICY.replace("\"minimum_margin_milli\":100",
        "\"minimum_margin_milli\":100,\"minimum_margin_milli\":100");
    assert_eq!(parse_policy(&duplicate), Err(CandidateError::Input));
}

#[test]
fn source_and_snapshot_modes_share_the_owned_complete_source_contract() {
    assert_eq!(command(&[]).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    let (options, args) = arguments(&["--cohort-chunks", "3"]);
    let source = options.validate(&args).unwrap();
    assert_eq!(format!("{options:?}"), "FaithfulnessCorpusArgs");
    let pin = "ab".repeat(32);
    let snapshot = parsed(&["--snapshot", "private.snapshot", "--expected-model-sha256", &pin,
        "--policy", "private-policy.json", "--cohort-chunks", "3"]).unwrap();
    let Int4Task::FaithfulnessCorpus(restored) = snapshot.task else { panic!("snapshot corpus required"); };
    let loaded = restored.validate(&snapshot.args).unwrap();
    assert!(source.common.snapshot_pin.is_none());
    assert!(loaded.common.snapshot_pin.is_some());
    assert_eq!(source.transport, loaded.transport);
    assert_eq!(source.document.mapping.chunks, loaded.document.mapping.chunks);
    assert_eq!(source.document.mapping.max_model_work, loaded.document.mapping.max_model_work);
    assert_eq!(source.document.mapping.aggregation, loaded.document.mapping.aggregation);
    assert_eq!(source.document.mapping.max_result_bytes, loaded.document.mapping.max_result_bytes);
    assert_eq!(source.document.mapping.max_retained_result_bytes, loaded.document.mapping.max_retained_result_bytes);
    assert_eq!(source.document.memory, loaded.document.memory);
    assert_eq!(options.document.budget(source.common).unwrap(), restored.document.budget(loaded.common).unwrap());
    for flags in [vec!["--policy", "policy"],
        vec!["--snapshot", "snapshot", "--policy", "policy"],
        vec!["--snapshot", "snapshot", "--expected-model-sha256", &pin, "--policy", "policy", "--source-dir", "source"],
        vec!["--snapshot", "snapshot", "--expected-model-sha256", &pin, "--policy", "policy", "--save-snapshot", "new"]] {
        assert!(parsed(&flags).is_err());
    }
}

#[test]
fn transport_capacity_and_cohorts_never_multiply_any_native_work_axis() {
    let (mut options, args) = arguments(&["--max-requests", "10"]);
    let serial = options.validate(&args).unwrap();
    options.stream.max_requests = 1000;
    options.document.mapping.cohort_chunks = Some(3);
    let grouped = options.validate(&args).unwrap();
    assert_eq!(serial.document.mapping.max_model_work, grouped.document.mapping.max_model_work);
    assert_eq!(grouped.document.mapping.max_model_work, options.document.scoring.work_ceiling());
    assert_eq!(serial.transport.max_work, grouped.transport.max_work);
    assert_eq!(options.document.budget(serial.common).unwrap(), options.document.budget(grouped.common).unwrap());
    for (count, limits) in [(10, serial), (1000, grouped)] {
        assert_eq!(limits.transport.max_output_bytes + (count + 4) * FRAME_ALLOWANCE, limits.output_bytes);
    }
}

#[test]
fn ragged_windows_quote_the_physical_bound_without_rejecting_useful_chunk_widths() {
    // A 64-chunk window also retains all chunk results and a cohort wrapper;
    // keep that explicit output bound below the map's 64 MiB window ceiling.
    let (options, args) = arguments(&["--max-result-bytes", "65536"]);
    let serial = options.validate(&args).unwrap();
    for chunks in [3, 16, 64] {
        let (mut options, args) = arguments(&["--max-result-bytes", "65536"]);
        options.document.mapping.cohort_chunks = Some(chunks);
        let grouped = options.validate(&args).unwrap();
        assert_eq!(grouped.document.memory.native.kv_bytes, serial.document.memory.native.kv_bytes * 64);
        assert_eq!(options.document.budget(grouped.common).unwrap().max_kv_bytes,
            serial.document.memory.native.kv_bytes * 32);
        assert_eq!(grouped.document.memory.max_output_bytes, serial.document.memory.max_output_bytes + 4096);
    }
}

#[test]
fn invalid_transport_native_work_and_geometry_refuse_before_io() {
    for flags in [vec!["--max-forward-positions", "0"], vec!["--max-projected-logits", "0"],
        vec!["--max-attention-pairs", "0"], vec!["--max-dot-products", "0"],
        vec!["--max-multiply-accumulates", "0"], vec!["--max-requests", "0"],
        vec!["--max-requests", "100001"], vec!["--max-total-input-bytes", "0"],
        vec!["--max-total-input-bytes", "17179869185"], vec!["--max-total-output-bytes", "1"],
        vec!["--max-total-output-bytes", "17179869185"], vec!["--max-line-bytes", "1048575"],
        vec!["--max-line-bytes", "4194305"], vec!["--cohort-chunks", "0"], vec!["--cohort-chunks", "65"],
        vec!["--max-chunks", "0"], vec!["--max-chunks", "257"],
        vec!["--max-chunks", "2", "--cohort-chunks", "3"], vec!["--max-chunk-bytes", "3"],
        vec!["--max-tokenizer-calls", "0"], vec!["--max-map-result-bytes", "0"],
        vec!["--max-retained-result-bytes", "0"], vec!["--minimum-classified-coverage-ppm", "1000001"],
        vec!["--timeout-seconds", "0"], vec!["--max-checkpoints", "1"], vec!["--preparation-mib", "1"]] {
        let (options, args) = arguments(&flags);
        assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments), "{flags:?}");
    }
    for policy in ["", "-"] {
        let (mut options, args) = arguments(&[]);
        options.stream.policy = PathBuf::from(policy);
        assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn corpus_prices_frozen_identities_transport_and_complete_results_before_input() {
    let (options, mut args) = arguments(&["--cohort-chunks", "3"]);
    let document = options.document.validate(&args).unwrap();
    let limits = options.validate(&args).unwrap();
    let buffers = corpus_buffer_bytes(limits.transport, limits.common.preparation_bytes, limits.io_bytes).unwrap();
    let metadata = retained_metadata_bytes(document.mapping.chunks.max_chunks).unwrap();
    assert_eq!(limits.common.minimum_memory_bytes,
        document.common.minimum_memory_bytes + buffers + metadata + limits.io_bytes);
    assert_eq!(limits.cli_preparation_bytes, document.cli_preparation_bytes + limits.io_bytes);
    assert_eq!(limits.transport.max_output_line_bytes as u64, document.memory.max_output_bytes + FRAME_ALLOWANCE);
    args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
    options.validate(&args).unwrap();
    args.memory_mib -= 1;
    assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn unsupported_prompt_sampling_and_cross_document_schedules_are_never_ignored() {
    for flags in [vec!["--prefill-rows", "1"], vec!["--cohort-heads", "1"],
        vec!["--cohort-documents", "1"], vec!["--max-new-tokens", "128"],
        vec!["--max-output-bytes", "65536"], vec!["--min-new-tokens", "0"],
        vec!["--logprobs"], vec!["--seed", "ab"], vec!["--task", "faithfulness"],
        vec!["--mode", "rubric"], vec!["--max-selected-rows", "128"], vec!["--defaults", "settings"]] {
        let mut all = vec!["--source-dir", "source", "--source-manifest", "manifest", "--policy", "policy"];
        all.extend_from_slice(&flags);
        assert!(parsed(&all).is_err(), "{flags:?}");
    }
}
