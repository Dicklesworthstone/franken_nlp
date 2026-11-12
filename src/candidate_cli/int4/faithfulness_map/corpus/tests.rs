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
pub(super) struct NoIo;
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
fn explicit_head_cap_accounts_for_every_policy_window_before_model_io() {
    assert_eq!(parse_policy_with_max_heads(POLICY, None), parse_policy(POLICY));
    assert_eq!(parse_policy_with_max_heads(POLICY, Some(32)), parse_policy(POLICY));
    for windows in [1, 2, 3, 31] {
        let mut value: serde_json::Value = serde_json::from_str(POLICY).unwrap();
        value["max_evidence_windows"] = serde_json::json!(windows);
        // Retaining one witness must not erase the other scored windows.
        value["max_evidence_spans"] = serde_json::json!(1);
        let text = value.to_string();
        let needed = if windows == 1 { 1 } else { windows + 1 };
        for cap in 0..=33 {
            assert_eq!(parse_policy_with_max_heads(&text, Some(cap)).is_ok(),
                (needed..=32).contains(&cap), "windows={windows}, cap={cap}");
        }
    }
}

#[test]
fn explicit_head_cap_admits_smaller_cohorts_without_renewing_any_work_counter() {
    let (legacy, mut args) = arguments(&["--cohort-chunks", "3"]);
    let old = legacy.validate(&args).unwrap();
    assert_eq!(legacy.stream.max_evidence_heads, None);
    let (explicit, _) = arguments(&["--cohort-chunks", "3", "--max-evidence-heads", "32"]);
    let unchanged = explicit.validate(&args).unwrap();
    assert_eq!(old.document.memory, unchanged.document.memory);
    assert_eq!(old.common.minimum_memory_bytes, unchanged.common.minimum_memory_bytes);
    assert_eq!(legacy.document.budget(old.common).unwrap(),
        explicit.document.budget_with_max_heads(unchanged.common, 32).unwrap());

    let (narrow, _) = arguments(&["--cohort-chunks", "3", "--max-evidence-heads", "1"]);
    let small = narrow.validate(&args).unwrap();
    assert_eq!(small.document.memory.native.kv_bytes, small.common.native.kv_bytes * 3);
    assert_eq!(narrow.document.budget_with_max_heads(small.common, 1).unwrap().max_kv_bytes,
        small.common.native.kv_bytes);
    assert!(small.document.memory.input_bytes < old.document.memory.input_bytes);
    assert!(small.common.minimum_memory_bytes < old.common.minimum_memory_bytes);
    assert_eq!(small.document.mapping.max_model_work, old.document.mapping.max_model_work);
    assert_eq!(small.transport, old.transport);
    assert_eq!(small.document.memory.max_output_bytes, old.document.memory.max_output_bytes);
    args.memory_mib = small.common.minimum_memory_bytes.div_ceil(MIB);
    narrow.validate(&args).unwrap();
    assert_eq!(legacy.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn source_and_snapshot_share_explicit_cap_and_keep_ragged_geometry_independent() {
    let (options, args) = arguments(&["--cohort-chunks", "3", "--max-evidence-heads", "3"]);
    let source = options.validate(&args).unwrap();
    let pin = "ab".repeat(32);
    let snapshot = parsed(&["--snapshot", "private.snapshot", "--expected-model-sha256", &pin,
        "--policy", "private-policy.json", "--cohort-chunks", "3", "--max-evidence-heads", "3"]).unwrap();
    let Int4Task::FaithfulnessCorpus(restored) = snapshot.task else { panic!("snapshot corpus required"); };
    let loaded = restored.validate(&snapshot.args).unwrap();
    assert_eq!(source.document.memory, loaded.document.memory);
    assert_eq!(options.document.budget_with_max_heads(source.common, 3).unwrap(),
        restored.document.budget_with_max_heads(loaded.common, 3).unwrap());
    assert_eq!(source.document.memory.native.kv_bytes, source.common.native.kv_bytes * 9);

    // A theoretical 3 * 32 peak above 64 only bounds the allocation. Each
    // actual window is still prepared in full and checked by the core.
    let (options, args) = arguments(&["--max-result-bytes", "65536", "--cohort-chunks", "32",
        "--max-evidence-heads", "3"]);
    let ragged = options.validate(&args).unwrap();
    assert_eq!(ragged.document.memory.native.kv_bytes, ragged.common.native.kv_bytes * 64);
    assert_eq!(options.document.mapping.cohort_chunks, Some(32));
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
        vec!["--timeout-seconds", "0"], vec!["--max-checkpoints", "1"], vec!["--preparation-mib", "1"],
        vec!["--max-evidence-heads", "0"], vec!["--max-evidence-heads", "33"]] {
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

#[test]
fn population_teams_price_one_workspace_and_preserve_all_evidence_and_work_limits() {
    use crate::native_engine::strict_int4::projection::int4_team_workspace_bytes;
    for max_heads in [None, Some(1), Some(3), Some(32)] {
        let (mut serial, args) = arguments(&[]); serial.stream.max_evidence_heads = max_heads;
        let baseline = serial.validate(&args).unwrap();
        let task = serial.document.budget_with_max_heads(baseline.common, max_heads.unwrap_or(32)).unwrap();
        for threads in [1, 2, 4, 64] {
            let width = threads.to_string();
            let (mut command, mut args) = arguments(&["--compute-threads", width.as_str()]);
            command.stream.max_evidence_heads = max_heads;
            let limits = command.validate(&args).unwrap(); let common = limits.common;
            let scratch = if threads == 1 { 0 } else { int4_team_workspace_bytes(threads, args.group_width).unwrap() };
            assert_eq!(common.compute.threads, threads);
            assert_eq!(common.compute.workspace_bytes, scratch);
            assert_eq!(limits.document.memory, baseline.document.memory);
            let mapping = |map: MapLimits| (map.mapping.chunks, map.mapping.max_model_work,
                map.mapping.max_retained_result_bytes, map.mapping.max_result_bytes, map.mapping.aggregation);
            let planning = |limits: JudgeLimits| (limits.per_head.max_candidates, limits.per_head.max_total_tokens,
                limits.per_head.max_nodes, limits.per_head.max_depth, limits.per_head.max_candidate_id_bytes,
                limits.per_head.max_projected_logits, limits.max_total_prompt_tokens,
                limits.max_total_candidate_tokens, limits.max_total_projected_logits, limits.max_output_bytes);
            assert_eq!(mapping(limits.document), mapping(baseline.document));
            assert_eq!(planning(limits.document.planning), planning(baseline.document.planning));
            assert_eq!(limits.transport, baseline.transport);
            assert_eq!(command.document.budget_with_max_heads(common, max_heads.unwrap_or(32)).unwrap(), task);
            let map = limits.document;
            let execution = map.memory.native.kv_bytes + map.memory.input_bytes
                + map.memory.workspace_bytes + map.memory.output_reservation_bytes + scratch;
            let loading = common.source_preparation_bytes + common.load.scratch_payload_bytes;
            let document_peak = map.cli_preparation_bytes + common.load.weight_payload_bytes + ALLOCATOR_BYTES
                + loading.max(execution) + CLEANUP_BYTES;
            assert_eq!(common.minimum_memory_bytes, document_peak
                + corpus_buffer_bytes(limits.transport, common.preparation_bytes, limits.io_bytes).unwrap()
                + retained_metadata_bytes(map.mapping.chunks.max_chunks).unwrap() + limits.io_bytes);
            args.memory_mib = common.minimum_memory_bytes.div_ceil(MIB);
            command.validate(&args).unwrap();
            args.memory_mib -= 1;
            assert_eq!(command.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
        }
    }
}

#[test]
fn invalid_widths_and_every_explicit_cohort_team_mix_refuse_before_private_io() {
    for width in ["0", "65", "18446744073709551615"] {
        let (command, args) = arguments(&["--compute-threads", width]);
        assert_eq!(command.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    }
    for chunks in ["1", "3", "64"] {
        let (command, args) = arguments(&["--compute-threads", "2", "--cohort-chunks", chunks]);
        assert_eq!(command.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    }
    let help = definition().render_long_help().to_string();
    for required in ["--compute-threads", "--cohort-chunks", "compute_schedule", "without renewing work or attempts"] {
        assert!(help.split_whitespace().collect::<Vec<_>>().join(" ").contains(required));
    }
}


#[test]
fn compact_corpus_flags_reach_the_complete_document_policy_and_quote_one_workspace() {
    use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};
    for schedule in [vec![], vec!["--cohort-chunks", "3", "--max-evidence-heads", "3"],
        vec!["--compute-threads", "2"]] {
        let (byte, byte_args) = arguments(&schedule);
        let baseline = byte.validate(&byte_args).unwrap();
        let mut flags = schedule;
        flags.extend(["--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "65536",
            "--max-tokenizer-output-tokens", "32768", "--max-tokenizer-pair-checks", "1048576",
            "--max-tokenizer-pair-bytes", "16777216"]);
        let (compact, mut args) = arguments(&flags);
        let selected = compact.document.input_encoding(&args).unwrap();
        let policy = SafeBpeLimits { max_input_bytes: 65536, max_output_tokens: 32768,
            max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 };
        assert_eq!(selected, UntrustedEncoding::SafeBpeV1(policy));
        let limits = compact.validate(&args).unwrap();
        assert_eq!(limits.document.memory.input_bytes - baseline.document.memory.input_bytes,
            policy.requirements().unwrap().workspace_bytes);
        assert_eq!(limits.document.memory.native, baseline.document.memory.native);
        assert_eq!(limits.document.mapping.chunks, baseline.document.mapping.chunks);
        assert_eq!(limits.document.mapping.max_model_work, baseline.document.mapping.max_model_work);
        assert_eq!(limits.transport, baseline.transport);
        assert_eq!(limits.cli_preparation_bytes, baseline.cli_preparation_bytes);
        assert!(limits.common.minimum_memory_bytes >= baseline.common.minimum_memory_bytes);
        args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
        compact.validate(&args).unwrap();
        args.memory_mib -= 1;
        assert_eq!(compact.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn compact_corpus_refuses_invalid_work_policy_and_snapshot_saving_before_private_io() {
    for flags in [
        vec!["--max-tokenizer-input-bytes", "65536"],
        vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "0"],
        vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "1048577"],
        vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-output-tokens", "0"],
        vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "0"],
        vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-bytes", "0"],
        vec!["--input-encoding", "safe-bpe-v1", "--save-snapshot", "must-not-be-created"],
    ] {
        let (command, args) = arguments(&flags);
        assert_eq!(command.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments), "{flags:?}");
    }
    let (command, args) = arguments(&["--input-encoding", "safe-bpe-v1"]);
    let source = command.validate(&args).unwrap();
    let pin = "ab".repeat(32);
    let loaded = parsed(&["--snapshot", "private.snapshot", "--expected-model-sha256", &pin,
        "--policy", "private-policy.json", "--input-encoding", "safe-bpe-v1"]).unwrap();
    let Int4Task::FaithfulnessCorpus(restored) = loaded.task else { panic!("faithfulness corpus required"); };
    let snapshot = restored.validate(&loaded.args).unwrap();
    assert_eq!(source.document.memory, snapshot.document.memory);
    assert_eq!(command.document.input_encoding(&args).unwrap(), restored.document.input_encoding(&loaded.args).unwrap());
}
