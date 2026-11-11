//! Strict wire and declared ownership tests; these fixtures perform no inference.
use super::*;
use crate::tasks::sentiment::SentimentAxis;

fn parsed(flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", "sentiment-corpus", "--memory-mib", "131072"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("source INT4 route required"); };
    Ok(command)
}
pub(super) fn command(flags: &[&str]) -> Int4Command {
    let mut all = vec!["--source-dir", "explicit-source", "--source-manifest", "explicit-manifest"];
    all.extend_from_slice(flags);
    parsed(&all).unwrap()
}
fn arguments(flags: &[&str]) -> (CorpusArgs, Int4Args) {
    let command = command(flags);
    let Int4Task::SentimentCorpus(options) = command.task else { panic!("sentiment corpus required"); };
    (options, command.args)
}
pub(super) struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused sentiment corpus read input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused sentiment corpus wrote output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused sentiment corpus flushed output"); }
}

#[test]
fn source_and_snapshot_routes_share_owned_complete_document_semantics() {
    assert_eq!(command(&[]).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    for chunks in [None, Some("2")] {
        let mut flags = vec!["--defaults", "private-axes.json", "--max-axes", "2"];
        if let Some(chunks) = chunks { flags.extend(["--cohort-chunks", chunks]); }
        let (options, args) = arguments(&flags);
        let source = options.validate(&args).unwrap();
        assert!(source.common.snapshot_pin.is_none());
        assert_eq!(format!("{options:?}"), "SentimentCorpusArgs");
        let pin = "ab".repeat(32);
        let mut snapshot_flags = vec!["--snapshot", "private.snapshot", "--expected-model-sha256", &pin];
        snapshot_flags.extend_from_slice(&flags);
        let snapshot = parsed(&snapshot_flags).unwrap();
        let Int4Task::SentimentCorpus(restored) = snapshot.task else { panic!("snapshot corpus required"); };
        let loaded = restored.validate(&snapshot.args).unwrap();
        assert!(loaded.common.snapshot_pin.is_some());
        assert_eq!(source.transport, loaded.transport);
        assert_eq!(source.document.mapping.chunks, loaded.document.mapping.chunks);
        assert_eq!(source.document.mapping.max_model_work, loaded.document.mapping.max_model_work);
        assert_eq!(source.document.mapping.aggregation, loaded.document.mapping.aggregation);
        assert_eq!(source.document.memory, loaded.document.memory);
        assert_eq!(options.document.budget_with_max_axes(source.common, options.stream.max_axes).unwrap(),
            restored.document.budget_with_max_axes(loaded.common, restored.stream.max_axes).unwrap());
    }
    let pin = "ab".repeat(32);
    for flags in [vec![], vec!["--snapshot", "private.snapshot"],
        vec!["--snapshot", "private.snapshot", "--expected-model-sha256", &pin, "--source-dir", "source"],
        vec!["--snapshot", "private.snapshot", "--expected-model-sha256", &pin, "--save-snapshot", "new.snapshot"]] {
        assert!(parsed(&flags).is_err());
    }
}

#[test]
fn transport_capacity_and_native_scheduling_never_multiply_the_lifetime_work_allowance() {
    let (mut options, args) = arguments(&["--max-requests", "10"]);
    let serial = options.validate(&args).unwrap();
    let serial_budget = options.document.budget_with_max_axes(serial.common, options.stream.max_axes).unwrap();
    options.stream.max_requests = 1000;
    options.document.mapping.cohort_chunks = Some(2);
    let grouped = options.validate(&args).unwrap();
    assert_eq!(serial.document.mapping.max_model_work, grouped.document.mapping.max_model_work);
    assert_eq!(grouped.document.mapping.max_model_work, options.document.scoring.work_ceiling());
    assert_eq!(serial.transport.max_work, grouped.transport.max_work);
    assert_eq!(grouped.transport.max_work.forward_positions, grouped.document.mapping.max_model_work.forward_positions);
    assert_eq!(grouped.transport.max_work.projected_logits, grouped.document.mapping.max_model_work.projected_logits);
    assert_eq!(serial_budget, options.document.budget_with_max_axes(grouped.common, options.stream.max_axes).unwrap());
    assert_eq!(serial_budget.max_kv_bytes, serial.common.native.kv_bytes * 4);
    assert_eq!(grouped.document.memory.native.kv_bytes, serial.document.memory.native.kv_bytes * 8);
    assert!(grouped.common.minimum_memory_bytes > serial.common.minimum_memory_bytes);
    for (count, limits) in [(10, serial), (1000, grouped)] {
        assert_eq!(limits.transport.max_output_bytes + (count + 4) * FRAME_ALLOWANCE, limits.output_bytes);
    }
}

#[test]
fn invalid_work_transport_and_declared_axis_geometry_refuse_before_any_io() {
    for flags in [vec!["--max-forward-positions", "0"], vec!["--max-projected-logits", "0"],
        vec!["--max-attention-pairs", "0"], vec!["--max-dot-products", "0"],
        vec!["--max-multiply-accumulates", "0"], vec!["--max-requests", "0"],
        vec!["--max-requests", "100001"], vec!["--max-total-input-bytes", "0"],
        vec!["--max-total-input-bytes", "17179869185"], vec!["--max-total-output-bytes", "1"],
        vec!["--max-total-output-bytes", "17179869185"], vec!["--max-line-bytes", "1048575"],
        vec!["--max-line-bytes", "4194305"], vec!["--defaults", "-"],
        vec!["--max-axes", "0"], vec!["--max-axes", "5"],
        vec!["--cohort-chunks", "0"], vec!["--cohort-chunks", "65"],
        vec!["--cohort-chunks", "17"], vec!["--max-axes", "2", "--cohort-chunks", "33"],
        vec!["--max-chunks", "0"], vec!["--max-chunks", "257"],
        vec!["--max-chunks", "2", "--cohort-chunks", "3"],
        vec!["--max-chunk-bytes", "3"], vec!["--max-tokenizer-calls", "0"],
        vec!["--max-map-result-bytes", "0"], vec!["--max-retained-result-bytes", "0"],
        vec!["--minimum-estimated-coverage-ppm", "1000001"],
        vec!["--timeout-seconds", "0"], vec!["--max-checkpoints", "1"], vec!["--preparation-mib", "1"]] {
        let (options, args) = arguments(&flags);
        assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments), "{flags:?}");
    }
    let (options, mut args) = arguments(&[]);
    args.seed = Some("ab".repeat(32));
    assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn unsupported_generation_sampling_and_cross_document_schedules_are_never_ignored() {
    for flags in [vec!["--prefill-rows", "1"], vec!["--cohort-heads", "1"],
        vec!["--cohort-documents", "1"], vec!["--max-new-tokens", "128"],
        vec!["--max-output-bytes", "65536"], vec!["--min-new-tokens", "0"],
        vec!["--logprobs"], vec!["--seed", "ab"], vec!["--task", "sentiment"],
        vec!["--mode", "sampled"], vec!["--focus-area", "reviews"], vec!["--max-labels", "2"]] {
        let mut all = vec!["--source-dir", "source", "--source-manifest", "manifest"];
        all.extend_from_slice(&flags);
        assert!(parsed(&all).is_err(), "{flags:?}");
    }
}

#[test]
fn corpus_memory_prices_complete_maps_and_the_narrowed_axis_ceiling_before_input() {
    let (wide, args) = arguments(&["--cohort-chunks", "2"]);
    let full = wide.validate(&args).unwrap();
    let (options, mut args) = arguments(&["--cohort-chunks", "2", "--max-axes", "1"]);
    let document = options.document.validate_with_max_axes(&args, 1).unwrap();
    let limits = options.validate(&args).unwrap();
    assert_eq!(limits.document.memory.native.kv_bytes, limits.common.native.kv_bytes * 2);
    assert_eq!(limits.document.mapping.max_model_work, full.document.mapping.max_model_work);
    let buffers = corpus_buffer_bytes(limits.transport, limits.common.preparation_bytes, limits.io_bytes).unwrap();
    let metadata = retained_metadata_bytes(document.mapping.chunks.max_chunks).unwrap();
    assert_eq!(limits.common.minimum_memory_bytes,
        document.common.minimum_memory_bytes + buffers + metadata + limits.io_bytes);
    assert_eq!(limits.cli_preparation_bytes, document.cli_preparation_bytes + limits.io_bytes);
    assert_eq!(limits.transport.max_output_line_bytes as u64, document.memory.max_output_bytes + FRAME_ALLOWANCE);
    assert!(limits.transport.max_output_line_bytes > limits.common.result_bytes);
    assert!(limits.io_bytes >= limits.transport.max_output_line_bytes as u64 + 2 * IO_BUFFER_BYTES as u64);
    assert!(limits.common.minimum_memory_bytes < full.common.minimum_memory_bytes);
    args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
    options.validate(&args).unwrap();
    assert_eq!(wide.validate(&args).err(), Some(CandidateError::Arguments));
    args.memory_mib -= 1;
    assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn defaults_freeze_policy_preserve_requested_axes_and_inject_only_the_host_budget() {
    let (options, args) = arguments(&[]);
    let limits = options.validate(&args).unwrap();
    let (fallback, policy) = options.load_defaults(None, limits, &args).unwrap();
    assert_eq!(fallback.axes, SentimentAxis::ALL);
    assert_eq!(policy, SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 });
    let (empty, empty_policy) = options.load_defaults(Some("{}"), limits, &args).unwrap();
    assert_eq!(empty.axes, fallback.axes); assert_eq!(empty.budget, fallback.budget); assert_eq!(empty_policy, policy);
    let input = r#"{"axes":["approach","valence"],"policy":{"minimum_peak_weight_ppm":650000,"maximum_normalized_entropy_ppm":800000}}"#;
    let (defaults, policy) = options.load_defaults(Some(input), limits, &args).unwrap();
    assert_eq!(defaults.axes, [SentimentAxis::Approach, SentimentAxis::Valence]);
    assert_eq!(policy, SentimentPolicy { minimum_peak_weight_ppm: 650_000, maximum_normalized_entropy_ppm: 800_000 });
    assert_eq!(defaults.budget, options.document.budget_with_max_axes(limits.common, 4).unwrap());
    let (narrow, args) = arguments(&["--max-axes", "1", "--cohort-chunks", "2"]);
    let limited = narrow.validate(&args).unwrap();
    assert_eq!(narrow.load_defaults(None, limited, &args).err(), Some(CandidateError::Planning));
    assert_eq!(narrow.load_defaults(Some(input), limited, &args).err(), Some(CandidateError::Planning));
    let (defaults, _) = narrow.load_defaults(Some(r#"{"axes":["valence"]}"#), limited, &args).unwrap();
    assert_eq!(defaults.axes, [SentimentAxis::Valence]);
    assert_eq!(defaults.budget.max_kv_bytes, limited.common.native.kv_bytes);
}

#[test]
fn defaults_reject_duplicate_keys_unknown_authority_and_invalid_dimensions() {
    let (options, args) = arguments(&[]);
    let limits = options.validate(&args).unwrap();
    for input in ["null", "[]", r#"{"axes":[]}"#, r#"{"axes":["happiness"]}"#,
        r#"{"axes":["valence","valence"]}"#, r#"{"axes":["valence"],"axes":["arousal"]}"#,
        r#"{"axes":["valence"],"document":"private source"}"#,
        r#"{"axes":["valence"],"budget":{}}"#, r#"{"axes":["valence"],"model_id":"other"}"#,
        r#"{"axes":["valence"],"eos_token_id":0}"#, r#"{"axes":["valence"],"mode":"sampled"}"#,
        r#"{"axes":["valence"],"score_space":"legal_mass"}"#,
        r#"{"policy":{"minimum_peak_weight_ppm":1000001,"maximum_normalized_entropy_ppm":1000000}}"#,
        r#"{"policy":{"minimum_peak_weight_ppm":0,"maximum_normalized_entropy_ppm":1000001}}"#,
        r#"{"policy":{"minimum_peak_weight_ppm":0,"minimum_peak_weight_ppm":1,"maximum_normalized_entropy_ppm":1000000}}"#] {
        assert_eq!(options.load_defaults(Some(input), limits, &args).err(), Some(CandidateError::Input), "{input}");
    }
    let input = " ".repeat(shared::DEFAULTS_BYTES + 1);
    assert_eq!(options.load_defaults(Some(&input), limits, &args).err(), Some(CandidateError::Input));
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn absent_host_feature_refuses_before_opening_private_defaults_or_source_files() {
    let (options, args) = arguments(&["--defaults", "never-open-private-axes.json"]);
    assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Unavailable));
}

#[test]
fn compact_corpus_policy_is_explicit_and_encoder_memory_is_not_multiplied_by_cohorts() {
    for schedule in [vec![], vec!["--cohort-chunks", "2", "--max-axes", "2"]] {
        let (byte_options, byte_args) = arguments(&schedule);
        let byte = byte_options.validate(&byte_args).unwrap();
        let mut flags = schedule;
        flags.extend(["--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "262144",
            "--max-tokenizer-output-tokens", "131072", "--max-tokenizer-pair-checks", "33554432",
            "--max-tokenizer-pair-bytes", "268435456"]);
        let (options, args) = arguments(&flags);
        let compact = options.validate(&args).unwrap();
        let policy = SafeBpeLimits { max_input_bytes: 262_144, max_output_tokens: 131_072,
            max_pair_checks: 33_554_432, max_pair_bytes: 268_435_456 };
        assert_eq!(options.document.input_encoding(&args).unwrap(), UntrustedEncoding::SafeBpeV1(policy));
        assert_eq!(compact.document.memory.input_bytes - byte.document.memory.input_bytes,
            policy.requirements().unwrap().workspace_bytes);
        assert_eq!(compact.document.memory.native, byte.document.memory.native);
        assert_eq!(compact.document.mapping.max_model_work, byte.document.mapping.max_model_work);
        assert_eq!(compact.transport, byte.transport);
        let (explicit, explicit_args) = arguments(&["--input-encoding", "byte-fallback"]);
        let (default, default_args) = arguments(&[]);
        assert_eq!(explicit.validate(&explicit_args).unwrap().document.memory,
            default.validate(&default_args).unwrap().document.memory);
    }
}

#[test]
fn compact_corpus_limits_refuse_before_defaults_model_or_source_io() {
    for flags in [vec!["--max-tokenizer-input-bytes", "128"],
        vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "0"],
        vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-output-tokens", "1048577"],
        vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "0"],
        vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-bytes", "536870913"],
        vec!["--input-encoding", "safe-bpe-v1", "--save-snapshot", "never-create.snapshot"]] {
        let (options, args) = arguments(&flags);
        assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments), "{flags:?}");
    }
    let (options, mut args) = arguments(&["--input-encoding", "safe-bpe-v1"]);
    args.memory_mib = 1;
    assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn resident_teams_preserve_axis_policy_work_and_one_encoder_quote() {
    for axes in ["1", "4"] {
        for compact in [false, true] {
            let mut base_flags = vec!["--max-axes", axes];
            if compact {
                base_flags.extend(["--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "262144",
                    "--max-tokenizer-output-tokens", "131072", "--max-tokenizer-pair-checks", "33554432",
                    "--max-tokenizer-pair-bytes", "268435456"]);
            }
            let (baseline, base_args) = arguments(&base_flags);
            let base = baseline.validate(&base_args).unwrap();
            let defaults = r#"{"axes":["valence"],"policy":{"minimum_peak_weight_ppm":650000,"maximum_normalized_entropy_ppm":800000}}"#;
            let (base_settings, base_policy) = baseline.load_defaults(Some(defaults), base, &base_args).unwrap();
            for threads in [1, 2, 4, 64] {
                let width = threads.to_string(); let mut flags = base_flags.clone();
                flags.extend(["--compute-threads", width.as_str()]);
                let (options, args) = arguments(&flags); let limits = options.validate(&args).unwrap();
                assert_eq!(args.compute.threads(), threads, "root dispatch must retain explicit team width");
                let workspace = if threads == 1 { 0 } else {
                    crate::native_engine::strict_int4::projection::int4_team_workspace_bytes(threads, args.group_width).unwrap()
                };
                assert_eq!(limits.common.compute.workspace_bytes, workspace);
                assert_eq!(limits.document.memory, base.document.memory);
                assert_eq!(limits.document.mapping.chunks, base.document.mapping.chunks);
                assert_eq!(limits.document.mapping.max_model_work, base.document.mapping.max_model_work);
                assert_eq!(limits.document.mapping.aggregation, base.document.mapping.aggregation);
                assert_eq!(limits.transport, base.transport);
                assert_eq!(options.document.input_encoding(&args).unwrap(),
                    baseline.document.input_encoding(&base_args).unwrap());
                let (settings, policy) = options.load_defaults(Some(defaults), limits, &args).unwrap();
                assert_eq!(settings.axes, base_settings.axes); assert_eq!(settings.budget, base_settings.budget);
                assert_eq!(policy, base_policy);
                let map = limits.document; let common = limits.common;
                let loading = common.source_preparation_bytes + common.load.scratch_payload_bytes;
                let execution = map.memory.native.kv_bytes + map.memory.input_bytes
                    + map.memory.workspace_bytes + map.memory.output_reservation_bytes + workspace;
                let document_peak = map.cli_preparation_bytes + common.load.weight_payload_bytes
                    + ALLOCATOR_BYTES + loading.max(execution) + CLEANUP_BYTES;
                assert_eq!(map.common.minimum_memory_bytes, document_peak);
                assert_eq!(common.minimum_memory_bytes, document_peak
                    + corpus_buffer_bytes(limits.transport, common.preparation_bytes, limits.io_bytes).unwrap()
                    + retained_metadata_bytes(map.mapping.chunks.max_chunks).unwrap() + limits.io_bytes);
            }
        }
    }
}

#[test]
fn invalid_or_underfunded_population_teams_refuse_before_private_io() {
    for compact in [false, true] {
        for mut flags in [vec!["--compute-threads", "0"], vec!["--compute-threads", "65"],
            vec!["--compute-threads", "2", "--cohort-chunks", "1"],
            vec!["--compute-threads", "4", "--cohort-chunks", "2"]] {
            if compact { flags.extend(["--input-encoding", "safe-bpe-v1"]); }
            let (options, args) = arguments(&flags);
            assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments), "{flags:?}");
        }
        let mut flags = vec!["--compute-threads", "64"];
        if compact { flags.extend(["--input-encoding", "safe-bpe-v1"]); }
        let (options, mut args) = arguments(&flags); let limits = options.validate(&args).unwrap();
        args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB) - 1;
        assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    }
    let (options, args) = arguments(&["--compute-threads", "1", "--cohort-chunks", "2"]);
    options.validate(&args).unwrap();
}
