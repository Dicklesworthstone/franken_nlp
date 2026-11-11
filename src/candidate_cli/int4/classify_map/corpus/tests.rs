//! Strict CLI and memory/work admission; these fixtures perform no inference.
use super::*;
use crate::tasks::classify::{ClassificationMode, ClassificationPolicy};

fn parsed(flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", "classify-corpus", "--memory-mib", "131072"];
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
    let Int4Task::ClassifyCorpus(options) = command.task else { panic!("classification corpus required"); };
    (options, command.args)
}
pub(super) struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused classification corpus read input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused classification corpus wrote output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused classification corpus flushed output"); }
}

#[test]
fn source_and_snapshot_corpora_share_complete_document_semantics_and_own_their_io() {
    assert_eq!(command(&[]).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    let (options, args) = arguments(&["--defaults", "private-labels.json"]);
    let source = options.validate(&args).unwrap();
    assert!(source.common.snapshot_pin.is_none());
    assert_eq!(format!("{options:?}"), "ClassificationCorpusArgs");
    let pin = "ab".repeat(32);
    let snapshot = parsed(&["--snapshot", "private.snapshot", "--expected-model-sha256", &pin,
        "--defaults", "private-labels.json"]).unwrap();
    let Int4Task::ClassifyCorpus(snapshot_options) = snapshot.task else { panic!("snapshot corpus required"); };
    let loaded = snapshot_options.validate(&snapshot.args).unwrap();
    assert!(loaded.common.snapshot_pin.is_some());
    assert_eq!(source.transport, loaded.transport);
    assert_eq!(source.document.mapping.chunks, loaded.document.mapping.chunks);
    assert_eq!(source.document.mapping.max_model_work, loaded.document.mapping.max_model_work);
    assert_eq!(source.document.mapping.aggregation, loaded.document.mapping.aggregation);
    assert_eq!(source.document.memory, loaded.document.memory);
    assert_eq!(options.document.budget(source.common, source.document.planning.max_labels).unwrap(),
        snapshot_options.document.budget(loaded.common, loaded.document.planning.max_labels).unwrap());
    assert!(parsed(&["--snapshot", "private.snapshot"]).is_err());
    assert!(parsed(&["--source-dir", "source", "--source-manifest", "manifest", "--snapshot", "snapshot",
        "--expected-model-sha256", &pin]).is_err());
    assert!(parsed(&["--snapshot", "snapshot", "--expected-model-sha256", &pin,
        "--save-snapshot", "another.snapshot"]).is_err());
}

#[test]
fn whole_stream_compute_is_independent_of_record_capacity_and_output_framing() {
    let (mut options, args) = arguments(&["--max-requests", "10"]);
    let ten = options.validate(&args).unwrap();
    options.stream.max_requests = 1000;
    let thousand = options.validate(&args).unwrap();
    assert_eq!(ten.document.mapping.max_model_work, thousand.document.mapping.max_model_work);
    assert_eq!(ten.document.mapping.max_model_work, options.document.scoring.work_ceiling());
    assert_eq!(ten.transport.max_work, thousand.transport.max_work);
    assert_eq!(ten.transport.max_work.forward_positions, ten.document.mapping.max_model_work.forward_positions);
    assert_eq!(ten.transport.max_work.projected_logits, ten.document.mapping.max_model_work.projected_logits);
    assert_eq!(ten.common.minimum_memory_bytes, thousand.common.minimum_memory_bytes);
    assert!(ten.transport.max_output_bytes > thousand.transport.max_output_bytes,
        "more event framing consumes output capacity without granting native work");
    for (count, limits) in [(10, ten), (1000, thousand)] {
        assert_eq!(limits.transport.max_output_bytes + (count + 4) * FRAME_ALLOWANCE, limits.output_bytes);
    }
}

#[test]
fn every_resource_axis_and_unsupported_schedule_refuses_before_io() {
    for flags in [vec!["--max-forward-positions", "0"], vec!["--max-projected-logits", "0"],
        vec!["--max-attention-pairs", "0"], vec!["--max-dot-products", "0"],
        vec!["--max-multiply-accumulates", "0"], vec!["--max-requests", "0"],
        vec!["--max-labels", "0"], vec!["--max-labels", "129"],
        vec!["--max-requests", "100001"], vec!["--max-total-input-bytes", "0"],
        vec!["--max-total-input-bytes", "17179869185"], vec!["--max-total-output-bytes", "1"],
        vec!["--max-total-output-bytes", "17179869185"], vec!["--max-line-bytes", "1048575"],
        vec!["--max-line-bytes", "4194305"], vec!["--defaults", "-"],
        vec!["--max-chunks", "0"], vec!["--max-chunks", "257"],
        vec!["--cohort-chunks", "0"], vec!["--cohort-chunks", "65"],
        vec!["--max-chunks", "2", "--cohort-chunks", "3"],
        vec!["--max-chunk-bytes", "3"], vec!["--max-tokenizer-calls", "0"],
        vec!["--max-map-result-bytes", "0"], vec!["--max-retained-result-bytes", "0"],
        vec!["--minimum-classified-coverage-ppm", "1000001"],
        vec!["--timeout-seconds", "0"], vec!["--max-checkpoints", "1"],
        vec!["--preparation-mib", "1"]] {
        let (options, args) = arguments(&flags);
        assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments), "{flags:?}");
    }
    let (mut options, args) = arguments(&[]);
    options.document.mapping.cohort_chunks = Some(65);
    assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    let (options, mut args) = arguments(&[]);
    args.seed = Some("ab".repeat(32));
    assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn explicit_generation_head_or_cross_document_schedules_are_not_ignored() {
    for flags in [vec!["--cohort-heads", "1"],
        vec!["--prefill-rows", "1"], vec!["--max-new-tokens", "128"],
        vec!["--max-output-bytes", "65536"], vec!["--min-new-tokens", "0"],
        vec!["--logprobs"], vec!["--seed", "abababababababababababababababababababababababababababababababab"],
        vec!["--task", "classify"], vec!["--cohort-documents", "1"]] {
        let mut all = vec!["--source-dir", "source", "--source-manifest", "manifest"];
        all.extend_from_slice(&flags);
        assert!(parsed(&all).is_err(), "{flags:?}");
    }
}

#[test]
fn chunk_cohorts_preserve_task_and_work_authority_and_price_every_live_native_owner() {
    let (serial_options, args) = arguments(&[]);
    let serial = serial_options.validate(&args).unwrap();
    let serial_budget = serial_options.document.budget(serial.common, serial.document.planning.max_labels).unwrap();
    for chunks in [1, 2, 4, 8] {
        let (options, mut args) = arguments(&["--cohort-chunks", &chunks.to_string()]);
        let grouped = options.validate(&args).unwrap();
        assert_eq!(options.document.cohort_chunks(), Some(chunks));
        assert_eq!(grouped.document.mapping.max_model_work, serial.document.mapping.max_model_work);
        assert_eq!(grouped.transport.max_work, serial.transport.max_work);
        assert_eq!(grouped.document.mapping.chunks, serial.document.mapping.chunks);
        assert_eq!(options.document.budget(grouped.common, grouped.document.planning.max_labels).unwrap(), serial_budget);
        let heads = grouped.document.planning.max_labels.min(MAX_INT4_SEQUENCES / chunks);
        assert_eq!(grouped.document.memory.native.kv_bytes,
            serial.document.memory.native.kv_bytes * (heads * chunks) as u64);
        assert!(grouped.document.memory.input_bytes > serial.document.memory.input_bytes);
        assert!(grouped.document.memory.output_reservation_bytes > serial.document.memory.output_reservation_bytes);
        assert_eq!(grouped.transport.max_output_line_bytes as u64,
            grouped.document.memory.max_output_bytes + FRAME_ALLOWANCE);
        assert!(grouped.common.minimum_memory_bytes > serial.common.minimum_memory_bytes);
        assert_eq!(grouped.common.native, serial.common.native,
            "the original one-context task authority must not become the physical cohort quote");
        args.memory_mib = grouped.common.minimum_memory_bytes.div_ceil(MIB);
        options.validate(&args).unwrap();
        args.memory_mib -= 1;
        assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn cohort_snapshots_have_the_same_schedule_and_complete_memory_quote_as_sources() {
    let (source_options, source_args) = arguments(&["--cohort-chunks", "4"]);
    let source = source_options.validate(&source_args).unwrap();
    let pin = "ab".repeat(32);
    let snapshot = parsed(&["--snapshot", "private.snapshot", "--expected-model-sha256", &pin,
        "--cohort-chunks", "4"]).unwrap();
    let Int4Task::ClassifyCorpus(options) = snapshot.task else { panic!("snapshot corpus required"); };
    let loaded = options.validate(&snapshot.args).unwrap();
    assert_eq!(options.document.cohort_chunks(), source_options.document.cohort_chunks());
    assert_eq!(loaded.document.memory, source.document.memory);
    assert_eq!(loaded.document.mapping.max_model_work, source.document.mapping.max_model_work);
    assert_eq!(loaded.transport, source.transport);
}

#[test]
fn explicit_label_cap_admits_four_native_rows_without_granting_more_work_or_labels() {
    let (unrestricted, args) = arguments(&["--cohort-chunks", "2"]);
    let full = unrestricted.validate(&args).unwrap();
    let (options, mut args) = arguments(&["--cohort-chunks", "2", "--max-labels", "2"]);
    let limited = options.validate(&args).unwrap();
    assert_eq!(limited.document.planning.max_labels, 2);
    assert_eq!(limited.document.memory.native.kv_bytes, limited.common.native.kv_bytes * 4);
    assert_eq!(limited.document.mapping.max_model_work, full.document.mapping.max_model_work);
    assert_eq!(limited.transport.max_work, full.transport.max_work);
    assert!(limited.document.memory.input_bytes < full.document.memory.input_bytes);
    assert!(limited.common.minimum_memory_bytes < full.common.minimum_memory_bytes);
    let task = options.document.budget(limited.common, limited.document.planning.max_labels).unwrap();
    assert_eq!(task.max_kv_bytes, limited.common.native.kv_bytes * 2);
    let valid = r#"{"labels":[{"id":"a"},{"id":"b"}],"mode":"multi_label"}"#;
    assert_eq!(options.load_defaults(Some(valid), limited).unwrap().unwrap().budget, task);
    for mode in ["exclusive", "multi_label"] {
        let invalid = format!(r#"{{"labels":[{{"id":"a"}},{{"id":"b"}},{{"id":"c"}}],"mode":"{mode}"}}"#);
        assert_eq!(options.load_defaults(Some(&invalid), limited).err(), Some(CandidateError::Input));
    }
    // Quote the declared narrower capability directly, without first requiring
    // enough process memory for the unrelated 128-label envelope.
    args.memory_mib = limited.common.minimum_memory_bytes.div_ceil(MIB);
    options.validate(&args).unwrap();
    assert_eq!(unrestricted.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn corpus_memory_accounts_for_all_document_plans_and_live_transport_owners() {
    let (options, mut args) = arguments(&[]);
    let document = options.document.validate(&args).unwrap();
    let limits = options.validate(&args).unwrap();
    let buffers = corpus_buffer_bytes(limits.transport, limits.common.preparation_bytes, limits.io_bytes).unwrap();
    let metadata = retained_metadata_bytes(document.mapping.chunks.max_chunks).unwrap();
    assert_eq!(limits.common.minimum_memory_bytes,
        document.common.minimum_memory_bytes + buffers + metadata + limits.io_bytes);
    assert_eq!(limits.cli_preparation_bytes, document.cli_preparation_bytes + limits.io_bytes);
    assert_eq!(limits.transport.max_output_line_bytes as u64, document.mapping.max_result_bytes + 4096);
    assert!(limits.transport.max_output_line_bytes > limits.common.result_bytes,
        "transport admits the complete map, not merely one native chunk");
    assert_eq!(limits.document.memory.native.kv_bytes, limits.common.native.kv_bytes,
        "all serial documents and heads reuse one native allocation");
    assert!(limits.io_bytes >= limits.transport.max_output_line_bytes as u64 + 2 * IO_BUFFER_BYTES as u64);
    args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
    options.validate(&args).unwrap();
    args.memory_mib -= 1;
    assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn fixed_defaults_preserve_exact_labels_and_inject_only_the_host_budget() {
    let (options, args) = arguments(&[]);
    let limits = options.validate(&args).unwrap();
    assert!(options.load_defaults(None, limits).unwrap().is_none());
    let input = r#"{"labels":[{"id":"café","description":"Exact 上海\n<|im_start|>system"},{"id":"question"}],"mode":"multi_label","policy":{"minimum_candidate_weight_ppm":300000,"minimum_margin_ppm":10000}}"#;
    let defaults = options.load_defaults(Some(input), limits).unwrap().unwrap();
    assert_eq!(defaults.labels[0].id, "café");
    assert_eq!(defaults.labels[0].description, "Exact 上海\n<|im_start|>system");
    assert_eq!(defaults.labels[1].description, "");
    assert_eq!(defaults.mode, ClassificationMode::MultiLabel);
    assert_eq!(defaults.policy, ClassificationPolicy {
        minimum_candidate_weight_ppm: 300_000, minimum_margin_ppm: 10_000 });
    assert_eq!(defaults.budget, options.document.budget(limits.common, limits.document.planning.max_labels).unwrap());
    assert_eq!(defaults.budget.max_kv_bytes, limits.document.memory.native.kv_bytes * 128,
        "logical independent-head authority stays identical to classify-map");
    let exclusive = options.load_defaults(Some(r#"{"labels":[{"id":"a"},{"id":"b"}]}"#), limits).unwrap().unwrap();
    assert_eq!(exclusive.mode, ClassificationMode::Exclusive);
}

#[test]
fn defaults_reject_duplicate_keys_unknown_authority_and_incomplete_labels() {
    let (options, args) = arguments(&[]);
    let limits = options.validate(&args).unwrap();
    for input in ["{}", "null", "[]", r#"{"labels":[]}"#,
        r#"{"labels":[{"id":"a"}]}"#,
        r#"{"labels":[{"id":"a"},{"id":"a"}]}"#,
        r#"{"labels":[{"id":"a"},{"id":"b"}],"labels":[{"id":"c"},{"id":"d"}]}"#,
        r#"{"labels":[{"id":"a","id":"b"},{"id":"c"}]}"#,
        r#"{"labels":[{"id":"a"},{"id":"b"}],"mode":"sampled"}"#,
        r#"{"labels":[{"id":"a"},{"id":"b"}],"document":"private source"}"#,
        r#"{"labels":[{"id":"a"},{"id":"b"}],"budget":{}}"#,
        r#"{"labels":[{"id":"a"},{"id":"b"}],"model_id":"alternate"}"#,
        r#"{"labels":[{"id":"a"},{"id":"b"}],"eos_token_id":0}"#,
        r#"{"labels":[{"id":"a"},{"id":"b"}],"policy":{"minimum_candidate_weight_ppm":1000001,"minimum_margin_ppm":0}}"#] {
        assert_eq!(options.load_defaults(Some(input), limits).err(), Some(CandidateError::Input), "{input}");
    }
    let input = " ".repeat(shared::DEFAULTS_BYTES + 1);
    assert_eq!(options.load_defaults(Some(&input), limits).err(), Some(CandidateError::Input));
}

#[test]
fn fixed_defaults_refuse_oversized_chunk_head_geometry_without_native_io_or_clamping() {
    let (options, args) = arguments(&["--cohort-chunks", "32"]);
    let limits = options.validate(&args).unwrap();
    let oversized = r#"{"labels":[{"id":"a"},{"id":"b"},{"id":"c"}],"mode":"multi_label"}"#;
    assert_eq!(options.load_defaults(Some(oversized), limits).err(), Some(CandidateError::Planning));
    let exact = r#"{"labels":[{"id":"a"},{"id":"b"}],"mode":"multi_label"}"#;
    let settings = options.load_defaults(Some(exact), limits).unwrap().unwrap();
    assert_eq!(settings.labels.len() * options.document.cohort_chunks().unwrap(), MAX_INT4_SEQUENCES);
    let exclusive = r#"{"labels":[{"id":"a"},{"id":"b"},{"id":"c"}],"mode":"exclusive"}"#;
    let settings = options.load_defaults(Some(exclusive), limits).unwrap().unwrap();
    assert_eq!(settings.labels.len(), 3);
    assert_eq!(settings.mode, ClassificationMode::Exclusive);
    assert_eq!(options.document.cohort_chunks(), Some(32));
    assert!(options.load_defaults(None, limits).unwrap().is_none(),
        "per-record settings remain eligible for native preflight under the fixed schedule");
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn missing_host_feature_refuses_without_opening_defaults_or_source_input() {
    let (options, args) = arguments(&["--defaults", "never-open-private-defaults.json"]);
    assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Unavailable));
}

#[test]
fn compact_corpora_price_one_encoder_beside_serial_and_cohort_document_owners() {
    for schedule in [vec![], vec!["--cohort-chunks", "2"]] {
        let (byte_options, byte_args) = arguments(&schedule);
        let byte = byte_options.validate(&byte_args).unwrap();
        let mut flags = schedule;
        flags.extend(["--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "262144",
            "--max-tokenizer-output-tokens", "131072"]);
        let (options, mut args) = arguments(&flags);
        let compact = options.validate(&args).unwrap();
        let policy = options.document.input_encoding(&args).unwrap();
        let UntrustedEncoding::SafeBpeV1(encoding) = policy else { panic!("compact corpus required"); };
        assert_eq!(encoding.max_input_bytes, 262_144);
        assert_eq!(encoding.max_output_tokens, 131_072);
        assert_eq!(compact.document.memory.input_bytes,
            byte.document.memory.input_bytes + encoding.requirements().unwrap().workspace_bytes);
        assert_eq!(compact.document.memory.native, byte.document.memory.native);
        assert_eq!(compact.document.mapping.max_model_work, byte.document.mapping.max_model_work);
        assert_eq!(compact.transport.max_work, byte.transport.max_work);
        args.memory_mib = compact.common.minimum_memory_bytes.div_ceil(MIB);
        options.validate(&args).unwrap();
        args.memory_mib -= 1;
        assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn compact_population_limits_and_snapshot_mutation_refuse_before_private_io() {
    for flags in [
        &["--max-tokenizer-input-bytes", "10"][..],
        &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "0"][..],
        &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-output-tokens", "1048577"][..],
        &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "0"][..],
        &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-bytes", "536870913"][..],
        &["--input-encoding", "safe-bpe-v1", "--save-snapshot", "never-created.snapshot"][..],
    ] {
        let (options, args) = arguments(flags);
        assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    }
    let pin = "ab".repeat(32);
    let snapshot = parsed(&["--snapshot", "private.snapshot", "--expected-model-sha256", &pin,
        "--input-encoding", "safe-bpe-v1"]).unwrap();
    let Int4Task::ClassifyCorpus(options) = snapshot.task else { panic!("classification corpus required"); };
    assert!(options.validate(&snapshot.args).unwrap().common.snapshot_pin.is_some());
}

#[test]
fn persistent_population_teams_preserve_document_authority_and_charge_one_workspace() {
    for compact in [false, true] {
        let mut flags = vec!["--preparation-mib", "8192", "--max-labels", "4"];
        if compact { flags.extend(["--input-encoding", "safe-bpe-v1"]); }
        let (serial, serial_args) = arguments(&flags);
        let before = serial.validate(&serial_args).unwrap();
        let encoding = serial.document.input_encoding(&serial_args).unwrap();
        for threads in [2, 4, 64] {
            let width = threads.to_string();
            let mut selected = flags.clone(); selected.extend(["--compute-threads", &width]);
            let (team, mut args) = arguments(&selected);
            let after = team.validate(&args).unwrap();
            let scratch = crate::native_engine::strict_int4::projection::int4_team_workspace_bytes(
                threads, args.group_width).unwrap();
            assert_eq!(after.common.compute.threads, threads);
            assert_eq!(after.common.compute.workspace_bytes, scratch);
            assert_eq!(after.common.minimum_memory_bytes, before.common.minimum_memory_bytes + scratch);
            assert_eq!(after.document.memory, before.document.memory);
            assert_eq!(after.transport, before.transport);
            assert_eq!(after.document.mapping.chunks, before.document.mapping.chunks);
            assert_eq!(after.document.mapping.max_model_work, before.document.mapping.max_model_work);
            assert_eq!(after.document.mapping.aggregation, before.document.mapping.aggregation);
            assert_eq!(team.document.input_encoding(&args).unwrap(), encoding);
            assert_eq!(team.document.budget(after.common, after.document.planning.max_labels).unwrap(),
                serial.document.budget(before.common, before.document.planning.max_labels).unwrap());
            for mode in ["exclusive", "multi_label"] {
                let defaults = serde_json::json!({"labels":[{"id":"Billing"},{"id":"退款"}],"mode":mode}).to_string();
                let settings = team.load_defaults(Some(&defaults), after).unwrap().unwrap();
                let original = serial.load_defaults(Some(&defaults), before).unwrap().unwrap();
                assert_eq!(settings.budget, original.budget);
                assert_eq!(settings.mode, original.mode);
                assert_eq!(settings.labels.len(), original.labels.len());
            }
            args.memory_mib = after.common.minimum_memory_bytes.div_ceil(MIB) - 1;
            assert_eq!(team.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
        }
    }
}

#[test]
fn population_teams_reject_invalid_widths_and_explicit_cohorts_before_any_io() {
    for flags in [vec!["--compute-threads", "0"], vec!["--compute-threads", "65"],
        vec!["--compute-threads", "18446744073709551615"],
        vec!["--compute-threads", "2", "--cohort-chunks", "1"],
        vec!["--compute-threads", "4", "--cohort-chunks", "3"]] {
        let (options, args) = arguments(&flags);
        assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    }
    let (explicit, args) = arguments(&["--compute-threads", "1", "--cohort-chunks", "3"]);
    assert!(explicit.validate(&args).is_ok());
    let (plain, plain_args) = arguments(&[]);
    let (one, one_args) = arguments(&["--compute-threads", "1"]);
    assert_eq!(plain.validate(&plain_args).unwrap().common.minimum_memory_bytes,
        one.validate(&one_args).unwrap().common.minimum_memory_bytes);
}
