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
        vec!["--max-requests", "100001"], vec!["--max-total-input-bytes", "0"],
        vec!["--max-total-input-bytes", "17179869185"], vec!["--max-total-output-bytes", "1"],
        vec!["--max-total-output-bytes", "17179869185"], vec!["--max-line-bytes", "1048575"],
        vec!["--max-line-bytes", "4194305"], vec!["--defaults", "-"],
        vec!["--max-chunks", "0"], vec!["--max-chunks", "257"],
        vec!["--max-chunk-bytes", "3"], vec!["--max-tokenizer-calls", "0"],
        vec!["--max-map-result-bytes", "0"], vec!["--max-retained-result-bytes", "0"],
        vec!["--minimum-classified-coverage-ppm", "1000001"],
        vec!["--timeout-seconds", "0"], vec!["--max-checkpoints", "1"],
        vec!["--preparation-mib", "1"]] {
        let (options, args) = arguments(&flags);
        assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments), "{flags:?}");
    }
    let (mut options, args) = arguments(&[]);
    options.document.mapping.cohort_chunks = Some(1);
    assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    let (options, mut args) = arguments(&[]);
    args.seed = Some("ab".repeat(32));
    assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn explicit_generation_chunk_or_cross_document_schedules_are_not_ignored() {
    for flags in [vec!["--cohort-chunks", "1"], vec!["--cohort-heads", "1"],
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

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn missing_host_feature_refuses_without_opening_defaults_or_source_input() {
    let (options, args) = arguments(&["--defaults", "never-open-private-defaults.json"]);
    assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Unavailable));
}
