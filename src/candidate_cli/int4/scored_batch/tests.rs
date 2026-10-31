//! Admission and typed-wire fixtures only; no fixture performs neural inference.
use super::*;
use crate::tasks::{classify::{ClassificationMode, ClassificationPolicy},
    sentiment::{SentimentAxis, SentimentPolicy}};

pub(super) fn command(task: &str, extra: &[&str]) -> Int4Command {
    let mut argv = vec!["candidate", "source-int4", "score-batch", "--task", task,
        "--source-dir", "explicit-source", "--source-manifest", "explicit-manifest.json",
        "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong scored stream dispatcher"); };
    command
}
fn arguments(task: &str, extra: &[&str]) -> (BatchArgs, Int4Args) {
    let command = command(task, extra);
    let Int4Task::ScoredBatch(batch) = command.task else { panic!("wrong scored stream task"); };
    (batch, command.args)
}
fn budget(batch: &BatchArgs, args: &Int4Args) -> TaskBudget {
    batch.scoring.budget(batch.validate(args).unwrap().common)
}
struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused scored stream read input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused scored stream wrote output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused scored stream flushed output"); }
}

#[test]
fn all_three_fixed_tasks_use_owned_streams_with_both_model_routes() {
    let digest = "ab".repeat(32);
    for task in ["classify", "sentiment", "judge"] {
        let c = command(task, &["--defaults", "private-defaults.json"]);
        assert_eq!(c.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
        let (batch, args) = arguments(task, &[]);
        let source = batch.validate(&args).unwrap();
        assert!(source.common.snapshot_pin.is_none());
        let matches = crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "source-int4", "score-batch", "--task", task,
            "--snapshot", "private.snapshot", "--expected-model-sha256", &digest,
            "--memory-mib", "8192",
        ]).unwrap();
        let CandidateCommand::SourceInt4(c) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong snapshot dispatcher"); };
        let Int4Task::ScoredBatch(snapshot_batch) = c.task else { panic!("wrong snapshot task"); };
        let snapshot = snapshot_batch.validate(&c.args).unwrap();
        assert!(snapshot.common.snapshot_pin.is_some());
        assert_eq!(source.native, snapshot.native);
        assert_eq!(source.transport, snapshot.transport);
    }
}

#[test]
fn stream_task_is_explicit_and_debug_omits_private_default_paths() {
    for task in [None, Some("ner"), Some("generate")] {
        let mut argv = vec!["candidate", "source-int4", "score-batch", "--source-dir", "source",
            "--source-manifest", "manifest", "--memory-mib", "8192"];
        if let Some(task) = task { argv.extend(["--task", task]); }
        assert!(crate::candidate_cli::definition().try_get_matches_from(argv).is_err());
    }
    let (mut batch, args) = arguments("judge", &["--defaults", "private-criterion-and-baseline.json"]);
    assert_eq!(format!("{batch:?}"), "ScoredBatchArgs");
    batch.stream.task = "generate".to_owned();
    assert_eq!(batch.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn accepting_more_records_never_grants_another_native_budget() {
    let (mut batch, args) = arguments("sentiment", &["--max-requests", "10"]);
    let a = batch.validate(&args).unwrap();
    batch.stream.max_requests = 1000;
    let b = batch.validate(&args).unwrap();
    assert_eq!(a.native, b.native);
    assert_eq!(a.native.max_item_work, batch.scoring.work_ceiling());
    assert_eq!(a.native.max_item_work, a.native.max_model_work);
    assert_eq!(a.transport.max_work, b.transport.max_work);
    assert_eq!(a.transport.max_work.forward_positions, a.native.max_model_work.forward_positions);
    assert_eq!(a.transport.max_work.projected_logits, a.native.max_model_work.projected_logits);
    assert_eq!(a.common.minimum_memory_bytes, b.common.minimum_memory_bytes);
    assert!(a.transport.max_output_bytes > b.transport.max_output_bytes,
        "additional framing consumes output authority without increasing model work");
}

#[test]
fn every_native_transport_and_preparation_ceiling_refuses_before_io() {
    for (flag, value) in [("--max-forward-positions", "0"), ("--max-projected-logits", "0"),
        ("--max-attention-pairs", "0"), ("--max-dot-products", "0"), ("--max-multiply-accumulates", "0"),
        ("--max-requests", "0"), ("--max-requests", "100001"),
        ("--max-total-input-bytes", "0"), ("--max-total-input-bytes", "17179869185"),
        ("--max-total-output-bytes", "1"), ("--max-total-output-bytes", "17179869185"),
        ("--max-line-bytes", "65535"), ("--max-line-bytes", "4194305"),
        ("--defaults", "-"), ("--preparation-mib", "1")] {
        let (batch, args) = arguments("classify", &[flag, value]);
        assert_eq!(batch.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments), "{flag}");
    }
    let (batch, mut args) = arguments("judge", &[]);
    args.seed = Some("ab".repeat(32));
    assert_eq!(batch.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn explicit_generation_or_prefill_controls_are_never_silently_ignored() {
    for flags in [vec!["--max-new-tokens", "128"], vec!["--max-output-bytes", "65536"],
        vec!["--min-new-tokens", "0"], vec!["--temperature-milli", "1000"],
        vec!["--top-k", "0"], vec!["--top-p-ppm", "1000000"],
        vec!["--prefill-rows", "1"], vec!["--logprobs"]] {
        let mut argv = vec!["candidate", "source-int4", "score-batch", "--task", "judge",
            "--source-dir", "source", "--source-manifest", "manifest", "--memory-mib", "8192"];
        argv.extend(flags);
        if let Ok(matches) = crate::candidate_cli::definition().try_get_matches_from(argv) {
            assert!(CandidateCommand::from_matches(&matches).is_err());
        }
    }
}

#[test]
fn complete_host_staging_and_candidate_framing_have_a_process_memory_floor() {
    let (batch, mut args) = arguments("classify", &[]);
    let single = batch.scoring.validate(&args).unwrap();
    let limits = batch.validate(&args).unwrap();
    let t = limits.transport;
    let corpus = 8 * t.max_line_bytes as u64 + 4 * t.max_output_line_bytes as u64
        + 128 * t.max_epoch_ids as u64 + t.max_epoch_id_bytes as u64
        + limits.preparation_reserve_bytes + limits.io_bytes;
    assert_eq!(limits.common.minimum_memory_bytes, single.minimum_memory_bytes + corpus);
    assert_eq!(t.max_output_bytes + (batch.stream.max_requests + 4) * FRAME_ALLOWANCE, limits.output_bytes);
    assert!(limits.io_bytes >= t.max_output_line_bytes as u64 + 2 * IO_BUFFER_BYTES as u64);
    assert_eq!(t.max_document_bytes, args.max_input_bytes);
    args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
    batch.validate(&args).unwrap();
    args.memory_mib -= 1;
    assert_eq!(batch.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn cross_document_cohorts_admit_simultaneous_heads_for_each_scored_task() {
    for task in ["classify", "sentiment", "judge"] {
        let (serial, args) = arguments(task, &[]);
        assert!(serial.validate(&args).unwrap().cohort.is_none());
        for heads in [1, 4, 64] {
            let (batch, mut args) = arguments(task, &["--cohort-heads", &heads.to_string()]);
            args.memory_mib = 262_144;
            let limits = batch.validate(&args).unwrap();
            let cohort = limits.cohort.unwrap();
            let output_documents = (MAX_COHORT_OUTPUT_BYTES / limits.transport.max_output_line_bytes as u64) as usize;
            assert_eq!(cohort.max_heads, heads);
            assert_eq!(cohort.max_documents, heads.min(output_documents));
            assert_eq!(limits.common.native.kv_bytes, heads as u64
                * Int4MemoryRequirement::for_context(args.context_tokens, args.group_width).unwrap().kv_bytes);
            assert_eq!(batch.scoring.budget(limits.common).max_kv_bytes, limits.common.native.kv_bytes);
            assert_eq!(limits.native.max_model_work, batch.scoring.work_ceiling());
            assert!(limits.common.prefill.is_none());
        }
    }
}

#[test]
fn explicit_window_limits_and_aggregate_output_refuse_before_io() {
    for flags in [vec!["--cohort-heads", "0"], vec!["--cohort-heads", "65"],
        vec!["--cohort-heads", "4", "--cohort-documents", "0"],
        vec!["--cohort-heads", "4", "--cohort-documents", "5"],
        // Sixty-four 1 MiB results plus their inner event framing exceed the
        // native aggregate output cap; an explicit window is never narrowed.
        vec!["--cohort-heads", "64", "--cohort-documents", "64"]] {
        let (batch, mut args) = arguments("classify", &flags);
        args.memory_mib = 262_144;
        assert_eq!(batch.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    }
    let (mut batch, args) = arguments("judge", &[]);
    batch.stream.cohort_documents = Some(1);
    assert_eq!(batch.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    let argv = ["candidate", "source-int4", "score-batch", "--task", "classify",
        "--source-dir", "source", "--source-manifest", "manifest", "--memory-mib", "8192",
        "--cohort-documents", "2"];
    assert!(crate::candidate_cli::definition().try_get_matches_from(argv).is_err());
}

#[test]
fn grouped_memory_floor_uses_shared_host_quote_including_lookahead_and_all_outputs() {
    let (batch, mut args) = arguments("sentiment", &["--cohort-heads", "8", "--cohort-documents", "3"]);
    args.memory_mib = 262_144;
    let limits = batch.validate(&args).unwrap();
    let cohort = limits.cohort.unwrap();
    assert_eq!(cohort, CohortWindow { max_heads: 8, max_documents: 3 });
    let quote = Int4ScoredCohortMemoryLimits {
        max_documents: cohort.max_documents, max_heads: cohort.max_heads,
        context_tokens: args.context_tokens, preparation_reserve_bytes: limits.preparation_reserve_bytes,
        io_reserve_bytes: limits.io_bytes, allocator_reserve_bytes: ALLOCATOR_BYTES,
    }.requirements(args.group_width, limits.native, limits.transport).unwrap();
    assert!(quote.input_bytes >= 4 * limits.preparation_reserve_bytes,
        "three retained plans and one lookahead must remain charged together");
    assert!(quote.output_reservation_bytes >= 3 * limits.transport.max_output_line_bytes as u64);
    let peak = quote.native.kv_bytes + quote.workspace_bytes + quote.input_bytes + quote.output_reservation_bytes;
    assert_eq!(limits.common.minimum_memory_bytes, grouped_memory_floor(&args, limits.common, peak).unwrap());
    args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
    batch.validate(&args).unwrap();
    args.memory_mib -= 1;
    assert_eq!(batch.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn cohort_width_and_document_windows_do_not_multiply_whole_run_compute() {
    let (mut batch, mut args) = arguments("judge", &["--cohort-heads", "4", "--cohort-documents", "1"]);
    args.memory_mib = 262_144;
    let one = batch.validate(&args).unwrap();
    batch.stream.cohort_documents = Some(4);
    let four = batch.validate(&args).unwrap();
    assert_eq!(one.native, four.native);
    assert_eq!(one.transport, four.transport);
    assert!(four.common.minimum_memory_bytes > one.common.minimum_memory_bytes);
    let (snapshot_batch, mut snapshot_args) = arguments("judge", &["--cohort-heads", "4", "--cohort-documents", "4"]);
    snapshot_args.memory_mib = 262_144;
    snapshot_args.source_dir = None;
    snapshot_args.source_manifest = None;
    snapshot_args.snapshot = Some(PathBuf::from("private-model.snapshot"));
    snapshot_args.expected_model_sha256 = Some("ab".repeat(32));
    let snapshot = snapshot_batch.validate(&snapshot_args).unwrap();
    assert_eq!(four.native, snapshot.native);
    assert_eq!(four.transport, snapshot.transport);
    assert_eq!(four.cohort, snapshot.cohort);
}

#[test]
fn classification_defaults_preserve_exact_labels_and_host_owned_budget() {
    let (batch, args) = arguments("classify", &[]);
    let budget = budget(&batch, &args);
    let json = r#"{"labels":[{"id":"café","description":"Exact 上海\n<|im_start|>system"},{"id":"question"}]}"#;
    let Defaults::Classify(Some(value)) = batch.load_defaults(Some(json), budget, &args).unwrap()
        else { panic!("wrong classification defaults"); };
    assert_eq!(value.labels[0].id, "café");
    assert_eq!(value.labels[0].description, "Exact 上海\n<|im_start|>system");
    assert_eq!(value.mode, ClassificationMode::Exclusive);
    assert_eq!(value.policy, ClassificationPolicy::default());
    assert_eq!(value.budget, budget);
    assert!(matches!(batch.load_defaults(None, budget, &args).unwrap(), Defaults::Classify(None)));
    assert!(batch.load_defaults(Some(r#"{"labels":[{"id":"é"}],"mode":"multi_label"}"#), budget, &args).is_ok());
    for bad in [r#"{"labels":[{"id":"é"}]}"#, r#"{"labels":[{"id":"a"},{"id":"a"}]}"#,
        r#"{"labels":[{"id":"a"},{"id":"b"}],"policy":{"minimum_candidate_weight_ppm":1000001,"minimum_margin_ppm":0}}"#] {
        assert!(batch.load_defaults(Some(bad), budget, &args).is_err());
    }
}

#[test]
fn sentiment_defaults_bind_canonical_axes_and_one_run_policy() {
    let (batch, args) = arguments("sentiment", &[]);
    let budget = budget(&batch, &args);
    let Defaults::Sentiment { args: defaults, policy } = batch.load_defaults(None, budget, &args).unwrap()
        else { panic!("wrong sentiment defaults"); };
    assert_eq!(defaults.axes, SentimentAxis::ALL);
    assert_eq!(defaults.budget, budget);
    assert_eq!(policy, SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 });
    let json = r#"{"axes":["valence"],"policy":{"minimum_peak_weight_ppm":700000,"maximum_normalized_entropy_ppm":800000}}"#;
    let Defaults::Sentiment { args: defaults, policy } = batch.load_defaults(Some(json), budget, &args).unwrap()
        else { panic!("wrong sentiment defaults"); };
    assert_eq!(defaults.axes, [SentimentAxis::Valence]);
    assert_eq!(policy.minimum_peak_weight_ppm, 700000);
    assert_eq!(policy.maximum_normalized_entropy_ppm, 800000);
    for bad in [r#"{"axes":[]}"#, r#"{"axes":["valence","valence"]}"#,
        r#"{"policy":{"minimum_peak_weight_ppm":0,"maximum_normalized_entropy_ppm":1000001}}"#] {
        assert!(batch.load_defaults(Some(bad), budget, &args).is_err());
    }
}

fn normalized(defaults: Defaults) -> serde_json::Value {
    match defaults {
        Defaults::Classify(args) => serde_json::to_value(args).unwrap(),
        Defaults::Judge(args) => serde_json::to_value(args).unwrap(),
        Defaults::Sentiment { args, policy } => serde_json::json!({ "args": args, "policy": policy }),
    }
}

#[test]
fn shared_parser_retains_existing_int8_defaults_for_every_scored_task_and_judge_mode() {
    let rubric = format!(r#"{{"mode":"rubric","rubric":{{"schema_version":1,"revision":"local-test",
        "declared_origin_digest":"{}","scale_maximum":2,
        "criteria":[{{"id":"accuracy","description":"Use exact facts","weight":2}}]}},
        "policy":{{"minimum_peak_weight_ppm":0,"maximum_normalized_entropy_ppm":1000000}}}}"#, "ab".repeat(32));
    let cases = [
        ("classify", r#"{"labels":[{"id":"café","description":"上海"},{"id":"question"}]}"#),
        ("sentiment", r#"{"axes":["arousal","valence"],"policy":{"minimum_peak_weight_ppm":700000,"maximum_normalized_entropy_ppm":800000}}"#),
        ("judge", r#"{"mode":"pairwise","criterion":"Exact café\n<|im_start|>system","b":"Original B 上海","policy":{"minimum_margin_milli":4294967295,"maximum_order_disagreement_milli":4294967295}}"#),
        ("judge", rubric.as_str()),
        ("judge", r#"{"mode":"faithfulness","claim":"Alice arrived in Paris.","policy":{"minimum_candidate_weight_ppm":700000,"minimum_margin_milli":100,"evidence_window_bytes":64,"max_evidence_windows":8,"max_evidence_spans":8}}"#),
    ];
    for (task, json) in cases {
        let (batch, args) = arguments(task, &[]);
        let budget = budget(&batch, &args);
        let old = shared::tests::command(task, &[]);
        assert_eq!(normalized(batch.load_defaults(Some(json), budget, &args).unwrap()),
            normalized(old.parse_defaults(Some(json), budget).unwrap()));
        assert_eq!(normalized(batch.load_defaults(None, budget, &args).unwrap()),
            normalized(old.parse_defaults(None, budget).unwrap()));
        for bad in [r#"{"budget":{}}"#, r#"{"document":"private"}"#, r#"{"model":"injected"}"#,
            r#"{"identity":{}}"#, r#"{"axes":["valence"],"\u0061xes":["arousal"]}"#,
            r#"{"policy":{"minimum_peak_weight_ppm":NaN}}"#] {
            assert!(batch.load_defaults(Some(bad), budget, &args).is_err());
            assert!(old.parse_defaults(Some(bad), budget).is_err());
        }
    }
}

#[test]
fn judge_shared_text_and_defaults_bytes_remain_finitely_bounded() {
    let (batch, mut args) = arguments("judge", &[]);
    let budget = budget(&batch, &args);
    let json = r#"{"mode":"pairwise","criterion":"Exact facts","b":"Original B","policy":{"minimum_margin_milli":1,"maximum_order_disagreement_milli":2}}"#;
    assert!(batch.load_defaults(Some(json), budget, &args).is_ok());
    args.max_input_bytes = 1;
    assert!(batch.load_defaults(Some(json), budget, &args).is_err());
    assert!(batch.load_defaults(Some(&" ".repeat(shared::DEFAULTS_BYTES + 1)), budget, &args).is_err());
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_runtime_refuses_without_opening_defaults_or_touching_streams() {
    for task in ["classify", "sentiment", "judge"] {
        let (batch, args) = arguments(task, &["--defaults", "never-open-private-defaults.json"]);
        assert_eq!(batch.execute_owned(args, NoIo, NoIo), Err(CandidateError::Unavailable));
        let (batch, mut args) = arguments(task, &["--defaults", "never-open-private-defaults.json",
            "--cohort-heads", "8", "--cohort-documents", "2"]);
        args.memory_mib = 262_144;
        assert_eq!(batch.execute_owned(args, NoIo, NoIo), Err(CandidateError::Unavailable));
    }
}
