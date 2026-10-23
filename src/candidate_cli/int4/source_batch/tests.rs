use super::*;
use crate::tasks::answer::{AnswerOptions, AnswerPassage};

pub(super) fn command(name: &str, extra: &[&str]) -> Int4Command {
    let mut argv = vec!["candidate", "source-int4", name, "--source-dir", "explicit-source",
        "--source-manifest", "explicit-manifest.json", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong source corpus dispatcher"); };
    command
}
fn arguments(name: &str, extra: &[&str]) -> (BatchArgs, Int4Args) {
    let command = command(name, extra);
    let Int4Task::SourceBatch(batch) = command.task else { panic!("wrong source corpus task"); };
    (batch, command.args)
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused source corpus read input"); }
}
struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused source corpus wrote output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused source corpus flushed output"); }
}

#[test]
fn all_four_batches_require_owned_execution_and_share_both_model_routes() {
    let digest = "ab".repeat(32);
    for name in ["ner-batch", "keyphrases-batch", "summarize-batch", "answer-batch"] {
        let command = command(name, &[]);
        assert_eq!(command.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
        let (batch, args) = arguments(name, &[]);
        let source = batch.validate(&args).unwrap();
        assert_eq!(batch.kind.name(), name.trim_end_matches("-batch"));
        assert!(source.common.snapshot_pin.is_none());
        let matches = crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "source-int4", name, "--snapshot", "local.snapshot",
            "--expected-model-sha256", &digest, "--memory-mib", "8192",
        ]).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong snapshot corpus dispatcher"); };
        let Int4Task::SourceBatch(batch) = command.task else { panic!("wrong snapshot corpus task"); };
        let snapshot = batch.validate(&command.args).unwrap();
        assert!(snapshot.common.snapshot_pin.is_some());
        assert_eq!(source.native.max_model_work, snapshot.native.max_model_work);
        assert_eq!(source.transport, snapshot.transport);
    }
}

#[test]
fn defaults_are_typed_and_qa_never_invents_a_passage_set() {
    for name in ["ner-batch", "keyphrases-batch", "summarize-batch", "answer-batch"] {
        let (batch, args) = arguments(name, &[]);
        let limits = batch.validate(&args).unwrap();
        let budget = batch.source.budget(&args, limits.common);
        let defaults = batch.load_defaults(None, budget, args.max_input_bytes).unwrap();
        if batch.kind == Kind::Answer { assert!(defaults.is_none()); }
        else {
            let json = serde_json::to_value(defaults.unwrap()).unwrap();
            assert_eq!(json["task"], batch.kind.name());
            assert_eq!(json["budget"]["max_kv_bytes"], budget.max_kv_bytes);
        }
    }
}

#[test]
fn defaults_refuse_other_tasks_budget_increases_and_smaller_fixed_kv_authority() {
    let (batch, args) = arguments("ner-batch", &[]);
    let budget = batch.source.budget(&args, batch.validate(&args).unwrap().common);
    let defaults = batch.load_defaults(None, budget, args.max_input_bytes).unwrap().unwrap();
    let encoded = serde_json::to_string(&defaults).unwrap();
    assert!(batch.load_defaults(Some(&encoded), budget, args.max_input_bytes).is_ok());
    let wrong = SourceBatchArgs::Summarize { options: SummaryOptions::default(), budget };
    assert!(batch.load_defaults(Some(&serde_json::to_string(&wrong).unwrap()), budget, args.max_input_bytes).is_err());
    for axis in 0..5 {
        let mut changed = budget;
        match axis { 0 => changed.max_input_tokens += 1, 1 => changed.max_output_tokens += 1,
            2 => changed.max_output_bytes += 1, 3 => changed.max_grammar_states += 1, _ => changed.max_kv_bytes -= 1 }
        let changed = SourceBatchArgs::Ner { options: NerOptions::default(), budget: changed };
        assert!(batch.load_defaults(Some(&serde_json::to_string(&changed).unwrap()), budget, args.max_input_bytes).is_err());
    }
    let duplicate = encoded.replacen("\"task\":", "\"task\":\"ner\",\"task\":", 1);
    assert!(batch.load_defaults(Some(&duplicate), budget, args.max_input_bytes).is_err());
    assert!(batch.load_defaults(Some(&" ".repeat(MAX_SOURCE_ARGUMENT_BYTES + 1)), budget, args.max_input_bytes).is_err());
}

#[test]
fn qa_defaults_preserve_separate_passages_and_reject_bad_evidence_ids() {
    let (batch, args) = arguments("answer-batch", &[]);
    let budget = batch.source.budget(&args, batch.validate(&args).unwrap().common);
    let passages = vec![AnswerPassage { id: "first".to_owned(), text: " Alice é 上海 ".to_owned() },
        AnswerPassage { id: "second".to_owned(), text: "Alice".to_owned() }];
    let defaults = SourceBatchArgs::Answer { passages: passages.clone(), options: AnswerOptions::default(), budget };
    let restored = batch.load_defaults(Some(&serde_json::to_string(&defaults).unwrap()), budget, args.max_input_bytes)
        .unwrap().unwrap();
    let SourceBatchArgs::Answer { passages: actual, .. } = restored else { panic!("wrong QA task"); };
    assert_eq!(actual[0].text, passages[0].text); assert_eq!(actual[1].id, passages[1].id);
    for bad in [vec![], vec![passages[0].clone(), passages[0].clone()],
        vec![AnswerPassage { id: "\n".to_owned(), text: "private".to_owned() }],
        vec![AnswerPassage { id: "x".repeat(129), text: "private".to_owned() }]] {
        let bad = SourceBatchArgs::Answer { passages: bad, options: AnswerOptions::default(), budget };
        assert!(batch.load_defaults(Some(&serde_json::to_string(&bad).unwrap()), budget, args.max_input_bytes).is_err());
    }
    assert!(batch.load_defaults(Some(&serde_json::to_string(&defaults).unwrap()), budget, 1).is_err());
}

#[test]
fn whole_run_scaling_preserves_independent_attention_and_group_dependent_dots() {
    let (mut batch, mut args) = arguments("summarize-batch", &["--max-requests", "8"]);
    let a = batch.validate(&args).unwrap();
    batch.stream.max_requests = 16;
    let b = batch.validate(&args).unwrap();
    assert_eq!(b.native.max_model_work, scale_work(a.native.max_model_work, 2).unwrap());
    assert_eq!(b.native.masks.max_visits_per_run, a.native.masks.max_visits_per_run * 2);
    assert_eq!(a.common.native.kv_bytes, b.common.native.kv_bytes);
    assert_eq!(a.common.minimum_memory_bytes, b.common.minimum_memory_bytes);
    assert_eq!(a.native.max_model_work.forward_positions, 8 * (args.context_tokens as u64 - 1));
    assert_eq!(a.transport.max_work.forward_positions, a.native.max_model_work.forward_positions);
    assert_eq!(a.transport.max_work.projected_logits, a.native.max_model_work.projected_logits);
    args.group_width = 16;
    let small = batch.validate(&args).unwrap().native.max_model_work;
    args.group_width = 128;
    let large = batch.validate(&args).unwrap().native.max_model_work;
    assert!(small.projections.dot_products > large.projections.dot_products);
    assert_eq!(small.projections.multiply_accumulates, large.projections.multiply_accumulates);
    assert_eq!(small.attention_pairs, large.attention_pairs);
}

#[test]
fn framing_and_hosted_staging_are_charged_for_the_whole_stream() {
    let (batch, args) = arguments("ner-batch", &["--max-requests", "8"]);
    let single = batch.source.validate(&args).unwrap();
    let limits = batch.validate(&args).unwrap();
    let t = limits.transport;
    let host = t.max_line_bytes as u64 * 8 + t.max_output_line_bytes as u64 * 4
        + t.max_epoch_ids as u64 * 128 + t.max_epoch_id_bytes as u64
        + limits.preparation_reserve_bytes + limits.io_bytes;
    assert_eq!(limits.common.minimum_memory_bytes, single.minimum_memory_bytes + host);
    assert_eq!(limits.transport.max_output_bytes + (8 + 4) * FRAME_ALLOWANCE, limits.output_bytes);
    for axis in 0..5 {
        let mut work = limits.native.max_model_work;
        match axis { 0 => work.forward_positions = u64::MAX, 1 => work.projected_logits = u64::MAX,
            2 => work.attention_pairs = u64::MAX, 3 => work.projections.dot_products = u64::MAX,
            _ => work.projections.multiply_accumulates = u64::MAX }
        assert!(scale_work(work, 2).is_err());
    }
    assert!(scale_work(limits.native.max_model_work, 0).is_err());
}

#[test]
fn invalid_limits_and_programmatic_policy_drift_refuse_before_all_io() {
    for flags in [vec!["--max-requests", "0"], vec!["--max-requests", "100001"],
        vec!["--max-total-input-bytes", "0"], vec!["--max-total-output-bytes", "1"],
        vec!["--max-line-bytes", "65535"], vec!["--max-line-bytes", "4194305"],
        vec!["--max-mask-node-visits", "1"], vec!["--defaults", "-"], vec!["--preparation-mib", "1"]] {
        let (batch, args) = arguments("ner-batch", &flags);
        assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
    }
    let (batch, mut args) = arguments("answer-batch", &[]); args.seed = Some("ab".repeat(32));
    assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
}

#[test]
fn source_options_and_text_controls_cannot_be_silently_ignored_by_corpus_routes() {
    for flags in [vec!["--options", "options.json"], vec!["--prefill-rows", "1"],
        vec!["--max-output-bytes", "65536"], vec!["--min-new-tokens", "0"], vec!["--logprobs"]] {
        let mut argv = vec!["candidate", "source-int4", "ner-batch", "--source-dir", "source",
            "--source-manifest", "manifest", "--memory-mib", "8192"];
        argv.extend(flags);
        let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
        assert!(CandidateCommand::from_matches(&matches).is_err());
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_runtime_refuses_owned_streams_without_opening_defaults_or_models() {
    for name in ["ner-batch", "keyphrases-batch", "summarize-batch", "answer-batch"] {
        let (batch, args) = arguments(name, &["--defaults", "private-defaults.json"]);
        assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Unavailable));
    }
}
