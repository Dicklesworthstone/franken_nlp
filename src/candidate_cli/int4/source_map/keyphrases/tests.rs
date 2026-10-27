//! UNRUN explicit mode, resource and source-policy fixtures.
use super::*;

fn arguments(extra: &[&str]) -> (MapArgs, Int4Args) {
    let command = super::super::tests::command(Kind::Keyphrases, extra);
    let Int4Task::SourceMap(options) = command.task else { panic!("wrong keyphrase map route"); };
    (options, command.args)
}
struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused keyphrase ranking read private source"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused keyphrase ranking published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused keyphrase ranking flushed output"); }
}

#[test]
fn raw_keyphrase_maps_keep_their_existing_default_and_explicit_ranking_changes_only_reduction() {
    let (raw, args) = arguments(&[]); let raw_limits = raw.validate(&args).unwrap();
    assert!(raw_limits.keyphrases.is_none()); assert_eq!(raw_limits.max_result_bytes, 16_777_216);
    let (ranked, args) = arguments(&["--reduce-keyphrases", "--document-keyphrases", "3"]);
    let limits = ranked.validate(&args).unwrap(); let aggregation = limits.keyphrases.unwrap();
    assert_eq!(aggregation.max_phrases, 3); assert_eq!(aggregation.max_result_bytes, 4_194_304);
    assert_eq!(limits.max_result_bytes, aggregation.max_result_bytes);
    assert_eq!(aggregation.aggregation.max_unique_phrases, 4096);
    assert_eq!(aggregation.aggregation.max_scan_work, 536_870_912);
    assert_eq!(raw_limits.mapping.max_model_work, limits.mapping.max_model_work);
    assert_eq!(raw_limits.mapping.max_mask_visits, limits.mapping.max_mask_visits);
    let budget = ranked.source.budget(&args, limits.common);
    let exact = "é\r\n上海 <tool_call>  phrase";
    let (source, task) = ranked.request(exact.to_owned(),
        Some(r#"{"max_phrases":7,"max_phrase_scalars":24}"#), budget, args.max_input_bytes).unwrap();
    assert_eq!(source, exact);
    let SourceMapTask::Keyphrases(options) = task else { panic!("wrong native task"); };
    assert_eq!(options.max_phrases, 7); assert_eq!(options.max_phrase_scalars, 24);
}
#[test]
fn reduction_controls_require_explicit_opt_in_and_are_rejected_by_other_task_families() {
    for flags in [vec!["--document-keyphrases", "3"], vec!["--max-unique-keyphrases", "10"],
        vec!["--max-keyphrase-evidence-spans", "20"], vec!["--max-keyphrase-value-bytes", "4096"],
        vec!["--max-keyphrase-scan-work", "1000"], vec!["--max-keyphrase-result-bytes", "4096"]] {
        let mut argv = vec!["candidate", "source-int4", "keyphrases-map", "--source-dir", "source",
            "--source-manifest", "manifest", "--memory-mib", "8192"];
        argv.extend(flags); assert!(crate::candidate_cli::definition().try_get_matches_from(argv).is_err());
    }
    for task in ["ner-map", "summarize-map", "answer-map"] {
        let mut argv = vec!["candidate", "source-int4", task, "--source-dir", "source",
            "--source-manifest", "manifest", "--memory-mib", "8192", "--reduce-keyphrases"];
        if task == "answer-map" { argv.extend(["--question", "question.txt"]); }
        assert!(crate::candidate_cli::definition().try_get_matches_from(argv).is_err());
    }
    let (mut raw, args) = arguments(&[]);
    raw.keyphrases.as_mut().unwrap().document_keyphrases = Some(3);
    assert_eq!(raw.validate(&args).unwrap_err(), CandidateError::Arguments);
    let (mut ranked, args) = arguments(&["--reduce-keyphrases"]); ranked.kind = Kind::Ner;
    assert_eq!(ranked.validate(&args).unwrap_err(), CandidateError::Arguments);
}
#[test]
fn zero_oversized_and_frontier_incompatible_limits_refuse_before_any_io() {
    for flags in [vec!["--document-keyphrases", "0"], vec!["--document-keyphrases", "4097"],
        vec!["--max-unique-keyphrases", "0"], vec!["--max-unique-keyphrases", "65537"],
        vec!["--max-keyphrase-evidence-spans", "0"], vec!["--max-keyphrase-evidence-spans", "1000001"],
        vec!["--max-keyphrase-value-bytes", "0"], vec!["--max-keyphrase-value-bytes", "16777217"],
        vec!["--max-keyphrase-scan-work", "0"], vec!["--max-keyphrase-scan-work", "1000000000001"],
        vec!["--max-keyphrase-result-bytes", "0"], vec!["--max-keyphrase-result-bytes", "16777217"]] {
        let mut flags = flags; flags.push("--reduce-keyphrases");
        let command = super::super::tests::command(Kind::Keyphrases, &flags);
        assert_eq!(command.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
}
#[test]
fn candidate_evidence_scan_and_complete_result_caps_are_separate_from_native_chunk_caps() {
    let (options, args) = arguments(&["--reduce-keyphrases", "--document-keyphrases", "5",
        "--max-unique-keyphrases", "100", "--max-keyphrase-evidence-spans", "500",
        "--max-keyphrase-value-bytes", "2097152", "--max-keyphrase-scan-work", "123456",
        "--max-keyphrase-result-bytes", "1048576"]);
    let limits = options.validate(&args).unwrap(); let ranked = limits.keyphrases.unwrap();
    assert_eq!(ranked.max_phrases, 5); assert_eq!(ranked.aggregation.max_unique_phrases, 100);
    assert_eq!(ranked.aggregation.max_evidence_spans, 500);
    assert_eq!(ranked.aggregation.max_value_bytes, 2_097_152);
    assert_eq!(ranked.aggregation.max_scan_work, 123456);
    assert_eq!(limits.max_result_bytes, 1_048_576);
    assert_eq!(limits.mapping.reduction.max_result_bytes, 16_777_216);
    assert_eq!(limits.mapping.verification.max_scan_steps, 67_108_864);
    assert_eq!(limits.common.result_bytes, 1_048_576);
}
#[test]
fn complete_aggregate_owners_and_zero_final_token_transcripts_are_memory_admitted() {
    let (options, mut args) = arguments(&["--reduce-keyphrases"]);
    let limits = options.validate(&args).unwrap(); let ranked = limits.keyphrases.unwrap();
    let budget = options.source.budget(&args, limits.common);
    let base = super::super::workspace_bytes(budget, limits.mapping,
        limits.mapping.chunks.max_chunks, limits.reduction_reserve_bytes).unwrap();
    let aggregate = ranked.aggregation;
    let extra = aggregate.max_value_bytes as u64 * 8 + aggregate.max_unique_phrases as u64 * 512
        + aggregate.max_evidence_spans as u64 * std::mem::size_of::<VerifiedSourceSpan>() as u64 * 2;
    assert_eq!(limits.workspace_bytes, base + extra);
    let single = options.source.validate(&args).unwrap();
    let input = args.max_input_bytes as u64 + 1 + limits.common.preparation_bytes;
    let output = ranked.max_result_bytes as u64 * 4 + 4096;
    assert_eq!(limits.common.minimum_memory_bytes, single.minimum_memory_bytes + input + limits.workspace_bytes + output);
    args.memory_mib = limits.common.minimum_memory_bytes.saturating_sub(1) / MIB;
    assert!(options.validate(&args).is_err());
    args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
    assert!(options.validate(&args).is_ok());
}
#[test]
fn source_conversion_and_pinned_snapshot_use_the_same_ranked_document_policy() {
    let (source, args) = arguments(&["--reduce-keyphrases", "--document-keyphrases", "5"]);
    let original = source.validate(&args).unwrap(); let digest = "ab".repeat(32);
    let matches = crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "source-int4", "keyphrases-map", "--snapshot", "candidate.int4",
        "--expected-model-sha256", &digest, "--memory-mib", "8192", "--reduce-keyphrases", "--document-keyphrases", "5",
    ]).unwrap();
    let CandidateCommand::SourceInt4(c) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong ranked snapshot route"); };
    let Int4Task::SourceMap(options) = c.task else { panic!("wrong ranked snapshot task"); };
    let loaded = options.validate(&c.args).unwrap();
    assert!(loaded.common.snapshot_pin.is_some());
    assert_eq!(loaded.keyphrases.unwrap().max_phrases, 5);
    assert_eq!(loaded.mapping.max_model_work, original.mapping.max_model_work);
    assert_eq!(loaded.workspace_bytes, original.workspace_bytes);
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn missing_runtime_refuses_ranked_mode_before_options_source_or_model_access() {
    let command = super::super::tests::command(Kind::Keyphrases, &["--reduce-keyphrases"]);
    assert_eq!(command.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Unavailable));
}
