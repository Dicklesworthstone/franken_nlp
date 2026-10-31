use super::*;

pub(super) fn command(kind: Kind, extra: &[&str]) -> Int4Command {
    let name = match kind { Kind::Ner => "ner-map", Kind::Keyphrases => "keyphrases-map",
        Kind::Summarize => "summarize-map", Kind::Answer => "answer-map" };
    let mut argv = vec!["candidate", "source-int4", name, "--source-dir", "explicit-source",
        "--source-manifest", "explicit-manifest.json", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong source map dispatcher"); };
    command
}
fn arguments(kind: Kind, extra: &[&str]) -> (MapArgs, Int4Args) {
    let command = command(kind, extra);
    let Int4Task::SourceMap(map) = command.task else { panic!("wrong source map task"); };
    (map, command.args)
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused source map reached input"); }
}
const KINDS: [Kind; 3] = [Kind::Ner, Kind::Keyphrases, Kind::Summarize];

#[test]
fn every_map_uses_whole_document_limits_on_source_and_pinned_snapshot_routes() {
    for kind in KINDS {
        let (map, args) = arguments(kind, &[]);
        let source = map.validate(&args).unwrap();
        assert_eq!(args.max_input_bytes, 1_048_576);
        assert_eq!(source.mapping.chunks.max_input_bytes, args.max_input_bytes);
        assert_eq!(source.mapping.chunks.context_tokens, args.context_tokens);
        assert_eq!(source.common.result_bytes, 1_048_576);
        assert_eq!(source.max_result_bytes, 16_777_216);
        assert_eq!(source.common.preparation_bytes, 1024 * MIB);
        assert_eq!(source.mapping.mask_visits_per_chunk, map.source.mask_visits());
        assert_eq!(source.mapping.max_mask_visits, 128_000_000_000);
        assert_eq!(source.mapping.reduction.map_batch_chunks, 1);
        assert_eq!(source.cohort_chunks, None);
        assert!(source.cohort.is_none());
        assert!(source.common.prefill.is_none());
        assert_eq!(source.summary.is_some(), kind == Kind::Summarize);
        let digest = "ab".repeat(32);
        let mut definition = definitions().into_iter().find(|c| c.get_name().starts_with(kind.name())).unwrap();
        let name = definition.get_name().to_owned();
        let matches = definition.try_get_matches_from_mut([name.as_str(),
            "--snapshot", "local.snapshot", "--expected-model-sha256", &digest, "--memory-mib", "8192"]).unwrap();
        let map = MapArgs::from_matches(kind, &matches).unwrap();
        let args = Int4Args::from_arg_matches(&matches).unwrap();
        let snapshot = map.validate(&args).unwrap();
        assert!(snapshot.common.snapshot_pin.is_some());
        assert_eq!(source.mapping.max_model_work, snapshot.mapping.max_model_work);
        assert_eq!(source.workspace_bytes, snapshot.workspace_bytes);
    }
    assert!(definition().try_get_matches_from(["source-int4", "answer-map"]).is_err());
}

#[test]
fn chunk_cohorts_preserve_task_budgets_and_all_document_authority() {
    for (kind, mode) in [(Kind::Ner, &[][..]), (Kind::Keyphrases, &[][..]),
        (Kind::Summarize, &["--max-summary-bullets", "3"][..]),
        (Kind::Keyphrases, &["--reduce-keyphrases", "--document-keyphrases", "3"][..]),
        (Kind::Answer, &["--question", "private-question.txt"][..])] {
        let mut flags = mode.to_vec(); flags.extend_from_slice(&["--max-result-bytes", "65536"]);
        let (serial, mut args) = arguments(kind, &flags);
        args.memory_mib = 262_144;
        let original = serial.validate(&args).unwrap();
        let budget = serial.source.budget(&args, original.common);
        for chunks in [1, 2, 8, 64] {
            let mut grouped = serial.clone(); grouped.cohort_chunks = Some(chunks);
            let limits = grouped.validate(&args).unwrap();
            assert_eq!(limits.cohort_chunks, Some(chunks));
            assert_eq!(limits.common.native, original.common.native);
            assert_eq!(grouped.source.budget(&args, limits.common), budget);
            assert_eq!(limits.mapping.chunks, original.mapping.chunks);
            assert_eq!(limits.mapping.max_model_work, original.mapping.max_model_work);
            assert_eq!(limits.mapping.max_mask_visits, original.mapping.max_mask_visits);
            assert_eq!(limits.mapping.mask_visits_per_chunk, original.mapping.mask_visits_per_chunk);
            assert_eq!(limits.mapping.mask_limits.max_trie_node_visits,
                original.mapping.mask_limits.max_trie_node_visits);
            assert_eq!(limits.mapping.verification, original.mapping.verification);
            assert_eq!(limits.workspace_bytes, original.workspace_bytes);
            assert_eq!(grouped.question, serial.question);
            let (got, want) = (grouped.planning(&args), serial.planning(&args));
            assert_eq!((got.max_input_bytes, got.max_context_tokens, got.max_passages),
                (want.max_input_bytes, want.max_context_tokens, want.max_passages));
            assert_eq!(limits.max_result_bytes, original.max_result_bytes);
            if let Some(summary) = limits.summary {
                let expected = original.summary.unwrap();
                assert_eq!(summary.max_bullets, 3);
                assert_eq!(summary.max_result_bytes, expected.max_result_bytes);
                assert_eq!(summary.aggregation, expected.aggregation);
            }
            if let Some(keyphrases) = limits.keyphrases {
                let expected = original.keyphrases.unwrap();
                assert_eq!(keyphrases.max_phrases, 3);
                assert_eq!(keyphrases.max_result_bytes, expected.max_result_bytes);
                let (got, want) = (keyphrases.aggregation, expected.aggregation);
                assert_eq!((got.max_unique_phrases, got.max_evidence_spans, got.max_value_bytes, got.max_scan_work),
                    (want.max_unique_phrases, want.max_evidence_spans, want.max_value_bytes, want.max_scan_work));
            }
            let mut reduction = limits.mapping.reduction;
            assert_eq!(reduction.map_batch_chunks, chunks);
            reduction.map_batch_chunks = 1;
            assert_eq!(reduction, original.mapping.reduction);
            let quote = limits.cohort.unwrap();
            assert_eq!(quote.native.kv_bytes, chunks as u64 * original.common.native.kv_bytes);
            assert_eq!(quote.max_result_bytes, chunks as u64 * budget.max_output_bytes + 4096 + chunks as u64);
        }
    }
}

#[test]
fn complete_chunk_cohort_memory_is_admitted_before_source_snapshot_or_output_io() {
    for (kind, mode) in [(Kind::Ner, &[][..]), (Kind::Keyphrases, &[][..]),
        (Kind::Summarize, &[][..]), (Kind::Keyphrases, &["--reduce-keyphrases"][..]),
        (Kind::Answer, &["--question", "private-question.txt"][..])] {
        for snapshot in [false, true] {
            let mut flags = mode.to_vec(); flags.extend_from_slice(&["--cohort-chunks", "4"]);
            let (map, mut args) = arguments(kind, &flags);
            args.memory_mib = 262_144;
            if snapshot {
                args.source_dir = None; args.source_manifest = None;
                args.snapshot = Some("local.snapshot".into()); args.expected_model_sha256 = Some("ab".repeat(32));
            }
            let limits = map.validate(&args).unwrap();
            let common = limits.common;
            let budget = map.source.budget(&args, common);
            let quote = Int4SourceMapCohortMemoryLimits { max_chunks: 4, context_tokens: args.context_tokens,
                max_output_tokens_per_chunk: u64::from(budget.max_output_tokens),
                max_result_bytes_per_chunk: budget.max_output_bytes,
            }.requirements(args.group_width).unwrap();
            assert_eq!(limits.cohort, Some(quote));
            let tokens = if limits.summary.is_some() || limits.keyphrases.is_some() || map.question.is_some() { 0 }
                else { map.mapping.max_chunks as u64 * u64::from(budget.max_output_tokens) };
            // Aggregates and QA omit native transcripts from final output;
            // complete map tokens and all candidates remain in workspace.
            let output = (limits.max_result_bytes as u64 + 4096) * 4 + tokens * 8 + 4096;
            let question_input = if map.question.is_some() { question::QUESTION_BYTES as u64 + 1 } else { 0 };
            let input = args.max_input_bytes as u64 + 1 + question_input + common.preparation_bytes;
            let execution = quote.native.kv_bytes + quote.native.rope_bytes + quote.native.scratch_payload_bound
                + ALLOCATOR_BYTES + quote.workspace_bytes + input + limits.workspace_bytes + output;
            let loading = if snapshot { common.snapshot_need.scratch_bytes }
                else { common.source_preparation_bytes + common.load.scratch_payload_bytes };
            let expected = common.preparation_bytes + common.load.weight_payload_bytes + ALLOCATOR_BYTES
                + loading.max(execution) + CLEANUP_BYTES;
            assert_eq!(common.minimum_memory_bytes, expected);
            args.memory_mib = expected.div_ceil(MIB);
            map.validate(&args).unwrap();
            args.memory_mib -= 1;
            assert_eq!(map.execute(args, &mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
        }
    }
}

#[test]
fn unsupported_synthesis_and_owned_job_routes_cannot_ignore_chunk_cohorts() {
    for command in definitions() {
        assert!(command.get_arguments().any(|arg| arg.get_id() == "cohort_chunks"));
    }
    let jobs = super::super::document_job::definition();
    for operation in jobs.get_subcommands() {
        for task in operation.get_subcommands() {
            assert!(!task.get_arguments().any(|arg| arg.get_id() == "cohort_chunks"),
                "{} {} must not advertise unimplemented cohort jobs", operation.get_name(), task.get_name());
        }
    }
    for flags in [&["--synthesize-summary"][..], &["--synthesize-summary", "--hierarchical-summary"][..]] {
        let summary = definitions().into_iter().find(|c| c.get_name() == "summarize-map").unwrap();
        let mut argv = vec!["summarize-map", "--source-dir", "source", "--source-manifest", "manifest",
            "--memory-mib", "8192", "--cohort-chunks", "2"];
        argv.extend_from_slice(flags);
        assert!(summary.try_get_matches_from(argv).is_err());
    }
    for flags in [&["--synthesize-answer"][..], &["--synthesize-answer", "--hierarchical-answer"][..]] {
        let answer = definitions().into_iter().find(|c| c.get_name() == "answer-map").unwrap();
        let mut argv = vec!["answer-map", "--source-dir", "source", "--source-manifest", "manifest",
            "--memory-mib", "8192", "--question", "private-question.txt", "--cohort-chunks", "2"];
        argv.extend_from_slice(flags);
        assert!(answer.try_get_matches_from(argv).is_err());
    }
    for (kind, flags) in [(Kind::Summarize, &["--synthesize-summary"][..]),
        (Kind::Summarize, &["--synthesize-summary", "--hierarchical-summary"][..]),
        (Kind::Answer, &["--question", "private-question.txt", "--synthesize-answer"][..]),
        (Kind::Answer, &["--question", "private-question.txt", "--synthesize-answer", "--hierarchical-answer"][..])] {
        let (mut map, args) = arguments(kind, flags); map.cohort_chunks = Some(2);
        assert_eq!(map.execute(args, &mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
    }
    for (kind, mode) in [(Kind::Ner, &[][..]), (Kind::Keyphrases, &[][..]),
        (Kind::Summarize, &[][..]), (Kind::Keyphrases, &["--reduce-keyphrases"][..]),
        (Kind::Answer, &["--question", "private-question.txt"][..])] {
        let (mut map, args) = arguments(kind, mode);
        map.validate_document_job(&args, false).unwrap();
        map.cohort_chunks = Some(1);
        assert!(map.validate_document_job(&args, false).is_err());
    }
}

#[test]
fn invalid_chunk_width_and_complete_window_output_fail_before_io() {
    for kind in KINDS {
        for flags in [vec!["--cohort-chunks", "0"], vec!["--cohort-chunks", "65"],
            vec!["--cohort-chunks", "2", "--max-chunks", "1"], vec!["--cohort-chunks", "64"]] {
            // Sixty-four one-MiB rows cannot fit their complete 64-MiB envelope.
            // The requested execution width must not be silently reduced.
            let mut command = command(kind, &flags); command.args.memory_mib = 262_144;
            let mut output = Vec::new();
            assert_eq!(command.execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
            assert!(output.is_empty());
        }
        let (mut map, args) = arguments(kind, &[]); map.cohort_chunks = Some(usize::MAX);
        assert_eq!(map.execute(args, &mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
    }
}

#[test]
fn invalid_complete_document_work_and_verification_refuse_before_io() {
    for flags in [vec!["--max-chunks", "0"], vec!["--max-chunks", "257"],
        vec!["--max-chunk-bytes", "3"], vec!["--max-tokenizer-calls", "0"],
        vec!["--max-tokenizer-calls", "1000001"], vec!["--max-map-result-bytes", "1048575"],
        vec!["--max-map-result-bytes", "67108865"], vec!["--max-live-value-bytes", "1"],
        vec!["--max-total-value-bytes", "1"], vec!["--reduction-reserve-mib", "0"],
        vec!["--max-total-mask-node-visits", "1"], vec!["--max-forward-positions", "0"],
        vec!["--max-forward-positions", "16777217"], vec!["--max-projected-logits", "1"],
        vec!["--max-attention-pairs", "0"], vec!["--max-dot-products", "0"],
        vec!["--max-multiply-accumulates", "0"], vec!["--max-source-fields", "0"],
        vec!["--max-source-evidence-spans", "1000001"], vec!["--max-source-scan-steps", "0"],
        vec!["--options", "-"], vec!["--preparation-mib", "256"]] {
        let mut output = Vec::new();
        assert_eq!(command(Kind::Ner, &flags).execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
}

#[test]
fn final_summary_selection_and_independent_reverification_have_finite_bounds() {
    let (map, args) = arguments(Kind::Summarize, &["--max-summary-bullets", "3"]);
    let limits = map.validate(&args).unwrap();
    let summary = limits.summary.unwrap();
    assert_eq!(summary.max_bullets, 3);
    assert_eq!(summary.max_result_bytes, limits.max_result_bytes);
    assert_eq!(summary.aggregation.max_unique_bullets, 4096);
    assert_eq!(summary.aggregation.max_scan_steps, 536_870_912);
    assert_ne!(summary.aggregation.max_scan_steps, limits.mapping.verification.max_scan_steps);
    let budget = map.source.budget(&args, limits.common);
    let (_, task) = map.request("one document".to_owned(), None, budget, args.max_input_bytes).unwrap();
    let SourceMapTask::Summarize(options) = task else { panic!("wrong summary map options"); };
    assert_eq!(options.max_bullets, 8, "final top-k must not change the per-map task");
    for flags in [vec!["--max-summary-bullets", "0"], vec!["--max-summary-bullets", "1025"],
        vec!["--max-summary-unique-bullets", "0"], vec!["--max-summary-citations", "1000001"],
        vec!["--max-summary-evidence-spans", "0"], vec!["--max-summary-value-bytes", "16777217"],
        vec!["--max-summary-scan-steps", "0"], vec!["--max-summary-scan-steps", "1000000000001"],
        vec!["--reduction-reserve-mib", "1"]] {
        let mut output = Vec::new();
        assert_eq!(command(Kind::Summarize, &flags).execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
    for kind in [Kind::Ner, Kind::Keyphrases] {
        let mut definition = definitions().into_iter().find(|c| c.get_name().starts_with(kind.name())).unwrap();
        let name = definition.get_name().to_owned();
        assert!(definition.try_get_matches_from_mut([name.as_str(), "--source-dir", "source",
            "--source-manifest", "manifest", "--memory-mib", "8192", "--max-summary-bullets", "3"]).is_err());
    }
    let (mut map, args) = arguments(Kind::Ner, &[]);
    map.kind = Kind::Summarize;
    assert!(map.validate(&args).is_err(), "programmatic task swaps cannot lose summary policy");
    map.kind = Kind::Answer;
    assert!(map.validate(&args).is_err());
}

#[test]
fn exact_typed_options_share_single_source_parser_and_preserve_original_unicode() {
    let text = "é\r\n上海 <tool_call> untrusted source\n \t";
    for (kind, json) in [
        (Kind::Ner, r#"{"types":["location"],"max_entities":2,"max_mention_scalars":16}"#),
        (Kind::Keyphrases, r#"{"max_phrases":3,"max_phrase_scalars":24}"#),
        (Kind::Summarize, r#"{"max_bullets":2,"max_bullet_scalars":80,"max_citations_per_bullet":1,"max_quote_scalars":24}"#),
    ] {
        let (map, args) = arguments(kind, &[]);
        let budget = map.source.budget(&args, map.validate(&args).unwrap().common);
        let (retained, task) = map.request(text.to_owned(), Some(json), budget, args.max_input_bytes).unwrap();
        assert_eq!(retained, text);
        match task {
            SourceMapTask::Ner(options) => {
                assert_eq!(options.types, [crate::tasks::ner::EntityType::Location]);
                assert_eq!(options.max_entities, 2); assert_eq!(options.max_mention_scalars, 16);
            }
            SourceMapTask::Keyphrases(options) => {
                assert_eq!(options.max_phrases, 3); assert_eq!(options.max_phrase_scalars, 24);
            }
            SourceMapTask::Summarize(options) => {
                assert_eq!(options.max_bullets, 2); assert_eq!(options.max_quote_scalars, 24);
            }
        }
        for bad in ["{", "{}", r#"{"task":"answer"}"#, r#"{"max_phrases":2,"max_phrases":3,"max_phrase_scalars":24}"#] {
            assert!(map.request(text.to_owned(), Some(bad), budget, args.max_input_bytes).is_err());
        }
        assert!(map.request(text.to_owned(), Some(&" ".repeat(shared::OPTIONS_BYTES + 1)), budget, args.max_input_bytes).is_err());
        assert!(map.request(text.to_owned(), Some(json), budget, text.len() - 1).is_err());
        assert!(map.request(String::new(), Some(json), budget, args.max_input_bytes).is_err());
        assert_eq!(map.request("  \r\n".to_owned(), Some(json), budget, args.max_input_bytes).unwrap().0, "  \r\n");
    }
}

#[test]
fn whole_document_admission_retains_both_preparations_and_guarded_full_output() {
    for kind in KINDS {
        let (map, mut args) = arguments(kind, &[]);
        let single = map.source.validate(&args).unwrap();
        let limits = map.validate(&args).unwrap();
        assert!(limits.common.minimum_memory_bytes
            > single.minimum_memory_bytes + single.preparation_bytes + limits.workspace_bytes);
        args.memory_mib = limits.common.minimum_memory_bytes.saturating_sub(1) / MIB;
        assert!(map.validate(&args).is_err());
        args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
        assert!(map.validate(&args).is_ok());
        let budget = map.source.budget(&args, limits.common);
        assert!(workspace_bytes(budget, limits.mapping, 0, limits.reduction_reserve_bytes).is_err());
        assert!(workspace_bytes(budget, limits.mapping, map.mapping.max_chunks + 1, limits.reduction_reserve_bytes).is_err());
        let mut overflow = budget; overflow.max_output_bytes = u64::MAX;
        assert!(workspace_bytes(overflow, limits.mapping, 2, limits.reduction_reserve_bytes).is_err());
    }
}

#[test]
fn explicit_group_width_remains_native_and_free_text_switches_cannot_leak() {
    let (map, mut args) = arguments(Kind::Ner, &[]);
    args.group_width = 16; let small = map.validate(&args).unwrap();
    args.group_width = 128; let large = map.validate(&args).unwrap();
    assert_eq!(small.mapping.max_model_work, large.mapping.max_model_work);
    assert!(small.common.load.groups > large.common.load.groups);
    for flags in [vec!["--max-output-bytes", "65536"], vec!["--prefill-rows", "1"],
        vec!["--min-new-tokens", "0"], vec!["--logit-bias", "42=0"]] {
        let mut argv = vec!["candidate", "source-int4", "ner-map", "--source-dir", "source",
            "--source-manifest", "manifest", "--memory-mib", "8192"];
        argv.extend(flags);
        let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
        assert!(CandidateCommand::from_matches(&matches).is_err());
    }
    let mut command = command(Kind::Ner, &[]); command.args.seed = Some("ab".repeat(32));
    assert_eq!(command.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_never_reads_options_document_or_model() {
    for kind in KINDS {
        let mut output = Vec::new();
        assert_eq!(command(kind, &[]).execute(&mut NoRead, &mut output), Err(CandidateError::Unavailable));
        assert!(output.is_empty());
    }
    for (kind, mode) in [(Kind::Ner, &[][..]), (Kind::Keyphrases, &[][..]),
        (Kind::Summarize, &[][..]), (Kind::Keyphrases, &["--reduce-keyphrases"][..]),
        (Kind::Answer, &["--question", "private-question.txt"][..])] {
        for chunks in ["1", "4"] {
            let mut flags = mode.to_vec(); flags.extend_from_slice(&["--cohort-chunks", chunks]);
            let mut command = command(kind, &flags); command.args.memory_mib = 262_144;
            let mut output = Vec::new();
            assert_eq!(command.execute(&mut NoRead, &mut output), Err(CandidateError::Unavailable));
            assert!(output.is_empty());
        }
    }
}
