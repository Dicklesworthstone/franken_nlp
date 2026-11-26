//! Command, typed-option and finite-resource checks, not inference fixtures.
use super::*;

pub(in crate::candidate_cli) fn command(task: &str, extra: &[&str]) -> MapCommand {
    let mut args = vec!["candidate", "map", "--task", task, "--model", "local.fnlpq", "--memory-mib", "8192"];
    args.extend_from_slice(extra);
    let matches = super::super::definition().try_get_matches_from(args).unwrap();
    let CandidateCommand::Map(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("map route was replaced") };
    command
}
#[test]
fn all_three_tasks_require_explicit_task_model_and_memory() {
    for task in ["ner", "keyphrases", "summarize"] {
        command(task, &[]).validate().unwrap();
        assert!(super::super::definition().try_get_matches_from(["candidate", "map", "--task", task]).is_err());
        assert!(super::super::definition().try_get_matches_from(["candidate", "map", "--task", task,
            "--model", "m"]).is_err());
    }
    assert!(super::super::definition().try_get_matches_from(["candidate", "map", "--model", "m",
        "--memory-mib", "8192"]).is_err());
}
#[test]
fn unsupported_tasks_missing_schema_and_sampling_are_refused() {
    assert!(command("ner", &["--schema", "private"]).validate().is_err());
    for task in ["generate", "extract", "classify"] {
        assert!(super::super::definition().try_get_matches_from(["candidate", "map", "--task", task,
            "--model", "m", "--memory-mib", "8192"]).is_err());
    }
    for flag in ["--seed", "--top-k", "--instruction"] {
        assert!(super::super::definition().try_get_matches_from(["candidate", "map", "--task", "ner",
            "--model", "m", "--memory-mib", "8192", flag, "private"]).is_err());
    }
}
#[test]
fn finite_limits_refuse_zero_overflow_and_inconsistent_envelopes() {
    for (flag, value) in [("--max-input-bytes", "3"), ("--max-chunks", "0"), ("--max-chunks", "257"),
        ("--max-chunk-bytes", "3"), ("--max-chunk-bytes", "1048577"),
        ("--max-tokenizer-calls", "0"), ("--max-tokenizer-calls", "1000001"),
        ("--max-map-result-bytes", "0"), ("--max-map-result-bytes", "67108865"),
        ("--max-live-value-bytes", "1048576"), ("--max-total-value-bytes", "1048576"),
        ("--reduction-reserve-mib", "0"), ("--reduction-reserve-mib", "18446744073709551615"),
        ("--max-total-mask-node-visits", "0"), ("--max-forward-positions", "0"),
        ("--max-projected-logits", "0"), ("--max-attention-pairs", "0"),
        ("--max-dot-products", "0"), ("--max-multiply-accumulates", "0"), ("--options", "-")] {
        assert!(command("ner", &[flag, value]).validate().is_err(), "{flag}");
    }
}
#[test]
fn retained_prompt_and_grammar_capacity_requires_aggregate_preparation() {
    let small = command("ner", &[]); small.validate().unwrap();
    assert!(command("ner", &["--max-chunks", "256"]).validate().is_err());
    command("ner", &["--max-chunks", "256", "--preparation-mib", "1024"]).validate().unwrap();
}
#[test]
fn raising_chunk_count_never_multiplies_compute_or_mask_authority() {
    let a = command("ner", &[]);
    let b = command("ner", &["--max-chunks", "256", "--preparation-mib", "1024"]);
    assert_eq!(a.work_ceiling(), b.work_ceiling());
    assert_eq!(a.max_total_mask_node_visits, b.max_total_mask_node_visits);
}
#[test]
fn every_whole_document_work_axis_has_an_exact_boundary() {
    let cmd = command("ner", &[]); let cap = cmd.work_ceiling();
    cmd.admit_work(cap, cmd.max_total_mask_node_visits).unwrap();
    for axis in 0..5 {
        let mut bad = cap;
        match axis { 0 => bad.forward_positions += 1, 1 => bad.projected_logits += 1,
            2 => bad.attention_pairs += 1, 3 => bad.projections.dot_products += 1,
            _ => bad.projections.multiply_accumulates += 1 }
        assert_eq!(cmd.admit_work(bad, 1), Err(CandidateError::Planning));
    }
    assert_eq!(cmd.admit_work(cap, cmd.max_total_mask_node_visits + 1), Err(CandidateError::Planning));
}
#[test]
fn options_are_exact_typed_data_not_a_document_or_execution_recipe() {
    let cmd = command("ner", &[]);
    let raw = r#"{"types":["person"],"max_entities":2,"max_mention_scalars":32}"#;
    let SourceMapTask::Ner(options) = cmd.map_task(Some(raw)).unwrap() else { panic!("wrong task") };
    assert_eq!(options.max_entities, 2);
    for bad in ["{}", r#"{"types":[],"max_entities":2,"max_mention_scalars":32}"#,
        r#"{"types":["person"],"max_entities":2,"\u006dax_entities":3,"max_mention_scalars":32}"#,
        r#"{"types":["person"],"max_entities":2,"max_mention_scalars":32,"document":"secret"}"#] {
        assert!(cmd.map_task(Some(bad)).is_err());
    }
    assert!(cmd.map_task(Some(&" ".repeat(source::OPTIONS_BYTES + 1))).is_err());
}
#[test]
fn help_does_not_mislabel_chunk_summaries_as_global_synthesis() {
    let help = definition().render_long_help().to_string();
    assert!(help.contains("not a global synthesized summary"));
    assert!(help.contains("No partial success"));
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_feature_never_opens_input_options_model_or_output() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private input read") } }
    struct NoWrite;
    impl Write for NoWrite {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output written") }
        fn flush(&mut self) -> io::Result<()> { panic!("output flushed") }
    }
    for task in ["ner", "keyphrases", "summarize"] {
        assert_eq!(command(task, &["--options", "never-open.json"]).execute(&mut NoRead, &mut NoWrite),
            Err(CandidateError::Unavailable));
    }
}

#[test]
fn compact_source_maps_and_exact_reducers_price_one_encoder_beside_all_retained_chunks() {
    for (task, extra) in [("ner", vec![]), ("keyphrases", vec!["--reduce-keyphrases"]),
        ("summarize", vec!["--reduce-summary"])] {
        let mut options = vec!["--input-encoding", "safe-bpe-v1", "--max-chunk-bytes", "4096"];
        options.extend(extra);
        let mut cmd = command(task, &options);
        let (_, limits) = cmd.validate().unwrap();
        let encoding = cmd.encoding.limits(&cmd).unwrap().unwrap();
        assert_eq!(encoding.call.max_input_bytes, 4096);
        assert_eq!(encoding.call.max_output_tokens, 4096);
        assert_eq!(encoding.complete.max_input_bytes, 8 * cmd.host.max_input_bytes as u64);
        assert!(encoding.workspace_bytes().unwrap() > 0);
        assert!(encoding.workspace_bytes().unwrap() < limits.preparation_bytes);
        cmd.max_chunk_bytes = Some(1_048_576);
        cmd.host.max_input_bytes = 1_048_576;
        cmd.host.preparation_mib = 512;
        assert!(cmd.validate().is_err());
    }
    let byte = command("ner", &[]);
    assert!(byte.encoding.limits(&byte).unwrap().is_none());
}
#[test]
fn compact_source_map_totals_are_finite_explicit_and_not_available_to_byte_input() {
    for flag in ["--max-total-tokenizer-input-bytes", "--max-total-tokenizer-output-tokens",
        "--max-total-tokenizer-pair-checks", "--max-total-tokenizer-pair-bytes"] {
        assert!(command("ner", &[flag, "1"]).validate().is_err());
        for value in ["0", "18446744073709551615"] {
            assert!(command("ner", &["--input-encoding", "safe-bpe-v1", flag, value]).validate().is_err());
        }
    }
    let cmd = command("ner", &["--input-encoding", "safe-bpe-v1",
        "--max-tokenizer-pair-checks", "1234", "--max-tokenizer-pair-bytes", "9999",
        "--max-total-tokenizer-input-bytes", "54321", "--max-total-tokenizer-output-tokens", "12345",
        "--max-total-tokenizer-pair-checks", "77777", "--max-total-tokenizer-pair-bytes", "888888"]);
    let limits = cmd.encoding.limits(&cmd).unwrap().unwrap();
    assert_eq!(limits.call.max_pair_checks, 1234);
    assert_eq!(limits.call.max_pair_bytes, 9999);
    assert_eq!(limits.complete.max_input_bytes, 54321);
    assert_eq!(limits.complete.max_output_tokens, 12345);
    assert_eq!(limits.complete.max_pair_checks, 77777);
    assert_eq!(limits.complete.max_pair_bytes, 888888);
}
#[test]
fn compact_passage_qa_admits_auxiliary_context_beside_source_before_io() {
    let cmd = command("answer", &["--question", "question.txt",
        "--input-encoding", "safe-bpe-v1", "--max-chunk-bytes", "4096"]);
    let (_, limits) = cmd.validate().unwrap();
    let encoding = cmd.encoding.limits(&cmd).unwrap().unwrap();
    assert_eq!(encoding.call.max_input_bytes, cmd.host.max_input_bytes);
    assert_eq!(encoding.call.max_output_tokens, cmd.host.max_input_bytes);
    assert!(encoding.workspace_bytes().unwrap() < limits.preparation_bytes);
    assert_eq!(cmd.question_mapping().unwrap().chunks.max_chunk_bytes, 4096);
    let byte = command("answer", &["--question", "question.txt"]);
    assert!(byte.encoding.limits(&byte).unwrap().is_none());
}
#[test]
fn compact_extraction_admits_schema_and_chunk_storage_before_any_private_io() {
    for grounded in [false, true] {
        let mut flags = vec!["--schema", "schema.json", "--input-encoding", "safe-bpe-v1",
            "--max-chunk-bytes", "4096"];
        if grounded { flags.push("--source-membership"); }
        let cmd = command("extract", &flags);
        let (_, limits) = cmd.validate().unwrap();
        let encoding = cmd.encoding.limits(&cmd).unwrap().unwrap();
        assert_eq!(encoding.call.max_input_bytes, 4096 + crate::candidate_cli::extract::SCHEMA_BYTES);
        assert_eq!(encoding.call.max_output_tokens, encoding.call.max_input_bytes);
        assert!(encoding.workspace_bytes().unwrap() < limits.preparation_bytes);
        assert_eq!(cmd.extraction_mapping().unwrap().mapping.chunks.max_chunk_bytes, 4096);
    }
}
#[test]
fn compact_synthesis_prices_complete_evidence_independently_of_original_chunk_bytes() {
    for (task, extra) in [("answer", vec!["--question", "question.txt", "--synthesize-answer"]),
        ("summarize", vec!["--synthesize-summary"]),
        ("summarize", vec!["--synthesize-summary", "--hierarchical-summary"])] {
        let mut flags = vec!["--input-encoding", "safe-bpe-v1", "--max-chunk-bytes", "4096"];
        flags.extend(extra);
        let cmd = command(task, &flags);
        let (_, limits) = cmd.validate().unwrap();
        let encoding = cmd.encoding.limits(&cmd).unwrap().unwrap();
        assert_eq!(encoding.call.max_input_bytes, cmd.host.max_input_bytes);
        assert_eq!(encoding.call.max_output_tokens, cmd.host.max_input_bytes);
        assert!(encoding.call.max_input_bytes > cmd.max_chunk_bytes.unwrap());
        assert!(encoding.workspace_bytes().unwrap() < limits.preparation_bytes);
        assert_eq!(encoding.complete.max_input_bytes, 8 * cmd.host.max_input_bytes as u64);
    }
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn compact_synthesis_with_disabled_runtime_refuses_before_private_io() {
    struct NoRead;
    impl Read for NoRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private input read") }
    }
    struct NoWrite;
    impl Write for NoWrite {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output written") }
        fn flush(&mut self) -> io::Result<()> { panic!("output flushed") }
    }
    for (task, extra) in [("answer", vec!["--question", "never-open.txt", "--synthesize-answer"]),
        ("summarize", vec!["--synthesize-summary"]),
        ("summarize", vec!["--synthesize-summary", "--hierarchical-summary"])] {
        let mut flags = vec!["--input-encoding", "safe-bpe-v1"];
        flags.extend(extra);
        assert_eq!(command(task, &flags).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Unavailable));
    }
}

#[test]
fn source_maps_and_exact_reducers_compose_packed_prompts_with_one_team() {
    for (task, reducer) in [("ner", None), ("keyphrases", None), ("summarize", None),
        ("keyphrases", Some("--reduce-keyphrases")), ("summarize", Some("--reduce-summary"))] {
        let baseline = command(task, &[]);
        let (_, serial) = baseline.validate().unwrap();
        for width in ["1", "2", "4"] { for rows in [None, Some("1"), Some("7")] {
            let mut flags = vec!["--compute-threads", width];
            if let Some(rows) = rows { flags.extend(["--prefill-rows", rows]); }
            if let Some(reducer) = reducer { flags.push(reducer); }
            let mut selected = command(task, &flags); selected.host.memory_mib = 16384;
            let (args, admitted) = selected.validate().unwrap();
            let prefill = selected.extraction.prefill().unwrap();
            assert_eq!(admitted.compute, args.compute.admit_prefill(prefill).unwrap());
            assert_eq!(admitted.compute.threads, width.parse::<usize>().unwrap());
            assert_eq!(selected.host.task_budget(admitted), baseline.host.task_budget(serial));
            assert_eq!(selected.work_ceiling(), baseline.work_ceiling());
            assert_eq!(selected.max_total_mask_node_visits, baseline.max_total_mask_node_visits);
            assert_eq!(selected.extraction.schema_reserve_per_chunk(), 0);
        } }
    }
}

#[test]
fn source_schedule_memory_keeps_coordinator_workers_and_the_complete_frontier() {
    for task in ["ner", "keyphrases", "summarize"] {
        let mut cmd = command(task, &["--compute-threads", "4", "--prefill-rows", "7",
            "--input-encoding", "safe-bpe-v1", "--max-chunk-bytes", "4096"]);
        cmd.host.memory_mib = 16384;
        let (args, mut limits) = cmd.validate().unwrap();
        let peak = cmd.extraction.schedule_peak_bytes(&cmd, &args, limits).unwrap();
        limits.memory_bytes = peak;
        cmd.extraction.check_schedule_memory(&cmd, &args, limits).unwrap();
        limits.memory_bytes -= 1;
        assert_eq!(cmd.extraction.check_schedule_memory(&cmd, &args, limits), Err(CandidateError::Arguments));
        let mut without_workers = limits; without_workers.compute.workspace_bytes = 0;
        assert_eq!(peak - cmd.extraction.schedule_peak_bytes(&cmd, &args, without_workers).unwrap(),
            limits.compute.workspace_bytes);
        cmd.max_chunks += 1;
        assert_eq!(cmd.extraction.schedule_peak_bytes(&cmd, &args, limits).unwrap() - peak,
            4 * cmd.host.max_result_bytes as u64 + 16 * args.max_new_tokens as u64);
    }
}

#[test]
fn unsupported_or_unfunded_source_schedules_refuse_before_private_io() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private source read") } }
    struct NoWrite;
    impl Write for NoWrite {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("partial result written") }
        fn flush(&mut self) -> io::Result<()> { panic!("partial result flushed") }
    }
    for task in ["ner", "keyphrases", "summarize"] {
        for flags in [vec!["--compute-threads", "0"], vec!["--compute-threads", "65"],
            vec!["--prefill-rows", "0"], vec!["--prefill-rows", "65"],
            vec!["--compute-threads", "4", "--prefill-rows", "7", "--max-weight-mib", "8192"]] {
            assert_eq!(command(task, &flags).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
        }
    }
    for (task, mode) in question_and_synthesis_modes() {
        for schedule in [vec!["--compute-threads", "0"], vec!["--prefill-rows", "65"],
            vec!["--compute-threads", "4", "--prefill-rows", "7", "--max-weight-mib", "8192"]] {
            let mut flags = mode.to_vec(); flags.extend(schedule);
            assert_eq!(command(task, &flags).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
        }
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_source_team_runtime_never_reads_or_publishes() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private source read") } }
    struct NoWrite;
    impl Write for NoWrite {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("result written") }
        fn flush(&mut self) -> io::Result<()> { panic!("result flushed") }
    }
    for task in ["ner", "keyphrases", "summarize"] {
        let mut cmd = command(task, &["--compute-threads", "4", "--prefill-rows", "7"]);
        cmd.host.memory_mib = 16384;
        assert_eq!(cmd.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Unavailable));
    }
}

pub(in crate::candidate_cli) fn question_and_synthesis_modes()
    -> [(&'static str, &'static [&'static str]); 5] {
    [
        ("answer", &["--question", "never-open.txt"]),
        ("answer", &["--question", "never-open.txt", "--synthesize-answer"]),
        ("answer", &["--question", "never-open.txt", "--synthesize-answer", "--hierarchical-answer"]),
        ("summarize", &["--synthesize-summary"]),
        ("summarize", &["--synthesize-summary", "--hierarchical-summary"]),
    ]
}

#[test]
fn synthesis_modes_register_distinct_groups_and_evidence_controls() {
    super::definition().debug_assert();
    for (task, mode, flag) in [
        ("summarize", vec!["--synthesize-summary"], "--max-summary-evidence-bytes"),
        ("answer", vec!["--question", "q", "--synthesize-answer"], "--max-answer-evidence-bytes"),
    ] {
        let mut selected = mode.clone(); selected.extend([flag, "1024"]);
        command(task, &selected).validate().unwrap();
        let mut wrong = vec!["candidate", "map", "--task", task, "--model", "m", "--memory-mib", "8192"];
        wrong.extend(mode);
        let other = if task == "answer" { "--max-summary-evidence-bytes" } else { "--max-answer-evidence-bytes" };
        wrong.extend([other, "1024"]);
        assert!(crate::candidate_cli::definition().try_get_matches_from(wrong).is_err());
    }
}

#[test]
fn document_questions_and_every_synthesis_mode_compose_schedules_without_renewing_authority() {
    for (task, mode) in question_and_synthesis_modes() {
        let baseline = command(task, mode);
        let (_, original) = baseline.validate().unwrap();
        for compact in [false, true] {
            for (width, rows) in [("1", Some("3")), ("2", None), ("4", Some("7"))] {
                let mut flags = mode.to_vec(); flags.extend(["--compute-threads", width]);
                if let Some(rows) = rows { flags.extend(["--prefill-rows", rows]); }
                if compact { flags.extend(["--input-encoding", "safe-bpe-v1"]); }
                let mut selected = command(task, &flags); selected.host.memory_mib = 32768;
                let (args, admitted) = selected.validate().unwrap();
                let prefill = selected.extraction.prefill().unwrap();
                assert_eq!(admitted.compute, args.compute.admit_prefill(prefill).unwrap());
                assert_eq!(selected.host.task_budget(admitted), baseline.host.task_budget(original));
                assert_eq!(selected.work_ceiling(), baseline.work_ceiling());
                assert_eq!(selected.max_total_mask_node_visits, baseline.max_total_mask_node_visits);
                assert_eq!(selected.max_chunks, baseline.max_chunks);
                assert_eq!(selected.max_map_result_bytes, baseline.max_map_result_bytes);
                assert_eq!(selected.question.synthesis.hierarchy.limits(selected.question.synthesis.synthesize_answer).unwrap(),
                    baseline.question.synthesis.hierarchy.limits(baseline.question.synthesis.synthesize_answer).unwrap());
                assert_eq!(selected.summary.synthesis.hierarchy.limits(selected.summary.synthesis.synthesize_summary).unwrap(),
                    baseline.summary.synthesis.hierarchy.limits(baseline.summary.synthesis.synthesize_summary).unwrap());
            }
        }
    }
}

#[test]
fn every_synthesized_schedule_prices_complete_results_beside_native_workers_and_prompt_storage() {
    for (task, mode) in question_and_synthesis_modes() {
        let mut flags = mode.to_vec();
        flags.extend(["--compute-threads", "4", "--prefill-rows", "7", "--input-encoding", "safe-bpe-v1"]);
        let mut selected = command(task, &flags); selected.host.memory_mib = 32768;
        let (args, mut limits) = selected.validate().unwrap();
        let peak = selected.extraction.schedule_peak_bytes(&selected, &args, limits).unwrap();
        limits.memory_bytes = peak;
        selected.extraction.check_schedule_memory(&selected, &args, limits).unwrap();
        limits.memory_bytes -= 1;
        assert_eq!(selected.extraction.check_schedule_memory(&selected, &args, limits), Err(CandidateError::Arguments));
        let mut coordinator = limits; coordinator.compute.workspace_bytes = 0;
        assert_eq!(peak - selected.extraction.schedule_peak_bytes(&selected, &args, coordinator).unwrap(),
            limits.compute.workspace_bytes);
        let original_work = selected.work_ceiling();
        selected.max_chunks += 1;
        assert!(selected.extraction.schedule_peak_bytes(&selected, &args, limits).unwrap() > peak);
        assert_eq!(selected.work_ceiling(), original_work);
        selected.reduction_reserve_mib = u64::MAX;
        assert_eq!(selected.extraction.schedule_peak_bytes(&selected, &args, limits), Err(CandidateError::Arguments));
    }
}

#[test]
fn hierarchy_memory_grows_with_permitted_passes_and_collected_evidence() {
    for (task, mode, pass_flag, evidence_flag) in [
        ("answer", vec!["--question", "q", "--synthesize-answer", "--hierarchical-answer"],
            "--answer-max-passes", "--max-answer-evidence-bytes"),
        ("summarize", vec!["--synthesize-summary", "--hierarchical-summary"],
            "--summary-max-passes", "--max-summary-evidence-bytes"),
    ] {
        let quote = |passes, evidence| {
            let mut flags = mode.clone();
            flags.extend(["--compute-threads", "2", "--prefill-rows", "3", pass_flag, passes, evidence_flag, evidence]);
            let mut selected = command(task, &flags); selected.host.memory_mib = 32768;
            let (args, limits) = selected.validate().unwrap();
            (selected.extraction.schedule_peak_bytes(&selected, &args, limits).unwrap(), selected.work_ceiling())
        };
        let (small, work) = quote("16", "1024");
        let (more_passes, pass_work) = quote("32", "1024");
        let (more_evidence, evidence_work) = quote("16", "8192");
        assert!(more_passes > small); assert!(more_evidence > small);
        assert_eq!(pass_work, work); assert_eq!(evidence_work, work);
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_question_and_synthesis_schedules_never_open_private_inputs() {
    struct NoIo;
    impl Read for NoIo {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private source/question read") }
    }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("result written") }
        fn flush(&mut self) -> io::Result<()> { panic!("result flushed") }
    }
    for (task, mode) in question_and_synthesis_modes() {
        let mut flags = mode.to_vec(); flags.extend(["--compute-threads", "4", "--prefill-rows", "7"]);
        let mut selected = command(task, &flags); selected.host.memory_mib = 32768;
        assert_eq!(selected.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Unavailable));
    }
}

#[test]
fn entity_union_requires_ner_and_admits_the_complete_workspace_before_io() {
    command("ner", &["--reduce-entities"]).validate().unwrap();
    assert!(command("summarize", &["--reduce-entities"]).validate().is_err());
    for (flag, value) in [("--max-entity-proposals", "0"), ("--max-entity-evidence-spans", "1048577"),
        ("--max-entity-scan-work", "0"), ("--reduction-reserve-mib", "1")] {
        assert!(command("ner", &["--reduce-entities", flag, value]).validate().is_err());
    }
}
