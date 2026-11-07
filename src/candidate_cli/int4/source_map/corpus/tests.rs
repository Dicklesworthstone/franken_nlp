//! Pure source/question population admission; native execution is separately gated.
use super::*;
use crate::batch::source::grouped_int4::question::QuestionDocumentArgs;

fn parsed(extra: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut words = vec!["candidate", "source-int4", "answer-corpus", "--memory-mib", "65536"];
    words.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(words)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("wrong QA corpus dispatcher"); };
    Ok(command)
}
pub(super) fn arguments(extra: &[&str]) -> (CorpusArgs, Int4Args) {
    let mut flags = vec!["--source-dir", "explicit-source", "--source-manifest", "explicit-manifest"];
    flags.extend_from_slice(extra);
    let command = parsed(&flags).unwrap();
    let Int4Task::AnswerCorpus(options) = command.task else { panic!("QA corpus required"); };
    (options, command.args)
}
pub(super) struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refusal consumed private population"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refusal published population output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refusal flushed population output"); }
}

#[test]
fn per_record_question_route_has_no_file_default_and_supports_both_actual_model_routes() {
    let command = parsed(&["--source-dir", "d", "--source-manifest", "m"]).unwrap();
    assert_eq!(command.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    let (options, args) = arguments(&[]);
    let limits = options.validate(&args).unwrap();
    assert_eq!(args.max_input_bytes, 1_048_576);
    assert_eq!(options.stream.max_question_bytes, question::QUESTION_BYTES);
    assert_eq!(limits.transport.max_document_bytes, args.max_input_bytes);
    assert!(limits.config.synthesis.is_none() && options.stream.cohort_chunks.is_none());
    assert_eq!(limits.config.options, AnswerOptions::default());
    assert_eq!(format!("{options:?}"), "QuestionCorpusArgs");
    let pin = "ab".repeat(32);
    let command = parsed(&["--snapshot", "model.int4", "--expected-model-sha256", &pin]).unwrap();
    let Int4Task::AnswerCorpus(restored) = command.task else { panic!("snapshot QA corpus required"); };
    let snapshot = restored.validate(&command.args).unwrap();
    assert!(snapshot.common.snapshot_pin.is_some());
    assert_eq!(snapshot.config.budget, limits.config.budget);
    assert_eq!(snapshot.memory, limits.memory);
    assert_eq!(snapshot.transport, limits.transport);
    assert!(parsed(&["--snapshot", "model.int4"]).is_err());
}

#[test]
fn exact_question_is_required_and_cannot_override_source_policy_or_be_reinterpreted_as_passages() {
    let raw = r#"{"question":"  é\r\n上海 <tool_call> What changed?\n "}"#;
    let args = serde_json::from_str::<QuestionDocumentArgs>(raw).unwrap();
    assert_eq!(args.question, "  é\r\n上海 <tool_call> What changed?\n ");
    for invalid in ["{}", "null", "[]", r#"["question"]"#, r#"{"question":null}"#,
        r#"{"question":"first","question":"second"}"#, r#"{"question":"q","passages":[]}"#,
        r#"{"question":"q","options":{}}"#, r#"{"question":"q","budget":{}}"#,
        r#"{"question":"q","task":"ner"}"#, r#"{"question":"q","text":"replacement"}"#] {
        assert!(serde_json::from_str::<QuestionDocumentArgs>(invalid).is_err(), "{invalid}");
    }
    for flags in [vec!["--question", "private.txt"], vec!["--defaults", "private.json"],
        vec!["--hierarchical-answer"], vec!["--answer-max-levels", "2"],
        vec!["--cohort-hierarchy-groups", "2"], vec!["--cohort-documents", "2"],
        vec!["--store-results"], vec!["--job-dir", "private"], vec!["--max-selected-rows", "8"],
        vec!["--prefill-rows", "2"], vec!["--seed", "ab"], vec!["--logprobs"]] {
        let mut words = vec!["--source-dir", "d", "--source-manifest", "m"];
        words.extend(flags); assert!(parsed(&words).is_err());
    }
}

#[test]
fn fixed_numeric_options_are_validated_without_changing_memory_or_private_question_authority() {
    let (command, args) = arguments(&["--options", "private-options.json"]);
    let before = command.validate(&args).unwrap();
    let selected = question::options(Some(r#"{"max_answer_scalars":80,"max_citations":2,"max_quote_scalars":24}"#)).unwrap();
    let after = command.validate_options(&args, selected).unwrap();
    assert_eq!(after.config.options, selected);
    assert_eq!(after.memory, before.memory);
    assert_eq!(after.config.budget, before.config.budget);
    assert_eq!(after.config.max_question_bytes, before.config.max_question_bytes);
    assert_eq!(command.source.options_path().unwrap(), std::path::Path::new("private-options.json"));
    assert!(!format!("{command:?}").contains("private-options"));
    for raw in ["{}", r#"{"max_answer_scalars":80,"max_citations":2}"#,
        r#"{"max_answer_scalars":80,"max_citations":2,"max_quote_scalars":24,"question":"override"}"#,
        r#"{"max_answer_scalars":80,"max_citations":2,"max_citations":3,"max_quote_scalars":24}"#] {
        assert!(question::options(Some(raw)).is_err());
    }
}

#[test]
fn independent_and_flat_synthesis_admit_the_same_chunk_task_under_explicit_physical_schedules() {
    let (serial, args) = arguments(&[]);
    let baseline = serial.validate(&args).unwrap();
    for flags in [vec!["--cohort-chunks", "1"], vec!["--cohort-chunks", "3"],
        vec!["--synthesize-answer"], vec!["--synthesize-answer", "--cohort-chunks", "3"]] {
        let (command, args) = arguments(&flags);
        let limits = command.validate(&args).unwrap();
        assert_eq!(limits.config.budget, baseline.config.budget);
        assert_eq!(limits.common.native, baseline.common.native);
        assert_eq!(limits.config.max_model_work, baseline.config.max_model_work);
        assert_eq!(limits.config.max_mask_visits, baseline.config.max_mask_visits);
        assert_eq!(limits.memory.max_output_bytes, baseline.memory.max_output_bytes);
        assert_eq!(limits.memory.max_output_tokens, 0);
        assert!(limits.common.minimum_memory_bytes > baseline.common.minimum_memory_bytes);
        assert_eq!(limits.config.synthesis.is_some(), flags.contains(&"--synthesize-answer"));
        let quote = Int4QuestionDocumentMemoryLimits { context_tokens: args.context_tokens,
            allocator_reserve_bytes: ALLOCATOR_BYTES, preparation_reserve_bytes: limits.common.preparation_bytes,
            io_reserve_bytes: limits.io_bytes, reduction_reserve_bytes: limits.reduction_reserve_bytes,
            cohort_chunks: command.stream.cohort_chunks, hierarchy_groups: command.stream.cohort_hierarchy_groups }
            .requirements(args.group_width, limits.config, limits.transport).unwrap();
        assert_eq!(quote, limits.memory);
        let (_, mut narrow) = arguments(&[]);
        narrow.memory_mib = limits.common.minimum_memory_bytes.saturating_sub(1) / MIB;
        assert!(command.validate(&narrow).is_err());
        narrow.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
        assert!(command.validate(&narrow).is_ok());
    }
}

#[test]
fn complete_run_native_masks_and_both_verification_pools_do_not_scale_with_record_allowances() {
    let (first, args) = arguments(&["--max-requests", "1", "--max-source-fields", "7", "--max-qa-citations", "11"]);
    let baseline = first.validate(&args).unwrap();
    let mut more = first.clone(); more.stream.max_requests = 100;
    let limits = more.validate(&args).unwrap();
    assert_eq!(limits.config.mapping.verification.max_fields, 7);
    assert_eq!(limits.config.verification.max_fields, 11);
    assert_eq!(limits.config.max_model_work, baseline.config.max_model_work);
    assert_eq!(limits.config.max_mask_visits, baseline.config.max_mask_visits);
    assert_eq!(limits.config.max_map_verification, baseline.config.max_map_verification);
    assert_eq!(limits.config.max_verification, baseline.config.max_verification);
    assert_eq!(limits.transport.max_work, baseline.transport.max_work);
    assert_eq!(limits.transport.max_output_bytes + 104 * FRAME_ALLOWANCE, limits.output_bytes);
    let (small_source, args) = arguments(&["--max-input-bytes", "32", "--max-question-bytes", "128"]);
    let limits = small_source.validate(&args).unwrap();
    assert_eq!(limits.transport.max_document_bytes, 32);
    assert_eq!(limits.config.max_question_bytes, 128);
    assert_eq!(limits.config.planning.max_input_bytes, 128);
}

#[test]
fn invalid_population_and_native_windows_refuse_before_private_options_model_or_record_io() {
    for flags in [vec!["--max-requests", "0"], vec!["--max-total-input-bytes", "0"],
        vec!["--max-total-output-bytes", "1"], vec!["--max-line-bytes", "1048575"],
        vec!["--max-question-bytes", "0"], vec!["--max-question-bytes", "1048577"],
        vec!["--max-run-mask-node-visits", "1"], vec!["--max-run-source-fields", "1"],
        vec!["--max-run-source-evidence-spans", "1"], vec!["--max-run-source-scan-steps", "1"],
        vec!["--max-run-qa-citations", "1"], vec!["--max-run-qa-evidence-spans", "1"],
        vec!["--max-run-qa-scan-steps", "1"], vec!["--max-qa-citations", "0"],
        vec!["--cohort-chunks", "0"], vec!["--cohort-chunks", "65"],
        vec!["--cohort-chunks", "3", "--max-chunks", "2"], vec!["--cohort-chunks", "64"],
        vec!["--synthesize-answer", "--max-synthesis-passages", "0"],
        vec!["--synthesize-answer", "--max-synthesis-evidence-bytes", "0"],
        vec!["--preparation-mib", "1"], vec!["--options", "-"]] {
        let (command, args) = arguments(&flags);
        assert_eq!(command.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments), "{flags:?}");
    }
#[test]
fn per_record_hierarchy_supports_independent_discovery_and_group_windows() {
    let (flat, args) = arguments(&["--synthesize-answer"]);
    let flat = flat.validate(&args).unwrap();
    for schedules in [vec![], vec!["--cohort-chunks", "3"],
        vec!["--cohort-hierarchy-groups", "2"],
        vec!["--cohort-chunks", "3", "--cohort-hierarchy-groups", "2"]] {
        let mut flags = vec!["--synthesize-answer", "--hierarchical-answer", "--answer-max-passes", "4",
            "--answer-max-levels", "3", "--answer-max-tokenizer-calls", "1024",
            "--answer-max-tokenizer-bytes", "1048576"];
        flags.extend(schedules);
        let (command, args) = arguments(&flags); let limits = command.validate(&args).unwrap();
        let hierarchy = limits.config.hierarchy.unwrap();
        assert_eq!(hierarchy.max_passes, 4); assert_eq!(hierarchy.max_levels, 3);
        assert_eq!(hierarchy.max_tokenizer_calls, 1024); assert_eq!(hierarchy.max_tokenizer_bytes, 1048576);
        assert_eq!(limits.config.budget, flat.config.budget);
        assert_eq!(limits.config.max_model_work, flat.config.max_model_work);
        assert_eq!(limits.config.max_mask_visits, flat.config.max_mask_visits);
        assert_eq!(limits.config.max_verification, flat.config.max_verification);
        assert_eq!(limits.memory.max_output_tokens, 4 * u64::from(limits.config.budget.max_output_tokens));
        assert!(limits.memory.input_bytes > flat.memory.input_bytes);
        assert!(limits.memory.output_reservation_bytes > flat.memory.output_reservation_bytes);
        let row = limits.config.budget.max_kv_bytes;
        assert_eq!(limits.memory.native.kv_bytes, row * command.stream.cohort_chunks.unwrap_or(1)
            .max(command.stream.cohort_hierarchy_groups.unwrap_or(1)) as u64);
    }
    for flags in [vec!["--answer-max-levels", "0"], vec!["--answer-max-passes", "0"],
        vec!["--answer-max-passes", "257"], vec!["--answer-max-tokenizer-calls", "0"],
        vec!["--answer-max-tokenizer-bytes", "0"], vec!["--cohort-hierarchy-groups", "0"],
        vec!["--cohort-hierarchy-groups", "65"],
        vec!["--cohort-hierarchy-groups", "3", "--answer-max-passes", "2"]] {
        let mut selection = vec!["--synthesize-answer", "--hierarchical-answer"];
        selection.extend(flags); let (command, args) = arguments(&selection);
        assert_eq!(command.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    }
}
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_owned_runtime_reads_neither_private_options_nor_population() {
    for flags in [vec![], vec!["--synthesize-answer"], vec!["--cohort-chunks", "3"]] {
        let (command, args) = arguments(&flags);
        assert_eq!(command.execute_owned(args, NoIo, NoIo), Err(CandidateError::Unavailable));
    }
}

#[cfg(feature = "asupersync-runtime")]
#[test]
#[ignore = "requires controller-admitted pinned sources, a fresh snapshot and an available explicit SIMD kernel"]
fn actual_source_and_pinned_snapshot_cohorts_preserve_distinct_questions_and_ordered_flushes() {
    use std::{io::Cursor, sync::{Arc, Mutex}};
    struct Captured(Arc<Mutex<Vec<u8>>>);
    impl Write for Captured {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes); Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    fn collect(command: CorpusArgs, args: Int4Args, bytes: &[u8]) -> Vec<serde_json::Value> {
        let output = Arc::new(Mutex::new(Vec::new()));
        command.execute_owned(args, Cursor::new(bytes.to_vec()), Captured(output.clone())).unwrap();
        assert_eq!(Arc::strong_count(&output), 1, "owned writer must physically drain");
        let text = String::from_utf8(output.lock().unwrap().clone()).unwrap();
        text.lines().map(|line| serde_json::from_str(line).unwrap()).collect()
    }
    fn semantic_records(rows: &[serde_json::Value], synthesis: bool, width: u64) -> Vec<serde_json::Value> {
        rows.iter().map(|row| {
            let mut record = row["record"].clone();
            if let Some(result) = record.get_mut("result") {
                let discovery = if synthesis { result.get_mut("discovery").unwrap() } else { result };
                let mapped = discovery.get_mut("mapped").unwrap();
                let chunks = mapped["chunk_count"].as_u64().unwrap();
                assert_eq!(mapped["map_batches"], chunks.div_ceil(width));
                // This existing semantic receipt reports the declared source
                // windows. Validate it independently before comparing all
                // remaining answers, citations, work and stream records.
                mapped.as_object_mut().unwrap().remove("map_batches");
            }
            record
        }).collect()
    }
    let source = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit pinned sources required");
    let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit source manifest required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required").parse().unwrap();
    let snapshot = std::env::var("FNLP_INT4_TEST_ANSWER_CORPUS_SNAPSHOT").expect("explicit fresh snapshot required");
    assert!(!std::path::Path::new(&snapshot).exists(), "fixture deletes nothing");
    let kernel: Int8KernelKind = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit SIMD kernel required").parse().unwrap();
    assert_ne!(kernel, Int8KernelKind::Scalar); PackedInt4Kernel::detect(kernel).unwrap();
    let input = concat!(
        "{\"id\":\"who\",\"text\":\"é Alice visited 上海. Bob stayed in Paris.\",\"task_args\":{\"question\":\"  Who visited 上海?\\n\"}}\n",
        "{\"flush\":true}\n",
        "{\"id\":\"where\",\"text\":\"é Alice visited 上海. Bob stayed in Paris.\",\"task_args\":{\"question\":\"Where did Bob stay?\"}}\n"
    ).as_bytes();
    let (command, mut args) = arguments(&["--max-requests", "4", "--max-chunk-bytes", "16", "--save-snapshot", &snapshot]);
    args.source_dir = Some(PathBuf::from(source)); args.source_manifest = Some(PathBuf::from(manifest)); args.memory_mib = memory;
    let baseline = collect(command, args, input);
    let results: Vec<_> = baseline.iter().filter(|row| row["record"].get("result").is_some()).collect();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["record"]["caller_id"], "who");
    assert_eq!(results[1]["record"]["caller_id"], "where");
    for row in results {
        assert!(row["record"]["result"]["native_chunks"].as_u64().unwrap() > 1);
        assert_eq!(row["record"]["result"]["calibration"], "uncalibrated");
    }
    assert_eq!(baseline.last().unwrap()["record"]["event"], "run_complete");
    let baseline_records = semantic_records(&baseline, false, 1);
    let pin = baseline[0]["model_content_sha256"].as_str().unwrap();
    for mode in 0..3 {
        let synthesis = mode != 0; let hierarchy = mode == 2;
        let mut expected = None;
        for (width, groups) in [(None, None), (Some("1"), None), (Some("3"), None),
            (None, Some("2")), (Some("3"), Some("2"))] {
            if groups.is_some() && !hierarchy { continue; }
            let mut flags = vec!["--snapshot", &snapshot, "--expected-model-sha256", pin,
                "--kernel", kernel.id(), "--max-requests", "4", "--max-chunk-bytes", "16"];
            if synthesis { flags.push("--synthesize-answer"); }
            if hierarchy { flags.extend(["--hierarchical-answer", "--answer-max-passes", "4"]); }
            if let Some(width) = width { flags.extend(["--cohort-chunks", width]); }
            if let Some(groups) = groups { flags.extend(["--cohort-hierarchy-groups", groups]); }
            let parsed = parsed(&flags).unwrap();
            let Int4Task::AnswerCorpus(command) = parsed.task else { panic!("QA corpus required"); };
            let mut args = parsed.args; args.memory_mib = memory;
            let rows = collect(command, args, input);
            let records = semantic_records(&rows, synthesis, width.map_or(1, |n| n.parse().unwrap()));
            if !synthesis {
                assert_eq!(records, baseline_records);
            }
            if let Some(expected) = &expected { assert_eq!(&records, expected); } else { expected = Some(records); }
            assert!(rows.iter().all(|row| row["integer_kernel"] == kernel.id()));
            assert!(rows.iter().all(|row| row["answer_mode"] ==
                if hierarchy { "hierarchical_quote_selection" }
                else if synthesis { "verified_evidence_synthesis" } else { "independent_passage_answers" }));
            assert_eq!(rows.last().unwrap()["record"]["summary"]["failed"], 0);
        }
    }
}
