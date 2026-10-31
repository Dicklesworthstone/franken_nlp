//! UNRUN admission and privacy fixtures; no native document result is fabricated.
use super::*;

mod hierarchy_cohorts;

pub(super) const MODES: &[(&str, &[&str], &str)] = &[
    ("ner", &[], "source_map"),
    ("keyphrases", &[], "source_map"),
    ("keyphrases", &["--reduce-keyphrases"], "keyphrase_ranking"),
    ("summarize", &["--map-only"], "source_map"),
    ("summarize", &[], "summary_selection"),
    ("summarize", &["--synthesize-summary"], "summary_synthesis"),
    ("summarize", &["--synthesize-summary", "--hierarchical-summary"], "summary_hierarchy"),
    ("answer", &[], "question_discovery"),
    ("answer", &["--synthesize-answer"], "question_synthesis"),
    ("answer", &["--synthesize-answer", "--hierarchical-answer"], "question_hierarchy"),
];
fn argv(operation: &str, task: &str) -> Vec<String> {
    let mut words: Vec<_> = ["document-job", operation, task,
        "--source-dir", "sources", "--source-manifest", "manifest.json", "--memory-mib", "65536",
        "--max-chunks", "8", "--job-dir", "private/job", "--key-file", "private/key",
        "--limits", "limits.json", "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"]
        .into_iter().map(str::to_owned).collect();
    if task == "answer" { words.extend(["--question", "private-question.txt"].into_iter().map(str::to_owned)); }
    words
}
fn parse(words: Vec<String>) -> Result<(JobCommand, Int4Args), clap::Error> {
    JobCommand::from_matches(&definition().try_get_matches_from(words)?)
}
pub(super) fn command(operation: &str, task: &str, extra: &[&str]) -> (JobCommand, Int4Args) {
    let mut words = argv(operation, task); words.extend(extra.iter().map(|word| (*word).to_owned()));
    parse(words).unwrap()
}
pub(super) fn lifetime(args: &Int4Args, map: MapLimits) -> JobLimits {
    JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: args.max_input_bytes,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: map.max_result_bytes,
        max_spool_bytes: 256 * MIB, max_materialized_bytes: 256 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 6,
        max_work: JobWork::from_int4(map.mapping.max_model_work, map.mapping.max_mask_visits) }
}
pub(super) struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused document job read private input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused document job published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused document job flushed output"); }
}

#[test]
fn every_document_mode_supports_start_resume_and_preserves_distinct_semantics() {
    for &(task, flags, mode) in MODES {
        for operation in ["start", "resume"] {
            let (c, args) = command(operation, task, flags); let map = c.args.validate(&args).unwrap();
            assert_eq!(c.mode.name(), operation); assert_eq!(c.args.map.kind.name(), task);
            assert_eq!(c.args.mode_name(map), mode);
            assert_eq!(map.mapping.chunks.max_chunks, 8);
            assert_eq!(map.mapping.chunks.max_input_bytes, args.max_input_bytes);
            assert!(c.args.storage.store_results);
            let fixed = lifetime(&args, map); let admitted = c.args.lifetime_limits(&args, map, fixed).unwrap();
            assert_eq!(admitted.map.mapping.max_model_work, map.mapping.max_model_work);
            assert_eq!(admitted.map.mapping.max_mask_visits, map.mapping.max_mask_visits);
            assert!(admitted.map.common.minimum_memory_bytes > map.common.minimum_memory_bytes);
        }
    }
}
#[test]
fn independently_pinned_snapshots_preserve_complete_document_admission() {
    for &(task, flags, mode) in MODES {
        let (source, source_args) = command("resume", task, flags);
        let original = source.args.validate(&source_args).unwrap();
        let mut words = argv("resume", task);
        let at = words.iter().position(|s| s == "--source-dir").unwrap(); words.drain(at..at + 4);
        words.extend(["--snapshot", "candidate.int4", "--expected-model-sha256"].into_iter().map(str::to_owned));
        words.push("ab".repeat(32)); words.extend(flags.iter().map(|word| (*word).to_owned()));
        let (snapshot, args) = parse(words).unwrap(); let reloaded = snapshot.args.validate(&args).unwrap();
        assert!(reloaded.common.snapshot_pin.is_some()); assert_eq!(snapshot.args.mode_name(reloaded), mode);
        assert_eq!(original.mapping.max_model_work, reloaded.mapping.max_model_work);
        assert_eq!(original.mapping.max_mask_visits, reloaded.mapping.max_mask_visits);
        assert_eq!(original.workspace_bytes, reloaded.workspace_bytes);
    }
}
#[test]
fn chunk_schedule_is_explicit_on_start_resume_and_does_not_change_document_authority() {
    for &(task, flags, mode) in MODES {
        let (serial, args) = command("start", task, flags);
        let original = serial.args.validate(&args).unwrap();
        assert!(original.cohort_chunks.is_none()); assert!(original.cohort.is_none());
        for operation in ["start", "resume"] {
            for width in ["1", "4"] {
                let mut flags = flags.to_vec(); flags.extend(["--cohort-chunks", width]);
                let (c, args) = command(operation, task, &flags); let map = c.args.validate(&args).unwrap();
                assert_eq!(c.mode.name(), operation); assert_eq!(c.args.mode_name(map), mode);
                assert_eq!(map.cohort_chunks, Some(width.parse().unwrap()));
                assert_eq!(map.mapping.reduction.map_batch_chunks, width.parse::<usize>().unwrap());
                assert_eq!(map.mapping.chunks, original.mapping.chunks);
                assert_eq!(map.mapping.max_model_work, original.mapping.max_model_work);
                assert_eq!(map.mapping.max_mask_visits, original.mapping.max_mask_visits);
                assert_eq!(map.mapping.verification, original.mapping.verification);
                assert_eq!(map.max_result_bytes, original.max_result_bytes);
                assert_eq!(map.workspace_bytes, original.workspace_bytes);
                assert_eq!(c.args.map.source.budget(&args, map.common), serial.args.map.source.budget(&args, original.common));
                let fixed = lifetime(&args, original);
                let admitted = c.args.lifetime_limits(&args, map, fixed).unwrap();
                assert_eq!(admitted.map.cohort, map.cohort);
                assert_eq!(admitted.buffer_bytes, serial.args.lifetime_limits(&args, original, fixed).unwrap().buffer_bytes);
            }
        }
    }
}
#[test]
fn durable_cohorts_price_native_phases_semantic_transcripts_private_inputs_and_population() {
    for &(task, flags, mode) in MODES {
        let mut flags = flags.to_vec(); flags.extend(["--cohort-chunks", "4"]);
        for snapshot in [false, true] {
            let (c, mut args) = command("resume", task, &flags);
            if snapshot {
                args.source_dir = None; args.source_manifest = None;
                args.snapshot = Some("candidate.int4".into()); args.expected_model_sha256 = Some("ab".repeat(32));
            }
            let map = c.args.validate(&args).unwrap(); let common = map.common;
            let quote = map.cohort.unwrap();
            let budget = c.args.map.source.budget(&args, common);
            let transcript_passes = match mode {
                "source_map" => map.mapping.chunks.max_chunks as u64,
                "summary_synthesis" => map.mapping.chunks.max_chunks as u64 + 1,
                "summary_hierarchy" => map.mapping.chunks.max_chunks as u64 + map.hierarchy.unwrap().max_passes as u64,
                "question_hierarchy" => map.question_hierarchy.unwrap().max_passes as u64,
                "summary_selection" | "keyphrase_ranking" | "question_discovery" | "question_synthesis" => 0,
                _ => panic!("unknown complete-document result"),
            };
            // No physical-cohort wrapper is persisted. Only native transcripts
            // owned by the selected semantic result contribute token storage.
            let output = map.max_result_bytes as u64 * 4
                + transcript_passes * u64::from(budget.max_output_tokens) * 8 + 4096;
            let question = if task == "answer" { crate::candidate_cli::source::OPTIONS_BYTES as u64 + 1 } else { 0 };
            let input = args.max_input_bytes as u64 + 1 + question + common.preparation_bytes;
            let execution = quote.native.kv_bytes + quote.native.rope_bytes + quote.native.scratch_payload_bound
                + ALLOCATOR_BYTES + quote.workspace_bytes + input + map.workspace_bytes + output;
            let loading = if snapshot { common.snapshot_need.scratch_bytes }
                else { common.source_preparation_bytes + common.load.scratch_payload_bytes };
            let expected = common.preparation_bytes + common.load.weight_payload_bytes + ALLOCATOR_BYTES
                + loading.max(execution) + CLEANUP_BYTES;
            assert_eq!(common.minimum_memory_bytes, expected);
            assert_eq!(quote.native.kv_bytes, common.native.kv_bytes * 4);
            let fixed = lifetime(&args, map); let all = c.args.lifetime_limits(&args, map, fixed).unwrap();
            assert_eq!(all.map.common.minimum_memory_bytes, expected + all.buffer_bytes);
            let mut tight = map; tight.common.memory_bytes = all.map.common.minimum_memory_bytes - 1;
            assert_eq!(c.args.lifetime_limits(&args, tight, fixed).unwrap_err().code, "job_population_memory");
            tight.common.memory_bytes += 1; assert!(c.args.lifetime_limits(&args, tight, fixed).is_ok());
            args.memory_mib = expected.div_ceil(MIB); c.args.validate(&args).unwrap();
            args.memory_mib -= 1;
            assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
        }
    }
}
#[test]
fn invalid_cohort_admission_refuses_all_document_modes_before_private_io() {
    for &(task, mode, _) in MODES {
        for flags in [vec!["--cohort-chunks", "0"], vec!["--cohort-chunks", "65"],
            vec!["--cohort-chunks", "9"], vec!["--cohort-chunks", "4", "--max-map-result-bytes", "1048575"]] {
            let mut mode = mode.to_vec(); mode.extend(flags);
            let (c, args) = command("resume", task, &mode);
            assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
        }
    }
    let mut words = argv("start", "ner");
    let at = words.iter().position(|word| word == "--max-chunks").unwrap() + 1;
    words[at] = "64".to_owned(); words.extend(["--cohort-chunks", "64"].into_iter().map(str::to_owned));
    let (c, mut args) = parse(words).unwrap(); args.memory_mib = 262_144;
    assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
}
#[test]
fn item_local_source_jobs_do_not_accept_document_chunk_schedules() {
    for definition in source_job::definitions() {
        for operation in ["start", "resume"] {
            let mut words = vec![definition.get_name().to_owned(), operation.to_owned()];
            words.extend(["--source-dir", "sources", "--source-manifest", "manifest.json", "--memory-mib", "65536",
                "--job-dir", "private/job", "--key-file", "private/key", "--limits", "limits.json",
                "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"].into_iter().map(str::to_owned));
            assert!(definition.clone().try_get_matches_from(words.clone()).is_ok());
            words.extend(["--cohort-chunks", "2"].into_iter().map(str::to_owned));
            assert!(definition.clone().try_get_matches_from(words).is_err());
        }
    }
}
#[test]
fn task_selection_hierarchy_and_repair_controls_are_explicit_and_family_scoped() {
    assert!(definition().try_get_matches_from(["document-job"]).is_err());
    assert!(definition().try_get_matches_from(["document-job", "start"]).is_err());
    assert!(parse(argv("start", "classify")).is_err());
    for (task, flags) in [("ner", vec!["--synthesize-summary"]),
        ("keyphrases", vec!["--question", "q"]),
        ("answer", vec!["--hierarchical-answer"]),
        ("summarize", vec!["--hierarchical-summary"]),
        ("summarize", vec!["--map-only", "--synthesize-summary"]),
        ("summarize", vec!["--map-only", "--max-summary-bullets", "3"]),
        ("answer", vec!["--map-only"])] {
        let mut words = argv("start", task); words.extend(flags.into_iter().map(str::to_owned));
        assert!(parse(words).is_err());
    }
    assert!(matches!(command("resume", "ner", &[]).0.mode, RunMode::Resume { discard_uncommitted: false }));
    assert!(matches!(command("resume", "ner", &["--discard-uncommitted-tail"]).0.mode,
        RunMode::Resume { discard_uncommitted: true }));
    let mut words = argv("start", "ner"); words.push("--discard-uncommitted-tail".to_owned());
    assert!(parse(words).is_err());
}
#[test]
fn explicit_storage_and_local_configuration_are_checked_before_any_io() {
    for axis in 0..7 {
        let (mut c, args) = command("start", "ner", &[]);
        match axis {
            0 => c.args.storage.store_results = false, 1 => c.args.storage.job_dir = PathBuf::from("-"),
            2 => c.args.storage.key_file = PathBuf::new(), 3 => c.args.storage.limits_file = PathBuf::from("-"),
            4 => c.args.storage.journal_memory_mib = 0, 5 => c.args.storage.serialization_memory_mib = 0,
            _ => c.args.lifetime.max_lifetime_source_fields = 0,
        }
        assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code, "explicit_document_job_storage");
    }
    for value in ["AB".repeat(16), "ab".repeat(15), "é".repeat(16), "x".repeat(32)] {
        let mut words = argv("start", "ner");
        let at = words.iter().position(|s| s == "--job-id").unwrap() + 1; words[at] = value;
        assert!(parse(words).is_err());
    }
}
#[test]
fn all_native_and_mask_axes_must_fit_the_unchanged_lifetime_limits() {
    let (mut c, args) = command("start", "ner", &[]); let map = c.args.validate(&args).unwrap();
    let fixed = lifetime(&args, map);
    for axis in 0..6 {
        let mut job = fixed;
        match axis { 0 => job.max_work.model.forward_positions -= 1,
            1 => job.max_work.model.projected_logits -= 1, 2 => job.max_work.model.attention_pairs -= 1,
            3 => job.max_work.model.projections.dot_products -= 1,
            4 => job.max_work.model.projections.multiply_accumulates -= 1,
            _ => job.max_work.mask_node_visits -= 1 }
        assert_eq!(c.args.lifetime_limits(&args, map, job).unwrap_err().code, "lifetime_document_work_limits");
    }
    let first = c.args.lifetime_limits(&args, map, fixed).unwrap();
    let mut larger = fixed; larger.max_items = 101; larger.max_attempts = 1000;
    c.args.storage.max_stream_mib = 1024; c.args.storage.max_input_lines = 1_000_000;
    let second = c.args.lifetime_limits(&args, map, larger).unwrap();
    assert_eq!(first.map.mapping.max_model_work, second.map.mapping.max_model_work);
    assert_eq!(first.map.mapping.max_mask_visits, second.map.mapping.max_mask_visits);
    assert!(second.buffer_bytes > first.buffer_bytes);
}
#[test]
fn verification_and_summary_reverification_never_renew_on_retry_or_resume() {
    for operation in ["start", "resume"] {
        for axis in 0..4 {
            let (mut c, args) = command(operation, "summarize", &[]);
            let map = c.args.validate(&args).unwrap(); let fixed = lifetime(&args, map);
            let per = c.args.map.document_verification(map).unwrap();
            match axis {
                0 => c.args.lifetime.max_lifetime_source_fields = per.max_fields * fixed.max_attempts as usize - 1,
                1 => c.args.lifetime.max_lifetime_source_matches = per.max_matches * fixed.max_attempts as usize - 1,
                2 => c.args.lifetime.max_lifetime_source_scan_steps = per.max_scan_steps * fixed.max_attempts - 1,
                _ => c.args.lifetime.max_lifetime_summary_scan_steps =
                    map.summary.unwrap().aggregation.max_scan_steps * fixed.max_attempts - 1,
            }
            assert_eq!(c.args.lifetime_limits(&args, map, fixed).unwrap_err().code, "lifetime_document_verification");
        }
    }
    let (c, args) = command("start", "ner", &[]); let map = c.args.validate(&args).unwrap();
    let per = GroundingBudget { max_fields: 1, max_matches: 1, max_scan_steps: u64::MAX };
    assert!(c.args.lifetime.check(per, 0, 2).is_err());
    assert!(c.args.lifetime.check(c.args.map.document_verification(map).unwrap(), 0, 0).is_err());
}

#[test]
fn ranked_keyphrase_reverification_has_a_separate_immutable_lifetime_bound() {
    for operation in ["start", "resume"] {
        let (mut c, args) = command(operation, "keyphrases", &["--reduce-keyphrases"]);
        let map = c.args.validate(&args).unwrap(); let fixed = lifetime(&args, map);
        let scan_work = map.keyphrases.unwrap().aggregation.max_scan_work;
        c.args.keyphrase_lifetime.as_mut().unwrap().max_lifetime_keyphrase_scan_work =
            Some(scan_work * fixed.max_attempts);
        assert!(c.args.lifetime_limits(&args, map, fixed).is_ok());
        c.args.keyphrase_lifetime.as_mut().unwrap().max_lifetime_keyphrase_scan_work =
            Some(scan_work * fixed.max_attempts - 1);
        assert_eq!(c.args.lifetime_limits(&args, map, fixed).unwrap_err().code, "lifetime_keyphrase_scan_work");
        let mut overflow = map; overflow.keyphrases.as_mut().unwrap().aggregation.max_scan_work = u64::MAX;
        assert_eq!(c.args.lifetime_limits(&args, overflow, fixed).unwrap_err().code, "lifetime_keyphrase_scan_work");
    }
    let (raw, args) = command("start", "keyphrases", &[]); let map = raw.args.validate(&args).unwrap();
    assert!(map.keyphrases.is_none()); assert_eq!(raw.args.mode_name(map), "source_map");
    assert!(raw.args.lifetime_limits(&args, map, lifetime(&args, map)).is_ok());
}

#[test]
fn keyphrase_ranking_and_lifetime_flags_cannot_silently_change_other_document_modes() {
    for (task, flags) in [("ner", vec!["--reduce-keyphrases"]),
        ("summarize", vec!["--max-lifetime-keyphrase-scan-work", "1"]),
        ("answer", vec!["--reduce-keyphrases"]),
        ("keyphrases", vec!["--max-lifetime-keyphrase-scan-work", "1"])] {
        let mut words = argv("start", task); words.extend(flags.into_iter().map(str::to_owned));
        assert!(parse(words).is_err());
    }
    let (c, args) = command("resume", "keyphrases", &["--reduce-keyphrases", "--max-lifetime-keyphrase-scan-work", "0"]);
    assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code, "lifetime_keyphrase_scan_work");
}

#[test]
fn retained_keyphrase_limit_is_the_complete_ranked_envelope_not_the_raw_map_frontier() {
    let (c, args) = command("resume", "keyphrases", &["--reduce-keyphrases", "--max-keyphrase-result-bytes", "2097152"]);
    let map = c.args.validate(&args).unwrap();
    assert_eq!(map.mapping.reduction.max_result_bytes, 16_777_216);
    assert_eq!(map.max_result_bytes, 2_097_152); assert_eq!(c.args.mode_name(map), "keyphrase_ranking");
    let fixed = lifetime(&args, map);
    assert!(c.args.lifetime_limits(&args, map, fixed).is_ok());
    let mut undersized = fixed; undersized.max_result_bytes -= 1;
    assert_eq!(c.args.lifetime_limits(&args, map, undersized).unwrap_err().code, "stored_document_result_ceiling");
}
#[test]
fn selected_task_verification_is_authoritative_instead_of_unused_map_defaults() {
    for (task, flags) in [("answer", vec!["--max-qa-citations", "128"]),
        ("summarize", vec!["--synthesize-summary", "--max-synthesis-fields", "128"])] {
        let mut flags = flags; flags.extend(["--max-source-fields", "1", "--max-lifetime-source-fields", "6"]);
        let (c, args) = command("resume", task, &flags); let map = c.args.validate(&args).unwrap();
        assert_eq!(map.mapping.verification.max_fields, 1);
        assert_eq!(c.args.map.document_verification(map).unwrap().max_fields, 128);
        assert_eq!(c.args.lifetime_limits(&args, map, lifetime(&args, map)).unwrap_err().code,
            "lifetime_document_verification");
    }
    let (c, args) = command("resume", "ner", &["--max-source-fields", "1", "--max-lifetime-source-fields", "6"]);
    let map = c.args.validate(&args).unwrap();
    assert!(c.args.lifetime_limits(&args, map, lifetime(&args, map)).is_ok());
}
#[test]
fn raw_summary_maps_reserve_retained_tokens_and_do_not_charge_selection_work() {
    let (raw, args) = command("start", "summarize", &["--map-only", "--max-lifetime-summary-scan-steps", "0"]);
    let map = raw.args.validate(&args).unwrap(); assert!(map.summary.is_none());
    assert_eq!(raw.args.mode_name(map), "source_map");
    assert!(raw.args.lifetime_limits(&args, map, lifetime(&args, map)).is_ok());
    let (selected, args) = command("start", "summarize", &["--max-lifetime-summary-scan-steps", "0"]);
    let summary = selected.args.validate(&args).unwrap(); assert!(summary.summary.is_some());
    assert!(map.common.minimum_memory_bytes > summary.common.minimum_memory_bytes);
    assert_eq!(selected.args.lifetime_limits(&args, summary, lifetime(&args, summary)).unwrap_err().code,
        "lifetime_document_verification");
}
#[test]
fn complete_population_and_output_caps_are_included_in_memory_admission() {
    let (c, args) = command("start", "ner", &[]); let map = c.args.validate(&args).unwrap();
    let fixed = lifetime(&args, map); let all = c.args.lifetime_limits(&args, map, fixed).unwrap();
    assert_eq!(all.map.common.minimum_memory_bytes, map.common.minimum_memory_bytes + all.buffer_bytes);
    let mut tight = map; tight.common.memory_bytes = all.map.common.minimum_memory_bytes - 1;
    assert_eq!(c.args.lifetime_limits(&args, tight, fixed).unwrap_err().code, "job_population_memory");
    tight.common.memory_bytes += 1; assert!(c.args.lifetime_limits(&args, tight, fixed).is_ok());
    let mut input = fixed; input.max_input_bytes_per_item += 1;
    assert_eq!(c.args.lifetime_limits(&args, map, input).unwrap_err().code, "stored_input_ceiling");
    let mut output = fixed; output.max_result_bytes -= 1;
    assert_eq!(c.args.lifetime_limits(&args, map, output).unwrap_err().code, "stored_document_result_ceiling");
    let mut huge = fixed; huge.max_snapshot_bytes = i64::MAX as u64;
    assert_eq!(c.args.lifetime_limits(&args, map, huge).unwrap_err().code, "job_memory_arithmetic");
}
#[test]
fn borrowed_dispatch_and_unconsented_owned_execution_never_touch_private_io() {
    let mut words = vec!["candidate".to_owned(), "source-int4".to_owned()];
    words.extend(argv("start", "answer"));
    let matches = crate::candidate_cli::definition().try_get_matches_from(words).unwrap();
    let CandidateCommand::SourceInt4(c) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong document dispatcher"); };
    assert_eq!(c.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    let (mut c, args) = command("resume", "answer", &[]); c.args.storage.store_results = false;
    let mut output = Vec::new(); let mut diagnostics = Vec::new();
    assert_eq!(c.run_owned(args, NoIo, &mut output, &mut diagnostics), ErrorCode::Usage.as_process_exit());
    assert!(output.is_empty());
    let report: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
    assert_eq!(report["durable_progress_may_exist"], true);
    let text = std::str::from_utf8(&diagnostics).unwrap();
    for private in ["private-question.txt", "private/key", "private/job", "0123456789abcdef"] {
        assert!(!text.contains(private));
    }
}
#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unavailable_profile_refuses_before_keys_questions_options_model_or_population() {
    for &(task, flags, _) in MODES {
        let (c, args) = command("resume", task, flags);
        assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code, "owned_document_job_profile_unavailable");
    }
}
