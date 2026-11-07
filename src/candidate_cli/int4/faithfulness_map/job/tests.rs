//! Pure command admission and immutable complete-source lifetime contracts.
use super::*;
use crate::jobs::JobWork;

fn argv(operation: &str) -> Vec<String> {
    ["faithfulness-document-job", operation, "--source-dir", "private-source", "--source-manifest", "private-manifest",
        "--memory-mib", "131072", "--max-chunks", "8", "--job-dir", "private-job",
        "--key-file", "private-key", "--limits", "private-limits", "--policy", "private-policy",
        "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"]
        .into_iter().map(str::to_owned).collect()
}
fn parse(words: Vec<String>) -> Result<(JobCommand, Int4Args), clap::Error> {
    JobCommand::from_matches(&definition().try_get_matches_from(words)?)
}
pub(super) fn command(operation: &str, extra: &[&str]) -> (JobCommand, Int4Args) {
    let mut words = argv(operation); words.extend(extra.iter().map(|word| (*word).to_owned()));
    parse(words).unwrap()
}
pub(super) fn lifetime(args: &Int4Args, map: MapLimits) -> JobLimits {
    JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: args.max_input_bytes,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: map.memory.max_output_bytes as usize,
        max_spool_bytes: 256 * MIB, max_materialized_bytes: 256 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 6,
        max_work: JobWork::from_int4(map.mapping.max_model_work, 0) }
}
pub(super) struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused faithfulness job read private population"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused faithfulness job published metadata"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused faithfulness job flushed metadata"); }
}

#[test]
fn serial_and_ragged_cohorts_keep_one_lifetime_and_price_frozen_chunk_descriptors() {
    for operation in ["start", "resume"] {
        for flags in [vec![], vec!["--cohort-chunks", "1"], vec!["--cohort-chunks", "3"], vec!["--cohort-chunks", "8"]] {
            let (c, args) = command(operation, &flags); let map = c.args.validate(&args).unwrap();
            let fixed = lifetime(&args, map); let limits = c.args.lifetime_limits(&args, map, fixed).unwrap();
            assert_eq!(c.mode.name(), operation);
            assert_eq!(limits.map.mapping.max_model_work, fixed.max_work.int4_model_work());
            assert_eq!(fixed.max_work.mask_node_visits, 0);
            assert_eq!(limits.map.common.minimum_memory_bytes,
                map.common.minimum_memory_bytes + limits.buffer_bytes + 8 * FAITHFULNESS_DOCUMENT_CHUNK_METADATA_BYTES);
            assert_eq!(limits.preparation_reserve_bytes, map.common.preparation_bytes);
            let mut more_items = fixed; more_items.max_items += 10; more_items.max_attempts += 20;
            let larger = c.args.lifetime_limits(&args, map, more_items).unwrap();
            assert_eq!(larger.map.mapping.max_model_work, limits.map.mapping.max_model_work);
            assert_eq!(larger.map.mapping.chunks, limits.map.mapping.chunks);
            assert!(larger.buffer_bytes > limits.buffer_bytes);
        }
    }
}

#[test]
fn explicit_head_cap_lowers_start_and_resume_admission_without_changing_the_lifetime() {
    for operation in ["start", "resume"] {
        let (legacy, mut args) = command(operation, &["--cohort-chunks", "3"]);
        let old = legacy.args.validate(&args).unwrap();
        assert_eq!(legacy.args.faithfulness.max_evidence_heads, None);
        let (explicit, _) = command(operation, &["--cohort-chunks", "3", "--max-evidence-heads", "32"]);
        let unchanged = explicit.args.validate(&args).unwrap();
        assert_eq!(old.memory, unchanged.memory);
        assert_eq!(old.common.minimum_memory_bytes, unchanged.common.minimum_memory_bytes);
        assert_eq!(legacy.args.map.budget(old.common).unwrap(),
            explicit.args.map.budget_with_max_heads(unchanged.common, 32).unwrap());

        let (narrow, _) = command(operation, &["--cohort-chunks", "3", "--max-evidence-heads", "1"]);
        let small = narrow.args.validate(&args).unwrap();
        let fixed = lifetime(&args, small);
        let admitted = narrow.args.lifetime_limits(&args, small, fixed).unwrap();
        let legacy_admitted = legacy.args.lifetime_limits(&args, old, fixed).unwrap();
        assert_eq!(small.memory.native.kv_bytes, small.common.native.kv_bytes * 3);
        assert_eq!(narrow.args.map.budget_with_max_heads(small.common, 1).unwrap().max_kv_bytes,
            small.common.native.kv_bytes);
        assert!(admitted.map.common.minimum_memory_bytes < legacy_admitted.map.common.minimum_memory_bytes);
        assert_eq!(admitted.map.mapping.max_model_work, legacy_admitted.map.mapping.max_model_work);
        assert_eq!(admitted.map.mapping.max_model_work, fixed.max_work.int4_model_work());
        assert_eq!(admitted.map.mapping.chunks, legacy_admitted.map.mapping.chunks);
        assert_eq!(small.memory.max_output_bytes, old.memory.max_output_bytes);
        args.memory_mib = admitted.map.common.minimum_memory_bytes.div_ceil(MIB);
        let small = narrow.args.validate(&args).unwrap();
        narrow.args.lifetime_limits(&args, small, fixed).unwrap();
        if let Ok(old) = legacy.args.validate(&args) {
            assert_eq!(legacy.args.lifetime_limits(&args, old, fixed).unwrap_err().code, "job_population_memory");
        }
    }
}

#[test]
fn exact_complete_output_and_every_lifetime_work_axis_are_required() {
    for flags in [vec![], vec!["--cohort-chunks", "3"]] {
        let (c, args) = command("resume", &flags); let map = c.args.validate(&args).unwrap();
        let mut fixed = lifetime(&args, map); fixed.max_result_bytes = c.args.result_bytes(map).unwrap() as usize;
        let admitted = c.args.lifetime_limits(&args, map, fixed).unwrap();
        assert_eq!(fixed.max_result_bytes as u64, map.mapping.max_result_bytes + if flags.is_empty() { 0 } else { 4096 });
        for axis in 0..5 {
            let mut short = fixed;
            match axis { 0 => short.max_work.model.forward_positions -= 1,
                1 => short.max_work.model.projected_logits -= 1, 2 => short.max_work.model.attention_pairs -= 1,
                3 => short.max_work.model.projections.dot_products -= 1,
                _ => short.max_work.model.projections.multiply_accumulates -= 1 }
            assert_eq!(c.args.lifetime_limits(&args, map, short).unwrap_err().code,
                "lifetime_faithfulness_document_work");
        }
        let mut masks = fixed; masks.max_work.mask_node_visits = 1;
        assert_eq!(c.args.lifetime_limits(&args, map, masks).unwrap_err().code, "lifetime_faithfulness_document_work");
        let mut output = fixed; output.max_result_bytes -= 1;
        assert_eq!(c.args.lifetime_limits(&args, map, output).unwrap_err().code, "stored_result_ceiling");
        let mut input = fixed; input.max_input_bytes_per_item += 1;
        assert_eq!(c.args.lifetime_limits(&args, map, input).unwrap_err().code, "stored_input_ceiling");
        let mut tight = map; tight.common.memory_bytes = admitted.map.common.minimum_memory_bytes - 1;
        assert_eq!(c.args.lifetime_limits(&args, tight, fixed).unwrap_err().exit, ErrorCode::AdmissionOrResourceLimit);
        tight.common.memory_bytes += 1; c.args.lifetime_limits(&args, tight, fixed).unwrap();
    }
}

#[test]
fn stored_combined_source_and_claim_ceiling_narrows_recipe_but_retains_the_original_memory_quote() {
    let (c, args) = command("resume", &[]); let map = c.args.validate(&args).unwrap();
    let mut fixed = lifetime(&args, map); fixed.max_input_bytes_per_item /= 2;
    let limited = c.args.lifetime_limits(&args, map, fixed).unwrap();
    assert_eq!(limited.map.mapping.chunks.max_input_bytes, fixed.max_input_bytes_per_item);
    assert_eq!(limited.map.mapping.max_model_work, map.mapping.max_model_work);
    assert_eq!(limited.map.memory.input_bytes, map.memory.input_bytes);
    assert_eq!(limited.map.common.minimum_memory_bytes,
        map.common.minimum_memory_bytes + limited.buffer_bytes + 8 * FAITHFULNESS_DOCUMENT_CHUNK_METADATA_BYTES);
}

#[test]
fn required_fixed_policy_has_no_claim_or_execution_override_route() {
    for operation in ["start", "resume"] {
        let mut words = argv(operation); let at = words.iter().position(|word| word == "--policy").unwrap();
        words.drain(at..at + 2); assert!(parse(words).is_err());
        for unsupported in [vec!["--claim", "fixed"], vec!["--defaults", "other"], vec!["--cohort-heads", "1"],
            vec!["--prefill-rows", "1"], vec!["--max-selected-rows", "7"], vec!["--max-new-tokens", "128"],
            vec!["--logprobs"], vec!["--task", "faithfulness"]] {
            let mut words = argv(operation); words.extend(unsupported.into_iter().map(str::to_owned));
            assert!(parse(words).is_err());
        }
    }
    let valid = r#"{"minimum_candidate_weight_ppm":700000,"minimum_margin_milli":100,"evidence_window_bytes":256,"max_evidence_windows":31,"max_evidence_spans":31}"#;
    assert!(super::super::corpus::parse_policy(valid).is_ok());
    for field in ["claim", "source", "budget", "chunks", "task_args", "prompt", "cohort_chunks"] {
        let mut injected: serde_json::Value = serde_json::from_str(valid).unwrap(); injected[field] = serde_json::json!({});
        assert_eq!(super::super::corpus::parse_policy(&injected.to_string()), Err(CandidateError::Input));
    }
}

#[test]
fn explicit_storage_policy_and_geometry_refuse_before_all_private_io() {
    for fault in 0..9 {
        let (mut c, args) = command("start", &[]);
        match fault { 0 => c.args.storage.store_results = false,
            1 => c.args.faithfulness.policy = PathBuf::from("-"),
            2 => c.args.storage.job_dir = PathBuf::new(), 3 => c.args.storage.key_file = PathBuf::from("-"),
            4 => c.args.storage.max_stream_mib = 0, 5 => c.args.storage.max_input_lines = 0,
            6 => c.args.storage.serialization_memory_mib = 0, 7 => c.args.storage.limits_file = PathBuf::from("-"),
            _ => c.args.faithfulness.policy = PathBuf::new() }
        assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code,
            "explicit_faithfulness_document_job_storage");
    }
    for flags in [vec!["--max-chunks", "0"], vec!["--cohort-chunks", "9"], vec!["--cohort-chunks", "0"],
        vec!["--max-tokenizer-calls", "0"], vec!["--minimum-classified-coverage-ppm", "1000001"],
        vec!["--max-forward-positions", "0"], vec!["--max-evidence-heads", "0"],
        vec!["--max-evidence-heads", "33"]] {
        let mut words = argv("start");
        if flags[0] == "--max-chunks" {
            let at = words.iter().position(|word| word == "--max-chunks").unwrap(); words[at + 1] = flags[1].to_owned();
        } else { words.extend(flags.iter().map(|word| (*word).to_owned())); }
        let (c, args) = parse(words).unwrap();
        assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
    }
}

#[test]
fn source_and_snapshot_keep_identical_task_authority_and_tail_repair_is_resume_only() {
    let (source, source_args) = command("resume", &[]); let original = source.args.validate(&source_args).unwrap();
    let mut words = argv("resume"); let at = words.iter().position(|s| s == "--source-dir").unwrap();
    words.drain(at..at + 4);
    words.extend(["--snapshot", "private-snapshot", "--expected-model-sha256"].into_iter().map(str::to_owned));
    words.push("ab".repeat(32));
    let (snapshot, args) = parse(words).unwrap(); let loaded = snapshot.args.validate(&args).unwrap();
    assert!(loaded.common.snapshot_pin.is_some());
    assert_eq!(original.mapping.chunks, loaded.mapping.chunks);
    assert_eq!(original.mapping.max_model_work, loaded.mapping.max_model_work);
    assert_eq!(source.args.map.budget(original.common).unwrap(), snapshot.args.map.budget(loaded.common).unwrap());
    assert!(definition().try_get_matches_from(["faithfulness-document-job"]).is_err());
    let mut start = argv("start"); start.push("--discard-uncommitted-tail".to_owned()); assert!(parse(start).is_err());
    let (resume, _) = command("resume", &["--discard-uncommitted-tail"]);
    assert!(matches!(resume.mode, RunMode::Resume { discard_uncommitted: true }));
}

#[test]
fn owned_root_dispatch_never_falls_through_to_a_single_document() {
    let mut words = vec!["candidate".to_owned(), "source-int4".to_owned()]; words.extend(argv("start"));
    let matches = crate::candidate_cli::definition().try_get_matches_from(words).unwrap();
    let CandidateCommand::SourceInt4(c) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("source INT4 required") };
    assert_eq!(c.task, Int4Task::FaithfulnessDocumentJob);
    assert_eq!(c.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
}

#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unsupported_owned_profile_refuses_before_private_input_or_configuration() {
    let (c, args) = command("resume", &[]);
    assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code,
        "owned_faithfulness_document_job_profile_unavailable");
}
