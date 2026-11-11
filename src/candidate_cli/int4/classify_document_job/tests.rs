//! Pure admission, immutable work and private-settings contract regressions.
use super::*;
use crate::{jobs::JobWork, tasks::classify::ClassificationMode};

fn argv(operation: &str) -> Vec<String> {
    ["classify-document-job", operation, "--source-dir", "private-source", "--source-manifest", "private-manifest",
        "--memory-mib", "131072", "--max-chunks", "8", "--defaults", "private-labels.json",
        "--job-dir", "private-job", "--key-file", "private-key", "--limits", "private-limits",
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
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused classification job read private population"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused classification job published metadata"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused classification job flushed metadata"); }
}

#[test]
fn complete_documents_and_optional_cohorts_keep_one_durable_work_allowance() {
    for operation in ["start", "resume"] {
        for flags in [vec![], vec!["--cohort-chunks", "1"], vec!["--cohort-chunks", "3"]] {
            let (c, args) = command(operation, &flags); let map = c.args.validate(&args).unwrap();
            let fixed = lifetime(&args, map); let limits = c.args.lifetime_limits(&args, map, fixed).unwrap();
            assert_eq!(c.mode.name(), operation);
            assert_eq!(map.mapping.chunks.max_chunks, 8);
            assert_eq!(map.mapping.chunks.max_input_bytes, args.max_input_bytes);
            assert_eq!(limits.map.mapping.max_model_work, fixed.max_work.int4_model_work());
            assert_eq!(fixed.max_work.mask_node_visits, 0);
            assert_eq!(limits.map.memory.max_output_bytes as usize, fixed.max_result_bytes);
            assert_eq!(limits.map.common.minimum_memory_bytes, map.common.minimum_memory_bytes + limits.buffer_bytes);
            assert_eq!(limits.preparation_reserve_bytes, map.common.preparation_bytes);
            let mut more_items = fixed; more_items.max_items += 10; more_items.max_attempts += 20;
            let larger = c.args.lifetime_limits(&args, map, more_items).unwrap();
            assert_eq!(larger.map.mapping.max_model_work, limits.map.mapping.max_model_work);
            assert!(larger.buffer_bytes > limits.buffer_bytes);
        }
    }
}

#[test]
fn exact_full_result_and_all_five_lifetime_counters_are_required() {
    for flags in [vec![], vec!["--cohort-chunks", "3"]] {
        let (c, args) = command("resume", &flags); let map = c.args.validate(&args).unwrap();
        let fixed = lifetime(&args, map);
        for axis in 0..5 {
            let mut short = fixed;
            match axis { 0 => short.max_work.model.forward_positions -= 1,
                1 => short.max_work.model.projected_logits -= 1, 2 => short.max_work.model.attention_pairs -= 1,
                3 => short.max_work.model.projections.dot_products -= 1,
                _ => short.max_work.model.projections.multiply_accumulates -= 1 }
            assert_eq!(c.args.lifetime_limits(&args, map, short).unwrap_err().code,
                "lifetime_classification_document_work");
        }
        let mut masks = fixed; masks.max_work.mask_node_visits = 1;
        assert_eq!(c.args.lifetime_limits(&args, map, masks).unwrap_err().code, "lifetime_classification_document_work");
        let mut output = fixed; output.max_result_bytes -= 1;
        assert_eq!(c.args.lifetime_limits(&args, map, output).unwrap_err().code, "stored_result_ceiling");
        let mut input = fixed; input.max_input_bytes_per_item += 1;
        assert_eq!(c.args.lifetime_limits(&args, map, input).unwrap_err().code, "stored_input_ceiling");
        let admitted = c.args.lifetime_limits(&args, map, fixed).unwrap();
        let mut tight = map; tight.common.memory_bytes = admitted.map.common.minimum_memory_bytes - 1;
        assert_eq!(c.args.lifetime_limits(&args, tight, fixed).unwrap_err().exit, ErrorCode::AdmissionOrResourceLimit);
        tight.common.memory_bytes += 1; c.args.lifetime_limits(&args, tight, fixed).unwrap();
    }
}

#[test]
fn settings_preserve_private_labels_modes_and_policy_but_cannot_supply_authority() {
    let (c, args) = command("start", &[]); let map = c.args.validate(&args).unwrap();
    let budget = c.args.map.budget(map.common, map.planning.max_labels).unwrap();
    let valid = serde_json::json!({"labels":[{"id":"Billing issue","description":"Invoices"},
        {"id":"退款","description":"退款申请"}],"mode":"multi_label",
        "policy":{"minimum_candidate_weight_ppm":600000,"minimum_margin_ppm":100000}});
    let settings = c.args.load_settings(&valid.to_string(), budget, &args, map).unwrap();
    assert_eq!(settings.mode, ClassificationMode::MultiLabel); assert_eq!(settings.labels[1].id, "退款");
    assert_eq!(settings.labels[1].description, "退款申请"); assert_eq!(settings.budget, budget);
    assert_eq!(settings.policy.minimum_candidate_weight_ppm, 600000);
    for field in ["document", "budget", "model", "scores", "aggregate", "chunks", "task_args", "prompt",
        "cohort_chunks", "minimum_classified_coverage_ppm"] {
        let mut injected = valid.clone(); injected[field] = serde_json::json!({});
        assert_eq!(c.args.load_settings(&injected.to_string(), budget, &args, map).err(), Some(CandidateError::Input));
    }
    for bad in ["{}", r#"{"labels":[],"labels":[{"id":"a"},{"id":"b"}]}"#,
        r#"{"labels":[{"id":"a"},{"id":"a"}]}"#,
        r#"{"labels":[{"id":"a"}]}"#,
        r#"{"labels":[{"id":"a"},{"id":"b"}],"mode":"sampled"}"#] {
        assert_eq!(c.args.load_settings(bad, budget, &args, map).err(), Some(CandidateError::Input));
    }
}

#[test]
fn fixed_multilabel_geometry_is_checked_before_model_or_population_loading() {
    let (c, args) = command("start", &["--cohort-chunks", "3"]); let map = c.args.validate(&args).unwrap();
    let budget = c.args.map.budget(map.common, map.planning.max_labels).unwrap();
    let labels = |count: usize| (0..count).map(|i| serde_json::json!({"id":format!("label_{i}")})).collect::<Vec<_>>();
    for (count, accepted) in [(21, true), (22, false)] {
        let json = serde_json::json!({"labels": labels(count), "mode":"multi_label"}).to_string();
        assert_eq!(c.args.load_settings(&json, budget, &args, map).is_ok(), accepted);
    }
    // Exclusive mode has one head even with a large finite label language.
    let exclusive = serde_json::json!({"labels":labels(128)}).to_string();
    assert!(c.args.load_settings(&exclusive, budget, &args, map).is_ok());
}

#[test]
fn explicit_storage_and_valid_geometry_precede_all_private_io() {
    for fault in 0..7 {
        let (mut c, args) = command("start", &[]);
        match fault { 0 => c.args.storage.store_results = false,
            1 => c.args.classification.defaults = PathBuf::from("-"),
            2 => c.args.storage.job_dir = PathBuf::new(), 3 => c.args.storage.key_file = PathBuf::from("-"),
            4 => c.args.storage.max_stream_mib = 0, 5 => c.args.storage.max_input_lines = 0,
            _ => c.args.storage.serialization_memory_mib = 0 }
        assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code,
            "explicit_classification_document_job_storage");
    }
    for flags in [vec!["--max-chunks", "0"], vec!["--cohort-chunks", "9"], vec!["--cohort-chunks", "0"],
        vec!["--max-tokenizer-calls", "0"], vec!["--minimum-classified-coverage-ppm", "1000001"],
        vec!["--max-forward-positions", "0"]] {
        let mut words = argv("start");
        if flags[0] == "--max-chunks" {
            let at = words.iter().position(|word| word == "--max-chunks").unwrap();
            words[at + 1] = flags[1].to_owned();
        } else { words.extend(flags.iter().map(|word| (*word).to_owned())); }
        let (c, args) = parse(words).unwrap();
        assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
    }
}

#[test]
fn compact_document_jobs_keep_one_encoder_reservation_and_unchanged_native_lifetime_work() {
    for operation in ["start", "resume"] {
        for schedule in [vec![], vec!["--cohort-chunks", "2"]] {
            let (byte_command, byte_args) = command(operation, &schedule);
            let byte = byte_command.args.validate(&byte_args).unwrap();
            let mut flags = schedule;
            flags.extend(["--input-encoding", "safe-bpe-v1",
                "--max-tokenizer-input-bytes", "262144", "--max-tokenizer-output-tokens", "131072"]);
            let (compact_command, args) = command(operation, &flags);
            let compact = compact_command.args.validate(&args).unwrap();
            let crate::tokenizer::untrusted::UntrustedEncoding::SafeBpeV1(policy) =
                compact_command.args.map.input_encoding(&args).unwrap() else { panic!("compact job required"); };
            assert_eq!(compact.memory.input_bytes,
                byte.memory.input_bytes + policy.requirements().unwrap().workspace_bytes);
            assert_eq!(compact.memory.native, byte.memory.native);
            assert_eq!(compact.mapping.max_model_work, byte.mapping.max_model_work);
            let fixed = lifetime(&args, compact);
            let admitted = compact_command.args.lifetime_limits(&args, compact, fixed).unwrap();
            assert_eq!(admitted.map.mapping.max_model_work, fixed.max_work.int4_model_work());
            assert_eq!(admitted.map.common.minimum_memory_bytes,
                compact.common.minimum_memory_bytes + admitted.buffer_bytes);
        }
    }
}

#[test]
fn invalid_or_unfunded_compact_jobs_refuse_before_keys_storage_or_population_io() {
    for operation in ["start", "resume"] {
        for flags in [
            &["--max-tokenizer-output-tokens", "32"][..],
            &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "0"][..],
            &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-output-tokens", "1048577"][..],
            &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "0"][..],
            &["--input-encoding", "safe-bpe-v1", "--save-snapshot", "never-created.snapshot"][..],
        ] {
            let (command, args) = command(operation, flags);
            assert_eq!(command.execute_owned(args, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
        }
        let (command, mut args) = command(operation, &["--input-encoding", "safe-bpe-v1"]);
        let minimum = command.args.validate(&args).unwrap().common.minimum_memory_bytes;
        args.memory_mib = minimum.div_ceil(MIB) - 1;
        assert_eq!(command.execute_owned(args, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
    }
}

#[test]
fn source_and_pinned_snapshot_share_task_authority_and_resume_requires_an_operation() {
    let (source, source_args) = command("resume", &[]); let original = source.args.validate(&source_args).unwrap();
    let mut words = argv("resume"); let at = words.iter().position(|s| s == "--source-dir").unwrap();
    words.drain(at..at + 4);
    words.extend(["--snapshot", "private-snapshot", "--expected-model-sha256"].into_iter().map(str::to_owned));
    words.push("ab".repeat(32));
    let (snapshot, args) = parse(words).unwrap(); let loaded = snapshot.args.validate(&args).unwrap();
    assert!(loaded.common.snapshot_pin.is_some());
    assert_eq!(original.mapping.chunks, loaded.mapping.chunks);
    assert_eq!(original.mapping.max_model_work, loaded.mapping.max_model_work);
    assert_eq!(source.args.map.budget(original.common, original.planning.max_labels).unwrap(),
        snapshot.args.map.budget(loaded.common, loaded.planning.max_labels).unwrap());
    assert!(definition().try_get_matches_from(["classify-document-job"]).is_err());
    let mut no_defaults = argv("start"); let at = no_defaults.iter().position(|s| s == "--defaults").unwrap();
    no_defaults.drain(at..at + 2); assert!(parse(no_defaults).is_err());
    let mut start_repair = argv("start"); start_repair.push("--discard-uncommitted-tail".to_owned());
    assert!(parse(start_repair).is_err());
    assert!(matches!(command("resume", &["--discard-uncommitted-tail"]).0.mode,
        RunMode::Resume { discard_uncommitted: true }));
}

#[test]
fn owned_root_dispatch_cannot_fall_through_to_single_document_execution() {
    let mut words = vec!["candidate".to_owned(), "source-int4".to_owned()]; words.extend(argv("start"));
    let matches = crate::candidate_cli::definition().try_get_matches_from(words).unwrap();
    let CandidateCommand::SourceInt4(c) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("source INT4 required"); };
    assert_eq!(c.task, Int4Task::ClassificationDocumentJob);
    assert_eq!(c.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    for unsupported in [vec!["--cohort-heads", "1"], vec!["--prefill-rows", "1"],
        vec!["--max-new-tokens", "128"], vec!["--logprobs"], vec!["--task", "sentiment"]] {
        let mut words = argv("start"); words.extend(unsupported.into_iter().map(str::to_owned));
        assert!(parse(words).is_err());
    }
}
