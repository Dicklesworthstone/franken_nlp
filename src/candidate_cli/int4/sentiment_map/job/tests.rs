//! Pure admission, fixed private settings and immutable lifetime regressions.
use super::*;
use crate::{jobs::JobWork, tasks::sentiment::SentimentAxis};

fn argv(operation: &str) -> Vec<String> {
    ["sentiment-document-job", operation, "--source-dir", "private-source", "--source-manifest", "private-manifest",
        "--memory-mib", "131072", "--max-chunks", "8", "--job-dir", "private-job",
        "--key-file", "private-key", "--limits", "private-limits",
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
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused sentiment job read private population"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused sentiment job published metadata"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused sentiment job flushed metadata"); }
}

#[test]
fn complete_documents_and_cohorts_keep_one_lifetime_and_price_durable_chunk_metadata() {
    for operation in ["start", "resume"] {
        for flags in [vec![], vec!["--cohort-chunks", "1"], vec!["--cohort-chunks", "3"]] {
            let (c, args) = command(operation, &flags); let map = c.args.validate(&args).unwrap();
            let fixed = lifetime(&args, map); let limits = c.args.lifetime_limits(&args, map, fixed).unwrap();
            assert_eq!(c.mode.name(), operation);
            assert_eq!(limits.map.mapping.max_model_work, fixed.max_work.int4_model_work());
            assert_eq!(fixed.max_work.mask_node_visits, 0);
            assert_eq!(limits.map.common.minimum_memory_bytes,
                map.common.minimum_memory_bytes + limits.buffer_bytes + 8 * SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES);
            assert_eq!(limits.preparation_reserve_bytes, map.common.preparation_bytes);
            let mut more_items = fixed; more_items.max_items += 10; more_items.max_attempts += 20;
            let larger = c.args.lifetime_limits(&args, map, more_items).unwrap();
            assert_eq!(larger.map.mapping.max_model_work, limits.map.mapping.max_model_work);
            assert_eq!(larger.map.mapping.chunks.max_chunks, limits.map.mapping.chunks.max_chunks);
            assert!(larger.buffer_bytes > limits.buffer_bytes);
        }
    }
}

#[test]
fn exact_serial_or_cohort_result_and_all_five_lifetime_counters_are_required() {
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
                "lifetime_sentiment_document_work");
        }
        let mut masks = fixed; masks.max_work.mask_node_visits = 1;
        assert_eq!(c.args.lifetime_limits(&args, map, masks).unwrap_err().code, "lifetime_sentiment_document_work");
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
fn stored_document_ceiling_narrows_the_recipe_without_renewing_work_or_shrinking_pre_io_quote() {
    let (c, args) = command("resume", &[]); let map = c.args.validate(&args).unwrap();
    let mut fixed = lifetime(&args, map); fixed.max_input_bytes_per_item /= 2;
    let limited = c.args.lifetime_limits(&args, map, fixed).unwrap();
    assert_eq!(limited.map.mapping.chunks.max_input_bytes, fixed.max_input_bytes_per_item);
    assert_eq!(limited.map.mapping.max_model_work, map.mapping.max_model_work);
    assert_eq!(limited.map.memory.input_bytes, map.memory.input_bytes);
    assert_eq!(limited.map.common.minimum_memory_bytes,
        map.common.minimum_memory_bytes + limited.buffer_bytes + 8 * SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES);
}

#[test]
fn private_defaults_preserve_axes_order_and_policy_but_cannot_supply_authority() {
    let (c, args) = command("start", &[]); let map = c.args.validate(&args).unwrap();
    let budget = c.args.map.budget(map.common).unwrap();
    let (fallback, policy) = c.args.load_settings(None, budget, &args).unwrap();
    let (empty, empty_policy) = c.args.load_settings(Some("{}"), budget, &args).unwrap();
    assert_eq!(fallback.axes, SentimentAxis::ALL); assert_eq!(empty.axes, fallback.axes);
    assert_eq!(empty_policy, policy); assert_eq!(fallback.budget, budget);
    let valid = serde_json::json!({"axes":["approach","valence"],
        "policy":{"minimum_peak_weight_ppm":650000,"maximum_normalized_entropy_ppm":800000}});
    let (settings, policy) = c.args.load_settings(Some(&valid.to_string()), budget, &args).unwrap();
    assert_eq!(settings.axes, [SentimentAxis::Approach, SentimentAxis::Valence]);
    assert_eq!(settings.budget, budget); assert_eq!(policy.minimum_peak_weight_ppm, 650_000);
    for field in ["document", "budget", "model", "scores", "aggregate", "chunks", "task_args", "prompt",
        "cohort_chunks", "minimum_estimated_coverage_ppm"] {
        let mut injected = valid.clone(); injected[field] = serde_json::json!({});
        assert_eq!(c.args.load_settings(Some(&injected.to_string()), budget, &args).err(), Some(CandidateError::Input));
    }
    for invalid in [r#"{"axes":[]}"#, r#"{"axes":["valence","valence"]}"#,
        r#"{"axes":["valence"],"axes":["arousal"]}"#, r#"{"axes":["politeness"]}"#,
        r#"{"policy":{"minimum_peak_weight_ppm":1000001}}"#,
        r#"{"policy":{"maximum_normalized_entropy_ppm":1000001}}"#] {
        assert_eq!(c.args.load_settings(Some(invalid), budget, &args).err(), Some(CandidateError::Input));
    }
}

#[test]
fn exact_fixed_axes_geometry_is_admitted_before_source_or_population_loading() {
    let mut words = argv("start"); let at = words.iter().position(|word| word == "--max-chunks").unwrap();
    words[at + 1] = "64".to_owned(); words.extend(["--cohort-chunks", "22"].into_iter().map(str::to_owned));
    let (c, args) = parse(words).unwrap(); let map = c.args.validate(&args).unwrap();
    let budget = c.args.map.budget(map.common).unwrap();
    assert!(c.args.load_settings(Some(r#"{"axes":["valence","arousal"]}"#), budget, &args).is_ok());
    assert_eq!(c.args.load_settings(Some(r#"{"axes":["valence","arousal","approach"]}"#), budget, &args).err(),
        Some(CandidateError::Planning));
    assert_eq!(c.args.load_settings(None, budget, &args).err(), Some(CandidateError::Planning));
}

#[test]
fn explicit_storage_and_valid_geometry_precede_all_private_io() {
    for fault in 0..8 {
        let (mut c, args) = command("start", &[]);
        match fault { 0 => c.args.storage.store_results = false,
            1 => c.args.sentiment.defaults = Some(PathBuf::from("-")),
            2 => c.args.storage.job_dir = PathBuf::new(), 3 => c.args.storage.key_file = PathBuf::from("-"),
            4 => c.args.storage.max_stream_mib = 0, 5 => c.args.storage.max_input_lines = 0,
            6 => c.args.storage.serialization_memory_mib = 0, _ => c.args.storage.limits_file = PathBuf::from("-") }
        assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code,
            "explicit_sentiment_document_job_storage");
    }
    for flags in [vec!["--max-chunks", "0"], vec!["--cohort-chunks", "9"], vec!["--cohort-chunks", "0"],
        vec!["--max-tokenizer-calls", "0"], vec!["--minimum-estimated-coverage-ppm", "1000001"],
        vec!["--max-forward-positions", "0"]] {
        let mut words = argv("start");
        if flags[0] == "--max-chunks" {
            let at = words.iter().position(|word| word == "--max-chunks").unwrap(); words[at + 1] = flags[1].to_owned();
        } else { words.extend(flags.iter().map(|word| (*word).to_owned())); }
        let (c, args) = parse(words).unwrap();
        assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
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
    assert_eq!(source.args.map.budget(original.common).unwrap(), snapshot.args.map.budget(loaded.common).unwrap());
    assert!(definition().try_get_matches_from(["sentiment-document-job"]).is_err());
    let mut start = argv("start"); start.push("--discard-uncommitted-tail".to_owned()); assert!(parse(start).is_err());
    let (resume, _) = command("resume", &["--discard-uncommitted-tail"]);
    assert!(matches!(resume.mode, RunMode::Resume { discard_uncommitted: true }));
}

#[test]
fn owned_root_dispatch_cannot_fall_through_to_single_document_execution() {
    let mut words = vec!["candidate".to_owned(), "source-int4".to_owned()]; words.extend(argv("start"));
    let matches = crate::candidate_cli::definition().try_get_matches_from(words).unwrap();
    let CandidateCommand::SourceInt4(c) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("source INT4 required"); };
    assert_eq!(c.task, Int4Task::SentimentDocumentJob);
    assert_eq!(c.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    for unsupported in [vec!["--cohort-heads", "1"], vec!["--prefill-rows", "1"],
        vec!["--max-new-tokens", "128"], vec!["--logprobs"], vec!["--task", "sentiment"]] {
        let mut words = argv("start"); words.extend(unsupported.into_iter().map(str::to_owned));
        assert!(parse(words).is_err());
    }
}
