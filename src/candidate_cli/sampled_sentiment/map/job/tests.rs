//! Durable CLI parsing, no-IO refusals and nonrenewable resource authority.
use super::*;
use std::io;
use crate::{
    batch::sentiment::sampled::quantized::long::int8_sampled_sentiment_document_preparation_bytes,
    jobs::{JobInputMemory, JobWork},
};

const SEED: &str = "4545454545454545454545454545454545454545454545454545454545454545";
fn argv(operation: &str) -> Vec<String> {
    ["sentiment-sampled-document-job", operation, "--model", "private-model.fnlpq",
        "--memory-mib", "131072", "--seed", SEED, "--job-dir", "private-job",
        "--key-file", "private-key", "--limits", "private-limits",
        "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"]
        .into_iter().map(str::to_owned).collect()
}
fn command(operation: &str, flags: &[&str]) -> SampledSentimentDocumentJobCommand {
    let mut words = argv(operation);
    words.extend(flags.iter().map(|word| (*word).to_owned()));
    SampledSentimentDocumentJobCommand::from_matches(&definition().try_get_matches_from(words).unwrap()).unwrap()
}
fn lifetime(command: &SampledSentimentDocumentJobCommand, document: &MapLimits) -> JobLimits {
    let work = command.args.document.sampled.work_ceiling();
    JobLimits { max_items: 3, max_id_bytes: 128,
        max_input_bytes_per_item: document.common.max_input_bytes, max_snapshot_bytes: 4 * MIB,
        max_result_bytes: document.mapping.max_result_bytes as usize,
        max_spool_bytes: 256 * MIB, max_materialized_bytes: 256 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 6,
        max_work: JobWork { model: work.checked_add(work).unwrap(),
            mask_node_visits: 2 * command.args.document.sampled.max_mask_node_visits } }
}
struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused job read private input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused job published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused job flushed output"); }
}

#[test]
fn complete_document_job_preserves_sampler_source_and_all_physical_schedule_options() {
    for operation in ["start", "resume"] {
        let c = command(operation, &["population.ndjson", "--samples", "7", "--temperature-milli", "800",
            "--top-p-ppm", "950000", "--top-k", "9", "--max-chunks", "11",
            "--max-chunk-bytes", "2048", "--max-tokenizer-calls", "1234",
            "--input-encoding", "safe-bpe-v1", "--compute-threads", "4", "--prefill-rows", "7"]);
        let document = c.args.validate().unwrap();
        assert_eq!(c.mode.name(), operation);
        assert_eq!(document.common.input, PathBuf::from("population.ndjson"));
        let options = c.args.document.sampled.options().unwrap();
        assert_eq!((options.samples, options.temperature_milli, options.top_p_ppm, options.top_k),
            (7, 800, 950000, Some(9)));
        assert_eq!(document.shared.compute.threads, 4);
        assert_eq!(document.prefill.unwrap().max_batch_rows, 7);
        assert_eq!(document.mapping.chunks.max_chunks, 11);
        assert_eq!(document.mapping.chunks.max_chunk_bytes, 2048);
        assert_eq!(document.mapping.chunks.max_tokenizer_calls, 1234);
        assert!(matches!(document.encoding, UntrustedEncoding::SafeBpeV1(_)));
    }
    let resumed = command("resume", &["--discard-uncommitted-tail"]);
    assert!(matches!(resumed.mode, RunMode::Resume { discard_uncommitted: true }));
    for flags in [vec!["--discard-uncommitted-tail"], vec!["--reuse-prefill"],
        vec!["--bins", "5"], vec!["--ndjson"]] {
        let mut words = argv("start"); words.extend(flags.into_iter().map(str::to_owned));
        assert!(definition().try_get_matches_from(words).is_err());
    }
}

#[test]
fn complete_document_job_projection_cap_is_explicit_for_start_and_resume() {
    for operation in ["start", "resume"] {
        let dense = command(operation, &[]);
        let selected = command(operation, &["--max-selected-rows", "4096"]);
        let a = dense.args.validate().unwrap(); let b = selected.args.validate().unwrap();
        assert_eq!(selected.args.document.sampled.selected_rows().unwrap(),
            Some(Int8JsonSparseLimits { max_rows_per_step: 4096 }));
        assert_eq!(a.memory, b.memory); assert_eq!(a.minimum_memory_bytes, b.minimum_memory_bytes);
        assert_eq!(dense.args.document.sampled.options().unwrap(), selected.args.document.sampled.options().unwrap());
        for cap in ["0", "166145"] { assert!(command(operation, &["--max-selected-rows", cap]).args.validate().is_err()); }
    }
}

#[test]
fn top_level_candidate_dispatch_selects_owned_complete_document_jobs() {
    for operation in ["start", "resume"] {
        let mut words = vec!["candidate".to_owned()];
        words.extend(argv(operation));
        let matches = crate::candidate_cli::definition().try_get_matches_from(words).unwrap();
        let CandidateCommand::SampledSentimentDocumentJob(mut selected) =
            CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("complete sampled document job route required"); };
        assert_eq!(selected.mode.name(), operation);
        selected.args.storage.store_results = false;
        let mut diagnostics = Vec::new();
        assert_eq!(selected.run_owned(NoIo, &mut NoIo, &mut diagnostics), ErrorCode::Usage.as_process_exit());
        let report: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
        assert_eq!(report["operation"], operation);
        assert_eq!(report["code"], "int8_sampled_document_job_storage");
    }
}

#[test]
fn all_six_lifetime_axes_remain_exact_as_transport_items_and_attempts_change() {
    let mut c = command("resume", &[]);
    let document = c.args.validate().unwrap();
    let fixed = lifetime(&c, &document);
    let a = c.args.lifetime_limits(document, fixed).unwrap();
    assert_eq!(a.native.max_model_work, fixed.max_work.model);
    assert_eq!(a.native.masks.max_visits_per_run, fixed.max_work.mask_node_visits);
    assert_eq!(a.native.max_item_work, a.document.mapping.max_model_work);
    let mut more = fixed; more.max_items += 20; more.max_attempts += 40;
    c.args.storage.max_stream_mib = 1024;
    c.args.storage.max_input_lines = 1_000_000;
    let b = c.args.lifetime_limits(c.args.validate().unwrap(), more).unwrap();
    assert_eq!(a.native.max_item_work, b.native.max_item_work);
    assert_eq!(a.native.max_model_work, b.native.max_model_work);
    assert_eq!(a.native.masks.max_visits_per_run, b.native.masks.max_visits_per_run);
    assert_eq!(a.native.max_sampler_bytes, b.native.max_sampler_bytes);
    assert!(b.buffer_bytes > a.buffer_bytes);
    let mut exact = fixed;
    exact.max_work = JobWork { model: a.native.max_item_work,
        mask_node_visits: a.native.masks.max_visits_per_item };
    c.args.lifetime_limits(c.args.validate().unwrap(), exact).unwrap();
    for axis in 0..6 {
        let mut short = exact;
        match axis {
            0 => short.max_work.model.forward_positions -= 1,
            1 => short.max_work.model.projected_logits -= 1,
            2 => short.max_work.model.attention_pairs -= 1,
            3 => short.max_work.model.projections.dot_products -= 1,
            4 => short.max_work.model.projections.multiply_accumulates -= 1,
            _ => short.max_work.mask_node_visits -= 1,
        }
        assert_eq!(c.args.lifetime_limits(c.args.validate().unwrap(), short).err().unwrap().code,
            "int8_sampled_document_job_lifetime_work", "axis {axis}");
    }
}

#[test]
fn exact_map_preparation_output_and_simultaneous_memory_are_independently_priced() {
    for flags in [&[][..], &["--compute-threads", "64"][..], &["--prefill-rows", "64"][..],
        &["--input-encoding", "safe-bpe-v1", "--compute-threads", "4", "--prefill-rows", "7"][..]] {
        let c = command("start", flags);
        let document = c.args.validate().unwrap();
        let fixed = lifetime(&c, &document);
        let limits = c.args.lifetime_limits(document, fixed).unwrap();
        let d = &limits.document;
        let prepared = int8_sampled_sentiment_document_preparation_bytes(d.task, limits.planning,
            d.mapping, &c.args.document.sampled.options().unwrap(), &d.encoding).unwrap();
        assert_eq!(limits.preparation_reserve_bytes, prepared + SAMPLED_SENTIMENT_JOB_RECIPE_BYTES as u64);
        assert!(prepared >= MIN_INT8_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES);
        assert_eq!(d.memory.max_output_bytes, d.mapping.max_result_bytes);
        assert!(d.memory.max_output_bytes > d.task.max_output_bytes);
        assert_eq!(limits.execution_bytes, limits.buffer_bytes + d.memory.native.kv_bytes
            + d.memory.workspace_bytes + d.memory.output_reservation_bytes + d.shared.compute.workspace_bytes);
        let execution = limits.execution_bytes + d.shared.weight_bytes + 192 * MIB + 65_536;
        let loading = d.shared.weight_bytes + 192 * MIB + 64 * 1024 + 65_536;
        assert_eq!(d.minimum_memory_bytes, d.shared.preparation_bytes + execution.max(loading));
        let exact = d.minimum_memory_bytes;
        let mut short = c.args.validate().unwrap();
        short.shared.memory_bytes = exact - 1;
        assert_eq!(c.args.lifetime_limits(short, fixed).err().unwrap().exit, ErrorCode::AdmissionOrResourceLimit);
        let mut enough = c.args.validate().unwrap();
        enough.shared.memory_bytes = exact;
        c.args.lifetime_limits(enough, fixed).unwrap();
    }
}

#[test]
fn immutable_input_and_per_chunk_and_document_result_ceilings_all_apply() {
    let c = command("start", &[]);
    let document = c.args.validate().unwrap();
    let fixed = lifetime(&c, &document);
    let mut narrow = fixed; narrow.max_input_bytes_per_item /= 2;
    let accepted = c.args.lifetime_limits(document, narrow).unwrap();
    assert_eq!(accepted.document.mapping.chunks.max_input_bytes, narrow.max_input_bytes_per_item);
    assert_eq!(accepted.planning.max_input_bytes, narrow.max_input_bytes_per_item);
    for case in 0..3 {
        let mut wrong = fixed;
        match case { 0 => wrong.max_input_bytes_per_item += 1,
            1 => wrong.max_id_bytes = 257, _ => wrong.max_result_bytes -= 1 }
        assert_eq!(c.args.lifetime_limits(c.args.validate().unwrap(), wrong).err().unwrap().code,
            if case == 2 { "int8_sampled_document_job_result_ceiling" }
            else { "int8_sampled_document_job_input_ceiling" });
    }
    let mut small_map = c.args.validate().unwrap();
    let mut short_leaf = fixed; short_leaf.max_result_bytes = small_map.task.max_output_bytes as usize - 1;
    small_map.mapping.max_result_bytes = 1;
    assert_eq!(c.args.lifetime_limits(small_map, short_leaf).err().unwrap().code,
        "int8_sampled_document_job_result_ceiling");
}

#[test]
fn retained_creation_and_replay_keep_the_same_work_and_replay_never_reads_external_input() {
    let original = command("resume", &[]);
    let document = original.args.validate().unwrap();
    let fixed = lifetime(&original, &document);
    let ordinary = original.args.lifetime_limits(document, fixed).unwrap();
    for (operation, flags) in [
        ("start", &["--spool-input", "--max-retained-input-mib", "8"][..]),
        ("resume", &["--use-retained-input", "--max-retained-input-mib", "8"][..]),
    ] {
        let retained = command(operation, flags);
        let streamed = retained.args.lifetime_limits(retained.args.validate().unwrap(), fixed).unwrap();
        assert_eq!(ordinary.buffer_bytes - streamed.buffer_bytes, 2 * fixed.max_snapshot_bytes);
        assert_eq!(ordinary.native.max_model_work, streamed.native.max_model_work);
        assert_eq!(ordinary.native.masks.max_visits_per_run, streamed.native.masks.max_visits_per_run);
        assert_eq!(retained.args.storage.input_retention.input_memory().unwrap(), JobInputMemory::RetainedRecord);
        let mut larger = fixed; larger.max_snapshot_bytes *= 100;
        let large = retained.args.lifetime_limits(retained.args.validate().unwrap(), larger).unwrap();
        assert_eq!(streamed.buffer_bytes, large.buffer_bytes);
        if operation == "resume" {
            let mut reader = job_cli::population_reader(retained.mode, retained.args.storage.input_retention,
                &PathBuf::from("-"), NoIo, IO_BUFFER_BYTES).unwrap();
            assert_eq!(reader.read(&mut [0]).unwrap(), 0);
        }
    }
}

#[test]
fn invalid_storage_mapping_sampler_and_retention_refuse_before_private_io() {
    for fault in 0..9 {
        let mut c = command("start", &[]);
        match fault {
            0 => c.args.storage.store_results = false,
            1 => c.args.storage.job_dir = PathBuf::new(),
            2 => c.args.storage.key_file = PathBuf::from("-"),
            3 => c.args.storage.limits_file = PathBuf::from("-"),
            4 => c.args.storage.max_stream_mib = 0,
            5 => c.args.storage.max_input_lines = 0,
            6 => c.args.storage.journal_memory_mib = 0,
            7 => c.args.storage.serialization_memory_mib = 0,
            _ => c.args.storage.max_stream_mib = u64::MAX,
        }
        assert_eq!(c.execute_owned(NoIo, &mut NoIo).unwrap_err().code,
            "int8_sampled_document_job_storage", "case {fault}");
    }
    for flags in [
        &["--samples", "0"][..], &["--max-chunks", "0"][..], &["--max-chunks", "257"][..],
        &["--max-map-result-bytes", "0"][..], &["--max-tokenizer-calls", "0"][..],
        &["--prefill-rows", "65"][..], &["--compute-threads", "65"][..],
        &["--max-mask-node-visits", "0"][..], &["--max-forward-positions", "0"][..],
    ] {
        assert_eq!(command("start", flags).execute_owned(NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
    }
    for (operation, flags) in [
        ("start", &["--use-retained-input", "--max-retained-input-mib", "8"][..]),
        ("resume", &["--spool-input", "--max-retained-input-mib", "8"][..]),
    ] {
        assert_eq!(command(operation, flags).execute_owned(NoIo, &mut NoIo).unwrap_err().code,
            "job_input_retention_options");
    }
}

#[test]
fn failed_command_reports_only_metadata_and_does_not_imply_durable_rollback() {
    let mut c = command("resume", &[]);
    c.args.document.sampled.seed = "private-invalid-seed".into();
    c.args.document.sampled.input = PathBuf::from("private-document-path");
    let mut diagnostics = Vec::new();
    let exit = c.run_owned(NoIo, &mut NoIo, &mut diagnostics);
    assert_eq!(exit, ErrorCode::Usage.as_process_exit());
    let report: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
    assert_eq!(report["operation"], "resume");
    assert_eq!(report["code"], "int8_sampled_document_job_options");
    assert_eq!(report["durable_progress_may_exist"], true);
    let text = String::from_utf8(diagnostics).unwrap();
    for secret in [SEED, "private-invalid-seed", "private-document-path", "private-key", "private-job"] {
        assert!(!text.contains(secret));
    }
}

#[test]
fn broken_diagnostics_preserve_an_already_observed_cancellation_exit() {
    use crate::native_engine::decode::DecodeCancellationKind::*;
    struct BrokenDiagnostics;
    impl Write for BrokenDiagnostics {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { Err(io::Error::other("broken diagnostic sink")) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("broken diagnostic sink")) }
    }
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, LinkedExit, Shutdown] {
        let exit = match cause {
            User | ParentCancelled | Shutdown => ErrorCode::Cancelled,
            Timeout | Deadline | PollQuota | CostBudget => ErrorCode::BudgetOrTimeout,
            ResourceUnavailable => ErrorCode::AdmissionOrResourceLimit,
            FailFast | RaceLost | LinkedExit => ErrorCode::Generic,
        };
        let failure = Failure { exit, code: "fixture_cancelled_job", cancellation: Some(cause) };
        assert_eq!(report_job_failure("resume", failure, &mut BrokenDiagnostics), exit.as_process_exit(),
            "cause {cause:?}");
    }
    let failure = Failure::usage("fixture_options");
    assert_eq!(report_job_failure("start", failure, &mut BrokenDiagnostics), ErrorCode::Generic.as_process_exit());
    let mut healthy = Vec::new();
    assert_eq!(report_job_failure("start", failure, &mut healthy), ErrorCode::Usage.as_process_exit());
}

#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unavailable_profile_refuses_valid_job_without_private_io() {
    assert_eq!(command("start", &[]).execute_owned(NoIo, &mut NoIo).unwrap_err().code,
        "int8_sampled_document_job_profile_unavailable");
}
