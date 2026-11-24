//! Model-free durable CLI policy fixtures; no private job or model is opened.
use super::*;
use crate::{jobs::JobWork, native_engine::{strict_int8::Int8Work, portable_int8::ProjectionWork},
    tasks::{extract::semantic::{ClaimRule, ClaimValueKind, SemanticLimits, SemanticVerificationSpec},
        ir::TaskBudget, judge::FaithfulnessPolicy}};

fn command(operation: &str, extra: &[&str]) -> JobCommand {
    let mut argv = vec!["candidate", "extract-semantic-job", operation,
        "--model", "must-not-open.fnlpq", "--memory-mib", "16384",
        "--schema", "schema.json", "--semantic-options", "semantic.json",
        "--max-input-bytes", "512", "--max-new-tokens", "128", "--max-result-bytes", "16384",
        "--job-dir", "private/job", "--job-id", "0123456789abcdef0123456789abcdef",
        "--key-file", "private/key", "--limits", "limits.json", "--store-results"];
    argv.extend_from_slice(extra);
    let matches = super::super::super::definition().try_get_matches_from(argv).unwrap();
    match CandidateCommand::from_matches(&matches).unwrap() {
        CandidateCommand::SemanticJob(command) => command,
        _ => panic!("wrong semantic retained-job route"),
    }
}
fn policy(limits: Limits) -> Int8SemanticOptions {
    Int8SemanticOptions {
        specification: SemanticVerificationSpec {
            schema_version: 1, experimental_opt_in: true, revision: "semantic-job-cli-test-v1".into(),
            claims: vec![ClaimRule { id: "amount-v1".into(), path: vec![], value_kind: ClaimValueKind::Number,
                prefix: "The amount is ".into(), suffix: ".".into() }],
            judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0,
                evidence_window_bytes: 512, max_evidence_windows: 1, max_evidence_spans: 1 },
            judge_budget: TaskBudget { max_input_tokens: 2016, max_output_tokens: 32,
                max_output_bytes: 65_536, max_grammar_states: 128, max_kv_bytes: limits.kv_bytes },
        },
        limits: SemanticLimits { max_fields: 2, max_claim_bytes: 256, max_total_claim_bytes: 1024,
            max_template_bytes: 2048, max_walk_nodes: 128, max_input_json_bytes: 16_384,
            max_total_prompt_tokens: 8192, max_total_projected_logits: 16_777_216,
            max_output_bytes: 1_048_576 },
        max_model_work: Int8Work { forward_positions: 4096, projected_logits: 64_000_000,
            attention_pairs: 4096 * 2048 * 44 * 48,
            projections: ProjectionWork { dot_products: 100_000_000_000,
                multiply_accumulates: 1_000_000_000_000_000 } },
    }
}
fn lifetime(command: &JobCommand, options: &Int8SemanticOptions) -> JobLimits {
    JobLimits { max_items: 3, max_id_bytes: 128,
        max_input_bytes_per_item: command.semantic.host.max_input_bytes,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: options.limits.max_output_bytes as usize,
        max_spool_bytes: 16 * MIB, max_materialized_bytes: 16 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 7,
        max_work: JobWork { model: options.max_model_work,
            mask_node_visits: command.semantic.host.max_mask_node_visits } }
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private population read"); }
}
struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("private result published"); }
    fn flush(&mut self) -> io::Result<()> { panic!("private result flushed"); }
}

#[test]
fn start_and_resume_keep_one_immutable_lifetime_across_all_physical_schedules() {
    for operation in ["start", "resume"] {
        for (threads, rows) in [("1", None), ("2", None), ("1", Some("4")), ("2", Some("4"))] {
            let mut extra = vec!["--compute-threads", threads, "--materialize"];
            if let Some(rows) = rows { extra.extend(["--prefill-rows", rows]); }
            let command = command(operation, &extra); let (args, limits) = command.validate().unwrap();
            let options = policy(limits); let job = lifetime(&command, &options);
            let envelope = command.lifetime_limits(&args, limits, &options, job).unwrap();
            assert_eq!(command.mode.name(), operation);
            assert_eq!(job_cli::job_id_hex(command.storage.job_id), "0123456789abcdef0123456789abcdef");
            assert!(command.storage.store_results && command.storage.materialize);
            assert_eq!(envelope.native.max_model_work, job.max_work.model);
            assert_eq!(envelope.native.masks.max_visits_per_run, job.max_work.mask_node_visits);
            assert_eq!(envelope.native.masks.max_visits_per_item, command.semantic.host.max_mask_node_visits);
            assert_eq!(envelope.encoding_work_limits, None);
            assert_eq!(envelope.native.max_model_work, options.max_model_work);
            assert_eq!(job.max_items, 3); assert_eq!(job.max_attempts, 7);
            assert_eq!(limits.compute, args.compute.admit_prefill(command.semantic.prefill().unwrap()).unwrap());
        }
    }
}

#[test]
fn every_lifetime_axis_and_complete_result_ceiling_must_cover_one_whole_attempt() {
    let command = command("start", &[]); let (args, limits) = command.validate().unwrap();
    let options = policy(limits); let original = lifetime(&command, &options);
    for index in 0..8 {
        let mut job = original;
        match index {
            0 => job.max_work.model.forward_positions -= 1,
            1 => job.max_work.model.projected_logits -= 1,
            2 => job.max_work.model.attention_pairs -= 1,
            3 => job.max_work.model.projections.dot_products -= 1,
            4 => job.max_work.model.projections.multiply_accumulates -= 1,
            5 => job.max_work.mask_node_visits -= 1,
            6 => job.max_result_bytes -= 1,
            _ => job.max_input_bytes_per_item += 1,
        }
        assert!(command.lifetime_limits(&args, limits, &options, job).is_err(), "axis {index}");
    }
    let mut options = options; options.specification.experimental_opt_in = false;
    assert!(command.lifetime_limits(&args, limits, &options, original).is_err());
}

#[test]
fn storage_consent_retention_schedule_and_tokenizer_refusals_precede_private_io() {
    for index in 0..6 {
        let mut command = command("start", &[]);
        match index {
            0 => command.storage.store_results = false,
            1 => command.storage.key_file = PathBuf::from("-"),
            2 => command.storage.job_dir = PathBuf::new(),
            3 => command.storage.limits_file = PathBuf::from("-"),
            4 => command.storage.journal_memory_mib = u64::MAX,
            _ => command.semantic.semantic_options = PathBuf::from("-"),
        }
        assert_eq!(command.execute_owned(NoRead, &mut NoWrite).err().unwrap().exit, ErrorCode::Usage);
    }
    for (operation, extra) in [
        ("start", vec!["--use-retained-input", "--max-retained-input-mib", "16"]),
        ("resume", vec!["--spool-input", "--max-retained-input-mib", "16"]),
        ("start", vec!["--compute-threads", "0"]), ("resume", vec!["--prefill-rows", "65"]),
        ("start", vec!["--max-total-tokenizer-input-bytes", "1"]),
        ("resume", vec!["--input-encoding", "safe-bpe-v1", "--max-total-tokenizer-pair-checks", "0"]),
    ] {
        assert_eq!(command(operation, &extra).execute_owned(NoRead, &mut NoWrite).err().unwrap().exit,
            ErrorCode::Usage, "{operation}: {extra:?}");
    }
}

#[test]
fn retained_replay_never_reads_input_and_compact_invocation_totals_do_not_replace_lifetime_work() {
    for operation in ["start", "resume"] {
        let retention = if operation == "start" { "--spool-input" } else { "--use-retained-input" };
        let command = command(operation, &[retention, "--max-retained-input-mib", "16",
            "--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "65536",
            "--max-tokenizer-output-tokens", "8192", "--max-total-tokenizer-input-bytes", "100000",
            "--max-total-tokenizer-output-tokens", "20000", "--max-total-tokenizer-pair-checks", "2000000",
            "--max-total-tokenizer-pair-bytes", "9000000"]);
        let (args, limits) = command.validate().unwrap(); let options = policy(limits);
        let job = lifetime(&command, &options);
        let envelope = command.lifetime_limits(&args, limits, &options, job).unwrap();
        assert_eq!(envelope.native.max_model_work, job.max_work.model);
        assert_eq!(envelope.encoding_work_limits, Some(SafeBpeWorkLimits {
            max_input_bytes: 100_000, max_output_tokens: 20_000,
            max_pair_checks: 2_000_000, max_pair_bytes: 9_000_000 }));
        assert_eq!(command.storage.input_retention.input_memory().unwrap(), crate::jobs::JobInputMemory::RetainedRecord);
        if operation == "resume" {
            let mut reader = job_cli::population_reader(command.mode, command.storage.input_retention,
                &command.semantic.input, NoRead, IO_BUFFER_BYTES).unwrap();
            let mut bytes = Vec::new();
            assert_eq!(reader.read_to_end(&mut bytes).unwrap(), 0);
            assert!(bytes.is_empty());
        }
    }
}
