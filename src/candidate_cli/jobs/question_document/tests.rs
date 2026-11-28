//! Parsing, private-IO refusal and immutable complete-document job authority.
use super::*;
use std::io;
use crate::{candidate_cli::CandidateCommand, jobs::{JobInputMemory, JobWork},
    tokenizer::untrusted::UntrustedEncoding};

fn argv(operation: &str) -> Vec<String> {
    ["answer-job", operation, "--model", "private-model.fnlpq", "--memory-mib", "131072",
        "--job-dir", "private-job", "--key-file", "private-key", "--limits", "private-limits",
        "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"]
        .into_iter().map(str::to_owned).collect()
}
fn parsed(operation: &str, flags: &[&str]) -> QuestionDocumentJobCommand {
    let mut words = argv(operation); words.extend(flags.iter().map(|word| (*word).to_owned()));
    QuestionDocumentJobCommand::from_matches(&definition().try_get_matches_from(words).unwrap()).unwrap()
}
fn lifetime(document: &QuestionDocumentLimits) -> JobLimits {
    let per = document.mapping.max_model_work;
    JobLimits { max_items: 3, max_id_bytes: 128,
        max_input_bytes_per_item: document.common.max_input_bytes + document.max_question_bytes + 4096,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: document.mapping.reduction.max_result_bytes,
        max_spool_bytes: 256 * MIB, max_materialized_bytes: 256 * MIB, max_journal_bytes: 64 * MIB,
        max_attempts: 6, max_work: JobWork { model: per.checked_add(per).unwrap(),
            mask_node_visits: document.mapping.max_mask_visits * 2 } }
}
struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused question job read private input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused question job published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused question job flushed output"); }
}

#[test]
fn start_and_resume_keep_exact_document_encoding_and_composed_physical_options() {
    for operation in ["start", "resume"] {
        let selected = parsed(operation, &["population.ndjson", "--options", "fixed-options.json",
            "--max-question-bytes", "4096", "--max-chunks", "11", "--max-chunk-bytes", "1536",
            "--max-tokenizer-calls", "1234", "--input-encoding", "safe-bpe-v1",
            "--compute-threads", "4", "--prefill-rows", "7"]);
        let document = selected.args.validate().unwrap();
        assert_eq!(selected.mode.name(), operation);
        assert_eq!(selected.args.input, PathBuf::from("population.ndjson"));
        assert_eq!(selected.args.document.options, Some(PathBuf::from("fixed-options.json")));
        assert_eq!(document.max_question_bytes, 4096);
        assert_eq!(document.mapping.chunks.max_chunks, 11);
        assert_eq!(document.mapping.chunks.max_chunk_bytes, 1536);
        assert_eq!(document.mapping.chunks.max_tokenizer_calls, 1234);
        assert_eq!(document.shared.compute.threads, 4);
        assert_eq!(document.prefill.unwrap().max_batch_rows, 7);
        assert!(matches!(document.encoding, UntrustedEncoding::SafeBpeV1(_)));
        assert!(document.encoding_work.is_some());
    }
    assert!(matches!(parsed("resume", &["--discard-uncommitted-tail"]).mode,
        RunMode::Resume { discard_uncommitted: true }));
    for flags in [vec!["--question", "fixed"], vec!["--synthesize-answer"], vec!["--cohort-chunks", "2"],
        vec!["--hierarchical-answer"], vec!["--reuse-prefill"], vec!["--max-selected-rows", "4096"],
        vec!["--discard-uncommitted-tail"]] {
        let mut words = argv("start"); words.extend(flags.into_iter().map(str::to_owned));
        assert!(definition().try_get_matches_from(words).is_err());
    }
}

#[test]
fn candidate_dispatch_and_storage_refusal_happen_before_private_io() {
    for operation in ["start", "resume"] {
        let mut words = vec!["candidate".to_owned()]; words.extend(argv(operation));
        let matches = crate::candidate_cli::definition().try_get_matches_from(words).unwrap();
        let CandidateCommand::QuestionDocumentJob(mut command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("owned complete-question job route required"); };
        command.args.storage.store_results = false;
        let mut diagnostics = Vec::new();
        assert_eq!(command.run_owned(NoIo, &mut NoIo, &mut diagnostics), ErrorCode::Usage.as_process_exit());
        let report: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
        assert_eq!(report["operation"], operation);
        assert_eq!(report["code"], "int8_question_document_job_storage");
        assert_eq!(report["durable_progress_may_exist"], true);
        let text = String::from_utf8(diagnostics).unwrap();
        for private in ["private-model", "private-key", "private-job", "private-limits"] {
            assert!(!text.contains(private));
        }
    }
    for fault in 0..5 {
        let mut command = parsed("start", &[]);
        match fault {
            0 => command.args.storage.job_dir = PathBuf::from("-"),
            1 => command.args.storage.key_file = PathBuf::new(),
            2 => command.args.storage.journal_memory_mib = 0,
            3 => command.args.document.max_question_bytes = 0,
            _ => command.args.storage.max_stream_mib = u64::MAX,
        }
        assert_eq!(command.execute_owned(NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
    }
}

#[test]
fn input_retention_is_explicit_and_cannot_fall_back_to_an_original_reader() {
    let start = parsed("start", &["--spool-input", "--max-retained-input-mib", "8"]);
    let resume = parsed("resume", &["--use-retained-input", "--max-retained-input-mib", "8"]);
    for command in [&start, &resume] {
        command.args.storage.input_retention.validate(command.mode, &command.args.input).unwrap();
        assert_eq!(command.args.storage.input_retention.input_memory().unwrap(), JobInputMemory::RetainedRecord);
    }
    let original = parsed("resume", &["original.ndjson"]);
    assert_eq!(original.args.storage.input_retention.input_memory().unwrap(), JobInputMemory::CompletePopulation);
    for (operation, flags) in [("start", vec!["--use-retained-input", "--max-retained-input-mib", "8"]),
        ("resume", vec!["--spool-input", "--max-retained-input-mib", "8"]),
        ("resume", vec!["original.ndjson", "--use-retained-input", "--max-retained-input-mib", "8"])] {
        let command = parsed(operation, &flags);
        assert!(command.args.storage.input_retention.validate(command.mode, &command.args.input).is_err());
    }
}

#[test]
fn transport_and_attempt_counts_do_not_multiply_six_immutable_native_counters() {
    let mut command = parsed("resume", &[]);
    let document = command.args.validate().unwrap(); let fixed = lifetime(&document);
    let before = command.args.lifetime_limits(document, fixed, AnswerOptions::default()).unwrap();
    let mut more = fixed; more.max_items = 60; more.max_attempts = 120;
    command.args.storage.max_input_lines = 1_000_000;
    command.args.storage.max_stream_mib = 1024;
    let after = command.args.lifetime_limits(command.args.validate().unwrap(), more, AnswerOptions::default()).unwrap();
    assert_eq!(before.config.max_model_work, fixed.max_work.model);
    assert_eq!(after.config.max_model_work, fixed.max_work.model);
    assert_eq!(before.config.max_mask_visits, fixed.max_work.mask_node_visits);
    assert_eq!(after.config.max_mask_visits, fixed.max_work.mask_node_visits);
    assert_eq!(after.config.max_verification.max_fields, after.document.verification.max_fields * 120);
    assert_eq!(after.config.max_verification.max_matches, after.document.verification.max_matches * 120);
    assert_eq!(after.config.max_verification.max_scan_steps, after.document.verification.max_scan_steps * 120);
    assert_eq!(after.config.verification, before.config.verification);
}

#[test]
fn every_native_or_mask_axis_must_cover_a_complete_document_before_any_io() {
    let command = parsed("start", &[]);
    let document = command.args.validate().unwrap(); let fixed = lifetime(&document);
    for axis in 0..6 {
        let mut short = fixed; short.max_work = JobWork {
            model: document.mapping.max_model_work, mask_node_visits: document.mapping.max_mask_visits };
        match axis {
            0 => short.max_work.model.forward_positions -= 1,
            1 => short.max_work.model.projected_logits -= 1,
            2 => short.max_work.model.attention_pairs -= 1,
            3 => short.max_work.model.projections.dot_products -= 1,
            4 => short.max_work.model.projections.multiply_accumulates -= 1,
            _ => short.max_work.mask_node_visits -= 1,
        }
        assert!(command.args.lifetime_limits(command.args.validate().unwrap(), short,
            AnswerOptions::default()).is_err(), "axis {axis}");
    }
}

#[test]
fn encoded_envelope_complete_result_and_all_verification_overflows_are_refused() {
    let command = parsed("start", &[]);
    let document = command.args.validate().unwrap(); let fixed = lifetime(&document);
    // A valid encoded envelope may exceed the decoded-source cap. Its full
    // parser and population payload remains explicitly quoted.
    assert!(fixed.max_input_bytes_per_item > document.common.max_input_bytes);
    command.args.lifetime_limits(document, fixed, AnswerOptions::default()).unwrap();
    for axis in 0..3 {
        let mut bad = fixed;
        match axis { 0 => bad.max_input_bytes_per_item = 1,
            1 => bad.max_result_bytes = 1, _ => bad.max_attempts = i64::MAX as u64 }
        assert!(command.args.lifetime_limits(command.args.validate().unwrap(), bad,
            AnswerOptions::default()).is_err());
    }
    for per in [GroundingBudget { max_fields: usize::MAX, max_matches: 1, max_scan_steps: 1 },
        GroundingBudget { max_fields: 1, max_matches: usize::MAX, max_scan_steps: 1 },
        GroundingBudget { max_fields: 1, max_matches: 1, max_scan_steps: u64::MAX }] {
        assert!(verification_lifetime(per, 2).is_err());
    }
}

#[test]
fn complete_job_peak_includes_separate_caller_and_transferred_preparation() {
    let command = parsed("start", &["--prefill-rows", "7", "--compute-threads", "4"]);
    let document = command.args.validate().unwrap(); let fixed = lifetime(&document);
    let exact = command.args.lifetime_limits(document, fixed, AnswerOptions::default()).unwrap();
    let d = &exact.document; let m = exact.memory.document;
    assert_eq!(exact.execution_bytes, m.input_bytes + m.native.kv_bytes + m.workspace_bytes
        + m.document_workspace_bytes + m.output_reservation_bytes + d.shared.compute.workspace_bytes);
    assert!(exact.minimum_memory_bytes >= d.shared.preparation_bytes + d.shared.weight_bytes + exact.execution_bytes);
    let mut too_small = command.args.validate().unwrap(); too_small.shared.memory_bytes = exact.minimum_memory_bytes - 1;
    assert_eq!(command.args.lifetime_limits(too_small, fixed, AnswerOptions::default()).err().unwrap().exit,
        ErrorCode::AdmissionOrResourceLimit);
}
