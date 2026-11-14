//! Pure CLI lifetime/admission contracts. These definitions do not load a model.
use super::*;
use crate::{candidate_cli::int4::{Int4Command, Int4JobCommand, Int4Task}, jobs::JobWork};

fn matches(mode: &str, extra: &[&str]) -> Result<clap::ArgMatches, clap::Error> {
    let mut values = vec!["source-int4", "answer-document-job", mode, "--source-dir", "/missing/source",
        "--source-manifest", "/missing/manifest", "--memory-mib", "131072", "--preparation-mib", "4096",
        "--store-results", "--job-dir", "/private/job", "--key-file", "/private/key",
        "--job-id", "11111111111111111111111111111111", "--limits", "/private/limits",
        "--max-input-bytes", "1048576", "--max-question-bytes", "1024"];
    values.extend_from_slice(extra);
    crate::candidate_cli::int4::definition().try_get_matches_from(values)
}
pub(super) fn parsed(mode: &str, extra: &[&str]) -> (JobCommand, Int4Args) {
    let command = Int4Command::from_matches(&matches(mode, extra).unwrap()).unwrap();
    assert_eq!(command.task, Int4Task::QuestionDocumentJob);
    match command.job {
        Some(Int4JobCommand::QuestionDocument(job)) => (job, command.args),
        _ => panic!("question document job routing"),
    }
}
pub(super) fn lifetime(command: &JobCommand, args: &Int4Args) -> JobLimits {
    let mapping = command.args.mapping.limits(args, &command.args.source, command.args.questions.cohort_chunks).unwrap();
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: args.max_input_bytes,
        max_snapshot_bytes: 8 * MIB, max_result_bytes: mapping.reduction.max_result_bytes,
        max_spool_bytes: 256 * MIB, max_materialized_bytes: 256 * MIB, max_journal_bytes: 8 * MIB,
        max_attempts: 4, max_work: JobWork::from_int4(mapping.max_model_work, mapping.max_mask_visits) }
}

#[test]
fn complete_question_modes_and_independent_schedules_keep_one_lifetime() {
    let cases: &[(&[&str], usize, Option<usize>, Option<usize>)] = &[
        (&[], 0, None, None),
        (&["--synthesize-answer"], 1, None, None),
        (&["--synthesize-answer", "--cohort-chunks", "3"], 1, Some(3), None),
        (&["--synthesize-answer", "--hierarchical-answer", "--answer-max-passes", "4"], 4, None, None),
        (&["--synthesize-answer", "--hierarchical-answer", "--answer-max-passes", "4",
            "--cohort-hierarchy-groups", "2"], 4, None, Some(2)),
        (&["--synthesize-answer", "--hierarchical-answer", "--answer-max-passes", "4",
            "--cohort-chunks", "3", "--cohort-hierarchy-groups", "2"], 4, Some(3), Some(2)),
    ];
    for &(extra, passes, chunks, groups) in cases {
        let (command, args) = parsed("start", extra);
        let job = lifetime(&command, &args);
        let common = command.args.validate(&args).unwrap();
        let limits = command.args.lifetime_limits(&args, common, AnswerOptions::default(), job).unwrap();
        assert_eq!(limits.config.continuation_passes(), passes);
        assert_eq!(command.args.questions.cohort_chunks, chunks);
        assert_eq!(command.args.questions.cohort_hierarchy_groups, groups);
        assert_eq!(limits.config.max_model_work, job.max_work.int4_model_work());
        assert_eq!(limits.config.max_mask_visits, job.max_work.mask_node_visits);
        assert_eq!(limits.config.budget.max_kv_bytes, common.native.kv_bytes);
        assert_eq!(limits.memory.document.max_output_tokens,
            if limits.config.hierarchy.is_some() { passes as u64 * u64::from(limits.config.budget.max_output_tokens) } else { 0 });
        assert!(limits.memory.document.native.kv_bytes >= limits.config.budget.max_kv_bytes);
    }
}

#[test]
fn question_job_worker_width_is_additive_and_preserves_source_question_and_lifetime_authority() {
    for flags in [&[][..], &["--synthesize-answer"][..],
        &["--synthesize-answer", "--hierarchical-answer", "--answer-max-passes", "4"][..]] {
        let (serial, args) = parsed("start", flags);
        let job = lifetime(&serial, &args);
        let common = serial.args.validate(&args).unwrap();
        let original = serial.args.lifetime_limits(&args, common, AnswerOptions::default(), job).unwrap();
        for operation in ["start", "resume"] {
            for threads in ["2", "4"] {
                let mut flags = flags.to_vec(); flags.extend(["--compute-threads", threads]);
                let (command, args) = parsed(operation, &flags);
                let common = command.args.validate(&args).unwrap();
                let limits = command.args.lifetime_limits(&args, common, AnswerOptions::default(), job).unwrap();
                assert_eq!(common.compute.threads, threads.parse::<usize>().unwrap());
                assert_eq!(limits.common.minimum_memory_bytes - original.common.minimum_memory_bytes,
                    common.compute.workspace_bytes);
                assert_eq!(limits.memory, original.memory);
                assert_eq!(limits.config.max_model_work, original.config.max_model_work);
                assert_eq!(limits.config.max_mask_visits, original.config.max_mask_visits);
                assert_eq!(limits.config.max_map_verification, original.config.max_map_verification);
                assert_eq!(limits.config.max_verification, original.config.max_verification);
                assert_eq!(limits.config.budget, original.config.budget);
                assert_eq!(limits.config.max_question_bytes, original.config.max_question_bytes);
                assert_eq!(limits.config.continuation_passes(), original.config.continuation_passes());
            }
        }
    }
}

#[test]
fn competing_question_job_schedules_and_invalid_worker_widths_refuse_before_private_io() {
    for flags in [&["--compute-threads", "0"][..], &["--compute-threads", "65"][..],
        &["--compute-threads", "2", "--cohort-chunks", "1"][..],
        &["--compute-threads", "2", "--synthesize-answer", "--hierarchical-answer",
            "--cohort-hierarchy-groups", "1"][..]] {
        let (command, args) = parsed("resume", flags);
        assert!(command.args.validate(&args).is_err());
    }
}

#[test]
fn changing_attempt_ceiling_scales_verification_without_multiplying_native_authority() {
    let (command, args) = parsed("start", &["--synthesize-answer"]);
    let common = command.args.validate(&args).unwrap();
    let mut job = lifetime(&command, &args); job.max_attempts = 2;
    let two = command.args.lifetime_limits(&args, common, AnswerOptions::default(), job).unwrap();
    job.max_attempts = 5;
    let five = command.args.lifetime_limits(&args, common, AnswerOptions::default(), job).unwrap();
    assert_eq!(two.config.max_model_work, five.config.max_model_work);
    assert_eq!(two.config.max_mask_visits, five.config.max_mask_visits);
    assert_eq!(two.memory, five.memory);
    assert_eq!(two.config.max_verification.max_fields, two.config.verification.max_fields * 2);
    assert_eq!(five.config.max_verification.max_fields, five.config.verification.max_fields * 5);
    assert_eq!(five.config.max_map_verification.max_matches, five.config.mapping.verification.max_matches * 5);
    assert_eq!(five.config.max_verification.max_scan_steps, five.config.verification.max_scan_steps * 5);
}

#[test]
fn encoded_record_and_result_caps_cannot_exceed_the_pre_io_declaration() {
    let (command, args) = parsed("start", &[]);
    let common = command.args.validate(&args).unwrap();
    for fault in 0..5 {
        let mut job = lifetime(&command, &args);
        match fault {
            0 => job.max_input_bytes_per_item = args.max_input_bytes + 1,
            1 => job.max_input_bytes_per_item = command.args.questions.max_question_bytes - 1,
            2 => job.max_result_bytes -= 1,
            3 => job.max_work.mask_node_visits = 1,
            _ => job.max_work.model.projections.multiply_accumulates = 1,
        }
        assert!(command.args.lifetime_limits(&args, common, AnswerOptions::default(), job).is_err(), "fault {fault}");
    }
    let mut job = lifetime(&command, &args); job.max_input_bytes_per_item -= 100;
    let narrowed = command.args.lifetime_limits(&args, common, AnswerOptions::default(), job).unwrap();
    assert_eq!(narrowed.config.mapping.chunks.max_input_bytes, job.max_input_bytes_per_item);
    assert_eq!(narrowed.config.max_question_bytes, 1024);
}

#[test]
fn verification_products_refuse_overflow_and_zero_attempts() {
    let per = GroundingBudget { max_fields: 2, max_matches: 3, max_scan_steps: 7 };
    assert!(verification_lifetime(per, 0).is_err());
    assert!(verification_lifetime(per, u64::MAX).is_err());
    let fields = GroundingBudget { max_fields: usize::MAX, ..per };
    assert!(verification_lifetime(fields, 2).is_err());
    let matches = GroundingBudget { max_matches: usize::MAX, ..per };
    assert!(verification_lifetime(matches, 2).is_err());
    let scans = GroundingBudget { max_scan_steps: u64::MAX, ..per };
    assert!(verification_lifetime(scans, 2).is_err());
}

#[test]
fn unsupported_projection_prefill_and_unretained_storage_refuse_before_io() {
    for extra in [&["--max-selected-rows", "64"][..], &["--prefill-rows", "2"][..]] {
        if let Ok(matches) = matches("start", extra) {
            if let Ok(command) = Int4Command::from_matches(&matches) {
                match command.job {
                    Some(Int4JobCommand::QuestionDocument(job)) => assert!(job.args.validate(&command.args).is_err()),
                    _ => panic!("unexpected accepted source task"),
                }
            }
        }
    }
    let (mut command, args) = parsed("start", &[]);
    command.args.storage.store_results = false;
    assert!(command.args.validate(&args).is_err());
    command.args.storage.store_results = true;
    command.args.questions.cohort_hierarchy_groups = Some(2);
    assert!(command.args.validate(&args).is_err());
    let (command, args) = parsed("resume", &["--discard-uncommitted-tail"]);
    assert!(matches!(command.mode, RunMode::Resume { discard_uncommitted: true }));
    assert!(command.args.validate(&args).is_ok());
}

#[test]
fn impossible_population_ownership_rejects_before_model_loading() {
    let (command, args) = parsed("start", &[]);
    let common = command.args.validate(&args).unwrap();
    let mut job = lifetime(&command, &args);
    job.max_snapshot_bytes = i64::MAX as u64;
    assert!(command.args.lifetime_limits(&args, common, AnswerOptions::default(), job).is_err());
}

#[test]
fn retained_record_replay_reduces_only_population_memory_and_preserves_question_policy() {
    let (ordinary, args) = parsed("start", &[]);
    let job = lifetime(&ordinary, &args);
    let common = ordinary.args.validate(&args).unwrap();
    let baseline = ordinary.args.lifetime_limits(&args, common, AnswerOptions::default(), job).unwrap();
    for (mode, extra) in [
        ("start", &["--spool-input", "--max-retained-input-mib", "8"][..]),
        ("resume", &["--use-retained-input", "--max-retained-input-mib", "8"][..]),
    ] {
        let (command, args) = parsed(mode, extra);
        let retention = command.args.storage.input_retention;
        retention.validate(command.mode, &args.input).unwrap();
        assert_eq!(retention.limits().unwrap().unwrap().max_spool_bytes, 8 * MIB);
        let common = command.args.validate(&args).unwrap();
        let declared = command.args.lifetime_limits(&args, common, AnswerOptions::default(), job).unwrap();
        let saved = if retention.use_retained_input { 2 * job.max_snapshot_bytes } else { 0 };
        let mut expected = baseline.memory;
        expected.population_buffer_bytes -= saved;
        expected.document.input_bytes -= saved;
        assert_eq!(declared.memory, expected);
        assert_eq!(declared.config.max_model_work, baseline.config.max_model_work);
        assert_eq!(declared.config.max_verification.max_scan_steps, baseline.config.max_verification.max_scan_steps);
    }
}

struct NeverRead;
impl Read for NeverRead {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> { panic!("private input must not be read") }
}
#[test]
fn retained_mode_mismatches_and_overflow_refuse_before_private_io() {
    for (mode, extra) in [
        ("start", &["--use-retained-input", "--max-retained-input-mib", "8"][..]),
        ("resume", &["--spool-input", "--max-retained-input-mib", "8"][..]),
        ("start", &["--max-retained-input-mib", "8"][..]),
        ("start", &["--spool-input", "--max-retained-input-mib", "18446744073709551615"][..]),
    ] {
        let (command, args) = parsed(mode, extra);
        let failure = command.execute_owned(args, NeverRead, &mut Vec::new()).unwrap_err();
        assert_eq!(failure.exit, ErrorCode::Usage);
        assert_eq!(failure.code, "job_input_retention_options");
    }
}

#[test]
fn retained_question_resume_selects_no_external_population_or_fallback() {
    let (command, args) = parsed("resume", &["--use-retained-input", "--max-retained-input-mib", "8"]);
    let mut reader = job_cli::population_reader(command.mode, command.args.storage.input_retention,
        &args.input, NeverRead, IO_BUFFER_BYTES).unwrap();
    assert_eq!(reader.read(&mut [0_u8; 1]).unwrap(), 0);
    assert!(job_cli::population_reader(command.mode, command.args.storage.input_retention,
        std::path::Path::new("/missing/original.ndjson"), NeverRead, IO_BUFFER_BYTES).is_err());
    assert!(matches("resume", &["original.ndjson", "--use-retained-input",
        "--max-retained-input-mib", "8"]).is_err());
}

#[test]
fn compact_question_jobs_keep_per_attempt_policy_separate_from_durable_lifetime() {
    for mode in ["start", "resume"] {
        for selection in [&[][..], &["--cohort-chunks", "2"][..],
            &["--synthesize-answer"][..], &["--synthesize-answer", "--cohort-chunks", "2"][..],
            &["--synthesize-answer", "--hierarchical-answer"][..],
            &["--synthesize-answer", "--hierarchical-answer", "--cohort-hierarchy-groups", "2"][..],
            &["--synthesize-answer", "--hierarchical-answer", "--cohort-chunks", "2", "--cohort-hierarchy-groups", "2"][..],
            &["--synthesize-answer", "--hierarchical-answer", "--compute-threads", "3"][..]] {
            let mut flags = selection.to_vec();
            flags.extend_from_slice(&["--input-encoding", "safe-bpe-v1",
                "--max-total-tokenizer-input-bytes", "123456",
                "--max-total-tokenizer-pair-checks", "234567"]);
            let (command, args) = parsed(mode, &flags);
            let common = command.args.validate(&args).unwrap();
            let job = lifetime(&command, &args);
            let admitted = command.args.lifetime_limits(&args, common, AnswerOptions::default(), job).unwrap();
            let encoding = admitted.encoding.unwrap();
            assert_eq!(encoding.complete.max_input_bytes, 123456);
            assert_eq!(encoding.complete.max_pair_checks, 234567);
            assert_eq!(encoding.call.max_input_bytes, args.max_input_bytes);
            assert_eq!(admitted.config.synthesis.is_some(), selection.contains(&"--synthesize-answer"));
            assert_eq!(admitted.config.hierarchy.is_some(), selection.contains(&"--hierarchical-answer"));
            assert!(encoding.workspace_bytes().unwrap() < common.preparation_bytes);
            assert_eq!(admitted.config.max_model_work, job.max_work.int4_model_work());
            assert_eq!(admitted.config.max_mask_visits, job.max_work.mask_node_visits);
            assert_eq!(admitted.config.max_verification.max_fields,
                admitted.config.verification.max_fields * job.max_attempts as usize);
            let mut insufficient = args.clone(); insufficient.preparation_mib = 1;
            assert!(command.args.validate(&insufficient).is_err());
        }
    }
}

#[test]
fn compact_question_jobs_refuse_unbounded_encoding_options_in_every_mode() {
    for operation in ["start", "resume"] {
        for mode in [&[][..], &["--synthesize-answer"][..],
            &["--synthesize-answer", "--hierarchical-answer"][..]] {
            for invalid in [
                &["--input-encoding", "safe-bpe-v1", "--max-total-tokenizer-input-bytes", "0"][..],
                &["--max-total-tokenizer-output-tokens", "1"][..],
            ] {
                let mut flags = mode.to_vec(); flags.extend_from_slice(invalid);
                let (command, args) = parsed(operation, &flags);
                assert!(command.args.validate(&args).is_err());
            }
        }
    }
}
