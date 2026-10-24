//! Metadata, key and resource fixtures; native execution is an explicit ignored gate.
use super::*;
use super::super::tests::{command, lifetime, NoIo};
use std::io::Cursor;
use crate::{
    batch::BatchFault,
    jobs::{JobId, JobProgress},
    tasks::redact::{grouped_int4::Int4RedactionError, pseudonym::PseudonymEncoding},
};

fn completed(id: JobId, limits: JobLimits) -> JobProgress {
    JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
        reserved_work: limits.max_work, spool_bytes: 1024, materialized: true }
}

#[test]
fn owned_execution_revalidates_options_before_runtime_keys_or_population_io() {
    let (mut job, args) = command("resume", &[]);
    let admitted = job.args.validate(&args).unwrap();
    job.args.storage.max_input_lines = 0;
    assert_eq!(execute(job.mode, args, job.args, admitted, NoIo, &mut NoIo).unwrap_err().code,
        "explicit_storage_or_job_options");
    let (job, mut args) = command("start", &[]);
    let admitted = job.args.validate(&args).unwrap(); args.seed = Some("ab".repeat(32));
    assert_eq!(execute(job.mode, args, job.args, admitted, NoIo, &mut NoIo).unwrap_err().code,
        "invalid_job_options");
}

#[test]
fn hosted_buffer_quote_matches_complete_population_and_private_recipe_capacity_exactly() {
    let (job, args) = command("start", &[]);
    let document = job.args.validate(&args).unwrap();
    let stored = lifetime(&args, document);
    let limits = job.args.lifetime_limits(&args, document, stored).unwrap();
    let host = host_limits(&args, limits, Duration::from_secs(19));
    check_host_quote(host, stored, limits.buffer_bytes).unwrap();
    assert_eq!(check_host_quote(host, stored, limits.buffer_bytes - 1).unwrap_err().code, "job_host_memory_quote");
    assert_eq!(RECIPE_RESERVE_BYTES, INT4_REDACTION_JOB_RECIPE_BYTES as u64 * 4);
    assert_eq!(host.native.context_tokens, args.context_tokens);
    assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
    assert_eq!(host.native.run.max_elapsed, Duration::from_secs(19));
    assert_eq!(host.preparation_reserve_bytes, document.common.preparation_bytes);
    let mut invalid = host; invalid.serialization_reserve_bytes = 0;
    assert_eq!(check_host_quote(invalid, stored, limits.buffer_bytes).unwrap_err().code, "job_host_admission");
}

#[test]
fn pseudonym_commitment_checks_before_population_io_and_the_job_key_is_never_a_substitute() {
    let (job, args) = command("start", &["population.ndjson", "--action", "pseudonymize",
        "--key-stdin", "--key-id", "rotation", "--namespace", "private-scope"]);
    let document = job.args.validate(&args).unwrap();
    let request = job.args.document.request(&args, document);
    let secret = key_scope(&job.args, &mut Cursor::new([0x80_u8; 32]), &request).unwrap().unwrap();
    let expected = secret.key.commitment();
    let context = Pseudonyms::full256(&secret.key, &secret.namespace, Some(&expected)).unwrap();
    assert_eq!(context.identity().encoding, PseudonymEncoding::Full256);
    let pseudonym = context.pseudonym(crate::tasks::redact::PiiKind::Email, "a@example.org").unwrap();
    assert_eq!(pseudonym, context.pseudonym(crate::tasks::redact::PiiKind::Email, "a@example.org").unwrap());
    let (pinned, args) = command("resume", &["population.ndjson", "--action", "pseudonymize",
        "--key-stdin", "--key-id", "rotation", "--namespace", "private-scope", "--expected-key-commitment", &expected]);
    let request = pinned.args.document.request(&args, pinned.args.validate(&args).unwrap());
    assert!(matches!(key_scope(&pinned.args, &mut Cursor::new([1_u8; 32]), &request), Err(CandidateError::Identity)));
    assert!(key_scope(&pinned.args, &mut Cursor::new([0x80_u8; 32]), &request).is_ok());
    assert!(matches!(key_scope(&pinned.args, &mut Cursor::new(Vec::<u8>::new()), &request), Err(CandidateError::Input)));
    let (plain, args) = command("start", &[]);
    let request = plain.args.document.request(&args, plain.args.validate(&args).unwrap());
    assert!(key_scope(&plain.args, &mut NoIo, &request).unwrap().is_none());
}

#[test]
fn all_cancellation_classes_survive_storage_processor_native_and_physical_scope_wrappers() {
    for (cause, exit) in [
        (DecodeCancellationKind::Timeout, ErrorCode::BudgetOrTimeout),
        (DecodeCancellationKind::Deadline, ErrorCode::BudgetOrTimeout),
        (DecodeCancellationKind::PollQuota, ErrorCode::BudgetOrTimeout),
        (DecodeCancellationKind::CostBudget, ErrorCode::BudgetOrTimeout),
        (DecodeCancellationKind::User, ErrorCode::Cancelled),
        (DecodeCancellationKind::Shutdown, ErrorCode::Cancelled),
        (DecodeCancellationKind::ParentCancelled, ErrorCode::Cancelled),
        (DecodeCancellationKind::ResourceUnavailable, ErrorCode::AdmissionOrResourceLimit),
        (DecodeCancellationKind::FailFast, ErrorCode::Generic),
        (DecodeCancellationKind::LinkedExit, ErrorCode::Generic),
        (DecodeCancellationKind::RaceLost, ErrorCode::Generic),
    ] {
        let storage = execution_failure(HostedJobError::Job(JobRunError::Storage(JobError::Cancelled(cause))));
        let processor = execution_failure(HostedJobError::Job(JobRunError::Processor(BatchFault::cancelled(cause))));
        let native = execution_failure(HostedJobError::Host(HostedError::Int4(
            Int4HostError::Redaction(Int4RedactionError::Cancelled(cause)))));
        let nested = execution_failure(HostedJobError::Host(HostedError::Scope {
            source: asupersync::cx::ScopedCpuError::WorkerCapExceeded { cap: 1 },
            physical_error: Some(Box::new(HostedError::Int4(
                Int4HostError::Redaction(Int4RedactionError::Cancelled(cause))))),
        }));
        for actual in [storage, processor, native, nested] {
            assert_eq!(actual.exit, exit); assert_eq!(actual.code, cancelled(cause).code);
        }
    }
    for (cause, expected) in [
        (DecodeCancellationKind::FailFast, ErrorCode::AdmissionOrResourceLimit),
        (DecodeCancellationKind::LinkedExit, ErrorCode::AdmissionOrResourceLimit),
        (DecodeCancellationKind::Deadline, ErrorCode::BudgetOrTimeout),
        (DecodeCancellationKind::User, ErrorCode::Cancelled),
    ] {
        let error = HostedJobError::Host(HostedError::Stopped {
            stop: crate::hosted::RunStop { kind: cause, runtime_reason: None, runtime_checkpoint_error: None },
            task_error: Some(Box::new(HostedError::Limits("fixture underlying admission"))),
        });
        assert_eq!(execution_failure(error).exit, expected);
    }
}

#[test]
fn authenticated_mismatch_diagnostics_name_only_the_changed_field_and_preserve_durable_failures() {
    for (field, code) in [(MismatchField::Job, "job_mismatch_id"),
        (MismatchField::Secret, "job_mismatch_secret"), (MismatchField::Execution, "job_mismatch_execution"),
        (MismatchField::Recipe, "job_mismatch_recipe"), (MismatchField::Population, "job_mismatch_population"),
        (MismatchField::Limits, "job_mismatch_limits")] {
        let failure = execution_failure(HostedJobError::Job(JobRunError::Storage(JobError::Mismatch(field))));
        assert_eq!(failure.exit, ErrorCode::ArtifactIntegrityOrFormatOrVersion); assert_eq!(failure.code, code);
    }
    for (error, exit, code) in [
        (JobError::WorkLimit, ErrorCode::BudgetOrTimeout, "job_lifetime_work_or_attempts"),
        (JobError::Authentication, ErrorCode::ArtifactIntegrityOrFormatOrVersion, "job_integrity"),
        (JobError::UncommittedTail, ErrorCode::StructuredTaskNoResult, "job_uncommitted_tail_or_stage"),
        (JobError::UnsafeStorage, ErrorCode::AdmissionOrResourceLimit, "unsafe_job_storage"),
    ] {
        let failure = execution_failure(HostedJobError::Job(JobRunError::Storage(error)));
        assert_eq!(failure.exit, exit); assert_eq!(failure.code, code);
    }
    let (mut job, args) = command("resume", &[]); job.args.storage.store_results = false;
    let mut diagnostics = Vec::new();
    assert_eq!(job.run_owned(args, NoIo, &mut NoIo, &mut diagnostics), ErrorCode::Usage.as_process_exit());
    let text = std::str::from_utf8(&diagnostics).unwrap();
    let event: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(event["durable_progress_may_exist"], true); assert_eq!(event["task"], "redact");
    assert_eq!(event["operation"], "resume"); assert_eq!(event["redaction_mode"], "complete_document");
    assert!(!text.contains("private-")); assert!(!text.contains("0123456789abcdef"));
}

#[test]
fn only_complete_bounded_authenticated_progress_can_form_a_success_report() {
    let (job, args) = command("start", &[]);
    let document = job.args.validate(&args).unwrap();
    let stored = lifetime(&args, document);
    let id = JobId([0x12; 16]);
    for case in 0..13 {
        let mut progress = completed(id, stored);
        match case {
            0 => progress.job_id = JobId([0x13; 16]),
            1 => progress.items = 0,
            2 => progress.committed -= 1,
            3 => progress.attempts = 1,
            4 => progress.attempts = stored.max_attempts + 1,
            5 => progress.reserved_work.model.forward_positions += 1,
            6 => progress.reserved_work.model.projected_logits += 1,
            7 => progress.reserved_work.model.attention_pairs += 1,
            8 => progress.reserved_work.model.projections.dot_products += 1,
            9 => progress.reserved_work.model.projections.multiply_accumulates += 1,
            10 => progress.reserved_work.mask_node_visits += 1,
            11 => progress.spool_bytes = stored.max_spool_bytes + 1,
            _ => progress.materialized = false,
        }
        assert!(matches!(progress_report(progress, id, stored, true, RunMode::Start, "redact"),
            Err(Failure { code: "job_completion_contract", .. })));
    }
}

#[test]
fn metadata_only_report_has_public_provenance_and_keeps_sensitive_results_in_the_owned_spool() {
    let (job, args) = command("resume", &[]);
    let document = job.args.validate(&args).unwrap();
    let stored = lifetime(&args, document); let id = JobId([0x12; 16]);
    let report = progress_report(completed(id, stored), id, stored, true,
        RunMode::Resume { discard_uncommitted: false }, "redact").unwrap();
    let response = Response {
        protocol: "fnlp-candidate-int4-redact-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "redact",
        redaction_mode: "complete_document", retained_output_is_sensitive: true,
        model_id: "Nanbeige4.2-3B", source_revision: "fixture-public-revision",
        source_root_sha256: "fixture-public-source", model_content_sha256: "fixture-public-model",
        candidate_format: "fixture-format", quant_recipe: "fixture-grouped-int4", group_width: 128,
        integer_kernel: "s_scalar_i32", snapshot: Some(SnapshotRecord::new("loaded", document.common.snapshot_need)),
        output: &report,
    };
    let mut bytes = Vec::new(); publish(&response, job_cli::REPORT_BYTES, &mut bytes).unwrap();
    let event: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(event["protocol"], "fnlp-candidate-int4-redact-job-v1");
    assert_eq!(event["retained_output_is_sensitive"], true);
    assert_eq!(event["snapshot"]["format"], INT4_SNAPSHOT_VERSION);
    assert_eq!(event["output"]["committed"], 2); assert_eq!(event["output"]["attempts"], 3);
    assert_eq!(event["output"]["reserved_work"], serde_json::to_value(stored.max_work).unwrap());
    for private in ["text", "map", "results", "source", "document", "prompt", "namespace", "key",
        "job_dir", "key_file", "limits_file", "snapshot_path", "ner_options", "task_args"] {
        assert!(event.get(private).is_none()); assert!(event["output"].get(private).is_none());
    }
    assert!(bytes.ends_with(b"\n"));
    assert_eq!(publish(&response, 1, &mut NoIo), Err(CandidateError::Output));
}

#[test]
fn report_delivery_or_flush_failure_cannot_erase_or_retry_durable_completion() {
    struct Broken { bytes: Vec<u8>, fail_write: bool }
    impl Write for Broken {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.fail_write { return Err(io::Error::other("closed report consumer")); }
            self.bytes.extend_from_slice(bytes); Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("report flush failed")); }
    }
    let (job, args) = command("start", &[]);
    let stored = lifetime(&args, job.args.validate(&args).unwrap()); let id = JobId([0x12; 16]);
    let report = progress_report(completed(id, stored), id, stored, false, RunMode::Start, "redact").unwrap();
    let before = serde_json::to_value(&report).unwrap();
    for fail_write in [false, true] {
        let mut sink = Broken { bytes: Vec::new(), fail_write };
        let failure = Failure::from(publish(&report, job_cli::REPORT_BYTES, &mut sink).unwrap_err());
        assert_eq!(failure.code, "report_delivery");
        assert_eq!(serde_json::to_value(&report).unwrap(), before);
        if !fail_write { assert_eq!(sink.bytes.iter().filter(|&&b| b == b'\n').count(), 1); }
    }
    assert_eq!(before["committed"], 2); assert_eq!(before["status"], "complete");
}

#[test]
#[ignore = "requires controller-admitted pinned sources, independent protected job key/limits, a fresh job directory and snapshot"]
fn source_to_snapshot_resume_reuses_commits_and_materialization_without_new_attempts() {
    fn configured(operation: &str, extra: &[&str]) -> (JobCommand, Int4Args) {
        let (mut job, mut args) = command(operation, extra);
        args.source_dir = Some(PathBuf::from(std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("pinned sources required")));
        args.source_manifest = Some(PathBuf::from(std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("pinned manifest required")));
        args.memory_mib = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("memory ceiling required").parse().unwrap();
        job.args.storage.job_dir = PathBuf::from(std::env::var("FNLP_INT4_TEST_REDACT_JOB_DIR").expect("fresh protected job directory required"));
        job.args.storage.key_file = PathBuf::from(std::env::var("FNLP_INT4_TEST_JOB_KEY_FILE").expect("independent protected job key required"));
        job.args.storage.limits_file = PathBuf::from(std::env::var("FNLP_INT4_TEST_REDACT_JOB_LIMITS").expect("immutable redaction job limits required"));
        job.args.storage.job_id = job_cli::parse_job_id(&std::env::var("FNLP_INT4_TEST_REDACT_JOB_ID").expect("fresh random job id required")).unwrap();
        (job, args)
    }
    let snapshot = std::env::var("FNLP_INT4_TEST_REDACT_JOB_SNAPSHOT").expect("fresh snapshot path required");
    assert!(!std::path::Path::new(&snapshot).exists(), "fixture deletes nothing");
    let input = b"{\"id\":\"a\",\"text\":\"Contact a@example.org.\"}\n{\"id\":\"b\",\"text\":\"Email b@example.org.\"}\n";
    let flags = ["--rules", "email", "--materialize", "--max-ner-chunk-bytes", "16"];
    let (job, mut args) = configured("start", &flags);
    let materialized = job.args.storage.job_dir.join("materialized.ndjson");
    assert!(!materialized.exists(), "controller provides a fresh directory");
    args.save_snapshot = Some(PathBuf::from(&snapshot));
    let mut first = Vec::new(); job.execute_owned(args, Cursor::new(input.to_vec()), &mut first).unwrap();
    let first: serde_json::Value = serde_json::from_slice(&first).unwrap();
    assert_eq!(first["output"]["committed"], 2); assert_eq!(first["output"]["materialized"], true);
    let committed = std::fs::read(&materialized).unwrap();
    assert!(!std::str::from_utf8(&committed).unwrap().contains("@example.org"));
    let pin = first["model_content_sha256"].as_str().unwrap().to_owned();
    let resume = |extra: &[&str]| {
        let (job, mut args) = configured("resume", extra);
        args.source_dir = None; args.source_manifest = None;
        args.snapshot = Some(PathBuf::from(&snapshot)); args.expected_model_sha256 = Some(pin.clone());
        (job, args)
    };
    let (job, args) = resume(&flags);
    let mut second = Vec::new(); job.execute_owned(args, Cursor::new(input.to_vec()), &mut second).unwrap();
    let second: serde_json::Value = serde_json::from_slice(&second).unwrap();
    assert_eq!(second["snapshot"]["operation"], "loaded");
    assert_eq!(second["output"]["attempts"], first["output"]["attempts"]);
    assert_eq!(second["output"]["reserved_work"], first["output"]["reserved_work"]);
    assert_eq!(std::fs::read(&materialized).unwrap(), committed);
    let (mut changed, args) = resume(&flags);
    changed.args.document.redaction.flags.no_verify = true;
    let mut output = Vec::new();
    assert_eq!(changed.execute_owned(args, Cursor::new(input.to_vec()), &mut output).unwrap_err().code, "job_mismatch_recipe");
    assert!(output.is_empty()); assert_eq!(std::fs::read(&materialized).unwrap(), committed);
}
