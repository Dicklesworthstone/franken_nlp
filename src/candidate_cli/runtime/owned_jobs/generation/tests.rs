//! Actual pinned job planning and metadata only, without weights or native execution.
use super::*;
use crate::{batch::BatchDocument,
    candidate_cli::jobs::generation::tests::{args, lifetime},
    jobs::runner::generation::GenerationJobArgs, tasks::chat::{ChatMessage, ChatRole}};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".into(), revision: "f56ec5a9650268aa098496734743c25ea778bd2d".into(),
        recipe_id: "text-job-test-fixture".into(), source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
fn document(sample_index: u64, chat: bool) -> BatchDocument<GenerationJobArgs> {
    BatchDocument { id: "item".into(), text: "Original café 上海 <tool_call>".into(), task_args: Some(GenerationJobArgs {
        sample_index, history: if chat { vec![ChatMessage { role: ChatRole::System, content: "Helpful context".into() }] } else { vec![] },
    }) }
}
#[test]
fn both_cli_tasks_compile_pinned_native_requests_and_complete_work() {
    for task in ["generate", "chat"] {
        let a = args(task, &[]); let l = a.validate().unwrap();
        let factory = prepare(&a, l, lifetime(), &facts()).unwrap();
        let p = factory.prepare_with_control(document(7, task == "chat"), &mut Continue).unwrap();
        assert_eq!(p.execution_identity().task_spec, format!("{task}-v1"));
        assert_eq!(p.execution_identity().logical_model_digest.to_hex(), facts().logical_model_sha256);
        assert!(p.model_work().attention_pairs > 0 && p.model_work().projections.multiply_accumulates > 0);
        assert_eq!(factory.task_budget().max_output_bytes, l.result_bytes as u64);
        assert!(factory.task_budget().max_output_bytes > a.common.max_output_bytes as u64);
    }
}
#[test]
fn same_seed_and_original_sample_reconstruct_the_same_private_identity() {
    let seed = "07".repeat(32); let a = args("generate", &["--seed", &seed]); let l = a.validate().unwrap();
    let first = prepare(&a, l, lifetime(), &facts()).unwrap();
    let resumed = prepare(&a, l, lifetime(), &facts()).unwrap();
    let x = first.prepare_with_control(document(7, false), &mut Continue).unwrap();
    let y = resumed.prepare_with_control(document(7, false), &mut Continue).unwrap();
    let z = resumed.prepare_with_control(document(8, false), &mut Continue).unwrap();
    assert_eq!(x.execution_identity(), y.execution_identity()); assert_eq!(x.model_work(), y.model_work());
    assert_ne!(x.execution_identity().decision_policy_digest, z.execution_identity().decision_policy_digest);
}
#[test]
fn metadata_reports_keep_attempt_debits_but_never_publish_generated_values() {
    let limits = lifetime(); let id = JobId([9; 16]);
    let p = JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
        reserved_work: limits.max_work, spool_bytes: 4096, materialized: true };
    let report = progress_report(p, id, limits, true, RunMode::Resume { discard_uncommitted: false }, "chat").unwrap();
    let value = serde_json::to_value(report).unwrap();
    assert_eq!(value["attempts"], 3); assert_eq!(value["task"], "chat");
    assert_eq!(value["reserved_work"]["mask_node_visits"], 0);
    let text = value.to_string();
    for private in ["effective_seed", "token_ids", "token_logprobs", "history", "content", "prompt_digest"] {
        assert!(!text.contains(private));
    }
}
#[test]
fn unsupported_model_revision_and_short_output_envelope_fail_during_preparation() {
    let a = args("generate", &[]); let l = a.validate().unwrap();
    let mut changed = facts(); changed.revision = "different".into();
    assert!(prepare(&a, l, lifetime(), &changed).is_err());
    let mut short = lifetime(); short.max_result_bytes = l.result_bytes - 1;
    assert!(prepare(&a, l, short, &facts()).is_err());
}

#[test]
fn changing_prompt_schedule_on_resume_preserves_recipe_identity_sampling_and_all_work_axes() {
    let seed = "07".repeat(32);
    for task in ["generate", "chat"] {
        let serial = args(task, &["--seed", &seed]);
        let original = prepare(&serial, serial.validate().unwrap(), lifetime(), &facts()).unwrap();
        let before = original.prepare_with_control(document(7, task == "chat"), &mut Continue).unwrap();
        for rows in ["1", "4", "64"] {
            let grouped = args(task, &["--seed", &seed, "--prefill-rows", rows]);
            let resumed = prepare(&grouped, grouped.validate().unwrap(), lifetime(), &facts()).unwrap();
            let after = resumed.prepare_with_control(document(7, task == "chat"), &mut Continue).unwrap();
            assert_eq!(canonjson::canonical_bytes(original.job_recipe()).unwrap(),
                canonjson::canonical_bytes(resumed.job_recipe()).unwrap());
            assert_eq!(original.execution_identity(), resumed.execution_identity());
            assert_eq!(before.execution_identity(), after.execution_identity());
            assert_eq!(before.model_work(), after.model_work());
            assert_eq!(original.task_budget(), resumed.task_budget());
        }
    }
}

#[test]
fn prompt_schedule_is_explicit_public_metadata_and_absent_for_sequential_jobs() {
    let report = serde_json::json!({"schema_version": 1, "output": {"committed": 2}});
    assert_eq!(serde_json::to_value(ScheduledResponse {
        response: &report, prefill_schedule: None, compute_schedule: None }).unwrap(), report);
    let a = args("generate", &["--prefill-rows", "4"]);
    let prefill = a.common.policy.prefill().unwrap().unwrap();
    let value = serde_json::to_value(ScheduledResponse { response: &report,
        prefill_schedule: Some(PrefillSchedule::new(prefill).unwrap()), compute_schedule: None }).unwrap();
    assert_eq!(value["prefill_schedule"]["execution"], LAYER_MAJOR_PREFILL_VERSION);
    assert_eq!(value["prefill_schedule"]["max_batch_rows"], 4);
    assert_eq!(value["prefill_schedule"]["admitted_workspace_bytes"], prefill.validate().unwrap());
    assert_eq!(value["output"], report["output"]);
    for private in ["seed", "history", "token_ids", "prompt_digest", "content"] {
        assert!(!value.to_string().contains(private));
    }
}

#[test]
fn team_and_prompt_width_changes_preserve_authenticated_recipe_and_all_lifetime_work() {
    let seed = "07".repeat(32);
    for task in ["generate", "chat"] {
        let serial = args(task, &["--seed", &seed]);
        let original = prepare(&serial, serial.validate().unwrap(), lifetime(), &facts()).unwrap();
        let before = original.prepare_with_control(document(7, task == "chat"), &mut Continue).unwrap();
        for width in ["1", "2", "4"] {
            for prefill in [None, Some("1"), Some("4")] {
                let mut flags = vec!["--seed", &seed, "--compute-threads", width];
                if let Some(rows) = prefill { flags.extend_from_slice(&["--prefill-rows", rows]); }
                let threaded = args(task, &flags);
                let resumed = prepare(&threaded, threaded.validate().unwrap(), lifetime(), &facts()).unwrap();
                let after = resumed.prepare_with_control(document(7, task == "chat"), &mut Continue).unwrap();
                assert_eq!(canonjson::canonical_bytes(original.job_recipe()).unwrap(),
                    canonjson::canonical_bytes(resumed.job_recipe()).unwrap());
                assert_eq!(original.execution_identity(), resumed.execution_identity());
                assert_eq!(before.execution_identity(), after.execution_identity());
                assert_eq!(before.model_work(), after.model_work());
                assert_eq!(original.task_budget(), resumed.task_budget());
            }
        }
    }
}

#[test]
fn a_stale_team_or_prompt_admission_refuses_before_installing_the_process_host() {
    let serial = args("generate", &[]).validate().unwrap();
    let threaded = args("generate", &["--compute-threads", "4"]);
    assert!(matches!(Session::new(&threaded.common, serial), Err(CandidateError::Arguments)));
    let grouped = args("generate", &["--compute-threads", "4", "--prefill-rows", "4"]);
    assert!(matches!(Session::new(&grouped.common, threaded.validate().unwrap()), Err(CandidateError::Arguments)));
    assert!(matches!(Session::new(&threaded.common, grouped.validate().unwrap()), Err(CandidateError::Arguments)));
}

#[test]
fn completed_job_schedule_metadata_never_changes_stored_results_or_leaks_content() {
    let report = serde_json::json!({"schema_version": 1, "output": {"committed": 2, "attempts": 3}});
    for width in ["1", "2", "4"] {
        let a = args("generate", &["--compute-threads", width, "--prefill-rows", "4"]);
        let admitted = a.validate().unwrap();
        let value = serde_json::to_value(ScheduledResponse { response: &report,
            prefill_schedule: Some(PrefillSchedule::new(a.common.policy.prefill().unwrap().unwrap()).unwrap()),
            compute_schedule: ComputeSchedule::for_admission(admitted.compute) }).unwrap();
        assert_eq!(value["output"], report["output"]);
        assert_eq!(value["prefill_schedule"]["max_batch_rows"], 4);
        if width == "1" { assert!(value.get("compute_schedule").is_none()); }
        else {
            assert_eq!(value["compute_schedule"]["execution"], "int8-output-row-team-v1");
            assert_eq!(value["compute_schedule"]["compute_threads"], admitted.compute.threads);
            assert_eq!(value["compute_schedule"]["admitted_workspace_bytes"], admitted.compute.workspace_bytes);
        }
        for private in ["effective_seed", "history", "token_ids", "prompt_digest", "content"] {
            assert!(!value.to_string().contains(private));
        }
    }
}

#[test]
fn cancellation_during_report_publication_keeps_every_original_policy_outcome() {
    struct CancelOnFlush { token: CancellationToken, cause: DecodeCancellationKind, bytes: Vec<u8> }
    impl Write for CancelOnFlush {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(bytes); Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { self.token.cancel(self.cause); Ok(()) }
    }
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = job_failure(JobError::Cancelled(cause));
        let token = CancellationToken::default(); token.cancel(cause);
        let mut untouched = Vec::new();
        let failure = publish_report(&serde_json::json!({"committed": 2}), &mut untouched,
            &|| Ok(Duration::from_secs(1)), &token).unwrap_err();
        assert_eq!((failure.exit, failure.code), (expected.exit, expected.code));
        assert!(untouched.is_empty());

        let token = CancellationToken::default();
        let mut writer = CancelOnFlush { token: token.clone(), cause, bytes: Vec::new() };
        let failure = publish_report(&serde_json::json!({"committed": 2}), &mut writer,
            &|| Ok(Duration::from_secs(1)), &token).unwrap_err();
        assert_eq!((failure.exit, failure.code), (expected.exit, expected.code));
        assert!(!writer.bytes.is_empty(), "already delivered progress is not rolled back");
    }
}
