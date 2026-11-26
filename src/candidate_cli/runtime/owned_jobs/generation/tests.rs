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

#[test]
fn compact_cli_factories_preserve_the_exact_admitted_policy_caps_and_workspace() {
    for task in ["generate", "chat"] {
        let a = args(task, &["--input-encoding", "safe-bpe-v1",
            "--max-total-tokenizer-input-bytes", "100000", "--max-total-tokenizer-output-tokens", "5000",
            "--max-total-tokenizer-pair-checks", "1000000", "--max-total-tokenizer-pair-bytes", "10000000"]);
        let limits = a.validate().unwrap();
        let factory = prepare(&a, limits, lifetime(), &facts()).unwrap();
        assert_eq!(factory.document_encoding(), &a.common.input_encoding_policy(limits).unwrap());
        assert_eq!(factory.encoding_work_limits(), a.encoding_work_limits().unwrap());
        assert_eq!(factory.encoding_workspace_bytes(), a.encoding_workspace_bytes(limits).unwrap());
        let recipe = serde_json::to_value(factory.job_recipe()).unwrap();
        assert_eq!(recipe["input_encoding"], serde_json::to_value(factory.document_encoding()).unwrap());
        assert_eq!(recipe["encoding_work_limits"], serde_json::to_value(factory.encoding_work_limits()).unwrap());
        let prepared = factory.prepare_with_control(document(7, task == "chat"), &mut Continue).unwrap();
        assert_eq!(prepared.execution_identity().task_spec, format!("{task}-v1"));
        assert!(prepared.model_work().attention_pairs > 0);
    }
}

#[test]
fn compact_retained_text_admits_a_complete_transcript_beyond_the_byte_prompt_capacity() {
    for task in ["generate", "chat"] {
        let byte = args(task, &["--context-tokens", "512", "--max-new-tokens", "8"]);
        let compact = args(task, &["--context-tokens", "512", "--max-new-tokens", "8",
            "--input-encoding", "safe-bpe-v1"]);
        let before = prepare(&byte, byte.validate().unwrap(), lifetime(), &facts()).unwrap();
        let after = prepare(&compact, compact.validate().unwrap(), lifetime(), &facts()).unwrap();
        let input = || {
            let mut item = document(7, task == "chat");
            item.text = "hello world ".repeat(128); item
        };
        assert!(before.prepare_with_control(input(), &mut Continue).is_err());
        let prepared = after.prepare_with_control(input(), &mut Continue).unwrap();
        assert!(prepared.model_work().forward_positions <= 512);
        assert!(byte.common.options(166_101).unwrap() == compact.common.options(166_101).unwrap());
    }
}

#[test]
fn equal_workspace_never_substitutes_for_exact_encoding_or_invocation_authority() {
    let a = args("generate", &["--input-encoding", "safe-bpe-v1"]);
    let limits = a.validate().unwrap(); let factory = prepare(&a, limits, lifetime(), &facts()).unwrap();
    let policy = a.common.input_encoding_policy(limits).unwrap();
    let total = a.encoding_work_limits().unwrap(); let workspace = a.encoding_workspace_bytes(limits).unwrap();
    check_encoding_quote(&factory, &policy, total, workspace).unwrap();
    assert!(check_encoding_quote(&factory, &policy, total, workspace - 1).is_err());
    assert!(check_encoding_quote(&factory, &UntrustedEncoding::ByteFallback, total, workspace).is_err());
    assert!(check_encoding_quote(&factory, &policy, None, workspace).is_err());
    for axis in 0..6 {
        let mut changed_policy = policy.clone(); let mut changed_total = total.unwrap();
        let UntrustedEncoding::SafeBpeV1(call) = &mut changed_policy else { unreachable!() };
        match axis { 0 => call.max_pair_checks -= 1, 1 => call.max_pair_bytes -= 1,
            2 => changed_total.max_input_bytes -= 1, 3 => changed_total.max_output_tokens -= 1,
            4 => changed_total.max_pair_checks -= 1, _ => changed_total.max_pair_bytes -= 1 }
        assert_eq!(call.requirements().unwrap().workspace_bytes + 1024, workspace);
        assert!(check_encoding_quote(&factory, &changed_policy, Some(changed_total), workspace).is_err());
    }
}

#[test]
fn compact_resume_changes_physical_widths_without_reseeding_or_changing_native_work() {
    let seed = "07".repeat(32);
    for task in ["generate", "chat"] {
        for seeded in [false, true] {
            let mut flags = vec!["--input-encoding", "safe-bpe-v1"];
            if seeded { flags.extend_from_slice(&["--seed", seed.as_str()]); }
            let a = args(task, &flags); let first = prepare(&a, a.validate().unwrap(), lifetime(), &facts()).unwrap();
            let before = first.prepare_with_control(document(7, task == "chat"), &mut Continue).unwrap();
            for schedule in [vec!["--prefill-rows", "4"], vec!["--compute-threads", "4"],
                vec!["--compute-threads", "4", "--prefill-rows", "4"]] {
                let mut changed = flags.clone(); changed.extend_from_slice(&schedule);
                let b = args(task, &changed); let resumed = prepare(&b, b.validate().unwrap(), lifetime(), &facts()).unwrap();
                let mut earlier = document(0, task == "chat"); earlier.id = "other".into();
                resumed.prepare_with_control(earlier, &mut Continue).unwrap();
                let after = resumed.prepare_with_control(document(7, task == "chat"), &mut Continue).unwrap();
                assert_eq!(before.execution_identity(), after.execution_identity());
                assert_eq!(before.model_work(), after.model_work());
                assert_eq!(canonjson::canonical_bytes(first.job_recipe()).unwrap(),
                    canonjson::canonical_bytes(resumed.job_recipe()).unwrap());
                assert_eq!(first.encoding_work_limits(), resumed.encoding_work_limits());
                assert_eq!(first.encoding_workspace_bytes(), resumed.encoding_workspace_bytes());
            }
        }
    }
}

#[test]
fn compact_completion_metadata_reports_only_public_encoding_and_schedule_settings() {
    let report = serde_json::json!({"schema_version": 1, "output": {"committed": 2, "attempts": 3}});
    let byte = EncodingReport { response: &report, input_encoding: None,
        invocation_encoding_work_limits: None, root_object_reconciliation: None };
    assert_eq!(serde_json::to_value(ScheduledResponse { response: &byte,
        prefill_schedule: None, compute_schedule: None }).unwrap(), report);
    let a = args("chat", &["--input-encoding", "safe-bpe-v1", "--compute-threads", "4", "--prefill-rows", "4"]);
    let limits = a.validate().unwrap(); let policy = a.common.input_encoding_policy(limits).unwrap();
    let encoded = EncodingReport { response: &report, input_encoding: input_encoding::reported_policy(&policy),
        invocation_encoding_work_limits: a.encoding_work_limits().unwrap(), root_object_reconciliation: None };
    let value = serde_json::to_value(ScheduledResponse { response: &encoded,
        prefill_schedule: Some(PrefillSchedule::new(a.common.policy.prefill().unwrap().unwrap()).unwrap()),
        compute_schedule: ComputeSchedule::for_admission(limits.compute) }).unwrap();
    assert_eq!(value["input_encoding"], serde_json::to_value(&policy).unwrap());
    assert_eq!(value["invocation_encoding_work_limits"], serde_json::to_value(a.encoding_work_limits().unwrap()).unwrap());
    assert_eq!(value["output"], report["output"]);
    assert_eq!(value["compute_schedule"]["compute_threads"], 4);
    assert_eq!(value["prefill_schedule"]["max_batch_rows"], 4);
    for private in ["effective_seed", "history", "content", "token_ids", "token_logprobs", "prompt_digest"] {
        assert!(!value.to_string().contains(private));
    }
}
