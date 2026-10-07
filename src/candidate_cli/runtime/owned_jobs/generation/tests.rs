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
