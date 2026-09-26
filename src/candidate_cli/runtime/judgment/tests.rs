//! Real pinned judge planning, not execution or model-quality evidence.
use super::*;
use crate::{candidate_cli::judge::tests::{command as cli_command, fixture},
    native_engine::decode::DecodeCancellationKind};
use serde_json::json;

struct Control { calls: usize, stop_at: usize }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        (self.calls >= self.stop_at).then_some(DecodeCancellationKind::Deadline)
    }
}
fn go() -> Control { Control { calls: 0, stop_at: usize::MAX } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "metadata-only-judge-fixture".to_owned(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
fn request(value: serde_json::Value, args: &ScoredArgs) -> (JudgeRequest, TaskBudget) {
    let (_, limits) = args.common().unwrap(); let budget = args.budget(limits);
    (command::request(&value.to_string(), budget, args.max_input_bytes).unwrap(), budget)
}
#[test]
fn all_three_modes_compile_complete_native_heads_and_actual_model_facts() {
    let cmd = cli_command(&[]);
    for (mode, heads) in [("pairwise",2), ("rubric",2), ("faithfulness",1)] {
        let (r, b) = request(fixture(mode), &cmd.args);
        let p = prepare(&r, &facts(), &cmd.args, b, &mut go()).unwrap();
        assert_eq!(p.head_count(), heads);
        assert_eq!(p.task_budget(), b);
        assert!(p.required_context() <= cmd.args.context_tokens);
        let id = p.execution_identity();
        assert_eq!(id.task_spec, "judge-v1");
        assert_eq!(id.logical_model_digest.to_hex(), facts().logical_model_sha256);
        assert_eq!(id.quant_recipe, facts().recipe_id);
        assert_eq!(id.numerics_profile, NumericsProfile::StrictQuantized { version: 1 });
        assert_ne!(id.prompt_digest, Sha256Digest::of_bytes(b"null"));
        assert_ne!(id.template_digest, Sha256Digest::of_bytes(b"null"));
        p.verify_identity(id).unwrap();
        let mut changed = id.clone(); changed.prompt_digest = Sha256Digest::of_bytes(b"other");
        assert!(p.verify_identity(&changed).is_err());
    }
}
#[test]
fn every_faithfulness_window_is_included_in_addition_to_the_full_source() {
    let cmd = cli_command(&[]); let mut v = fixture("faithfulness");
    v["source"] = json!("abcdefgh"); v["claim"] = json!("abc");
    v["policy"]["evidence_window_bytes"] = json!(4);
    let (r, b) = request(v, &cmd.args);
    let plan = prepare(&r, &facts(), &cmd.args, b, &mut go()).unwrap();
    assert_eq!(plan.head_count(), 3); // Whole source plus both disjoint windows.
}
#[test]
fn both_orders_reuse_context_without_discarding_their_summed_work() {
    let cmd = cli_command(&[]); let mut v = fixture("pairwise");
    v["a"] = json!("a".repeat(600)); v["b"] = json!("b".repeat(600));
    let (r, b) = request(v, &cmd.args);
    let p = prepare(&r, &facts(), &cmd.args, b, &mut go()).unwrap();
    assert_eq!(p.head_count(), 2);
    assert!(p.required_context() <= cmd.args.context_tokens);
    assert!(p.planned_work().forward_positions > cmd.args.context_tokens as u64);
}
#[test]
fn every_work_axis_and_largest_context_are_checked_before_a_model_load() {
    let mut cmd = cli_command(&[]); let (r, b) = request(fixture("pairwise"), &cmd.args);
    let p = prepare(&r, &facts(), &cmd.args, b, &mut go()).unwrap();
    let w = p.planned_work();
    for axis in 0..5 {
        let mut args = cli_command(&[]).args;
        match axis { 0 => args.max_forward_positions = w.forward_positions-1,
            1 => args.max_projected_logits = w.projected_logits-1,
            2 => args.max_attention_pairs = w.attention_pairs-1,
            3 => args.max_dot_products = w.projections.dot_products-1,
            _ => args.max_multiply_accumulates = w.projections.multiply_accumulates-1 }
        assert!(prepare(&r, &facts(), &args, b, &mut go()).is_err());
    }
    cmd.args.context_tokens = p.required_context()-1;
    assert!(prepare(&r, &facts(), &cmd.args, b, &mut go()).is_err());
}
#[test]
fn late_planning_cancellation_and_foreign_model_do_not_return_a_plan() {
    let cmd = cli_command(&[]); let (r, b) = request(fixture("pairwise"), &cmd.args);
    let mut control = go(); prepare(&r, &facts(), &cmd.args, b, &mut control).unwrap();
    for stop_at in [1,2,control.calls] {
        assert!(prepare(&r, &facts(), &cmd.args, b, &mut Control { calls:0, stop_at }).is_err());
    }
    let mut changed = facts(); changed.model_id = "another model".to_owned();
    assert!(prepare(&r, &changed, &cmd.args, b, &mut go()).is_err());
}
#[test]
fn changed_criterion_or_policy_changes_the_sealed_execution_identity() {
    let cmd = cli_command(&[]); let (r, b) = request(fixture("pairwise"), &cmd.args);
    let base = prepare(&r, &facts(), &cmd.args, b, &mut go()).unwrap();
    let mut v = fixture("pairwise"); v["criterion"] = json!("Prefer concise text. <think> é");
    let (r, _) = request(v, &cmd.args); let p = prepare(&r, &facts(), &cmd.args, b, &mut go()).unwrap();
    assert_ne!(p.execution_identity().prompt_digest, base.execution_identity().prompt_digest);
    let mut v = fixture("pairwise"); v["policy"]["minimum_margin_milli"] = json!(100);
    let (r, _) = request(v, &cmd.args); let p = prepare(&r, &facts(), &cmd.args, b, &mut go()).unwrap();
    assert_ne!(p.execution_identity().decision_policy_digest, base.execution_identity().decision_policy_digest);
}
