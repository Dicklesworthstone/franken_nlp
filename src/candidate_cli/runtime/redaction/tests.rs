//! Actual pinned source preparation and arithmetic, not full-model inference.
use super::*;
use crate::candidate_cli::redact::tests::command;
use crate::native_engine::decode::DecodeCancellationKind;
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "fixture-only".to_owned(), source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
#[test]
fn verified_request_prices_two_independent_contexts_and_uses_ner_identity() {
    let cmd = command(&[]); let (_, limits) = cmd.validate().unwrap();
    let (p, _) = planner().unwrap(); let id = source_identity(&facts(), &p, BuiltInTask::Ner).unwrap();
    let config = detector("Alice <tool_call> é 上海", NerOptions::default(), &cmd.request(), &id, &p, &cmd, limits, &mut Continue).unwrap();
    assert_eq!(id.task_spec, "ner-v1"); assert_eq!(config.max_mask_visits, 2 * config.mask_visits_per_pass);
    assert_eq!(config.per_pass.max_kv_bytes, limits.kv_bytes);
    assert!(config.max_model_work.forward_positions > cmd.host.context_tokens as u64);
    assert!(config.planning.max_input_bytes >= cmd.host.max_result_bytes);
    Int8Redactor::new(&p, id, config).unwrap();
}
#[test]
fn no_verify_reserves_only_the_actual_original_plan() {
    let cmd = command(&["--no-verify"]); let (_, limits) = cmd.validate().unwrap();
    let (p, _) = planner().unwrap(); let id = source_identity(&facts(), &p, BuiltInTask::Ner).unwrap();
    let config = detector("Alice", NerOptions::default(), &cmd.request(), &id, &p, &cmd, limits, &mut Continue).unwrap();
    assert_eq!(config.max_mask_visits, config.mask_visits_per_pass);
    assert!(config.max_model_work.forward_positions < cmd.host.context_tokens as u64);
}
#[test]
fn verification_sums_every_work_axis_and_checks_overflow() {
    let first = constrained_int8::planned_work(20, 8).unwrap();
    let second = constrained_int8::planned_work(120, 8).unwrap();
    assert_eq!(work_ceiling(first, true, 128, 8).unwrap(), first.checked_add(second).unwrap());
    assert_eq!(work_ceiling(first, false, 128, 8).unwrap(), first);
    for axis in 0..5 {
        let mut w = first;
        match axis { 0 => w.forward_positions = u64::MAX, 1 => w.projected_logits = u64::MAX,
            2 => w.attention_pairs = u64::MAX, 3 => w.projections.dot_products = u64::MAX,
            _ => w.projections.multiply_accumulates = u64::MAX }
        assert!(work_ceiling(w, true, 128, 8).is_err());
    }
}
#[test]
fn preparation_cancellation_and_overlong_source_do_not_load_weights() {
    struct Stop;
    impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) } }
    let cmd = command(&[]); let (_, limits) = cmd.validate().unwrap();
    let (p, _) = planner().unwrap(); let id = source_identity(&facts(), &p, BuiltInTask::Ner).unwrap();
    assert!(detector("Alice", NerOptions::default(), &cmd.request(), &id, &p, &cmd, limits, &mut Stop).is_err());
    assert!(detector(&"x".repeat(cmd.host.context_tokens), NerOptions::default(), &cmd.request(), &id, &p, &cmd, limits, &mut Continue).is_err());
}
