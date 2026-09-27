//! Actual pinned preflight/preparation comparisons only. No model-success fixture.
use super::*;
use crate::{candidate_cli::map::tests::command,
    native_engine::decode::DecodeCancellationKind,
    tasks::source_planning::quantized::long::question::{SourceQuestion, Int8QuestionPreflight, PreparedInt8Question}};

struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "question-planning-fixture".to_owned(), source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
fn assets(cmd: &MapCommand) -> (SourceTaskPlanner, ExecutionIdentity, TaskBudget, SourceQuestion, Int8SourceMapLimits) {
    let (_, limits) = cmd.validate().unwrap();
    let (planner, _) = planner().unwrap();
    let identity = source_identity(&facts(), &planner, BuiltInTask::Answer).unwrap();
    let budget = cmd.host.task_budget(limits);
    let question = cmd.question.task("Who is named?".to_owned(), None).unwrap();
    (planner, identity, budget, question, cmd.question_mapping().unwrap())
}
fn paired<'s>(cmd: &MapCommand, p: &SourceTaskPlanner, id: &ExecutionIdentity, b: TaskBudget,
    q: &SourceQuestion, source: &'s str, mapping: Int8SourceMapLimits)
    -> (Int8QuestionPreflight, PreparedInt8Question<'s>) {
    let ctx = PlanContext::new(id, b).unwrap();
    let expected = p.preflight_int8_question_with_control(source, q, b, &ctx, cmd.host.planning(), mapping, &mut Continue).unwrap();
    let prepared = p.plan_int8_question_with_control(source, q, b, &ctx, cmd.host.planning(), mapping, &mut Continue).unwrap();
    assert_eq!(expected, prepared.preflight_metadata());
    (expected, prepared)
}
#[test]
fn cli_question_preflight_equals_real_native_plans_with_a_long_unicode_source() {
    let cmd = command("answer", &["--question", "q"]);
    let (p, id, b, q, mapping) = assets(&cmd);
    let source = "Alice <tool_call> é 上海 😀\r\n".repeat(100);
    assert!(source.len() > cmd.host.context_tokens);
    let (expected, prepared) = paired(&cmd, &p, &id, b, &q, &source, mapping);
    assert!(expected.native_chunks() > 1);
    assert_eq!(expected.source_span().byte_end, source.len());
    assert_eq!(expected.source_span().scalar_end, source.chars().count());
    assert_eq!(prepared.execution_identities().len(), expected.native_chunks());
    for actual in prepared.execution_identities() {
        assert_eq!(actual.task_spec, "answer-v1"); assert_eq!(actual.logical_model_digest, id.logical_model_digest);
        assert_ne!(actual.prompt_digest, id.prompt_digest);
    }
}
#[test]
fn question_only_changes_are_committed_and_blank_ranges_are_not_native_answers() {
    let cmd = command("answer", &["--question", "q", "--max-chunk-bytes", "6"]);
    let (p, id, b, mut q, mapping) = assets(&cmd);
    let (expected, a) = paired(&cmd, &p, &id, b, &q, "éAéA      éAéA", mapping);
    assert_eq!(expected.chunk_count(), 3); assert_eq!(expected.native_chunks(), 2); assert_eq!(expected.whitespace_chunks(), 1);
    assert_eq!(expected.reserved_mask_visits(), mapping.mask_visits_per_chunk * 2);
    q.question = "Which person?".to_owned();
    let (_, other) = paired(&cmd, &p, &id, b, &q, "éAéA      éAéA", mapping);
    for (a, other) in a.execution_identities().zip(other.execution_identities()) {
        assert_ne!(a.prompt_digest, other.prompt_digest); assert_ne!(a.taskir_digest, other.taskir_digest);
    }
}
#[test]
fn every_whole_run_native_axis_and_mask_ceiling_refuses_before_weight_loading() {
    let cmd = command("answer", &["--question", "q", "--max-chunk-bytes", "64"]);
    let (p, id, b, q, mapping) = assets(&cmd); let source = "Alice é 上海\r\n".repeat(16);
    let ctx = PlanContext::new(&id, b).unwrap();
    let expected = p.preflight_int8_question_with_control(&source, &q, b, &ctx, cmd.host.planning(), mapping, &mut Continue).unwrap();
    let mut exact = mapping; exact.max_model_work = expected.planned_work(); exact.max_mask_visits = expected.reserved_mask_visits();
    assert!(p.preflight_int8_question_with_control(&source, &q, b, &ctx, cmd.host.planning(), exact, &mut Continue).is_ok());
    for axis in 0..6 {
        let mut bad = exact;
        match axis { 0 => bad.max_model_work.forward_positions -= 1, 1 => bad.max_model_work.projected_logits -= 1,
            2 => bad.max_model_work.attention_pairs -= 1, 3 => bad.max_model_work.projections.dot_products -= 1,
            4 => bad.max_model_work.projections.multiply_accumulates -= 1, _ => bad.max_mask_visits -= 1 }
        assert!(p.preflight_int8_question_with_control(&source, &q, b, &ctx, cmd.host.planning(), bad, &mut Continue).is_err(), "{axis}");
    }
}
#[test]
fn oversized_question_empty_source_chunk_exhaustion_and_deadline_have_no_preflight_success() {
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
    }
    let cmd = command("answer", &["--question", "q", "--max-chunk-bytes", "6"]);
    let (p, id, b, mut q, mapping) = assets(&cmd); let ctx = PlanContext::new(&id, b).unwrap();
    for source in ["", "                  "] {
        assert!(p.preflight_int8_question_with_control(source, &q, b, &ctx, cmd.host.planning(), mapping, &mut Continue).is_err());
    }
    assert!(p.preflight_int8_question_with_control("Alice", &q, b, &ctx, cmd.host.planning(), mapping, &mut Stop).is_err());
    let mut single = mapping; single.chunks.max_chunks = 1;
    assert!(p.preflight_int8_question_with_control("Alice Bob Carol", &q, b, &ctx, cmd.host.planning(), single, &mut Continue).is_err());
    q.question = "q".repeat(b.max_input_tokens as usize);
    assert!(p.preflight_int8_question_with_control("Alice", &q, b, &ctx, cmd.host.planning(), mapping, &mut Continue).is_err());
}
