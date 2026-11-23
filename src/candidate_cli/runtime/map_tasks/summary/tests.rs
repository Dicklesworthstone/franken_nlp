//! Real pinned preflight and exact request association. No model runs.
use super::*;
use crate::{candidate_cli::{map::tests::command, runtime::source_tasks},
    native_engine::decode::DecodeCancellationKind,
    tasks::source_planning::quantized::{capacity::Int8SourceMapCapacity, long::SourceMapTask}};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "summary-planning-fixture".to_owned(), source_root_sha256: "ab".repeat(32),
        logical_model_sha256: "cd".repeat(32) }
}
fn assets(command: &MapCommand) -> (SourceTaskPlanner, ExecutionIdentity, TaskBudget, SourceMapTask,
    Int8SourceMapCapacity, Int8SourceMapLimits) {
    let (_, limits) = command.validate().unwrap();
    let (planner, _) = source_tasks::planner().unwrap();
    let identity = source_identity(&facts(), &planner, command.kind().unwrap()).unwrap();
    let budget = command.host.task_budget(limits); let task = command.map_task(None).unwrap();
    let capacity = planner.int8_map_capacity_with_control(&task, budget,
        &PlanContext::new(&identity, budget).unwrap(), command.host.planning(), &mut Continue).unwrap();
    let mapping = command.mapping(capacity).unwrap();
    (planner, identity, budget, task, capacity, mapping)
}
fn preflight(source: &str, planner: &SourceTaskPlanner, command: &MapCommand,
    _capacity: Int8SourceMapCapacity, mapping: Int8SourceMapLimits, budget: TaskBudget,
    control: &mut impl DecodeStepControl) -> Result<Preflight, CandidateError> {
    let identity = source_identity(&facts(), planner, command.kind()?)?;
    let task = command.map_task(None)?;
    let context = PlanContext::new(&identity, budget).map_err(|_| CandidateError::Planning)?;
    super::super::preflight(source, planner, command, &task, &context, mapping, budget, control)
}
#[test]
fn summary_cli_preflight_matches_every_real_native_plan_and_retains_model_identity() {
    let cmd = command("summarize", &["--reduce-summary", "--summary-bullets", "1", "--max-chunk-bytes", "64"]);
    let (planner, identity, budget, task, capacity, mapping) = assets(&cmd);
    let source = "é 上海 Alice <tool_call>\r\n".repeat(8);
    let expected = preflight(&source, &planner, &cmd, capacity, mapping, budget, &mut Continue).unwrap();
    let prepared = planner.plan_int8_map_with_control(&source, &task, budget,
        &PlanContext::new(&identity, budget).unwrap(), cmd.host.planning(), mapping, &mut Continue).unwrap();
    prepared.check_summary(cmd.summary.limits(cmd.max_map_result_bytes).unwrap().unwrap()).unwrap();
    assert!(prepared.chunk_count() > 1); assert_eq!(expected.chunk_count(), prepared.chunk_count());
    assert_eq!(expected, prepared.preflight_metadata());
    let reduction = cmd.summary.limits(cmd.max_map_result_bytes).unwrap().unwrap();
    assert_eq!(expected.summary_with_control(reduction, &mut Continue).unwrap(),
        prepared.preflight_metadata().summary_with_control(reduction, &mut Continue).unwrap());
    for actual in prepared.execution_identities() {
        assert_eq!(actual.task_spec, "summarize-v1"); assert_eq!(actual.logical_model_digest, identity.logical_model_digest);
        assert_ne!(actual.prompt_digest, identity.prompt_digest);
    }
}
#[test]
fn opting_into_reduction_neither_adds_model_work_nor_relaxes_any_native_axis() {
    let plain = command("summarize", &["--max-chunk-bytes", "64"]);
    let reduced = command("summarize", &["--reduce-summary", "--max-chunk-bytes", "64"]);
    let (planner, _, budget, _, capacity, mapping) = assets(&plain);
    let source = "Alice é 上海\r\n".repeat(8);
    let a = preflight(&source, &planner, &plain, capacity, mapping, budget, &mut Continue).unwrap();
    let b = preflight(&source, &planner, &reduced, capacity, mapping, budget, &mut Continue).unwrap();
    assert_eq!(a, b);
    let work = a.planned_work();
    for axis in 0..5 {
        let mut cmd = command("summarize", &["--reduce-summary", "--max-chunk-bytes", "64"]);
        match axis { 0 => cmd.max_forward_positions = work.forward_positions - 1,
            1 => cmd.max_projected_logits = work.projected_logits - 1,
            2 => cmd.max_attention_pairs = work.attention_pairs - 1,
            3 => cmd.max_dot_products = work.projections.dot_products - 1,
            _ => cmd.max_multiply_accumulates = work.projections.multiply_accumulates - 1 }
        assert!(preflight(&source, &planner, &cmd, capacity, mapping, budget, &mut Continue).is_err());
    }
}
#[test]
fn summary_mode_preserves_empty_input_and_preparation_cancellation_refusals() {
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
    }
    let cmd = command("summarize", &["--reduce-summary"]);
    let (planner, _, budget, _, capacity, mapping) = assets(&cmd);
    assert!(preflight("", &planner, &cmd, capacity, mapping, budget, &mut Continue).is_err());
    assert!(preflight("Alice", &planner, &cmd, capacity, mapping, budget, &mut Stop).is_err());
}
#[test]
fn complete_summary_admission_distinguishes_same_geometry_and_unused_reducer_allowances() {
    let cmd = command("summarize", &["--reduce-summary"]);
    let (planner, _, budget, _, capacity, mapping) = assets(&cmd);
    let a = preflight("Alice", &planner, &cmd, capacity, mapping, budget, &mut Continue).unwrap();
    let b = preflight("Bobby", &planner, &cmd, capacity, mapping, budget, &mut Continue).unwrap();
    assert_eq!(a.source_span(), b.source_span()); assert_eq!(a.planned_work(), b.planned_work());
    let limits = cmd.summary.limits(cmd.max_map_result_bytes).unwrap().unwrap();
    let expected = a.summary_with_control(limits, &mut Continue).unwrap();
    assert_ne!(expected, b.summary_with_control(limits, &mut Continue).unwrap());
    let mut changed = limits; changed.aggregation.max_scan_steps -= 1;
    assert_ne!(expected, a.summary_with_control(changed, &mut Continue).unwrap());
}
