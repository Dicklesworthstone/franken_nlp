//! Pinned actual document preflight. These definitions do not execute a model.
use super::*;
use crate::{candidate_cli::map::tests::command,
    native_engine::decode::DecodeCancellationKind, tasks::summarize::SummaryOptions};
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(), recipe_id: "metadata-only-unit-fixture".to_owned(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
struct Control { calls: usize, stop: usize }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1; (self.calls >= self.stop).then_some(DecodeCancellationKind::Deadline)
    }
}
fn continuing() -> Control { Control { calls: 0, stop: usize::MAX } }
fn preflight(c: &MapCommand, text: &str, control: &mut Control) -> Result<Int8SummarySynthesisPreflight, CandidateError> {
    let (_, limits) = c.validate()?;
    let (p, _) = planner()?;
    let budget = c.host.task_budget(limits);
    let id = source_identity(&facts(), &p, crate::tasks::BuiltInTask::Summarize)?;
    let ctx = PlanContext::new(&id, budget).map_err(|_| CandidateError::Planning)?;
    let options = SummaryOptions::default();
    let request = c.summary.synthesis.request(options, c.host.planning())?;
    let capacity = p.int8_map_capacity_with_control(&SourceMapTask::Summarize(options), budget, &ctx,
        c.host.planning(), &mut continuing()).map_err(|_| CandidateError::Planning)?;
    prepare_metadata(text, &p, request, budget, &ctx, c, c.mapping(capacity)?, control)
}
#[test]
fn preflight_prices_the_complete_document_and_one_more_native_pass_before_loading() {
    let c = command("summarize", &["--synthesize-summary", "--max-chunk-bytes", "6"]);
    let text = "éAéAéAéA"; let p = preflight(&c, text, &mut continuing()).unwrap();
    assert_eq!(p.chunk_count(), 2); assert_eq!(p.source_span().byte_end, text.len());
    assert_eq!(p.source_span().scalar_end, text.chars().count());
    assert_eq!(p.reserved_mask_visits().unwrap(), 3 * c.host.max_mask_node_visits);
    assert!(p.reserved_model_work().unwrap().forward_positions > p.synthesis_reserved_work().forward_positions);
}
#[test]
fn every_work_axis_and_masks_including_synthesis_have_an_exact_cli_boundary() {
    let mut c = command("summarize", &["--synthesize-summary", "--max-chunk-bytes", "6"]);
    let text = "éAéAéAéA"; let p = preflight(&c, text, &mut continuing()).unwrap();
    let work = p.reserved_model_work().unwrap();
    c.max_forward_positions = work.forward_positions; c.max_projected_logits = work.projected_logits;
    c.max_attention_pairs = work.attention_pairs; c.max_dot_products = work.projections.dot_products;
    c.max_multiply_accumulates = work.projections.multiply_accumulates;
    c.max_total_mask_node_visits = p.reserved_mask_visits().unwrap();
    preflight(&c, text, &mut continuing()).unwrap();
    for axis in 0..6 {
        match axis { 0 => c.max_forward_positions -= 1, 1 => c.max_projected_logits -= 1,
            2 => c.max_attention_pairs -= 1, 3 => c.max_dot_products -= 1,
            4 => c.max_multiply_accumulates -= 1, _ => c.max_total_mask_node_visits -= 1 }
        assert!(preflight(&c, text, &mut continuing()).is_err(), "{axis}");
        c.max_forward_positions = work.forward_positions; c.max_projected_logits = work.projected_logits;
        c.max_attention_pairs = work.attention_pairs; c.max_dot_products = work.projections.dot_products;
        c.max_multiply_accumulates = work.projections.multiply_accumulates;
        c.max_total_mask_node_visits = p.reserved_mask_visits().unwrap();
    }
}
#[test]
fn final_preparation_deadline_and_empty_source_return_no_preflight_receipt() {
    let c = command("summarize", &["--synthesize-summary", "--max-chunk-bytes", "6"]);
    let mut control = continuing(); preflight(&c, "éAéA", &mut control).unwrap();
    for stop in [1, control.calls] {
        assert_eq!(preflight(&c, "éAéA", &mut Control { calls: 0, stop }).err(), Some(CandidateError::Timeout));
    }
    assert!(preflight(&c, "", &mut continuing()).is_err());
}
fn tree_preflight(c: &MapCommand, text: &str, control: &mut Control) -> Result<Int8SummaryHierarchyPreflight, CandidateError> {
    let (_, limits) = c.validate()?; let (p, _) = planner()?;
    let budget = c.host.task_budget(limits);
    let id = source_identity(&facts(), &p, crate::tasks::BuiltInTask::Summarize)?;
    let ctx = PlanContext::new(&id, budget).map_err(|_| CandidateError::Planning)?;
    let options = SummaryOptions::default();
    let request = c.summary.synthesis.request(options, c.host.planning())?;
    let capacity = p.int8_map_capacity_with_control(&SourceMapTask::Summarize(options), budget, &ctx,
        c.host.planning(), &mut continuing()).map_err(|_| CandidateError::Planning)?;
    match prepare_mode(text, &p, request, budget, &ctx, c, c.mapping(capacity)?, control)? {
        Expected::Hierarchical(expected) => Ok(expected), Expected::Single(_) => Err(CandidateError::Arguments),
    }
}
fn tree_command() -> MapCommand {
    command("summarize", &["--synthesize-summary", "--hierarchical-summary", "--summary-max-passes", "4", "--max-chunk-bytes", "6"])
}
#[test]
fn hierarchical_route_reserves_all_allowed_calls_before_any_model_loading() {
    let text = "éAéAéAéA";
    let single = command("summarize", &["--synthesize-summary", "--max-chunk-bytes", "6"]);
    let single = preflight(&single, text, &mut continuing()).unwrap();
    let c = tree_command(); let tree = tree_preflight(&c, text, &mut continuing()).unwrap();
    assert_eq!(tree.chunk_count(), 2); assert_eq!(tree.source_span(), single.source_span());
    assert_eq!(tree.reserved_mask_visits(), 6 * c.host.max_mask_node_visits);
    let mut work = single.reserved_model_work().unwrap();
    for _ in 0..3 { work = work.checked_add(single.synthesis_reserved_work()).unwrap(); }
    assert_eq!(tree.reserved_model_work(), work);
}
#[test]
fn hierarchy_whole_run_admission_rejects_one_unit_short_on_every_axis() {
    let text = "éAéAéAéA"; let expected = tree_preflight(&tree_command(), text, &mut continuing()).unwrap();
    let w = expected.reserved_model_work();
    for axis in 0..7 {
        let mut c = tree_command();
        c.max_forward_positions = w.forward_positions; c.max_projected_logits = w.projected_logits;
        c.max_attention_pairs = w.attention_pairs; c.max_dot_products = w.projections.dot_products;
        c.max_multiply_accumulates = w.projections.multiply_accumulates; c.max_total_mask_node_visits = expected.reserved_mask_visits();
        match axis { 0 => c.max_forward_positions -= 1, 1 => c.max_projected_logits -= 1,
            2 => c.max_attention_pairs -= 1, 3 => c.max_dot_products -= 1, 4 => c.max_multiply_accumulates -= 1,
            5 => c.max_total_mask_node_visits -= 1, _ => () }
        assert_eq!(tree_preflight(&c, text, &mut continuing()).is_ok(), axis == 6);
    }
}
#[test]
fn hierarchy_preparation_cancellation_preserves_deadline_through_the_cli() {
    let c = tree_command(); let mut control = continuing(); tree_preflight(&c, "éAéA", &mut control).unwrap();
    for stop in [1, control.calls] {
        assert_eq!(tree_preflight(&c, "éAéA", &mut Control { calls: 0, stop }).err(), Some(CandidateError::Timeout));
    }
}
