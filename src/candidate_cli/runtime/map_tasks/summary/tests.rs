//! Real pinned preflight plus private output-corruption fixtures. No model runs.
use super::*;
use crate::{candidate_cli::{map::tests::command, runtime::source_tasks},
    corpus::summarize::{CorpusSummaryResult, SummaryWarning, SUMMARY_REDUCTION_POLICY},
    native_engine::decode::DecodeCancellationKind,
    tasks::{ir::ScoreSpace, mapreduce::CHUNK_PROFILE,
        source_planning::quantized::long::SourceMapTask,
        summarize::{CitationGuarantee, SummarySemanticSupport}},
    validation::grounded_fields::VerifiedSourceSpan,
};
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
#[test]
fn summary_cli_preflight_matches_every_real_native_plan_and_retains_model_identity() {
    let cmd = command("summarize", &["--reduce-summary", "--summary-bullets", "1", "--max-chunk-bytes", "64"]);
    let (planner, identity, budget, task, capacity, mapping) = assets(&cmd);
    let source = "é 上海 Alice <tool_call>\r\n".repeat(8);
    let expected = preflight(&source, &planner, &cmd, capacity, mapping, budget, &mut Continue).unwrap();
    let prepared = planner.plan_int8_map_with_control(&source, &task, budget,
        &PlanContext::new(&identity, budget).unwrap(), cmd.host.planning(), mapping, &mut Continue).unwrap();
    prepared.check_summary(cmd.summary.limits(cmd.max_map_result_bytes).unwrap().unwrap()).unwrap();
    assert!(prepared.chunk_count() > 1); assert_eq!(expected.chunks, prepared.chunk_count());
    assert_eq!(expected.work, prepared.planned_work()); assert_eq!(expected.masks, prepared.reserved_mask_visits());
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
    assert_eq!(a.work, b.work); assert_eq!(a.masks, b.masks); assert_eq!(a.chunks, b.chunks);
    for axis in 0..5 {
        let mut cmd = command("summarize", &["--reduce-summary", "--max-chunk-bytes", "64"]);
        match axis { 0 => cmd.max_forward_positions = a.work.forward_positions - 1,
            1 => cmd.max_projected_logits = a.work.projected_logits - 1,
            2 => cmd.max_attention_pairs = a.work.attention_pairs - 1,
            3 => cmd.max_dot_products = a.work.projections.dot_products - 1,
            _ => cmd.max_multiply_accumulates = a.work.projections.multiply_accumulates - 1 }
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
// A private verifier fixture only: it is never returned by the native host and
// is not a model-success, summary-quality, or performance assertion.
fn envelope() -> (Preflight, Int8CorpusSummaryRun) {
    let work = constrained_int8::planned_work(10, 2).unwrap();
    let expected = Preflight { chunks: 1, source_bytes: 3, source_scalars: 2, work, masks: 100 };
    let summary = CorpusSummaryResult { schema_version: 1, task_spec_version: SUMMARIZE_TASK_VERSION.to_owned(),
        reduction_policy: SUMMARY_REDUCTION_POLICY.to_owned(), citation_guarantee: CitationGuarantee::StructuralSourceMembership,
        semantic_support: SummarySemanticSupport::NotAssessed, score_space: ScoreSpace::NotComputed,
        mapped_chunks: 1, requested_max_bullets: 16, omitted_bullets: 0, bullets: Vec::new(),
        forward_positions: work.forward_positions, projected_logits: work.projected_logits, mask_node_visit_charge: 20,
        untrusted_fields: ["bullets".to_owned()], warnings: [SummaryWarning::LossyMapAndFinalSelection,
            SummaryWarning::SingleContextEquivalenceNotEstablished, SummaryWarning::SemanticConflictsNotReconciled,
            SummaryWarning::ChunkBoundariesMaySplitContext, SummaryWarning::SupportFrequencyIsNotImportanceOrConfidence] };
    (expected, Int8CorpusSummaryRun { schema_version: 1, execution: INT8_CORPUS_SUMMARY_EXECUTION,
        numerics_profile: STRICT_INT8_PROFILE, chunk_profile: CHUNK_PROFILE,
        semantics: "exact-bullet-evidence-union-no-neural-synthesis-v1",
        source_span: VerifiedSourceSpan { byte_start: 0, byte_end: 3, scalar_start: 0, scalar_end: 2 },
        map_batches: 1, reduce_calls: 0, reduction_levels: 0, planned_model_work: work, model_work: work,
        reserved_mask_node_visits: 100, mask_node_visit_charge: 20, verification_scan_steps: 0, summary })
}
#[test]
fn delivery_rejects_partial_coverage_changed_profiles_or_substituted_work() {
    let (expected, result) = envelope(); check_completed(&expected, &result, 16).unwrap();
    for axis in 0..12 {
        let (expected, mut result) = envelope();
        match axis { 0 => result.source_span.byte_end -= 1, 1 => result.source_span.scalar_end += 1,
            2 => result.summary.mapped_chunks += 1, 3 => result.summary.requested_max_bullets = 1,
            4 => result.model_work.forward_positions += 1, 5 => result.model_work.projected_logits += 1,
            6 => result.model_work.attention_pairs += 1, 7 => result.model_work.projections.dot_products += 1,
            8 => result.model_work.projections.multiply_accumulates += 1,
            9 => result.reserved_mask_node_visits += 1, 10 => result.numerics_profile = "hf-bf16-eager",
            _ => result.summary.mask_node_visit_charge += 1 }
        assert!(check_completed(&expected, &result, 16).is_err(), "{axis}");
    }
}
