//! Pinned preflight and private delivery-corruption fixtures. No model runs.
use super::*;
use crate::{candidate_cli::{map::tests::command, runtime::source_tasks},
    native_engine::decode::DecodeCancellationKind,
    tasks::{corpus_keyphrases::CorpusKeyphraseResult, source_planning::quantized::long::SourceMapTask},
    validation::grounded_fields::VerifiedSourceSpan,
};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "keyphrase-planning-fixture".to_owned(), source_root_sha256: "ab".repeat(32),
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
fn preflight_matches_every_real_native_plan_without_loading_a_model() {
    let cmd = command("keyphrases", &["--reduce-keyphrases", "--document-keyphrases", "1", "--max-chunk-bytes", "64"]);
    let (planner, identity, budget, task, capacity, mapping) = assets(&cmd);
    let source = "é 上海 Alice <tool_call>\r\n".repeat(8);
    let expected = preflight(&source, &planner, &cmd, capacity, mapping, budget, &mut Continue).unwrap();
    let prepared = planner.plan_int8_map_with_control(&source, &task, budget,
        &PlanContext::new(&identity, budget).unwrap(), cmd.host.planning(), mapping, &mut Continue).unwrap();
    prepared.check_keyphrases(cmd.keyphrases.limits(cmd.max_map_result_bytes).unwrap().unwrap()).unwrap();
    assert!(prepared.chunk_count() > 1); assert_eq!(expected.chunks, prepared.chunk_count());
    assert_eq!(expected.work, prepared.planned_work()); assert_eq!(expected.masks, prepared.reserved_mask_visits());
    for actual in prepared.execution_identities() {
        assert_eq!(actual.task_spec, "keyphrases-v1"); assert_eq!(actual.logical_model_digest, identity.logical_model_digest);
        assert_ne!(actual.prompt_digest, identity.prompt_digest);
    }
}
#[test]
fn reduction_preserves_native_work_and_whole_document_limits_on_all_five_axes() {
    let plain = command("keyphrases", &["--max-chunk-bytes", "64"]);
    let reduced = command("keyphrases", &["--reduce-keyphrases", "--max-chunk-bytes", "64"]);
    let (planner, _, budget, _, capacity, mapping) = assets(&plain);
    let source = "Alice é 上海\r\n".repeat(8);
    let a = preflight(&source, &planner, &plain, capacity, mapping, budget, &mut Continue).unwrap();
    let b = preflight(&source, &planner, &reduced, capacity, mapping, budget, &mut Continue).unwrap();
    assert_eq!(a.work, b.work); assert_eq!(a.masks, b.masks); assert_eq!(a.chunks, b.chunks);
    for axis in 0..5 {
        let mut cmd = command("keyphrases", &["--reduce-keyphrases", "--max-chunk-bytes", "64"]);
        match axis { 0 => cmd.max_forward_positions = a.work.forward_positions - 1,
            1 => cmd.max_projected_logits = a.work.projected_logits - 1,
            2 => cmd.max_attention_pairs = a.work.attention_pairs - 1,
            3 => cmd.max_dot_products = a.work.projections.dot_products - 1,
            _ => cmd.max_multiply_accumulates = a.work.projections.multiply_accumulates - 1 }
        assert!(preflight(&source, &planner, &cmd, capacity, mapping, budget, &mut Continue).is_err());
    }
}
#[test]
fn empty_input_and_preparation_cancellation_never_become_empty_success() {
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
    }
    let cmd = command("keyphrases", &["--reduce-keyphrases"]);
    let (planner, _, budget, _, capacity, mapping) = assets(&cmd);
    assert!(preflight("", &planner, &cmd, capacity, mapping, budget, &mut Continue).is_err());
    assert!(preflight("Alice", &planner, &cmd, capacity, mapping, budget, &mut Stop).is_err());
}
// Only a private verifier fixture, never a native-success or relevance receipt.
fn envelope() -> (Preflight, Int8CorpusKeyphraseRun) {
    let work = constrained_int8::planned_work(10, 2).unwrap();
    let expected = Preflight { chunks: 1, source_bytes: 3, source_scalars: 2, work, masks: 100 };
    let keyphrases = CorpusKeyphraseResult { schema_version: 1, task_spec_version: KEYPHRASES_TASK_VERSION.to_owned(),
        ranking_policy: CORPUS_KEYPHRASE_POLICY.to_owned(), score_space: ScoreSpace::NotComputed,
        grounding: ExtractionGrounding::SourceMembership, mapped_chunks: 1, omitted_candidates: 0, phrases: Vec::new(),
        forward_positions: work.forward_positions, projected_logits: work.projected_logits, mask_node_visit_charge: 20,
        warnings: [CorpusKeyphraseWarning::ChunkBoundariesMaySplitPhrases,
            CorpusKeyphraseWarning::SingleContextEquivalenceNotEstablished, CorpusKeyphraseWarning::ModelSelectionIsNotRecallGuarantee] };
    (expected, Int8CorpusKeyphraseRun { schema_version: 1, execution: INT8_CORPUS_KEYPHRASE_EXECUTION,
        numerics_profile: STRICT_INT8_PROFILE, chunk_profile: CHUNK_PROFILE, semantics: INT8_CORPUS_KEYPHRASE_SEMANTICS,
        source_span: VerifiedSourceSpan { byte_start: 0, byte_end: 3, scalar_start: 0, scalar_end: 2 },
        map_batches: 1, reduce_calls: 0, reduction_levels: 0, requested_max_phrases: 16,
        planned_model_work: work, model_work: work, reserved_mask_node_visits: 100,
        mask_node_visit_charge: 20, verification_scan_work: 0, keyphrases })
}
#[test]
fn delivery_rejects_partial_coverage_profiles_policies_and_substituted_work() {
    let limits = Int8KeyphraseLimits::default();
    let (expected, result) = envelope(); check_completed(&expected, &result, limits).unwrap();
    for axis in 0..18 {
        let (expected, mut result) = envelope();
        match axis { 0 => result.source_span.byte_end -= 1, 1 => result.source_span.scalar_end += 1,
            2 => result.keyphrases.mapped_chunks += 1, 3 => result.requested_max_phrases = 1,
            4 => result.model_work.forward_positions += 1, 5 => result.model_work.projected_logits += 1,
            6 => result.model_work.attention_pairs += 1, 7 => result.model_work.projections.dot_products += 1,
            8 => result.model_work.projections.multiply_accumulates += 1,
            9 => result.reserved_mask_node_visits += 1, 10 => result.numerics_profile = "hf-bf16-eager",
            11 => result.keyphrases.mask_node_visit_charge += 1,
            12 => result.keyphrases.ranking_policy = "foreign-policy".to_owned(),
            13 => result.chunk_profile = "foreign-partition", 14 => result.semantics = "neural-reranking",
            15 => result.verification_scan_work = limits.aggregation.max_scan_work + 1,
            16 => result.keyphrases.omitted_candidates = usize::MAX,
            _ => result.keyphrases.warnings.swap(0, 1) }
        assert!(check_completed(&expected, &result, limits).is_err(), "{axis}");
    }
}
