//! Pure admission, planning and one-invocation work contracts, without weights.
use super::*;
use crate::native_engine::portable_int8::ProjectionWork;
fn host() -> Int4SentimentAssayHostLimits {
    Int4SentimentAssayHostLimits { native: NativeLimits { context_tokens: 2048, allocator_reserve_bytes: 65536,
        run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 1000, cleanup_reserve_bytes: 65536 } },
        task: TaskBudget { max_input_tokens: 2044, max_output_tokens: 5, max_output_bytes: 100_000,
            max_grammar_states: 17, max_kv_bytes: Int4MemoryRequirement::for_context(2048, 128).unwrap().kv_bytes },
        max_axes: 4, preparation_reserve_bytes: 512 * 1024 * 1024,
        max_model_work: Int4Work { forward_positions: u64::MAX, projected_logits: u64::MAX,
            attention_pairs: u64::MAX, projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } } }
}
fn row() -> Int4Work { Int4Work { forward_positions: 2, projected_logits: 3, attention_pairs: 4,
    projections: ProjectionWork { dot_products: 5, multiply_accumulates: 6 } } }
fn axis(work: &mut Int4Work, index: usize) -> &mut u64 {
    match index { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits,
        2 => &mut work.attention_pairs, 3 => &mut work.projections.dot_products,
        4 => &mut work.projections.multiply_accumulates, _ => panic!("unknown axis") }
}
#[test]
fn both_models_reuse_one_engine_peak_while_retaining_both_complete_replays() {
    let host = host(); let limits = metrics::Limits::default();
    let left = host.requirements(&[3], limits).unwrap(); let right = host.requirements(&[10752], limits).unwrap();
    let pair = host.requirements(&[3, 10752], limits).unwrap();
    assert_eq!(pair, host.requirements(&[10752, 3], limits).unwrap());
    assert_eq!(pair.native, right.native); assert_eq!(pair.workspace_bytes, right.workspace_bytes);
    assert_eq!(pair.native.kv_bytes, left.native.kv_bytes); assert_eq!(pair.output_reservation_bytes, left.output_reservation_bytes);
    assert_eq!(pair.input_bytes - left.input_bytes, native_sentiment::required_replay_bytes(limits, 4, 2).unwrap()
        - native_sentiment::required_replay_bytes(limits, 4, 1).unwrap());
    assert_eq!(left.output_reservation_bytes, 4 * limits.max_output_bytes as u64 + 4096);
    let mut fewer = host; fewer.max_axes -= 1;
    assert!(fewer.requirements(&[3], limits).unwrap().input_bytes < left.input_bytes);
}
#[test]
fn a_complete_first_arm_cannot_renew_any_work_allowance_for_the_second() {
    let complete = row().checked_add(row()).unwrap();
    for field in 0..5 {
        let mut ceiling = complete; *axis(&mut ceiling, field) -= 1;
        let mut spent = Int4Work::default(); charge_work(&mut spent, row(), ceiling).unwrap();
        let before = spent;
        assert!(matches!(charge_work(&mut spent, row(), ceiling),
            Err(HostedError::Int4(Int4HostError::Sentiment(Int4SentimentError::WorkBudget)))));
        assert_eq!(spent, before);
        *axis(&mut spent, field) = u64::MAX; let before = spent;
        assert!(charge_work(&mut spent, row(), host().max_model_work).is_err()); assert_eq!(spent, before);
    }
}
#[test]
fn undersized_complete_resources_refuse_before_private_io_or_planning() {
    for field in 0..14 {
        let mut value = host();
        match field { 0 => value.task.max_kv_bytes -= 1, 1 => value.native.context_tokens = 0,
            2 => value.preparation_reserve_bytes = 0, 3 => value.native.allocator_reserve_bytes = 0,
            4 => value.max_axes = 0, 5 => value.task.max_output_tokens = 4, 6 => value.native.run.max_checkpoints = 1,
            7 => value.task.max_grammar_states = 16, 8 => value.task.max_input_tokens += 1,
            _ => *axis(&mut value.max_model_work, field - 9) = 0 }
        assert!(value.requirements(&[128], metrics::Limits::default()).is_err(), "field {field}");
    }
    for widths in [&[][..], &[0][..], &[10753][..], &[1, 2, 3][..]] {
        assert!(host().requirements(widths, metrics::Limits::default()).is_err());
    }
}
#[test]
fn admitted_fixed_bins_cover_conservative_nodes_and_compile_all_four_axes() {
    use crate::{execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
        tasks::{ir::PlanContext, sentiment::{SentimentAxis, SentimentPlanner, SentimentPolicy, SentimentRequest}}};
    let host = host(); let limits = planning_limits(host).unwrap();
    assert_eq!(limits.per_axis.max_nodes, 17); assert_eq!(limits.max_total_nodes, 68);
    let registry = crate::tokenizer::pinned_controls::pinned().unwrap();
    let eos_token_id = registry.template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    let planner = SentimentPlanner::pinned(registry.template_controls(), SentimentOptions {
        mode: ScoringMode::FullVocabulary, eos_token_id,
        policy: SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 } }).unwrap();
    let d = Sha256Digest::of_bytes(b"synthetic-planning-fixture");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
        task_spec: "sentiment-v1".to_owned(), taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(),
        schema_digest: d, numerics_profile: NumericsProfile::HfBf16Eager, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: "fixture".to_owned(),
        host_class: None, compiler_identity: None };
    let request = SentimentRequest { document: "Private fixture é上海😀".to_owned(), axes: SentimentAxis::ALL.to_vec(), budget: host.task };
    let context = PlanContext::new(&identity, host.task).unwrap();
    let prepared = planner.plan(&request, &context, limits).unwrap();
    assert_eq!(prepared.planned_work().prefix_evaluations, 40);
    assert_eq!(prepared.planned_work().scored_edges, 56);
    let mut undersized = limits; undersized.per_axis.max_nodes = 15;
    assert!(planner.plan(&request, &context, undersized).is_err());
}
