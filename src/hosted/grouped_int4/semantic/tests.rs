//! UNRUN shared admission fixtures; no native quality or inference evidence.
use super::*;

fn fixture() -> (Int4SemanticHostLimits, TaskBudget, TaskBudget, SemanticLimits, JudgeLimits) {
    let kv = Int4MemoryRequirement::for_context(256, 32).unwrap().kv_bytes;
    let extraction = TaskBudget { max_input_tokens: 224, max_output_tokens: 32,
        max_output_bytes: 65_536, max_grammar_states: 4096, max_kv_bytes: kv };
    let judge = TaskBudget { max_input_tokens: 240, max_output_tokens: 8,
        max_output_bytes: 16_384, max_grammar_states: 1, max_kv_bytes: kv };
    let semantic = SemanticLimits { max_fields: 3, max_claim_bytes: 128,
        max_total_claim_bytes: 1024, max_template_bytes: 1024, max_walk_nodes: 128,
        max_input_json_bytes: 8192, max_total_prompt_tokens: 2048,
        max_total_projected_logits: 1_000_000, max_output_bytes: 131_072 };
    let planning = JudgeLimits { max_total_prompt_tokens: 2048, max_total_candidate_tokens: 256,
        max_total_projected_logits: 1_000_000, max_output_bytes: 16_384, ..JudgeLimits::default() };
    let host = Int4SemanticHostLimits {
        native: NativeLimits { context_tokens: 256, allocator_reserve_bytes: 65_536,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 1000,
                cleanup_reserve_bytes: 65_536 } },
        masks: MaskWorkLimits { max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 },
        max_mask_node_visits: 32_000, preparation_reserve_bytes: 256 << 20,
    };
    (host, extraction, judge, semantic, planning)
}

#[test]
fn both_stages_need_the_whole_resident_context_even_when_each_prompt_is_short() {
    let (host, extraction, judge, semantic, planning) = fixture();
    let required = host.requirements(32, extraction, judge, semantic, planning, 1024).unwrap();
    assert_eq!(required.native.kv_bytes, extraction.max_kv_bytes);
    for stage in 0..2 {
        let (mut extraction, mut judge) = (extraction, judge);
        if stage == 0 { extraction.max_kv_bytes -= 1; } else { judge.max_kv_bytes -= 1; }
        assert!(host.requirements(32, extraction, judge, semantic, planning, 1024).is_err());
    }
    let mut wrong = host; wrong.native.context_tokens += 1;
    assert!(wrong.requirements(32, extraction, judge, semantic, planning, 1024).is_err());
    assert!(host.requirements(0, extraction, judge, semantic, planning, 1024).is_err());
}

#[test]
fn more_field_plans_increase_memory_even_with_one_fixed_aggregate_prompt_budget() {
    let (host, extraction, judge, mut semantic, planning) = fixture();
    semantic.max_fields = 1;
    let one = host.requirements(32, extraction, judge, semantic, planning, 16_384).unwrap();
    semantic.max_fields = 4;
    let four = host.requirements(32, extraction, judge, semantic, planning, 16_384).unwrap();
    assert_eq!(one.native, four.native);
    assert_eq!(one.native_workspace_bytes, four.native_workspace_bytes);
    assert_eq!(one.output_bytes, four.output_bytes);
    // The common prompt ceiling cannot hide separately retained source copies,
    // scorer graphs and complete per-field outputs that all coexist.
    assert!(four.preparation_bytes - one.preparation_bytes >= 3 * (16_384 + 4 * judge.max_output_bytes));
    assert!(four.total_bytes().unwrap() > one.total_bytes().unwrap());
}

#[test]
fn final_result_capacity_does_not_substitute_for_retained_sources_and_judge_results() {
    let (host, extraction, judge, semantic, planning) = fixture();
    let small = host.requirements(32, extraction, judge, semantic, planning, 1024).unwrap();
    let large_source = host.requirements(32, extraction, judge, semantic, planning, 32_768).unwrap();
    assert_eq!(small.output_bytes, large_source.output_bytes);
    assert!(large_source.preparation_bytes - small.preparation_bytes >=
        semantic.max_fields as u64 * (32_768 - 1024));
    let mut wider = judge; wider.max_output_bytes *= 2;
    let mut plans = planning; plans.max_output_bytes *= 2;
    let large_results = host.requirements(32, extraction, wider, semantic, plans, 1024).unwrap();
    assert_eq!(small.output_bytes, large_results.output_bytes);
    assert!(large_results.preparation_bytes > small.preparation_bytes);
}

#[test]
fn native_intermediates_and_final_output_remain_independent_simultaneous_claims() {
    let (host, extraction, judge, semantic, planning) = fixture();
    let required = host.requirements(32, extraction, judge, semantic, planning, 1024).unwrap();
    assert!(required.preparation_bytes > host.preparation_reserve_bytes);
    assert!(required.output_bytes >= 4 * semantic.max_output_bytes);
    assert!(required.native_workspace_bytes > required.native.scratch_payload_bound);
    let total = required.total_bytes().unwrap();
    for part in [required.native.kv_bytes, required.native_workspace_bytes,
        required.preparation_bytes, required.output_bytes] { assert!(total > part); }
    let impossible = Int4SemanticHostRequirements { preparation_bytes: u64::MAX, ..required };
    assert!(impossible.total_bytes().is_err());
}

#[test]
fn zero_controls_missing_reserves_and_variable_memory_overflow_refuse_before_execution() {
    for axis in 0..10 {
        let (mut host, extraction, judge, mut semantic, mut planning) = fixture();
        match axis { 0 => host.preparation_reserve_bytes = 0, 1 => host.native.allocator_reserve_bytes = 0,
            2 => host.max_mask_node_visits = 0, 3 => host.masks.max_trie_node_visits = 0,
            4 => host.masks.checkpoint_interval_nodes = 0, 5 => semantic.max_fields = 0,
            6 => planning.max_total_candidate_tokens = 0, 7 => semantic.max_output_bytes = u64::MAX,
            8 => host.preparation_reserve_bytes = u64::MAX, _ => planning.max_output_bytes = 0 }
        assert!(host.requirements(32, extraction, judge, semantic, planning, 1024).is_err(), "axis {axis}");
    }
    let (host, extraction, judge, semantic, planning) = fixture();
    assert!(host.requirements(32, extraction, judge, semantic, planning, 0).is_err());
    if usize::BITS >= 64 {
        assert!(host.requirements(32, extraction, judge, semantic, planning, usize::MAX).is_err());
    }
}

#[test]
fn typed_cancellation_stays_available_without_exposing_private_claim_diagnostics() {
    let cause = crate::native_engine::decode::DecodeCancellationKind::Deadline;
    let error = Int4HostError::SemanticExtraction(Int4SemanticError::Cancelled(cause));
    assert_eq!(error.to_string(), format!("{error:?}"));
    assert_eq!(error.to_string(), "INT4 extraction second-reader execution or finalization refused");
    assert!(error.source().is_some());
    let Int4HostError::SemanticExtraction(inner) = error else { panic!("wrong cause"); };
    assert_eq!(inner.cancellation(), Some(cause));
}

#[test]
fn source_planners_vocabulary_and_real_combined_output_cross_one_owned_boundary() {
    fn send<T: Send + 'static>() {}
    send::<SemanticInput>();
    send::<PreparedInt4SemanticExtraction>();
    send::<Int4SemanticExtractionRun>();
    send::<Arc<ExtractionVocabulary>>();
    let _entry = NlpEngine::execute_int4_semantic_extract;
}
