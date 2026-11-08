//! Admission and ownership contracts only; these fixtures perform no inference.
use super::*;

fn fixture() -> (Int4SourceSemanticHostLimits, TaskBudget, TaskBudget, SourceSemanticLimits,
    JudgeLimits, SourceSemanticInputFacts) {
    let kv = Int4MemoryRequirement::for_context(256, 32).unwrap().kv_bytes;
    let source = TaskBudget { max_input_tokens: 224, max_output_tokens: 32,
        max_output_bytes: 65_536, max_grammar_states: 4096, max_kv_bytes: kv };
    let judge = TaskBudget { max_input_tokens: 240, max_output_tokens: 8,
        max_output_bytes: 16_384, max_grammar_states: 4096, max_kv_bytes: kv };
    let semantic = SourceSemanticLimits { max_statements: 3, max_source_bytes: 32_768,
        max_claim_bytes: 128, max_total_claim_bytes: 1024, max_total_prompt_tokens: 2048,
        max_total_projected_logits: 1_000_000, max_output_bytes: 131_072 };
    let planning = JudgeLimits { max_total_prompt_tokens: 2048, max_total_candidate_tokens: 256,
        max_total_projected_logits: 1_000_000, max_output_bytes: 16_384, ..JudgeLimits::default() };
    let host = Int4SourceSemanticHostLimits {
        native: NativeLimits { context_tokens: 256, allocator_reserve_bytes: 65_536,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 1000,
                cleanup_reserve_bytes: 65_536 } },
        masks: MaskWorkLimits { max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 },
        max_mask_node_visits: 32_000, preparation_reserve_bytes: 256 << 20,
    };
    let input = SourceSemanticInputFacts { source_bytes: 1024, question_bytes: 0, passage_count: 0 };
    (host, source, judge, semantic, planning, input)
}

#[test]
fn both_stages_require_the_complete_physical_context_and_one_native_workspace() {
    let (host, source, judge, semantic, planning, input) = fixture();
    let required = host.requirements(32, source, judge, semantic, planning, input).unwrap();
    assert_eq!(required.native, Int4MemoryRequirement::for_context(256, 32).unwrap());
    assert_eq!(required.native.kv_bytes, source.max_kv_bytes);
    assert_eq!(required.native_workspace_bytes,
        required.native.rope_bytes + required.native.scratch_payload_bound + host.native.allocator_reserve_bytes);
    for stage in 0..2 {
        let (mut source, mut judge) = (source, judge);
        if stage == 0 { source.max_kv_bytes -= 1; } else { judge.max_kv_bytes -= 1; }
        assert!(host.requirements(32, source, judge, semantic, planning, input).is_err());
    }
    let mut wider = host; wider.native.context_tokens += 1;
    assert!(wider.requirements(32, source, judge, semantic, planning, input).is_err());
    for width in [0, usize::MAX] {
        assert!(host.requirements(width, source, judge, semantic, planning, input).is_err());
    }
}

#[test]
fn all_statement_plans_and_results_coexist_even_with_a_fixed_aggregate_prompt_budget() {
    let (host, source, judge, mut semantic, planning, input) = fixture();
    semantic.max_statements = 1;
    let one = host.requirements(32, source, judge, semantic, planning, input).unwrap();
    semantic.max_statements = 4;
    let four = host.requirements(32, source, judge, semantic, planning, input).unwrap();
    assert_eq!(one.native, four.native);
    assert_eq!(one.native_workspace_bytes, four.native_workspace_bytes);
    assert_eq!(one.output_bytes, four.output_bytes);
    assert!(four.preparation_bytes - one.preparation_bytes >=
        3 * (input.source_bytes as u64 + 4 * judge.max_output_bytes));
    assert!(four.total_bytes().unwrap() > one.total_bytes().unwrap());
}

#[test]
fn complete_evidence_question_and_passage_layout_are_priced_independently_of_output() {
    let (host, source, judge, semantic, planning, input) = fixture();
    let document = host.requirements(32, source, judge, semantic, planning, input).unwrap();
    let passages = SourceSemanticInputFacts { question_bytes: 32, passage_count: 2, ..input };
    let answer = host.requirements(32, source, judge, semantic, planning, passages).unwrap();
    assert_eq!(document.native, answer.native);
    assert_eq!(document.output_bytes, answer.output_bytes);
    assert!(answer.preparation_bytes > document.preparation_bytes);
    let more_evidence = host.requirements(32, source, judge, semantic, planning,
        SourceSemanticInputFacts { source_bytes: 16_384, ..passages }).unwrap();
    assert_eq!(answer.output_bytes, more_evidence.output_bytes);
    assert!(more_evidence.preparation_bytes - answer.preparation_bytes >=
        semantic.max_statements as u64 * (16_384 - 1024));
    let mut wider_judge = judge; wider_judge.max_output_bytes *= 2;
    let mut wider_plans = planning; wider_plans.max_output_bytes *= 2;
    let more_results = host.requirements(32, source, wider_judge, semantic, wider_plans, passages).unwrap();
    assert_eq!(answer.output_bytes, more_results.output_bytes);
    assert!(more_results.preparation_bytes > answer.preparation_bytes);
}

#[test]
fn original_source_and_combined_output_limits_are_independent_simultaneous_charges() {
    let (host, source, judge, semantic, planning, input) = fixture();
    let required = host.requirements(32, source, judge, semantic, planning, input).unwrap();
    assert!(required.preparation_bytes > host.preparation_reserve_bytes);
    assert!(required.output_bytes >= 4 * semantic.max_output_bytes);
    let total = required.total_bytes().unwrap();
    for part in [required.native.kv_bytes, required.native_workspace_bytes,
        required.preparation_bytes, required.output_bytes] { assert!(total > part); }
    for stage in 0..2 {
        let (mut source, mut judge) = (source, judge);
        if stage == 0 { source.max_output_bytes = semantic.max_output_bytes + 1; }
        else { judge.max_output_bytes = semantic.max_output_bytes + 1; }
        assert!(host.requirements(32, source, judge, semantic, planning, input).is_err());
    }
    assert!(Int4SourceSemanticHostRequirements { preparation_bytes: u64::MAX, ..required }.total_bytes().is_err());
}

#[test]
fn malformed_evidence_size_facts_and_each_missing_control_refuse_before_execution() {
    for axis in 0..23 {
        let (mut host, source, judge, mut semantic, mut planning, mut input) = fixture();
        match axis {
            0 => host.preparation_reserve_bytes = 0,
            1 => host.native.allocator_reserve_bytes = 0,
            2 => host.max_mask_node_visits = 0,
            3 => host.masks.max_trie_node_visits = 0,
            4 => host.masks.checkpoint_interval_nodes = 0,
            5 => semantic.max_statements = 0,
            6 => semantic.max_total_prompt_tokens = 0,
            7 => planning.max_total_prompt_tokens = 0,
            8 => planning.max_total_candidate_tokens = 0,
            9 => planning.max_total_projected_logits = 0,
            10 => planning.max_output_bytes = 0,
            11 => planning.per_head.max_candidates = 0,
            12 => planning.per_head.max_total_tokens = 0,
            13 => planning.per_head.max_nodes = 0,
            14 => planning.per_head.max_depth = 0,
            15 => planning.per_head.max_candidate_id_bytes = 0,
            16 => planning.per_head.max_projected_logits = 0,
            17 => input.source_bytes = 0,
            18 => input.source_bytes = semantic.max_source_bytes + 1,
            19 => { input.question_bytes = semantic.max_claim_bytes + 1; input.passage_count = 1; },
            20 => input.passage_count = 1,
            21 => input.question_bytes = 1,
            _ => { input.question_bytes = 1; input.passage_count = 1025; },
        }
        assert!(host.requirements(32, source, judge, semantic, planning, input).is_err(), "axis {axis}");
    }
}

#[test]
fn retained_plan_arithmetic_and_reservation_addition_cannot_wrap() {
    let (host, source, judge, semantic, planning, input) = fixture();
    let mut impossible = host; impossible.preparation_reserve_bytes = u64::MAX;
    assert!(impossible.requirements(32, source, judge, semantic, planning, input).is_err());
    let mut impossible = host; impossible.native.allocator_reserve_bytes = u64::MAX;
    assert!(impossible.requirements(32, source, judge, semantic, planning, input).is_err());
    if usize::BITS >= 64 {
        let mut oversized = planning; oversized.max_total_candidate_tokens = usize::MAX;
        assert!(host.requirements(32, source, judge, semantic, oversized, input).is_err());
        let mut oversized = planning; oversized.per_head.max_candidate_id_bytes = usize::MAX;
        assert!(host.requirements(32, source, judge, semantic, oversized, input).is_err());
    }
}

#[test]
fn cancellation_is_typed_and_private_cause_details_are_not_displayed() {
    let cause = crate::native_engine::decode::DecodeCancellationKind::Deadline;
    let error = Int4HostError::SourceSemantic(Int4SourceSemanticError::Cancelled(cause));
    assert_eq!(error.to_string(), format!("{error:?}"));
    assert_eq!(error.to_string(), "INT4 summary or answer second-reader execution or finalization refused");
    assert!(error.source().is_some());
    let Int4HostError::SourceSemantic(inner) = error else { panic!("wrong cause"); };
    assert_eq!(inner.cancellation(), Some(cause));
}

#[test]
fn sealed_preparation_vocabulary_and_combined_output_can_cross_one_owned_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Input>(); send::<PreparedInt4SourceSemantic>(); send::<Int4SourceSemanticRun>();
    send::<Arc<ExtractionVocabulary>>();
    let _entry = NlpEngine::execute_int4_semantic_source;
}
