//! Selected-row document policy preserves original sample addresses while
//! changing exact native admission and completion authority.
use super::*;

fn selected(cap: usize) -> Int8SampledSentimentPlanner {
    Int8SampledSentimentPlanner::for_model(&facts(), options(), budget(),
        SampledSentimentLimits::default()).unwrap()
        .with_selected_rows(Int8JsonSparseLimits { max_rows_per_step: cap }).unwrap()
}

#[test]
fn selected_documents_preserve_semantic_draw_addresses_and_complete_source_geometry() {
    let source = request("abcdefghé\r\n好 test");
    let dense = planner().plan_document_for_item_with_control(&source, limits(), "item-é", &mut Continue).unwrap();
    let sparse = selected(32).plan_document_for_item_with_control(&source, limits(), "item-é", &mut Continue).unwrap();
    assert_eq!(dense.chunk_count(), sparse.chunk_count());
    assert_eq!(dense.expected.source_span, sparse.expected.source_span);
    assert_eq!(dense.expected.geometry, sparse.expected.geometry);
    assert_ne!(dense.expected.request_binding, sparse.expected.request_binding);
    assert_eq!(dense.preparation_requirement_bytes(), sparse.preparation_requirement_bytes());
    assert_eq!(dense.planned_work().forward_positions, sparse.planned_work().forward_positions);
    assert_eq!(dense.planned_work().attention_pairs, sparse.planned_work().attention_pairs);
    assert!(sparse.planned_work().projections.dot_products < dense.planned_work().projections.dot_products);
    assert!(sparse.planned_work().projections.multiply_accumulates < dense.planned_work().projections.multiply_accumulates);
    assert_eq!(sparse.planned_work().projected_logits, sparse.max_output_tokens() * 32);
    for (cold, selected) in dense.chunks.iter().zip(&sparse.chunks) {
        assert_eq!((cold.id, cold.span, cold.tokens), (selected.id, selected.span, selected.tokens));
        assert_ne!(cold.plan.execution_identity(), selected.plan.execution_identity());
        for (a, b) in cold.plan.inner.axes.iter().zip(&selected.plan.inner.axes) {
            assert_eq!(a.prompt, b.prompt);
            assert_eq!(a.identity, b.identity, "physical projection policy must not reseed a chunk");
        }
    }
}

#[test]
fn sparse_complete_document_accounts_actual_rows_and_keeps_every_replicate() {
    let sparse = selected(32).plan_document_with_control(&request("abcdefghé"), limits(), &mut Continue).unwrap();
    let dense = prepare("abcdefghé");
    let run = execute(&sparse);
    assert_eq!(run.execution, INT8_SAMPLED_SENTIMENT_MAP_SPARSE_EXECUTION);
    assert_eq!(run.model_work.projected_logits, run.generated_tokens * 2);
    assert_eq!(run.planned_model_work.projected_logits, sparse.max_output_tokens() * 32);
    assert_eq!(run.planned_mask_node_visit_charge, sparse.max_output_tokens() * MASK_CHARGE);
    assert_eq!(run.aggregate, execute(&dense).aggregate);
    assert_eq!(run.chunks.len(), sparse.chunk_count());
    assert_eq!(sparse.selected_rows(), Some(Int8JsonSparseLimits { max_rows_per_step: 32 }));
    sparse.preflight_metadata().verify_completed_with_control(&run, &mut Continue).unwrap();
    assert!(dense.preflight_metadata().verify_completed_with_control(&run, &mut Continue).is_err());
    let different_cap = selected(33).plan_document_with_control(&request("abcdefghé"), limits(), &mut Continue).unwrap();
    assert!(different_cap.preflight_metadata().verify_completed_with_control(&run, &mut Continue).is_err());
}

#[test]
fn selected_head_budget_admits_without_full_projection_and_rejects_a_short_cap_before_callbacks() {
    let map = selected(32).plan_document_with_control(&request("abcdefghabcdefgh"), limits(), &mut Continue).unwrap();
    let mut capped = limits(); capped.max_model_work = map.planned_work();
    assert!(selected(32).plan_document_with_control(&request("abcdefghabcdefgh"), capped, &mut Continue).is_ok());
    assert!(matches!(planner().plan_document_with_control(&request("abcdefghabcdefgh"), capped, &mut Continue),
        Err(Int8SampledSentimentMapError::Work)));
    let mut short = native_budget(&map); short.json.max_projected_logits -= 1;
    let mut calls = 0;
    assert!(map.execute_chunks(MASK_CHARGE, short, sampler_bytes(), &mut Continue, |plan, _, _| {
        calls += 1; Ok(synthetic(plan, &[-20, 20]))
    }).is_err());
    assert_eq!(calls, 0);
}

#[test]
fn selected_document_rejects_mixed_profiles_and_forged_selected_work() {
    let map = selected(32).plan_document_with_control(&request("abcdefghabcdefgh"), limits(), &mut Continue).unwrap();
    for fault in 0..3 {
        let mut visits = 0;
        let result = map.execute_chunks(MASK_CHARGE, native_budget(&map), sampler_bytes(), &mut Continue,
            |plan, _, _| {
                visits += 1;
                let mut run = synthetic(plan, &[-20, 20]);
                if visits == 2 {
                    match fault {
                        0 => run.execution = INT8_SAMPLED_SENTIMENT_EXECUTION.to_owned(),
                        1 => run.dimensions[0].samples[0].model_work.projected_logits = 0,
                        _ => run.dimensions[0].samples[0].model_work.projections.multiply_accumulates -= 1,
                    }
                }
                Ok(run)
            });
        assert!(result.is_err(), "fault {fault}");
        assert_eq!(visits, 2);
    }
}
