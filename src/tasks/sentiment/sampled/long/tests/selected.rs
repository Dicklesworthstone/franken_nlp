//! Complete-source selected projection preserves original semantic addresses.
use super::*;

fn selected(rows: usize) -> SampledSentimentPlanner {
    let compiler = Compiler::pinned(&options(), budget(), SampledSentimentLimits::default()).unwrap();
    let identity = unbound_identity(&compiler);
    SampledSentimentPlanner { compiler, identity, group_width: 32, options: options(),
        ceiling: budget(), limits: SampledSentimentLimits::default(),
        selected_rows: None, admission_identity: None }
        .with_selected_rows(Int4JsonSparseLimits { max_rows_per_step: rows }).unwrap()
}

#[test]
fn whole_document_item_and_original_chunk_draw_addresses_do_not_include_projection_policy() {
    let source = request("é\r\n好 test é\r\n好 test");
    for item in [None, Some("document-42")] {
        let prepare = |planner: &SampledSentimentPlanner| match item {
            None => planner.plan_document_with_control(&source, limits(), &mut Continue),
            Some(id) => planner.plan_document_for_item_with_control(&source, limits(), id, &mut Continue),
        }.unwrap();
        let dense = prepare(planner());
        for cap in [7, 97, NANBEIGE_VOCAB_SIZE] {
            let sparse = prepare(&selected(cap));
            assert_eq!(dense.chunk_count(), sparse.chunk_count());
            assert_eq!(dense.expected.geometry, sparse.expected.geometry);
            assert_ne!(dense.expected.request_binding, sparse.expected.request_binding);
            assert_eq!(dense.preparation_requirement_bytes(), sparse.preparation_requirement_bytes());
            assert_eq!(sparse.selected_rows(), Some(Int4JsonSparseLimits { max_rows_per_step: cap }));
            for (a, b) in dense.chunks.iter().zip(&sparse.chunks) {
                assert_eq!((a.id, a.span, a.tokens), (b.id, b.span, b.tokens));
                assert_ne!(a.plan.execution_identity(), b.plan.execution_identity());
                for (a, b) in a.plan.axes.iter().zip(&b.plan.axes) {
                    assert_eq!(a.prompt, b.prompt);
                    assert_eq!(canonjson::canonical_bytes(&a.identity).unwrap(),
                        canonjson::canonical_bytes(&b.identity).unwrap());
                }
            }
            assert!(fits_work(sparse.planned_work(), dense.planned_work()));
            assert_eq!(sparse.planned_work().projected_logits,
                sparse.sample_count() as u64 * options().max_sample_tokens as u64 * cap as u64);
        }
    }
}

#[test]
fn selected_complete_map_seals_refuse_dense_or_different_cap_results_even_at_full_width() {
    let source = request("abcdefghabcdefgh");
    let dense = planner().plan_document_with_control(&source, limits(), &mut Continue).unwrap();
    let a = selected(NANBEIGE_VOCAB_SIZE).plan_document_with_control(&source, limits(), &mut Continue).unwrap();
    let b = selected(97).plan_document_with_control(&source, limits(), &mut Continue).unwrap();
    let dense_result = fixture_run(&dense); let selected_result = fixture_run(&a); let other_result = fixture_run(&b);
    a.expected.verify_completed_with_control(&selected_result, false, &mut Continue).unwrap();
    b.expected.verify_completed_with_control(&other_result, false, &mut Continue).unwrap();
    assert_eq!(a.planned_work(), dense.planned_work());
    assert_eq!(selected_result.execution, SAMPLED_SENTIMENT_MAP_SPARSE_EXECUTION);
    assert!(a.expected.verify_completed_with_control(&dense_result, false, &mut Continue).is_err());
    assert!(a.expected.verify_completed_with_control(&other_result, false, &mut Continue).is_err());
    assert!(dense.expected.verify_completed_with_control(&selected_result, false, &mut Continue).is_err());
    let mut changed = selected_result.clone();
    changed.chunks[0].run.execution = SAMPLED_SENTIMENT_EXECUTION.to_owned();
    changed.result_binding = binding::result(&changed, a.max_result_bytes(), &mut Continue).unwrap();
    assert!(a.expected.verify_completed_with_control(&changed, false, &mut Continue).is_err());
    assert!(a.expected.verify_completed_with_control(&selected_result, true, &mut Continue).is_err());
}
