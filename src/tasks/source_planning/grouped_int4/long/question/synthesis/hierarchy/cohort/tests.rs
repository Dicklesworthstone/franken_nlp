//! Pinned admission and private phase fixtures, not native-model parity evidence.
use super::*;
use super::super::tests as fixtures;
use crate::{native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, strict_int4::Int4MemoryBudget},
    tasks::source_planning::grouped_int4::long::tests as maps};

fn stats(discovery: &Int4QuestionRun) -> Int4SourceMapCohortStats {
    Int4SourceMapCohortStats { cohorts: discovery.native_chunks as u64, chunks: discovery.native_chunks as u64,
        group_steps: discovery.model_work.forward_positions,
        projection_groups: discovery.model_work.projected_logits / NANBEIGE_VOCAB_SIZE as u64,
        planned_work: discovery.planned_model_work, model_work: discovery.model_work }
}

#[test]
fn full_hierarchy_context_and_every_peak_component_are_admitted_before_discovery() {
    let planner = maps::planner();
    let mut prepared = fixtures::prepare(&planner, fixtures::SOURCE, maps::limits(), fixtures::limits(), 32, &mut fixtures::go()).unwrap();
    let context = prepared.base.planning.max_context_tokens;
    let discovery = prepared.base.map.cohort_requirements(context).unwrap();
    let scalar = Int4MemoryRequirement::for_context(context, 32).unwrap();
    let need = prepared.cohort_requirements(context).unwrap();
    assert_eq!(need.native.kv_bytes, discovery.native.kv_bytes.max(scalar.kv_bytes));
    assert_eq!(need.native.rope_bytes, discovery.native.rope_bytes.max(scalar.rope_bytes));
    assert_eq!(need.native.scratch_payload_bound, discovery.native.scratch_payload_bound.max(scalar.scratch_payload_bound));
    assert_eq!(need.workspace_bytes, discovery.workspace_bytes);
    assert!(prepared.cohort_requirements(context - 1).is_err());
    let exact = Int4SourceMapCohortBudget { context_limit: context, native_memory: Int4MemoryBudget {
        max_kv_bytes: need.native.kv_bytes, max_rope_bytes: need.native.rope_bytes,
        max_scratch_payload_bytes: need.native.scratch_payload_bound },
        max_workspace_bytes: need.workspace_bytes, max_result_bytes: need.max_result_bytes };
    prepared.preflight_cohorts(&planner, exact, &mut fixtures::go()).unwrap();
    for axis in 0..5 {
        let mut short = exact;
        match axis { 0 => short.native_memory.max_kv_bytes -= 1, 1 => short.native_memory.max_rope_bytes -= 1,
            2 => short.native_memory.max_scratch_payload_bytes -= 1,
            3 => short.max_workspace_bytes -= 1, _ => short.max_result_bytes -= 1 }
        assert!(prepared.preflight_cohorts(&planner, short, &mut fixtures::go()).is_err(), "{axis}");
    }
    prepared.base.budget.max_kv_bytes = scalar.kv_bytes - 1;
    assert!(prepared.cohort_requirements(context).is_err());
}

#[test]
fn common_hierarchy_continuation_preserves_all_passes_and_skips_scalar_creation_without_evidence() {
    let planner = maps::planner();
    for mode in [0, 1] {
        let (expected, original) = fixtures::complete(mode);
        let original_json = serde_json::to_value(&original).unwrap();
        let mut prepared = fixtures::prepare(&planner, fixtures::SOURCE, maps::limits(), fixtures::limits(), 32, &mut fixtures::go()).unwrap();
        prepared.expected = expected; // Same narrow grouping seam as the original fixture.
        let (_, continuation) = prepared.into_phases();
        let mut calls = 0;
        let document = continuation.execute(original.discovery, &mut fixtures::go(), |continuation, collection, remaining, control| {
            calls += 1;
            execution::reduce(continuation.base.source, collection, expected, 16 << 20,
                remaining, &mut fixtures::script(expected), control)
        }).unwrap();
        assert_eq!(calls, usize::from(mode == 0));
        expected.verify_completed(&document).unwrap();
        assert_eq!(serde_json::to_value(&document).unwrap(), original_json);
        let ledger = stats(&document.discovery);
        let run = finish(document, ledger, 16 << 20, &mut fixtures::go()).unwrap();
        assert_eq!(serde_json::to_value(&run.document).unwrap(), original_json);
        assert_eq!(run.discovery_cohorts.model_work, run.document.discovery.model_work);
        if mode == 0 {
            assert_eq!(run.document.passes.len(), 3);
            assert!(canonjson::canonical_string(&run).unwrap().contains("generated_token_ids"));
            assert_ne!(run.discovery_cohorts.model_work, run.document.model_work);
        }
    }
}

#[test]
fn wrapper_keeps_exhaustion_and_final_abstention_separate_from_the_discovery_ledger() {
    for mode in [2, 3] {
        let (expected, document) = fixtures::complete(mode);
        let ledger = stats(&document.discovery);
        let run = finish(document, ledger, 16 << 20, &mut fixtures::go()).unwrap();
        expected.verify_completed(&run.document).unwrap();
        assert_ne!(run.discovery_cohorts.model_work, run.document.model_work);
        assert_eq!(run.document.status, if mode == 2 { QuestionHierarchyStatus::EvidenceExhausted } else { QuestionHierarchyStatus::Abstained });
        let mut wrong = run.discovery_cohorts; wrong.model_work = run.document.model_work;
        assert!(finish(run.document, wrong, 16 << 20, &mut fixtures::go()).is_err());
    }
}
