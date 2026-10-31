//! Pinned planning and private phase fixtures; no native-model or throughput claims.
use super::*;
use super::super::tests as fixtures;
use crate::{native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, strict_int4::Int4MemoryBudget},
    tasks::source_planning::grouped_int4::long::tests as maps};

const SOURCE: &str = "ééé                        ééé";
fn budget(prepared: &PreparedInt4QuestionSynthesis<'_>) -> Int4SourceMapCohortBudget {
    let context_limit = prepared.planning.max_context_tokens;
    let need = prepared.cohort_requirements(context_limit).unwrap();
    Int4SourceMapCohortBudget { context_limit, native_memory: Int4MemoryBudget {
        max_kv_bytes: need.native.kv_bytes, max_rope_bytes: need.native.rope_bytes,
        max_scratch_payload_bytes: need.native.scratch_payload_bound },
        max_workspace_bytes: need.workspace_bytes, max_result_bytes: need.max_result_bytes }
}
fn stats(discovery: &Int4QuestionRun) -> Int4SourceMapCohortStats {
    // These fixtures use one source chunk per window; whitespace has no row.
    Int4SourceMapCohortStats { cohorts: discovery.native_chunks as u64, chunks: discovery.native_chunks as u64,
        group_steps: discovery.model_work.forward_positions,
        projection_groups: discovery.model_work.projected_logits / NANBEIGE_VOCAB_SIZE as u64,
        planned_work: discovery.planned_model_work, model_work: discovery.model_work }
}

#[test]
fn whole_scalar_context_is_admitted_before_any_discovery_even_for_sparse_windows() {
    let planner = maps::planner();
    for width in [1, 2, 6] {
        let mut mapping = fixtures::mapping(); mapping.reduction.map_batch_chunks = width;
        let mut prepared = fixtures::prepare(&planner, SOURCE, &fixtures::request(), mapping, 32, &mut fixtures::continuing()).unwrap();
        let context = prepared.planning.max_context_tokens;
        let discovery = prepared.map.cohort_requirements(context).unwrap();
        let scalar = Int4MemoryRequirement::for_context(context, 32).unwrap();
        let need = prepared.cohort_requirements(context).unwrap();
        assert_eq!(need.native.kv_bytes, discovery.native.kv_bytes.max(scalar.kv_bytes));
        assert_eq!(need.native.rope_bytes, discovery.native.rope_bytes.max(scalar.rope_bytes));
        assert_eq!(need.native.scratch_payload_bound, discovery.native.scratch_payload_bound.max(scalar.scratch_payload_bound));
        assert_eq!(need.workspace_bytes, discovery.workspace_bytes);
        assert_eq!(need.max_result_bytes, discovery.max_result_bytes);
        assert!(prepared.cohort_requirements(context - 1).is_err());
        let exact = budget(&prepared);
        prepared.preflight_cohorts(&planner, exact, &mut fixtures::continuing()).unwrap();
        for axis in 0..5 {
            let mut short = exact;
            match axis { 0 => short.native_memory.max_kv_bytes -= 1,
                1 => short.native_memory.max_rope_bytes -= 1,
                2 => short.native_memory.max_scratch_payload_bytes -= 1,
                3 => short.max_workspace_bytes -= 1, _ => short.max_result_bytes -= 1 }
            assert!(prepared.preflight_cohorts(&planner, short, &mut fixtures::continuing()).is_err(), "{axis}");
        }
        prepared.budget.max_kv_bytes = scalar.kv_bytes - 1;
        assert!(prepared.cohort_requirements(context).is_err());
    }
}

#[test]
fn shared_continuation_preserves_answer_abstention_and_zero_allocation_no_evidence_semantics() {
    let planner = maps::planner();
    for mode in 0..3 {
        let prepared = fixtures::prepare(&planner, SOURCE, &fixtures::request(), fixtures::mapping(), 32,
            &mut fixtures::continuing()).unwrap();
        let expected = prepared.preflight_metadata();
        let (_, continuation) = prepared.into_phases();
        let (_, original) = fixtures::synthetic(mode);
        let original_json = serde_json::to_value(&original).unwrap();
        let mut calls = 0;
        let document = continuation.execute(original.discovery, &planner, &mut fixtures::continuing(), |plan, _| {
            calls += 1;
            let raw = fixtures::raw(plan, &[("evidence_0", "é"), ("evidence_1", "é")],
                (mode != 2).then_some("é"), "final answer");
            let model_work = constrained_int4::planned_work(32, plan.prompt_tokens(), raw.generated_token_ids.len()).unwrap();
            Ok(Int4SourceTaskRun { schema_version: 1, execution: INT4_SOURCE_EXECUTION.to_owned(), group_width: 32,
                result: SourceTaskResult::Answer(raw), model_work })
        }).unwrap();
        assert_eq!(calls, usize::from(mode != 1));
        expected.verify_completed(&document).unwrap();
        assert_eq!(serde_json::to_value(&document).unwrap(), original_json);
        let ledger = stats(&document.discovery);
        let run = finish(document, ledger, 16 << 20, &mut fixtures::continuing()).unwrap();
        assert_eq!(serde_json::to_value(&run.document).unwrap(), original_json);
        assert_eq!(run.discovery_cohorts.model_work, run.document.discovery.model_work);
        assert!(run.discovery_cohorts.chunks < run.document.discovery.mapped.root().value().chunks().len() as u64);
        assert!(!canonjson::canonical_string(&run).unwrap().contains("generated_token_ids"));
        if mode != 1 { assert_ne!(run.discovery_cohorts.model_work, run.document.model_work); }
    }
}

#[test]
fn wrapper_rejects_final_pass_work_or_whitespace_charged_as_discovery() {
    for axis in 0..3 {
        let (_, document) = fixtures::synthetic(0); let mut ledger = stats(&document.discovery);
        match axis { 0 => ledger.model_work = document.model_work,
            1 => ledger.planned_work = document.reserved_model_work,
            _ => ledger.chunks += document.discovery.whitespace_chunks as u64 }
        assert!(finish(document, ledger, 16 << 20, &mut fixtures::continuing()).is_err());
    }
}
