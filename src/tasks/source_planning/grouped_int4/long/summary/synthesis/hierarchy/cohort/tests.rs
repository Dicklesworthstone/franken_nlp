//! Source-only admission and shared hierarchy continuation fixtures.
use super::*;
use super::super::tests as hierarchy;
use crate::tasks::source_planning::grouped_int4::long::tests as maps;

#[test]
fn full_scalar_phase_is_admitted_without_renewing_any_hierarchy_pass() {
    let planner = maps::planner();
    for width in [16, 32, 127] { for batch in [1, 2, 4] {
        let mut mapping = maps::limits(); mapping.reduction.map_batch_chunks = batch;
        let prepared = hierarchy::prepare(&planner, "éabc----βdef", mapping, hierarchy::limits(), width).unwrap();
        let context = prepared.base.planning.max_context_tokens; let expected = prepared.preflight_metadata();
        let map = prepared.base.map.cohort_requirements(context).unwrap();
        let scalar = Int4MemoryRequirement::for_context(context, width).unwrap();
        let peak = prepared.cohort_requirements(context).unwrap();
        assert_eq!(peak.native.kv_bytes, map.native.kv_bytes.max(scalar.kv_bytes));
        assert_eq!(peak.native.rope_bytes, map.native.rope_bytes.max(scalar.rope_bytes));
        assert_eq!(peak.native.scratch_payload_bound, map.native.scratch_payload_bound.max(scalar.scratch_payload_bound));
        assert_eq!(peak.workspace_bytes, map.workspace_bytes); assert_eq!(peak.max_result_bytes, map.max_result_bytes);
        assert_eq!(prepared.preflight_metadata(), expected);
        let mut work = prepared.base.map.planned_work();
        for _ in 0..expected.limits.max_passes { work = add_work(work, expected.discovery.synthesis_reserve).unwrap(); }
        assert_eq!(expected.reserved_model_work(), work); assert_eq!(expected.reserved_mask_visits(), 6000);
    } }
}

#[test]
fn hierarchy_cannot_defer_context_or_task_kv_refusal_until_after_discovery() {
    let planner = maps::planner();
    let mut prepared = hierarchy::prepare(&planner, "source", maps::limits(), hierarchy::limits(), 32).unwrap();
    let context = prepared.base.planning.max_context_tokens;
    assert!(prepared.base.map.cohort_requirements(context - 1).is_ok());
    assert!(matches!(prepared.cohort_requirements(context - 1), Err(Int4SourceMapError::Admission)));
    prepared.base.budget.max_kv_bytes = Int4MemoryRequirement::for_context(context, 32).unwrap().kv_bytes - 1;
    assert!(prepared.base.map.cohort_requirements(context).is_ok());
    assert!(matches!(prepared.cohort_requirements(context), Err(Int4SourceMapError::Admission)));
}

#[test]
fn empty_discovery_preserves_native_transcripts_without_allocating_a_reduction_engine() {
    let (tail, discovery) = hierarchy::discovered(true); let expected = tail.expected;
    let run = tail.execute(discovery, |_, _, _, _| panic!("empty evidence reached scalar allocation"), &mut maps::Continue).unwrap();
    expected.verify_completed(&run).unwrap();
    assert_eq!(run.status, SummarySynthesisStatus::NoEvidenceCollected); assert_eq!(run.initial_evidence.segments, 0);
    assert!(run.passes.is_empty()); assert!(run.levels.is_empty()); assert!(run.final_pass.is_none());
    assert_eq!(run.model_work, run.discovery.model_work);
    for chunk in run.discovery.mapped.root().value().chunks() {
        let SourceTaskResult::Summarize(raw) = &chunk.native.result else { panic!() };
        assert_eq!(raw.generated_token_ids.len(), 2);
    }
}

#[test]
fn nonempty_tail_passes_only_verified_quotes_and_one_remaining_verification_budget() {
    let (tail, discovery) = hierarchy::discovered(false);
    let map_used = discovery.verification_used; let total = tail.base.request.limits.verification;
    let mut calls = 0;
    let result = tail.execute(discovery, |_, collection, remaining, _| {
        calls += 1;
        assert_eq!(collection.text, "éabc\n\nβdef");
        assert!(remaining.max_fields < total.max_fields - map_used.fields);
        assert!(remaining.max_matches < total.max_matches - map_used.matches);
        assert!(remaining.max_scan_steps < total.max_scan_steps - map_used.scan_steps);
        Err(Int4SourceError::Cancelled(DecodeCancellationKind::Deadline).into())
    }, &mut maps::Continue);
    assert_eq!(calls, 1);
    assert_eq!(result.err().unwrap().cancellation(), Some(DecodeCancellationKind::Deadline));
}
