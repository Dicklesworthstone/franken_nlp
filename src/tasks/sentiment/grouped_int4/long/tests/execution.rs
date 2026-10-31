use super::*;
use super::super::execution::{Collection, value_bytes};

#[test]
fn complete_map_retains_every_native_chunk_and_one_explicit_document_estimator() {
    let request = source("private-é\r\n上海-abcdefghijklmnop", &[SentimentAxis::Valence, SentimentAxis::Arousal]);
    let prepared = prepare(&request, map_limits()).unwrap();
    let output = finish(&prepared).unwrap();
    prepared.preflight_metadata().verify_completed(&output).unwrap();
    assert_eq!(output.chunks.len(), prepared.chunk_count());
    assert_eq!(output.model_work, prepared.planned_work());
    assert_eq!(output.planned_model_work, output.model_work);
    assert!(!output.single_context_equivalence);
    assert_eq!(output.aggregate.weight_unit, "original_unicode_scalar_count");
    assert_eq!(output.aggregate.interpretation, "descriptive_chunk_candidate_mixture_not_full_document_inference");
    assert_eq!(output.aggregate.calibration, "uncalibrated_not_a_psychological_measurement");
    assert_eq!(output.aggregate.dimensions.len(), 2);
    let independently_reduced = aggregate_with_control(output.chunks.iter().map(|chunk| SentimentChunkView {
        chunk_id: chunk.chunk_id, span: chunk.source_span, result: &chunk.run.result }), output.chunks.len(),
        request.document.len(), request.document.chars().count(), map_limits().aggregation, &mut Continue).unwrap();
    assert_eq!(output.aggregate, independently_reduced);
    let bytes = canonjson::canonical_bytes(&output).unwrap();
    assert!(!std::str::from_utf8(&bytes).unwrap().contains(&request.document));
    for dimension in &output.aggregate.dimensions {
        assert_eq!(dimension.estimated_scalars, request.document.chars().count());
        assert_eq!(dimension.abstained_scalars, 0);
        assert!(dimension.coverage_passed && dimension.mixture_policy_passed);
        assert!((dimension.between_chunk_variance).abs() < 1e-12);
        assert!((dimension.within_chunk_variance - 0.5).abs() < 1e-12);
        assert_eq!(dimension.mixture_weights.len(), 5);
    }
}

#[test]
fn missing_reordered_or_corrupt_native_runs_never_yield_a_partial_map() {
    let request = source("0123456789abcdefghijklmn", &[SentimentAxis::Valence]);
    let prepared = prepare(&request, map_limits()).unwrap();
    let mut missing = Collection::new(&prepared).unwrap();
    missing.push(0, run(&prepared.plans[0]), &mut Continue).unwrap();
    assert!(matches!(missing.finish(&mut Continue), Err(Int4SentimentMapError::Admission)));
    let mut reordered = Collection::new(&prepared).unwrap();
    assert!(reordered.push(1, run(&prepared.plans[1]), &mut Continue).is_err());
    assert!(reordered.push(0, run(&prepared.plans[0]), &mut Continue).is_err());
    assert!(reordered.finish(&mut Continue).is_err());
    for changed in 0..5 {
        let mut collected = Collection::new(&prepared).unwrap();
        let mut result = run(&prepared.plans[0]);
        match changed { 0 => result.execution = "foreign".to_owned(),
            1 => result.group_width = 16, 2 => result.model_work.projected_logits += 1,
            3 => result.head_count += 1, _ => result.result.dimensions.clear() }
        assert!(collected.push(0, result, &mut Continue).is_err());
        assert!(collected.finish(&mut Continue).is_err());
    }
}

#[test]
fn a_late_cancelled_chunk_poisons_the_complete_collection_and_retains_its_cause() {
    let request = source("0123456789abcdef", &[SentimentAxis::Valence]);
    let prepared = prepare(&request, map_limits()).unwrap();
    let mut collected = Collection::new(&prepared).unwrap();
    collected.push(0, run(&prepared.plans[0]), &mut Continue).unwrap();
    let error = collected.push(1, run(&prepared.plans[1]), &mut Stop(DecodeCancellationKind::Deadline)).unwrap_err();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    assert!(collected.push(1, run(&prepared.plans[1]), &mut Continue).is_err());
    assert!(collected.finish(&mut Continue).is_err());
}

#[test]
fn retained_frontier_and_final_envelope_have_separate_nonrenewable_caps() {
    let request = source("0123456789abcdefghijklmn", &[SentimentAxis::Valence]);
    let prepared = prepare(&request, map_limits()).unwrap(); let output = finish(&prepared).unwrap();
    let first = value_bytes(&output.chunks[0], MAX_BYTES).unwrap();
    let mut limits = map_limits(); limits.max_retained_result_bytes = first;
    let limited = prepare(&request, limits).unwrap();
    let mut collected = Collection::new(&limited).unwrap();
    collected.push(0, run(&limited.plans[0]), &mut Continue).unwrap();
    assert!(matches!(collected.push(1, run(&limited.plans[1]), &mut Continue), Err(Int4SentimentMapError::Output)));
    assert!(collected.finish(&mut Continue).is_err());
    limits = map_limits(); limits.max_result_bytes = 1;
    let limited = prepare(&request, limits).unwrap();
    assert!(matches!(finish(&limited), Err(Int4SentimentMapError::Output)));
    assert!(matches!(value_bytes(&output, 1), Err(Int4SentimentMapError::Output)));
}

#[test]
fn completed_metadata_refuses_coverage_geometry_and_accounting_tampering() {
    let request = source("0123456789abcdef", &[SentimentAxis::Valence]);
    let prepared = prepare(&request, map_limits()).unwrap(); let original = finish(&prepared).unwrap();
    let expected = prepared.preflight_metadata(); expected.verify_completed(&original).unwrap();
    for field in 0..11 {
        let mut changed = original.clone();
        match field { 0 => { changed.chunks.pop(); }, 1 => changed.chunks[1].chunk_id = 0,
            2 => changed.chunks[1].source_span.byte_start += 1,
            3 => changed.chunks[0].source_span.scalar_end += 1,
            4 => changed.model_work.projected_logits += 1,
            5 => changed.retained_result_bytes += 1,
            6 => changed.chunk_limits.reserved_tokens += 1,
            7 => changed.tokenizer_checks += 1, 8 => changed.aggregate.source_bytes += 1,
            9 => changed.group_width = 16, _ => changed.single_context_equivalence = true }
        assert!(expected.verify_completed(&changed).is_err(), "changed field {field}");
    }
}

#[test]
fn every_future_cohort_window_is_admitted_before_execution_can_start() {
    let request = source("0123456789abcdefghijklmn", &[SentimentAxis::Valence, SentimentAxis::Arousal]);
    let mut prepared = prepare(&request, map_limits()).unwrap();
    let mut identities = prepared.identities().unwrap();
    let budget = cohort_budget(&prepared);
    let windows = prepared.cohort_windows(&identities, 1, budget, &mut Continue).unwrap();
    assert_eq!(windows.len(), prepared.chunk_count());
    assert!(windows.iter().all(|window| window.contexts.len() == 2));
    let last = identities.len() - 1;
    identities[last].prompt_digest = Sha256Digest::of_bytes(b"late-foreign-admission");
    assert!(prepared.cohort_windows(&identities, 1, budget, &mut Continue).is_err());
    identities = prepared.identities().unwrap();
    // Private mutation simulates a late plan whose per-document authority is
    // insufficient. All earlier windows still fit, but no engine is started.
    prepared.plans[last].budget.max_kv_bytes = 1;
    assert!(prepared.cohort_windows(&identities, 1, budget, &mut Continue).is_err());
}

#[test]
fn exact_cohort_preflight_prices_all_native_axes_and_transient_task_workspace() {
    let request = source("0123456789abcdefghijklmn", &[SentimentAxis::Valence, SentimentAxis::Arousal]);
    let prepared = prepare(&request, map_limits()).unwrap(); let identities = prepared.identities().unwrap();
    let generous = cohort_budget(&prepared);
    prepared.cohort_windows(&identities, 2, generous, &mut Continue).unwrap();
    for axis in 0..5 {
        let mut budget = generous;
        match axis { 0 => budget.memory.max_kv_bytes = 1, 1 => budget.memory.max_rope_bytes = 1,
            2 => budget.memory.max_scratch_payload_bytes = 1,
            3 => budget.max_task_workspace_bytes = 1, _ => budget.native.max_forward_positions -= 1 }
        assert!(prepared.cohort_windows(&identities, 2, budget, &mut Continue).is_err());
    }
    let error = prepared.cohort_windows(&identities, 1, generous, &mut Stop(DecodeCancellationKind::PollQuota))
        .err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::PollQuota));
}

#[test]
fn declared_cohort_geometry_never_silently_splits_excess_heads() {
    let request = source("0123456789abcdef", &SentimentAxis::ALL);
    let prepared = prepare(&request, map_limits()).unwrap(); let identities = prepared.identities().unwrap();
    assert!(prepared.cohort_windows(&identities, 0, cohort_budget(&prepared), &mut Continue).is_err());
    assert!(prepared.cohort_windows(&identities, 17, cohort_budget(&prepared), &mut Continue).is_err());
    prepared.cohort_windows(&identities, 16, cohort_budget(&prepared), &mut Continue).unwrap();
    let request = source("0123456789abcdef", &[SentimentAxis::Valence]);
    let prepared = prepare(&request, map_limits()).unwrap(); let identities = prepared.identities().unwrap();
    prepared.cohort_windows(&identities, 64, cohort_budget(&prepared), &mut Continue).unwrap();
    assert!(prepared.cohort_windows(&identities, 65, cohort_budget(&prepared), &mut Continue).is_err());
}

#[test]
fn serial_and_closed_document_cohort_finalizers_produce_the_same_complete_map() {
    let request = source("0123456789abcdefghijklmn", &[SentimentAxis::Approach, SentimentAxis::Arousal, SentimentAxis::Valence]);
    let prepared = prepare(&request, map_limits()).unwrap(); let identities = prepared.identities().unwrap();
    let serial = finish(&prepared).unwrap();
    let budget = cohort_budget(&prepared);
    let windows = prepared.cohort_windows(&identities, 2, budget, &mut Continue).unwrap();
    assert!(windows.len() > 1);
    let mut collected = Collection::new(&prepared).unwrap();
    for window in windows {
        let documents: Vec<_> = (window.start..window.end).map(|index| (&prepared.plans[index], &identities[index])).collect();
        let native_budget = Int4ScoringCohortBudget { native: Int4RunBudget::exact(window.work),
            max_kv_bytes: budget.memory.max_kv_bytes, max_output_bytes: budget.max_output_bytes };
        let (schedules, work) = task::document_layout(&documents, &window.contexts, native_budget).unwrap();
        let native = fixtures::run(documents.iter().flat_map(|(plan, _)| plan.inner.heads.iter()
            .zip(&plan.schedules).map(move |(head, &schedule)|
                (head.scorer.score(&mut fixtures::Constant, plan.inner.options.mode).unwrap(), schedule))));
        let finalized = task::finish_documents(&documents, native, &schedules, work,
            budget.max_output_bytes, &mut Continue).unwrap();
        for (index, output) in (window.start..window.end).zip(finalized.output) {
            collected.push(index, output, &mut Continue).unwrap();
        }
    }
    let grouped = collected.finish(&mut Continue).unwrap();
    assert_eq!(canonjson::canonical_bytes(&serial).unwrap(), canonjson::canonical_bytes(&grouped).unwrap());
    assert!(prepared.plans.iter().all(|plan| plan.task_budget() == request.budget));
    prepared.preflight_metadata().verify_completed(&grouped).unwrap();
}
