//! Sealed question planning and hosted resource quotes without model loading.
//! Whitespace belongs to the logical frontier, never to a fabricated native row.
use super::*;
use super::super::tests::fixture;
use crate::{native_engine::{decode::{DecodeCancellationKind, DecodeStepControl}, strict_int4::Int4Work},
    tasks::source_planning::grouped_int4::long::question::{PreparedInt4Question,
        cohort::Int4QuestionCohortRun}};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn planning_fixture(width: usize) -> (SourceTaskPlanner, Int4SourceMapConfig<SourceQuestion>) {
    let (planner, mut config) = fixture(width);
    config.budget.max_output_tokens = 64;
    config.mapping.chunks.max_chunk_bytes = 6;
    config.mapping.chunks.max_chunk_tokens = 6;
    config.mapping.chunks.reserved_tokens = 64;
    config.mapping.mask_visits_per_chunk = 1_000_000_000;
    config.mapping.max_mask_visits = config.mapping.mask_visits_per_chunk * config.mapping.chunks.max_chunks as u64;
    // Bound the complete allowed population using independent full-context
    // rows, rather than giving a fixture the first row's renewed allowance.
    let row = Int4Work::for_sequence(width, 0, config.native.context_tokens,
        config.budget.max_output_tokens as usize * NANBEIGE_VOCAB_SIZE).unwrap();
    config.mapping.max_model_work = Int4Work::default();
    for _ in 0..config.mapping.chunks.max_chunks {
        config.mapping.max_model_work = config.mapping.max_model_work.checked_add(row).unwrap();
    }
    (planner, config)
}
fn prepare<'s>(planner: &SourceTaskPlanner, source: &'s str,
    config: &Int4SourceMapConfig<SourceQuestion>, group_width: usize) -> PreparedInt4Question<'s> {
    planner.plan_int4_question_with_control(source, &config.task, config.budget,
        &PlanContext::new(&config.identity, config.budget).unwrap(), config.planning,
        config.mapping, group_width, &mut Continue).unwrap()
}

#[test]
fn sparse_source_windows_quote_only_real_native_rows_and_keep_the_complete_frontier() {
    let source = "ééé                        ééé";
    for group_width in [16, 32, 128] {
        let (planner, mut config) = planning_fixture(group_width);
        let row_budget = config.budget;
        let row = question_requirements(&config, source.len(), group_width, &planner).unwrap();
        let mut one_native_peak = None;
        let mut identities = None;
        let mut whole_frontier = None;
        for width in [1, 2, 4, 6] {
            config.mapping.reduction.map_batch_chunks = width;
            let declared = config.cohort_requirements(group_width).unwrap();
            let prepared = prepare(&planner, source, &config, group_width);
            let expected = prepared.preflight_metadata();
            assert_eq!((expected.chunk_count(), expected.native_chunks(), expected.whitespace_chunks()), (6, 2, 4));
            assert_eq!(prepared.max_kv_bytes(), row_budget.max_kv_bytes);
            let actual = prepared.cohort_requirements(config.native.context_tokens).unwrap();
            check_peak(actual, declared).unwrap();
            assert!(actual.native.kv_bytes <= declared.native.kv_bytes);
            assert!(actual.native.rope_bytes <= declared.native.rope_bytes);
            assert!(actual.native.scratch_payload_bound <= declared.native.scratch_payload_bound);
            assert!(actual.workspace_bytes <= declared.workspace_bytes);
            assert!(actual.max_result_bytes <= declared.max_result_bytes);
            assert_eq!(declared.native.kv_bytes, row.kv_bytes * width as u64);
            assert_eq!(config.budget, row_budget);
            if width < 6 {
                if let Some(previous) = one_native_peak { assert_eq!(actual, previous); }
                else { one_native_peak = Some(actual); }
            } else {
                let one = one_native_peak.unwrap();
                assert_eq!(actual.native.kv_bytes, one.native.kv_bytes * 2);
                assert_eq!(actual.native.rope_bytes, one.native.rope_bytes);
                assert!(actual.workspace_bytes > one.workspace_bytes);
            }
            let observed: Vec<_> = prepared.execution_identities().cloned().collect();
            if let Some(previous) = &identities { assert_eq!(&observed, previous); }
            else { identities = Some(observed); }
            let frontier = config.question_workspace_bytes(expected.chunk_count()).unwrap();
            assert!(frontier > config.question_workspace_bytes(expected.native_chunks()).unwrap());
            if let Some(previous) = whole_frontier { assert_eq!(frontier, previous); }
            else { whole_frontier = Some(frontier); }
        }
    }
}

#[test]
fn trailing_whitespace_has_a_real_output_frontier_but_no_extra_native_reservation() {
    let (planner, mut config) = planning_fixture(32);
    config.mapping.reduction.map_batch_chunks = 4;
    let compact = prepare(&planner, "ééé", &config, 32);
    let padded = prepare(&planner, "ééé                  ", &config, 32);
    let compact_meta = compact.preflight_metadata(); let padded_meta = padded.preflight_metadata();
    assert_eq!(compact_meta.native_chunks(), 1);
    assert_eq!(padded_meta.native_chunks(), 1);
    assert!(padded_meta.chunk_count() > compact_meta.chunk_count());
    assert_eq!(compact.cohort_requirements(config.native.context_tokens).unwrap(),
        padded.cohort_requirements(config.native.context_tokens).unwrap());
    assert_eq!(compact_meta.planned_work(), padded_meta.planned_work());
    assert_eq!(compact_meta.reserved_mask_visits(), padded_meta.reserved_mask_visits());
    assert!(config.question_workspace_bytes(padded_meta.chunk_count()).unwrap()
        > config.question_workspace_bytes(compact_meta.chunk_count()).unwrap());
    assert_eq!(padded.max_result_bytes(), config.mapping.reduction.max_result_bytes as u64);
}

#[test]
fn private_question_capacity_is_charged_without_changing_execution_identity_or_native_work() {
    let (planner, mut config) = planning_fixture(32);
    config.mapping.reduction.map_batch_chunks = 2;
    let source = "ééé      ééé";
    let original = prepare(&planner, source, &config, 32);
    let original_meta = original.preflight_metadata();
    let original_ids: Vec<_> = original.execution_identities().cloned().collect();
    let original_quote = original.cohort_requirements(config.native.context_tokens).unwrap();
    let before_capacity = config.task.question.capacity();
    let before_capture = input_bytes(source.len(), &config).unwrap();
    let mut larger = String::with_capacity(before_capacity + 8192); larger.push_str(&config.task.question);
    config.task.question = larger;
    assert_eq!(input_bytes(source.len(), &config).unwrap() - before_capture,
        (config.task.question.capacity() - before_capacity) as u64);
    let prepared = prepare(&planner, source, &config, 32);
    assert_eq!(prepared.preflight_metadata(), original_meta);
    assert_eq!(prepared.execution_identities().cloned().collect::<Vec<_>>(), original_ids);
    assert_eq!(prepared.cohort_requirements(config.native.context_tokens).unwrap(), original_quote);
}

#[test]
fn qa_specific_evidence_admission_cannot_be_replaced_by_the_raw_source_window_quote() {
    let (planner, mut config) = planning_fixture(32);
    config.mapping.reduction.map_batch_chunks = 2;
    let prepared = prepare(&planner, "ééé                        ééé", &config, 32);
    let count = prepared.preflight_metadata().chunk_count();
    let native = config.cohort_requirements(32).unwrap();
    let raw = config.workspace_bytes(count).unwrap();
    let qa = config.question_workspace_bytes(count).unwrap();
    assert!(qa > raw);
    config.task.verification.max_fields *= 2;
    config.task.verification.max_matches *= 2;
    assert!(config.question_workspace_bytes(count).unwrap() > qa);
    assert_eq!(config.cohort_requirements(32).unwrap(), native);
    assert_eq!(config.workspace_bytes(count).unwrap(), raw);
}

#[test]
fn cohort_schedule_cannot_widen_the_original_question_row_budget_or_capture_contract() {
    let (planner, mut config) = planning_fixture(32);
    config.mapping.reduction.map_batch_chunks = 4;
    let original = config.budget;
    let declared = config.cohort_requirements(32).unwrap();
    assert_eq!(config.budget, original);
    config.budget.max_kv_bytes -= 1;
    assert_eq!(config.cohort_requirements(32).unwrap(), declared);
    assert!(question_requirements(&config, 12, 32, &planner).is_err());
    config.budget = original;
    config.preparation_reserve_bytes = 0;
    assert!(question_requirements(&config, 12, 32, &planner).is_err());
    config.preparation_reserve_bytes = u64::MAX;
    assert!(input_bytes(12, &config).is_err());
}

#[test]
fn complete_question_cohort_output_keeps_owned_physical_lifetime_types() {
    fn send<T: Send + 'static>() {}
    send::<Int4SourceMapConfig<SourceQuestion>>();
    send::<Charged<SourceMapInput<SourceQuestion>>>();
    send::<Int4QuestionCohortRun>();
    send::<HostedOutput<Int4QuestionCohortRun>>();
    let _host = NlpEngine::answer_int4_document_cohorts;
}
