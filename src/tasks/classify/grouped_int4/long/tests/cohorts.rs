//! Actual pinned plans and semantic finalizers with private synthetic logits.
//! These tests exercise complete orchestration, not native numerical parity.
use super::*;
use crate::{execution_identity::Sha256Digest,
    native_engine::{lmhead::scoring::{CandidateLogits, ProjectionRows},
        strict_int4::{Int4MemoryBudget, Int4MemoryRequirement,
            scoring::cohort::{Int4ScoredCohort, Int4ScoringCohortBudget, required_task_workspace_bytes, task}}}};
use super::super::cohort::Window;

fn source(mode: ClassificationMode) -> ClassificationRequest {
    let mut request = request(mode);
    request.document = "0123456789abcdefghijklmn".to_owned();
    request.policy = ClassificationPolicy { minimum_candidate_weight_ppm: 600_000, minimum_margin_ppm: 200_000 };
    request
}
fn budget(plan: &PreparedInt4ClassificationMap<'_>) -> Int4ClassificationMapCohortBudget {
    Int4ClassificationMapCohortBudget { native: Int4RunBudget::exact(plan.planned_work()),
        memory: Int4MemoryBudget { max_kv_bytes: u64::MAX, max_rope_bytes: u64::MAX,
            max_scratch_payload_bytes: u64::MAX }, max_task_workspace_bytes: u64::MAX,
        max_output_bytes: MAX_BYTES }
}
fn collect<C: DecodeStepControl, F>(plan: &PreparedInt4ClassificationMap<'_>, admitted: &[ExecutionIdentity],
    chunks: usize, budget: Int4ClassificationMapCohortBudget, control: &mut C, mut execute: F)
    -> Result<Int4ClassificationMapCohortRun, Int4ClassificationMapError>
where F: FnMut(&Window, &[(&PreparedInt4Classification, &ExecutionIdentity)], &mut C)
    -> Result<Int4ScoredCohort<Vec<Int4ClassificationRun>>, Int4ClassificationMapError> {
    let windows = plan.cohort_windows(admitted, chunks, budget, control)?;
    let result = plan.collect_cohorts_with_control(&windows, control, |window, control| {
        let documents: Vec<_> = (window.start..window.end).map(|row| (&plan.plans[row], &admitted[row])).collect();
        execute(window, &documents, control)
    })?;
    plan.preflight_metadata().verify_completed_cohorts(&result, chunks)?;
    Ok(result)
}
struct Prefer(u32);
impl CandidateLogits for Prefer {
    type Error = &'static str;
    fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let ProjectionRows::FullVocabulary { vocabulary_size } = rows else { panic!("full denominator required") };
        let mut logits = vec![0.0; vocabulary_size];
        if prefix.is_empty() { logits[self.0 as usize] = 8.0; }
        Ok(logits)
    }
}
fn native_heads(plan: &PreparedInt4Classification, chunk: usize)
    -> crate::native_engine::strict_int4::scoring::cohort::Int4CandidateCohortRun {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    fixtures::run(plan.inner.heads.iter().zip(&plan.schedules).enumerate().map(|(head_index, (head, &schedule))| {
        let code = if plan.head_count() == 1 { b'A' + (chunk % 3) as u8 }
            else if head_index == 0 || (head_index == 2 && chunk % 2 == 0) { b'Y' } else { b'N' };
        let token = tokenizer.tokenizer().encode_byte_fallback_only(&[code]).unwrap()[0];
        (head.classifier.scorer.score(&mut Prefer(token), ScoringMode::FullVocabulary).unwrap(), schedule)
    }))
}
fn serial(plan: &PreparedInt4ClassificationMap<'_>) -> Int4ClassificationMapRun {
    plan.collect_with_control(&mut Continue, |index, row, control| {
        let mut heads = native_heads(row, index).heads.into_iter();
        row.execute_heads(control, |_, _, _, _| heads.next().ok_or(Int4ClassificationError::Accounting)).map_err(Into::into)
    }).unwrap()
}
fn window_run<C: DecodeStepControl>(window: &Window, documents: &[(&PreparedInt4Classification, &ExecutionIdentity)],
    cap: u64, control: &mut C) -> Result<Int4ScoredCohort<Vec<Int4ClassificationRun>>, Int4ClassificationMapError> {
    let (schedules, work) = task::document_layout(documents, &window.contexts, Int4ScoringCohortBudget {
        native: Int4RunBudget::exact(window.work), max_kv_bytes: u64::MAX, max_output_bytes: cap })?;
    let mut runs = documents.iter().enumerate().map(|(index, (plan, _))| native_heads(plan, window.start + index));
    let mut native = runs.next().expect("nonempty admitted window");
    for run in runs {
        native.model_work = native.model_work.checked_add(run.model_work)?;
        native.group_steps += run.group_steps; native.projection_groups += run.projection_groups;
        native.heads.extend(run.heads);
    }
    task::finish_documents(documents, native, &schedules, work, cap, control).map_err(Into::into)
}

#[test]
fn ragged_cohorts_preserve_exact_complete_maps_and_independent_label_decisions() {
    for mode in [ClassificationMode::Exclusive, ClassificationMode::MultiLabel] {
        let request = source(mode); let prepared = prepare(&request, mapping()).unwrap();
        assert_eq!(prepared.chunk_count(), 3);
        let identities = prepared.identities().unwrap(); let budget = budget(&prepared);
        let expected = serial(&prepared);
        let mut windows = Vec::new();
        let actual = collect(&prepared, &identities, 2, budget, &mut Continue,
            |window, documents, control| {
                windows.push((window.start, window.end, window.contexts.len()));
                window_run(window, documents, budget.max_output_bytes, control)
            }).unwrap();
        let heads = prepared.head_count();
        assert_eq!(windows, [(0, 2, 2 * heads), (2, 3, heads)]);
        assert_eq!(canonjson::canonical_bytes(&actual.output).unwrap(), canonjson::canonical_bytes(&expected).unwrap());
        assert_eq!(actual.chunk_cohorts, 2); assert_eq!(actual.output.model_work, prepared.planned_work());
        assert_eq!(actual.group_steps, prepared.planned_work().forward_positions);
        prepared.preflight_metadata().verify_completed_cohorts(&actual, 2).unwrap();
        if let ClassificationAggregateResult::MultiLabel(labels) = &actual.output.aggregate.result {
            assert!(labels.selected_ids.iter().any(|id| id == "a"));
            assert!(labels.excluded_ids.iter().any(|id| id == "b"));
            assert!(labels.labels.iter().map(|label| label.positive_candidate_weight).sum::<f64>() > 1.5);
        }
    }
}

#[test]
fn every_late_identity_and_declared_geometry_is_checked_before_any_native_window() {
    let request = source(ClassificationMode::MultiLabel); let prepared = prepare(&request, mapping()).unwrap();
    let mut identities = prepared.identities().unwrap(); let budget = budget(&prepared);
    identities.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"foreign-last-chunk");
    let mut calls = 0;
    let result = collect(&prepared, &identities, 1, budget, &mut Continue, |_, _, _| {
        calls += 1; Err(Int4ClassificationMapError::Admission)
    });
    assert!(result.is_err()); assert_eq!(calls, 0);
    let identities = prepared.identities().unwrap();
    for chunks in [0, 22, 65, usize::MAX] {
        assert!(prepared.cohort_windows(&identities, chunks, budget, &mut Continue).is_err());
    }
    prepared.cohort_windows(&identities, 21, budget, &mut Continue).unwrap();
    let request = source(ClassificationMode::Exclusive); let prepared = prepare(&request, mapping()).unwrap();
    prepared.cohort_windows(&prepared.identities().unwrap(), 64, self::budget(&prepared), &mut Continue).unwrap();
}

#[test]
fn every_document_work_axis_and_exact_peak_are_nonrenewable() {
    let request = source(ClassificationMode::MultiLabel); let mut prepared = prepare(&request, mapping()).unwrap();
    let identities = prepared.identities().unwrap(); let generous = budget(&prepared);
    let windows = prepared.cohort_windows(&identities, 2, generous, &mut Continue).unwrap();
    let mut exact = generous;
    exact.memory = Int4MemoryBudget { max_kv_bytes: 0, max_rope_bytes: 0, max_scratch_payload_bytes: 0 };
    exact.max_task_workspace_bytes = 0;
    for window in windows {
        let required = Int4MemoryRequirement::for_cohort(&window.contexts, prepared.group_width()).unwrap();
        exact.memory.max_kv_bytes = exact.memory.max_kv_bytes.max(required.kv_bytes);
        exact.memory.max_rope_bytes = exact.memory.max_rope_bytes.max(required.rope_bytes);
        exact.memory.max_scratch_payload_bytes = exact.memory.max_scratch_payload_bytes.max(required.scratch_payload_bound);
        exact.max_task_workspace_bytes = exact.max_task_workspace_bytes.max(required_task_workspace_bytes(
            window.work, window.contexts.len(), exact.max_output_bytes).unwrap());
    }
    prepared.cohort_windows(&identities, 2, exact, &mut Continue).unwrap();
    for axis in 0..9 {
        let mut cap = exact;
        prepared.limits.max_model_work = prepared.planned_work();
        match axis {
            0 => cap.native.max_forward_positions -= 1,
            1 => prepared.limits.max_model_work.projected_logits -= 1,
            2 => cap.native.max_attention_pairs -= 1,
            3 => cap.native.max_projection_work.dot_products -= 1,
            4 => cap.native.max_projection_work.multiply_accumulates -= 1,
            5 => cap.memory.max_kv_bytes -= 1,
            6 => cap.memory.max_rope_bytes -= 1,
            7 => cap.memory.max_scratch_payload_bytes -= 1,
            _ => cap.max_task_workspace_bytes -= 1,
        }
        let mut calls = 0;
        assert!(collect(&prepared, &identities, 2, cap, &mut Continue, |_, _, _| {
            calls += 1; Err(Int4ClassificationMapError::Admission)
        }).is_err(), "axis {axis}");
        assert_eq!(calls, 0, "axis {axis}");
    }
}

#[test]
fn presave_admission_refuses_a_larger_future_window_without_native_execution() {
    let mut request = source(ClassificationMode::MultiLabel);
    // The scalar boundary leaves seven bytes in the first chunk and eight in
    // the second. A first-window-only check would incorrectly admit this map.
    request.document = "1234567𐐀abcd".to_owned();
    let prepared = prepare(&request, mapping()).unwrap();
    let identities = prepared.identities().unwrap(); let generous = budget(&prepared);
    let windows = prepared.cohort_windows(&identities, 1, generous, &mut Continue).unwrap();
    assert_eq!(windows.len(), 2);
    let first = Int4MemoryRequirement::for_cohort(&windows[0].contexts, prepared.group_width()).unwrap();
    let second = Int4MemoryRequirement::for_cohort(&windows[1].contexts, prepared.group_width()).unwrap();
    let first_workspace = required_task_workspace_bytes(windows[0].work,
        windows[0].contexts.len(), generous.max_output_bytes).unwrap();
    let second_workspace = required_task_workspace_bytes(windows[1].work,
        windows[1].contexts.len(), generous.max_output_bytes).unwrap();
    assert!(first.kv_bytes < second.kv_bytes && first.rope_bytes < second.rope_bytes);
    assert!(first_workspace < second_workspace);
    for axis in 0..3 {
        let mut limit = generous;
        match axis { 0 => limit.memory.max_kv_bytes = first.kv_bytes,
            1 => limit.memory.max_rope_bytes = first.rope_bytes,
            _ => limit.max_task_workspace_bytes = first_workspace }
        // This is the all-window checker shared by the public pre-save
        // preflight and native execution; no weights or engine are needed.
        assert!(prepared.cohort_windows(&identities, 1, limit, &mut Continue).is_err());
        let mut calls = 0;
        assert!(collect(&prepared, &identities, 1, limit, &mut Continue, |_, _, _| {
            calls += 1; Err(Int4ClassificationMapError::Admission)
        }).is_err());
        assert_eq!(calls, 0);
    }
}

#[test]
fn per_chunk_task_authority_covers_the_sum_of_independent_head_contexts() {
    let mut request = source(ClassificationMode::MultiLabel);
    let prepared = prepare(&request, mapping()).unwrap();
    let kv = prepared.plans.iter().map(|plan| plan.cohort_contexts().unwrap().iter().sum::<usize>() as u64
        * crate::native_engine::kv::KV_BYTES_PER_TOKEN as u64).max().unwrap();
    drop(prepared);
    request.budget.max_kv_bytes = kv;
    let prepared = prepare(&request, mapping()).unwrap();
    prepared.cohort_windows(&prepared.identities().unwrap(), 2, budget(&prepared), &mut Continue).unwrap();
    drop(prepared); request.budget.max_kv_bytes -= 1;
    let prepared = prepare(&request, mapping()).unwrap();
    assert!(prepared.cohort_windows(&prepared.identities().unwrap(), 2, budget(&prepared), &mut Continue).is_err());
}

#[test]
fn late_failure_cancellation_and_malformed_native_windows_return_no_partial_document() {
    let request = source(ClassificationMode::MultiLabel); let prepared = prepare(&request, mapping()).unwrap();
    let identities = prepared.identities().unwrap(); let budget = budget(&prepared);
    for fault in 0..7 {
        let mut calls = 0;
        let result = collect(&prepared, &identities, 2, budget, &mut Continue,
            |window, documents, control| {
                calls += 1;
                let mut run = window_run(window, documents, budget.max_output_bytes, control)?;
                if window.start == 2 {
                    match fault {
                        0 => return Err(StrictInt4Error::Cancelled(DecodeCancellationKind::Deadline).into()),
                        1 => { run.output.pop(); }, 2 => run.output[0].model_work.projected_logits += 1,
                        3 => run.group_steps = 0, 4 => run.projection_groups = u64::MAX,
                        5 => run.execution = "foreign-window".to_owned(),
                        _ => {
                            // A physical projection group is bounded by
                            // evaluated prefixes, not vocabulary-sized logits.
                            run.projection_groups = documents.iter().flat_map(|(plan, _)| &plan.schedules)
                                .map(|schedule| schedule.scoring.prefix_evaluations as u64).sum::<u64>() + 1;
                            assert!(run.projection_groups <= window.work.projected_logits);
                        }
                    }
                }
                Ok(run)
            });
        let error = result.unwrap_err(); assert_eq!(calls, 2);
        if fault == 0 { assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); }
    }
    struct Stop(bool);
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.0.then_some(DecodeCancellationKind::Shutdown)
        }
    }
    let mut calls = 0;
    let error = collect(&prepared, &identities, 2, budget, &mut Stop(false),
        |window, documents, control| {
            calls += 1; let run = window_run(window, documents, budget.max_output_bytes, control)?;
            control.0 = true; Ok(run)
        }).unwrap_err();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Shutdown)); assert_eq!(calls, 1);
}

#[test]
fn cohort_collection_enforces_retained_document_and_physical_wrapper_boundaries() {
    let request = source(ClassificationMode::Exclusive); let mut prepared = prepare(&request, mapping()).unwrap();
    let identities = prepared.identities().unwrap(); let budget = budget(&prepared);
    let baseline = serial(&prepared);
    prepared.limits.max_retained_result_bytes = baseline.retained_result_bytes;
    prepared.limits.max_result_bytes = canonjson::canonical_bytes(&baseline).unwrap().len() as u64;
    let grouped = collect(&prepared, &identities, 2, budget, &mut Continue,
        |window, documents, control| window_run(window, documents, budget.max_output_bytes, control)).unwrap();
    assert_eq!(grouped.output, baseline);
    for field in 0..5 {
        let mut corrupt = grouped.clone();
        match field { 0 => corrupt.chunk_cohorts += 1, 1 => corrupt.output.chunks.swap(0, 1),
            2 => corrupt.projection_groups = 0, 3 => corrupt.output.retained_result_bytes += 1,
            _ => corrupt.execution.push('x') }
        assert!(prepared.preflight_metadata().verify_completed_cohorts(&corrupt, 2).is_err());
    }
    prepared.limits.max_retained_result_bytes -= 1;
    assert!(matches!(collect(&prepared, &identities, 2, budget, &mut Continue,
        |window, documents, control| window_run(window, documents, budget.max_output_bytes, control)),
        Err(Int4ClassificationMapError::Output)));
    prepared.limits.max_retained_result_bytes += 1; prepared.limits.max_result_bytes -= 1;
    assert!(matches!(collect(&prepared, &identities, 2, budget, &mut Continue,
        |window, documents, control| window_run(window, documents, budget.max_output_bytes, control)),
        Err(Int4ClassificationMapError::Output)));
}
