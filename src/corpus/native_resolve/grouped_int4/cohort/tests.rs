//! Model-free planning and explicitly synthetic native completion fixtures.
//! They exercise the real source/ticket/finalizer contract, not model quality.
use super::*;
use super::super::tests::{Continue, planner, docs, graph, limits, prepared, admitted, synthetic, all_same};
use crate::corpus::resolve::PairDecision;

fn budget(plan: &PreparedInt4Resolution<'_, '_, '_>, pairs: usize) -> Int4ResolutionCohortBudget {
    let q = Int4ResolutionCohortMemoryLimits { max_pairs: pairs,
        context_tokens: plan.required_context_tokens().max(1), max_model_work: plan.work,
        max_result_bytes: plan.max_result_bytes() }.requirements(plan.group_width()).unwrap();
    Int4ResolutionCohortBudget { native: Int4RunBudget::exact(plan.work),
        memory: Int4MemoryBudget { max_kv_bytes: q.native.kv_bytes,
            max_rope_bytes: q.native.rope_bytes, max_scratch_payload_bytes: q.native.scratch_payload_bound },
        max_task_workspace_bytes: q.workspace_bytes, max_output_bytes: q.max_result_bytes }
}
fn raw(window: &Window, width: usize) -> Int4CandidateCohortRun {
    Int4CandidateCohortRun { schema_version: 1, execution: native::INT4_SCORING_COHORT_EXECUTION.to_owned(),
        numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: width,
        heads: window.schedules.iter().map(|&s| synthetic(s, width, PairDecision::Same)).collect(),
        group_steps: window.minimum_group_steps, projection_groups: window.minimum_projection_groups,
        model_work: window.work }
}
fn complete(plan: PreparedInt4Resolution<'_, '_, '_>, pairs: usize) -> Int4ResolutionCohortRun {
    let ids = admitted(&plan); let cap = budget(&plan, pairs);
    let width = plan.group_width(); let expected = plan.work;
    plan.collect_cohorts_with_control(&ids, pairs, cap, &mut Continue, |_, window, progress, control|
        finish_window(raw(window, width), window, progress, width, planner().eos, expected,
            cap.max_output_bytes.min(MAX_OUTPUT_BYTES), control)).unwrap()
}
fn first_window(plan: &PreparedInt4Resolution<'_, '_, '_>, pairs: usize,
    cap: Int4ResolutionCohortBudget) -> Window {
    let n = pairs.min(plan.pair_count());
    Window::checked(&plan.inner.pairs[..n], &plan.schedules[..n], planner(), plan.task_budget(),
        plan.group_width(), cap, &mut Continue).unwrap()
}

#[test]
fn widths_and_ragged_windows_preserve_every_serial_semantic_byte() {
    let d = docs(); let g = graph(&d);
    for width in [16, 32] {
        let serial = canonjson::canonical_bytes(&all_same(prepared(&g, width))).unwrap();
        for pairs in [1, 2, 3, 32] {
            let result = complete(prepared(&g, width), pairs);
            assert_eq!(result.execution, INT4_RESOLUTION_COHORT_EXECUTION);
            assert_eq!(result.pair_cohorts, 3_usize.div_ceil(pairs));
            assert_eq!(result.output.head_count, 6);
            assert_eq!(result.output.rewound_positions, 12);
            assert_eq!(canonjson::canonical_bytes(&result.output).unwrap(), serial);
            assert!(result.group_steps > 0); assert!(result.projection_groups >= 4);
        }
    }
}

#[test]
fn both_pair_orders_stay_together_and_conflicts_keep_the_shared_cluster_policy() {
    let d = docs(); let g = graph(&d);
    let scalar = prepared(&g, 32); let ids = admitted(&scalar);
    let scalar = scalar.execute_heads(&ids, &mut Continue, |pair, order, _, s, _|
        Ok(synthetic(s, 32, if pair == 1 && order == 1 { PairDecision::Different } else { PairDecision::Same }))).unwrap();
    for width in [1, 2, 3] {
        let plan = prepared(&g, 32); let ids = admitted(&plan); let cap = budget(&plan, width);
        let expected = plan.work; let mut next_pair = 0;
        let result = plan.collect_cohorts_with_control(&ids, width, cap, &mut Continue,
            |pairs, window, progress, control| {
                assert_eq!(window.schedules.len(), pairs.len() * 2);
                let mut run = raw(window, 32);
                for (index, schedule) in window.schedules.iter().copied().enumerate() {
                    let pair = next_pair + index / 2;
                    run.heads[index] = synthetic(schedule, 32,
                        if pair == 1 && index % 2 == 1 { PairDecision::Different } else { PairDecision::Same });
                }
                next_pair += pairs.len();
                finish_window(run, window, progress, 32, planner().eos, expected,
                    cap.max_output_bytes.min(MAX_OUTPUT_BYTES), control)
            }).unwrap();
        assert_eq!(next_pair, 3);
        assert_eq!(canonjson::canonical_bytes(&result.output).unwrap(), canonjson::canonical_bytes(&scalar).unwrap());
        assert!(result.output.result.judgments.iter().any(|j| j.decision == PairDecision::Uncertain));
    }
}

#[test]
fn invalid_widths_and_late_identity_changes_refuse_before_any_window() {
    let d = docs(); let g = graph(&d);
    for mutation in 0..7 {
        let plan = prepared(&g, 32); let cap = budget(&plan, 1); let mut ids = admitted(&plan);
        let pairs = match mutation { 0 => 0, 1 => 33, 2 => usize::MAX, _ => 1 };
        match mutation {
            3 => ids.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"wrong final prompt"),
            4 => ids.reverse(), 5 => { ids.pop(); }, 6 => ids.push(ids[0].clone()), _ => {},
        }
        let mut calls = 0;
        let result = plan.collect_cohorts_with_control(&ids, pairs, cap, &mut Continue, |_, _, _, _| {
            calls += 1; Err(Int4ResolveError::Accounting)
        });
        assert!(result.is_err(), "mutation {mutation}"); assert_eq!(calls, 0);
    }
}

#[test]
fn a_larger_late_window_is_admitted_before_the_first_native_callback() {
    let mut d = docs();
    d[2].text = format!("Smith {}", "a long original surrounding context ".repeat(8));
    let g = graph(&d); let plan = prepared(&g, 32); let ids = admitted(&plan);
    let mut cap = budget(&plan, 1);
    let first = first_window(&plan, 1, cap);
    let required = Int4MemoryRequirement::for_cohort(&first.contexts, 32).unwrap();
    let second = Window::checked(&plan.inner.pairs[1..2], &plan.schedules[1..2], planner(),
        plan.task_budget(), 32, cap, &mut Continue).unwrap();
    let later = Int4MemoryRequirement::for_cohort(&second.contexts, 32).unwrap();
    assert!(later.kv_bytes > required.kv_bytes);
    cap.memory.max_kv_bytes = required.kv_bytes;
    let mut calls = 0;
    let result = plan.collect_cohorts_with_control(&ids, 1, cap, &mut Continue, |_, _, _, _| {
        calls += 1; Err(Int4ResolveError::Accounting)
    });
    assert!(result.is_err()); assert_eq!(calls, 0);
}

#[test]
fn exact_physical_memory_and_workspace_axes_refuse_before_execution() {
    let d = docs(); let g = graph(&d);
    for axis in 0..4 {
        let plan = prepared(&g, 32); let ids = admitted(&plan); let mut cap = budget(&plan, 3);
        let window = first_window(&plan, 3, cap);
        let required = Int4MemoryRequirement::for_cohort(&window.contexts, 32).unwrap();
        cap.memory = Int4MemoryBudget { max_kv_bytes: required.kv_bytes,
            max_rope_bytes: required.rope_bytes, max_scratch_payload_bytes: required.scratch_payload_bound };
        cap.max_task_workspace_bytes = native::required_task_workspace_bytes(window.work,
            window.contexts.len(), cap.max_output_bytes.min(MAX_OUTPUT_BYTES)).unwrap();
        plan.preflight_cohort_windows(&ids, 3, cap, &mut Continue).unwrap();
        match axis { 0 => cap.memory.max_kv_bytes -= 1, 1 => cap.memory.max_rope_bytes -= 1,
            2 => cap.memory.max_scratch_payload_bytes -= 1, _ => cap.max_task_workspace_bytes -= 1 }
        let mut calls = 0;
        assert!(plan.collect_cohorts_with_control(&ids, 3, cap, &mut Continue, |_, _, _, _| {
            calls += 1; Err(Int4ResolveError::Accounting)
        }).is_err());
        assert_eq!(calls, 0, "axis {axis}");
    }
}

#[test]
fn physical_head_sum_does_not_enlarge_the_original_single_head_kv_budget() {
    let d = docs(); let g = graph(&d); let mut plan = prepared(&g, 32);
    let ids = admitted(&plan); let cap = budget(&plan, 3);
    let one_head = (plan.required_context_tokens() as u64) * KV_BYTES_PER_TOKEN as u64;
    assert!(cap.memory.max_kv_bytes > one_head);
    plan.inner.limits.per_head.max_kv_bytes = one_head;
    plan.preflight_cohort_windows(&ids, 3, cap, &mut Continue).unwrap();
    plan.inner.limits.per_head.max_kv_bytes = one_head - 1;
    assert!(plan.preflight_cohort_windows(&ids, 3, cap, &mut Continue).is_err());
}

#[test]
fn pair_windows_never_renew_whole_snapshot_native_authority() {
    let d = docs(); let g = graph(&d);
    for axis in 0..4 {
        let plan = prepared(&g, 32); let ids = admitted(&plan); let mut cap = budget(&plan, 1);
        match axis { 0 => cap.native.max_forward_positions -= 1, 1 => cap.native.max_attention_pairs -= 1,
            2 => cap.native.max_projection_work.dot_products -= 1,
            _ => cap.native.max_projection_work.multiply_accumulates -= 1 }
        let mut calls = 0;
        assert!(plan.collect_cohorts_with_control(&ids, 1, cap, &mut Continue, |_, _, _, _| {
            calls += 1; Err(Int4ResolveError::Accounting)
        }).is_err());
        assert_eq!(calls, 0, "axis {axis}");
    }
}

#[test]
fn every_window_receipt_axis_is_checked_before_a_second_window() {
    let d = docs(); let g = graph(&d);
    for mutation in 0..20 {
        let plan = prepared(&g, 32); let ids = admitted(&plan); let cap = budget(&plan, 1);
        let expected = plan.work; let mut calls = 0;
        let result = plan.collect_cohorts_with_control(&ids, 1, cap, &mut Continue,
            |_, window, progress, control| {
                calls += 1; let mut run = raw(window, 32);
                match mutation {
                    0 => run.schema_version = 2, 1 => run.execution = "wrong".to_owned(),
                    2 => run.numerics_profile = HF_BF16_EAGER_PROFILE.to_owned(), 3 => run.group_width = 16,
                    4 => { run.heads.pop(); }, 5 => run.model_work.forward_positions -= 1,
                    6 => run.model_work.projected_logits -= 1, 7 => run.model_work.attention_pairs -= 1,
                    8 => run.model_work.projections.dot_products -= 1,
                    9 => run.model_work.projections.multiply_accumulates -= 1,
                    10 => run.group_steps = window.minimum_group_steps - 1,
                    11 => run.group_steps = window.work.forward_positions + 1,
                    12 => run.projection_groups = window.minimum_projection_groups - 1,
                    13 => run.projection_groups = window.projection_steps + 1,
                    14 => run.heads[1].rewound_positions = 0,
                    15 => run.heads[1].scores.full_vocab_denominators_computed = false,
                    16 => run.heads[1].scores.eos_token_id ^= 1,
                    17 => run.heads[1].scores.candidates[0].sequence_score = f64::NAN,
                    18 => run.heads[1].scores.candidates.reverse(),
                    _ => run.heads[1].scores.work.scored_edges -= 1,
                }
                finish_window(run, window, progress, 32, planner().eos, expected,
                    cap.max_output_bytes.min(MAX_OUTPUT_BYTES), control)
            });
        assert!(result.is_err(), "mutation {mutation}"); assert_eq!(calls, 1);
    }
}

#[test]
fn a_late_failed_window_retains_its_cancellation_and_publishes_no_partial_graph() {
    let d = docs(); let g = graph(&d); let plan = prepared(&g, 32);
    let ids = admitted(&plan); let cap = budget(&plan, 1); let expected = plan.work; let mut calls = 0;
    let error = plan.collect_cohorts_with_control(&ids, 1, cap, &mut Continue,
        |_, window, progress, control| {
            calls += 1;
            if calls == 2 { return Err(StrictInt4Error::Cancelled(DecodeCancellationKind::Deadline).into()); }
            finish_window(raw(window, 32), window, progress, 32, planner().eos, expected,
                cap.max_output_bytes.min(MAX_OUTPUT_BYTES), control)
        }).err().unwrap();
    assert_eq!(calls, 2); assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
}

#[test]
fn cancellation_after_a_window_prevents_the_next_window() {
    struct Control(bool);
    impl DecodeStepControl for Control {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.0.then_some(DecodeCancellationKind::Shutdown)
        }
    }
    let d = docs(); let g = graph(&d); let plan = prepared(&g, 32);
    let ids = admitted(&plan); let cap = budget(&plan, 1); let expected = plan.work; let mut calls = 0;
    let error = plan.collect_cohorts_with_control(&ids, 1, cap, &mut Control(false),
        |_, window, progress, control| {
            calls += 1;
            let result = finish_window(raw(window, 32), window, progress, 32, planner().eos, expected,
                cap.max_output_bytes.min(MAX_OUTPUT_BYTES), control)?;
            control.0 = true; Ok(result)
        }).err().unwrap();
    assert_eq!(calls, 1); assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Shutdown));
}

#[test]
fn empty_graph_uses_no_window_and_preserves_the_no_model_semantic_result() {
    let d = docs(); let g = graph(&d[..1]);
    let serial = canonjson::canonical_bytes(&all_same(prepared(&g, 32))).unwrap();
    for pairs in [1, 32] {
        let plan = prepared(&g, 32); let ids = admitted(&plan); let mut cap = budget(&plan, pairs);
        cap.memory = Int4MemoryBudget { max_kv_bytes: 0, max_rope_bytes: 0, max_scratch_payload_bytes: 0 };
        let result = plan.collect_cohorts_with_control(&ids, pairs, cap, &mut Continue,
            |_, _, _, _| panic!("empty graph must allocate no native window")).unwrap();
        assert_eq!(result.pair_cohorts, 0); assert_eq!(result.group_steps, 0); assert_eq!(result.projection_groups, 0);
        assert!(!result.output.model_evaluated);
        assert_eq!(canonjson::canonical_bytes(&result.output).unwrap(), serial);
    }
}

#[test]
fn wrapper_publication_cap_is_exact_even_without_native_work() {
    let d = docs(); let g = graph(&d[..1]);
    let bytes = canonjson::canonical_bytes(&complete(prepared(&g, 32), 1)).unwrap().len() as u64;
    for short in [false, true] {
        let plan = prepared(&g, 32); let ids = admitted(&plan); let mut cap = budget(&plan, 1);
        cap.max_output_bytes = bytes - u64::from(short);
        let result = plan.collect_cohorts_with_control(&ids, 1, cap, &mut Continue,
            |_, _, _, _| panic!("empty graph must allocate no native window"));
        if short { assert!(result.is_err()); }
        else { assert_eq!(canonjson::canonical_bytes(&result.unwrap()).unwrap().len() as u64, bytes); }
    }
}

#[test]
fn quote_prices_two_heads_per_pair_without_multiplying_the_single_head_context() {
    let q = Int4ResolutionCohortMemoryLimits { max_pairs: 3, context_tokens: 1024,
        max_model_work: limits().max_model_work, max_result_bytes: 1_048_576 }.requirements(32).unwrap();
    let expected = Int4MemoryRequirement::for_cohort(&[1024; 6], 32).unwrap();
    assert_eq!(q.native, expected);
    assert_eq!(q.native.kv_bytes, 6 * 1024 * KV_BYTES_PER_TOKEN as u64);
    assert_eq!(q.max_result_bytes, 1_048_576 + INT4_RESOLUTION_COHORT_WRAPPER_BYTES);
    let mut work = limits().max_model_work; work.forward_positions = 6 * 1026;
    assert_eq!(q.workspace_bytes, native::required_task_workspace_bytes(work, 6, q.max_result_bytes).unwrap());
}

#[test]
fn quote_rejects_bad_geometry_and_prices_native_staging_for_small_semantic_outputs() {
    let base = Int4ResolutionCohortMemoryLimits { max_pairs: 32, context_tokens: 512,
        max_model_work: limits().max_model_work, max_result_bytes: 1 };
    let q = base.requirements(32).unwrap();
    assert_eq!(q.max_result_bytes, 64 * HEAD_RESULT_BYTES + INT4_RESOLUTION_COHORT_WRAPPER_BYTES);
    for changed in [Int4ResolutionCohortMemoryLimits { max_pairs: 0, ..base },
        Int4ResolutionCohortMemoryLimits { max_pairs: 33, ..base },
        Int4ResolutionCohortMemoryLimits { max_pairs: usize::MAX, ..base },
        Int4ResolutionCohortMemoryLimits { context_tokens: 0, ..base },
        Int4ResolutionCohortMemoryLimits { context_tokens: usize::MAX, ..base },
        Int4ResolutionCohortMemoryLimits { max_result_bytes: 0, ..base },
        Int4ResolutionCohortMemoryLimits { max_result_bytes: MAX_OUTPUT_BYTES + 1, ..base }] {
        assert!(changed.requirements(32).is_err());
    }
    assert!(base.requirements(0).is_err());
    let largest = Int4ResolutionCohortMemoryLimits { max_result_bytes: MAX_OUTPUT_BYTES, ..base }.requirements(32).unwrap();
    assert_eq!(largest.max_result_bytes, MAX_OUTPUT_BYTES + INT4_RESOLUTION_COHORT_WRAPPER_BYTES);
}

#[test]
fn quote_respects_a_tighter_total_forward_allowance_for_workspace() {
    let d = docs(); let g = graph(&d); let plan = prepared(&g, 32);
    let base = Int4ResolutionCohortMemoryLimits { max_pairs: 32, context_tokens: 8192,
        max_model_work: plan.work, max_result_bytes: plan.max_result_bytes() };
    let quoted = base.requirements(32).unwrap();
    assert_eq!(quoted.workspace_bytes, native::required_task_workspace_bytes(plan.work, 64,
        quoted.max_result_bytes.min(MAX_OUTPUT_BYTES)).unwrap());
    let unlimited = Int4ResolutionCohortMemoryLimits { max_model_work: limits().max_model_work, ..base }.requirements(32).unwrap();
    assert!(quoted.workspace_bytes < unlimited.workspace_bytes);
}

#[test]
fn public_wrapper_keeps_only_native_statistics_and_existing_snapshot_semantics() {
    let d = docs(); let g = graph(&d); let result = complete(prepared(&g, 32), 2);
    let public = canonjson::canonical_string(&result).unwrap();
    for absent in ["packing_set_digest", "prompt_digest", "logical_model_digest", "taskir_digest", "source_root_sha256"] {
        assert!(!public.contains(absent));
    }
    assert_eq!(result.output.result.dependency_scope, DependencyScope::CorpusGlobal);
    assert_eq!(result.output.result.score_space, ScoreSpace::FullVocabSequenceLogprob);
    assert_eq!(result.output.result.calibration, "uncalibrated");
}

#[test]
fn an_empty_graph_still_observes_terminal_cancellation_before_publication() {
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::LinkedExit) }
    }
    let d = docs(); let g = graph(&d[..1]); let plan = prepared(&g, 32);
    let ids = admitted(&plan); let cap = budget(&plan, 1);
    let error = plan.collect_cohorts_with_control(&ids, 1, cap, &mut Stop,
        |_, _, _, _| panic!("cancelled empty graph must do no work")).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::LinkedExit));
}
