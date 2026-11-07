//! Closed finalizer fixtures keep selected identities, caps and original
//! schema/source authority intact across both physical cohort schedules.
use super::*;

fn limits(cap: usize) -> selected_head::Int4JsonSparseLimits {
    selected_head::Int4JsonSparseLimits { max_rows_per_step: cap }
}
fn prefill() -> Int4PrefillLimits {
    Int4PrefillLimits { max_batch_rows: 64,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(64).unwrap() }
}
fn sparse_row(plan: &Int4ExtractPlan, json: &str, projected: u64) -> Int4JsonRun {
    let mut row = raw_row(plan, json);
    let work = plan.completed_work(row.output.token_ids.len(), projected).unwrap();
    row.execution = selected_head::INT4_SPARSE_JSON_EXECUTION.to_owned();
    row.output.projected_logits = projected; row.output.forward_positions = work.forward_positions;
    row.model_work = work; row
}
fn selected_raw(requests: &[Int4ExtractCohortRequest<'_>], json: &[&str], projected: &[u64], packed: bool)
    -> Int4JsonCohortRun {
    assert_eq!(requests.len(), json.len()); assert_eq!(requests.len(), projected.len());
    let mut run = raw(requests, json);
    run.execution = if packed { INT4_JSON_SELECTED_PACKED_COHORT_EXECUTION } else { INT4_JSON_SELECTED_COHORT_EXECUTION }.to_owned();
    run.model_work = Int4Work::default(); run.projection_groups = 0;
    for (slot, (&json, &projected)) in json.iter().zip(projected).enumerate() {
        run.sequences[slot] = sparse_row(requests[slot].prepared, json, projected);
        run.model_work = run.model_work.checked_add(run.sequences[slot].model_work).unwrap();
        // Fixture has one JSON token followed by a separately projected EOS.
        run.projection_groups += (projected - 1).div_ceil(32) + 1;
    }
    if packed { run.group_steps = run.model_work.forward_positions.div_ceil(64).max(2); }
    run
}

#[test]
fn independent_selected_caps_keep_complete_source_occurrences_and_serial_task_results() {
    let plans = [source_plan("é Alice Alice", &[]).with_selected_rows(limits(3)).unwrap(),
        source_plan("Bob", &[]).with_selected_rows(limits(7)).unwrap(),
        plan(r#"{"type":"boolean"}"#, 127).with_selected_rows(limits(2)).unwrap()];
    let requests = requests(&plans); let json = ["\"Alice\"", "\"Bob\"", "true"]; let projected = [3, 6, 3];
    let before: Vec<_> = plans.iter().map(|plan| canonjson::canonical_bytes(plan.execution_identity()).unwrap()).collect();
    let ordinary = finish(&requests, selected_raw(&requests, &json, &projected, false), 1 << 20).unwrap();
    let packed = finish_scheduled(&requests, selected_raw(&requests, &json, &projected, true), 1 << 20, Some(prefill())).unwrap();
    assert_eq!(ordinary.execution, INT4_EXTRACT_SELECTED_COHORT_EXECUTION);
    assert_eq!(packed.execution, INT4_EXTRACT_SELECTED_PACKED_COHORT_EXECUTION);
    assert_eq!(ordinary.sequences, packed.sequences);
    assert_eq!(ordinary.planned_work, packed.planned_work); assert_eq!(ordinary.model_work, packed.model_work);
    assert_eq!(ordinary.projection_groups, 6);
    for (slot, plan) in plans.iter().enumerate() {
        assert_eq!(ordinary.sequences[slot], plan.finalize(sparse_row(plan, json[slot], projected[slot])).unwrap());
        plan.verify_completed(&ordinary.sequences[slot]).unwrap();
        assert_eq!(before[slot], canonjson::canonical_bytes(plan.execution_identity()).unwrap());
    }
    let evidence = &ordinary.sequences[0].result.source_fields[0];
    assert_eq!(evidence.occurrence, SourceOccurrence::Ambiguous); assert_eq!(evidence.spans.len(), 2);
    assert_eq!(evidence.spans[0].byte_start, 3); assert_eq!(evidence.spans[0].scalar_start, 2);
}

#[test]
fn mixed_head_policies_and_foreign_selected_admission_fail_without_rebinding_a_plan() {
    let dense = plan(r#"{"type":"boolean"}"#, 127);
    let dense_id = dense.execution_identity().clone();
    let selected = plan(r#"{"type":"boolean"}"#, 127).with_selected_rows(limits(3)).unwrap();
    let plans = [dense, selected]; let mixed = requests(&plans);
    assert!(selected_policy(&mixed).is_err());
    assert!(finish(&mixed, raw(&mixed, &["true", "true"]), 1 << 20).is_err());
    let vocabulary = ExtractionVocabulary::pinned(crate::tasks::extract::tests::registry().template_controls()).unwrap();
    let wrong = Int4ExtractCohortRequest { admitted_identity: &dense_id, ..mixed[1] };
    assert!(preflight(&wrong, &vocabulary).is_err());
    let plans = [plan(r#"{"type":"boolean"}"#, 127).with_selected_rows(limits(3)).unwrap(),
        plan(r#"{"type":"boolean"}"#, 127).with_selected_rows(limits(4)).unwrap()];
    let rows = requests(&plans);
    assert!(selected_policy(&rows).unwrap());
    let wrong = Int4ExtractCohortRequest { admitted_identity: rows[0].admitted_identity, ..rows[1] };
    assert!(preflight(&wrong, &vocabulary).is_err());
}

#[test]
fn selected_plan_admission_keeps_every_work_axis_mask_and_control_check() {
    let plans = [plan(r#"{"type":"boolean"}"#, 127).with_selected_rows(limits(3)).unwrap()];
    let requests = requests(&plans);
    let mut vocabulary = ExtractionVocabulary::pinned(crate::tasks::extract::tests::registry().template_controls()).unwrap();
    preflight(&requests[0], &vocabulary).unwrap();
    for axis in 0..8 {
        let mut budget = requests[0].budget;
        match axis {
            0 => budget.native.max_forward_positions -= 1, 1 => budget.native.max_attention_pairs -= 1,
            2 => budget.native.max_projection_work.dot_products -= 1,
            3 => budget.native.max_projection_work.multiply_accumulates -= 1,
            4 => budget.json.max_forward_positions -= 1, 5 => budget.json.max_projected_logits -= 1,
            6 => budget.json.max_total_mask_node_visits = 1,
            _ => budget.json.mask_limits.max_trie_node_visits = 0,
        }
        assert!(preflight(&Int4ExtractCohortRequest { budget, ..requests[0] }, &vocabulary).is_err(), "axis {axis}");
    }
    vocabulary.controls.insert(55);
    assert!(preflight(&requests[0], &vocabulary).is_err());
}

#[test]
fn selected_markers_actual_work_caps_and_group_counts_cannot_bypass_finalization() {
    let plans = [source_plan("Alice", &[]).with_selected_rows(limits(33)).unwrap(),
        source_plan("Bob", &[]).with_selected_rows(limits(3)).unwrap()];
    let requests = requests(&plans); let json = ["\"Alice\"", "\"Bob\""]; let projected = [34, 3];
    for packed in [false, true] {
        for axis in 0..13 {
            let mut run = selected_raw(&requests, &json, &projected, packed);
            match axis {
                0 => run.execution = if packed { INT4_JSON_PACKED_COHORT_EXECUTION } else { INT4_JSON_COHORT_EXECUTION }.to_owned(),
                1 => run.planned_work.projected_logits += 1, 2 => run.model_work.attention_pairs += 1,
                3 => run.group_steps = 0, 4 => run.group_steps = run.model_work.forward_positions + 1,
                5 => run.projection_groups = 3, 6 => run.projection_groups = 6,
                7 => run.sequences[0].execution = INT4_JSON_EXECUTION.to_owned(),
                8 => run.sequences[0].model_work.projections.multiply_accumulates += 1,
                9 => run.sequences[0].output.projected_logits = 100,
                10 => { run.sequences[1].output.token_ids.pop(); },
                11 => run.sequences[1].output.json = "\"Mallory\"".to_owned(),
                _ => run.sequences.swap(0, 1),
            }
            assert!(finish_scheduled(&requests, run, 1 << 20, packed.then(prefill)).is_err(), "packed {packed}, axis {axis}");
        }
        finish_scheduled(&requests, selected_raw(&requests, &json, &projected, packed), 1 << 20, packed.then(prefill)).unwrap();
    }
}

#[test]
fn selected_complete_envelopes_are_exact_bounded_and_do_not_export_private_identity() {
    let plans = [source_plan("Alice Alice", &[]).with_selected_rows(limits(3)).unwrap()];
    let requests = requests(&plans);
    for packed in [false, true] {
        let run = selected_raw(&requests, &["\"Alice\""], &[3], packed);
        let output = finish_scheduled(&requests, run.clone(), 1 << 20, packed.then(prefill)).unwrap();
        let bytes = canonjson::canonical_bytes(&output).unwrap(); let cap = bytes.len() as u64;
        finish_scheduled(&requests, run.clone(), cap, packed.then(prefill)).unwrap();
        assert!(finish_scheduled(&requests, run, cap - 1, packed.then(prefill)).is_err());
        let text = String::from_utf8(bytes).unwrap();
        for private in ["prompt_digest", "taskir_digest", "decision_policy_digest", "confidence"] { assert!(!text.contains(private)); }
    }
}
