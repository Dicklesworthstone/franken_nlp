//! Independent semantic finalizers over synthetic native records, not inference.
use super::*;
use crate::native_engine::{constrained::JsonWorkBudget, strict_int4::Int4RunBudget};
use super::super::tests::{planner, prepare, ner, keyphrases, summary, answer, synthetic};

fn fixtures() -> Vec<(SourceTaskRequest, &'static str, &'static str)> {
    vec![
        (ner("é上海 and 上海"), "é上海 and 上海", r#"[{"text":"上海","type":"location"}]"#),
        (keyphrases("Rust compiler Rust"), "Rust compiler Rust", r#"["compiler","Rust","Rust"]"#),
        (summary("Alice"), "Alice", r#"[{"citations":["Alice"],"text":"An unverified assertion."}]"#),
        (answer(&[("one", "é上海"), ("two", "上海")]), "é上海\n\n上海",
            r#"{"answer":"A location.","answerable":true,"citations":["上海"]}"#),
    ]
}
fn requests(plans: &[PreparedInt4SourceTask]) -> Vec<Int4SourceCohortRequest<'_>> {
    plans.iter().map(|plan| {
        let work = plan.planned_work();
        Int4SourceCohortRequest { prepared: plan, admitted_identity: plan.execution_identity(),
            budget: Int4JsonBudget { native: Int4RunBudget::exact(work),
                json: JsonWorkBudget { max_forward_positions: work.forward_positions,
                    max_projected_logits: work.projected_logits, max_kv_bytes: plan.max_kv_bytes(),
                    max_total_mask_node_visits: 1_000_000_000,
                    mask_limits: crate::grammar::mask::MaskWorkLimits::default() } } }
    }).collect()
}
fn raw(plans: &[PreparedInt4SourceTask], fixtures: &[(SourceTaskRequest, &str, &str)]) -> Int4ExtractCohortRun {
    let mut planned_work = Int4Work::default(); let mut model_work = Int4Work::default(); let mut group_steps = 0;
    let sequences = plans.iter().zip(fixtures).map(|(plan, (request, source, json))| {
        let output = synthetic(plan, request, source, json);
        planned_work = planned_work.checked_add(plan.planned_work()).unwrap();
        model_work = model_work.checked_add(output.model_work).unwrap();
        group_steps = group_steps.max(output.model_work.forward_positions);
        output
    }).collect();
    Int4ExtractCohortRun { schema_version: 1, execution: INT4_EXTRACT_COHORT_EXECUTION.to_owned(),
        sequences, group_steps, projection_groups: 2, planned_work, model_work }
}

#[test]
fn every_source_task_preflights_identity_and_work_without_allocating_an_engine() {
    let planner = planner(); let fixtures = fixtures();
    let plans: Vec<_> = fixtures.iter().map(|(request, _, _)| prepare(&planner, request)).collect();
    let requests = requests(&plans);
    let vocabulary = ExtractionVocabulary::pinned(&planner.controls).unwrap();
    for request in &requests {
        preflight(request, &vocabulary).unwrap();
        let mut wrong = request.admitted_identity.clone();
        wrong.prompt_digest = Sha256Digest::of_bytes(b"another source-task prompt");
        let changed = Int4SourceCohortRequest { admitted_identity: &wrong,
            prepared: request.prepared, budget: request.budget };
        assert!(preflight(&changed, &vocabulary).is_err());
        for axis in 0..3 {
            let mut budget = request.budget;
            match axis { 0 => budget.native.max_forward_positions -= 1,
                1 => budget.json.max_total_mask_node_visits = 1,
                _ => budget.json.mask_limits.max_trie_node_visits = 0 }
            let changed = Int4SourceCohortRequest { prepared: request.prepared,
                admitted_identity: request.admitted_identity, budget };
            assert!(preflight(&changed, &vocabulary).is_err(), "axis {axis}");
        }
        preflight(request, &vocabulary).unwrap();
    }
}

#[test]
fn all_four_original_semantic_finalizers_remain_independent_and_ordered() {
    let planner = planner(); let fixtures = fixtures();
    let plans: Vec<_> = fixtures.iter().map(|(request, _, _)| prepare(&planner, request)).collect();
    let output = finish(&requests(&plans), raw(&plans, &fixtures), 8 << 20).unwrap();
    assert_eq!(output.sequences.len(), 4);
    for ((plan, (request, source, json)), actual) in plans.iter().zip(&fixtures).zip(&output.sequences) {
        assert_eq!(*actual, plan.finish(synthetic(plan, request, source, json)).unwrap());
        plan.verify_identity(plan.execution_identity()).unwrap();
    }
}

#[test]
fn late_semantic_refusal_yields_no_partial_collection() {
    let planner = planner(); let mut fixtures = fixtures();
    let plans: Vec<_> = fixtures.iter().map(|(request, _, _)| prepare(&planner, request)).collect();
    fixtures[2].2 = r#"[{"citations":[],"text":"A claim."}]"#;
    assert!(finish(&requests(&plans), raw(&plans, &fixtures), 8 << 20).is_err());
}

#[test]
fn cross_task_swaps_work_drift_and_impossible_group_counts_fail_closed() {
    let planner = planner(); let fixtures = fixtures();
    let plans: Vec<_> = fixtures.iter().map(|(request, _, _)| prepare(&planner, request)).collect();
    for axis in 0..8 {
        let mut raw = raw(&plans, &fixtures);
        match axis { 0 => raw.sequences.swap(0, 1), 1 => raw.planned_work.projected_logits += 1,
            2 => raw.model_work.attention_pairs += 1, 3 => raw.group_steps += 1,
            4 => raw.projection_groups = 1, 5 => raw.projection_groups = 9,
            6 => raw.execution.push('x'), _ => raw.sequences[3].group_width = 127 }
        assert!(finish(&requests(&plans), raw, 8 << 20).is_err(), "axis {axis}");
    }
}

#[test]
fn final_document_identity_drift_is_not_repaired_by_grouping() {
    let planner = planner(); let fixtures = fixtures();
    let plans: Vec<_> = fixtures.iter().map(|(request, _, _)| prepare(&planner, request)).collect();
    let mut requests = requests(&plans);
    let mut wrong = plans[3].execution_identity().clone(); wrong.schema_digest = Sha256Digest::of_bytes(b"other schema");
    requests[3].admitted_identity = &wrong;
    assert!(finish(&requests, raw(&plans, &fixtures), 8 << 20).is_err());
}

#[test]
fn every_source_occurrence_and_group_counter_counts_toward_the_complete_output_bound() {
    let planner = planner(); let fixtures = fixtures();
    let plans: Vec<_> = fixtures.iter().map(|(request, _, _)| prepare(&planner, request)).collect();
    let requests = requests(&plans);
    let output = finish(&requests, raw(&plans, &fixtures), 8 << 20).unwrap();
    let cap = canonjson::canonical_bytes(&output).unwrap().len() as u64;
    assert!(finish(&requests, raw(&plans, &fixtures), cap).is_ok());
    assert!(finish(&requests, raw(&plans, &fixtures), cap - 1).is_err());
}
