//! UNRUN durable scheduling and pinned-planning fixtures. No model execution,
//! native parity, throughput or recovery evidence is claimed by these tests.
use super::*;

fn selected_task(index: usize) -> Int4SourceDocumentTask {
    if index < 7 { return task(index); }
    let keyphrases = Int4KeyphraseLimits::default();
    Int4SourceDocumentTask::Keyphrases { options: KeyphraseOptions::default(),
        max_lifetime_scan_work: keyphrases.aggregation.max_scan_work * limits().max_attempts,
        limits: keyphrases }
}
fn assert_peak_fits(actual: Int4SourceMapCohortRequirement, declared: Int4SourceMapCohortRequirement) {
    assert!(actual.native.kv_bytes <= declared.native.kv_bytes);
    assert!(actual.native.rope_bytes <= declared.native.rope_bytes);
    assert!(actual.native.scratch_payload_bound <= declared.native.scratch_payload_bound);
    assert!(actual.workspace_bytes <= declared.workspace_bytes);
    assert!(actual.max_result_bytes <= declared.max_result_bytes);
}

#[test]
fn all_eight_routes_keep_complete_semantic_receipts_and_lifetime_authority() {
    for index in 0..8 {
        let mut map = mapping(); map.reduction.map_batch_chunks = 2;
        let serial = build(selected_task(index), map, limits(), lifetime()).unwrap();
        let original = prepare(&serial, SOURCE);
        let result_bytes = serial.max_result_bytes();
        let retained_tokens = serial.output_token_ceiling().unwrap();
        let cohorts = serial.with_chunk_cohorts().unwrap();
        assert!(cohorts.uses_chunk_cohorts());
        assert!(cohorts.owns(&original).is_err());
        let prepared = prepare(&cohorts, SOURCE);
        assert_eq!(prepared.expected, original.expected, "route {index}");
        assert_eq!(prepared.work, original.work, "route {index}");
        assert_eq!(prepared.identities, original.identities, "route {index}");
        assert_eq!(cohorts.max_result_bytes(), result_bytes);
        assert_eq!(cohorts.output_token_ceiling().unwrap(), retained_tokens);
        let native = cohorts.plan(SOURCE, &mut Continue).unwrap();
        native.compare(&prepared, &mut Continue).unwrap();
        let exact = native.cohort_requirements(cohorts.planning_limits().max_context_tokens, None).unwrap();
        assert_peak_fits(exact, cohorts.cohort_requirements().unwrap());
        // Enabling an already-enabled policy does not invalidate its receipts.
        drop(native);
        cohorts.with_chunk_cohorts().unwrap().owns(&prepared).unwrap();
    }
}

#[test]
fn schedule_and_width_are_keyed_while_serial_recipe_shape_is_unchanged() {
    let serial = factory(6);
    let original = serde_json::to_value(serial.job_recipe()).unwrap();
    assert!(original.get("chunk_cohorts").is_none());
    let binding = freeze(&serial).binding;
    let cohorts = serial.with_chunk_cohorts().unwrap();
    assert!(binding.compare(&freeze(&cohorts).binding).is_err());
    let mut cohort_recipe = serde_json::to_value(cohorts.job_recipe()).unwrap();
    assert_eq!(cohort_recipe.as_object_mut().unwrap().remove("chunk_cohorts").unwrap(),
        "native-document-chunk-cohorts-v1");
    assert_eq!(cohort_recipe, original);
    assert_eq!(cohort_recipe["task"]["request"]["question"]["question"], question().question);
    let mut map = mapping(); map.reduction.map_batch_chunks = 2;
    let wider = build(task(6), map, limits(), lifetime()).unwrap().with_chunk_cohorts().unwrap();
    assert!(freeze(&cohorts).binding.compare(&freeze(&wider).binding).is_err());
    let mut changed_question = question_request(); changed_question.question.question.push('!');
    let changed = build(Int4SourceDocumentTask::QuestionHierarchy { request: changed_question,
        limits: QuestionHierarchyLimits { max_passes: 4, ..QuestionHierarchyLimits::default() } },
        mapping(), limits(), lifetime()).unwrap().with_chunk_cohorts().unwrap();
    assert!(freeze(&cohorts).binding.compare(&freeze(&changed).binding).is_err());
}

#[test]
fn sparse_question_windows_keep_their_native_rows_and_complete_pass_debit() {
    let source = "Alice   \n        \nBob";
    for index in [2, 5, 6] {
        let mut map = mapping(); map.reduction.map_batch_chunks = 4;
        let serial = build(task(index), map, limits(), lifetime()).unwrap();
        let original = prepare(&serial, source);
        let cohorts = serial.with_chunk_cohorts().unwrap();
        let prepared = prepare(&cohorts, source);
        assert!(prepared.chunk_count() > prepared.execution_identities().len());
        assert_eq!(prepared.expected, original.expected);
        assert_eq!(prepared.work, original.work);
        assert_eq!(prepared.identities, original.identities);
        let exact = cohorts.plan(source, &mut Continue).unwrap()
            .cohort_requirements(cohorts.planning_limits().max_context_tokens, None).unwrap();
        let declared = cohorts.cohort_requirements().unwrap();
        assert_peak_fits(exact, declared);
        assert!(exact.native.kv_bytes < declared.native.kv_bytes);
        assert!(cohorts.prepare_with_control(document(" \n\t  "), &mut Continue).is_err());
    }
}

#[test]
fn invalid_declared_width_or_original_task_kv_refuses_before_document_preparation() {
    let mut map = mapping(); map.reduction.map_batch_chunks = map.chunks.max_chunks + 1;
    assert!(build(task(0), map, limits(), lifetime()).unwrap().with_chunk_cohorts().is_err());
    let mut map = mapping(); map.reduction.map_batch_chunks = 65; map.chunks.max_chunks = 65;
    map.max_mask_visits = map.mask_visits_per_chunk * map.chunks.max_chunks as u64;
    let mut job = limits(); job.max_work.mask_node_visits = map.max_mask_visits * job.max_attempts;
    assert!(build(task(0), map, job, lifetime()).unwrap().with_chunk_cohorts().is_err());
    let mut planner = factory(5); planner.budget.max_kv_bytes -= 1;
    assert!(planner.with_chunk_cohorts().is_err());
}

#[test]
fn hierarchy_windows_preserve_semantic_debits_but_invalidate_old_schedule_receipts() {
    for index in [4, 6] {
        let serial = factory(index);
        let old = prepare(&serial, SOURCE);
        let bytes = serial.max_result_bytes();
        let tokens = serial.output_token_ceiling().unwrap();
        let grouped = serial.with_hierarchy_cohorts(3).unwrap();
        assert!(grouped.uses_native_cohorts());
        assert!(!grouped.uses_chunk_cohorts());
        assert!(grouped.owns(&old).is_err());
        let prepared = prepare(&grouped, SOURCE);
        assert_eq!(prepared.expected, old.expected);
        assert_eq!(prepared.work, old.work);
        assert_eq!(prepared.identities, old.identities);
        assert_eq!(grouped.max_result_bytes(), bytes);
        assert_eq!(grouped.output_token_ceiling().unwrap(), tokens);
        let plan = grouped.plan(SOURCE, &mut Continue).unwrap();
        let exact = plan.cohort_requirements(grouped.planning_limits().max_context_tokens,
            grouped.hierarchy_cohort_schedule()).unwrap();
        assert_eq!(exact, grouped.cohort_requirements().unwrap());
        drop(plan);
        let grouped = grouped.with_hierarchy_cohorts(3).unwrap();
        grouped.owns(&prepared).unwrap();
        assert!(grouped.with_hierarchy_cohorts(2).unwrap().owns(&prepared).is_err());
    }
}

#[test]
fn hierarchy_schedule_is_keyed_independently_of_discovery_and_private_question() {
    let serial = factory(6);
    let original = serde_json::to_value(serial.job_recipe()).unwrap();
    assert!(original.get("hierarchy_cohorts").is_none());
    let binding = freeze(&serial).binding;
    let grouped = serial.with_hierarchy_cohorts(2).unwrap();
    let grouped_binding = freeze(&grouped).binding;
    assert!(binding.compare(&grouped_binding).is_err());
    let mut recipe = serde_json::to_value(grouped.job_recipe()).unwrap();
    let schedule = recipe.as_object_mut().unwrap().remove("hierarchy_cohorts").unwrap();
    assert_eq!(schedule["execution"], "native-document-hierarchy-group-cohorts-v1");
    assert_eq!(schedule["max_groups"], 2);
    assert_eq!(recipe, original);
    assert!(grouped_binding.compare(&freeze(&factory(6).with_hierarchy_cohorts(3).unwrap()).binding).is_err());
    let both = grouped.with_chunk_cohorts().unwrap();
    assert!(grouped_binding.compare(&freeze(&both).binding).is_err());
    assert_eq!(both.hierarchy_cohort_schedule(), Some(Int4HierarchyCohortSchedule {
        max_groups: 2, discovery_cohorts: true }));
    let reversed = factory(6).with_chunk_cohorts().unwrap().with_hierarchy_cohorts(2).unwrap();
    assert_eq!(serde_json::to_value(both.job_recipe()).unwrap(), serde_json::to_value(reversed.job_recipe()).unwrap());
    assert_eq!(both.cohort_requirements().unwrap(), reversed.cohort_requirements().unwrap());
    let prepared = prepare(&both, SOURCE);
    both.with_chunk_cohorts().unwrap().owns(&prepared).unwrap();
}

#[test]
fn combined_discovery_and_hierarchy_admit_the_larger_phase_and_sparse_question_rows() {
    for index in [4, 6] {
        let mut map = mapping(); map.reduction.map_batch_chunks = 4;
        let serial = build(task(index), map, limits(), lifetime()).unwrap();
        let discovery = serial.discovery_cohort_requirements().unwrap();
        let grouped = serial.with_hierarchy_cohorts(2).unwrap();
        let groups = grouped.cohort_requirements().unwrap();
        let both = grouped.with_chunk_cohorts().unwrap();
        let peak = both.cohort_requirements().unwrap();
        assert_peak_fits(discovery, peak);
        assert_peak_fits(groups, peak);
        assert_eq!(peak.native.kv_bytes, discovery.native.kv_bytes.max(groups.native.kv_bytes));
        let plan = both.plan("Alice   \n        \nBob", &mut Continue).unwrap();
        let actual = plan.cohort_requirements(both.planning_limits().max_context_tokens,
            both.hierarchy_cohort_schedule()).unwrap();
        assert_peak_fits(actual, peak);
    }
}

#[test]
fn hierarchy_only_schedule_rejects_other_tasks_and_unfunded_widths() {
    for index in [0, 1, 2, 3, 5] { assert!(factory(index).with_hierarchy_cohorts(2).is_err()); }
    for width in [0, 5, 65, usize::MAX] { assert!(factory(4).with_hierarchy_cohorts(width).is_err()); }
    let mut planner = factory(6); planner.budget.max_kv_bytes -= 1;
    assert!(planner.with_hierarchy_cohorts(2).is_err());
    let map = factory(0).plan(SOURCE, &mut Continue).unwrap();
    assert!(map.cohort_requirements(8192, Some(Int4HierarchyCohortSchedule {
        max_groups: 2, discovery_cohorts: false })).is_err());
}
