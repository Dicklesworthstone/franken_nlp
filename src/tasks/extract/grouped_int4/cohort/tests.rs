//! Compiler and independent finalizer regressions, not native-model evidence.
use super::*;
use crate::{native_engine::{constrained::JsonWorkBudget, strict_int4::Int4RunBudget},
    validation::grounded_fields::SourceOccurrence};
use super::super::tests::{plan, source_plan, raw as raw_row};

fn requests(plans: &[Int4ExtractPlan]) -> Vec<Int4ExtractCohortRequest<'_>> {
    plans.iter().map(|plan| {
        let work = plan.planned_work();
        Int4ExtractCohortRequest { prepared: plan, admitted_identity: plan.execution_identity(),
            budget: Int4JsonBudget { native: Int4RunBudget::exact(work),
                json: JsonWorkBudget { max_forward_positions: work.forward_positions,
                    max_projected_logits: work.projected_logits, max_kv_bytes: plan.max_kv_bytes(),
                    max_total_mask_node_visits: 1_000_000_000,
                    mask_limits: crate::grammar::mask::MaskWorkLimits::default() } } }
    }).collect()
}
fn raw(requests: &[Int4ExtractCohortRequest<'_>], json: &[&str]) -> Int4JsonCohortRun {
    assert_eq!(requests.len(), json.len());
    let mut planned_work = Int4Work::default(); let mut model_work = Int4Work::default(); let mut group_steps = 0;
    let sequences = requests.iter().zip(json).map(|(request, json)| {
        let result = raw_row(request.prepared, json);
        planned_work = planned_work.checked_add(request.prepared.planned_work()).unwrap();
        model_work = model_work.checked_add(result.model_work).unwrap();
        group_steps = group_steps.max(result.model_work.forward_positions);
        result
    }).collect();
    Int4JsonCohortRun { schema_version: 1, execution: INT4_JSON_COHORT_EXECUTION.to_owned(),
        sequences, group_steps, projection_groups: 2, planned_work, model_work }
}

#[test]
fn every_document_binds_its_full_identity_and_vocabulary_control_registry() {
    let plans = [plan(r#"{"type":"boolean"}"#, 127)]; let requests = requests(&plans);
    let registry = crate::tasks::extract::tests::registry();
    let mut vocabulary = ExtractionVocabulary::pinned(registry.template_controls()).unwrap();
    check_request(&requests[0], &vocabulary).unwrap();
    let mut changed = plans[0].execution_identity().clone();
    changed.prompt_digest = Sha256Digest::of_bytes(b"a different private prompt");
    let wrong = Int4ExtractCohortRequest { prepared: &plans[0], admitted_identity: &changed, budget: requests[0].budget };
    assert!(check_request(&wrong, &vocabulary).is_err());
    vocabulary.controls.insert(55);
    assert!(check_request(&requests[0], &vocabulary).is_err());
}

#[test]
fn per_document_kv_authority_is_the_meet_of_task_and_invocation_limits() {
    let plans = [plan(r#"{"type":"boolean"}"#, 127)]; let mut requests = requests(&plans);
    requests[0].budget.json.max_kv_bytes = u64::MAX;
    assert_eq!(native_request(&requests[0]).budget.json.max_kv_bytes, plans[0].max_kv_bytes());
    requests[0].budget.json.max_kv_bytes = 123;
    let row = native_request(&requests[0]);
    assert_eq!(row.budget.json.max_kv_bytes, 123);
    assert_eq!(row.identity, plans[0].execution_identity());
    assert_eq!(row.prompt.len(), plans[0].prompt_tokens());
    assert_eq!(row.options.max_new_tokens, plans[0].options().max_new_tokens);
}

#[test]
fn impossible_row_work_and_mask_budgets_fail_preflight_without_native_allocation() {
    let plans = [plan(r#"{"type":"boolean"}"#, 127)]; let requests = requests(&plans);
    let registry = crate::tasks::extract::tests::registry();
    let vocabulary = ExtractionVocabulary::pinned(registry.template_controls()).unwrap();
    preflight(&requests[0], &vocabulary).unwrap();
    for axis in 0..8 {
        let mut budget = requests[0].budget;
        match axis { 0 => budget.native.max_forward_positions -= 1,
            1 => budget.native.max_attention_pairs -= 1,
            2 => budget.native.max_projection_work.dot_products -= 1,
            3 => budget.native.max_projection_work.multiply_accumulates -= 1,
            4 => budget.json.max_forward_positions -= 1,
            5 => budget.json.max_projected_logits -= 1,
            6 => budget.json.max_total_mask_node_visits = 1,
            _ => budget.json.mask_limits.max_trie_node_visits = 0 }
        let request = Int4ExtractCohortRequest { budget, ..requests[0] };
        assert!(preflight(&request, &vocabulary).is_err(), "axis {axis}");
    }
    preflight(&requests[0], &vocabulary).unwrap();
}

#[test]
fn independent_schemas_and_unicode_source_occurrences_survive_in_document_order() {
    let plans = [source_plan("é Alice Alice", &[]), source_plan("Bob", &[]),
        plan(r#"{"type":"boolean"}"#, 127)];
    let requests = requests(&plans); let json = ["\"Alice\"", "\"Bob\"", "true"];
    let output = finish(&requests, raw(&requests, &json), 1 << 20).unwrap();
    assert_eq!(output.sequences.len(), 3);
    for ((plan, json), actual) in plans.iter().zip(json).zip(&output.sequences) {
        assert_eq!(*actual, plan.finalize(raw_row(plan, json)).unwrap());
    }
    let evidence = &output.sequences[0].result.source_fields[0];
    assert_eq!(evidence.occurrence, SourceOccurrence::Ambiguous); assert_eq!(evidence.spans.len(), 2);
    assert_eq!(evidence.spans[0].byte_start, 3); assert_eq!(evidence.spans[0].scalar_start, 2);
    let mut swapped = raw(&requests, &json); swapped.sequences.swap(0, 1);
    assert!(finish(&requests, swapped, 1 << 20).is_err());
}

#[test]
fn exact_decimal_and_integer_domains_do_not_change_under_grouping() {
    let plans = [plan(r#"{"type":"number"}"#, 32), plan(r#"{"type":"integer"}"#, 32)];
    let requests = requests(&plans);
    let decimal = "1.2345678901234567890123456789012345678e37";
    let output = finish(&requests, raw(&requests, &[decimal, "18446744073709551615"]), 1 << 20).unwrap();
    assert_eq!(output.sequences[0].result.output.json, decimal);
    for outside in ["-9223372036854775809", "18446744073709551616", decimal] {
        assert!(finish(&requests, raw(&requests, &[decimal, outside]), 1 << 20).is_err());
    }
}

#[test]
fn later_source_or_schema_failure_discards_the_whole_typed_collection() {
    let plans = [source_plan("Alice", &[]), source_plan("Bob", &[])]; let requests = requests(&plans);
    for invalid in ["\"Mallory\"", "null", "\"Bob"] {
        assert!(finish(&requests, raw(&requests, &["\"Alice\"", invalid]), 1 << 20).is_err());
    }
}

#[test]
fn native_versions_exact_work_and_physical_counters_are_reconciled() {
    let plans = [plan(r#"{"type":"boolean"}"#, 32)]; let requests = requests(&plans);
    for axis in 0..12 {
        let mut run = raw(&requests, &["true"]);
        match axis { 0 => run.schema_version = 2, 1 => run.execution.push('x'),
            2 => { run.sequences.clear(); }, 3 => run.planned_work.projected_logits += 1,
            4 => run.model_work.attention_pairs += 1, 5 => run.group_steps += 1,
            6 => run.projection_groups = 1, 7 => run.projection_groups = 3,
            8 => run.sequences[0].group_width = 127, 9 => run.sequences[0].model_work.projections.dot_products += 1,
            10 => { run.sequences[0].output.token_ids.pop(); }, _ => run.sequences[0].output.token_ids[0] = 3 }
        assert!(finish(&requests, run, 1 << 20).is_err(), "axis {axis}");
    }
}

#[test]
fn complete_outer_size_is_exact_and_no_private_identity_is_exported() {
    let plans = [source_plan("Alice Alice", &[])]; let requests = requests(&plans);
    let before = canonjson::canonical_bytes(plans[0].execution_identity()).unwrap();
    let output = finish(&requests, raw(&requests, &["\"Alice\""]), 1 << 20).unwrap();
    let bytes = canonjson::canonical_bytes(&output).unwrap(); let cap = bytes.len() as u64;
    assert!(finish(&requests, raw(&requests, &["\"Alice\""]), cap).is_ok());
    assert!(finish(&requests, raw(&requests, &["\"Alice\""]), cap - 1).is_err());
    let text = String::from_utf8(bytes).unwrap();
    for private in ["prompt_digest", "taskir_digest", "decision_policy_digest", "confidence"] { assert!(!text.contains(private)); }
    assert_eq!(before, canonjson::canonical_bytes(plans[0].execution_identity()).unwrap());
}
