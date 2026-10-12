//! Closed configuration and quota tests without a model or job filesystem.
use super::*;
use crate::candidate_cli::jobs::tests as job_tests;
fn lifetime() -> JobLimits { let mut l = job_tests::lifetime(); l.max_result_bytes = 4 << 20; l }
fn root(task: &str) -> serde_json::Value {
    let aggregation = if task == "summarize" {
        serde_json::json!({"max_unique_bullets":256,"max_citations":1024,"max_evidence_spans":4096,
            "max_value_bytes":1048576,"max_scan_steps":100000})
    } else {
        serde_json::json!({"max_unique_phrases":256,"max_evidence_spans":4096,
            "max_value_bytes":1048576,"max_scan_work":100000})
    };
    let mut reduction = serde_json::json!({"schema_version":1,"task":task,"aggregation":aggregation,
        "max_result_bytes":2097152,"max_lifetime_scan_work":100000 * lifetime().max_attempts});
    reduction[if task == "summarize" { "max_bullets" } else { "max_phrases" }] = serde_json::json!(8);
    serde_json::json!({"schema_version":1,"max_chunks":8,"max_chunk_bytes":8,"max_tokenizer_calls":1024,
        "max_result_bytes":4194304,"max_live_value_bytes":16777216,"max_total_value_bytes":67108864,
        "reduction_reserve_mib":64,"reduction":reduction})
}
fn load_root(task: &str, root: &serde_json::Value, lifetime: JobLimits) -> Result<(Int8SourceMapLimits, u64), Failure> {
    let args = job_tests::args(task, &["--source-document-config", "d", "--preparation-mib", "1024"]);
    let (_, limits) = args.validate()?;
    super::super::load(&root.to_string(), &args, args.host.task_budget(limits), lifetime, limits)
}
#[test]
fn both_reduction_families_parse_and_keep_native_lifetime_work_unchanged() {
    for task in ["summarize", "keyphrases"] {
        let value = root(task); let (mapping, _) = load_root(task, &value, lifetime()).unwrap();
        assert_eq!(mapping.max_model_work, lifetime().max_work.model);
        let reduction = super::super::load_reduction(&value.to_string(), task, mapping, lifetime()).unwrap().unwrap();
        match reduction {
            PreparedReduction::Summary { limits, max_lifetime_scan_work } => {
                assert_eq!(task, "summarize"); assert_eq!(limits.max_bullets, 8);
                assert_eq!(max_lifetime_scan_work, limits.aggregation.max_scan_steps * lifetime().max_attempts);
            }
            PreparedReduction::Keyphrases { limits, max_lifetime_scan_work } => {
                assert_eq!(task, "keyphrases"); assert_eq!(limits.max_phrases, 8);
                assert_eq!(max_lifetime_scan_work, limits.aggregation.max_scan_work * lifetime().max_attempts);
            }
        }
    }
}
#[test]
fn only_absence_selects_legacy_maps_and_explicit_null_is_refused() {
    let mut old = root("summarize"); old.as_object_mut().unwrap().remove("reduction");
    let (mapping, _) = load_root("summarize", &old, lifetime()).unwrap();
    assert!(super::super::load_reduction(&old.to_string(), "summarize", mapping, lifetime()).unwrap().is_none());
    let parsed = super::super::parse(&old.to_string()).unwrap();
    assert!(!serde_json::to_value(parsed).unwrap().as_object().unwrap().contains_key("reduction"));
    for invalid in [serde_json::Value::Null, serde_json::json!({}), serde_json::json!([]), serde_json::json!(true)] {
        let mut bad = old.clone(); bad["reduction"] = invalid;
        assert!(super::super::parse(&bad.to_string()).is_err());
    }
}
#[test]
fn partial_unknown_and_duplicate_nested_reduction_fields_are_rejected() {
    for task in ["summarize", "keyphrases"] {
        let value = root(task);
        let keys: Vec<_> = value["reduction"].as_object().unwrap().keys().cloned().collect();
        for key in keys {
            let mut bad = value.clone(); bad["reduction"].as_object_mut().unwrap().remove(&key);
            assert!(super::super::parse(&bad.to_string()).is_err(), "missing {key}");
        }
        for key in ["model", "max_attempts", "max_work", "schema"] {
            let mut bad = value.clone(); bad["reduction"][key] = serde_json::json!(1);
            assert!(super::super::parse(&bad.to_string()).is_err());
            let mut bad = value.clone(); bad["reduction"]["aggregation"][key] = serde_json::json!(1);
            assert!(super::super::parse(&bad.to_string()).is_err());
        }
        let needle = if task == "summarize" { "\"max_bullets\":8" } else { "\"max_phrases\":8" };
        let text = value.to_string(); assert!(text.contains(needle));
        assert!(super::super::parse(&text.replacen(needle, &format!("{needle},{needle}"), 1)).is_err());
    }
}
#[test]
fn a_reducer_cannot_change_the_jobs_fixed_task_or_version() {
    let summary = root("summarize"); let phrases = root("keyphrases");
    assert!(load_root("keyphrases", &summary, lifetime()).is_err());
    assert!(load_root("summarize", &phrases, lifetime()).is_err());
    assert!(load_root("ner", &summary, lifetime()).is_err());
    let mut changed = summary; changed["reduction"]["schema_version"] = serde_json::json!(2);
    assert!(load_root("summarize", &changed, lifetime()).is_err());
}
#[test]
fn the_entire_attempt_lifetime_must_fit_verification_work() {
    for task in ["summarize", "keyphrases"] {
        let mut value = root(task);
        value["reduction"]["max_lifetime_scan_work"] = serde_json::json!(100000 * lifetime().max_attempts - 1);
        assert_eq!(load_root(task, &value, lifetime()).err().unwrap().code, "source_document_reduction_lifetime");
        let mut job = lifetime(); job.max_attempts = u64::MAX;
        assert!(load_root(task, &root(task), job).is_err());
    }
}
#[test]
fn top_k_and_aggregate_limits_cannot_be_used_to_bypass_output_bounds() {
    for task in ["summarize", "keyphrases"] {
        let item_key = if task == "summarize" { "max_bullets" } else { "max_phrases" };
        let scan_key = if task == "summarize" { "max_scan_steps" } else { "max_scan_work" };
        for (key, value) in [(item_key, 0), (item_key, 65537), ("max_result_bytes", 0), ("max_result_bytes", 4194305)] {
            let mut bad = root(task); bad["reduction"][key] = serde_json::json!(value);
            assert!(load_root(task, &bad, lifetime()).is_err());
        }
        for (key, value) in [(scan_key, 0), ("max_value_bytes", 4194305), ("max_evidence_spans", 0)] {
            let mut bad = root(task); bad["reduction"][item_key] = serde_json::json!(1);
            bad["reduction"]["aggregation"][key] = serde_json::json!(value);
            assert!(load_root(task, &bad, lifetime()).is_err());
        }
    }
}
#[test]
fn final_selection_does_not_change_chunking_or_native_reservations() {
    for task in ["summarize", "keyphrases"] {
        let mut changed = root(task); let (before, reserve) = load_root(task, &changed, lifetime()).unwrap();
        changed["reduction"][if task == "summarize" { "max_bullets" } else { "max_phrases" }] = serde_json::json!(1);
        let (after, same_reserve) = load_root(task, &changed, lifetime()).unwrap();
        assert_eq!(before.chunks, after.chunks); assert_eq!(before.reduction, after.reduction);
        assert_eq!(before.max_model_work, after.max_model_work); assert_eq!(before.max_mask_visits, after.max_mask_visits);
        assert_eq!(reserve, same_reserve);
    }
}
