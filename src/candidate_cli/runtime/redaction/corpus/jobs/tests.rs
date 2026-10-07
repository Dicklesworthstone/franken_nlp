//! Pinned configuration and metadata checks only, not model execution.
use super::*;
use crate::candidate_cli::redact::retention::tests::{command, lifetime};
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".into(), revision: "f56ec5a9650268aa098496734743c25ea778bd2d".into(),
        recipe_id: "retained-redaction-planning-fixture".into(), source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
#[test]
fn live_and_retained_modes_build_the_same_pinned_detector_and_rule_scope() {
    let (p, _) = planner().unwrap();
    for chunked in [false, true] {
        let extras: &[&str] = if chunked { &["--chunked"] } else { &[] };
        let retained = command(extras); let mut args = vec!["--ndjson"]; args.extend_from_slice(extras);
        let live = crate::candidate_cli::redact::tests::command(&args);
        let (_, a) = live.validate().unwrap(); let (_, b) = retained.validate().unwrap();
        let identity = source_identity(&facts(), &p, BuiltInTask::Ner).unwrap();
        let left = prepare(&live, a, live.corpus.envelope(&live, a).unwrap(), &p, identity.clone(), NerOptions::default(), live.request()).unwrap();
        let right = prepare(&retained, b, retained.corpus.envelope(&retained, b).unwrap(), &p, identity, NerOptions::default(), retained.request()).unwrap();
        match (left, right) {
            (PreparedCorpus::Short(x), PreparedCorpus::Short(y)) => {
                assert_eq!(x.ner_identity, y.ner_identity); assert_eq!(x.max_model_work, y.max_model_work);
                assert_eq!(x.detector.per_pass, y.detector.per_pass); assert_eq!(x.detector.ner, y.detector.ner);
                assert_eq!(canonjson::canonical_bytes(&x.request).unwrap(), canonjson::canonical_bytes(&y.request).unwrap());
            }
            (PreparedCorpus::Document(x), PreparedCorpus::Document(y)) => {
                assert_eq!(x.ner_identity, y.ner_identity); assert_eq!(x.item_model_work(), y.item_model_work());
                assert_eq!(x.detector.mapping.chunks, y.detector.mapping.chunks);
                assert_eq!(x.detector.mapping.reduction, y.detector.mapping.reduction);
                assert_eq!(x.detector.mapping.max_mask_visits, y.detector.mapping.max_mask_visits);
                assert_eq!(canonjson::canonical_bytes(&x.request).unwrap(), canonjson::canonical_bytes(&y.request).unwrap());
            }
            _ => panic!("retention changed the native document mode"),
        }
    }
}
fn progress(limits: JobLimits, id: JobId) -> JobProgress {
    JobProgress { job_id: id, items: 2, committed: 2, attempts: 3, reserved_work: limits.max_work,
        spool_bytes: 8192, materialized: true }
}
#[test]
fn metadata_contains_lifetime_work_and_sensitivity_but_no_redacted_documents_or_keys() {
    let c = command(&[]); let limits = lifetime(&c); let id = c.retention.job_id.unwrap();
    let output = report(progress(limits, id), id, limits, true, true, true).unwrap();
    let value = serde_json::to_value(output).unwrap();
    assert_eq!(value["operation"], "resume"); assert_eq!(value["attempts"], 3);
    assert_eq!(value["redaction_mode"], "chunked"); assert_eq!(value["retained_output_is_sensitive"], true);
    assert_eq!(value["reserved_work"]["mask_node_visits"], limits.max_work.mask_node_visits);
    let text = value.to_string();
    for private in ["redacted_text", "entities", "key_id", "key_commitment", "scope_commitment", "namespace", "source_text"] {
        assert!(!text.contains(private));
    }
}
#[test]
fn incomplete_or_divergent_job_progress_is_not_a_success_report() {
    let c = command(&[]); let limits = lifetime(&c); let id = c.retention.job_id.unwrap();
    for axis in 0..7 {
        let mut p = progress(limits, id);
        match axis { 0 => p.job_id = JobId([4; 16]), 1 => p.items = 0, 2 => p.committed -= 1,
            3 => p.attempts = 1, 4 => p.attempts = limits.max_attempts + 1,
            5 => p.reserved_work.mask_node_visits += 1, _ => p.materialized = false }
        assert!(report(p, id, limits, false, true, false).is_err());
    }
}
