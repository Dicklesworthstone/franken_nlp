//! UNRUN owned runtime contracts; these fixtures do not fabricate native NLP.
use super::*;
use super::super::tests::{command, lifetime, NoIo};
use crate::jobs::{JobId, JobProgress};

#[test]
fn runtime_rechecks_storage_consent_before_host_schema_model_or_population() {
    let (mut c, args) = command("resume", &["--source-membership"]);
    let map = c.args.validate(&args).unwrap(); c.args.storage.store_results = false;
    assert_eq!(execute(c.mode, args, c.args, map, NoIo, &mut NoIo).unwrap_err().code,
        "explicit_extraction_document_job_storage");
}
#[test]
fn exact_schema_request_uses_the_same_native_context_and_grounding_policy() {
    for grounding in [false, true] {
        let extra = if grounding { vec!["--source-membership"] } else { Vec::new() };
        let (c, args) = command("resume", &extra); let map = c.args.validate(&args).unwrap();
        let budget = c.args.map.extraction.budget(&args, map.common);
        let schema = if grounding { " {\"type\":\"string\",\"const\":\"上海\",\"x-fnlp-source\":\"verbatim\"}\n" }
            else { " {\"type\":\"boolean\"}\n" };
        let request = c.args.map.request(schema.to_owned(), budget).unwrap();
        assert_eq!(request.schema, schema); assert_eq!(request.budget.max_kv_bytes, map.common.native.kv_bytes);
        assert_eq!(request.budget.max_output_tokens, budget.max_output_tokens);
        assert_eq!(request.grounding == crate::batch::extract::ExtractionBatchGrounding::SourceMembership, grounding);
    }
}
#[test]
fn complete_population_memory_quote_matches_host_and_preserves_invocation_limits() {
    let (c, args) = command("resume", &[]); let map = c.args.validate(&args).unwrap();
    let fixed = lifetime(&args, map); let limits = c.args.lifetime_limits(&args, map, fixed).unwrap();
    let host = host_limits(&args, limits, Duration::from_secs(30));
    check_host_quote(host, fixed, limits.buffer_bytes).unwrap();
    assert_eq!(check_host_quote(host, fixed, limits.buffer_bytes - 1).unwrap_err().code, "job_host_memory_quote");
    let mut invalid = host; invalid.journal_reserve_bytes = 0;
    assert_eq!(check_host_quote(invalid, fixed, limits.buffer_bytes).unwrap_err().code, "job_host_admission");
    assert_eq!(host.native.context_tokens, args.context_tokens);
    assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
    assert_eq!(host.native.run.max_elapsed, Duration::from_secs(30));
    assert_eq!(host.preparation_reserve_bytes, map.common.preparation_bytes);
    assert_eq!(host.transport.max_lines, c.args.storage.max_input_lines);
}
#[test]
fn whole_document_progress_report_does_not_emit_schema_source_values_or_private_paths() {
    let (c, args) = command("resume", &["--source-membership"]);
    let map = c.args.validate(&args).unwrap(); let fixed = lifetime(&args, map); let id = JobId([0x12; 16]);
    let progress = JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
        reserved_work: fixed.max_work, spool_bytes: 1024, materialized: true };
    let report = progress_report(progress, id, fixed, true,
        RunMode::Resume { discard_uncommitted: true }, "extract").unwrap();
    let content = "ab".repeat(32); let source = "cd".repeat(32);
    let response = Response {
        protocol: "fnlp-candidate-int4-extract-document-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "extract",
        durable_unit: "complete_original_document", extraction_mode: "independent_schema_chunk_values",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture-public-revision", source_root_sha256: &source,
        model_content_sha256: &content, candidate_format: "fixture-candidate-format", quant_recipe: "fixture-grouped-int4",
        group_width: args.group_width, integer_kernel: map.common.kernel.kind().id(),
        snapshot: Some(SnapshotRecord::new("loaded", map.common.snapshot_need)), output: &report,
    };
    let mut bytes = Vec::new(); publish(&response, job_cli::REPORT_BYTES, &mut bytes).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["protocol"], "fnlp-candidate-int4-extract-document-job-v1");
    assert_eq!(value["durable_unit"], "complete_original_document");
    assert_eq!(value["extraction_mode"], "independent_schema_chunk_values");
    assert_eq!(value["model_content_sha256"], content); assert_eq!(value["snapshot"]["format"], INT4_SNAPSHOT_VERSION);
    assert_eq!(value["output"]["attempts"], 3); assert_eq!(value["output"]["committed"], 2);
    assert_eq!(value["output"]["reserved_work"], serde_json::to_value(fixed.max_work).unwrap());
    for forbidden in ["text", "schema", "values", "spans", "recipe", "key", "job_dir", "results", "documents"] {
        assert!(value.get(forbidden).is_none()); assert!(value["output"].get(forbidden).is_none());
    }
    assert!(bytes.ends_with(b"\n")); assert_eq!(publish(&response, 1, &mut NoIo), Err(CandidateError::Output));
}
