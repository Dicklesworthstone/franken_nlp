//! UNRUN owned runtime contracts; these fixtures do not fabricate native NLP.
use super::*;
use super::super::tests::{command, lifetime, NoIo};
use crate::jobs::{JobId, JobProgress};

#[test]
fn runtime_rechecks_storage_consent_before_host_schema_model_or_population() {
    let (mut c, args) = command("resume", &["--source-membership", "--cohort-chunks", "4"]);
    let map = c.args.validate(&args).unwrap(); c.args.storage.store_results = false;
    assert_eq!(execute(c.mode, args, c.args, map, NoIo, &mut NoIo).unwrap_err().code,
        "explicit_extraction_document_job_storage");
}
#[test]
fn runtime_rechecks_cohort_memory_before_host_schema_or_population_io() {
    let (c, mut args) = command("resume", &["--cohort-chunks", "4"]);
    let map = c.args.validate(&args).unwrap();
    args.memory_mib = map.common.minimum_memory_bytes.div_ceil(MIB) - 1;
    assert_eq!(execute(c.mode, args, c.args, map, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
}

#[test]
fn changing_physical_team_width_on_resume_preserves_the_complete_authenticated_recipe() {
    use crate::native_engine::strict_int4::weights::projection_fixture;
    // Real pinned schema compilation and recipe serialization, without native
    // execution or private job IO. Semantic policy changes remain distinct.
    let mut recipes = Vec::new();
    for flags in [&[][..], &["--max-selected-rows", "32"][..],
        &["--input-encoding", "safe-bpe-v1"][..]] {
        let mut expected = None;
        for operation in ["start", "resume"] {
            for width in [1, 2, 4, 64] {
                let width = width.to_string();
                let mut options = flags.to_vec(); options.extend(["--compute-threads", &width]);
                let (c, args) = command(operation, &options);
                let map = c.args.validate(&args).unwrap();
                let fixed = lifetime(&args, map);
                let limits = c.args.lifetime_limits(&args, map, fixed).unwrap();
                let budget = c.args.map.extraction.budget(&args, map.common);
                let request = c.args.map.request(" {\"type\":\"boolean\"}\n".to_owned(), budget).unwrap();
                let (weights, _) = projection_fixture(args.group_width);
                let planner = Int4DocumentExtractionJobPlanner::for_model(weights.identity(), budget,
                    c.args.map.extraction.compiler(), c.args.map.extraction.source_limits(),
                    request, map.mapping, fixed, c.args.lifetime.verification()).unwrap()
                    .with_document_encoding(c.args.map.encoding.policy().unwrap()).unwrap();
                let planner = match c.args.map.extraction.selected_rows().unwrap() {
                    Some(rows) => planner.with_selected_rows(rows).unwrap(), None => planner,
                };
                planner.check_job_limits(fixed).unwrap();
                let recipe = crate::canonjson::canonical_bytes(planner.job_recipe()).unwrap();
                let identity = crate::canonjson::canonical_bytes(planner.execution_identity()).unwrap();
                let semantic = (recipe, identity, fixed.max_work, limits.buffer_bytes);
                if let Some(expected) = &expected { assert_eq!(&semantic, expected); }
                else { expected = Some(semantic); }
            }
        }
        recipes.push(expected.unwrap().0);
    }
    assert_ne!(recipes[0], recipes[1], "selected-row policy remains authenticated");
    assert_ne!(recipes[0], recipes[2], "compact encoding remains authenticated");
}
#[test]
fn exact_schema_request_uses_the_same_native_context_and_grounding_policy() {
    for grounding in [false, true] {
        for width in [None, Some("4")] {
            let mut extra = if grounding { vec!["--source-membership"] } else { Vec::new() };
            if let Some(width) = width { extra.extend(["--cohort-chunks", width]); }
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
}
#[test]
fn complete_population_memory_quote_matches_host_and_preserves_invocation_limits() {
    for extra in [&[][..], &["--cohort-chunks", "4"][..]] {
        let (c, args) = command("resume", extra); let map = c.args.validate(&args).unwrap();
        let fixed = lifetime(&args, map); let limits = c.args.lifetime_limits(&args, map, fixed).unwrap();
        let host = host_limits(&args, limits, Duration::from_secs(30));
        check_host_quote(host, fixed, limits.buffer_bytes, JobInputMemory::CompletePopulation).unwrap();
        assert_eq!(check_host_quote(host, fixed, limits.buffer_bytes - 1, JobInputMemory::CompletePopulation)
            .unwrap_err().code, "job_host_memory_quote");
        let mut invalid = host; invalid.journal_reserve_bytes = 0;
        assert_eq!(check_host_quote(invalid, fixed, limits.buffer_bytes, JobInputMemory::CompletePopulation)
            .unwrap_err().code, "job_host_admission");
        assert_eq!(host.native.context_tokens, args.context_tokens);
        assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
        assert_eq!(host.native.run.max_elapsed, Duration::from_secs(30));
        assert_eq!(host.preparation_reserve_bytes, map.common.preparation_bytes);
        assert_eq!(host.transport.max_lines, c.args.storage.max_input_lines);
    }
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
    let mut response = Response {
        protocol: "fnlp-candidate-int4-extract-document-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "extract",
        durable_unit: "complete_original_document", extraction_mode: "independent_schema_chunk_values",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture-public-revision", source_root_sha256: &source,
        model_content_sha256: &content, candidate_format: "fixture-candidate-format", quant_recipe: "fixture-grouped-int4",
        group_width: args.group_width, integer_kernel: map.common.kernel.kind().id(),
        snapshot: Some(SnapshotRecord::new("loaded", map.common.snapshot_need)), chunk_schedule: None, output: &report,
        root_array_union: None, max_selected_rows: None, root_object_reconciliation: None, input_encoding: None,
        compute_schedule: None,
    };
    let mut bytes = Vec::new(); publish(&response, job_cli::REPORT_BYTES, &mut bytes).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["protocol"], "fnlp-candidate-int4-extract-document-job-v1");
    assert_eq!(value["durable_unit"], "complete_original_document");
    assert_eq!(value["extraction_mode"], "independent_schema_chunk_values");
    assert_eq!(value["model_content_sha256"], content); assert_eq!(value["snapshot"]["format"], INT4_SNAPSHOT_VERSION);
    assert_eq!(value["output"]["attempts"], 3); assert_eq!(value["output"]["committed"], 2);
    assert_eq!(value["output"]["reserved_work"], serde_json::to_value(fixed.max_work).unwrap());
    assert!(value.get("chunk_schedule").is_none());
    assert!(value.get("root_array_union").is_none());
    assert!(value.get("root_object_reconciliation").is_none());
    assert!(value.get("max_selected_rows").is_none());
    assert!(value.get("input_encoding").is_none());
    assert!(value.get("compute_schedule").is_none());
    for forbidden in ["text", "schema", "values", "spans", "recipe", "key", "job_dir", "results", "documents"] {
        assert!(value.get(forbidden).is_none()); assert!(value["output"].get(forbidden).is_none());
    }
    assert!(bytes.ends_with(b"\n")); assert_eq!(publish(&response, 1, &mut NoIo), Err(CandidateError::Output));
    for width in [2, 4, 64] {
        response.compute_schedule = compute::Schedule::for_threads(width);
        let threaded = serde_json::to_value(&response).unwrap();
        assert_eq!(threaded["compute_schedule"], serde_json::json!({
            "execution": "int4-output-row-team-v1", "declared_compute_threads": width }));
        assert_eq!(threaded["output"], value["output"]);
        assert_eq!(threaded["extraction_mode"], value["extraction_mode"]);
    }
    response.compute_schedule = None;
    response.chunk_schedule = Some(ChunkSchedule::new(4));
    let grouped = serde_json::to_value(&response).unwrap();
    assert_eq!(grouped["chunk_schedule"], serde_json::json!({
        "execution": "native_chunk_cohorts", "phase": "map", "declared_max_chunks": 4 }));
    assert_eq!(grouped["output"], value["output"]);
    assert_eq!(grouped["extraction_mode"], value["extraction_mode"]);
    response.max_selected_rows = Some(8192);
    let selected = serde_json::to_value(&response).unwrap();
    assert_eq!(selected["max_selected_rows"], 8192);
    assert_eq!(selected["output"], value["output"]);
    assert_eq!(selected["chunk_schedule"], grouped["chunk_schedule"]);
    let encoding = crate::tokenizer::untrusted::UntrustedEncoding::SafeBpeV1(
        crate::tokenizer::untrusted::SafeBpeLimits { max_input_bytes: 65536, max_output_tokens: 32768,
            max_pair_checks: 100000, max_pair_bytes: 1000000 });
    response.input_encoding = input_encoding::reported_policy(&encoding);
    let mut compact_bytes = Vec::new();
    publish(&response, job_cli::REPORT_BYTES, &mut compact_bytes).unwrap();
    let mut compact: serde_json::Value = serde_json::from_slice(&compact_bytes).unwrap();
    assert_eq!(compact["input_encoding"], serde_json::json!({
        "kind": "safe_bpe_v1", "limits": { "max_input_bytes": 65536, "max_output_tokens": 32768,
            "max_pair_checks": 100000, "max_pair_bytes": 1000000 } }));
    assert_eq!(compact["output"], selected["output"]);
    for forbidden in ["text", "schema", "values", "spans", "recipe", "key", "job_dir", "results", "documents"] {
        assert!(compact.get(forbidden).is_none()); assert!(compact["output"].get(forbidden).is_none());
    }
    let compact_text = std::str::from_utf8(&compact_bytes).unwrap();
    for private in ["private-schema", "private/job", "private/key", "schema_digest", "binding"] {
        assert!(!compact_text.contains(private));
    }
    assert!(compact_bytes.ends_with(b"\n"));
    compact.as_object_mut().unwrap().remove("input_encoding");
    assert_eq!(compact, selected, "encoding provenance must not change progress or other public metadata");
    response.input_encoding = None;
    let (object, object_args) = command("resume", &["--reconcile-root-object"]);
    let object_limits = object.args.reconciliation.limits(object.args.map.mapping).unwrap().unwrap();
    response.root_object_reconciliation = Some(RootObjectSelection::new(object_limits));
    response.extraction_mode = "root_object_exact_reconciliation";
    let object = serde_json::to_value(&response).unwrap();
    assert_eq!(object["extraction_mode"], "root_object_exact_reconciliation");
    assert_eq!(object["root_object_reconciliation"]["policy"],
        crate::tasks::extract::object_reconcile::ROOT_OBJECT_RECONCILIATION);
    assert_eq!(object["root_object_reconciliation"]["limits"], serde_json::to_value(object_limits).unwrap());
    assert_eq!(object["output"], value["output"]);
    assert!(object.get("input_encoding").is_none());
    let bytes = serde_json::to_string(&object).unwrap();
    for private in ["private-schema", "private/job", "private/key", "schema_digest", "binding"] {
        assert!(!bytes.contains(private));
    }
    assert_eq!(object_args.group_width, args.group_width);
}

#[test]
fn object_runtime_rechecks_complete_schema_workspace_before_private_io() {
    let (c, mut args) = command("resume", &["--reconcile-root-object", "--cohort-chunks", "3",
        "--max-selected-rows", "32"]);
    let map = c.args.validate(&args).unwrap();
    args.memory_mib = map.common.minimum_memory_bytes.div_ceil(MIB) - 1;
    assert_eq!(execute(c.mode, args, c.args, map, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
}

#[test]
fn deadline_control_refuses_object_schema_work_without_a_new_time_budget() {
    let mut expired = DeadlineControl { started: Instant::now(), deadline: Duration::ZERO };
    assert_eq!(expired.prefill_checkpoint(0), Some(DecodeCancellationKind::Deadline));
    let mut live = DeadlineControl { started: Instant::now(), deadline: Duration::from_secs(30) };
    assert_eq!(live.prefill_checkpoint(0), None);
}
