//! Host quotation, typed cancellation and metadata-publication boundaries.
use super::*;
use super::super::tests::{command, lifetime, NoIo};
use crate::{jobs::{JobId, JobProgress}, native_engine::decode::DecodeCancellationKind};

#[test]
fn host_population_quote_and_frozen_chunk_identity_storage_match_admission() {
    for flags in [vec![], vec!["--cohort-chunks", "3"], vec!["--cohort-chunks", "8"]] {
        let (c, args) = command("resume", &flags); let map = c.args.validate(&args).unwrap();
        let fixed = lifetime(&args, map); let limits = c.args.lifetime_limits(&args, map, fixed).unwrap();
        let host = host_limits(&args, limits, Duration::from_secs(7));
        assert_eq!(host.required_buffer_bytes(fixed).unwrap(), limits.buffer_bytes);
        assert_eq!(host.native.run.max_elapsed, Duration::from_secs(7));
        assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
        assert_eq!(limits.map.common.minimum_memory_bytes, map.common.minimum_memory_bytes
            + limits.buffer_bytes + map.mapping.chunks.max_chunks as u64 * FAITHFULNESS_DOCUMENT_CHUNK_METADATA_BYTES);
    }
}

#[test]
fn runtime_rechecks_programmatic_storage_refusal_before_any_private_io() {
    let (mut c, args) = command("resume", &[]); let map = c.args.validate(&args).unwrap();
    c.args.storage.store_results = false;
    assert_eq!(execute(c.mode, args, c.args, map, NoIo, &mut NoIo).unwrap_err().code,
        "explicit_faithfulness_document_job_storage");
}

#[test]
fn all_typed_load_and_snapshot_cancellation_causes_survive_nested_physical_errors() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost, ParentCancelled,
        ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = job_failure(JobError::Cancelled(cause));
        for error in [HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Judge(crate::tasks::judge::grouped_int4::Int4JudgeError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::FaithfulnessMap(
                crate::tasks::judge::grouped_int4::faithfulness_map::Int4FaithfulnessMapError::Judge(
                    crate::tasks::judge::grouped_int4::Int4JudgeError::Cancelled(cause)))),
            HostedError::Scope { source: asupersync::cx::ScopedCpuError::WorkerCapExceeded { cap: 1 },
                physical_error: Some(Box::new(HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause))))) }] {
            let actual = model_failure(error); assert_eq!(actual.exit, expected.exit); assert_eq!(actual.code, expected.code);
        }
    }
}

#[test]
fn completion_reports_only_public_model_and_durable_progress_metadata() {
    let (c, args) = command("resume", &["--cohort-chunks", "3"]); let map = c.args.validate(&args).unwrap();
    let fixed = lifetime(&args, map); let id = JobId([0x12; 16]);
    let progress = JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
        reserved_work: fixed.max_work, spool_bytes: 1024, materialized: true };
    let report = progress_report(progress, id, fixed, true, c.mode, "faithfulness").unwrap();
    let mut response = Response { protocol: "fnlp-candidate-int4-faithfulness-document-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "faithfulness",
        durable_unit: "complete_original_document", model_id: "Nanbeige4.2-3B",
        source_revision: "public-revision", source_root_sha256: "public-source", model_content_sha256: "public-model",
        candidate_format: "candidate-int4", quant_recipe: "strict-int4", group_width: args.group_width,
        integer_kernel: map.common.kernel.kind().id(), snapshot: None, cohort_chunks: Some(3),
        max_evidence_heads: None, compute_schedule: None, output: &report };
    let mut bytes = Vec::new(); publish_report(&response, &mut bytes, &available).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["durable_unit"], "complete_original_document"); assert_eq!(value["cohort_chunks"], 3);
    assert_eq!(value["task"], "faithfulness"); assert_eq!(value["output"]["committed"], 2);
    assert_eq!(value["output"]["attempts"], 3);
    assert_eq!(value["output"]["reserved_work"], serde_json::to_value(fixed.max_work).unwrap());
    for forbidden in ["text", "source", "claim", "scores", "citations", "policy", "recipe", "key", "job_dir", "documents", "aggregate", "max_evidence_heads", "compute_schedule"] {
        assert!(value.get(forbidden).is_none()); assert!(value["output"].get(forbidden).is_none());
    }
    assert!(bytes.ends_with(b"\n"));
    response.max_evidence_heads = Some(1);
    assert_eq!(serde_json::to_value(&response).unwrap()["max_evidence_heads"], 1);
}

fn available() -> Result<Duration, CandidateError> { Ok(Duration::from_secs(1)) }
#[test]
fn oversize_or_expired_metadata_never_reaches_the_writer() {
    let oversized = "x".repeat(job_cli::REPORT_BYTES);
    assert_eq!(publish_report(&oversized, &mut NoIo, &available).unwrap_err().exit, ErrorCode::Generic);
    let checks = std::cell::Cell::new(0);
    let deadline = || {
        checks.set(checks.get() + 1);
        if checks.get() == 1 { available() } else { Err(CandidateError::Timeout) }
    };
    assert_eq!(publish_report(&serde_json::json!({"status":"complete"}), &mut NoIo, &deadline)
        .unwrap_err().exit, ErrorCode::BudgetOrTimeout);
    assert_eq!(checks.get(), 2, "serialization cannot renew the original publication deadline");
}

#[test]
fn flush_failure_or_a_deadline_during_delivery_never_reports_success() {
    struct FlushFails { bytes: Vec<u8> }
    impl Write for FlushFails {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.bytes.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("fixture flush")) }
    }
    let mut writer = FlushFails { bytes: Vec::new() };
    assert!(publish_report(&true, &mut writer, &available).is_err()); assert!(!writer.bytes.is_empty());
    let checks = std::cell::Cell::new(0);
    let deadline = || { checks.set(checks.get() + 1);
        if checks.get() < 3 { available() } else { Err(CandidateError::Timeout) } };
    let mut bytes = Vec::new();
    assert_eq!(publish_report(&true, &mut bytes, &deadline).unwrap_err().exit, ErrorCode::BudgetOrTimeout);
    assert_eq!(bytes, b"true\n"); assert_eq!(checks.get(), 3);
}

#[test]
fn physical_width_preserves_real_recipe_identity_evidence_policy_and_lifetime_on_resume() {
    use crate::native_engine::strict_int4::weights::projection_fixture;
    // Real model-bound factories consume fixture metadata only. No native
    // forward, substitute judge, numerical parity or quality claim is made.
    let (weights, _) = projection_fixture(128);
    let mut head_recipes = Vec::new();
    for max_heads in [None, Some(1), Some(3), Some(32)] {
        let head_text = max_heads.map(|heads: usize| heads.to_string());
        let policy = FaithfulnessPolicy { minimum_candidate_weight_ppm: 713_000, minimum_margin_milli: 137,
            evidence_window_bytes: 256, max_evidence_windows: match max_heads { Some(1) => 1, Some(3) => 2, _ => 31 },
            max_evidence_spans: 1 };
        let mut expected = None;
        for operation in ["start", "resume"] {
            for threads in [1, 2, 4, 64] {
                let width = threads.to_string();
                let mut flags = vec!["--compute-threads", width.as_str(),
                    "--minimum-classified-coverage-ppm", "850000"];
                if let Some(heads) = &head_text { flags.extend(["--max-evidence-heads", heads.as_str()]); }
                let (c, args) = command(operation, &flags);
                let map = c.args.validate(&args).unwrap(); let fixed = lifetime(&args, map);
                let limits = c.args.lifetime_limits(&args, map, fixed).unwrap();
                let budget = c.args.map.budget_with_max_heads(map.common, max_heads.unwrap_or(32)).unwrap();
                let policy = super::super::super::corpus::parse_policy_with_max_heads(
                    &serde_json::to_string(&policy).unwrap(), c.args.faithfulness.max_evidence_heads).unwrap();
                let compiler = Int4JudgePlanner::for_model(weights.identity(), budget, map.planning).unwrap();
                let planner = Int4FaithfulnessDocumentJobPlanner::new(compiler, policy, limits.map.mapping, fixed).unwrap();
                let planner = if let Some(heads) = max_heads { planner.with_max_heads(heads).unwrap() } else { planner };
                planner.check_job_limits(fixed).unwrap();
                assert_eq!(planner.max_heads(), max_heads.unwrap_or(32));
                assert_eq!(planner.policy(), policy);
                assert_eq!(planner.document_limits().aggregation.minimum_classified_coverage_ppm, 850_000);
                let recipe = serde_json::to_value(planner.job_recipe()).unwrap();
                assert!(recipe.get("compute_threads").is_none()); assert!(recipe.get("compute_schedule").is_none());
                match max_heads {
                    Some(heads) => assert_eq!(recipe["max_heads"], heads),
                    None => assert!(recipe.get("max_heads").is_none()),
                }
                let semantic = (crate::canonjson::canonical_bytes(planner.job_recipe()).unwrap(),
                    crate::canonjson::canonical_bytes(planner.execution_identity()).unwrap(),
                    fixed.max_work, fixed.max_attempts, limits.buffer_bytes, planner.max_result_bytes());
                if let Some(expected) = &expected { assert_eq!(&semantic, expected); }
                else { expected = Some(semantic); }
                let config = compute::resource_config(map.common);
                assert_eq!(config.scoped_cpu_children_per_coordinator, threads - 1);
                assert_eq!(config.thread_ceiling, threads + 1);
            }
        }
        head_recipes.push(expected.unwrap().0);
    }
    assert_ne!(head_recipes[0], head_recipes[3], "explicit 32 remains distinct authenticated authority");
    let omitted: serde_json::Value = serde_json::from_slice(&head_recipes[0]).unwrap();
    let mut explicit: serde_json::Value = serde_json::from_slice(&head_recipes[3]).unwrap();
    explicit.as_object_mut().unwrap().remove("max_heads");
    assert_eq!(omitted, explicit);
}

#[test]
fn team_progress_metadata_changes_only_the_optional_physical_schedule() {
    let mut expected = None;
    for threads in [1, 2, 4, 64] {
        let width = threads.to_string();
        let (c, args) = command("resume", &["--compute-threads", width.as_str(), "--max-evidence-heads", "3"]);
        let map = c.args.validate(&args).unwrap(); let fixed = lifetime(&args, map); let id = JobId([0x12; 16]);
        let progress = JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
            reserved_work: fixed.max_work, spool_bytes: 1024, materialized: true };
        let report = progress_report(progress, id, fixed, true, c.mode, "faithfulness").unwrap();
        let response = Response { protocol: "fnlp-candidate-int4-faithfulness-document-job-v1", schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "faithfulness",
            durable_unit: "complete_original_document", model_id: "Nanbeige4.2-3B",
            source_revision: "public-revision", source_root_sha256: "public-source", model_content_sha256: "public-model",
            candidate_format: "candidate-int4", quant_recipe: "strict-int4", group_width: args.group_width,
            integer_kernel: map.common.kernel.kind().id(), snapshot: None, cohort_chunks: None,
            max_evidence_heads: Some(3), compute_schedule: compute::Schedule::for_threads(threads), output: &report };
        let mut bytes = Vec::new(); publish_report(&response, &mut bytes, &available).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        if threads == 1 { assert!(value.get("compute_schedule").is_none()); }
        else {
            assert_eq!(value["compute_schedule"]["execution"], "int4-output-row-team-v1");
            assert_eq!(value["compute_schedule"]["declared_compute_threads"], threads);
        }
        assert!(value.get("compute").is_none());
        value.as_object_mut().unwrap().remove("compute_schedule");
        if let Some(expected) = &expected { assert_eq!(&value, expected); }
        else { expected = Some(value); }
    }
}

#[test]
fn runtime_rechecks_team_cohort_conflicts_before_private_job_io() {
    let (mut c, args) = command("resume", &["--compute-threads", "4"]);
    let map = c.args.validate(&args).unwrap(); c.args.map.mapping.cohort_chunks = Some(1);
    assert_eq!(execute(c.mode, args, c.args, map, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
}
