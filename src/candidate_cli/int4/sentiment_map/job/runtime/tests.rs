//! Host quotation, typed cancellation and metadata-publication boundaries.
use super::*;
use super::super::tests::{command, lifetime, NoIo};
use crate::{jobs::{JobId, JobProgress}, native_engine::decode::DecodeCancellationKind};

#[test]
fn host_population_quote_and_frozen_chunk_identity_storage_match_admission() {
    for flags in [vec![], vec!["--cohort-chunks", "3"], vec!["--cohort-chunks", "4", "--max-axes", "1"]] {
        let (c, args) = command("resume", &flags); let map = c.args.validate(&args).unwrap();
        let fixed = lifetime(&args, map); let limits = c.args.lifetime_limits(&args, map, fixed).unwrap();
        let host = host_limits(&args, limits, Duration::from_secs(7));
        assert_eq!(host.required_buffer_bytes(fixed).unwrap(), limits.buffer_bytes);
        assert_eq!(host.native.run.max_elapsed, Duration::from_secs(7));
        assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
        assert_eq!(limits.map.common.minimum_memory_bytes, map.common.minimum_memory_bytes
            + limits.buffer_bytes + map.mapping.chunks.max_chunks as u64 * SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES);
    }
}

#[test]
fn runtime_rechecks_programmatic_storage_refusal_before_any_private_io() {
    let (mut c, args) = command("resume", &[]); let map = c.args.validate(&args).unwrap();
    c.args.storage.store_results = false;
    assert_eq!(execute(c.mode, args, c.args, map, NoIo, &mut NoIo).unwrap_err().code,
        "explicit_sentiment_document_job_storage");
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
    let report = progress_report(progress, id, fixed, true, c.mode, "sentiment").unwrap();
    let response = Response { protocol: "fnlp-candidate-int4-sentiment-document-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "sentiment",
        durable_unit: "complete_original_document", model_id: "Nanbeige4.2-3B",
        source_revision: "public-revision", source_root_sha256: "public-source", model_content_sha256: "public-model",
        candidate_format: "candidate-int4", quant_recipe: "strict-int4", group_width: args.group_width,
        integer_kernel: map.common.kernel.kind().id(), snapshot: None, cohort_chunks: Some(3),
        input_encoding: None, compute_schedule: None, output: &report };
    let mut bytes = Vec::new(); publish_report(&response, &mut bytes, &available).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["durable_unit"], "complete_original_document"); assert_eq!(value["cohort_chunks"], 3);
    assert_eq!(value["task"], "sentiment"); assert_eq!(value["output"]["committed"], 2);
    assert_eq!(value["output"]["attempts"], 3);
    assert!(value.get("input_encoding").is_none());
    assert_eq!(value["output"]["reserved_work"], serde_json::to_value(fixed.max_work).unwrap());
    for forbidden in ["text", "axes", "scores", "policy", "recipe", "key", "job_dir", "documents", "aggregate"] {
        assert!(value.get(forbidden).is_none()); assert!(value["output"].get(forbidden).is_none());
    }
    assert!(bytes.ends_with(b"\n"));
    let encoding = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 262_144,
        max_output_tokens: 131_072, max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 });
    let compact = Response { input_encoding: Some(&encoding), ..response };
    let mut compact_bytes = Vec::new(); publish_report(&compact, &mut compact_bytes, &available).unwrap();
    let compact: serde_json::Value = serde_json::from_slice(&compact_bytes).unwrap();
    assert_eq!(compact["input_encoding"], serde_json::to_value(&encoding).unwrap());
    assert_eq!(compact["output"], value["output"]);
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
fn runtime_rechecks_explicit_axis_capability_before_private_io() {
    for axes in [0, 5] {
        let (mut c, args) = command("resume", &[]); let map = c.args.validate(&args).unwrap();
        c.args.sentiment.max_axes = Some(axes);
        assert_eq!(execute(c.mode, args, c.args, map, NoIo, &mut NoIo).unwrap_err().code,
            "sentiment_document_job_geometry");
    }
    let (mut c, args) = command("start", &[]); let map = c.args.validate(&args).unwrap();
    c.args.sentiment.max_axes = Some(4);
    c.args.map.mapping.max_chunks = 64; c.args.map.mapping.cohort_chunks = Some(22);
    assert_eq!(execute(c.mode, args, c.args, map, NoIo, &mut NoIo).unwrap_err().code,
        "sentiment_document_job_geometry");
}

#[test]
fn changing_team_width_preserves_authenticated_recipe_policy_and_lifetime_on_resume() {
    use crate::native_engine::strict_int4::weights::projection_fixture;
    let (weights, _) = projection_fixture(128);
    for defaults in ["{}", r#"{"axes":["approach","valence"],"policy":{"minimum_peak_weight_ppm":650000,"maximum_normalized_entropy_ppm":800000}}"#] {
        let mut recipes = Vec::new();
        for compact in [false, true] {
            let mut expected = None;
            for operation in ["start", "resume"] {
                for threads in [1, 2, 4, 64] {
                    let width = threads.to_string();
                    let mut flags = vec!["--compute-threads", width.as_str()];
                    if compact { flags.extend(["--input-encoding", "safe-bpe-v1"]); }
                    let (command, args) = command(operation, &flags);
                    let map = command.args.validate(&args).unwrap();
                    let fixed = lifetime(&args, map);
                    let limits = command.args.lifetime_limits(&args, map, fixed).unwrap();
                    let budget = command.args.map.budget_with_max_axes(map.common,
                        command.args.sentiment.max_axes.unwrap_or(4)).unwrap();
                    let (settings, policy) = command.args.load_settings(Some(defaults), budget, &args).unwrap();
                    let options = crate::candidate_cli::int4::scored_batch::runtime::sentiment_options(policy).unwrap();
                    let compiler = Int4SentimentPlanner::for_model(weights.identity(), options, budget, map.planning)
                        .unwrap().with_document_encoding(command.args.map.input_encoding(&args).unwrap()).unwrap();
                    let planner = Int4SentimentDocumentJobPlanner::new(compiler, settings, map.mapping, fixed).unwrap();
                    planner.check_job_limits(fixed).unwrap();
                    let semantic = (crate::canonjson::canonical_bytes(planner.job_recipe()).unwrap(),
                        crate::canonjson::canonical_bytes(planner.execution_identity()).unwrap(),
                        fixed.max_work, fixed.max_attempts, limits.buffer_bytes);
                    if let Some(expected) = &expected { assert_eq!(&semantic, expected); }
                    else { expected = Some(semantic); }
                    let config = compute::resource_config(map.common);
                    assert_eq!(config.scoped_cpu_children_per_coordinator, threads - 1);
                    assert_eq!(config.thread_ceiling, threads + 1);
                }
            }
            recipes.push(expected.unwrap().0);
        }
        assert_ne!(recipes[0], recipes[1], "compact policy remains authenticated while worker width may change");
    }
}

#[test]
fn team_progress_metadata_changes_only_the_optional_physical_schedule() {
    let mut expected = None;
    for threads in [1, 2, 4, 64] {
        let width = threads.to_string();
        let (c, args) = command("resume", &["--compute-threads", width.as_str()]);
        let map = c.args.validate(&args).unwrap(); let fixed = lifetime(&args, map); let id = JobId([0x12; 16]);
        let progress = JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
            reserved_work: fixed.max_work, spool_bytes: 1024, materialized: true };
        let report = progress_report(progress, id, fixed, true, c.mode, "sentiment").unwrap();
        let response = Response { protocol: "fnlp-candidate-int4-sentiment-document-job-v1", schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "sentiment",
            durable_unit: "complete_original_document", model_id: "Nanbeige4.2-3B",
            source_revision: "public-revision", source_root_sha256: "public-source", model_content_sha256: "public-model",
            candidate_format: "candidate-int4", quant_recipe: "strict-int4", group_width: args.group_width,
            integer_kernel: map.common.kernel.kind().id(), snapshot: None, cohort_chunks: None,
            input_encoding: None, compute_schedule: compute::Schedule::for_threads(threads), output: &report };
        let mut bytes = Vec::new(); publish_report(&response, &mut bytes, &available).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        if threads == 1 { assert!(value.get("compute_schedule").is_none()); }
        else {
            assert_eq!(value["compute_schedule"]["execution"], "int4-output-row-team-v1");
            assert_eq!(value["compute_schedule"]["declared_compute_threads"], threads);
        }
        assert!(value.get("compute").is_none(), "no native projection receipt is fabricated");
        value.as_object_mut().unwrap().remove("compute_schedule");
        if let Some(expected) = &expected { assert_eq!(&value, expected); }
        else { expected = Some(value); }
    }
}

#[test]
fn runtime_rechecks_team_cohort_conflicts_before_private_job_io() {
    let (mut c, args) = command("resume", &["--compute-threads", "4"]);
    let map = c.args.validate(&args).unwrap();
    c.args.map.mapping.cohort_chunks = Some(1);
    assert_eq!(execute(c.mode, args, c.args, map, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
}
