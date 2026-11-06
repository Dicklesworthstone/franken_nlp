//! Typed failures and complete-response publication without model IO.
use super::*;
use super::super::tests::{command, request_json};
use crate::tasks::judge::FaithfulnessAggregationError;

#[test]
fn typed_cancellation_survives_planning_loading_reduction_and_physical_wrappers() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = if matches!(cause, Timeout | Deadline) { CandidateError::Timeout } else { CandidateError::Execution };
        let judge = || Int4JudgeError::Cancelled(cause);
        let map = || Int4FaithfulnessMapError::Judge(judge());
        assert_eq!(judge_error(judge()), expected);
        assert_eq!(planning_error(map()), expected);
        assert_eq!(planning_error(Int4FaithfulnessMapError::Aggregation(FaithfulnessAggregationError::Cancelled(cause))), expected);
        for fallback in [CandidateError::Model, CandidateError::Execution] {
            assert_eq!(host_error(HostedError::Int4(Int4HostError::FaithfulnessMap(map())), fallback), expected);
            assert_eq!(host_error(HostedError::Int4(Int4HostError::Judge(judge())), fallback), expected);
            assert_eq!(host_error(HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause))), fallback), expected);
            assert_eq!(host_error(HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause))), fallback), expected);
            assert_eq!(host_error(HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))), fallback), expected);
            assert_eq!(host_error(HostedError::Stopped { stop: crate::hosted::RunStop {
                kind: cause, runtime_reason: None, runtime_checkpoint_error: None }, task_error: None }, fallback), expected);
            assert_eq!(host_error(HostedError::Scope {
                source: asupersync::cx::ScopedCpuError::WorkerCapExceeded { cap: 1 },
                physical_error: Some(Box::new(HostedError::Int4(Int4HostError::FaithfulnessMap(map())))),
            }, fallback), expected);
        }
    }
    assert_eq!(planning_error(Int4FaithfulnessMapError::Work), CandidateError::Planning);
    assert_eq!(host_error(HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Integrity)),
        CandidateError::Model), CandidateError::Model);
    for error in [HostedError::ModelIdentity, HostedError::ResourceDomain] {
        assert_eq!(host_error(error, CandidateError::Execution), CandidateError::Identity);
    }
    assert_eq!(host_error(HostedError::Reservation(crate::ReservationError::ArithmeticOverflow),
        CandidateError::Execution), CandidateError::Memory);
}

#[test]
fn host_and_cli_share_the_same_serial_and_ragged_window_quotes() {
    for flags in [vec![], vec!["--cohort-chunks", "1"], vec!["--cohort-chunks", "3"],
        vec!["--cohort-chunks", "64", "--max-result-bytes", "524288"]] {
        for width in [1, 3, 128, 10752] {
            let mut command = command(&flags);
            command.args.group_width = width; command.args.max_load_mib = 65536;
            let Int4Task::FaithfulnessMap(options) = command.task else { panic!("wrong task") };
            let limits = options.validate(&command.args).unwrap(); let common = limits.common;
            let config = Int4FaithfulnessMapHostConfig {
                native: NativeLimits { context_tokens: command.args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
                    run: RunLimits { max_elapsed: Duration::from_secs(command.args.timeout_seconds),
                        max_checkpoints: command.args.max_checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES } },
                planning: limits.planning, mapping: limits.mapping, preparation_reserve_bytes: common.preparation_bytes,
            };
            let task = options.budget(common).unwrap();
            let quote = if let Some(chunks) = options.mapping.cohort_chunks {
                config.cohort_requirements(width, task, chunks).unwrap().memory
            } else { config.requirements(width, task).unwrap() };
            assert_eq!(quote, limits.memory);
        }
    }
}

#[test]
fn malformed_or_other_task_inputs_refuse_before_source_snapshot_or_save_io() {
    // The process broker fixes its configuration on first installation.
    // Exercise real pre-IO admission in a fresh test process so unrelated
    // runtime fixtures cannot replace the expected input refusal.
    const CHILD: &str = "FNLP_FAITHFULNESS_MAP_WIRE_FIXTURE_CHILD";
    const COMPLETE: &str = "FNLP_FAITHFULNESS_MAP_WIRE_FIXTURE_COMPLETE";
    let full_name = concat!(module_path!(), "::malformed_or_other_task_inputs_refuse_before_source_snapshot_or_save_io");
    let name = full_name.split_once("::").expect("crate-qualified test name").1;
    if std::env::var(CHILD).ok().as_deref() != Some(name) {
        let result = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", name, "--nocapture"]).env(CHILD, name).output().expect("isolated runtime fixture");
        assert!(result.status.success(), "{}\n{}", String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr));
        assert!(String::from_utf8_lossy(&result.stdout).contains(COMPLETE), "fixture must actually run");
        return;
    }
    assert!(crate::installed_process_resources().is_none());
    let mut value: serde_json::Value = serde_json::from_str(&request_json()).unwrap();
    value["mode"] = serde_json::json!("pairwise");
    for route in 0..3 {
        let mut command = command(&[]);
        if route == 1 {
            command.args.source_dir = None; command.args.source_manifest = None;
            command.args.snapshot = Some("fixture-must-not-be-opened.snapshot".into());
            command.args.expected_model_sha256 = Some("ab".repeat(32));
        } else if route == 2 {
            command.args.save_snapshot = Some("fixture-must-not-be-created.snapshot".into());
        }
        let text = value.to_string(); let mut output = Vec::new();
        assert_eq!(command.execute(&mut io::Cursor::new(text.as_bytes()), &mut output), Err(CandidateError::Input));
        assert!(output.is_empty());
    }
    println!("{COMPLETE}");
}

fn available() -> Result<Duration, CandidateError> { Ok(Duration::from_secs(1)) }
fn provenance() -> Provenance<'static> {
    Provenance { protocol: "fnlp-candidate-int4-faithfulness-map-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "faithfulness-map",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "fixture-source",
        model_content_sha256: "fixture-model", candidate_format: "fixture-format", quant_recipe: "fixture:group=128",
        group_width: 128, integer_kernel: "s_scalar_i32", snapshot: None }
}

#[test]
fn publication_preserves_evidence_without_leaking_source_claim_or_local_paths() {
    let provenance = provenance();
    let semantic = serde_json::json!({"single_context_equivalence":false,
        "aggregate":{"relation":null,"abstention":"conflicting_evidence","evidence":[
            {"chunk_id":0,"quote":"approved witness","relation":"entailed"},
            {"chunk_id":1,"quote":"contrary witness","relation":"contradicted"}]}});
    let mut serial = Vec::new();
    let response = Response { provenance: &provenance, output: &semantic, cohort: None };
    publish_complete(&response, 4096, &mut serial, &available).unwrap();
    assert_eq!(serial.last(), Some(&b'\n'));
    let value: serde_json::Value = serde_json::from_slice(&serial).unwrap();
    assert_eq!(value["output"], semantic);
    assert_eq!(value["task"], "faithfulness-map"); assert_eq!(value["evidence"], "non_authoritative");
    assert_eq!(value["protocol"], "fnlp-candidate-int4-faithfulness-map-v1");
    for absent in ["source", "claim", "prompt", "prompt_digest", "source_dir", "snapshot_path", "snapshot", "cohort"] {
        assert!(value.get(absent).is_none());
    }
    let mut exact = Vec::new();
    publish_complete(&response, serial.len(), &mut exact, &available).unwrap(); assert_eq!(exact, serial);
    let mut refused = Vec::new();
    assert_eq!(publish_complete(&response, serial.len() - 1, &mut refused, &available), Err(CandidateError::Output));
    assert!(refused.is_empty());
    let response = Response { provenance: &provenance, output: &semantic,
        cohort: Some(Cohort { schema_version: 1,
            execution: crate::tasks::judge::grouped_int4::faithfulness_map::INT4_FAITHFULNESS_MAP_COHORT_EXECUTION,
            max_chunks: 3, chunk_cohorts: 2, group_steps: 7, projection_groups: 5 }) };
    let mut grouped = Vec::new();
    publish_complete(&response, 8192, &mut grouped, &available).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&grouped).unwrap();
    assert_eq!(value["output"], semantic);
    assert_eq!(value["cohort"]["max_chunks"], 3); assert_eq!(value["cohort"]["chunk_cohorts"], 2);
}

#[test]
fn serialization_deadline_or_failure_never_publishes_a_response_prefix() {
    use std::cell::Cell;
    struct Expires<'a>(&'a Cell<bool>);
    impl Serialize for Expires<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.0.set(true); serializer.serialize_str("complete evidence")
        }
    }
    struct Refuse;
    impl Serialize for Refuse {
        fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
            Err(<S::Error as serde::ser::Error>::custom("private serializer detail"))
        }
    }
    let expired = Cell::new(false);
    let remaining = || if expired.get() { Err(CandidateError::Timeout) } else { available() };
    let mut output = Vec::new();
    assert_eq!(publish_complete(&Expires(&expired), 128, &mut output, &remaining), Err(CandidateError::Timeout));
    assert!(expired.get()); assert!(output.is_empty());
    expired.set(false);
    assert_eq!(publish_complete(&Expires(&expired), 128, &mut output, &|| Err(CandidateError::Timeout)),
        Err(CandidateError::Timeout));
    assert!(!expired.get()); assert!(output.is_empty());
    assert_eq!(publish_complete(&Response { provenance: &provenance(), output: &Refuse, cohort: None },
        4096, &mut output, &available), Err(CandidateError::Output));
    assert!(output.is_empty());
    let mut control = DeadlineControl { started: Instant::now(), deadline: Duration::ZERO };
    assert_eq!(control.checkpoint(0), Some(DecodeCancellationKind::Deadline));
    assert_eq!(control.prefill_checkpoint(0), Some(DecodeCancellationKind::Deadline));
}

#[test]
fn physical_write_flush_and_final_deadline_remain_inside_publication() {
    use std::cell::Cell;
    struct Writer<'a> { flushed: &'a Cell<bool>, bytes: Vec<u8>, fail_write: bool, fail_flush: bool }
    impl Write for Writer<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.fail_write { return Err(io::Error::other("private write detail")); }
            self.bytes.extend_from_slice(bytes); Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            self.flushed.set(true);
            if self.fail_flush { Err(io::Error::other("private flush detail")) } else { Ok(()) }
        }
    }
    let flushed = Cell::new(false);
    let mut writer = Writer { flushed: &flushed, bytes: Vec::new(), fail_write: false, fail_flush: false };
    let deadline = || if flushed.get() { Err(CandidateError::Timeout) } else { available() };
    assert_eq!(publish_complete(&serde_json::json!({"complete":true}), 128, &mut writer, &deadline),
        Err(CandidateError::Timeout));
    assert!(flushed.get()); assert_eq!(writer.bytes.last(), Some(&b'\n'));
    for fail_write in [true, false] {
        flushed.set(false);
        let mut writer = Writer { flushed: &flushed, bytes: Vec::new(), fail_write, fail_flush: !fail_write };
        assert_eq!(publish_complete(&serde_json::json!({"complete":true}), 128, &mut writer, &available),
            Err(CandidateError::Output));
        assert_eq!(flushed.get(), !fail_write);
    }
}

#[test]
fn generic_one_off_dispatch_cannot_skip_document_admission() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("input accessed"); } }
    let command = command(&[]);
    let Int4Task::FaithfulnessMap(options) = command.task else { panic!("wrong task") };
    let limits = options.validate(&command.args).unwrap();
    assert_eq!(super::super::super::runtime::execute_task(command.task, command.args, limits.common,
        &mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
}
