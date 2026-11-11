//! Model-free native-boundary, deadline and complete-publication contracts.
use super::*;
use super::super::tests::command;
use crate::tasks::classify::{ClassificationAggregationError, ClassificationPlanningError};

#[test]
fn cancellation_survives_planning_aggregation_loading_and_nested_physical_failures() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost, ParentCancelled,
        ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = if matches!(cause, Timeout | Deadline) { CandidateError::Timeout } else { CandidateError::Execution };
        let classification = || Int4ClassificationError::Planning(ClassificationPlanningError::Cancelled(cause));
        let map = || Int4ClassificationMapError::Classification(classification());
        assert_eq!(classification_error(classification()), expected);
        assert_eq!(planning_error(map()), expected);
        assert_eq!(planning_error(Int4ClassificationMapError::Aggregation(
            ClassificationAggregationError::Cancelled(cause))), expected);
        for fallback in [CandidateError::Model, CandidateError::Execution] {
            assert_eq!(host_error(HostedError::Int4(Int4HostError::ClassificationMap(map())), fallback), expected);
            assert_eq!(host_error(HostedError::Int4(Int4HostError::Classification(classification())), fallback), expected);
            assert_eq!(host_error(HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause))), fallback), expected);
            assert_eq!(host_error(HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause))), fallback), expected);
            assert_eq!(host_error(HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))), fallback), expected);
            assert_eq!(host_error(HostedError::Stopped { stop: crate::hosted::RunStop {
                kind: cause, runtime_reason: None, runtime_checkpoint_error: None }, task_error: None }, fallback), expected);
            assert_eq!(host_error(HostedError::Scope {
                source: asupersync::cx::ScopedCpuError::WorkerCapExceeded { cap: 1 },
                physical_error: Some(Box::new(HostedError::Int4(Int4HostError::ClassificationMap(map())))),
            }, fallback), expected);
        }
    }
    assert_eq!(planning_error(Int4ClassificationMapError::Work), CandidateError::Planning);
    assert_eq!(host_error(HostedError::Int4(Int4HostError::ClassificationMap(Int4ClassificationMapError::Work)),
        CandidateError::Execution), CandidateError::Execution);
    assert_eq!(host_error(HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Integrity)),
        CandidateError::Model), CandidateError::Model);
    for error in [HostedError::ModelIdentity, HostedError::ResourceDomain] {
        assert_eq!(host_error(error, CandidateError::Execution), CandidateError::Identity);
    }
    assert_eq!(host_error(HostedError::Reservation(crate::ReservationError::ArithmeticOverflow),
        CandidateError::Execution), CandidateError::Memory);
}

#[test]
fn runtime_configuration_uses_the_exact_pure_quote_admitted_before_input_io() {
    for flags in [vec![], vec!["--cohort-chunks", "1"], vec!["--cohort-chunks", "3"],
        vec!["--cohort-chunks", "64", "--max-result-bytes", "1048512"]] {
        for width in [1, 3, 128, 10752] {
            let mut command = command(&flags);
            command.args.group_width = width; command.args.max_load_mib = 65536;
            let Int4Task::ClassifyMap(options) = command.task else { panic!("map required"); };
            let limits = options.validate(&command.args).unwrap(); let common = limits.common;
            let config = Int4ClassificationMapHostConfig {
                native: NativeLimits { context_tokens: command.args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
                    run: RunLimits { max_elapsed: Duration::from_secs(command.args.timeout_seconds),
                        max_checkpoints: command.args.max_checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES } },
                planning: limits.planning, mapping: limits.mapping, preparation_reserve_bytes: common.preparation_bytes,
            };
            let task = options.budget(common, limits.planning.max_labels).unwrap();
            let quote = if let Some(chunks) = options.mapping.cohort_chunks {
                let max_heads = limits.planning.max_labels.min(options.cohort_head_limit().unwrap());
                let quote = config.cohort_requirements(width, task, max_heads, chunks).unwrap();
                // The CLI admits unknown input conservatively. The host and
                // pre-save preflight both quote the actual independent heads.
                for heads in [1, max_heads] {
                    let actual = config.cohort_requirements(width, task, heads, chunks).unwrap();
                    assert!(actual.memory.native.kv_bytes <= quote.memory.native.kv_bytes);
                    assert!(actual.memory.workspace_bytes <= quote.memory.workspace_bytes);
                    assert!(actual.memory.input_bytes <= quote.memory.input_bytes);
                    assert_eq!(actual.memory.output_reservation_bytes, quote.memory.output_reservation_bytes);
                    let budget = cohort_budget(actual, config.mapping.max_model_work);
                    assert_eq!(budget.native.max_forward_positions, config.mapping.max_model_work.forward_positions);
                    assert_eq!(budget.native.max_attention_pairs, config.mapping.max_model_work.attention_pairs);
                    assert_eq!(budget.native.max_projection_work, config.mapping.max_model_work.projections);
                    assert_eq!(budget.memory.max_kv_bytes, actual.memory.native.kv_bytes);
                    assert_eq!(budget.memory.max_rope_bytes, actual.memory.native.rope_bytes);
                    assert_eq!(budget.memory.max_scratch_payload_bytes, actual.memory.native.scratch_payload_bound);
                    assert_eq!(budget.max_task_workspace_bytes, actual.task_workspace_bytes);
                    assert_eq!(budget.max_output_bytes, actual.max_cohort_output_bytes);
                }
                quote.memory
            } else {
                let quote = config.requirements(width, task).unwrap();
                assert_eq!(quote.native.kv_bytes, common.native.kv_bytes);
                quote
            };
            assert_eq!(quote, limits.memory);
        }
    }
}

fn provenance() -> Provenance<'static> {
    Provenance { protocol: "fnlp-candidate-int4-classify-map-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "classify-map",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "fixture-source",
        model_content_sha256: "fixture-model", candidate_format: "fixture-format", quant_recipe: "fixture:group=128",
        group_width: 128, integer_kernel: "s_scalar_i32", snapshot: None }
}
fn available() -> Result<Duration, CandidateError> { Ok(Duration::from_secs(1)) }

#[test]
fn complete_envelope_keeps_aggregate_separate_and_includes_the_newline_in_its_cap() {
    let provenance = provenance();
    let semantic = serde_json::json!({"single_context_equivalence":false,"chunks":[],
        "aggregate":{"calibration":"uncalibrated","result":{"mode":"exclusive"}}});
    let response = Response { provenance: &provenance, output: &semantic, cohort: None };
    let mut encoded = Vec::new(); publish_complete(&response, 4096, &mut encoded, &available).unwrap();
    assert_eq!(encoded.last(), Some(&b'\n'));
    let value: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(value["output"], semantic);
    assert_eq!(value["protocol"], "fnlp-candidate-int4-classify-map-v1");
    assert_eq!(value["task"], "classify-map"); assert_eq!(value["evidence"], "non_authoritative");
    for absent in ["document", "prompt", "prompt_digest", "source_dir", "snapshot_path", "cohort", "snapshot"] {
        assert!(value.get(absent).is_none());
    }
    let mut exact = Vec::new();
    publish_complete(&response, encoded.len(), &mut exact, &available).unwrap(); assert_eq!(exact, encoded);
    let mut refused = Vec::new();
    assert_eq!(publish_complete(&response, encoded.len() - 1, &mut refused, &available), Err(CandidateError::Output));
    assert!(refused.is_empty());
}

#[test]
fn cohort_metadata_never_wraps_or_changes_the_original_semantic_output() {
    let provenance = provenance();
    let semantic = serde_json::json!({"single_context_equivalence":false,"chunks":[{"chunk_id":0}],
        "aggregate":{"calibration":"uncalibrated","result":{"mode":"multi_label"}}});
    let serial = Response { provenance: &provenance, output: &semantic, cohort: None };
    let grouped = Response { provenance: &provenance, output: &semantic, cohort: Some(Cohort {
        schema_version: 1, execution: "fixture-native-chunk-cohorts", max_chunks: 3,
        chunk_cohorts: 2, group_steps: 7, projection_groups: 5 }) };
    let mut a = Vec::new(); let mut b = Vec::new();
    publish_complete(&serial, 4096, &mut a, &available).unwrap();
    publish_complete(&grouped, 4096, &mut b, &available).unwrap();
    let mut exact = Vec::new();
    publish_complete(&grouped, b.len(), &mut exact, &available).unwrap(); assert_eq!(exact, b);
    let mut refused = Vec::new();
    assert_eq!(publish_complete(&grouped, b.len() - 1, &mut refused, &available), Err(CandidateError::Output));
    assert!(refused.is_empty());
    let serial: serde_json::Value = serde_json::from_slice(&a).unwrap();
    let grouped: serde_json::Value = serde_json::from_slice(&b).unwrap();
    assert_eq!(serial["output"], grouped["output"]); assert_eq!(grouped["output"], semantic);
    assert_eq!(canonjson::canonical_bytes(&serial["output"]).unwrap(), canonjson::canonical_bytes(&grouped["output"]).unwrap());
    assert!(grouped["output"].get("output").is_none()); assert!(serial.get("cohort").is_none());
    assert_eq!(grouped["cohort"]["max_chunks"], 3); assert_eq!(grouped["cohort"]["chunk_cohorts"], 2);
    assert_eq!(grouped["cohort"]["group_steps"], 7); assert_eq!(grouped["cohort"]["projection_groups"], 5);
}

#[test]
fn late_serialization_or_prior_deadline_refuses_before_the_first_output_byte() {
    use std::cell::Cell;
    struct DuringSerialization<'a>(&'a Cell<bool>);
    impl Serialize for DuringSerialization<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.0.set(true); serializer.serialize_bool(true)
        }
    }
    let serialized = Cell::new(false); let value = DuringSerialization(&serialized);
    let deadline = || if serialized.get() { Err(CandidateError::Timeout) } else { available() };
    let mut output = Vec::new();
    assert_eq!(publish_complete(&value, 128, &mut output, &deadline), Err(CandidateError::Timeout));
    assert!(serialized.get()); assert!(output.is_empty());
    serialized.set(false);
    assert_eq!(publish_complete(&value, 128, &mut output, &|| Err(CandidateError::Timeout)), Err(CandidateError::Timeout));
    assert!(!serialized.get(), "an expired run must not begin serialization"); assert!(output.is_empty());
    let mut control = DeadlineControl { started: Instant::now(), deadline: Duration::ZERO };
    assert_eq!(control.checkpoint(0), Some(DecodeCancellationKind::Deadline));
    assert_eq!(control.prefill_checkpoint(0), Some(DecodeCancellationKind::Deadline));
}

#[test]
fn completion_verification_keeps_typed_deadlines_and_all_other_terminal_causes() {
    use crate::tasks::classify::ClassificationPlanningError;
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let error = Int4ClassificationMapError::from(ClassificationPlanningError::Cancelled(cause));
        let expected = if matches!(cause, Timeout | Deadline) { CandidateError::Timeout }
            else { CandidateError::Execution };
        assert_eq!(completion_error(error), expected);
    }
    assert_eq!(completion_error(Int4ClassificationMapError::Admission), CandidateError::Execution);
}

#[test]
fn complete_publication_polls_inside_large_strings_before_any_transport_byte() {
    use std::cell::Cell;
    let checks = Cell::new(0);
    let remaining = || {
        checks.set(checks.get() + 1);
        if checks.get() == 2 { Err(CandidateError::Timeout) } else { available() }
    };
    let value = "x".repeat(3 * COMPLETION_CHECKPOINT_BYTES);
    let mut output = Vec::new();
    assert_eq!(publish_complete(&value, value.len() + 3, &mut output, &remaining), Err(CandidateError::Timeout));
    assert_eq!(checks.get(), 2); assert!(output.is_empty());
    let mut expected = serde_json::to_vec(&value).unwrap(); expected.push(b'\n');
    publish_complete(&value, expected.len(), &mut output, &available).unwrap();
    assert_eq!(output, expected);
}

#[test]
fn serializer_failure_never_publishes_a_completed_prefix() {
    struct Refuse;
    impl Serialize for Refuse {
        fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
            Err(<S::Error as serde::ser::Error>::custom("private fixture serialization detail"))
        }
    }
    let mut output = Vec::new();
    assert_eq!(publish_complete(&Response { provenance: &provenance(), output: &Refuse, cohort: None },
        4096, &mut output, &available), Err(CandidateError::Output));
    assert!(output.is_empty(), "even a staged provenance prefix must remain private on failure");
}

#[test]
fn physical_write_flush_and_post_flush_deadline_all_remain_part_of_completion() {
    use std::cell::Cell;
    struct Writer<'a> { flushed: &'a Cell<bool>, bytes: Vec<u8>, fail_write: bool, fail_flush: bool }
    impl Write for Writer<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.fail_write { return Err(io::Error::other("private writer detail")); }
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
    let calls = Cell::new(0);
    let available = || { calls.set(calls.get() + 1); Ok(Duration::from_secs(1)) };
    let mut refused = Vec::new();
    assert_eq!(publish_complete(&serde_json::json!({"complete":true}), 1, &mut refused, &available),
        Err(CandidateError::Output));
    assert!(refused.is_empty()); assert_eq!(calls.get(), 1);
}

#[test]
fn excessive_actual_label_geometry_stops_before_model_io_and_snapshot_creation() {
    const CHILD: &str = "FNLP_CLASSIFY_MAP_GEOMETRY_FIXTURE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact",
                "candidate_cli::int4::classify_map::runtime::tests::excessive_actual_label_geometry_stops_before_model_io_and_snapshot_creation",
                "--nocapture"])
            .env(CHILD, "1").status().unwrap();
        assert!(status.success()); return;
    }
    let request = serde_json::to_vec(&serde_json::json!({"document":"a short source", "mode":"multi_label",
        "labels":(0..33).map(|index| serde_json::json!({"id":format!("label-{index}")})).collect::<Vec<_>>()})).unwrap();
    for route in 0..3 {
        let mut command = command(&["--cohort-chunks", "2"]);
        if route == 1 {
            command.args.source_dir = None; command.args.source_manifest = None;
            command.args.snapshot = Some(PathBuf::from("/nonexistent/classification-map-snapshot-must-not-open"));
            command.args.expected_model_sha256 = Some("ab".repeat(32));
        } else {
            command.args.source_dir = Some(PathBuf::from("/nonexistent/classification-map-source-must-not-open"));
            command.args.source_manifest = Some(PathBuf::from("/nonexistent/classification-map-manifest-must-not-open"));
            if route == 2 {
                command.args.save_snapshot = Some(PathBuf::from("/nonexistent/classification-map-snapshot-must-not-create"));
            }
        }
        let mut output = Vec::new();
        assert_eq!(command.execute(&mut io::Cursor::new(&request), &mut output), Err(CandidateError::Planning));
        assert!(output.is_empty());
    }
}

#[test]
fn generic_one_off_dispatch_cannot_bypass_document_admission() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("input accessed"); } }
    let command = command(&[]);
    let Int4Task::ClassifyMap(options) = command.task else { panic!("classification map required"); };
    let limits = options.validate(&command.args).unwrap();
    assert_eq!(super::super::super::runtime::execute_task(command.task, command.args, limits.common,
        &mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
}

mod model;
