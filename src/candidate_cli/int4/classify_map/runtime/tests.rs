//! Typed failures, resource quotes and complete-document publication.
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
fn host_configuration_and_cli_use_the_same_pure_memory_quote() {
    for flags in [vec![], vec!["--cohort-chunks", "1"], vec!["--cohort-chunks", "3"],
        vec!["--cohort-chunks", "64", "--max-result-bytes", "524288"]] {
        for width in [1, 3, 128, 10752] {
            let mut command = command(&flags);
            command.args.group_width = width; command.args.max_load_mib = 65536;
            let Int4Task::ClassifyMap(options) = command.task else { panic!("classification map required"); };
            let limits = options.validate(&command.args).unwrap(); let common = limits.common;
            let config = Int4ClassificationMapHostConfig {
                native: NativeLimits { context_tokens: command.args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
                    run: RunLimits { max_elapsed: Duration::from_secs(command.args.timeout_seconds),
                        max_checkpoints: command.args.max_checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES } },
                planning: limits.planning, mapping: limits.mapping,
                preparation_reserve_bytes: common.preparation_bytes,
            };
            let budget = options.budget(common, limits.planning.max_labels).unwrap();
            let quote = if let Some(chunks) = options.mapping.cohort_chunks {
                let quote = config.cohort_requirements(width, budget, options.cohort_head_limit().unwrap(), chunks).unwrap();
                // The largest admissible shape has been priced; smaller actual
                // exclusive/multilabel windows cannot need a larger native peak.
                let actual = config.cohort_requirements(width, budget, 1, chunks).unwrap();
                assert!(actual.memory.native.kv_bytes <= quote.memory.native.kv_bytes);
                assert!(actual.memory.workspace_bytes <= quote.memory.workspace_bytes);
                assert!(actual.memory.input_bytes <= quote.memory.input_bytes);
                assert_eq!(actual.memory.output_reservation_bytes, quote.memory.output_reservation_bytes);
                quote.memory
            } else {
                let quote = config.requirements(width, budget).unwrap();
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
fn complete_publication_retains_semantics_and_provenance_without_source_or_path_leaks() {
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
    assert!(value.get("cohort").is_none(), "serial publication retains the existing wire shape");
    for absent in ["document", "prompt", "prompt_digest", "source_dir", "snapshot_path", "snapshot"] {
        assert!(value.get(absent).is_none());
    }
    let mut exact = Vec::new();
    publish_complete(&response, encoded.len(), &mut exact, &available).unwrap(); assert_eq!(exact, encoded);
    let mut refused = Vec::new();
    assert_eq!(publish_complete(&response, encoded.len() - 1, &mut refused, &available), Err(CandidateError::Output));
    assert!(refused.is_empty());
    struct FlushFailure;
    impl Write for FlushFailure {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("private fixture path")) }
    }
    assert_eq!(publish_complete(&response, encoded.len(), &mut FlushFailure, &available), Err(CandidateError::Output));
    let response = Response { provenance: &provenance, output: &semantic,
        cohort: Some(Cohort { schema_version: 1,
            execution: crate::tasks::classify::grouped_int4::long::INT4_CLASSIFICATION_MAP_COHORT_EXECUTION,
            max_chunks: 3, chunk_cohorts: 2, group_steps: 7, projection_groups: 5 }) };
    let mut grouped = Vec::new(); publish_complete(&response, 8192, &mut grouped, &available).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&grouped).unwrap();
    assert_eq!(value["output"], semantic, "physical grouping must stay outside the semantic output");
    assert_eq!(value["cohort"]["max_chunks"], 3); assert_eq!(value["cohort"]["chunk_cohorts"], 2);
    let mut exact = Vec::new();
    publish_complete(&response, grouped.len(), &mut exact, &available).unwrap(); assert_eq!(exact, grouped);
    assert_eq!(publish_complete(&response, grouped.len() - 1, &mut refused, &available), Err(CandidateError::Output));
    assert!(refused.is_empty());
}

#[test]
fn oversized_actual_window_refuses_before_source_or_snapshot_loading_and_persistence() {
    let json = r#"{"document":"short source","labels":[{"id":"a"},{"id":"b"},{"id":"c"}],"mode":"multi_label"}"#;
    for route in 0..3 {
        let mut command = command(&["--cohort-chunks", "32"]);
        if route == 1 {
            command.args.source_dir = None; command.args.source_manifest = None;
            command.args.snapshot = Some("fixture-must-not-be-opened.snapshot".into());
            command.args.expected_model_sha256 = Some("ab".repeat(32));
        } else if route == 2 {
            command.args.save_snapshot = Some("fixture-must-not-be-created.snapshot".into());
        }
        let mut output = Vec::new();
        assert_eq!(command.execute(&mut io::Cursor::new(json.as_bytes()), &mut output), Err(CandidateError::Planning));
        assert!(output.is_empty());
    }
}

#[test]
fn serialization_must_finish_before_the_publication_deadline() {
    use std::cell::Cell;
    struct ExpiresDuringSerialization<'a>(&'a Cell<bool>);
    impl Serialize for ExpiresDuringSerialization<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.0.set(true); serializer.serialize_str("complete document")
        }
    }
    let expired = Cell::new(false);
    let available = || if expired.get() { Err(CandidateError::Timeout) } else { Ok(Duration::from_secs(1)) };
    let mut output = Vec::new();
    assert_eq!(publish_complete(&ExpiresDuringSerialization(&expired), 128, &mut output, &available),
        Err(CandidateError::Timeout));
    assert!(expired.get());
    assert!(output.is_empty(), "a fully serialized but late response must not become visible");
    expired.set(false);
    assert_eq!(publish_complete(&ExpiresDuringSerialization(&expired), 128, &mut output,
        &|| Err(CandidateError::Timeout)), Err(CandidateError::Timeout));
    assert!(!expired.get(), "an expired run must not begin serialization"); assert!(output.is_empty());
    let mut control = DeadlineControl { started: Instant::now(), deadline: Duration::ZERO };
    assert_eq!(control.checkpoint(0), Some(DecodeCancellationKind::Deadline));
    assert_eq!(control.prefill_checkpoint(0), Some(DecodeCancellationKind::Deadline));
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
fn publication_rechecks_the_deadline_after_physical_flush_and_never_writes_an_oversized_result() {
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
