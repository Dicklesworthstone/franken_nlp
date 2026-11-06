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
            assert_eq!(host_error(HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause))), fallback), expected);
            assert_eq!(host_error(HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause))), fallback), expected);
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
    for width in [1, 3, 128, 10752] {
        let mut command = command(&[]);
        command.args.group_width = width; command.args.max_load_mib = 65536;
        let Int4Task::ClassifyMap(options) = command.task else { panic!("map required"); };
        let limits = options.validate(&command.args).unwrap(); let common = limits.common;
        let config = Int4ClassificationMapHostConfig {
            native: NativeLimits { context_tokens: command.args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
                run: RunLimits { max_elapsed: Duration::from_secs(command.args.timeout_seconds),
                    max_checkpoints: command.args.max_checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES } },
            planning: limits.planning, mapping: limits.mapping, preparation_reserve_bytes: common.preparation_bytes,
        };
        let quote = config.requirements(width, options.budget(common, limits.planning.max_labels).unwrap()).unwrap();
        assert_eq!(quote, limits.memory);
        assert_eq!(quote.native.kv_bytes, common.native.kv_bytes);
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
    let response = Response { provenance: &provenance, output: &semantic };
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
fn serializer_failure_never_publishes_a_completed_prefix() {
    struct Refuse;
    impl Serialize for Refuse {
        fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
            Err(<S::Error as serde::ser::Error>::custom("private fixture serialization detail"))
        }
    }
    let mut output = Vec::new();
    assert_eq!(publish_complete(&Response { provenance: &provenance(), output: &Refuse },
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
