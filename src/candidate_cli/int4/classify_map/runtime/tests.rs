//! Typed failures, resource quotes and complete-document publication.
use super::*;
use super::super::tests::command;

#[test]
fn map_preserves_native_cancellation_and_identity_failures() {
    use crate::tasks::classify::ClassificationPlanningError;
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout, DecodeCancellationKind::User] {
        let expected = if cause == DecodeCancellationKind::User { CandidateError::Execution } else { CandidateError::Timeout };
        let failure = Int4ClassificationError::Planning(ClassificationPlanningError::Cancelled(cause));
        assert_eq!(classification_error(failure.clone()), expected);
        assert_eq!(planning_error(Int4ClassificationMapError::Classification(failure.clone())), expected);
        assert_eq!(execution_error(HostedError::Int4(Int4HostError::ClassificationMap(
            Int4ClassificationMapError::Classification(failure)))), expected);
    }
    assert_eq!(planning_error(Int4ClassificationMapError::Work), CandidateError::Planning);
    assert_eq!(execution_error(HostedError::ModelIdentity), CandidateError::Identity);
    assert_eq!(execution_error(HostedError::ResourceDomain), CandidateError::Identity);
}

#[test]
fn host_configuration_and_cli_use_the_same_pure_memory_quote() {
    let command = command(&[]);
    let Int4Task::ClassifyMap(options) = command.task else { panic!("classification map required"); };
    let limits = options.validate(&command.args).unwrap(); let common = limits.common;
    let config = Int4ClassificationMapHostConfig {
        native: NativeLimits { context_tokens: command.args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: Duration::from_secs(command.args.timeout_seconds),
                max_checkpoints: command.args.max_checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES } },
        planning: limits.planning, mapping: limits.mapping,
        preparation_reserve_bytes: common.preparation_bytes,
    };
    assert_eq!(config.requirements(command.args.group_width, options.budget(common)).unwrap(), limits.memory);
}

#[test]
fn complete_publication_retains_semantics_and_provenance_without_source_or_path_leaks() {
    let provenance = Provenance { protocol: "fnlp-candidate-int4-classify-map-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "classify-map",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "fixture-source",
        model_content_sha256: "fixture-model", candidate_format: "fixture-format", quant_recipe: "fixture:group=128",
        group_width: 128, integer_kernel: "s_scalar_i32", snapshot: None };
    let semantic = serde_json::json!({"single_context_equivalence":false,"chunks":[],
        "aggregate":{"calibration":"uncalibrated","mode":"exclusive","labels":[]}});
    let response = Response { provenance: &provenance, output: &semantic };
    let available = || Ok(Duration::from_secs(1));
    let mut encoded = Vec::new(); publish_complete(&response, 4096, &mut encoded, &available).unwrap();
    assert_eq!(encoded.last(), Some(&b'\n'));
    let value: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(value["output"], semantic);
    assert_eq!(value["protocol"], "fnlp-candidate-int4-classify-map-v1");
    for absent in ["document", "prompt", "prompt_digest", "source_dir", "snapshot_path"] { assert!(value.get(absent).is_none()); }
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
    assert!(output.is_empty(), "a fully serialized but late response must not become visible");
    assert_eq!(publish_complete(&serde_json::json!({"complete":true}), 128, &mut output,
        &|| Err(CandidateError::Timeout)), Err(CandidateError::Timeout));
    assert!(output.is_empty());
}

#[test]
fn publication_rechecks_the_deadline_after_physical_flush_and_never_writes_an_oversized_result() {
    use std::cell::Cell;
    struct Writer<'a> { flushed: &'a Cell<bool>, bytes: Vec<u8> }
    impl Write for Writer<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.bytes.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { self.flushed.set(true); Ok(()) }
    }
    let flushed = Cell::new(false);
    let mut writer = Writer { flushed: &flushed, bytes: Vec::new() };
    let available = || if flushed.get() { Err(CandidateError::Timeout) } else { Ok(Duration::from_secs(1)) };
    assert_eq!(publish_complete(&serde_json::json!({"complete":true}), 128, &mut writer, &available),
        Err(CandidateError::Timeout));
    assert_eq!(writer.bytes.last(), Some(&b'\n'));
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
