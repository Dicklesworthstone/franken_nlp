//! Typed failure and publication contracts; native inference has its own gate.
use super::*;
use super::super::tests::command;

#[test]
fn map_retains_pinned_full_vocabulary_options_and_typed_native_cancellation() {
    use crate::{native_engine::lmhead::scoring::ScoringMode, tasks::sentiment::SentimentPolicy};
    let policy = SentimentPolicy { minimum_peak_weight_ppm: 700000, maximum_normalized_entropy_ppm: 600000 };
    let options = sentiment_options(policy).unwrap();
    assert_eq!(options.mode, ScoringMode::FullVocabulary); assert_eq!(options.policy, policy);
    let controls = crate::tokenizer::pinned_controls::pinned().unwrap();
    let eos = controls.template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    assert_eq!(options.eos_token_id, eos);
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout, DecodeCancellationKind::User] {
        let expected = if cause == DecodeCancellationKind::User { CandidateError::Execution } else { CandidateError::Timeout };
        assert_eq!(planning_error(Int4SentimentMapError::Sentiment(Int4SentimentError::Cancelled(cause))), expected);
        assert_eq!(execution_error(HostedError::Int4(Int4HostError::SentimentMap(
            Int4SentimentMapError::Sentiment(Int4SentimentError::Cancelled(cause))))), expected);
    }
    assert_eq!(planning_error(Int4SentimentMapError::Work), CandidateError::Planning);
    assert_eq!(execution_error(HostedError::ModelIdentity), CandidateError::Identity);
    assert_eq!(execution_error(HostedError::ResourceDomain), CandidateError::Identity);
}

#[test]
fn actual_host_configuration_uses_the_same_pure_pre_io_memory_quote() {
    use crate::tasks::sentiment::SentimentPolicy;
    for flags in [vec![], vec!["--cohort-chunks", "3"], vec!["--cohort-chunks", "64"]] {
        let command = command(&flags);
        let Int4Task::SentimentMap(options) = command.task else { panic!("map required"); };
        let limits = options.validate(&command.args).unwrap(); let common = limits.common;
        let config = Int4SentimentMapHostConfig {
            native: NativeLimits { context_tokens: command.args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
                run: RunLimits { max_elapsed: Duration::from_secs(command.args.timeout_seconds),
                    max_checkpoints: command.args.max_checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES } },
            planning: limits.planning, mapping: limits.mapping,
            options: sentiment_options(SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 }).unwrap(),
            preparation_reserve_bytes: common.preparation_bytes,
        };
        let quote = config.requirements(command.args.group_width, options.budget(common).unwrap(), 4,
            options.mapping.cohort_chunks).unwrap();
        assert_eq!(quote.native, limits.memory.native);
        assert_eq!(quote.workspace_bytes, limits.memory.workspace_bytes);
        assert_eq!(quote.input_bytes, limits.memory.input_bytes);
        assert_eq!(quote.output_reservation_bytes, limits.memory.output_reservation_bytes);
        assert_eq!(quote.max_output_bytes, limits.memory.max_output_bytes);
    }
}

#[test]
fn map_publication_keeps_semantic_output_separate_from_cohorts_and_checks_complete_newline_cap() {
    let provenance = Provenance { protocol: "fnlp-candidate-int4-sentiment-map-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "sentiment-map",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "fixture-source",
        model_content_sha256: "fixture-model", candidate_format: "fixture-format", quant_recipe: "fixture:group=128",
        group_width: 128, integer_kernel: "s_scalar_i32", snapshot: None };
    let semantic = serde_json::json!({"single_context_equivalence":false,"chunks":[],
        "aggregate":{"calibration":"uncalibrated","dimensions":[]}});
    let response = Response { provenance: &provenance, output: &semantic, cohort: Some(Cohort {
        schema_version: 1, execution: "fixture-native-cohorts".to_owned(),
        max_chunks: 3, chunk_cohorts: 2, group_steps: 7, projection_groups: 5 }) };
    let available = || Ok(Duration::from_secs(1));
    let mut encoded = Vec::new(); publish_complete(&response, 4096, &mut encoded, &available).unwrap();
    assert_eq!(encoded.last(), Some(&b'\n'));
    let value: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(value["output"], semantic); assert_eq!(value["cohort"]["chunk_cohorts"], 2);
    assert_eq!(value["protocol"], "fnlp-candidate-int4-sentiment-map-v1");
    for absent in ["document", "prompt", "prompt_digest", "source_dir", "snapshot_path"] { assert!(value.get(absent).is_none()); }
    let mut exact = Vec::new(); publish_complete(&response, encoded.len(), &mut exact, &available).unwrap(); assert_eq!(exact, encoded);
    let mut refused = Vec::new();
    assert_eq!(publish_complete(&response, encoded.len() - 1, &mut refused, &available), Err(CandidateError::Output));
    assert!(refused.is_empty());
    struct FlushFailure;
    impl Write for FlushFailure {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("private fixture path")) }
    }
    assert_eq!(publish_complete(&response, encoded.len(), &mut FlushFailure, &available), Err(CandidateError::Output));
    let mut serial = Vec::new();
    publish_complete(&Response { provenance: &provenance, output: &semantic, cohort: None }, 4096, &mut serial, &available).unwrap();
    assert!(serde_json::from_slice::<serde_json::Value>(&serial).unwrap().get("cohort").is_none());
}

#[test]
fn final_publication_checks_elapsed_deadline_after_physical_flush() {
    use std::cell::Cell;
    struct Writer<'a> { flushed: &'a Cell<bool>, bytes: Vec<u8> }
    impl Write for Writer<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(bytes); Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { self.flushed.set(true); Ok(()) }
    }
    let flushed = Cell::new(false);
    let mut writer = Writer { flushed: &flushed, bytes: Vec::new() };
    let expired = || if flushed.get() { Err(CandidateError::Timeout) } else { Ok(Duration::from_secs(1)) };
    assert_eq!(publish_complete(&serde_json::json!({"complete":true}), 128, &mut writer, &expired),
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
fn deadline_before_or_during_serialization_never_starts_document_publication() {
    use std::cell::Cell;
    struct SlowResult<'a>(&'a Cell<bool>);
    impl Serialize for SlowResult<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.0.set(true); serializer.serialize_str("completed document result")
        }
    }
    let serialized = Cell::new(false);
    let deadline = || if serialized.get() { Err(CandidateError::Timeout) } else { Ok(Duration::from_secs(1)) };
    let mut output = Vec::new();
    assert_eq!(publish_complete(&SlowResult(&serialized), 128, &mut output, &deadline), Err(CandidateError::Timeout));
    assert!(serialized.get()); assert!(output.is_empty());
    serialized.set(false);
    assert_eq!(publish_complete(&SlowResult(&serialized), 128, &mut output,
        &|| Err(CandidateError::Timeout)), Err(CandidateError::Timeout));
    assert!(!serialized.get(), "expired invocations must refuse before serialization");
    assert!(output.is_empty());
}

mod model;
