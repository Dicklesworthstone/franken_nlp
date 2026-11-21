use super::*;

#[test]
fn all_cancellation_causes_keep_the_frozen_exit_category_through_task_and_host() {
    use DecodeCancellationKind::*;
    for (cause, exit) in [
        (Timeout, ErrorCode::BudgetOrTimeout), (Deadline, ErrorCode::BudgetOrTimeout),
        (PollQuota, ErrorCode::BudgetOrTimeout), (CostBudget, ErrorCode::BudgetOrTimeout),
        (User, ErrorCode::Cancelled), (ParentCancelled, ErrorCode::Cancelled),
        (Shutdown, ErrorCode::Cancelled), (ResourceUnavailable, ErrorCode::AdmissionOrResourceLimit),
        (FailFast, ErrorCode::Generic), (LinkedExit, ErrorCode::Generic), (RaceLost, ErrorCode::Generic),
    ] {
        for failure in [
            cancelled(cause), task_failure(SampledSentimentError::Cancelled(cause)),
            task_failure(SampledSentimentError::Native(Int4JsonError::Decode(JsonDecodeError::Cancelled(cause)))),
            host_failure(HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause)))),
            host_failure(HostedError::Int4(Int4HostError::SampledSentiment(
                SampledSentimentError::Native(Int4JsonError::Native(StrictInt4Error::Cancelled(cause)))))),
        ] {
            assert_eq!(failure.exit, exit);
            assert_eq!(failure.cancellation, Some(cause));
        }
    }
}

#[test]
fn budget_resource_and_structured_no_result_failures_are_distinct() {
    use SampledSentimentError::*;
    let exhausted_encoding = task_failure(TokenizerWork);
    assert_eq!(exhausted_encoding.exit, ErrorCode::BudgetOrTimeout);
    assert_eq!(exhausted_encoding.code, "sampled_sentiment_encoding_budget");
    assert_eq!(exhausted_encoding.cancellation, None);
    assert_eq!(host_failure(HostedError::Int4(Int4HostError::SampledSentiment(TokenizerWork))).code,
        "sampled_sentiment_encoding_budget");
    assert_eq!(admission_failure(CandidateError::Planning).exit, ErrorCode::AdmissionOrResourceLimit);
    assert_eq!(host_failure(HostedError::Int4(Int4HostError::Native(StrictInt4Error::Memory))).exit,
        ErrorCode::AdmissionOrResourceLimit);
    for error in [Limits, WorkBudget, Allocation,
        Native(Int4JsonError::Decode(JsonDecodeError::AllocationRefused)),
        Native(Int4JsonError::Native(StrictInt4Error::Context)),
        Native(Int4JsonError::Native(StrictInt4Error::Memory))] {
        assert_eq!(task_failure(error).exit, ErrorCode::AdmissionOrResourceLimit);
    }
    for error in [OutputBudget,
        Native(Int4JsonError::Decode(JsonDecodeError::BudgetExceeded("output before EOS"))),
        Native(Int4JsonError::Decode(JsonDecodeError::Mask(MaskOracleError::WorkBudgetExceeded {
            max_trie_node_visits: 1 }))),
        Native(Int4JsonError::Native(StrictInt4Error::Work))] {
        assert_eq!(task_failure(error).exit, ErrorCode::BudgetOrTimeout);
    }
    for error in [JsonDecodeError::NoLegalToken, JsonDecodeError::IndependentValidation] {
        let failure = task_failure(Native(Int4JsonError::Decode(error)));
        assert_eq!(failure.exit, ErrorCode::StructuredTaskNoResult);
        assert_eq!(failure.cancellation, None);
    }
    assert_eq!(task_failure(Schema).exit, ErrorCode::SchemaOrRecipeCompile);
    assert_eq!(task_failure(NoResult).exit, ErrorCode::StructuredTaskNoResult);
    assert_eq!(task_failure(Identity).exit, ErrorCode::ArtifactIntegrityOrFormatOrVersion);
}

#[test]
fn invariant_or_completion_failures_do_not_become_ordinary_no_result() {
    for error in [SampledSentimentError::Accounting, SampledSentimentError::Serialization,
        SampledSentimentError::Native(Int4JsonError::WorkMismatch),
        SampledSentimentError::Native(Int4JsonError::Decode(JsonDecodeError::InvalidLogits)),
        SampledSentimentError::Native(Int4JsonError::Decode(JsonDecodeError::IllegalTransition))] {
        assert_eq!(task_failure(error).exit, ErrorCode::Generic);
    }
    for error in [HostedError::Panicked, HostedError::CompletionMissing] {
        assert_eq!(host_failure(error).exit, ErrorCode::Generic);
    }
    assert_eq!(host_failure(HostedError::Limits("PRIVATE" )).exit, ErrorCode::AdmissionOrResourceLimit);
}

#[test]
fn compact_documents_use_actual_tokens_instead_of_the_byte_input_lower_bound() {
    use crate::tokenizer::{embedded::EmbeddedTokenizer, pinned_controls,
        untrusted::{SafeBpeLimits, SafeBpeWork, UntrustedDocumentEncoder, UntrustedEncoding}};
    let source = "The result was helpful and the service was excellent. ".repeat(4);
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let controls = pinned_controls::pinned().unwrap();
    let policy = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 4096,
        max_output_tokens: 4096, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 });
    let mut work = SafeBpeWork::default();
    let document = UntrustedDocumentEncoder::new(tokenizer.tokenizer(), controls.template_controls())
        .encode_with_policy(source.as_bytes(), &policy, &mut work, &mut || true).unwrap();
    assert_eq!(document.bytes(), source.as_bytes());
    assert!(document.ids().len() < source.len());
    let capacity = document.ids().len();
    assert_eq!(admit_source_bytes(&UntrustedEncoding::ByteFallback, source.len(), capacity),
        Err(CandidateError::Planning));
    // This only skips the byte-only early rejection. The planner still checks
    // the actual tokenized document plus every trusted prompt segment.
    assert_eq!(admit_source_bytes(&policy, source.len(), capacity), Ok(()));
    assert_eq!(admit_source_bytes(&UntrustedEncoding::ByteFallback, capacity, capacity), Ok(()));
    assert_eq!(work.input_bytes(), source.len());
}

#[test]
fn optional_encoding_and_compute_metadata_leave_default_result_bytes_unchanged() {
    use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};
    let output = serde_json::json!({"fixture_only":true,"samples_per_axis":2,"dimensions":[]});
    let provenance = Provenance {
        protocol: "fnlp-candidate-int4-sentiment-sampled-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        task: "sentiment-sampled", model_id: "Nanbeige4.2-3B", source_revision: "fixture",
        source_root_sha256: "fixture-source", model_content_sha256: "fixture-model",
        candidate_format: "fixture-format", quant_recipe: "fixture-recipe", group_width: 128,
        integer_kernel: "s_scalar_i32", snapshot: None, max_selected_rows: None,
    };
    let response = Response { provenance: &provenance, output: &output };
    let mut original = Vec::new();
    publish_checked(&response, 4096, &mut original, &|| Ok(())).unwrap();
    let mut serial = Vec::new();
    publish_checked(&input_encoding::Response { response: &compute::Response {
        response: &response, compute: None }, input_encoding: None },
        4096, &mut serial, &|| Ok(())).unwrap();
    assert_eq!(serial, original);
    let baseline: serde_json::Value = serde_json::from_slice(&original).unwrap();
    assert!(baseline.get("max_selected_rows").is_none());
    let receipt = crate::hosted::Int4ComputeReceipt { schedule: "int4-output-row-team-v1",
        compute_threads: 4, completed_projections: 31, completed_rows: 1000, completed_worker_rows: 750 };
    let policy = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 4096,
        max_output_tokens: 2048, max_pair_checks: 8192, max_pair_bytes: 65536 });
    for (compute, encoding) in [(Some(&receipt), None), (None, Some(&policy)), (Some(&receipt), Some(&policy))] {
        let envelope = input_encoding::Response { response: &compute::Response {
            response: &response, compute }, input_encoding: encoding };
        let mut bytes = Vec::new();
        publish_checked(&envelope, 4096, &mut bytes, &|| Ok(())).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["output"], output);
        assert_eq!(value.get("compute").is_some(), compute.is_some());
        assert_eq!(value.get("input_encoding").is_some(), encoding.is_some());
        if compute.is_some() { assert_eq!(value["compute"], serde_json::to_value(receipt).unwrap()); }
        if encoding.is_some() { assert_eq!(value["input_encoding"], serde_json::to_value(&policy).unwrap()); }
        for private in ["seed", "effective_seed", "document", "prompt_digest", "source_dir", "snapshot"] {
            assert!(value.get(private).is_none());
        }
        let mut refused = Vec::new();
        assert_eq!(publish_checked(&envelope, 4096, &mut refused, &|| Err(CandidateError::Timeout)),
            Err(CandidateError::Timeout));
        assert!(refused.is_empty());
    }
    let selected = Provenance { max_selected_rows: Some(4096), ..provenance };
    let mut metadata = serde_json::to_value(Response { provenance: &selected, output: &output }).unwrap();
    assert_eq!(metadata.as_object_mut().unwrap().remove("max_selected_rows"), Some(serde_json::json!(4096)));
    assert_eq!(metadata, baseline);
}

#[test]
fn complete_optional_metadata_fits_the_admitted_envelope_above_an_exact_result_cap() {
    use crate::{native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, quant_algebra::MAX_MODEL_K},
        tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding}};
    // This is a wire-size fixture, not an invented inference or compute receipt.
    let output = serde_json::json!({"fixture_only":true,"justification":"x".repeat(8192)});
    let result_cap = serde_json::to_vec(&output).unwrap().len();
    let digest = "f".repeat(64);
    let recipe = format!("{GROUPED_INT4_RECIPE}:group={MAX_MODEL_K}");
    let provenance = Provenance {
        protocol: "fnlp-candidate-int4-sentiment-sampled-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        task: "sentiment-sampled", model_id: "Nanbeige4.2-3B",
        source_revision: crate::artifact::converter::PINNED_REVISION,
        source_root_sha256: &digest, model_content_sha256: &digest,
        candidate_format: INT4_SOURCE_FORMAT, quant_recipe: &recipe, group_width: MAX_MODEL_K,
        integer_kernel: Int8KernelKind::ALL.into_iter().max_by_key(|kind| kind.id().len()).unwrap().id(),
        snapshot: Some(SnapshotRecord { operation: "loaded", format: INT4_SNAPSHOT_VERSION,
            file_bytes: u64::MAX }),
        max_selected_rows: Some(166144),
    };
    let receipt = crate::hosted::Int4ComputeReceipt { schedule: "int4-output-row-team-v1",
        compute_threads: 64, completed_projections: u64::MAX, completed_rows: u64::MAX,
        completed_worker_rows: u64::MAX };
    let policy = UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: SafeBpeLimits::MAX_INPUT_BYTES,
        max_output_tokens: SafeBpeLimits::MAX_OUTPUT_TOKENS,
        max_pair_checks: SafeBpeLimits::MAX_PAIR_CHECKS, max_pair_bytes: SafeBpeLimits::MAX_PAIR_BYTES,
    });
    let response = Response { provenance: &provenance, output: &output };
    let envelope = input_encoding::Response { response: &compute::Response {
        response: &response, compute: Some(&receipt) }, input_encoding: Some(&policy) };
    let complete_bytes = serde_json::to_vec(&envelope).unwrap().len() + 1;
    assert!(complete_bytes > result_cap);
    assert!(complete_bytes - result_cap <= 4096);
    let mut bytes = Vec::new();
    publish_checked(&envelope, result_cap + 4096, &mut bytes, &|| Ok(())).unwrap();
    assert_eq!(bytes.len(), complete_bytes);
    let mut refused = Vec::new();
    assert_eq!(publish_checked(&envelope, complete_bytes - 1, &mut refused, &|| Ok(())),
        Err(CandidateError::Output));
    assert!(refused.is_empty());
}

#[test]
fn compact_team_metadata_cannot_turn_an_expired_publication_into_success() {
    use std::cell::Cell;
    struct Expiring<'a> { expired: &'a Cell<bool> }
    impl Serialize for Expiring<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.expired.set(true);
            serde_json::json!({"output":"fixture"}).serialize(serializer)
        }
    }
    struct ExpiringFlush<'a> { expired: &'a Cell<bool>, bytes: Vec<u8> }
    impl Write for ExpiringFlush<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(bytes); Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { self.expired.set(true); Ok(()) }
    }
    let expired = Cell::new(false);
    let check = || if expired.get() { Err(CandidateError::Timeout) } else { Ok(()) };
    let receipt = crate::hosted::Int4ComputeReceipt { schedule: "int4-output-row-team-v1",
        compute_threads: 2, completed_projections: 17, completed_rows: 100, completed_worker_rows: 50 };
    let mut bytes = Vec::new();
    assert_eq!(publish_checked(&compute::Response { response: &Expiring { expired: &expired },
        compute: Some(&receipt) }, 4096, &mut bytes, &check), Err(CandidateError::Timeout));
    assert!(bytes.is_empty());
    expired.set(false);
    let mut sink = ExpiringFlush { expired: &expired, bytes: Vec::new() };
    assert_eq!(publish_checked(&compute::Response { response: &serde_json::json!({"output":"fixture"}),
        compute: Some(&receipt) }, 4096, &mut sink, &check), Err(CandidateError::Timeout));
    assert!(!sink.bytes.is_empty());
}
