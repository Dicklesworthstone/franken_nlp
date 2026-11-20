//! Boundary fixtures exercise no native model, tokenizer or source IO.
use super::*;
use super::super::tests::{command, NoRead, NoWrite};
use crate::{hosted::RunStop, native_engine::{constrained::JsonDecodeError,
    constrained_int4::Int4JsonError}, tasks::sentiment::sampled::SampledSentimentError};

#[test]
fn host_quote_matches_all_cli_owners_and_prices_prefill_workspace_once() {
    for flags in [
        &[][..], &["--reuse-prefill"][..], &["--prefill-rows", "1"][..],
        &["--reuse-prefill", "--prefill-rows", "64"][..],
        &["--input-encoding", "safe-bpe-v1", "--compute-threads", "4"][..],
        &["--input-encoding", "safe-bpe-v1", "--reuse-prefill", "--prefill-rows", "4"][..],
    ] {
        let (args, options) = command(flags);
        let limits = options.validate(&args).unwrap();
        let encoding = options.input_encoding(&args).unwrap();
        let config = host_config(&args, options, limits, Duration::from_secs(17)).unwrap();
        check_host_quote(&args, options, limits, &config, &encoding).unwrap();
        assert_eq!(config.native.run.max_elapsed, Duration::from_secs(17));
        assert_eq!(config.native.run.max_checkpoints, args.max_checkpoints);
        assert_eq!(config.native.run.cleanup_reserve_bytes, CLEANUP_BYTES);
        assert_eq!(config.max_mask_node_visits, options.sampled.max_mask_node_visits);
        assert_eq!(config.mapping.max_model_work, options.sampled.work_ceiling());
        assert_eq!(config.options, options.sampled.options(&args).unwrap());
        assert_eq!(config.reuse_prefill, options.sampled.reuse_prefill);
        let task = options.sampled.budget(&args, limits.common, 4).unwrap();
        let quote = config.requirements_with_encoding(args.group_width, task, 4, &encoding).unwrap();
        let extra = config.prefill.map(|p| p.validate().unwrap()).unwrap_or(0);
        assert_eq!(quote.workspace_bytes, limits.memory.workspace_bytes + extra);
        assert_eq!(quote.native, limits.memory.native);
        assert_eq!(quote.input_bytes, limits.memory.input_bytes);
        assert_eq!(quote.max_output_tokens,
            (options.mapping.max_chunks * 4 * options.sampled.samples * args.max_new_tokens) as u64);
        for field in 0..6 {
            let mut underquoted = limits;
            match field {
                0 => underquoted.memory.input_bytes -= 1,
                1 => underquoted.memory.workspace_bytes -= 1,
                2 => underquoted.memory.output_reservation_bytes -= 1,
                3 => underquoted.memory.max_output_bytes -= 1,
                4 => underquoted.memory.max_output_tokens -= 1,
                _ => underquoted.memory.native.kv_bytes -= 1,
            }
            assert_eq!(check_host_quote(&args, options, underquoted, &config, &encoding).unwrap_err().code,
                "sampled_sentiment_map_memory_quote");
        }
        if extra != 0 {
            let mut double = limits; double.memory.workspace_bytes += extra;
            assert_eq!(check_host_quote(&args, options, double, &config, &encoding).unwrap_err().code,
                "sampled_sentiment_map_memory_quote");
        }
    }
}

#[test]
fn runtime_boundary_revalidates_mutated_options_before_runtime_or_private_io() {
    let (args, mut options) = command(&[]);
    let admitted = options.validate(&args).unwrap();
    options.mapping.max_chunks = 0;
    assert_eq!(execute(args, options, admitted, &mut NoRead, &mut NoWrite).unwrap_err().exit, ErrorCode::Usage);
}

#[test]
fn task_and_host_preserve_every_cancellation_cause_and_the_underlying_failure() {
    use DecodeCancellationKind::*;
    for (cause, exit) in [
        (Timeout, ErrorCode::BudgetOrTimeout), (Deadline, ErrorCode::BudgetOrTimeout),
        (PollQuota, ErrorCode::BudgetOrTimeout), (CostBudget, ErrorCode::BudgetOrTimeout),
        (User, ErrorCode::Cancelled), (ParentCancelled, ErrorCode::Cancelled),
        (Shutdown, ErrorCode::Cancelled), (ResourceUnavailable, ErrorCode::AdmissionOrResourceLimit),
        (FailFast, ErrorCode::Generic), (LinkedExit, ErrorCode::Generic), (RaceLost, ErrorCode::Generic),
    ] {
        for error in [SampledSentimentMapError::Cancelled(cause),
            SampledSentimentMapError::Sampled(SampledSentimentError::Cancelled(cause)),
            SampledSentimentMapError::Sampled(SampledSentimentError::Native(
                Int4JsonError::Decode(JsonDecodeError::Cancelled(cause))))] {
            let failure = map_failure(error);
            assert_eq!(failure.exit, exit);
            assert_eq!(failure.cancellation, Some(cause));
        }
        let failure = host_failure(HostedError::Int4(Int4HostError::SampledSentimentMap(
            SampledSentimentMapError::Cancelled(cause))));
        assert_eq!(failure.exit, exit);
        assert_eq!(failure.cancellation, Some(cause));
        let failure = host_failure(HostedError::Stopped {
            stop: RunStop { kind: cause, runtime_reason: None, runtime_checkpoint_error: None }, task_error: None });
        assert_eq!(failure.exit, exit);
        assert_eq!(failure.cancellation, Some(cause));
    }
    for cause in [FailFast, LinkedExit] {
        let failure = host_failure(HostedError::Stopped {
            stop: RunStop { kind: cause, runtime_reason: None, runtime_checkpoint_error: None },
            task_error: Some(Box::new(HostedError::Int4(Int4HostError::SampledSentimentMap(
                SampledSentimentMapError::Output)))) });
        assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
        assert_eq!(failure.code, "sampled_sentiment_map_budget");
        assert_eq!(failure.cancellation, Some(cause));
    }
}

#[test]
fn partition_refusal_exhausted_work_no_result_and_invariants_remain_distinct() {
    use SampledSentimentMapError::*;
    for error in [Limits, Work, Allocation, Chunk(MapReduceError::AllocationRefused), EmptySource, Chunk(MapReduceError::InvalidLimits),
        Chunk(MapReduceError::InputBudget), Chunk(MapReduceError::ChunkBudget), Chunk(MapReduceError::TokenBudget)] {
        assert_eq!(map_failure(error).exit, ErrorCode::AdmissionOrResourceLimit);
    }
    for error in [Output, Chunk(MapReduceError::TokenizerBudget), Sampled(SampledSentimentError::TokenizerWork)] {
        assert_eq!(map_failure(error).exit, ErrorCode::BudgetOrTimeout);
    }
    assert_eq!(map_failure(Sampled(SampledSentimentError::NoResult)).exit, ErrorCode::StructuredTaskNoResult);
    assert_eq!(map_failure(Sampled(SampledSentimentError::Schema)).exit, ErrorCode::SchemaOrRecipeCompile);
    assert_eq!(map_failure(Sampled(SampledSentimentError::Identity)).exit, ErrorCode::ArtifactIntegrityOrFormatOrVersion);
    for error in [Admission, Aggregation, Serialization, Chunk(MapReduceError::InvalidSpan),
        Sampled(SampledSentimentError::Accounting)] {
        let failure = map_failure(error);
        assert_eq!(failure.exit, ErrorCode::Generic);
        assert_eq!(failure.cancellation, None);
    }
}

#[test]
fn cancellation_or_expiry_during_staging_publishes_no_document_prefix() {
    use std::cell::Cell;
    struct Expiring<'a>(&'a Cell<bool>);
    impl Serialize for Expiring<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.0.set(true); serde_json::json!({"fixture_only":true}).serialize(serializer)
        }
    }
    struct Cancelling<'a>(&'a CancellationToken, DecodeCancellationKind);
    impl Serialize for Cancelling<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.0.cancel(self.1); serde_json::json!({"fixture_only":true}).serialize(serializer)
        }
    }
    let expired = Cell::new(false);
    let remaining = || if expired.get() { Err(CandidateError::Timeout) } else { Ok(Duration::from_secs(1)) };
    let failure = publish_complete(&Expiring(&expired), 4096, &mut NoWrite, &remaining,
        &CancellationToken::default()).unwrap_err();
    assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::Deadline));
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Deadline,
        DecodeCancellationKind::PollQuota, DecodeCancellationKind::ResourceUnavailable] {
        let token = CancellationToken::default();
        let failure = publish_complete(&Cancelling(&token, cause), 4096, &mut NoWrite,
            &|| Ok(Duration::from_secs(1)), &token).unwrap_err();
        assert_eq!(failure.cancellation, Some(cause));
        assert_eq!(failure.exit, cancelled(cause).exit);
        // Once cancelled, no subsequent serialization or output attempt can
        // publish the already completed document under a renewed allowance.
        let failure = publish_complete(&serde_json::json!({"fixture_only":true}), 4096,
            &mut NoWrite, &|| Ok(Duration::from_secs(1)), &token).unwrap_err();
        assert_eq!(failure.cancellation, Some(cause));
    }
    let failure = publish_complete(&serde_json::json!({"fixture_only":true}), 4096,
        &mut NoWrite, &|| Err(CandidateError::Timeout), &CancellationToken::default()).unwrap_err();
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::Deadline));
}

#[test]
fn cancellation_at_flush_and_output_failure_never_report_complete_success() {
    struct CancelAtFlush<'a> { token: &'a CancellationToken, bytes: Vec<u8> }
    impl Write for CancelAtFlush<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.bytes.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { self.token.cancel(DecodeCancellationKind::User); Ok(()) }
    }
    struct BrokenOutput;
    impl Write for BrokenOutput {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { Err(io::Error::other("private sink detail")) }
        fn flush(&mut self) -> io::Result<()> { unreachable!() }
    }
    let value = serde_json::json!({"fixture_only":true});
    let token = CancellationToken::default();
    let mut writer = CancelAtFlush { token: &token, bytes: Vec::new() };
    let failure = publish_complete(&value, 4096, &mut writer,
        &|| Ok(Duration::from_secs(1)), &token).unwrap_err();
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::User));
    assert_eq!(failure.exit, ErrorCode::Cancelled);
    assert!(!writer.bytes.is_empty());
    let failure = publish_complete(&value, 4096, &mut BrokenOutput,
        &|| Ok(Duration::from_secs(1)), &CancellationToken::default()).unwrap_err();
    assert_eq!(failure.exit, ErrorCode::Generic);
    assert!(!failure.code.contains("private"));
}

#[test]
fn response_keeps_complete_result_and_optional_physical_metadata_inside_its_quote() {
    // Only a serialization-size fixture: no fabricated model proof or native receipt.
    let output = serde_json::json!({"fixture_only":true,"all_chunks":"x".repeat(8192)});
    let digest = "f".repeat(64);
    let result_bytes = serde_json::to_vec(&output).unwrap().len();
    let mut response = Response { protocol: "fnlp-candidate-int4-sentiment-sampled-map-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "sentiment-sampled-map",
        model_id: "Nanbeige4.2-3B", source_revision: crate::artifact::converter::PINNED_REVISION,
        source_root_sha256: &digest, model_content_sha256: &digest,
        candidate_format: INT4_SOURCE_FORMAT, quant_recipe: "fixture-quant-recipe", group_width: 128,
        integer_kernel: "s_scalar_i32", snapshot: None, prefill_schedule: None, prefill_reuse: None, output: &output };
    let original = serde_json::to_value(&response).unwrap();
    for field in ["snapshot", "prefill_schedule", "prefill_reuse", "compute", "seed", "effective_seed",
        "document", "source_dir", "prompt_digest", "request_binding", "result_binding"] {
        assert!(original.get(field).is_none());
    }
    response.snapshot = Some(SnapshotRecord { operation: "loaded", format: INT4_SNAPSHOT_VERSION, file_bytes: u64::MAX });
    let prefill = Int4PrefillLimits { max_batch_rows: 64,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(64).unwrap() + MIB };
    response.prefill_schedule = Some(PrefillSchedule::new(prefill).unwrap());
    response.prefill_reuse = Some(crate::tasks::sentiment::sampled::SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION);
    let mut bytes = Vec::new();
    publish_complete(&response, result_bytes + 4096, &mut bytes,
        &|| Ok(Duration::from_secs(1)), &CancellationToken::default()).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["output"], output);
    assert_eq!(value["prefill_schedule"]["execution"], INT4_LAYER_MAJOR_PREFILL_VERSION);
    assert_eq!(value["prefill_schedule"]["max_batch_rows"], 64);
    assert_eq!(value["prefill_schedule"]["admitted_workspace_bytes"], prefill.validate().unwrap());
    assert!(bytes.len() > result_bytes && bytes.len() <= result_bytes + 4096);
    let failure = publish_complete(&response, bytes.len() - 1, &mut NoWrite,
        &|| Ok(Duration::from_secs(1)), &CancellationToken::default()).unwrap_err();
    assert_eq!(failure.exit, ErrorCode::Generic);
    assert!(PrefillSchedule::new(Int4PrefillLimits { max_batch_rows: 0, max_extra_scratch_bytes: u64::MAX }).is_err());
}
