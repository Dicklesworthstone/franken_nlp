//! Hosted quote consistency, typed failures and private provenance boundaries.
use super::*;
use super::super::tests::command;
use super::super::super::tests::{NoRead, NoWrite};
use crate::{hosted::{Int4HostError, RunStop},
    tasks::sentiment::sampled::{SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION,
        long::SampledSentimentMapError}};

#[test]
fn shared_host_quote_matches_every_cli_memory_owner_before_any_io() {
    for snapshot in [false, true] {
        for flags in [
            &[][..], &["--reuse-prefill"][..], &["--prefill-rows", "1"][..],
            &["--reuse-prefill", "--prefill-rows", "64"][..],
            &["--compute-threads", "2"][..], &["--compute-threads", "64"][..],
            &["--input-encoding", "safe-bpe-v1", "--compute-threads", "4"][..],
            &["--input-encoding", "safe-bpe-v1", "--reuse-prefill", "--prefill-rows", "4"][..],
        ] {
            let (args, options) = command(flags, snapshot);
            let limits = options.validate(&args).unwrap();
            let config = host_config(&args, options, limits, Duration::from_secs(19)).unwrap();
            check_host_quote(&args, limits, &config).unwrap();
            assert_eq!(config.document.native.run.max_elapsed, Duration::from_secs(19));
            assert_eq!(config.document.native.run.max_checkpoints, args.max_checkpoints);
            assert_eq!(config.max_model_work, options.document.sampled.work_ceiling());
            assert_eq!(config.max_model_work, config.document.mapping.max_model_work);
            assert_eq!(config.max_mask_node_visits, options.document.sampled.max_mask_node_visits);
            assert_eq!(config.max_mask_node_visits, config.document.max_mask_node_visits);
            assert_eq!(config.encoding, options.document.input_encoding(&args).unwrap());
            assert_eq!(config.document.options, options.document.sampled.options(&args).unwrap());
            assert_eq!(config.document.reuse_prefill, options.document.sampled.reuse_prefill);
            assert_eq!(config.transport.max_work, limits.transport.max_work);
            assert_eq!(config.transport.max_output_bytes, limits.transport.max_output_bytes);
            assert_eq!(config.io_reserve_bytes, limits.io_bytes);
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
                assert_eq!(check_host_quote(&args, underquoted, &config).unwrap_err().code,
                    "sampled_sentiment_corpus_memory_quote");
            }
        }
    }
}

#[test]
fn runtime_revalidates_mutated_transport_before_runtime_and_private_io() {
    let (args, mut options) = command(&[], false);
    let limits = options.validate(&args).unwrap();
    options.stream.max_requests = 0;
    assert_eq!(execute(args, options, limits, NoRead, NoWrite).unwrap_err().exit, ErrorCode::Usage);
}

#[test]
fn every_cancellation_cause_survives_batch_and_map_failure_wrapping() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, LinkedExit, Shutdown] {
        let expected = cancelled(cause);
        for error in [
            HostedError::BatchSetup(BatchFault::cancelled(cause)),
            HostedError::Batch(crate::batch::BatchRunError {
                fault: BatchFault::cancelled(cause), summary: BatchSummary::default() }),
            HostedError::Int4(Int4HostError::SampledSentimentMap(SampledSentimentMapError::Cancelled(cause))),
            HostedError::Stopped { stop: RunStop { kind: cause, runtime_reason: None,
                runtime_checkpoint_error: None }, task_error: None },
        ] {
            let actual = execution_failure(error);
            assert_eq!(actual.exit, expected.exit);
            assert_eq!(actual.cancellation, Some(cause));
        }
    }
    for cause in [FailFast, LinkedExit] {
        let actual = execution_failure(HostedError::Stopped {
            stop: RunStop { kind: cause, runtime_reason: None, runtime_checkpoint_error: None },
            task_error: Some(Box::new(HostedError::BatchSetup(BatchCode::WorkLimit.into()))) });
        assert_eq!(actual.exit, ErrorCode::BudgetOrTimeout);
        assert_eq!(actual.code, "sampled_sentiment_corpus_budget");
        assert_eq!(actual.cancellation, Some(cause));
    }
}

#[test]
fn malformed_input_budget_no_result_and_delivery_failures_remain_distinct() {
    for code in [BatchCode::InvalidJson, BatchCode::InvalidEnvelope, BatchCode::DuplicateId] {
        assert_eq!(batch_failure(code.into()).exit, ErrorCode::InputDecodeOrParse);
    }
    for code in [BatchCode::WorkLimit, BatchCode::OutputLimit, BatchCode::LineLimit] {
        assert_eq!(batch_failure(code.into()).exit, ErrorCode::BudgetOrTimeout);
    }
    assert_eq!(batch_failure(BatchCode::Admission.into()).exit, ErrorCode::AdmissionOrResourceLimit);
    assert_eq!(batch_failure(BatchCode::Execution.into()).exit, ErrorCode::StructuredTaskNoResult);
    assert_eq!(batch_failure(BatchCode::OutputIo.into()).exit, ErrorCode::Generic);
    assert_eq!(batch_failure(BatchCode::Cancelled.into()).code, "sampled_sentiment_corpus_missing_cancellation");
}

#[test]
fn candidate_frames_omit_private_seed_and_bind_optional_public_execution() {
    let digest = "f".repeat(64);
    let mut provenance = Provenance {
        protocol: "fnlp-candidate-int4-sentiment-sampled-corpus-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        task: "sentiment-sampled-map", model_id: "Nanbeige4.2-3B",
        source_revision: crate::artifact::converter::PINNED_REVISION,
        source_root_sha256: &digest, model_content_sha256: &digest,
        candidate_format: INT4_SOURCE_FORMAT, quant_recipe: "fixture-recipe", group_width: 128,
        integer_kernel: "s_scalar_i32", snapshot: None, compute_schedule: None,
        prefill_schedule: None, prefill_reuse: None, input_encoding: None,
    };
    let plain = serde_json::to_value(&provenance).unwrap();
    for name in ["seed", "effective_seed", "document", "source_dir", "snapshot_path",
        "request_binding", "result_binding", "input_encoding", "prefill_reuse",
        "snapshot", "compute_schedule", "prefill_schedule"] {
        assert!(plain.get(name).is_none(), "unexpected provenance field {name}");
    }
    let prefill = Int4PrefillLimits { max_batch_rows: 64,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(64).unwrap() };
    let encoding = UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 1024, max_output_tokens: 1024,
        max_pair_checks: 65536, max_pair_bytes: 1_048_576 });
    provenance.snapshot = Some(SnapshotRecord { operation: "loaded", format: INT4_SNAPSHOT_VERSION, file_bytes: u64::MAX });
    provenance.prefill_schedule = Some(PrefillSchedule::new(prefill).unwrap());
    provenance.prefill_reuse = Some(SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION);
    provenance.input_encoding = Some(&encoding);
    let mut output = Vec::new();
    {
        let mut writer = CandidateWriter::new(&mut output, &provenance, 8192, 16_384).unwrap();
        // Framing fixture only: this is not a fabricated model-success claim.
        writer.write_all(b"{\"event\":\"fixture_only\"}\n").unwrap();
        writer.flush().unwrap();
    }
    let frame: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(frame["record"]["event"], "fixture_only");
    assert_eq!(frame["prefill_reuse"], SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION);
    assert_eq!(frame["prefill_schedule"]["admitted_workspace_bytes"], prefill.validate().unwrap());
    assert!(serde_json::to_vec(&provenance).unwrap().len() as u64 + 16 < FRAME_ALLOWANCE);
}
