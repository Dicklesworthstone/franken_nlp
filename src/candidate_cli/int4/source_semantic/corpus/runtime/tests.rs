//! Exact pre-IO quotations and typed failure boundaries; no model execution.
use super::*;
use super::super::tests::{command, options, NoRead, NoWrite};
use crate::{hosted::{Int4SourceSemanticHostLimits, RunStop},
    tasks::{source_planning::grouped_int4::Int4SourceError, judge::grouped_int4::Int4JudgeError}};

#[test]
fn both_tasks_price_source_snapshot_save_compact_and_team_peaks_exactly_once() {
    for kind in [Kind::Summarize, Kind::Answer] {
        for route in 0..3 {
            for flags in [&[][..], &["--compute-threads", "4"][..],
                &["--input-encoding", "safe-bpe-v1"][..],
                &["--input-encoding", "safe-bpe-v1", "--compute-threads", "4"][..]] {
                let mut flags = flags.to_vec();
                if route == 1 { flags.extend(["--save-snapshot", "private-new-snapshot"]); }
                let (args, batch) = command(kind, &flags, route == 2);
                let options = options(kind);
                let limits = batch.policy_limits(&args, batch.validate(&args).unwrap(), &options).unwrap();
                let remaining = Duration::from_secs(19);
                let quote = requirements(&args, &batch, limits, &options, remaining).unwrap();
                let corpus = host_limits(&args, limits, remaining);
                let encoding = batch.semantic.input_encoding.policy().unwrap();
                let one = Int4SourceSemanticHostLimits {
                    native: corpus.native, masks: limits.native.masks.per_mask,
                    max_mask_node_visits: limits.native.masks.max_visits_per_item,
                    preparation_reserve_bytes: corpus.preparation_reserve_bytes,
                }.requirements_with_encoding(args.group_width, limits.task, options.judge_budget,
                    options.limits, judge_limits(&options), input_ceiling(kind, &args, &options).unwrap(),
                    &encoding).unwrap();
                let transport = 8 * limits.transport.max_line_bytes as u64
                    + 4 * limits.transport.max_output_line_bytes as u64
                    + 128 * limits.transport.max_epoch_ids as u64
                    + limits.transport.max_epoch_id_bytes as u64 + limits.io_bytes;
                assert_eq!(quote.input_bytes, one.preparation_bytes + transport);
                assert_eq!(quote.preparation_bytes, one.preparation_bytes);
                assert_eq!(quote.workspace_bytes, one.native_workspace_bytes);
                assert_eq!(quote.output_reservation_bytes, one.output_bytes);
                assert_eq!(quote.native, one.native);
                assert_eq!(quote.team_workspace_bytes, limits.common.compute.workspace_bytes);
                assert_eq!(quote.total_bytes().unwrap(), one.total_bytes().unwrap()
                    + transport + limits.common.compute.workspace_bytes);
                let planning = batch.semantic.source.planning(&args);
                let task = batch.load_task(None, limits.task, planning, &options).unwrap();
                let config = Int4SourceSemanticCorpusConfig { task, planning,
                    judge_planning: judge_limits(&options), options, encoding, work: limits.native, limits: corpus };
                let hosted = if limits.common.compute.threads > 1 {
                    config.requirements_with_threads(args.group_width, limits.common.compute.threads)
                } else { config.requirements(args.group_width) }.unwrap();
                assert_eq!(hosted, quote, "CLI and resident host must price the same full QA/source envelope");
                let admitted = admit_memory(&args, limits, quote).unwrap();
                let loading = if route == 2 { limits.common.snapshot_need.scratch_bytes }
                    else { limits.common.source_preparation_bytes + limits.common.load.scratch_payload_bytes };
                let saving = if route == 1 { limits.common.snapshot_need.scratch_bytes } else { 0 };
                assert_eq!(admitted.common.minimum_memory_bytes, limits.common.preparation_bytes
                    + limits.common.load.weight_payload_bytes + ALLOCATOR_BYTES
                    + loading.max(saving).max(quote.total_bytes().unwrap()) + CLEANUP_BYTES);
                let mut short = limits; short.common.memory_bytes = admitted.common.minimum_memory_bytes - 1;
                assert_eq!(admit_memory(&args, short, quote).unwrap_err().exit, ErrorCode::AdmissionOrResourceLimit);
                let mut wrong = quote; wrong.team_workspace_bytes += 1;
                assert_eq!(admit_memory(&args, limits, wrong).unwrap_err().exit,
                    ErrorCode::ArtifactIntegrityOrFormatOrVersion);
            }
        }
    }
}

#[test]
fn qa_admission_prices_question_and_every_passage_beyond_summary_source_bytes() {
    let (args, batch) = command(Kind::Summarize, &[], false);
    let options = options(Kind::Summarize);
    let limits = batch.policy_limits(&args, batch.validate(&args).unwrap(), &options).unwrap();
    let summary = requirements(&args, &batch, limits, &options, Duration::from_secs(1)).unwrap();
    let mut answer = batch; answer.kind = Kind::Answer;
    let qa = requirements(&args, &answer, limits, &options, Duration::from_secs(1)).unwrap();
    let facts = input_ceiling(Kind::Answer, &args, &options).unwrap();
    assert_eq!(qa.input_bytes - summary.input_bytes, 8 * facts.question_bytes as u64
        + 2048 * facts.passage_count as u64);
    assert_eq!(qa.native, summary.native);
    assert_eq!(qa.output_reservation_bytes, summary.output_reservation_bytes);
    assert_eq!(qa.workspace_bytes, summary.workspace_bytes);
}

#[test]
fn more_requests_and_native_work_do_not_multiply_resident_source_and_review_memory() {
    for kind in [Kind::Summarize, Kind::Answer] {
        let (args, mut batch) = command(kind, &["--input-encoding", "safe-bpe-v1", "--compute-threads", "4"], false);
        let options = options(kind); let initial = batch.validate(&args).unwrap();
        let a = batch.policy_limits(&args, initial, &options).unwrap();
        let quote = requirements(&args, &batch, a, &options, Duration::from_secs(1)).unwrap();
        let work = options.max_model_work.checked_add(options.max_model_work).unwrap();
        batch.stream.max_requests += 10;
        batch.stream.max_total_forward_positions = Some(work.forward_positions);
        batch.stream.max_total_projected_logits = Some(work.projected_logits);
        batch.stream.max_total_attention_pairs = Some(work.attention_pairs);
        batch.stream.max_total_dot_products = Some(work.projections.dot_products);
        batch.stream.max_total_multiply_accumulates = Some(work.projections.multiply_accumulates);
        batch.stream.max_total_mask_node_visits = Some(2 * batch.semantic.source.mask_visits());
        let b = batch.policy_limits(&args, initial, &options).unwrap();
        let larger = requirements(&args, &batch, b, &options, Duration::from_secs(1)).unwrap();
        assert_eq!(quote, larger); assert_eq!(b.native.max_model_work, work);
        assert_eq!(b.native.masks.max_visits_per_item, a.native.masks.max_visits_per_item);
        assert_eq!(b.native.masks.max_visits_per_run, 2 * a.native.masks.max_visits_per_run);
        assert_eq!(b.task, a.task);
    }
}

#[test]
fn runtime_revalidation_and_expired_deadlines_precede_private_file_opens() {
    for kind in [Kind::Summarize, Kind::Answer] {
        let (args, mut batch) = command(kind, &[], false);
        let initial = batch.validate(&args).unwrap(); batch.stream.max_requests = 0;
        assert_eq!(execute(args, batch, initial, NoRead, NoWrite).unwrap_err().exit, ErrorCode::Usage);
    }
    let failure = read_local(std::path::Path::new("private-policy-that-must-not-open"), OPTIONS_BYTES,
        &|| Err(cancelled(DecodeCancellationKind::Deadline))).unwrap_err();
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::Deadline));
    assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
}

#[test]
fn all_original_cancellation_causes_survive_source_review_native_and_nested_wrapper_errors() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, LinkedExit, Shutdown] {
        let expected = cancelled(cause);
        for error in [
            HostedError::BatchSetup(BatchFault::cancelled(cause)),
            HostedError::Batch(crate::batch::BatchRunError {
                fault: BatchFault::cancelled(cause), summary: BatchSummary::default() }),
            HostedError::Stopped { stop: RunStop { kind: cause, runtime_reason: None,
                runtime_checkpoint_error: None }, task_error: None },
            HostedError::Int4(Int4HostError::SourceSemantic(Int4SourceSemanticError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::SourceTask(Int4SourceError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Judge(Int4JudgeError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))),
            HostedError::Scope {
                source: asupersync::cx::ScopedCpuError::WorkerCapExceeded { cap: 1 },
                physical_error: Some(Box::new(HostedError::Scope {
                    source: asupersync::cx::ScopedCpuError::WorkerCapExceeded { cap: 1 },
                    physical_error: Some(Box::new(HostedError::Int4(
                        Int4HostError::SourceSemantic(Int4SourceSemanticError::Cancelled(cause))))),
                })),
            },
        ] {
            let actual = host_failure(error);
            assert_eq!(actual.cancellation, Some(cause)); assert_eq!(actual.exit, expected.exit);
            let mut diagnostics = Vec::new(); report_failure(Kind::Answer, actual, &mut diagnostics);
            let report: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
            assert_eq!(report["cancellation"], serde_json::to_value(cause).unwrap());
            assert_eq!(report["exit_code"], serde_json::to_value(expected.exit).unwrap());
        }
    }
    for cause in [FailFast, LinkedExit] {
        let failure = host_failure(HostedError::Stopped {
            stop: RunStop { kind: cause, runtime_reason: None, runtime_checkpoint_error: None },
            task_error: Some(Box::new(HostedError::BatchSetup(BatchCode::WorkLimit.into()))),
        });
        assert_eq!(failure.cancellation, Some(cause)); assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
    }
}

#[test]
fn input_resource_budget_no_result_and_invariant_failures_remain_distinct() {
    for code in [BatchCode::InvalidJson, BatchCode::InvalidEnvelope, BatchCode::DuplicateId] {
        assert_eq!(batch_failure(code.into()).exit, ErrorCode::InputDecodeOrParse);
    }
    for code in [BatchCode::WorkLimit, BatchCode::OutputLimit] {
        assert_eq!(batch_failure(code.into()).exit, ErrorCode::BudgetOrTimeout);
    }
    assert_eq!(batch_failure(BatchCode::Admission.into()).exit, ErrorCode::AdmissionOrResourceLimit);
    assert_eq!(batch_failure(BatchCode::Execution.into()).exit, ErrorCode::StructuredTaskNoResult);
    assert_eq!(batch_failure(BatchCode::OutputIo.into()).exit, ErrorCode::Generic);
    assert_eq!(semantic_failure(Int4SourceSemanticError::Accounting).exit, ErrorCode::Generic);
    assert_eq!(semantic_failure(Int4SourceSemanticError::InputMismatch).exit, ErrorCode::Generic);
    assert_eq!(semantic_failure(Int4SourceSemanticError::WorkBudget).exit, ErrorCode::BudgetOrTimeout);
    assert_eq!(semantic_failure(Int4SourceSemanticError::ClaimBudget).exit, ErrorCode::BudgetOrTimeout);
    assert_eq!(semantic_failure(Int4SourceSemanticError::Contract).exit, ErrorCode::Usage);
    assert_eq!(batch_failure(BatchCode::Cancelled.into()).cancellation, None);
    assert_eq!(batch_failure(BatchCode::Cancelled.into()).code, "semantic_source_batch_missing_cancellation");
}

#[test]
fn every_framed_result_keeps_the_explicit_diagnostic_scope_without_private_options_or_evidence() {
    for kind in [Kind::Summarize, Kind::Answer] {
        let digest = "f".repeat(64);
        let provenance = Provenance {
            protocol: PROTOCOL, schema_version: 1, scope: "real-source-current-candidate-int4",
            evidence: "non_authoritative", task: task_name(kind), execution: INT4_SOURCE_SEMANTIC_EXECUTION,
            same_model_correlated: true, calibration: SOURCE_SEMANTIC_CALIBRATION,
            evidence_scope: SOURCE_SEMANTIC_SCOPE,
            model_id: "Nanbeige4.2-3B", source_revision: crate::artifact::converter::PINNED_REVISION,
            source_root_sha256: &digest, model_content_sha256: &digest,
            candidate_format: INT4_SOURCE_FORMAT, quant_recipe: "fixture-recipe", group_width: 128,
            integer_kernel: "s_scalar_i32", snapshot: None, compute_schedule: None, input_encoding: None,
        };
        let value = serde_json::to_value(&provenance).unwrap();
        for key in ["options", "semantic_options", "claims", "document", "question", "passages",
            "source_dir", "snapshot_path", "request_binding", "result_binding", "policy_digest",
            "input_encoding", "compute_schedule", "snapshot"] {
            assert!(value.get(key).is_none(), "unexpected public field {key}");
        }
        assert_eq!(value["same_model_correlated"], true);
        assert_eq!(value["calibration"], SOURCE_SEMANTIC_CALIBRATION);
        assert_eq!(value["evidence_scope"], SOURCE_SEMANTIC_SCOPE);
        let mut bytes = Vec::new();
        {
            let mut writer = CandidateWriter::new(&mut bytes, &provenance, 8192, 16384).unwrap();
            writer.write_all(b"{\"event\":\"fixture_only\"}\n").unwrap(); writer.flush().unwrap();
        }
        let frame: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(frame["record"]["event"], "fixture_only"); assert_eq!(frame["task"], task_name(kind));
        assert!(serde_json::to_vec(&provenance).unwrap().len() as u64 + 16 < FRAME_ALLOWANCE);
    }
}
