//! Pure host-quote, cancellation and metadata contracts. No model or job runs.
use super::*;
use super::super::tests::{command, options, lifetime, NoRead, NoWrite};
use crate::{batch::BatchFault, hosted::{Int4SourceSemanticHostLimits, RunStop},
    jobs::{JobId, JobInputMemory, JobProgress,
        runner::source::grouped_int4::semantic::SEMANTIC_SOURCE_JOB_RECIPE_BYTES}};

#[test]
fn runtime_revalidates_before_process_initialization_or_private_io() {
    for kind in [Kind::Summarize, Kind::Answer] {
        let (mut command, args) = command(kind, "start", false, &[]);
        let admitted = command.args.validate(&args).unwrap(); command.args.storage.max_stream_mib = 0;
        let failure = execute(command.mode, args, command.args, admitted, NoRead, &mut NoWrite).unwrap_err();
        assert_eq!(failure.code, "semantic_source_job_storage_options"); assert_eq!(failure.exit, ErrorCode::Usage);
    }
}

#[test]
fn both_tasks_price_complete_stage_owners_load_paths_retention_and_teams_exactly_once() {
    for kind in [Kind::Summarize, Kind::Answer] { for route in 0..3 { for compact in [false, true] {
        for team in [false, true] { for retained in [false, true] {
            let mut flags = Vec::new();
            if route == 1 { flags.extend(["--save-snapshot", "private-new.snapshot"]); }
            if compact { flags.extend(["--input-encoding", "safe-bpe-v1"]); }
            if team { flags.extend(["--compute-threads", "4"]); }
            if retained { flags.extend(["--spool-input", "--max-retained-input-mib", "8"]); }
            let (command, args) = command(kind, "start", route == 2, &flags); let options = options(kind);
            let initial = command.args.validate(&args).unwrap(); let job = lifetime(&command.args, &args, &options);
            let limits = command.args.lifetime_limits(&args, initial, &options, job).unwrap();
            let remaining = Duration::from_secs(19);
            let quote = requirements(&args, &command.args, limits, &options, job, remaining).unwrap();
            let host = host_limits(&args, limits, remaining);
            let input = if retained { JobInputMemory::RetainedRecord } else { JobInputMemory::CompletePopulation };
            let ordinary = host.required_buffer_bytes_for_input(job, input).unwrap();
            let semantic = Int4SourceSemanticHostLimits { native: host.native, masks: limits.native.masks.per_mask,
                max_mask_node_visits: limits.native.masks.max_visits_per_item,
                preparation_reserve_bytes: host.preparation_reserve_bytes,
            }.requirements_with_encoding(args.group_width, limits.task, options.judge_budget, options.limits,
                judge_limits(&options), command.args.input_ceiling(limits, &options),
                &command.args.semantic.input_encoding.policy().unwrap()).unwrap();
            let recipe = 2 * SEMANTIC_SOURCE_JOB_RECIPE_BYTES as u64;
            assert_eq!(quote.preparation_bytes, semantic.preparation_bytes + recipe);
            assert_eq!(quote.input_bytes, ordinary - host.preparation_reserve_bytes + semantic.preparation_bytes + recipe);
            assert_eq!(quote.output_reservation_bytes, semantic.output_bytes);
            assert_eq!(quote.workspace_bytes, semantic.native_workspace_bytes); assert_eq!(quote.native, semantic.native);
            assert_eq!(quote.team_workspace_bytes, limits.common.compute.workspace_bytes);
            assert_eq!(quote.total_bytes().unwrap(), semantic.total_bytes().unwrap()
                + ordinary - host.preparation_reserve_bytes + recipe + quote.team_workspace_bytes);
            assert_eq!(host.native.context_tokens, args.context_tokens);
            assert_eq!(host.native.run.max_elapsed, remaining);
            assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
            let common = limits.common;
            let loading = if route == 2 { common.snapshot_need.scratch_bytes }
                else { common.source_preparation_bytes + common.load.scratch_payload_bytes };
            let saving = if route == 1 { common.snapshot_need.scratch_bytes } else { 0 };
            let admitted = admit_memory(&args, limits, quote).unwrap();
            assert_eq!(admitted.common.minimum_memory_bytes, common.preparation_bytes + common.load.weight_payload_bytes
                + ALLOCATOR_BYTES + loading.max(saving).max(quote.total_bytes().unwrap()) + CLEANUP_BYTES);
            assert_eq!(admitted.native.max_model_work, job.max_work.int4_model_work());
            assert_eq!(admitted.native.masks.max_visits_per_run, job.max_work.mask_node_visits);
            let mut exact = limits; exact.common.memory_bytes = admitted.common.minimum_memory_bytes;
            assert!(admit_memory(&args, exact, quote).is_ok()); exact.common.memory_bytes -= 1;
            assert_eq!(admit_memory(&args, exact, quote).unwrap_err().exit, ErrorCode::AdmissionOrResourceLimit);
            let process = compute::resource_config(admitted.common);
            assert_eq!(process.scoped_cpu_children_per_coordinator, if team { 3 } else { 0 });
            assert_eq!(process.thread_ceiling, if team { 5 } else { 2 });
            assert_eq!(process.memory_ceiling_bytes, common.memory_bytes);
        } }
    } } }
}

#[test]
fn retained_replay_keeps_complete_reviews_and_qa_memory_beyond_document_source_bytes() {
    for kind in [Kind::Summarize, Kind::Answer] {
        let flags = ["--input-encoding", "safe-bpe-v1", "--compute-threads", "4"];
        let (original, args) = command(kind, "resume", true, &flags); let options = options(kind);
        let initial = original.args.validate(&args).unwrap(); let job = lifetime(&original.args, &args, &options);
        let limits = original.args.lifetime_limits(&args, initial, &options, job).unwrap();
        let complete = requirements(&args, &original.args, limits, &options, job, Duration::from_secs(1)).unwrap();
        let mut flags = flags.to_vec(); flags.extend(["--use-retained-input", "--max-retained-input-mib", "8"]);
        let (retained, replay_args) = command(kind, "resume", true, &flags);
        let replay_limits = retained.args.lifetime_limits(&replay_args,
            retained.args.validate(&replay_args).unwrap(), &options, job).unwrap();
        let replay = requirements(&replay_args, &retained.args, replay_limits, &options, job, Duration::from_secs(1)).unwrap();
        assert_eq!(complete.input_bytes - replay.input_bytes, 2 * job.max_snapshot_bytes);
        assert_eq!(complete.total_bytes().unwrap() - replay.total_bytes().unwrap(), 2 * job.max_snapshot_bytes);
        assert_eq!(complete.preparation_bytes, replay.preparation_bytes); assert_eq!(complete.native, replay.native);
        assert_eq!(complete.workspace_bytes, replay.workspace_bytes);
        assert_eq!(complete.output_reservation_bytes, replay.output_reservation_bytes);
        assert_eq!(complete.team_workspace_bytes, replay.team_workspace_bytes);
        assert_eq!(limits.native.max_model_work, replay_limits.native.max_model_work);
        let mut huge_population = job; huge_population.max_snapshot_bytes *= 1024;
        assert_eq!(requirements(&replay_args, &retained.args, replay_limits, &options,
            huge_population, Duration::from_secs(1)).unwrap(), replay);
        for field in 0..2 {
            let mut wrong = replay;
            if field == 0 { wrong.native.kv_bytes += 1; } else { wrong.team_workspace_bytes += 1; }
            assert_eq!(admit_memory(&replay_args, replay_limits, wrong).unwrap_err().exit,
                ErrorCode::ArtifactIntegrityOrFormatOrVersion);
        }
    }
    let (mut summary, args) = command(Kind::Summarize, "start", false, &[]); let options = options(Kind::Summarize);
    let limits = summary.args.lifetime_limits(&args, summary.args.validate(&args).unwrap(), &options,
        lifetime(&summary.args, &args, &options)).unwrap();
    let job = lifetime(&summary.args, &args, &options);
    let base = requirements(&args, &summary.args, limits, &options, job, Duration::from_secs(1)).unwrap();
    summary.args.kind = Kind::Answer;
    let qa = requirements(&args, &summary.args, limits, &options, job, Duration::from_secs(1)).unwrap();
    let facts = summary.args.input_ceiling(limits, &options);
    assert_eq!(qa.input_bytes - base.input_bytes, 8 * facts.question_bytes as u64 + 2048 * facts.passage_count as u64);
    assert_eq!(qa.native, base.native); assert_eq!(qa.output_reservation_bytes, base.output_reservation_bytes);
}

#[test]
fn all_typed_cancellation_causes_survive_durable_native_and_physical_wrappers() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = job_failure(JobError::Cancelled(cause)).exit;
        for error in [
            HostedJobError::Job(JobRunError::Storage(JobError::Cancelled(cause))),
            HostedJobError::Job(JobRunError::Processor(BatchFault::cancelled(cause))),
            HostedJobError::Host(HostedError::Stopped { stop: RunStop { kind: cause,
                runtime_reason: None, runtime_checkpoint_error: None }, task_error: None }),
            HostedJobError::Host(HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause)))),
            HostedJobError::Host(HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause)))),
            HostedJobError::Host(HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause)))),
            HostedJobError::Host(HostedError::Int4(Int4HostError::SourceSemantic(Int4SourceSemanticError::Cancelled(cause)))),
            HostedJobError::Host(HostedError::Scope {
                source: asupersync::cx::ScopedCpuError::WorkerCapExceeded { cap: 1 },
                physical_error: Some(Box::new(HostedError::Scope {
                    source: asupersync::cx::ScopedCpuError::WorkerCapExceeded { cap: 1 },
                    physical_error: Some(Box::new(HostedError::Int4(
                        Int4HostError::SourceSemantic(Int4SourceSemanticError::Cancelled(cause))))),
                })),
            }),
        ] {
            let failure = execution_failure(error);
            assert_eq!(failure.cancellation, Some(cause)); assert_eq!(failure.exit, expected);
        }
        let stop = CancellationToken::default(); stop.cancel(cause);
        let failure = publish_report(&true, &mut NoWrite, &|| Err(cancelled(Deadline)), &stop).unwrap_err();
        assert_eq!(failure.cancellation, Some(cause)); assert_eq!(failure.exit, expected);
    }
    for cause in [FailFast, LinkedExit] {
        let failure = host_error(HostedError::Stopped { stop: RunStop { kind: cause,
            runtime_reason: None, runtime_checkpoint_error: None }, task_error: Some(Box::new(
                HostedError::Int4(Int4HostError::SourceSemantic(Int4SourceSemanticError::WorkBudget)))) });
        assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
        assert_eq!(failure.code, "semantic_source_job_budget"); assert_eq!(failure.cancellation, Some(cause));
    }
}

#[test]
fn publication_checks_staging_and_delivery_without_retrying_acknowledged_work() {
    use DecodeCancellationKind::{Deadline, Shutdown};
    let available = || Ok(Duration::from_secs(1)); let stop = CancellationToken::default();
    assert!(publish_report(&"x".repeat(job_cli::REPORT_BYTES), &mut NoWrite, &available, &stop).is_err());
    struct CancelWhileSerializing(CancellationToken);
    impl Serialize for CancelWhileSerializing {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.0.cancel(Shutdown); serializer.serialize_bool(true)
        }
    }
    let stopped = CancellationToken::default();
    assert_eq!(publish_report(&CancelWhileSerializing(stopped.clone()), &mut NoWrite,
        &available, &stopped).unwrap_err().cancellation, Some(Shutdown));
    let checks = std::cell::Cell::new(0);
    let expires_while_staging = || { checks.set(checks.get() + 1);
        if checks.get() == 1 { available() } else { Err(cancelled(Deadline)) } };
    assert_eq!(publish_report(&true, &mut NoWrite, &expires_while_staging, &stop).unwrap_err().cancellation, Some(Deadline));
    struct CancelOnFlush { bytes: Vec<u8>, stop: CancellationToken, flushes: usize }
    impl Write for CancelOnFlush {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.bytes.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { self.flushes += 1; self.stop.cancel(Shutdown); Ok(()) }
    }
    let stop = CancellationToken::default();
    let mut writer = CancelOnFlush { bytes: Vec::new(), stop: stop.clone(), flushes: 0 };
    assert_eq!(publish_report(&true, &mut writer, &available, &stop).unwrap_err().cancellation, Some(Shutdown));
    assert_eq!(writer.bytes, b"true\n"); assert_eq!(writer.flushes, 1);
    struct BrokenFlush { bytes: Vec<u8>, flushes: usize }
    impl Write for BrokenFlush {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.bytes.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { self.flushes += 1; Err(io::ErrorKind::BrokenPipe.into()) }
    }
    let mut broken = BrokenFlush { bytes: Vec::new(), flushes: 0 };
    let failure = publish_report(&true, &mut broken, &available, &CancellationToken::default()).unwrap_err();
    assert_eq!(failure.exit, ErrorCode::Generic); assert_eq!(failure.cancellation, None);
    assert_eq!(broken.bytes, b"true\n"); assert_eq!(broken.flushes, 1);
}

#[test]
fn both_metadata_envelopes_keep_correlated_scope_progress_and_no_private_inputs() {
    for kind in [Kind::Summarize, Kind::Answer] {
        let (command, args) = command(kind, "resume", true, &[]); let options = options(kind);
        let initial = command.args.validate(&args).unwrap(); let limits = lifetime(&command.args, &args, &options);
        let id = JobId([0x12; 16]);
        let progress = JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
            reserved_work: limits.max_work, spool_bytes: 1024, materialized: true };
        let report = progress_report(progress, id, limits, true,
            RunMode::Resume { discard_uncommitted: true }, task_name(kind)).unwrap();
        let content = "ab".repeat(32); let source = "cd".repeat(32);
        let response = Response { protocol: "fnlp-candidate-int4-semantic-source-job-v1", schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
            execution: INT4_SOURCE_SEMANTIC_EXECUTION, same_model_correlated: true,
            calibration: SOURCE_SEMANTIC_CALIBRATION, evidence_scope: SOURCE_SEMANTIC_SCOPE,
            model_id: "Nanbeige4.2-3B", source_revision: "fixture-public-revision", source_root_sha256: &source,
            model_content_sha256: &content, candidate_format: INT4_SOURCE_FORMAT, quant_recipe: "fixture-grouped-int4",
            group_width: args.group_width, integer_kernel: initial.kernel.kind().id(),
            snapshot: Some(SnapshotRecord::new("loaded", initial.snapshot_need)),
            compute_schedule: None, input_encoding: None, output: &report };
        let mut bytes = Vec::new();
        publish_report(&response, &mut bytes, &|| Ok(Duration::from_secs(1)), &CancellationToken::default()).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["same_model_correlated"], true); assert_eq!(value["calibration"], SOURCE_SEMANTIC_CALIBRATION);
        assert_eq!(value["evidence_scope"], SOURCE_SEMANTIC_SCOPE); assert_eq!(value["execution"], INT4_SOURCE_SEMANTIC_EXECUTION);
        assert_eq!(value["snapshot"]["format"], INT4_SNAPSHOT_VERSION); assert_eq!(value["output"]["task"], task_name(kind));
        assert_eq!(value["output"]["operation"], "resume"); assert_eq!(value["output"]["attempts"], 3);
        assert_eq!(value["output"]["committed"], 2);
        assert_eq!(value["output"]["reserved_work"], serde_json::to_value(limits.max_work).unwrap());
        assert!(bytes.ends_with(b"\n")); assert!(value.get("compute_schedule").is_none()); assert!(value.get("input_encoding").is_none());
        for private in ["text", "question", "passages", "source_options", "semantic_options", "claims", "key", "job_dir", "results"] {
            assert!(value.get(private).is_none()); assert!(value["output"].get(private).is_none());
        }
        let text = std::str::from_utf8(&bytes).unwrap();
        for marker in ["private-review", "private.snapshot", "private/key", "private/job"] { assert!(!text.contains(marker)); }
        let policy = UntrustedEncoding::SafeBpeV1(crate::tokenizer::untrusted::SafeBpeLimits {
            max_input_bytes: 8192, max_output_tokens: 4096, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 });
        let mut scheduled = response; scheduled.compute_schedule = compute::Schedule::for_threads(4);
        scheduled.input_encoding = Some(&policy);
        let mut scheduled = serde_json::to_value(&scheduled).unwrap();
        assert_eq!(scheduled.as_object_mut().unwrap().remove("compute_schedule"),
            compute::Schedule::for_threads(4).map(serde_json::to_value).transpose().unwrap());
        assert_eq!(scheduled.as_object_mut().unwrap().remove("input_encoding"), Some(serde_json::to_value(&policy).unwrap()));
        assert_eq!(scheduled, value);
    }
}
