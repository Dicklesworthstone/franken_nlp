//! Pinned configuration and metadata checks only, not model execution.
use super::*;
use crate::candidate_cli::redact::retention::tests::{command, lifetime};
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".into(), revision: "f56ec5a9650268aa098496734743c25ea778bd2d".into(),
        recipe_id: "retained-redaction-planning-fixture".into(), source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
#[test]
fn live_and_retained_modes_build_the_same_pinned_detector_and_rule_scope() {
    let (p, _) = planner().unwrap();
    for chunked in [false, true] {
        let extras: &[&str] = if chunked { &["--chunked"] } else { &[] };
        let retained = command(extras); let mut args = vec!["--ndjson"]; args.extend_from_slice(extras);
        let live = crate::candidate_cli::redact::tests::command(&args);
        let (_, a) = live.validate().unwrap(); let (_, b) = retained.validate().unwrap();
        let identity = source_identity(&facts(), &p, BuiltInTask::Ner).unwrap();
        let left = prepare(&live, a, live.corpus.envelope(&live, a).unwrap(), &p, identity.clone(), NerOptions::default(), live.request()).unwrap();
        let right = prepare(&retained, b, retained.corpus.envelope(&retained, b).unwrap(), &p, identity, NerOptions::default(), retained.request()).unwrap();
        match (left, right) {
            (PreparedCorpus::Short(x), PreparedCorpus::Short(y)) => {
                assert_eq!(x.ner_identity, y.ner_identity); assert_eq!(x.max_model_work, y.max_model_work);
                assert_eq!(x.detector.per_pass, y.detector.per_pass); assert_eq!(x.detector.ner, y.detector.ner);
                assert_eq!(canonjson::canonical_bytes(&x.request).unwrap(), canonjson::canonical_bytes(&y.request).unwrap());
            }
            (PreparedCorpus::Document(x), PreparedCorpus::Document(y)) => {
                assert_eq!(x.ner_identity, y.ner_identity); assert_eq!(x.item_model_work(), y.item_model_work());
                assert_eq!(x.detector.mapping.chunks, y.detector.mapping.chunks);
                assert_eq!(x.detector.mapping.reduction, y.detector.mapping.reduction);
                assert_eq!(x.detector.mapping.max_mask_visits, y.detector.mapping.max_mask_visits);
                assert_eq!(canonjson::canonical_bytes(&x.request).unwrap(), canonjson::canonical_bytes(&y.request).unwrap());
            }
            _ => panic!("retention changed the native document mode"),
        }
    }
}
fn progress(limits: JobLimits, id: JobId) -> JobProgress {
    JobProgress { job_id: id, items: 2, committed: 2, attempts: 3, reserved_work: limits.max_work,
        spool_bytes: 8192, materialized: true }
}
#[test]
fn metadata_contains_lifetime_work_and_sensitivity_but_no_redacted_documents_or_keys() {
    let c = command(&[]); let limits = lifetime(&c); let id = c.retention.job_id.unwrap();
    let output = report(progress(limits, id), id, limits, true, true, true).unwrap();
    let value = serde_json::to_value(output).unwrap();
    assert_eq!(value["operation"], "resume"); assert_eq!(value["attempts"], 3);
    assert_eq!(value["redaction_mode"], "chunked"); assert_eq!(value["retained_output_is_sensitive"], true);
    assert_eq!(value["reserved_work"]["mask_node_visits"], limits.max_work.mask_node_visits);
    let text = value.to_string();
    for private in ["redacted_text", "entities", "key_id", "key_commitment", "scope_commitment", "namespace", "source_text"] {
        assert!(!text.contains(private));
    }
}
#[test]
fn incomplete_or_divergent_job_progress_is_not_a_success_report() {
    let c = command(&[]); let limits = lifetime(&c); let id = c.retention.job_id.unwrap();
    for axis in 0..7 {
        let mut p = progress(limits, id);
        match axis { 0 => p.job_id = JobId([4; 16]), 1 => p.items = 0, 2 => p.committed -= 1,
            3 => p.attempts = 1, 4 => p.attempts = limits.max_attempts + 1,
            5 => p.reserved_work.mask_node_visits += 1, _ => p.materialized = false }
        assert!(report(p, id, limits, false, true, false).is_err());
    }
}

fn compact_command(extra: &[&str]) -> RedactCommand {
    let mut args = vec!["--chunked", "--input-encoding", "safe-bpe-v1",
        "--max-ner-chunk-bytes", "128", "--max-new-tokens", "64"];
    args.extend_from_slice(extra);
    let mut command = command(&args);
    command.host.memory_mib = 16_384;
    command
}
fn admitted_host(command: &RedactCommand, common: &CandidateArgs, limits: Limits, envelope: CorpusEnvelope) -> JobHostLimits {
    JobHostLimits { native: crate::hosted::NativeLimits { context_tokens: common.context_tokens,
        allocator_reserve_bytes: 64 * MIB, run: crate::hosted::RunLimits {
            max_elapsed: std::time::Duration::from_secs(common.timeout_seconds),
            max_checkpoints: common.max_checkpoints, cleanup_reserve_bytes: 65536 } },
        transport: PopulationReadLimits { max_stream_bytes: envelope.transport.max_input_bytes,
            max_lines: command.retention.max_lines() },
        preparation_reserve_bytes: limits.preparation_bytes, io_reserve_bytes: 2 * IO_BUFFER_BYTES as u64,
        journal_reserve_bytes: command.retention.journal_bytes().unwrap(),
        serialization_reserve_bytes: command.retention.serialization_bytes().unwrap() }
}

#[test]
fn compact_live_start_and_both_resume_sources_keep_one_exact_policy_and_complete_document_allowance() {
    let mut live = crate::candidate_cli::redact::tests::command(&["--ndjson", "--chunked",
        "--input-encoding", "safe-bpe-v1", "--max-ner-chunk-bytes", "128", "--max-new-tokens", "64"]);
    live.host.memory_mib = 16_384;
    let (_, live_limits) = live.validate().unwrap();
    let (factory, _) = planner().unwrap();
    let live_factory = configure_planner(&live, live_limits, factory).unwrap();
    let live_identity = source_identity(&facts(), &live_factory, BuiltInTask::Ner).unwrap();
    let live_envelope = live.corpus.envelope(&live, live_limits).unwrap();
    let PreparedCorpus::Document(live_batch) = prepare(&live, live_limits, live_envelope, &live_factory,
        live_identity.clone(), NerOptions::default(), live.request()).unwrap() else { panic!("document"); };
    for extra in [vec![], vec!["--resume"],
        vec!["--resume", "--use-retained-input", "--max-retained-input-mib", "4"]] {
        let retained = compact_command(&extra); let (_, limits) = retained.validate().unwrap();
        let (factory, _) = planner().unwrap();
        let factory = configure_planner(&retained, limits, factory).unwrap();
        let identity = source_identity(&facts(), &factory, BuiltInTask::Ner).unwrap();
        assert_eq!(identity, live_identity);
        let envelope = retained.corpus.envelope(&retained, limits).unwrap();
        let PreparedCorpus::Document(batch) = prepare(&retained, limits, envelope, &factory,
            identity, NerOptions::default(), retained.request()).unwrap() else { panic!("document"); };
        assert_eq!(batch.item_model_work(), live_batch.item_model_work());
        assert_eq!(batch.max_model_work, live_batch.max_model_work);
        assert_eq!(batch.max_mask_visits, live_batch.max_mask_visits);
        assert_eq!(batch.detector.mapping.chunks, live_batch.detector.mapping.chunks);
        assert_eq!(canonjson::canonical_bytes(&batch.request).unwrap(), canonjson::canonical_bytes(&live_batch.request).unwrap());
        assert_eq!(factory.document_encoding(), live_factory.document_encoding());
        assert_eq!(factory.map_document_encoding_limits(), live_factory.map_document_encoding_limits());
        assert_eq!(factory.map_document_encoding_work().unwrap(), Some(SafeBpeWork::default()));
        let lifetime = lifetime(&retained);
        retained.retention.check_limits(&retained, envelope, lifetime).unwrap();
        let mut oversized = lifetime;
        oversized.max_input_bytes_per_item = retained.host.max_input_bytes + 1;
        assert!(retained.retention.check_limits(&retained, envelope, oversized).is_err(),
            "verification headroom does not increase original retained-envelope authority");
    }
}

#[test]
fn retained_peak_prices_full_output_rows_and_changes_only_the_actual_input_owner() {
    let command = compact_command(&[]);
    let (common, limits) = command.validate().unwrap();
    let envelope = command.corpus.envelope(&command, limits).unwrap();
    let lifetime = lifetime(&command);
    let host = admitted_host(&command, &common, limits, envelope);
    let original = retained_peak_bytes(&command, &common, limits, host, lifetime, JobInputMemory::CompletePopulation).unwrap();
    let retained = retained_peak_bytes(&command, &common, limits, host, lifetime, JobInputMemory::RetainedRecord).unwrap();
    assert_eq!(original - retained, 2 * lifetime.max_snapshot_bytes,
        "retained replay removes only the complete original population, not its index or single record");

    let detector = command.long.config(&command.host, limits, NerOptions::default()).unwrap();
    let rows = detector.mapping.max_model_work.projected_logits / crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE as u64;
    assert!(rows > u64::from(detector.per_chunk.max_output_tokens));
    let native = Int8MemoryRequirement::for_context(common.context_tokens).unwrap();
    let kv_bytes = native.kv_bytes;
    let shared_without_rows = crate::candidate_cli::compute::request_peak_bytes(limits, native,
        command.host.max_result_bytes as u64, 0, 0, 0).unwrap();
    let buffers = host.required_buffer_bytes_for_input(lifetime, JobInputMemory::RetainedRecord).unwrap();
    let count = detector.mapping.chunks.max_chunks as u64;
    let temporaries = 4 * (detector.per_chunk.max_output_bytes * count + detector.mapping.reduction.max_live_value_bytes as u64)
        + 8 * u64::from(detector.per_chunk.max_output_tokens) * count + command.edit_reserve_mib * MIB;
    assert_eq!(retained - shared_without_rows - buffers - temporaries - 4096 - 4 * (MIB - 1024), 8 * rows,
        "the admitted native output rows survive alongside map-token staging");
    assert!(retained > limits.weight_bytes + 2 * limits.preparation_bytes + kv_bytes);
    let mut larger = lifetime; larger.max_snapshot_bytes *= 2;
    assert_eq!(retained_peak_bytes(&command, &common, limits, host, larger, JobInputMemory::RetainedRecord).unwrap(), retained);
    let mut more_preparation = host; more_preparation.preparation_reserve_bytes += 1;
    assert_eq!(retained_peak_bytes(&command, &common, limits, more_preparation, lifetime, JobInputMemory::RetainedRecord).unwrap(),
        retained + 1);
    let mut invalid = lifetime; invalid.max_items = 0;
    assert_eq!(retained_peak_bytes(&command, &common, limits, host, invalid, JobInputMemory::RetainedRecord),
        Err(CandidateError::Memory));
}

#[test]
fn retained_runtime_rejects_stale_authority_before_the_limits_file_or_either_secret() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> { panic!("private read"); } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> { panic!("output write"); }
        fn flush(&mut self) -> std::io::Result<()> { panic!("output flush"); }
    }
    for axis in 0..4 {
        let command = compact_command(&[]);
        let (common, mut limits) = command.validate().unwrap();
        let mut envelope = command.corpus.envelope(&command, limits).unwrap();
        match axis { 0 => limits.memory_bytes -= 1, 1 => envelope.transport.max_requests -= 1,
            2 => envelope.max_model_work.attention_pairs -= 1, _ => envelope.max_mask_visits -= 1 }
        assert_eq!(execute(command, common, limits, envelope, NoIo, NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn retained_metadata_adds_exact_encoding_policy_without_document_work_or_private_payloads() {
    let command = compact_command(&[]);
    let (_, limits) = command.validate().unwrap();
    let lifetime = lifetime(&command); let id = command.retention.job_id.unwrap();
    let report = report(progress(lifetime, id), id, lifetime, true, true, true).unwrap();
    let facts = facts();
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: &report };
    let byte = PopulationResponse { response: &response, input_encoding: None };
    assert_eq!(serde_json::to_vec(&byte).unwrap(), serde_json::to_vec(&response).unwrap());
    let (factory, _) = planner().unwrap();
    let factory = configure_planner(&command, limits, factory).unwrap();
    let compact = PopulationResponse { response: &response, input_encoding: population_encoding(&factory).unwrap() };
    let value = serde_json::to_value(&compact).unwrap();
    assert_eq!(value["input_encoding"]["policy"], serde_json::to_value(factory.document_encoding()).unwrap());
    assert_eq!(value["input_encoding"]["complete_document_limits"],
        serde_json::to_value(factory.map_document_encoding_limits().unwrap()).unwrap());
    assert!(value["input_encoding"].get("work").is_none());
    assert_eq!(value["output"]["reserved_work"], serde_json::to_value(lifetime.max_work).unwrap());
    let text = value.to_string();
    for private in ["redacted_text", "entities", "key_commitment", "scope_commitment", "namespace", "source_text", "prompt_digest"] {
        assert!(!text.contains(private));
    }
}

#[test]
fn retained_processor_and_storage_cancellation_keep_the_original_timeout_cause() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = match cause { Deadline | Timeout => CandidateError::Timeout, _ => CandidateError::Batch };
        assert_eq!(job_execution_error(HostedJobError::Job(JobRunError::Storage(JobError::Cancelled(cause)))), expected);
        assert_eq!(job_execution_error(HostedJobError::Host(HostedError::Native(
            crate::native_engine::strict_int8::StrictInt8Error::Cancelled(cause)))), expected);
        assert_eq!(job_execution_error(HostedJobError::Job(JobRunError::Processor(crate::batch::BatchFault::cancelled(cause)))), expected);
    }
}

fn scheduled_command(chunked: bool, compact: bool, extra: &[&str]) -> RedactCommand {
    let mut flags = Vec::new();
    if chunked { flags.push("--chunked"); }
    if compact { flags.extend(["--input-encoding", "safe-bpe-v1"]); }
    flags.extend_from_slice(extra);
    let mut command = command(&flags); command.host.memory_mib = 16_384;
    command
}

#[test]
fn retained_schedules_add_only_one_worker_and_prompt_workspace_for_either_input_owner() {
    use crate::native_engine::strict_int8::{prefill::Int8PrefillLimits,
        projection::int8_team_workspace_bytes_with_prefill};
    for (chunked, compact) in [(false, false), (true, false), (true, true)] {
        let plain = scheduled_command(chunked, compact, &[]);
        let threaded = scheduled_command(chunked, compact, &["--compute-threads", "4", "--prefill-rows", "7"]);
        let (plain_args, a) = plain.validate().unwrap();
        let (threaded_args, b) = threaded.validate().unwrap();
        let plain_envelope = plain.corpus.envelope(&plain, a).unwrap();
        let threaded_envelope = threaded.corpus.envelope(&threaded, b).unwrap();
        let plain_host = admitted_host(&plain, &plain_args, a, plain_envelope);
        let threaded_host = admitted_host(&threaded, &threaded_args, b, threaded_envelope);
        let job_limits = lifetime(&plain);
        let extra = int8_team_workspace_bytes_with_prefill(4, 7).unwrap()
            + Int8PrefillLimits::required_extra_scratch_bytes(7).unwrap();
        for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
            let serial = retained_peak_bytes(&plain, &plain_args, a, plain_host, job_limits, input).unwrap();
            let scheduled = retained_peak_bytes(&threaded, &threaded_args, b, threaded_host, job_limits, input).unwrap();
            assert_eq!(scheduled, serial + extra);
            assert!(scheduled > b.weight_bytes + 2 * b.preparation_bytes);
        }
        let original = retained_peak_bytes(&threaded, &threaded_args, b, threaded_host,
            job_limits, JobInputMemory::CompletePopulation).unwrap();
        let replay = retained_peak_bytes(&threaded, &threaded_args, b, threaded_host,
            job_limits, JobInputMemory::RetainedRecord).unwrap();
        assert_eq!(original - replay, 2 * job_limits.max_snapshot_bytes);
        let floor = threaded.corpus.compact_base_peak_bytes(&threaded, b, threaded.edit_reserve_mib * MIB).unwrap();
        let input_bytes = JobInputMemory::RetainedRecord.buffer_bytes(job_limits).unwrap();
        assert_eq!(replay, floor + input_bytes,
            "the final pre-secret quote adds the actual retained-record owner, not another team");
    }
}

#[test]
fn retained_schedule_readmission_precedes_limits_file_both_keys_and_original_or_saved_input() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> { panic!("private read"); } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> { panic!("output"); }
        fn flush(&mut self) -> std::io::Result<()> { panic!("flush"); }
    }
    for resume in [false, true] { for axis in 0..5 {
        let mut flags = vec!["--compute-threads", "4", "--prefill-rows", "7"];
        if resume { flags.extend(["--resume", "--use-retained-input", "--max-retained-input-mib", "4"]); }
        let mut command = scheduled_command(true, true, &flags);
        let (common, mut limits) = command.validate().unwrap();
        let mut envelope = command.corpus.envelope(&command, limits).unwrap();
        match axis {
            0 => limits.compute.workspace_bytes -= 1,
            1 => limits.compute.threads = 3,
            2 => command.prefill = Default::default(),
            3 => command.compute = Default::default(),
            _ => envelope.max_model_work.projections.multiply_accumulates -= 1,
        }
        assert_eq!(execute(command, common, limits, envelope, NoIo, NoIo), Err(CandidateError::Arguments));
    } }
}
