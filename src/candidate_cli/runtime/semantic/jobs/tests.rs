//! Real pinned factory/recipe and complete job-admission regressions, without weights.
use super::*;
use crate::{jobs::JobWork, native_engine::portable_int8::ProjectionWork,
    tasks::{extract::semantic::{ClaimRule, ClaimValueKind, SemanticLimits, SemanticVerificationSpec},
        ir::TaskBudget, judge::FaithfulnessPolicy}};

fn job_command(operation: &str, extra: &[&str]) -> JobCommand {
    let mut argv = vec!["extract-semantic-job", operation,
        "--model", "must-not-open.fnlpq", "--memory-mib", "16384",
        "--schema", "schema.json", "--semantic-options", "semantic.json",
        "--max-input-bytes", "512", "--max-new-tokens", "128", "--max-result-bytes", "16384",
        "--job-dir", "private/job", "--job-id", "0123456789abcdef0123456789abcdef",
        "--key-file", "private/key", "--limits", "limits.json", "--store-results"];
    argv.extend_from_slice(extra);
    let matches = command::definition().try_get_matches_from(argv).unwrap();
    JobCommand::from_matches(&matches).unwrap()
}
fn policy(limits: Limits) -> Int8SemanticOptions {
    Int8SemanticOptions {
        specification: SemanticVerificationSpec {
            schema_version: 1, experimental_opt_in: true, revision: "semantic-job-runtime-test-v1".into(),
            claims: vec![ClaimRule { id: "amount-v1".into(), path: vec![], value_kind: ClaimValueKind::Number,
                prefix: "The amount is ".into(), suffix: ".".into() }],
            judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0,
                evidence_window_bytes: 512, max_evidence_windows: 1, max_evidence_spans: 1 },
            judge_budget: TaskBudget { max_input_tokens: 2016, max_output_tokens: 32,
                max_output_bytes: 65_536, max_grammar_states: 128, max_kv_bytes: limits.kv_bytes },
        },
        limits: SemanticLimits { max_fields: 2, max_claim_bytes: 256, max_total_claim_bytes: 1024,
            max_template_bytes: 2048, max_walk_nodes: 128, max_input_json_bytes: 16_384,
            max_total_prompt_tokens: 8192, max_total_projected_logits: 16_777_216,
            max_output_bytes: 1_048_576 },
        max_model_work: Int8Work { forward_positions: 4096, projected_logits: 64_000_000,
            attention_pairs: 4096 * 2048 * 44 * 48,
            projections: ProjectionWork { dot_products: 100_000_000_000,
                multiply_accumulates: 1_000_000_000_000_000 } },
    }
}
fn lifetime(command: &JobCommand, options: &Int8SemanticOptions) -> JobLimits {
    JobLimits { max_items: 3, max_id_bytes: 128,
        max_input_bytes_per_item: command.semantic.host.max_input_bytes,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: options.limits.max_output_bytes as usize,
        max_spool_bytes: 16 * MIB, max_materialized_bytes: 16 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 7,
        max_work: JobWork { model: options.max_model_work,
            mask_node_visits: command.semantic.host.max_mask_node_visits } }
}
fn host(args: &CandidateArgs, envelope: JobEnvelope) -> JobHostLimits {
    JobHostLimits {
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: 64 * MIB,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: 65_536 } },
        transport: envelope.transport, preparation_reserve_bytes: envelope.preparation_bytes,
        io_reserve_bytes: envelope.io_bytes, journal_reserve_bytes: envelope.journal_bytes,
        serialization_reserve_bytes: envelope.serialization_bytes,
    }
}
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".into(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".into(),
        recipe_id: "semantic-job-metadata-only-fixture".into(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}

#[test]
fn pinned_durable_recipe_is_invariant_across_start_resume_and_all_four_schedules() {
    for compact in [false, true] {
        let mut recipe = None;
        for operation in ["start", "resume"] {
            for (threads, rows) in [("1", None), ("2", None), ("1", Some("4")), ("2", Some("4"))] {
                let mut extra = vec!["--compute-threads", threads];
                if let Some(rows) = rows { extra.extend(["--prefill-rows", rows]); }
                if compact { extra.extend(["--input-encoding", "safe-bpe-v1",
                    "--max-tokenizer-input-bytes", "65536", "--max-tokenizer-output-tokens", "8192"]); }
                let command = job_command(operation, &extra); let (args, limits) = command.validate().unwrap();
                let options = policy(limits); let job = lifetime(&command, &options);
                let envelope = command.lifetime_limits(&args, limits, &options, job).unwrap();
                let schema = " {\"type\":\"number\"} \n";
                let request = semantic_cli::fixed_request(&command.semantic, schema.into(),
                    command.semantic.host.task_budget(limits), &options).unwrap();
                let mut changed_budget = request.clone();
                changed_budget.budget.max_output_tokens += 1;
                assert_eq!(prepare(&facts(), &command.semantic, limits, changed_budget, options.clone(),
                    envelope.native, envelope.encoding_work_limits).err().unwrap().code,
                    "semantic_job_extraction_budget");
                let factory = prepare(&facts(), &command.semantic, limits, request, options.clone(),
                    envelope.native, envelope.encoding_work_limits).unwrap();
                let host = host(&args, envelope); let prefill = command.semantic.prefill().unwrap();
                let input = command.storage.input_retention.input_memory().unwrap();
                let expected = requirements(&command, limits, &options, envelope, host, job, input, prefill).unwrap();
                let actual = check_factory(&command, limits, &options, Sha256Digest::of_bytes(schema.as_bytes()),
                    envelope, host, job, input, prefill, &factory).unwrap();
                assert_eq!(actual, expected);
                if !compact && operation == "start" && threads == "1" && rows.is_none() {
                    // This policy mutation changes no memory geometry. The
                    // actual factory settings must still match the declaration.
                    let mut altered = factory.config().clone();
                    altered.judge_planning.per_head.max_projected_logits -= 1;
                    let same_quote = int8_semantic_job_requirements(altered.extraction.budget, &altered.options,
                        altered.judge_planning, &altered.encoding, altered.encoding_work_limits,
                        altered.work, host, job, input, prefill, None).unwrap();
                    assert_eq!(same_quote, expected);
                    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
                    let eos = controls.entries().iter().find(|entry| entry.special
                        && entry.surface == crate::template::IM_END).unwrap().id;
                    let mut identity = candidate_identity(&facts()).unwrap(); identity.task_spec = "extract-v1".into();
                    let altered = Int8SemanticExtractionJobPlanner::pinned(controls, eos, identity, altered).unwrap();
                    assert_eq!(check_factory(&command, limits, &options, Sha256Digest::of_bytes(schema.as_bytes()),
                        envelope, host, job, input, prefill, &altered).err().unwrap().code,
                        "semantic_job_factory_mismatch");
                }
                assert_eq!(factory.config().work.max_model_work, job.max_work.model);
                assert_eq!(factory.execution_identity().logical_model_digest, factory.judge_identity().logical_model_digest);
                assert_eq!(factory.encoding_work_limits(), envelope.encoding_work_limits);
                assert_eq!(factory.encoding_workspace_bytes().unwrap() > 0, compact);
                let current = canonjson::canonical_bytes(factory.job_recipe()).unwrap();
                if let Some(previous) = &recipe { assert_eq!(&current, previous); }
                recipe = Some(current);
                let mut changed = options.clone(); changed.specification.claims[0].prefix.push_str("changed ");
                assert_eq!(check_factory(&command, limits, &changed, Sha256Digest::of_bytes(schema.as_bytes()),
                    envelope, host, job, input, prefill, &factory).err().unwrap().code,
                    "semantic_job_factory_mismatch");
                assert_eq!(check_factory(&command, limits, &options, Sha256Digest::of_bytes(b"{\"type\":\"number\"}"),
                    envelope, host, job, input, prefill, &factory).err().unwrap().code,
                    "semantic_job_factory_mismatch");
                if compact {
                    let mut changed = envelope;
                    changed.encoding_work_limits.as_mut().unwrap().max_pair_checks += 1;
                    assert_eq!(check_factory(&command, limits, &options, Sha256Digest::of_bytes(schema.as_bytes()),
                        changed, host, job, input, prefill, &factory).err().unwrap().code,
                        "semantic_job_factory_mismatch");
                }
            }
        }
    }
}

#[test]
fn complete_and_retained_input_quotes_keep_recipe_reviews_output_and_caller_owner_additive() {
    for extra in [vec![], vec!["--input-encoding", "safe-bpe-v1", "--compute-threads", "4",
        "--prefill-rows", "8", "--max-tokenizer-input-bytes", "65536",
        "--max-tokenizer-output-tokens", "8192"]] {
        let command = job_command("start", &extra); let (args, mut limits) = command.validate().unwrap();
        let options = policy(limits); let job = lifetime(&command, &options);
        let envelope = command.lifetime_limits(&args, limits, &options, job).unwrap();
        let host = host(&args, envelope); let prefill = command.semantic.prefill().unwrap();
        let complete = requirements(&command, limits, &options, envelope, host, job,
            JobInputMemory::CompletePopulation, prefill).unwrap();
        let retained = requirements(&command, limits, &options, envelope, host, job,
            JobInputMemory::RetainedRecord, prefill).unwrap();
        assert_eq!(complete.preparation_bytes, retained.preparation_bytes);
        assert_eq!(complete.output_bytes, retained.output_bytes);
        assert!(complete.input_bytes > retained.input_bytes);
        assert!(complete.preparation_bytes > limits.preparation_bytes
            + 2 * crate::jobs::runner::extract::semantic::SEMANTIC_EXTRACTION_JOB_RECIPE_BYTES as u64);
        for quote in [complete, retained] {
            let required = minimum_memory(&args, limits, quote, prefill).unwrap();
            assert!(required > limits.weight_bytes + limits.preparation_bytes + quote.preparation_bytes);
            limits.memory_bytes = required;
            admit_memory(&args, limits, quote, prefill).unwrap();
            limits.memory_bytes -= 1;
            assert_eq!(admit_memory(&args, limits, quote, prefill), Err(CandidateError::Memory));
            let mut wrong = quote; wrong.native.kv_bytes -= 1;
            assert_eq!(minimum_memory(&args, limits, wrong, prefill), Err(CandidateError::Arguments));
            let mut wrong = quote; wrong.team_workspace_bytes += 1;
            assert_eq!(minimum_memory(&args, limits, wrong, prefill), Err(CandidateError::Arguments));
        }
        let mut changed = job; changed.max_work.model.attention_pairs += 1;
        assert!(requirements(&command, limits, &options, envelope, host, changed,
            JobInputMemory::CompletePopulation, prefill).is_err());
    }
}

#[test]
fn durable_cancellation_cause_survives_storage_native_supervision_and_report_delivery() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        assert_eq!(job_failure(JobError::Cancelled(cause)).cancellation, Some(cause));
        assert_eq!(host_failure(HostedJobError::Job(JobRunError::Storage(JobError::Cancelled(cause))))
            .cancellation, Some(cause));
        let nested = HostedError::Scope {
            source: asupersync::cx::ScopedCpuError::WorkerCapExceeded { cap: 1 },
            physical_error: Some(Box::new(HostedError::SemanticExtraction(Int8SemanticError::Cancelled(cause)))),
        };
        assert_eq!(host_failure(HostedJobError::Host(nested)).cancellation, Some(cause));
        let cancellation = CancellationToken::default(); cancellation.cancel(cause);
        let mut bytes = Vec::new();
        let failure = publish_report(&serde_json::json!({"status":"complete"}), &mut bytes,
            &|| Ok(Duration::from_secs(1)), &cancellation).err().unwrap();
        assert_eq!(failure.cancellation, Some(cause));
        assert!(bytes.is_empty());
    }
    assert_eq!(host_failure(HostedJobError::Job(JobRunError::WorkContract)).code, "semantic_job_work_contract");
    assert_eq!(host_failure(HostedJobError::Job(JobRunError::Stopped)).cancellation, None);
}

#[test]
fn metadata_only_completion_omits_byte_defaults_and_reports_declared_compact_invocation_limits() {
    let response = serde_json::json!({"output":{"status":"complete","task":"extract-semantic"}});
    let mut metadata = Metadata { response: &response, protocol: command::PROTOCOL,
        execution: INT8_SEMANTIC_EXECUTION, same_model_correlated: true,
        calibration: "experimental_uncalibrated_diagnostic_only",
        compute_threads: None, prefill_rows: None, input_encoding: None,
        invocation_encoding_work_limits: None };
    let value = serde_json::to_value(&metadata).unwrap();
    for key in ["input_encoding", "invocation_encoding_work_limits", "compute_threads", "prefill_rows",
        "encoding_work", "schema", "semantic_options", "extraction", "semantic"] {
        assert!(value.get(key).is_none(), "{key}");
    }
    assert_eq!(value["same_model_correlated"], true);
    let command = job_command("resume", &["--input-encoding", "safe-bpe-v1"]);
    let encoding = command.semantic.encoding_policy().unwrap();
    metadata.input_encoding = Some(&encoding);
    metadata.invocation_encoding_work_limits = command.encoding_work_limits().unwrap();
    metadata.compute_threads = Some(2); metadata.prefill_rows = Some(4);
    let value = serde_json::to_value(&metadata).unwrap();
    assert_eq!(value["invocation_encoding_work_limits"]["max_input_bytes"], 64 * MIB);
    assert_eq!(value["compute_threads"], 2); assert_eq!(value["prefill_rows"], 4);
    let mut output = Vec::new();
    publish_report(&metadata, &mut output, &|| Ok(Duration::from_secs(1)),
        &CancellationToken::default()).unwrap();
    assert_eq!(output.iter().filter(|&&byte| byte == b'\n').count(), 1);
    struct FlushFailure(Vec<u8>);
    impl Write for FlushFailure {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.0.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("private flush detail")) }
    }
    let mut failed = FlushFailure(Vec::new());
    assert!(publish_report(&metadata, &mut failed, &|| Ok(Duration::from_secs(1)),
        &CancellationToken::default()).is_err());
    assert_eq!(failed.0, output);
}
