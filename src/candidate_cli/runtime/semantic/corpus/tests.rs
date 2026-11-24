//! Model-free host/CLI binding checks. These do not qualify native parity.
use super::*;
use crate::tasks::{extract::semantic::{ClaimRule, ClaimValueKind, SemanticLimits,
    SemanticVerificationSpec}, ir::TaskBudget, judge::FaithfulnessPolicy};

fn batch_command(extra: &[&str]) -> BatchCommand {
    let mut argv = vec!["extract-semantic-batch", "--model", "must-not-open.fnlpq",
        "--memory-mib", "16384", "--schema", "schema.json", "--semantic-options", "semantic.json",
        "--max-input-bytes", "512", "--max-new-tokens", "128", "--max-result-bytes", "16384"];
    argv.extend_from_slice(extra);
    let matches = command::definition().try_get_matches_from(argv).unwrap();
    BatchCommand::from_matches(&matches).unwrap()
}
fn policy(limits: Limits) -> Int8SemanticOptions {
    Int8SemanticOptions {
        specification: SemanticVerificationSpec {
            schema_version: 1, experimental_opt_in: true, revision: "semantic-batch-runtime-test-v1".into(),
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
            projections: crate::native_engine::portable_int8::ProjectionWork {
                dot_products: 100_000_000_000, multiply_accumulates: 1_000_000_000_000_000 } },
    }
}
fn host(args: &CandidateArgs, limits: Limits, envelope: BatchEnvelope) -> CorpusLimits {
    CorpusLimits {
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: 64 * MIB,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: 65_536 } },
        transport: envelope.transport, preparation_reserve_bytes: limits.preparation_bytes,
        io_reserve_bytes: envelope.io_bytes,
    }
}
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".into(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".into(),
        recipe_id: "semantic-batch-metadata-only-fixture".into(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}

#[test]
fn actual_pinned_factory_binds_the_same_policy_and_quote_in_all_four_schedules() {
    for compact in [false, true] {
        let mut original = None;
        for (threads, rows) in [("1", None), ("2", None), ("1", Some("4")), ("2", Some("4"))] {
            let mut extra = vec!["--compute-threads", threads];
            if let Some(rows) = rows { extra.extend(["--prefill-rows", rows]); }
            if compact { extra.extend(["--input-encoding", "safe-bpe-v1",
                "--max-tokenizer-input-bytes", "65536", "--max-tokenizer-output-tokens", "8192"]); }
            let command = batch_command(&extra); let (args, limits) = command.validate().unwrap();
            let options = policy(limits); let envelope = command.policy_limits(&args, limits, &options).unwrap();
            let host = host(&args, limits, envelope); let prefill = command.semantic.prefill().unwrap();
            let schema = " {\"type\":\"number\"} \n";
            let request = semantic_cli::fixed_request(&command.semantic, schema.into(),
                command.semantic.host.task_budget(limits), &options).unwrap();
            let planner = planner(&facts(), &command.semantic, limits, request,
                options.clone(), envelope.native, envelope.encoding_work_limits).unwrap();
            let expected = requirements(&command, limits, &options, envelope, host, prefill).unwrap();
            let actual = check_factory(&command, limits, &options, envelope, host, prefill, &planner).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(planner.base_execution_identity().logical_model_digest, planner.judge_identity().logical_model_digest);
            assert_eq!(planner.base_execution_identity().task_spec, "extract-v1");
            assert_eq!(planner.judge_identity().task_spec, "judge-v1");
            let identity = planner.base_execution_identity().clone();
            if let Some(previous) = &original { assert_eq!(&identity, previous); }
            original = Some(identity);
            assert_eq!(planner.encoding_work_limits(), envelope.encoding_work_limits);
            assert_eq!(planner.encoding_workspace_bytes().unwrap() > 0, compact);
            let mut changed = options.clone();
            changed.specification.claims[0].prefix.push_str("changed ");
            assert_eq!(check_factory(&command, limits, &changed, envelope, host, prefill, &planner)
                .err().unwrap().code, "semantic_batch_factory_mismatch");
            let mut changed = envelope;
            changed.native.max_model_work.attention_pairs += 1;
            assert_eq!(check_factory(&command, limits, &options, changed, host, prefill, &planner)
                .err().unwrap().code, "semantic_batch_factory_mismatch");
            if compact {
                let mut changed = envelope;
                changed.encoding_work_limits.as_mut().unwrap().max_pair_checks += 1;
                assert_eq!(check_factory(&command, limits, &options, changed, host, prefill, &planner)
                    .err().unwrap().code, "semantic_batch_factory_mismatch");
            }
        }
    }
}

#[test]
fn whole_process_floor_prices_retained_population_native_team_and_combined_output_additively() {
    for extra in [vec![], vec!["--compute-threads", "4", "--prefill-rows", "8",
        "--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "65536",
        "--max-tokenizer-output-tokens", "8192"]] {
        let command = batch_command(&extra); let (args, mut limits) = command.validate().unwrap();
        let options = policy(limits); let envelope = command.policy_limits(&args, limits, &options).unwrap();
        let host = host(&args, limits, envelope); let prefill = command.semantic.prefill().unwrap();
        let quote = requirements(&command, limits, &options, envelope, host, prefill).unwrap();
        let required = minimum_memory(&args, limits, quote, prefill).unwrap();
        assert!(required > limits.weight_bytes + limits.preparation_bytes + quote.preparation_bytes);
        assert!(quote.output_bytes >= options.limits.max_output_bytes);
        limits.memory_bytes = required;
        admit_memory(&args, limits, quote, prefill).unwrap();
        limits.memory_bytes -= 1;
        assert_eq!(admit_memory(&args, limits, quote, prefill), Err(CandidateError::Memory));
        let mut bad = quote; bad.native.kv_bytes -= 1;
        assert_eq!(minimum_memory(&args, limits, bad, prefill), Err(CandidateError::Arguments));
        let mut bad = quote; bad.team_workspace_bytes += 1;
        assert_eq!(minimum_memory(&args, limits, bad, prefill), Err(CandidateError::Arguments));
        let mut bad = quote; bad.preparation_bytes = u64::MAX; bad.input_bytes = u64::MAX;
        assert_eq!(minimum_memory(&args, limits, bad, prefill), Err(CandidateError::Arguments));
        let mut bad = limits; bad.compute.threads += 1;
        assert_eq!(minimum_memory(&args, bad, quote, prefill), Err(CandidateError::Arguments));
    }
}

#[test]
fn typed_population_cancellation_survives_nested_physical_completion_and_supervision() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for native in [false, true] {
            let mut error = if native {
                HostedError::Native(crate::native_engine::strict_int8::StrictInt8Error::Cancelled(cause))
            } else { HostedError::SemanticExtraction(Int8SemanticError::Cancelled(cause)) };
            for _ in 0..2 {
                error = HostedError::Scope {
                    source: asupersync::cx::ScopedCpuError::WorkerCapExceeded { cap: 1 },
                    physical_error: Some(Box::new(error)),
                };
            }
            let failure = host_failure(error);
            assert_eq!(failure.cancellation, Some(cause));
            assert_eq!(failure.exit, cancelled(cause).exit);
        }
    }
    for cause in [FailFast, LinkedExit] {
        let error = HostedError::Stopped { stop: crate::hosted::RunStop {
            kind: cause, runtime_reason: None, runtime_checkpoint_error: None,
        }, task_error: Some(Box::new(HostedError::SemanticExtraction(Int8SemanticError::WorkBudget))) };
        let failure = host_failure(error);
        assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
        assert_eq!(failure.cancellation, Some(cause));
    }
}

#[test]
fn provenance_omits_byte_defaults_and_distinguishes_declared_invocation_policy_from_actual_work() {
    let facts = facts();
    let mut provenance = Provenance { protocol: command::PROTOCOL, schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative", task: "extract-semantic",
        execution: INT8_SEMANTIC_EXECUTION, same_model_correlated: true,
        calibration: "experimental_uncalibrated_diagnostic_only", model_id: &facts.model_id,
        source_revision: &facts.revision, source_root_sha256: &facts.source_root_sha256,
        logical_model_sha256: &facts.logical_model_sha256, quant_recipe: &facts.recipe_id,
        compute_threads: None, prefill_rows: None, input_encoding: None,
        invocation_encoding_work_limits: None };
    let value = serde_json::to_value(&provenance).unwrap();
    for key in ["compute_threads", "prefill_rows", "input_encoding", "invocation_encoding_work_limits",
        "encoding_work", "independently_verified", "accepted"] {
        assert!(value.get(key).is_none(), "{key}");
    }
    assert_eq!(value["protocol"], command::PROTOCOL);
    assert_eq!(value["same_model_correlated"], true);
    let command = batch_command(&["--input-encoding", "safe-bpe-v1"]);
    let encoding = command.semantic.encoding_policy().unwrap();
    provenance.input_encoding = Some(&encoding);
    provenance.invocation_encoding_work_limits = command.encoding_work_limits().unwrap();
    provenance.compute_threads = Some(2); provenance.prefill_rows = Some(4);
    let value = serde_json::to_value(provenance).unwrap();
    assert_eq!(value["invocation_encoding_work_limits"]["max_input_bytes"], 64 * MIB);
    assert_eq!(value["compute_threads"], 2); assert_eq!(value["prefill_rows"], 4);
    assert!(value.get("encoding_work").is_none());
}
