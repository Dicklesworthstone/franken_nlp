//! Admission, pinned preparation and publication fixtures. Ordinary tests do
//! not load weights. The ignored controller-only gate requires independent
//! artifact pins and supplies no parity evidence until actually executed.
use super::*;
use std::cell::Cell;
use crate::{
    native_engine::{constrained::JsonDecodeOutput, decode::DecodeStepControl,
        lmhead::{NANBEIGE_VOCAB_SIZE, scoring::ScoringWork}},
    tasks::extract::{ExtractResult, ExtractionGrounding, quantized::Int8ExtractRun,
        semantic::{ClaimRule, ClaimValueKind, SemanticFieldResult, SemanticFieldStatus,
            SemanticLimits, SemanticNotChecked, SemanticVerificationReceipt, SemanticVerificationSpec,
            CLAIM_RENDER_VERSION, SEMANTIC_VERIFICATION_VERSION}},
    tasks::judge::{FaithfulnessPolicy, FAITHFULNESS_VERSION},
    tokenizer::untrusted::UntrustedEncoding,
};

fn command() -> SemanticCommand {
    let matches = command::definition().try_get_matches_from([
        "extract-semantic", "--model", "must-not-open.fnlpq", "--memory-mib", "8192",
        "--schema", "schema.json", "--semantic-options", "semantic.json",
        "--max-input-bytes", "512", "--max-new-tokens", "128", "--max-result-bytes", "16384",
    ]).unwrap();
    SemanticCommand::from_matches(&matches).unwrap()
}
fn options(limits: Limits) -> Int8SemanticOptions {
    let mut work = Int8Work::default();
    work.forward_positions = 4096; work.projected_logits = 64_000_000;
    // Every charged position may attend to the full 2,048-position context
    // in all 44 effective layers and all 48 query heads.
    work.attention_pairs = work.forward_positions * 2048 * 44 * 48;
    work.projections.dot_products = 100_000_000_000;
    work.projections.multiply_accumulates = 1_000_000_000_000_000;
    Int8SemanticOptions {
        specification: SemanticVerificationSpec {
            schema_version: 1, experimental_opt_in: true, revision: "semantic-runtime-test-v1".into(),
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
        max_model_work: work,
    }
}
fn host(command: &SemanticCommand, args: &CandidateArgs, limits: Limits) -> Int8SemanticHostLimits {
    Int8SemanticHostLimits {
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: 64 * MIB,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: 65_536 } },
        masks: command.host.masks(), max_mask_node_visits: command.host.max_mask_node_visits,
        preparation_reserve_bytes: limits.preparation_bytes,
    }
}
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".into(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".into(),
        recipe_id: "semantic-metadata-only-fixture".into(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

#[test]
fn source_schema_checks_exact_numbers_claim_rules_and_complete_evidence_before_model_io() {
    let command = command(); let (_, limits) = command.validate().unwrap();
    let mut options = options(limits);
    let number = "0.12345678901234567890123456789012345678";
    let schema = format!(" {{\"type\":\"number\",\"const\":{number}}}\n");
    let source = "The amount is 0.12345678901234567890123456789012345678.";
    let request = source_schema(&command, &options, schema.clone(), source, command.host.task_budget(limits)).unwrap();
    assert_eq!(request.schema.as_bytes(), schema.as_bytes());
    for invalid in [r#"{"type":"number","type":"string"}"#, r#"{"$ref":"https://example.invalid/schema"}"#,
        r#"{"type":"string","x-fnlp-source":"verbatim"}"#] {
        assert!(matches!(source_schema(&command, &options, invalid.into(), source, request.budget),
            Err(CandidateError::Planning)));
    }
    options.specification.claims[0].value_kind = ClaimValueKind::String;
    assert!(matches!(source_schema(&command, &options, schema.clone(), source, request.budget),
        Err(CandidateError::Planning)));
    options.specification.claims[0].value_kind = ClaimValueKind::Number;
    options.specification.judge_policy.evidence_window_bytes = 1;
    assert!(matches!(source_schema(&command, &options, schema.clone(), source, request.budget),
        Err(CandidateError::Planning)));
    assert!(matches!(source_schema(&command, &options, schema,
        &"x".repeat(command.host.max_input_bytes + 1), request.budget), Err(CandidateError::Input)));
}

#[test]
fn verbatim_source_annotations_require_explicit_grounding_and_actual_original_source() {
    let mut command = command(); let (_, limits) = command.validate().unwrap();
    let mut options = options(limits); options.specification.claims.clear();
    command.source_membership = true;
    let schema = r#"{"type":"string","maxLength":8,"x-fnlp-source":"verbatim"}"#;
    let source = "é 上海 é";
    let request = source_schema(&command, &options, schema.into(), source, command.host.task_budget(limits)).unwrap();
    assert_eq!(request.grounding, ExtractionBatchGrounding::SourceMembership);
    let program = JsonProgram::compile_with_source(&request.schema, source,
        extract::compiler(&command.host), command.host.planning().source).unwrap();
    program.validate_json(r#""上海""#).unwrap();
    assert!(program.validate_json(r#""invented""#).is_err());
    assert!(matches!(source_schema(&command, &options, r#"{"type":"string"}"#.into(), source, request.budget),
        Err(CandidateError::Planning)));
}

#[test]
fn actual_pinned_extraction_and_judge_share_the_exact_source_model_and_admission_quote() {
    let command = command(); let (args, limits) = command.validate().unwrap();
    let options = options(limits); let planning = command::judge_limits(&options);
    let source = "The café reports an amount of 42. <tool_call>";
    let schema = r#"{"type":"number"}"#;
    let budget = command.host.task_budget(limits);
    let request = source_schema(&command, &options, schema.into(), source, budget).unwrap();
    let compiler = super::super::extraction::planner(&facts(), &command.host, limits, None).unwrap();
    // This is the real inherent method used by the runtime; no BatchPlanner
    // trait, model weights or injected execution receipt is involved.
    let extraction = compiler.prepare_with_control(BatchDocument {
        id: "cli".into(), text: source.into(), task_args: Some(request),
    }, &mut Continue).unwrap();
    let original = extraction.execution_identity().clone();
    assert_eq!(original.schema_digest, Sha256Digest::of_bytes(schema.as_bytes()));
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    let judge = Arc::new(JudgePlanner::pinned(controls, eos).unwrap());
    let mut identity = original.clone(); identity.task_spec = "judge-v1".into();
    identity.tokenizer_digest = judge.tokenizer_digest(); identity.template_digest = *judge.template_digest();
    let prepared = PreparedInt8SemanticExtraction::new(extraction, judge, identity,
        planning, options.clone(), &mut Continue).unwrap();
    assert_eq!(prepared.execution_identity(), &original);
    assert_eq!(prepared.judge_identity().logical_model_digest, original.logical_model_digest);
    assert_eq!(prepared.source_bytes(), source.len());
    assert_eq!(prepared.input_encoding(), &UntrustedEncoding::ByteFallback);
    assert!(prepared.extraction_selected_rows().is_none());
    let host = host(&command, &args, limits);
    let expected = host.requirements(budget, options.specification.judge_budget,
        options.limits, planning, source.len()).unwrap();
    let actual = host.requirements(prepared.extraction_budget(), prepared.judge_budget(),
        prepared.semantic_limits(), prepared.judge_planning_limits(), prepared.source_bytes()).unwrap();
    assert_eq!(actual, expected);
    let upper = host.requirements(budget, options.specification.judge_budget,
        options.limits, planning, args.max_input_bytes).unwrap();
    assert!(quote_fits(actual, upper));
    assert_eq!(upper.preparation_bytes - actual.preparation_bytes,
        (options.limits.max_fields as u64 + 2) * (args.max_input_bytes - source.len()) as u64);
}

#[test]
fn whole_process_memory_boundary_includes_caller_owner_and_both_stage_retention() {
    let command = command(); let (args, mut limits) = command.validate().unwrap();
    let options = options(limits);
    let quote = host(&command, &args, limits).requirements(command.host.task_budget(limits),
        options.specification.judge_budget, options.limits, command::judge_limits(&options),
        args.max_input_bytes).unwrap();
    let required = minimum_memory(&args, limits, quote, None).unwrap();
    assert!(required > limits.weight_bytes + limits.preparation_bytes + quote.preparation_bytes);
    limits.memory_bytes = required;
    admit_memory(&args, limits, quote, None).unwrap();
    limits.memory_bytes -= 1;
    assert_eq!(admit_memory(&args, limits, quote, None), Err(CandidateError::Arguments));
    let mut too_large = quote; too_large.preparation_bytes += 1;
    assert!(!quote_fits(too_large, quote));
    let mut wrong_native = quote; wrong_native.native.kv_bytes -= 1;
    assert_eq!(minimum_memory(&args, limits, wrong_native, None), Err(CandidateError::Arguments));
    let overflow = Int8SemanticHostRequirements { preparation_bytes: u64::MAX, ..quote };
    assert_eq!(minimum_memory(&args, limits, overflow, None), Err(CandidateError::Arguments));
    let mut wrong_schedule = limits; wrong_schedule.compute.threads = 2;
    assert_eq!(minimum_memory(&args, wrong_schedule, quote, None), Err(CandidateError::Arguments));
}

// Private receipt fixture for accounting/serialization only. Public execution
// obtains this type exclusively from the consumed native preparation.
fn completion() -> (Int8SemanticExtractionRun, Int8Work) {
    let original = Int8Work::for_sequence(0, 8, 3 * NANBEIGE_VOCAB_SIZE).unwrap();
    let actual = Int8Work::for_sequence(0, 7, 2 * NANBEIGE_VOCAB_SIZE).unwrap();
    (Int8SemanticExtractionRun {
        schema_version: 1, execution: INT8_SEMANTIC_EXECUTION, numerics_profile: STRICT_INT8_PROFILE,
        extraction_reserved_work: original, semantic_model_work: Int8Work::default(),
        reserved_model_work: original, model_work: actual, judge_head_count: 0, rewound_positions: 0,
        input_encoding: None,
        extraction: Int8ExtractRun { schema_version: 1,
            execution: crate::tasks::extract::quantized::INT8_EXTRACT_VERSION.into(), model_work: actual,
            result: ExtractResult { schema_version: 1, task_spec_version: "extract-v1".into(),
                score_space: crate::tasks::ir::ScoreSpace::NotComputed, grounding: ExtractionGrounding::NotRequested,
                source_fields: Vec::new(),
                output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT8_PROFILE.into(),
                    token_ids: vec![1, 2], json: "1".into(), forward_positions: actual.forward_positions,
                    projected_logits: actual.projected_logits, mask_node_visit_charge: 1 } } },
        semantic: SemanticVerificationReceipt {
            schema_version: 1, algorithm: SEMANTIC_VERIFICATION_VERSION.into(),
            claim_rendering: CLAIM_RENDER_VERSION.into(), judge_algorithm: FAITHFULNESS_VERSION.into(),
            specification_revision: "semantic-runtime-test-v1".into(),
            calibration: "experimental_uncalibrated_diagnostic_only".into(), same_model_correlated: true,
            attempted_fields: 0, checked_fields: 0, not_checked_fields: 1,
            fields: vec![SemanticFieldResult { json_pointer: "$".into(), status: SemanticFieldStatus::NotChecked,
                not_checked: Some(SemanticNotChecked::NoClaimRule), claim_rule_id: None, rendered_claim: None,
                same_model_correlated: None, judge: None }],
            work: ScoringWork { prefix_evaluations: 0, scored_edges: 0, projected_logits: 0 },
        },
    }, original)
}
fn axis(work: &mut Int8Work, index: usize) -> &mut u64 {
    match index { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits,
        2 => &mut work.attention_pairs, 3 => &mut work.projections.dot_products,
        _ => &mut work.projections.multiply_accumulates }
}

#[test]
fn result_boundary_rejects_all_five_refunds_corrupted_debits_and_promoted_provenance() {
    let (run, original) = completion();
    check_result(&run, original, original).unwrap();
    assert_eq!(check_result(&run, original, run.model_work), Err(CandidateError::Execution));
    for index in 0..5 {
        let mut short = original; *axis(&mut short, index) -= 1;
        assert_eq!(check_result(&run, original, short), Err(CandidateError::Execution));
        for slot in 0..3 {
            let (mut bad, original) = completion();
            let work = match slot { 0 => &mut bad.extraction_reserved_work,
                1 => &mut bad.reserved_model_work, _ => &mut bad.model_work };
            *axis(work, index) += 1;
            assert_eq!(check_result(&bad, original, original), Err(CandidateError::Execution));
        }
    }
    for mutation in 0..4 {
        let (mut bad, original) = completion();
        match mutation { 0 => bad.execution = "another-backend", 1 => bad.numerics_profile = "hf-bf16-eager",
            2 => bad.semantic.same_model_correlated = false, _ => bad.semantic.calibration = "qualified".into() }
        assert_eq!(check_result(&bad, original, original), Err(CandidateError::Execution));
    }
}

#[test]
fn every_typed_cancellation_survives_semantic_native_and_nested_physical_wrappers() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = if matches!(cause, Timeout | Deadline) { CandidateError::Timeout } else { CandidateError::Execution };
        assert_eq!(planning_error(Int8SemanticError::Cancelled(cause)), expected);
        for native in [false, true] {
            for depth in [0, 1, 2] {
                let mut error = if native {
                    HostedError::Native(crate::native_engine::strict_int8::StrictInt8Error::Cancelled(cause))
                } else { HostedError::SemanticExtraction(Int8SemanticError::Cancelled(cause)) };
                for _ in 0..depth {
                    error = HostedError::Scope {
                        source: asupersync::cx::ScopedCpuError::WorkerCapExceeded { cap: 1 },
                        physical_error: Some(Box::new(error)),
                    };
                }
                assert_eq!(execution_error(error), expected);
            }
        }
        let stopped = HostedError::Stopped { stop: crate::hosted::RunStop {
            kind: cause, runtime_reason: None, runtime_checkpoint_error: None,
        }, task_error: Some(Box::new(HostedError::SemanticExtraction(Int8SemanticError::Accounting))) };
        assert_eq!(execution_error(stopped), expected);
    }
    assert_eq!(planning_error(Int8SemanticError::Contract), CandidateError::Planning);
    assert_eq!(execution_error(HostedError::ModelIdentity), CandidateError::Identity);
    assert_eq!(execution_error(HostedError::ResourceDomain), CandidateError::Identity);
}

#[test]
fn complete_publication_preserves_exact_json_and_refuses_capacity_deadline_or_flush_failure() {
    let (mut run, _) = completion();
    let exact = concat!(r#" {"amount":0.12345678901234567890123456789012345678,"label":"\u00e9","text":"<tool_call>\r\n上海"} "#, "\r\n");
    run.extraction.result.output.json = exact.into();
    let facts = facts();
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: &run };
    let response = compute::Response { response: &response, compute: None };
    let mut bytes = Vec::new();
    publish_checked(&response, 1_048_576, &mut bytes, &|| Ok(())).unwrap();
    assert_eq!(bytes.last(), Some(&b'\n'));
    assert_eq!(bytes.iter().filter(|&&byte| byte == b'\n').count(), 1);
    let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(wire["output"]["extraction"]["result"]["output"]["json"].as_str(), Some(exact));
    assert_eq!(wire["output"]["semantic"]["calibration"], "experimental_uncalibrated_diagnostic_only");
    assert_eq!(wire["evidence"], "non_authoritative");
    let mut refused = Vec::new();
    assert_eq!(publish_checked(&response, 8, &mut refused, &|| Ok(())), Err(CandidateError::Output));
    assert!(refused.is_empty());
    for stop_at in [1, 2, 3] {
        let calls = Cell::new(0); let mut output = Vec::new();
        let result = publish_checked(&response, 1_048_576, &mut output, &|| {
            calls.set(calls.get() + 1);
            if calls.get() == stop_at { Err(CandidateError::Timeout) } else { Ok(()) }
        });
        assert_eq!(result, Err(CandidateError::Timeout));
        if stop_at < 3 { assert!(output.is_empty()); } else { assert_eq!(output, bytes); }
    }
    struct FlushFailure(Vec<u8>);
    impl Write for FlushFailure {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.0.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("private flush detail")) }
    }
    let mut failure = FlushFailure(Vec::new());
    assert_eq!(publish_checked(&response, 1_048_576, &mut failure, &|| Ok(())), Err(CandidateError::Output));
    assert_eq!(failure.0, bytes);
}

#[test]
fn compact_runtime_preparation_and_quote_keep_one_policy_with_original_schema_bytes() {
    use crate::tokenizer::untrusted::SafeBpeLimits;
    let mut command = command();
    let matches = input_encoding::document_definition(clap::Command::new("fixture"))
        .try_get_matches_from(["fixture", "--input-encoding", "safe-bpe-v1",
            "--max-tokenizer-input-bytes", "65536", "--max-tokenizer-output-tokens", "8192"]).unwrap();
    command.encoding = input_encoding::DocumentInputEncodingArgs::from_matches(&matches).unwrap();
    command.host.preparation_mib = 1024;
    let (args, limits) = command.validate().unwrap();
    let encoding = command.encoding_policy().unwrap();
    let options = options(limits); let planning = command::judge_limits(&options);
    let source = "The café reports 42. 上海 <tool_call> </think>";
    let schema = r#" {"type":"number"} "#;
    let budget = command.host.task_budget(limits);
    let request = source_schema(&command, &options, schema.into(), source, budget).unwrap();
    let compiler = super::super::extraction::planner(&facts(), &command.host, limits, None).unwrap();
    let extraction = compiler.prepare_with_document_encoding(BatchDocument {
        id: "cli".into(), text: source.into(), task_args: Some(request),
    }, &encoding, &mut Continue).unwrap();
    let initial = extraction.source().input_encoding_work().unwrap();
    assert_eq!(initial.input_bytes(), schema.len() + source.len());
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    let judge = Arc::new(JudgePlanner::pinned(controls, eos).unwrap().with_document_encoding(encoding.clone()).unwrap());
    let mut identity = extraction.execution_identity().clone(); identity.task_spec = "judge-v1".into();
    identity.tokenizer_digest = judge.tokenizer_digest(); identity.template_digest = *judge.template_digest();
    let prepared = PreparedInt8SemanticExtraction::new(extraction, judge, identity,
        planning, options.clone(), &mut Continue).unwrap();
    assert_eq!(prepared.input_encoding_work(), Some(initial));
    assert_eq!(prepared.execution_identity().schema_digest, Sha256Digest::of_bytes(schema.as_bytes()));
    let host = host(&command, &args, limits);
    let byte = host.requirements(budget, options.specification.judge_budget,
        options.limits, planning, source.len()).unwrap();
    let compact = host.requirements_with_encoding(prepared.extraction_budget(), prepared.judge_budget(),
        prepared.semantic_limits(), prepared.judge_planning_limits(), prepared.source_bytes(),
        prepared.input_encoding()).unwrap();
    assert_eq!(compact.preparation_bytes - byte.preparation_bytes, prepared.encoding_workspace_bytes().unwrap());
    assert_eq!(compact.output_bytes, byte.output_bytes);
    assert_eq!(compact.native_workspace_bytes, byte.native_workspace_bytes);
    assert_eq!(minimum_memory(&args, limits, compact, None).unwrap() - minimum_memory(&args, limits, byte, None).unwrap(),
        prepared.encoding_workspace_bytes().unwrap());
    let UntrustedEncoding::SafeBpeV1(actual) = prepared.input_encoding() else { panic!("compact policy"); };
    assert_eq!(actual.max_input_bytes, 65_536);
    assert_eq!(actual.max_output_tokens, 8192);
    assert_eq!(actual.max_pair_checks, SafeBpeLimits::MAX_PAIR_CHECKS);
}

fn scheduled_command(extra: &[&str]) -> SemanticCommand {
    let mut argv = vec!["extract-semantic", "--model", "must-not-open.fnlpq", "--memory-mib", "8192",
        "--schema", "schema.json", "--semantic-options", "semantic.json",
        "--max-input-bytes", "512", "--max-new-tokens", "128", "--max-result-bytes", "16384"];
    argv.extend_from_slice(extra);
    let matches = command::definition().try_get_matches_from(argv).unwrap();
    SemanticCommand::from_matches(&matches).unwrap()
}

#[test]
fn complete_semantic_memory_admits_coordinator_and_persistent_children_separately() {
    for compact in [false, true] {
        let policy = if compact { vec!["--input-encoding", "safe-bpe-v1"] } else { vec![] };
        let sequential = scheduled_command(&policy);
        let (serial_args, serial_limits) = sequential.validate().unwrap();
        let options = options(serial_limits); let planning = command::judge_limits(&options);
        let serial = host(&sequential, &serial_args, serial_limits).requirements_with_schedule(
            sequential.host.task_budget(serial_limits), options.specification.judge_budget, options.limits,
            planning, serial_args.max_input_bytes, &sequential.encoding_policy().unwrap(), None).unwrap();
        let base = minimum_memory(&serial_args, serial_limits, serial, None).unwrap();

        let mut flags = policy;
        flags.extend(["--compute-threads", "4", "--prefill-rows", "4"]);
        let scheduled = scheduled_command(&flags);
        let (args, mut limits) = scheduled.validate().unwrap();
        let prefill = scheduled.prefill().unwrap();
        let quote = host(&scheduled, &args, limits).requirements_with_schedule(
            scheduled.host.task_budget(limits), options.specification.judge_budget, options.limits,
            planning, args.max_input_bytes, &scheduled.encoding_policy().unwrap(), prefill).unwrap();
        let coordinator = prefill.unwrap().validate().unwrap();
        assert_eq!(quote.native, serial.native);
        assert_eq!(quote.preparation_bytes, serial.preparation_bytes);
        assert_eq!(quote.output_bytes, serial.output_bytes);
        let required = minimum_memory(&args, limits, quote, prefill).unwrap();
        assert_eq!(required - base, coordinator + limits.compute.workspace_bytes);
        limits.memory_bytes = required;
        admit_memory(&args, limits, quote, prefill).unwrap();
        limits.memory_bytes -= 1;
        assert_eq!(admit_memory(&args, limits, quote, prefill), Err(CandidateError::Arguments));
        assert_eq!(minimum_memory(&args, limits, serial, prefill), Err(CandidateError::Arguments));
        let mut missing_children = limits; missing_children.compute.workspace_bytes -= 1;
        assert_eq!(minimum_memory(&args, missing_children, quote, prefill), Err(CandidateError::Arguments));
        assert_eq!(minimum_memory(&args, limits, quote, None), Err(CandidateError::Arguments));
    }
}

#[test]
fn semantic_runtime_refuses_changed_schedule_before_options_source_or_model_io() {
    struct NoInput;
    impl Read for NoInput {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private input opened before admission"); }
    }
    for changed in 0..2 {
        let command = scheduled_command(&["--compute-threads", "4", "--prefill-rows", "4"]);
        let (mut args, mut limits) = command.validate().unwrap();
        if changed == 0 { limits.compute.workspace_bytes -= 1; }
        else { args.compute = compute::Args::default(); }
        let mut output = Vec::new();
        assert_eq!(execute(command, args, limits, &mut NoInput, &mut output), Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
}

#[test]
fn team_publication_attaches_only_the_admitted_compute_receipt_to_exact_semantic_output() {
    let (run, _) = completion();
    let facts = facts();
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: &run };
    let serial = compute::Response { response: &response, compute: None };
    let mut serial_bytes = Vec::new();
    publish_checked(&serial, 1_048_576, &mut serial_bytes, &|| Ok(())).unwrap();
    let serial: serde_json::Value = serde_json::from_slice(&serial_bytes).unwrap();
    assert!(serial.get("compute").is_none());
    // Private publication fixture only; these counters are not model evidence.
    let compute = crate::hosted::Int8ComputeReceipt { schedule: "int8-output-row-team-v1",
        compute_threads: 4, completed_projections: 9, completed_rows: 42, completed_worker_rows: 30 };
    let team = compute::Response { response: &response, compute: Some(&compute) };
    let mut team_bytes = Vec::new();
    publish_checked(&team, 1_048_576, &mut team_bytes, &|| Ok(())).unwrap();
    let team: serde_json::Value = serde_json::from_slice(&team_bytes).unwrap();
    assert_eq!(team["output"], serial["output"]);
    assert_eq!(team["compute"]["compute_threads"], 4);
    assert_eq!(team["compute"]["completed_worker_rows"], 30);
    assert_eq!(team["evidence"], "non_authoritative");
}

#[test]
#[ignore = "controller-only real INT8 artifact and fresh resource-host process; no parity claim until executed"]
fn complete_semantic_results_match_across_workers_and_prompt_schedules() {
    use crate::native_engine::simd::config::process_kernel;
    let path = std::env::var_os("FNLP_INT8_SEMANTIC_TEAM_MODEL").expect("admitted INT8 artifact required");
    let logical = std::env::var("FNLP_INT8_SEMANTIC_TEAM_LOGICAL_SHA256")
        .expect("independent logical model pin required");
    let source_root = std::env::var("FNLP_INT8_SEMANTIC_TEAM_SOURCE_ROOT_SHA256")
        .expect("independent source root pin required");
    Sha256Digest::from_hex(&logical).unwrap(); Sha256Digest::from_hex(&source_root).unwrap();
    let mut command = scheduled_command(&["--compute-threads", "4", "--prefill-rows", "3"]);
    command.host.model = path.into(); command.host.memory_mib = 16_384;
    let (args, limits) = command.validate().unwrap();
    let schedule = command.prefill().unwrap();
    let session = Session::new_with_prefill(&args, limits, schedule).unwrap();
    let options = options(limits); let planning = command::judge_limits(&options);
    let budget = command.host.task_budget(limits);
    let quote = host_limits(&session, &args, &command, limits).unwrap().requirements_with_schedule(
        budget, options.specification.judge_budget, options.limits, planning,
        args.max_input_bytes, &UntrustedEncoding::ByteFallback, schedule).unwrap();
    admit_memory(&args, limits, quote, schedule).unwrap();
    let facts = session.facts(&args).unwrap();
    assert_eq!(facts.logical_model_sha256, logical); assert_eq!(facts.source_root_sha256, source_root);
    assert_eq!(facts.revision, "f56ec5a9650268aa098496734743c25ea778bd2d");
    let compiler = super::super::extraction::planner(&facts, &command.host, limits, None).unwrap();
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|entry| entry.special && entry.surface == crate::template::IM_END)
        .unwrap().id;
    let judge = Arc::new(JudgePlanner::pinned(controls, eos).unwrap());
    let vocabulary = super::super::extraction::vocabulary().unwrap();
    let model = session.load(&args, limits, &facts, CancellationToken::default()).unwrap();
    // A constrained non-null scalar must create one actual correlated review.
    // No supplied extraction, fake judge score or repaired output enters this path.
    let schema = r#"{"type":"number","const":42}"#;
    let source = "The amount is 42.";
    let prepare = || {
        let request = source_schema(&command, &options, schema.into(), source, budget).unwrap();
        let extraction = compiler.prepare_with_control(BatchDocument {
            id: "semantic-real-model-gate".into(), text: source.into(), task_args: Some(request),
        }, &mut session.control()).unwrap();
        let mut identity = extraction.execution_identity().clone(); identity.task_spec = "judge-v1".into();
        identity.tokenizer_digest = judge.tokenizer_digest(); identity.template_digest = *judge.template_digest();
        PreparedInt8SemanticExtraction::new(extraction, Arc::clone(&judge), identity,
            planning, options.clone(), &mut session.control()).unwrap()
    };
    let baseline = session.engine.resources().memory_snapshot().committed_bytes;
    let drained = || {
        let memory = session.engine.resources().memory_snapshot();
        assert_eq!(memory.committed_bytes, baseline); assert_eq!(memory.reserved_bytes, 0);
        assert_eq!(memory.recorded_leaks, 0);
        assert_eq!(session.engine.resources().outstanding_closure_snapshot().active_closures, 0);
        assert_eq!(Arc::strong_count(&judge), 1); assert_eq!(Arc::strong_count(&vocabulary), 1);
    };
    let run = |width: usize, prefill: Option<Int8PrefillLimits>, cancellation: CancellationToken|
        -> Result<compute::Completion<Int8SemanticExtractionRun>, HostedError> {
        let prepared = prepare(); let reserved = prepared.extraction_reserved_work();
        let host = host_limits(&session, &args, &command, limits).unwrap();
        let result = if width == 1 {
            compute::Completion::Serial(match prefill {
                None => session.engine.execute_int8_semantic_extract(&model, prepared,
                    Arc::clone(&vocabulary), host, cancellation),
                Some(prefill) => session.engine.execute_int8_semantic_extract_layer_major(&model,
                    prepared, Arc::clone(&vocabulary), host, prefill, cancellation),
            }?)
        } else {
            compute::Completion::Team(match prefill {
                None => session.engine.execute_int8_semantic_extract_with_threads(&model, prepared,
                    Arc::clone(&vocabulary), host, process_kernel(), width, cancellation),
                Some(prefill) => session.engine.execute_int8_semantic_extract_layer_major_with_threads(&model,
                    prepared, Arc::clone(&vocabulary), host, process_kernel(), width, prefill, cancellation),
            }?)
        };
        let output = result.result();
        check_result(output, reserved, options.max_model_work).unwrap();
        output.verify_input_encoding(&UntrustedEncoding::ByteFallback, Default::default()).unwrap();
        assert_eq!(output.semantic.attempted_fields, 1); assert_eq!(output.semantic.fields.len(), 1);
        assert!(output.semantic.fields[0].rendered_claim.is_some());
        assert!(output.semantic.fields[0].judge.is_some()); assert!(output.judge_head_count > 0);
        assert!(output.extraction.model_work.forward_positions > 0);
        assert!(output.semantic_model_work.forward_positions > 0);
        Ok(result)
    };
    let prefill = |rows| Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() };
    let result = run(1, None, CancellationToken::default()).unwrap();
    let expected = crate::canonjson::canonical_bytes(result.result()).unwrap();
    let work = result.result().model_work; let review_work = result.result().semantic_model_work;
    drop(result); drained();
    for width in [1, 2, 4] { for schedule in [None, Some(prefill(1)), Some(prefill(3))] {
        if width == 1 && schedule.is_none() { continue; }
        let result = run(width, schedule, CancellationToken::default()).unwrap();
        assert_eq!(crate::canonjson::canonical_bytes(result.result()).unwrap(), expected);
        assert_eq!(result.result().model_work, work);
        assert_eq!(result.result().semantic_model_work, review_work);
        if let Some(receipt) = result.compute() {
            assert_eq!(receipt.compute_threads, width);
            assert_eq!(receipt.completed_rows, work.projections.dot_products);
            assert!(receipt.completed_projections > 0);
        }
        drop(result); drained();
    } }
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for width in [1, 4] {
            let cancellation = CancellationToken::default(); cancellation.cancel(cause);
            let error = run(width, Some(prefill(3)), cancellation).err()
                .expect("a real two-stage request must preserve pre-cancellation");
            assert!(matches!(error, HostedError::Stopped { stop, .. } if stop.kind == cause));
            drained();
        }
    }
    drop(vocabulary); drop(judge); drop(compiler); drop(model);
    let memory = session.engine.resources().memory_snapshot();
    assert_eq!(memory.committed_bytes, limits.preparation_bytes); assert_eq!(memory.reserved_bytes, 0);
    assert_eq!(memory.recorded_leaks, 0);
    assert_eq!(session.engine.resources().outstanding_closure_snapshot().active_closures, 0);
}
