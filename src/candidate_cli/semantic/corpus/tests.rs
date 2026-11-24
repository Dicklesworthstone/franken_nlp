//! Fixed-policy, whole-population admission fixtures; no model results are injected.
use super::*;
use crate::tasks::{extract::semantic::{ClaimRule, ClaimValueKind, SemanticLimits,
    SemanticVerificationSpec}, ir::TaskBudget, judge::FaithfulnessPolicy};

fn command(extra: &[&str]) -> BatchCommand {
    let mut argv = vec!["candidate", "extract-semantic-batch", "--model", "must-not-open.fnlpq",
        "--memory-mib", "16384", "--schema", "schema.json", "--semantic-options", "semantic.json",
        "--max-input-bytes", "512", "--max-new-tokens", "128", "--max-result-bytes", "16384"];
    argv.extend_from_slice(extra);
    let matches = super::super::super::definition().try_get_matches_from(argv).unwrap();
    match CandidateCommand::from_matches(&matches).unwrap() {
        CandidateCommand::SemanticBatch(command) => command,
        _ => panic!("wrong semantic population route"),
    }
}
fn options(limits: Limits) -> Int8SemanticOptions {
    Int8SemanticOptions {
        specification: SemanticVerificationSpec {
            schema_version: 1, experimental_opt_in: true, revision: "semantic-batch-cli-test-v1".into(),
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

#[test]
fn explicit_population_route_preserves_fixed_policy_and_does_not_multiply_work_by_request_count() {
    let selected = command(&["population.ndjson", "--max-requests", "7"]);
    let (args, limits) = selected.validate().unwrap();
    let options = options(limits);
    let envelope = selected.policy_limits(&args, limits, &options).unwrap();
    assert_eq!(selected.semantic.input, std::path::Path::new("population.ndjson"));
    assert_eq!(envelope.native.max_model_work, options.max_model_work);
    assert_eq!(envelope.native.masks.max_visits_per_run, selected.semantic.host.max_mask_node_visits);
    assert_eq!(envelope.transport.max_requests, 7);
    assert_eq!(envelope.transport.max_output_line_bytes, options.limits.max_output_bytes as usize + 4096);
    assert_eq!(envelope.transport.max_work.forward_positions, options.max_model_work.forward_positions);
    assert_eq!(envelope.transport.max_work.projected_logits, options.max_model_work.projected_logits);
    assert_eq!(envelope.encoding_work_limits, None);
    assert_eq!(limits.compute.threads, 1);
    assert!(selected.semantic.prefill().unwrap().is_none());
    for index in 0..6 {
        let mut short = command(&[]);
        let work = options.max_model_work;
        match index {
            0 => short.stream.max_total_forward_positions = Some(work.forward_positions - 1),
            1 => short.stream.max_total_projected_logits = Some(work.projected_logits - 1),
            2 => short.stream.max_total_attention_pairs = Some(work.attention_pairs - 1),
            3 => short.stream.max_total_dot_products = Some(work.projections.dot_products - 1),
            4 => short.stream.max_total_multiply_accumulates = Some(work.projections.multiply_accumulates - 1),
            _ => short.stream.max_total_mask_node_visits = Some(short.semantic.host.max_mask_node_visits - 1),
        }
        assert!(short.policy_limits(&args, limits, &options).is_err(), "underfunded axis {index}");
    }
}

#[test]
fn all_six_run_flags_are_explicit_and_independent_from_per_attempt_limits() {
    let command = command(&["--max-total-forward-positions", "8192",
        "--max-total-projected-logits", "128000000", "--max-total-attention-pairs", "100000000000",
        "--max-total-dot-products", "200000000000", "--max-total-multiply-accumulates", "2000000000000000",
        "--max-total-mask-node-visits", "18446744073709551615"]);
    let (args, limits) = command.validate().unwrap();
    let options = options(limits); let envelope = command.policy_limits(&args, limits, &options).unwrap();
    assert_eq!(envelope.native.max_model_work, Int8Work { forward_positions: 8192,
        projected_logits: 128_000_000, attention_pairs: 100_000_000_000,
        projections: ProjectionWork { dot_products: 200_000_000_000,
            multiply_accumulates: 2_000_000_000_000_000 } });
    assert_eq!(envelope.native.masks.max_visits_per_run, u64::MAX);
    assert_eq!(envelope.native.masks.max_visits_per_item, command.semantic.host.max_mask_node_visits);
    assert_eq!(options.max_model_work.forward_positions, 4096);
}

#[test]
fn invalid_transport_schedule_and_compact_totals_refuse_before_private_io() {
    struct NoIo;
    impl Read for NoIo {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("population input read"); }
    }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("population output written"); }
        fn flush(&mut self) -> io::Result<()> { panic!("population output flushed"); }
    }
    for extra in [
        vec!["--max-requests", "0"], vec!["--max-line-bytes", "1"],
        vec!["--max-total-forward-positions", "0"], vec!["--max-total-mask-node-visits", "0"],
        vec!["--compute-threads", "65"], vec!["--prefill-rows", "0"],
        vec!["--max-total-tokenizer-input-bytes", "1"],
        vec!["--input-encoding", "safe-bpe-v1", "--max-total-tokenizer-pair-bytes", "0"],
    ] {
        let failure = command(&extra).execute_owned(NoIo, NoIo).err().unwrap();
        assert_eq!(failure.exit, ErrorCode::Usage, "{extra:?}");
        assert_eq!(failure.cancellation, None);
    }
    let compact = command(&["--input-encoding", "safe-bpe-v1",
        "--max-tokenizer-input-bytes", "65536", "--max-tokenizer-output-tokens", "8192",
        "--max-total-tokenizer-input-bytes", "100000", "--max-total-tokenizer-output-tokens", "20000",
        "--max-total-tokenizer-pair-checks", "2000000", "--max-total-tokenizer-pair-bytes", "9000000"]);
    compact.validate().unwrap();
    assert_eq!(compact.encoding_work_limits().unwrap(), Some(SafeBpeWorkLimits {
        max_input_bytes: 100_000, max_output_tokens: 20_000,
        max_pair_checks: 2_000_000, max_pair_bytes: 9_000_000 }));
    assert!(matches!(compact.semantic.encoding_policy().unwrap(),
        UntrustedEncoding::SafeBpeV1(call) if call.max_input_bytes == 65_536 && call.max_output_tokens == 8192));
}

#[test]
fn fixed_schema_keeps_exact_numbers_and_defers_membership_to_real_documents() {
    let mut command = command(&[]);
    let (_, limits) = command.validate().unwrap(); let mut options = options(limits);
    let schema = " {\"type\":\"number\",\"const\":0.12345678901234567890123456789012345678}\n";
    let request = fixed_request(&command.semantic, schema.into(),
        command.semantic.host.task_budget(limits), &options).unwrap();
    assert_eq!(request.schema.as_bytes(), schema.as_bytes());
    for invalid in [r#"{"type":"number","type":"string"}"#,
        r#"{"$ref":"https://example.invalid/schema"}"#,
        r#"{"type":"string","x-fnlp-source":"verbatim"}"#] {
        assert!(fixed_request(&command.semantic, invalid.into(), request.budget, &options).is_err());
    }
    options.specification.claims.clear();
    command.semantic.source_membership = true;
    let schema = r#"{"type":"string","maxLength":8,"x-fnlp-source":"verbatim"}"#;
    let request = fixed_request(&command.semantic, schema.into(), request.budget, &options).unwrap();
    assert_eq!(request.grounding, crate::batch::extract::ExtractionBatchGrounding::SourceMembership);
    assert_eq!(request.schema, schema);
    assert!(fixed_request(&command.semantic, r#"{"type":"string"}"#.into(), request.budget, &options).is_err());
}
