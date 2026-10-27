//! UNRUN exact-schema, resource-quote and publication fixtures; no inference.
use super::*;

const SCHEMA: &str = r#"{"type":"object","properties":{"amount":{"type":"integer"}},"required":["amount"],"additionalProperties":false}"#;

#[test]
fn schema_rule_type_and_evidence_partition_are_checked_before_model_loading() {
    let (command, args, common, options) = super::super::tests::fixture();
    let budget = command.extraction.budget(&args, common);
    assert_eq!(source_schema(&command, &options, SCHEMA.to_owned(), "The amount is 7.", budget).unwrap().schema, SCHEMA);
    let mut invalid = options.clone();
    invalid.specification.claims[0].value_kind = crate::tasks::extract::semantic::ClaimValueKind::String;
    assert!(source_schema(&command, &invalid, SCHEMA.to_owned(), "The amount is 7.", budget).is_err());
    let mut invalid = options.clone();
    invalid.specification.claims[0].path = vec![crate::tasks::extract::semantic::ClaimPathStep::Property("absent".to_owned())];
    assert!(source_schema(&command, &invalid, SCHEMA.to_owned(), "The amount is 7.", budget).is_err());
    let mut invalid = options;
    invalid.specification.judge_policy.evidence_window_bytes = 1;
    invalid.specification.judge_policy.max_evidence_windows = 1;
    invalid.specification.judge_policy.max_evidence_spans = 1;
    assert!(source_schema(&command, &invalid, SCHEMA.to_owned(), "The amount is 7.", budget).is_err());
}

#[test]
fn exact_decimal_schema_authority_is_not_rounded_through_options_json() {
    let (command, args, common, mut options) = super::super::tests::fixture();
    options.specification.claims.clear();
    let schema = r#"{"type":"number","const":12345678901234567890123456789012345678}"#;
    let request = source_schema(&command, &options, schema.to_owned(), "source",
        command.extraction.budget(&args, common)).unwrap();
    assert_eq!(request.schema, schema);
}

#[test]
fn whole_request_memory_matches_the_host_with_both_preparation_owners_live() {
    let (command, args, common, options) = super::super::tests::fixture();
    let common = command.admit_options(&args, common, &options).unwrap();
    let budget = command.extraction.budget(&args, common);
    let host = host_limits(&args, &command, common, Duration::from_secs(1));
    let quote = host.requirements(args.group_width, budget, options.specification.judge_budget,
        options.limits, judge_limits(&options), 100).unwrap();
    let run = quote.total_bytes().unwrap();
    let admitted = admit_memory(&args, common, quote).unwrap();
    let resident = common.load.weight_payload_bytes + ALLOCATOR_BYTES;
    assert!(admitted.minimum_memory_bytes >= common.preparation_bytes + resident + run + CLEANUP_BYTES);
    assert!(quote.preparation_bytes >= host.preparation_reserve_bytes);
    assert!(quote.output_bytes >= options.limits.max_output_bytes);
    assert_eq!(quote.native, common.native);
    let mut too_small = common; too_small.memory_bytes = admitted.minimum_memory_bytes - 1;
    assert_eq!(admit_memory(&args, too_small, quote).err(), Some(CandidateError::Memory));
}

#[test]
fn retained_source_and_judge_outputs_grow_with_all_fields_not_one_field() {
    let (command, args, common, mut options) = super::super::tests::fixture();
    let budget = command.extraction.budget(&args, common);
    let host = host_limits(&args, &command, common, Duration::from_secs(1));
    let before = host.requirements(args.group_width, budget, options.specification.judge_budget,
        options.limits, judge_limits(&options), 100).unwrap();
    options.limits.max_fields += 1;
    let after = host.requirements(args.group_width, budget, options.specification.judge_budget,
        options.limits, judge_limits(&options), 100).unwrap();
    assert!(after.preparation_bytes > before.preparation_bytes);
    let source = host.requirements(args.group_width, budget, options.specification.judge_budget,
        options.limits, judge_limits(&options), 101).unwrap();
    assert!(source.preparation_bytes >= after.preparation_bytes + options.limits.max_fields as u64);
}

#[test]
fn publication_preserves_original_extraction_and_correlated_diagnostics_without_private_configuration() {
    let value = serde_json::json!({
        "extraction":{"result":{"output":{"json":"{\"amount\":7}"}}},
        "semantic":{"same_model_correlated":true,"calibration":"experimental_uncalibrated_diagnostic_only",
            "fields":[{"json_pointer":"$/amount","status":"contradicted"}]}
    });
    let response = Response { protocol: "fnlp-candidate-int4-semantic-extraction-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", model_id: "Nanbeige4.2-3B",
        source_revision: "fixture", source_root_sha256: "fixture-root", model_content_sha256: "fixture-model",
        candidate_format: "fixture-format", quant_recipe: "fixture-group", group_width: 128,
        integer_kernel: "s_scalar_i32", snapshot: None, output: &value };
    let mut bytes = Vec::new();
    publish(&response, 8192, &mut bytes).unwrap();
    let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(wire["output"], value);
    assert_eq!(wire["evidence"], "non_authoritative");
    for absent in ["semantic_options", "schema_path", "source_dir", "source_manifest", "prompt_digest",
        "decision_policy_digest", "judge_identity", "snapshot"] {
        assert!(wire.get(absent).is_none());
    }
}

#[test]
fn cancellation_causes_survive_private_error_mapping_and_physical_wrapper_failures() {
    for (cause, expected) in [(DecodeCancellationKind::Deadline, CandidateError::Timeout),
        (DecodeCancellationKind::Timeout, CandidateError::Timeout),
        (DecodeCancellationKind::User, CandidateError::Execution)] {
        assert_eq!(planning_error(Int4SemanticError::Cancelled(cause)), expected);
        assert_eq!(execution_error(HostedError::Int4(Int4HostError::SemanticExtraction(
            Int4SemanticError::Cancelled(cause)))), expected);
    }
}
