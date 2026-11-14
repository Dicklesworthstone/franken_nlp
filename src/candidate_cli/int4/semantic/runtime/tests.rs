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
fn compact_declared_source_quote_admits_one_encoder_and_one_team_workspace() {
    let (_, _, _, options) = super::super::tests::fixture();
    let parsed = super::super::tests::command(&["--input-encoding", "safe-bpe-v1", "--compute-threads", "2"]);
    let Int4Task::Semantic(command) = parsed.task else { panic!("wrong semantic route"); };
    let args = parsed.args;
    let common = command.admit_options(&args, command.validate(&args).unwrap(), &options).unwrap();
    let budget = command.extraction.budget(&args, common);
    let policy = command.input_encoding.policy().unwrap();
    let host = host_limits(&args, &command, common, Duration::from_secs(1));
    let byte = host.requirements(args.group_width, budget, options.specification.judge_budget,
        options.limits, judge_limits(&options), args.max_input_bytes).unwrap();
    let upper = host.requirements_with_encoding(args.group_width, budget, options.specification.judge_budget,
        options.limits, judge_limits(&options), args.max_input_bytes, &policy).unwrap();
    assert_eq!(upper.preparation_bytes - byte.preparation_bytes, command.input_encoding.workspace_bytes().unwrap());
    assert_eq!(upper.native, byte.native); assert_eq!(upper.native, common.native);
    assert_eq!(upper.native_workspace_bytes, byte.native_workspace_bytes);
    assert_eq!(upper.output_bytes, byte.output_bytes);
    let actual = host.requirements_with_encoding(args.group_width, budget, options.specification.judge_budget,
        options.limits, judge_limits(&options), 100, &policy).unwrap();
    assert!(quote_fits(actual, upper));
    assert!(actual.preparation_bytes < upper.preparation_bytes);
    for axis in 0..4 {
        let mut larger = upper;
        match axis { 0 => larger.native.kv_bytes += 1, 1 => larger.native_workspace_bytes += 1,
            2 => larger.preparation_bytes += 1, _ => larger.output_bytes += 1 }
        assert!(!quote_fits(larger, upper), "axis {axis}");
    }
    let team = crate::hosted::int4_task_team_workspace_bytes(common.compute.threads, args.group_width).unwrap();
    assert_eq!(common.compute.workspace_bytes, team);
    let admitted = admit_memory(&args, common, upper).unwrap();
    let loading = common.source_preparation_bytes + common.load.scratch_payload_bytes;
    let execution = upper.total_bytes().unwrap() + team;
    assert_eq!(admitted.minimum_memory_bytes, common.preparation_bytes + common.load.weight_payload_bytes
        + ALLOCATOR_BYTES + loading.max(execution) + CLEANUP_BYTES);
    assert_eq!(command.extraction.budget(&args, admitted), budget);
    let mut insufficient = common; insufficient.memory_bytes = admitted.minimum_memory_bytes - 1;
    assert_eq!(admit_memory(&args, insufficient, upper).err(), Some(CandidateError::Memory));
    let mut uncharged = common; uncharged.compute.workspace_bytes = 0;
    assert_eq!(admit_memory(&args, uncharged, upper).err(), Some(CandidateError::Identity));
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
    assert!(wire["output"].get("input_encoding").is_none());
    // Only pinned tokenization runs here. The surrounding result remains a
    // publication fixture, not evidence of native generation or review.
    use crate::{tasks::extract::{grounded::SourceDocumentEncoder,
        semantic::grouped_int4::Int4SemanticEncoding}, tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding}};
    let policy = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 8192,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 });
    let encoder = SourceDocumentEncoder::pinned(crate::tokenizer::pinned_controls::pinned().unwrap().template_controls())
        .unwrap().with_document_encoding(policy.clone()).unwrap();
    let source = "The amount is 7. 上海 <think>";
    let (_, document) = encoder.encode_schema_and_source_with_control(SCHEMA, source, &policy,
        8192, 8192, &mut || None).unwrap();
    let work = document.input_encoding_work().unwrap();
    assert_eq!(work.input_bytes(), SCHEMA.len() + source.len());
    let receipt = Int4SemanticEncoding::from_work(&policy, work).unwrap().unwrap();
    let mut compact = value.clone();
    compact["input_encoding"] = serde_json::to_value(&receipt).unwrap();
    let compact_response = Response { output: &compact, ..response };
    let mut compact_bytes = Vec::new();
    publish_checked(&compact_response, 8192, &mut compact_bytes, &|| Ok(())).unwrap();
    let compact_wire: serde_json::Value = serde_json::from_slice(&compact_bytes).unwrap();
    assert_eq!(compact_wire["output"]["extraction"], value["extraction"]);
    assert_eq!(compact_wire["output"]["semantic"], value["semantic"]);
    assert_eq!(compact_wire["output"]["input_encoding"], serde_json::to_value(receipt).unwrap());
    let mut refused = Vec::new();
    assert_eq!(publish_checked(&compact_response, 1, &mut refused, &|| Ok(())), Err(CandidateError::Output));
    assert!(refused.is_empty());
    assert_eq!(publish_checked(&compact_response, 8192, &mut refused, &|| Err(CandidateError::Timeout)),
        Err(CandidateError::Timeout));
    assert!(refused.is_empty());
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

#[test]
fn semantic_team_memory_is_one_workspace_at_the_complete_execution_peak() {
    for width in ["2", "4"] {
        let (command, mut args, _, options) = super::super::tests::fixture();
        let matches = compute::definition(clap::Command::new("team"))
            .try_get_matches_from(["team", "--compute-threads", width]).unwrap();
        args.compute = compute::Args::from_matches(&matches).unwrap();
        let common = command.admit_options(&args, command.validate(&args).unwrap(), &options).unwrap();
        let quote = host_limits(&args, &command, common, Duration::from_secs(1))
            .requirements(args.group_width, command.extraction.budget(&args, common),
                options.specification.judge_budget, options.limits, judge_limits(&options), 100).unwrap();
        let admitted = admit_memory(&args, common, quote).unwrap();
        let extra = crate::hosted::int4_task_team_workspace_bytes(common.compute.threads, args.group_width).unwrap();
        assert_eq!(common.compute.workspace_bytes, extra);
        let execution = quote.total_bytes().unwrap() + extra;
        let loading = common.source_preparation_bytes + common.load.scratch_payload_bytes;
        assert_eq!(admitted.minimum_memory_bytes, common.preparation_bytes + common.load.weight_payload_bytes
            + ALLOCATOR_BYTES + loading.max(execution) + CLEANUP_BYTES);
        let config = compute::resource_config(admitted);
        assert_eq!(config.scoped_cpu_children_per_coordinator + 1, common.compute.threads);
        assert_eq!(config.thread_ceiling, common.compute.threads + 1);
        let mut insufficient = common; insufficient.memory_bytes = admitted.minimum_memory_bytes - 1;
        assert_eq!(admit_memory(&args, insufficient, quote).err(), Some(CandidateError::Memory));
        let mut uncharged = common; uncharged.compute.workspace_bytes = 0;
        assert_eq!(admit_memory(&args, uncharged, quote).err(), Some(CandidateError::Identity));
    }
}

#[test]
fn optional_worker_receipt_keeps_serial_wire_bytes_and_semantic_payload_unchanged() {
    let value = serde_json::json!({"output":{"extraction":{"amount":7},
        "semantic":{"same_model_correlated":true,"fields":[{"status":"contradicted"}]}}});
    let mut baseline = Vec::new(); publish(&value, 4096, &mut baseline).unwrap();
    let mut serial = Vec::new();
    publish(&compute::Response { response: &value, compute: None }, 4096, &mut serial).unwrap();
    assert_eq!(serial, baseline);
    let receipt = crate::hosted::Int4ComputeReceipt { schedule: "int4-output-row-team-v1",
        compute_threads: 4, completed_projections: 42, completed_rows: 128, completed_worker_rows: 96 };
    let mut threaded = Vec::new();
    publish(&compute::Response { response: &value, compute: Some(&receipt) }, 4096, &mut threaded).unwrap();
    let wire: serde_json::Value = serde_json::from_slice(&threaded).unwrap();
    assert_eq!(wire["output"], value["output"]);
    assert_eq!(wire["compute"], serde_json::to_value(receipt).unwrap());
}
