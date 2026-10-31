use super::*;
use super::super::tests::{command, NoRead, NoWrite};
use super::super::super::native::MIB;

#[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
mod model;

#[test]
fn extraction_complete_peak_prices_both_models_exact_gold_and_one_native_workspace() {
    let options = command(false, &[]); let (limits, extract) = options.validate().unwrap();
    let single = Plan::new(&options, limits, extract).unwrap();
    let mut options = command(true, &[]); options.comparison.as_mut().unwrap().baseline_group_width = 127;
    let paired = Plan::new(&options, limits, extract).unwrap(); let baseline = paired.baseline.as_ref().unwrap();
    let host = paired.host.requirements(&[128, 127], limits, extract).unwrap();
    let execution = host.native.kv_bytes + host.workspace_bytes + host.input_bytes + host.output_reservation_bytes;
    assert_eq!(paired.minimum_memory_bytes, paired.raw_staging_bytes + paired.candidate.memory.resident_bytes
        + baseline.memory.resident_bytes + execution.max(paired.candidate.memory.staging_bytes.max(baseline.memory.staging_bytes))
        + CLEANUP_BYTES);
    assert!(paired.minimum_memory_bytes >= single.minimum_memory_bytes + baseline.memory.resident_bytes);
    assert_eq!(paired.host.task, single.host.task); assert_eq!(paired.host.max_model_work, single.host.max_model_work);
    assert_eq!(paired.host.max_mask_node_visits, single.host.max_mask_node_visits);
    assert_eq!(paired.host.compiler, single.host.compiler);
    assert_eq!(paired.host.source.verification, single.host.source.verification);
    assert_eq!(paired.host.source.verification.max_scan_steps, extract.max_verification_steps);
    assert_eq!(host.native.kv_bytes, paired.host.task.max_kv_bytes);
}

#[test]
fn extraction_declared_memory_ceiling_is_checked_before_any_input_or_model_io() {
    let mut options = command(true, &[]); let (limits, extract) = options.validate().unwrap();
    let required = Plan::new(&options, limits, extract).unwrap().minimum_memory_bytes;
    options.args.memory_mib = Some(required.div_ceil(MIB)); assert!(Plan::new(&options, limits, extract).is_ok());
    options.args.memory_mib = Some(required.div_ceil(MIB) - 1);
    let mut errors = Vec::new();
    assert_eq!(options.run(&mut NoRead, &mut NoWrite, &mut errors), ErrorCode::AdmissionOrResourceLimit.as_process_exit());
    let value: serde_json::Value = serde_json::from_slice(&errors).unwrap();
    assert_eq!(value["protocol"], EXTRACT_PROTOCOL); assert_eq!(value["reason"], "native_memory_ceiling");
    let mut options = command(false, &[]); options.args.memory_mib = Some(u64::MAX);
    assert_eq!(options.validate().unwrap_err().code, ErrorCode::Usage);
}

#[test]
fn exact_gold_node_capacity_is_intersected_with_encoded_input_and_schema_roots_are_priced() {
    let options = command(false, &["--max-rows", "16", "--max-input-bytes", "65536", "--max-line-bytes", "8192",
        "--max-json-nodes-per-row", "256", "--max-fields-per-row", "128"]);
    let (limits, extract) = options.validate().unwrap(); let initial = Plan::new(&options, limits, extract).unwrap();
    let mut expanded = extract; expanded.max_json_nodes_per_row = 512;
    let wider = Plan::new(&options, limits, expanded).unwrap();
    assert_eq!(wider.raw_staging_bytes - initial.raw_staging_bytes, 16 * 256 * 128);
    assert!(wider.minimum_memory_bytes > initial.minimum_memory_bytes);
    // Once encoded input is the tighter bound, increasing the per-row node
    // capacity cannot manufacture an unpriced rows*nodes cross product.
    let mut saturated = extract; saturated.max_json_nodes_per_row = 4096;
    let capped = Plan::new(&options, limits, saturated).unwrap(); saturated.max_json_nodes_per_row = 8192;
    assert_eq!(Plan::new(&options, limits, saturated).unwrap().raw_staging_bytes, capped.raw_staging_bytes);
    let mut more_schemas = extract; more_schemas.max_schemas += 1;
    let catalogue = Plan::new(&options, limits, more_schemas).unwrap();
    assert!(catalogue.raw_staging_bytes > initial.raw_staging_bytes);
    assert!(catalogue.minimum_memory_bytes > initial.minimum_memory_bytes);
    assert_eq!(catalogue.candidate.memory, initial.candidate.memory);
    // The active parser stack remains priced even when a tight retained-node
    // cap admits very little JSON and the input fails before schema validation.
    let mut deeper = extract; deeper.max_json_depth += 1;
    let stack = Plan::new(&options, limits, deeper).unwrap();
    assert_eq!(stack.raw_staging_bytes - initial.raw_staging_bytes, 2 * limits.max_line_bytes as u64);
    assert!(stack.minimum_memory_bytes > initial.minimum_memory_bytes);
}

fn preflight_population() -> native_extract::Dataset {
    let schema = r#"{"type":"object","additionalProperties":false,"required":["amount"],"properties":{"amount":{"type":"number"}}}"#;
    let expected = r#"{"amount":12345678901234567890123456789012345678}"#;
    let mut input = Vec::new(); let mut splits = serde_json::Map::new();
    for (split, count) in [("development", 1), ("calibration", 1), ("locked_test", 5)] {
        let mut ids = Vec::new();
        for index in 0..count {
            let id = format!("{split}-{index}"); ids.push(id.clone());
            input.extend(canonjson::canonical_bytes(&serde_json::json!({"schema_version":1,"id":id,"split":split,
                "text":format!("Owned exact-decimal source {split} {index}"),"schema_id":"amount","language":"en",
                "expected_json":expected})).unwrap()); input.push(b'\n');
        }
        splits.insert(split.to_owned(), serde_json::json!(ids));
    }
    let suite = serde_json::json!({"schema_version":1,"task":"structured_extraction","dataset_id":"preflight",
        "dataset_version":"1","named_population":"owned exact JSON fixture", "schemas":[{
            "id":"amount","schema_json":schema,"grounding":"structural"}],
        "expected_input_sha256":Sha256Digest::of_bytes(&input),"splits":splits});
    let options = command(false, &[]); let (limits, extract) = options.validate().unwrap();
    native_extract::Dataset::parse(&canonjson::canonical_bytes(&suite).unwrap(), &input,
        limits, extract, options.args.compiler(), options.args.source(limits)).unwrap()
}

#[test]
fn extraction_exact_json_preflight_counts_only_locked_documents_and_both_paired_arms() {
    let dataset = preflight_population(); assert_eq!(dataset.len(), 7); assert_eq!(dataset.replay_len(), 5);
    let key = CommitmentKey::new("fixture", [0x42; 32]).unwrap();
    let options = command(false, &[]).args.statistics.options();
    let mut limits = Limits::default(); limits.max_bootstrap_draws = 1999;
    assert!(preflight_replay(&dataset, &key, "fixture", options, limits, false).is_ok());
    assert_eq!(preflight_replay(&dataset, &key, "fixture", options, limits, true).unwrap_err().code,
        ErrorCode::AdmissionOrResourceLimit);
    limits.max_bootstrap_draws = 2000;
    assert!(preflight_replay(&dataset, &key, "fixture", options, limits, true).is_ok());
    assert!(preflight_replay(&dataset, &key, "PRIVATE invalid\nnamespace", options, limits, false).is_err());
}

#[test]
fn direct_extraction_and_batch_planning_causes_keep_typed_sanitized_failures() {
    use crate::{batch::{BatchCode, BatchFault}, hosted::{HostedError, Int4HostError},
        native_engine::{constrained::JsonDecodeError, constrained_int4::Int4JsonError,
            decode::DecodeCancellationKind, strict_int4::StrictInt4Error},
        tasks::extract::{ExtractError, grouped_int4::Int4ExtractError}};
    let cases = [
        (Int4ExtractError::Extraction(ExtractError::AllocationRefused), ErrorCode::AdmissionOrResourceLimit),
        (Int4ExtractError::Extraction(ExtractError::OutputBudgetExceeded), ErrorCode::BudgetOrTimeout),
        (Int4ExtractError::Native(Int4JsonError::Decode(JsonDecodeError::BudgetExceeded("PRIVATE-axis"))), ErrorCode::BudgetOrTimeout),
        (Int4ExtractError::Native(Int4JsonError::Native(StrictInt4Error::Work)), ErrorCode::BudgetOrTimeout),
        (Int4ExtractError::Native(Int4JsonError::Decode(JsonDecodeError::Cancelled(DecodeCancellationKind::User))), ErrorCode::Cancelled),
        (Int4ExtractError::Native(Int4JsonError::Native(StrictInt4Error::Cancelled(DecodeCancellationKind::Deadline))), ErrorCode::BudgetOrTimeout),
        (Int4ExtractError::Native(Int4JsonError::Native(StrictInt4Error::Allocation)), ErrorCode::AdmissionOrResourceLimit),
        (Int4ExtractError::Native(Int4JsonError::WorkMismatch), ErrorCode::ArtifactIntegrityOrFormatOrVersion),
        (Int4ExtractError::Identity, ErrorCode::ArtifactIntegrityOrFormatOrVersion),
        (Int4ExtractError::Extraction(ExtractError::InvalidResult), ErrorCode::StructuredTaskNoResult),
    ];
    for (cause, expected) in cases {
        let error = host_failure(HostedError::Stopped { stop: crate::hosted::RunStop {
            kind: DecodeCancellationKind::FailFast, runtime_reason: None, runtime_checkpoint_error: None },
            task_error: Some(Box::new(HostedError::Int4(Int4HostError::Extraction(cause)))) });
        assert_eq!(error.code, expected); assert!(!error.reason.contains("PRIVATE"));
    }
    for (cause, expected) in [(BatchFault::cancelled(DecodeCancellationKind::Deadline), ErrorCode::BudgetOrTimeout),
        (BatchFault::cancelled(DecodeCancellationKind::User), ErrorCode::Cancelled),
        (BatchCode::Allocation.into(), ErrorCode::AdmissionOrResourceLimit),
        (BatchCode::WorkLimit.into(), ErrorCode::BudgetOrTimeout),
        (BatchCode::InvalidExecution.into(), ErrorCode::ArtifactIntegrityOrFormatOrVersion),
        (BatchCode::Planning.into(), ErrorCode::StructuredTaskNoResult)] {
        assert_eq!(host_failure(HostedError::BatchSetup(cause)).code, expected);
    }
}

#[test]
fn extraction_output_is_complete_protocol_scoped_and_admitted_with_the_final_newline() {
    use super::super::super::native::runtime::encode;
    let report = serde_json::json!({"score_space":"not_computed","measurement":{"status":"insufficient_data",
        "split":"locked_test","rows":1,"minimum_samples":20}});
    let bytes = encode(EXTRACT_PROTOCOL, "eval", &report, Limits::default()).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["protocol"], EXTRACT_PROTOCOL); assert_eq!(value["evidence"], "native_model_replay");
    assert_eq!(value["activation_authority"], false); assert_eq!(bytes.last(), Some(&b'\n'));
    let mut exact = Limits::default(); exact.max_output_bytes = bytes.len();
    assert_eq!(encode(EXTRACT_PROTOCOL, "eval", &report, exact).unwrap(), bytes);
    exact.max_output_bytes -= 1;
    assert!(matches!(encode(EXTRACT_PROTOCOL, "eval", &report, exact),
        Err(Failure { code: ErrorCode::AdmissionOrResourceLimit, .. })));
}
