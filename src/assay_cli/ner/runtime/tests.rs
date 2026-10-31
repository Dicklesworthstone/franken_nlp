use super::*;
use super::super::tests::{command, NoRead, NoWrite};
use super::super::super::native::MIB;

#[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
mod model;

#[test]
fn ner_pre_io_peak_owns_both_residents_and_one_complete_source_host_envelope() {
    let options = command(false, &[]); let (limits, ner) = options.validate().unwrap();
    let single = Plan::new(&options, limits, ner).unwrap();
    let mut options = command(true, &[]); options.comparison.as_mut().unwrap().baseline_group_width = 127;
    let paired = Plan::new(&options, limits, ner).unwrap();
    let baseline = paired.baseline.as_ref().unwrap();
    let host = paired.host.requirements(&[128, 127], limits, ner).unwrap();
    let execution = host.native.kv_bytes + host.workspace_bytes + host.input_bytes + host.output_reservation_bytes;
    assert_eq!(paired.minimum_memory_bytes, paired.raw_staging_bytes + paired.candidate.memory.resident_bytes
        + baseline.memory.resident_bytes + execution.max(paired.candidate.memory.staging_bytes.max(baseline.memory.staging_bytes))
        + CLEANUP_BYTES);
    assert!(paired.minimum_memory_bytes >= single.minimum_memory_bytes + baseline.memory.resident_bytes);
    assert_eq!(paired.host.native.context_tokens, single.host.native.context_tokens);
    assert_eq!(paired.host.task, single.host.task);
    assert_eq!(paired.host.max_model_work, single.host.max_model_work);
    assert_eq!(paired.host.max_mask_node_visits, single.host.max_mask_node_visits);
    assert_eq!(paired.host.planning.source.verification.max_scan_steps, ner.max_verification_steps);
    assert_eq!(paired.host.planning.source.verification, single.host.planning.source.verification);
    assert_eq!(host.native.kv_bytes, paired.host.task.max_kv_bytes);
}

#[test]
fn ner_full_cap_quote_and_rounded_memory_ceiling_refuse_before_any_input_access() {
    let mut options = command(true, &[]); let (limits, ner) = options.validate().unwrap();
    let plan = Plan::new(&options, limits, ner).unwrap();
    options.args.memory_mib = Some(plan.minimum_memory_bytes.div_ceil(MIB));
    assert!(Plan::new(&options, limits, ner).is_ok());
    options.args.memory_mib = Some(plan.minimum_memory_bytes.div_ceil(MIB) - 1);
    let mut errors = Vec::new();
    assert_eq!(options.run(&mut NoRead, &mut NoWrite, &mut errors), ErrorCode::AdmissionOrResourceLimit.as_process_exit());
    let value: serde_json::Value = serde_json::from_slice(&errors).unwrap();
    assert_eq!(value["protocol"], NER_PROTOCOL); assert_eq!(value["reason"], "native_memory_ceiling");
    let options = command(false, &[]); let initial = Plan::new(&options, limits, ner).unwrap();
    let mut wider = ner; wider.max_spans_per_row += 1;
    let expanded = Plan::new(&options, limits, wider).unwrap();
    assert_eq!(expanded.raw_staging_bytes - initial.raw_staging_bytes, limits.max_rows as u64 * 64);
    assert!(expanded.minimum_memory_bytes > initial.minimum_memory_bytes);
    assert_eq!(expanded.candidate.memory, initial.candidate.memory);
    for field in 0..3 {
        let mut smaller = limits;
        match field { 0 => smaller.max_input_bytes -= 1, 1 => smaller.max_rows -= 1,
            _ => smaller.max_output_bytes -= 1 }
        let quote = Plan::new(&options, smaller, ner).unwrap();
        assert!(quote.raw_staging_bytes < initial.raw_staging_bytes);
        assert!(quote.minimum_memory_bytes < initial.minimum_memory_bytes);
    }
}

fn preflight_population() -> native_ner::Dataset {
    let mut input = Vec::new(); let mut splits = serde_json::Map::new();
    for (split, count) in [("development", 1), ("calibration", 1), ("locked_test", 5)] {
        let mut ids = Vec::new();
        for index in 0..count {
            let id = format!("{split}-{index}"); ids.push(id.clone());
            input.extend(canonjson::canonical_bytes(&serde_json::json!({"schema_version":1,"id":id,"split":split,
                "text":format!("Alice owns source {split} {index}."),"language":"en",
                "spans":[{"entity_type":"person","byte_start":0,"byte_end":5}]})).unwrap()); input.push(b'\n');
        }
        splits.insert(split.to_owned(), serde_json::json!(ids));
    }
    let suite = serde_json::json!({"schema_version":1,"task":"named_entity_recognition","dataset_id":"preflight",
        "dataset_version":"1","named_population":"owned source fixture", "options":{
            "types":["person"],"max_entities":4,"max_mention_scalars":64},
        "expected_input_sha256":Sha256Digest::of_bytes(&input),"splits":splits});
    native_ner::Dataset::parse(&canonjson::canonical_bytes(&suite).unwrap(), &input,
        Limits::default(), native_ner::NerLimits::default()).unwrap()
}

#[test]
fn ner_population_draws_count_only_locked_documents_and_both_paired_arms_before_model_io() {
    let dataset = preflight_population();
    assert_eq!(dataset.len(), 7); assert_eq!(dataset.replay_len(), 5);
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
fn ner_nested_source_errors_keep_resource_work_integrity_and_cancellation_categories() {
    use crate::{grammar::{mask::MaskOracleError, schema::SchemaError},
        hosted::{HostedError, Int4HostError},
        native_engine::{constrained::JsonDecodeError, constrained_int4::Int4JsonError,
            decode::DecodeCancellationKind, strict_int4::StrictInt4Error},
        tasks::{extract::{ExtractError, grouped_int4::Int4ExtractError}, ner::NerError,
            source_planning::{SourcePlanningError, grouped_int4::Int4SourceError}}};
    let cases = [
        (Int4SourceError::Planning(SourcePlanningError::AllocationRefused), ErrorCode::AdmissionOrResourceLimit),
        (Int4SourceError::Planning(SourcePlanningError::Ner(NerError::AllocationRefused)), ErrorCode::AdmissionOrResourceLimit),
        (Int4SourceError::Planning(SourcePlanningError::Ner(NerError::Extraction(ExtractError::Decode(
            JsonDecodeError::Cancelled(DecodeCancellationKind::User))))), ErrorCode::Cancelled),
        (Int4SourceError::Planning(SourcePlanningError::Extraction(ExtractError::Schema(SchemaError::Resource {
            pointer: "PRIVATE-pointer".to_owned(), reason: "PRIVATE-source".to_owned() }))), ErrorCode::AdmissionOrResourceLimit),
        (Int4SourceError::Extraction(Int4ExtractError::Native(Int4JsonError::Decode(JsonDecodeError::BudgetExceeded("PRIVATE-axis")))),
            ErrorCode::BudgetOrTimeout),
        (Int4SourceError::Extraction(Int4ExtractError::Native(Int4JsonError::Decode(JsonDecodeError::Mask(
            MaskOracleError::WorkBudgetExceeded { max_trie_node_visits: 1 })))), ErrorCode::BudgetOrTimeout),
        (Int4SourceError::Extraction(Int4ExtractError::Native(Int4JsonError::Native(StrictInt4Error::Work))), ErrorCode::BudgetOrTimeout),
        (Int4SourceError::Extraction(Int4ExtractError::Native(Int4JsonError::Native(StrictInt4Error::Cancelled(
            DecodeCancellationKind::Deadline)))), ErrorCode::BudgetOrTimeout),
        (Int4SourceError::Extraction(Int4ExtractError::Native(Int4JsonError::Native(StrictInt4Error::Allocation))),
            ErrorCode::AdmissionOrResourceLimit),
        (Int4SourceError::Extraction(Int4ExtractError::Native(Int4JsonError::WorkMismatch)), ErrorCode::ArtifactIntegrityOrFormatOrVersion),
        (Int4SourceError::Extraction(Int4ExtractError::Identity), ErrorCode::ArtifactIntegrityOrFormatOrVersion),
        (Int4SourceError::InvalidResult, ErrorCode::StructuredTaskNoResult),
    ];
    for (cause, expected) in cases {
        // A fail-fast wrapper must retain the physical source cause, rather
        // than collapsing a bounded failure to a generic cancellation.
        let error = host_failure(HostedError::Stopped { stop: crate::hosted::RunStop {
            kind: DecodeCancellationKind::FailFast, runtime_reason: None, runtime_checkpoint_error: None },
            task_error: Some(Box::new(HostedError::Int4(Int4HostError::SourceTask(cause)))) });
        assert_eq!(error.code, expected); assert!(!error.reason.contains("PRIVATE"));
    }
}

#[test]
fn ner_complete_output_uses_its_own_protocol_and_never_exceeds_outer_cap() {
    use super::super::super::native::runtime::encode;
    let report = serde_json::json!({"score_space":"not_computed","measurement":{"status":"insufficient_data",
        "split":"locked_test","rows":1,"minimum_samples":20}});
    let bytes = encode(NER_PROTOCOL, "eval", &report, Limits::default()).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["protocol"], NER_PROTOCOL); assert_eq!(value["evidence"], "native_model_replay");
    assert_eq!(value["activation_authority"], false); assert_eq!(bytes.last(), Some(&b'\n'));
    let mut exact = Limits::default(); exact.max_output_bytes = bytes.len();
    assert_eq!(encode(NER_PROTOCOL, "eval", &report, exact).unwrap(), bytes);
    exact.max_output_bytes -= 1;
    assert!(matches!(encode(NER_PROTOCOL, "eval", &report, exact),
        Err(Failure { code: ErrorCode::AdmissionOrResourceLimit, .. })));
}
