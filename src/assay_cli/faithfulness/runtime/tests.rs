use super::*;
use super::super::tests::{command, NoRead, NoWrite};
use super::super::super::native::MIB;
#[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))] mod model;

#[test]
fn complete_peak_admits_both_models_one_engine_and_every_prepared_head_before_io() {
    let options = command(false, &[]); let limits = options.validate().unwrap(); let one = Plan::new(&options, limits).unwrap();
    let mut options = command(true, &[]); options.comparison.as_mut().unwrap().baseline_group_width = 127;
    let pair = Plan::new(&options, limits).unwrap(); let baseline = pair.baseline.as_ref().unwrap();
    let need = pair.host.requirements(&[128, 127], limits).unwrap();
    let execution = need.native.kv_bytes + need.workspace_bytes + need.input_bytes + need.output_reservation_bytes;
    assert_eq!(pair.minimum_memory_bytes, pair.raw_staging_bytes + pair.candidate.memory.resident_bytes
        + baseline.memory.resident_bytes + execution.max(pair.candidate.memory.staging_bytes.max(baseline.memory.staging_bytes))
        + CLEANUP_BYTES);
    assert!(pair.minimum_memory_bytes >= one.minimum_memory_bytes + baseline.memory.resident_bytes);
    assert_eq!(pair.host.task, one.host.task); assert_eq!(pair.host.max_model_work, one.host.max_model_work);
    options.args.memory_mib = Some(pair.minimum_memory_bytes.div_ceil(MIB)); assert!(Plan::new(&options, limits).is_ok());
    options.args.memory_mib = Some(pair.minimum_memory_bytes.div_ceil(MIB) - 1);
    let mut errors = Vec::new();
    assert_eq!(options.run(&mut NoRead, &mut NoWrite, &mut errors), ErrorCode::AdmissionOrResourceLimit.as_process_exit());
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&errors).unwrap()["reason"], "native_memory_ceiling");
}
fn population() -> native_faithfulness::Dataset {
    let mut bytes = Vec::new(); let mut splits = serde_json::Map::new();
    for (split, sources, claims) in [("development", 1, 1), ("calibration", 1, 1), ("locked_test", 5, 2)] {
        let mut ids = Vec::new();
        for source in 0..sources { for claim in 0..claims {
            let id = format!("{split}-{source}-{claim}"); ids.push(id.clone());
            bytes.extend(canonjson::canonical_bytes(&serde_json::json!({"schema_version":1,"id":id,"split":split,
                "source":format!("Exact owned {split} source {source}."),"claim":format!("Claim {claim}."),
                "relation":"unsupported"})).unwrap()); bytes.push(b'\n');
        } }
        splits.insert(split.to_owned(), serde_json::json!(ids));
    }
    let suite = serde_json::json!({"schema_version":1,"task":"faithfulness","dataset_id":"preflight",
        "dataset_version":"1","named_population":"owned fixture", "policy":{
            "minimum_candidate_weight_ppm":500000,"minimum_margin_milli":100,"evidence_window_bytes":128,
            "max_evidence_windows":31,"max_evidence_spans":31},
        "expected_input_sha256":Sha256Digest::of_bytes(&bytes),"splits":splits});
    native_faithfulness::Dataset::parse(&canonjson::canonical_bytes(&suite).unwrap(), &bytes, Limits::default()).unwrap()
}
#[test]
fn replay_preflight_counts_sources_and_both_arms_before_opening_any_model() {
    let dataset = population(); assert_eq!(dataset.validated_rows(), 12);
    assert_eq!(dataset.replay_len(), 10); assert_eq!(dataset.locked_test_sources(), 5);
    let key = CommitmentKey::new("fixture", [0x42; 32]).unwrap(); let options = command(false, &[]).args.statistics.options();
    let mut limits = Limits::default(); limits.max_bootstrap_draws = 1999;
    assert!(preflight_replay(&dataset, &key, "fixture", options, limits, false, 32).is_ok());
    assert_eq!(preflight_replay(&dataset, &key, "fixture", options, limits, true, 32).unwrap_err().code,
        ErrorCode::AdmissionOrResourceLimit);
    limits.max_bootstrap_draws = 2000;
    assert!(preflight_replay(&dataset, &key, "fixture", options, limits, true, 32).is_ok());
    assert!(preflight_replay(&dataset, &key, "private invalid\nnamespace", options, limits, false, 32).is_err());
    assert!(preflight_replay(&dataset, &key, "fixture", options, limits, false, 0).is_err());
}
#[test]
fn judge_failures_preserve_cancel_work_identity_and_resource_categories() {
    use crate::{hosted::{HostedError, Int4HostError}, native_engine::{decode::DecodeCancellationKind,
        strict_int4::{StrictInt4Error, scoring::Int4ScoringError}},
        tasks::judge::{JudgeError, grouped_int4::Int4JudgeError}};
    for (cause, expected) in [
        (Int4JudgeError::Cancelled(DecodeCancellationKind::User), ErrorCode::Cancelled),
        (Int4JudgeError::Cancelled(DecodeCancellationKind::Deadline), ErrorCode::BudgetOrTimeout),
        (Int4JudgeError::WorkBudget, ErrorCode::BudgetOrTimeout),
        (Int4JudgeError::Task(JudgeError::Limit("PRIVATE-axis")), ErrorCode::BudgetOrTimeout),
        (Int4JudgeError::Task(JudgeError::AllocationRefused), ErrorCode::AdmissionOrResourceLimit),
        (Int4JudgeError::Scoring(Int4ScoringError::Native(StrictInt4Error::Allocation)), ErrorCode::AdmissionOrResourceLimit),
        (Int4JudgeError::Scoring(Int4ScoringError::Native(StrictInt4Error::Cancelled(DecodeCancellationKind::Shutdown))), ErrorCode::Cancelled),
        (Int4JudgeError::Identity, ErrorCode::ArtifactIntegrityOrFormatOrVersion),
        (Int4JudgeError::Accounting, ErrorCode::ArtifactIntegrityOrFormatOrVersion),
        (Int4JudgeError::Task(JudgeError::InvalidScores), ErrorCode::StructuredTaskNoResult),
    ] {
        let failure = host_failure(HostedError::Int4(Int4HostError::Judge(cause)));
        assert_eq!(failure.code, expected); assert!(!failure.reason.contains("PRIVATE"));
    }
}
