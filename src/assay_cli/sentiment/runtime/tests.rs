use super::*;
use super::super::tests::{command, NoRead, NoWrite};
use super::super::super::native::MIB;

#[test]
fn memory_peak_admits_two_resident_models_one_engine_and_all_reports_before_io() {
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
fn population() -> native_sentiment::Dataset {
    let mut bytes = Vec::new(); let mut splits = serde_json::Map::new();
    for (split, count) in [("development", 1), ("calibration", 1), ("locked_test", 5)] {
        let mut ids = Vec::new();
        for index in 0..count {
            let id = format!("{split}-{index}"); ids.push(id.clone());
            bytes.extend(canonjson::canonical_bytes(&serde_json::json!({"schema_version":1,"id":id,"split":split,
                "text":format!("Exact owned {split} source {index}."),
                "gold":[{"axis":"valence","value_milli":133},{"axis":"arousal","value_milli":-250}]})).unwrap()); bytes.push(b'\n');
        }
        splits.insert(split.to_owned(), serde_json::json!(ids));
    }
    let suite = serde_json::json!({"schema_version":1,"task":"sentiment","dataset_id":"preflight",
        "dataset_version":"1","named_population":"owned fixture","axes":["valence","arousal"],
        "policy":{"minimum_peak_weight_ppm":0,"maximum_normalized_entropy_ppm":1000000},
        "expected_input_sha256":Sha256Digest::of_bytes(&bytes),"splits":splits});
    native_sentiment::Dataset::parse(&canonjson::canonical_bytes(&suite).unwrap(), &bytes, Limits::default()).unwrap()
}
#[test]
fn complete_document_axis_and_arm_bootstrap_allowance_is_checked_before_snapshot_io() {
    let dataset = population(); assert_eq!(dataset.validated_rows(), 7); assert_eq!(dataset.replay_len(), 5);
    let key = CommitmentKey::new("fixture", [0x42; 32]).unwrap(); let options = command(false, &[]).args.statistics.options();
    let mut limits = Limits::default(); limits.max_bootstrap_draws = 3999;
    assert!(preflight_replay(&dataset, &key, "fixture", options, limits, false).is_ok());
    assert_eq!(preflight_replay(&dataset, &key, "fixture", options, limits, true).unwrap_err().code,
        ErrorCode::AdmissionOrResourceLimit);
    limits.max_bootstrap_draws = 4000;
    assert!(preflight_replay(&dataset, &key, "fixture", options, limits, true).is_ok());
    assert!(preflight_replay(&dataset, &key, "private invalid\nnamespace", options, limits, false).is_err());
}
#[test]
fn sentiment_failures_keep_nested_cancellation_work_resource_and_identity_categories() {
    use crate::{hosted::{HostedError, Int4HostError}, native_engine::{decode::DecodeCancellationKind,
        strict_int4::{StrictInt4Error, scoring::Int4ScoringError}},
        tasks::sentiment::{SentimentError, SentimentPlanningError, grouped_int4::Int4SentimentError}};
    for (cause, expected) in [
        (Int4SentimentError::Cancelled(DecodeCancellationKind::User), ErrorCode::Cancelled),
        (Int4SentimentError::Planning(SentimentPlanningError::Cancelled(DecodeCancellationKind::Deadline)), ErrorCode::BudgetOrTimeout),
        (Int4SentimentError::WorkBudget, ErrorCode::BudgetOrTimeout),
        (Int4SentimentError::Task(SentimentError::LimitExceeded("PRIVATE-axis")), ErrorCode::BudgetOrTimeout),
        (Int4SentimentError::Planning(SentimentPlanningError::Task(SentimentError::LimitExceeded("PRIVATE-axis"))), ErrorCode::BudgetOrTimeout),
        (Int4SentimentError::Planning(SentimentPlanningError::Budget), ErrorCode::AdmissionOrResourceLimit),
        (Int4SentimentError::Task(SentimentError::AllocationRefused), ErrorCode::AdmissionOrResourceLimit),
        (Int4SentimentError::Scoring(Int4ScoringError::Native(StrictInt4Error::Allocation)), ErrorCode::AdmissionOrResourceLimit),
        (Int4SentimentError::Scoring(Int4ScoringError::Native(StrictInt4Error::Cancelled(DecodeCancellationKind::Shutdown))), ErrorCode::Cancelled),
        (Int4SentimentError::Identity, ErrorCode::ArtifactIntegrityOrFormatOrVersion),
        (Int4SentimentError::Planning(SentimentPlanningError::Identity), ErrorCode::ArtifactIntegrityOrFormatOrVersion),
        (Int4SentimentError::Accounting, ErrorCode::ArtifactIntegrityOrFormatOrVersion),
        (Int4SentimentError::Task(SentimentError::InvalidScores), ErrorCode::StructuredTaskNoResult),
    ] {
        let failure = host_failure(HostedError::Int4(Int4HostError::Sentiment(cause)));
        assert_eq!(failure.code, expected); assert!(!failure.reason.contains("PRIVATE"));
    }
}
