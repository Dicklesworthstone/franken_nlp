use super::*;
use super::super::tests::{application, calibration_command, command};

#[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
mod model;

#[test]
fn native_peak_includes_raw_inputs_all_resident_models_and_shared_host_quote() {
    let single = command(false, &[]); let limits = single.validate().unwrap();
    let single_plan = Plan::new(&single, limits).unwrap();
    let mut paired = command(true, &[]); paired.comparison.as_mut().unwrap().baseline_group_width = 127;
    let paired_plan = Plan::new(&paired, paired.validate().unwrap()).unwrap();
    let baseline = paired_plan.baseline.as_ref().unwrap();
    let host = paired_plan.host.requirements(&[128, 127], limits).unwrap();
    let execution = host.native.kv_bytes + host.workspace_bytes + host.input_bytes + host.output_reservation_bytes;
    assert_eq!(paired_plan.minimum_memory_bytes, paired_plan.raw_staging_bytes
        + paired_plan.candidate.memory.resident_bytes + baseline.memory.resident_bytes
        + execution.max(paired_plan.candidate.memory.staging_bytes.max(baseline.memory.staging_bytes)) + CLEANUP_BYTES);
    assert!(paired_plan.minimum_memory_bytes >= single_plan.minimum_memory_bytes + baseline.memory.resident_bytes);
    assert_eq!(single_plan.host.task, paired_plan.host.task);
    assert_eq!(single_plan.host.max_model_work, paired_plan.host.max_model_work);
    assert_eq!(paired_plan.host.native.run.max_checkpoints, paired.args.max_checkpoints);
    // Two models share the native engine peak, but each resident has its own charge.
    assert_eq!(host.native.kv_bytes, single_plan.host.task.max_kv_bytes);
}

#[test]
fn rounded_memory_authority_is_checked_before_any_io_and_overflow_refuses() {
    let mut options = command(true, &[]); let limits = options.validate().unwrap();
    let required = Plan::new(&options, limits).unwrap().minimum_memory_bytes;
    options.args.memory_mib = Some(required.div_ceil(MIB));
    assert!(Plan::new(&options, limits).is_ok());
    options.args.memory_mib = Some(required.div_ceil(MIB) - 1);
    assert!(matches!(Plan::new(&options, limits), Err(Failure { code: ErrorCode::AdmissionOrResourceLimit, .. })));
    options.args.memory_mib = Some(u64::MAX);
    assert!(matches!(Plan::new(&options, limits), Err(Failure { code: ErrorCode::Usage, .. })));
}

#[test]
fn native_input_and_complete_report_bounds_increase_both_outer_and_host_reservations() {
    let options = command(false, &[]); let limits = options.validate().unwrap();
    let initial = Plan::new(&options, limits).unwrap();
    for field in 0..3 {
        let mut smaller = limits;
        match field { 0 => smaller.max_input_bytes -= 1, 1 => smaller.max_rows -= 1,
            _ => smaller.max_output_bytes -= 1 }
        let quote = Plan::new(&options, smaller).unwrap();
        assert!(quote.raw_staging_bytes < initial.raw_staging_bytes);
        assert!(quote.minimum_memory_bytes < initial.minimum_memory_bytes);
        assert_eq!(quote.candidate.memory, initial.candidate.memory);
    }
}

#[test]
fn label_cap_prices_gold_vectors_and_complete_native_heads_before_input_io() {
    let binary = command(false, &[]); let limits = binary.validate().unwrap();
    assert_eq!(binary.args.max_labels, 2);
    let initial = Plan::new(&binary, limits).unwrap();
    let expanded = command(false, &["--max-labels", "3"]);
    let expanded = Plan::new(&expanded, expanded.validate().unwrap()).unwrap();
    assert_eq!(expanded.host.max_labels, 3);
    assert_eq!(expanded.raw_staging_bytes - initial.raw_staging_bytes,
        limits.max_rows as u64 * 64 + 512);
    assert!(expanded.minimum_memory_bytes > initial.minimum_memory_bytes);
    assert_eq!(expanded.candidate.memory, initial.candidate.memory);
    assert_eq!(expanded.host.task, initial.host.task);
    assert_eq!(expanded.host.max_model_work, initial.host.max_model_work);
    for cap in ["0", "129"] {
        assert_eq!(command(false, &["--max-labels", cap]).validate().unwrap_err().code, ErrorCode::Usage);
    }
}

#[test]
fn native_fit_and_each_authenticated_artifact_are_in_the_complete_pre_io_peak() {
    let raw = command(false, &[]); let limits = raw.validate().unwrap();
    let raw_plan = Plan::new(&raw, limits).unwrap();
    let fit = calibration_command(&[]); let fit_plan = Plan::new(&fit, fit.validate().unwrap()).unwrap();
    assert!(fit_plan.minimum_memory_bytes > raw_plan.minimum_memory_bytes);
    assert_eq!(fit_plan.raw_staging_bytes, raw_plan.raw_staging_bytes);
    assert_eq!(fit_plan.candidate.memory, raw_plan.candidate.memory);
    let mut applied = command(false, &[]); applied.application = Some(application());
    let applied_plan = Plan::new(&applied, applied.validate().unwrap()).unwrap();
    assert!(applied_plan.raw_staging_bytes > raw_plan.raw_staging_bytes);
    assert!(applied_plan.minimum_memory_bytes > raw_plan.minimum_memory_bytes);
    let mut paired = command(true, &[]); paired.application = Some(application());
    let one = Plan::new(&paired, paired.validate().unwrap()).unwrap();
    let comparison = paired.comparison.as_mut().unwrap();
    comparison.baseline_calibration = Some(PathBuf::from("absent-baseline-calibration"));
    comparison.expected_baseline_calibration_sha256 = Some(super::super::tests::BASELINE_PIN.to_owned());
    comparison.baseline_as_of = Some("2026-06-01".to_owned());
    comparison.baseline_shift_assessment = Some("in-distribution".to_owned());
    let both = Plan::new(&paired, paired.validate().unwrap()).unwrap();
    assert!(both.minimum_memory_bytes > one.minimum_memory_bytes); assert!(both.raw_staging_bytes > one.raw_staging_bytes);
    paired.args.memory_mib = Some(both.minimum_memory_bytes.div_ceil(MIB) - 1);
    assert!(matches!(Plan::new(&paired, limits), Err(Failure { code: ErrorCode::AdmissionOrResourceLimit, .. })));
}

#[test]
fn calibration_transport_requires_exact_pin_native_fit_envelope_and_authenticated_inner_report() {
    let key = CommitmentKey::new("fixture", [0x42; 32]).unwrap(); let limits = Limits::default();
    let context = || assay::calibrate::ApplyOptions {
        observed_date: assay::calibrate::parse_date("2026-06-01").unwrap(),
        shift: crate::calibration::ShiftAssessment::InDistribution,
    };
    let valid_transport = encode(NATIVE_PROTOCOL, "calibrate", &serde_json::json!({"body":{},"seal":{}}), limits).unwrap();
    let envelope: serde_json::Value = serde_json::from_slice(&valid_transport).unwrap();
    for field in 0..8 {
        let mut value = envelope.clone();
        match field {
            0 => value["protocol"] = serde_json::json!("fnlp.assay.scored"),
            1 => value["operation"] = serde_json::json!("eval"),
            2 => value["evidence"] = serde_json::json!("provided_observations"),
            3 => value["activation_authority"] = serde_json::json!(true),
            4 => value["schema_version"] = serde_json::json!(2),
            5 => { value.as_object_mut().unwrap().remove("activation_authority"); }
            6 => value["unknown"] = serde_json::json!("PRIVATE-untrusted"),
            _ => {} // A correctly framed but unauthenticated inner report also refuses.
        }
        let bytes = canonjson::canonical_bytes(&value).unwrap();
        let Err(error) = decode_application(&bytes, Sha256Digest::of_bytes(&bytes), &key, "fixture", context(), limits)
            else { panic!("untrusted calibration transport became a prepared application"); };
        assert_eq!(error.code, ErrorCode::ArtifactIntegrityOrFormatOrVersion);
        assert!(!error.reason.contains("PRIVATE"));
    }
    let duplicate = b"{\"schema_version\":1,\"schema_version\":1}";
    assert!(decode_application(duplicate, Sha256Digest::of_bytes(duplicate), &key, "fixture", context(), limits).is_err());
    let Err(error) = decode_application(b"not-json", Sha256Digest::of_bytes(b"independent-pin"), &key, "fixture", context(), limits)
        else { panic!("mismatched pin accepted"); };
    assert_eq!(error.reason, AssayError::InputDigest.code());
    let mut cap = limits; cap.max_output_bytes = valid_transport.len() - 1;
    let Err(error) = decode_application(&valid_transport, Sha256Digest::of_bytes(&valid_transport), &key, "fixture", context(), cap)
        else { panic!("oversized calibration accepted"); };
    assert_eq!(error.code, ErrorCode::AdmissionOrResourceLimit);
}

#[test]
fn actual_host_causes_keep_budget_integrity_and_cancellation_exit_categories() {
    let work = host_failure(HostedError::Int4(Int4HostError::Classification(Int4ClassificationError::WorkBudget)));
    assert_eq!(work.code, ErrorCode::BudgetOrTimeout);
    let allocation = host_failure(HostedError::Int4(Int4HostError::Classification(
        Int4ClassificationError::Native(StrictInt4Error::Allocation))));
    assert_eq!(allocation.code, ErrorCode::AdmissionOrResourceLimit);
    let digest = host_failure(HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Integrity)));
    assert_eq!(digest.code, ErrorCode::ArtifactIntegrityOrFormatOrVersion);
    let input = host_failure(HostedError::Assay(AssayError::InputDigest));
    assert_eq!(input.code, ErrorCode::ArtifactIntegrityOrFormatOrVersion);
    for (kind, expected) in [(DecodeCancellationKind::Deadline, ErrorCode::BudgetOrTimeout),
        (DecodeCancellationKind::PollQuota, ErrorCode::BudgetOrTimeout),
        (DecodeCancellationKind::User, ErrorCode::Cancelled),
        (DecodeCancellationKind::ResourceUnavailable, ErrorCode::AdmissionOrResourceLimit),
        (DecodeCancellationKind::RaceLost, ErrorCode::Generic)] {
        let stopped = HostedError::Stopped { stop: crate::hosted::RunStop { kind,
            runtime_reason: None, runtime_checkpoint_error: None }, task_error: None };
        assert_eq!(host_failure(stopped).code, expected);
    }
}

#[test]
fn nested_planner_and_scorer_causes_preserve_budget_allocation_and_cancellation_categories() {
    use ClassificationPlanningError as Planning;
    use Int4ClassificationError as Classification;
    use Int4ScoringError as Scoring;
    let cases = [
        (Classification::Planning(Planning::WorkBudget), ErrorCode::BudgetOrTimeout),
        (Classification::Planning(Planning::AllocationRefused), ErrorCode::AdmissionOrResourceLimit),
        (Classification::Planning(Planning::Task(ClassificationError::AllocationRefused)), ErrorCode::AdmissionOrResourceLimit),
        (Classification::Planning(Planning::Task(ClassificationError::Scoring(ScoringError::AllocationRefused))),
            ErrorCode::AdmissionOrResourceLimit),
        (Classification::Scoring(Scoring::Allocation), ErrorCode::AdmissionOrResourceLimit),
        (Classification::Scoring(Scoring::Scoring(ScoringError::AllocationRefused)), ErrorCode::AdmissionOrResourceLimit),
        (Classification::Scoring(Scoring::Native(StrictInt4Error::Work)), ErrorCode::BudgetOrTimeout),
        (Classification::Scoring(Scoring::Native(StrictInt4Error::Allocation)), ErrorCode::AdmissionOrResourceLimit),
        (Classification::Scoring(Scoring::Native(StrictInt4Error::Memory)), ErrorCode::AdmissionOrResourceLimit),
        (Classification::Scoring(Scoring::Native(StrictInt4Error::Context)), ErrorCode::AdmissionOrResourceLimit),
        (Classification::Scoring(Scoring::Native(StrictInt4Error::Cancelled(DecodeCancellationKind::Deadline))),
            ErrorCode::BudgetOrTimeout),
        (Classification::Scoring(Scoring::Native(StrictInt4Error::Cancelled(DecodeCancellationKind::User))), ErrorCode::Cancelled),
        (Classification::Planning(Planning::Cancelled(DecodeCancellationKind::ResourceUnavailable)),
            ErrorCode::AdmissionOrResourceLimit),
    ];
    for (cause, expected) in cases {
        assert_eq!(host_failure(HostedError::Int4(Int4HostError::Classification(cause))).code, expected);
    }
}

fn preflight_population() -> native_classify::Dataset {
    let mut input = Vec::new(); let mut splits = serde_json::Map::new();
    for (split, count) in [("development", 1), ("calibration", 1), ("locked_test", 5)] {
        let mut ids = Vec::new();
        for index in 0..count {
            let id = format!("{split}-{index}"); ids.push(id.clone());
            input.extend(canonjson::canonical_bytes(&serde_json::json!({"schema_version":1,
                "id":id,"split":split,"text":format!("Owned preflight document {split} {index}"),
                "positive":index % 2 == 0})).unwrap()); input.push(b'\n');
        }
        splits.insert(split.to_owned(), serde_json::json!(ids));
    }
    let suite = serde_json::json!({"schema_version":1,"task":"binary_classification","dataset_id":"preflight",
        "dataset_version":"1","named_population":"owned fixture","positive_label":{"id":"yes","description":""},
        "negative_label":{"id":"no","description":""},"expected_input_sha256":Sha256Digest::of_bytes(&input),"splits":splits});
    native_classify::Dataset::parse(&canonjson::canonical_bytes(&suite).unwrap(), &input, Limits::default()).unwrap()
}

fn preflight_label_population(task: &str) -> native_classify::Dataset {
    let mut input = Vec::new(); let mut splits = serde_json::Map::new();
    for (split, count) in [("development", 1), ("calibration", 1), ("locked_test", 5)] {
        let mut ids = Vec::new();
        for index in 0..count {
            let id = format!("{split}-{index}"); ids.push(id.clone());
            let mut row = serde_json::json!({"schema_version":1,"id":id,"split":split,
                "text":format!("Owned label-set preflight document {split} {index}")});
            if task == "multiclass_classification" { row["label"] = serde_json::json!("alpha"); }
            else { row["labels"] = serde_json::json!(if index % 2 == 0 { vec!["alpha", "gamma"] } else { vec![] }); }
            input.extend(canonjson::canonical_bytes(&row).unwrap()); input.push(b'\n');
        }
        splits.insert(split.to_owned(), serde_json::json!(ids));
    }
    let suite = serde_json::json!({"schema_version":1,"task":task,"dataset_id":"preflight-labels",
        "dataset_version":"1","named_population":"owned label-set fixture",
        "labels":[{"id":"gamma","description":""},{"id":"alpha","description":""},{"id":"beta","description":""}],
        "expected_input_sha256":Sha256Digest::of_bytes(&input),"splits":splits});
    native_classify::Dataset::parse(&canonjson::canonical_bytes(&suite).unwrap(), &input, Limits::default()).unwrap()
}

#[test]
fn task_preflight_admits_explicit_label_caps_and_refuses_binary_calibration_reuse() {
    for task in ["multiclass_classification", "multilabel_classification"] {
        let dataset = preflight_label_population(task);
        assert_eq!(preflight_task(&dataset, &command(false, &[])).unwrap_err().code,
            ErrorCode::AdmissionOrResourceLimit);
        for paired in [false, true] {
            assert!(preflight_task(&dataset, &command(paired, &["--max-labels", "3"])).is_ok());
        }
        let fit = calibration_command(&["--max-labels", "3"]);
        let mut apply = command(false, &["--max-labels", "3"]); apply.application = Some(application());
        let baseline = command(true, &["--max-labels", "3", "--baseline-calibration", "must-not-open-artifact",
            "--expected-baseline-calibration-sha256", super::super::tests::BASELINE_PIN,
            "--baseline-as-of", "2026-06-01", "--baseline-shift-assessment", "in-distribution"]);
        for invalid in [fit, apply, baseline] {
            invalid.validate().unwrap();
            let error = preflight_task(&dataset, &invalid).unwrap_err();
            assert_eq!(error.code, ErrorCode::Usage);
            assert_eq!(error.reason, "native_calibration_requires_binary_task");
        }
    }
    assert!(preflight_task(&preflight_population(), &calibration_command(&[])).is_ok());
}

#[test]
fn label_reliability_bootstrap_admission_counts_documents_without_label_pseudoreplication() {
    let key = CommitmentKey::new("fixture", [0x42; 32]).unwrap();
    let mut options = EvalOptions::default(); options.minimum_samples = 5; options.bootstrap.resamples = 16;
    for task in ["multiclass_classification", "multilabel_classification"] {
        let dataset = preflight_label_population(task);
        let mut limits = Limits::default(); limits.max_bootstrap_draws = 160;
        assert!(preflight_replay(&dataset, &key, "fixture", options, limits, true).is_ok());
        limits.max_bootstrap_draws = 159;
        assert_eq!(preflight_replay(&dataset, &key, "fixture", options, limits, true).unwrap_err().code,
            ErrorCode::AdmissionOrResourceLimit);
    }
}

#[test]
fn parsed_population_preflight_prices_combined_bootstrap_draws_before_snapshot_loads() {
    let dataset = preflight_population(); let key = CommitmentKey::new("fixture", [0x42; 32]).unwrap();
    let mut options = EvalOptions::default(); options.minimum_samples = 5; options.bootstrap.resamples = 16;
    let mut limits = Limits::default(); limits.max_bootstrap_draws = 160;
    assert!(preflight_replay(&dataset, &key, "fixture", options, limits, true).is_ok());
    limits.max_bootstrap_draws -= 1;
    assert_eq!(preflight_replay(&dataset, &key, "fixture", options, limits, true).unwrap_err().code,
        ErrorCode::AdmissionOrResourceLimit);
    limits.max_bootstrap_draws = 80;
    assert!(preflight_replay(&dataset, &key, "fixture", options, limits, false).is_ok());
    limits.max_bootstrap_draws -= 1;
    assert_eq!(preflight_replay(&dataset, &key, "fixture", options, limits, false).unwrap_err().code,
        ErrorCode::AdmissionOrResourceLimit);
}

#[test]
fn receipt_namespace_preflight_uses_the_actual_commitment_contract_without_a_model() {
    let dataset = preflight_population(); let key = CommitmentKey::new("fixture", [0x42; 32]).unwrap();
    for namespace in ["", "_leading", "private space", "private\ncontrol", "non-ascii-雪"] {
        assert_eq!(preflight_replay(&dataset, &key, namespace, EvalOptions::default(), Limits::default(), false)
            .unwrap_err().code, AssayError::Commitment.exit_code());
    }
    let mut namespace = "a".repeat(256);
    assert!(preflight_replay(&dataset, &key, &namespace, EvalOptions::default(), Limits::default(), false).is_ok());
    namespace.push('a');
    assert_eq!(preflight_replay(&dataset, &key, &namespace, EvalOptions::default(), Limits::default(), false)
        .unwrap_err().code, AssayError::Commitment.exit_code());
}

#[test]
fn native_report_framing_is_complete_bounded_and_not_supplied_observation_evidence() {
    let report = serde_json::json!({"classification":{"examples":20},"activation_authority":false});
    let limits = Limits::default(); let bytes = encode(NATIVE_PROTOCOL, "eval", &report, limits).unwrap();
    assert_eq!(bytes.last(), Some(&b'\n'));
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["protocol"], NATIVE_PROTOCOL); assert_eq!(value["evidence"], "native_model_replay");
    assert_eq!(value["activation_authority"], false);
    let mut exact = limits; exact.max_output_bytes = bytes.len();
    assert_eq!(encode(NATIVE_PROTOCOL, "eval", &report, exact).unwrap(), bytes);
    exact.max_output_bytes -= 1;
    assert!(encode(NATIVE_PROTOCOL, "eval", &report, exact).is_err());
}

#[test]
fn failed_report_flush_returns_failure_without_retry_or_another_frame() {
    #[derive(Default)] struct RejectFlush { bytes: Vec<u8>, writes: usize, flushes: usize }
    impl Write for RejectFlush {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.writes += 1; self.bytes.extend_from_slice(bytes); Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.flushes += 1; Err(std::io::ErrorKind::BrokenPipe.into())
        }
    }
    let mut output = RejectFlush::default();
    assert!(publish(None, b"{\"complete\":true}\n", &mut output).is_err());
    assert_eq!(output.bytes, b"{\"complete\":true}\n");
    assert_eq!(output.writes, 1); assert_eq!(output.flushes, 1);
}
