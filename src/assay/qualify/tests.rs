//! Owned paired score fixtures; no native replay or activation qualification.
use super::*;
use crate::{assay::{Observation, SuiteManifest, Split, tests as fixtures},
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    receipt::CommitmentKey};

const NAMESPACE: &str = "paired-fixture";
const HOST: &str = "caller-declared-fixture-host";

fn gates() -> Gates {
    parse_gates(br#"{"schema_version":1,"max_brier_increase":0.0,"max_ece_increase":0.0,"max_selective_risk":0.0,"min_acceptance_rate":0.4,"minimum_accepted":2}"#,
        Limits::default()).unwrap()
}
fn rebuild(dataset: &Dataset, rows: &[Observation], change: impl FnOnce(&mut SuiteManifest)) -> Dataset {
    let input = fixtures::encode_rows(rows);
    let mut manifest = dataset.manifest().clone();
    manifest.expected_input_sha256 = Sha256Digest::of_bytes(&input); change(&mut manifest);
    Dataset::parse(&crate::canonjson::canonical_bytes(&manifest).unwrap(), &input, Limits::default()).unwrap()
}
fn scored(dataset: &Dataset, score: impl Fn(&Observation) -> f64) -> Dataset {
    let mut rows = dataset.rows().to_vec();
    for row in &mut rows { if row.split == Split::LockedTest { row.probability = score(row); } }
    rebuild(dataset, &rows, |_| {})
}
fn run(candidate: &Dataset, baseline: &Dataset, policy: Gates, limits: Limits)
    -> Result<QualificationReport, AssayError> {
    compare(candidate, baseline, &fixtures::key(), NAMESPACE, HOST, policy, fixtures::options(), limits)
}
fn wire(report: &QualificationReport) -> serde_json::Value {
    serde_json::to_value(report).unwrap()
}

#[test]
fn paired_gates_measure_identical_better_and_worse_observations_without_native_authority() {
    let baseline = fixtures::dataset();
    let identical = wire(&run(&baseline, &baseline, gates(), Limits::default()).unwrap());
    assert_eq!(identical["decision"]["status"], "passed");
    assert_eq!(identical["decision"]["paired"]["rows"], 8);
    assert_eq!(identical["decision"]["paired"]["equal"], 8);
    assert_eq!(identical["decision"]["paired"]["mean_brier_change"], 0.0);
    assert_eq!(identical["evidence"], "provided_observations");
    assert_eq!(identical["activation_authority"], false);
    assert_eq!(identical["receipt_grade"], "audit_only");
    assert_eq!(identical["gate_basis"], "point_estimates");
    assert_eq!(identical["host_scope_evidence"], "caller_declared");

    let improved = scored(&baseline, |row| if row.positive { 0.9 } else { 0.1 });
    let improved = wire(&run(&improved, &baseline, gates(), Limits::default()).unwrap());
    assert_eq!(improved["decision"]["status"], "passed");
    let paired = &improved["decision"]["paired"];
    assert_eq!(paired["candidate_better"], 8); assert_eq!(paired["candidate_worse"], 0);
    assert!((paired["mean_brier_change"].as_f64().unwrap() + 0.03).abs() < 1e-12);
    assert!(improved["decision"]["comparisons"].as_array().unwrap().iter().all(|row| row["passed"] == true));

    let worsened = scored(&baseline, |row| if row.positive { 0.2 } else { 0.8 });
    let worsened = wire(&run(&worsened, &baseline, gates(), Limits::default()).unwrap());
    assert_eq!(worsened["decision"]["status"], "rejected");
    assert_eq!(worsened["decision"]["paired"]["candidate_worse"], 8);
    assert!((worsened["decision"]["paired"]["mean_brier_change"].as_f64().unwrap() - 0.6).abs() < 1e-12);
    assert!(worsened["decision"]["comparisons"].as_array().unwrap().iter().any(|row| row["passed"] == false));
}

#[test]
fn zero_acceptance_is_rejected_without_inventing_a_selective_risk() {
    let baseline = fixtures::dataset(); let candidate = scored(&baseline, |_| 0.1);
    let mut policy = gates(); policy.max_brier_increase = 1.0; policy.max_ece_increase = 1.0;
    policy.max_selective_risk = 1.0; policy.min_acceptance_rate = 0.0; policy.minimum_accepted = 1;
    let report = wire(&run(&candidate, &baseline, policy, Limits::default()).unwrap());
    assert_eq!(report["decision"]["status"], "rejected");
    assert_eq!(report["candidate"]["measurement"]["metrics"]["accepted"], 0);
    assert!(report["candidate"]["measurement"]["metrics"]["selective_risk"].is_null());
    assert!(report["candidate"]["measurement"]["metrics"]["selective_risk_interval"].is_null());
    assert!(report["decision"]["comparisons"].as_array().unwrap().iter()
        .any(|row| row["observed"].is_null() && row["passed"] == false));
}

#[test]
fn negative_increase_gates_require_an_actual_paired_improvement() {
    let baseline = fixtures::dataset();
    let candidate = scored(&baseline, |row| if row.positive { 0.9 } else { 0.1 });
    let mut policy = gates(); policy.max_brier_increase = -0.02; policy.max_ece_increase = -0.05;
    assert_eq!(wire(&run(&candidate, &baseline, policy, Limits::default()).unwrap())["decision"]["status"], "passed");
    assert_eq!(wire(&run(&baseline, &baseline, policy, Limits::default()).unwrap())["decision"]["status"], "rejected");
}

#[test]
fn a_tiny_paired_population_has_no_numeric_gate_verdict_or_paired_summary() {
    let (suite, observations) = fixtures::fixture(5, 4);
    let tiny = Dataset::parse(&suite, &observations, Limits::default()).unwrap();
    let report = wire(&run(&tiny, &tiny, gates(), Limits::default()).unwrap());
    assert_eq!(report["decision"]["status"], "insufficient_data");
    assert_eq!(report["candidate"]["measurement"]["status"], "insufficient_data");
    assert_eq!(report["baseline"]["measurement"]["status"], "insufficient_data");
    assert!(report["decision"].get("comparisons").is_none());
    assert!(report["decision"].get("paired").is_none());
    assert_eq!(report["activation_authority"], false);
}

#[test]
fn gates_reject_ambiguous_missing_unknown_nonfinite_and_out_of_domain_values() {
    let valid = crate::canonjson::canonical_bytes(&gates()).unwrap();
    let text = std::str::from_utf8(&valid).unwrap();
    for prefix in [r#"{"schema_version":1,"#, r#"{"schema\u005fversion":1,"#, r#"{"unknown":0,"#] {
        let malformed = format!("{prefix}{}", &text[1..]);
        assert!(parse_gates(malformed.as_bytes(), Limits::default()).is_err());
    }
    for (name, value) in [("schema_version", serde_json::json!(2)),
        ("max_brier_increase", serde_json::json!(-1.01)), ("max_brier_increase", serde_json::json!(1.01)),
        ("max_ece_increase", serde_json::json!(-1.01)), ("max_ece_increase", serde_json::json!(1.01)),
        ("max_selective_risk", serde_json::json!(-0.01)), ("max_selective_risk", serde_json::json!(1.01)),
        ("min_acceptance_rate", serde_json::json!(-0.01)), ("min_acceptance_rate", serde_json::json!(1.01)),
        ("minimum_accepted", serde_json::json!(0)), ("minimum_accepted", serde_json::json!(100_001))] {
        let mut changed = crate::canonjson::parse_str(text).unwrap(); changed[name] = value;
        assert!(parse_gates(&serde_json::to_vec(&changed).unwrap(), Limits::default()).is_err(), "{name}");
    }
    for name in ["schema_version", "max_brier_increase", "max_ece_increase", "max_selective_risk", "min_acceptance_rate", "minimum_accepted"] {
        let mut changed = crate::canonjson::parse_str(text).unwrap(); changed.as_object_mut().unwrap().remove(name);
        assert!(parse_gates(&serde_json::to_vec(&changed).unwrap(), Limits::default()).is_err());
    }
    let dataset = fixtures::dataset();
    let mut forged = gates(); forged.max_brier_increase = f64::NAN;
    assert!(run(&dataset, &dataset, forged, Limits::default()).is_err());
    let mut forged = gates(); forged.max_ece_increase = f64::INFINITY;
    assert!(run(&dataset, &dataset, forged, Limits::default()).is_err());
    let mut limited = Limits::default(); limited.max_manifest_bytes = valid.len() - 1;
    assert!(parse_gates(&valid, limited).is_err());
}

#[test]
fn every_non_model_execution_semantic_must_match_exactly() {
    let baseline = fixtures::dataset();
    for axis in 0..14 {
        let changed = rebuild(&baseline, baseline.rows(), |suite| {
            let id = &mut suite.execution_identity; let different = Sha256Digest::of_bytes(b"changed private semantic");
            match axis {
                0 => id.source_revision = "different-source".to_owned(),
                1 => id.tokenizer_digest = different,
                2 => id.template_digest = different,
                3 => id.task_spec = "different-task".to_owned(),
                4 => id.taskir_digest = different,
                5 => id.prompt_digest = different,
                6 => id.grammar_compiler_version = "different-grammar".to_owned(),
                7 => id.schema_digest = different,
                8 => id.kv_dtype = "different-kv".to_owned(),
                9 => id.sampler_version = "different-sampler".to_owned(),
                10 => id.thinking_mode = ThinkingMode::Enabled,
                11 => id.tool_mode = ToolMode::Xml,
                12 => id.calibration_digest = different,
                _ => id.decision_policy_digest = different,
            }
        });
        assert!(matches!(run(&changed, &baseline, gates(), Limits::default()), Err(AssayError::Scope)), "axis {axis}");
    }
}

#[test]
fn candidate_model_recipe_packing_and_numerics_changes_are_compared_instead_of_rejected() {
    let baseline = fixtures::dataset();
    for axis in 0..6 {
        let changed = rebuild(&baseline, baseline.rows(), |suite| {
            let id = &mut suite.execution_identity;
            match axis {
                0 => id.logical_model_digest = Sha256Digest::of_bytes(b"candidate weights"),
                1 => id.artifact_format = "candidate-format-v2".to_owned(),
                2 => id.quant_recipe = "candidate-recipe-v2".to_owned(),
                3 => id.packing_set_digest = Sha256Digest::of_bytes(b"candidate packing"),
                4 => { id.numerics_profile = NumericsProfile::Fast { version: 1 };
                    id.host_class = Some("declared-host".to_owned()); id.compiler_identity = Some("declared-compiler".to_owned()); }
                _ => id.backend_semantic_version = "candidate-backend-v2".to_owned(),
            }
        });
        let report = wire(&run(&changed, &baseline, gates(), Limits::default()).unwrap());
        assert_eq!(report["decision"]["status"], "passed", "axis {axis}");
    }
}

#[test]
fn paired_comparison_refuses_foreign_population_labels_gold_units_ids_and_splits() {
    let baseline = fixtures::dataset();
    for axis in 0..7 {
        let mut rows = baseline.rows().to_vec();
        match axis {
            0 => rows[0].positive = !rows[0].positive,
            1 => rows[0].unit_sha256 = Sha256Digest::of_bytes(b"another evaluation unit"),
            2 => rows[0].id = "replacement-private-id".to_owned(),
            3 => { let a = rows.iter().position(|row| row.split == Split::Development).unwrap();
                let b = rows.iter().position(|row| row.split == Split::Calibration).unwrap();
                rows[a].split = Split::Calibration; rows[b].split = Split::Development; }
            _ => (),
        }
        let changed = rebuild(&baseline, &rows, |suite| {
            if axis == 2 || axis == 3 {
                suite.splits.development = rows.iter().filter(|r| r.split == Split::Development).map(|r| r.id.clone()).collect();
                suite.splits.calibration = rows.iter().filter(|r| r.split == Split::Calibration).map(|r| r.id.clone()).collect();
                suite.splits.locked_test = rows.iter().filter(|r| r.split == Split::LockedTest).map(|r| r.id.clone()).collect();
            }
            match axis {
                4 => suite.named_population = "different population".to_owned(),
                5 => suite.positive_label = "different positive event".to_owned(),
                6 => suite.dataset_version = "different-version".to_owned(),
                _ => (),
            }
        });
        assert!(matches!(run(&changed, &baseline, gates(), Limits::default()), Err(AssayError::Scope)), "axis {axis}");
    }
    let mut duplicated = baseline.rows().to_vec(); duplicated[1].unit_sha256 = duplicated[0].unit_sha256;
    let input = fixtures::encode_rows(&duplicated); let mut suite = baseline.manifest().clone();
    suite.expected_input_sha256 = Sha256Digest::of_bytes(&input);
    assert!(matches!(Dataset::parse(&crate::canonjson::canonical_bytes(&suite).unwrap(), &input, Limits::default()),
        Err(AssayError::DuplicateUnit)));
}

#[test]
fn row_order_changes_only_exact_byte_bindings_not_paired_measurements_or_gates() {
    let baseline = fixtures::dataset(); let candidate = scored(&baseline, |row| if row.positive { 0.9 } else { 0.1 });
    let original = wire(&run(&candidate, &baseline, gates(), Limits::default()).unwrap());
    let mut rows = candidate.rows().to_vec(); rows.reverse();
    let reversed = rebuild(&candidate, &rows, |suite| { suite.splits.calibration.reverse(); suite.splits.locked_test.reverse(); });
    let replay = wire(&run(&reversed, &baseline, gates(), Limits::default()).unwrap());
    assert_eq!(original["decision"], replay["decision"]);
    assert_eq!(original["candidate"]["measurement"], replay["candidate"]["measurement"]);
    assert_eq!(original["baseline"]["measurement"], replay["baseline"]["measurement"]);
    assert_ne!(original["candidate"]["bindings"]["input"], replay["candidate"]["bindings"]["input"]);
}

#[test]
fn both_bootstrap_populations_and_the_complete_report_obey_the_limits() {
    let dataset = fixtures::dataset(); let mut limits = Limits::default();
    limits.max_bootstrap_draws = 2 * 8 * fixtures::options().bootstrap.resamples as u64 - 1;
    assert!(matches!(run(&dataset, &dataset, gates(), limits), Err(AssayError::InputLimit)));
    limits.max_bootstrap_draws += 1;
    let report = run(&dataset, &dataset, gates(), limits).unwrap();
    let encoded = crate::assay::render(&report, Limits::default()).unwrap();
    limits.max_output_bytes = encoded.len(); assert!(run(&dataset, &dataset, gates(), limits).is_ok());
    limits.max_output_bytes -= 1;
    assert!(matches!(run(&dataset, &dataset, gates(), limits), Err(AssayError::Output)));
    assert_eq!(encoded.last(), Some(&b'\n'));
}

#[test]
fn qualification_scope_is_declared_and_private_bindings_remain_keyed() {
    let dataset = fixtures::dataset();
    let report = run(&dataset, &dataset, gates(), Limits::default()).unwrap();
    let exported = String::from_utf8(crate::assay::render(&report, Limits::default()).unwrap()).unwrap();
    for row in dataset.rows() { assert!(!exported.contains(&row.id)); assert!(!exported.contains(&row.unit_sha256.to_hex())); }
    for digest in [dataset.manifest().expected_input_sha256, dataset.manifest().execution_identity.prompt_digest,
        dataset.manifest().execution_identity.schema_digest, dataset.manifest().execution_identity.taskir_digest,
        dataset.partition().split_digests().locked_test] { assert!(!exported.contains(&digest.to_hex())); }
    assert!(exported.contains("caller_declared")); assert!(exported.contains(HOST));
    let original = wire(&report);
    let alternate = CommitmentKey::new("assay-tests", [8; 32]).unwrap();
    let changed = wire(&compare(&dataset, &dataset, &alternate, NAMESPACE, HOST, gates(), fixtures::options(), Limits::default()).unwrap());
    assert_ne!(original["policy_commitment"], changed["policy_commitment"]);
    for host in ["", "injected\nhost", " leading-host"] {
        assert!(compare(&dataset, &dataset, &fixtures::key(), NAMESPACE, host, gates(), fixtures::options(), Limits::default()).is_err());
    }
    let mut changed_policy = gates(); changed_policy.max_brier_increase = 0.1;
    let changed = wire(&run(&dataset, &dataset, changed_policy, Limits::default()).unwrap());
    assert_ne!(original["policy_commitment"], changed["policy_commitment"]);
}

#[test]
fn the_report_seal_authenticates_claim_scope_policy_both_measurements_and_decision() {
    let dataset = fixtures::dataset(); let key = fixtures::key();
    let report = run(&dataset, &dataset, gates(), Limits::default()).unwrap();
    assert_eq!(report.seal, commit(&key, NAMESPACE, "qualification_report_v1", &report.body, Limits::default()).unwrap());
    for axis in 0..10 {
        let mut altered = report.clone();
        match axis {
            0 => altered.body.schema_version = 2,
            1 => altered.body.evidence = "native_replay".to_owned(),
            2 => altered.body.activation_authority = true,
            3 => altered.body.receipt_grade = ReceiptGrade::Replayable,
            4 => altered.body.declared_host_scope = "another-declared-host".to_owned(),
            5 => altered.body.gates.max_brier_increase = 0.1,
            6 => altered.body.policy_commitment.hmac_sha256 = Sha256Digest::of_bytes(b"changed policy"),
            7 | 8 => {
                let evaluation = if axis == 7 { &mut altered.body.candidate } else { &mut altered.body.baseline };
                let Measurement::Measured { metrics } = &mut evaluation.measurement else { panic!("measured fixture"); };
                metrics.brier += 0.1;
            }
            _ => {
                let Decision::Passed { paired, .. } = &mut altered.body.decision else { panic!("passed fixture"); };
                paired.equal -= 1; paired.candidate_better += 1;
            }
        }
        assert_ne!(altered.seal, commit(&key, NAMESPACE, "qualification_report_v1", &altered.body, Limits::default()).unwrap(), "axis {axis}");
    }
    assert_ne!(report.seal, commit(&key, "different-scope", "qualification_report_v1", &report.body, Limits::default()).unwrap());
    let alternate = CommitmentKey::new("assay-tests", [8; 32]).unwrap();
    assert_ne!(report.seal, commit(&alternate, NAMESPACE, "qualification_report_v1", &report.body, Limits::default()).unwrap());
    let different_host = compare(&dataset, &dataset, &key, NAMESPACE, "other-host", gates(), fixtures::options(), Limits::default()).unwrap();
    assert_ne!(report.seal, different_host.seal);
    assert_ne!(report.body.policy_commitment, different_host.body.policy_commitment);
}
