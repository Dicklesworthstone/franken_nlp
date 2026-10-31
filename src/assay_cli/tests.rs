use super::*;

fn command(args: &[&str]) -> ScoredCommand {
    let definition = definitions().into_iter().find(|c| c.get_name() == args[0]).unwrap();
    let matches = definition.try_get_matches_from(args).unwrap();
    match AssayCommand::from_matches(args[0], &matches).unwrap() {
        AssayCommand::Scored(command) => command,
        AssayCommand::Native(_) => panic!("supplied-observation fixture selected a native backend"),
        AssayCommand::NativeNer(_) => panic!("supplied-observation fixture selected native NER"),
        AssayCommand::NativeExtract(_) => panic!("supplied-observation fixture selected native extraction"),
    }
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> { panic!("refused operation reached input"); }
}
struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> { panic!("refused operation published output"); }
    fn flush(&mut self) -> std::io::Result<()> { panic!("refused operation flushed output"); }
}

#[test]
fn schema_is_explicit_model_free_and_available_without_files_or_a_runtime() {
    for name in ["eval", "calibrate", "qualify"] {
        let mut out = Vec::new(); let mut errors = Vec::new();
        assert_eq!(command(&[name, "scored", "--schema"]).run(&mut NoRead, &mut out, &mut errors), ExitCode::SUCCESS);
        assert!(errors.is_empty()); assert_eq!(out.last(), Some(&b'\n'));
        let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(value["protocol"], PROTOCOL); assert_eq!(value["operation"], name);
        assert_eq!(value["native_model_replay"], false); assert_eq!(value["activation_authority"], false);
        let suite = &value["json_schema"]["$defs"]["suite"];
        assert_eq!(suite["additionalProperties"], false);
        assert!(suite["required"].as_array().unwrap().iter().any(|v| v == "expected_input_sha256"));
        let gates = &value["json_schema"]["$defs"]["qualification_gates"];
        assert_eq!(gates["additionalProperties"], false);
        assert_eq!(gates["required"].as_array().unwrap().len(), 6);
        let qualified = &value["json_schema"]["$defs"]["qualification_report"];
        assert_eq!(qualified["additionalProperties"], false);
        assert_eq!(qualified["properties"]["gate_basis"]["const"], "point_estimates");
        assert_eq!(qualified["properties"]["activation_authority"]["const"], false);
    }
}

#[test]
fn a_backend_and_protected_key_are_mandatory_and_no_secret_value_option_exists() {
    for name in ["eval", "calibrate", "qualify"] {
        for args in [vec![name], vec![name, "scored"], vec![name, "native"],
            vec![name, "scored", "--schema", "--key-id", "unexpected"],
            vec![name, "scored", "--key", "private-secret"],
            vec![name, "scored", "--model", "model.fnlpq"]] {
            let definition = definitions().into_iter().find(|c| c.get_name() == name).unwrap();
            assert!(definition.try_get_matches_from(args).is_err());
        }
    }
}

fn minimal_eval(flags: &[&str]) -> ScoredCommand {
    let mut args = vec!["eval", "scored", "--suite", "absent-suite", "--key-file", "absent-key",
        "--key-id", "fixture-key", "--namespace", "fixture-suite"];
    args.extend_from_slice(flags); command(&args)
}

#[test]
fn invalid_limits_statistics_and_apply_context_refuse_before_input_or_key_reads() {
    for flags in [vec!["--max-input-bytes", "0"], vec!["--max-rows", "0"],
        vec!["--minimum-samples", "0"], vec!["--acceptance-threshold", "NaN"],
        vec!["--confidence-level", "1"], vec!["--bootstrap-resamples", "1"],
        vec!["--bootstrap-resamples", "1001"], vec!["--max-bootstrap-draws", "0"]] {
        let mut errors = Vec::new();
        assert_ne!(minimal_eval(&flags).run(&mut NoRead, &mut NoWrite, &mut errors), ExitCode::SUCCESS);
        assert!(!String::from_utf8(errors).unwrap().contains("absent-"));
    }
    let mut options = minimal_eval(&[]);
    let ScoredCommand::Eval(ref mut eval) = options else { unreachable!() };
    eval.as_of = Some("2026-01-01".to_owned());
    assert_eq!(options.run(&mut NoRead, &mut NoWrite, &mut Vec::new()), ErrorCode::Usage.as_process_exit());
}

#[test]
fn paired_comparison_requires_baseline_files_and_an_explicit_safe_host_scope() {
    for invalid in 0..5 {
        let mut options = command(&["qualify", "scored", "--suite", "absent-candidate-suite",
            "--key-file", "absent-key", "--key-id", "fixture-key", "--namespace", "fixture-suite",
            "--baseline-suite", "absent-baseline-suite", "--baseline-observations", "absent-baseline",
            "--gates", "absent-gates", "--host-scope", "declared-workstation"]);
        let ScoredCommand::Qualify(ref mut comparison) = options else { unreachable!() };
        match invalid {
            0 => comparison.baseline_suite = None,
            1 => comparison.baseline_observations = Some(PathBuf::from("-")),
            2 => comparison.gates = Some(PathBuf::new()),
            3 => comparison.host_scope = Some("private\ninvalid".to_owned()),
            _ => comparison.host_scope = Some("not a scope id".to_owned()),
        }
        let mut errors = Vec::new();
        assert_eq!(options.run(&mut NoRead, &mut NoWrite, &mut errors), ErrorCode::Usage.as_process_exit());
        let errors = String::from_utf8(errors).unwrap();
        assert!(!errors.contains("private")); assert!(!errors.contains("absent-"));
    }
}

#[test]
fn bounded_reader_observes_at_most_one_byte_beyond_the_admitted_ceiling() {
    let mut source = &b"abcdef"[..];
    let error = read_bytes(&mut source, 3).unwrap_err();
    assert_eq!(error.code, ErrorCode::AdmissionOrResourceLimit); assert_eq!(source, b"ef");
    assert_eq!(read_bytes(&mut &b"abc"[..], 3).unwrap(), b"abc");
}

#[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
mod files {
    use super::*;
    use std::{fs::{self, OpenOptions}, os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt},
        sync::atomic::{AtomicU64, Ordering}};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct Fixture { root: PathBuf, suite: PathBuf, key: PathBuf, observations: Vec<u8>, manifest: serde_json::Value }
    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!("fnlp-assay-cli-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
            let key = root.join("key"); write(&key, &[0x42; 32]);
            let mut rows = Vec::new(); let mut splits = serde_json::Map::new();
            for (split, prefix, count) in [("development", "dev", 2), ("calibration", "cal", 20), ("locked_test", "test", 20)] {
                let mut ids = Vec::new();
                for index in 0..count {
                    let id = format!("{prefix}-{index}"); ids.push(id.clone());
                    let row = serde_json::json!({"schema_version":1,"id":id,"split":split,
                        "probability":if index % 2 == 0 { 0.7 } else { 0.3 },"positive":index % 2 == 0,
                        "unit_sha256":Sha256Digest::of_bytes(id.as_bytes())});
                    rows.extend(canonjson::canonical_bytes(&row).unwrap()); rows.push(b'\n');
                }
                splits.insert(split.to_owned(), serde_json::json!(ids));
            }
            let digest = |label: &str| Sha256Digest::of_bytes(label.as_bytes()).to_hex();
            let identity = serde_json::json!({"schema_version":1,"source_revision":"f56ec5a9650268aa098496734743c25ea778bd2d",
                "logical_model_digest":digest("model"),"artifact_format":"fixture-fnlpq-v1","quant_recipe":"fixture-int8-v1",
                "packing_set_digest":digest("packing"),"tokenizer_digest":digest("tokenizer"),"template_digest":digest("template"),
                "task_spec":"binary-event-fixture-v1","taskir_digest":digest("taskir"),"prompt_digest":digest("prompt"),
                "grammar_compiler_version":"fixture-grammar-v1","schema_digest":digest("schema"),"numerics_profile":"strict-quantized-v1",
                "kv_dtype":"bf16","sampler_version":"greedy-v1","thinking_mode":"disabled","tool_mode":"none",
                "calibration_digest":digest("uncalibrated"),"decision_policy_digest":digest("no-policy"),
                "backend_semantic_version":"fixture-cpu-v1","host_class":null,"compiler_identity":null});
            let manifest = serde_json::json!({"schema_version":1,"task":"binary_event","dataset_id":"fixture-suite",
                "dataset_version":"1","named_population":"synthetic-fixture-only","positive_label":"yes","negative_label":"no",
                "execution_identity":identity,"expected_input_sha256":Sha256Digest::of_bytes(&rows),"splits":splits});
            let suite = root.join("suite.json"); write(&suite, &canonjson::canonical_bytes(&manifest).unwrap());
            Self { root, suite, key, observations: rows, manifest }
        }
        fn args(&self, name: &str) -> Vec<String> {
            vec![name.to_owned(), "scored".to_owned(), "--suite".to_owned(), self.suite.to_string_lossy().into_owned(),
                "--key-file".to_owned(), self.key.to_string_lossy().into_owned(), "--key-id".to_owned(), "fixture-key".to_owned(),
                "--namespace".to_owned(), "fixture-suite".to_owned()]
        }
        fn run(&self, name: &str, extra: &[&str], data: &[u8]) -> (ExitCode, Vec<u8>, Vec<u8>) {
            let mut args = self.args(name); args.extend(extra.iter().map(|s| (*s).to_owned()));
            let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
            let (mut output, mut errors) = (Vec::new(), Vec::new());
            let code = command(&refs).run(&mut &*data, &mut output, &mut errors); (code, output, errors)
        }
    }
    fn write(path: &Path, bytes: &[u8]) {
        let mut file = OpenOptions::new().write(true).create_new(true).mode(0o600).open(path).unwrap();
        file.write_all(bytes).unwrap(); file.sync_all().unwrap();
    }

    #[test]
    fn supplied_observations_produce_a_complete_scoped_private_report() {
        let fixture = Fixture::new();
        let (code, out, errors) = fixture.run("eval", &[], &fixture.observations);
        assert_eq!(code, ExitCode::SUCCESS); assert!(errors.is_empty());
        let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(value["protocol"], PROTOCOL); assert_eq!(value["operation"], "eval");
        assert_eq!(value["report"]["evidence"], "provided_observations");
        assert_eq!(value["report"]["activation_authority"], false);
        assert_eq!(value["report"]["measurement"]["status"], "measured");
        assert_eq!(value["report"]["bindings"]["metadata"]["dataset_id"], "fixture-suite");
        assert_eq!(value["report"]["bindings"]["metadata"]["model_recipe_id"], "fixture-int8-v1");
        assert_eq!(value["report"]["bindings"]["metadata"]["prompt_template_sha256"],
            Sha256Digest::of_bytes(b"template").to_hex());
        let text = String::from_utf8(out).unwrap();
        for private in ["dev-0".to_owned(), Sha256Digest::of_bytes(b"dev-0").to_hex(),
            Sha256Digest::of_bytes(&fixture.observations).to_hex()] { assert!(!text.contains(&private)); }
    }

    #[test]
    fn calibration_report_round_trips_under_an_independent_pin_and_explicit_validity() {
        for method in ["temperature", "isotonic"] {
            let fixture = Fixture::new(); let artifact = fixture.root.join("calibration.json");
            let artifact_path = artifact.to_str().unwrap();
            let (code, out, errors) = fixture.run("calibrate", &["--method", method, "--valid-from", "2026-01-01",
                "--valid-through", "2026-12-31", "--shift-policy", "conservative-abstain", "-o", artifact_path], &fixture.observations);
            assert_eq!(code, ExitCode::SUCCESS); assert!(out.is_empty()); assert!(errors.is_empty());
            let bytes = fs::read(&artifact).unwrap(); assert_eq!(fs::metadata(&artifact).unwrap().permissions().mode() & 0o777, 0o600);
            let pin = Sha256Digest::of_bytes(&bytes).to_hex();
            let mut flags = vec!["--calibration", artifact_path, "--expected-calibration-sha256", &pin,
                "--as-of", "2026-06-01", "--shift-assessment", "in-distribution"];
            let (code, out, errors) = fixture.run("eval", &flags, &fixture.observations);
            assert_eq!(code, ExitCode::SUCCESS); assert!(errors.is_empty());
            assert_eq!(serde_json::from_slice::<serde_json::Value>(&out).unwrap()["report"]["status"], "calibrated");
            flags[5] = "2027-01-01";
            let (code, out, errors) = fixture.run("eval", &flags, &fixture.observations);
            assert_eq!(code, ExitCode::SUCCESS); assert!(errors.is_empty());
            assert_eq!(serde_json::from_slice::<serde_json::Value>(&out).unwrap()["report"]["status"], "abstained");
            let wrong = "00".repeat(32); flags[3] = &wrong;
            let (code, out, _) = fixture.run("eval", &flags, &fixture.observations);
            assert_eq!(code, ErrorCode::ArtifactIntegrityOrFormatOrVersion.as_process_exit()); assert!(out.is_empty());
        }
    }

    #[test]
    fn paired_qualification_is_scoped_and_rejects_gold_or_baseline_byte_substitution() {
        let fixture = Fixture::new();
        let mut baseline_rows = Vec::new();
        let mut mismatched_rows = Vec::new();
        let mut changed_gold = false;
        for line in std::str::from_utf8(&fixture.observations).unwrap().lines() {
            let mut row: serde_json::Value = serde_json::from_str(line).unwrap();
            row["probability"] = serde_json::json!(if row["positive"].as_bool().unwrap() { 0.6 } else { 0.4 });
            baseline_rows.extend(canonjson::canonical_bytes(&row).unwrap()); baseline_rows.push(b'\n');
            if !changed_gold && row["split"] == "locked_test" {
                row["positive"] = serde_json::json!(!row["positive"].as_bool().unwrap());
                changed_gold = true;
            }
            mismatched_rows.extend(canonjson::canonical_bytes(&row).unwrap()); mismatched_rows.push(b'\n');
        }
        let baseline_data = fixture.root.join("baseline.ndjson"); write(&baseline_data, &baseline_rows);
        let baseline_suite = fixture.root.join("baseline-suite.json");
        let mut manifest = fixture.manifest.clone();
        manifest["execution_identity"]["logical_model_digest"] = serde_json::json!(Sha256Digest::of_bytes(b"baseline-model"));
        manifest["expected_input_sha256"] = serde_json::json!(Sha256Digest::of_bytes(&baseline_rows));
        write(&baseline_suite, &canonjson::canonical_bytes(&manifest).unwrap());
        let gates = fixture.root.join("gates.json");
        let declared = serde_json::json!({"schema_version":1,"max_brier_increase":0.0,
            "max_ece_increase":0.0,"max_selective_risk":0.1,"min_acceptance_rate":0.4,"minimum_accepted":5});
        write(&gates, &canonjson::canonical_bytes(&declared).unwrap());
        let flags = ["--baseline-suite", baseline_suite.to_str().unwrap(),
            "--baseline-observations", baseline_data.to_str().unwrap(), "--gates", gates.to_str().unwrap(),
            "--host-scope", "caller-declared-fixture-host", "--acceptance-threshold", "0.5"];
        let (code, out, errors) = fixture.run("qualify", &flags, &fixture.observations);
        assert_eq!(code, ExitCode::SUCCESS); assert!(errors.is_empty());
        let report: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(report["operation"], "qualify");
        assert_eq!(report["report"]["decision"]["status"], "passed");
        assert_eq!(report["report"]["receipt_grade"], "audit_only");
        assert_eq!(report["report"]["host_scope_evidence"], "caller_declared");
        assert_eq!(report["report"]["gate_basis"], "point_estimates");
        assert_eq!(report["report"]["decision"]["comparisons"].as_array().unwrap().len(), 5);
        assert_eq!(report["report"]["decision"]["paired"]["candidate_better"], 20);
        assert_eq!(report["report"]["decision"]["paired"]["candidate_worse"], 0);
        assert!(report["report"]["seal"]["hmac_sha256"].as_str().is_some());
        assert_eq!(report["report"]["activation_authority"], false);
        assert_eq!(report["report"]["evidence"], "provided_observations");

        let strict_gates = fixture.root.join("strict-gates.json");
        let mut strict = declared; strict["max_brier_increase"] = serde_json::json!(-0.5);
        write(&strict_gates, &canonjson::canonical_bytes(&strict).unwrap());
        let mut rejected = flags; rejected[5] = strict_gates.to_str().unwrap();
        let (code, out, errors) = fixture.run("qualify", &rejected, &fixture.observations);
        assert_eq!(code, ExitCode::SUCCESS); assert!(errors.is_empty());
        assert_eq!(serde_json::from_slice::<serde_json::Value>(&out).unwrap()["report"]["decision"]["status"], "rejected");

        let mut tiny = flags.to_vec(); tiny.extend(["--minimum-samples", "21"]);
        let (code, out, errors) = fixture.run("qualify", &tiny, &fixture.observations);
        assert_eq!(code, ExitCode::SUCCESS); assert!(errors.is_empty());
        let report: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(report["report"]["decision"]["status"], "insufficient_data");
        assert!(report["report"]["decision"].get("paired").is_none());

        let bad_data = fixture.root.join("different-gold.ndjson"); write(&bad_data, &mismatched_rows);
        let bad_suite = fixture.root.join("different-gold-suite.json");
        manifest["expected_input_sha256"] = serde_json::json!(Sha256Digest::of_bytes(&mismatched_rows));
        write(&bad_suite, &canonjson::canonical_bytes(&manifest).unwrap());
        let mut mismatched = flags; mismatched[1] = bad_suite.to_str().unwrap(); mismatched[3] = bad_data.to_str().unwrap();
        let (code, out, errors) = fixture.run("qualify", &mismatched, &fixture.observations);
        assert_eq!(code, ErrorCode::ArtifactIntegrityOrFormatOrVersion.as_process_exit()); assert!(out.is_empty());
        assert!(!String::from_utf8(errors).unwrap().contains("test-0"));

        let stale_data = fixture.root.join("stale-baseline.ndjson"); baseline_rows.push(b' '); write(&stale_data, &baseline_rows);
        let mut stale = flags; stale[3] = stale_data.to_str().unwrap();
        let (code, out, _) = fixture.run("qualify", &stale, &fixture.observations);
        assert_eq!(code, ErrorCode::ArtifactIntegrityOrFormatOrVersion.as_process_exit()); assert!(out.is_empty());
    }

    #[test]
    fn stale_bytes_tiny_samples_output_limits_and_existing_destinations_never_emit_partial_success() {
        let fixture = Fixture::new();
        let mut stale = fixture.observations.clone(); stale.push(b' ');
        let (code, out, _) = fixture.run("eval", &[], &stale);
        assert_eq!(code, ErrorCode::ArtifactIntegrityOrFormatOrVersion.as_process_exit()); assert!(out.is_empty());
        let (code, out, _) = fixture.run("eval", &["--minimum-samples", "21"], &fixture.observations);
        assert_eq!(code, ExitCode::SUCCESS);
        assert_eq!(serde_json::from_slice::<serde_json::Value>(&out).unwrap()["report"]["measurement"]["status"], "insufficient_data");
        let (code, out, _) = fixture.run("eval", &["--max-output-bytes", "128"], &fixture.observations);
        assert_eq!(code, ErrorCode::AdmissionOrResourceLimit.as_process_exit()); assert!(out.is_empty());
        let existing = fixture.root.join("existing"); write(&existing, b"keep");
        let mut args = fixture.args("eval"); args.extend(["-o".to_owned(), existing.to_string_lossy().into_owned()]);
        let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
        assert_ne!(command(&refs).run(&mut NoRead, &mut NoWrite, &mut Vec::new()), ExitCode::SUCCESS);
        assert_eq!(fs::read(existing).unwrap(), b"keep");
    }

    #[test]
    fn strict_input_failures_cannot_echo_private_rows_or_json_fragments() {
        let fixture = Fixture::new();
        let private = "private-document-fragment-never-echo";
        let bad = format!("{{\"schema_version\":1,\"id\":\"{private}\",\"unexpected\":\"{private}\"}}\n");
        let mut manifest = fixture.manifest.clone(); manifest["expected_input_sha256"] = serde_json::json!(Sha256Digest::of_bytes(bad.as_bytes()));
        let path = fixture.root.join("malformed-suite.json"); write(&path, &canonjson::canonical_bytes(&manifest).unwrap());
        let mut args = fixture.args("eval"); args[3] = path.to_string_lossy().into_owned();
        let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
        let (mut out, mut errors) = (Vec::new(), Vec::new());
        assert_eq!(command(&refs).run(&mut bad.as_bytes(), &mut out, &mut errors), ErrorCode::InputDecodeOrParse.as_process_exit());
        assert!(out.is_empty());
        let text = String::from_utf8(errors).unwrap(); assert!(!text.contains(private)); assert!(!text.contains(path.to_str().unwrap()));
        let event: serde_json::Value = serde_json::from_str(&text).unwrap(); assert_eq!(event["event"], "assay_error");
    }
}
