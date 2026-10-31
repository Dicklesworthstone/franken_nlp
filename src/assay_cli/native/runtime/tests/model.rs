//! Explicit opt-in integration source; owning a model does not establish quality.
use super::*;
use std::{fs::{self, OpenOptions},
    os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt},
    sync::atomic::{AtomicU64, Ordering}};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Source { snapshot: PathBuf, pin: String, memory: u64, kernel: Int8KernelKind, width: usize }
impl Source {
    fn explicit() -> Self {
        let snapshot = std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("explicit existing snapshot required").into();
        let pin = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model-content pin required");
        let memory = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit paired memory ceiling required")
            .parse().expect("memory ceiling must be MiB");
        let kernel = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit kernel required")
            .parse().expect("canonical kernel ID required");
        let width = std::env::var("FNLP_INT4_TEST_GROUP_WIDTH").unwrap_or_else(|_| "128".to_owned())
            .parse().expect("positive snapshot group width required");
        PackedInt4Kernel::detect(kernel).expect("requested kernel must be supported");
        Self { snapshot, pin, memory, kernel, width }
    }
}

struct Fixture {
    root: PathBuf, suite: PathBuf, cases: PathBuf, key: PathBuf, gates: PathBuf,
    rows: Vec<u8>, manifest: serde_json::Value, private: Vec<String>,
}
impl Fixture {
    fn new(locked: usize) -> Self {
        let root = std::env::temp_dir().join(format!("fnlp-native-assay-{}-{}",
            std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
        let (suite, cases, key, gates) = (root.join("suite.json"), root.join("cases.ndjson"),
            root.join("key"), root.join("gates.json"));
        write(&key, &[0x42; 32]);
        let mut rows = Vec::new(); let mut private = Vec::new(); let mut splits = serde_json::Map::new();
        for (split, count) in [("development", 1), ("calibration", 1), ("locked_test", locked)] {
            let mut ids = Vec::new();
            for ordinal in 0..count {
                let id = format!("private-native-{split}-{ordinal}"); ids.push(id.clone()); private.push(id.clone());
                let positive = ordinal % 2 == 0;
                let text = format!("Private fixture {split} document {ordinal}. {}",
                    if positive { "The product works well and I am satisfied." }
                    else { "The product broke immediately and I am disappointed." });
                private.push(text.clone()); private.push(Sha256Digest::of_bytes(text.as_bytes()).to_hex());
                let row = serde_json::json!({"schema_version":1,"id":id,"split":split,"text":text,"positive":positive});
                rows.extend(canonjson::canonical_bytes(&row).unwrap()); rows.push(b'\n');
            }
            splits.insert(split.to_owned(), serde_json::json!(ids));
        }
        let manifest = serde_json::json!({"schema_version":1,"task":"binary_classification",
            "dataset_id":"owned-native-fixture","dataset_version":"1","named_population":"synthetic-integration-only",
            "positive_label":{"id":"positive","description":"An approving product review."},
            "negative_label":{"id":"negative","description":"A disapproving product review."},
            "expected_input_sha256":Sha256Digest::of_bytes(&rows),"splits":splits});
        private.push(Sha256Digest::of_bytes(&rows).to_hex());
        write(&suite, &canonjson::canonical_bytes(&manifest).unwrap()); write(&cases, &rows);
        // These neutral equality gates test replay wiring, not model usefulness.
        let policy = serde_json::json!({"schema_version":1,"reliability":{"schema_version":1,
            "max_brier_increase":0.0,"max_ece_increase":0.0,"max_selective_risk":1.0,
            "min_acceptance_rate":0.0,"minimum_accepted":1},
            "max_accuracy_decrease":0.0,"max_macro_f1_decrease":0.0});
        write(&gates, &canonjson::canonical_bytes(&policy).unwrap());
        Self { root, suite, cases, key, gates, rows, manifest, private }
    }
    fn options(&self, source: &Source, paired: bool) -> Command {
        let mut options = command(paired, &["--minimum-samples", "5", "--acceptance-threshold", "0",
            "--bootstrap-resamples", "16", "--max-rows", "16", "--max-input-bytes", "65536",
            "--max-suite-bytes", "32768", "--max-line-bytes", "8192", "--max-output-bytes", "524288"]);
        options.args.common.suite = Some(self.suite.clone()); options.args.common.observations = self.cases.clone();
        options.args.common.key_file = Some(self.key.clone()); options.args.snapshot = Some(source.snapshot.clone());
        options.args.expected_model_sha256 = Some(source.pin.clone()); options.args.group_width = source.width;
        options.args.memory_mib = Some(source.memory); options.args.kernel = source.kernel;
        options.args.timeout_seconds = 86_400;
        if let Some(comparison) = &mut options.comparison {
            comparison.baseline_snapshot = Some(source.snapshot.clone());
            comparison.baseline_model_sha256 = Some(source.pin.clone()); comparison.baseline_group_width = source.width;
            comparison.gates = Some(self.gates.clone());
        }
        options
    }
    fn run(&self, mut options: Command, name: &str) -> (ExitCode, Option<serde_json::Value>, Vec<u8>) {
        let path = self.root.join(name); options.args.common.output = Some(path.clone());
        let (mut out, mut errors) = (Vec::new(), Vec::new());
        let code = options.run(&mut &b""[..], &mut out, &mut errors);
        assert!(out.is_empty(), "file output must not also publish a report to stdout");
        let report = if path.exists() {
            let bytes = fs::read(&path).unwrap();
            assert_eq!(fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
            assert_eq!(bytes.last(), Some(&b'\n'));
            let text = std::str::from_utf8(&bytes).unwrap();
            for private in &self.private { assert!(!text.contains(private)); }
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["protocol"], NATIVE_PROTOCOL); assert_eq!(value["evidence"], "native_model_replay");
            assert_eq!(value["activation_authority"], false);
            Some(value)
        } else { None };
        (code, report, errors)
    }
}
fn write(path: &Path, bytes: &[u8]) {
    let mut file = OpenOptions::new().write(true).create_new(true).mode(0o600).open(path).unwrap();
    file.write_all(bytes).unwrap(); file.sync_all().unwrap();
}
fn work(value: &serde_json::Value) -> [u64; 5] {
    [value["forward_positions"].as_u64().unwrap(), value["projected_logits"].as_u64().unwrap(),
        value["attention_pairs"].as_u64().unwrap(), value["projections"]["dot_products"].as_u64().unwrap(),
        value["projections"]["multiply_accumulates"].as_u64().unwrap()]
}

#[test]
#[ignore = "requires controller-admitted independently pinned INT4 snapshot for actual native Assay eval and paired replay"]
fn model_present_native_assay_replays_and_paired_work_are_exact_and_private() {
    const CHILD: &str = "FNLP_INT4_ASSAY_REPLAY_CHILD";
    const COMPLETE: &str = "FNLP_INT4_ASSAY_REPLAY_COMPLETE";
    let full_name = concat!(module_path!(), "::model_present_native_assay_replays_and_paired_work_are_exact_and_private");
    let name = full_name.split_once("::").expect("crate-qualified fixture name").1;
    if std::env::var(CHILD).ok().as_deref() != Some(name) {
        let result = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", name, "--ignored", "--nocapture"]).env(CHILD, name)
            .output().expect("isolated native Assay fixture must start");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(result.status.success(), "isolated native Assay fixture failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&result.stderr));
        assert!(stdout.contains(COMPLETE), "exact fixture filter ran no completed fixture");
        return;
    }
    assert!(crate::installed_process_resources().is_none());
    let source = Source::explicit(); let fixture = Fixture::new(5);
    let paired_options = fixture.options(&source, true);
    Plan::new(&paired_options, paired_options.validate().unwrap())
        .expect("explicit memory ceiling must admit both resident models and complete replay/report peak");
    let (code, evaluated, errors) = fixture.run(fixture.options(&source, false), "evaluation.json");
    assert_eq!(code, ExitCode::SUCCESS, "{}", String::from_utf8_lossy(&errors)); assert!(errors.is_empty());
    let evaluated = evaluated.unwrap(); let eval = &evaluated["report"];
    assert_eq!(eval["evidence"], native_classify::NATIVE_EVIDENCE); assert_eq!(eval["activation_authority"], false);
    assert_eq!(eval["executed_rows"], 7); assert_eq!(eval["classification"]["rows"], 5);
    assert_eq!(eval["measurement"]["status"], "measured");
    assert_eq!(eval["bindings"]["execution_collection_rows"], 7);
    assert_eq!(eval["bindings"]["observation_collection_rows"], 7);
    assert_eq!(eval["bindings"]["observation_collection_format"], native_classify::OBSERVATION_COLLECTION_FORMAT);
    let single_work = work(&eval["model_work"]); assert!(single_work.into_iter().all(|n| n > 0));

    let (code, qualified, errors) = fixture.run(paired_options, "qualification.json");
    assert_eq!(code, ExitCode::SUCCESS, "{}", String::from_utf8_lossy(&errors)); assert!(errors.is_empty());
    let qualified = qualified.unwrap(); let report = &qualified["report"];
    assert_eq!(report["host_scope_evidence"], "caller_declared"); assert_eq!(report["activation_authority"], false);
    assert_eq!(report["decision"]["status"], "passed");
    for arm in ["candidate", "baseline"] {
        assert_eq!(report[arm]["executed_rows"], 7); assert_eq!(report[arm]["measurement"], eval["measurement"]);
        assert_eq!(report[arm]["classification"], eval["classification"]);
        assert_eq!(report[arm]["bindings"]["execution_collection"], eval["bindings"]["execution_collection"]);
        assert_eq!(report[arm]["bindings"]["observation_collection"], eval["bindings"]["observation_collection"]);
        assert_eq!(report[arm]["bindings"]["observation_collection_rows"], 7);
        assert_eq!(work(&report[arm]["model_work"]), single_work);
    }
    let paired = &report["decision"]["paired"];
    assert_eq!(paired["rows"], 5); assert_eq!(paired["equal"], 5);
    assert_eq!(paired["candidate_better"], 0); assert_eq!(paired["candidate_worse"], 0);
    assert_eq!(paired["mean_brier_change"].as_f64(), Some(0.0));
    for gate in report["decision"]["comparisons"].as_array().unwrap() {
        assert_eq!(gate["passed"], true);
        if ["brier_increase", "ece_increase", "accuracy_decrease", "macro_f1_decrease"]
            .contains(&gate["name"].as_str().unwrap()) { assert_eq!(gate["observed"].as_f64(), Some(0.0)); }
    }
    assert_eq!(work(&report["total_model_work"]), single_work.map(|n| n.checked_mul(2).unwrap()));

    // A one-arm allowance must not be silently renewed for the baseline.
    let mut limited = fixture.options(&source, true);
    limited.args.max_forward_positions = single_work[0]; limited.args.max_projected_logits = single_work[1];
    limited.args.max_attention_pairs = single_work[2]; limited.args.max_dot_products = single_work[3];
    limited.args.max_multiply_accumulates = single_work[4];
    let (code, partial, errors) = fixture.run(limited, "refused-paired-budget.json");
    assert_eq!(code, ErrorCode::BudgetOrTimeout.as_process_exit()); assert!(partial.is_none());
    assert!(!errors.is_empty());

    // A smaller real replay reports its lack of statistical support, not an invented scorecard.
    let tiny = Fixture::new(1);
    let (code, insufficient, errors) = tiny.run(tiny.options(&source, false), "insufficient.json");
    assert_eq!(code, ExitCode::SUCCESS, "{}", String::from_utf8_lossy(&errors)); assert!(errors.is_empty());
    let insufficient = insufficient.unwrap();
    assert_eq!(insufficient["report"]["measurement"]["status"], "insufficient_data");
    assert!(insufficient["report"]["measurement"].get("metrics").is_none());
    assert!(insufficient["report"]["classification"].is_null());

    let mut wrong_source = fixture.manifest.clone();
    wrong_source["expected_input_sha256"] = serde_json::json!(Sha256Digest::of_bytes(b"different frozen case bytes"));
    let wrong_suite = fixture.root.join("wrong-pin-suite.json");
    write(&wrong_suite, &canonjson::canonical_bytes(&wrong_source).unwrap());
    let mut options = fixture.options(&source, false); options.args.common.suite = Some(wrong_suite);
    let (code, report, errors) = fixture.run(options, "refused-source.json");
    assert_eq!(code, ErrorCode::ArtifactIntegrityOrFormatOrVersion.as_process_exit()); assert!(report.is_none());
    assert!(!errors.is_empty());
    let mut wrong_pin = source.pin.clone();
    wrong_pin.replace_range(..1, if source.pin.starts_with('0') { "1" } else { "0" });
    let mut options = fixture.options(&source, false); options.args.expected_model_sha256 = Some(wrong_pin);
    let (code, report, errors) = fixture.run(options, "refused-model.json");
    assert_eq!(code, ErrorCode::ArtifactIntegrityOrFormatOrVersion.as_process_exit()); assert!(report.is_none());
    assert!(!errors.is_empty());
    assert_eq!(fs::read(&fixture.cases).unwrap(), fixture.rows);
    println!("{COMPLETE}");
}
