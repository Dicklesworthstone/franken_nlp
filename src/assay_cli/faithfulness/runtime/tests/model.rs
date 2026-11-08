//! Opt-in real native wiring. Identical-model agreement is not task quality.
use super::*;
use std::{fs::{self, OpenOptions}, os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt},
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
            .parse().expect("positive group width required");
        PackedInt4Kernel::detect(kernel).expect("explicit kernel must be supported");
        Self { snapshot, pin, memory, kernel, width }
    }
}
struct Fixture { root: PathBuf, suite: PathBuf, cases: PathBuf, key: PathBuf, gates: PathBuf,
    private: Vec<String>, input: Vec<u8>, manifest: serde_json::Value }
impl Fixture {
    fn new(sources: usize) -> Self {
        let root = std::env::temp_dir().join(format!("fnlp-faithfulness-assay-{}-{}",
            std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
        let (suite, cases, key, gates) = (root.join("suite.json"), root.join("cases.ndjson"), root.join("key"), root.join("gates.json"));
        write(&key, &[0x42; 32]);
        let mut input = Vec::new(); let mut private = Vec::new(); let mut splits = serde_json::Map::new();
        for (split, count, claims) in [("development", 1, 1), ("calibration", 1, 1), ("locked_test", sources, 2)] {
            let mut ids = Vec::new();
            for source in 0..count { for claim in 0..claims {
                let id = format!("private-{split}-{source}-{claim}"); ids.push(id.clone()); private.push(id.clone());
                let text = format!("Private record {split} {source}: Élodie arrived. The gate is blue.");
                let (label, statements) = [
                    ("entailed", ["Élodie arrived.", "The gate is blue."]),
                    ("contradicted", ["Élodie did not arrive.", "The gate is red."]),
                    ("unsupported", ["Élodie owns a boat.", "The gate weighs ten tons."]),
                ][source % 3];
                private.extend([text.clone(), statements[claim].to_owned(), Sha256Digest::of_bytes(text.as_bytes()).to_hex()]);
                input.extend(canonjson::canonical_bytes(&serde_json::json!({"schema_version":1,"id":id,"split":split,
                    "source":text,"claim":statements[claim],"relation":label})).unwrap()); input.push(b'\n');
            } }
            splits.insert(split.to_owned(), serde_json::json!(ids));
        }
        let manifest = serde_json::json!({"schema_version":1,"task":"faithfulness","dataset_id":"owned-faithfulness",
            "dataset_version":"1","named_population":"synthetic integration only", "policy":{
                "minimum_candidate_weight_ppm":500000,"minimum_margin_milli":100,"evidence_window_bytes":32,
                "max_evidence_windows":31,"max_evidence_spans":31},
            "expected_input_sha256":Sha256Digest::of_bytes(&input),"splits":splits});
        private.push(Sha256Digest::of_bytes(&input).to_hex());
        write(&suite, &canonjson::canonical_bytes(&manifest).unwrap()); write(&cases, &input);
        write(&gates, &canonjson::canonical_bytes(&serde_json::json!({"schema_version":1,
            "max_accuracy_decrease":0.0,"max_macro_f1_decrease":0.0,"max_brier_increase":0.0,"max_coverage_decrease":0.0})).unwrap());
        Self { root, suite, cases, key, gates, private, input, manifest }
    }
    fn options(&self, source: &Source, paired: bool) -> Command {
        let mut options = command(paired, &["--minimum-samples", "5", "--bootstrap-resamples", "16",
            "--max-rows", "32", "--max-input-bytes", "65536", "--max-suite-bytes", "32768",
            "--max-line-bytes", "8192", "--max-output-bytes", "524288"]);
        options.args.common.suite = Some(self.suite.clone()); options.args.common.observations = self.cases.clone();
        options.args.common.key_file = Some(self.key.clone()); options.args.snapshot = Some(source.snapshot.clone());
        options.args.expected_model_sha256 = Some(source.pin.clone()); options.args.group_width = source.width;
        options.args.memory_mib = Some(source.memory); options.args.kernel = source.kernel; options.args.timeout_seconds = 86_400;
        if let Some(comparison) = &mut options.comparison {
            comparison.baseline_snapshot = Some(source.snapshot.clone()); comparison.baseline_model_sha256 = Some(source.pin.clone());
            comparison.baseline_group_width = source.width; comparison.gates = Some(self.gates.clone());
        }
        options
    }
    fn run(&self, mut options: Command, name: &str) -> (ExitCode, Option<serde_json::Value>, Vec<u8>) {
        let path = self.root.join(name); options.args.common.output = Some(path.clone());
        let (mut output, mut errors) = (Vec::new(), Vec::new());
        let code = options.run(&mut &b""[..], &mut output, &mut errors); assert!(output.is_empty());
        let report = if path.exists() {
            let bytes = fs::read(&path).unwrap(); assert_eq!(bytes.last(), Some(&b'\n'));
            assert_eq!(fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
            let text = std::str::from_utf8(&bytes).unwrap();
            for private in &self.private { assert!(!text.contains(private)); }
            let report: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(report["protocol"], FAITHFULNESS_PROTOCOL); assert_eq!(report["activation_authority"], false);
            Some(report)
        } else { None };
        for private in &self.private { assert!(!String::from_utf8_lossy(&errors).contains(private)); }
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
#[ignore = "requires controller-admitted independently pinned INT4 snapshot; no native quality claim"]
fn model_present_faithfulness_replay_binds_all_heads_and_repeated_source_claims() {
    const CHILD: &str = "FNLP_INT4_FAITHFULNESS_ASSAY_CHILD";
    const COMPLETE: &str = "FNLP_INT4_FAITHFULNESS_ASSAY_COMPLETE";
    let full_name = concat!(module_path!(), "::model_present_faithfulness_replay_binds_all_heads_and_repeated_source_claims");
    let name = full_name.split_once("::").expect("crate-qualified fixture name").1;
    if std::env::var(CHILD).ok().as_deref() != Some(name) {
        let result = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", name, "--ignored", "--nocapture"]).env(CHILD, name).output().expect("isolated fixture must start");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(result.status.success(), "isolated fixture failed:\n{stdout}\n{}", String::from_utf8_lossy(&result.stderr));
        assert!(stdout.contains(COMPLETE), "exact fixture filter ran no complete fixture"); return;
    }
    assert!(crate::installed_process_resources().is_none());
    let source = Source::explicit(); let fixture = Fixture::new(6);
    let pair_options = fixture.options(&source, true); let limits = pair_options.validate().unwrap();
    Plan::new(&pair_options, limits).expect("explicit memory ceiling must admit paired peak");
    let (code, evaluated, errors) = fixture.run(fixture.options(&source, false), "evaluation.json");
    assert_eq!(code, ExitCode::SUCCESS, "{}", String::from_utf8_lossy(&errors)); assert!(errors.is_empty());
    let evaluated = evaluated.unwrap(); let eval = &evaluated["report"];
    assert_eq!(eval["evidence"], native_faithfulness::NATIVE_EVIDENCE); assert_eq!(eval["executed_rows"], 12);
    assert_eq!(eval["validated_rows"], 14); assert!(eval["executed_heads"].as_u64().unwrap() > 12);
    assert_eq!(eval["measurement"]["status"], "measured"); assert_eq!(eval["measurement"]["quality"]["source_clusters"], 6);
    assert_eq!(eval["measurement"]["quality"]["labels"]["rows"], 12);
    assert_eq!(eval["bindings"]["execution_collection_rows"], 12); assert_eq!(eval["bindings"]["observation_collection_rows"], 12);
    let single = work(&eval["model_work"]); assert!(single.into_iter().all(|n| n > 0));
    let (code, paired, errors) = fixture.run(pair_options, "qualification.json");
    assert_eq!(code, ExitCode::SUCCESS, "{}", String::from_utf8_lossy(&errors)); assert!(errors.is_empty());
    let paired = paired.unwrap(); let report = &paired["report"];
    assert_eq!(report["decision"]["status"], "passed");
    for arm in ["candidate", "baseline"] {
        assert_eq!(report[arm]["measurement"], eval["measurement"]);
        assert_eq!(report[arm]["bindings"]["execution_collection"], eval["bindings"]["execution_collection"]);
        assert_eq!(report[arm]["bindings"]["observation_collection"], eval["bindings"]["observation_collection"]);
        assert_eq!(work(&report[arm]["model_work"]), single);
    }
    for name in ["accuracy_change", "macro_f1_change", "brier_change", "coverage_change"] {
        assert_eq!(report["decision"]["paired"][name].as_f64(), Some(0.0));
    }
    assert_eq!(work(&report["total_model_work"]), single.map(|value| value * 2));
    let mut limited = fixture.options(&source, true);
    limited.args.max_forward_positions = single[0]; limited.args.max_projected_logits = single[1];
    limited.args.max_attention_pairs = single[2]; limited.args.max_dot_products = single[3];
    limited.args.max_multiply_accumulates = single[4];
    let (code, partial, _) = fixture.run(limited, "refused-paired-work.json");
    assert_eq!(code, ErrorCode::BudgetOrTimeout.as_process_exit()); assert!(partial.is_none());
    let tiny = Fixture::new(1);
    let (code, insufficient, errors) = tiny.run(tiny.options(&source, false), "insufficient.json");
    assert_eq!(code, ExitCode::SUCCESS, "{}", String::from_utf8_lossy(&errors));
    assert_eq!(insufficient.unwrap()["report"]["measurement"]["status"], "insufficient_data");
    let mut wrong = fixture.manifest.clone(); wrong["expected_input_sha256"] = serde_json::json!(Sha256Digest::of_bytes(b"foreign"));
    let suite = fixture.root.join("wrong-pin.json"); write(&suite, &canonjson::canonical_bytes(&wrong).unwrap());
    let mut options = fixture.options(&source, false); options.args.common.suite = Some(suite);
    options.args.snapshot = Some(fixture.root.join("must-not-open-model"));
    let (code, report, errors) = fixture.run(options, "refused-pin.json");
    assert_eq!(code, ErrorCode::ArtifactIntegrityOrFormatOrVersion.as_process_exit()); assert!(report.is_none());
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&errors).unwrap()["reason"], "observation_digest_mismatch");
    assert_eq!(fs::read(&fixture.cases).unwrap(), fixture.input); println!("{COMPLETE}");
}
