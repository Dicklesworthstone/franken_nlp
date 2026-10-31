//! Opt-in real native NER wiring; equality is not a model-quality claim.
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
        let root = std::env::temp_dir().join(format!("fnlp-native-ner-assay-{}-{}",
            std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
        let (suite, cases, key, gates) = (root.join("suite.json"), root.join("cases.ndjson"),
            root.join("key"), root.join("gates.json"));
        write(&key, &[0x42; 32]);
        let mut rows = Vec::new(); let mut private = Vec::new(); let mut splits = serde_json::Map::new();
        for (split, count) in [("development", 1), ("calibration", 1), ("locked_test", locked)] {
            let mut ids = Vec::new();
            for ordinal in 0..count {
                let id = format!("private-ner-{split}-{ordinal}"); ids.push(id.clone()); private.push(id.clone());
                let (name, language) = [("Élodie", "fr"), ("Alice", "en"), ("李雷", "zh")][ordinal % 3];
                // One held-out repeated name exercises unresolved occurrences
                // without presuming what this particular model will extract.
                let text = if ordinal == 4 { format!("{name} met {name}. Owned source {split} {ordinal}.") }
                    else { format!("{name} arrived. Owned source {split} {ordinal}.") };
                private.push(text.clone()); private.push(Sha256Digest::of_bytes(text.as_bytes()).to_hex());
                rows.extend(canonjson::canonical_bytes(&serde_json::json!({"schema_version":1,"id":id,
                    "split":split,"text":text,"language":language,"spans":[{
                        "entity_type":"person","byte_start":0,"byte_end":name.len()}]})).unwrap());
                rows.push(b'\n');
            }
            splits.insert(split.to_owned(), serde_json::json!(ids));
        }
        let manifest = serde_json::json!({"schema_version":1,"task":"named_entity_recognition",
            "dataset_id":"owned-ner-fixture","dataset_version":"1","named_population":"synthetic-integration-only",
            "options":{"types":["person"],"max_entities":4,"max_mention_scalars":64},
            "expected_input_sha256":Sha256Digest::of_bytes(&rows),"splits":splits});
        private.push(Sha256Digest::of_bytes(&rows).to_hex());
        write(&suite, &canonjson::canonical_bytes(&manifest).unwrap()); write(&cases, &rows);
        let policy = serde_json::json!({"schema_version":1,"max_exact_f1_decrease":0.0,
            "max_type_relaxed_f1_decrease":0.0,"max_exact_recall_decrease":0.0,"max_document_accuracy_decrease":0.0});
        write(&gates, &canonjson::canonical_bytes(&policy).unwrap());
        Self { root, suite, cases, key, gates, rows, manifest, private }
    }
    fn options(&self, source: &Source, paired: bool) -> Command {
        let mut options = command(paired, &["--minimum-samples", "5", "--bootstrap-resamples", "16",
            "--max-rows", "16", "--max-input-bytes", "65536", "--max-suite-bytes", "32768",
            "--max-line-bytes", "8192", "--max-output-bytes", "524288", "--max-spans-per-row", "4"]);
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
        assert!(out.is_empty(), "file output must not also publish to stdout");
        let report = if path.exists() {
            let bytes = fs::read(&path).unwrap();
            assert_eq!(fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
            assert_eq!(bytes.last(), Some(&b'\n'));
            let text = std::str::from_utf8(&bytes).unwrap();
            for private in &self.private { assert!(!text.contains(private)); }
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["protocol"], NER_PROTOCOL); assert_eq!(value["evidence"], "native_model_replay");
            assert_eq!(value["activation_authority"], false); Some(value)
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
#[ignore = "requires controller-admitted independently pinned INT4 snapshot for native span eval and paired NER replay"]
fn model_present_ner_assay_replays_locked_sources_and_accounts_paired_work_without_content_leakage() {
    const CHILD: &str = "FNLP_INT4_NER_ASSAY_CHILD";
    const COMPLETE: &str = "FNLP_INT4_NER_ASSAY_COMPLETE";
    let full_name = concat!(module_path!(), "::model_present_ner_assay_replays_locked_sources_and_accounts_paired_work_without_content_leakage");
    let name = full_name.split_once("::").expect("crate-qualified fixture name").1;
    if std::env::var(CHILD).ok().as_deref() != Some(name) {
        let result = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", name, "--ignored", "--nocapture"]).env(CHILD, name)
            .output().expect("isolated NER fixture must start");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(result.status.success(), "isolated NER fixture failed:\n{stdout}\n{}", String::from_utf8_lossy(&result.stderr));
        assert!(stdout.contains(COMPLETE), "exact fixture filter ran no completed fixture"); return;
    }
    assert!(crate::installed_process_resources().is_none());
    let source = Source::explicit(); let fixture = Fixture::new(5);
    let paired_options = fixture.options(&source, true); let (limits, ner) = paired_options.validate().unwrap();
    Plan::new(&paired_options, limits, ner).expect("explicit memory ceiling must admit complete paired peak");
    let (code, evaluated, errors) = fixture.run(fixture.options(&source, false), "evaluation.json");
    assert_eq!(code, ExitCode::SUCCESS, "{}", String::from_utf8_lossy(&errors)); assert!(errors.is_empty());
    let evaluated = evaluated.unwrap(); let eval = &evaluated["report"];
    assert_eq!(eval["evidence"], native_ner::NATIVE_EVIDENCE); assert_eq!(eval["score_space"], "not_computed");
    assert_eq!(eval["executed_rows"], 5); assert_eq!(eval["validated_rows"], 7);
    assert_eq!(eval["measurement"]["status"], "measured"); assert_eq!(eval["measurement"]["metrics"]["rows"], 5);
    assert_eq!(eval["measurement"]["metrics"]["exact"]["support"], 5);
    assert_eq!(eval["bindings"]["execution_collection_rows"], 5);
    assert_eq!(eval["bindings"]["observation_collection_rows"], 5);
    assert_eq!(eval["bindings"]["observation_collection_format"], native_ner::OBSERVATION_COLLECTION_FORMAT);
    let single = work(&eval["model_work"]); assert!(single.into_iter().all(|n| n > 0));
    let masks = eval["mask_node_visits"].as_u64().unwrap(); assert!(masks > 0);
    let verification = eval["total_source_verification_steps"].as_u64().unwrap();
    assert_eq!(verification, eval["independent_verification_steps"].as_u64().unwrap() * 3);
    let (code, qualified, errors) = fixture.run(paired_options, "qualification.json");
    assert_eq!(code, ExitCode::SUCCESS, "{}", String::from_utf8_lossy(&errors)); assert!(errors.is_empty());
    let qualified = qualified.unwrap(); let report = &qualified["report"];
    assert_eq!(report["host_scope_evidence"], "caller_declared"); assert_eq!(report["decision"]["status"], "passed");
    for arm in ["candidate", "baseline"] {
        assert_eq!(report[arm]["measurement"], eval["measurement"]);
        assert_eq!(report[arm]["bindings"]["execution_collection"], eval["bindings"]["execution_collection"]);
        assert_eq!(report[arm]["bindings"]["observation_collection"], eval["bindings"]["observation_collection"]);
        assert_eq!(report[arm]["executed_rows"], 5); assert_eq!(report[arm]["validated_rows"], 7);
        assert_eq!(work(&report[arm]["model_work"]), single);
    }
    let paired = &report["decision"]["paired"];
    assert_eq!(paired["rows"], 5); assert_eq!(paired["equal"], 5);
    assert_eq!(paired["candidate_better"], 0); assert_eq!(paired["candidate_worse"], 0);
    for name in ["exact_f1_change", "type_relaxed_f1_change", "exact_recall_change", "document_accuracy_change"] {
        assert_eq!(paired[name].as_f64(), Some(0.0));
    }
    assert_eq!(work(&report["total_model_work"]), single.map(|value| value.checked_mul(2).unwrap()));
    assert_eq!(report["total_mask_node_visits"].as_u64(), Some(masks * 2));
    assert_eq!(report["total_source_verification_steps"].as_u64(), Some(verification * 2));

    // Work authority for one arm must never renew for the second arm.
    let mut limited = fixture.options(&source, true);
    limited.args.max_forward_positions = single[0]; limited.args.max_projected_logits = single[1];
    limited.args.max_attention_pairs = single[2]; limited.args.max_dot_products = single[3];
    limited.args.max_multiply_accumulates = single[4];
    let (code, partial, errors) = fixture.run(limited, "refused-paired-work.json");
    assert_eq!(code, ErrorCode::BudgetOrTimeout.as_process_exit()); assert!(partial.is_none()); assert!(!errors.is_empty());

    let tiny = Fixture::new(1);
    let (code, insufficient, errors) = tiny.run(tiny.options(&source, false), "insufficient.json");
    assert_eq!(code, ExitCode::SUCCESS, "{}", String::from_utf8_lossy(&errors)); assert!(errors.is_empty());
    let insufficient = insufficient.unwrap();
    assert_eq!(insufficient["report"]["measurement"]["status"], "insufficient_data");
    assert_eq!(insufficient["report"]["executed_rows"], 1);
    assert!(insufficient["report"]["measurement"].get("metrics").is_none());

    // The complete input pin and gold scalar-boundary validation precede a
    // deliberately nonexistent model path and publish no successful prefix.
    let mut wrong = fixture.manifest.clone(); wrong["expected_input_sha256"] = serde_json::json!(Sha256Digest::of_bytes(b"foreign"));
    let suite = fixture.root.join("wrong-pin.json"); write(&suite, &canonjson::canonical_bytes(&wrong).unwrap());
    let mut options = fixture.options(&source, false); options.args.common.suite = Some(suite);
    options.args.snapshot = Some(fixture.root.join("must-not-open-model"));
    let (code, report, errors) = fixture.run(options, "refused-source-pin.json");
    assert_eq!(code, ErrorCode::ArtifactIntegrityOrFormatOrVersion.as_process_exit()); assert!(report.is_none());
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&errors).unwrap()["reason"], "observation_digest_mismatch");
    let mut rows = Vec::new();
    for (index, line) in std::str::from_utf8(&fixture.rows).unwrap().lines().enumerate() {
        let mut row: serde_json::Value = serde_json::from_str(line).unwrap();
        if index == 0 { row["spans"][0]["byte_end"] = serde_json::json!(1); }
        rows.extend(canonjson::canonical_bytes(&row).unwrap()); rows.push(b'\n');
    }
    let mut manifest = fixture.manifest.clone(); manifest["expected_input_sha256"] = serde_json::json!(Sha256Digest::of_bytes(&rows));
    let suite = fixture.root.join("bad-span-suite.json"); let cases = fixture.root.join("bad-span.ndjson");
    write(&suite, &canonjson::canonical_bytes(&manifest).unwrap()); write(&cases, &rows);
    let mut options = fixture.options(&source, false); options.args.common.suite = Some(suite); options.args.common.observations = cases;
    options.args.snapshot = Some(fixture.root.join("must-not-open-model"));
    let (code, report, errors) = fixture.run(options, "refused-gold-boundary.json");
    assert_eq!(code, ErrorCode::InputDecodeOrParse.as_process_exit()); assert!(report.is_none());
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&errors).unwrap()["reason"], "invalid_observation");
    assert_eq!(fs::read(&fixture.cases).unwrap(), fixture.rows);
    println!("{COMPLETE}");
}
