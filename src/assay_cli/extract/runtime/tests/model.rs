//! Opt-in native schema-diverse replay; paired equality is not a quality claim.
use super::*;
use std::{fs::{self, OpenOptions}, os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt},
    sync::atomic::{AtomicU64, Ordering}};

static NEXT: AtomicU64 = AtomicU64::new(0);
const DECIMAL: &str = "12345678901234567890123456789012345678";
const INVOICE: &str = r#"{"type":"object","additionalProperties":false,"required":["private_amount","private_currency"],"properties":{"private_amount":{"type":"number","const":12345678901234567890123456789012345678},"private_currency":{"type":"string","enum":["CNY","EUR"]}}}"#;
const PEOPLE: &str = r#"{"type":"object","additionalProperties":false,"required":["private_name"],"properties":{"private_name":{"type":"string","maxLength":64,"x-fnlp-source":"verbatim"}}}"#;
const FLAGS: &str = r#"{"type":"array","maxItems":2,"items":{"type":"boolean"}}"#;

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
        let root = std::env::temp_dir().join(format!("fnlp-native-extract-assay-{}-{}",
            std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
        let (suite, cases, key, gates) = (root.join("suite.json"), root.join("cases.ndjson"),
            root.join("key"), root.join("gates.json"));
        write(&key, &[0x42; 32]);
        let mut rows = Vec::new(); let mut splits = serde_json::Map::new();
        let mut private = [DECIMAL, "private_amount", "private_currency", "private_name"]
            .into_iter().map(str::to_owned).collect::<Vec<_>>();
        for (split, count) in [("development", 1), ("calibration", 1), ("locked_test", locked)] {
            let mut ids = Vec::new();
            for ordinal in 0..count {
                let id = format!("private-extract-{split}-{ordinal}"); ids.push(id.clone()); private.push(id.clone());
                let (schema_id, language, text, expected_json) = match ordinal % 3 {
                    0 => ("invoice", "en", format!("Owned invoice {split} {ordinal}: amount {DECIMAL} CNY."),
                        format!("{{\"private_amount\":{DECIMAL},\"private_currency\":\"CNY\"}}")),
                    1 => ("people", "fr", if ordinal == 4 { format!("Zoë met Zoë. Owned source {split} {ordinal}.") }
                        else { format!("Zoë arrived. Owned source {split} {ordinal}.") }, r#"{"private_name":"Zoë"}"#.to_owned()),
                    _ => ("flags", "en", format!("Owned status {split} {ordinal}: the flag is true."), "[true]".to_owned()),
                };
                private.push(text.clone()); private.push(Sha256Digest::of_bytes(text.as_bytes()).to_hex());
                rows.extend(canonjson::canonical_bytes(&serde_json::json!({"schema_version":1,"id":id,
                    "split":split,"text":text,"schema_id":schema_id,"language":language,"expected_json":expected_json})).unwrap());
                rows.push(b'\n');
            }
            splits.insert(split.to_owned(), serde_json::json!(ids));
        }
        let manifest = serde_json::json!({"schema_version":1,"task":"structured_extraction",
            "dataset_id":"owned-extraction-fixture","dataset_version":"1","named_population":"synthetic-integration-only",
            "schemas":[{"id":"invoice","schema_json":INVOICE,"grounding":"structural"},
                {"id":"people","schema_json":PEOPLE,"grounding":"source_membership"},
                {"id":"flags","schema_json":FLAGS,"grounding":"structural"}],
            "expected_input_sha256":Sha256Digest::of_bytes(&rows),"splits":splits});
        private.push(Sha256Digest::of_bytes(&rows).to_hex());
        write(&suite, &canonjson::canonical_bytes(&manifest).unwrap()); write(&cases, &rows);
        let policy = serde_json::json!({"schema_version":1,"max_field_f1_decrease":0.0,
            "max_field_recall_decrease":0.0,"max_document_accuracy_decrease":0.0});
        write(&gates, &canonjson::canonical_bytes(&policy).unwrap());
        Self { root, suite, cases, key, gates, rows, manifest, private }
    }
    fn options(&self, source: &Source, paired: bool) -> Command {
        let mut options = command(paired, &["--minimum-samples", "5", "--bootstrap-resamples", "16",
            "--max-rows", "16", "--max-input-bytes", "65536", "--max-suite-bytes", "32768",
            "--max-line-bytes", "8192", "--max-output-bytes", "524288", "--max-schemas", "3",
            "--max-schema-bytes", "4096", "--max-fields-per-row", "16", "--max-json-nodes-per-row", "128",
            "--max-json-depth", "16"]);
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
            assert_eq!(value["protocol"], EXTRACT_PROTOCOL); assert_eq!(value["evidence"], "native_model_replay");
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
#[ignore = "requires controller-admitted independently pinned INT4 snapshot for native exact-JSON eval and paired replay"]
fn model_present_extraction_assay_preserves_exact_json_source_authority_and_paired_work() {
    const CHILD: &str = "FNLP_INT4_EXTRACTION_ASSAY_CHILD";
    const COMPLETE: &str = "FNLP_INT4_EXTRACTION_ASSAY_COMPLETE";
    let full_name = concat!(module_path!(), "::model_present_extraction_assay_preserves_exact_json_source_authority_and_paired_work");
    let name = full_name.split_once("::").expect("crate-qualified fixture name").1;
    if std::env::var(CHILD).ok().as_deref() != Some(name) {
        let result = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", name, "--ignored", "--nocapture"]).env(CHILD, name)
            .output().expect("isolated extraction fixture must start");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(result.status.success(), "isolated extraction fixture failed:\n{stdout}\n{}", String::from_utf8_lossy(&result.stderr));
        assert!(stdout.contains(COMPLETE), "exact fixture filter ran no completed fixture"); return;
    }
    assert!(crate::installed_process_resources().is_none());
    let source = Source::explicit(); let fixture = Fixture::new(5);
    let paired_options = fixture.options(&source, true); let (limits, extract) = paired_options.validate().unwrap();
    Plan::new(&paired_options, limits, extract).expect("explicit memory ceiling must admit the complete paired peak");
    let (code, evaluated, errors) = fixture.run(fixture.options(&source, false), "evaluation.json");
    assert_eq!(code, ExitCode::SUCCESS, "{}", String::from_utf8_lossy(&errors)); assert!(errors.is_empty());
    let evaluated = evaluated.unwrap(); let eval = &evaluated["report"];
    assert_eq!(eval["evidence"], native_extract::NATIVE_EVIDENCE); assert_eq!(eval["score_space"], "not_computed");
    assert_eq!(eval["executed_rows"], 5); assert_eq!(eval["validated_rows"], 7);
    assert_eq!(eval["measurement"]["status"], "measured"); assert_eq!(eval["measurement"]["metrics"]["rows"], 5);
    assert_eq!(eval["measurement"]["metrics"]["fields"]["support"], 7);
    assert_eq!(eval["measurement"]["metrics"]["by_schema"].as_array().unwrap().len(), 3);
    assert_eq!(eval["measurement"]["metrics"]["by_language"].as_array().unwrap().len(), 2);
    assert_eq!(eval["bindings"]["execution_collection_rows"], 5);
    assert_eq!(eval["bindings"]["observation_collection_rows"], 5);
    assert_eq!(eval["bindings"]["observation_collection_format"], native_extract::OBSERVATION_COLLECTION_FORMAT);
    let single = work(&eval["model_work"]); assert!(single.into_iter().all(|n| n > 0));
    let masks = eval["mask_node_visits"].as_u64().unwrap(); assert!(masks > 0);
    let verification = eval["total_source_verification_steps"].as_u64().unwrap();
    let gold = eval["gold_verification_steps"].as_u64().unwrap(); assert!(gold > 0);
    assert_eq!(verification, gold + eval["independent_verification_steps"].as_u64().unwrap() * 3);
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
    for name in ["field_f1_change", "field_recall_change", "document_accuracy_change"] {
        assert_eq!(paired[name].as_f64(), Some(0.0));
    }
    assert_eq!(work(&report["total_model_work"]), single.map(|value| value.checked_mul(2).unwrap()));
    assert_eq!(report["total_mask_node_visits"].as_u64(), Some(masks * 2));
    assert_eq!(report["total_source_verification_steps"].as_u64(), Some(verification * 2 - gold));

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

    // Exact source pin and gold schema validation precede nonexistent models.
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
        if index == 0 { row["expected_json"] = serde_json::json!(r#"{"private_amount":"wrong-type","private_currency":"CNY"}"#); }
        rows.extend(canonjson::canonical_bytes(&row).unwrap()); rows.push(b'\n');
    }
    let mut manifest = fixture.manifest.clone(); manifest["expected_input_sha256"] = serde_json::json!(Sha256Digest::of_bytes(&rows));
    let suite = fixture.root.join("bad-gold-suite.json"); let cases = fixture.root.join("bad-gold.ndjson");
    write(&suite, &canonjson::canonical_bytes(&manifest).unwrap()); write(&cases, &rows);
    let mut options = fixture.options(&source, false); options.args.common.suite = Some(suite); options.args.common.observations = cases;
    options.args.snapshot = Some(fixture.root.join("must-not-open-model"));
    let (code, report, errors) = fixture.run(options, "refused-gold-schema.json");
    assert_eq!(code, ErrorCode::InputDecodeOrParse.as_process_exit()); assert!(report.is_none()); assert!(!errors.is_empty());
    assert_eq!(fs::read(&fixture.cases).unwrap(), fixture.rows);
    println!("{COMPLETE}");
}
