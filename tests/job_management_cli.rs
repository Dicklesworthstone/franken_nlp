//! Actual compiled CLI/storage integration; no model or synthetic native driver.
#![cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
use franken_nlp::{canonjson, jobs::{FrozenManifest, JobContract, JobId, JobInput, JobLimits,
    JobSecret, JobWork, OwnedJob}, execution_identity::{ExecutionIdentity, Sha256Digest, NumericsProfile, ThinkingMode, ToolMode},
    native_engine::decode::{DecodeCancellationKind, DecodeStepControl}};
use std::{fs, os::unix::fs::{DirBuilderExt, PermissionsExt}, path::{Path, PathBuf},
    process::{Command, Stdio}, sync::atomic::{AtomicU64, Ordering}};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Control;
impl DecodeStepControl for Control { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn input(id: &str) -> JobInput<'_> { JobInput { id, original: b"private original source", normalized: b"private original source" } }
fn fixture() -> PathBuf {
    let root = std::env::temp_dir().join(format!("fnlp-management-cli-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
    let key = JobSecret::from_bytes([17; 32]);
    let limits = JobLimits { max_items: 4, max_id_bytes: 128, max_input_bytes_per_item: 65536,
        max_snapshot_bytes: 1 << 20, max_result_bytes: 16384, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 32 << 20, max_attempts: 4, max_work: JobWork::default() };
    let d = Sha256Digest::of_bytes(b"stored-cli-fixture");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(),
        logical_model_digest: d, artifact_format: "fixture".to_owned(), quant_recipe: "fixture".to_owned(),
        packing_set_digest: d, tokenizer_digest: d, template_digest: d, task_spec: "fixture-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::DiagnosticF32, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d,
        decision_policy_digest: d, backend_semantic_version: "fixture-v1".to_owned(), host_class: None, compiler_identity: None };
    let manifest = FrozenManifest::freeze(&key, JobContract { job_id: JobId([7; 16]), execution: &identity,
        recipe: &"private recipe never required by inspection", limits }, [input("a"), input("b")], &mut Control).unwrap();
    let mut job = OwnedJob::create(&root, key, manifest, &mut Control).unwrap();
    for id in ["a", "b"] {
        job.begin(&input(id), JobWork::default(), &mut Control).unwrap()
            .commit(&"private committed output", &mut Control).unwrap();
    }
    drop(job);
    fs::write(root.join("protected.key"), [17; 32]).unwrap();
    fs::set_permissions(root.join("protected.key"), fs::Permissions::from_mode(0o600)).unwrap();
    fs::write(root.join("limits.json"), canonjson::canonical_bytes(&limits).unwrap()).unwrap();
    root
}
fn command(root: &Path, operation: &str, memory: u64) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_fnlp"));
    command.args(["job", operation, "--job-id", "07070707070707070707070707070707", "--json"])
        .arg("--job-dir").arg(root).arg("--key-file").arg(root.join("protected.key"))
        .arg("--limits").arg(root.join("limits.json")).arg("--memory-bytes").arg(memory.to_string())
        .stdin(Stdio::null());
    command
}
#[test]
fn real_cli_authenticates_and_publishes_without_original_inputs_or_a_model() {
    let root = fixture();
    for operation in ["status", "verify", "materialize"] {
        let result = command(&root, operation, 256 * 1024 * 1024).output().unwrap();
        assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
        assert!(result.stderr.is_empty());
        let text = std::str::from_utf8(&result.stdout).unwrap();
        let report = canonjson::parse_str(text).unwrap();
        assert_eq!(report["report"]["committed"], 2); assert_eq!(report["report"]["attempts"], 2);
        assert!(!text.contains("private committed output")); assert!(!text.contains("private original source"));
    }
    assert_eq!(fs::read(root.join("materialized.ndjson")).unwrap(), b"\"private committed output\"\n\"private committed output\"\n");
    let again = command(&root, "materialize", 256 * 1024 * 1024).output().unwrap();
    assert!(again.status.success());
}
#[test]
fn actual_process_budget_refuses_before_any_job_mutation() {
    let root = fixture(); let before = fs::read(root.join("results.spool")).unwrap();
    let result = command(&root, "materialize", 1).output().unwrap();
    assert_eq!(result.status.code(), Some(9)); assert!(result.stdout.is_empty());
    assert!(!root.join("materialized.ndjson").exists());
    assert_eq!(fs::read(root.join("results.spool")).unwrap(), before);
}
#[test]
fn root_dispatch_redacts_invalid_job_arguments_and_schema_never_opens_files() {
    let result = Command::new(env!("CARGO_BIN_EXE_fnlp"))
        .args(["job", "status", "--secret", "PRIVATE_SENTINEL_7"]).output().unwrap();
    assert_eq!(result.status.code(), Some(2)); assert!(result.stdout.is_empty());
    assert!(!String::from_utf8_lossy(&result.stderr).contains("PRIVATE_SENTINEL_7"));
    let schema = Command::new(env!("CARGO_BIN_EXE_fnlp")).args(["job", "schema"])
        .stdin(Stdio::null()).output().unwrap();
    assert!(schema.status.success()); assert!(schema.stderr.is_empty());
    let data = canonjson::parse_str(std::str::from_utf8(&schema.stdout).unwrap()).unwrap();
    assert_eq!(data["can_execute_or_resume"], false);
}
