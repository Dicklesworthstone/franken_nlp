//! Bounded input, private-key and resource fixtures; no synthetic neural success.
use super::*;
use std::io::Cursor;
use crate::tasks::redact::{PiiKind, pseudonym::{Pseudonyms, PseudonymEncoding}};

fn parsed(flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", "redact", "--memory-mib", "8192"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("wrong INT4 redaction dispatcher"); };
    Ok(command)
}
pub(super) fn command(extra: &[&str]) -> Int4Command {
    let mut flags = vec!["--source-dir", "explicit-source", "--source-manifest", "explicit-manifest"];
    flags.extend_from_slice(extra); parsed(&flags).unwrap()
}
pub(super) fn options(command: &Int4Command) -> &RedactArgs {
    let Int4Task::Redact(options) = &command.task else { panic!("redaction required"); };
    options
}
struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refusal consumed document or private key"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refusal published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refusal flushed output"); }
}

#[test]
fn defaults_verify_the_complete_ner_rule_union_on_both_model_input_routes() {
    let source = command(&[]); let selected = options(&source);
    let limits = selected.validate(&source.args).unwrap();
    let request = selected.request(&source.args, limits);
    assert!(request.verify);
    assert!(!request.actions.include_map);
    assert_eq!(request.actions.default_action, RedactionAction::Mask);
    assert_eq!(request.rules.enabled.len(), 5);
    assert_eq!(shared::ner_options(None).unwrap(), crate::tasks::ner::NerOptions::default());
    assert!(request.rule_budget.max_input_bytes >= limits.common.result_bytes);
    assert_eq!(selected.source.budget(&source.args, limits.common).max_kv_bytes, limits.common.native.kv_bytes);
    assert_eq!(source.args.max_new_tokens, 512);
    assert_eq!(source.args.preparation_mib, 512);
    let pin = "ab".repeat(32);
    let reload = parsed(&["--snapshot", "explicit.snapshot", "--expected-model-sha256", &pin]).unwrap();
    let reloaded = options(&reload).validate(&reload.args).unwrap();
    assert_eq!(reloaded.common.snapshot_pin.unwrap().model_content_sha256, [0xab; 32]);
    assert_eq!(reloaded.common.native.kv_bytes, limits.common.native.kv_bytes);
    for invalid in [vec![], vec!["--snapshot", "s"], vec!["--model", "int8.fnlpq"],
        vec!["--snapshot", "s", "--expected-model-sha256", &pin, "--source-dir", "d"],
        vec!["--snapshot", "s", "--expected-model-sha256", &pin, "--save-snapshot", "new"]] {
        assert!(parsed(&invalid).is_err());
    }
}

#[test]
fn transformed_verification_rule_scope_action_and_sensitive_map_are_explicit() {
    let command = command(&["--rules", "date,email", "--no-verify", "--include-map", "--action", "placeholder"]);
    let options = options(&command); let limits = options.validate(&command.args).unwrap();
    let request = options.request(&command.args, limits);
    assert!(!request.verify);
    assert!(request.actions.include_map);
    assert_eq!(request.actions.default_action, RedactionAction::Placeholder);
    assert_eq!(request.rules.enabled, [PiiKind::Date, PiiKind::Email].into_iter().collect());
    assert_eq!(request.edit_budget.max_output_bytes, limits.common.result_bytes);
    assert_eq!(request.edit_budget.max_regions, 4096);
    assert_eq!(request.rule_budget.max_work, 128 * 1024 * 1024);
    assert_eq!(format!("{:?}", options), "RedactArgs");
}

#[test]
fn private_key_scope_and_source_stdin_conflicts_refuse_before_any_io() {
    for flags in [vec!["--key-stdin"], vec!["--action", "pseudonymize"],
        vec!["--action", "pseudonymize", "--key-stdin", "--key-id", "k", "--namespace", "n"],
        vec!["--key-id", "k"], vec!["--namespace", "n"], vec!["--expected-key-commitment", "ab"],
        vec!["document.txt", "--action", "pseudonymize", "--key-stdin", "--key-id", "bad id", "--namespace", "n"],
        vec!["document.txt", "--action", "pseudonymize", "--key-stdin", "--key-id", "k", "--namespace", ""]] {
        assert_eq!(command(&flags).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
    let valid = command(&["document.txt", "--action", "pseudonymize", "--key-stdin",
        "--key-id", "rotation-1", "--namespace", "customer-set"]);
    options(&valid).validate(&valid.args).unwrap();
    for flag in ["--key", "--key-file", "--rules-only", "--chunked", "--ndjson", "--store-results",
        "--resume", "--job-dir", "--seed", "--logprobs", "--prefill-rows"] {
        let flags = ["--source-dir", "d", "--source-manifest", "m", flag, "private"];
        assert!(parsed(&flags).is_err());
    }
}

#[test]
fn bounded_binary_key_and_commitment_contract_is_shared_with_int8() {
    let command = command(&["document.txt", "--action", "pseudonymize", "--key-stdin",
        "--key-id", "rotation", "--namespace", "scope"]);
    let bytes = [0x80_u8; 32];
    let (key, namespace) = options(&command).key(&mut Cursor::new(bytes)).unwrap().unwrap();
    let legacy = crate::candidate_cli::redact::tests::command(&["document.txt", "--action", "pseudonymize",
        "--key-stdin", "--key-id", "rotation", "--namespace", "scope"]);
    let (legacy_key, legacy_namespace) = legacy.key(&mut Cursor::new(bytes)).unwrap().unwrap();
    assert_eq!(namespace, legacy_namespace);
    assert_eq!(key.commitment(), legacy_key.commitment());
    for count in [0, 31, 4097] { assert!(options(&command).key(&mut Cursor::new(vec![0; count])).is_err()); }
    let expected = key.commitment();
    let checked = command_with_commitment(&expected);
    assert!(options(&checked).key(&mut Cursor::new(bytes)).is_ok());
    assert!(matches!(options(&checked).key(&mut Cursor::new([1_u8; 32])), Err(CandidateError::Identity)));
    let context = Pseudonyms::full256(&key, &namespace, Some(&expected)).unwrap();
    assert_eq!(context.identity().encoding, PseudonymEncoding::Full256);
}
fn command_with_commitment(expected: &str) -> Int4Command {
    command(&["missing-document.txt", "--action", "pseudonymize", "--key-stdin",
        "--key-id", "rotation", "--namespace", "scope", "--expected-key-commitment", expected])
}

#[test]
fn typed_ner_options_cannot_override_prompts_or_budgets_or_consume_key_stdin() {
    let selected = command(&["--ner-options", "entities.json"]);
    assert_eq!(options(&selected).source.options_path().unwrap(), std::path::Path::new("entities.json"));
    for value in ["-", ""] {
        let command = command(&["--ner-options", value]);
        assert!(options(&command).validate(&command.args).is_err());
    }
    for text in ["{}", r#"{"types":["person"],"types":["date"]}"#,
        r#"{"types":["person"],"\u0074ypes":["date"]}"#,
        r#"{"types":["person","person"],"max_entities":2,"max_mention_scalars":32}"#,
        r#"{"types":["person"],"max_entities":2,"max_mention_scalars":32,"prompt":"override"}"#] {
        assert!(shared::ner_options(Some(text)).is_err());
    }
    let text = r#"{"types":["person"],"max_entities":2,"max_mention_scalars":32}"#;
    assert_eq!(shared::ner_options(Some(text)).unwrap().max_entities, 2);
    assert!(shared::ner_options(Some(&" ".repeat(crate::candidate_cli::source::OPTIONS_BYTES + 1))).is_err());
}

#[test]
fn invalid_work_edit_or_native_limits_refuse_before_private_io() {
    for flags in [vec!["--max-detections", "0"], vec!["--max-detections", "16385"],
        vec!["--max-rule-work", "0"], vec!["--max-rule-work", "1000000000001"],
        vec!["--edit-reserve-mib", "1"], vec!["--edit-reserve-mib", "18446744073709551615"],
        vec!["--max-new-tokens", "0"], vec!["--context-tokens", "512"],
        vec!["--max-result-bytes", "0"], vec!["--max-mask-node-visits", "1"],
        vec!["--mask-step-node-visits", "0"], vec!["--group-width", "0"]] {
        assert_eq!(command(&flags).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn memory_admission_prices_captured_source_secrets_two_preparations_and_edit_peak() {
    for width in [3, 128, 10752] {
        let mut command = command(&[]);
        command.args.group_width = width;
        command.args.max_load_mib = 65536; command.args.memory_mib = 65536;
        let selected = options(&command);
        let base = selected.source.validate(&command.args).unwrap();
        let admitted = selected.validate(&command.args).unwrap();
        let task = selected.source.budget(&command.args, base);
        let extra = base.preparation_bytes + command.args.max_input_bytes as u64 + 1 + 4096
            + task.max_output_bytes * 4 + u64::from(task.max_output_tokens) * 8 + 64 * MIB;
        assert_eq!(admitted.common.minimum_memory_bytes, base.minimum_memory_bytes + extra);
        assert_eq!(admitted.edit_bytes, 64 * MIB);
        assert_eq!(task.max_kv_bytes,
            Int4MemoryRequirement::for_context(command.args.context_tokens, width).unwrap().kv_bytes);
        command.args.memory_mib = admitted.common.minimum_memory_bytes.saturating_sub(1) / MIB;
        assert!(options(&command).validate(&command.args).is_err());
        command.args.memory_mib = admitted.common.minimum_memory_bytes.div_ceil(MIB);
        assert!(options(&command).validate(&command.args).is_ok());
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_does_not_consume_document_or_private_key() {
    for command in [command(&[]), command(&["document.txt", "--action", "pseudonymize", "--key-stdin",
        "--key-id", "k", "--namespace", "n"])] {
        assert_eq!(command.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Unavailable));
    }
}

#[cfg(feature = "asupersync-runtime")]
#[test]
#[ignore = "requires controller-admitted pinned sources, a fresh snapshot and an available explicit SIMD kernel"]
fn source_scalar_and_pinned_snapshot_simd_match_after_fresh_redaction_verification() {
    let source = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit pinned source directory required");
    let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit source manifest required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required")
        .parse().expect("memory ceiling must be MiB");
    let snapshot = std::env::var("FNLP_INT4_TEST_REDACT_SNAPSHOT").expect("explicit new snapshot required");
    assert!(!std::path::Path::new(&snapshot).exists(), "fixture requires a fresh path and deletes nothing");
    let kernel: Int8KernelKind = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit SIMD kernel required")
        .parse().expect("canonical kernel ID required");
    assert_ne!(kernel, Int8KernelKind::Scalar);
    PackedInt4Kernel::detect(kernel).expect("requested kernel must be available on the actual host");
    let document = "Contact via alice@example.org.";
    let mut convert = command(&["--rules", "email", "--save-snapshot", &snapshot]);
    convert.args.source_dir = Some(PathBuf::from(source));
    convert.args.source_manifest = Some(PathBuf::from(manifest));
    convert.args.memory_mib = memory;
    let mut bytes = Vec::new(); convert.execute(&mut document.as_bytes(), &mut bytes).unwrap();
    let scalar: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(scalar["output"]["ner_passes"], 2);
    assert_eq!(scalar["output"]["result"]["verification"], "clean_declared_union");
    assert!(!scalar["output"]["result"]["text"].as_str().unwrap().contains("alice@example.org"));
    let pin = scalar["model_content_sha256"].as_str().unwrap();
    let mut reload = parsed(&["--rules", "email", "--snapshot", &snapshot,
        "--expected-model-sha256", pin, "--kernel", kernel.id()]).unwrap();
    reload.args.memory_mib = memory;
    let mut bytes = Vec::new(); reload.execute(&mut document.as_bytes(), &mut bytes).unwrap();
    let selected: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(selected["output"], scalar["output"]);
    assert_eq!(selected["integer_kernel"], kernel.id());
    assert_eq!(selected["snapshot"]["operation"], "loaded");
}
