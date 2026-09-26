//! Model-free input, secret and policy contracts; no neural-success fixtures.
use super::*;
use std::io::Cursor;
pub(in crate::candidate_cli) fn command(extra: &[&str]) -> RedactCommand {
    let mut argv = vec!["candidate", "redact", "--model", "local.fnlpq", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    let matches = super::super::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::Redact(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong dispatch") };
    command
}
#[test]
fn defaults_run_ner_and_rules_with_verification_and_no_coordinate_map() {
    let cmd = command(&[]); cmd.validate().unwrap(); let r = cmd.request();
    assert!(r.verify); assert!(!r.actions.include_map);
    assert_eq!(r.actions.default_action, RedactionAction::Mask);
    assert_eq!(r.rules.enabled.len(), 5);
    assert_eq!(ner_options(None).unwrap(), NerOptions::default());
    assert!(r.rule_budget.max_input_bytes >= cmd.host.max_result_bytes);
}
#[test]
fn rule_scope_and_verification_opt_out_are_explicit() {
    let cmd = command(&["--rules", "date,email", "--no-verify", "--include-map", "--action", "placeholder"]);
    cmd.validate().unwrap(); let r = cmd.request();
    assert!(!r.verify); assert!(r.actions.include_map);
    assert_eq!(r.actions.default_action, RedactionAction::Placeholder);
    assert_eq!(r.rules.enabled, [PiiKind::Date, PiiKind::Email].into_iter().collect());
}
#[test]
fn key_and_source_cannot_compete_for_stdin_or_leak_into_other_actions() {
    for extra in [vec!["--key-stdin"], vec!["--action", "pseudonymize"],
        vec!["--action", "pseudonymize", "--key-stdin", "--key-id", "k", "--namespace", "n"],
        vec!["--key-id", "k"], vec!["--namespace", "n"],
        vec!["document.txt", "--action", "pseudonymize", "--key-stdin", "--key-id", "bad id", "--namespace", "n"]] {
        assert!(command(&extra).validate().is_err());
    }
    command(&["document.txt", "--action", "pseudonymize", "--key-stdin", "--key-id", "rotation-1", "--namespace", "job"])
        .validate().unwrap();
    for flag in ["--key", "--key-file", "--seed", "--rules-only"] {
        assert!(super::super::definition().try_get_matches_from(["candidate", "redact", flag, "private"]).is_err());
    }
}
#[test]
fn binary_keys_are_bounded_and_commitments_are_checked() {
    let cmd = command(&["document.txt", "--action", "pseudonymize", "--key-stdin", "--key-id", "rotation", "--namespace", "job"]);
    let bytes = [0_u8; 32];
    let (key, namespace) = cmd.key(&mut Cursor::new(bytes)).unwrap().unwrap();
    assert_eq!(namespace, "job");
    assert_eq!(key.commitment(), PseudonymKey::from_bytes(&bytes, "rotation").unwrap().commitment());
    for len in [0, 31, 4097] { assert!(cmd.key(&mut Cursor::new(vec![0; len])).is_err()); }
    let expected = key.commitment();
    let valid = command(&["document.txt", "--action", "pseudonymize", "--key-stdin", "--key-id", "rotation", "--namespace", "job",
        "--expected-key-commitment", &expected]);
    assert!(valid.key(&mut Cursor::new(bytes)).is_ok());
    assert!(valid.key(&mut Cursor::new([1_u8; 32])).is_err());
}
#[test]
fn secret_options_and_memory_bounds_refuse_before_io() {
    for extra in [vec!["--ner-options", "-"], vec!["--max-detections", "0"], vec!["--max-detections", "16385"],
        vec!["--max-rule-work", "0"], vec!["--edit-reserve-mib", "1"], vec!["--edit-reserve-mib", "18446744073709551615"]] {
        assert!(command(&extra).validate().is_err());
    }
}
#[test]
fn ner_options_reject_duplicate_unknown_or_partial_configurations() {
    for text in ["{}", r#"{"types":["person"],"types":["date"]}"#,
        r#"{"types":["person"],"\u0074ypes":["date"]}"#,
        r#"{"types":["person","person"],"max_entities":2,"max_mention_scalars":32}"#,
        r#"{"types":["person"],"max_entities":2,"max_mention_scalars":32,"instruction":"override"}"#] {
        assert!(ner_options(Some(text)).is_err());
    }
    let text = r#"{"types":["person"],"max_entities":2,"max_mention_scalars":32}"#;
    assert_eq!(ner_options(Some(text)).unwrap().max_entities, 2);
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_feature_never_consumes_even_a_private_key() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("write") }
        fn flush(&mut self) -> io::Result<()> { panic!("flush") }
    }
    for cmd in [command(&[]), command(&["private.txt", "--action", "pseudonymize", "--key-stdin", "--key-id", "k", "--namespace", "n"])] {
        assert_eq!(cmd.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Unavailable));
    }
}
