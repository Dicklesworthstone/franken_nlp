//! CLI consent, immutable limits and no-IO boundaries, not native execution.
use super::*;
use crate::{candidate_cli::CandidateCommand, jobs::JobWork};
use std::io::{self, Read, Write};

pub(in crate::candidate_cli) fn command(extra: &[&str]) -> RedactCommand {
    let mut args = vec!["--ndjson", "--store-results", "--job-dir", "protected",
        "--job-id", "0123456789abcdef0123456789abcdef", "--job-key-file", "protected.key", "--job-limits", "limits.json"];
    args.extend_from_slice(extra);
    crate::candidate_cli::redact::tests::command(&args)
}
pub(in crate::candidate_cli) fn lifetime(c: &RedactCommand) -> JobLimits {
    let (_, l) = c.validate().unwrap(); let e = c.corpus.envelope(c, l).unwrap();
    JobLimits { max_items: 10, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 1 << 20, max_result_bytes: c.host.max_result_bytes,
        max_spool_bytes: 16 << 20, max_materialized_bytes: 16 << 20, max_journal_bytes: 1 << 20,
        max_attempts: 20, max_work: JobWork { model: e.max_model_work, mask_node_visits: e.max_mask_visits } }
}
struct NoIo;
impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("input/key read") } }
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("private output write") }
    fn flush(&mut self) -> io::Result<()> { panic!("output flush") }
}
#[test]
fn defaults_retain_neither_single_documents_nor_live_corpora() {
    for args in [vec![], vec!["--ndjson"], vec!["--ndjson", "--chunked"]] {
        let c = crate::candidate_cli::redact::tests::command(&args);
        c.validate().unwrap(); assert!(!c.retention.store_results); assert!(!c.retention.resume);
    }
}
#[test]
fn retention_requires_explicit_consent_and_the_complete_storage_contract() {
    let base = ["--ndjson", "--store-results", "--job-dir", "protected", "--job-id",
        "0123456789abcdef0123456789abcdef", "--job-key-file", "protected.key", "--job-limits", "limits.json"];
    let parse = |args: &[&str]| {
        let mut argv = vec!["candidate", "redact", "--model", "m", "--memory-mib", "8192"];
        argv.extend_from_slice(args); crate::candidate_cli::definition().try_get_matches_from(argv)
    };
    assert!(parse(&base).is_ok());
    for range in [0..1, 1..2, 2..4, 4..6, 6..8, 8..10] {
        let args: Vec<_> = base.iter().enumerate().filter(|(i, _)| !range.contains(i)).map(|(_, &s)| s).collect();
        assert!(parse(&args).is_err());
    }
    let mut repair = base.to_vec(); repair.push("--discard-uncommitted-tail");
    assert!(parse(&repair).is_err()); repair.push("--resume"); assert!(parse(&repair).is_ok());
}
#[test]
fn start_resume_and_chunking_do_not_rewrite_redaction_policy_or_native_work() {
    for chunked in [false, true] {
        let extra: &[&str] = if chunked { &["--chunked"] } else { &[] };
        let start = command(extra);
        let mut extra = extra.to_vec(); extra.push("--resume"); let resume = command(&extra);
        let (_, a) = start.validate().unwrap(); let (_, b) = resume.validate().unwrap();
        assert_eq!(canonjson::canonical_bytes(&start.request()).unwrap(), canonjson::canonical_bytes(&resume.request()).unwrap());
        assert_eq!(start.corpus.envelope(&start, a).unwrap().max_model_work,
            resume.corpus.envelope(&resume, b).unwrap().max_model_work);
        assert_eq!(start.retention.job_id, resume.retention.job_id);
    }
}
#[test]
fn invalid_paths_repairs_and_memory_authority_refuse_without_any_io() {
    for axis in 0..7 {
        let mut c = command(&[]);
        match axis { 0 => c.retention.job_dir = Some("-".into()), 1 => c.retention.job_key_file = Some("".into()),
            2 => c.retention.job_limits = None, 3 => c.retention.discard_uncommitted_tail = true,
            4 => c.retention.max_job_input_lines = Some(0), 5 => c.retention.journal_memory_mib = Some(u64::MAX),
            _ => c.retention.serialization_memory_mib = Some(0) }
        assert!(c.validate().is_err());
        let mut diagnostics = Vec::new();
        assert_ne!(c.run_owned(NoIo, NoIo, &mut diagnostics), ExitCode::SUCCESS);
        assert!(String::from_utf8(diagnostics).unwrap().contains("Durable progress may exist"));
    }
}
#[test]
fn immutable_limits_cannot_spend_transformed_text_headroom_on_original_input() {
    let c = command(&[]); let (_, l) = c.validate().unwrap(); let e = c.corpus.envelope(&c, l).unwrap();
    let limits = lifetime(&c); c.retention.check_limits(&c, e, limits).unwrap();
    assert!(c.host.max_result_bytes > c.host.max_input_bytes);
    for axis in 0..6 {
        let mut bad = limits;
        match axis { 0 => bad.max_items = e.transport.max_requests + 1,
            1 => bad.max_id_bytes = e.transport.max_id_bytes + 1,
            2 => bad.max_input_bytes_per_item = c.host.max_input_bytes + 1,
            3 => bad.max_snapshot_bytes = e.transport.max_input_bytes + 1,
            4 => bad.max_result_bytes = c.host.max_result_bytes - 1,
            _ => bad.max_spool_bytes = e.output_bytes }
        assert!(c.retention.check_limits(&c, e, bad).is_err());
    }
}
#[test]
fn job_authentication_never_substitutes_for_the_pseudonym_key_source() {
    let mut c = command(&["--action", "pseudonymize"]); assert!(c.validate().is_err());
    c = command(&["input.ndjson", "--action", "pseudonymize", "--key-stdin", "--key-id", "pii-v1", "--namespace", "corpus"]);
    c.validate().unwrap();
    assert!(c.key(&mut &[][..]).is_err());
    let (key, scope) = c.key(&mut &[7; 32][..]).unwrap().unwrap();
    assert_eq!(scope, "corpus"); assert!(!key.commitment().is_empty());
    assert_eq!(c.retention.job_key_file.as_ref().unwrap(), &PathBuf::from("protected.key"));
}
#[test]
fn job_identifiers_are_exact_and_no_key_bytes_enter_arguments() {
    for text in ["", "ABCDABCDABCDABCDABCDABCDABCDABCD", "0123456789abcdef0123456789abcde", "-1", "０１２３"] {
        assert!(parse_job_id(text).is_err());
    }
    let value = "0123456789abcdef0123456789abcdef";
    assert_eq!(crate::candidate_cli::jobs::job_id_hex(parse_job_id(value).unwrap()), value);
    for flag in ["--job-key", "--key", "--rules-only"] {
        assert!(crate::candidate_cli::definition().try_get_matches_from(["candidate", "redact", "--model", "m",
            "--memory-mib", "8192", flag, "secret"]).is_err());
    }
    let m = crate::candidate_cli::definition().try_get_matches_from(["candidate", "redact", "--model", "m", "--memory-mib", "8192"]).unwrap();
    assert!(matches!(CandidateCommand::from_matches(&m).unwrap(), CandidateCommand::Redact(_)));
}
#[cfg(not(all(feature = "asupersync-runtime", feature = "metadata-store", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unavailable_profile_refuses_before_job_or_pseudonym_keys_and_output() {
    for args in [vec![], vec!["--chunked", "--resume"],
        vec!["input.ndjson", "--action", "pseudonymize", "--key-stdin", "--key-id", "pii", "--namespace", "scope"]] {
        let c = command(&args); c.validate().unwrap();
        assert_ne!(c.run_owned(NoIo, NoIo, &mut Vec::new()), ExitCode::SUCCESS);
    }
}
