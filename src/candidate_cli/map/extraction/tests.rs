//! CLI boundaries and shared budgets only; no native inference claims.
use super::*;
use crate::candidate_cli::{definition as root, map::tests::command};

fn extract(extra: &[&str]) -> MapCommand {
    let mut flags = vec!["--schema", "schema.json"]; flags.extend_from_slice(extra);
    command("extract", &flags)
}
#[test]
fn extraction_requires_schema_model_and_memory_and_keeps_both_grounding_modes() {
    for extra in [&[][..], &["--source-membership"][..]] {
        let c = extract(extra); c.validate().unwrap();
        assert_eq!(c.kind().unwrap(), BuiltInTask::Extract);
        assert_eq!(c.extraction.source_membership, !extra.is_empty());
        assert!(c.map_task(None).is_err());
    }
    for argv in [vec!["candidate", "map", "--task", "extract", "--model", "m", "--memory-mib", "8192"],
        vec!["candidate", "map", "--task", "extract", "--schema", "s", "--memory-mib", "8192"],
        vec!["candidate", "map", "--task", "extract", "--schema", "s", "--model", "m"]] {
        assert!(root().try_get_matches_from(argv).is_err());
    }
    assert!(command("extract", &["--schema", "-"]).validate().is_err());
}
#[test]
fn extraction_flags_cannot_silently_change_other_map_tasks() {
    for task in ["ner", "keyphrases", "summarize"] {
        assert!(command(task, &["--schema", "s"]).validate().is_err());
        assert!(command(task, &["--schema", "s", "--source-membership"]).validate().is_err());
    }
    assert!(command("answer", &["--question", "q", "--schema", "s"]).validate().is_err());
    assert!(extract(&["--question", "q"]).validate().is_err());
    assert!(extract(&["--reduce-summary"]).validate().is_err());
    assert!(root().try_get_matches_from(["candidate", "map", "--task", "extract", "--schema", "s",
        "--options", "o", "--model", "m", "--memory-mib", "8192"]).is_err());
    // Defense in depth even when a crate-internal caller mutates parsed data.
    let mut c = extract(&[]); c.options = Some(PathBuf::from("o"));
    assert!(c.validate().is_err()); assert!(c.extraction_mapping().is_err());
}
#[test]
fn verification_options_require_schema_and_have_explicit_finite_domains() {
    for flag in ["--max-extraction-fields", "--max-extraction-evidence-spans", "--max-extraction-scan-steps"] {
        assert!(root().try_get_matches_from(["candidate", "map", "--task", "ner", "--model", "m",
            "--memory-mib", "8192", flag, "1"]).is_err());
        assert!(extract(&[flag, "0"]).validate().is_err());
        extract(&[flag, "1"]).validate().unwrap();
    }
    for flag in ["--max-extraction-fields", "--max-extraction-evidence-spans"] {
        extract(&[flag, "1000000"]).validate().unwrap();
        assert!(extract(&[flag, "1000001"]).validate().is_err());
    }
    extract(&["--max-extraction-scan-steps", "1000000000000"]).validate().unwrap();
    assert!(extract(&["--max-extraction-scan-steps", "1000000000001"]).validate().is_err());
}
#[test]
fn mapping_keeps_all_document_ceilings_without_renewing_per_chunk() {
    let a = extract(&[]).extraction_mapping().unwrap();
    let b = extract(&["--max-chunks", "256", "--preparation-mib", "1024"]).extraction_mapping().unwrap();
    assert_eq!(a.mapping.max_model_work, b.mapping.max_model_work);
    assert_eq!(a.mapping.max_mask_visits, b.mapping.max_mask_visits);
    assert_eq!(a.verification, b.verification);
    assert_eq!(a.mapping.chunks.reserved_tokens, 512);
    assert_eq!(a.mapping.chunks.context_tokens, 2048);
    let c = extract(&["--max-extraction-fields", "17", "--max-extraction-evidence-spans", "19",
        "--max-extraction-scan-steps", "23", "--max-chunk-bytes", "32"]).extraction_mapping().unwrap();
    assert_eq!(c.verification.max_fields, 17); assert_eq!(c.verification.max_matches, 19);
    assert_eq!(c.verification.max_scan_steps, 23); assert_eq!(c.mapping.chunks.max_chunk_bytes, 32);
}
#[test]
fn complete_schema_prompt_storage_is_an_additional_preparation_charge() {
    let c = extract(&[]);
    assert_eq!(c.extraction.schema_reserve_per_chunk(), 2 * schema::SCHEMA_BYTES as u64);
    assert_eq!(command("ner", &[]).extraction.schema_reserve_per_chunk(), 0);
    command("ner", &["--max-chunks", "101"]).validate().unwrap();
    assert!(extract(&["--max-chunks", "101"]).validate().is_err());
    extract(&["--max-chunks", "101", "--preparation-mib", "1024"]).validate().unwrap();
}
#[test]
fn help_names_independent_json_and_required_field_limits() {
    let help = super::super::definition().render_long_help().to_string();
    assert!(help.contains("--task extract --schema FILE"));
    assert!(help.contains("independent exact-schema JSON strings"));
    assert!(help.contains("Required fields must be satisfiable in each chunk"));
    assert!(help.contains("No partial success"));
}
struct NoRead;
impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("source read") } }
struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output written") }
    fn flush(&mut self) -> io::Result<()> { panic!("output flushed") }
}
#[test]
fn invalid_schema_path_is_refused_before_input_or_output() {
    assert_eq!(command("extract", &["--schema", "-"]).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_runtime_never_opens_schema_source_model_or_output() {
    assert_eq!(extract(&["--source-membership"]).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Unavailable));
}
