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
#[test]
fn document_prompt_morsels_and_workers_preserve_task_and_whole_document_budgets() {
    let serial = extract(&[]);
    let (_, baseline) = serial.validate().unwrap();
    for width in ["1", "2", "7", "64"] {
        for rows in ["1", "4", "64"] {
            let mut c = extract(&["--compute-threads", width, "--prefill-rows", rows]);
            c.host.memory_mib = 16384;
            let (args, limits) = c.validate().unwrap();
            let prefill = c.extraction.prefill().unwrap().unwrap();
            assert_eq!(limits.compute.threads, width.parse::<usize>().unwrap());
            assert_eq!(args.compute.admit_prefill(Some(prefill)).unwrap(), limits.compute);
            assert_eq!(prefill.max_batch_rows, rows.parse::<usize>().unwrap());
            assert_eq!(c.host.task_budget(limits), serial.host.task_budget(baseline));
            assert_eq!(c.work_ceiling(), serial.work_ceiling());
            assert_eq!(c.max_total_mask_node_visits, serial.max_total_mask_node_visits);
            assert_eq!(c.extraction_mapping().unwrap().verification, serial.extraction_mapping().unwrap().verification);
        }
    }
}
#[test]
fn invalid_or_unfunded_document_schedules_refuse_before_private_io() {
    for flags in [vec!["--compute-threads", "0"], vec!["--compute-threads", "65"],
        vec!["--prefill-rows", "0"], vec!["--prefill-rows", "65"],
        vec!["--compute-threads", "4", "--prefill-rows", "4", "--max-weight-mib", "8192"]] {
        assert_eq!(extract(&flags).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    }
    for task in ["ner", "keyphrases", "summarize"] {
        assert_eq!(command(task, &["--compute-threads", "2"]).execute(&mut NoRead, &mut NoWrite),
            Err(CandidateError::Arguments));
    }
}
#[test]
fn scheduled_document_peak_accounts_for_prompt_workers_and_the_complete_frontier() {
    let mut c = extract(&["--compute-threads", "4", "--prefill-rows", "4"]);
    c.host.memory_mib = 16384;
    let (args, mut limits) = c.validate().unwrap();
    let peak = c.extraction.schedule_peak_bytes(&c, &args, limits).unwrap();
    limits.memory_bytes = peak;
    c.extraction.check_schedule_memory(&c, &args, limits).unwrap();
    limits.memory_bytes -= 1;
    assert_eq!(c.extraction.check_schedule_memory(&c, &args, limits), Err(CandidateError::Arguments));
    let mut no_children = limits; no_children.compute.workspace_bytes = 0;
    assert_eq!(peak - c.extraction.schedule_peak_bytes(&c, &args, no_children).unwrap(),
        limits.compute.workspace_bytes);
    let mut larger = c;
    larger.max_chunks += 1;
    let per_chunk = larger.host.max_result_bytes as u64 * 4 + args.max_new_tokens as u64 * 16;
    assert_eq!(larger.extraction.schedule_peak_bytes(&larger, &args, limits).unwrap() - peak, per_chunk);
    larger.reduction_reserve_mib = u64::MAX;
    assert_eq!(larger.extraction.schedule_peak_bytes(&larger, &args, limits), Err(CandidateError::Arguments));
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_runtime_never_opens_schema_source_model_or_output() {
    assert_eq!(extract(&["--source-membership"]).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Unavailable));
}
