//! CLI contract fixtures. These do not execute a model or confer snapshot trust.
use super::*;

fn parsed(task: &str, flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", task, "--memory-mib", "8192"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("wrong INT4 command"); };
    Ok(command)
}
fn snapshot(task: &str) -> Int4Command {
    parsed(task, &["--snapshot", "explicit-model.int4snap", "--expected-model-sha256", &"ab".repeat(32)]).unwrap()
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused snapshot request reached input IO"); }
}

#[test]
fn source_and_external_pin_snapshot_modes_are_exclusive_and_complete() {
    let hash = "ab".repeat(32);
    for task in ["generate", "chat"] {
        for flags in [vec!["--snapshot", "s"], vec!["--expected-model-sha256", &hash],
            vec!["--snapshot", "s", "--expected-model-sha256", &hash, "--source-dir", "d"],
            vec!["--snapshot", "s", "--expected-model-sha256", &hash, "--source-manifest", "m"],
            vec!["--snapshot", "s", "--expected-model-sha256", &hash, "--save-snapshot", "new"],
            vec!["--source-dir", "d", "--source-manifest", "m", "--expected-model-sha256", &hash],
            vec!["--save-snapshot", "new"],
        ] { assert!(parsed(task, &flags).is_err()); }
        let command = snapshot(task);
        let pin = command.args.validate().unwrap().snapshot_pin.unwrap();
        assert_eq!(pin.group_width, 128); assert_eq!(pin.model_content_sha256, [0xab; 32]);
        assert!(command.args.source_dir.is_none()); assert!(command.args.source_manifest.is_none());
        let source = parsed(task, &["--source-dir", "d", "--source-manifest", "m", "--save-snapshot", "new"]).unwrap();
        assert!(source.args.validate().unwrap().snapshot_pin.is_none());
    }
}

#[test]
fn malformed_pins_empty_paths_and_programmatic_mode_mixing_refuse_before_io() {
    for hash in ["".to_owned(), "a".repeat(63), "a".repeat(65), "AB".repeat(32), "gg".repeat(32), "auto".to_owned()] {
        let mut command = snapshot("generate"); command.args.expected_model_sha256 = Some(hash);
        assert_eq!(command.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
    }
    for axis in 0..8 {
        let mut command = snapshot("chat");
        match axis {
            0 => command.args.snapshot = Some(PathBuf::new()),
            1 => command.args.snapshot = Some(PathBuf::from("-")),
            2 => command.args.source_dir = Some(PathBuf::from("d")),
            3 => command.args.source_manifest = Some(PathBuf::from("m")),
            4 => command.args.save_snapshot = Some(PathBuf::from("new")),
            5 => command.args.expected_model_sha256 = None,
            6 => command.args.snapshot = None,
            _ => command.args.snapshot_checkpoints = 1,
        }
        assert_eq!(command.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
    }
    for path in [PathBuf::new(), PathBuf::from("-")] {
        let mut args = super::tests::args("generate", &[]); args.save_snapshot = Some(path);
        assert!(args.validate().is_err());
    }
}

#[test]
fn snapshot_storage_does_not_change_sampling_prompt_limits_or_native_memory() {
    let hash = "ab".repeat(32); let seed = "cd".repeat(32);
    let options = ["--seed", &seed, "--top-k", "30", "--top-p-ppm", "900000",
        "--stop", " END", "--logprobs", "--presence-penalty-milli", "250"];
    let source = super::tests::args("generate", &options);
    let mut flags = vec!["--snapshot", "s", "--expected-model-sha256", &hash]; flags.extend_from_slice(&options);
    let restored = parsed("generate", &flags).unwrap().args;
    let a = source.validate().unwrap(); let b = restored.validate().unwrap();
    assert!(source.options(166_101).unwrap() == restored.options(166_101).unwrap());
    assert_eq!(a.max_prompt_tokens, b.max_prompt_tokens);
    assert_eq!(a.result_bytes, b.result_bytes); assert_eq!(a.native, b.native);
    assert_eq!(a.load.weight_payload_bytes, b.load.weight_payload_bytes);
    assert!(b.minimum_memory_bytes <= a.minimum_memory_bytes);
}

#[test]
fn snapshot_budget_derives_only_from_pinned_geometry_and_counts_the_eof_probe() {
    for width in [1, 3, 16, 32, 128, 10752] {
        let mut args = snapshot("generate").args;
        args.group_width = width; args.memory_mib = 65536; args.max_load_mib = 65536;
        let limits = args.validate().unwrap();
        let pin = limits.snapshot_pin.unwrap(); assert_eq!(pin.group_width, width);
        let read = limits.snapshot_budget(true).unwrap(); let write = limits.snapshot_budget(false).unwrap();
        assert_eq!(read.max_io_bytes, write.max_io_bytes + 1);
        assert_eq!(write.max_io_bytes, limits.load.weight_payload_bytes + 128);
        assert_eq!(read.max_scratch_bytes, 65536);
        assert_eq!(read.max_weight_bytes, limits.load.weight_payload_bytes);
        limits.snapshot_need.check(read, true).unwrap(); limits.snapshot_need.check(write, false).unwrap();
        args.max_load_mib = (limits.load.weight_payload_bytes + read.max_scratch_bytes - 1) / MIB;
        assert!(args.validate().is_err());
    }
}

#[test]
fn persistence_checkpoint_budget_is_separate_and_finite() {
    let mut args = super::tests::args("generate", &["--save-snapshot", "new", "--snapshot-checkpoints", "42"]);
    assert!(args.validate().is_ok()); assert_eq!(args.snapshot_checkpoints, 42);
    assert_eq!(args.load_checkpoints, 1_000_000_000); assert_eq!(args.max_checkpoints, 1_000_000_000);
    for count in [0, 1, 1_000_000_000_001, u64::MAX] {
        args.snapshot_checkpoints = count; assert!(args.validate().is_err());
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn feature_disabled_reload_fails_before_any_input_or_snapshot_access() {
    for task in ["generate", "chat"] {
        let mut output = Vec::new();
        assert_eq!(snapshot(task).execute(&mut NoRead, &mut output), Err(CandidateError::Unavailable));
        assert!(output.is_empty());
    }
}

#[cfg(feature = "asupersync-runtime")]
#[test]
fn snapshot_cli_admission_matches_actual_resident_host_charges() {
    use crate::hosted::{Int4LoadRequirements, RunLimits};
    for width in [1, 3, 16, 32, 128, 10752] {
        let mut args = snapshot("generate").args;
        args.group_width = width; args.memory_mib = 65536; args.max_load_mib = 65536;
        let limits = args.validate().unwrap();
        let required = Int4LoadRequirements::for_snapshot(limits.snapshot_pin.unwrap(), limits.snapshot_budget(true).unwrap(),
            ALLOCATOR_BYTES, RunLimits { max_elapsed: std::time::Duration::from_secs(1), max_checkpoints: 100,
                cleanup_reserve_bytes: CLEANUP_BYTES }).unwrap();
        assert_eq!(required.resident_bytes, limits.load.weight_payload_bytes + ALLOCATOR_BYTES);
        assert_eq!(required.staging_bytes, limits.snapshot_need.scratch_bytes);
        assert!(limits.minimum_memory_bytes >= limits.preparation_bytes + required.resident_bytes + required.staging_bytes + CLEANUP_BYTES);
    }
}

#[cfg(feature = "asupersync-runtime")]
#[test]
#[ignore = "requires controller-admitted source/model execution, new snapshot path and disk space"]
fn source_save_then_snapshot_only_reload_keep_model_identity_and_generation() {
    let source = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit source directory");
    let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit source manifest");
    let path = std::env::var("FNLP_INT4_TEST_SNAPSHOT").expect("explicit NEW path; test never deletes it");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling").parse().unwrap();
    let prompt = "Reply with a short greeting.";
    let mut command = parsed("generate", &["--source-dir", &source, "--source-manifest", &manifest,
        "--save-snapshot", &path, "--context-tokens", "256", "--max-new-tokens", "4"]).unwrap();
    command.args.memory_mib = memory;
    let mut output = Vec::new(); command.execute(&mut prompt.as_bytes(), &mut output).unwrap();
    let saved: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(saved["snapshot"]["operation"], "saved");
    let expected = saved["model_content_sha256"].as_str().unwrap();
    // No source flags/path available to this command: failure cannot convert.
    let mut reload = parsed("generate", &["--snapshot", &path, "--expected-model-sha256", expected,
        "--context-tokens", "256", "--max-new-tokens", "4"]).unwrap();
    reload.args.memory_mib = memory;
    let mut output = Vec::new(); reload.execute(&mut prompt.as_bytes(), &mut output).unwrap();
    let restored: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(restored["snapshot"]["operation"], "loaded");
    assert_eq!(saved["snapshot"]["file_bytes"], restored["snapshot"]["file_bytes"]);
    for field in ["model_content_sha256", "source_root_sha256", "candidate_format", "quant_recipe", "group_width", "output"] {
        assert_eq!(saved[field], restored[field], "{field}");
    }
    // Existing destinations are never replaced, even by an identical model.
    let original_len = std::fs::metadata(&path).unwrap().len();
    let mut existing = parsed("generate", &["--source-dir", &source, "--source-manifest", &manifest,
        "--save-snapshot", &path, "--context-tokens", "256", "--max-new-tokens", "4"]).unwrap();
    existing.args.memory_mib = memory;
    let mut output = Vec::new(); assert!(existing.execute(&mut prompt.as_bytes(), &mut output).is_err());
    assert!(output.is_empty()); assert_eq!(std::fs::metadata(path).unwrap().len(), original_len);
}
