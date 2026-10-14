//! CLI/admission fixtures. Only the explicitly ignored test loads a real model.
use super::*;
fn parsed(task: &str, flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", task, "--source-dir", "d",
        "--source-manifest", "m", "--memory-mib", "8192"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("wrong INT4 dispatcher"); };
    Ok(command)
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused prefill reached input"); }
}
#[test]
fn text_prefill_is_explicit_bounded_and_never_accepted_for_classification() {
    for task in ["generate", "chat"] {
        assert!(parsed(task, &[]).unwrap().args.validate().unwrap().prefill.is_none());
        for rows in [1, 3, 4, 64] {
            let command = parsed(task, &["--prefill-rows", &rows.to_string()]).unwrap();
            let need = command.args.validate().unwrap().prefill.unwrap();
            assert_eq!(need.max_batch_rows, rows);
            assert_eq!(need.max_extra_scratch_bytes, Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap());
        }
        for rows in [0, 65, usize::MAX] {
            let command = parsed(task, &["--prefill-rows", &rows.to_string()]).unwrap();
            let mut output = Vec::new();
            assert_eq!(command.execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
            assert!(output.is_empty());
        }
    }
    for rows in [1, 4, 64] { assert!(parsed("classify", &["--prefill-rows", &rows.to_string()]).is_err()); }
    // Programmatic construction must not bypass the scored parser's refusal.
    let scoring = match parsed("classify", &[]).unwrap().task {
        Int4Task::Classify(scoring) => scoring, _ => unreachable!(),
    };
    let args = parsed("generate", &["--prefill-rows", "4", "--preparation-mib", "512"]).unwrap().args;
    let command = Int4Command { task: Int4Task::Classify(scoring), args };
    assert_eq!(command.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
}
fn set_mode(args: &mut Int4Args, mode: usize) {
    match mode {
        0 => (),
        1 => {
            args.source_dir = None; args.source_manifest = None;
            args.snapshot = Some(PathBuf::from("explicit-snapshot"));
            args.expected_model_sha256 = Some("ab".repeat(32));
        }
        _ => args.save_snapshot = Some(PathBuf::from("new-snapshot")),
    }
}
fn expected_floor(args: &Int4Args, limits: Int4Limits) -> u64 {
    let load = if limits.snapshot_pin.is_some() { limits.snapshot_need.scratch_bytes }
        else { limits.source_preparation_bytes + limits.load.scratch_payload_bytes };
    let save = if args.save_snapshot.is_some() { limits.snapshot_need.scratch_bytes } else { 0 };
    let execute = limits.native.kv_bytes + limits.native.rope_bytes + limits.native.scratch_payload_bound
        + SAMPLER_BYTES + ALLOCATOR_BYTES + limits.result_bytes as u64 * 4 + args.max_new_tokens as u64 * 8 + 4096
        + limits.prefill.map(|p| p.max_extra_scratch_bytes).unwrap_or(0);
    limits.preparation_bytes + limits.load.weight_payload_bytes + ALLOCATOR_BYTES
        + load.max(save).max(execute) + CLEANUP_BYTES
}
#[test]
fn process_floor_prices_only_simultaneous_extra_workspace_for_every_model_input_mode() {
    for mode in 0..3 { for width in [1, 3, 128, 10752] { for context in [128, 2048] {
        let mut base = parsed("generate", &[]).unwrap().args;
        base.group_width = width; base.context_tokens = context; base.memory_mib = 65536; base.max_load_mib = 65536;
        set_mode(&mut base, mode);
        let sequential = base.validate().unwrap();
        assert_eq!(sequential.minimum_memory_bytes, expected_floor(&base, sequential));
        for rows in [1, 4, 64] {
            let mut grouped = parsed("generate", &["--prefill-rows", &rows.to_string()]).unwrap().args;
            grouped.group_width = width; grouped.context_tokens = context;
            grouped.memory_mib = 65536; grouped.max_load_mib = 65536; set_mode(&mut grouped, mode);
            let actual = grouped.validate().unwrap();
            assert_eq!(actual.minimum_memory_bytes, expected_floor(&grouped, actual));
            assert_eq!(actual.preparation_bytes, sequential.preparation_bytes);
            assert_eq!(actual.source_preparation_bytes, sequential.source_preparation_bytes);
            assert_eq!(actual.load, sequential.load); assert_eq!(actual.native, sequential.native);
            assert_eq!(actual.snapshot_pin, sequential.snapshot_pin);
            assert_eq!(actual.snapshot_need.file_bytes, sequential.snapshot_need.file_bytes);
            assert!(actual.minimum_memory_bytes >= sequential.minimum_memory_bytes);
            // Exact boundary before IO, not a weight-only or per-morsel limit.
            grouped.memory_mib = (actual.minimum_memory_bytes - 1) / MIB;
            assert!(grouped.validate().is_err());
            grouped.memory_mib = actual.minimum_memory_bytes.div_ceil(MIB);
            assert!(grouped.validate().is_ok());
        }
    } } }
}
#[test]
fn scheduling_keeps_seed_stops_penalties_and_checked_kernel_unchanged() {
    let seed = "ab".repeat(32);
    for task in ["generate", "chat"] { for kind in Int8KernelKind::ALL {
        if PackedInt4Kernel::detect(kind).is_err() { continue; }
        let flags = ["--kernel", kind.id(), "--seed", &seed, "--temperature-milli", "600", "--top-k", "20",
            "--top-p-ppm", "950000", "--logprobs", "--min-new-tokens", "2", "--stop", " END\n",
            "--ban-token", "7", "--repetition-penalty-milli", "1200", "--presence-penalty-milli", "-250",
            "--frequency-penalty-milli", "750", "--logit-bias", "42=-1500"];
        let serial = parsed(task, &flags).unwrap();
        let mut flags = flags.to_vec(); flags.extend_from_slice(&["--prefill-rows", "4"]);
        let grouped = parsed(task, &flags).unwrap();
        assert!(serial.args.options(166_101).unwrap() == grouped.args.options(166_101).unwrap());
        assert_eq!(serial.args.validate().unwrap().kernel.kind(), grouped.args.validate().unwrap().kernel.kind());
    } }
}
#[cfg(feature = "asupersync-runtime")]
#[test]
fn stale_or_underpriced_schedule_refuses_before_input_loading_or_persistence() {
    for axis in 0..4 {
        let mut command = parsed("generate", &["--prefill-rows", "4"]).unwrap();
        let mut limits = command.args.validate().unwrap();
        match axis {
            0 => command.args = parsed("generate", &[]).unwrap().args,
            1 => limits.prefill = None,
            2 => limits.prefill.as_mut().unwrap().max_extra_scratch_bytes -= 1,
            _ => limits.prefill.as_mut().unwrap().max_batch_rows = 1,
        }
        let mut output = Vec::new();
        assert_eq!(super::super::runtime::execute_task(command.task, command.args, limits, &mut NoRead, &mut output),
            Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn prefill_is_unavailable_without_runtime_before_input_or_stdout() {
    for task in ["generate", "chat"] {
        let mut output = Vec::new();
        assert_eq!(parsed(task, &["--prefill-rows", "4"]).unwrap().execute(&mut NoRead, &mut output),
            Err(CandidateError::Unavailable));
        assert!(output.is_empty());
    }
}
#[cfg(feature = "asupersync-runtime")]
#[test]
#[ignore = "requires controller-admitted independent-pin snapshot and full-model prefill parity execution"]
fn model_present_snapshot_text_prefill_matches_sequential_complete_results() {
    let snapshot = std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model-content pin required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required")
        .parse().expect("memory ceiling must be MiB");
    let kernel: Int8KernelKind = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit kernel required")
        .parse().expect("canonical kernel ID required");
    PackedInt4Kernel::detect(kernel).expect("requested kernel must be supported");
    let seed = "11".repeat(32);
    for task in ["generate", "chat"] {
        let mut expected = None;
        for rows in [None, Some(1), Some(4), Some(64)] {
            let mut flags = vec!["--context-tokens", "256", "--max-new-tokens", "4", "--max-output-bytes", "256",
                "--seed", &seed, "--top-k", "20", "--logprobs"];
            let row = rows.map(|n| n.to_string());
            if let Some(row) = &row { flags.extend_from_slice(&["--prefill-rows", row]); }
            let mut command = parsed(task, &flags).unwrap();
            command.args.source_dir = None; command.args.source_manifest = None;
            command.args.snapshot = Some(PathBuf::from(&snapshot)); command.args.expected_model_sha256 = Some(pin.clone());
            command.args.memory_mib = memory; command.args.kernel = kernel;
            let text = if task == "generate" { "Reply with a short greeting.".to_owned() }
                else { serde_json::json!([{"role":"system","content":"Reply briefly."},
                    {"role":"user","content":"Say hello."},{"role":"assistant","content":"Hello!"},
                    {"role":"user","content":"Say it once more."}]).to_string() };
            let mut output = Vec::new();
            command.execute(&mut text.as_bytes(), &mut output).unwrap();
            let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(value["scope"], "real-source-current-candidate-int4");
            assert_eq!(value["evidence"], "non_authoritative");
            assert_eq!(value["model_content_sha256"], pin);
            assert_eq!(value["snapshot"]["operation"], "loaded");
            // Includes tokens, full-vocabulary scores, finish, seed and group
            // work, not just a visually plausible assistant string.
            if let Some(expected) = &expected { assert_eq!(&value, expected); }
            else { expected = Some(value); }
        }
    }
}
