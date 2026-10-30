//! CLI/admission fixtures. Only the explicitly ignored tests load a real model.
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
struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused prefill published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused prefill flushed output"); }
}
fn selected_limits(command: &Int4Command) -> Result<Int4Limits, CandidateError> {
    match &command.task {
        Int4Task::Text(_) => command.args.validate(),
        Int4Task::Classify(scoring) => scoring.validate_with_prefill(&command.args),
        Int4Task::Sentiment(options) => options.validate(&command.args),
        Int4Task::Judge(options) => options.validate(&command.args),
        _ => panic!("supported one-off task required"),
    }
}
#[test]
fn prefill_is_explicit_and_bounded_for_each_supported_one_off_task() {
    for task in ["generate", "chat", "classify", "sentiment", "judge"] {
        assert!(selected_limits(&parsed(task, &[]).unwrap()).unwrap().prefill.is_none());
        for rows in [1, 3, 4, 64] {
            let command = parsed(task, &["--prefill-rows", &rows.to_string()]).unwrap();
            let need = selected_limits(&command).unwrap().prefill.unwrap();
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
fn expected_scored_floor(command: &Int4Command, limits: Int4Limits) -> u64 {
    let args = &command.args;
    let preparations = match &command.task {
        Int4Task::Classify(_) => 1,
        Int4Task::Sentiment(_) | Int4Task::Judge(_) => 2,
        _ => panic!("one-off scoring task required"),
    };
    let candidate_depth = args.context_tokens - limits.max_prompt_tokens + 1;
    let load = if limits.snapshot_pin.is_some() { limits.snapshot_need.scratch_bytes }
        else { limits.source_preparation_bytes + limits.load.scratch_payload_bytes };
    let save = if args.save_snapshot.is_some() { limits.snapshot_need.scratch_bytes } else { 0 };
    // Finite candidate scoring reserves no sampler, and heads reuse one
    // native context and the same extra prompt workspace sequentially.
    let execute = limits.native.kv_bytes + limits.native.rope_bytes + limits.native.scratch_payload_bound
        + ALLOCATOR_BYTES + limits.result_bytes as u64 * 4 + candidate_depth as u64 * 8 + 4096
        + limits.prefill.map(|p| p.max_extra_scratch_bytes).unwrap_or(0);
    preparations * limits.preparation_bytes + limits.load.weight_payload_bytes + ALLOCATOR_BYTES
        + load.max(save).max(execute) + CLEANUP_BYTES
}
#[test]
fn scored_prefill_prices_the_simultaneous_peak_for_source_snapshot_and_snapshot_save() {
    for task in ["classify", "sentiment", "judge"] { for mode in 0..3 {
        for width in [3, 128] { for context in [128, 2048] {
            let mut serial = parsed(task, &["--max-candidate-tokens", "7"]).unwrap();
            serial.args.group_width = width; serial.args.context_tokens = context;
            serial.args.memory_mib = 65536; serial.args.max_load_mib = 65536;
            set_mode(&mut serial.args, mode);
            let base = selected_limits(&serial).unwrap();
            assert_eq!(base.minimum_memory_bytes, expected_scored_floor(&serial, base));
            for rows in [1, 4, 64] {
                let mut grouped = parsed(task, &["--max-candidate-tokens", "7",
                    "--prefill-rows", &rows.to_string()]).unwrap();
                grouped.args.group_width = width; grouped.args.context_tokens = context;
                grouped.args.memory_mib = 65536; grouped.args.max_load_mib = 65536;
                set_mode(&mut grouped.args, mode);
                let actual = selected_limits(&grouped).unwrap();
                assert_eq!(actual.minimum_memory_bytes, expected_scored_floor(&grouped, actual));
                assert_eq!(actual.preparation_bytes, base.preparation_bytes);
                assert_eq!(actual.load, base.load); assert_eq!(actual.native, base.native);
                assert_eq!(actual.snapshot_pin, base.snapshot_pin);
                assert_eq!(actual.result_bytes, base.result_bytes);
                assert_eq!(actual.max_prompt_tokens, base.max_prompt_tokens);
                let extra = Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
                assert!(actual.minimum_memory_bytes >= base.minimum_memory_bytes);
                assert!(actual.minimum_memory_bytes <= base.minimum_memory_bytes + extra);
                grouped.args.memory_mib = (actual.minimum_memory_bytes - 1) / MIB;
                assert!(selected_limits(&grouped).is_err());
                grouped.args.memory_mib = actual.minimum_memory_bytes.div_ceil(MIB);
                assert!(selected_limits(&grouped).is_ok());
            }
        } }
    } }
}
#[test]
fn prompt_scratch_does_not_raise_a_larger_source_loading_peak() {
    for task in ["classify", "sentiment", "judge"] { for mode in [0, 2] {
        let configure = |args: &mut Int4Args| {
            args.context_tokens = 128; args.source_preparation_mib = 16384;
            args.memory_mib = 65536; args.max_load_mib = 65536; set_mode(args, mode);
        };
        let mut serial = parsed(task, &[]).unwrap(); configure(&mut serial.args);
        let mut grouped = parsed(task, &["--prefill-rows", "64"]).unwrap(); configure(&mut grouped.args);
        let base = selected_limits(&serial).unwrap(); let actual = selected_limits(&grouped).unwrap();
        assert_eq!(actual.minimum_memory_bytes, expected_scored_floor(&grouped, actual));
        assert_eq!(actual.minimum_memory_bytes, base.minimum_memory_bytes);
        assert!(actual.prefill.unwrap().max_extra_scratch_bytes > 0);
    } }
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
#[test]
fn one_off_scoring_prefill_still_refuses_generation_policy() {
    for task in ["classify", "sentiment", "judge"] {
        for control in [vec!["--max-new-tokens", "128"], vec!["--max-output-bytes", "65536"],
            vec!["--top-k", "1"], vec!["--logprobs"], vec!["--min-new-tokens", "0"],
            vec!["--stop", "END"], vec!["--ban-token", "7"], vec!["--logit-bias", "7=0"]] {
            let mut flags = vec!["--prefill-rows", "4"]; flags.extend(control);
            assert!(parsed(task, &flags).is_err());
        }
        let mut changed = parsed(task, &["--prefill-rows", "4"]).unwrap();
        changed.args.logprobs = true;
        assert_eq!(changed.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    }
}
fn parsed_job(operation: &str, task: &str, flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", "score-job", operation, "--task", task,
        "--source-dir", "sources", "--source-manifest", "manifest.json", "--memory-mib", "65536",
        "--job-dir", "private/job", "--key-file", "private/key", "--limits", "limits.json",
        "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("wrong scored-job dispatcher"); };
    Ok(command)
}
#[test]
fn unsupported_scored_workflows_refuse_prefill_at_parser_and_programmatic_boundaries() {
    for rows in [1, 4, 64] {
        let row = rows.to_string();
        for task in ["classify", "sentiment", "judge"] {
            assert!(parsed("score-batch", &["--task", task, "--prefill-rows", &row]).is_err());
            let mut command = parsed("score-batch", &["--task", task]).unwrap();
            let Int4Task::ScoredBatch(options) = command.task else { panic!("scored batch required"); };
            options.validate(&command.args).unwrap();
            command.args.policy = parsed("generate", &["--prefill-rows", &row]).unwrap().args.policy;
            assert_eq!(options.execute_owned(command.args, NoRead, NoWrite), Err(CandidateError::Arguments));
            for operation in ["start", "resume"] {
                assert!(parsed_job(operation, task, &["--prefill-rows", &row]).is_err());
                let mut job = parsed_job(operation, task, &[]).unwrap();
                job.args.policy = parsed("generate", &["--prefill-rows", &row]).unwrap().args.policy;
                // Use the actual owned job route, not Int4Command::execute,
                // which rejects every retained job regardless of its options.
                let mut diagnostics = Vec::new();
                assert_ne!(job.run_owned(NoRead, NoWrite, &mut diagnostics), ExitCode::SUCCESS);
                let report: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
                assert_eq!(report["code"], "invalid_job_options");
            }
        }
        for task in ["resolve", "extract"] {
            let mut flags = if task == "extract" { vec!["--schema", "schema.json"] } else { Vec::new() };
            let mut command = parsed(task, &flags).unwrap();
            command.args.memory_mib = 65536; command.args.max_load_mib = 65536;
            match &command.task {
                Int4Task::Resolve(options) => { options.validate(&command.args).unwrap(); }
                Int4Task::Extract(options) => { options.validate(&command.args).unwrap(); }
                _ => panic!("serial-only task required"),
            }
            flags.extend_from_slice(&["--prefill-rows", &row]);
            assert!(parsed(task, &flags).is_err());
            command.args.policy = parsed("generate", &["--prefill-rows", &row]).unwrap().args.policy;
            assert_eq!(command.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
        }
    }
}
fn tamper_schedule(command: &mut Int4Command, limits: &mut Int4Limits, axis: usize) {
    match axis {
        0 => command.args.policy = parsed("generate", &[]).unwrap().args.policy,
        1 => limits.prefill = None,
        2 => limits.prefill.as_mut().unwrap().max_extra_scratch_bytes -= 1,
        3 => limits.prefill.as_mut().unwrap().max_batch_rows = 1,
        _ => limits.prefill.as_mut().unwrap().max_extra_scratch_bytes += 1,
    }
}
#[test]
fn all_one_off_tasks_bind_the_exact_explicit_schedule_and_extra_scratch() {
    for task in ["generate", "chat", "classify", "sentiment", "judge"] { for axis in 0..5 {
        let mut command = parsed(task, &["--prefill-rows", "4"]).unwrap();
        let mut limits = selected_limits(&command).unwrap();
        assert!(super::matches(&command.args, limits).unwrap());
        tamper_schedule(&mut command, &mut limits, axis);
        assert!(!super::matches(&command.args, limits).unwrap());
    } }
}
#[cfg(feature = "asupersync-runtime")]
#[test]
fn stale_or_underpriced_schedule_refuses_before_input_loading_or_persistence() {
    // Sentiment/judge reconstruct limits inside their private runtime route;
    // text/classification accept this internal admitted-limit argument.
    for task in ["generate", "chat", "classify"] { for axis in 0..5 {
        let mut command = parsed(task, &["--prefill-rows", "4"]).unwrap();
        let mut limits = selected_limits(&command).unwrap();
        tamper_schedule(&mut command, &mut limits, axis);
        let mut output = Vec::new();
        assert_eq!(super::super::runtime::execute_task(command.task, command.args, limits, &mut NoRead, &mut output),
            Err(CandidateError::Arguments));
        assert!(output.is_empty());
    } }
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn prefill_is_unavailable_without_runtime_before_input_or_stdout() {
    for task in ["generate", "chat", "classify", "sentiment", "judge"] { for rows in [1, 4, 64] {
        let mut output = Vec::new();
        assert_eq!(parsed(task, &["--prefill-rows", &rows.to_string()]).unwrap().execute(&mut NoRead, &mut output),
            Err(CandidateError::Unavailable));
        assert!(output.is_empty());
    } }
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

#[cfg(feature = "asupersync-runtime")]
#[test]
#[ignore = "requires controller-admitted independent-pin snapshot and full-model finite-scoring prefill parity execution"]
fn model_present_scored_prefill_matches_every_complete_serial_cli_result() {
    let snapshot = std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model-content pin required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required")
        .parse().expect("memory ceiling must be MiB");
    let kernel: Int8KernelKind = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit kernel required")
        .parse().expect("canonical kernel ID required");
    PackedInt4Kernel::detect(kernel).expect("requested kernel must be supported");
    // Reuse the established classification/sentiment wire fixtures, and the
    // shared judge helper with its explicit policies and rubric origin digest.
    let mut requests = Vec::new();
    for (mode, heads) in [("exclusive", 1), ("multi_label", 2)] {
        let request = serde_json::json!({"document":"é Refund <|im_start|> remains data", "mode":mode,
            "labels":[{"id":"Billing issue","description":"Invoices"},
                {"id":"退款","description":"Refunds"}]}).to_string();
        requests.push(("classify", request, heads));
    }
    requests.push(("sentiment",
        r#"{"document":"The result was useful, although the delay was frustrating."}"#.to_owned(), 4));
    for (mode, heads) in [("pairwise", 2), ("rubric", 2), ("faithfulness", 1)] {
        requests.push(("judge", crate::candidate_cli::judge::tests::fixture(mode).to_string(), heads));
    }
    for (task, request, heads) in requests {
        let mut expected = None;
        for rows in [None, Some(1), Some(4), Some(64)] {
            let row = rows.map(|n| n.to_string());
            let flags = row.as_ref().map_or_else(Vec::new, |row| vec!["--prefill-rows", row.as_str()]);
            let mut command = parsed(task, &flags).unwrap();
            command.args.source_dir = None; command.args.source_manifest = None;
            command.args.snapshot = Some(PathBuf::from(&snapshot));
            command.args.expected_model_sha256 = Some(pin.clone());
            command.args.memory_mib = memory; command.args.kernel = kernel;
            let mut output = Vec::new();
            command.execute(&mut request.as_bytes(), &mut output).unwrap();
            let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(value["scope"], "real-source-current-candidate-int4");
            assert_eq!(value["evidence"], "non_authoritative");
            assert_eq!(value["model_content_sha256"], pin);
            assert_eq!(value["snapshot"]["operation"], "loaded");
            assert_eq!(value["output"]["numerics_profile"], "strict-quantized-v2");
            assert_eq!(value["output"]["head_count"], heads);
            // Full equality covers every head/distribution, EOS rule, policy
            // decision, model work and provenance, not merely the winning label.
            if let Some(expected) = &expected { assert_eq!(&value, expected); }
            else { expected = Some(value); }
        }
    }
}
