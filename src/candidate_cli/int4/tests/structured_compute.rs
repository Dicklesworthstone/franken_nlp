use super::*;

const TASKS: &[&str] = &["classify", "sentiment", "judge", "extract", "ner", "keyphrases", "summarize", "answer"];

fn parsed(task: &str, extra: &[&str]) -> Int4Command {
    let mut flags = Vec::new();
    if task == "extract" { flags.extend(["--schema", "private-schema.json"]); }
    flags.extend_from_slice(extra);
    command(task, &flags)
}
fn limits(command: &Int4Command) -> Result<Int4Limits, CandidateError> {
    match &command.task {
        Int4Task::Classify(options) => options.validate_with_threaded_prefill(&command.args),
        Int4Task::Sentiment(options) => options.validate(&command.args),
        Int4Task::Judge(options) => options.validate(&command.args),
        Int4Task::Extract(options) => options.validate_population_with_threaded_prefill(&command.args),
        Int4Task::Source(_, options) => options.validate_population_with_threaded_prefill(&command.args),
        _ => panic!("expected one-off structured task"),
    }
}
fn snapshot(command: &mut Int4Command) {
    command.args.source_dir = None; command.args.source_manifest = None;
    command.args.snapshot = Some(PathBuf::from("private.snapshot"));
    command.args.expected_model_sha256 = Some("ab".repeat(32));
}

#[test]
fn all_structured_routes_admit_one_fixed_worker_workspace_without_changing_native_capacity() {
    for task in TASKS {
        let mut serial = parsed(task, &["--compute-threads", "1"]);
        let mut team = parsed(task, &["--compute-threads", "4"]);
        snapshot(&mut serial); snapshot(&mut team);
        let before = limits(&serial).unwrap(); let after = limits(&team).unwrap();
        let extra = crate::native_engine::strict_int4::projection::int4_team_workspace_bytes(4, 128).unwrap();
        assert_eq!(after.compute, compute::Admission { threads: 4, workspace_bytes: extra });
        assert_eq!(after.native, before.native);
        assert_eq!(after.load, before.load);
        assert_eq!(after.max_prompt_tokens, before.max_prompt_tokens);
        assert_eq!(after.minimum_memory_bytes, before.minimum_memory_bytes + extra);
        assert!(after.prefill.is_none() && after.cohort.is_none());
        team.args.memory_mib = (after.minimum_memory_bytes - 1) / MIB;
        assert!(limits(&team).is_err());
        team.args.memory_mib = after.minimum_memory_bytes.div_ceil(MIB);
        limits(&team).unwrap();
    }
}

#[test]
fn invalid_worker_widths_and_unbounded_prompt_schedules_refuse_before_private_io() {
    for task in TASKS {
        for flags in [
            &["--compute-threads", "0"][..], &["--compute-threads", "65"][..],
            &["--compute-threads", "18446744073709551615"][..],
            &["--compute-threads", "2", "--prefill-rows", "0"][..],
            &["--compute-threads", "4", "--prefill-rows", "65"][..],
        ] {
            let mut output = Vec::new();
            assert_eq!(parsed(task, flags).execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
            assert!(output.is_empty());
        }
        // Width one keeps ordinary prompt scheduling available.
        limits(&parsed(task, &["--compute-threads", "1", "--prefill-rows", "4"])).unwrap();
    }
    for task in ["classify", "sentiment", "judge"] {
        let mut output = Vec::new();
        assert_eq!(parsed(task, &["--compute-threads", "2", "--cohort-heads", "2"])
            .execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
}

#[test]
fn one_shot_prefill_teams_charge_both_prompt_and_worker_storage_before_io() {
    for task in ["classify", "extract", "ner", "keyphrases", "summarize", "answer"] {
        for rows in ["1", "2", "3", "64"] {
            let mut serial = parsed(task, &["--prefill-rows", rows]);
            snapshot(&mut serial);
            let before = limits(&serial).unwrap();
            for width in ["2", "4"] {
                let mut command = parsed(task, &["--compute-threads", width, "--prefill-rows", rows,
                    "--input-encoding", "safe-bpe-v1"]);
                // Compare like-for-like input preparation, keeping compact
                // semantics separate from this purely physical schedule.
                serial.args.input_encoding = command.args.input_encoding.clone();
                snapshot(&mut command);
                let compact_serial = limits(&serial).unwrap();
                let admitted = limits(&command).unwrap();
                let workers = crate::native_engine::strict_int4::projection::int4_prefill_team_workspace_bytes(
                    width.parse().unwrap(), 128, rows.parse().unwrap()).unwrap();
                assert_eq!(admitted.compute.workspace_bytes, workers);
                assert_eq!(admitted.minimum_memory_bytes, compact_serial.minimum_memory_bytes + workers);
                assert_eq!(admitted.native, before.native);
                assert_eq!(admitted.max_prompt_tokens, before.max_prompt_tokens);
                assert_eq!(admitted.prefill.unwrap().max_batch_rows, rows.parse::<usize>().unwrap());
                command.args.memory_mib = (admitted.minimum_memory_bytes - 1) / MIB;
                let mut output = Vec::new();
                assert_eq!(command.execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
                assert!(output.is_empty());
            }
        }
    }
    // Broader scored workflows retain their separate scheduling gate until
    // their own hosted drivers are composed.
    for task in ["sentiment", "judge"] {
        let mut output = Vec::new();
        assert_eq!(parsed(task, &["--compute-threads", "2", "--prefill-rows", "1"])
            .execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
}

#[test]
fn worker_teams_compose_with_supported_compact_and_exact_selected_row_policies() {
    for task in TASKS.iter().copied().filter(|task| *task != "judge") {
        let command = parsed(task, &["--compute-threads", "4", "--input-encoding", "safe-bpe-v1"]);
        limits(&command).unwrap();
        assert!(!command.args.input_encoding.is_default());
    }
    for task in ["extract", "ner", "keyphrases", "summarize", "answer"] {
        limits(&parsed(task, &["--compute-threads", "4", "--max-selected-rows", "16",
            "--input-encoding", "safe-bpe-v1"])).unwrap();
    }
}

#[test]
fn composed_validators_cannot_silently_ignore_a_requested_worker_team() {
    for task in ["classify", "extract", "ner"] {
        let command = parsed(task, &["--compute-threads", "4"]);
        limits(&command).unwrap();
        match &command.task {
            Int4Task::Classify(options) => {
                assert!(options.validate(&command.args).is_err());
                assert!(options.validate_batch(&command.args).is_err());
                assert!(options.validate_with_prefill(&command.args).is_err());
            }
            Int4Task::Extract(options) => {
                assert!(options.validate(&command.args).is_err());
                assert!(options.validate_full_projection_with_prefill(&command.args).is_err());
                assert!(options.validate_with_projection(&command.args).is_err());
            }
            Int4Task::Source(_, options) => {
                assert!(options.validate(&command.args).is_err());
                assert!(options.validate_full_projection_with_prefill(&command.args).is_err());
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(feature = "asupersync-runtime")]
#[test]
fn stale_structured_worker_admission_refuses_before_runtime_or_private_input() {
    for task in ["classify", "extract", "ner"] {
        let command = parsed(task, &["--compute-threads", "4"]);
        let mut admitted = limits(&command).unwrap();
        assert!(compute::matches(&command.task, &command.args, admitted).unwrap());
        admitted.compute.workspace_bytes -= 1;
        assert!(!compute::matches(&command.task, &command.args, admitted).unwrap());
        let mut output = Vec::new();
        assert_eq!(runtime::execute_task(command.task, command.args, admitted, &mut NoRead, &mut output),
            Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
    for task in ["classify", "extract", "ner", "keyphrases", "summarize", "answer"] {
        let command = parsed(task, &["--compute-threads", "4", "--prefill-rows", "3"]);
        let mut admitted = limits(&command).unwrap();
        assert!(compute::matches(&command.task, &command.args, admitted).unwrap());
        admitted.compute.workspace_bytes = crate::native_engine::strict_int4::projection::int4_team_workspace_bytes(
            4, command.args.group_width).unwrap();
        assert!(!compute::matches(&command.task, &command.args, admitted).unwrap());
        let mut output = Vec::new();
        assert_eq!(runtime::execute_task(command.task, command.args, admitted, &mut NoRead, &mut output),
            Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
}
