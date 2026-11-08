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
        Int4Task::Classify(options) => options.validate_one_off_with_prefill(&command.args),
        Int4Task::Sentiment(options) => options.validate(&command.args),
        Int4Task::Judge(options) => options.validate(&command.args),
        Int4Task::Extract(options) => options.validate_one_off_with_prefill(&command.args),
        Int4Task::Source(_, options) => options.validate_with_prefill(&command.args),
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
fn invalid_worker_widths_and_mixed_prompt_schedules_refuse_before_private_io() {
    for task in TASKS {
        for flags in [
            &["--compute-threads", "0"][..], &["--compute-threads", "65"][..],
            &["--compute-threads", "18446744073709551615"][..],
            &["--compute-threads", "2", "--prefill-rows", "1"][..],
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
}
