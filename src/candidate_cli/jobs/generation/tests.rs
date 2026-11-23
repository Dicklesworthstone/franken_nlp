//! CLI, consent and pre-IO boundaries; no model or filesystem execution.
use super::*;
use crate::native_engine::strict_int8::Int8Work;
use crate::jobs::JobWork;

fn parse(operation: &str, task: &str, extra: &[&str]) -> Result<TextJobCommand, clap::Error> {
    let mut argv = vec!["candidate", "text-job", operation, "--task", task, "--model", "local.fnlpq",
        "--memory-mib", "8192", "--store-results", "--job-dir", "protected-job",
        "--job-id", "0123456789abcdef0123456789abcdef", "--key-file", "job.key", "--limits", "limits.json"];
    argv.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::TextJob(command) = CandidateCommand::from_matches(&matches)? else { panic!("text job route") };
    Ok(command)
}
pub(in crate::candidate_cli) fn args(task: &str, extra: &[&str]) -> TextJobArgs {
    parse("start", task, extra).unwrap().into_parts().1
}
pub(in crate::candidate_cli) fn lifetime() -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192, max_snapshot_bytes: 65536,
        max_result_bytes: 1 << 20, max_spool_bytes: 4 << 20, max_materialized_bytes: 4 << 20,
        max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork { model: Int8Work::for_sequence(0, 8192, 100_000_000).unwrap(), mask_node_visits: 0 } }
}
#[test]
fn generation_and_chat_have_distinct_start_and_authenticated_resume_routes() {
    for task in ["generate", "chat"] {
        let (mode, a) = parse("start", task, &[]).unwrap().into_parts();
        assert!(matches!(mode, RunMode::Start)); a.validate().unwrap(); assert_eq!(a.task_name(), task);
        let (mode, a) = parse("resume", task, &["--discard-uncommitted-tail"]).unwrap().into_parts();
        assert!(matches!(mode, RunMode::Resume { discard_uncommitted: true })); a.validate().unwrap();
    }
    assert!(parse("start", "generate", &["--discard-uncommitted-tail"]).is_err());
    assert!(parse("start", "judge", &[]).is_err());
}
#[test]
fn text_job_input_replay_never_implies_a_new_seed_or_lifetime_budget() {
    for task in ["generate", "chat"] {
        let (mode, a) = parse("start", task, &["--spool-input", "--max-retained-input-mib", "8"]).unwrap().into_parts();
        a.input_retention.validate(mode, &a.common.input).unwrap();
        let (mode, b) = parse("resume", task, &["--use-retained-input", "--max-retained-input-mib", "4"]).unwrap().into_parts();
        b.input_retention.validate(mode, &b.common.input).unwrap();
        assert!(a.common.options(166_101).unwrap() == b.common.options(166_101).unwrap());
        a.check_lifetime(lifetime(), a.validate().unwrap()).unwrap();
        b.check_lifetime(lifetime(), b.validate().unwrap()).unwrap();
        assert!(parse("resume", task, &["--use-retained-input", "--max-retained-input-mib", "4", "population.ndjson"]).is_err());
    }
}
#[test]
fn all_generation_controls_survive_the_retained_parser_and_explicit_resume() {
    let seed = "07".repeat(32);
    let switches = ["--seed", seed.as_str(), "--temperature-milli", "850", "--top-k", "32", "--top-p-ppm", "900000",
        "--stop", " END\n", "--min-new-tokens", "2", "--ban-token", "7", "--logit-bias", "8=-500",
        "--repetition-penalty-milli", "1100", "--presence-penalty-milli", "-200", "--frequency-penalty-milli", "250", "--logprobs"];
    for task in ["generate", "chat"] {
        let a = args(task, &switches);
        let (_, b) = parse("resume", task, &switches).unwrap().into_parts();
        a.validate().unwrap(); b.validate().unwrap();
        let options = a.common.options(166_101).unwrap();
        assert!(options == b.common.options(166_101).unwrap());
        assert_eq!(options.stop_suffixes, vec![b" END\n".to_vec()]);
        assert_eq!(options.min_new_tokens, 2); assert!(options.capture_logprobs);
        assert_eq!(options.logit_bias_milli.get(&8), Some(&-500));
        assert!(matches!(options.sampling, GenerationSampling::Seeded { effective_seed, .. } if effective_seed == [7; 32]));
    }
}
#[test]
fn omitted_seed_is_greedy_never_a_new_random_seed_on_resume() {
    for operation in ["start", "resume"] {
        let (_, a) = parse(operation, "generate", &[]).unwrap().into_parts();
        assert!(matches!(a.common.options(166_101).unwrap().sampling, GenerationSampling::Greedy));
    }
    assert!(parse("start", "generate", &["--temperature-milli", "900"]).is_err());
    assert!(args("generate", &["--seed", "BAD"]).validate().is_err());
}
#[test]
fn explicit_retention_keys_paths_and_memory_are_mandatory() {
    for axis in 0..8 {
        let mut a = args("generate", &[]);
        match axis { 0 => a.store_results = false, 1 => a.job_dir = "-".into(), 2 => a.key_file = "".into(),
            3 => a.limits_file = "-".into(), 4 => a.max_stream_mib = 0, 5 => a.max_input_lines = 0,
            6 => a.journal_memory_mib = u64::MAX, _ => a.serialization_memory_mib = 0 }
        assert!(a.validate().is_err());
    }
}
#[test]
fn whole_original_envelopes_and_complete_result_storage_have_separate_caps() {
    let a = args("generate", &[]); let limits = a.validate().unwrap();
    a.check_lifetime(lifetime(), limits).unwrap();
    let mut j = lifetime(); j.max_input_bytes_per_item = a.common.max_input_bytes + 1;
    assert!(a.check_lifetime(j, limits).is_err());
    let mut j = lifetime(); j.max_result_bytes = limits.result_bytes - 1;
    assert!(a.check_lifetime(j, limits).is_err());
    let mut j = lifetime(); j.max_snapshot_bytes = a.max_stream_mib * MIB + 1;
    assert!(a.check_lifetime(j, limits).is_err());
}
#[test]
fn every_native_work_axis_is_required_before_input_or_model_io() {
    let a = args("generate", &[]); let limits = a.validate().unwrap();
    for axis in 0..5 {
        let mut j = lifetime();
        match axis { 0 => j.max_work.model.forward_positions = 0, 1 => j.max_work.model.projected_logits = 0,
            2 => j.max_work.model.attention_pairs = 0, 3 => j.max_work.model.projections.dot_products = 0,
            _ => j.max_work.model.projections.multiply_accumulates = 0 }
        assert!(a.check_lifetime(j, limits).is_err());
    }
}
#[test]
fn transport_changes_do_not_grant_more_generation_or_lifetime_authority() {
    let a = args("chat", &[]); let b = args("chat", &["--max-input-lines", "200000", "--max-stream-mib", "128"]);
    a.validate().unwrap(); b.validate().unwrap();
    assert!(a.common.options(166_101).unwrap() == b.common.options(166_101).unwrap());
    assert_eq!(a.validate().unwrap().result_bytes, b.validate().unwrap().result_bytes);
}
#[test]
fn invalid_prompt_schedules_refuse_before_reading_keys_or_population() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("write") }
        fn flush(&mut self) -> io::Result<()> { panic!("flush") }
    }
    for operation in ["start", "resume"] {
        for rows in ["0", "65", "18446744073709551615"] {
            let Ok(command) = parse(operation, "generate", &["--prefill-rows", rows]) else { continue; };
            let (mode, a) = command.into_parts();
            assert_eq!(execute_owned(mode, a, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
        }
        let (mode, mut a) = parse(operation, "generate", &["--prefill-rows", "64"]).unwrap().into_parts();
        a.common.memory_mib = 1024; a.common.max_weight_mib = 512;
        assert_eq!(execute_owned(mode, a, NoIo, &mut NoIo).unwrap_err().code, "text_job_native_memory");
    }
    for flag in ["--cohort-rows", "--active-rows", "--schema", "--defaults", "--key"] {
        assert!(parse("start", "generate", &[flag, "4"]).is_err());
    }
}
#[test]
fn bounded_prompt_schedule_is_additional_memory_and_not_generation_policy() {
    use crate::native_engine::strict_int8::prefill::Int8PrefillLimits;
    for task in ["generate", "chat"] {
        let serial = args(task, &[]); let serial_limits = serial.validate().unwrap();
        let seed = "07".repeat(32);
        for rows in [1, 4, 64] {
            let spelling = rows.to_string();
            let grouped = args(task, &["--prefill-rows", &spelling]);
            let grouped_limits = grouped.validate().unwrap();
            assert_eq!(grouped.common.policy.prefill().unwrap().unwrap().max_batch_rows, rows);
            assert_eq!(grouped.execution_memory_bytes(grouped_limits).unwrap()
                - serial.execution_memory_bytes(serial_limits).unwrap(),
                Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap());
            assert!(serial.common.options(166_101).unwrap() == grouped.common.options(166_101).unwrap());
            let (_, resumed) = parse("resume", task, &["--prefill-rows", &spelling, "--seed", &seed])
                .unwrap().into_parts();
            let seeded = args(task, &["--seed", &seed]);
            resumed.validate().unwrap();
            assert!(seeded.common.options(166_101).unwrap() == resumed.common.options(166_101).unwrap());
            assert_eq!(serial_limits.result_bytes, grouped_limits.result_bytes);
            grouped.check_lifetime(lifetime(), grouped_limits).unwrap();
        }
    }
}
#[test]
fn persistent_team_survives_start_and_resume_parsing_for_both_text_tasks() {
    for operation in ["start", "resume"] {
        for task in ["generate", "chat"] {
            for width in ["1", "2", "4", "64"] {
                for prefill in [None, Some("4")] {
                    let mut flags = vec!["--compute-threads", width];
                    if let Some(rows) = prefill { flags.extend_from_slice(&["--prefill-rows", rows]); }
                    let (_, a) = parse(operation, task, &flags).unwrap().into_parts();
                    let admitted = a.validate().unwrap();
                    assert_eq!(admitted.compute.threads, width.parse::<usize>().unwrap());
                    assert_eq!(admitted.compute,
                        a.common.compute.admit_prefill(a.common.policy.prefill().unwrap()).unwrap());
                    assert_eq!(admitted.compute.workspace_bytes == 0, width == "1");
                    a.check_lifetime(lifetime(), admitted).unwrap();
                }
            }
        }
    }
}
#[test]
fn persistent_worker_buffers_are_additional_to_prompt_sampler_and_owned_job_memory() {
    use crate::native_engine::strict_int8::projection::int8_team_workspace_bytes_with_prefill;
    for task in ["generate", "chat"] {
        for prefill in [None, Some("1"), Some("4"), Some("64")] {
            let serial_flags = prefill.map(|rows| vec!["--prefill-rows", rows]).unwrap_or_default();
            let serial = args(task, &serial_flags); let original = serial.validate().unwrap();
            let rows = prefill.map(|rows| rows.parse::<usize>().unwrap()).unwrap_or(1);
            for width in ["2", "4", "64"] {
                let mut flags = serial_flags.clone(); flags.extend_from_slice(&["--compute-threads", width]);
                let parallel = args(task, &flags); let admitted = parallel.validate().unwrap();
                let expected = int8_team_workspace_bytes_with_prefill(width.parse().unwrap(), rows).unwrap();
                assert_eq!(parallel.execution_memory_bytes(admitted).unwrap()
                    - serial.execution_memory_bytes(original).unwrap(), expected);
                assert_eq!(admitted.compute.workspace_bytes, expected);
                assert_eq!(admitted.result_bytes, original.result_bytes);
                assert!(parallel.common.options(166_101).unwrap() == serial.common.options(166_101).unwrap());
            }
            let mut same_flags = serial_flags;
            same_flags.extend_from_slice(&["--compute-threads", "1"]);
            let one = args(task, &same_flags); let admitted = one.validate().unwrap();
            assert_eq!(one.execution_memory_bytes(admitted).unwrap(), serial.execution_memory_bytes(original).unwrap());
        }
    }
}
#[test]
fn invalid_or_underfunded_persistent_jobs_refuse_before_private_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("write") }
        fn flush(&mut self) -> io::Result<()> { panic!("flush") }
    }
    for operation in ["start", "resume"] {
        for width in ["0", "65", "18446744073709551615"] {
            let Ok(command) = parse(operation, "generate", &["--compute-threads", width]) else { continue; };
            let (mode, a) = command.into_parts();
            assert_eq!(execute_owned(mode, a, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
        }
        for task in ["generate", "chat"] {
            let (mode, mut a) = parse(operation, task,
                &["--compute-threads", "4", "--prefill-rows", "4"]).unwrap().into_parts();
            a.common.memory_mib = 1024; a.common.max_weight_mib = 512;
            assert!(a.validate().is_err(), "refusal must precede feature-graph dispatch");
            assert!(execute_owned(mode, a, NoIo, &mut NoIo).is_err());
        }
    }
}
#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unavailable_feature_graph_never_opens_key_config_input_or_output() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("write") }
        fn flush(&mut self) -> io::Result<()> { panic!("flush") }
    }
    for task in ["generate", "chat"] {
        for flags in [vec![], vec!["--prefill-rows", "4"], vec!["--compute-threads", "4"],
            vec!["--compute-threads", "4", "--prefill-rows", "4"]] {
            assert_eq!(execute_owned(RunMode::Start, args(task, &flags), NoIo, &mut NoIo).unwrap_err().code,
                "owned_text_job_profile_unavailable");
        }
    }
}
