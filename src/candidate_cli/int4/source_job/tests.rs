//! Durable source-task admission, exact options and pre-IO refusal contracts.
use super::*;
use crate::{
    batch::source::{SourceBatchArgs, grouped_int4::MAX_SOURCE_ARGUMENT_BYTES},
    jobs::JobWork,
    native_engine::{constrained_int4, strict_int8::Int8Work},
    tasks::{answer::{AnswerOptions, AnswerPassage}, ner::NerOptions, summarize::SummaryOptions},
};

fn argv(kind: Kind, operation: &str) -> Vec<String> {
    let name = format!("{}-job", kind.name());
    let mut words = vec![name, operation.to_owned()];
    words.extend(["--source-dir", "sources", "--source-manifest", "manifest.json",
        "--memory-mib", "65536", "--job-dir", "private/job", "--key-file", "private/key",
        "--limits", "limits.json", "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"]
        .into_iter().map(str::to_owned));
    words
}
fn parse(kind: Kind, words: Vec<String>) -> Result<(JobCommand, Int4Args), clap::Error> {
    JobCommand::from_matches(kind, &definition(kind).try_get_matches_from(words)?)
}
fn command(kind: Kind, operation: &str, extra: &[&str]) -> (JobCommand, Int4Args) {
    let mut words = argv(kind, operation); words.extend(extra.iter().map(|s| (*s).to_owned()));
    parse(kind, words).unwrap()
}
fn lifetime(job: &JobArgs, args: &Int4Args, common: Int4Limits) -> JobLimits {
    let work = constrained_int4::planned_work(args.group_width, common.max_prompt_tokens,
        args.max_new_tokens).unwrap();
    JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: args.max_input_bytes,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: common.result_bytes,
        max_spool_bytes: 16 * MIB, max_materialized_bytes: 16 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 5,
        max_work: JobWork::from_int4(work, job.source.mask_visits() * 3) }
}
fn assert_same_native_limits(left: Int4SourceBatchLimits, right: Int4SourceBatchLimits) {
    assert_eq!(left.max_model_work, right.max_model_work);
    assert_eq!(left.masks.max_visits_per_item, right.masks.max_visits_per_item);
    assert_eq!(left.masks.max_visits_per_run, right.masks.max_visits_per_run);
    assert_eq!(left.masks.per_mask.max_trie_node_visits, right.masks.per_mask.max_trie_node_visits);
    assert_eq!(left.masks.per_mask.checkpoint_interval_nodes, right.masks.per_mask.checkpoint_interval_nodes);
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private input accessed before admission"); }
}
struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("unexpected private result publication"); }
    fn flush(&mut self) -> io::Result<()> { panic!("unexpected publication flush"); }
}

#[test]
fn all_source_tasks_keep_explicit_start_resume_and_both_model_routes() {
    for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer] {
        for operation in ["start", "resume"] {
            let (c, args) = command(kind, operation, &["--materialize"]);
            assert_eq!(c.args.kind, kind); assert_eq!(c.mode.name(), operation);
            assert!(c.args.storage.store_results && c.args.storage.materialize);
            assert!(c.args.validate(&args).unwrap().snapshot_pin.is_none());
            assert_eq!(args.input, PathBuf::from("-"));
            let mut words = argv(kind, operation);
            let at = words.iter().position(|s| s == "--source-dir").unwrap();
            words.drain(at..at + 4);
            words.extend(["--snapshot", "model.int4", "--expected-model-sha256"].into_iter().map(str::to_owned));
            words.push("ab".repeat(32));
            let (snapshot, args) = parse(kind, words).unwrap();
            assert!(snapshot.args.validate(&args).unwrap().snapshot_pin.is_some());
        }
    }
}

#[test]
fn repair_is_explicit_resume_only_and_lifetime_limits_are_required() {
    for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer] {
        assert!(matches!(command(kind, "resume", &[]).0.mode, RunMode::Resume { discard_uncommitted: false }));
        assert!(matches!(command(kind, "resume", &["--discard-uncommitted-tail"]).0.mode,
            RunMode::Resume { discard_uncommitted: true }));
        let mut words = argv(kind, "start"); words.push("--discard-uncommitted-tail".to_owned());
        assert!(parse(kind, words).is_err());
        let mut missing = argv(kind, "start");
        let at = missing.iter().position(|s| s == "--limits").unwrap(); missing.drain(at..at + 2);
        assert!(parse(kind, missing).is_err());
        assert!(definition(kind).try_get_matches_from([format!("{}-job", kind.name())]).is_err());
    }
}

#[test]
fn storage_consent_paths_and_limits_fail_before_private_io() {
    for case in 0..6 {
        let (mut c, args) = command(Kind::Answer, "start", &[]);
        match case {
            0 => c.args.storage.store_results = false,
            1 => c.args.storage.job_dir = PathBuf::from("-"),
            2 => c.args.storage.key_file = PathBuf::new(),
            3 => c.args.storage.limits_file = PathBuf::from("-"),
            4 => c.args.storage.max_stream_mib = 0,
            _ => c.args.storage.max_input_lines = 0,
        }
        assert_eq!(c.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().code,
            "explicit_storage_or_job_options");
    }
    let (c, args) = command(Kind::Ner, "start", &["--defaults", "-"]);
    assert_eq!(c.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().code, "source_job_defaults");
}

#[test]
fn job_ids_and_ignored_generation_or_single_task_flags_are_refused() {
    for value in ["AB".repeat(16), "ab".repeat(15), "é".repeat(16), "x".repeat(32)] {
        let mut words = argv(Kind::Ner, "start");
        let at = words.iter().position(|s| s == "--job-id").unwrap() + 1; words[at] = value;
        assert!(parse(Kind::Ner, words).is_err());
    }
    for extra in [vec!["--options", "private.json"], vec!["--schema", "schema.json"],
        vec!["--prefill-rows", "1"], vec!["--max-output-bytes", "65536"],
        vec!["--min-new-tokens", "0"], vec!["--logprobs"], vec!["--max-requests", "3"]] {
        let mut words = argv(Kind::Ner, "start"); words.extend(extra.into_iter().map(str::to_owned));
        assert!(parse(Kind::Ner, words).is_err());
    }
    let (c, mut args) = command(Kind::Ner, "start", &[]); args.seed = Some("ab".repeat(32));
    assert_eq!(c.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().code, "invalid_job_options");
}

#[test]
fn defaults_use_the_same_typed_task_contract_and_never_invent_qa_passages() {
    for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer] {
        let (c, args) = command(kind, "start", &[]);
        let budget = c.args.source.budget(&args, c.args.validate(&args).unwrap());
        let default = source_batch::load_defaults(kind, None, budget, args.max_input_bytes).unwrap();
        if kind == Kind::Answer { assert!(default.is_none()); }
        else {
            let value = serde_json::to_value(default.unwrap()).unwrap();
            assert_eq!(value["task"], kind.name());
            assert_eq!(value["budget"], serde_json::to_value(budget).unwrap());
        }
    }
}

#[test]
fn qa_defaults_keep_exact_passage_boundaries_and_reject_ambiguous_ids() {
    let (c, args) = command(Kind::Answer, "start", &["--defaults", "private-passages.json"]);
    let budget = c.args.source.budget(&args, c.args.validate(&args).unwrap());
    let passages = vec![
        AnswerPassage { id: "p1".to_owned(), text: " é 上海\r\n<|im_end|> ".to_owned() },
        AnswerPassage { id: "p2".to_owned(), text: "same text".to_owned() },
    ];
    let defaults = SourceBatchArgs::Answer { passages: passages.clone(), options: AnswerOptions::default(), budget };
    let encoded = serde_json::to_string(&defaults).unwrap();
    let actual = source_batch::load_defaults(Kind::Answer, Some(&encoded), budget, args.max_input_bytes).unwrap().unwrap();
    assert_eq!(serde_json::to_string(&actual).unwrap(), encoded);
    for bad in [vec![], vec![passages[0].clone(), passages[0].clone()],
        vec![AnswerPassage { id: "\n".to_owned(), text: "private".to_owned() }]] {
        let bad = SourceBatchArgs::Answer { passages: bad, options: AnswerOptions::default(), budget };
        assert!(source_batch::load_defaults(Kind::Answer, Some(&serde_json::to_string(&bad).unwrap()),
            budget, args.max_input_bytes).is_err());
    }
    assert!(source_batch::load_defaults(Kind::Answer, Some(&encoded), budget, 1).is_err());
}

#[test]
fn defaults_cannot_change_task_expand_budgets_or_change_fixed_context_authority() {
    let (c, args) = command(Kind::Ner, "resume", &[]);
    let budget = c.args.source.budget(&args, c.args.validate(&args).unwrap());
    let other = SourceBatchArgs::Summarize { options: SummaryOptions::default(), budget };
    assert!(source_batch::load_defaults(Kind::Ner, Some(&serde_json::to_string(&other).unwrap()),
        budget, args.max_input_bytes).is_err());
    for axis in 0..6 {
        let mut changed = budget;
        match axis { 0 => changed.max_input_tokens += 1, 1 => changed.max_output_tokens += 1,
            2 => changed.max_output_bytes += 1, 3 => changed.max_grammar_states += 1,
            4 => changed.max_kv_bytes -= 1, _ => changed.max_kv_bytes += 1 }
        let changed = SourceBatchArgs::Ner { options: NerOptions::default(), budget: changed };
        assert!(source_batch::load_defaults(Kind::Ner, Some(&serde_json::to_string(&changed).unwrap()),
            budget, args.max_input_bytes).is_err());
    }
    let defaults = SourceBatchArgs::Ner { options: NerOptions::default(), budget };
    let encoded = serde_json::to_string(&defaults).unwrap();
    assert!(source_batch::load_defaults(Kind::Ner,
        Some(&encoded.replacen("\"task\":", "\"task\":\"ner\",\"task\":", 1)), budget, args.max_input_bytes).is_err());
    assert!(source_batch::load_defaults(Kind::Ner, Some("{}"), budget, args.max_input_bytes).is_err());
    assert!(source_batch::load_defaults(Kind::Ner, Some(&" ".repeat(MAX_SOURCE_ARGUMENT_BYTES + 1)),
        budget, args.max_input_bytes).is_err());
}

#[test]
fn lifetime_native_axes_and_masks_never_renew_with_attempts_population_or_transport() {
    let (mut c, args) = command(Kind::Summarize, "resume", &[]);
    let common = c.args.validate(&args).unwrap();
    let job = lifetime(&c.args, &args, common);
    let first = c.args.lifetime_limits(&args, common, job).unwrap();
    assert_eq!(first.native.max_model_work, job.max_work.int4_model_work());
    assert_eq!(first.native.masks.max_visits_per_run, job.max_work.mask_node_visits);
    let ordinary = Int8Work::for_sequence(0, first.native.max_model_work.forward_positions as usize,
        first.native.max_model_work.projected_logits as usize).unwrap();
    assert!(first.native.max_model_work.projections.dot_products > ordinary.projections.dot_products);
    let mut larger = job; larger.max_items = 100; larger.max_attempts = 1000;
    c.args.storage.max_input_lines = 1_000_000; c.args.storage.max_stream_mib = 1024;
    let next = c.args.lifetime_limits(&args, common, larger).unwrap();
    assert_eq!(next.native.max_model_work, first.native.max_model_work);
    assert_eq!(next.native.masks.max_visits_per_run, first.native.masks.max_visits_per_run);
    assert!(next.buffer_bytes > first.buffer_bytes);
}

#[test]
fn empty_native_axes_and_insufficient_mask_lifetime_fail_closed() {
    let (c, args) = command(Kind::Keyphrases, "start", &[]);
    let common = c.args.validate(&args).unwrap();
    let job = lifetime(&c.args, &args, common);
    for axis in 0..6 {
        let mut changed = job;
        match axis {
            0 => changed.max_work.model.forward_positions = 0,
            1 => changed.max_work.model.projected_logits = 0,
            2 => changed.max_work.model.attention_pairs = 0,
            3 => changed.max_work.model.projections.dot_products = 0,
            4 => changed.max_work.model.projections.multiply_accumulates = 0,
            _ => changed.max_work.mask_node_visits = c.args.source.mask_visits() - 1,
        }
        assert_eq!(c.args.lifetime_limits(&args, common, changed).unwrap_err().code, "lifetime_native_or_mask_limits");
    }
}

#[test]
fn stored_input_and_result_envelopes_remain_inside_cli_admission() {
    let (c, args) = command(Kind::Ner, "start", &[]);
    let common = c.args.validate(&args).unwrap();
    let job = lifetime(&c.args, &args, common);
    let mut too_large = job; too_large.max_input_bytes_per_item += 1;
    assert_eq!(c.args.lifetime_limits(&args, common, too_large).unwrap_err().code, "stored_input_ceiling");
    let mut too_small = job; too_small.max_result_bytes -= 1;
    assert_eq!(c.args.lifetime_limits(&args, common, too_small).unwrap_err().code, "stored_result_ceiling");
    let mut invalid = job; invalid.max_attempts = 0;
    assert_eq!(c.args.lifetime_limits(&args, common, invalid).unwrap_err().code, "job_limits");
}

#[test]
fn complete_population_and_two_preparation_lifetimes_fit_one_process_ceiling() {
    let (c, args) = command(Kind::Answer, "start", &[]);
    let common = c.args.validate(&args).unwrap();
    let job = lifetime(&c.args, &args, common);
    let full = c.args.lifetime_limits(&args, common, job).unwrap();
    assert!(full.buffer_bytes > common.preparation_bytes + 2 * job.max_snapshot_bytes);
    let mut boundary = common; boundary.memory_bytes = full.common.minimum_memory_bytes - 1;
    assert_eq!(c.args.lifetime_limits(&args, boundary, job).unwrap_err().code, "job_population_memory");
    boundary.memory_bytes += 1;
    assert!(c.args.lifetime_limits(&args, boundary, job).is_ok());
    let mut huge = job; huge.max_snapshot_bytes = i64::MAX as u64;
    assert_eq!(c.args.lifetime_limits(&args, common, huge).unwrap_err().code, "job_memory_arithmetic");
}

#[test]
fn immutable_limit_parser_rejects_duplicate_unknown_or_incomplete_contracts() {
    let (c, args) = command(Kind::Ner, "start", &[]);
    let common = c.args.validate(&args).unwrap();
    let job = lifetime(&c.args, &args, common);
    let encoded = serde_json::to_string(&job).unwrap();
    assert_eq!(job_cli::parse_limits(&encoded).unwrap(), job);
    assert!(job_cli::parse_limits(&encoded.replacen('{', "{\"max_items\":1,", 1)).is_err());
    let mut extra = serde_json::to_value(job).unwrap(); extra["task"] = serde_json::json!("ner");
    assert!(job_cli::parse_limits(&extra.to_string()).is_err());
    assert!(job_cli::parse_limits("{}").is_err());
}

#[test]
fn failure_report_names_task_and_preserves_possible_durable_progress_without_private_paths() {
    let (mut c, args) = command(Kind::Answer, "resume", &["--defaults", "private-passages.json"]);
    c.args.storage.store_results = false;
    let mut output = Vec::new(); let mut diagnostics = Vec::new();
    assert_eq!(c.run_owned(args, NoRead, &mut output, &mut diagnostics), ErrorCode::Usage.as_process_exit());
    assert!(output.is_empty());
    let report: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
    assert_eq!(report["task"], "answer"); assert_eq!(report["operation"], "resume");
    assert_eq!(report["durable_progress_may_exist"], true);
    let encoded = std::str::from_utf8(&diagnostics).unwrap();
    for private in ["private", "passages", "limits.json", "0123456789abcdef"] { assert!(!encoded.contains(private)); }
}

#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unavailable_profile_refuses_all_four_jobs_before_private_file_access() {
    for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer] {
        let (c, args) = command(kind, "resume", &["--defaults", "private-defaults.json"]);
        assert_eq!(c.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().code, "owned_job_profile_unavailable");
    }
}


#[test]
fn persistent_source_job_width_preserves_every_semantic_and_lifetime_limit() {
    for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer] {
        for operation in ["start", "resume"] {
            let (serial, serial_args) = command(kind, operation, &[]);
            let serial_common = serial.args.validate(&serial_args).unwrap();
            let fixed = lifetime(&serial.args, &serial_args, serial_common);
            let serial_limits = serial.args.lifetime_limits(&serial_args, serial_common, fixed).unwrap();
            for width in [1, 2, 4, 64] {
                let width_text = width.to_string();
                let (c, args) = command(kind, operation, &["--compute-threads", &width_text]);
                let common = c.args.validate(&args).unwrap();
                let limits = c.args.lifetime_limits(&args, common, fixed).unwrap();
                assert_eq!(common.compute.threads, width);
                assert_eq!(common.compute, args.compute.admit(args.group_width).unwrap());
                assert_eq!(common.compute.workspace_bytes == 0, width == 1);
                assert_eq!(common.native, serial_common.native);
                assert_eq!(c.args.source.budget(&args, common),
                    serial.args.source.budget(&serial_args, serial_common));
                assert_same_native_limits(limits.native, serial_limits.native);
                assert_eq!(limits.buffer_bytes, serial_limits.buffer_bytes);
                assert_eq!(limits.transport, serial_limits.transport);
                assert!(limits.common.minimum_memory_bytes >= serial_limits.common.minimum_memory_bytes);
                #[cfg(feature = "asupersync-runtime")]
                {
                    let host = compute::resource_config(common);
                    assert_eq!(host.scoped_cpu_children_per_coordinator, width - 1);
                    assert_eq!(host.thread_ceiling, width + 1);
                    assert_eq!(host.runtime_workers, 1);
                    assert_eq!(host.max_blocking_coordinators, 1);
                }
            }
        }
    }
}

#[test]
fn worker_width_and_incompatible_schedules_refuse_before_private_source_job_io() {
    for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer] {
        for width in ["0", "65", "18446744073709551615"] {
            let mut words = argv(kind, "start");
            words.extend(["--compute-threads", width].into_iter().map(str::to_owned));
            if let Ok((c, args)) = parse(kind, words) {
                assert!(c.execute_owned(args, NoRead, &mut NoWrite).is_err());
            }
        }
        for option in ["--prefill-rows", "--cohort-heads", "--cohort-documents", "--max-selected-rows"] {
            let mut words = argv(kind, "resume");
            words.extend(["--compute-threads", "4", option, "1"].into_iter().map(str::to_owned));
            assert!(parse(kind, words).is_err());
        }
        let (c, mut args) = command(kind, "resume", &["--compute-threads", "4"]);
        args.memory_mib = 1;
        assert!(c.execute_owned(args, NoRead, &mut NoWrite).is_err());
    }
}

#[test]
fn resumed_source_jobs_keep_lifetime_work_when_worker_width_changes() {
    for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer] {
        let (start, start_args) = command(kind, "start",
            &["--compute-threads", "4", "--spool-input", "--max-retained-input-mib", "16"]);
        let (resume, resume_args) = command(kind, "resume",
            &["--compute-threads", "2", "--use-retained-input", "--max-retained-input-mib", "16"]);
        let start_common = start.args.validate(&start_args).unwrap();
        let resume_common = resume.args.validate(&resume_args).unwrap();
        let fixed = lifetime(&start.args, &start_args, start_common);
        let first = start.args.lifetime_limits(&start_args, start_common, fixed).unwrap();
        let next = resume.args.lifetime_limits(&resume_args, resume_common, fixed).unwrap();
        assert_same_native_limits(first.native, next.native);
        assert_eq!(first.buffer_bytes, next.buffer_bytes);
        assert_eq!(start.args.source.budget(&start_args, start_common),
            resume.args.source.budget(&resume_args, resume_common));
        let mut boundary = resume_common;
        boundary.memory_bytes = next.common.minimum_memory_bytes - 1;
        assert_eq!(resume.args.lifetime_limits(&resume_args, boundary, fixed).unwrap_err().code,
            "job_population_memory");
    }
}


#[test]
fn compact_source_jobs_parse_fixed_call_and_invocation_caps_for_all_tasks_and_widths() {
    use crate::tokenizer::untrusted::UntrustedEncoding;
    for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer] {
        for operation in ["start", "resume"] {
            for width in ["1", "4"] {
                let (c, args) = command(kind, operation, &["--compute-threads", width,
                    "--max-input-bytes", "8192", "--input-encoding", "safe-bpe-v1",
                    "--max-tokenizer-pair-checks", "7777", "--max-tokenizer-pair-bytes", "65536",
                    "--max-total-tokenizer-input-bytes", "32768",
                    "--max-total-tokenizer-output-tokens", "16384",
                    "--max-total-tokenizer-pair-checks", "100000",
                    "--max-total-tokenizer-pair-bytes", "1000000"]);
                let common = c.args.validate(&args).unwrap();
                let UntrustedEncoding::SafeBpeV1(call) = args.input_encoding.policy(
                    args.max_input_bytes, common.max_prompt_tokens).unwrap() else { panic!("wrong input policy"); };
                assert_eq!(call.max_input_bytes, 8192);
                assert_eq!(call.max_output_tokens, common.max_prompt_tokens);
                assert_eq!(call.max_pair_checks, 7777);
                assert_eq!(call.max_pair_bytes, 65536);
                assert_eq!(c.args.encoding.limits(&args).unwrap(), Some(SafeBpeWorkLimits {
                    max_input_bytes: 32768, max_output_tokens: 16384,
                    max_pair_checks: 100000, max_pair_bytes: 1000000 }));
                assert_eq!(c.args.encoding.workspace_bytes(&args, common.max_prompt_tokens).unwrap(),
                    call.requirements().unwrap().workspace_bytes + 1024);
                assert_eq!(common.compute.threads.to_string(), width);
            }
        }
    }
}

#[test]
fn invalid_or_unselected_compact_work_caps_refuse_before_private_source_job_io() {
    for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer] {
        for flag in ["--max-total-tokenizer-input-bytes", "--max-total-tokenizer-output-tokens",
            "--max-total-tokenizer-pair-checks", "--max-total-tokenizer-pair-bytes"] {
            let (c, args) = command(kind, "start", &[flag, "1"]);
            assert_eq!(c.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().code,
                "source_job_input_encoding");
            for value in ["0", "18446744073709551615"] {
                let (c, args) = command(kind, "resume",
                    &["--input-encoding", "safe-bpe-v1", flag, value]);
                assert_eq!(c.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().code,
                    "source_job_tokenizer_work");
            }
        }
        for flag in ["--max-tokenizer-pair-checks", "--max-tokenizer-pair-bytes"] {
            let (c, args) = command(kind, "resume",
                &["--input-encoding", "safe-bpe-v1", "--max-input-bytes", "8192", flag, "0"]);
            assert_eq!(c.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().exit, ErrorCode::Usage);
        }
    }
}

#[test]
fn compact_workspace_adds_to_full_and_retained_input_without_renewing_native_work() {
    for retained in [false, true] {
        let mut options = vec!["--max-input-bytes", "8192", "--compute-threads", "4"];
        if retained { options.extend(["--use-retained-input", "--max-retained-input-mib", "16"]); }
        let (byte, byte_args) = command(Kind::Answer, "resume", &options);
        let common = byte.args.validate(&byte_args).unwrap();
        let fixed = lifetime(&byte.args, &byte_args, common);
        let baseline = byte.args.lifetime_limits(&byte_args, common, fixed).unwrap();
        assert_eq!(byte.args.encoding.limits(&byte_args).unwrap(), None);
        assert_eq!(baseline.encoding_workspace_bytes, 0);
        options.extend(["--input-encoding", "safe-bpe-v1"]);
        let (compact, args) = command(Kind::Answer, "resume", &options);
        let common = compact.args.validate(&args).unwrap();
        let selected = compact.args.lifetime_limits(&args, common, fixed).unwrap();
        assert!(selected.encoding_workspace_bytes > 1024);
        assert_eq!(selected.buffer_bytes, baseline.buffer_bytes);
        assert_same_native_limits(selected.native, baseline.native);
        assert_eq!(selected.common.compute, baseline.common.compute);
        assert_eq!(selected.common.native, baseline.common.native);
        assert_eq!(selected.common.minimum_memory_bytes,
            baseline.common.minimum_memory_bytes + selected.encoding_workspace_bytes);
        let mut short = common; short.memory_bytes = selected.common.minimum_memory_bytes - 1;
        assert_eq!(compact.args.lifetime_limits(&args, short, fixed).unwrap_err().code,
            "job_population_memory");
        short.memory_bytes += 1;
        assert!(compact.args.lifetime_limits(&args, short, fixed).is_ok());
    }
}
