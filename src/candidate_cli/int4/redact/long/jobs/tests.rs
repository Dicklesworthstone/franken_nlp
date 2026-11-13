//! Durable command authority, complete-population limits and independent key scopes.
use super::*;
use crate::tasks::redact::batch::RedactionBatchArgs;

fn argv(operation: &str) -> Vec<String> {
    ["redact-job", operation, "--source-dir", "explicit-source", "--source-manifest", "explicit-manifest",
        "--memory-mib", "65536", "--store-results", "--job-dir", "private-job", "--key-file", "private-job-key",
        "--job-id", "0123456789abcdef0123456789abcdef", "--limits", "private-limits"]
        .into_iter().map(str::to_owned).collect()
}
fn parse(words: Vec<String>) -> Result<(JobCommand, Int4Args), clap::Error> {
    JobCommand::from_matches(&definition().try_get_matches_from(words)?)
}
pub(super) fn command(operation: &str, extra: &[&str]) -> (JobCommand, Int4Args) {
    let mut words = argv(operation); words.extend(extra.iter().map(|s| (*s).to_owned()));
    parse(words).unwrap()
}
pub(super) fn lifetime(args: &Int4Args, document: DocumentLimits) -> JobLimits {
    let mut item = document.mapping.max_model_work;
    item.projected_logits -= item.projected_logits % NANBEIGE_VOCAB_SIZE as u64;
    JobLimits { max_items: 3, max_id_bytes: 128,
        max_input_bytes_per_item: args.max_input_bytes * 2 + 1024,
        max_snapshot_bytes: 16 * MIB, max_result_bytes: document.common.result_bytes,
        max_spool_bytes: 16 * MIB, max_materialized_bytes: 16 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 5,
        max_work: JobWork::from_int4(item, document.mapping.max_mask_visits) }
}
pub(super) struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refusal consumed private input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refusal published private output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refusal flushed output"); }
}

#[test]
fn complete_dispatch_requires_owned_input_and_supports_both_model_routes_and_modes() {
    for operation in ["start", "resume"] {
        let (job, args) = command(operation, &["--materialize"]);
        assert_eq!(job.mode.name(), operation);
        assert!(job.args.storage.store_results && job.args.storage.materialize);
        assert!(job.args.validate(&args).unwrap().common.snapshot_pin.is_none());
        let mut words = argv(operation);
        let at = words.iter().position(|s| s == "--source-dir").unwrap(); words.drain(at..at + 4);
        words.extend(["--snapshot", "pinned.int4", "--expected-model-sha256"].into_iter().map(str::to_owned));
        words.push("ab".repeat(32));
        let (snapshot, args) = parse(words).unwrap();
        assert!(snapshot.args.validate(&args).unwrap().common.snapshot_pin.is_some());
        let mut root = vec!["candidate".to_owned(), "source-int4".to_owned()];
        root.extend(argv(operation));
        let matches = crate::candidate_cli::definition().try_get_matches_from(root).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong candidate dispatcher"); };
        assert!(matches!(command.task, Int4Task::RedactJob));
        assert_eq!(command.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn compact_job_options_survive_start_resume_and_pinned_snapshot_dispatch() {
    let flags = ["--input-encoding", "safe-bpe-v1", "--max-input-bytes", "4096",
        "--max-result-bytes", "65536", "--max-ner-chunk-bytes", "128",
        "--max-tokenizer-pair-checks", "1000", "--max-tokenizer-pair-bytes", "8192",
        "--max-total-tokenizer-input-bytes", "200000", "--max-total-tokenizer-output-tokens", "100000",
        "--max-total-tokenizer-pair-checks", "3000", "--max-total-tokenizer-pair-bytes", "32768"];
    for operation in ["start", "resume"] {
        let (job, args) = command(operation, &flags);
        let document = job.args.validate(&args).unwrap();
        let selected = document.encoding.unwrap();
        assert_eq!(selected.call.max_input_bytes, 128);
        assert_eq!(selected.call.max_output_tokens, 128);
        assert_eq!(selected.call.max_pair_checks, 1000);
        assert_eq!(selected.call.max_pair_bytes, 8192);
        assert_eq!(selected.complete.max_input_bytes, 200000);
        assert_eq!(selected.complete.max_output_tokens, 100000);
        assert_eq!(selected.complete.max_pair_checks, 3000);
        assert_eq!(selected.complete.max_pair_bytes, 32768);
        let mut words = argv(operation);
        let at = words.iter().position(|s| s == "--source-dir").unwrap(); words.drain(at..at + 4);
        words.extend(["--snapshot", "pinned.int4", "--expected-model-sha256"].into_iter().map(str::to_owned));
        words.push("ab".repeat(32)); words.extend(flags.iter().map(|s| (*s).to_owned()));
        let (snapshot, args) = parse(words).unwrap();
        let reload = snapshot.args.validate(&args).unwrap();
        assert_eq!(reload.encoding, Some(selected));
        assert!(reload.common.snapshot_pin.is_some());
        let mut root = vec!["candidate".to_owned(), "source-int4".to_owned()];
        root.extend(argv(operation)); root.extend(flags.iter().map(|s| (*s).to_owned()));
        let matches = crate::candidate_cli::definition().try_get_matches_from(root).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong compact job dispatcher"); };
        assert_eq!(command.args.input_encoding, args.input_encoding);
        assert!(matches!(command.task, Int4Task::RedactJob));
    }
}

#[test]
fn compact_job_tokenizer_allowances_do_not_multiply_lifetime_native_or_attempt_authority() {
    for operation in ["start", "resume"] {
        let (plain, args) = command(operation, &[]);
        let baseline = plain.args.validate(&args).unwrap();
        assert!(baseline.encoding.is_none());
        let stored = lifetime(&args, baseline);
        let original = plain.args.lifetime_limits(&args, baseline, stored).unwrap();
        let (job, args) = command(operation, &["--input-encoding", "safe-bpe-v1"]);
        let document = job.args.validate(&args).unwrap();
        let compact = document.encoding.unwrap();
        assert_eq!(compact.complete.max_input_bytes,
            8 * (args.max_input_bytes as u64 + document.common.result_bytes as u64));
        assert_eq!(compact.complete.max_output_tokens, compact.complete.max_input_bytes);
        let limits = job.args.lifetime_limits(&args, document, stored).unwrap();
        assert_eq!(limits.document.encoding, Some(compact));
        assert_eq!(limits.model_work, original.model_work);
        assert_eq!(limits.masks, original.masks);
        assert_eq!(limits.common.native, original.common.native);
        let mut population = stored; population.max_items = 999; population.max_attempts = 9999;
        let expanded = job.args.lifetime_limits(&args, document, population).unwrap();
        assert_eq!(expanded.document.encoding, Some(compact));
        assert_eq!(expanded.model_work, limits.model_work);
        assert_eq!(expanded.masks, limits.masks);
        for schedule in [vec!["--cohort-chunks", "3"], vec!["--compute-threads", "3"]] {
            let mut flags = vec!["--input-encoding", "safe-bpe-v1"]; flags.extend(schedule);
            let (scheduled, args) = command(operation, &flags);
            assert_eq!(scheduled.args.validate(&args).unwrap().encoding, Some(compact));
        }
    }
    let (retained, args) = command("resume", &["--use-retained-input", "--max-retained-input-mib", "16",
        "--input-encoding", "safe-bpe-v1"]);
    assert!(retained.args.validate(&args).unwrap().encoding.is_some());
}

#[test]
fn invalid_compact_job_limits_refuse_before_job_keys_limits_or_population_io() {
    for operation in ["start", "resume"] {
        for flag in ["--max-total-tokenizer-input-bytes", "--max-total-tokenizer-output-tokens",
            "--max-total-tokenizer-pair-checks", "--max-total-tokenizer-pair-bytes"] {
            let (without_policy, args) = command(operation, &[flag, "1"]);
            assert_eq!(without_policy.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code,
                "invalid_job_options");
            for invalid in ["0", "18446744073709551615"] {
                let (compact, args) = command(operation, &["--input-encoding", "safe-bpe-v1", flag, invalid]);
                assert_eq!(compact.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code,
                    "invalid_job_options");
            }
        }
        let (compact, args) = command(operation, &["--input-encoding", "safe-bpe-v1",
            "--max-ner-chunks", "1", "--max-ner-chunk-bytes", "1048576", "--preparation-mib", "512"]);
        assert_eq!(compact.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code, "invalid_job_options");
        for flag in ["--max-tokenizer-pair-checks", "--max-tokenizer-pair-bytes"] {
            let mut words = argv(operation); words.extend([flag, "1"].into_iter().map(str::to_owned));
            assert!(parse(words).is_err());
        }
    }
}

#[test]
fn repair_is_resume_only_and_retention_paths_and_immutable_limits_are_explicit() {
    assert!(matches!(command("resume", &[]).0.mode, RunMode::Resume { discard_uncommitted: false }));
    assert!(matches!(command("resume", &["--discard-uncommitted-tail"]).0.mode,
        RunMode::Resume { discard_uncommitted: true }));
    for required in ["--job-dir", "--key-file", "--job-id", "--limits"] {
        let mut words = argv("start");
        let at = words.iter().position(|s| s == required).unwrap(); words.drain(at..at + 2);
        assert!(parse(words).is_err());
    }
    let mut words = argv("start"); words.push("--discard-uncommitted-tail".to_owned());
    assert!(parse(words).is_err());
    assert!(definition().try_get_matches_from(["redact-job"]).is_err());
    for operation in ["status", "verify", "materialize", "partition", "merge"] {
        assert!(definition().try_get_matches_from(["redact-job", operation]).is_err());
    }
    // Independently requested snapshot setup keeps the existing INT4 job contract.
    assert!(command("resume", &["--save-snapshot", "new-model.int4"]).0.args.storage.store_results);
}

#[test]
fn storage_and_private_key_refusals_happen_before_any_owned_io() {
    for case in 0..8 {
        let (mut job, args) = command("start", &[]);
        match case {
            0 => job.args.storage.store_results = false,
            1 => job.args.storage.job_dir = PathBuf::from("-"),
            2 => job.args.storage.key_file = PathBuf::new(),
            3 => job.args.storage.limits_file = PathBuf::from("-"),
            4 => job.args.storage.max_stream_mib = 0,
            5 => job.args.storage.max_input_lines = 0,
            6 => job.args.storage.journal_memory_mib = 0,
            _ => job.args.storage.serialization_memory_mib = 0,
        }
        assert_eq!(job.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code,
            "explicit_storage_or_job_options");
    }
    for flags in [vec!["--key-stdin"], vec!["--action", "pseudonymize"],
        vec!["--action", "pseudonymize", "--key-stdin", "--key-id", "k", "--namespace", "n"]] {
        let (job, args) = command("start", &flags);
        assert_eq!(job.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code, "invalid_job_options");
    }
    let (job, args) = command("start", &["population.ndjson", "--action", "pseudonymize",
        "--key-stdin", "--key-id", "rotation", "--namespace", "private-scope"]);
    assert_eq!(job.args.storage.key_file, PathBuf::from("private-job-key"));
    assert!(job.args.validate(&args).is_ok());
}

#[test]
fn retained_job_cohorts_keep_exact_lifetime_authority_and_one_admitted_peak_in_both_modes() {
    for operation in ["start", "resume"] {
        let (serial, args) = command(operation, &[]);
        let document = serial.args.validate(&args).unwrap();
        let stored = lifetime(&args, document);
        let baseline = serial.args.lifetime_limits(&args, document, stored).unwrap();
        for width in ["1", "3"] {
            let (cohorts, args) = command(operation, &["--cohort-chunks", width]);
            let document = cohorts.args.validate(&args).unwrap();
            let selected = cohorts.args.lifetime_limits(&args, document, stored).unwrap();
            let quote = document.cohort.unwrap();
            let original = baseline.common.native;
            let native_delta = quote.native.kv_bytes + quote.native.rope_bytes + quote.native.scratch_payload_bound
                - original.kv_bytes - original.rope_bytes - original.scratch_payload_bound;
            assert_eq!(document.mapping.reduction.map_batch_chunks, width.parse::<usize>().unwrap());
            assert_eq!(selected.common.minimum_memory_bytes,
                baseline.common.minimum_memory_bytes + native_delta + quote.workspace_bytes
                    + 4 * INT4_LONG_REDACTION_COHORT_ENVELOPE_BYTES);
            assert_eq!(selected.model_work, stored.max_work.int4_model_work());
            assert_eq!(selected.masks, stored.max_work.mask_node_visits);
            assert_eq!(selected.common.native, baseline.common.native);
            assert_eq!(selected.common.result_bytes, baseline.common.result_bytes);
            assert_eq!(selected.buffer_bytes, baseline.buffer_bytes);
            assert!(matches!(cohorts.args.lifetime_limits(&args, baseline.document, stored),
                Err(Failure { code: "redaction_chunk_schedule", .. })));
        }
        for flags in [vec!["--cohort-chunks", "0"], vec!["--cohort-chunks", "65"],
            vec!["--cohort-chunks", "3", "--max-ner-chunks", "2"], vec!["--cohort-chunks", "64"]] {
            let (job, args) = command(operation, &flags);
            assert_eq!(job.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code, "invalid_job_options");
        }
    }
}

#[test]
fn retained_resume_authenticates_pseudonym_scope_without_an_external_population() {
    let (job, args) = command("resume", &["--use-retained-input", "--max-retained-input-mib", "16",
        "--action", "pseudonymize", "--key-stdin", "--key-id", "rotation", "--namespace", "private-scope"]);
    assert_eq!(args.input, PathBuf::from("-"));
    assert!(job.args.validate(&args).is_ok());
    let (external, args) = command("resume", &["--action", "pseudonymize", "--key-stdin",
        "--key-id", "rotation", "--namespace", "private-scope"]);
    assert!(external.args.validate(&args).is_err());
    let (wrong_mode, args) = command("start", &["--use-retained-input", "--max-retained-input-mib", "16"]);
    assert_eq!(wrong_mode.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code,
        "job_input_retention_options");
}

#[test]
fn fixed_redaction_options_and_original_text_cap_remain_independent_of_encoded_envelopes() {
    let (job, args) = command("start", &["--max-input-bytes", "32", "--rules", "email,date",
        "--action", "placeholder", "--include-map", "--no-verify", "--ner-options", "typed-options.json"]);
    let document = job.args.validate(&args).unwrap();
    let stored = lifetime(&args, document);
    assert!(stored.max_input_bytes_per_item > args.max_input_bytes);
    let limits = job.args.lifetime_limits(&args, document, stored).unwrap();
    let request = job.args.document.request(&args, limits.document);
    assert_eq!(request.actions.default_action, RedactionAction::Placeholder);
    assert!(request.actions.include_map); assert!(!request.verify);
    assert_eq!(limits.document.mapping.chunks.max_input_bytes, limits.common.result_bytes);
    assert_eq!(request.rule_budget.max_input_bytes, limits.common.result_bytes);
    assert_eq!(job.args.document.redaction.source.options_path().unwrap(), std::path::Path::new("typed-options.json"));
    assert!(serde_json::from_str::<RedactionBatchArgs>("{}").is_ok());
    for text in [r#"{"verify":false}"#, r#"{"action":"mask"}"#, r#"{"namespace":"changed"}"#,
        r#"{"key":"secret"}"#, r#"{"types":["person"]}"#, r#"{"max_chunks":1}"#] {
        assert!(serde_json::from_str::<RedactionBatchArgs>(text).is_err());
    }
}

#[test]
fn all_lifetime_work_is_copied_exactly_and_items_attempts_or_transport_never_replenish_it() {
    let (mut job, args) = command("resume", &[]);
    let document = job.args.validate(&args).unwrap();
    let mut stored = lifetime(&args, document);
    stored.max_work.model.projected_logits += 17;
    let original = job.args.lifetime_limits(&args, document, stored).unwrap();
    assert_eq!(original.model_work, stored.max_work.int4_model_work());
    assert_eq!(original.masks, stored.max_work.mask_node_visits);
    assert_eq!(original.model_work.projected_logits % NANBEIGE_VOCAB_SIZE as u64, 17);
    stored.max_items = 999; stored.max_attempts = 9999;
    job.args.storage.max_stream_mib = 1; job.args.storage.max_input_lines = 1;
    let changed = job.args.lifetime_limits(&args, document, stored).unwrap();
    assert_eq!(changed.model_work, original.model_work); assert_eq!(changed.masks, original.masks);
    assert_eq!(changed.transport.max_stream_bytes, MIB); assert_eq!(changed.transport.max_lines, 1);
    assert!(changed.buffer_bytes > original.buffer_bytes);
}

#[test]
fn every_native_axis_mask_and_retained_result_bound_must_fit_one_complete_attempt() {
    let (job, args) = command("start", &[]);
    let document = job.args.validate(&args).unwrap();
    let stored = lifetime(&args, document);
    for axis in 0..6 {
        let mut narrow = stored;
        match axis {
            0 => narrow.max_work.model.forward_positions -= 1,
            1 => narrow.max_work.model.projected_logits -= 1,
            2 => narrow.max_work.model.attention_pairs -= 1,
            3 => narrow.max_work.model.projections.dot_products -= 1,
            4 => narrow.max_work.model.projections.multiply_accumulates -= 1,
            _ => narrow.max_work.mask_node_visits -= 1,
        }
        assert!(matches!(job.args.lifetime_limits(&args, document, narrow),
            Err(Failure { code: "lifetime_native_or_mask_limits", .. })));
    }
    let mut small = stored; small.max_result_bytes = document.common.result_bytes - 1;
    assert!(matches!(job.args.lifetime_limits(&args, document, small),
        Err(Failure { code: "stored_result_ceiling", .. })));
    let mut overflow = stored; overflow.max_snapshot_bytes = i64::MAX as u64;
    assert!(matches!(job.args.lifetime_limits(&args, document, overflow),
        Err(Failure { code: "job_memory_arithmetic", .. })));
}

#[test]
fn population_memory_replaces_capture_but_retains_full_map_native_secret_and_two_preparations() {
    let (job, mut args) = command("start", &[]);
    let document = job.args.validate(&args).unwrap();
    let stored = lifetime(&args, document);
    let limits = job.args.lifetime_limits(&args, document, stored).unwrap();
    let expected = document.common.minimum_memory_bytes - args.max_input_bytes as u64 - 1
        - document.common.preparation_bytes + limits.buffer_bytes + RECIPE_RESERVE_BYTES;
    assert_eq!(limits.common.minimum_memory_bytes, expected);
    assert_eq!(limits.document.workspace_bytes, document.workspace_bytes);
    assert_eq!(limits.common.native, document.common.native);
    assert_eq!(limits.preparation_reserve_bytes, document.common.preparation_bytes);
    let mut larger = stored; larger.max_result_bytes *= 2;
    let expanded = job.args.lifetime_limits(&args, document, larger).unwrap();
    assert_eq!(expanded.common.minimum_memory_bytes - expected, stored.max_result_bytes as u64 * 4);
    assert_eq!(expanded.common.result_bytes, limits.common.result_bytes);
    args.memory_mib = expected.saturating_sub(1) / MIB;
    let smaller = job.args.validate(&args).unwrap();
    assert!(matches!(job.args.lifetime_limits(&args, smaller, stored),
        Err(Failure { code: "job_population_memory", .. })));
    args.memory_mib = expected.div_ceil(MIB);
    let admitted = job.args.validate(&args).unwrap();
    assert!(job.args.lifetime_limits(&args, admitted, stored).is_ok());
}

#[test]
fn live_stream_unrequested_storage_and_generation_options_cannot_silently_change_a_job() {
    for extra in [vec!["--max-requests", "3"], vec!["--ndjson"], vec!["--chunked"],
        vec!["--max-corpus-mask-node-visits", "9"], vec!["--rules-only"],
        vec!["--defaults", "private"], vec!["--spool-input"], vec!["--key", "secret"],
        vec!["--prefill-rows", "1"], vec!["--seed", "ab"], vec!["--logprobs"]] {
        let mut words = argv("start"); words.extend(extra.into_iter().map(str::to_owned));
        assert!(parse(words).is_err());
    }
    for bad in ["AB".repeat(16), "ab".repeat(15), "é".repeat(16), "x".repeat(32)] {
        let mut words = argv("start"); let at = words.iter().position(|s| s == "--job-id").unwrap() + 1;
        words[at] = bad; assert!(parse(words).is_err());
    }
    let (job, mut args) = command("start", &[]); args.seed = Some("ab".repeat(32));
    assert_eq!(job.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code, "invalid_job_options");
}

#[test]
fn exact_kernel_selection_is_hardware_checked_without_changing_job_or_storage_authority() {
    let (baseline, args) = command("start", &[]);
    let baseline = baseline.args.validate(&args).unwrap();
    for kind in Int8KernelKind::ALL {
        let (job, args) = command("start", &["--kernel", kind.id()]);
        assert_eq!(args.kernel, kind);
        match PackedInt4Kernel::detect(kind) {
            Ok(_) => {
                let selected = job.args.validate(&args).unwrap();
                assert_eq!(selected.common.kernel.kind(), kind);
                assert_eq!(selected.common.minimum_memory_bytes, baseline.common.minimum_memory_bytes);
                assert_eq!(selected.common.native, baseline.common.native);
                assert_eq!(selected.mapping.max_model_work, baseline.mapping.max_model_work);
            }
            Err(_) => assert_eq!(job.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code, "invalid_job_options"),
        }
    }
}

#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unavailable_owned_profile_refuses_without_reading_keys_or_the_population() {
    for operation in ["start", "resume"] {
        let (job, args) = command(operation, &[]);
        assert_eq!(job.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code, "owned_job_profile_unavailable");
    }
}

#[test]
fn retained_job_team_admits_one_workspace_without_replenishing_any_lifetime_axis() {
    for operation in ["start", "resume"] {
        let (serial, args) = command(operation, &[]);
        let document = serial.args.validate(&args).unwrap();
        let stored = lifetime(&args, document);
        let baseline = serial.args.lifetime_limits(&args, document, stored).unwrap();
        let request = crate::canonjson::canonical_bytes(
            &serial.args.document.request(&args, baseline.document)).unwrap();
        for threads in [2, 3, 64] {
            let width = threads.to_string();
            let (threaded, mut args) = command(operation, &["--compute-threads", &width]);
            let document = threaded.args.validate(&args).unwrap();
            let selected = threaded.args.lifetime_limits(&args, document, stored).unwrap();
            let extra = crate::native_engine::strict_int4::projection::int4_team_workspace_bytes(
                threads, args.group_width).unwrap();
            assert_eq!(selected.common.compute.threads, threads);
            assert_eq!(selected.common.compute.workspace_bytes, extra);
            assert!(document.cohort.is_none());
            assert_eq!(document.mapping.reduction.map_batch_chunks, 1);
            assert_eq!(selected.common.native, baseline.common.native);
            assert_eq!(selected.common.result_bytes, baseline.common.result_bytes);
            assert_eq!(selected.document.workspace_bytes, baseline.document.workspace_bytes);
            assert_eq!(selected.document.mapping.max_model_work, baseline.document.mapping.max_model_work);
            assert_eq!(selected.document.mapping.verification, baseline.document.mapping.verification);
            assert_eq!(selected.buffer_bytes, baseline.buffer_bytes);
            assert_eq!(selected.model_work, stored.max_work.int4_model_work());
            assert_eq!(selected.masks, stored.max_work.mask_node_visits);
            assert_eq!(selected.transport, baseline.transport);
            assert_eq!(crate::canonjson::canonical_bytes(
                &threaded.args.document.request(&args, selected.document)).unwrap(), request);
            assert!(selected.common.minimum_memory_bytes >= baseline.common.minimum_memory_bytes);
            assert!(selected.common.minimum_memory_bytes <= baseline.common.minimum_memory_bytes + extra);
            let mut more_attempts = stored; more_attempts.max_items *= 2; more_attempts.max_attempts *= 2;
            let unchanged = threaded.args.lifetime_limits(&args, document, more_attempts).unwrap();
            assert_eq!(unchanged.model_work, selected.model_work);
            assert_eq!(unchanged.masks, selected.masks);
            for axis in 0..6 {
                let mut narrow = stored;
                match axis {
                    0 => narrow.max_work.model.forward_positions -= 1,
                    1 => narrow.max_work.model.projected_logits -= 1,
                    2 => narrow.max_work.model.attention_pairs -= 1,
                    3 => narrow.max_work.model.projections.dot_products -= 1,
                    4 => narrow.max_work.model.projections.multiply_accumulates -= 1,
                    _ => narrow.max_work.mask_node_visits -= 1,
                }
                assert!(matches!(threaded.args.lifetime_limits(&args, document, narrow),
                    Err(Failure { code: "lifetime_native_or_mask_limits", .. })));
            }
            args.memory_mib = selected.common.minimum_memory_bytes.saturating_sub(1) / MIB;
            assert!(threaded.args.validate(&args)
                .and_then(|document| threaded.args.lifetime_limits(&args, document, stored)).is_err());
            args.memory_mib = selected.common.minimum_memory_bytes.div_ceil(MIB);
            let admitted = threaded.args.validate(&args).unwrap();
            threaded.args.lifetime_limits(&args, admitted, stored).unwrap();
        }
    }
}

#[test]
fn population_teams_keep_retained_input_and_pseudonym_key_authority_separate() {
    for operation in ["start", "resume"] {
        for flags in [vec!["--compute-threads", "0"], vec!["--compute-threads", "65"],
            vec!["--compute-threads", "3", "--cohort-chunks", "1"],
            vec!["--compute-threads", "3", "--cohort-chunks", "3"],
            vec!["--compute-threads", "3", "--action", "pseudonymize", "--key-stdin",
                "--key-id", "rotation", "--namespace", "private-scope"]] {
            let (job, args) = command(operation, &flags);
            assert_eq!(job.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code, "invalid_job_options");
        }
        for chunks in ["1", "3"] {
            let (grouped, args) = command(operation, &["--cohort-chunks", chunks, "--compute-threads", "1"]);
            assert!(grouped.args.validate(&args).unwrap().cohort.is_some());
        }
        let mut words = argv(operation);
        let at = words.iter().position(|s| s == "--source-dir").unwrap(); words.drain(at..at + 4);
        words.extend(["--snapshot", "pinned.int4", "--expected-model-sha256"]
            .into_iter().map(str::to_owned));
        words.push("ab".repeat(32));
        words.extend(["--compute-threads", "3"].into_iter().map(str::to_owned));
        let (snapshot, args) = parse(words).unwrap();
        let admitted = snapshot.args.validate(&args).unwrap();
        assert!(admitted.common.snapshot_pin.is_some());
        assert_eq!(admitted.common.compute.threads, 3);
    }
    let (retained, args) = command("resume", &["--compute-threads", "3",
        "--use-retained-input", "--max-retained-input-mib", "16", "--action", "pseudonymize",
        "--key-stdin", "--key-id", "rotation", "--namespace", "private-scope"]);
    let admitted = retained.args.validate(&args).unwrap();
    assert_eq!(admitted.common.compute.threads, 3);
    assert_eq!(retained.args.storage.key_file, PathBuf::from("private-job-key"));
    assert_eq!(args.input, PathBuf::from("-"));
    let help = definition().find_subcommand_mut("resume").unwrap().render_long_help().to_string()
        .split_whitespace().collect::<Vec<_>>().join(" ");
    for phrase in ["--compute-threads", "--use-retained-input", "entire retained invocation",
        "Physical worker count", "never renew"] {
        assert!(help.contains(phrase), "missing {phrase}");
    }
}
