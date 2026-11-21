use super::*;

const SEED: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn argv<'a>(extra: &'a [&'a str]) -> Vec<&'a str> {
    let mut argv = vec!["candidate", "source-int4", "sentiment-sampled",
        "--source-dir", "private-source", "--source-manifest", "private-manifest",
        "--memory-mib", "16384", "--seed", SEED];
    argv.extend_from_slice(extra);
    argv
}

fn command(extra: &[&str]) -> (Int4Args, SampledArgs) {
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv(extra)).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong candidate dispatcher"); };
    let Int4Task::SampledSentiment(sampled) = command.task else { panic!("wrong task dispatcher"); };
    (command.args, sampled)
}

struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused request reached input IO"); }
}

#[test]
fn selected_rows_preserve_exact_sampling_and_full_memory_on_every_single_request_schedule() {
    use crate::tasks::sentiment::sampled::{SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION,
        SAMPLED_SENTIMENT_SPARSE_REUSED_PREFILL_EXECUTION};
    for compact in [false, true] { for route in 0..3 {
        for schedule in [&[][..], &["--reuse-prefill"][..], &["--compute-threads", "4"][..],
            &["--reuse-prefill", "--compute-threads", "4"][..], &["--prefill-rows", "4"][..],
            &["--reuse-prefill", "--prefill-rows", "4"][..]] {
            let mut flags = schedule.to_vec();
            if compact { flags.extend(["--input-encoding", "safe-bpe-v1"]); }
            let (mut args, dense) = command(&flags);
            if route == 1 { args.save_snapshot = Some(PathBuf::from("new-private-snapshot")); }
            if route == 2 {
                args.source_dir = None; args.source_manifest = None;
                args.snapshot = Some(PathBuf::from("private-snapshot"));
                args.expected_model_sha256 = Some(SEED.to_owned());
            }
            flags.extend(["--max-selected-rows", "4096"]);
            let (_, selected) = command(&flags);
            let baseline = dense.validate(&args).unwrap();
            let admitted = selected.validate(&args).unwrap();
            assert_eq!(dense.selected_rows().unwrap(), None);
            assert_eq!(selected.selected_rows().unwrap(), Some(Int4JsonSparseLimits { max_rows_per_step: 4096 }));
            assert_eq!(selected.options(&args).unwrap(), dense.options(&args).unwrap());
            assert_eq!(selected.budget(&args, admitted, 4).unwrap(), dense.budget(&args, baseline, 4).unwrap());
            assert_eq!(selected.work_ceiling(), dense.work_ceiling());
            assert_eq!(selected.masks(), dense.masks());
            assert_eq!(selected.max_mask_node_visits, dense.max_mask_node_visits);
            assert_eq!(admitted.native, baseline.native);
            assert_eq!(admitted.compute, baseline.compute);
            assert_eq!(admitted.preparation_bytes, baseline.preparation_bytes);
            assert_eq!(admitted.minimum_memory_bytes, baseline.minimum_memory_bytes);
            assert_eq!(selected.prefill_reuse_execution(),
                selected.reuse_prefill.then_some(SAMPLED_SENTIMENT_SPARSE_REUSED_PREFILL_EXECUTION));
            assert_eq!(dense.prefill_reuse_execution(),
                dense.reuse_prefill.then_some(SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION));
        }
    } }
}

#[test]
fn every_sampled_family_refuses_invalid_selected_policy_before_any_private_io() {
    struct NoWrite;
    impl Write for NoWrite {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused sampled command published output"); }
        fn flush(&mut self) -> io::Result<()> { panic!("refused sampled command flushed output"); }
    }
    for (task, operation) in [
        ("sentiment-sampled", None), ("sentiment-sampled-map", None),
        ("sentiment-sampled-batch", None), ("sentiment-sampled-corpus", None),
        ("sentiment-sampled-job", Some("start")), ("sentiment-sampled-job", Some("resume")),
        ("sentiment-sampled-document-job", Some("start")),
        ("sentiment-sampled-document-job", Some("resume")),
    ] {
        let mut words = vec!["candidate", "source-int4", task];
        if let Some(operation) = operation { words.push(operation); }
        words.extend(["--source-dir", "private-source", "--source-manifest", "private-manifest",
            "--memory-mib", "131072", "--seed", SEED]);
        if operation.is_some() {
            words.extend(["--store-results", "--job-dir", "private-job", "--key-file", "private-key",
                "--limits", "private-limits", "--job-id", "0123456789abcdef0123456789abcdef"]);
        }
        for cap in ["1", "166144"] {
            let mut valid = words.clone(); valid.extend(["--max-selected-rows", cap]);
            let matches = crate::candidate_cli::definition().try_get_matches_from(valid).unwrap();
            assert!(CandidateCommand::from_matches(&matches).is_ok());
        }
        for flags in [&["--max-selected-rows", "0"][..], &["--max-selected-rows", "166145"][..],
            &["--max-selected-rows", "18446744073709551615"][..],
            &["--max-selected-rows", "18446744073709551616"][..],
            &["--max-selected-rows", "4096", "--compute-threads", "2", "--prefill-rows", "4"][..]] {
            let mut invalid = words.clone(); invalid.extend_from_slice(flags);
            if let Ok(matches) = crate::candidate_cli::definition().try_get_matches_from(invalid) {
                let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
                    else { panic!("selected policy escaped the INT4 dispatcher"); };
                let mut diagnostics = Vec::new();
                assert_eq!(command.run_owned(NoRead, NoWrite, &mut diagnostics), ErrorCode::Usage.as_process_exit(),
                    "{task} {operation:?} {flags:?}");
                let text = String::from_utf8(diagnostics).unwrap();
                assert!(!text.contains("private-") && !text.contains(SEED));
            }
        }
    }
}

#[test]
fn prompt_reuse_is_explicit_and_preserves_cold_admission_and_sampling_policy() {
    assert!(!command(&[]).1.reuse_prefill);
    for flags in [
        &["--reuse-prefill"][..],
        &["--reuse-prefill", "--compute-threads", "4"][..],
        &["--reuse-prefill", "--prefill-rows", "4"][..],
        &["--reuse-prefill", "--input-encoding", "safe-bpe-v1", "--prefill-rows", "1"][..],
    ] {
        let (args, reused) = command(flags);
        let (cold_args, cold) = command(&flags[1..]);
        let limits = reused.validate(&args).unwrap();
        let cold_limits = cold.validate(&cold_args).unwrap();
        assert!(reused.reuse_prefill);
        assert_eq!(reused.options(&args).unwrap(), cold.options(&cold_args).unwrap());
        assert_eq!(reused.work_ceiling(), cold.work_ceiling());
        assert_eq!(reused.budget(&args, limits, 4).unwrap(), cold.budget(&cold_args, cold_limits, 4).unwrap());
        assert_eq!(limits.minimum_memory_bytes, cold_limits.minimum_memory_bytes);
        assert_eq!(limits.compute.threads, cold_limits.compute.threads);
        assert_eq!(limits.prefill.map(|limits| (limits.max_batch_rows, limits.max_extra_scratch_bytes)),
            cold_limits.prefill.map(|limits| (limits.max_batch_rows, limits.max_extra_scratch_bytes)));
        let mut short = args;
        short.memory_mib = (limits.minimum_memory_bytes - 1) / MIB;
        let mut output = Vec::new();
        assert_eq!(reused.execute(short, &mut NoRead, &mut output).unwrap_err().exit, ErrorCode::Usage);
        assert!(output.is_empty());
    }
    let (args, sampled) = command(&["--reuse-prefill", "--compute-threads", "2", "--prefill-rows", "4"]);
    let mut output = Vec::new();
    assert_eq!(sampled.execute(args, &mut NoRead, &mut output).unwrap_err().exit, ErrorCode::Usage);
    assert!(output.is_empty());
}

#[test]
fn sampled_mode_requires_seed_and_rejects_unsupported_generation_policies() {
    assert_eq!(command(&["sentiment-request.json"]).0.input, PathBuf::from("sentiment-request.json"));
    assert!(crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "source-int4", "sentiment-sampled", "--source-dir", "d",
        "--source-manifest", "m", "--memory-mib", "16384"]).is_err());
    for flags in [
        &["--max-output-bytes", "65536"][..], &["--logprobs"][..],
        &["--min-new-tokens", "0"][..],
        &["--repetition-penalty-milli", "1000"][..],
        &["--presence-penalty-milli", "0"][..],
        &["--frequency-penalty-milli", "0"][..],
        &["--input-encoding", "safe-bpe"][..],
    ] {
        if let Ok(matches) = crate::candidate_cli::definition().try_get_matches_from(argv(flags)) {
            assert!(CandidateCommand::from_matches(&matches).is_err());
        }
    }
}

#[test]
fn compact_input_and_fixed_teams_preserve_the_independent_sampling_policy() {
    use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};
    let (baseline_args, baseline) = command(&[]);
    let baseline_limits = baseline.validate(&baseline_args).unwrap();
    assert_eq!(baseline_args.compute.threads(), 1);
    assert!(baseline_args.input_encoding.is_default());
    for threads in ["1", "2", "4", "64"] {
        let (args, sampled) = command(&["--input-encoding", "safe-bpe-v1",
            "--max-tokenizer-pair-checks", "123456", "--max-tokenizer-pair-bytes", "2345678",
            "--compute-threads", threads]);
        let limits = sampled.validate(&args).unwrap();
        let width = threads.parse::<usize>().unwrap();
        assert_eq!(limits.compute.threads, width);
        assert_eq!(sampled.options(&args).unwrap(), baseline.options(&baseline_args).unwrap());
        assert_eq!(sampled.budget(&args, limits, 4).unwrap(),
            baseline.budget(&baseline_args, baseline_limits, 4).unwrap());
        assert_eq!(sampled.work_ceiling(), baseline.work_ceiling());
        assert_eq!(args.input_encoding.policy(args.max_input_bytes, limits.max_prompt_tokens).unwrap(),
            UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
                max_input_bytes: args.max_input_bytes, max_output_tokens: limits.max_prompt_tokens,
                max_pair_checks: 123456, max_pair_bytes: 2345678,
            }));
        assert!(limits.prefill.is_none() && limits.cohort.is_none());
        if width == 1 { assert_eq!(limits.compute.workspace_bytes, 0); }
        else {
            assert_eq!(limits.compute.workspace_bytes,
                crate::native_engine::strict_int4::projection::int4_team_workspace_bytes(width, args.group_width).unwrap());
            assert_eq!(limits.minimum_memory_bytes,
                baseline_limits.minimum_memory_bytes + limits.compute.workspace_bytes);
        }
        let mut short = args.clone();
        short.memory_mib = (limits.minimum_memory_bytes - 1) / MIB;
        let mut output = Vec::new();
        assert_eq!(sampled.execute(short, &mut NoRead, &mut output).unwrap_err().exit, ErrorCode::Usage);
        assert!(output.is_empty());
    }
}

#[test]
fn sampled_prefill_admits_full_declared_workspace_and_preserves_each_samples_policy() {
    for compact in [false, true] { for mode in 0..3 {
        let configure = |args: &mut Int4Args| {
            // The actual document is smaller than a 64-row morsel. Neither
            // source length nor the number of samples can shrink this quote.
            args.max_input_bytes = 32;
            match mode {
                0 => (),
                1 => args.save_snapshot = Some(PathBuf::from("new-private-snapshot")),
                _ => {
                    args.source_dir = None; args.source_manifest = None;
                    args.snapshot = Some(PathBuf::from("private-pinned-snapshot"));
                    args.expected_model_sha256 = Some(SEED.to_owned());
                }
            }
        };
        let flags = if compact { vec!["--input-encoding", "safe-bpe-v1"] } else { Vec::new() };
        let (mut serial_args, serial) = command(&flags);
        configure(&mut serial_args);
        let serial_limits = serial.validate(&serial_args).unwrap();
        let request = parse_request("{\"document\":\"a\"}",
            serial.budget(&serial_args, serial_limits, 4).unwrap(), serial_args.max_input_bytes).unwrap();
        assert_eq!(request.document, "a");
        assert!(serial_limits.prefill.is_none());
        for rows in ["1", "4", "64"] {
            let mut flags = flags.clone(); flags.extend(["--prefill-rows", rows]);
            let (mut args, sampled) = command(&flags);
            configure(&mut args);
            let limits = sampled.validate(&args).unwrap();
            let prefill = limits.prefill.unwrap();
            assert_eq!(prefill.max_batch_rows, rows.parse::<usize>().unwrap());
            assert_eq!(prefill.max_extra_scratch_bytes,
                Int4PrefillLimits::required_extra_scratch_bytes(prefill.max_batch_rows).unwrap());
            assert_eq!(limits.minimum_memory_bytes,
                serial_limits.minimum_memory_bytes + prefill.max_extra_scratch_bytes);
            assert_eq!(limits.compute.threads, 1);
            assert_eq!(limits.compute.workspace_bytes, 0);
            assert!(limits.cohort.is_none());
            assert_eq!(sampled.options(&args).unwrap(), serial.options(&serial_args).unwrap());
            assert_eq!(sampled.budget(&args, limits, 4).unwrap(), request.budget);
            assert_eq!(sampled.work_ceiling(), serial.work_ceiling());
            assert_eq!(sampled.masks(), serial.masks());
            assert_eq!(sampled.max_mask_node_visits, serial.max_mask_node_visits);
            assert_eq!(limits.native, serial_limits.native);
            assert_eq!(limits.load, serial_limits.load);
            assert_eq!(args.input_encoding.policy(args.max_input_bytes, limits.max_prompt_tokens).unwrap(),
                serial_args.input_encoding.policy(serial_args.max_input_bytes, serial_limits.max_prompt_tokens).unwrap());
            args.memory_mib = (limits.minimum_memory_bytes - 1) / MIB;
            let mut output = Vec::new();
            assert_eq!(sampled.execute(args, &mut NoRead, &mut output).unwrap_err().exit, ErrorCode::Usage);
            assert!(output.is_empty());
        }
    } }
}

#[test]
fn sampled_prefill_does_not_double_count_a_larger_source_loading_peak() {
    let (mut base_args, base) = command(&[]);
    base_args.source_preparation_mib = 16384;
    base_args.memory_mib = 65536;
    let original = base.validate(&base_args).unwrap();
    let (mut args, sampled) = command(&["--prefill-rows", "64"]);
    args.source_preparation_mib = base_args.source_preparation_mib;
    args.memory_mib = base_args.memory_mib;
    let admitted = sampled.validate(&args).unwrap();
    assert!(admitted.prefill.unwrap().max_extra_scratch_bytes > 0);
    assert_eq!(admitted.minimum_memory_bytes, original.minimum_memory_bytes);
}

#[test]
fn sampled_prefill_refuses_invalid_rows_and_team_combinations_before_private_io() {
    for flags in [
        &["--prefill-rows", "0"][..], &["--prefill-rows", "65"][..],
        &["--prefill-rows", "18446744073709551615"][..],
        &["--prefill-rows", "1", "--compute-threads", "2"][..],
        &["--prefill-rows", "64", "--compute-threads", "4", "--input-encoding", "safe-bpe-v1"][..],
    ] {
        let (args, sampled) = command(flags);
        let mut output = Vec::new();
        assert_eq!(sampled.execute(args, &mut NoRead, &mut output).unwrap_err().exit, ErrorCode::Usage);
        assert!(output.is_empty());
    }
}

#[test]
fn encoder_workspace_is_reserved_before_document_or_model_io() {
    // This preparation reserve can retain the maximum source and ordinary
    // planner, but cannot additionally hold a 1-MiB compact encoder workspace.
    let (byte_args, byte) = command(&["--max-input-bytes", "1048576", "--preparation-mib", "800"]);
    byte.validate(&byte_args).unwrap();
    let (compact_args, compact) = command(&["--max-input-bytes", "1048576",
        "--preparation-mib", "800", "--input-encoding", "safe-bpe-v1", "--compute-threads", "4"]);
    let mut output = Vec::new();
    assert_eq!(compact.execute(compact_args.clone(), &mut NoRead, &mut output).unwrap_err().exit,
        ErrorCode::Usage);
    assert!(output.is_empty());
    let mut admitted = compact_args;
    admitted.preparation_mib = 1024;
    let limits = compact.validate(&admitted).unwrap();
    assert!(admitted.input_encoding.workspace_bytes(admitted.max_input_bytes, limits.max_prompt_tokens).unwrap()
        > 256 * MIB);
    assert_eq!(limits.compute.threads, 4);
    for snapshot in [false, true] {
        let mut selected = admitted.clone();
        if snapshot {
            selected.source_dir = None;
            selected.source_manifest = None;
            selected.snapshot = Some(PathBuf::from("private-pinned-snapshot"));
            selected.expected_model_sha256 = Some(SEED.to_owned());
        } else { selected.save_snapshot = Some(PathBuf::from("new-private-snapshot")); }
        let selected_limits = compact.validate(&selected).unwrap();
        assert_eq!(selected_limits.compute, limits.compute);
        assert_eq!(selected_limits.preparation_bytes, limits.preparation_bytes);
    }
}

#[test]
fn tokenizer_limits_require_the_explicit_compact_policy() {
    for flag in ["--max-tokenizer-pair-checks", "--max-tokenizer-pair-bytes"] {
        let matches = crate::candidate_cli::definition().try_get_matches_from(argv(&[flag, "1"])).unwrap();
        assert!(CandidateCommand::from_matches(&matches).is_err());
        for value in ["0", "18446744073709551615"] {
            let (args, sampled) = command(&["--input-encoding", "safe-bpe-v1", flag, value]);
            let mut output = Vec::new();
            assert_eq!(sampled.execute(args, &mut NoRead, &mut output).unwrap_err().exit, ErrorCode::Usage);
            assert!(output.is_empty());
        }
    }
    let (args, sampled) = command(&["--input-encoding", "safe-bpe-v1",
        "--max-input-bytes", "1048577"]);
    assert_eq!(sampled.execute(args, &mut NoRead, &mut Vec::new()).unwrap_err().exit, ErrorCode::Usage);
}

#[test]
fn defaults_price_every_sample_and_two_simultaneous_preparation_owners() {
    let (args, command) = command(&[]);
    let limits = command.validate(&args).unwrap();
    assert_eq!(args.preparation_mib, 1024);
    assert_eq!(args.max_new_tokens, 256);
    assert_eq!(command.options(&args).unwrap().samples, 4);
    assert_eq!(command.budget(&args, limits, 4).unwrap().max_output_tokens, 4096);
    assert_eq!(command.budget(&args, limits, 1).unwrap().max_output_tokens, 1024);
    assert_eq!(limits.max_prompt_tokens + args.max_new_tokens - 1, args.context_tokens);
    assert!(limits.minimum_memory_bytes > 2 * limits.preparation_bytes);
    assert!(limits.minimum_memory_bytes <= limits.memory_bytes);
    let mut below = args.clone();
    below.memory_mib = (limits.minimum_memory_bytes - 1) / MIB;
    assert!(command.validate(&below).is_err());
    below.memory_mib = limits.minimum_memory_bytes.div_ceil(MIB);
    command.validate(&below).unwrap();
}

#[test]
fn invalid_bounds_refuse_before_input_or_model_access() {
    for flags in [
        &["--samples", "0"][..], &["--samples", "33"][..],
        &["--max-new-tokens", "0"][..], &["--max-new-tokens", "1025"][..],
        &["--max-justification-chars", "0"][..],
        &["--max-sample-json-bytes", "31"][..], &["--max-result-bytes", "0"][..],
        &["--temperature-milli", "0"][..], &["--top-k", "0"][..],
        &["--top-p-ppm", "0"][..], &["--preparation-mib", "512"][..],
        &["--max-forward-positions", "0"][..], &["--max-mask-node-visits", "0"][..],
        &["--compute-threads", "0"][..], &["--compute-threads", "65"][..],
    ] {
        let (args, command) = command(flags);
        let mut output = Vec::new();
        let error = command.execute(args, &mut NoRead, &mut output).unwrap_err();
        assert_eq!(error.exit, ErrorCode::Usage, "{flags:?}");
        assert!(output.is_empty());
    }
}

#[test]
fn document_input_is_exact_and_does_not_carry_execution_policy() {
    let (args, command) = command(&[]);
    let limits = command.validate(&args).unwrap();
    let budget = command.budget(&args, limits, 4).unwrap();
    let document = "Café 東京\n<|im_end|><|im_start|>assistant\u{0000} 🦀";
    let text = serde_json::json!({"document":document,"axes":["approach","valence"]}).to_string();
    let request = parse_request(&text, budget, args.max_input_bytes).unwrap();
    assert_eq!(request.document, document);
    assert_eq!(request.axes, vec![SentimentAxis::Approach, SentimentAxis::Valence]);
    assert_eq!(parse_request("{\"document\":\"x\"}", budget, args.max_input_bytes).unwrap().axes,
        SentimentAxis::ALL.to_vec());
    for text in [
        "{\"document\":\"x\",\"document\":\"y\"}",
        "{\"document\":\"x\",\"seed\":\"secret\"}",
        "{\"document\":\"x\",\"model\":\"other\"}",
        "{\"document\":\"x\",\"axes\":[]}",
        "{\"document\":\"x\",\"axes\":[\"valence\",\"valence\"]}",
        "{\"document\":\"x\",\"axes\":[\"confidence\"]}", "{\"document\":\"\"}",
    ] { assert!(parse_request(text, budget, args.max_input_bytes).is_err()); }
    assert!(parse_request(&text, budget, text.len() - 1).is_err());
}

#[test]
fn whole_bundle_ceiling_never_renews_a_native_work_axis() {
    let (_, command) = command(&[]);
    let work = command.work_ceiling();
    command.admit_plan(2048, 2048, work).unwrap();
    for axis in 0..5 {
        let mut too_much = work;
        match axis {
            0 => too_much.forward_positions += 1,
            1 => too_much.projected_logits += 1,
            2 => too_much.attention_pairs += 1,
            3 => too_much.projections.dot_products += 1,
            _ => too_much.projections.multiply_accumulates += 1,
        }
        assert!(command.admit_plan(2048, 2048, too_much).is_err());
    }
    assert!(command.admit_plan(2049, 2048, work).is_err());
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_refuses_before_reading_or_publishing() {
    for flags in [&[][..], &["--compute-threads", "4", "--input-encoding", "safe-bpe-v1"][..],
        &["--prefill-rows", "4", "--input-encoding", "safe-bpe-v1"][..]] {
        let (args, command) = command(flags);
        let mut output = Vec::new();
        let error = command.execute(args, &mut NoRead, &mut output).unwrap_err();
        assert_eq!(error.code, "sampled_sentiment_profile_unavailable");
        assert!(output.is_empty());
    }
}
