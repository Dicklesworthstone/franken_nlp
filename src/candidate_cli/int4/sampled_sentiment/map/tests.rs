//! Pure CLI admission/shape checks; no source, snapshot or model is opened.
use super::*;

pub(super) const SEED: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn argv<'a>(extra: &'a [&'a str]) -> Vec<&'a str> {
    let mut words = vec!["candidate", "source-int4", "sentiment-sampled-map",
        "--source-dir", "private-source", "--source-manifest", "private-manifest",
        "--memory-mib", "65536", "--seed", SEED];
    words.extend_from_slice(extra); words
}

pub(super) fn command(extra: &[&str]) -> (Int4Args, MapArgs) {
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv(extra)).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong candidate dispatcher"); };
    let Int4Task::SampledSentimentMap(options) = command.task else { panic!("wrong task dispatcher"); };
    (command.args, options)
}

pub(super) struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused map reached private input IO"); }
}
pub(super) struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused map published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused map flushed output"); }
}

#[test]
fn map_dispatch_requires_a_seed_and_preserves_explicit_sample_controls() {
    let (args, options) = command(&["complete-request.json", "--samples", "3",
        "--temperature-milli", "700", "--top-k", "17", "--top-p-ppm", "800000",
        "--max-new-tokens", "99", "--max-justification-chars", "80"]);
    assert_eq!(args.input, PathBuf::from("complete-request.json"));
    let limits = options.validate(&args).unwrap();
    let sampling = options.sampled.options(&args).unwrap();
    assert_eq!(sampling.samples, 3);
    assert_eq!(sampling.temperature_milli, 700);
    assert_eq!(sampling.top_k, Some(17));
    assert_eq!(sampling.top_p_ppm, 800000);
    assert_eq!(sampling.max_sample_tokens, 99);
    assert_eq!(sampling.max_justification_chars, 80);
    assert_eq!(limits.mapping.chunks.max_input_bytes, args.max_input_bytes);
    assert_eq!(limits.mapping.chunks.max_chunks, 64);
    assert_eq!(limits.memory.max_output_tokens, 64 * 4 * 3 * 99);
    assert_eq!(limits.memory.max_output_bytes, 16 * MIB);
    assert_eq!(limits.mapping.max_model_work, options.sampled.work_ceiling());
    assert!(crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "source-int4", "sentiment-sampled-map", "--source-dir", "d",
        "--source-manifest", "m", "--memory-mib", "65536"]).is_err());
    for flags in [
        &["--max-output-bytes", "65536"][..], &["--logprobs"][..],
        &["--min-new-tokens", "0"][..], &["--repetition-penalty-milli", "1000"][..],
        &["--input-encoding", "safe-bpe"][..], &["--cohort-width", "2"][..],
    ] {
        if let Ok(matches) = crate::candidate_cli::definition().try_get_matches_from(argv(flags)) {
            assert!(CandidateCommand::from_matches(&matches).is_err());
        }
    }
}

#[test]
fn complete_map_bounds_do_not_renew_the_native_or_mask_allowance_per_chunk() {
    let (args, two) = command(&["--max-chunks", "2", "--max-chunk-bytes", "128",
        "--max-tokenizer-calls", "41", "--max-map-result-bytes", "4194304",
        "--max-retained-result-bytes", "2097152"]);
    let two_limits = two.validate(&args).unwrap();
    let mut four = two; four.mapping.max_chunks = 4;
    let four_limits = four.validate(&args).unwrap();
    assert_eq!(four_limits.mapping.max_model_work, two_limits.mapping.max_model_work);
    assert_eq!(four.sampled.max_mask_node_visits, two.sampled.max_mask_node_visits);
    assert_eq!(four.sampled.budget(&args, four_limits.common, 4).unwrap(),
        two.sampled.budget(&args, two_limits.common, 4).unwrap());
    assert_eq!(four_limits.memory.max_output_tokens, two_limits.memory.max_output_tokens * 2);
    assert_eq!(four_limits.memory.native, two_limits.memory.native);
    assert!(four_limits.memory.input_bytes > two_limits.memory.input_bytes);
    assert_eq!(two_limits.mapping.chunks.max_chunk_bytes, 128);
    assert_eq!(two_limits.mapping.chunks.max_tokenizer_calls, 41);
    assert_eq!(two_limits.planning.max_result_bytes, MIB);
    assert_eq!(two_limits.mapping.max_retained_result_bytes, 2 * MIB);
    assert_eq!(two_limits.mapping.max_result_bytes, 4 * MIB);
    assert_eq!(two_limits.envelope_bytes as u64, two_limits.memory.max_output_bytes + 4096);
}

#[test]
fn compact_encoding_has_one_explicit_document_work_allowance_and_workspace() {
    let (byte_args, byte) = command(&[]);
    let byte_limits = byte.validate(&byte_args).unwrap();
    assert_eq!(byte.input_encoding(&byte_args).unwrap(), UntrustedEncoding::ByteFallback);
    let (args, compact) = command(&["--input-encoding", "safe-bpe-v1",
        "--max-tokenizer-input-bytes", "123456", "--max-tokenizer-output-tokens", "65432",
        "--max-tokenizer-pair-checks", "222222", "--max-tokenizer-pair-bytes", "3333333"]);
    let expected = SafeBpeLimits { max_input_bytes: 123456, max_output_tokens: 65432,
        max_pair_checks: 222222, max_pair_bytes: 3333333 };
    assert!(expected.max_output_tokens > args.context_tokens);
    assert_eq!(compact.input_encoding(&args).unwrap(), UntrustedEncoding::SafeBpeV1(expected));
    let limits = compact.validate(&args).unwrap();
    assert_eq!(limits.memory.input_bytes,
        byte_limits.memory.input_bytes + expected.requirements().unwrap().workspace_bytes);
    assert_eq!(limits.memory.native, byte_limits.memory.native);
    assert_eq!(limits.memory.max_output_tokens, byte_limits.memory.max_output_tokens);
    assert_eq!(compact.sampled.options(&args).unwrap(), byte.sampled.options(&byte_args).unwrap());
    assert_eq!(limits.mapping.max_model_work, byte_limits.mapping.max_model_work);
    let (default_args, default) = command(&["--input-encoding", "safe-bpe-v1"]);
    let UntrustedEncoding::SafeBpeV1(default_policy) = default.input_encoding(&default_args).unwrap()
        else { panic!("compact policy missing"); };
    assert_eq!(default_policy.max_input_bytes, SafeBpeLimits::MAX_INPUT_BYTES);
    assert_eq!(default_policy.max_output_tokens, SafeBpeLimits::MAX_OUTPUT_TOKENS);
    for flag in ["--max-tokenizer-input-bytes", "--max-tokenizer-output-tokens"] {
        let (args, refused) = command(&[flag, "1024"]);
        assert_eq!(refused.execute(args, &mut NoRead, &mut NoWrite).unwrap_err().exit, ErrorCode::Usage);
    }
}

#[test]
fn serial_prefill_and_team_schedules_preserve_sampling_and_charge_the_complete_peak() {
    for snapshot in [false, true] {
        let configure = |args: &mut Int4Args| {
            if snapshot {
                args.source_dir = None; args.source_manifest = None;
                args.snapshot = Some(PathBuf::from("private-pinned-snapshot"));
                args.expected_model_sha256 = Some(SEED.to_owned());
            }
        };
        let (mut baseline_args, baseline) = command(&[]);
        configure(&mut baseline_args);
        let baseline_limits = baseline.validate(&baseline_args).unwrap();
        for flags in [
            &["--reuse-prefill"][..], &["--prefill-rows", "1"][..],
            &["--reuse-prefill", "--prefill-rows", "4"][..], &["--prefill-rows", "64"][..],
            &["--reuse-prefill", "--compute-threads", "2"][..], &["--compute-threads", "64"][..],
        ] {
            let (mut args, options) = command(flags); configure(&mut args);
            let limits = options.validate(&args).unwrap();
            assert_eq!(limits.memory, baseline_limits.memory);
            assert_eq!(options.sampled.options(&args).unwrap(), baseline.sampled.options(&baseline_args).unwrap());
            assert_eq!(limits.mapping.max_model_work, baseline_limits.mapping.max_model_work);
            assert_eq!(options.sampled.max_mask_node_visits, baseline.sampled.max_mask_node_visits);
            let prefill = limits.common.prefill.map(|p| p.validate().unwrap()).unwrap_or(0);
            let loading = if snapshot { limits.common.snapshot_need.scratch_bytes }
                else { limits.common.source_preparation_bytes + limits.common.load.scratch_payload_bytes };
            let execution = limits.memory.input_bytes + limits.memory.native.kv_bytes
                + limits.memory.workspace_bytes + limits.memory.output_reservation_bytes
                + prefill + limits.common.compute.workspace_bytes;
            assert_eq!(limits.common.minimum_memory_bytes, limits.cli_preparation_bytes
                + limits.common.load.weight_payload_bytes + ALLOCATOR_BYTES + loading.max(execution) + CLEANUP_BYTES);
            let mut enough = args.clone();
            enough.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
            options.validate(&enough).unwrap();
            args.memory_mib = (limits.common.minimum_memory_bytes - 1) / MIB;
            assert_eq!(options.execute(args, &mut NoRead, &mut NoWrite).unwrap_err().exit, ErrorCode::Usage);
        }
    }
}

#[test]
fn loading_and_execution_are_distinct_peaks_and_snapshot_saving_is_refused() {
    let (mut args, base) = command(&[]);
    args.source_preparation_mib = 32768;
    let cold = base.validate(&args).unwrap();
    let (mut selected_args, selected) = command(&["--prefill-rows", "64"]);
    selected_args.source_preparation_mib = args.source_preparation_mib;
    let packed = selected.validate(&selected_args).unwrap();
    assert!(packed.common.prefill.unwrap().validate().unwrap() > 0);
    assert_eq!(packed.common.minimum_memory_bytes, cold.common.minimum_memory_bytes);
    for snapshot in [false, true] {
        let (mut args, options) = command(&[]);
        args.save_snapshot = Some(PathBuf::from("private-new-snapshot"));
        if snapshot {
            args.source_dir = None; args.source_manifest = None;
            args.snapshot = Some(PathBuf::from("private-existing-snapshot"));
            args.expected_model_sha256 = Some(SEED.to_owned());
        }
        assert_eq!(options.execute(args, &mut NoRead, &mut NoWrite).unwrap_err().exit, ErrorCode::Usage);
    }
}

#[test]
fn invalid_map_geometry_work_and_schedule_fail_before_private_io() {
    for flags in [
        &["--max-chunks", "0"][..], &["--max-chunks", "257"][..],
        &["--max-chunk-bytes", "3"][..], &["--max-chunk-bytes", "1048577"][..],
        &["--max-tokenizer-calls", "0"][..], &["--max-tokenizer-calls", "1000001"][..],
        &["--max-map-result-bytes", "0"][..], &["--max-map-result-bytes", "67108865"][..],
        &["--max-retained-result-bytes", "1048575"][..],
        &["--max-forward-positions", "0"][..], &["--max-mask-node-visits", "0"][..],
        &["--prefill-rows", "0"][..], &["--prefill-rows", "65"][..],
        &["--prefill-rows", "1", "--compute-threads", "2"][..],
        &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "0"][..],
        &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-output-tokens", "1048577"][..],
    ] {
        let (args, options) = command(flags);
        assert_eq!(options.execute(args, &mut NoRead, &mut NoWrite).unwrap_err().exit, ErrorCode::Usage,
            "flags: {flags:?}");
    }
}

#[test]
fn typed_request_preserves_all_original_utf8_and_rejects_partial_or_ambiguous_input() {
    let (args, options) = command(&[]);
    let limits = options.validate(&args).unwrap();
    let budget = options.sampled.budget(&args, limits.common, 4).unwrap();
    let document = "  é\r\n🙂\t\nlast  ";
    let encoded = serde_json::json!({"document":document}).to_string();
    let request = parse_request(&encoded, budget, encoded.len()).unwrap();
    assert_eq!(request.document.as_bytes(), document.as_bytes());
    assert_eq!(request.axes, SentimentAxis::ALL);
    assert_eq!(request.budget, budget);
    let explicit = parse_request(r#"{"document":" ","axes":["approach","valence"]}"#, budget, 4096).unwrap();
    assert_eq!(explicit.document, " ");
    assert_eq!(explicit.axes, [SentimentAxis::Approach, SentimentAxis::Valence]);
    for malformed in [
        r#"{"document":""}"#, r#"{"document":"a","document":"b"}"#,
        r#"{"document":"a","axes":[]}"#, r#"{"document":"a","axes":["valence","valence"]}"#,
        r#"{"document":"a","axes":["unknown"]}"#, r#"{"document":"a","extra":true}"#,
        "{\"document\":\"a\"}\n{\"document\":\"b\"}", r#"{"document":"unfinished""#,
    ] {
        assert!(matches!(parse_request(malformed, budget, 4096), Err(CandidateError::Input)));
    }
    assert!(matches!(parse_request(&encoded, budget, encoded.len() - 1), Err(CandidateError::Input)));
}
