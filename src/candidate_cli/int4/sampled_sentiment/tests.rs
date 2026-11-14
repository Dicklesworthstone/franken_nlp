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
fn sampled_mode_requires_seed_and_rejects_unsupported_generation_policies() {
    assert_eq!(command(&["sentiment-request.json"]).0.input, PathBuf::from("sentiment-request.json"));
    assert!(crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "source-int4", "sentiment-sampled", "--source-dir", "d",
        "--source-manifest", "m", "--memory-mib", "16384"]).is_err());
    for flags in [
        &["--max-output-bytes", "65536"][..], &["--logprobs"][..],
        &["--prefill-rows", "1"][..], &["--min-new-tokens", "0"][..],
        &["--repetition-penalty-milli", "1000"][..],
        &["--presence-penalty-milli", "0"][..],
        &["--frequency-penalty-milli", "0"][..],
        &["--compute-threads", "2"][..], &["--input-encoding", "safe-bpe"][..],
    ] {
        if let Ok(matches) = crate::candidate_cli::definition().try_get_matches_from(argv(flags)) {
            assert!(CandidateCommand::from_matches(&matches).is_err());
        }
    }
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
    let (args, command) = command(&[]);
    let mut output = Vec::new();
    let error = command.execute(args, &mut NoRead, &mut output).unwrap_err();
    assert_eq!(error.code, "sampled_sentiment_profile_unavailable");
    assert!(output.is_empty());
}
