use super::*;

const SEED: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn argv<'a>(extra: &'a [&'a str]) -> Vec<&'a str> {
    let mut values = vec!["candidate", "source-int4", "sentiment-sampled-batch",
        "--source-dir", "private-source", "--source-manifest", "private-manifest",
        "--memory-mib", "16384", "--seed", SEED];
    values.extend_from_slice(extra);
    values
}
fn command(extra: &[&str]) -> (Int4Args, BatchArgs) {
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv(extra)).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong candidate dispatcher"); };
    let Int4Task::SampledSentimentBatch(batch) = command.task else { panic!("wrong resident task"); };
    (command.args, batch)
}

struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused stream reached input IO"); }
}
struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused stream published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused stream flushed output"); }
}

#[test]
fn resident_command_keeps_the_explicit_sampler_and_encoding_controls() {
    let (args, batch) = command(&["population.ndjson", "--samples", "7", "--compute-threads", "4",
        "--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "123456",
        "--max-tokenizer-pair-bytes", "2345678"]);
    assert_eq!(args.input, PathBuf::from("population.ndjson"));
    assert_eq!(batch.sampled.options(&args).unwrap().samples, 7);
    let limits = batch.validate(&args).unwrap();
    assert_eq!(limits.common.compute.threads, 4);
    assert!(limits.common.prefill.is_none() && limits.common.cohort.is_none());
    let crate::tokenizer::untrusted::UntrustedEncoding::SafeBpeV1(encoding) =
        args.input_encoding.policy(args.max_input_bytes, limits.common.max_prompt_tokens).unwrap()
        else { panic!("compact policy lost"); };
    assert_eq!(encoding.max_pair_checks, 123456);
    assert_eq!(encoding.max_pair_bytes, 2345678);
    assert_eq!(command(&[]).0.input, PathBuf::from("-"));
    assert!(crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "source-int4", "sentiment-sampled-batch", "--source-dir", "d",
        "--source-manifest", "m", "--memory-mib", "16384"]).is_err());
}

#[test]
fn more_transport_records_never_grant_more_native_mask_or_sampler_work() {
    let (args, small) = command(&["--max-requests", "2"]);
    let a = small.validate(&args).unwrap();
    let (args, large) = command(&["--max-requests", "200"]);
    let b = large.validate(&args).unwrap();
    assert_eq!(a.native.max_item_work, b.native.max_item_work);
    assert_eq!(a.native.max_model_work, b.native.max_model_work);
    assert_eq!(a.native.max_item_work, a.native.max_model_work);
    assert_eq!(a.native.masks.max_visits_per_item, a.native.masks.max_visits_per_run);
    assert_eq!(a.native.masks.max_visits_per_run, b.native.masks.max_visits_per_run);
    assert_eq!(a.native.max_sampler_bytes, b.native.max_sampler_bytes);
    assert_eq!(a.transport.max_work, b.transport.max_work);
    assert_eq!(a.common.minimum_memory_bytes, b.common.minimum_memory_bytes);
    assert_eq!(a.common.native, b.common.native);
    assert_eq!(a.transport.max_output_bytes - b.transport.max_output_bytes, 198 * FRAME_ALLOWANCE);
}

#[test]
fn complete_framing_and_live_corpus_staging_are_admitted_before_io() {
    let (args, batch) = command(&[]);
    let one = batch.sampled.validate(&args).unwrap();
    let limits = batch.validate(&args).unwrap();
    let framing = (batch.stream.max_requests + 4) * FRAME_ALLOWANCE;
    assert_eq!(limits.transport.max_output_bytes + framing, limits.output_bytes);
    assert_eq!(limits.transport.max_output_line_bytes, limits.common.result_bytes + 4096);
    assert_eq!(limits.preparation_reserve_bytes, one.preparation_bytes);
    let staging = 8 * limits.transport.max_line_bytes as u64
        + 4 * limits.transport.max_output_line_bytes as u64
        + 128 * limits.transport.max_epoch_ids as u64
        + limits.transport.max_epoch_id_bytes as u64 + limits.io_bytes;
    assert_eq!(limits.common.minimum_memory_bytes, one.minimum_memory_bytes + staging);
    let mut short = args.clone();
    short.memory_mib = (limits.common.minimum_memory_bytes - 1) / MIB;
    assert_eq!(batch.execute_owned(short, NoRead, NoWrite).unwrap_err().exit, ErrorCode::Usage);
    let mut exact = args;
    exact.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
    batch.validate(&exact).unwrap();
}

#[test]
fn invalid_transport_and_policy_refuse_without_private_io() {
    for flags in [
        &["--max-requests", "0"][..], &["--max-requests", "100001"][..],
        &["--max-total-input-bytes", "0"][..], &["--max-total-output-bytes", "4096"][..],
        &["--max-line-bytes", "1"][..], &["--max-line-bytes", "4194305"][..],
        &["--max-total-input-bytes", "18446744073709551615"][..],
        &["--max-total-output-bytes", "18446744073709551615"][..],
        &["--compute-threads", "65"][..], &["--samples", "0"][..],
        &["--max-mask-node-visits", "0"][..],
    ] {
        let (args, batch) = command(flags);
        assert_eq!(batch.execute_owned(args, NoRead, NoWrite).unwrap_err().exit, ErrorCode::Usage, "{flags:?}");
    }
    for flags in [
        &["--prefill-rows", "1"][..], &["--logprobs"][..],
        &["--max-output-bytes", "65536"][..], &["--min-new-tokens", "0"][..],
        &["--repetition-penalty-milli", "1000"][..], &["--max-tokenizer-pair-checks", "1"][..],
    ] {
        if let Ok(matches) = crate::candidate_cli::definition().try_get_matches_from(argv(flags)) {
            assert!(CandidateCommand::from_matches(&matches).is_err(), "{flags:?}");
        }
    }
    // Programmatic construction gets the same pre-IO rejection as the parser.
    let (_, batch) = command(&[]);
    let mut one_shot = argv(&["--prefill-rows", "1"]);
    one_shot[2] = "sentiment-sampled";
    let matches = crate::candidate_cli::definition().try_get_matches_from(one_shot).unwrap();
    let CandidateCommand::SourceInt4(one_shot) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong candidate dispatcher"); };
    assert_eq!(batch.execute_owned(one_shot.args, NoRead, NoWrite).unwrap_err().exit, ErrorCode::Usage);
}

#[test]
fn a_drained_stream_with_rejected_items_has_nonzero_process_status() {
    completed(BatchSummary::default()).unwrap();
    let summary = BatchSummary { succeeded: 2, failed: 1, ..BatchSummary::default() };
    let error = completed(summary).unwrap_err();
    assert_eq!(error.code, "sampled_sentiment_batch_items_failed");
    assert_ne!(error.exit, ErrorCode::Ok);
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn an_unavailable_runtime_never_reads_or_publishes_the_population() {
    for flags in [&[][..], &["--compute-threads", "4", "--input-encoding", "safe-bpe-v1"][..]] {
        let (args, batch) = command(flags);
        let error = batch.execute_owned(args, NoRead, NoWrite).unwrap_err();
        assert_eq!(error.code, "sampled_sentiment_batch_profile_unavailable");
        assert_eq!(error.exit, ErrorCode::Usage);
    }
}
