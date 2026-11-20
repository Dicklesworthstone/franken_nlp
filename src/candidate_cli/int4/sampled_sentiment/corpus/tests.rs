use super::*;

const SEED: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn argv<'a>(extra: &'a [&'a str]) -> Vec<&'a str> {
    let mut values = vec!["candidate", "source-int4", "sentiment-sampled-batch",
        "--source-dir", "private-source", "--source-manifest", "private-manifest",
        "--memory-mib", "16384", "--seed", SEED];
    values.extend_from_slice(extra);
    values
}
fn command(extra: &[&str]) -> (Int4Args, BatchArgs) { command_for_route(extra, 0) }
fn command_for_route(extra: &[&str], route: usize) -> (Int4Args, BatchArgs) {
    let mut values = vec!["candidate", "source-int4", "sentiment-sampled-batch",
        "--memory-mib", "16384", "--seed", SEED];
    match route {
        0 => values.extend(["--source-dir", "private-source", "--source-manifest", "private-manifest"]),
        1 => values.extend(["--source-dir", "private-source", "--source-manifest", "private-manifest",
            "--save-snapshot", "new-private-snapshot"]),
        2 => values.extend(["--snapshot", "private-pinned-snapshot", "--expected-model-sha256", SEED]),
        _ => panic!("unknown private route fixture"),
    }
    values.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(values).unwrap();
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
fn resident_prefill_reuse_preserves_cold_work_memory_and_sampler_admission() {
    assert!(!command(&[]).1.sampled.reuse_prefill);
    for route in 0..3 {
        for schedule in [&[][..], &["--compute-threads", "4"][..], &["--prefill-rows", "1"][..],
            &["--prefill-rows", "64", "--input-encoding", "safe-bpe-v1"][..],
            &["--compute-threads", "4", "--input-encoding", "safe-bpe-v1"][..]] {
            let (cold_args, cold) = command_for_route(schedule, route);
            let before = cold.validate(&cold_args).unwrap();
            let mut flags = schedule.to_vec(); flags.push("--reuse-prefill");
            let (args, batch) = command_for_route(&flags, route);
            assert!(batch.sampled.reuse_prefill);
            let after = batch.validate(&args).unwrap();
            assert_eq!(after.common.minimum_memory_bytes, before.common.minimum_memory_bytes);
            assert_eq!(after.common.native, before.common.native);
            let schedule = |limits: Option<Int4PrefillLimits>| limits.map(|prefill|
                (prefill.max_batch_rows, prefill.max_extra_scratch_bytes));
            assert_eq!(schedule(after.common.prefill), schedule(before.common.prefill));
            assert_eq!(after.common.compute.threads, before.common.compute.threads);
            assert_eq!(after.common.compute.workspace_bytes, before.common.compute.workspace_bytes);
            assert_eq!(after.preparation_reserve_bytes, before.preparation_reserve_bytes);
            assert_eq!(after.native.max_item_work, before.native.max_item_work);
            assert_eq!(after.native.max_model_work, before.native.max_model_work);
            assert_eq!(after.native.max_sampler_bytes, before.native.max_sampler_bytes);
            assert_eq!(after.native.masks.per_mask, before.native.masks.per_mask);
            assert_eq!(after.native.masks.max_visits_per_item, before.native.masks.max_visits_per_item);
            assert_eq!(after.native.masks.max_visits_per_run, before.native.masks.max_visits_per_run);
            assert_eq!(after.transport.max_work, before.transport.max_work);
            assert_eq!(after.transport.max_output_bytes, before.transport.max_output_bytes);
            assert_eq!(after.output_bytes, before.output_bytes);
            assert_eq!(batch.sampled.options(&args).unwrap(), cold.sampled.options(&cold_args).unwrap());
            assert_eq!(batch.sampled.budget(&args, after.common, SentimentAxis::ALL.len()).unwrap(),
                cold.sampled.budget(&cold_args, before.common, SentimentAxis::ALL.len()).unwrap());
            assert_eq!(args.input_encoding.policy(args.max_input_bytes, after.common.max_prompt_tokens).unwrap(),
                cold_args.input_encoding.policy(cold_args.max_input_bytes, before.common.max_prompt_tokens).unwrap());
            // Savings are physical work receipts, never authority to under-admit
            // the resident engine, output or declared prompt/team workspace.
            let mut short = args;
            short.memory_mib = (after.common.minimum_memory_bytes - 1) / MIB;
            assert_eq!(batch.execute_owned(short, NoRead, NoWrite).unwrap_err().exit, ErrorCode::Usage);
        }
    }
    assert!(definition().render_long_help().to_string().contains("--reuse-prefill"));
}

#[test]
fn reuse_does_not_bypass_invalid_schedules_or_native_budgets_before_population_io() {
    for flags in [
        &["--reuse-prefill", "--prefill-rows", "4", "--compute-threads", "2"][..],
        &["--reuse-prefill", "--prefill-rows", "0"][..],
        &["--reuse-prefill", "--compute-threads", "65"][..],
        &["--reuse-prefill", "--samples", "0"][..],
        &["--reuse-prefill", "--max-forward-positions", "0"][..],
        &["--reuse-prefill", "--max-mask-node-visits", "0"][..],
    ] {
        let (args, batch) = command(flags);
        assert_eq!(batch.execute_owned(args, NoRead, NoWrite).unwrap_err().exit, ErrorCode::Usage, "{flags:?}");
    }
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
        &["--prefill-rows", "0"][..], &["--prefill-rows", "65"][..],
        &["--prefill-rows", "18446744073709551615"][..],
        &["--prefill-rows", "1", "--compute-threads", "2"][..],
        &["--prefill-rows", "64", "--compute-threads", "4", "--input-encoding", "safe-bpe-v1"][..],
    ] {
        let (args, batch) = command(flags);
        assert!(batch.validate(&args).is_err(), "{flags:?}");
        assert_eq!(batch.execute_owned(args, NoRead, NoWrite).unwrap_err().exit, ErrorCode::Usage, "{flags:?}");
    }
    for flags in [
        &["--logprobs"][..],
        &["--max-output-bytes", "65536"][..], &["--min-new-tokens", "0"][..],
        &["--repetition-penalty-milli", "1000"][..], &["--max-tokenizer-pair-checks", "1"][..],
    ] {
        if let Ok(matches) = crate::candidate_cli::definition().try_get_matches_from(argv(flags)) {
            assert!(CandidateCommand::from_matches(&matches).is_err(), "{flags:?}");
        }
    }
    // Transferring one-shot arguments still rejects an invalid declared
    // schedule before any corpus/model IO.
    let (_, batch) = command(&[]);
    let mut one_shot = argv(&["--prefill-rows", "65"]);
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
    for route in 0..3 {
        for flags in [&[][..], &["--compute-threads", "4", "--input-encoding", "safe-bpe-v1"][..],
            &["--prefill-rows", "4"][..], &["--prefill-rows", "64", "--input-encoding", "safe-bpe-v1"][..],
            &["--reuse-prefill"][..], &["--reuse-prefill", "--compute-threads", "4"][..],
            &["--reuse-prefill", "--prefill-rows", "4", "--input-encoding", "safe-bpe-v1"][..]] {
            let (args, batch) = command_for_route(flags, route);
            let error = batch.execute_owned(args, NoRead, NoWrite).unwrap_err();
            assert_eq!(error.code, "sampled_sentiment_batch_profile_unavailable");
            assert_eq!(error.exit, ErrorCode::Usage);
        }
    }
}

#[test]
fn prefill_boundaries_preserve_corpus_policy_and_price_full_workspace_for_each_model_route() {
    for compact in [false, true] {
        for route in 0..3 {
            let mut base_flags = vec!["--max-input-bytes", "32", "--max-requests", "2"];
            if compact { base_flags.extend(["--input-encoding", "safe-bpe-v1"]); }
            let (base_args, baseline) = command_for_route(&base_flags, route);
            let base = baseline.validate(&base_args).unwrap();
            assert!(base.common.prefill.is_none());
            assert_eq!(base.common.snapshot_pin.is_some(), route == 2);
            assert_eq!(base_args.save_snapshot.is_some(), route == 1);
            // Price the simultaneous execution owners independently. The
            // workspace grows this stage only; it cannot enlarge a higher
            // source-load or snapshot-save peak by the same amount.
            let budget = baseline.sampled.budget(&base_args, base.common, SentimentAxis::ALL.len()).unwrap();
            let output_charge = budget.max_output_bytes * 4 + u64::from(budget.max_output_tokens) * 8 + 4096;
            let execute_peak = base.common.native.kv_bytes + base.common.native.rope_bytes
                + base.common.native.scratch_payload_bound + base.native.max_sampler_bytes
                + ALLOCATOR_BYTES + output_charge;
            let load_peak = if route == 2 { base.common.snapshot_need.scratch_bytes }
                else { base.common.source_preparation_bytes + base.common.load.scratch_payload_bytes };
            let save_peak = if route == 1 { base.common.snapshot_need.scratch_bytes } else { 0 };
            let model_io_peak = load_peak.max(save_peak);
            for rows in [1, 4, 64] {
                let row_text = rows.to_string();
                let mut flags = base_flags.clone(); flags.extend(["--prefill-rows", row_text.as_str()]);
                let (args, batch) = command_for_route(&flags, route);
                let limits = batch.validate(&args).unwrap();
                let prefill = limits.common.prefill.expect("explicit corpus schedule retained");
                let workspace = Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
                assert_eq!(prefill.max_batch_rows, rows);
                assert_eq!(prefill.max_extra_scratch_bytes, workspace);
                let growth = (execute_peak + workspace).max(model_io_peak) - execute_peak.max(model_io_peak);
                assert_eq!(limits.common.minimum_memory_bytes, base.common.minimum_memory_bytes + growth);
                assert_eq!(limits.common.compute.threads, 1);
                assert_eq!(limits.common.compute.workspace_bytes, 0);
                assert!(limits.common.cohort.is_none());
                assert_eq!(limits.common.native, base.common.native);
                assert_eq!(limits.common.load, base.common.load);
                assert_eq!(limits.common.preparation_bytes, base.common.preparation_bytes);
                assert_eq!(limits.native.max_model_work, base.native.max_model_work);
                assert_eq!(limits.native.max_item_work, base.native.max_item_work);
                assert_eq!(limits.native.masks.per_mask, base.native.masks.per_mask);
                assert_eq!(limits.native.masks.max_visits_per_item, base.native.masks.max_visits_per_item);
                assert_eq!(limits.native.masks.max_visits_per_run, base.native.masks.max_visits_per_run);
                assert_eq!(limits.native.max_sampler_bytes, base.native.max_sampler_bytes);
                assert_eq!(limits.transport.max_work, base.transport.max_work);
                assert_eq!(limits.transport.max_requests, base.transport.max_requests);
                assert_eq!(limits.transport.max_output_bytes, base.transport.max_output_bytes);
                assert_eq!(batch.sampled.options(&args).unwrap(), baseline.sampled.options(&base_args).unwrap());
                assert_eq!(batch.sampled.budget(&args, limits.common, SentimentAxis::ALL.len()).unwrap(),
                    baseline.sampled.budget(&base_args, base.common, SentimentAxis::ALL.len()).unwrap());
                assert_eq!(args.input_encoding.policy(args.max_input_bytes, limits.common.max_prompt_tokens).unwrap(),
                    base_args.input_encoding.policy(base_args.max_input_bytes, base.common.max_prompt_tokens).unwrap());
                // A short source and only two records cannot reduce the full
                // declared prompt workspace or borrow any transport reservation.
                let mut short = args.clone();
                short.memory_mib = (limits.common.minimum_memory_bytes - 1) / MIB;
                assert_eq!(batch.execute_owned(short, NoRead, NoWrite).unwrap_err().exit, ErrorCode::Usage);
                let mut exact = args;
                exact.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
                batch.validate(&exact).unwrap();
            }
        }
    }
}

#[test]
fn corpus_prefill_remains_inside_the_execution_peak_when_source_loading_is_larger() {
    let (mut base_args, baseline) = command(&[]);
    base_args.source_preparation_mib = 16384;
    base_args.memory_mib = 65536;
    let base = baseline.validate(&base_args).unwrap();
    let (mut args, batch) = command(&["--prefill-rows", "64"]);
    args.source_preparation_mib = base_args.source_preparation_mib;
    args.memory_mib = base_args.memory_mib;
    let limits = batch.validate(&args).unwrap();
    assert!(limits.common.prefill.unwrap().max_extra_scratch_bytes > 0);
    assert_eq!(limits.common.minimum_memory_bytes, base.common.minimum_memory_bytes);
    assert_eq!(limits.native.max_model_work, base.native.max_model_work);
    assert_eq!(limits.native.masks.per_mask, base.native.masks.per_mask);
    assert_eq!(limits.native.masks.max_visits_per_item, base.native.masks.max_visits_per_item);
    assert_eq!(limits.native.masks.max_visits_per_run, base.native.masks.max_visits_per_run);
}
