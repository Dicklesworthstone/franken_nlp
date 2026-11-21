//! Parser and exact pre-IO memory/work admission for complete document streams.
use super::*;
use super::super::tests::{NoRead, NoWrite, SEED};

pub(super) fn command(extra: &[&str], snapshot: bool) -> (Int4Args, CorpusArgs) {
    let mut words = vec!["candidate", "source-int4", "sentiment-sampled-corpus",
        "--memory-mib", "65536", "--seed", SEED];
    if snapshot { words.extend(["--snapshot", "private-snapshot", "--expected-model-sha256", SEED]); }
    else { words.extend(["--source-dir", "private-source", "--source-manifest", "private-manifest"]); }
    words.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(words).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong candidate dispatcher"); };
    let Int4Task::SampledSentimentCorpus(options) = command.task else { panic!("wrong corpus task"); };
    (command.args, options)
}

#[test]
fn parser_preserves_sampler_complete_document_encoding_and_transport_limits() {
    let (args, options) = command(&["population.ndjson", "--samples", "3", "--reuse-prefill",
        "--compute-threads", "4", "--max-chunks", "7", "--max-chunk-bytes", "128",
        "--max-tokenizer-calls", "41", "--max-requests", "17",
        "--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "123456",
        "--max-tokenizer-output-tokens", "65432", "--max-tokenizer-pair-checks", "222222",
        "--max-tokenizer-pair-bytes", "3333333"], false);
    let limits = options.validate(&args).unwrap();
    assert_eq!(args.input, PathBuf::from("population.ndjson"));
    assert_eq!(options.document.sampled.options(&args).unwrap().samples, 3);
    assert!(options.document.sampled.reuse_prefill);
    assert_eq!(limits.common.compute.threads, 4);
    assert_eq!(limits.document.mapping.chunks.max_chunks, 7);
    assert_eq!(limits.document.mapping.chunks.max_chunk_bytes, 128);
    assert_eq!(limits.document.mapping.chunks.max_tokenizer_calls, 41);
    assert_eq!(limits.transport.max_requests, 17);
    assert_eq!(limits.transport.max_document_bytes, args.max_input_bytes);
    assert_eq!(options.document.input_encoding(&args).unwrap(), UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 123456, max_output_tokens: 65432, max_pair_checks: 222222, max_pair_bytes: 3333333 }));
    let (default_args, defaults) = command(&[], true);
    assert_eq!(default_args.input, PathBuf::from("-"));
    assert_eq!(defaults.document.input_encoding(&default_args).unwrap(), UntrustedEncoding::ByteFallback);
    assert!(defaults.validate(&default_args).unwrap().common.snapshot_pin.is_some());
    assert!(crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "source-int4", "sentiment-sampled-corpus", "--source-dir", "d",
        "--source-manifest", "m", "--memory-mib", "65536"]).is_err());
}

#[test]
fn larger_populations_never_multiply_native_masks_or_resident_workspaces() {
    let (args, mut options) = command(&["--max-requests", "2"], false);
    let before = options.validate(&args).unwrap();
    options.stream.max_requests = 200;
    let after = options.validate(&args).unwrap();
    assert_eq!(before.document.mapping.max_model_work, after.document.mapping.max_model_work);
    assert_eq!(before.transport.max_work, after.transport.max_work);
    assert_eq!(before.memory, after.memory);
    assert_eq!(before.common.minimum_memory_bytes, after.common.minimum_memory_bytes);
    assert_eq!(before.transport.max_output_bytes - after.transport.max_output_bytes, 198 * FRAME_ALLOWANCE);
    assert_eq!(after.document.mapping.max_model_work, options.document.sampled.work_ceiling());
    let mut more_chunks = options; more_chunks.document.mapping.max_chunks *= 2;
    let chunked = more_chunks.validate(&args).unwrap();
    assert_eq!(chunked.document.mapping.max_model_work, after.document.mapping.max_model_work);
    assert_eq!(more_chunks.document.sampled.max_mask_node_visits, options.document.sampled.max_mask_node_visits);
    assert_eq!(chunked.transport.max_work, after.transport.max_work);
    assert_eq!(chunked.memory.native, after.memory.native);
    assert!(chunked.memory.input_bytes > after.memory.input_bytes);
}

#[test]
fn document_work_defaults_preserve_the_existing_whole_stream_policy() {
    let (args, options) = command(&[], false);
    assert_eq!(options.work, DocumentWorkArgs::default());
    assert_eq!(options.work.apply(options.document.sampled).unwrap(), options.document.sampled);
    let limits = options.validate(&args).unwrap();
    assert_eq!(limits.native.max_item_work, limits.native.max_model_work);
    assert_eq!(limits.native.max_model_work, options.document.sampled.work_ceiling());
    assert_eq!(limits.native.masks.max_visits_per_item, options.document.sampled.max_mask_node_visits);
    assert_eq!(limits.native.masks.max_visits_per_run, options.document.sampled.max_mask_node_visits);
}

#[test]
fn each_document_cap_reaches_planning_without_replacing_the_corpus_ledger() {
    let (args, options) = command(&[
        "--max-document-forward-positions", "32768",
        "--max-document-projected-logits", "1000000000",
        "--max-document-attention-pairs", "100000000000",
        "--max-document-dot-products", "100000000000000",
        "--max-document-multiply-accumulates", "1000000000000000",
        "--max-document-mask-node-visits", "32000000000",
    ], false);
    let limits = options.validate(&args).unwrap();
    let item = Int4Work { forward_positions: 32768, projected_logits: 1_000_000_000,
        attention_pairs: 100_000_000_000,
        projections: ProjectionWork { dot_products: 100_000_000_000_000,
            multiply_accumulates: 1_000_000_000_000_000 } };
    assert_eq!(limits.native.max_item_work, item);
    assert_eq!(limits.document.mapping.max_model_work, item);
    assert_eq!(limits.document.planning.max_total_prompt_tokens, 32768);
    assert_eq!(limits.native.masks.max_visits_per_item, 32_000_000_000);
    let run = options.document.sampled.work_ceiling();
    assert_eq!(limits.native.max_model_work, run);
    assert_eq!(limits.native.masks.max_visits_per_run, options.document.sampled.max_mask_node_visits);
    assert_eq!(limits.transport.max_work,
        BatchWork { forward_positions: run.forward_positions, projected_logits: run.projected_logits });
    let (_, mut defaults) = command(&[], false);
    let original = defaults.validate(&args).unwrap();
    assert_eq!(limits.transport.max_work, original.transport.max_work);
    assert!(limits.memory.input_bytes < original.memory.input_bytes);
    assert_eq!(limits.memory.native, original.memory.native);
    defaults.work = options.work;
    assert_eq!(defaults.validate(&args).unwrap().memory, limits.memory);
}

#[test]
fn larger_corpus_totals_do_not_expand_an_explicit_document_geometry() {
    let (args, mut options) = command(&["--input-encoding", "safe-bpe-v1"], false);
    let sampled = options.document.sampled;
    options.work = DocumentWorkArgs {
        max_document_forward_positions: Some(sampled.max_forward_positions),
        max_document_projected_logits: Some(sampled.max_projected_logits),
        max_document_attention_pairs: Some(sampled.max_attention_pairs),
        max_document_dot_products: Some(sampled.max_dot_products),
        max_document_multiply_accumulates: Some(sampled.max_multiply_accumulates),
        max_document_mask_node_visits: Some(sampled.max_mask_node_visits),
    };
    let before = options.validate(&args).unwrap();
    options.document.sampled.max_forward_positions *= 64;
    options.document.sampled.max_projected_logits *= 2;
    options.document.sampled.max_attention_pairs *= 2;
    options.document.sampled.max_dot_products *= 2;
    options.document.sampled.max_multiply_accumulates *= 2;
    options.document.sampled.max_mask_node_visits *= 64;
    options.stream.max_requests = 2000;
    let after = options.validate(&args).unwrap();
    assert_eq!(after.native.max_item_work, before.native.max_item_work);
    assert_eq!(after.native.masks.max_visits_per_item, before.native.masks.max_visits_per_item);
    assert_eq!(after.native.max_model_work, options.document.sampled.work_ceiling());
    assert_eq!(after.native.masks.max_visits_per_run, 64 * before.native.masks.max_visits_per_run);
    assert_eq!(after.transport.max_work.forward_positions, 64 * before.transport.max_work.forward_positions);
    assert_eq!(after.document.planning.max_total_prompt_tokens, before.document.planning.max_total_prompt_tokens);
    assert_eq!(after.memory, before.memory);
    assert_eq!(after.common.minimum_memory_bytes, before.common.minimum_memory_bytes);
    assert_eq!(options.document.sampled.options(&args).unwrap(), sampled.options(&args).unwrap());
    assert_eq!(options.document.sampled.budget(&args, after.common, 4).unwrap(),
        sampled.budget(&args, before.common, 4).unwrap());
}

#[test]
fn zero_expanding_and_unbounded_document_overrides_refuse_before_private_io() {
    let (_, defaults) = command(&[], false);
    let sampled = defaults.document.sampled;
    for (flag, total) in [
        ("--max-document-forward-positions", sampled.max_forward_positions),
        ("--max-document-projected-logits", sampled.max_projected_logits),
        ("--max-document-attention-pairs", sampled.max_attention_pairs),
        ("--max-document-dot-products", sampled.max_dot_products),
        ("--max-document-multiply-accumulates", sampled.max_multiply_accumulates),
        ("--max-document-mask-node-visits", sampled.max_mask_node_visits),
    ] {
        for value in [0, total + 1, u64::MAX] {
            let text = value.to_string();
            let (args, options) = command(&[flag, &text], false);
            assert_eq!(options.execute_owned(args, NoRead, NoWrite).unwrap_err().exit, ErrorCode::Usage);
        }
    }
    for flags in [
        &["--max-forward-positions", "18446744073709551615"][..],
        &["--max-mask-node-visits", "18446744073709551615"][..],
        &["--max-forward-positions", "33554432", "--max-document-forward-positions", "16777217"][..],
        &["--max-mask-node-visits", "2000000000000", "--max-document-mask-node-visits", "1000000000001"][..],
        &["--max-forward-positions", "0", "--max-document-forward-positions", "1"][..],
        &["--max-mask-node-visits", "0", "--max-document-mask-node-visits", "1"][..],
    ] {
        let (args, options) = command(flags, false);
        assert_eq!(options.execute_owned(args, NoRead, NoWrite).unwrap_err().exit, ErrorCode::Usage);
    }
}

#[test]
fn full_transport_native_and_scheduled_owners_share_one_pre_io_peak() {
    for snapshot in [false, true] {
        for flags in [
            &[][..], &["--reuse-prefill"][..], &["--compute-threads", "2"][..],
            &["--compute-threads", "64", "--reuse-prefill"][..],
            &["--prefill-rows", "1"][..], &["--prefill-rows", "64", "--reuse-prefill"][..],
            &["--input-encoding", "safe-bpe-v1", "--compute-threads", "4"][..],
            &["--input-encoding", "safe-bpe-v1", "--prefill-rows", "4"][..],
        ] {
            let (mut args, options) = command(flags, snapshot);
            let limits = options.validate(&args).unwrap();
            let one = options.document.validate(&args).unwrap();
            let prefill = limits.common.prefill.map(|p| p.validate().unwrap()).unwrap_or(0);
            let transport = 8 * limits.transport.max_line_bytes as u64
                + 4 * limits.transport.max_output_line_bytes as u64
                + 128 * limits.transport.max_epoch_ids as u64
                + limits.transport.max_epoch_id_bytes as u64 + limits.io_bytes;
            assert_eq!(limits.memory.input_bytes, one.memory.input_bytes + transport);
            assert_eq!(limits.memory.workspace_bytes, one.memory.workspace_bytes
                + prefill + limits.common.compute.workspace_bytes);
            assert_eq!(limits.memory.output_reservation_bytes, one.memory.output_reservation_bytes);
            assert_eq!(limits.memory.native, one.memory.native);
            assert_eq!(limits.output_bytes, limits.transport.max_output_bytes
                + (options.stream.max_requests + 4) * FRAME_ALLOWANCE);
            let loading = if snapshot { limits.common.snapshot_need.scratch_bytes }
                else { limits.common.source_preparation_bytes + limits.common.load.scratch_payload_bytes };
            let execution = limits.memory.input_bytes + limits.memory.native.kv_bytes
                + limits.memory.workspace_bytes + limits.memory.output_reservation_bytes;
            assert_eq!(limits.common.minimum_memory_bytes, limits.cli_preparation_bytes
                + limits.common.load.weight_payload_bytes + ALLOCATOR_BYTES + loading.max(execution) + CLEANUP_BYTES);
            args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
            options.validate(&args).unwrap();
            args.memory_mib = (limits.common.minimum_memory_bytes - 1) / MIB;
            assert_eq!(options.execute_owned(args, NoRead, NoWrite).unwrap_err().exit, ErrorCode::Usage);
        }
    }
}

#[test]
fn prefix_reuse_never_reduces_declared_cold_resources_or_changes_the_sampler() {
    for snapshot in [false, true] {
        for flags in [&[][..], &["--prefill-rows", "4"][..],
            &["--input-encoding", "safe-bpe-v1", "--compute-threads", "4"][..]] {
            let (args, mut options) = command(flags, snapshot);
            let cold = options.validate(&args).unwrap();
            let sampling = options.document.sampled.options(&args).unwrap();
            options.document.sampled.reuse_prefill = true;
            let reused = options.validate(&args).unwrap();
            assert_eq!(cold.memory, reused.memory);
            assert_eq!(cold.common.minimum_memory_bytes, reused.common.minimum_memory_bytes);
            assert_eq!(cold.document.mapping.max_model_work, reused.document.mapping.max_model_work);
            assert_eq!(cold.transport.max_work, reused.transport.max_work);
            assert_eq!(sampling, options.document.sampled.options(&args).unwrap());
        }
    }
}

#[test]
fn invalid_bounds_and_unsupported_compositions_refuse_before_private_io() {
    for flags in [
        &["--max-requests", "0"][..], &["--max-requests", "100001"][..],
        &["--max-total-input-bytes", "0"][..], &["--max-total-output-bytes", "4096"][..],
        &["--max-line-bytes", "1"][..], &["--max-line-bytes", "4194305"][..],
        &["--max-total-input-bytes", "18446744073709551615"][..],
        &["--max-total-output-bytes", "18446744073709551615"][..],
        &["--max-chunks", "0"][..], &["--max-chunks", "257"][..],
        &["--max-chunk-bytes", "3"][..], &["--max-map-result-bytes", "0"][..],
        &["--max-mask-node-visits", "0"][..], &["--max-forward-positions", "0"][..],
        &["--max-projected-logits", "0"][..], &["--max-attention-pairs", "0"][..],
        &["--max-dot-products", "0"][..], &["--max-multiply-accumulates", "0"][..],
        &["--compute-threads", "0"][..], &["--compute-threads", "65"][..],
        &["--prefill-rows", "0"][..], &["--prefill-rows", "65"][..],
        &["--prefill-rows", "4", "--compute-threads", "2"][..],
        &["--reuse-prefill", "--prefill-rows", "1", "--compute-threads", "2"][..],
        &["--max-tokenizer-input-bytes", "1024"][..],
        &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-output-tokens", "0"][..],
        &["--save-snapshot", "new-private-snapshot"][..],
    ] {
        let (args, options) = command(flags, false);
        assert_eq!(options.execute_owned(args, NoRead, NoWrite).unwrap_err().exit, ErrorCode::Usage, "{flags:?}");
    }
    for flags in [&["--logprobs"][..], &["--max-output-bytes", "65536"][..],
        &["--min-new-tokens", "0"][..], &["--cohort-heads", "2"][..],
        &["--cohort-chunks", "2"][..], &["--defaults", "private-defaults"][..]] {
        let mut words = vec!["candidate", "source-int4", "sentiment-sampled-corpus",
            "--source-dir", "d", "--source-manifest", "m", "--memory-mib", "65536", "--seed", SEED];
        words.extend_from_slice(flags);
        if let Ok(matches) = crate::candidate_cli::definition().try_get_matches_from(words) {
            assert!(CandidateCommand::from_matches(&matches).is_err(), "{flags:?}");
        }
    }
}

#[test]
fn a_terminal_summary_with_any_failed_document_has_nonzero_process_status() {
    completed(BatchSummary::default()).unwrap();
    let failure = completed(BatchSummary { succeeded: 3, failed: 1, ..BatchSummary::default() }).unwrap_err();
    assert_eq!(failure.code, "sampled_sentiment_corpus_items_failed");
    assert_ne!(failure.exit, ErrorCode::Ok);
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_never_reads_a_population_or_publishes_a_frame() {
    for snapshot in [false, true] {
        for flags in [&[][..], &["--reuse-prefill", "--prefill-rows", "4"][..],
            &["--input-encoding", "safe-bpe-v1", "--compute-threads", "4"][..]] {
            let (args, options) = command(flags, snapshot);
            let failure = options.execute_owned(args, NoRead, NoWrite).unwrap_err();
            assert_eq!(failure.code, "sampled_sentiment_corpus_profile_unavailable");
            assert_eq!(failure.exit, ErrorCode::Usage);
        }
    }
}
