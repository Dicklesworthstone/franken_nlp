//! Admission regressions that never open a model or private input.
use super::*;
use std::io;

const SEED: &str = "4545454545454545454545454545454545454545454545454545454545454545";

fn parsed(extra: &[&str]) -> Result<Command, clap::Error> {
    let mut argv = vec!["sentiment-sampled-corpus", "--model", "private-model.fnlpq",
        "--memory-mib", "65536", "--seed", SEED];
    argv.extend_from_slice(extra);
    Command::from_matches(&definition().try_get_matches_from(argv)?)
}
fn command(extra: &[&str]) -> Command { parsed(extra).unwrap() }

struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused population read input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused population wrote output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused population flushed output"); }
}

#[test]
fn root_dispatch_selects_complete_document_population_and_requires_explicit_identity() {
    let matches = crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "sentiment-sampled-corpus", "--model", "m", "--memory-mib", "65536", "--seed", SEED,
    ]).unwrap();
    assert!(matches!(CandidateCommand::from_matches(&matches).unwrap(),
        CandidateCommand::SampledSentimentCorpus(_)));
    let base = ["sentiment-sampled-corpus", "--model", "m", "--memory-mib", "65536", "--seed", SEED];
    for missing in [1, 3, 5] {
        let argv = base.iter().enumerate().filter_map(|(index, value)|
            (index != missing && index != missing + 1).then_some(*value)).collect::<Vec<_>>();
        assert!(definition().try_get_matches_from(argv).is_err());
    }
    for extra in [vec!["--reuse-prefill"], vec!["--bins", "5"],
        vec!["--store-results"], vec!["--logprobs"]] {
        assert!(parsed(&extra).is_err());
    }
}

#[test]
fn document_population_selected_rows_do_not_renew_or_replace_declared_run_work() {
    let (dense, dense_limits, dense_run) = command(&[]).admit().unwrap();
    let (selected, selected_limits, selected_run) = command(&["--max-selected-rows", "4096"]).admit().unwrap();
    assert_eq!(selected.sampled.selected_rows().unwrap(), Some(Int8JsonSparseLimits { max_rows_per_step: 4096 }));
    assert_eq!(dense.sampled.options().unwrap(), selected.sampled.options().unwrap());
    assert_eq!(dense_limits.memory, selected_limits.memory);
    assert_eq!(dense_limits.minimum_memory_bytes, selected_limits.minimum_memory_bytes);
    assert_eq!(dense_run.max_model_work, selected_run.max_model_work);
    assert_eq!(dense_run.max_mask_node_visits, selected_run.max_mask_node_visits);
    assert_eq!(dense_run.transport.max_work, selected_run.transport.max_work);
    assert_eq!(dense_run.transport_bytes, selected_run.transport_bytes);
    for cap in ["0", "166145"] { assert!(command(&["--max-selected-rows", cap]).admit().is_err()); }
}

#[test]
fn complete_document_limits_tighten_without_replacing_population_ceiling_or_sample_policy() {
    let (baseline, baseline_limits, population) = command(&[]).admit().unwrap();
    let (document, limits, envelope) = command(&[
        "--max-document-forward-positions", "32768",
        "--max-document-projected-logits", "123456",
        "--max-document-attention-pairs", "234567",
        "--max-document-dot-products", "345678",
        "--max-document-multiply-accumulates", "456789",
        "--max-document-mask-node-visits", "567890",
    ]).admit().unwrap();
    assert_eq!(document.sampled.options().unwrap(), baseline.sampled.options().unwrap());
    assert_eq!(limits.task, baseline_limits.task);
    assert_eq!(limits.mapping.chunks, baseline_limits.mapping.chunks);
    assert_eq!(envelope.max_model_work, population.max_model_work);
    assert_eq!(envelope.max_mask_node_visits, population.max_mask_node_visits);
    assert_eq!(envelope.transport.max_work, population.transport.max_work);
    let work = limits.mapping.max_model_work;
    assert_eq!((work.forward_positions, work.projected_logits, work.attention_pairs), (32768, 123456, 234567));
    assert_eq!((work.projections.dot_products, work.projections.multiply_accumulates), (345678, 456789));
    assert_eq!(document.sampled.max_mask_node_visits, 567890);
    assert_eq!(document.sampled.task_limits().max_total_prompt_tokens, 32768);
}

#[test]
fn every_document_ceiling_refuses_zero_and_widening_before_any_io() {
    for (name, total) in [
        ("--max-document-forward-positions", 1_048_576_u64),
        ("--max-document-projected-logits", 100_000_000_000),
        ("--max-document-attention-pairs", 10_000_000_000_000),
        ("--max-document-dot-products", 10_000_000_000_000_000),
        ("--max-document-multiply-accumulates", 100_000_000_000_000_000),
        ("--max-document-mask-node-visits", 1_000_000_000_000),
    ] {
        for value in ["0".to_owned(), (total + 1).to_string()] {
            let mut diagnostics = Vec::new();
            let status = command(&[name, &value]).run_owned(NoIo, NoIo, &mut diagnostics);
            assert_eq!(status, ErrorCode::Usage.as_process_exit(), "{name}");
            let text = String::from_utf8(diagnostics).unwrap();
            assert!(text.contains("int8_sampled_sentiment_corpus_options"));
            assert!(!text.contains("private-model"));
            assert!(!text.contains(SEED));
        }
    }
}

#[test]
fn transport_reservation_and_candidate_frames_are_added_to_complete_document_peak() {
    let command = command(&["--max-requests", "30", "--max-output-mib", "64"]);
    let map = command.document.validate().unwrap();
    let (_, limits, envelope) = command.admit().unwrap();
    assert_eq!(envelope.transport.max_document_bytes, 1_048_576);
    assert_eq!(envelope.transport.max_output_line_bytes, map.envelope_bytes);
    assert_eq!(envelope.transport.max_output_bytes + 34 * FRAME_ALLOWANCE, 64 * MIB);
    assert_eq!(limits.minimum_memory_bytes, map.minimum_memory_bytes + envelope.transport_bytes);
    assert!(envelope.transport_bytes > envelope.io_bytes);
    assert_eq!(limits.memory, map.memory);
    assert!(super::tests::command(&["--max-requests", "100000", "--max-output-mib", "1"]).admit().is_err());
    assert!(super::tests::command(&["--max-line-bytes", "1024"]).admit().is_err());
}

#[test]
fn full_population_peak_is_refused_even_when_one_document_would_fit() {
    let (_, admitted, envelope) = command(&[]).admit().unwrap();
    let mut short = command(&[]);
    let document_peak = admitted.minimum_memory_bytes - envelope.transport_bytes;
    short.document.sampled.memory_mib = document_peak.div_ceil(MIB);
    assert!(short.document.sampled.memory_mib * MIB < admitted.minimum_memory_bytes);
    let mut diagnostics = Vec::new();
    assert_eq!(short.run_owned(NoIo, NoIo, &mut diagnostics), ErrorCode::Usage.as_process_exit());
}

#[test]
fn prompt_morsels_and_team_width_keep_population_work_and_semantic_samples() {
    let (serial, baseline, population) = command(&[]).admit().unwrap();
    for width in ["1", "2", "4", "64"] {
        let (scheduled, limits, envelope) = command(&["--compute-threads", width, "--prefill-rows", "4"])
            .admit().unwrap();
        assert_eq!(scheduled.sampled.options().unwrap(), serial.sampled.options().unwrap());
        assert_eq!(limits.task, baseline.task);
        assert_eq!(limits.mapping.chunks, baseline.mapping.chunks);
        assert_eq!(limits.mapping.max_model_work, baseline.mapping.max_model_work);
        assert_eq!(limits.mapping.max_retained_result_bytes, baseline.mapping.max_retained_result_bytes);
        assert_eq!(limits.mapping.max_result_bytes, baseline.mapping.max_result_bytes);
        assert_eq!(envelope.max_model_work, population.max_model_work);
        assert_eq!(envelope.max_mask_node_visits, population.max_mask_node_visits);
        assert_eq!(envelope.transport, population.transport);
        assert_eq!(limits.shared.compute,
            limits.common.compute.admit_prefill(limits.prefill).unwrap());
        assert!(limits.minimum_memory_bytes > baseline.minimum_memory_bytes);
    }
}

#[test]
fn compact_population_policy_keeps_explicit_cumulative_encoding_ceiling() {
    let (byte, _, _) = command(&[]).admit().unwrap();
    let (compact, limits, _) = command(&["--input-encoding", "safe-bpe-v1",
        "--max-tokenizer-input-bytes", "524288", "--max-tokenizer-output-tokens", "262144",
        "--max-tokenizer-pair-checks", "32000", "--max-tokenizer-pair-bytes", "64000"])
        .admit().unwrap();
    assert_eq!(compact.sampled.options().unwrap(), byte.sampled.options().unwrap());
    let UntrustedEncoding::SafeBpeV1(policy) = limits.encoding else { panic!("compact population required"); };
    assert_eq!((policy.max_input_bytes, policy.max_output_tokens), (524288, 262144));
    assert_eq!((policy.max_pair_checks, policy.max_pair_bytes), (32000, 64000));
    assert!(parsed(&["--max-tokenizer-input-bytes", "1"]).is_err());
}
