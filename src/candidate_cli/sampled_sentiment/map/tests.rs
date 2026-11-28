//! CLI contract and complete-document admission, without fabricated inference.
use super::*;
use std::io;

const SEED: &str = "4545454545454545454545454545454545454545454545454545454545454545";

fn parsed(extra: &[&str]) -> Result<CandidateCommand, clap::Error> {
    let mut argv = vec!["candidate", "sentiment-sampled-map", "--model", "private-model.fnlpq",
        "--memory-mib", "65536", "--seed", SEED];
    argv.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    CandidateCommand::from_matches(&matches)
}

fn command(extra: &[&str]) -> SampledSentimentMapCommand {
    let CandidateCommand::SampledSentimentMap(command) = parsed(extra).unwrap()
        else { panic!("complete sampled map required"); };
    command
}

struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused map read private input") }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused map published output") }
    fn flush(&mut self) -> io::Result<()> { panic!("refused map flushed output") }
}

#[test]
fn sampled_document_has_explicit_identity_and_does_not_select_distribution_or_corpus() {
    let command = command(&[]);
    let limits = command.validate().unwrap();
    assert_eq!(limits.mapping.chunks.max_input_bytes, 1_048_576);
    assert_eq!(limits.mapping.chunks.reserved_tokens, command.sampled.max_new_tokens - 1);
    assert_eq!(limits.encoding, UntrustedEncoding::ByteFallback);
    let base = ["candidate", "sentiment-sampled-map", "--model", "m", "--memory-mib", "65536", "--seed", SEED];
    for missing in [2, 4, 6] {
        let args = base.iter().enumerate().filter_map(|(index, arg)|
            (index != missing && index != missing + 1).then_some(*arg)).collect::<Vec<_>>();
        assert!(crate::candidate_cli::definition().try_get_matches_from(args).is_err());
    }
    for extra in [vec!["--ndjson"], vec!["--reuse-prefill"], vec!["--bins", "5"],
        vec!["--logprobs"], vec!["--source", "model"]] {
        assert!(parsed(&extra).is_err());
    }
}

#[test]
fn selected_document_projection_is_explicit_and_keeps_every_memory_owner() {
    let dense = command(&["--compute-threads", "4", "--prefill-rows", "7"]);
    let selected = command(&["--compute-threads", "4", "--prefill-rows", "7", "--max-selected-rows", "4096"]);
    let dense_limits = dense.validate().unwrap();
    let selected_limits = selected.validate().unwrap();
    assert_eq!(selected.sampled.selected_rows().unwrap(), Some(Int8JsonSparseLimits { max_rows_per_step: 4096 }));
    assert_eq!(dense.sampled.options().unwrap(), selected.sampled.options().unwrap());
    assert_eq!(dense_limits.memory, selected_limits.memory);
    assert_eq!(dense_limits.minimum_memory_bytes, selected_limits.minimum_memory_bytes);
    assert_eq!(dense_limits.mapping.chunks, selected_limits.mapping.chunks);
    assert_eq!(dense_limits.mapping.max_model_work, selected_limits.mapping.max_model_work);
    assert_eq!(dense_limits.mapping.max_retained_result_bytes, selected_limits.mapping.max_retained_result_bytes);
    assert_eq!(dense_limits.mapping.max_result_bytes, selected_limits.mapping.max_result_bytes);
    for cap in ["0", "166145"] {
        assert_eq!(command(&["--max-selected-rows", cap]).execute(&mut NoIo, &mut NoIo)
            .unwrap_err().exit, ErrorCode::Usage);
    }
}

#[test]
fn one_document_encoding_allowance_is_independent_of_native_context_and_sample_policy() {
    let default = command(&[]);
    let compact = command(&["--input-encoding", "safe-bpe-v1",
        "--max-tokenizer-input-bytes", "524288", "--max-tokenizer-output-tokens", "262144",
        "--max-tokenizer-pair-checks", "32000", "--max-tokenizer-pair-bytes", "64000"]);
    let byte_limits = default.validate().unwrap();
    let compact_limits = compact.validate().unwrap();
    assert_eq!(default.sampled.options().unwrap(), compact.sampled.options().unwrap());
    assert_eq!(byte_limits.task, compact_limits.task);
    assert_eq!(byte_limits.memory.native, compact_limits.memory.native);
    assert!(compact_limits.memory.input_bytes > byte_limits.memory.input_bytes);
    let UntrustedEncoding::SafeBpeV1(policy) = compact_limits.encoding else { panic!("compact policy"); };
    assert_eq!((policy.max_input_bytes, policy.max_output_tokens), (524288, 262144));
    assert_eq!((policy.max_pair_checks, policy.max_pair_bytes), (32000, 64000));
    assert!(parsed(&["--max-tokenizer-input-bytes", "1"]).is_err());
    assert!(parsed(&["--max-tokenizer-pair-checks", "1"]).is_err());
}

#[test]
fn physical_team_and_prompt_schedule_preserve_semantics_and_charge_every_worker() {
    let serial = command(&[]);
    let baseline = serial.validate().unwrap();
    let mut parallel = command(&["--compute-threads", "64", "--prefill-rows", "64"]);
    let admitted = parallel.validate().unwrap();
    assert_eq!(serial.sampled.options().unwrap(), parallel.sampled.options().unwrap());
    assert_eq!(baseline.task, admitted.task);
    assert_eq!(baseline.memory, admitted.memory);
    assert!(admitted.shared.compute.workspace_bytes > MIB);
    assert!(admitted.minimum_memory_bytes > baseline.minimum_memory_bytes);
    assert_eq!(admitted.shared.compute,
        admitted.common.compute.admit_prefill(admitted.prefill).unwrap());
    let coordinator = admitted.minimum_memory_bytes - admitted.shared.compute.workspace_bytes;
    parallel.sampled.memory_mib = coordinator.div_ceil(MIB);
    assert!(parallel.sampled.memory_mib * MIB < admitted.minimum_memory_bytes);
    assert_eq!(parallel.execute(&mut NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
}

#[test]
fn complete_plan_caps_are_priced_together_and_invalid_requests_fail_before_io() {
    let smaller = command(&["--max-chunks", "2"]).validate().unwrap();
    let larger = command(&["--max-chunks", "16"]).validate().unwrap();
    assert_eq!(larger.memory.max_output_tokens, smaller.memory.max_output_tokens * 8);
    assert!(larger.memory.input_bytes > smaller.memory.input_bytes);
    assert!(larger.memory.output_reservation_bytes > smaller.memory.output_reservation_bytes);
    for case in 0..12 {
        let mut invalid = command(&[]);
        match case {
            0 => invalid.mapping.max_chunks = 0,
            1 => invalid.mapping.max_chunks = 257,
            2 => invalid.mapping.max_chunk_bytes = Some(3),
            3 => invalid.mapping.max_tokenizer_calls = 0,
            4 => invalid.mapping.max_map_result_bytes = 0,
            5 => invalid.mapping.max_retained_result_bytes = invalid.sampled.max_result_bytes as u64 - 1,
            6 => invalid.sampled.samples = 0,
            7 => invalid.sampled.seed = "private-invalid-seed".into(),
            8 => invalid.sampled.memory_mib = 1,
            9 => invalid.sampled.prefill_rows = Some(65),
            10 => invalid.sampled.max_forward_positions = 0,
            _ => invalid.mapping.max_map_result_bytes = 64 * MIB + 1,
        }
        let failure = invalid.execute(&mut NoIo, &mut NoIo).unwrap_err();
        assert_eq!(failure.exit, ErrorCode::Usage, "case {case}");
        assert_eq!(failure.code, "int8_sampled_sentiment_map_options", "case {case}");
    }
}

#[test]
fn executable_route_keeps_private_diagnostics_and_emits_no_partial_document() {
    let mut invalid = command(&[]);
    invalid.sampled.seed = "private-invalid-seed".into();
    invalid.sampled.input = PathBuf::from("private-document-path");
    let mut diagnostics = Vec::new();
    let status = CandidateCommand::SampledSentimentMap(invalid).run(&mut NoIo, &mut NoIo, &mut diagnostics);
    assert_eq!(status, ErrorCode::Usage.as_process_exit());
    assert_eq!(String::from_utf8(diagnostics).unwrap(),
        "fnlp candidate INT8 sampled sentiment: int8_sampled_sentiment_map_options\n");
}
