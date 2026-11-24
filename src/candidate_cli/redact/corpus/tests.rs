//! Model-free CLI admission/ownership contracts, not native success fixtures.
use super::*;
use super::super::tests::command;

#[test]
fn ndjson_is_explicit_and_corpus_flags_cannot_be_ignored_by_single_mode() {
    let single = command(&[]); single.validate().unwrap(); assert!(!single.corpus.ndjson);
    let stream = command(&["--ndjson", "--max-requests", "2"]);
    let (_, limits) = stream.validate().unwrap(); let envelope = stream.corpus.envelope(&stream, limits).unwrap();
    assert_eq!(envelope.transport.max_requests, 2); assert!(stream.request().verify);
    for flag in ["--max-requests", "--max-input-mib", "--max-output-mib", "--max-line-bytes",
        "--max-corpus-forward-positions", "--max-corpus-projected-logits", "--max-corpus-attention-pairs",
        "--max-corpus-dot-products", "--max-corpus-multiply-accumulates", "--max-corpus-mask-node-visits"] {
        assert!(super::super::super::definition().try_get_matches_from(
            ["candidate", "redact", "--model", "local.fnlpq", "--memory-mib", "8192", flag, "1"]).is_err());
    }
}
#[test]
fn short_mode_prices_both_independent_contexts_and_every_document() {
    let c = command(&["--ndjson", "--max-requests", "3"]); let (_, limits) = c.validate().unwrap();
    let d = short_detector(&c, limits, NerOptions::default()).unwrap();
    let e = c.corpus.envelope(&c, limits).unwrap();
    assert_eq!(e.max_model_work, scale_work(d.max_model_work, 3).unwrap());
    assert_eq!(e.max_mask_visits, c.host.max_mask_node_visits * 6);
    let off = command(&["--ndjson", "--no-verify", "--max-requests", "3"]);
    let (_, limits) = off.validate().unwrap(); let one = short_detector(&off, limits, NerOptions::default()).unwrap();
    assert_eq!(d.max_model_work, one.max_model_work.checked_add(one.max_model_work).unwrap());
    assert!(!off.request().verify);
}
#[test]
fn chunked_mode_uses_document_ceiling_not_one_chunk_or_one_fictitious_context() {
    let c = command(&["--ndjson", "--chunked", "--max-requests", "2"]);
    let (_, limits) = c.validate().unwrap(); let d = c.long.config(&c.host, limits, NerOptions::default()).unwrap();
    let e = c.corpus.envelope(&c, limits).unwrap();
    assert_eq!(e.max_model_work.forward_positions, d.mapping.max_model_work.forward_positions * 2);
    assert_eq!(e.max_model_work.projections.multiply_accumulates, d.mapping.max_model_work.projections.multiply_accumulates * 2);
    assert_eq!(e.max_mask_visits, d.mapping.max_mask_visits * 2);
    assert_eq!(e.transport.max_work.forward_positions, e.max_model_work.forward_positions);
    assert_eq!(e.max_model_work.projected_logits % NANBEIGE_VOCAB_SIZE as u64, 0);
}
#[test]
fn complete_output_budget_includes_all_candidate_frames_and_owned_buffers() {
    let c = command(&["--ndjson", "--max-requests", "2", "--max-output-mib", "1"]);
    let (_, limits) = c.validate().unwrap(); let e = c.corpus.envelope(&c, limits).unwrap();
    assert_eq!(e.transport.max_output_bytes + 6 * FRAME_ALLOWANCE, e.output_bytes);
    assert_eq!(e.io_bytes, e.transport.max_output_line_bytes as u64 + FRAME_ALLOWANCE + IO_BUFFER_BYTES as u64 * 2);
    assert_eq!(e.transport.max_document_bytes, c.host.max_input_bytes);
    assert!(e.transport.max_output_line_bytes > c.host.max_result_bytes);
}
#[test]
fn transport_and_derived_work_overflow_refuse_before_io() {
    for args in [vec!["--ndjson", "--max-requests", "0"], vec!["--ndjson", "--max-input-mib", "0"],
        vec!["--ndjson", "--max-output-mib", "1"], vec!["--ndjson", "--max-line-bytes", "1"],
        vec!["--ndjson", "--chunked", "--max-requests", "100000"]] {
        assert!(command(&args).validate().is_err());
    }
    let work = Int8Work { forward_positions: u64::MAX, ..Int8Work::default() };
    assert!(scale_work(work, 2).is_err());
}
#[test]
fn whole_corpus_caps_can_tighten_but_never_expand_the_derived_authority() {
    let mut c = command(&["--ndjson", "--max-requests", "3"]);
    let (_, limits) = c.validate().unwrap(); let d = short_detector(&c, limits, NerOptions::default()).unwrap();
    c.corpus.max_corpus_forward_positions = Some(d.max_model_work.forward_positions * 2);
    c.corpus.max_corpus_mask_node_visits = Some(d.max_mask_visits * 2);
    let e = c.corpus.envelope(&c, limits).unwrap();
    assert_eq!(e.max_model_work.forward_positions, d.max_model_work.forward_positions * 2);
    assert_eq!(e.max_mask_visits, d.max_mask_visits * 2);
    let mut total = 20; tighten(&mut total, Some(100), 10).unwrap(); assert_eq!(total, 20);
    assert!(tighten(&mut total, Some(9), 10).is_err());
    for axis in 0..6 {
        let mut c = command(&["--ndjson", "--max-requests", "3"]);
        match axis { 0 => c.corpus.max_corpus_forward_positions = Some(1),
            1 => c.corpus.max_corpus_projected_logits = Some(1),
            2 => c.corpus.max_corpus_attention_pairs = Some(1),
            3 => c.corpus.max_corpus_dot_products = Some(1),
            4 => c.corpus.max_corpus_multiply_accumulates = Some(1),
            _ => c.corpus.max_corpus_mask_node_visits = Some(1) }
        assert!(c.validate().is_err());
    }
}
struct NoIo;
impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("must not read") } }
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("must not write") }
    fn flush(&mut self) -> io::Result<()> { panic!("must not flush") }
}
#[test]
fn borrowed_dispatch_never_collects_or_detaches_a_corpus() {
    let c = command(&["--ndjson"]);
    assert_eq!(c.execute(&mut NoIo, &mut NoIo).unwrap_err(), CandidateError::Arguments);
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_runtime_refuses_owned_mode_before_key_options_source_and_model_io() {
    let c = command(&["records.ndjson", "--ndjson", "--action", "pseudonymize", "--key-stdin",
        "--key-id", "test", "--namespace", "test", "--ner-options", "not-read.json"]);
    assert_eq!(c.execute_owned(NoIo, NoIo).unwrap_err(), CandidateError::Unavailable);
}
#[test]
fn binary_key_and_corpus_stdin_are_exclusive_and_commitment_is_checked_once() {
    let invalid = command(&["--ndjson", "--action", "pseudonymize", "--key-stdin", "--key-id", "rotation", "--namespace", "corpus"]);
    assert!(invalid.validate().is_err());
    let expected = "0".repeat(64);
    let c = command(&["records.ndjson", "--ndjson", "--chunked", "--action", "pseudonymize", "--key-stdin",
        "--key-id", "rotation", "--namespace", "corpus", "--expected-key-commitment", &expected]);
    c.validate().unwrap();
    let error = c.key(&mut std::io::Cursor::new([7_u8; 32])).err().unwrap();
    assert_eq!(error, CandidateError::Identity);
}


fn compact(extra: &[&str]) -> RedactCommand {
    let mut flags = vec!["--ndjson", "--chunked", "--input-encoding", "safe-bpe-v1"];
    flags.extend_from_slice(extra);
    let mut command = command(&flags);
    command.host.memory_mib = 16_384;
    command
}
fn retained(extra: &[&str]) -> RedactCommand {
    let mut flags = vec!["--store-results", "--job-dir", "private-job",
        "--job-id", "0123456789abcdef0123456789abcdef",
        "--job-key-file", "private-key", "--job-limits", "limits.json"];
    flags.extend_from_slice(extra);
    compact(&flags)
}

#[test]
fn compact_populations_keep_per_original_document_encoding_and_whole_corpus_native_work() {
    let a = compact(&["--max-requests", "2"]);
    let b = compact(&["--max-requests", "3"]);
    let (common, la) = a.validate().unwrap(); let (_, lb) = b.validate().unwrap();
    assert_eq!(a.encoding_policy(la).unwrap(), b.encoding_policy(lb).unwrap());
    let ea = a.corpus.envelope(&a, la).unwrap(); let eb = b.corpus.envelope(&b, lb).unwrap();
    assert_eq!(common.max_input_bytes, a.host.max_input_bytes);
    assert_eq!(ea.transport.max_document_bytes, a.host.max_input_bytes);
    assert_eq!(ea.max_model_work.forward_positions / 2, eb.max_model_work.forward_positions / 3);
    assert_eq!(ea.max_model_work.projected_logits / 2, eb.max_model_work.projected_logits / 3);
    assert_eq!(ea.max_mask_visits / 2, eb.max_mask_visits / 3);
    let (_, retained_limits) = retained(&[]).validate().unwrap();
    assert_eq!(a.encoding_policy(la).unwrap(), retained(&[]).encoding_policy(retained_limits).unwrap());

    for mut unsupported in [compact(&[]), retained(&[])] {
        unsupported.long.chunked = false;
        assert!(matches!(unsupported.validate(), Err(CandidateError::Arguments)));
    }
}

#[test]
fn compact_live_peak_prices_transport_native_output_rows_and_map_tokens_separately() {
    let mut command = compact(&["--max-requests", "2"]);
    let (_, limits) = command.validate().unwrap();
    let envelope = command.corpus.envelope(&command, limits).unwrap();
    let detector = command.long.config(&command.host, limits, NerOptions::default()).unwrap();
    let native = Int8MemoryRequirement::for_context(command.host.context_tokens).unwrap();
    let count = detector.mapping.chunks.max_chunks as u64;
    let rows = detector.mapping.max_model_work.projected_logits / NANBEIGE_VOCAB_SIZE as u64;
    let temporary = 4 * (detector.per_chunk.max_output_bytes * count
        + detector.mapping.reduction.max_live_value_bytes as u64)
        + 8 * u64::from(detector.per_chunk.max_output_tokens) * count + command.edit_reserve_mib * MIB;
    let t = envelope.transport;
    let expected = limits.weight_bytes + 128 * MIB + 64 * MIB
        + 2 * limits.preparation_bytes + native.kv_bytes + native.rope_bytes
        + native.scratch_payload_bound + 64 * MIB
        + 4 * detector.max_result_bytes + 8 * rows + 4096 + 65_536 + 64 * 1024
        + 8 * t.max_line_bytes as u64 + 4 * t.max_output_line_bytes as u64
        + 128 * t.max_epoch_ids as u64 + t.max_epoch_id_bytes as u64 + envelope.io_bytes
        + temporary + 4096;
    assert!(rows > 0);
    assert_eq!(command.corpus.compact_base_peak_bytes(&command, limits, command.edit_reserve_mib * MIB).unwrap(), expected);
    command.host.memory_mib = expected.div_ceil(MIB) - 1;
    assert!(command.host.common(command.input.clone()).is_ok());
    assert!(matches!(command.validate(), Err(CandidateError::Arguments)));
    command.host.memory_mib += 1;
    assert!(command.validate().is_ok());
}

#[test]
fn compact_retained_floor_prices_fixed_storage_and_does_not_pretend_to_cover_population() {
    let command = retained(&[]);
    let (_, limits) = command.validate().unwrap();
    let native = Int8MemoryRequirement::for_context(command.host.context_tokens).unwrap();
    let detector = command.long.config(&command.host, limits, NerOptions::default()).unwrap();
    let rows = detector.mapping.max_model_work.projected_logits / NANBEIGE_VOCAB_SIZE as u64;
    let base = compute::request_peak_bytes(limits, native, detector.max_result_bytes,
        rows, 0, limits.preparation_bytes).unwrap();
    let temporary = command.long.temporary_bytes(&command.host, command.edit_reserve_mib * MIB).unwrap();
    let expected = base + temporary + 4096 + 2 * IO_BUFFER_BYTES as u64
        + command.retention.journal_bytes().unwrap() + command.retention.serialization_bytes().unwrap()
        + 4 * (MIB - 1024);
    assert_eq!(command.corpus.compact_base_peak_bytes(&command, limits, command.edit_reserve_mib * MIB).unwrap(), expected);
    // The bounded JobLimits file is deliberately unopened here; runtime adds
    // its actual CompletePopulation/RetainedRecord owner before private keys.
    let larger = retained(&["--journal-memory-mib", "128", "--serialization-memory-mib", "32"]);
    let (_, larger_limits) = larger.validate().unwrap();
    let larger_peak = larger.corpus.compact_base_peak_bytes(&larger, larger_limits, larger.edit_reserve_mib * MIB).unwrap();
    assert_eq!(larger_peak - expected, 80 * MIB);
    let resumed = retained(&["--resume", "--use-retained-input", "--max-retained-input-mib", "1"]);
    let (_, resumed_limits) = resumed.validate().unwrap();
    assert_eq!(resumed.encoding_policy(resumed_limits).unwrap(), command.encoding_policy(limits).unwrap());
}

#[test]
fn compact_population_work_and_memory_refusals_precede_private_io() {
    for extra in [
        vec!["--max-total-tokenizer-pair-checks", "0"],
        vec!["--max-total-tokenizer-input-bytes", "18446744073709551615"],
        vec!["--max-corpus-forward-positions", "1"],
        vec!["--max-requests", "100000"],
    ] {
        let command = compact(&extra);
        assert_eq!(command.execute_owned(NoIo, NoIo), Err(CandidateError::Arguments));
    }
    let mut low = compact(&[]);
    low.host.memory_mib = 1024;
    assert_eq!(low.execute_owned(NoIo, NoIo), Err(CandidateError::Arguments));
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn compact_populations_without_runtime_do_not_consume_key_options_or_population() {
    let command = compact(&["private.ndjson", "--action", "pseudonymize", "--key-stdin",
        "--key-id", "rotation", "--namespace", "scope", "--ner-options", "private-options.json"]);
    assert_eq!(command.execute_owned(NoIo, NoIo), Err(CandidateError::Unavailable));
    assert_eq!(retained(&[]).execute_owned(NoIo, NoIo), Err(CandidateError::Unavailable));
}

fn scheduled_live(extra: &[&str]) -> RedactCommand {
    let mut flags = vec!["--ndjson", "--max-requests", "3"];
    flags.extend_from_slice(extra);
    let mut command = command(&flags);
    command.host.memory_mib = 16_384;
    command
}

#[test]
fn live_schedules_preserve_item_policy_and_nonrenewable_corpus_allowances() {
    for mode in 0..3 {
        let base = match mode {
            0 => vec![],
            1 => vec!["--chunked"],
            _ => vec!["--chunked", "--input-encoding", "safe-bpe-v1"],
        };
        let ordinary = scheduled_live(&base);
        let (_, ordinary_limits) = ordinary.validate().unwrap();
        let envelope = ordinary.corpus.envelope(&ordinary, ordinary_limits).unwrap();
        for schedule in [
            vec!["--compute-threads", "4"],
            vec!["--prefill-rows", "7"],
            vec!["--compute-threads", "4", "--prefill-rows", "7"],
        ] {
            let mut flags = base.clone(); flags.extend(schedule);
            let command = scheduled_live(&flags);
            let (common, limits) = command.validate().unwrap();
            let actual = command.corpus.envelope(&command, limits).unwrap();
            assert_eq!(actual.transport, envelope.transport);
            assert_eq!(actual.max_model_work, envelope.max_model_work);
            assert_eq!(actual.max_mask_visits, envelope.max_mask_visits);
            assert_eq!(actual.output_bytes, envelope.output_bytes);
            assert_eq!(actual.io_bytes, envelope.io_bytes);
            assert_eq!(command.encoding_policy(limits).unwrap(), ordinary.encoding_policy(ordinary_limits).unwrap());
            assert_eq!(serde_json::to_vec(&command.request()).unwrap(),
                serde_json::to_vec(&ordinary.request()).unwrap());
            assert_eq!(common.compute, command.compute);
            assert!(common.policy.prefill().unwrap().is_none());
        }
    }
}

#[test]
fn live_scheduling_prices_one_team_full_transport_and_both_detector_stages_before_io() {
    use crate::native_engine::strict_int8::{prefill::Int8PrefillLimits,
        projection::int8_team_workspace_bytes_with_prefill};
    for chunked in [false, true] {
        let mut flags = vec!["--compute-threads", "4", "--prefill-rows", "7"];
        if chunked { flags.push("--chunked"); }
        let mut command = scheduled_live(&flags);
        let (_, limits) = command.validate().unwrap();
        let envelope = command.corpus.envelope(&command, limits).unwrap();
        let native = Int8MemoryRequirement::for_context(command.host.context_tokens).unwrap();
        let edit = command.edit_reserve_mib * MIB;
        let (output_rows, temporary) = if chunked {
            let d = command.long.config(&command.host, limits, NerOptions::default()).unwrap();
            let chunks = d.mapping.chunks.max_chunks as u64;
            (d.mapping.max_model_work.projected_logits / NANBEIGE_VOCAB_SIZE as u64,
                4 * (d.per_chunk.max_output_bytes * chunks + d.mapping.reduction.max_live_value_bytes as u64)
                    + 8 * u64::from(d.per_chunk.max_output_tokens) * chunks + edit)
        } else {
            let d = short_detector(&command, limits, NerOptions::default()).unwrap();
            (d.max_model_work.projected_logits / NANBEIGE_VOCAB_SIZE as u64,
                4 * d.per_pass.max_output_bytes + 8 * u64::from(d.per_pass.max_output_tokens) + edit)
        };
        let t = envelope.transport;
        let expected = limits.weight_bytes + 192 * MIB + 2 * limits.preparation_bytes
            + native.kv_bytes + native.rope_bytes + native.scratch_payload_bound + 64 * MIB
            + int8_team_workspace_bytes_with_prefill(4, 7).unwrap()
            + Int8PrefillLimits::required_extra_scratch_bytes(7).unwrap()
            + 4 * command.host.max_result_bytes as u64 + 8 * output_rows + 4096
            + 65_536 + 64 * 1024 + 8 * t.max_line_bytes as u64 + 4 * t.max_output_line_bytes as u64
            + 128 * t.max_epoch_ids as u64 + t.max_epoch_id_bytes as u64 + envelope.io_bytes
            + temporary + 4096;
        assert_eq!(command.corpus.compact_base_peak_bytes(&command, limits, edit).unwrap(), expected);
        command.corpus.max_requests = Some(4);
        let (_, more) = command.validate().unwrap();
        assert_eq!(command.corpus.compact_base_peak_bytes(&command, more, edit).unwrap(), expected,
            "a second document or flush does not allocate another native engine or numerical team");
        command.host.memory_mib = expected.div_ceil(MIB) - 1;
        assert_eq!(command.execute_owned(NoIo, NoIo), Err(CandidateError::Arguments));
        let mut exact = scheduled_live(&flags); exact.host.memory_mib = expected.div_ceil(MIB);
        assert!(exact.validate().is_ok());
    }
}

#[test]
fn invalid_live_and_retained_schedules_fail_before_owned_private_io() {
    for flags in [
        vec!["--compute-threads", "0"], vec!["--compute-threads", "65"],
        vec!["--prefill-rows", "0"], vec!["--prefill-rows", "65"],
        vec!["--compute-threads", "4", "--max-corpus-mask-node-visits", "1"],
        vec!["--prefill-rows", "7", "--input-encoding", "safe-bpe-v1"],
    ] {
        assert_eq!(scheduled_live(&flags).execute_owned(NoIo, NoIo), Err(CandidateError::Arguments));
    }
    for flags in [
        vec!["--compute-threads", "0"], vec!["--prefill-rows", "65"],
        vec!["--compute-threads", "4", "--prefill-rows", "0"],
    ] {
        let command = retained(&flags);
        assert!(matches!(command.validate(), Err(CandidateError::Arguments)));
        assert_eq!(command.execute_owned(NoIo, NoIo), Err(CandidateError::Arguments));
    }
    assert!(retained(&["--compute-threads", "1"]).validate().is_ok());
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn scheduled_live_mode_without_runtime_never_reads_key_options_population_or_model() {
    for chunked in [false, true] {
        let mut flags = vec!["private.ndjson", "--compute-threads", "4", "--prefill-rows", "7",
            "--action", "pseudonymize", "--key-stdin", "--key-id", "k", "--namespace", "scope",
            "--ner-options", "private-options.json"];
        if chunked { flags.extend(["--chunked", "--input-encoding", "safe-bpe-v1"]); }
        assert_eq!(scheduled_live(&flags).execute_owned(NoIo, NoIo), Err(CandidateError::Unavailable));
    }
}

#[test]
fn retained_start_and_resume_accept_physical_schedules_without_changing_saved_policy() {
    for mode in [vec![], vec!["--resume"],
        vec!["--resume", "--use-retained-input", "--max-retained-input-mib", "4"]] {
        let ordinary = retained(&mode); let (_, ordinary_limits) = ordinary.validate().unwrap();
        let envelope = ordinary.corpus.envelope(&ordinary, ordinary_limits).unwrap();
        for schedule in [
            vec!["--compute-threads", "4"], vec!["--prefill-rows", "7"],
            vec!["--compute-threads", "4", "--prefill-rows", "7"],
        ] {
            let mut flags = mode.clone(); flags.extend(schedule);
            let command = retained(&flags); let (common, limits) = command.validate().unwrap();
            let current = command.corpus.envelope(&command, limits).unwrap();
            assert_eq!(common.compute, command.compute);
            assert!(common.policy.prefill().unwrap().is_none());
            assert_eq!(current.transport, envelope.transport);
            assert_eq!(current.max_model_work, envelope.max_model_work);
            assert_eq!(current.max_mask_visits, envelope.max_mask_visits);
            assert_eq!(command.encoding_policy(limits).unwrap(), ordinary.encoding_policy(ordinary_limits).unwrap());
            assert_eq!(serde_json::to_vec(&command.request()).unwrap(),
                serde_json::to_vec(&ordinary.request()).unwrap());
        }
    }
}
