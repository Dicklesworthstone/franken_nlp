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
