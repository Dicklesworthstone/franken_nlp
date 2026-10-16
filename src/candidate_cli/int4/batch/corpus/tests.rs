//! Synthetic corpus transport/accounting cases; no native model is involved.
use super::*;
use std::io::{BufReader, Cursor};

fn options(rows: usize) -> CorpusArgs {
    CorpusArgs { task: BatchTask::Generate, cohort_rows: rows, active_rows: None, max_records: 1000,
        max_total_input_bytes: 1 << 20, max_total_output_bytes: 1 << 20,
        max_total_forward_positions: u64::MAX, max_total_projected_logits: u64::MAX,
        max_total_attention_pairs: u64::MAX, max_total_group_dots: u64::MAX,
        max_total_multiply_accumulates: u64::MAX, max_total_execution_checkpoints: 1000 }
}
fn line(id: usize) -> String { format!("{{\"id\":\"doc-{id}\",\"sample_index\":7,\"prompt\":\" é 上海 \"}}\n") }
fn work(positions: usize) -> Int4Work { Int4Work::for_sequence(128, 0, positions, NANBEIGE_VOCAB_SIZE).unwrap() }

#[test]
fn more_than_64_records_use_bounded_windows_and_monotone_delivery_addresses() {
    let text: String = (0..137).map(line).collect();
    for rows in [1, 2, 7, 64] {
        let mut reader = BufReader::with_capacity(3, Cursor::new(text.as_bytes()));
        let mut state = CorpusInput::new(options(rows)).unwrap();
        let mut observed = 0;
        loop {
            let window = state.next_window(&mut reader, 1024, || Ok(())).unwrap();
            assert_eq!(window.first_sequence, observed + 1);
            assert!(window.records.len() <= rows);
            if window.records.is_empty() { break; }
            for record in window.records {
                assert_eq!(record.id, format!("doc-{observed}")); assert_eq!(record.sample_index, 7);
                let wire::Content::Generate(prompt) = record.content else { panic!("wrong task") };
                assert_eq!(prompt, " é 上海 "); observed += 1;
            }
        }
        assert_eq!(observed, 137); assert_eq!(state.records, 137);
        assert_eq!(state.consumed, text.len() as u64); assert!(state.eof);
    }
}

#[test]
fn full_window_does_not_parse_or_wait_for_a_following_record() {
    let first = line(0); let text = format!("{first}not-json\n");
    let mut reader = Cursor::new(text.as_bytes()); let mut state = CorpusInput::new(options(1)).unwrap();
    assert_eq!(state.next_window(&mut reader, 1024, || Ok(())).unwrap().records.len(), 1);
    assert_eq!(reader.position(), first.len() as u64); assert!(!state.eof);
    assert!(state.next_window(&mut reader, 1024, || Ok(())).is_err());
    let stopped = reader.position();
    assert!(state.next_window(&mut reader, 1024, || panic!("failed reader resumed")).is_err());
    assert_eq!(reader.position(), stopped);
}

#[test]
fn duplicate_ids_across_windows_and_empty_or_malformed_records_fail_stop() {
    let repeated = format!("{}{}", line(0), line(0));
    let mut input = Cursor::new(repeated.as_bytes()); let mut state = CorpusInput::new(options(1)).unwrap();
    state.next_window(&mut input, 1024, || Ok(())).unwrap();
    assert!(state.next_window(&mut input, 1024, || Ok(())).is_err());
    assert_eq!(state.records, 1); assert!(state.failed);
    for bytes in [b"\n".as_slice(), b"\xff\n".as_slice(),
        b"{\"id\":\"a\",\"id\":\"b\",\"prompt\":\"x\"}\n".as_slice(),
        b"{\"id\":\"a\",\"prompt\":\"x\",\"seed\":3}\n".as_slice()] {
        let mut state = CorpusInput::new(options(2)).unwrap();
        assert!(state.next_window(&mut Cursor::new(bytes), 1024, || Ok(())).is_err());
        assert!(state.failed);
    }
}

#[test]
fn exact_record_and_crlf_byte_caps_require_observed_eof() {
    let a = line(0).replace('\n', "\r\n"); let b = line(1).trim_end().to_owned();
    let text = format!("{a}{b}"); let mut opts = options(1);
    opts.max_records = 2; opts.max_total_input_bytes = text.len() as u64;
    let mut state = CorpusInput::new(opts).unwrap(); let mut reader = Cursor::new(text.as_bytes());
    for _ in 0..2 { assert_eq!(state.next_window(&mut reader, 1024, || Ok(())).unwrap().records.len(), 1); }
    assert!(!state.eof);
    assert!(state.next_window(&mut reader, 1024, || Ok(())).unwrap().records.is_empty());
    assert!(state.eof); assert_eq!(state.consumed, text.len() as u64);
    opts.max_total_input_bytes -= 1;
    let mut state = CorpusInput::new(opts).unwrap(); let mut reader = Cursor::new(text.as_bytes());
    state.next_window(&mut reader, 1024, || Ok(())).unwrap();
    assert!(state.next_window(&mut reader, 1024, || Ok(())).is_err());
    opts.max_records = 1; opts.max_total_input_bytes += 1;
    let mut state = CorpusInput::new(opts).unwrap(); let mut reader = Cursor::new(text.as_bytes());
    state.next_window(&mut reader, 1024, || Ok(())).unwrap();
    assert!(state.next_window(&mut reader, 1024, || Ok(())).is_err());
    assert_eq!(reader.position(), a.len() as u64);
}

#[test]
fn empty_corpus_is_distinct_from_blank_record_and_needs_no_window_execution() {
    let mut state = CorpusInput::new(options(8)).unwrap();
    let result = state.next_window(&mut Cursor::new(b""), 1024, || Ok(())).unwrap();
    assert!(result.records.is_empty());
    let summary = CorpusLedger::default().summary(&state, 0).unwrap();
    assert_eq!(summary.records, 0); assert_eq!(summary.windows, 0);
    assert_eq!(summary.reserved_work, Int4Work::default());
    assert_eq!(summary.event, "corpus_complete");
}

#[test]
fn cancellation_in_the_next_window_cannot_reset_prior_consumption_or_seen_ids() {
    let text: String = (0..3).map(line).collect();
    let mut state = CorpusInput::new(options(1)).unwrap(); let mut reader = Cursor::new(text.as_bytes());
    state.next_window(&mut reader, 1024, || Ok(())).unwrap();
    let bytes = state.consumed;
    assert!(matches!(state.next_window(&mut reader, 1024, || Err(CandidateError::Timeout)), Err(CandidateError::Timeout)));
    assert_eq!(state.consumed, bytes); assert_eq!(state.ids.len(), 1); assert!(state.failed);
}

#[test]
fn every_cumulative_work_axis_and_checkpoint_ceiling_is_atomic() {
    let planned = work(4); let twice = planned.checked_add(planned).unwrap();
    for axis in 0..6 {
        let mut opts = options(1);
        match axis {
            0 => opts.max_total_forward_positions = twice.forward_positions - 1,
            1 => opts.max_total_projected_logits = twice.projected_logits - 1,
            2 => opts.max_total_attention_pairs = twice.attention_pairs - 1,
            3 => opts.max_total_group_dots = twice.projections.dot_products - 1,
            4 => opts.max_total_multiply_accumulates = twice.projections.multiply_accumulates - 1,
            _ => opts.max_total_execution_checkpoints = 19,
        }
        let mut ledger = CorpusLedger::default();
        ledger.admit(1, 1, planned, 10, opts).unwrap(); ledger.complete(planned, work(1)).unwrap();
        assert!(ledger.admit(2, 1, planned, 10, opts).is_err());
        assert_eq!(ledger.reserved, planned); assert_eq!(ledger.checkpoints, 10);
        assert_eq!(ledger.actual, work(1)); assert_eq!(ledger.completed_records, 1); assert!(ledger.pending.is_none());
    }
}

#[test]
fn early_stop_reconciles_actual_work_without_refunding_the_reserved_worst_case() {
    let mut opts = options(1); opts.max_total_forward_positions = 8;
    let mut ledger = CorpusLedger::default(); let planned = work(4);
    for sequence in [1, 2] {
        ledger.admit(sequence, 1, planned, 10, opts).unwrap();
        ledger.complete(planned, work(1)).unwrap();
    }
    assert_eq!(ledger.reserved.forward_positions, 8); assert_eq!(ledger.actual.forward_positions, 2);
    assert_eq!(ledger.checkpoints, 20);
    assert!(ledger.admit(3, 1, work(1), 1, opts).is_err());
}

#[test]
fn pending_failed_or_misrouted_windows_never_produce_a_complete_corpus() {
    let mut input = CorpusInput::new(options(1)).unwrap();
    let text = line(0); input.next_window(&mut Cursor::new(text.as_bytes()), 1024, || Ok(())).unwrap();
    let mut ledger = CorpusLedger::default();
    assert!(ledger.summary(&input, 0).is_err());
    ledger.admit(1, 1, work(4), 10, options(1)).unwrap();
    assert!(ledger.admit(1, 1, work(4), 10, options(1)).is_err());
    assert!(ledger.complete(work(3), work(1)).is_err());
    assert!(ledger.complete(work(4), work(5)).is_err());
    input.eof = true; assert!(ledger.summary(&input, 0).is_err());
    ledger.complete(work(4), work(1)).unwrap();
    assert!(ledger.complete(work(4), work(1)).is_err());
    assert!(ledger.admit(1, 1, work(1), 1, options(1)).is_err());
    assert_eq!(ledger.summary(&input, 100).unwrap().output_bytes_before_footer, 100);
    input.failed = true; assert!(ledger.summary(&input, 100).is_err());
}

#[test]
fn chat_role_validation_is_shared_and_refuses_a_late_bad_window() {
    let mut opts = options(1); opts.task = BatchTask::Chat;
    let text = b"{\"id\":\"a\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}\n{\"id\":\"b\",\"messages\":[{\"role\":\"assistant\",\"content\":\"bad\"}]}\n";
    let mut state = CorpusInput::new(opts).unwrap(); let mut reader = Cursor::new(text);
    assert_eq!(state.next_window(&mut reader, 1024, || Ok(())).unwrap().records.len(), 1);
    assert!(state.next_window(&mut reader, 1024, || Ok(())).is_err()); assert_eq!(state.records, 1);
}

#[test]
fn cumulative_output_admission_preserves_space_for_the_terminal_frame() {
    let mut bytes = Vec::new(); let mut out = wire::Output::new(&mut bytes, wire::FOOTER_BYTES as u64 + 7);
    out.admit_frame(7).unwrap(); out.write_all(b"1234567").unwrap();
    assert!(out.admit_frame(1).is_err());
    out.write_all(&vec![0; wire::FOOTER_BYTES]).unwrap();
    assert!(out.write_all(b"x").is_err());
}

#[test]
fn active_rows_is_a_bounded_subset_of_each_corpus_window() {
    let mut opts = options(8);
    for rows in [1, 2, 8] {
        opts.active_rows = Some(rows);
        assert!(opts.check().is_ok());
    }
    for rows in [0, 9, usize::MAX] {
        opts.active_rows = Some(rows);
        assert!(opts.check().is_err());
    }
    opts.active_rows = None;
    assert!(opts.check().is_ok());
}
