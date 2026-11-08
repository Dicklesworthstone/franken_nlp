//! Real pinned planning and bounded transport fixtures; no model loading.
use super::*;
use std::io::Cursor;
fn command(extra: &[&str]) -> TextBatchCommand {
    let mut args = vec!["text-batch", "--model", "local.fnlpq", "--memory-mib", "8192", "--cohort-rows", "3"];
    args.extend_from_slice(extra);
    TextBatchCommand::from_matches(&wire::definition().try_get_matches_from(args).unwrap()).unwrap()
}
fn planner(command: &TextBatchCommand) -> Planner {
    let facts = ArtifactIdentity { model_id: "Nanbeige4.2-3B".into(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".into(), recipe_id: "metadata-only-unit-fixture".into(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) };
    Planner::new(&command.common, command.validate().unwrap(), &facts).unwrap()
}
fn line(id: &str) -> String { format!("{{\"id\":\"{id}\",\"sample_index\":7,\"prompt\":\"hello\"}}\n") }
#[test]
fn full_cohorts_do_not_read_the_next_record_and_keep_global_sequences() {
    let command = command(&[]); let planner = planner(&command);
    let prefix = line("a") + &line("b");
    let mut input = Cursor::new(prefix.clone() + &line("c")); let mut bytes = 0; let mut ledger = Ledger::default();
    let first = read_record(&mut input, &command, &mut bytes).unwrap().unwrap();
    let cohort = collect(first, &mut input, &planner, &command, &mut ledger, &mut bytes, 2, &mut NeverCancelled).unwrap();
    assert_eq!(cohort.ids, ["a", "b"]); assert_eq!(cohort.first_request_seq, 1); assert!(!cohort.eof);
    assert_eq!(bytes, prefix.len() as u64); assert_eq!(input.position(), bytes);
    let first = read_record(&mut input, &command, &mut bytes).unwrap().unwrap();
    let tail = collect(first, &mut input, &planner, &command, &mut ledger, &mut bytes, 2, &mut NeverCancelled).unwrap();
    assert_eq!(tail.ids, ["c"]); assert_eq!(tail.first_request_seq, 3); assert!(tail.eof); assert_eq!(ledger.records, 3);
}
#[test]
fn partial_tail_keeps_item_and_sample_identity_not_cohort_slot_identity() {
    let command = command(&[]); let planner = planner(&command);
    let mut input = Cursor::new(line("a") + &line("b")); let mut bytes = 0; let mut ledger = Ledger::default();
    let first = read_record(&mut input, &command, &mut bytes).unwrap().unwrap();
    let cohort = collect(first, &mut input, &planner, &command, &mut ledger, &mut bytes, 3, &mut NeverCancelled).unwrap();
    assert!(cohort.eof); assert_eq!(cohort.prepared.len(), 2);
    for (id, plan) in cohort.ids.iter().zip(&cohort.prepared) {
        let record = wire::parse_record(command.task, &line(id), command.common.max_input_bytes).unwrap();
        let (_, individual) = planner.prepare(record).unwrap();
        assert_eq!(plan.execution_identity(), individual.execution_identity());
    }
}
#[test]
fn duplicate_later_row_never_returns_an_executable_cohort() {
    let command = command(&[]); let planner = planner(&command);
    let mut input = Cursor::new(line("a") + &line("a")); let mut bytes = 0; let mut ledger = Ledger::default();
    let first = read_record(&mut input, &command, &mut bytes).unwrap().unwrap();
    assert!(collect(first, &mut input, &planner, &command, &mut ledger, &mut bytes, 3, &mut NeverCancelled).is_err());
    assert_eq!(ledger.records, 1); // no refund/retry or terminal-success path
}
#[test]
fn reaching_record_ceiling_does_not_swallow_an_excess_record() {
    let command = command(&["--max-records", "1"]); let planner = planner(&command);
    let mut input = Cursor::new(line("a") + &line("b")); let mut bytes = 0; let mut ledger = Ledger::default();
    let first = read_record(&mut input, &command, &mut bytes).unwrap().unwrap();
    let cohort = collect(first, &mut input, &planner, &command, &mut ledger, &mut bytes, 3, &mut NeverCancelled).unwrap();
    assert_eq!(cohort.ids, ["a"]); assert!(!cohort.eof);
    let excess = read_record(&mut input, &command, &mut bytes).unwrap().unwrap();
    assert!(collect(excess, &mut input, &planner, &command, &mut ledger, &mut bytes, 3, &mut NeverCancelled).is_err());
    assert_eq!(ledger.records, 1);
}
#[test]
fn whole_group_delivery_is_admitted_once_before_any_output() {
    let mut bytes = Vec::new(); let output = Output::new(&mut bytes, FOOTER_BYTES as u64 + 199);
    assert!(output.admit_frame(100).is_ok());
    assert!(output.admit_frame(delivery_bytes(2, 100).unwrap()).is_err());
    assert_eq!(output.written, 0);
    assert!(delivery_bytes(0, 100).is_err()); assert!(delivery_bytes(65, 100).is_err());
    assert!(delivery_bytes(2, usize::MAX).is_err());
}
#[test]
fn exhausted_preparation_deadline_stops_without_consuming_more_rows() {
    struct Expired;
    impl DecodeStepControl for Expired {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
    }
    let command = command(&[]); let planner = planner(&command);
    let mut input = Cursor::new(line("a") + &line("b")); let mut bytes = 0; let mut ledger = Ledger::default();
    let first = read_record(&mut input, &command, &mut bytes).unwrap().unwrap(); let consumed = bytes;
    assert!(collect(first, &mut input, &planner, &command, &mut ledger, &mut bytes, 3,
        &mut Expired).is_err());
    assert_eq!(ledger.records, 0); assert_eq!(bytes, consumed); assert_eq!(input.position(), consumed);
}
#[test]
fn packed_strategy_preserves_partial_tail_plans_and_whole_corpus_work() {
    let ordinary = command(&[]); let ordinary_planner = planner(&ordinary);
    for width in ["1", "4", "64"] {
        let selected = command(&["--prefill-rows", width]); let selected_planner = planner(&selected);
        let mut input = Cursor::new(line("a") + &line("b")); let mut bytes = 0; let mut ledger = Ledger::default();
        let first = read_record(&mut input, &selected, &mut bytes).unwrap().unwrap();
        let cohort = collect(first, &mut input, &selected_planner, &selected, &mut ledger, &mut bytes, 3,
            &mut NeverCancelled).unwrap();
        assert!(cohort.eof); assert_eq!(cohort.ids, ["a", "b"]); assert_eq!(ledger.records, 2);
        let mut positions = 0; let mut logits = 0;
        for (id, prepared) in cohort.ids.iter().zip(&cohort.prepared) {
            let record = wire::parse_record(ordinary.task, &line(id), ordinary.common.max_input_bytes).unwrap();
            let (_, scalar) = ordinary_planner.prepare(record).unwrap();
            assert_eq!(prepared.execution_identity(), scalar.execution_identity());
            assert_eq!(prepared.planned_work(), scalar.planned_work());
            positions += scalar.planned_work().forward_positions; logits += scalar.planned_work().projected_logits;
        }
        assert_eq!(ledger.work.forward_positions, positions); assert_eq!(ledger.work.projected_logits, logits);
    }
}

#[test]
fn compact_record_identity_is_the_same_in_serial_fixed_and_refilling_schedules() {
    let text = "é <think>literal</think> The quick brown fox jumps over the lazy dog.";
    let (work, _) = super::super::tests::measured_encoding(text);
    let checks = work.pair_checks().to_string(); let bytes_cap = work.pair_bytes().to_string();
    let flags = ["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", &checks,
        "--max-tokenizer-pair-bytes", &bytes_cap];
    let mut serial = command(&flags); serial.cohort_rows = None;
    let serial_planner = planner(&serial);
    let line = |id: &str| format!("{}\n", serde_json::json!({"id": id, "sample_index": 7, "prompt": text}));
    let (_, first) = serial_planner.prepare(wire::parse_record(serial.task, &line("a"),
        serial.common.max_input_bytes).unwrap()).unwrap();
    let (_, second) = serial_planner.prepare(wire::parse_record(serial.task, &line("b"),
        serial.common.max_input_bytes).unwrap()).unwrap();
    let expected = [first, second];
    for active in [None, Some(1), Some(2)] {
        let mut selected = command(&flags); selected.active_rows = active;
        let selected_planner = planner(&selected);
        let mut input = Cursor::new(line("a") + &line("b"));
        let mut bytes = 0; let mut ledger = Ledger::default();
        let first = read_record(&mut input, &selected, &mut bytes).unwrap().unwrap();
        let cohort = collect(first, &mut input, &selected_planner, &selected, &mut ledger,
            &mut bytes, 3, &mut NeverCancelled).unwrap();
        assert!(cohort.eof); assert_eq!(cohort.ids, ["a", "b"]);
        assert_eq!(selected.refill_strategy(2).unwrap().map(|(width, _)| width), active);
        for (plan, expected) in cohort.prepared.iter().zip(&expected) {
            assert_eq!(plan.execution_identity(), expected.execution_identity());
            assert_eq!(plan.planned_work(), expected.planned_work());
        }
        assert_eq!(ledger.work.forward_positions, expected.iter().map(|plan| plan.planned_work().forward_positions).sum::<u64>());
        assert_eq!(ledger.work.projected_logits, expected.iter().map(|plan| plan.planned_work().projected_logits).sum::<u64>());
    }
}

#[test]
fn fresh_compact_record_work_cannot_renew_native_corpus_limits_at_a_window_boundary() {
    let text = "The quick brown fox jumps over the lazy dog.";
    let (work, _) = super::super::tests::measured_encoding(text);
    let checks = work.pair_checks().to_string(); let bytes_cap = work.pair_bytes().to_string();
    let flags = ["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", &checks,
        "--max-tokenizer-pair-bytes", &bytes_cap];
    let line = |id: &str| format!("{}\n", serde_json::json!({"id": id, "prompt": text}));
    for active in [None, Some(1)] {
        for cap_positions in [false, true] {
            let mut selected = command(&flags); selected.cohort_rows = Some(2); selected.active_rows = active;
            let selected_planner = planner(&selected);
            let (_, one) = selected_planner.prepare(wire::parse_record(selected.task, &line("a"),
                selected.common.max_input_bytes).unwrap()).unwrap();
            if cap_positions { selected.max_total_forward_positions = 2 * one.planned_work().forward_positions; }
            else { selected.max_total_projected_logits = 2 * one.planned_work().projected_logits; }
            selected.validate().unwrap();
            let prefix = line("a") + &line("b");
            let mut input = Cursor::new(prefix.clone() + &line("c") + &line("d"));
            let mut bytes = 0; let mut ledger = Ledger::default();
            let first = read_record(&mut input, &selected, &mut bytes).unwrap().unwrap();
            let cohort = collect(first, &mut input, &selected_planner, &selected, &mut ledger,
                &mut bytes, 2, &mut NeverCancelled).unwrap();
            assert_eq!(cohort.ids, ["a", "b"]); assert_eq!(input.position(), prefix.len() as u64);
            let reserved_positions = ledger.work.forward_positions; let reserved_logits = ledger.work.projected_logits;
            let next = read_record(&mut input, &selected, &mut bytes).unwrap().unwrap();
            let consumed = bytes;
            assert!(matches!(collect(next, &mut input, &selected_planner, &selected, &mut ledger,
                &mut bytes, 2, &mut NeverCancelled), Err(CandidateError::Batch)));
            assert_eq!(ledger.records, 2); assert_eq!(ledger.work.forward_positions, reserved_positions);
            assert_eq!(ledger.work.projected_logits, reserved_logits);
            assert_eq!(bytes, consumed); assert_eq!(input.position(), consumed);
            assert_eq!(consumed, (prefix.len() + line("c").len()) as u64);
        }
    }
}

#[test]
fn cancellation_inside_compact_encoding_returns_no_cohort_and_does_not_read_lookahead() {
    struct Expires { checkpoints: usize, stop: usize }
    impl DecodeStepControl for Expires {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.checkpoints += 1;
            (self.checkpoints >= self.stop).then_some(DecodeCancellationKind::Deadline)
        }
    }
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(16);
    let (work, encoder_checkpoints) = super::super::tests::measured_encoding(&text);
    // The selected stop is well past fixed preparation checkpoints, but before
    // even the encoder's independently measured checkpoint sequence can finish.
    assert!(encoder_checkpoints > 32);
    let checks = work.pair_checks().to_string(); let bytes_cap = work.pair_bytes().to_string();
    let selected = command(&["--input-encoding", "safe-bpe-v1",
        "--max-tokenizer-pair-checks", &checks, "--max-tokenizer-pair-bytes", &bytes_cap]);
    let selected_planner = planner(&selected);
    let first_line = format!("{}\n", serde_json::json!({"id": "a", "prompt": text}));
    let mut input = Cursor::new(first_line.clone() + &line("must-remain-unread"));
    let mut bytes = 0; let mut ledger = Ledger::default();
    let first = read_record(&mut input, &selected, &mut bytes).unwrap().unwrap();
    let mut control = Expires { checkpoints: 0, stop: encoder_checkpoints / 2 };
    assert!(matches!(collect(first, &mut input, &selected_planner, &selected, &mut ledger,
        &mut bytes, 3, &mut control), Err(CandidateError::Timeout)));
    assert_eq!(control.checkpoints, control.stop);
    assert_eq!(ledger.records, 0); assert_eq!(ledger.work.forward_positions, 0);
    assert_eq!(ledger.work.projected_logits, 0);
    assert_eq!(bytes, first_line.len() as u64); assert_eq!(input.position(), bytes);
}
