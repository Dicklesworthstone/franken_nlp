//! Real pinned planning and bounded transport fixtures; no model loading.
use super::*;
use std::io::Cursor;
fn command(extra: &[&str]) -> TextBatchCommand {
    let mut args = vec!["text-batch", "--model", "local.fnlpq", "--memory-mib", "8192", "--cohort-rows", "3"];
    args.extend_from_slice(extra);
    TextBatchCommand::from_arg_matches(&wire::definition().try_get_matches_from(args).unwrap()).unwrap()
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
    let cohort = collect(first, &mut input, &planner, &command, &mut ledger, &mut bytes, 2, &mut || Ok(())).unwrap();
    assert_eq!(cohort.ids, ["a", "b"]); assert_eq!(cohort.first_request_seq, 1); assert!(!cohort.eof);
    assert_eq!(bytes, prefix.len() as u64); assert_eq!(input.position(), bytes);
    let first = read_record(&mut input, &command, &mut bytes).unwrap().unwrap();
    let tail = collect(first, &mut input, &planner, &command, &mut ledger, &mut bytes, 2, &mut || Ok(())).unwrap();
    assert_eq!(tail.ids, ["c"]); assert_eq!(tail.first_request_seq, 3); assert!(tail.eof); assert_eq!(ledger.records, 3);
}
#[test]
fn partial_tail_keeps_item_and_sample_identity_not_cohort_slot_identity() {
    let command = command(&[]); let planner = planner(&command);
    let mut input = Cursor::new(line("a") + &line("b")); let mut bytes = 0; let mut ledger = Ledger::default();
    let first = read_record(&mut input, &command, &mut bytes).unwrap().unwrap();
    let cohort = collect(first, &mut input, &planner, &command, &mut ledger, &mut bytes, 3, &mut || Ok(())).unwrap();
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
    assert!(collect(first, &mut input, &planner, &command, &mut ledger, &mut bytes, 3, &mut || Ok(())).is_err());
    assert_eq!(ledger.records, 1); // no refund/retry or terminal-success path
}
#[test]
fn reaching_record_ceiling_does_not_swallow_an_excess_record() {
    let command = command(&["--max-records", "1"]); let planner = planner(&command);
    let mut input = Cursor::new(line("a") + &line("b")); let mut bytes = 0; let mut ledger = Ledger::default();
    let first = read_record(&mut input, &command, &mut bytes).unwrap().unwrap();
    let cohort = collect(first, &mut input, &planner, &command, &mut ledger, &mut bytes, 3, &mut || Ok(())).unwrap();
    assert_eq!(cohort.ids, ["a"]); assert!(!cohort.eof);
    let excess = read_record(&mut input, &command, &mut bytes).unwrap().unwrap();
    assert!(collect(excess, &mut input, &planner, &command, &mut ledger, &mut bytes, 3, &mut || Ok(())).is_err());
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
    let command = command(&[]); let planner = planner(&command);
    let mut input = Cursor::new(line("a") + &line("b")); let mut bytes = 0; let mut ledger = Ledger::default();
    let first = read_record(&mut input, &command, &mut bytes).unwrap().unwrap(); let consumed = bytes;
    assert!(collect(first, &mut input, &planner, &command, &mut ledger, &mut bytes, 3,
        &mut || Err(CandidateError::Timeout)).is_err());
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
            &mut || Ok(())).unwrap();
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
