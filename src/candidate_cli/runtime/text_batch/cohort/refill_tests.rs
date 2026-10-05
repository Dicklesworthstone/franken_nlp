//! Pinned planning and bounded input/output admission, without model loading.
use super::*;
use std::io::Cursor;
fn command(active: &str) -> TextBatchCommand {
    TextBatchCommand::from_arg_matches(&wire::definition().try_get_matches_from(["text-batch",
        "--model", "local.fnlpq", "--memory-mib", "8192", "--cohort-rows", "4", "--active-rows", active]).unwrap()).unwrap()
}
fn planner(command: &TextBatchCommand) -> Planner {
    let facts = ArtifactIdentity { model_id: "Nanbeige4.2-3B".into(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".into(), recipe_id: "metadata-only-unit-fixture".into(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) };
    Planner::new(&command.common, command.validate().unwrap(), &facts).unwrap()
}
fn line(id: &str) -> String { format!("{{\"id\":\"{id}\",\"sample_index\":7,\"prompt\":\"hello\"}}\n") }
#[test]
fn read_ahead_is_the_admitted_queue_not_the_smaller_live_slot_count() {
    let command = command("1"); let planner = planner(&command);
    let prefix = line("a") + &line("b") + &line("c") + &line("d");
    let mut input = Cursor::new(prefix.clone() + &line("e")); let mut bytes = 0; let mut ledger = Ledger::default();
    let first = read_record(&mut input, &command, &mut bytes).unwrap().unwrap();
    let window = collect(first, &mut input, &planner, &command, &mut ledger, &mut bytes, 4, &mut || Ok(())).unwrap();
    assert_eq!(window.ids, ["a", "b", "c", "d"]); assert_eq!(ledger.records, 4);
    assert_eq!(bytes, prefix.len() as u64); assert_eq!(input.position(), bytes);
    assert_eq!(command.refill_strategy(window.ids.len()).unwrap().unwrap().0, 1);
    let expected: u64 = window.prepared.iter().map(|plan| plan.planned_work().forward_positions).sum();
    assert_eq!(ledger.work.forward_positions, expected);
}
#[test]
fn partial_window_keeps_full_item_sample_and_corpus_work_identity() {
    let command = command("3"); let planner = planner(&command);
    let mut input = Cursor::new(line("a") + &line("b")); let mut bytes = 0; let mut ledger = Ledger::default();
    let first = read_record(&mut input, &command, &mut bytes).unwrap().unwrap();
    let window = collect(first, &mut input, &planner, &command, &mut ledger, &mut bytes, 4, &mut || Ok(())).unwrap();
    assert!(window.eof); assert_eq!(window.ids.len(), 2);
    let (active, tokens) = command.refill_strategy(2).unwrap().unwrap(); assert_eq!(active, 2); assert_eq!(tokens.max_batch_rows, 2);
    for (id, plan) in window.ids.iter().zip(&window.prepared) {
        let record = wire::parse_record(command.task, &line(id), command.common.max_input_bytes).unwrap();
        let (_, individual) = planner.prepare(record).unwrap();
        assert_eq!(plan.execution_identity(), individual.execution_identity());
    }
    assert_eq!(ledger.records, 2);
}
#[test]
fn one_live_slot_does_not_admit_only_one_frames_output_budget() {
    let command = command("1"); let mut bytes = Vec::new();
    let output = Output::new(&mut bytes, FOOTER_BYTES as u64 + 299);
    assert_eq!(command.refill_strategy(4).unwrap().unwrap().0, 1);
    assert!(output.admit_frame(100).is_ok());
    assert!(output.admit_frame(delivery_bytes(4, 100).unwrap()).is_err()); assert_eq!(output.written, 0);
}
