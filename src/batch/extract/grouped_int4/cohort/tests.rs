//! Model-free admission and ordering regressions; no native execution claim.
use super::*;
use super::super::tests::{boolean, document, planner};
use crate::native_engine::{decode::DecodeCancellationKind, portable_int8::ProjectionWork};

const WIDTH: usize = 127;

fn row_work() -> Int4Work {
    Int4Work { forward_positions: 2, projected_logits: 3, attention_pairs: 4,
        projections: ProjectionWork { dot_products: 5, multiply_accumulates: 6 } }
}
fn axis(work: &mut Int4Work, index: usize) -> &mut u64 {
    match index { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits,
        2 => &mut work.attention_pairs, 3 => &mut work.projections.dot_products,
        4 => &mut work.projections.multiply_accumulates, _ => panic!("unknown work axis") }
}
fn work_limits() -> Int4ExtractionBatchLimits {
    Int4ExtractionBatchLimits {
        max_model_work: Int4Work { forward_positions: u64::MAX, projected_logits: u64::MAX,
            attention_pairs: u64::MAX, projections: ProjectionWork {
                dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits {
            max_trie_node_visits: 1_000_000, checkpoint_interval_nodes: 64 },
            max_visits_per_item: 64_000_000, max_visits_per_run: u64::MAX },
    }
}
fn limits() -> Int4ExtractionCohortBatchLimits {
    let required = Int4MemoryRequirement::for_cohort(&[4096; 3], WIDTH).unwrap();
    Int4ExtractionCohortBatchLimits { max_documents: 3, context_limit: 4096,
        native_memory: Int4MemoryBudget { max_kv_bytes: required.kv_bytes,
            max_rope_bytes: required.rope_bytes, max_scratch_payload_bytes: required.scratch_payload_bound },
        max_workspace_bytes: 1 << 24, max_item_result_bytes: 100_000, max_result_bytes: 1 << 20 }
}
fn vocabulary() -> ExtractionVocabulary {
    let controls = pinned_controls::pinned().unwrap();
    ExtractionVocabulary::pinned(controls.template_controls()).unwrap()
}
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

#[test]
fn each_native_or_mask_ceiling_preserves_previously_reserved_documents_atomically() {
    let row = row_work(); let twice = row.checked_add(row).unwrap();
    for field in 0..6 {
        let mut cap = work_limits(); cap.max_model_work = twice;
        cap.masks.max_visits_per_run = 2 * cap.masks.max_visits_per_item;
        if field < 5 { *axis(&mut cap.max_model_work, field) -= 1; }
        else { cap.masks.max_visits_per_run -= 1; }
        let mut account = Account::default(); account.reserve(row, cap).unwrap();
        let before = account;
        let error = account.reserve(row, cap).unwrap_err();
        assert_eq!(error.fault.code, BatchCode::WorkLimit, "axis {field}");
        assert!(!error.stop); assert_eq!(account, before);
        assert_eq!(account.work, row); assert_eq!(account.masks, cap.masks.max_visits_per_item);
    }
    let mut cap = work_limits(); cap.max_model_work = twice;
    cap.masks.max_visits_per_run = 2 * cap.masks.max_visits_per_item;
    let mut account = Account::default();
    account.reserve(row, cap).unwrap(); account.reserve(row, cap).unwrap();
    let before = account;
    assert_eq!(account.reserve(row, cap).unwrap_err().fault.code, BatchCode::WorkLimit);
    assert_eq!(account, before); // Exhaustion cannot renew the preceding reservations.
}

#[test]
fn arithmetic_overflow_cannot_partially_debit_another_native_or_mask_axis() {
    for field in 0..6 {
        let mut account = Account::default();
        if field < 5 { *axis(&mut account.work, field) = u64::MAX; }
        else { account.masks = u64::MAX; }
        let before = account;
        let error = account.reserve(row_work(), work_limits()).unwrap_err();
        assert_eq!(error.fault.code, BatchCode::WorkLimit); assert!(!error.stop);
        assert_eq!(account, before, "axis {field}");
    }
}

#[test]
fn row_admission_prices_its_exact_planned_kv_and_keeps_task_and_output_caps() {
    let planner = planner(WIDTH, None); let prepared = planner.prepare(boolean()).unwrap();
    let limits = limits(); limits.validate(WIDTH).unwrap();
    let (context, bytes) = price_plan(&prepared, limits).unwrap();
    assert_eq!(context as u64, prepared.planned_work().forward_positions);
    assert_eq!(context, prepared.plan.prompt_tokens() + prepared.plan.options().max_new_tokens - 1);
    assert_eq!(bytes, context as u64 * KV_BYTES_PER_TOKEN as u64);
    assert!(context < limits.context_limit);
    assert!(bytes < prepared.plan.max_kv_bytes());
    assert!(bytes < limits.native_memory.max_kv_bytes);
    for field in 0..3 {
        let mut cap = limits;
        match field { 0 => cap.context_limit = context - 1,
            1 => cap.native_memory.max_kv_bytes = bytes - 1,
            _ => cap.max_item_result_bytes = prepared.plan.max_result_bytes() - 1 }
        let error = price_plan(&prepared, cap).unwrap_err();
        let expected = if field == 2 { BatchCode::OutputLineLimit } else { BatchCode::Admission };
        assert_eq!(error.fault.code, expected); assert!(!error.stop);
    }
}

#[test]
fn impossible_row_mask_budget_is_rejected_before_charging_or_disturbing_pending_work() {
    let planner = planner(WIDTH, None); let vocabulary = vocabulary();
    let row = planner.prepare(boolean()).unwrap().planned_work();
    let mut cap = work_limits(); cap.max_model_work = row.checked_add(row).unwrap();
    cap.masks.max_visits_per_run = 2 * cap.masks.max_visits_per_item;
    let mut account = Account::default();
    let first = prepare_document(&planner, boolean(), &vocabulary, limits(), cap,
        &mut account, &mut Continue).unwrap();
    let before = account;
    let mut impossible = cap; impossible.masks.max_visits_per_item = 1;
    let error = prepare_document(&planner, boolean(), &vocabulary, limits(), impossible,
        &mut account, &mut Continue).err().unwrap();
    assert_eq!(error.fault.code, BatchCode::WorkLimit); assert!(!error.stop);
    assert_eq!(account, before);

    // A rejected row cannot discard or modify the sealed work waiting before it.
    let second = prepare_document(&planner, boolean(), &vocabulary, limits(), cap,
        &mut account, &mut Continue).unwrap();
    assert_eq!(first.execution_identity(), second.execution_identity());
    assert_eq!(first.planned_work(), second.planned_work());
    assert_eq!(account.work, cap.max_model_work);
    assert_eq!(account.masks, cap.masks.max_visits_per_run);
    let before = account;
    let error = prepare_document(&planner, boolean(), &vocabulary, limits(), cap,
        &mut account, &mut Continue).err().unwrap();
    assert_eq!(error.fault.code, BatchCode::WorkLimit); assert!(!error.stop);
    assert_eq!(account, before);
}

#[test]
fn malformed_schema_and_insufficient_output_admission_never_reserve_model_work() {
    let planner = planner(WIDTH, None); let vocabulary = vocabulary();
    let mut account = Account::default();
    let invalid = document(r#"{"type":"boolean","type":"string"}"#,
        "PRIVATE_SOURCE", ExtractionBatchGrounding::Structural);
    let error = prepare_document(&planner, invalid, &vocabulary, limits(), work_limits(),
        &mut account, &mut Continue).err().unwrap();
    assert!(!error.stop); assert_eq!(account, Account::default());
    let mut insufficient = limits(); insufficient.max_item_result_bytes = 1;
    let error = prepare_document(&planner, boolean(), &vocabulary, insufficient, work_limits(),
        &mut account, &mut Continue).err().unwrap();
    assert_eq!(error.fault.code, BatchCode::OutputLineLimit); assert!(!error.stop);
    assert_eq!(account, Account::default());
}

#[test]
fn cancellation_keeps_the_original_cause_and_does_not_refund_completed_reservations() {
    struct Stop { calls: usize, at: usize }
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            let call = self.calls; self.calls += 1;
            (call == self.at).then_some(DecodeCancellationKind::CostBudget)
        }
    }
    let planner = planner(WIDTH, None); let vocabulary = vocabulary();
    let row = planner.prepare(boolean()).unwrap().planned_work(); let cap = work_limits();
    for at in 0..3 {
        let mut account = Account::default(); let mut control = Stop { calls: 0, at };
        let error = prepare_document(&planner, boolean(), &vocabulary, limits(), cap,
            &mut account, &mut control).err().unwrap();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Cancelled);
        assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::CostBudget));
        assert_eq!(control.calls, at + 1);
        let expected = if at == 2 { Account { work: row, masks: cap.masks.max_visits_per_item } }
            else { Account::default() };
        assert_eq!(account, expected, "checkpoint {at}");
    }
}

fn context(sequence: u64, epoch: u64, line: u64, offset: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: sequence, epoch, input_line: line, byte_offset: offset }
}
fn request(context: BatchRequestContext) -> GroupedRequest<()> { GroupedRequest { prepared: (), context } }

#[test]
fn delivery_coordinates_accept_gaps_but_never_replays_or_mixed_epochs() {
    let initial = context(1, 1, 1, 0);
    validate_contexts(&[request(initial)], None, 1).unwrap(); // The first line starts at byte zero.
    let rows = [request(context(3, 1, 4, 80)), request(context(5, 1, 7, 200))];
    validate_contexts(&rows, Some(initial), 2).unwrap(); // Errors may occupy intervening positions.
    let previous = rows[1].context;
    validate_contexts(&[request(context(7, 2, 10, 400))], Some(previous), 2).unwrap();
    assert_eq!(validate_contexts::<()>(&[], None, 2).unwrap_err().code, BatchCode::InvalidExecution);
    assert!(validate_contexts(&rows, Some(initial), 1).is_err());
    for changed in [context(6, 0, 8, 300), context(0, 2, 8, 300), context(6, 2, 0, 300),
        context(5, 2, 8, 300), context(6, 2, 7, 300), context(6, 2, 8, 200)] {
        assert_eq!(validate_contexts(&[request(changed)], Some(previous), 2).unwrap_err().code,
            BatchCode::InvalidExecution);
    }
    let prior_epoch = context(7, 2, 10, 400);
    assert!(validate_contexts(&[request(context(8, 1, 11, 500))], Some(prior_epoch), 2).is_err());
    for changed in [context(6, 2, 8, 300), context(3, 1, 8, 300),
        context(6, 1, 4, 300), context(6, 1, 8, 80)] {
        assert!(validate_contexts(&[request(rows[0].context), request(changed)], Some(initial), 2).is_err());
    }
}

#[test]
fn statistics_accumulate_actual_work_separately_from_planned_reservations() {
    let actual = row_work(); let planned = actual.checked_add(actual).unwrap();
    let first = add_statistics(Int4ExtractionCohortBatchStats::default(), 3, 7, 3, planned, actual).unwrap();
    let second = add_statistics(first, 1, 4, 2, planned, actual).unwrap();
    assert_eq!(second.cohorts, 2); assert_eq!(second.documents, 4);
    assert_eq!(second.group_steps, 11); assert_eq!(second.projection_groups, 5);
    assert_eq!(second.planned_work, planned.checked_add(planned).unwrap());
    assert_eq!(second.model_work, actual.checked_add(actual).unwrap());
    assert_ne!(second.model_work, second.planned_work);
    assert_eq!(first.cohorts, 1); assert_eq!(first.model_work, actual);
}

#[test]
fn statistics_refuse_overflow_in_every_counter_and_native_work_axis() {
    for field in 0..4 {
        let mut previous = Int4ExtractionCohortBatchStats::default();
        match field { 0 => previous.cohorts = u64::MAX, 1 => previous.documents = u64::MAX,
            2 => previous.group_steps = u64::MAX, _ => previous.projection_groups = u64::MAX }
        let before = previous;
        let error = add_statistics(previous, 1, 1, 1, row_work(), row_work()).unwrap_err();
        assert_eq!(error.code, BatchCode::SequenceOverflow); assert_eq!(previous, before);
    }
    for field in 0..10 {
        let mut previous = Int4ExtractionCohortBatchStats::default();
        let work = if field < 5 { &mut previous.planned_work } else { &mut previous.model_work };
        *axis(work, field % 5) = u64::MAX;
        let before = previous;
        assert!(add_statistics(previous, 1, 1, 1, row_work(), row_work()).is_err());
        assert_eq!(previous, before);
    }
}
