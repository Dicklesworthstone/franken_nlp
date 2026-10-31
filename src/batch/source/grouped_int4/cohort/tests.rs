//! Pinned source planning and cohort admission; not model parity evidence.
use super::*;
use super::super::tests::{args, budget, compiler, document, planner};
use crate::{native_engine::{decode::DecodeCancellationKind, portable_int8::ProjectionWork},
    tokenizer::pinned_controls};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn vocabulary() -> ExtractionVocabulary {
    let controls = pinned_controls::pinned().unwrap();
    ExtractionVocabulary::pinned(controls.template_controls()).unwrap()
}
fn work_limits() -> Int4SourceBatchLimits {
    Int4SourceBatchLimits {
        max_model_work: Int4Work { forward_positions: u64::MAX, projected_logits: u64::MAX,
            attention_pairs: u64::MAX, projections: ProjectionWork {
                dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
        masks: SourceMaskBudget { per_mask: crate::grammar::mask::MaskWorkLimits {
            max_trie_node_visits: 10000, checkpoint_interval_nodes: 64 },
            max_visits_per_item: 2_000_000, max_visits_per_run: u64::MAX },
    }
}
fn limits() -> Int4SourceCohortBatchLimits {
    let required = Int4MemoryRequirement::for_cohort(&[4096; 3], 32).unwrap();
    Int4SourceCohortBatchLimits { max_documents: 3, context_limit: 4096,
        native_memory: Int4MemoryBudget { max_kv_bytes: required.kv_bytes,
            max_rope_bytes: required.rope_bytes, max_scratch_payload_bytes: required.scratch_payload_bound },
        max_workspace_bytes: 1 << 24, max_item_result_bytes: 100_000, max_result_bytes: 1 << 20 }
}
fn row_work() -> Int4Work {
    Int4Work { forward_positions: 2, projected_logits: 3, attention_pairs: 4,
        projections: ProjectionWork { dot_products: 5, multiply_accumulates: 6 } }
}
fn axis(work: &mut Int4Work, index: usize) -> &mut u64 {
    match index { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits,
        2 => &mut work.attention_pairs, 3 => &mut work.projections.dot_products,
        4 => &mut work.projections.multiply_accumulates, _ => panic!("unknown work axis") }
}

#[test]
fn all_source_kinds_keep_serial_task_identity_and_exact_per_document_kv() {
    let planner = planner(); let vocabulary = vocabulary();
    for kind in [BuiltInTask::Ner, BuiltInTask::Keyphrases, BuiltInTask::Summarize, BuiltInTask::Answer] {
        let compiler = compiler(&planner, kind);
        let serial = compiler.prepare_with_control(document(), &mut Continue).unwrap();
        let mut account = Account::default();
        let prepared = prepare_document(&compiler, document(), &vocabulary, limits(), work_limits(),
            &mut account, &mut Continue).unwrap();
        prepared.verify_identity(serial.execution_identity()).unwrap();
        assert_eq!(prepared.task_budget(), budget());
        assert_eq!(prepared.planned_work(), serial.planned_work());
        let (context, kv) = price_plan(&prepared, limits()).unwrap();
        assert_eq!(context, prepared.prompt_tokens() + budget().max_output_tokens as usize - 1);
        assert_eq!(kv, context as u64 * KV_BYTES_PER_TOKEN as u64);
        assert!(kv < prepared.max_kv_bytes());
        assert!(kv < limits().native_memory.max_kv_bytes);
        assert_eq!(account.work, serial.planned_work());
        assert_eq!(account.masks, work_limits().masks.max_visits_per_item);
    }
}

#[test]
fn per_record_options_and_passages_stay_in_each_original_source_identity() {
    let planner = planner(); let vocabulary = vocabulary();
    for kind in [BuiltInTask::Keyphrases, BuiltInTask::Answer] {
        let compiler = compiler(&planner, kind); let mut account = Account::default();
        let first = prepare_document(&compiler, document(), &vocabulary, limits(), work_limits(),
            &mut account, &mut Continue).unwrap();
        let mut other = document(); other.text = "Different question or source: Bob visited 北京.".to_owned();
        let mut smaller = args(kind);
        match &mut smaller {
            SourceBatchArgs::Keyphrases { budget, options } => {
                budget.max_output_tokens = 64; options.max_phrases = 2;
            }
            SourceBatchArgs::Answer { budget, passages, .. } => {
                budget.max_output_tokens = 64;
                passages[0].text = "Distinct evidence belongs only to Bob.".to_owned();
            }
            _ => unreachable!(),
        }
        other.task_args = Some(smaller);
        let second = prepare_document(&compiler, other, &vocabulary, limits(), work_limits(),
            &mut account, &mut Continue).unwrap();
        assert_eq!(second.task_budget().max_output_tokens, 64);
        assert_eq!(first.task_budget(), budget());
        assert_ne!(first.execution_identity().prompt_digest, second.execution_identity().prompt_digest);
        assert!(first.verify_identity(second.execution_identity()).is_err());
        assert_eq!(second.planned_work().projected_logits, 64 * NANBEIGE_VOCAB_SIZE as u64);
        assert_eq!(account.work, first.planned_work().checked_add(second.planned_work()).unwrap());
    }
}

#[test]
fn rejected_source_records_preserve_pending_plans_and_the_nonrenewable_ledger() {
    let planner = planner(); let compiler = compiler(&planner, BuiltInTask::Keyphrases);
    let vocabulary = vocabulary(); let cap = work_limits(); let mut account = Account::default();
    let first = prepare_document(&compiler, document(), &vocabulary, limits(), cap,
        &mut account, &mut Continue).unwrap();
    let before = account;
    let mut wrong_task = document(); wrong_task.task_args = Some(args(BuiltInTask::Answer));
    let error = prepare_document(&compiler, wrong_task, &vocabulary, limits(), cap,
        &mut account, &mut Continue).err().unwrap();
    assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::Planning); assert_eq!(account, before);
    let mut insufficient = limits(); insufficient.max_item_result_bytes = 1;
    let error = prepare_document(&compiler, document(), &vocabulary, insufficient, cap,
        &mut account, &mut Continue).err().unwrap();
    assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::OutputLineLimit); assert_eq!(account, before);
    let mut impossible_mask = cap; impossible_mask.masks.max_visits_per_item = 1;
    let error = prepare_document(&compiler, document(), &vocabulary, limits(), impossible_mask,
        &mut account, &mut Continue).err().unwrap();
    assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::WorkLimit); assert_eq!(account, before);
    let second = prepare_document(&compiler, document(), &vocabulary, limits(), cap,
        &mut account, &mut Continue).unwrap();
    first.verify_identity(second.execution_identity()).unwrap();
    assert_eq!(account.work, before.work.checked_add(before.work).unwrap());
    assert_eq!(account.masks, 2 * before.masks);
}

#[test]
fn every_native_and_mask_counter_reserves_atomically_and_never_refunds() {
    let row = row_work(); let twice = row.checked_add(row).unwrap();
    for field in 0..6 {
        let mut cap = work_limits(); cap.max_model_work = twice;
        cap.masks.max_visits_per_run = 2 * cap.masks.max_visits_per_item;
        if field < 5 { *axis(&mut cap.max_model_work, field) -= 1; }
        else { cap.masks.max_visits_per_run -= 1; }
        let mut account = Account::default(); account.reserve(row, cap).unwrap();
        let before = account; let error = account.reserve(row, cap).unwrap_err();
        assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::WorkLimit); assert_eq!(account, before);
        let mut exhausted = before;
        if field < 5 { *axis(&mut exhausted.work, field) = u64::MAX; }
        else { exhausted.masks = u64::MAX; }
        let before = exhausted;
        assert_eq!(exhausted.reserve(row, work_limits()).unwrap_err().fault.code, BatchCode::WorkLimit);
        assert_eq!(exhausted, before);
    }
    let mut cap = work_limits(); cap.max_model_work = twice;
    cap.masks.max_visits_per_run = 2 * cap.masks.max_visits_per_item;
    let mut account = Account::default(); account.reserve(row, cap).unwrap(); account.reserve(row, cap).unwrap();
    let before = account;
    assert!(account.reserve(row, cap).is_err()); assert_eq!(account, before);
}

#[test]
fn cancellation_before_admission_preserves_cause_and_existing_work() {
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::CostBudget) }
    }
    let planner = planner(); let compiler = compiler(&planner, BuiltInTask::Answer);
    let vocabulary = vocabulary(); let mut account = Account::default();
    prepare_document(&compiler, document(), &vocabulary, limits(), work_limits(), &mut account, &mut Continue).unwrap();
    let before = account;
    let error = prepare_document(&compiler, document(), &vocabulary, limits(), work_limits(),
        &mut account, &mut Stop).err().unwrap();
    assert!(error.stop); assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::CostBudget));
    assert_eq!(account, before);
}

fn context(seq: u64, epoch: u64, line: u64, offset: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: seq, epoch, input_line: line, byte_offset: offset }
}
fn request(context: BatchRequestContext) -> GroupedRequest<()> { GroupedRequest { prepared: (), context } }

#[test]
fn ordered_source_windows_allow_rejected_row_gaps_but_refuse_replay_or_mixed_epochs() {
    let initial = context(1, 1, 1, 0);
    validate_contexts(&[request(initial)], None, 1).unwrap();
    let rows = [request(context(3, 1, 4, 80)), request(context(5, 1, 7, 200))];
    validate_contexts(&rows, Some(initial), 2).unwrap();
    let previous = rows[1].context;
    validate_contexts(&[request(context(7, 2, 10, 400))], Some(previous), 2).unwrap();
    assert!(validate_contexts::<()>(&[], None, 2).is_err());
    assert!(validate_contexts(&rows, Some(initial), 1).is_err());
    for changed in [context(5, 2, 8, 300), context(6, 2, 7, 300), context(6, 2, 8, 200),
        context(0, 2, 8, 300), context(6, 0, 8, 300), context(6, 2, 0, 300)] {
        assert!(validate_contexts(&[request(changed)], Some(previous), 2).is_err());
    }
    assert!(validate_contexts(&[request(rows[0].context), request(context(6, 2, 8, 300))], Some(initial), 2).is_err());
}

#[test]
fn physical_group_counters_accumulate_without_replacing_planned_work() {
    let actual = row_work(); let planned = actual.checked_add(actual).unwrap();
    let first = add_statistics(Int4SourceCohortBatchStats::default(), 3, 7, 3, planned, actual).unwrap();
    let second = add_statistics(first, 1, 4, 2, planned, actual).unwrap();
    assert_eq!((second.cohorts, second.documents, second.group_steps, second.projection_groups), (2, 4, 11, 5));
    assert_eq!(second.planned_work, planned.checked_add(planned).unwrap());
    assert_eq!(second.model_work, actual.checked_add(actual).unwrap());
    assert_ne!(second.planned_work, second.model_work);
    for field in 0..10 {
        let mut previous = first;
        let work = if field < 5 { &mut previous.planned_work } else { &mut previous.model_work };
        *axis(work, field % 5) = u64::MAX;
        assert!(add_statistics(previous, 1, 1, 1, planned, actual).is_err());
    }
}
