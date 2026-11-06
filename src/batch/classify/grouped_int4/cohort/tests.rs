//! Pure geometry and memory authority; these fixtures never fabricate scores.
use super::*;
use super::super::tests::{args, document, mapping, planning, task, work};
use crate::native_engine::strict_int4::Int4MemoryBudget;

fn memory() -> Int4ClassificationMapMemoryLimits {
    Int4ClassificationMapMemoryLimits { context_tokens: 1024,
        preparation_reserve_bytes: 128 * 1024 * 1024, allocator_reserve_bytes: 65536 }
}
fn ceiling(labels: usize) -> TaskBudget {
    TaskBudget { max_kv_bytes: task().max_kv_bytes * labels as u64, ..task() }
}
fn native_budget(schedule: CohortSchedule) -> Int4ClassificationMapCohortBudget {
    let native = schedule.required.memory.native;
    Int4ClassificationMapCohortBudget { native: Int4RunBudget::exact(mapping().max_model_work),
        memory: Int4MemoryBudget { max_kv_bytes: native.kv_bytes, max_rope_bytes: native.rope_bytes,
            max_scratch_payload_bytes: native.scratch_payload_bound },
        max_task_workspace_bytes: schedule.required.task_workspace_bytes,
        max_output_bytes: schedule.required.max_cohort_output_bytes }
}

#[test]
fn explicit_label_authority_quotes_four_rows_and_rejects_every_third_label_override() {
    let planning = ClassificationLimits { max_labels: 2, ..planning() };
    let task = ceiling(2);
    let schedule = CohortSchedule::new(32, task, planning, mapping(), 2, memory()).unwrap();
    assert_eq!(schedule.max_heads, 2);
    assert_eq!(schedule.required.memory.native, Int4MemoryRequirement::for_cohort(&[1024; 4], 32).unwrap());
    assert_eq!(schedule.required.memory.native.kv_bytes, 2 * task.max_kv_bytes);
    for mode in [ClassificationMode::Exclusive, ClassificationMode::MultiLabel] {
        let mut valid = args(); valid.mode = mode; valid.budget = task;
        let request = build_request(document(Some(valid.clone())), &None, mapping(), task, planning).unwrap();
        schedule.check_request(request.mode, request.labels.len(), request.budget).unwrap();
        valid.labels.push(ClassificationLabel { id: "third".to_owned(), description: "another class".to_owned() });
        let failure = build_request(document(Some(valid)), &None, mapping(), task, planning).err().unwrap();
        assert_eq!(failure.fault.code, BatchCode::Planning);
        assert!(!failure.stop, "an unadmitted record never spends a native attempt");
    }
}

#[test]
fn all_declared_windows_are_bounded_and_actual_head_products_never_silently_shrink() {
    for chunks in [1, 2, 3, 16, 32, 64] {
        let schedule = CohortSchedule::new(32, ceiling(4), planning(), mapping(), chunks, memory()).unwrap();
        assert_eq!(schedule.max_heads, 4.min(64 / chunks));
        assert_eq!(schedule.chunks, chunks);
        schedule.check_request(ClassificationMode::Exclusive, 4, task()).unwrap();
        schedule.check_request(ClassificationMode::MultiLabel, schedule.max_heads, ceiling(schedule.max_heads)).unwrap();
        let error = schedule.check_request(ClassificationMode::MultiLabel, schedule.max_heads + 1, ceiling(4)).unwrap_err();
        assert_eq!(error.fault.code, BatchCode::Planning);
        assert!(!error.stop);
    }
    for chunks in [0, 65, usize::MAX] {
        assert!(CohortSchedule::new(32, ceiling(4), planning(), mapping(), chunks, memory()).is_err());
    }
    let mut too_short = mapping(); too_short.chunks.max_chunks = 1;
    assert!(CohortSchedule::new(32, ceiling(4), planning(), too_short, 2, memory()).is_err());
    let mut wrong_context = memory(); wrong_context.context_tokens += 1;
    assert!(CohortSchedule::new(32, ceiling(4), planning(), mapping(), 2, wrong_context).is_err());
}

#[test]
fn a_chunk_budget_covers_its_heads_without_claiming_the_whole_window_allocation() {
    let schedule = CohortSchedule::new(32, ceiling(4), planning(), mapping(), 8, memory()).unwrap();
    let task = ceiling(2);
    assert!(task.max_kv_bytes < schedule.required.memory.native.kv_bytes);
    schedule.check_request(ClassificationMode::MultiLabel, 2, task).unwrap();
    let short = TaskBudget { max_kv_bytes: task.max_kv_bytes - 1, ..task };
    assert_eq!(schedule.check_request(ClassificationMode::MultiLabel, 2, short).unwrap_err().fault.code, BatchCode::Admission);
    assert_eq!(schedule.check_request(ClassificationMode::Exclusive, 0, task).unwrap_err().fault.code, BatchCode::Planning);
    let short_ceiling = TaskBudget { max_kv_bytes: ceiling(4).max_kv_bytes - 1, ..ceiling(4) };
    assert!(CohortSchedule::new(32, short_ceiling, planning(), mapping(), 8, memory()).is_err());
}

#[test]
fn native_window_capacity_and_result_frontier_are_admitted_independently_of_work() {
    let schedule = CohortSchedule::new(32, ceiling(4), planning(), mapping(), 2, memory()).unwrap();
    let admitted = native_budget(schedule);
    check_cohort_budget(schedule.required, work(4), admitted).unwrap();
    for axis in 0..10 {
        let mut short = admitted;
        match axis { 0 => short.memory.max_kv_bytes -= 1, 1 => short.memory.max_rope_bytes -= 1,
            2 => short.memory.max_scratch_payload_bytes -= 1, 3 => short.max_task_workspace_bytes -= 1,
            4 => short.max_output_bytes -= 1, 5 => short.max_output_bytes += 1,
            6 => short.native.max_forward_positions -= 1, 7 => short.native.max_attention_pairs -= 1,
            8 => short.native.max_projection_work.dot_products -= 1,
            _ => short.native.max_projection_work.multiply_accumulates -= 1 }
        let failure = check_cohort_budget(schedule.required, work(4), short).unwrap_err();
        assert!(failure.stop); assert_eq!(failure.fault.code, BatchCode::Admission, "axis {axis}");
    }
    let serial = memory().requirements(32, ceiling(4), planning(), mapping()).unwrap();
    assert!(schedule.required.memory.input_bytes > serial.input_bytes);
    assert!(schedule.required.memory.max_output_bytes > serial.max_output_bytes);
    assert!(schedule.required.memory.output_reservation_bytes > serial.output_reservation_bytes);
}
