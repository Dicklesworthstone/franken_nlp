use super::*;

fn setup(chunks: usize, labels: usize)
    -> (TaskBudget, ClassificationLimits, Int4ClassificationMapLimits, CorpusLimits,
        Int4ClassificationMapCohortMemoryRequirement, DocumentCapacity) {
    let (mut task, mut planning, mapping, limits) = fixture();
    task.max_kv_bytes *= labels as u64;
    planning.max_labels = labels;
    let quote = int4_classification_document_corpus_cohort_requirements(32, task, planning, mapping, limits, chunks).unwrap();
    let capacity = DocumentCapacity { context: limits.native.context_tokens, kv_bytes: quote.memory.native.kv_bytes,
        mapping, max_heads: cohort_head_limit(labels, chunks).unwrap(), cohort_chunks: Some(chunks),
        max_result_bytes: quote.memory.max_output_bytes };
    (task, planning, mapping, limits, quote, capacity)
}

#[test]
fn corpus_cohorts_share_the_complete_native_quote_without_adding_a_serial_cache() {
    for width in [3, 32, 128, 10752] {
        for (chunks, labels) in [(1, 1), (2, 2), (3, 4), (8, 16)] {
            let (task, planning, mapping, limits, _, _) = setup(chunks, labels);
            let heads = cohort_head_limit(labels, chunks).unwrap();
            let base = Int4ClassificationMapMemoryLimits { context_tokens: limits.native.context_tokens,
                preparation_reserve_bytes: limits.preparation_reserve_bytes,
                allocator_reserve_bytes: limits.native.allocator_reserve_bytes }
                .cohort_requirements(width, task, planning, mapping, heads, chunks).unwrap();
            let actual = int4_classification_document_corpus_cohort_requirements(width, task, planning, mapping, limits, chunks).unwrap();
            assert_eq!(actual.memory.native, base.memory.native);
            assert_eq!(actual.memory.workspace_bytes, base.memory.workspace_bytes);
            assert_eq!(actual.task_workspace_bytes, base.task_workspace_bytes);
            assert_eq!(actual.max_cohort_output_bytes, base.max_cohort_output_bytes);
            assert_eq!(actual.memory.max_output_bytes, base.memory.max_output_bytes);
            assert_eq!(actual.memory.output_reservation_bytes, base.memory.output_reservation_bytes);
            assert_eq!(actual.memory.input_bytes, base.memory.input_bytes + limits.reservation_bytes().unwrap()
                + mapping.chunks.max_chunks as u64 * CLASSIFICATION_DOCUMENT_CHUNK_METADATA_BYTES);
        }
    }
}

#[test]
fn a_whole_window_and_one_chunks_head_bundle_have_separate_kv_authority() {
    let (task, _, _, _, quote, capacity) = setup(2, 2);
    let identities = [identity(), identity()];
    let mut request = request(&identities);
    request.task_budget = task; request.head_count = 2; request.cohort_chunks = Some(2);
    request.kv_reservation_bytes = quote.memory.native.kv_bytes;
    request.max_result_bytes = quote.memory.max_output_bytes;
    assert_eq!(request.kv_reservation_bytes, 2 * task.max_kv_bytes);
    assert!(request.max_result_bytes > capacity.mapping.max_result_bytes);
    check_request(32, capacity, &request).unwrap();
    request.task_budget.max_kv_bytes -= 1;
    assert_eq!(check_request(32, capacity, &request).unwrap_err().fault.code, BatchCode::Admission);
    request.head_count = 1;
    check_request(32, capacity, &request).unwrap();
}

#[test]
fn admission_refuses_changed_schedule_geometry_peak_storage_and_complete_result_caps() {
    let (task, _, _, _, quote, capacity) = setup(2, 2);
    let identities = [identity(), identity()];
    for fault in 0..9 {
        let mut request = request(&identities);
        request.task_budget = task; request.head_count = 2; request.cohort_chunks = Some(2);
        request.kv_reservation_bytes = quote.memory.native.kv_bytes;
        request.max_result_bytes = quote.memory.max_output_bytes;
        match fault { 0 => request.cohort_chunks = None, 1 => request.cohort_chunks = Some(1),
            2 => request.head_count = 0, 3 => request.head_count = 3,
            4 => request.kv_reservation_bytes -= 1, 5 => request.required_context = 1025,
            6 => request.max_result_bytes = capacity.mapping.max_result_bytes,
            7 => request.max_result_bytes += 1, _ => request.group_width = 128 }
        assert!(check_request(32, capacity, &request).unwrap_err().stop, "fault {fault}");
    }
}

#[test]
fn whole_cohort_wrapper_and_complete_peak_are_refused_before_any_input_io() {
    let (task, planning, mapping, limits, quote, _) = setup(2, 2);
    let mut short = limits;
    short.transport.max_output_line_bytes = quote.memory.max_output_bytes as usize - 1;
    assert!(int4_classification_document_corpus_cohort_requirements(32, task, planning, mapping, short, 2).is_err());
    short.transport.max_output_line_bytes += 1;
    int4_classification_document_corpus_cohort_requirements(32, task, planning, mapping, short, 2).unwrap();
    let mut wrong_context = limits; wrong_context.native.context_tokens += 1;
    assert!(int4_classification_document_corpus_cohort_requirements(32, task, planning, mapping, wrong_context, 2).is_err());
    let mut short_task = task; short_task.max_kv_bytes -= 1;
    assert!(int4_classification_document_corpus_cohort_requirements(32, short_task, planning, mapping, limits, 2).is_err());
    for chunks in [0, 9, 65, usize::MAX] {
        assert!(int4_classification_document_corpus_cohort_requirements(32, task, planning, mapping, limits, chunks).is_err());
    }
}
