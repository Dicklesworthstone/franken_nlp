//! Pure complete-document job admission, without model or storage claims.
use super::*;
use std::time::Duration;
use crate::{jobs::JobWork, native_engine::lmhead::NANBEIGE_VOCAB_SIZE,
    tasks::{classify::ClassificationAggregationPolicy, mapreduce::ChunkLimits,
        sentiment::SentimentAggregationPolicy}, tokenizer::untrusted::SafeBpeLimits};

fn limits() -> JobHostLimits {
    JobHostLimits { native: NativeLimits { context_tokens: 1024, allocator_reserve_bytes: 65536,
        run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 10000,
            cleanup_reserve_bytes: 65536 } },
        transport: PopulationReadLimits { max_stream_bytes: 128 << 20, max_lines: 32 },
        preparation_reserve_bytes: 128 << 20, io_reserve_bytes: 65536,
        journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 2 << 20 }
}
fn task() -> TaskBudget {
    TaskBudget { max_input_tokens: 768, max_output_tokens: 16, max_output_bytes: 65536,
        max_grammar_states: 4096,
        max_kv_bytes: Int8MemoryRequirement::for_context(1024).unwrap().kv_bytes * 4 }
}
fn work() -> Int8Work { Int8Work::for_sequence(0, 4096, 64 * NANBEIGE_VOCAB_SIZE).unwrap() }
fn job() -> JobLimits {
    JobLimits { max_items: 4, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 64 << 20, max_result_bytes: 2 << 20,
        max_spool_bytes: 16 << 20, max_materialized_bytes: 16 << 20,
        max_journal_bytes: 1 << 20, max_attempts: 16,
        max_work: JobWork { model: work(), mask_node_visits: 0 } }
}
fn chunks() -> ChunkLimits {
    ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 512, max_chunk_tokens: 768,
        context_tokens: 1024, reserved_tokens: 16, max_chunks: 8, max_tokenizer_calls: 1024 }
}
fn classification() -> (ClassificationLimits, Int8ClassificationMapLimits) {
    (ClassificationLimits { max_labels: 4, max_input_bytes: 4096,
        max_total_label_bytes: 1024, max_context_tokens: 1024, ..Default::default() },
        Int8ClassificationMapLimits { chunks: chunks(), max_model_work: work(),
            max_retained_result_bytes: 1 << 20, max_result_bytes: 1 << 20,
            aggregation: ClassificationAggregationPolicy::default() })
}
fn sentiment() -> (SentimentLimits, Int8SentimentMapLimits) {
    (SentimentLimits::default(), Int8SentimentMapLimits { chunks: chunks(), max_model_work: work(),
        max_retained_result_bytes: 1 << 20, max_result_bytes: 1 << 20,
        aggregation: SentimentAggregationPolicy::default() })
}

#[test]
fn durable_population_journal_and_map_frontier_are_simultaneous_owners() {
    let (planning, mapping) = classification();
    let base = Int8ClassificationMapHostConfig { native: limits().native, planning, mapping,
        preparation_reserve_bytes: limits().preparation_reserve_bytes }.requirements(task()).unwrap();
    let (sentiment_planning, sentiment_mapping) = sentiment();
    let sentiment_base = Int8SentimentMapMemoryLimits { context_tokens: 1024,
        preparation_reserve_bytes: limits().preparation_reserve_bytes,
        allocator_reserve_bytes: limits().native.allocator_reserve_bytes, max_axes: 4 }
        .requirements(task(), sentiment_planning, sentiment_mapping).unwrap();
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        let buffers = limits().required_buffer_bytes_for_input(job(), input).unwrap();
        let required = int8_classification_document_job_requirements(task(), planning, mapping,
            &UntrustedEncoding::ByteFallback, limits(), job(), input).unwrap();
        assert_eq!(required.input_bytes, base.input_bytes + buffers
            + chunks().max_chunks as u64 * CLASSIFICATION_DOCUMENT_CHUNK_METADATA_BYTES);
        assert_eq!(required.native, base.native);
        assert_eq!(required.workspace_bytes, base.workspace_bytes);
        assert_eq!(required.output_reservation_bytes, base.output_reservation_bytes);
        let required = int8_sentiment_document_job_requirements(task(), sentiment_planning, sentiment_mapping,
            &UntrustedEncoding::ByteFallback, limits(), job(), input).unwrap();
        assert_eq!(required.input_bytes, sentiment_base.input_bytes + buffers
            + chunks().max_chunks as u64 * INT8_SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES);
        assert_eq!(required.native, sentiment_base.native);
        assert_eq!(required.workspace_bytes, sentiment_base.workspace_bytes);
        assert_eq!(required.output_reservation_bytes, sentiment_base.output_reservation_bytes);
    }
}

#[test]
fn retained_replay_discounts_only_population_storage() {
    let (planning, mapping) = classification();
    let a = int8_classification_document_job_requirements(task(), planning, mapping,
        &UntrustedEncoding::ByteFallback, limits(), job(), JobInputMemory::CompletePopulation).unwrap();
    let b = int8_classification_document_job_requirements(task(), planning, mapping,
        &UntrustedEncoding::ByteFallback, limits(), job(), JobInputMemory::RetainedRecord).unwrap();
    let difference = limits().required_buffer_bytes_for_input(job(), JobInputMemory::CompletePopulation).unwrap()
        - limits().required_buffer_bytes_for_input(job(), JobInputMemory::RetainedRecord).unwrap();
    assert!(difference > 0);
    assert_eq!(a.input_bytes - b.input_bytes, difference);
    assert_eq!(a.native, b.native); assert_eq!(a.workspace_bytes, b.workspace_bytes);
    assert_eq!(a.output_reservation_bytes, b.output_reservation_bytes);
    assert!(b.output_reservation_bytes > task().max_output_bytes * 8);
}

#[test]
fn compact_source_planning_and_packed_prompts_keep_separate_workspaces() {
    let (planning, mapping) = classification();
    let call = SafeBpeLimits { max_input_bytes: 262144, max_output_tokens: 131072,
        max_pair_checks: 1 << 25, max_pair_bytes: 1 << 28 };
    let byte = int8_classification_document_job_requirements(task(), planning, mapping,
        &UntrustedEncoding::ByteFallback, limits(), job(), JobInputMemory::RetainedRecord).unwrap();
    let compact = int8_classification_document_job_requirements(task(), planning, mapping,
        &UntrustedEncoding::SafeBpeV1(call), limits(), job(), JobInputMemory::RetainedRecord).unwrap();
    assert_eq!(compact.input_bytes - byte.input_bytes, call.requirements().unwrap().workspace_bytes);
    assert_eq!(compact.native, byte.native);
    assert_eq!(compact.output_reservation_bytes, byte.output_reservation_bytes);
    let prefill = Int8PrefillLimits { max_batch_rows: 4,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(4).unwrap() };
    let scheduled = compact.with_prefill(prefill).unwrap();
    assert_eq!(scheduled.input_bytes, compact.input_bytes);
    assert_eq!(scheduled.workspace_bytes - compact.workspace_bytes, prefill.validate().unwrap());
    assert!(dispatch::int8_prefill_team_workspace_bytes(4, 4).unwrap() > 0);
}

#[test]
fn complete_results_and_all_native_axes_must_fit_the_immutable_job() {
    let (cp, cm) = classification(); let (sp, sm) = sentiment();
    for axis in 0..7 {
        let mut selected = job();
        match axis {
            0 => selected.max_input_bytes_per_item = chunks().max_input_bytes - 1,
            1 => selected.max_result_bytes = cm.max_result_bytes as usize - 1,
            2 => selected.max_work.model.forward_positions -= 1,
            3 => selected.max_work.model.projected_logits -= 1,
            4 => selected.max_work.model.attention_pairs -= 1,
            5 => selected.max_work.model.projections.dot_products -= 1,
            _ => selected.max_work.model.projections.multiply_accumulates -= 1,
        }
        for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
            assert!(int8_classification_document_job_requirements(task(), cp, cm,
                &UntrustedEncoding::ByteFallback, limits(), selected, input).is_err(), "classification {axis}");
            assert!(int8_sentiment_document_job_requirements(task(), sp, sm,
                &UntrustedEncoding::ByteFallback, limits(), selected, input).is_err(), "sentiment {axis}");
        }
    }
}

#[test]
fn invalid_capacity_and_overflow_refuse_before_private_io() {
    let (planning, mapping) = classification();
    for axis in 0..6 {
        let mut selected = limits();
        match axis {
            0 => selected.journal_reserve_bytes = 0,
            1 => selected.serialization_reserve_bytes = 0,
            2 => selected.preparation_reserve_bytes = u64::MAX,
            3 => selected.io_reserve_bytes = u64::MAX,
            4 => selected.native.context_tokens = 1023,
            _ => selected.native.allocator_reserve_bytes = 0,
        }
        assert!(int8_classification_document_job_requirements(task(), planning, mapping,
            &UntrustedEncoding::ByteFallback, selected, job(), JobInputMemory::RetainedRecord).is_err());
    }
}

#[test]
fn document_factory_owners_cross_the_physical_runtime_boundary() {
    fn send_sync<T: Send + Sync + 'static>() {}
    send_sync::<Int8ClassificationDocumentJobPlanner>();
    send_sync::<Int8SentimentDocumentJobPlanner>();
    fn send<T: Send + 'static>() {}
    send::<ClassificationDocumentTask>(); send::<SentimentDocumentTask>();
}
