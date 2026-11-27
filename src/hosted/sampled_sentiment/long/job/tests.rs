//! Pure complete-document durable admission. Native execution remains gated.
use super::*;
use crate::{
    batch::{extract::ExtractionMaskBudget,
        sentiment::sampled::quantized::long::int8_sampled_sentiment_document_job_preparation_bytes},
    grammar::mask::MaskWorkLimits,
    jobs::population::PopulationReadLimits,
    native_engine::portable_int8::ProjectionWork,
    tasks::mapreduce::ChunkLimits,
    tokenizer::untrusted::SafeBpeLimits,
};
use std::time::Duration;

fn task() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 1024, max_output_bytes: 65536,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn planning() -> SampledSentimentLimits {
    SampledSentimentLimits { max_input_bytes: 1024, max_result_bytes: 65536, ..Default::default() }
}
fn options() -> SampledSentimentOptions {
    SampledSentimentOptions { effective_seed: [41; 32], samples: 2, temperature_milli: 1000,
        top_k: None, top_p_ppm: 1_000_000, max_sample_tokens: 128, max_justification_chars: 64 }
}
fn work() -> Int8SampledSentimentBatchLimits {
    let work = Int8Work { forward_positions: u64::MAX / 8, projected_logits: u64::MAX / 8,
        attention_pairs: u64::MAX / 8, projections: ProjectionWork {
            dot_products: u64::MAX / 8, multiply_accumulates: u64::MAX / 8 } };
    Int8SampledSentimentBatchLimits { max_item_work: work, max_model_work: work,
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: 1000,
            checkpoint_interval_nodes: 64 }, max_visits_per_item: 1_000_000, max_visits_per_run: 4_000_000 },
        max_sampler_bytes: json_sampling_workspace_bytes().unwrap() }
}
fn mapping() -> Int8SampledSentimentMapLimits {
    Int8SampledSentimentMapLimits { chunks: ChunkLimits { max_input_bytes: 8192, max_chunk_bytes: 1024,
        max_chunk_tokens: 1024, context_tokens: 4096, reserved_tokens: 128,
        max_chunks: 16, max_tokenizer_calls: 4096 }, max_model_work: work().max_item_work,
        max_retained_result_bytes: 4 << 20, max_result_bytes: 8 << 20 }
}
fn limits() -> JobLimits {
    JobLimits { max_items: 4, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 1 << 20, max_result_bytes: 8 << 20, max_spool_bytes: 64 << 20,
        max_materialized_bytes: 64 << 20, max_journal_bytes: 8 << 20, max_attempts: 8,
        max_work: JobWork { model: work().max_model_work, mask_node_visits: work().masks.max_visits_per_run } }
}
fn host(encoding: &UntrustedEncoding) -> JobHostLimits {
    let preparation = int8_sampled_sentiment_document_job_preparation_bytes(task(), planning(), mapping(),
        &options(), encoding).unwrap();
    JobHostLimits { native: NativeLimits { context_tokens: 4096, allocator_reserve_bytes: 64 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(60), max_checkpoints: 1_000_000,
            cleanup_reserve_bytes: 1 << 20 } },
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 4 },
        preparation_reserve_bytes: preparation, io_reserve_bytes: 128 << 10,
        journal_reserve_bytes: 64 << 20, serialization_reserve_bytes: 64 << 20 }
}
fn quote(encoding: &UntrustedEncoding, input: JobInputMemory, prefill: Option<Int8PrefillLimits>,
    threads: Option<usize>) -> Int8SampledSentimentDocumentJobRequirements {
    int8_sampled_sentiment_document_job_requirements(task(), planning(), mapping(), &options(), encoding,
        work(), limits(), host(encoding), input, prefill, threads).unwrap()
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}

#[test]
fn complete_output_and_retained_record_memory_have_separate_explicit_authority() {
    let encoding = UntrustedEncoding::ByteFallback;
    let complete = quote(&encoding, JobInputMemory::CompletePopulation, None, None);
    let retained = quote(&encoding, JobInputMemory::RetainedRecord, None, None);
    assert_eq!(complete.document.max_output_bytes, mapping().max_result_bytes);
    assert!(complete.document.max_output_bytes > task().max_output_bytes);
    assert_eq!(complete.document.max_output_tokens, 16 * 4 * 2 * 128);
    assert_eq!(complete.document.output_reservation_bytes,
        mapping().max_result_bytes * 4 + complete.document.max_output_tokens * 8 + 4096);
    assert_eq!(complete.document.input_bytes,
        host(&encoding).required_buffer_bytes_for_input(limits(), JobInputMemory::CompletePopulation).unwrap());
    assert_eq!(retained.document.input_bytes,
        host(&encoding).required_buffer_bytes_for_input(limits(), JobInputMemory::RetainedRecord).unwrap());
    assert_eq!(retained.document.native, complete.document.native);
    assert_eq!(retained.document.workspace_bytes, complete.document.workspace_bytes);
    assert_eq!(retained.document.output_reservation_bytes, complete.document.output_reservation_bytes);
}

#[test]
fn compact_encoding_and_composed_prefill_teams_are_each_charged_once() {
    let encoding = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 65536,
        max_output_tokens: 65536, max_pair_checks: 4_000_000, max_pair_bytes: 64_000_000 });
    let plain = quote(&UntrustedEncoding::ByteFallback, JobInputMemory::RetainedRecord, None, None);
    let compact = quote(&encoding, JobInputMemory::RetainedRecord, None, None);
    let UntrustedEncoding::SafeBpeV1(policy) = &encoding else { unreachable!() };
    assert_eq!(compact.document.input_bytes - plain.document.input_bytes, policy.requirements().unwrap().workspace_bytes);
    for rows in [1, 4, 64] {
        let layered = quote(&encoding, JobInputMemory::RetainedRecord, Some(prefill(rows)), None);
        assert_eq!(layered.document.workspace_bytes, compact.document.workspace_bytes + prefill(rows).max_extra_scratch_bytes);
        assert_eq!(layered.document.input_bytes, compact.document.input_bytes);
        for width in [2, 4, 64] {
            let team = quote(&encoding, JobInputMemory::RetainedRecord, Some(prefill(rows)), Some(width));
            assert_eq!(team.document, layered.document);
            assert_eq!(team.team_workspace_bytes, dispatch::int8_prefill_team_workspace_bytes(width, rows).unwrap());
            assert_eq!(team.total_bytes().unwrap(), layered.total_bytes().unwrap() + team.team_workspace_bytes);
        }
    }
}

#[test]
fn every_lifetime_work_axis_is_exact_before_any_job_storage_opens() {
    let encoding = UntrustedEncoding::ByteFallback;
    for axis in 0..6 {
        let mut job = limits();
        match axis {
            0 => job.max_work.model.forward_positions -= 1,
            1 => job.max_work.model.projected_logits -= 1,
            2 => job.max_work.model.attention_pairs -= 1,
            3 => job.max_work.model.projections.dot_products -= 1,
            4 => job.max_work.model.projections.multiply_accumulates -= 1,
            _ => job.max_work.mask_node_visits -= 1,
        }
        assert!(int8_sampled_sentiment_document_job_requirements(task(), planning(), mapping(),
            &options(), &encoding, work(), job, host(&encoding), JobInputMemory::RetainedRecord,
            None, None).is_err(), "lifetime work axis {axis}");
    }
}

#[test]
fn preparation_output_source_and_schedule_underquotes_refuse_before_io() {
    let encoding = UntrustedEncoding::ByteFallback;
    for axis in 0..10 {
        let mut host = host(&encoding); let mut job = limits(); let mut map = mapping();
        let mut chunk = planning(); let mut prefill_limit = None; let mut threads = None;
        match axis {
            0 => host.preparation_reserve_bytes -= 1,
            1 => job.max_result_bytes -= 1,
            2 => job.max_input_bytes_per_item -= 1,
            3 => map.max_model_work.attention_pairs -= 1,
            4 => job.max_id_bytes = 257,
            5 => chunk.max_input_bytes = job.max_input_bytes_per_item + 1,
            6 => { job.max_result_bytes = 32768; map.max_result_bytes = 32768; },
            7 => prefill_limit = Some(Int8PrefillLimits { max_batch_rows: 0, max_extra_scratch_bytes: u64::MAX }),
            8 => prefill_limit = Some(Int8PrefillLimits { max_extra_scratch_bytes: prefill(4).max_extra_scratch_bytes - 1,
                ..prefill(4) }),
            _ => threads = Some(1),
        }
        assert!(int8_sampled_sentiment_document_job_requirements(task(), chunk, map, &options(),
            &encoding, work(), job, host, JobInputMemory::CompletePopulation, prefill_limit, threads).is_err(),
            "admission relation {axis}");
    }
}

#[test]
fn owned_document_job_methods_and_claims_cross_the_same_dispatch_boundary() {
    use std::io::Cursor;
    fn send<T: Send + 'static>() {}
    send::<JobExecution<Cursor<Vec<u8>>>>();
    let _serial = NlpEngine::job_int8_sampled_sentiment_documents::<Cursor<Vec<u8>>>;
    let _team = NlpEngine::job_int8_sampled_sentiment_documents_with_threads::<Cursor<Vec<u8>>>;
}
