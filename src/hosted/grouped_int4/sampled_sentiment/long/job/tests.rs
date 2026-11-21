use super::*;
use crate::{
    batch::{extract::ExtractionMaskBudget, sentiment::sampled::long::sampled_sentiment_document_preparation_bytes},
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
fn work() -> SampledSentimentBatchLimits {
    let work = Int4Work { forward_positions: u64::MAX / 8, projected_logits: u64::MAX / 8,
        attention_pairs: u64::MAX / 8, projections: ProjectionWork {
            dot_products: u64::MAX / 8, multiply_accumulates: u64::MAX / 8 } };
    SampledSentimentBatchLimits { max_item_work: work, max_model_work: work,
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: 1000,
            checkpoint_interval_nodes: 64 }, max_visits_per_item: 1_000_000, max_visits_per_run: 4_000_000 },
        max_sampler_bytes: json_sampling_workspace_bytes().unwrap() }
}
fn mapping() -> SampledSentimentMapLimits {
    SampledSentimentMapLimits { chunks: ChunkLimits { max_input_bytes: 8192, max_chunk_bytes: 1024,
        max_chunk_tokens: 1024, context_tokens: 4096, reserved_tokens: 128,
        max_chunks: 16, max_tokenizer_calls: 4096 }, max_model_work: work().max_item_work,
        max_retained_result_bytes: 4 << 20, max_result_bytes: 8 << 20 }
}
fn limits() -> JobLimits {
    JobLimits { max_items: 4, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 1 << 20, max_result_bytes: 8 << 20, max_spool_bytes: 64 << 20,
        max_materialized_bytes: 64 << 20, max_journal_bytes: 8 << 20, max_attempts: 8,
        max_work: JobWork::from_int4(work().max_model_work, work().masks.max_visits_per_run) }
}
fn host(encoding: &UntrustedEncoding) -> JobHostLimits {
    let preparation = sampled_sentiment_document_preparation_bytes(32, task(), planning(), mapping(),
        &options(), encoding).unwrap() + SAMPLED_SENTIMENT_JOB_RECIPE_BYTES as u64;
    JobHostLimits { native: NativeLimits { context_tokens: 4096, allocator_reserve_bytes: 64 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(60), max_checkpoints: 1_000_000,
            cleanup_reserve_bytes: 1 << 20 } },
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 4 },
        preparation_reserve_bytes: preparation, io_reserve_bytes: 128 << 10,
        journal_reserve_bytes: 64 << 20, serialization_reserve_bytes: 64 << 20 }
}
fn quote(encoding: &UntrustedEncoding, input: JobInputMemory, prefill: Option<Int4PrefillLimits>,
    threads: Option<usize>) -> SampledSentimentDocumentJobRequirements {
    int4_sampled_sentiment_document_job_requirements(32, task(), planning(), mapping(), &options(), encoding,
        work(), limits(), host(encoding), input, prefill, threads).unwrap()
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
fn compact_encoding_prompt_morsels_and_workers_are_charged_once() {
    let encoding = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 65536,
        max_output_tokens: 65536, max_pair_checks: 4_000_000, max_pair_bytes: 64_000_000 });
    let plain = quote(&UntrustedEncoding::ByteFallback, JobInputMemory::RetainedRecord, None, None);
    let compact = quote(&encoding, JobInputMemory::RetainedRecord, None, None);
    let UntrustedEncoding::SafeBpeV1(policy) = &encoding else { unreachable!() };
    assert_eq!(compact.document.input_bytes - plain.document.input_bytes, policy.requirements().unwrap().workspace_bytes);
    let team = quote(&encoding, JobInputMemory::RetainedRecord, None, Some(4));
    assert_eq!(team.document, compact.document);
    assert_eq!(team.team_workspace_bytes, int4_task_team_workspace_bytes(4, 32).unwrap());
    assert_eq!(team.total_bytes().unwrap(), compact.total_bytes().unwrap() + team.team_workspace_bytes);
    let prefill = Int4PrefillLimits { max_batch_rows: 2, max_extra_scratch_bytes: 64 << 20 };
    let layered = quote(&encoding, JobInputMemory::RetainedRecord, Some(prefill), None);
    assert_eq!(layered.document.workspace_bytes, compact.document.workspace_bytes + prefill.validate().unwrap());
    assert!(int4_sampled_sentiment_document_job_requirements(32, task(), planning(), mapping(), &options(),
        &encoding, work(), limits(), host(&encoding), JobInputMemory::RetainedRecord, Some(prefill), Some(2)).is_err());
}

#[test]
fn underquoted_preparation_output_source_and_lifetime_work_refuse_before_io() {
    let encoding = UntrustedEncoding::ByteFallback;
    for axis in 0..8 {
        let mut host = host(&encoding); let mut job = limits(); let mut map = mapping();
        let mut chunk = planning();
        match axis { 0 => host.preparation_reserve_bytes -= 1,
            1 => job.max_result_bytes -= 1, 2 => job.max_input_bytes_per_item -= 1,
            3 => job.max_work.mask_node_visits -= 1,
            4 => map.max_model_work.attention_pairs -= 1,
            5 => job.max_id_bytes = 257,
            6 => chunk.max_input_bytes = job.max_input_bytes_per_item + 1,
            _ => { job.max_result_bytes = 32768; map.max_result_bytes = 32768; } }
        assert!(int4_sampled_sentiment_document_job_requirements(32, task(), chunk, map, &options(),
            &encoding, work(), job, host, JobInputMemory::CompletePopulation, None, None).is_err(),
            "invalid admission relation {axis} reached the IO boundary");
    }
}

#[test]
fn public_complete_document_job_methods_share_the_owned_host_boundary() {
    use std::io::Cursor;
    let _serial = NlpEngine::job_int4_sampled_sentiment_documents::<Cursor<Vec<u8>>>;
    let _prefill = NlpEngine::job_int4_sampled_sentiment_documents_layer_major::<Cursor<Vec<u8>>>;
    let _team = NlpEngine::job_int4_sampled_sentiment_documents_with_threads::<Cursor<Vec<u8>>>;
}
