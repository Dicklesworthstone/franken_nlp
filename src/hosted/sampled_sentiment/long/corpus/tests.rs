//! Pure population admission fixtures. No fixture is model-quality evidence.
use super::*;
use crate::{
    batch::{BatchWork, sentiment::sampled::quantized::long::Int8SampledSentimentDocumentPlanFacts},
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{constrained_int8::planned_work, strict_int8::STRICT_INT8_EXECUTION},
    tasks::mapreduce::ChunkLimits,
    tokenizer::untrusted::SafeBpeLimits,
};

const MIB: u64 = 1 << 20;
fn fixture() -> Int8SampledSentimentDocumentBatchHostConfig {
    let context = 2048;
    let task = TaskBudget { max_input_tokens: 1793, max_output_tokens: 256,
        max_output_bytes: 65_536, max_grammar_states: 4096,
        max_kv_bytes: Int8MemoryRequirement::for_context(context).unwrap().kv_bytes };
    let one = planned_work(task.max_input_tokens as usize, 32).unwrap();
    let item = (0..64).try_fold(Int8Work::default(), |total, _| total.checked_add(one)).unwrap();
    let run = item.checked_add(item).unwrap();
    Int8SampledSentimentDocumentBatchHostConfig {
        document: Int8SampledSentimentMapHostConfig {
            native: NativeLimits { context_tokens: context, allocator_reserve_bytes: MIB,
                run: RunLimits { max_elapsed: Duration::from_secs(120), max_checkpoints: 1_000_000,
                    cleanup_reserve_bytes: 65_536 } },
            planning: SampledSentimentLimits { max_input_bytes: 4096, max_result_bytes: 65_536,
                ..SampledSentimentLimits::default() },
            mapping: Int8SampledSentimentMapLimits { chunks: ChunkLimits { max_input_bytes: 4096,
                max_chunk_bytes: 512, max_chunk_tokens: 512, context_tokens: context,
                reserved_tokens: 1536, max_chunks: 8, max_tokenizer_calls: 256 },
                max_model_work: item, max_retained_result_bytes: MIB, max_result_bytes: 2 * MIB },
            options: SampledSentimentOptions { effective_seed: [17; 32], samples: 2,
                temperature_milli: 1000, top_k: None, top_p_ppm: 1_000_000,
                max_sample_tokens: 32, max_justification_chars: 64 },
            preparation_reserve_bytes: 600 * MIB,
            max_sampler_bytes: json_sampling_workspace_bytes().unwrap(),
            mask_limits: MaskWorkLimits { max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 256 },
            max_mask_node_visits: 1_000_000_000_000, prefill: None,
        },
        task, encoding: UntrustedEncoding::ByteFallback,
        transport: BatchLimits { max_line_bytes: 65_536, max_document_bytes: 4096,
            max_output_line_bytes: 2 * MIB as usize + 4096,
            max_work: BatchWork { forward_positions: run.forward_positions, projected_logits: run.projected_logits },
            ..BatchLimits::default() },
        io_reserve_bytes: 65_536, max_model_work: run, max_mask_node_visits: 2_000_000_000_000,
    }
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
fn identity() -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"int8-sampled-document-population-admission");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(),
        logical_model_digest: digest, artifact_format: "fixture".to_owned(),
        quant_recipe: "nanbeige42-int8-v1".to_owned(), packing_set_digest: digest,
        tokenizer_digest: digest, template_digest: digest, task_spec: "sentiment-v1".to_owned(),
        taskir_digest: digest, prompt_digest: digest, grammar_compiler_version: "fixture".to_owned(), schema_digest: digest,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: constrained_int8::JSON_SAMPLER_VERSION.to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: digest, decision_policy_digest: digest,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn capacity(config: &Int8SampledSentimentDocumentBatchHostConfig) -> DocumentCapacity {
    let required = config.requirements().unwrap();
    DocumentCapacity { context: config.document.native.context_tokens, kv_bytes: required.native.kv_bytes,
        mapping: config.document.mapping, work: config.work_limits(), preparation_bytes: required.input_bytes,
        output_reservation_bytes: required.output_reservation_bytes,
        max_output_bytes: required.max_output_bytes, max_output_tokens: required.max_output_tokens,
        samples_per_axis: config.document.options.samples, max_sample_tokens: config.document.options.max_sample_tokens }
}
fn request<'a>(config: &Int8SampledSentimentDocumentBatchHostConfig, identity: &'a ExecutionIdentity)
    -> Int8SampledSentimentDocumentAdmission<'a> {
    let required = config.requirements().unwrap();
    Int8SampledSentimentDocumentAdmission { facts: Int8SampledSentimentDocumentPlanFacts {
        identity, model_work: config.document.mapping.max_model_work,
        sample_count: 64, chunk_count: 8, required_context: 1824, original_task: config.task,
        max_result_bytes: required.max_output_bytes, max_output_tokens: required.max_output_tokens,
        preparation_bytes: required.input_bytes },
        kv_reservation_bytes: required.native.kv_bytes, mask_node_visits: 1_000_000 }
}

#[test]
fn transport_is_added_once_and_longer_streams_reuse_the_complete_document_peak() {
    let mut config = fixture();
    let map = config.document.requirements(config.task, 4).unwrap();
    let stream = config.requirements().unwrap();
    let transport = CorpusLimits { native: config.document.native, transport: config.transport,
        preparation_reserve_bytes: config.document.preparation_reserve_bytes,
        io_reserve_bytes: config.io_reserve_bytes }.reservation_bytes().unwrap();
    assert_eq!(stream.input_bytes, map.input_bytes + transport - config.document.preparation_reserve_bytes);
    assert_eq!(stream.native, map.native);
    assert_eq!(stream.workspace_bytes, map.workspace_bytes);
    assert_eq!(stream.output_reservation_bytes, map.output_reservation_bytes);
    config.max_model_work = config.max_model_work.checked_add(config.max_model_work).unwrap();
    config.max_mask_node_visits *= 2;
    config.transport.max_requests *= 2;
    assert_eq!(config.requirements().unwrap(), stream);
}

#[test]
fn compact_prefill_and_team_quotes_price_each_coexisting_owner_once() {
    let mut config = fixture();
    let base = config.requirements().unwrap();
    let policy = SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 4096,
        max_pair_checks: 1 << 20, max_pair_bytes: 16 << 20 };
    config.encoding = UntrustedEncoding::SafeBpeV1(policy);
    let compact = config.requirements().unwrap();
    assert_eq!(compact.input_bytes, base.input_bytes + policy.requirements().unwrap().workspace_bytes);
    assert_eq!(compact.native, base.native);
    for rows in [1, 4, 64] {
        config.document.prefill = Some(prefill(rows));
        let scheduled = config.requirements().unwrap();
        assert_eq!(scheduled.workspace_bytes, compact.workspace_bytes + prefill(rows).max_extra_scratch_bytes);
        assert_eq!(scheduled.input_bytes, compact.input_bytes);
        for width in [2, 4, 64] {
            let threaded = config.requirements_with_threads(width).unwrap();
            assert_eq!(threaded.workspace_bytes, scheduled.workspace_bytes
                + dispatch::int8_prefill_team_workspace_bytes(width, rows).unwrap());
            assert_eq!(threaded.native, compact.native);
            assert_eq!(threaded.output_reservation_bytes, compact.output_reservation_bytes);
        }
    }
    for width in [0, 1, 65, usize::MAX] { assert!(config.requirements_with_threads(width).is_err()); }
}

#[test]
fn all_native_mask_transport_and_memory_limits_refuse_before_input() {
    for axis in 0..5 {
        let mut config = fixture();
        let item = config.document.mapping.max_model_work;
        match axis {
            0 => config.max_model_work.forward_positions = item.forward_positions - 1,
            1 => config.max_model_work.projected_logits = item.projected_logits - 1,
            2 => config.max_model_work.attention_pairs = item.attention_pairs - 1,
            3 => config.max_model_work.projections.dot_products = item.projections.dot_products - 1,
            _ => config.max_model_work.projections.multiply_accumulates = item.projections.multiply_accumulates - 1,
        }
        assert!(config.requirements().is_err(), "native axis {axis}");
    }
    for axis in 0..11 {
        let mut config = fixture();
        match axis {
            0 => config.max_mask_node_visits = config.document.max_mask_node_visits - 1,
            1 => config.document.mask_limits.max_trie_node_visits = 0,
            2 => config.document.mask_limits.checkpoint_interval_nodes = 0,
            3 => config.document.max_sampler_bytes -= 1,
            4 => config.transport.max_output_line_bytes = config.document.mapping.max_result_bytes as usize - 1,
            5 => config.io_reserve_bytes = 0,
            6 => config.document.preparation_reserve_bytes = 1,
            7 => config.document.native.allocator_reserve_bytes = u64::MAX,
            8 => config.task.max_kv_bytes -= 1,
            9 => config.document.prefill = Some(Int8PrefillLimits { max_batch_rows: 0, max_extra_scratch_bytes: u64::MAX }),
            _ => config.document.prefill = Some(Int8PrefillLimits { max_extra_scratch_bytes: prefill(4).max_extra_scratch_bytes - 1,
                ..prefill(4) }),
        }
        assert!(config.requirements().is_err(), "limit axis {axis}");
    }
}

#[test]
fn complete_sample_geometry_full_kv_and_every_work_axis_are_bound_before_execution() {
    let config = fixture(); let identity = identity(); let cap = capacity(&config);
    check_request(cap, &request(&config, &identity)).unwrap();
    for axis in 0..15 {
        let mut changed = request(&config, &identity);
        match axis {
            0 => changed.kv_reservation_bytes -= 1,
            1 => changed.facts.original_task.max_kv_bytes -= 1,
            2 => changed.facts.required_context = cap.context + 1,
            3 => changed.facts.chunk_count = 0,
            4 => changed.facts.chunk_count = config.document.mapping.chunks.max_chunks + 1,
            5 => changed.facts.sample_count -= 1,
            6 => changed.facts.max_output_tokens -= 1,
            7 => changed.facts.model_work.forward_positions += 1,
            8 => changed.facts.model_work.projected_logits -= 1,
            9 => changed.facts.model_work.attention_pairs += 1,
            10 => changed.facts.model_work.projections.dot_products += 1,
            11 => changed.facts.model_work.projections.multiply_accumulates += 1,
            12 => changed.mask_node_visits = config.document.max_mask_node_visits + 1,
            13 => changed.facts.preparation_bytes = cap.preparation_bytes + 1,
            _ => changed.facts.preparation_bytes = 0,
        }
        assert!(check_request(cap, &changed).unwrap_err().stop, "binding {axis}");
    }
    let mut wrong = identity.clone(); wrong.sampler_version = "distribution-scoring".into();
    assert!(check_request(cap, &request(&config, &wrong)).unwrap_err().stop);
    wrong = identity; wrong.numerics_profile = NumericsProfile::StrictQuantized { version: 2 };
    assert!(check_request(cap, &request(&config, &wrong)).unwrap_err().stop);
}

#[test]
fn result_bytes_and_serialization_peak_are_an_independent_document_boundary() {
    let config = fixture(); let identity = identity(); let cap = capacity(&config);
    let mut changed = request(&config, &identity);
    changed.facts.max_result_bytes -= 1;
    assert_eq!(check_request(cap, &changed).unwrap_err().fault.code, BatchCode::OutputLineLimit);
    let changed = request(&config, &identity);
    let expected = changed.facts.max_result_bytes * 4 + changed.facts.max_output_tokens * 8 + 4096;
    let too_small = DocumentCapacity { output_reservation_bytes: expected - 1, ..cap };
    assert_eq!(check_request(too_small, &changed).unwrap_err().fault.code, BatchCode::OutputLineLimit);
}

#[test]
fn retained_stream_owners_cross_dispatch_and_debug_keeps_seed_private() {
    fn send<T: Send + 'static>() {}
    send::<StreamExecution<std::io::Cursor<Vec<u8>>, Vec<u8>>>();
    send::<stream::deferred::DeferredCompletion>();
    let _serial = NlpEngine::batch_int8_sampled_sentiment_documents::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _team = NlpEngine::batch_int8_sampled_sentiment_documents_with_threads::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let shown = format!("{:?}", fixture());
    assert!(shown.contains("REDACTED")); assert!(!shown.contains("[17, 17"));
}
