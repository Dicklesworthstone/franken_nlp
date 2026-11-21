//! Pure whole-document admission plus explicitly gated physical-lifetime tests.
//! Tiny weights never stand in for native inference or sentiment quality.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::{BatchWork, sentiment::sampled::long::SampledSentimentDocumentPlanFacts},
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{constrained_int4::planned_work, generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::mapreduce::ChunkLimits,
    tokenizer::untrusted::SafeBpeLimits,
};

const WIDTH: usize = 32;
const MIB: u64 = 1 << 20;

fn fixture(width: usize) -> SampledSentimentDocumentBatchHostConfig {
    let context = 2048;
    let task = TaskBudget { max_input_tokens: 1793, max_output_tokens: 256,
        max_output_bytes: 65_536, max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(context, width).unwrap().kv_bytes };
    let one = planned_work(width, task.max_input_tokens as usize, 32).unwrap();
    let item = (0..64).try_fold(Int4Work::default(), |total, _| total.checked_add(one)).unwrap();
    let run = item.checked_add(item).unwrap();
    SampledSentimentDocumentBatchHostConfig {
        document: SampledSentimentMapHostConfig {
            native: NativeLimits { context_tokens: context, allocator_reserve_bytes: MIB,
                run: RunLimits { max_elapsed: Duration::from_secs(120), max_checkpoints: 1_000_000,
                    cleanup_reserve_bytes: 65_536 } },
            planning: SampledSentimentLimits { max_input_bytes: 4096, max_result_bytes: 65_536,
                ..SampledSentimentLimits::default() },
            mapping: SampledSentimentMapLimits { chunks: ChunkLimits { max_input_bytes: 4096,
                max_chunk_bytes: 512, max_chunk_tokens: 512, context_tokens: context,
                reserved_tokens: 1536, max_chunks: 8, max_tokenizer_calls: 256 },
                max_model_work: item, max_retained_result_bytes: MIB, max_result_bytes: 2 * MIB },
            options: SampledSentimentOptions { effective_seed: [17; 32], samples: 2,
                temperature_milli: 1000, top_k: None, top_p_ppm: 1_000_000,
                max_sample_tokens: 32, max_justification_chars: 64 },
            preparation_reserve_bytes: 600 * MIB,
            max_sampler_bytes: json_sampling_workspace_bytes().unwrap(),
            mask_limits: MaskWorkLimits { max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 256 },
            max_mask_node_visits: 1_000_000_000_000, prefill: None, reuse_prefill: false,
        },
        task, encoding: UntrustedEncoding::ByteFallback,
        transport: BatchLimits { max_line_bytes: 65_536, max_document_bytes: 4096,
            max_output_line_bytes: 2 * MIB as usize + 4096,
            max_work: BatchWork { forward_positions: run.forward_positions, projected_logits: run.projected_logits },
            ..BatchLimits::default() },
        io_reserve_bytes: 65_536, max_model_work: run, max_mask_node_visits: 2_000_000_000_000,
    }
}
fn prefill(rows: usize) -> Int4PrefillLimits {
    Int4PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
fn identity(width: usize) -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"sampled-document-host-admission-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
        logical_model_digest: digest, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
        quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", digest, width)).unwrap()),
        tokenizer_digest: digest, template_digest: digest, task_spec: "sentiment-v1".to_owned(),
        taskir_digest: digest, prompt_digest: digest, grammar_compiler_version: "fixture".to_owned(), schema_digest: digest,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: constrained_int4::JSON_SAMPLER_VERSION.to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: digest, decision_policy_digest: digest,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn capacity(config: &SampledSentimentDocumentBatchHostConfig, width: usize) -> DocumentCapacity {
    let required = config.requirements(width).unwrap();
    DocumentCapacity { context: config.document.native.context_tokens, kv_bytes: required.native.kv_bytes,
        mapping: config.document.mapping, work: config.work_limits(), preparation_bytes: required.input_bytes,
        output_reservation_bytes: required.output_reservation_bytes,
        max_output_bytes: required.max_output_bytes, max_output_tokens: required.max_output_tokens }
}
fn request<'a>(config: &SampledSentimentDocumentBatchHostConfig, id: &'a ExecutionIdentity)
    -> SampledSentimentDocumentAdmission<'a> {
    let required = config.requirements(WIDTH).unwrap();
    SampledSentimentDocumentAdmission { facts: SampledSentimentDocumentPlanFacts {
        identity: id, model_work: config.document.mapping.max_model_work, group_width: WIDTH,
        sample_count: 64, chunk_count: 8, required_context: 1824, original_task: config.task,
        max_result_bytes: required.max_output_bytes, max_output_tokens: required.max_output_tokens,
        preparation_bytes: required.input_bytes },
        kv_reservation_bytes: required.native.kv_bytes, mask_node_visits: 1_000_000 }
}

#[test]
fn transport_is_added_once_without_multiplying_native_state_or_the_document_work_allowance() {
    for width in [1, 3, 32, 128] {
        let mut config = fixture(width);
        let map = config.document.requirements(width, config.task, 4).unwrap();
        let stream = config.requirements(width).unwrap();
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
        assert_eq!(config.requirements(width).unwrap(), stream,
            "longer streams reuse payload; they do not allocate another native model or plan set");
    }
}

#[test]
fn compact_prefill_and_team_quotes_share_the_declared_complete_document_peak() {
    let mut config = fixture(WIDTH);
    let base = config.requirements(WIDTH).unwrap();
    let policy = SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 4096,
        max_pair_checks: 1 << 20, max_pair_bytes: 16 << 20 };
    config.encoding = UntrustedEncoding::SafeBpeV1(policy);
    let compact = config.requirements(WIDTH).unwrap();
    assert_eq!(compact.input_bytes, base.input_bytes + policy.requirements().unwrap().workspace_bytes);
    assert_eq!(compact.native, base.native);
    for reuse in [false, true] {
        config.document.reuse_prefill = reuse;
        assert_eq!(config.requirements(WIDTH).unwrap(), compact);
        for width in [2, 4, 64] {
            let team = config.requirements_with_threads(WIDTH, width).unwrap();
            assert_eq!(team.input_bytes, compact.input_bytes);
            assert_eq!(team.workspace_bytes, compact.workspace_bytes + int4_task_team_workspace_bytes(width, WIDTH).unwrap());
        }
        config.document.prefill = Some(prefill(4));
        let prefilled = config.requirements(WIDTH).unwrap();
        assert_eq!(prefilled.workspace_bytes, compact.workspace_bytes + prefill(4).max_extra_scratch_bytes);
        assert_eq!(prefilled.input_bytes, compact.input_bytes);
        assert!(config.requirements_with_threads(WIDTH, 4).is_err());
        config.document.prefill = None;
    }
}

#[test]
fn every_run_work_axis_mask_transport_and_geometry_refuses_before_io() {
    for axis in 0..5 {
        let mut config = fixture(WIDTH);
        let item = config.document.mapping.max_model_work;
        match axis { 0 => config.max_model_work.forward_positions = item.forward_positions - 1,
            1 => config.max_model_work.projected_logits = item.projected_logits - 1,
            2 => config.max_model_work.attention_pairs = item.attention_pairs - 1,
            3 => config.max_model_work.projections.dot_products = item.projections.dot_products - 1,
            _ => config.max_model_work.projections.multiply_accumulates = item.projections.multiply_accumulates - 1 }
        assert!(config.requirements(WIDTH).is_err(), "work axis {axis}");
    }
    for axis in 0..11 {
        let mut config = fixture(WIDTH);
        match axis {
            0 => config.max_mask_node_visits = config.document.max_mask_node_visits - 1,
            1 => config.document.max_mask_node_visits = 0,
            2 => config.document.max_sampler_bytes -= 1,
            3 => config.task.max_kv_bytes -= 1,
            4 => config.io_reserve_bytes = 0,
            5 => config.document.native.allocator_reserve_bytes = u64::MAX,
            6 => config.transport.max_output_line_bytes = config.document.mapping.max_result_bytes as usize - 1,
            7 => config.document.mapping.chunks.max_chunks = 0,
            8 => config.document.native.run.max_elapsed = Duration::ZERO,
            9 => config.transport.max_epoch_id_bytes = 0,
            _ => config.document.prefill = Some(Int4PrefillLimits { max_batch_rows: 4,
                max_extra_scratch_bytes: prefill(4).max_extra_scratch_bytes - 1 }),
        }
        assert!(config.requirements(WIDTH).is_err(), "resource axis {axis}");
    }
    for threads in [0, 1, 65, usize::MAX] {
        assert!(fixture(WIDTH).requirements_with_threads(WIDTH, threads).is_err());
    }
}

#[test]
fn complete_output_can_exceed_one_chunk_but_never_its_owned_document_reservation() {
    let config = fixture(WIDTH); let id = identity(WIDTH); let capacity = capacity(&config, WIDTH);
    let admitted = request(&config, &id);
    assert!(admitted.facts.max_result_bytes > admitted.facts.original_task.max_output_bytes);
    assert!(admitted.facts.max_output_tokens > u64::from(admitted.facts.original_task.max_output_tokens));
    check_request(WIDTH, capacity, &admitted).unwrap();
    for axis in 0..4 {
        let mut request = request(&config, &id);
        match axis { 0 => request.facts.max_result_bytes += 1,
            1 => request.facts.max_output_tokens += 1,
            2 => request.facts.max_output_tokens = 0,
            _ => request.facts.max_output_tokens = u64::MAX }
        assert!(check_request(WIDTH, capacity, &request).is_err());
    }
    let mut short = capacity; short.output_reservation_bytes -= 1;
    assert_eq!(check_request(WIDTH, short, &admitted).unwrap_err().fault.code, BatchCode::OutputLineLimit);
}

#[test]
fn every_complete_document_binding_is_checked_before_result_claims() {
    let config = fixture(WIDTH); let id = identity(WIDTH); let cap = capacity(&config, WIDTH);
    for axis in 0..10 {
        let mut request = request(&config, &id);
        match axis {
            0 => request.facts.required_context = cap.context + 1,
            1 => request.kv_reservation_bytes -= 1,
            2 => request.facts.original_task.max_kv_bytes -= 1,
            3 => request.facts.preparation_bytes += 1,
            4 => request.mask_node_visits = 0,
            5 => request.mask_node_visits = cap.work.masks.max_visits_per_item + 1,
            6 => request.facts.chunk_count += 1,
            7 => request.facts.sample_count = 0,
            8 => request.facts.group_width = 128,
            _ => request.facts.model_work.projections.multiply_accumulates += 1,
        }
        assert!(check_request(WIDTH, cap, &request).unwrap_err().stop, "binding {axis}");
    }
    let mut wrong = identity(WIDTH); wrong.sampler_version = "distribution-scoring".to_owned();
    assert!(check_request(WIDTH, cap, &request(&config, &wrong)).unwrap_err().stop);
}

#[test]
fn complete_stream_owners_are_send_and_seed_debug_is_redacted() {
    fn send<T: Send + 'static>() {}
    send::<StreamOwners>(); send::<StreamExecution<std::io::Cursor<Vec<u8>>, Vec<u8>>>();
    send::<DeferredDocumentStream<std::io::Cursor<Vec<u8>>, Vec<u8>>>();
    let _serial = NlpEngine::batch_int4_sampled_sentiment_documents::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _team = NlpEngine::batch_int4_sampled_sentiment_documents_with_threads::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let shown = format!("{:?}", fixture(WIDTH));
    assert!(shown.contains("REDACTED")); assert!(!shown.contains("[17, 17"));
}

mod lifecycle;
mod model;
