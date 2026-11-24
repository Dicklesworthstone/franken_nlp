//! Complete long-document recipe and durable work, not chunk inference evidence.
use super::*;
use crate::tasks::{mapreduce::{ChunkLimits, ExecutionLimits}, source_planning::quantized::long::Int8SourceMapLimits,
    redact::{corpus::LongRedactionBatchConfig, long::LongRedactionConfig}};
fn planner() -> SourceTaskPlanner {
    let tokenizer = crate::tokenizer::embedded::EmbeddedTokenizer::pinned().unwrap();
    let controls = crate::tokenizer::pinned_controls::pinned().unwrap();
    SourceTaskPlanner::pinned(controls.template_controls(), tokenizer.eos_token_id().unwrap()).unwrap()
}
fn long_config() -> LongRedactionBatchConfig {
    let c = config();
    LongRedactionBatchConfig { ner_identity: c.ner_identity,
        detector: LongRedactionConfig { ner: c.detector.ner, per_chunk: c.detector.per_pass, planning: c.detector.planning,
            mapping: Int8SourceMapLimits { chunks: ChunkLimits { max_chunks: 8, ..ChunkLimits::default() },
                reduction: ExecutionLimits::default(), max_model_work: c.detector.max_model_work,
                mask_limits: c.detector.mask_limits, mask_visits_per_chunk: 1000, max_mask_visits: 2000 },
            max_result_bytes: c.detector.max_result_bytes },
        request: c.request, max_model_work: c.max_model_work, max_mask_visits: c.max_mask_visits }
}
#[test]
fn short_and_chunked_jobs_cannot_share_a_replay_contract() {
    let base = config(); let long = long_config();
    let short_recipe = RedactionJobRecipe::short(&base, None).unwrap();
    let long_recipe = RedactionJobRecipe::long(&long, None, &planner()).unwrap();
    assert_eq!(freeze(&base, &short_recipe, INPUT).binding.compare(&freeze(&base, &long_recipe, INPUT).binding),
        Err(JobError::Mismatch(MismatchField::Recipe)));
    // Preserve the existing short recipe's detector shape, without a new tag.
    let value = serde_json::to_value(short_recipe).unwrap();
    assert!(value["detector"].get("per_pass").is_some());
    assert!(value["detector"].get("Short").is_none());
}
#[test]
fn every_chunk_and_reduction_bound_changes_the_authenticated_recipe() {
    let base = config(); let original = freeze(&base, &RedactionJobRecipe::long(&long_config(), None, &planner()).unwrap(), INPUT);
    for axis in 0..15 {
        let mut c = long_config(); let m = &mut c.detector.mapping;
        match axis {
            0 => m.chunks.max_input_bytes += 1, 1 => m.chunks.max_chunk_bytes += 1,
            2 => m.chunks.max_chunk_tokens += 1, 3 => m.chunks.context_tokens += 1,
            4 => m.chunks.reserved_tokens += 1, 5 => m.chunks.max_chunks += 1, 6 => m.chunks.max_tokenizer_calls += 1,
            7 => m.reduction.map_batch_chunks += 1, 8 => m.reduction.reduce_fan_in += 1,
            9 => m.reduction.max_reduction_levels += 1, 10 => m.reduction.max_task_calls += 1,
            11 => m.reduction.max_value_bytes += 1, 12 => m.reduction.max_live_value_bytes += 1,
            13 => m.reduction.max_total_value_bytes += 1, _ => m.reduction.max_result_bytes += 1,
        }
        let recipe = RedactionJobRecipe::long(&c, None, &planner()).unwrap();
        assert_eq!(original.binding.compare(&freeze(&base, &recipe, INPUT).binding), Err(JobError::Mismatch(MismatchField::Recipe)));
    }
}
#[test]
fn chunked_debit_uses_complete_document_work_and_exact_full_row_rounding() {
    let mut c = long_config(); c.detector.mapping.max_model_work.projected_logits += 17;
    let recipe = RedactionJobRecipe::long(&c, None, &planner()).unwrap(); let expected = c.item_model_work();
    let work = recipe.check_work(BatchWork { forward_positions: expected.forward_positions,
        projected_logits: expected.projected_logits }, false).unwrap();
    assert_eq!(work.model, expected); assert_eq!(work.mask_node_visits, c.detector.mapping.max_mask_visits);
    assert_eq!(work.model.projected_logits % crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE as u64, 0);
    assert!(recipe.check_work(BatchWork { forward_positions: expected.forward_positions,
        projected_logits: expected.projected_logits }, true).is_err());
}
#[test]
fn verification_and_map_mask_allowances_cannot_change_on_resume() {
    let base = config(); let original = freeze(&base, &RedactionJobRecipe::long(&long_config(), None, &planner()).unwrap(), INPUT);
    for axis in 0..4 {
        let mut c = long_config();
        match axis { 0 => c.request.verify = false, 1 => c.detector.mapping.mask_visits_per_chunk += 1,
            2 => c.detector.mapping.max_mask_visits += 1, _ => c.detector.per_chunk.max_kv_bytes += 1 }
        let changed = RedactionJobRecipe::long(&c, None, &planner()).unwrap();
        assert_eq!(original.binding.compare(&freeze(&base, &changed, INPUT).binding), Err(JobError::Mismatch(MismatchField::Recipe)));
    }
}
#[test]
fn long_document_host_owns_the_same_native_processor_and_key_scope() {
    fn send<T: Send + 'static>() {}
    send::<StreamInput<(Arc<SourceTaskPlanner>, Arc<ExtractionVocabulary>, RedactionCorpusConfig<LongRedactionBatchConfig>,
        Option<RedactionPseudonyms>, SourceJobRequest), std::io::Cursor<Vec<u8>>, ()>>();
    let _entry = NlpEngine::job_int8_redact_document::<std::io::Cursor<Vec<u8>>>;
}

#[test]
fn every_compact_encoding_limit_is_authenticated_before_any_resumed_attempt() {
    use crate::tokenizer::untrusted::{SafeBpeLimits, SafeBpeWorkLimits, UntrustedEncoding};
    let call = SafeBpeLimits { max_input_bytes: 128, max_output_tokens: 128,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 };
    let complete = SafeBpeWorkLimits { max_input_bytes: 16384, max_output_tokens: 16384,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 };
    let factory = |call, complete| planner().with_map_document_encoding(UntrustedEncoding::SafeBpeV1(call), complete).unwrap();
    let base = config(); let long = long_config();
    let original_factory = factory(call, complete);
    let recipe = RedactionJobRecipe::long(&long, None, &original_factory).unwrap();
    let original = freeze(&base, &recipe, INPUT);
    original.binding.compare(&freeze(&base,
        &RedactionJobRecipe::long(&long, None, &factory(call, complete)).unwrap(), INPUT).binding).unwrap();
    for axis in 0..8 {
        let mut call = call; let mut complete = complete;
        match axis {
            0 => call.max_input_bytes += 1, 1 => call.max_output_tokens += 1,
            2 => call.max_pair_checks += 1, 3 => call.max_pair_bytes += 1,
            4 => complete.max_input_bytes += 1, 5 => complete.max_output_tokens += 1,
            6 => complete.max_pair_checks += 1, _ => complete.max_pair_bytes += 1,
        }
        let changed = RedactionJobRecipe::long(&long, None, &factory(call, complete)).unwrap();
        assert_eq!(original.binding.compare(&freeze(&base, &changed, INPUT).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "{axis}");
    }
    let byte = RedactionJobRecipe::long(&long, None, &planner()).unwrap();
    assert_eq!(original.binding.compare(&freeze(&base, &byte, INPUT).binding),
        Err(JobError::Mismatch(MismatchField::Recipe)));
    assert!(serde_json::to_value(&byte).unwrap().get("input_encoding").is_none());
    let value = serde_json::to_value(&recipe).unwrap();
    assert_eq!(value["input_encoding"]["policy"], serde_json::to_value(original_factory.document_encoding()).unwrap());
    assert_eq!(value["input_encoding"]["complete_document_limits"], serde_json::to_value(complete).unwrap());
    assert_eq!(value["input_encoding"]["budget_scope"],
        "original_preflight_execution_and_edited_verification_per_document_attempt");
    assert!(value["input_encoding"].get("work").is_none());
}

#[test]
fn a_private_job_recipe_never_freezes_an_unbounded_or_spent_factory_as_document_work() {
    use crate::tokenizer::untrusted::{SafeBpeLimits, SafeBpeWorkLimits, UntrustedEncoding};
    let policy = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 128, max_output_tokens: 128,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 });
    let unbounded = planner().with_document_encoding(policy.clone()).unwrap();
    assert!(RedactionJobRecipe::long(&long_config(), None, &unbounded).is_err());
    let factory = planner().with_map_document_encoding(policy, SafeBpeWorkLimits {
        max_input_bytes: 16384, max_output_tokens: 16384, max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 }).unwrap();
    let before = canonjson::canonical_bytes(&RedactionJobRecipe::long(&long_config(), None, &factory).unwrap()).unwrap();
    factory.source_encoder().encode_with_control("private source", 128, 128, &mut || None).unwrap();
    assert!(factory.map_document_encoding_work().unwrap().unwrap().input_bytes() > 0);
    let after = canonjson::canonical_bytes(&RedactionJobRecipe::long(&long_config(), None, &factory).unwrap()).unwrap();
    assert_eq!(before, after, "the recipe authenticates immutable allowance, never an observed-work refund or renewal");
    let text = String::from_utf8(after).unwrap();
    assert!(!text.contains("private source"));
}
