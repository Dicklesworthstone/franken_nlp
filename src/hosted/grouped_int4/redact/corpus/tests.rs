//! Resident corpus admission/ownership fixtures; no model execution.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::{ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::{SourcePlanningLimits, grouped_int4::long::Int4SourceMapLimits},
        redact::{actions::RedactionAction, grouped_int4::long::Int4LongRedactionConfig,
            pseudonym::PseudonymKey}},
    validation::grounded_fields::{GroundingBudget, SourceFieldEvidence, VerifiedSourceSpan},
};

fn config(width: usize) -> RedactionCorpusConfig<Int4DocumentRedactionBatchConfig> {
    let d = Sha256Digest::of_bytes(b"int4-redaction-corpus-host-fixture");
    let identity = ExecutionIdentity {
        schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width,
        )).unwrap()), tokenizer_digest: d, template_digest: d, task_spec: NER_TASK_VERSION.to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(),
        schema_digest: d, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None,
    };
    let per = constrained_int4::planned_work(width, 64, 64).unwrap();
    let mut item = per;
    for _ in 1..6 { item = item.checked_add(per).unwrap(); }
    let mut total = item;
    for _ in 1..4 { total = total.checked_add(item).unwrap(); }
    let mut request = RedactionRequest::default();
    request.actions.default_action = RedactionAction::Mask;
    request.edit_budget.max_output_bytes = 65_536;
    request.rule_budget.max_input_bytes = 65_536;
    RedactionCorpusConfig {
        batch: Int4DocumentRedactionBatchConfig {
            ner_identity: identity,
            detector: Int4LongRedactionConfig {
                ner: Default::default(),
                per_chunk: TaskBudget { max_input_tokens: 64, max_output_tokens: 64, max_output_bytes: 65_536,
                    max_grammar_states: 4096, max_kv_bytes: Int4MemoryRequirement::for_context(128, width).unwrap().kv_bytes },
                planning: SourcePlanningLimits { max_input_bytes: 128, max_context_tokens: 128, ..Default::default() },
                mapping: Int4SourceMapLimits {
                    chunks: ChunkLimits { max_input_bytes: 65_536, max_chunk_bytes: 128,
                        max_chunk_tokens: 64, context_tokens: 128, reserved_tokens: 64,
                        max_chunks: 3, ..Default::default() },
                    reduction: ExecutionLimits { max_value_bytes: 65_536, max_live_value_bytes: 131_072,
                        max_result_bytes: 65_536, ..Default::default() },
                    max_model_work: item, mask_limits: Default::default(), mask_visits_per_chunk: 1000,
                    max_mask_visits: 6000,
                    verification: GroundingBudget { max_fields: 32, max_matches: 128, max_scan_steps: 1_000_000 },
                },
                max_result_bytes: 65_536,
            },
            request, max_model_work: total, max_mask_visits: 24_000,
        },
        edit_reserve_bytes: 1 << 20,
    }
}
fn limits() -> CorpusLimits {
    CorpusLimits {
        native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 65_536,
            run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100_000, cleanup_reserve_bytes: 65_536 } },
        transport: crate::batch::BatchLimits::default(),
        preparation_reserve_bytes: 1 << 20, io_reserve_bytes: 1 << 20,
    }
}
fn fixed(c: &RedactionCorpusConfig<Int4DocumentRedactionBatchConfig>, width: usize) -> FixedItem {
    FixedItem::new(&c.batch, width, Int4MemoryRequirement::for_context(128, width).unwrap().kv_bytes, 131_072)
}
fn request(fixed: &FixedItem) -> Int4RedactionAdmission<'_> {
    Int4RedactionAdmission { identity: &fixed.identity, model_work: fixed.model_work,
        mask_node_visits: fixed.mask_visits, mask_limits: fixed.mask_limits,
        kv_reservation_bytes: fixed.kv_bytes, max_result_bytes: fixed.result_bytes }
}

#[test]
fn multiple_independent_contexts_admit_without_concatenated_attention_or_kv_geometry() {
    for width in [3, 127, 128, 10752] {
        let c = config(width); let f = fixed(&c, width);
        assert!(f.model_work.forward_positions > 128);
        let concatenated = Int4Work::for_sequence(width, 0, f.model_work.forward_positions as usize,
            f.model_work.projected_logits as usize).unwrap();
        assert_ne!(f.model_work.attention_pairs, concatenated.attention_pairs);
        check_admission(&f, &request(&f)).unwrap();
    }
}

#[test]
fn independently_loose_cap_axes_are_not_mistaken_for_an_executable_sequence() {
    let mut c = config(127);
    c.batch.detector.mapping.max_model_work.projected_logits =
        (c.batch.detector.mapping.max_model_work.forward_positions + 10) * NANBEIGE_VOCAB_SIZE as u64;
    let f = fixed(&c, 127);
    assert!(f.model_work.projected_logits / NANBEIGE_VOCAB_SIZE as u64 > f.model_work.forward_positions);
    // All actual execution still has to fit every independent cap.
    check_admission(&f, &request(&f)).unwrap();
}

#[test]
fn exact_task_model_profile_and_full_identity_remain_fixed_for_every_item() {
    let c = config(127); let f = fixed(&c, 127);
    for axis in 0..6 {
        let mut identity = f.identity.clone();
        match axis {
            0 => identity.task_spec = "extract-v1".to_owned(),
            1 => identity.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            2 => identity.backend_semantic_version = "other".to_owned(),
            3 => identity.quant_recipe = format!("{GROUPED_INT4_RECIPE}:group=128"),
            4 => identity.logical_model_digest = Sha256Digest::of_bytes(b"foreign"),
            _ => identity.prompt_digest = Sha256Digest::of_bytes(b"foreign-template-contract"),
        }
        let mut req = request(&f); req.identity = &identity;
        let error = check_admission(&f, &req).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
    }
    let mut req = request(&f); req.kv_reservation_bytes -= 1;
    assert_eq!(check_admission(&f, &req).unwrap_err().fault.code, BatchCode::Admission);
}

#[test]
fn all_five_item_work_and_mask_authorities_must_equal_the_frozen_configuration() {
    let c = config(127); let f = fixed(&c, 127);
    for axis in 0..8 {
        let mut req = request(&f);
        match axis {
            0 => req.model_work.forward_positions += 1,
            1 => req.model_work.projected_logits += NANBEIGE_VOCAB_SIZE as u64,
            2 => req.model_work.attention_pairs += 1,
            3 => req.model_work.projections.dot_products += 1,
            4 => req.model_work.projections.multiply_accumulates += 1,
            5 => req.mask_node_visits += 1,
            6 => req.mask_limits.max_trie_node_visits += 1,
            _ => req.mask_limits.checkpoint_interval_nodes += 1,
        }
        let error = check_admission(&f, &req).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidExecution);
    }
}

#[test]
fn whole_vocabulary_row_floor_drops_only_unusable_fractional_logits() {
    let mut c = config(127);
    c.batch.detector.mapping.max_model_work.projected_logits += NANBEIGE_VOCAB_SIZE as u64 - 1;
    let f = fixed(&c, 127);
    assert_eq!(f.model_work.projected_logits % NANBEIGE_VOCAB_SIZE as u64, 0);
    assert_eq!(c.batch.detector.mapping.max_model_work.projected_logits - f.model_work.projected_logits,
        NANBEIGE_VOCAB_SIZE as u64 - 1);
    check_admission(&f, &request(&f)).unwrap();
    let mut fractional = fixed(&c, 127); fractional.model_work.projected_logits += 1;
    assert!(check_admission(&fractional, &request(&fractional)).is_err());
}

#[test]
fn output_frame_refusals_and_mutated_result_caps_do_not_release_a_success_permit() {
    let c = config(127); let f = fixed(&c, 127);
    for bytes in [0, f.output_line_bytes + 1] {
        let mut req = request(&f); req.max_result_bytes = bytes;
        let error = check_admission(&f, &req).unwrap_err();
        assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::OutputLineLimit);
    }
    let mut req = request(&f); req.max_result_bytes -= 1;
    let error = check_admission(&f, &req).unwrap_err();
    assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidExecution);
}

#[test]
fn complete_resident_kv_context_and_nonzero_reserves_are_required_once_for_the_stream() {
    let required = Int4MemoryRequirement::for_context(128, 127).unwrap();
    validate_limits(&config(127), limits(), required.kv_bytes).unwrap();
    for axis in 0..7 {
        let mut c = config(127); let mut l = limits();
        match axis {
            0 => c.edit_reserve_bytes = 0,
            1 => l.native.allocator_reserve_bytes = 0,
            2 => c.batch.detector.per_chunk.max_kv_bytes = required.kv_bytes - 1,
            3 => c.batch.detector.planning.max_context_tokens += 1,
            4 => c.batch.detector.mapping.chunks.context_tokens += 1,
            5 => c.batch.detector.max_result_bytes = 0,
            _ => l.transport.max_output_line_bytes = 65_535,
        }
        assert!(validate_limits(&c, l, required.kv_bytes).is_err(), "axis {axis}");
    }
    assert!(validate_limits(&config(127), limits(), 0).is_err());
}

#[test]
fn stream_buffers_and_one_complete_map_peak_remain_distinct_from_per_item_output() {
    let c = config(127); let l = limits();
    let t = l.transport;
    assert_eq!(l.reservation_bytes().unwrap(), 8 * t.max_line_bytes as u64
        + 4 * t.max_output_line_bytes as u64 + 128 * t.max_epoch_ids as u64
        + t.max_epoch_id_bytes as u64 + l.preparation_reserve_bytes + l.io_reserve_bytes);
    let expected = 4 * (3 * 65_536 + 131_072) + 8 * 64 * 3
        + 32 * std::mem::size_of::<SourceFieldEvidence>() as u64
        + 2 * 128 * std::mem::size_of::<VerifiedSourceSpan>() as u64 + c.edit_reserve_bytes;
    assert_eq!(long::document_workspace_bytes(&c.batch.detector, c.edit_reserve_bytes).unwrap(), expected);
    assert!(long::document_workspace_bytes(&c.batch.detector, 0).is_err());
    let mut oversized = config(127); oversized.batch.detector.per_chunk.max_output_bytes = u64::MAX;
    assert!(long::document_workspace_bytes(&oversized.batch.detector, oversized.edit_reserve_bytes).is_err());
    let mut l = limits(); l.preparation_reserve_bytes = u64::MAX;
    assert!(l.reservation_bytes().is_err());
}

#[test]
fn one_full256_scope_prices_retained_namespace_capacity_and_rejects_the_wrong_commitment() {
    let mut c = config(127); c.batch.request.actions.default_action = RedactionAction::Pseudonymize;
    let mut namespace = String::with_capacity(4096); namespace.push_str("corpus-scope");
    let secret = RedactionPseudonyms { key: Arc::new(PseudonymKey::from_bytes(&[7; 32], "key-id").unwrap()), namespace };
    assert_eq!(secret_bytes(Some(&secret)).unwrap(), secret.namespace.capacity() as u64 + 2048);
    c.batch.request.actions.expected_key_commitment = Some(secret.key.commitment());
    let context = pseudonym_context(Some(&secret), &c.batch.request).unwrap().unwrap();
    assert_eq!(context.identity().key_commitment, secret.key.commitment());
    c.batch.request.actions.expected_key_commitment = Some("wrong-key-commitment".to_owned());
    assert!(pseudonym_context(Some(&secret), &c.batch.request).is_err());
    assert!(pseudonym_context(None, &c.batch.request).is_err());
    assert_eq!(secret_bytes(None).unwrap(), 0);
    let empty = RedactionPseudonyms { key: secret.key.clone(), namespace: String::new() };
    assert!(secret_bytes(Some(&empty)).is_err());
}

#[test]
fn owned_stream_capture_keeps_reader_writer_key_and_configuration_in_one_send_package() {
    fn send<T: Send + 'static>() {}
    send::<CorpusInput<std::io::Cursor<Vec<u8>>, Vec<u8>>>();
    send::<Preparation>();
    send::<RedactionPseudonyms>();
    send::<RedactionCorpusConfig<Int4DocumentRedactionBatchConfig>>();
    let _host = NlpEngine::batch_int4_redact_documents::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
}
