//! Admission and completion validation only; no neural execution is simulated.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{decode::DecodeCancellationKind, generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::{ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::{SourcePlanningLimits, grouped_int4::{Int4SourceError,
            long::Int4SourceMapLimits}},
        redact::{actions::RedactionAction, grouped_int4::long::Int4LongRedactionPreflight,
            pipeline::LeakReport, pseudonym::PseudonymKey, redact_rules}},
};

fn config(width: usize) -> RedactConfig<Int4LongRedactionConfig> {
    let d = Sha256Digest::of_bytes(b"int4-document-redaction-host-fixture");
    let context = 128;
    let mut request = RedactionRequest::default();
    request.actions.default_action = RedactionAction::Mask;
    request.edit_budget.max_output_bytes = 65_536;
    request.rule_budget.max_input_bytes = 65_536;
    let per = constrained_int4::planned_work(width, 64, 64).unwrap();
    let mut complete = per;
    for _ in 1..6 { complete = complete.checked_add(per).unwrap(); }
    RedactConfig {
        ner_identity: ExecutionIdentity {
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
        },
        request,
        detector: Int4LongRedactionConfig {
            ner: Default::default(),
            per_chunk: TaskBudget { max_input_tokens: 64, max_output_tokens: 64, max_output_bytes: 65_536,
                max_grammar_states: 4096, max_kv_bytes: Int4MemoryRequirement::for_context(context, width).unwrap().kv_bytes },
            planning: SourcePlanningLimits { max_input_bytes: 128, max_context_tokens: context, ..Default::default() },
            mapping: Int4SourceMapLimits {
                chunks: ChunkLimits { max_input_bytes: 65_536, max_chunk_bytes: 128,
                    max_chunk_tokens: 64, context_tokens: context, reserved_tokens: 64,
                    max_chunks: 3, ..Default::default() },
                reduction: ExecutionLimits { max_value_bytes: 65_536, max_live_value_bytes: 131_072,
                    max_result_bytes: 65_536, ..Default::default() },
                max_model_work: complete, mask_limits: Default::default(),
                mask_visits_per_chunk: 1000, max_mask_visits: 6000,
                verification: GroundingBudget { max_fields: 32, max_matches: 128, max_scan_steps: 1_000_000 },
            },
            max_result_bytes: 65_536,
        },
        native: NativeLimits { context_tokens: context, allocator_reserve_bytes: 65_536,
            run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100_000, cleanup_reserve_bytes: 65_536 } },
        preparation_reserve_bytes: 1 << 20, edit_reserve_bytes: 1 << 20,
    }
}
fn check(c: &RedactConfig<Int4LongRedactionConfig>, bytes: usize, width: usize)
    -> Result<Int4MemoryRequirement, HostedError> {
    document_requirements(c, bytes, width, c.ner_identity.tokenizer_digest, c.ner_identity.template_digest)
}

#[test]
fn document_exceeds_one_chunk_but_actual_group_and_full_resident_kv_still_bind() {
    for width in [3, 127, 128, 10752] {
        let mut c = config(width);
        let required = check(&c, 256, width).unwrap();
        assert_eq!(required, Int4MemoryRequirement::for_context(c.native.context_tokens, width).unwrap());
        assert!(c.detector.mapping.max_model_work.forward_positions > c.native.context_tokens as u64);
        assert!(check(&c, 256, if width == 127 { 128 } else { 127 }).is_err());
        c.detector.per_chunk.max_kv_bytes = Int4MemoryRequirement::for_context(64, width).unwrap().kv_bytes;
        assert!(check(&c, 256, width).is_err(), "every chunk prices the complete allocated KV");
    }
}

#[test]
fn task_profile_recipe_backend_and_assets_cannot_be_relabelled() {
    for axis in 0..5 {
        let mut c = config(127);
        match axis {
            0 => c.ner_identity.task_spec = "redact-v1".to_owned(),
            1 => c.ner_identity.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            2 => c.ner_identity.backend_semantic_version = "other".to_owned(),
            3 => c.ner_identity.kv_dtype = "int8".to_owned(),
            _ => c.ner_identity.packing_set_digest = Sha256Digest::of_bytes(b"foreign"),
        }
        assert!(matches!(check(&c, 256, 127), Err(HostedError::ModelIdentity)));
    }
    let c = config(127); let other = Sha256Digest::of_bytes(b"other-assets");
    assert!(document_requirements(&c, 256, 127, other, c.ner_identity.template_digest).is_err());
    assert!(document_requirements(&c, 256, 127, c.ner_identity.tokenizer_digest, other).is_err());
}

#[test]
fn whole_source_chunk_context_preparation_and_final_envelope_limits_refuse() {
    assert!(check(&config(127), 0, 127).is_err());
    assert!(check(&config(127), 65_537, 127).is_err());
    for axis in 0..12 {
        let mut c = config(127);
        match axis {
            0 => c.preparation_reserve_bytes = 0,
            1 => c.edit_reserve_bytes = 0,
            2 => c.native.allocator_reserve_bytes = 0,
            3 => c.request.rule_budget.max_input_bytes = 255,
            4 => c.detector.planning.max_input_bytes = 127,
            5 => c.detector.planning.max_context_tokens = 129,
            6 => c.detector.mapping.chunks.context_tokens = 129,
            7 => c.detector.mapping.chunks.reserved_tokens = 63,
            8 => c.detector.mapping.reduction.max_value_bytes = 65_535,
            9 => c.detector.max_result_bytes = 65_535,
            10 => c.detector.mapping.mask_limits.max_trie_node_visits = 0,
            _ => c.detector.mapping.mask_limits.checkpoint_interval_nodes = 0,
        }
        assert!(check(&c, 256, 127).is_err(), "axis {axis}");
    }
}

#[test]
fn all_five_native_axes_and_both_stage_mask_allowance_are_nonzero_and_checked() {
    for axis in 0..5 {
        let mut c = config(127);
        let work = &mut c.detector.mapping.max_model_work;
        match axis {
            0 => work.forward_positions = 0,
            1 => work.projected_logits = 0,
            2 => work.attention_pairs = 0,
            3 => work.projections.dot_products = 0,
            _ => work.projections.multiply_accumulates = 0,
        }
        assert!(check(&c, 256, 127).is_err());
    }
    let mut c = config(127);
    c.detector.mapping.max_mask_visits = 1000;
    assert!(check(&c, 256, 127).is_err());
    c.request.verify = false; check(&c, 256, 127).unwrap();
    c.request.verify = true; c.detector.mapping.mask_visits_per_chunk = u64::MAX;
    c.detector.mapping.max_mask_visits = u64::MAX;
    assert!(check(&c, 256, 127).is_err());
    for axis in 0..3 {
        let mut c = config(127);
        match axis { 0 => c.detector.mapping.verification.max_fields = 0,
            1 => c.detector.mapping.verification.max_matches = 1_000_001,
            _ => c.detector.mapping.verification.max_scan_steps = 0 }
        assert!(check(&c, 256, 127).is_err());
    }
}

#[test]
fn one_complete_map_and_both_coordinate_representations_are_reserved_at_peak() {
    let c = config(127);
    let expected = 4 * (3 * 65_536 + 131_072) + 8 * 64 * 3
        + 32 * std::mem::size_of::<SourceFieldEvidence>() as u64
        + 2 * 128 * std::mem::size_of::<VerifiedSourceSpan>() as u64 + c.edit_reserve_bytes;
    assert_eq!(document_temporary_bytes(&c).unwrap(), expected);
    for axis in 0..5 {
        let mut c = config(127);
        match axis {
            0 => c.detector.mapping.chunks.max_chunks = 257,
            1 => c.detector.mapping.reduction.max_live_value_bytes = 0,
            2 => c.detector.per_chunk.max_output_bytes = u64::MAX,
            3 => c.detector.mapping.reduction.max_live_value_bytes = usize::MAX,
            _ => c.edit_reserve_bytes = u64::MAX,
        }
        assert!(document_temporary_bytes(&c).is_err(), "axis {axis}");
    }
}

#[test]
fn captured_source_and_secret_capacity_remain_separate_from_map_workspace() {
    let c = config(127);
    let mut source = String::with_capacity(8192); source.push_str("Alice");
    let mut namespace = String::with_capacity(4096); namespace.push_str("corpus");
    let secret = RedactionPseudonyms { key: Arc::new(PseudonymKey::from_bytes(&[7; 32], "key").unwrap()), namespace };
    assert_eq!(input_bytes(&source, &c, None).unwrap(), source.capacity() as u64 + c.preparation_reserve_bytes);
    assert_eq!(input_bytes(&source, &c, Some(&secret)).unwrap(), source.capacity() as u64
        + c.preparation_reserve_bytes + secret.namespace.capacity() as u64 + 2048);
    let mut overflow = config(127); overflow.preparation_reserve_bytes = u64::MAX;
    assert!(input_bytes(&source, &overflow, Some(&secret)).is_err());
}

// The native-shaped receipt below is only a completion-validator input.
// Its edited text comes from the real rules-only pipeline, not model inference.
fn completed(c: &RedactConfig<Int4LongRedactionConfig>, source: &str) -> Int4LongRedactionRun {
    let mut result = redact_rules(source, &c.request, None).unwrap();
    result.model_types = c.detector.ner.types.iter().copied().collect();
    let width = 127;
    let stage = |text: &str, chunks: usize| {
        let one = constrained_int4::planned_work(width, 7, 8).unwrap();
        let work = if chunks == 2 { one.checked_add(one).unwrap() } else { one };
        Int4LongRedactionStage {
            preflight: Int4LongRedactionPreflight { source_bytes: text.len(), source_scalars: text.chars().count(),
                chunks, group_width: width, planned_model_work: work, reserved_mask_node_visits: chunks as u64 * 1000 },
            model_work: work, mask_node_visit_charge: chunks as u64 * 500,
            map_verification_used: SourceMapVerification { fields: 1, matches: 1, scan_steps: 10 },
            detection_verification_used: SourceMapVerification { fields: 1, matches: 1, scan_steps: 11 },
        }
    };
    let original = stage(source, 2);
    let verification = c.request.verify.then(|| stage(result.text(), 1));
    let reserved_model_work = verification.as_ref().map_or(original.preflight.planned_model_work,
        |v| original.preflight.planned_model_work.checked_add(v.preflight.planned_model_work).unwrap());
    let reserved_masks = original.preflight.reserved_mask_node_visits
        + verification.as_ref().map_or(0, |v| v.preflight.reserved_mask_node_visits);
    let masks = original.mask_node_visit_charge + verification.as_ref().map_or(0, |v| v.mask_node_visit_charge);
    let map = verification.as_ref().map_or(original.map_verification_used,
        |v| add_verification(original.map_verification_used, v.map_verification_used).unwrap());
    let detection = verification.as_ref().map_or(original.detection_verification_used,
        |v| add_verification(original.detection_verification_used, v.detection_verification_used).unwrap());
    Int4LongRedactionRun { schema_version: 1, execution: INT4_LONG_REDACTION_EXECUTION,
        numerics_profile: STRICT_INT4_PROFILE, group_width: width,
        detector_scope: "whole-source-rules-independent-ner-chunks-v1", result, original, verification,
        reserved_model_work, model_work: reserved_model_work, reserved_mask_node_visits: reserved_masks,
        mask_node_visit_charge: masks, map_verification_used: map, detection_verification_used: detection,
        verification_scan_steps: detection.scan_steps,
        warnings: ["ner_chunk_boundaries_may_split_entities", "detector_recall_not_established",
            "clean_scan_and_pseudonyms_are_not_anonymization"],
    }
}

#[test]
fn completed_validation_binds_actual_transformed_geometry_and_optional_verification() {
    let source = "é Alice a@example.org 上海";
    for verify in [true, false] {
        let mut c = config(127); c.request.verify = verify;
        let run = completed(&c, source);
        validate_completed(&run, &c, source, 127).unwrap();
        assert!(!run.result.text().contains("a@example.org"));
        assert_eq!(run.verification.is_some(), verify);
        if let Some(stage) = &run.verification {
            assert_eq!(stage.preflight.source_bytes, run.result.text().len());
            assert_eq!(stage.preflight.source_scalars, run.result.text().chars().count());
        }
    }
}

#[test]
fn corrupt_stage_geometry_policy_and_aggregate_receipts_never_become_success() {
    let source = "é Alice a@example.org 上海";
    let c = config(127);
    for axis in 0..15 {
        let mut run = completed(&c, source);
        match axis {
            0 => run.schema_version = 2,
            1 => run.group_width = 128,
            2 => run.original.preflight.source_bytes += 1,
            3 => run.original.preflight.source_scalars += 1,
            4 => run.original.preflight.chunks = 4,
            5 => run.original.preflight.reserved_mask_node_visits += 1,
            6 => run.verification.as_mut().unwrap().preflight.source_bytes += 1,
            7 => run.result.verification = VerificationStatus::NotRequested,
            8 => run.model_work.forward_positions += 1,
            9 => run.reserved_mask_node_visits += 1,
            10 => run.map_verification_used.fields += 1,
            11 => run.detection_verification_used.matches += 1,
            12 => run.verification_scan_steps += 1,
            13 => run.result.model_types.clear(),
            _ => run.warnings[0] = "no-warning",
        }
        assert!(validate_completed(&run, &c, source, 127).is_err(), "axis {axis}");
    }
}

#[test]
fn every_stage_actual_axis_must_fit_its_own_reservation_even_with_global_slack() {
    let c = config(127); let source = "Alice";
    for axis in 0..5 {
        let mut run = completed(&c, source);
        let actual = &mut run.original.model_work;
        match axis {
            0 => actual.forward_positions += 1,
            1 => actual.projected_logits += 1,
            2 => actual.attention_pairs += 1,
            3 => actual.projections.dot_products += 1,
            _ => actual.projections.multiply_accumulates += 1,
        }
        run.model_work = run.original.model_work.checked_add(run.verification.as_ref().unwrap().model_work).unwrap();
        assert!(within(run.model_work, c.detector.mapping.max_model_work));
        assert!(validate_completed(&run, &c, source, 127).is_err());
    }
}

#[test]
fn independent_grounding_pools_and_native_mask_limits_never_renew_between_stages() {
    let source = "Alice"; let c = config(127);
    let run = completed(&c, source);
    for axis in 0..4 {
        let mut short = config(127);
        match axis {
            0 => short.detector.mapping.verification.max_fields = 1,
            1 => short.request.grounding_budget.max_scan_steps = 21,
            2 => short.detector.mapping.max_mask_visits = 2999,
            _ => short.detector.mapping.max_model_work = run.original.preflight.planned_model_work,
        }
        assert!(validate_completed(&run, &short, source, 127).is_err(), "axis {axis}");
    }
    assert!(add_verification(SourceMapVerification { fields: usize::MAX, matches: 0, scan_steps: 0 },
        SourceMapVerification { fields: 1, matches: 0, scan_steps: 0 }).is_err());
    assert!(add_verification(SourceMapVerification { fields: 0, matches: 0, scan_steps: u64::MAX },
        SourceMapVerification { fields: 0, matches: 0, scan_steps: 1 }).is_err());
}

#[test]
fn residuals_collapse_under_the_output_guard_and_both_error_branches_preserve_cancel_kind() {
    let c = config(127);
    let error = document_error(Int4LongRedactionError::Redaction(Int4RedactionError::Residual(LeakReport {
        schema_version: 1, residuals: Vec::new(), rules: c.request.rules, model_types: Default::default(),
    })));
    assert!(matches!(error, HostedError::Int4(Int4HostError::Redaction(
        Int4RedactionError::Redaction(RedactError::VerificationResidual { count: 0 })))));
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::CostBudget] {
        let HostedError::Int4(Int4HostError::Redaction(inner)) =
            document_error(Int4LongRedactionError::Redaction(Int4RedactionError::Cancelled(cause)))
            else { panic!("redaction cause lost"); };
        assert_eq!(inner.cancellation(), Some(cause));
        let HostedError::Int4(Int4HostError::SourceMap(inner)) =
            document_error(Int4LongRedactionError::Map(Int4SourceError::Cancelled(cause).into()))
            else { panic!("source-map cause lost"); };
        assert_eq!(inner.cancellation(), Some(cause));
    }
}

#[test]
fn complete_owned_input_and_result_cross_one_physical_completion_boundary() {
    fn send<T: Send + 'static>() {}
    send::<RedactInput<Int4LongRedactionConfig>>();
    send::<RedactionPseudonyms>();
    send::<HostedOutput<Int4LongRedactionRun>>();
    let _host = NlpEngine::redact_int4_document;
}

