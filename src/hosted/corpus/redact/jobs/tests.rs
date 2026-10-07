//! Real HMAC manifests and admission arithmetic, not native inference evidence.
use super::*;
use crate::{
    canonjson, execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    jobs::{FrozenManifest, JobContract, JobId, JobInput, JobLimits, JobSecret, JobError, MismatchField},
    native_engine::{constrained_int8, decode::DecodeCancellationKind, strict_int8::STRICT_INT8_EXECUTION},
    tasks::{ir::TaskBudget, ner::NerOptions, source_planning::SourcePlanningLimits,
        redact::{PiiKind, actions::RedactionAction, pseudonym::PseudonymKey, quantized::Int8RedactionConfig}},
};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn config() -> Int8RedactionBatchConfig {
    let d = Sha256Digest::of_bytes(b"durable-redaction-fixture");
    let id = ExecutionIdentity { schema_version: 1, source_revision: "fixture".into(), logical_model_digest: d,
        artifact_format: "fixture".into(), quant_recipe: "int8-fixture".into(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "ner-v1".into(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".into(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".into(),
        sampler_version: "fixture".into(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.into(),
        host_class: None, compiler_identity: None };
    let work = constrained_int8::planned_work(128, 16).unwrap();
    let work = work.checked_add(work).unwrap();
    let mut request = RedactionRequest::default(); request.edit_budget.max_output_bytes = 65536;
    Int8RedactionBatchConfig { ner_identity: id, detector: Int8RedactionConfig {
        ner: NerOptions::default(), per_pass: TaskBudget { max_input_tokens: 8192, max_output_tokens: 16,
            max_output_bytes: 65536, max_grammar_states: 4096, max_kv_bytes: 1 << 31 },
        planning: SourcePlanningLimits::default(), max_model_work: work, mask_limits: MaskWorkLimits::default(),
        mask_visits_per_pass: 1000, max_mask_visits: 2000, max_result_bytes: 65536 },
        request, max_model_work: work.checked_add(work).unwrap(), max_mask_visits: 4000 }
}
const INPUT: &[u8] = br#"{"id":"item","text":"Alice lives here."}"#;
fn freeze(config: &Int8RedactionBatchConfig, recipe: &RedactionJobRecipe, input: &[u8]) -> FrozenManifest {
    let limits = JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65536, max_result_bytes: 65536, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork { model: config.max_model_work, mask_node_visits: config.max_mask_visits } };
    FrozenManifest::freeze(&JobSecret::from_bytes([9; 32]), JobContract { job_id: JobId([8; 16]),
        execution: &config.ner_identity, recipe, limits },
        [JobInput { id: "item", original: input, normalized: input }], &mut Continue).unwrap()
}
#[test]
fn detector_rules_actions_maps_and_verification_are_authenticated() {
    let base = config(); let recipe = RedactionJobRecipe::short(&base, None).unwrap();
    let original = freeze(&base, &recipe, INPUT);
    original.binding.compare(&freeze(&base, &recipe, INPUT).binding).unwrap();
    for axis in 0..6 {
        let mut changed = config();
        match axis {
            0 => changed.request.verify = false,
            1 => changed.request.actions.include_map = true,
            2 => changed.request.actions.default_action = RedactionAction::Mask,
            3 => { changed.request.rules.enabled.clear(); },
            4 => changed.request.rule_budget.max_work += 1,
            _ => changed.request.edit_budget.max_regions -= 1,
        }
        let recipe = RedactionJobRecipe::short(&changed, None).unwrap();
        assert_eq!(original.binding.compare(&freeze(&base, &recipe, INPUT).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
    }
    assert_eq!(original.binding.compare(&freeze(&base, &recipe, br#"{"id":"item","text":"Bob"}"#).binding),
        Err(JobError::Mismatch(MismatchField::Population)));
}
#[test]
fn actual_key_and_namespace_not_just_public_key_label_are_frozen() {
    let config = config();
    let a = PseudonymKey::from_bytes(&[1; 32], "same-id").unwrap();
    let b = PseudonymKey::from_bytes(&[2; 32], "same-id").unwrap();
    let first = Pseudonyms::full256(&a, "original", None).unwrap();
    let recipe = RedactionJobRecipe::short(&config, Some(&first)).unwrap();
    let original = freeze(&config, &recipe, INPUT);
    for context in [Pseudonyms::full256(&b, "original", None).unwrap(), Pseudonyms::full256(&a, "changed", None).unwrap()] {
        let changed = RedactionJobRecipe::short(&config, Some(&context)).unwrap();
        assert_eq!(original.binding.compare(&freeze(&config, &changed, INPUT).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
    }
    let reopened = Pseudonyms::full256(&a, "original", None).unwrap();
    original.binding.compare(&freeze(&config, &RedactionJobRecipe::short(&config, Some(&reopened)).unwrap(), INPUT).binding).unwrap();
    assert_eq!(first.pseudonym(PiiKind::Person, "Alice").unwrap(), reopened.pseudonym(PiiKind::Person, "Alice").unwrap());
    let encoded = canonjson::canonical_string(&recipe).unwrap();
    assert!(!encoded.contains("original")); // Only the keyed scope commitment, not namespace/source/key bytes.
}
#[test]
fn omitted_wrong_and_unpreflighted_key_contexts_never_become_new_job_scopes() {
    let mut config = config(); config.request.actions.default_action = RedactionAction::Pseudonymize;
    assert!(RedactionJobRecipe::short(&config, None).is_err());
    let key = PseudonymKey::from_bytes(&[1; 32], "k").unwrap();
    let context = Pseudonyms::full256(&key, "scope", None).unwrap();
    config.request.actions.expected_key_commitment = Some("0".repeat(64));
    assert!(RedactionJobRecipe::short(&config, Some(&context)).is_err());
    config.request.actions.expected_key_commitment = Some(key.commitment());
    assert!(RedactionJobRecipe::short(&config, Some(&context)).is_ok());
    let short = Pseudonyms::preflight128(&key, "scope", &[(PiiKind::Person, "Alice")],
        crate::tasks::redact::pseudonym::PseudonymBudget::default(), None).unwrap();
    assert!(RedactionJobRecipe::short(&config, Some(&short)).is_err());
}
#[test]
fn planning_and_all_native_counters_are_bound_without_cloning_source() {
    let base = config(); let before = canonjson::canonical_bytes(&RedactionJobRecipe::short(&base, None).unwrap()).unwrap();
    for axis in 0..17 {
        let mut c = config();
        match axis {
            0 => c.detector.per_pass.max_input_tokens += 1,
            1 => c.detector.per_pass.max_output_tokens += 1,
            2 => c.detector.planning.compiler.max_states += 1,
            3 => c.detector.planning.compiler.max_transitions += 1,
            4 => c.detector.planning.compiler.max_mask_bytes += 1,
            5 => c.detector.planning.max_context_tokens += 1,
            6 => c.detector.mask_limits.checkpoint_interval_nodes += 1,
            7 => c.detector.mask_limits.max_trie_node_visits += 1,
            8 => c.detector.max_mask_visits += 1,
            9 => c.detector.max_result_bytes += 1,
            10 => c.max_model_work.forward_positions += 1,
            11 => c.max_model_work.projected_logits += 1,
            12 => c.max_model_work.attention_pairs += 1,
            13 => c.max_model_work.projections.dot_products += 1,
            14 => c.max_model_work.projections.multiply_accumulates += 1,
            15 => c.max_mask_visits += 1,
            _ => c.request.grounding_budget = crate::validation::grounded_fields::GroundingBudget::default(),
        }
        if axis == 16 { continue; } // Grounding limits are serialized in request, not a second projection.
        assert_ne!(before, canonjson::canonical_bytes(&RedactionJobRecipe::short(&c, None).unwrap()).unwrap());
    }
}
#[test]
fn durable_debit_reserves_both_passes_and_never_uses_observed_work() {
    let config = config(); let recipe = RedactionJobRecipe::short(&config, None).unwrap();
    let batch = BatchWork { forward_positions: config.detector.max_model_work.forward_positions,
        projected_logits: config.detector.max_model_work.projected_logits };
    let work = recipe.check_work(batch, false).unwrap();
    assert_eq!(work.model, config.detector.max_model_work); assert_eq!(work.mask_node_visits, 2000);
    assert!(recipe.check_work(batch, true).unwrap_err().stop);
    for wrong in [BatchWork { forward_positions: batch.forward_positions - 1, ..batch },
        BatchWork { projected_logits: batch.projected_logits - 1, ..batch }] {
        assert!(recipe.check_work(wrong, false).unwrap_err().stop);
    }
    let mut one = config; one.request.verify = false;
    assert_eq!(RedactionJobRecipe::short(&one, None).unwrap().check_work(batch, false).unwrap().mask_node_visits, 1000);
}
#[test]
fn full_context_outputs_and_edit_storage_are_checked_before_population_ingestion() {
    capacity(64, 4096, 8192, 1, 64, 4096, 8192).unwrap();
    for args in [(65, 4096, 8192, 1, 64, 4096, 8192), (64, 4095, 8192, 1, 64, 4096, 8192),
        (64, 4096, 8193, 1, 64, 4096, 8192), (64, 4096, 8192, 0, 64, 4096, 8192),
        (64, 4096, 8192, 1, 64, 0, 8192)] {
        assert!(capacity(args.0, args.1, args.2, args.3, args.4, args.5, args.6).is_err());
    }
}
#[test]
fn serialization_bound_counts_escaped_bytes_and_leaves_room_for_the_job_envelope() {
    assert!(recipe::bounded(&"x".repeat(RECIPE_BYTES - 2)).is_ok());
    assert!(recipe::bounded(&"x".repeat(RECIPE_BYTES - 1)).is_err());
    assert!(recipe::bounded(&"\n".repeat(RECIPE_BYTES / 2)).is_err());
}
#[test]
fn owned_redaction_jobs_cross_the_existing_runtime_boundary() {
    fn send<T: Send + 'static>() {}
    send::<StreamInput<(Arc<SourceTaskPlanner>, Arc<ExtractionVocabulary>, RedactionCorpusConfig,
        Option<RedactionPseudonyms>, SourceJobRequest), std::io::Cursor<Vec<u8>>, ()>>();
    let _entry = NlpEngine::job_int8_redact::<std::io::Cursor<Vec<u8>>>;
}
