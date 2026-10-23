//! Host configuration and admission checks with real pinned planner data.
//! Synthetic identities exercise policy only; no fixture supplies model or
//! inference authority to the public resident-engine entrypoint.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::BatchLimits,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::{answer::{AnswerOptions, AnswerPassage}, BuiltInTask},
    template::{IM_START, IM_END, THINK_START, THINK_END},
    tokenizer::{bpe::EncodeOptions, embedded::EmbeddedTokenizer, specials::ArchivedControlRegistries},
};

fn planner() -> SourceTaskPlanner {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = tokenizer.tokenizer().encode_ids_with_options(surface,
            EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let registry = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    SourceTaskPlanner::pinned(registry.template_controls(), tokenizer.eos_token_id().unwrap()).unwrap()
}
fn masks() -> ExtractionMaskBudget {
    ExtractionMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 },
        max_visits_per_item: 4000, max_visits_per_run: 16_000 }
}
fn config(planner: &SourceTaskPlanner, task: BuiltInTask, width: usize) -> Int4SourceCorpusConfig {
    let d = Sha256Digest::of_bytes(b"source-host-policy-fixture");
    Int4SourceCorpusConfig {
        identity: ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
            logical_model_digest: d, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
            quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
            packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
                INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width,
            )).unwrap()),
            tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
            task_spec: task.spec().identity(), taskir_digest: d, prompt_digest: d,
            grammar_compiler_version: "none".to_owned(), schema_digest: d,
            numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
            sampler_version: "constrained-json".to_owned(), thinking_mode: ThinkingMode::Disabled,
            tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
            backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None },
        task_ceiling: TaskBudget { max_input_tokens: 4096, max_output_tokens: 128,
            max_output_bytes: 65_536, max_grammar_states: 8192, max_kv_bytes: 4096 * KV_BYTES_PER_TOKEN as u64 },
        planning: SourcePlanningLimits::default(), defaults: None,
        native_work: Int4SourceBatchLimits { max_model_work: constrained_int4::planned_work(width, 8, 4).unwrap(),
            masks: masks() },
    }
}
fn request(config: &Int4SourceCorpusConfig) -> Int4SourceAdmission<'_> {
    Int4SourceAdmission { identity: &config.identity, model_work: config.native_work.max_model_work,
        kv_reservation_bytes: config.task_ceiling.max_kv_bytes, max_result_bytes: 65_536,
        mask_node_visits: masks().max_visits_per_item, mask_limits: masks().per_mask }
}
fn check(config: &Int4SourceCorpusConfig, request: &Int4SourceAdmission<'_>) -> Result<(), BatchItemFailure> {
    check_request_for_task(32, config.task_ceiling.max_kv_bytes, 65_536,
        masks(), request, &config.identity.task_spec)
}

#[test]
fn all_four_tasks_share_full_resident_kv_and_exact_task_admission() {
    let planner = planner();
    for task in [BuiltInTask::Ner, BuiltInTask::Keyphrases, BuiltInTask::Summarize, BuiltInTask::Answer] {
        let config = config(&planner, task, 32);
        config.validate_planning(&planner, 32, config.task_ceiling.max_kv_bytes).unwrap();
        check(&config, &request(&config)).unwrap();
        let mut substituted = config.identity.clone(); substituted.task_spec = "extract-v1".to_owned();
        let mut request = request(&config); request.identity = &substituted;
        let error = check(&config, &request).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
    }
}

#[test]
fn group_backend_mode_and_pinned_assets_must_match_before_allocation() {
    let planner = planner();
    for axis in 0..8 {
        let mut config = config(&planner, BuiltInTask::Answer, 32);
        match axis {
            0 => config.identity.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            1 => config.identity.backend_semantic_version.push('x'),
            2 => config.identity.template_digest = Sha256Digest::of_bytes(b"other-template"),
            3 => config.identity.tokenizer_digest = Sha256Digest::of_bytes(b"other-tokenizer"),
            4 => config.identity.task_spec = "extract-v1".to_owned(),
            5 => config.identity.thinking_mode = ThinkingMode::Enabled,
            6 => config.identity.tool_mode = ToolMode::Json,
            _ => config.identity.packing_set_digest = Sha256Digest::of_bytes(b"other-partition"),
        }
        assert!(config.validate_planning(&planner, 32, config.task_ceiling.max_kv_bytes).is_err());
    }
    let config = config(&planner, BuiltInTask::Ner, 16);
    assert!(config.validate_planning(&planner, 32, config.task_ceiling.max_kv_bytes).is_err());
}

#[test]
fn every_native_axis_and_mask_ceiling_is_nonzero_at_host_setup() {
    let planner = planner();
    for axis in 0..9 {
        let mut config = config(&planner, BuiltInTask::Keyphrases, 32);
        match axis {
            0 => config.native_work.max_model_work.forward_positions = 0,
            1 => config.native_work.max_model_work.projected_logits = 0,
            2 => config.native_work.max_model_work.attention_pairs = 0,
            3 => config.native_work.max_model_work.projections.dot_products = 0,
            4 => config.native_work.max_model_work.projections.multiply_accumulates = 0,
            5 => config.native_work.masks.per_mask.max_trie_node_visits = 0,
            6 => config.native_work.masks.per_mask.checkpoint_interval_nodes = 0,
            7 => config.native_work.masks.max_visits_per_item = 0,
            _ => config.native_work.masks.max_visits_per_run = config.native_work.masks.max_visits_per_item - 1,
        }
        assert!(config.validate_planning(&planner, 32, config.task_ceiling.max_kv_bytes).is_err());
    }
}

#[test]
fn complete_allocated_kv_must_fit_even_when_the_actual_request_is_short() {
    let planner = planner(); let config = config(&planner, BuiltInTask::Summarize, 32);
    let kv = config.task_ceiling.max_kv_bytes;
    for allocated in [0, kv + 1, u64::MAX] {
        assert!(config.validate_planning(&planner, 32, allocated).is_err());
    }
    for declared in [0, kv - 1, kv + 1] {
        let mut request = request(&config); request.kv_reservation_bytes = declared;
        assert_eq!(check(&config, &request).unwrap_err().fault.code, BatchCode::Admission);
    }
}

#[test]
fn defaults_cannot_switch_task_or_hide_unbounded_private_passages() {
    let planner = planner(); let mut config = config(&planner, BuiltInTask::Answer, 32);
    config.defaults = Some(SourceBatchArgs::Keyphrases { options: Default::default(), budget: config.task_ceiling });
    assert!(config.validate_planning(&planner, 32, config.task_ceiling.max_kv_bytes).is_err());
    config.defaults = Some(SourceBatchArgs::Answer { passages: vec![AnswerPassage {
        id: "passage".to_owned(), text: "x".repeat(source_batch::MAX_SOURCE_ARGUMENT_BYTES),
    }], options: AnswerOptions::default(), budget: config.task_ceiling });
    assert!(config.validate_planning(&planner, 32, config.task_ceiling.max_kv_bytes).is_err());
}

#[test]
fn source_admission_rechecks_group_work_masks_and_output_delivery_cap() {
    let planner = planner(); let config = config(&planner, BuiltInTask::Ner, 32);
    for axis in 0..5 {
        let mut request = request(&config);
        match axis {
            0 => request.model_work = constrained_int4::planned_work(16, 8, 4).unwrap(),
            1 => request.model_work.attention_pairs -= 1,
            2 => request.model_work.projections.multiply_accumulates -= 1,
            3 => request.mask_node_visits -= 1,
            _ => request.mask_limits.checkpoint_interval_nodes += 1,
        }
        let error = check(&config, &request).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidExecution);
    }
    let mut request = request(&config); request.max_result_bytes = 65_537;
    let error = check(&config, &request).unwrap_err();
    assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::OutputLineLimit);
}

#[test]
fn input_storage_and_public_corpus_entrypoint_cross_one_owned_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Charged<SourceInput<std::io::Cursor<Vec<u8>>, Vec<u8>>>>();
    send::<Int4SourceCorpusConfig>();
    let _entry = NlpEngine::batch_int4_source::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let planner = planner(); let config = config(&planner, BuiltInTask::Keyphrases, 32);
    let limits = CorpusLimits { native: NativeLimits { context_tokens: 4096, allocator_reserve_bytes: 65_536,
        run: RunLimits { max_elapsed: Duration::from_secs(10), max_checkpoints: 1000, cleanup_reserve_bytes: 65_536 } },
        transport: BatchLimits::default(), preparation_reserve_bytes: 1024 * 1024, io_reserve_bytes: 65_536 };
    let required = requirements(32, config.native_work, limits).unwrap();
    assert_eq!(required.input_bytes, limits.reservation_bytes().unwrap());
    assert_eq!(required.native.kv_bytes, config.task_ceiling.max_kv_bytes);
}
