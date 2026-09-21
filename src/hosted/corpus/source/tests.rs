//! Real pinned planner/configuration and host admission types, without loading
//! weights. Compile-time ownership checks are not actual runtime test receipts.
use super::*;
use std::io::Cursor;
use crate::{
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{kv::KV_BYTES_PER_TOKEN, strict_int8::STRICT_INT8_EXECUTION},
    tasks::{answer::{AnswerOptions, AnswerPassage}, BuiltInTask},
    template::{IM_START, IM_END, THINK_START, THINK_END},
    tokenizer::{bpe::EncodeOptions, embedded::EmbeddedTokenizer, specials::ArchivedControlRegistries},
};
fn planner() -> SourceTaskPlanner {
    let t = EmbeddedTokenizer::pinned().unwrap();
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = t.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let registry = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    SourceTaskPlanner::pinned(registry.template_controls(), t.eos_token_id().unwrap()).unwrap()
}
fn model() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: "fixture".to_owned(),
        recipe_id: "nanbeige42-int8-v1".to_owned(), source_root_sha256: "0".repeat(64), logical_model_sha256: "1".repeat(64) }
}
fn config(p: &SourceTaskPlanner, kind: BuiltInTask) -> SourceCorpusConfig {
    let m = model(); let d = Sha256Digest::of_bytes(b"source-host-fixture");
    SourceCorpusConfig {
        identity: ExecutionIdentity { schema_version: 1, source_revision: m.revision,
            logical_model_digest: Sha256Digest::from_hex(&m.logical_model_sha256).unwrap(),
            artifact_format: "fixture".to_owned(), quant_recipe: m.recipe_id, packing_set_digest: d,
            tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(), task_spec: kind.spec().identity(),
            taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
            numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
            thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d,
            decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None },
        task_ceiling: TaskBudget { max_input_tokens: 4096, max_output_tokens: 128, max_output_bytes: 65536,
            max_grammar_states: 8192, max_kv_bytes: 4096 * KV_BYTES_PER_TOKEN as u64 },
        planning: SourcePlanningLimits::default(), defaults: None,
        native_work: Int8SourceBatchLimits {
            max_model_work: Int8Work::for_sequence(0, 32, 4 * NANBEIGE_VOCAB_SIZE).unwrap(),
            masks: crate::batch::source::SourceMaskBudget {
                per_mask: crate::grammar::mask::MaskWorkLimits { max_trie_node_visits: 10000, checkpoint_interval_nodes: 64 },
                max_visits_per_item: 100000, max_visits_per_run: 1000000 } },
    }
}
#[test]
fn every_source_family_uses_the_same_actual_model_and_native_admission_contract() {
    let p = planner();
    for kind in [BuiltInTask::Ner, BuiltInTask::Keyphrases, BuiltInTask::Summarize, BuiltInTask::Answer] {
        let c = config(&p, kind); let kv = c.task_ceiling.max_kv_bytes;
        c.validate(&p, &model(), kv).unwrap();
        check_request(&model(), kv, 0, 65536, &c.identity, c.native_work.max_model_work, kv, 0, 65536).unwrap();
    }
}
#[test]
fn wrong_model_backend_task_or_pinned_assets_are_not_repaired_by_the_host() {
    let p = planner();
    for axis in 0..7 {
        let mut c = config(&p, BuiltInTask::Keyphrases);
        match axis { 0 => c.identity.source_revision.push('x'), 1 => c.identity.quant_recipe.push('x'),
            2 => c.identity.logical_model_digest = Sha256Digest::of_bytes(b"different"),
            3 => c.identity.numerics_profile = NumericsProfile::HfBf16Eager,
            4 => c.identity.backend_semantic_version.push('x'), 5 => c.identity.template_digest = Sha256Digest::of_bytes(b"wrong"),
            _ => c.identity.task_spec = "extract-v1".to_owned() }
        assert!(c.validate(&p, &model(), c.task_ceiling.max_kv_bytes).is_err());
    }
}
#[test]
fn whole_kv_reservation_and_each_native_mask_axis_are_checked_before_allocation() {
    let p = planner(); let c = config(&p, BuiltInTask::Answer);
    assert!(c.validate(&p, &model(), 0).is_err());
    assert!(c.validate(&p, &model(), c.task_ceiling.max_kv_bytes + 1).is_err());
    for axis in 0..5 {
        let mut c = config(&p, BuiltInTask::Answer);
        match axis { 0 => c.native_work.max_model_work.attention_pairs = 0,
            1 => c.native_work.max_model_work.projections.multiply_accumulates = 0,
            2 => c.native_work.masks.max_visits_per_run = 0, 3 => c.native_work.masks.max_visits_per_item = 0,
            _ => c.native_work.masks.per_mask.checkpoint_interval_nodes = 0 }
        assert!(c.validate(&p, &model(), c.task_ceiling.max_kv_bytes).is_err());
    }
}
#[test]
fn unbounded_default_passages_and_cross_task_defaults_refuse_at_host_preflight() {
    let p = planner(); let mut c = config(&p, BuiltInTask::Answer);
    c.defaults = Some(SourceBatchArgs::Answer { passages: vec![AnswerPassage { id: "p1".to_owned(),
        text: "x".repeat(source_batch::MAX_SOURCE_ARGUMENT_BYTES) }], options: AnswerOptions { max_answer_scalars: 128, max_citations: 4, max_quote_scalars: 32 }, budget: c.task_ceiling });
    assert!(c.validate(&p, &model(), c.task_ceiling.max_kv_bytes).is_err());
    c.defaults = Some(SourceBatchArgs::Keyphrases { options: Default::default(), budget: c.task_ceiling });
    assert!(c.validate(&p, &model(), c.task_ceiling.max_kv_bytes).is_err());
}
#[test]
fn corpus_outputs_use_reserved_kv_and_cannot_acquire_an_unpriced_sampler() {
    let p = planner(); let c = config(&p, BuiltInTask::Summarize); let kv = c.task_ceiling.max_kv_bytes;
    for (request_kv, sampler) in [(kv - 1, 0), (kv + 1, 0), (kv, 1)] {
        let error = check_request(&model(), kv, 0, 65536, &c.identity, c.native_work.max_model_work,
            request_kv, sampler, 65536).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
    }
    let error = check_request(&model(), kv, 0, 65536, &c.identity, c.native_work.max_model_work, kv, 0, 65537).unwrap_err();
    assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::OutputLineLimit);
}
#[test]
fn input_package_is_owned_send_and_the_public_method_uses_the_same_corpus_types() {
    fn send<T: Send + 'static>() {}
    type Input = StreamInput<(Arc<SourceTaskPlanner>, Arc<ExtractionVocabulary>, SourceCorpusConfig), Cursor<Vec<u8>>, Vec<u8>>;
    send::<Charged<Input>>(); send::<SourceCorpusConfig>();
    let _entry = NlpEngine::batch_int8_source::<Cursor<Vec<u8>>, Vec<u8>>;
}
#[test]
fn actual_request_arguments_stay_data_and_never_deserialize_execution_identity() {
    let p = planner(); let c = config(&p, BuiltInTask::Keyphrases);
    let mut args = serde_json::to_value(SourceBatchArgs::Keyphrases { options: Default::default(), budget: c.task_ceiling }).unwrap();
    args["identity"] = serde_json::to_value(c.identity).unwrap();
    assert!(serde_json::from_value::<SourceBatchArgs>(args).is_err());
}
