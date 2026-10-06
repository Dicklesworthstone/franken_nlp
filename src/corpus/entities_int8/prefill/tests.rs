//! Real pinned preparation and zero-work finalization only; no model parity.
use super::*;
use crate::{execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::strict_int8::STRICT_INT8_EXECUTION, tokenizer::pinned_controls,
    corpus::entities_int8::long::{self, Int8DocumentEntityConfig, PreparedInt8DocumentEntityCorpus},
    tasks::mapreduce::ChunkLimits};

struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn limits(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
fn config() -> Int8EntityConfig {
    let cap = Int8Work::for_sequence(0, 16_384, 16_384 * 166_144).unwrap();
    Int8EntityConfig {
        ner: NerOptions::default(), ner_budget: TaskBudget { max_input_tokens: 8192, max_output_tokens: 64,
            max_output_bytes: 1 << 20, max_grammar_states: 100_000, max_kv_bytes: 2 * 1024 * 1024 * 1024 },
        source_planning: SourcePlanningLimits::default(), masks: SourceMaskBudget { per_mask: Default::default(),
            max_visits_per_item: 1_000_000_000, max_visits_per_run: 1_000_000_000_000 },
        resolution: ResolveOptions::default(), graph: ResolveLimits::default(),
        scoring: Int8ResolveLimits { planning: Default::default(), max_model_work: cap },
        verification: GroundingBudget::default(), max_model_work: cap, max_result_bytes: 16 << 20,
    }
}
fn planners() -> (Arc<SourceTaskPlanner>, ExecutionIdentity, Arc<ResolutionPlanner>, ExecutionIdentity) {
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    let source = Arc::new(SourceTaskPlanner::pinned(controls, eos).unwrap());
    let resolver = Arc::new(ResolutionPlanner::pinned(controls, eos).unwrap());
    let d = Sha256Digest::of_bytes(b"prefill-entity-planner-fixture");
    let a = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "fixture-int8".to_owned(), packing_set_digest: d,
        tokenizer_digest: source.tokenizer_digest(), template_digest: *source.template_digest(), task_spec: NER_TASK_VERSION.to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "fixture".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    let mut b = a.clone(); b.task_spec = resolve::RESOLVE_VERSION.to_owned(); b.template_digest = resolver.template_digest();
    (source, a, resolver, b)
}
fn documents() -> Vec<EntityDocument> {
    vec![EntityDocument { id: "z".into(), text: "Alice 上海 Alice".into() },
        EntityDocument { id: "a".into(), text: "Bob é".into() }]
}
fn prepare(documents: Vec<EntityDocument>) -> PreparedInt8EntityCorpus {
    let (source, a, resolver, b) = planners();
    prepare_int8_entities(documents, source, a, resolver, b, config(), &mut Continue).unwrap()
}
fn prepare_long(documents: Vec<EntityDocument>) -> PreparedInt8DocumentEntityCorpus {
    let (source, a, resolver, b) = planners();
    let config = Int8DocumentEntityConfig { entities: config(),
        chunks: ChunkLimits { max_input_bytes: 1 << 20, max_chunk_bytes: 6, max_chunk_tokens: 6,
            context_tokens: 8192, reserved_tokens: 64, max_chunks: 64, max_tokenizer_calls: 1024 },
        max_snapshot_chunks: 128 };
    long::prepare_int8_document_entities(documents, source, a, resolver, b, config, &mut Continue).unwrap()
}

#[test]
fn configuration_validates_native_geometry_before_changing_the_schedule() {
    let mut short = limits(4); short.max_extra_scratch_bytes -= 1;
    for invalid in [short, Int8PrefillLimits { max_batch_rows: 0, max_extra_scratch_bytes: u64::MAX },
        Int8PrefillLimits { max_batch_rows: 65, max_extra_scratch_bytes: u64::MAX }] {
        let mut schedule = None;
        assert!(configure(&mut schedule, invalid).is_err()); assert!(schedule.is_none());
    }
}
#[test]
fn repeated_configuration_cannot_replace_the_admission_geometry() {
    let mut schedule = None; configure(&mut schedule, limits(4)).unwrap();
    for next in [limits(4), limits(8)] {
        assert!(matches!(configure(&mut schedule, next), Err(Int8EntityError::InvalidLimits)));
        assert_eq!(schedule.unwrap().max_batch_rows, 4);
        assert_eq!(schedule.unwrap().max_extra_scratch_bytes, limits(4).max_extra_scratch_bytes);
    }
}
#[test]
fn whole_document_prefill_preserves_every_private_source_witness_and_both_stage_identities() {
    for rows in [1, 4, 64] {
        let plan = prepare(documents()); assert!(plan.prefill_limits().is_none());
        let source = plan.source_identity().clone(); let resolution = plan.resolution_identity().clone();
        let work = plan.ner_reserved_work(); let masks = plan.reserved_mask_visits();
        let context = plan.required_ner_context_tokens(); let payload = plan.retained_input_bytes().unwrap();
        let before: Vec<_> = plan.inputs.iter().map(|i| (i.witness, i.work, i.prompt_tokens,
            i.document.id.clone(), i.document.text.clone())).collect();
        let plan = plan.with_layer_major_prefill(limits(rows)).unwrap();
        let after: Vec<_> = plan.inputs.iter().map(|i| (i.witness, i.work, i.prompt_tokens,
            i.document.id.clone(), i.document.text.clone())).collect();
        assert_eq!(before, after); assert_eq!(plan.source_identity(), &source);
        assert_eq!(plan.resolution_identity(), &resolution); assert_eq!(plan.ner_reserved_work(), work);
        assert_eq!(plan.reserved_mask_visits(), masks); assert_eq!(plan.required_ner_context_tokens(), context);
        assert_eq!(plan.retained_input_bytes().unwrap(), payload);
        assert_eq!(plan.prefill_limits().unwrap().max_batch_rows, rows);
    }
}
#[test]
fn chunked_prefill_keeps_original_document_geometry_context_and_work() {
    let plan = prepare_long(documents()); assert!(plan.prefill_limits().is_none());
    assert!(plan.chunk_count() > plan.document_count());
    let source = plan.source_identity().clone(); let resolution = plan.resolution_identity().clone();
    let work = plan.ner_reserved_work(); let masks = plan.reserved_mask_visits();
    let context = plan.required_ner_context_tokens(); let chunks = plan.chunk_count();
    let count = plan.document_count(); let bytes = plan.input_bytes(); let retained = plan.retained_input_bytes().unwrap();
    let plan = plan.with_layer_major_prefill(limits(4)).unwrap();
    assert_eq!(plan.source_identity(), &source); assert_eq!(plan.resolution_identity(), &resolution);
    assert_eq!(plan.ner_reserved_work(), work); assert_eq!(plan.reserved_mask_visits(), masks);
    assert_eq!(plan.required_ner_context_tokens(), context); assert_eq!(plan.chunk_count(), chunks);
    assert_eq!(plan.document_count(), count); assert_eq!(plan.input_bytes(), bytes);
    assert_eq!(plan.retained_input_bytes().unwrap(), retained);
    assert_eq!(plan.prefill_limits().unwrap().max_batch_rows, 4);
}
#[test]
fn choosing_grouped_ner_never_refunds_its_reserved_work_to_resolution() {
    let plan = prepare(documents());
    let before = remaining_scoring(plan.config(), plan.ner_reserved_work()).unwrap().max_model_work;
    let plan = plan.with_layer_major_prefill(limits(4)).unwrap();
    let after = remaining_scoring(plan.config(), plan.ner_reserved_work()).unwrap().max_model_work;
    assert_eq!(after, before);
    assert_eq!(plus(after, plan.ner_reserved_work()).unwrap(), plan.config().max_model_work);
    let plan = prepare_long(documents());
    let before = remaining_scoring(&plan.config().entities, plan.ner_reserved_work()).unwrap().max_model_work;
    let plan = plan.with_layer_major_prefill(limits(4)).unwrap();
    assert_eq!(remaining_scoring(&plan.config().entities, plan.ner_reserved_work()).unwrap().max_model_work, before);
}
#[test]
fn empty_whole_and_chunked_snapshots_have_identical_zero_work_results() {
    let serial = prepare(Vec::new()).finalize_without_model(&mut Continue).unwrap();
    let grouped = prepare(Vec::new()).with_layer_major_prefill(limits(4)).unwrap()
        .finalize_without_model(&mut Continue).unwrap();
    assert_eq!(canonjson::canonical_bytes(&serial).unwrap(), canonjson::canonical_bytes(&grouped).unwrap());
    assert_eq!(grouped.model_work, Int8Work::default()); assert!(!grouped.resolution.model_evaluated);
    let serial = prepare_long(Vec::new()).finalize_without_model(&mut Continue).unwrap();
    let grouped = prepare_long(Vec::new()).with_layer_major_prefill(limits(4)).unwrap()
        .finalize_without_model(&mut Continue).unwrap();
    assert_eq!(canonjson::canonical_bytes(&serial).unwrap(), canonjson::canonical_bytes(&grouped).unwrap());
    assert_eq!(grouped.output.model_work, Int8Work::default()); assert!(!grouped.output.resolution.model_evaluated);
}
#[test]
fn grouped_nonempty_snapshots_still_require_real_ner_and_cannot_skip_to_clusters() {
    assert!(matches!(prepare(documents()).with_layer_major_prefill(limits(4)).unwrap()
        .finalize_without_model(&mut Continue), Err(Int8EntityError::Accounting)));
    assert!(matches!(prepare_long(documents()).with_layer_major_prefill(limits(4)).unwrap()
        .finalize_without_model(&mut Continue), Err(Int8EntityError::Accounting)));
}
#[test]
fn both_consumed_plan_builders_reject_repeated_schedule_changes() {
    assert!(matches!(prepare(Vec::new()).with_layer_major_prefill(limits(4)).unwrap()
        .with_layer_major_prefill(limits(8)), Err(Int8EntityError::InvalidLimits)));
    assert!(matches!(prepare_long(Vec::new()).with_layer_major_prefill(limits(4)).unwrap()
        .with_layer_major_prefill(limits(8)), Err(Int8EntityError::InvalidLimits)));
}
