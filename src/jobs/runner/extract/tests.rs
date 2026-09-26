//! Real pinned planning and authenticated recipe binding, not model inference.
use super::*;
use crate::{
    canonjson,
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    jobs::{FrozenManifest, JobContract, JobId, JobInput, JobLimits, JobSecret, MismatchField},
    native_engine::{decode::DecodeCancellationKind, strict_int8::STRICT_INT8_EXECUTION},
    batch::extract::ExtractionBatchGrounding,
    tokenizer::{pinned_controls, embedded::EmbeddedTokenizer},
    tasks::ir::PromptSegmentKind,
};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 2048, max_output_tokens: 32, max_output_bytes: 65536,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"synthetic-durable-extract-factory");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "fixture-int8".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn native() -> Int8ExtractionBatchLimits {
    Int8ExtractionBatchLimits { max_model_work: Int8Work::for_sequence(0, 4096, 64 * 166144).unwrap(),
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits {
            max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 256 },
            max_visits_per_item: 100_000_000, max_visits_per_run: 200_000_000 } }
}
fn args(schema: &str, grounding: ExtractionBatchGrounding) -> ExtractionBatchArgs {
    ExtractionBatchArgs { schema: schema.to_owned(), grounding, budget: budget() }
}
fn factory(defaults: Option<ExtractionBatchArgs>) -> Int8ExtractionJobPlanner {
    let controls = pinned_controls::pinned().unwrap();
    Int8ExtractionJobPlanner::pinned(controls.template_controls(), 166101, identity(), budget(),
        CompileLimits::default(), SourceRuntimeLimits::default(), defaults, native()).unwrap()
}
fn document(text: &str, task_args: Option<ExtractionBatchArgs>) -> BatchDocument<ExtractionBatchArgs> {
    BatchDocument { id: "item".to_owned(), text: text.to_owned(), task_args }
}
fn job_limits() -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192, max_snapshot_bytes: 65536,
        max_result_bytes: 65536, max_spool_bytes: 1 << 20, max_materialized_bytes: 1 << 20,
        max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork { model: native().max_model_work, mask_node_visits: 200_000_000 } }
}
fn freeze(p: &Int8ExtractionJobPlanner, input: &[u8]) -> FrozenManifest {
    FrozenManifest::freeze(&JobSecret::from_bytes([7; 32]), JobContract { job_id: JobId([9; 16]),
        execution: p.execution_identity(), recipe: &p.recipe, limits: job_limits() },
        [JobInput { id: "item", original: input, normalized: input }], &mut Continue).unwrap()
}

#[test]
fn actual_template_and_tokenizer_are_frozen_not_the_callers_placeholders() {
    let p = factory(Some(args(r#"{"type":"string","maxLength":8}"#, ExtractionBatchGrounding::Structural)));
    let plan = p.compiler.prepare_with_control(document("Alice", None), &mut Continue).unwrap();
    assert_ne!(p.identity.template_digest, identity().template_digest);
    assert_ne!(p.identity.tokenizer_digest, identity().tokenizer_digest);
    assert_eq!(p.identity.template_digest, plan.execution_identity().template_digest);
    assert_eq!(p.identity.tokenizer_digest, plan.execution_identity().tokenizer_digest);
    assert_eq!(p.identity.logical_model_digest, identity().logical_model_digest);
    assert_eq!(p.identity.numerics_profile, NumericsProfile::StrictQuantized { version: 1 });
    plan.verify_identity(plan.execution_identity()).unwrap();
}

#[test]
fn schema_bytes_and_exact_decimal_constants_survive_the_entire_recipe() {
    let schema = " {\"type\":\"number\",\"const\":12345678901234567890123456789012345678} \n";
    let p = factory(Some(args(schema, ExtractionBatchGrounding::Structural)));
    let value = serde_json::to_value(&p.recipe).unwrap();
    assert_eq!(value["defaults"]["schema"].as_str(), Some(schema));
    let plan = p.compiler.prepare_with_control(document("exact number", None), &mut Continue).unwrap();
    assert_eq!(plan.execution_identity().schema_digest, Sha256Digest::of_bytes(schema.as_bytes()));
    assert_eq!(plan.extraction_plan().max_result_bytes(), budget().max_output_bytes);
}

#[test]
fn source_binding_and_control_exclusion_use_the_actual_original_record() {
    let schema = r#"{"type":"string","maxLength":64,"x-fnlp-source":"verbatim"}"#;
    let p = factory(Some(args(schema, ExtractionBatchGrounding::SourceMembership)));
    let text = "é Alice <tool_call> <|im_start|> 上海";
    let plan = p.compiler.prepare_with_control(document(text, None), &mut Continue).unwrap();
    assert_eq!(plan.source().text(), text);
    let controls = pinned_controls::pinned().unwrap();
    let t = EmbeddedTokenizer::pinned().unwrap();
    let segment = plan.task_plan().ir().prompt_segments().iter()
        .find(|s| s.kind() == PromptSegmentKind::Document).unwrap();
    assert!(segment.token_ids().iter().all(|id| !controls.template_controls().contains(*id)));
    assert_eq!(t.tokenizer().decode_bytes(segment.token_ids()).unwrap(), text.as_bytes());
    let other = p.compiler.prepare_with_control(document("Bob", None), &mut Continue).unwrap();
    assert_ne!(other.execution_identity().prompt_digest, plan.execution_identity().prompt_digest);
}

#[test]
fn exact_schema_defaults_and_original_records_are_authenticated_independently() {
    let a = factory(Some(args(r#"{"type":"string","maxLength":8}"#, ExtractionBatchGrounding::Structural)));
    let b = factory(Some(args(r#"{"type":"string","maxLength":9}"#, ExtractionBatchGrounding::Structural)));
    let original = br#"{"id":"item","text":"Alice"}"#;
    let first = freeze(&a, original);
    let same = freeze(&a, original);
    first.binding.compare(&same.binding).unwrap();
    assert_eq!(first.binding.compare(&freeze(&b, original).binding), Err(crate::jobs::JobError::Mismatch(MismatchField::Recipe)));
    assert_eq!(first.binding.compare(&freeze(&a, br#"{"id":"item","text":"Bob"}"#).binding),
        Err(crate::jobs::JobError::Mismatch(MismatchField::Population)));
}

#[test]
fn every_compiler_mask_and_native_work_axis_changes_the_frozen_recipe() {
    let original = || Int8ExtractionJobRecipe::new(budget(), CompileLimits::default(),
        SourceRuntimeLimits::default(), native(), None);
    let before = canonjson::canonical_bytes(&original()).unwrap();
    for axis in 0..15 {
        let mut c = CompileLimits::default(); let mut n = native();
        match axis {
            0 => c.max_schema_bytes += 1, 1 => c.max_string_bytes += 1,
            2 => c.max_array_items += 1, 3 => c.max_output_bytes += 1,
            4 => c.max_states += 1, 5 => c.max_transitions += 1, 6 => c.max_mask_bytes += 1,
            7 => n.max_model_work.forward_positions += 1, 8 => n.max_model_work.projected_logits += 1,
            9 => n.max_model_work.attention_pairs += 1, 10 => n.max_model_work.projections.dot_products += 1,
            11 => n.max_model_work.projections.multiply_accumulates += 1,
            12 => n.masks.per_mask.max_trie_node_visits += 1,
            13 => n.masks.per_mask.checkpoint_interval_nodes += 1, _ => n.masks.max_visits_per_item += 1,
        }
        let changed = Int8ExtractionJobRecipe::new(budget(), c, SourceRuntimeLimits::default(), n, None);
        assert_ne!(before, canonjson::canonical_bytes(&changed).unwrap(), "axis {axis}");
    }
    let mut changed = original(); changed.native.max_visits_per_run += 1;
    assert_ne!(before, canonjson::canonical_bytes(&changed).unwrap());
}

#[test]
fn defaults_overrides_do_not_mutate_future_recipe_or_other_records() {
    let p = factory(Some(args(r#"{"type":"string","maxLength":8}"#, ExtractionBatchGrounding::Structural)));
    let before = canonjson::canonical_bytes(&p.recipe).unwrap();
    let first = p.compiler.prepare_with_control(document("Alice", None), &mut Continue).unwrap();
    let override_args = args(r#"{"type":"integer"}"#, ExtractionBatchGrounding::Structural);
    let changed = p.compiler.prepare_with_control(document("Alice", Some(override_args)), &mut Continue).unwrap();
    let last = p.compiler.prepare_with_control(document("Alice", None), &mut Continue).unwrap();
    assert_ne!(first.execution_identity().schema_digest, changed.execution_identity().schema_digest);
    assert_eq!(first.execution_identity(), last.execution_identity());
    assert_eq!(before, canonjson::canonical_bytes(&p.recipe).unwrap());
}

#[test]
fn malformed_and_unfunded_factories_fail_before_native_construction() {
    let registry = pinned_controls::pinned().unwrap();
    for axis in 0..6 {
        let mut n = native();
        match axis { 0 => n.max_model_work.forward_positions = 0,
            1 => n.max_model_work.projected_logits = 0, 2 => n.max_model_work.attention_pairs = 0,
            3 => n.max_model_work.projections.dot_products = 0,
            4 => n.max_model_work.projections.multiply_accumulates = 0, _ => n.masks.max_visits_per_run = 1 }
        assert!(Int8ExtractionJobPlanner::pinned(registry.template_controls(), 166101, identity(), budget(),
            CompileLimits::default(), SourceRuntimeLimits::default(), None, n).is_err());
    }
    let mut id = identity(); id.numerics_profile = NumericsProfile::HfBf16Eager;
    assert!(Int8ExtractionJobPlanner::pinned(registry.template_controls(), 166101, id, budget(),
        CompileLimits::default(), SourceRuntimeLimits::default(), None, native()).is_err());
    let mut defaults = args(r#"{"type":"string"}"#, ExtractionBatchGrounding::Structural);
    defaults.budget.max_output_bytes += 1;
    assert!(Int8ExtractionJobPlanner::pinned(registry.template_controls(), 166101, identity(), budget(),
        CompileLimits::default(), SourceRuntimeLimits::default(), Some(defaults), native()).is_err());
}

#[test]
fn preparation_observes_shared_cancellation_and_missing_schema_is_not_invented() {
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
    }
    let p = factory(None);
    let stopped = p.compiler.prepare_with_control(document("Alice", None), &mut Stop).err().unwrap();
    assert!(stopped.stop); assert_eq!(stopped.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    assert!(p.compiler.prepare_with_control(document("Alice", None), &mut Continue).is_err());
    let duplicate = args(r#"{"type":"string","type":"integer"}"#, ExtractionBatchGrounding::Structural);
    assert!(p.compiler.prepare_with_control(document("Alice", Some(duplicate)), &mut Continue).is_err());
}
