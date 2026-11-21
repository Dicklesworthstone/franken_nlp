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

mod encoding {
    use super::*;
    use crate::tokenizer::untrusted::SafeBpeLimits;
    const SCHEMA: &str = " {\"type\":\"string\",\"maxLength\":64,\"x-fnlp-source\":\"verbatim\"} \n";
    fn call() -> SafeBpeLimits {
        SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 2048,
            max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 }
    }
    fn total() -> SafeBpeWorkLimits {
        SafeBpeWorkLimits { max_input_bytes: 65536, max_output_tokens: 32768,
            max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 }
    }
    fn compact(call: SafeBpeLimits, total: SafeBpeWorkLimits) -> Int8ExtractionJobPlanner {
        factory(Some(args(SCHEMA, ExtractionBatchGrounding::SourceMembership)))
            .with_document_encoding(UntrustedEncoding::SafeBpeV1(call), Some(total)).unwrap()
    }
    fn prepare(p: &mut Int8ExtractionJobPlanner, text: &str, control: &mut impl DecodeStepControl)
        -> Result<PreparedInt8BatchExtraction, BatchItemFailure> {
        p.compiler.prepare_with_invocation_encoding_work(document(text, None),
            &mut p.encoding_work, p.recipe.encoding_work_limits, control)
    }

    #[test]
    fn byte_recipe_stays_unchanged_and_compact_requires_a_cumulative_owner() {
        let original = factory(None);
        let bytes = canonjson::canonical_bytes(&original.recipe).unwrap();
        let explicit = factory(None).with_document_encoding(UntrustedEncoding::ByteFallback, None).unwrap();
        assert_eq!(bytes, canonjson::canonical_bytes(&explicit.recipe).unwrap());
        assert_eq!(explicit.encoding_workspace_bytes().unwrap(), 0);
        let value = serde_json::to_value(&explicit.recipe).unwrap();
        assert!(value.get("input_encoding").is_none());
        assert!(value.get("encoding_work_limits").is_none());
        assert!(factory(None).with_document_encoding(UntrustedEncoding::SafeBpeV1(call()), None).is_err());
        assert!(factory(None).with_document_encoding(UntrustedEncoding::ByteFallback, Some(total())).is_err());
        let p = compact(call(), total());
        assert_eq!(p.encoding_workspace_bytes().unwrap(), call().requirements().unwrap().workspace_bytes);
        assert!(p.compiler.prepare(document("Alice", None)).is_err());
    }

    #[test]
    fn every_fixed_call_and_invocation_ceiling_authenticates_before_resume() {
        let original = br#"{"id":"item","text":"Alice"}"#;
        let p = compact(call(), total());
        let frozen = freeze(&p, original);
        for axis in 0..8 {
            let mut call = call(); let mut total = total();
            match axis {
                0 => call.max_input_bytes -= 1, 1 => call.max_output_tokens -= 1,
                2 => call.max_pair_checks -= 1, 3 => call.max_pair_bytes -= 1,
                4 => total.max_input_bytes -= 1, 5 => total.max_output_tokens -= 1,
                6 => total.max_pair_checks -= 1, _ => total.max_pair_bytes -= 1,
            }
            let changed = compact(call, total);
            assert_eq!(frozen.binding.compare(&freeze(&changed, original).binding),
                Err(crate::jobs::JobError::Mismatch(MismatchField::Recipe)), "axis {axis}");
        }
    }

    #[test]
    fn one_ledger_spans_exact_schema_source_and_factory_handoff_without_identity_drift() {
        let text = "é Alice <tool_call> <|im_start|> 上海\r\n";
        let record_bytes = SCHEMA.len() + text.len();
        let mut cap = total(); cap.max_input_bytes = (2 * record_bytes) as u64;
        let mut p = compact(call(), cap);
        let recipe = canonjson::canonical_bytes(&p.recipe).unwrap();
        let first = prepare(&mut p, text, &mut Continue).unwrap();
        let spent = p.encoding_work;
        assert_eq!(spent.input_bytes(), record_bytes);
        assert_eq!(first.source().text(), text);
        assert_eq!(first.source().document_encoding(), &UntrustedEncoding::SafeBpeV1(call()));
        assert_eq!(first.execution_identity().schema_digest, Sha256Digest::of_bytes(SCHEMA.as_bytes()));
        let mut moved = p;
        let second = prepare(&mut moved, text, &mut Continue).unwrap();
        assert_eq!(first.execution_identity(), second.execution_identity());
        assert_eq!(moved.encoding_work, spent.checked_add(spent).unwrap());
        assert_eq!(recipe, canonjson::canonical_bytes(&moved.recipe).unwrap());
        let exhausted = moved.encoding_work;
        let error = prepare(&mut moved, text, &mut Continue).err().unwrap();
        assert_eq!(error.fault.code, BatchCode::WorkLimit); assert!(error.stop);
        assert_eq!(moved.encoding_work, exhausted);
        let mut changed = cap; changed.max_pair_bytes += 1;
        assert!(moved.with_document_encoding(UntrustedEncoding::SafeBpeV1(call()), Some(changed)).is_err());
    }

    #[test]
    fn failed_encoding_spends_completed_work_without_a_fresh_record_allowance() {
        let text = "Alice private café Shanghai";
        let mut tight = call(); tight.max_output_tokens = 1;
        let mut p = compact(tight, total());
        assert!(prepare(&mut p, text, &mut Continue).is_err());
        let first = p.encoding_work;
        assert!(first.input_bytes() > 0 && first.output_tokens() > 0);
        assert!(prepare(&mut p, text, &mut Continue).is_err());
        assert_eq!(p.encoding_work, first.checked_add(first).unwrap());
    }

    #[test]
    fn cancellation_after_encoding_keeps_spent_work_and_every_original_cause() {
        struct Count { seen: usize, stop: usize, cause: DecodeCancellationKind }
        impl DecodeStepControl for Count {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
                self.seen += 1; (self.seen == self.stop).then_some(self.cause)
            }
        }
        let text = "Alice <think> é";
        let mut cap = total(); cap.max_input_bytes = (SCHEMA.len() + text.len()) as u64;
        let mut counter = Count { seen: 0, stop: usize::MAX, cause: DecodeCancellationKind::User };
        let mut baseline = compact(call(), cap);
        prepare(&mut baseline, text, &mut counter).unwrap();
        use DecodeCancellationKind::*;
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let mut p = compact(call(), cap);
            let error = prepare(&mut p, text, &mut Count { seen: 0, stop: counter.seen, cause }).err().unwrap();
            assert_eq!(error.fault.cancellation, Some(cause)); assert!(error.stop);
            assert_eq!(p.encoding_work, baseline.encoding_work);
            let spent = p.encoding_work;
            assert_eq!(prepare(&mut p, text, &mut Continue).err().unwrap().fault.code, BatchCode::WorkLimit);
            assert_eq!(p.encoding_work, spent);
        }
    }
}
