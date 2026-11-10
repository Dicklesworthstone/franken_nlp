//! Model-free contract fixtures using the real pinned schema/source compiler.
//! These do not execute native inference or establish model/schedule parity.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    native_engine::{decode::DecodeCancellationKind, generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::{ProjectionWork, int4::convert::GROUPED_INT4_RECIPE}},
};

const WIDTH: usize = 127;
const SOURCE: &str = "éabc----βdef";
const SCHEMA: &str = r#"{"type":"string","maxLength":64,"x-fnlp-source":"verbatim"}"#;
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 65_536,
        max_grammar_states: 8192, max_kv_bytes: Int4MemoryRequirement::for_context(8192, WIDTH).unwrap().kv_bytes }
}
fn maximum_work() -> Int4Work {
    let n = u64::MAX / 4;
    Int4Work { forward_positions: n, projected_logits: n, attention_pairs: n,
        projections: ProjectionWork { dot_products: n, multiply_accumulates: n } }
}
fn config(grounded: bool) -> Int4ExtractionDocumentBatchConfig {
    let verification = GroundingBudget::default();
    Int4ExtractionDocumentBatchConfig {
        request: ExtractionBatchArgs { schema: if grounded { SCHEMA } else { r#"{"type":"boolean"}"# }.to_owned(),
            grounding: if grounded { ExtractionBatchGrounding::SourceMembership } else { ExtractionBatchGrounding::Structural },
            budget: budget() },
        limits: Int4ExtractionMapLimits { mapping: Int4ExtractionMappingLimits {
            chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 8, max_chunk_tokens: 8,
                context_tokens: 8192, reserved_tokens: 64, max_chunks: 8, max_tokenizer_calls: 1024 },
            reduction: ExecutionLimits { map_batch_chunks: 3, reduce_fan_in: 2,
                max_value_bytes: 4 << 20, ..ExecutionLimits::default() },
            max_model_work: maximum_work(), mask_limits: MaskWorkLimits::default(),
            mask_visits_per_chunk: 1000, max_mask_visits: 8000 }, verification, root_array_union: None },
        max_model_work: maximum_work(), max_mask_visits: 32_000,
        max_verification: GroundingBudget { max_fields: verification.max_fields * 4,
            max_matches: verification.max_matches * 4, max_scan_steps: verification.max_scan_steps * 4 },
    }
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"private live complete-document fixture; not model authority");
    let packing = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, WIDTH)).unwrap());
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={WIDTH}"),
        packing_set_digest: packing, tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "uncompiled".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn build(config: Int4ExtractionDocumentBatchConfig) -> Result<Int4ExtractionDocumentBatchPlanner, BatchFault> {
    let controls = pinned_controls::pinned().unwrap();
    let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
    let compiler = Int4ExtractionBatchPlanner::pinned(controls.template_controls(), eos, identity(), WIDTH,
        budget(), CompileLimits::default(), SourceRuntimeLimits::default(), None)?;
    Int4ExtractionDocumentBatchPlanner::configured(compiler, config)
}
fn factory(grounded: bool) -> Int4ExtractionDocumentBatchPlanner { build(config(grounded)).unwrap() }
// Exercise closed preparation commitments without constructing weights or a
// public alternate inference path. Native capacity checks are separate gates.
fn snapshot(factory: &Int4ExtractionDocumentBatchPlanner, source: &str) -> PreparedInt4ExtractionDocument {
    let plan = factory.plan(source, &mut Continue, false).unwrap();
    let (expected, identities) = factory.snapshot(&plan, &mut Continue).unwrap();
    PreparedInt4ExtractionDocument { source: source.to_owned(), expected, identities, owner: Arc::clone(&factory.owner) }
}
fn context(sequence: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: sequence, epoch, input_line: sequence, byte_offset: 0 }
}
fn axis(work: &mut Int4Work, index: usize) -> &mut u64 {
    match index { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits, 2 => &mut work.attention_pairs,
        3 => &mut work.projections.dot_products, _ => &mut work.projections.multiply_accumulates }
}

#[test]
fn all_real_unicode_chunks_keep_exact_source_schema_and_independent_identities() {
    for grounded in [false, true] {
        let factory = factory(grounded); let prepared = snapshot(&factory, SOURCE);
        assert_eq!(prepared.source, SOURCE);
        assert_eq!(prepared.chunk_count(), 2);
        assert_eq!(prepared.execution_identities().len(), 2);
        let span = prepared.expected.mapped().source_span();
        assert_eq!((span.byte_start, span.byte_end), (0, SOURCE.len()));
        assert_eq!((span.scalar_start, span.scalar_end), (0, SOURCE.chars().count()));
        assert_eq!(prepared.reserved_mask_visits(), 2000);
        for identity in prepared.execution_identities() {
            assert_eq!(identity.schema_digest, Sha256Digest::of_bytes(factory.request().schema.as_bytes()));
            assert_eq!(identity.grammar_compiler_version, if grounded { SOURCE_JSON_RUNTIME_VERSION } else { JSON_RUNTIME_VERSION });
        }
        assert_ne!(prepared.identities[0].prompt_digest, prepared.identities[1].prompt_digest);
        let plan = factory.plan(SOURCE, &mut Continue, true).unwrap();
        factory.compare(&plan, &prepared, &mut Continue).unwrap();
        let one = snapshot(&factory, "Alice");
        assert_eq!(one.chunk_count(), 1);
        assert!(one.planned_work().forward_positions < prepared.planned_work().forward_positions);
    }
}

#[test]
fn union_factory_refuses_non_array_defaults_and_freezes_policy_before_document_execution() {
    let mut selected = config(false);
    selected.limits.root_array_union = Some(super::super::tests::union_limits());
    assert!(build(selected).is_err());
    let mut selected = config(false);
    selected.request.schema = r#"{"type":"array","items":{"type":"integer"},"maxItems":8}"#.to_owned();
    selected.limits.root_array_union = Some(super::super::tests::union_limits());
    let factory = build(selected).unwrap();
    let prepared = snapshot(&factory, SOURCE);
    assert!(prepared.expected.mapped().root_array_binding.is_some());
    let private = prepared.expected.mapped().root_array_binding.unwrap().to_hex();
    assert!(!format!("{:?}", prepared.expected).contains(&private));
    let plan = factory.plan(SOURCE, &mut Continue, true).unwrap();
    factory.compare(&plan, &prepared, &mut Continue).unwrap();
    let mut changed = config(false);
    changed.request.schema = factory.request().schema.clone();
    changed.limits.root_array_union = factory.document_limits().root_array_union;
    changed.limits.root_array_union.as_mut().unwrap().max_origins += 1;
    let other = build(changed).unwrap();
    let other_plan = other.plan(SOURCE, &mut Continue, true).unwrap();
    assert_ne!(other_plan.preflight_metadata(), prepared.expected);
    assert!(other.compare(&other_plan, &prepared, &mut Continue).is_err());
}

#[test]
fn changed_tail_or_factory_cannot_reuse_frozen_preparation_and_schedule_rotates_seal() {
    let factory = factory(true); let mut prepared = snapshot(&factory, SOURCE);
    prepared.source = "éabc----βdeg".to_owned();
    let changed = factory.plan(&prepared.source, &mut Continue, true).unwrap();
    assert_eq!(factory.compare(&changed, &prepared, &mut Continue).err().unwrap().fault.code, BatchCode::Admission);
    let other = build(config(true)).unwrap();
    assert_eq!(other.owns(&prepared).err().unwrap().fault.code, BatchCode::Admission);
    let clean = snapshot(&factory, SOURCE);
    drop(changed);
    let cohort = factory.with_chunk_cohorts().unwrap();
    assert!(cohort.uses_chunk_cohorts());
    assert_eq!(cohort.owns(&clean).err().unwrap().fault.code, BatchCode::Admission);
    let next = snapshot(&cohort, SOURCE);
    cohort.owns(&next).unwrap();
    assert_eq!(next.expected, clean.expected);
    assert_eq!(next.identities, clean.identities);
}

#[test]
fn selecting_complete_legal_rows_rotates_the_factory_seal_and_preserves_peak_memory() {
    for cap in [65, NANBEIGE_VOCAB_SIZE] {
        let dense = factory(true).with_chunk_cohorts().unwrap();
        let original = snapshot(&dense, SOURCE); let peak = dense.cohort_requirements().unwrap();
        let selected = dense.with_selected_rows(Int4JsonSparseLimits { max_rows_per_step: cap }).unwrap();
        assert_eq!(selected.selected_rows(), Some(Int4JsonSparseLimits { max_rows_per_step: cap }));
        assert!(selected.uses_chunk_cohorts()); assert_eq!(selected.cohort_requirements().unwrap(), peak);
        assert_eq!(selected.owns(&original).err().unwrap().fault.code, BatchCode::Admission);
        let next = snapshot(&selected, SOURCE); selected.owns(&next).unwrap();
        assert_eq!(next.expected.selected_rows(), selected.selected_rows());
        assert_eq!(next.expected.planned_work().forward_positions, original.expected.planned_work().forward_positions);
        assert_eq!(next.expected.reserved_mask_visits(), original.expected.reserved_mask_visits());
        if cap == NANBEIGE_VOCAB_SIZE { assert_eq!(next.planned_work(), original.planned_work()); }
        assert_ne!(next.expected, original.expected);
        for (before, after) in original.identities.iter().zip(&next.identities) {
            assert_eq!(before.prompt_digest, after.prompt_digest);
            assert_ne!(before.decision_policy_digest, after.decision_policy_digest);
        }
        let plan = selected.plan(SOURCE, &mut Continue, true).unwrap();
        selected.compare(&plan, &next, &mut Continue).unwrap();
        assert!(selected.compare(&plan, &original, &mut Continue).is_err());
        assert!(selected.with_selected_rows(Int4JsonSparseLimits { max_rows_per_step: cap }).is_err());
    }
    for cap in [0, NANBEIGE_VOCAB_SIZE + 1] {
        assert!(factory(true).with_selected_rows(Int4JsonSparseLimits { max_rows_per_step: cap }).is_err());
    }
}

#[test]
fn selected_population_admits_exact_work_below_dense_projection_and_debits_across_flushes() {
    let rows = Int4JsonSparseLimits { max_rows_per_step: 7 };
    let measured = factory(false).with_selected_rows(rows).unwrap();
    let expected = snapshot(&measured, SOURCE).expected;
    let mut cap = config(false);
    cap.limits.mapping.max_model_work = expected.planned_work();
    cap.max_model_work = expected.planned_work();
    cap.max_mask_visits = expected.reserved_mask_visits();
    cap.limits.mapping.max_mask_visits = expected.reserved_mask_visits();
    let dense = build(cap.clone()).unwrap();
    assert_eq!(dense.plan(SOURCE, &mut Continue, false).err().unwrap().fault.code, BatchCode::WorkLimit);
    let selected = build(cap.clone()).unwrap().with_selected_rows(rows).unwrap();
    let prepared = snapshot(&selected, SOURCE);
    assert_eq!(prepared.expected, expected);
    let mut ledger = Ledger::default();
    ledger.begin(&cap, expected, context(1, 1)).unwrap();
    ledger.failed = false;
    assert_eq!(ledger.begin(&cap, expected, context(2, 99)).err().unwrap().fault.code, BatchCode::WorkLimit);
    assert_eq!(ledger.work, expected.planned_work());
}

#[test]
fn document_args_are_strict_map_only_and_cannot_replace_any_policy() {
    for text in [r#"{"id":"x","text":"source"}"#, r#"{"id":"x","text":"source","task_args":null}"#,
        r#"{"id":"x","text":"source","task_args":{}}"#] {
        assert!(serde_json::from_str::<BatchDocument<ExtractionDocumentArgs>>(text).is_ok());
    }
    for args in ["[]", "true", "0", r#""""#, r#"{"schema":"{}"}"#, r#"{"budget":{}}"#,
        r#"{"grounding":"structural"}"#, r#"{"x":1,"x":2}"#] {
        let text = format!(r#"{{"id":"x","text":"source","task_args":{args}}}"#);
        assert!(serde_json::from_str::<BatchDocument<ExtractionDocumentArgs>>(&text).is_err(), "{args}");
    }
}

#[test]
fn each_of_nine_whole_run_axes_exhausts_atomically_across_flush_epochs() {
    for factory in [factory(true), factory(true).with_selected_rows(Int4JsonSparseLimits { max_rows_per_step: 65 }).unwrap()] {
    let prepared = snapshot(&factory, SOURCE); let expected = prepared.expected;
    let twice_work = expected.planned_work().checked_add(expected.planned_work()).unwrap();
    let twice_verification = add_verification(factory.config.limits.verification, factory.config.limits.verification).unwrap();
    for dimension in 0..9 {
        let mut cap = factory.config.clone();
        cap.limits.mapping.max_model_work = expected.planned_work();
        cap.limits.mapping.max_mask_visits = expected.reserved_mask_visits();
        cap.max_model_work = twice_work; cap.max_mask_visits = expected.reserved_mask_visits() * 2;
        cap.max_verification = twice_verification;
        match dimension {
            0..=4 => *axis(&mut cap.max_model_work, dimension) -= 1,
            5 => cap.max_mask_visits -= 1,
            6 => cap.max_verification.max_fields -= 1,
            7 => cap.max_verification.max_matches -= 1,
            _ => cap.max_verification.max_scan_steps -= 1,
        }
        cap.validate().unwrap();
        // Exact actual masks are charged; the configured per-document maximum
        // can be larger and is never substituted for the prepared reservation.
        let mut ledger = Ledger::default();
        ledger.begin(&cap, expected, context(1, 1)).unwrap();
        assert_eq!(ledger.work, expected.planned_work());
        assert_eq!(ledger.masks, expected.reserved_mask_visits());
        assert_eq!(ledger.verification, cap.limits.verification);
        ledger.failed = false; // Simulate the only allowed reset: successful completion.
        let before = (ledger.work, ledger.masks, ledger.verification, ledger.sequence);
        let failure = ledger.begin(&cap, expected, context(2, 99)).err().unwrap();
        assert_eq!(failure.fault.code, BatchCode::WorkLimit, "dimension {dimension}");
        assert!(failure.stop && ledger.failed);
        assert_eq!((ledger.work, ledger.masks, ledger.verification, ledger.sequence), before);
    }
    }
}

#[test]
fn failed_attempt_cancellation_and_replayed_context_never_refund_or_renew() {
    let factory = factory(true); let prepared = snapshot(&factory, SOURCE);
    let mut ledger = Ledger::default();
    ledger.begin(&factory.config, prepared.expected, context(7, 1)).unwrap();
    let reserved = (ledger.work, ledger.masks, ledger.verification);
    // A failure/unwind leaves the latch closed before the next callback.
    assert_eq!(ledger.begin(&factory.config, prepared.expected, context(8, 2)).err().unwrap().fault.code,
        BatchCode::InvalidExecution);
    assert_eq!((ledger.work, ledger.masks, ledger.verification), reserved);
    ledger.failed = false;
    assert_eq!(ledger.begin(&factory.config, prepared.expected, context(7, 100)).err().unwrap().fault.code,
        BatchCode::InvalidExecution);
    assert_eq!((ledger.work, ledger.masks, ledger.verification), reserved);
    for invalid in [context(1, 0), BatchRequestContext { input_line: 0, ..context(1, 1) }] {
        let mut ledger = Ledger::default();
        assert!(ledger.begin(&factory.config, prepared.expected, invalid).is_err());
        assert_eq!(ledger.work, Int4Work::default());
        assert_eq!(ledger.masks, 0); assert!(ledger.failed);
    }
}

#[test]
fn rejected_empty_oversize_or_unplannable_rows_leave_later_valid_source_plannable() {
    let factory = factory(true);
    for text in [String::new(), "a".repeat(factory.document_limits().mapping.chunks.max_input_bytes + 1), "a".repeat(100)] {
        let failure = factory.plan(&text, &mut Continue, false).err().unwrap();
        assert!(!failure.stop);
        assert!(matches!(failure.fault.code, BatchCode::DocumentLimit | BatchCode::Planning));
        assert_eq!(snapshot(&factory, SOURCE).chunk_count(), 2);
    }
    let bad = map_failure(Int4ExtractionMapError::Chunk(MapReduceError::AllocationRefused), false);
    assert!(bad.stop); assert_eq!(bad.fault.code, BatchCode::Allocation);
    struct Cancel;
    impl DecodeStepControl for Cancel {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::User) }
    }
    let failure = factory.plan(SOURCE, &mut Cancel, false).err().unwrap();
    assert!(failure.stop); assert!(failure.fault.cancellation.is_some());
}

#[test]
fn transferred_schema_capacity_is_preserved_and_run_caps_need_only_one_document() {
    let mut cfg = config(true);
    cfg.request.schema.reserve(32_768);
    let original_capacity = cfg.request.schema.capacity();
    cfg.max_model_work = cfg.limits.mapping.max_model_work;
    cfg.max_mask_visits = cfg.limits.mapping.max_mask_visits;
    cfg.max_verification = cfg.limits.verification;
    let factory = build(cfg).unwrap();
    assert_eq!(factory.request().schema.capacity(), original_capacity);
    assert_eq!(factory.retained_defaults_bytes().unwrap(),
        (original_capacity + std::mem::size_of::<ExtractionBatchArgs>()) as u64);
    assert!(factory.compiler.base_planner().compiler.defaults.is_none());
    assert_eq!(factory.max_verification(), factory.document_limits().verification);
    assert_eq!(snapshot(&factory, SOURCE).chunk_count(), 2);
}

#[test]
fn cohort_quote_is_declared_peak_while_every_actual_window_is_smaller_or_equal() {
    let factory = factory(true).with_chunk_cohorts().unwrap();
    let peak = factory.cohort_requirements().unwrap();
    assert_eq!(peak.native.kv_bytes, factory.document_budget().max_kv_bytes * 3);
    let plan = factory.plan(SOURCE, &mut Continue, false).unwrap();
    let actual = plan.cohort_requirements(8192).unwrap();
    assert!(actual.native.kv_bytes < peak.native.kv_bytes);
    assert!(actual.workspace_bytes <= peak.workspace_bytes);
    assert!(actual.max_result_bytes <= peak.max_result_bytes);
    let mut invalid = config(true);
    invalid.limits.mapping.reduction.map_batch_chunks = 65;
    invalid.limits.mapping.chunks.max_chunks = 128;
    assert!(build(invalid).unwrap().with_chunk_cohorts().is_err());
    let mut invalid = config(true);
    invalid.request.budget.max_kv_bytes -= 1;
    assert_eq!(build(invalid).err().unwrap().code, BatchCode::InvalidLimits);
}

const OBJECT: &str = r#"{"type":"object","additionalProperties":false,"properties":{"first":{"type":"string","maxLength":16},"last":{"type":"string","maxLength":16}},"required":["first","last"]}"#;
fn object_limits() -> RootObjectLimits {
    RootObjectLimits { max_properties: 16, max_origins: 128, max_input_bytes: 1 << 20,
        max_json_bytes: 65_536, max_nodes: 65_536, max_result_bytes: 1 << 20 }
}
fn object_config() -> Int4ExtractionDocumentBatchConfig {
    let mut cfg = config(false);
    cfg.request.schema = OBJECT.to_owned();
    cfg.limits.mapping.reduction.map_batch_chunks = 1;
    cfg
}

#[test]
fn explicit_object_population_binds_original_partial_schema_prompt_and_complete_source() {
    let ordinary = build(object_config()).unwrap();
    let old = snapshot(&ordinary, SOURCE);
    let old_base = ordinary.base_execution_identity().clone();
    let object = ordinary.with_root_object_reconciliation(object_limits()).unwrap();
    assert_eq!(object.root_object_reconciliation(), Some(object_limits()));
    assert_eq!(object.request().schema, OBJECT);
    assert_eq!(object.base_execution_identity().task_spec, "extract-v1");
    assert_ne!(object.base_execution_identity().template_digest, old_base.template_digest);
    assert!(object.owns(&old).is_err());
    let prepared = snapshot(&object, SOURCE);
    assert!(matches!(prepared.expected, DocumentPreflight::Object(_)));
    assert_eq!(prepared.expected.execution_version(), super::super::object::INT4_OBJECT_MAP_EXECUTION);
    assert_eq!(prepared.chunk_count(), 2);
    let plan = object.plan(SOURCE, &mut Continue, false).unwrap();
    object.compare(&plan, &prepared, &mut Continue).unwrap();
    let mapped = plan.mapped();
    assert_eq!(mapped.chunks.chunks().iter().map(|chunk| chunk.text()).collect::<String>(), SOURCE);
    let partial = RootObjectSchema::derive(OBJECT, object.compiler_limits(), object_limits(), &mut Continue).unwrap();
    let partial_digest = Sha256Digest::of_bytes(partial.partial_schema().as_bytes());
    assert_ne!(partial_digest, Sha256Digest::of_bytes(OBJECT.as_bytes()));
    for row in &mapped.plans {
        assert_eq!(row.execution_identity().schema_digest, partial_digest);
        assert_eq!(row.execution_identity().template_digest, object.base_execution_identity().template_digest);
    }
    let wrong_source = object.plan("éabc----βdeg", &mut Continue, true).unwrap();
    assert!(object.compare(&wrong_source, &prepared, &mut Continue).is_err());
    let ordinary = build(object_config()).unwrap();
    let exact_schema = ordinary.plan(SOURCE, &mut Continue, true).unwrap();
    assert_ne!(exact_schema.preflight_metadata(), prepared.expected);
    assert!(object.compare(&exact_schema, &prepared, &mut Continue).is_err());
}

#[test]
fn original_required_fields_and_reconciliation_limits_remain_authority_after_partial_schema_matches() {
    let object = build(object_config()).unwrap().with_root_object_reconciliation(object_limits()).unwrap();
    let prepared = snapshot(&object, SOURCE);
    // Both requests derive the same partial schema and chunk prompts, but the
    // final required set is different and may never borrow a previous proof.
    let mut changed = object_config();
    changed.request.schema = OBJECT.replace("[\"first\",\"last\"]", "[\"first\"]");
    let different = build(changed).unwrap().with_root_object_reconciliation(object_limits()).unwrap();
    let other = snapshot(&different, SOURCE);
    assert_eq!(prepared.identities, other.identities);
    assert_eq!(prepared.expected.mapped(), other.expected.mapped());
    assert_ne!(prepared.expected, other.expected);
    let mut limits = object_limits(); limits.max_origins += 1;
    let different = build(object_config()).unwrap().with_root_object_reconciliation(limits).unwrap();
    let changed_policy = snapshot(&different, SOURCE);
    assert_eq!(prepared.identities, changed_policy.identities);
    assert_ne!(prepared.expected, changed_policy.expected);
    assert!(!format!("{:?}", prepared.expected).contains(SOURCE));
}

#[test]
fn selected_rows_and_native_cohorts_preserve_object_semantics_in_both_builder_orders() {
    let selected = Int4JsonSparseLimits { max_rows_per_step: 65 };
    for object_first in [false, true] {
        let mut cfg = object_config(); cfg.limits.mapping.reduction.map_batch_chunks = 3;
        let plain = build(cfg).unwrap();
        let peak = plain.cohort_requirements().unwrap();
        let object = if object_first {
            plain.with_root_object_reconciliation(object_limits()).unwrap().with_selected_rows(selected).unwrap()
        } else {
            plain.with_selected_rows(selected).unwrap().with_root_object_reconciliation(object_limits()).unwrap()
        }.with_chunk_cohorts().unwrap();
        assert_eq!(object.selected_rows(), Some(selected));
        assert_eq!(object.root_object_reconciliation(), Some(object_limits()));
        assert!(object.uses_chunk_cohorts());
        assert_eq!(object.cohort_requirements().unwrap(), peak);
        let prepared = snapshot(&object, SOURCE);
        assert_eq!(prepared.expected.execution_version(), super::super::object::INT4_OBJECT_MAP_SELECTED_EXECUTION);
        assert_eq!(prepared.expected.selected_rows(), Some(selected));
        let plan = object.plan(SOURCE, &mut Continue, true).unwrap();
        object.compare(&plan, &prepared, &mut Continue).unwrap();
        let actual = plan.cohort_requirements(8192).unwrap();
        assert!(actual.native.kv_bytes <= peak.native.kv_bytes);
        assert!(actual.workspace_bytes <= peak.workspace_bytes);
        assert!(object.with_root_object_reconciliation(object_limits()).is_err());
    }
}

#[test]
fn object_population_rejects_competing_reducers_bad_grounding_and_invalid_bounds_before_sources() {
    assert!(factory(false).with_root_object_reconciliation(object_limits()).is_err());
    for case in 0..5 {
        let mut cfg = object_config(); let mut limits = object_limits();
        match case {
            0 => cfg.request.grounding = ExtractionBatchGrounding::SourceMembership,
            1 => limits.max_properties = 1,
            2 => limits.max_origins = 1,
            3 => limits.max_json_bytes = cfg.limits.mapping.reduction.max_result_bytes + 1,
            _ => limits.max_result_bytes = cfg.limits.mapping.reduction.max_result_bytes + 1,
        }
        assert!(build(cfg).unwrap().with_root_object_reconciliation(limits).is_err(), "case {case}");
    }
    let mut cfg = config(false);
    cfg.request.schema = r#"{"type":"array","items":{"type":"integer"},"maxItems":8}"#.to_owned();
    cfg.limits.root_array_union = Some(super::super::tests::union_limits());
    assert!(build(cfg).unwrap().with_root_object_reconciliation(object_limits()).is_err());
}

#[test]
fn object_attempts_charge_complete_work_mask_and_global_verification_once_per_document() {
    let object = build(object_config()).unwrap().with_root_object_reconciliation(object_limits()).unwrap();
    let expected = snapshot(&object, SOURCE).expected;
    for dimension in 0..9 {
        let mut cap = object.config.clone();
        cap.max_model_work = expected.planned_work().checked_add(expected.planned_work()).unwrap();
        cap.max_mask_visits = expected.reserved_mask_visits() * 2;
        cap.max_verification = add_verification(cap.limits.verification, cap.limits.verification).unwrap();
        match dimension {
            0..=4 => *axis(&mut cap.max_model_work, dimension) -= 1,
            5 => cap.max_mask_visits -= 1,
            6 => cap.max_verification.max_fields -= 1,
            7 => cap.max_verification.max_matches -= 1,
            _ => cap.max_verification.max_scan_steps -= 1,
        }
        let mut ledger = Ledger::default();
        ledger.begin(&cap, expected, context(1, 1)).unwrap();
        assert_eq!(ledger.work, expected.planned_work());
        assert_eq!(ledger.masks, expected.reserved_mask_visits());
        assert_eq!(ledger.verification, object.document_limits().verification);
        ledger.failed = false;
        let before = (ledger.work, ledger.masks, ledger.verification);
        let failure = ledger.begin(&cap, expected, context(2, 99)).unwrap_err();
        assert_eq!(failure.fault.code, BatchCode::WorkLimit);
        assert!(failure.stop && ledger.failed);
        assert_eq!((ledger.work, ledger.masks, ledger.verification), before);
    }
    let mut ledger = Ledger::default();
    ledger.begin(&object.config, expected, context(1, 1)).unwrap();
    assert_eq!(ledger.begin(&object.config, expected, context(2, 2)).unwrap_err().fault.code, BatchCode::InvalidExecution);
    for error in [RootObjectError::Conflict, RootObjectError::MissingRequired, RootObjectError::Source, RootObjectError::Limit] {
        let failure = document_failure(Int4ObjectExtractionMapError::Reconciliation(error), true);
        assert!(failure.stop); assert_eq!(failure.fault.code, BatchCode::InvalidExecution);
    }
}

#[test]
fn cancelled_object_planning_never_returns_partial_metadata_and_preserves_cause() {
    let object = build(object_config()).unwrap().with_root_object_reconciliation(object_limits()).unwrap();
    struct Cancel;
    impl DecodeStepControl for Cancel {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
    }
    let failure = object.plan(SOURCE, &mut Cancel, false).err().unwrap();
    assert!(failure.stop); assert!(failure.fault.cancellation.is_some());
    let failure = document_failure(Int4ObjectExtractionMapError::Reconciliation(
        RootObjectError::Cancelled(DecodeCancellationKind::User)), false);
    assert!(failure.stop); assert!(failure.fault.cancellation.is_some());
}

#[test]
fn ordinary_document_result_keeps_identical_wire_and_cannot_satisfy_object_completion() {
    use super::super::tests as fixture;
    let compiler = fixture::planner();
    let args = fixture::args(r#"{"type":"boolean"}"#, ExtractionBatchGrounding::Structural);
    let plan = fixture::prepare(&compiler, SOURCE, &args, fixture::limits());
    let expected = DocumentPreflight::Map(plan.preflight_metadata());
    let identities = plan.execution_identities().cloned().collect::<Vec<_>>();
    let mut script = fixture::Script::new(&args); script.json = Some("true".to_owned());
    let run = plan.execute_with_driver(&identities, script, &mut Continue).unwrap();
    let old_bytes = canonjson::canonical_bytes(&run).unwrap();
    let document = Int4ExtractionDocumentRun::Map(run);
    assert_eq!(canonjson::canonical_bytes(&document).unwrap(), old_bytes);
    assert_eq!(document.verification_used(), document.mapped().verification_used);
    expected.verify_completed(&document).unwrap();
    let object = build(object_config()).unwrap().with_root_object_reconciliation(object_limits()).unwrap();
    assert!(snapshot(&object, SOURCE).expected.verify_completed(&document).is_err());
}
