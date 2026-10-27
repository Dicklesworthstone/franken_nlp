//! UNRUN pinned schema/source planning and private corruption fixtures.
//! Synthetic identities/results never enter public native execution; these
//! fixtures do not establish inference, semantic accuracy or calibration.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::{BatchDocument, extract::{ExtractionBatchArgs, ExtractionBatchGrounding,
        grouped_int4::Int4ExtractionBatchPlanner}},
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::{CompileLimits, runtime::{JsonProgram, SourceRuntimeLimits}},
    native_engine::{constrained::JsonDecodeOutput,
        generation::grouped_int4::INT4_SOURCE_FORMAT, lmhead::NANBEIGE_VOCAB_SIZE,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE,
        strict_int4::STRICT_INT4_EXECUTION},
    tasks::{extract::{ExtractionGrounding, grouped_int4::INT4_EXTRACT_VERSION}, ir::ScoreSpace},
    tokenizer::pinned_controls,
};

const WIDTH: usize = 127;
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn maximum() -> Int4Work {
    Int4Work { forward_positions: u64::MAX, projected_logits: u64::MAX, attention_pairs: u64::MAX,
        projections: crate::native_engine::portable_int8::ProjectionWork {
            dot_products: u64::MAX, multiply_accumulates: u64::MAX } }
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"synthetic-semantic-int4-planning-only");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={WIDTH}"),
        packing_set_digest: Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, WIDTH,
        )).unwrap()),
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn rule() -> ClaimRule {
    ClaimRule { id: "amount-v1".to_owned(), path: vec![], value_kind: ClaimValueKind::Number,
        prefix: "The document reports an amount of ".to_owned(), suffix: ".".to_owned() }
}
fn options(rules: Vec<ClaimRule>) -> Int4SemanticOptions {
    Int4SemanticOptions { specification: SemanticVerificationSpec {
        schema_version: 1, experimental_opt_in: true, revision: "review-v1".to_owned(), claims: rules,
        judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0,
            evidence_window_bytes: 4096, max_evidence_windows: 8, max_evidence_spans: 8 }, judge_budget: budget() },
        limits: SemanticLimits::default(), max_model_work: maximum() }
}
fn extraction(schema: &str, source: &str, grounded: bool) -> PreparedInt4BatchExtraction {
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    let factory = Int4ExtractionBatchPlanner::pinned(controls, eos, identity(), WIDTH, budget(),
        CompileLimits::default(), SourceRuntimeLimits::default(), None).unwrap();
    factory.prepare_with_control(BatchDocument { id: "original".to_owned(), text: source.to_owned(),
        task_args: Some(ExtractionBatchArgs { schema: schema.to_owned(), budget: budget(),
            grounding: if grounded { ExtractionBatchGrounding::SourceMembership }
                else { ExtractionBatchGrounding::Structural } }) }, &mut Continue).unwrap()
}
fn judge_identity(extraction: &PreparedInt4BatchExtraction) -> ExecutionIdentity {
    // This private configuration fixture does not construct Int4JudgePlanner
    // or a model. Public preparation obtains both from actual model identity.
    let mut id = extraction.execution_identity().clone();
    id.task_spec = "judge-v1".to_owned(); id.template_digest = Sha256Digest::of_bytes(b"judge-template-fixture"); id
}
fn result(prepared: &PreparedInt4BatchExtraction, schema: &str, json: &str, grounded: bool) -> Int4ExtractRun {
    let native = prepared.extraction_plan(); let options = native.options();
    let token = (0..NANBEIGE_VOCAB_SIZE as u32)
        .find(|id| *id != options.eos_token_id && !options.excluded_token_ids.contains(id)).unwrap();
    let work = constrained_int4::planned_work(WIDTH, native.prompt_tokens(), 2).unwrap();
    let program = if grounded { JsonProgram::compile_with_source(schema, prepared.source().text(),
        CompileLimits::default(), SourceRuntimeLimits::default()).unwrap() }
        else { JsonProgram::compile(schema, CompileLimits::default()).unwrap() };
    Int4ExtractRun { schema_version: 1, execution: INT4_EXTRACT_VERSION.to_owned(), group_width: WIDTH, model_work: work,
        result: ExtractResult { schema_version: if grounded { 2 } else { 1 }, task_spec_version: "extract-v1".to_owned(),
            score_space: ScoreSpace::NotComputed, grounding: if grounded { ExtractionGrounding::SourceMembership }
                else { ExtractionGrounding::NotRequested }, source_fields: program.source_fields(json).unwrap(),
            output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT4_PROFILE.to_owned(),
                token_ids: vec![token, options.eos_token_id], json: json.to_owned(), forward_positions: work.forward_positions,
                projected_logits: work.projected_logits, mask_node_visit_charge: 1 } } }
}

#[test]
fn semantic_schema_stays_attached_to_exact_original_task_and_source() {
    let a = extraction(r#"{"type":"number"}"#, "Amount café <tool_call> 上海", false);
    let b = extraction(r#"{"type":"boolean"}"#, "Another amount", false);
    a.extraction_plan().semantic_schema(a.task_plan(), a.source()).unwrap();
    assert!(a.extraction_plan().semantic_schema(a.task_plan(), b.source()).is_err());
    assert!(a.extraction_plan().semantic_schema(b.task_plan(), a.source()).is_err());
    let opts = options(vec![rule()]);
    check_configuration(&a, &judge_identity(&a), WIDTH, budget(), JudgeLimits::default(), &opts).unwrap();
    assert!(check_configuration(&a, &judge_identity(&a), 128, budget(), JudgeLimits::default(), &opts).is_err());
}

#[test]
fn exact_decimal_claim_rendering_keeps_original_native_extraction_unchanged() {
    let schema = r#"{"type":"number"}"#; let number = "12345678901234567890123456789012345678";
    let prepared = extraction(schema, "The report gives a precise amount.", false);
    let original = result(&prepared, schema, number, false);
    let before = original.clone();
    let fields = prepare_fields(&prepared, &original, &options(vec![rule()])).unwrap();
    assert_eq!(original, before); assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].claim.as_deref(), Some("The document reports an amount of 1.2345678901234567890123456789012345678e37."));
    assert_eq!(fields[0].rule_id.as_deref(), Some("amount-v1"));
    assert_eq!(fields[0].not_checked, None);
}

#[test]
fn pre_model_rule_validation_refuses_wrong_types_paths_duplicates_and_missing_opt_in() {
    let program = JsonProgram::compile(r#"{"type":"number"}"#, CompileLimits::default()).unwrap();
    let schema = program.declarative_schema();
    options(vec![rule()]).validate_source_schema(schema, "source").unwrap();
    let mut bad = rule(); bad.path = vec![ClaimPathStep::Property("missing".to_owned())];
    assert!(options(vec![bad]).validate_source_schema(schema, "source").is_err());
    let mut bad = rule(); bad.value_kind = ClaimValueKind::String;
    assert!(options(vec![bad]).validate_source_schema(schema, "source").is_err());
    assert!(options(vec![rule(), rule()]).validate_source_schema(schema, "source").is_err());
    let mut bad = options(vec![rule()]); bad.specification.experimental_opt_in = false;
    assert!(bad.validate_source_schema(schema, "source").is_err());
    let mut bad = options(vec![rule()]); bad.specification.judge_policy.evidence_window_bytes = 1;
    bad.specification.judge_policy.max_evidence_windows = 1;
    bad.specification.judge_policy.max_evidence_spans = 1;
    assert!(bad.validate_source_schema(schema, "source longer than one evidence window").is_err());
}

#[test]
fn task_owned_identity_fields_can_differ_but_model_partition_tokenizer_and_modes_cannot() {
    let a = identity(); let mut b = a.clone();
    b.task_spec = "judge-v1".to_owned(); b.template_digest = Sha256Digest::of_bytes(b"judge-template");
    b.prompt_digest = Sha256Digest::of_bytes(b"different complete prompt");
    b.decision_policy_digest = Sha256Digest::of_bytes(b"different judgment policy");
    same_model(&a, &b).unwrap();
    for axis in 0..11 {
        let mut wrong = b.clone();
        match axis { 0 => wrong.source_revision.push('x'), 1 => wrong.quant_recipe.push('x'),
            2 => wrong.logical_model_digest = Sha256Digest::of_bytes(b"different"),
            3 => wrong.packing_set_digest = Sha256Digest::of_bytes(b"different"),
            4 => wrong.tokenizer_digest = Sha256Digest::of_bytes(b"different"),
            5 => wrong.backend_semantic_version.push('x'), 6 => wrong.kv_dtype.push('x'),
            7 => wrong.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            8 => wrong.thinking_mode = ThinkingMode::Enabled, 9 => wrong.tool_mode = ToolMode::Xml,
            _ => wrong.host_class = Some("foreign-host".to_owned()) }
        assert!(same_model(&a, &wrong).is_err(), "axis {axis}");
    }
}

#[test]
fn no_claim_results_still_require_valid_judge_limits_and_compatible_task_ceiling() {
    let prepared = extraction(r#"{"type":"null"}"#, "source", false);
    let id = judge_identity(&prepared); let opts = options(vec![]);
    for axis in 0..10 {
        let mut limits = JudgeLimits::default();
        match axis { 0 => limits.max_total_prompt_tokens = 0, 1 => limits.max_total_candidate_tokens = 0,
            2 => limits.max_total_projected_logits = 0, 3 => limits.max_output_bytes = 0,
            4 => limits.per_head.max_candidates = 0, 5 => limits.per_head.max_total_tokens = 0,
            6 => limits.per_head.max_nodes = 0, 7 => limits.per_head.max_depth = 0,
            8 => limits.per_head.max_candidate_id_bytes = 0, _ => limits.per_head.max_projected_logits = 0 }
        assert!(check_configuration(&prepared, &id, WIDTH, budget(), limits, &opts).is_err(), "axis {axis}");
    }
    for axis in 0..5 {
        let mut ceiling = budget();
        match axis { 0 => ceiling.max_input_tokens -= 1, 1 => ceiling.max_output_tokens -= 1,
            2 => ceiling.max_output_bytes -= 1, 3 => ceiling.max_grammar_states -= 1, _ => ceiling.max_kv_bytes -= 1 }
        assert!(check_configuration(&prepared, &id, WIDTH, ceiling, JudgeLimits::default(), &opts).is_err(), "axis {axis}");
    }
}

#[test]
fn unconfigured_verbatim_null_empty_and_absent_fields_are_not_invented_judgments() {
    for (schema, json, grounded, reason) in [
        (r#"{"type":"number"}"#, "1", false, SemanticNotChecked::NoClaimRule),
        (r#"{"type":"null"}"#, "null", false, SemanticNotChecked::NullValue),
        (r#"{"type":"array","maxItems":1,"items":{"type":"number"}}"#, "[]", false, SemanticNotChecked::EmptyContainer),
        (r#"{"type":"string","x-fnlp-source":"verbatim"}"#, r#""Alice""#, true, SemanticNotChecked::VerbatimMembershipOnly),
    ] {
        let prepared = extraction(schema, "Alice", grounded);
        let original = result(&prepared, schema, json, grounded);
        let fields = prepare_fields(&prepared, &original, &options(vec![])).unwrap();
        assert_eq!(fields.len(), 1); assert!(fields[0].claim.is_none()); assert_eq!(fields[0].not_checked, Some(reason));
    }
    let schema = r#"{"type":"object","additionalProperties":false,"properties":{"amount":{"type":"number"},"optional":{"type":"number"}},"required":["amount"]}"#;
    let prepared = extraction(schema, "The amount is 1.", false);
    let mut first = rule(); first.path = vec![ClaimPathStep::Property("amount".to_owned())];
    let mut optional = rule(); optional.id = "optional-v1".to_owned();
    optional.path = vec![ClaimPathStep::Property("optional".to_owned())];
    let fields = prepare_fields(&prepared, &result(&prepared, schema, r#"{"amount":1}"#, false),
        &options(vec![first, optional])).unwrap();
    assert_eq!(fields.len(), 1); assert_eq!(fields[0].pointer, "/amount");
}

#[test]
fn every_native_axis_keeps_maximum_extraction_debit_and_sums_independent_judge_heads() {
    let reserved = Int4Work::for_sequence(WIDTH, 0, 64, 32 * NANBEIGE_VOCAB_SIZE).unwrap();
    let actual = Int4Work::for_sequence(WIDTH, 0, 34, 2 * NANBEIGE_VOCAB_SIZE).unwrap();
    let head = Int4Work::for_sequence(WIDTH, 0, 48, 8 * NANBEIGE_VOCAB_SIZE).unwrap();
    let judges = plus(head, head).unwrap();
    let exact = plus(reserved, judges).unwrap();
    assert_eq!(reserve_combined(reserved, judges, exact).unwrap(), exact);
    assert_eq!(judges.attention_pairs, 2 * head.attention_pairs);
    assert_ne!(judges.attention_pairs,
        Int4Work::for_sequence(WIDTH, 0, 96, 16 * NANBEIGE_VOCAB_SIZE).unwrap().attention_pairs);
    for axis in 0..5 {
        let mut small = exact;
        match axis { 0 => small.forward_positions -= 1, 1 => small.projected_logits -= 1,
            2 => small.attention_pairs -= 1, 3 => small.projections.dot_products -= 1,
            _ => small.projections.multiply_accumulates -= 1 }
        assert!(matches!(reserve_combined(reserved, judges, small), Err(Int4SemanticError::WorkBudget)), "axis {axis}");
    }
    assert!(reserve_combined(reserved, judges, plus(actual, judges).unwrap()).is_err());
    assert!(plus(maximum(), head).is_err());
}

#[test]
fn altered_result_group_work_json_task_or_native_profile_never_reaches_claim_planning() {
    let schema = r#"{"type":"number"}"#;
    let prepared = extraction(schema, "The amount is 1.", false);
    for axis in 0..8 {
        let mut run = result(&prepared, schema, "1", false);
        match axis { 0 => run.result.output.json = "true".to_owned(), 1 => run.model_work.attention_pairs += 1,
            2 => run.result.task_spec_version = "ner-v1".to_owned(), 3 => { run.result.output.token_ids.pop(); },
            4 => run.result.output.numerics_profile = "strict-quantized-v1".to_owned(),
            5 => run.group_width = 128, 6 => run.model_work.projections.dot_products -= 1,
            _ => run.execution = "portable-int8-extract".to_owned() }
        assert!(prepare_fields(&prepared, &run, &options(vec![rule()])).is_err(), "axis {axis}");
    }
}

#[test]
fn nested_claims_share_field_byte_and_json_walk_allowances() {
    let schema = r#"{"type":"array","maxItems":2,"items":{"type":"number"}}"#;
    let prepared = extraction(schema, "The amounts are 1 and 2.", false);
    let original = result(&prepared, schema, "[1,2]", false);
    let mut each = rule(); each.path = vec![ClaimPathStep::EachItem];
    let opts = options(vec![each]);
    let fields = prepare_fields(&prepared, &original, &opts).unwrap();
    assert_eq!(fields.len(), 2); assert_eq!(fields[0].pointer, "/0"); assert_eq!(fields[1].pointer, "/1");
    for axis in 0..5 {
        let mut small = opts.clone();
        match axis { 0 => small.limits.max_fields = 1, 1 => small.limits.max_claim_bytes = 1,
            2 => small.limits.max_total_claim_bytes = fields[0].claim.as_ref().unwrap().len(),
            3 => small.limits.max_input_json_bytes = 1, _ => small.limits.max_walk_nodes = 1 }
        assert!(prepare_fields(&prepared, &original, &small).is_err(), "axis {axis}");
    }
}

#[test]
fn every_declared_work_or_dynamic_output_bound_is_explicit_and_positive() {
    for axis in 0..14 {
        let mut bad = options(vec![]);
        match axis { 0 => bad.max_model_work.forward_positions = 0, 1 => bad.max_model_work.projected_logits = 0,
            2 => bad.max_model_work.attention_pairs = 0, 3 => bad.max_model_work.projections.dot_products = 0,
            4 => bad.max_model_work.projections.multiply_accumulates = 0, 5 => bad.limits.max_fields = 0,
            6 => bad.limits.max_claim_bytes = 0, 7 => bad.limits.max_total_claim_bytes = 0,
            8 => bad.limits.max_template_bytes = 0, 9 => bad.limits.max_walk_nodes = 0,
            10 => bad.limits.max_input_json_bytes = 0, 11 => bad.limits.max_total_prompt_tokens = 0,
            12 => bad.limits.max_total_projected_logits = 0, _ => bad.limits.max_output_bytes = 0 }
        assert!(bad.validate().is_err(), "axis {axis}");
    }
    let mut bad = options(vec![]); bad.limits.max_fields = 4097; assert!(bad.validate().is_err());
    let mut bad = options(vec![]); bad.limits.max_output_bytes = u64::MAX; assert!(bad.validate().is_err());
}

#[test]
fn typed_cancellation_and_redacted_errors_survive_the_composed_boundary() {
    struct Stop(DecodeCancellationKind);
    impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) } }
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Deadline, DecodeCancellationKind::ResourceUnavailable] {
        let error = poll(&mut Stop(cause)).err().unwrap(); assert_eq!(error.cancellation(), Some(cause));
        let error = Int4SemanticError::Judge(Int4JudgeError::Cancelled(cause));
        assert_eq!(error.cancellation(), Some(cause));
    }
    let private = Int4SemanticError::Semantic(SemanticError::Contract("PRIVATE_SOURCE"));
    assert!(!format!("{private:?} {private}").contains("PRIVATE_SOURCE"));
    fn send<T: Send + 'static>() {}
    send::<PreparedInt4SemanticExtraction>(); send::<Int4SemanticExtractionRun>();
}
