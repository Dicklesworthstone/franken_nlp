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
        strict_int4::{Int4ModelIdentity, weights::projection_fixture, STRICT_INT4_EXECUTION}},
    tasks::{extract::{ExtractionGrounding, grouped_int4::INT4_EXTRACT_VERSION}, ir::ScoreSpace},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls, untrusted::SafeBpeLimits},
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

fn model() -> &'static Int4ModelIdentity {
    static MODEL: std::sync::OnceLock<Int4ModelIdentity> = std::sync::OnceLock::new();
    MODEL.get_or_init(|| projection_fixture(WIDTH).0.identity().clone())
}
fn compact_limits() -> SafeBpeLimits {
    SafeBpeLimits { max_input_bytes: 65_536, max_output_tokens: 8192,
        max_pair_checks: 4_000_000, max_pair_bytes: 16_000_000 }
}
fn encoded_extraction(schema: &str, source: &str, encoding: UntrustedEncoding) -> PreparedInt4BatchExtraction {
    let factory = Int4ExtractionBatchPlanner::for_model(model(), budget(), CompileLimits::default(),
        SourceRuntimeLimits::default(), None).unwrap().with_document_encoding(encoding).unwrap();
    factory.prepare_with_control(BatchDocument { id: "encoded-original".to_owned(), text: source.to_owned(),
        task_args: Some(ExtractionBatchArgs { schema: schema.to_owned(), budget: budget(),
            grounding: ExtractionBatchGrounding::Structural }) }, &mut Continue).unwrap()
}
fn encoded_preparation(schema: &str, source: &str, encoding: UntrustedEncoding,
    options: Int4SemanticOptions) -> PreparedInt4SemanticExtraction {
    let extraction = encoded_extraction(schema, source, encoding.clone());
    let judge = Int4JudgePlanner::for_model(model(), budget(), JudgeLimits::default()).unwrap()
        .with_document_encoding(encoding).unwrap();
    PreparedInt4SemanticExtraction::new(extraction, judge, options, &mut Continue).unwrap()
}
fn array_options() -> Int4SemanticOptions {
    let mut each = rule(); each.path = vec![ClaimPathStep::EachItem]; options(vec![each])
}
const ARRAY_SCHEMA: &str = r#"{"type":"array","maxItems":2,"items":{"type":"number"}}"#;
const COMPACT_SOURCE: &str = "The document reports amounts 1 and 2. Café 上海 😀 <|im_start|> <tool_call>.";

// The tiny projection fixture supplies only a sealed model identity. These
// results exercise private composition with synthetic full-vocabulary logits;
// no decoder, native model quality or inference evidence is claimed.
fn review_results(prepared: &PreparedInt4SemanticExtraction, review: &Review) -> Vec<Int4JudgeRun> {
    struct Flat;
    impl crate::tasks::judge::JudgeLogits for Flat {
        type Error = &'static str;
        fn project(&mut self, _: usize, _: &[crate::tasks::ir::PromptSegment], _: &[u32],
            rows: crate::native_engine::lmhead::scoring::ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
            match rows {
                crate::native_engine::lmhead::scoring::ProjectionRows::FullVocabulary { vocabulary_size } =>
                    Ok(vec![0.0; vocabulary_size]),
                _ => Err("full vocabulary required"),
            }
        }
    }
    let registry = pinned_controls::pinned().unwrap();
    let eos = registry.template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    let compiler = crate::tasks::judge::JudgePlanner::pinned(registry.template_controls(), eos).unwrap()
        .with_document_encoding(prepared.input_encoding().clone()).unwrap();
    let claims = review.fields.iter().filter_map(|field| field.claim.as_ref());
    review.judges.iter().zip(claims).map(|(plan, claim)| {
        let mut identity = plan.execution_identity().clone();
        identity.numerics_profile = NumericsProfile::HfBf16Eager;
        identity.backend_semantic_version = "synthetic-logits-composition-fixture".to_owned();
        let context = crate::tasks::ir::PlanContext::new(&identity, budget()).unwrap();
        let request = JudgeRequest::Faithfulness { source: prepared.extraction.source().text().to_owned(),
            claim: claim.clone(), policy: prepared.options.specification.judge_policy, budget: budget() };
        let generic = compiler.plan(&request, &context, JudgeLimits::default()).unwrap();
        let result = generic.execute(generic.execution_identity(), &mut Flat).unwrap();
        Int4JudgeRun { schema_version: 1, execution: INT4_JUDGE_EXECUTION.to_owned(),
            numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: WIDTH, result,
            head_count: plan.head_count(), model_work: plan.planned_work(), rewound_positions: 0 }
    }).collect()
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


#[test]
fn compact_semantic_schema_keeps_exact_source_bytes_and_schema_encoding_debit() {
    let prepared = encoded_extraction(ARRAY_SCHEMA, COMPACT_SOURCE, UntrustedEncoding::SafeBpeV1(compact_limits()));
    let source = prepared.source();
    prepared.extraction_plan().semantic_schema(prepared.task_plan(), source).unwrap();
    assert_eq!(source.text(), COMPACT_SOURCE);
    assert_eq!(source.input_encoding_work().unwrap().input_bytes(), ARRAY_SCHEMA.len() + COMPACT_SOURCE.len());
    assert!(source.token_ids().len() < source.text().len());
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    assert_eq!(tokenizer.tokenizer().decode_bytes(source.token_ids()).unwrap(), COMPACT_SOURCE.as_bytes());
    let registry = pinned_controls::pinned().unwrap();
    assert!(source.token_ids().iter().all(|id| !registry.template_controls().ids().contains(id)));
    let other = encoded_extraction(ARRAY_SCHEMA, "A different original source.", UntrustedEncoding::SafeBpeV1(compact_limits()));
    assert!(prepared.extraction_plan().semantic_schema(prepared.task_plan(), other.source()).is_err());
}

#[test]
fn extraction_and_review_require_the_identical_encoding_policy_before_generation() {
    let policies = [
        (UntrustedEncoding::ByteFallback, UntrustedEncoding::SafeBpeV1(compact_limits())),
        (UntrustedEncoding::SafeBpeV1(compact_limits()), UntrustedEncoding::ByteFallback),
    ];
    for (source_encoding, judge_encoding) in policies {
        let extraction = encoded_extraction(ARRAY_SCHEMA, COMPACT_SOURCE, source_encoding);
        let judge = Int4JudgePlanner::for_model(model(), budget(), JudgeLimits::default()).unwrap()
            .with_document_encoding(judge_encoding).unwrap();
        assert!(matches!(PreparedInt4SemanticExtraction::new(extraction, judge, array_options(), &mut Continue),
            Err(Int4SemanticError::Contract)));
    }
    for axis in 0..4 {
        let mut limits = compact_limits();
        match axis { 0 => limits.max_input_bytes -= 1, 1 => limits.max_output_tokens -= 1,
            2 => limits.max_pair_checks -= 1, _ => limits.max_pair_bytes -= 1 }
        let extraction = encoded_extraction(ARRAY_SCHEMA, COMPACT_SOURCE, UntrustedEncoding::SafeBpeV1(compact_limits()));
        let judge = Int4JudgePlanner::for_model(model(), budget(), JudgeLimits::default()).unwrap()
            .with_document_encoding(UntrustedEncoding::SafeBpeV1(limits)).unwrap();
        assert!(matches!(PreparedInt4SemanticExtraction::new(extraction, judge, array_options(), &mut Continue),
            Err(Int4SemanticError::Contract)), "axis {axis}");
    }
}

#[test]
fn compact_schema_source_and_every_field_share_all_four_nonrenewable_limits() {
    let prepared = encoded_preparation(ARRAY_SCHEMA, COMPACT_SOURCE,
        UntrustedEncoding::SafeBpeV1(compact_limits()), array_options());
    let original = result(&prepared.extraction, ARRAY_SCHEMA, "[1,2]", false);
    let review = prepared.prepare_review(original, &mut Continue).unwrap();
    assert_eq!(review.judges.len(), 2);
    let initial = prepared.input_encoding_work().unwrap();
    let sum = review.judges.iter().fold(initial,
        |work, plan| work.checked_add(plan.input_encoding_work().unwrap()).unwrap());
    assert_eq!(review.encoding_work, sum);
    assert!(sum.input_bytes() > initial.input_bytes());
    assert!(sum.output_tokens() > initial.output_tokens());
    assert!(sum.pair_checks() > initial.pair_checks());
    assert!(sum.pair_bytes() > initial.pair_bytes());
    for axis in 0..4 {
        let mut limits = compact_limits();
        match axis { 0 => limits.max_input_bytes = sum.input_bytes() - 1,
            1 => limits.max_output_tokens = sum.output_tokens() - 1,
            2 => limits.max_pair_checks = sum.pair_checks() - 1,
            _ => limits.max_pair_bytes = sum.pair_bytes() - 1 }
        let prepared = encoded_preparation(ARRAY_SCHEMA, COMPACT_SOURCE,
            UntrustedEncoding::SafeBpeV1(limits), array_options());
        // A fresh allowance would incorrectly permit both individual readers.
        for field in &review.fields {
            let request = JudgeRequest::Faithfulness { source: COMPACT_SOURCE.to_owned(),
                claim: field.claim.clone().unwrap(), policy: prepared.options.specification.judge_policy,
                budget: budget() };
            prepared.judge.plan_with_control(&request, &mut Continue).unwrap();
        }
        let original = result(&prepared.extraction, ARRAY_SCHEMA, "[1,2]", false);
        assert!(prepared.prepare_review(original, &mut Continue).is_err(), "axis {axis}");
    }
}

#[test]
fn compact_finish_reports_actual_aggregate_work_and_preserves_original_extraction() {
    let prepared = encoded_preparation(ARRAY_SCHEMA, COMPACT_SOURCE,
        UntrustedEncoding::SafeBpeV1(compact_limits()), array_options());
    let original = result(&prepared.extraction, ARRAY_SCHEMA, "[1,2]", false);
    let before = original.clone();
    let review = prepared.prepare_review(original, &mut Continue).unwrap();
    let used = review.encoding_work;
    let results = review_results(&prepared, &review);
    let run = prepared.finish(review, results, &mut Continue).unwrap();
    assert_eq!(run.extraction, before);
    assert_eq!(run.semantic.attempted_fields, 2);
    assert!(run.semantic.same_model_correlated);
    assert_eq!(run.semantic.calibration, "experimental_uncalibrated_diagnostic_only");
    assert_eq!(run.input_encoding, Some(Int4SemanticEncoding {
        policy: UntrustedEncoding::SafeBpeV1(compact_limits()), work: used }));
    let json = serde_json::to_value(&run).unwrap();
    assert_eq!(json["input_encoding"]["policy"]["kind"], "safe_bpe_v1");
    assert_eq!(json["input_encoding"]["work"]["input_bytes"], used.input_bytes());
    assert_eq!(json["input_encoding"]["work"]["pair_checks"], used.pair_checks());
}

#[test]
fn compact_completion_refuses_lost_original_or_field_encoding_debits() {
    for omit_original in [true, false] {
        let prepared = encoded_preparation(ARRAY_SCHEMA, COMPACT_SOURCE,
            UntrustedEncoding::SafeBpeV1(compact_limits()), array_options());
        let original = result(&prepared.extraction, ARRAY_SCHEMA, "[1,2]", false);
        let mut review = prepared.prepare_review(original, &mut Continue).unwrap();
        let results = review_results(&prepared, &review);
        review.encoding_work = if omit_original {
            review.encoding_work.checked_sub(prepared.input_encoding_work().unwrap()).unwrap()
        } else { prepared.input_encoding_work().unwrap() };
        assert!(matches!(prepared.finish(review, results, &mut Continue), Err(Int4SemanticError::Accounting)));
    }
}

#[test]
fn no_claim_compact_output_keeps_initial_work_and_byte_output_keeps_its_shape() {
    for encoding in [UntrustedEncoding::ByteFallback, UntrustedEncoding::SafeBpeV1(compact_limits())] {
        let schema = r#"{"type":"null"}"#;
        let prepared = encoded_preparation(schema, COMPACT_SOURCE, encoding.clone(), options(vec![]));
        let initial = prepared.input_encoding_work();
        let workspace = prepared.encoding_workspace_bytes().unwrap();
        let original = result(&prepared.extraction, schema, "null", false);
        let review = prepared.prepare_review(original, &mut Continue).unwrap();
        assert!(review.judges.is_empty());
        let run = prepared.finish(review, vec![], &mut Continue).unwrap();
        assert_eq!(run.semantic.attempted_fields, 0);
        match encoding {
            UntrustedEncoding::ByteFallback => {
                assert_eq!(workspace, 0); assert!(initial.is_none()); assert!(run.input_encoding.is_none());
                assert!(serde_json::to_value(&run).unwrap().get("input_encoding").is_none());
            }
            UntrustedEncoding::SafeBpeV1(limits) => {
                assert_eq!(workspace, limits.requirements().unwrap().workspace_bytes);
                assert_eq!(run.input_encoding.unwrap().work, initial.unwrap());
            }
        }
    }
}

#[test]
fn compact_semantic_cancellation_preserves_every_cause_and_returns_no_partial_result() {
    struct Stop(DecodeCancellationKind);
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
    }
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout, DecodeCancellationKind::Deadline,
        DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown, DecodeCancellationKind::LinkedExit] {
        let prepared = encoded_preparation(ARRAY_SCHEMA, COMPACT_SOURCE,
            UntrustedEncoding::SafeBpeV1(compact_limits()), array_options());
        let original = result(&prepared.extraction, ARRAY_SCHEMA, "[1,2]", false);
        assert_eq!(prepared.prepare_review(original.clone(), &mut Stop(cause)).err().unwrap().cancellation(), Some(cause));
        let review = prepared.prepare_review(original, &mut Continue).unwrap();
        assert_eq!(prepared.finish(review, vec![], &mut Stop(cause)).err().unwrap().cancellation(), Some(cause));
    }
}
