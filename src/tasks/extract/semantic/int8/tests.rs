//! Pinned planning and private result-corruption tests, not native-success or
//! accuracy evidence. The public pipeline cannot accept these synthetic runs.
use super::*;
use crate::{
    batch::{BatchDocument, extract::{ExtractionBatchArgs, ExtractionBatchGrounding,
        quantized::Int8ExtractionBatchPlanner}},
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::{CompileLimits, runtime::{JsonProgram, SourceRuntimeLimits}},
    native_engine::{constrained::JsonDecodeOutput, constrained_int8,
        lmhead::NANBEIGE_VOCAB_SIZE, strict_int8::STRICT_INT8_EXECUTION},
    tasks::{extract::{ExtractionGrounding, quantized::INT8_EXTRACT_VERSION}, ir::ScoreSpace},
    tokenizer::pinned_controls,
};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn maximum() -> Int8Work {
    let mut w = Int8Work::default();
    w.forward_positions = u64::MAX; w.projected_logits = u64::MAX; w.attention_pairs = u64::MAX;
    w.projections.dot_products = u64::MAX; w.projections.multiply_accumulates = u64::MAX; w
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"semantic-int8-planning-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".into(), logical_model_digest: d,
        artifact_format: "fixture".into(), quant_recipe: "int8-fixture".into(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".into(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".into(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".into(),
        sampler_version: "fixture".into(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.into(),
        host_class: None, compiler_identity: None }
}
fn rule() -> ClaimRule {
    ClaimRule { id: "amount-v1".into(), path: vec![], value_kind: ClaimValueKind::Number,
        prefix: "The document reports an amount of ".into(), suffix: ".".into() }
}
fn options(rules: Vec<ClaimRule>) -> Int8SemanticOptions {
    Int8SemanticOptions { specification: SemanticVerificationSpec {
        schema_version: 1, experimental_opt_in: true, revision: "review-v1".into(), claims: rules,
        judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0,
            evidence_window_bytes: 4096, max_evidence_windows: 8, max_evidence_spans: 8 }, judge_budget: budget() },
        limits: SemanticLimits::default(), max_model_work: maximum() }
}
fn extraction(schema: &str, source: &str, grounded: bool) -> PreparedInt8BatchExtraction {
    let r = pinned_controls::pinned().unwrap(); let controls = r.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    let factory = Int8ExtractionBatchPlanner::pinned(controls, eos, identity(), budget(),
        CompileLimits::default(), SourceRuntimeLimits::default(), None).unwrap();
    factory.prepare_with_control(BatchDocument { id: "original".into(), text: source.into(), task_args: Some(ExtractionBatchArgs {
        schema: schema.into(), grounding: if grounded { ExtractionBatchGrounding::SourceMembership }
            else { ExtractionBatchGrounding::Structural }, budget: budget() }) }, &mut Continue).unwrap()
}
fn prepare(schema: &str, source: &str, grounded: bool, options: Int8SemanticOptions)
    -> Result<PreparedInt8SemanticExtraction, Int8SemanticError> {
    let extraction = extraction(schema, source, grounded);
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    let judge = Arc::new(JudgePlanner::pinned(controls, eos).unwrap());
    let mut id = extraction.execution_identity().clone(); id.task_spec = "judge-v1".into();
    id.template_digest = *judge.template_digest(); id.tokenizer_digest = judge.tokenizer_digest();
    PreparedInt8SemanticExtraction::new(extraction, judge, id, JudgeLimits::default(), options, &mut Continue)
}
fn result(p: &PreparedInt8SemanticExtraction, schema: &str, json: &str, grounded: bool) -> Int8ExtractRun {
    let native = p.extraction.extraction_plan(); let options = native.options();
    let token = (0..NANBEIGE_VOCAB_SIZE as u32).find(|id| *id != options.eos_token_id && !options.excluded_token_ids.contains(id)).unwrap();
    let work = constrained_int8::planned_work(native.prompt_tokens(), 2).unwrap();
    let program = if grounded { JsonProgram::compile_with_source(schema, p.extraction.source().text(),
        CompileLimits::default(), SourceRuntimeLimits::default()).unwrap() }
        else { JsonProgram::compile(schema, CompileLimits::default()).unwrap() };
    Int8ExtractRun { schema_version: 1, execution: INT8_EXTRACT_VERSION.into(), model_work: work,
        result: ExtractResult { schema_version: if grounded { 2 } else { 1 }, task_spec_version: "extract-v1".into(),
            score_space: ScoreSpace::NotComputed, grounding: if grounded { ExtractionGrounding::SourceMembership }
                else { ExtractionGrounding::NotRequested }, source_fields: program.source_fields(json).unwrap(),
            output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT8_PROFILE.into(),
                token_ids: vec![token, options.eos_token_id], json: json.into(), forward_positions: work.forward_positions,
                projected_logits: work.projected_logits, mask_node_visit_charge: 1 } } }
}
#[test]
fn exact_large_decimal_claims_preserve_original_extraction_and_use_real_judge_plans() {
    let schema = r#"{"type":"number"}"#; let number = "12345678901234567890123456789012345678";
    let p = prepare(schema, "The report gives a precise amount.", false, options(vec![rule()])).unwrap();
    let run = result(&p, schema, number, false);
    let review = p.prepare_review(run.clone(), &mut Continue).unwrap();
    assert_eq!(review.extraction, run); assert_eq!(review.fields.len(), 1); assert_eq!(review.judges.len(), 1);
    assert_eq!(review.fields[0].claim.as_deref(), Some("The document reports an amount of 1.2345678901234567890123456789012345678e37."));
    assert_eq!(review.judges[0].execution_identity().task_spec, "judge-v1");
    assert_eq!(review.judges[0].execution_identity().numerics_profile, NumericsProfile::StrictQuantized { version: 1 });
    assert_eq!(review.work, review.judges[0].planned_work());
}
#[test]
fn invalid_rule_paths_types_duplicates_and_missing_opt_in_fail_before_native_work() {
    let schema = r#"{"type":"number"}"#;
    let mut bad = rule(); bad.path = vec![ClaimPathStep::Property("missing".into())];
    assert!(prepare(schema, "source", false, options(vec![bad])).is_err());
    let mut bad = rule(); bad.value_kind = ClaimValueKind::String;
    assert!(prepare(schema, "source", false, options(vec![bad])).is_err());
    assert!(prepare(schema, "source", false, options(vec![rule(), rule()])).is_err());
    let mut bad = options(vec![rule()]); bad.specification.experimental_opt_in = false;
    assert!(prepare(schema, "source", false, bad).is_err());
}
#[test]
fn semantic_schema_cannot_be_detached_from_original_task_or_source() {
    let a = extraction(r#"{"type":"number"}"#, "amount one", false);
    let b = extraction(r#"{"type":"boolean"}"#, "amount two", false);
    assert!(a.extraction_plan().semantic_schema(a.task_plan(), b.source()).is_err());
    assert!(a.extraction_plan().semantic_schema(b.task_plan(), a.source()).is_err());
}
#[test]
fn same_model_comparison_keeps_all_non_task_fields_including_host_and_source_revision() {
    let a = identity(); let mut b = a.clone(); b.task_spec = "judge-v1".into();
    same_model(&a, &b).unwrap();
    for axis in 0..7 {
        let mut b = b.clone();
        match axis { 0 => b.source_revision.push('x'), 1 => b.quant_recipe.push('x'),
            2 => b.logical_model_digest = Sha256Digest::of_bytes(b"different"),
            3 => b.packing_set_digest = Sha256Digest::of_bytes(b"different"),
            4 => b.tokenizer_digest = Sha256Digest::of_bytes(b"different"),
            5 => b.backend_semantic_version.push('x'), _ => b.kv_dtype.push('x') }
        assert!(same_model(&a, &b).is_err(), "axis {axis}");
    }
}
#[test]
fn unconfigured_verbatim_null_and_empty_fields_stay_explicitly_unchecked() {
    for (schema, json, grounded, reason) in [
        (r#"{"type":"number"}"#, "1", false, SemanticNotChecked::NoClaimRule),
        (r#"{"type":"null"}"#, "null", false, SemanticNotChecked::NullValue),
        (r#"{"type":"array","maxItems":1,"items":{"type":"number"}}"#, "[]", false, SemanticNotChecked::EmptyContainer),
        (r#"{"type":"string","x-fnlp-source":"verbatim"}"#, r#""Alice""#, true, SemanticNotChecked::VerbatimMembershipOnly),
    ] {
        let p = prepare(schema, "Alice", grounded, options(vec![])).unwrap();
        let original = result(&p, schema, json, grounded);
        let review = p.prepare_review(original.clone(), &mut Continue).unwrap();
        assert!(review.judges.is_empty());
        let output = p.finish(review, vec![], &mut Continue).unwrap();
        assert_eq!(output.extraction, original); assert_eq!(output.semantic.attempted_fields, 0);
        assert_eq!(output.semantic.checked_fields, 0); assert_eq!(output.semantic.fields[0].not_checked, Some(reason));
        assert_eq!(output.semantic_model_work, Int8Work::default()); assert!(output.semantic.same_model_correlated);
        assert_eq!(output.reserved_model_work, p.extraction_reserved_work());
    }
}
#[test]
fn all_native_axes_reserve_maximum_extraction_work_not_early_eos_actuals() {
    let schema = r#"{"type":"number"}"#;
    let mut p = prepare(schema, "The amount is 42.", false, options(vec![rule()])).unwrap();
    let original = result(&p, schema, "42", false);
    let review = p.prepare_review(original.clone(), &mut Continue).unwrap();
    let exact = plus(p.extraction_reserved_work(), review.work).unwrap();
    p.options.max_model_work = exact;
    p.prepare_review(original.clone(), &mut Continue).unwrap();
    for axis in 0..5 {
        p.options.max_model_work = exact; let w = &mut p.options.max_model_work;
        match axis { 0 => w.forward_positions -= 1, 1 => w.projected_logits -= 1, 2 => w.attention_pairs -= 1,
            3 => w.projections.dot_products -= 1, _ => w.projections.multiply_accumulates -= 1 }
        assert!(matches!(p.prepare_review(original.clone(), &mut Continue), Err(Int8SemanticError::WorkBudget)), "axis {axis}");
    }
    p.options.max_model_work = plus(original.model_work, review.work).unwrap();
    assert!(matches!(p.prepare_review(original, &mut Continue), Err(Int8SemanticError::WorkBudget)));
}
#[test]
fn nested_array_fields_share_claim_and_complete_judge_planning_limits() {
    let schema = r#"{"type":"array","maxItems":2,"items":{"type":"number"}}"#;
    let mut r = rule(); r.path = vec![ClaimPathStep::EachItem];
    let mut p = prepare(schema, "The values are 1 and 2.", false, options(vec![r])).unwrap();
    let original = result(&p, schema, "[1,2]", false);
    let review = p.prepare_review(original.clone(), &mut Continue).unwrap();
    assert_eq!(review.fields[0].pointer, "/0"); assert_eq!(review.fields[1].pointer, "/1");
    assert_eq!(review.judges.len(), 2);
    p.options.limits.max_fields = 1;
    assert!(p.prepare_review(original.clone(), &mut Continue).is_err());
    p.options.limits.max_fields = 256; p.options.limits.max_total_prompt_tokens = 1;
    assert!(p.prepare_review(original.clone(), &mut Continue).is_err());
    p.options.limits.max_total_prompt_tokens = 1_000_000; p.options.limits.max_total_projected_logits = 1;
    assert!(p.prepare_review(original, &mut Continue).is_err());
}
#[test]
fn replaced_native_extraction_metadata_or_invalid_json_never_enters_a_judge() {
    let schema = r#"{"type":"number"}"#;
    let p = prepare(schema, "The amount is 1.", false, options(vec![rule()])).unwrap();
    for axis in 0..5 {
        let mut run = result(&p, schema, "1", false);
        match axis { 0 => run.result.output.json = "true".into(), 1 => run.model_work.attention_pairs += 1,
            2 => run.result.task_spec_version = "ner-v1".into(), 3 => { run.result.output.token_ids.pop(); },
            _ => run.result.output.numerics_profile = "hf-bf16-eager".into() }
        assert!(p.prepare_review(run, &mut Continue).is_err());
    }
    let review = p.prepare_review(result(&p, schema, "1", false), &mut Continue).unwrap();
    assert!(p.finish(review, vec![], &mut Continue).is_err());
}
#[test]
fn bounded_output_and_late_cancellation_do_not_publish_partial_second_reader_results() {
    let schema = r#"{"type":"number"}"#;
    let mut p = prepare(schema, "source", false, options(vec![])).unwrap();
    let review = p.prepare_review(result(&p, schema, "1", false), &mut Continue).unwrap();
    p.options.limits.max_output_bytes = 1; assert!(p.finish(review, vec![], &mut Continue).is_err());
    struct Count { calls: usize, stop: Option<usize> }
    impl DecodeStepControl for Count { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1; (self.stop == Some(self.calls)).then_some(DecodeCancellationKind::Deadline)
    } }
    let p = prepare(schema, "source", false, options(vec![rule()])).unwrap(); let original = result(&p, schema, "1", false);
    let mut c = Count { calls: 0, stop: None }; p.prepare_review(original.clone(), &mut c).unwrap();
    for stop in 1..=c.calls {
        let error = p.prepare_review(original.clone(), &mut Count { calls: 0, stop: Some(stop) }).err().unwrap();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
        assert!(!format!("{error:?}").contains("source"));
    }
}
