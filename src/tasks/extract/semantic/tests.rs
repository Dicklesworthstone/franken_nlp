//! Source-only wiring regressions. Synthetic logits are not quality evidence.
use super::*;
use crate::{
    grammar::{CompileLimits, runtime::{SOURCE_JSON_RUNTIME_VERSION, SourceRuntimeLimits}},
    native_engine::{constrained::{JsonDecodeOptions, JsonDecodeOutput}, hf_bf16_eager::HF_BF16_EAGER_PROFILE,
        lmhead::{NANBEIGE_VOCAB_SIZE, scoring::ProjectionRows}},
    tasks::{BuiltInTask, extract::SourceDocumentEncoder, ir::{Candidate, DecodeStrategy, DependencyScope,
        FinitePostcondition, GrammarReference, PromptSegment, TaskIR}},
    template::{IM_START, IM_END, THINK_START, THINK_END},
    tokenizer::{bpe::EncodeOptions, embedded::EmbeddedTokenizer, specials::ArchivedControlRegistries},
};
const SCHEMA: &str = r#"{"type":"object","additionalProperties":false,"properties":{"amount":{"type":"number"},"name":{"type":"string","x-fnlp-source":"verbatim"},"tax":{"type":"number"},"unmapped":{"type":"boolean"}},"required":["amount","name","tax","unmapped"]}"#;
struct Fixture {
    plan: ExtractPlan, task: TaskPlan, source: SourceDocument, extraction: ExtractResult,
    extraction_identity: ExecutionIdentity, judge_base: ExecutionIdentity, planner: JudgePlanner,
    encoder: SourceDocumentEncoder, spec: SemanticVerificationSpec, budget: TaskBudget, entailed_token: u32,
}
impl Fixture {
    fn new() -> Self {
        let tokenizer = EmbeddedTokenizer::pinned().unwrap();
        let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
            let ids = tokenizer.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
            assert_eq!(ids.len(), 1);
            serde_json::json!({"id":ids[0], "special":surface == IM_START || surface == IM_END, "surface":surface})
        }).collect();
        let eos = entries[1]["id"].as_u64().unwrap() as u32;
        let special: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
        let registry = ArchivedControlRegistries::from_archived_json(
            &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":special}).to_string(),
            &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
        let encoder = SourceDocumentEncoder::pinned(registry.template_controls()).unwrap();
        let source = encoder.encode("Alice paid 12.50 and tax was 2.50.", 4096, 4096).unwrap();
        let budget = TaskBudget { max_input_tokens: 4096, max_output_tokens: 64, max_output_bytes: 1_000_000,
            max_grammar_states: 4096, max_kv_bytes: 1 << 30 };
        let base = super::super::tests::identity();
        let ir = TaskIR::new(vec![PromptSegment::new(PromptSegmentKind::TaskInstruction, vec![1]),
            PromptSegment::new(PromptSegmentKind::Document, source.token_ids().to_vec()),
            PromptSegment::new(PromptSegmentKind::AnswerScaffold, vec![2])], DecodeStrategy::ConstrainedJson,
            GrammarReference::json_schema(Sha256Digest::of_bytes(SCHEMA.as_bytes()), SOURCE_JSON_RUNTIME_VERSION), None,
            vec![FinitePostcondition::JsonValid, FinitePostcondition::SourceSpansVerified, FinitePostcondition::OutputWithinBudget],
            budget, DependencyScope::ItemLocal).unwrap();
        let task = TaskPlan::new(BuiltInTask::Extract.spec(), &PlanContext::new(&base, budget).unwrap(), ir).unwrap();
        let plan = ExtractPlan::from_task_plan_with_source(&task, SCHEMA,
            JsonDecodeOptions { max_new_tokens: 64, eos_token_id: eos, excluded_token_ids: Default::default() },
            CompileLimits::default(), registry.template_controls(), &source, SourceRuntimeLimits::default()).unwrap();
        let extraction_identity = plan.bind_identity(base).unwrap();
        let output = JsonDecodeOutput { schema_version: 1, numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(),
            token_ids: vec![source.token_ids()[0], eos], json: r#"{"amount":1.25e1,"name":"Alice","tax":2.5e0,"unmapped":true}"#.to_owned(),
            forward_positions: 4, projected_logits: 4 * NANBEIGE_VOCAB_SIZE as u64, mask_node_visit_charge: 20 };
        let extraction = plan.finalize(output).unwrap();
        let planner = JudgePlanner::pinned(registry.template_controls(), eos).unwrap();
        let mut judge_base = extraction_identity.clone(); judge_base.task_spec = "judge-v1".to_owned();
        judge_base.template_digest = *planner.template_digest(); judge_base.tokenizer_digest = planner.tokenizer_digest();
        let claims = [("amount", "Paid amount was "), ("tax", "Tax amount was ")].iter().map(|(name, prefix)| ClaimRule {
            id: (*name).to_owned(), path: vec![ClaimPathStep::Property((*name).to_owned())], value_kind: ClaimValueKind::Number,
            prefix: (*prefix).to_owned(), suffix: ".".to_owned(),
        }).collect();
        let spec = SemanticVerificationSpec { schema_version: 1, experimental_opt_in: true, revision: "invoice-v1".to_owned(), claims,
            judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 500000, minimum_margin_milli: 100,
                evidence_window_bytes: 128, max_evidence_windows: 31, max_evidence_spans: 31 }, judge_budget: budget };
        let ids = tokenizer.tokenizer().encode_byte_fallback_only(b"E").unwrap(); assert_eq!(ids.len(), 1);
        Self { plan, task, source, extraction, extraction_identity, judge_base, planner, encoder, spec, budget, entailed_token: ids[0] }
    }
    fn prepare(&self) -> PreparedSemanticVerification {
        self.plan.prepare_semantic_verification(&self.task, &self.source, &self.extraction, &self.extraction_identity,
            &self.spec, &self.planner, &PlanContext::new(&self.judge_base, self.budget).unwrap(),
            JudgeLimits::default(), SemanticLimits::default()).unwrap()
    }
    fn model(&self) -> Model { Model { token: self.entailed_token, calls: 0, fail_after: usize::MAX, uncertain: false } }
}
struct Model { token: u32, calls: usize, fail_after: usize, uncertain: bool }
impl JudgeLogits for Model {
    type Error = &'static str;
    fn project(&mut self, _: usize, _: &[PromptSegment], prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        self.calls += 1;
        if self.calls > self.fail_after { return Err("private extraction fixture detail"); }
        let ProjectionRows::FullVocabulary { vocabulary_size } = rows else { panic!("full denominator required") };
        let mut logits = vec![0.0; vocabulary_size];
        if prefix.is_empty() && !self.uncertain { logits[self.token as usize] = 8.0; }
        Ok(logits)
    }
}
#[test]
fn real_field_pipeline_retains_extraction_and_exact_claims_with_explicit_coverage() {
    let f = Fixture::new(); let prepared = f.prepare(); let ids: Vec<_> = prepared.judge_identities().cloned().collect(); let mut model = f.model();
    let result = prepared.execute(&ids, &mut model).unwrap();
    assert_eq!(result.extraction, f.extraction); assert_eq!(result.semantic.attempted_fields, 2);
    assert_eq!(result.semantic.checked_fields, 2); assert_eq!(result.semantic.not_checked_fields, 2); assert_eq!(model.calls, 8);
    assert_eq!(result.semantic.fields[0].rendered_claim.as_deref(), Some("Paid amount was 1.25e1."));
    assert_eq!(result.semantic.fields[1].not_checked, Some(SemanticNotChecked::VerbatimMembershipOnly));
    assert_eq!(result.semantic.fields[3].not_checked, Some(SemanticNotChecked::NoClaimRule));
    assert_eq!(result.semantic.fields[0].same_model_correlated, Some(true));
}
#[test]
fn judge_abstention_is_not_mislabeled_unsupported_or_successful_verification() {
    let f = Fixture::new(); let prepared = f.prepare(); let ids: Vec<_> = prepared.judge_identities().cloned().collect(); let mut model = f.model(); model.uncertain = true;
    let result = prepared.execute(&ids, &mut model).unwrap();
    assert_eq!(result.semantic.attempted_fields, 2); assert_eq!(result.semantic.checked_fields, 0);
    assert_eq!(result.semantic.fields[0].status, SemanticFieldStatus::NotChecked);
    assert_eq!(result.semantic.fields[0].not_checked, Some(SemanticNotChecked::JudgeAbstained));
    assert!(result.semantic.fields[0].judge.is_some());
}
#[test]
fn later_identity_mismatch_refuses_before_any_field_callback() {
    let f = Fixture::new(); let prepared = f.prepare(); let mut ids: Vec<_> = prepared.judge_identities().cloned().collect();
    ids[1].logical_model_digest = Sha256Digest::of_bytes(b"wrong model"); let mut model = f.model();
    assert!(prepared.execute(&ids, &mut model).is_err()); assert_eq!(model.calls, 0);
}
#[test]
fn failed_second_field_returns_no_composite_or_silently_skipped_field() {
    let f = Fixture::new(); let prepared = f.prepare(); let ids: Vec<_> = prepared.judge_identities().cloned().collect(); let mut model = f.model(); model.fail_after = 4;
    assert!(prepared.execute(&ids, &mut model).is_err()); assert_eq!(model.calls, 5);
}
#[test]
fn substituting_source_or_another_model_fails_before_claim_execution() {
    let f = Fixture::new(); let wrong = f.encoder.encode("Mallory paid nothing.", 4096, 4096).unwrap();
    let context = PlanContext::new(&f.judge_base, f.budget).unwrap();
    assert!(f.plan.prepare_semantic_verification(&f.task, &wrong, &f.extraction, &f.extraction_identity,
        &f.spec, &f.planner, &context, JudgeLimits::default(), SemanticLimits::default()).is_err());
    let mut other = f.judge_base.clone(); other.logical_model_digest = Sha256Digest::of_bytes(b"different model");
    assert!(f.plan.prepare_semantic_verification(&f.task, &f.source, &f.extraction, &f.extraction_identity,
        &f.spec, &f.planner, &PlanContext::new(&other, f.budget).unwrap(), JudgeLimits::default(), SemanticLimits::default()).is_err());
}
#[test]
fn opt_in_and_complete_output_limit_are_enforced() {
    let mut f = Fixture::new(); f.spec.experimental_opt_in = false;
    assert!(f.plan.prepare_semantic_verification(&f.task, &f.source, &f.extraction, &f.extraction_identity,
        &f.spec, &f.planner, &PlanContext::new(&f.judge_base, f.budget).unwrap(), JudgeLimits::default(), SemanticLimits::default()).is_err());
    f.spec.experimental_opt_in = true; let mut prepared = f.prepare(); prepared.max_output_bytes = 1;
    let ids: Vec<_> = prepared.judge_identities().cloned().collect();
    assert!(matches!(prepared.execute(&ids, &mut f.model()), Err(SemanticError::Limit("complete_semantic_output"))));
}
#[test]
fn no_claim_rules_means_no_model_calls_not_an_all_fields_verified_claim() {
    let mut f = Fixture::new(); f.spec.claims.clear(); let prepared = f.prepare(); let mut model = f.model();
    let result = prepared.execute(&[], &mut model).unwrap();
    assert_eq!(model.calls, 0); assert_eq!(result.semantic.checked_fields, 0); assert_eq!(result.semantic.not_checked_fields, 4);
}
#[test]
fn strict_specification_parser_rejects_duplicate_and_unknown_keys() {
    assert!(SemanticVerificationSpec::from_json(r#"{"schema_version":1,"schema_version":1}"#, 4096).is_err());
    let f = Fixture::new(); let mut json = serde_json::to_value(&f.spec).unwrap(); json["auto_accept"] = serde_json::json!(true);
    assert!(SemanticVerificationSpec::from_json(&json.to_string(), 65536).is_err());
}
