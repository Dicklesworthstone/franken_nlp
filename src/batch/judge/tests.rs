//! Exercise the REAL pinned judge planner/scorer via the batch transport.
//! Synthetic logits test wiring only; no accuracy or native qualification.
use super::*;
use crate::{
    execution_identity::Sha256Digest,
    native_engine::lmhead::{NANBEIGE_VOCAB_SIZE, scoring::ProjectionRows},
    tasks::{ir::PromptSegment, judge::JudgeLogits},
    tokenizer::{bpe::EncodeOptions, embedded::EmbeddedTokenizer, specials::ArchivedControlRegistries},
    template::{IM_START, IM_END, THINK_START, THINK_END},
};
use std::io::Cursor;
fn budget() -> TaskBudget { TaskBudget { max_input_tokens: 4096, max_output_tokens: 16,
    max_output_bytes: 1_000_000, max_grammar_states: 4096, max_kv_bytes: 1 << 30 } }
fn policy() -> FaithfulnessPolicy { FaithfulnessPolicy { minimum_candidate_weight_ppm: 500000,
    minimum_margin_milli: 100, evidence_window_bytes: 128, max_evidence_windows: 31, max_evidence_spans: 31 } }
fn fixture() -> (JudgePlanner, ExecutionIdentity, u32) {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = tokenizer.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let controls = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    let planner = JudgePlanner::pinned(controls.template_controls(), entries[1]["id"].as_u64().unwrap() as u32).unwrap();
    let d = Sha256Digest::of_bytes(b"fixture");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "bf16-verbatim".to_owned(), packing_set_digest: d,
        tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(), task_spec: "judge-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::HfBf16Eager, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None };
    let label = tokenizer.tokenizer().encode_byte_fallback_only(b"E").unwrap()[0];
    (planner, identity, label)
}
struct Model { token: u32, calls: usize }
impl JudgeLogits for Model {
    type Error = &'static str;
    fn project(&mut self, _: usize, _: &[PromptSegment], prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        self.calls += 1;
        let ProjectionRows::FullVocabulary { vocabulary_size } = rows else { panic!("complete denominator required") };
        let mut logits = vec![0.0; vocabulary_size]; if prefix.is_empty() { logits[self.token as usize] = 8.0; }
        Ok(logits)
    }
}
struct Provider<'a> { compiler: JudgeBatchPlanner<'a>, model: Model }
impl BatchProcessor for Provider<'_> {
    type Args = JudgeBatchArgs; type Prepared = PreparedBatchJudge; type Output = JudgeResult;
    fn prepare(&mut self, doc: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> { self.compiler.prepare(doc) }
    fn planned_work(&self, p: &Self::Prepared) -> BatchWork { p.work }
    fn execute<C: DecodeStepControl>(&mut self, p: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        p.plan.execute(p.execution_identity(), &mut self.model).map_err(|_| BatchItemFailure::fatal(BatchCode::Execution))
    }
}
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
#[test]
fn multiple_documents_use_one_planner_and_retain_actual_faithfulness_evidence() {
    let (planner, identity, token) = fixture();
    let defaults = JudgeBatchArgs::Faithfulness { claim: "The statement is true.".to_owned(), policy: policy(), budget: budget() };
    let compiler = JudgeBatchPlanner::new(&planner, identity, budget(), JudgeLimits::default(), Some(defaults)).unwrap();
    let mut provider = Provider { compiler, model: Model { token, calls: 0 } };
    let input = "{\"id\":\"a\",\"text\":\"é yes\"}\n{\"id\":\"b\",\"text\":\"Second source <think>\"}\n";
    let mut output = Vec::new();
    let result = run_ndjson(&mut Cursor::new(input.as_bytes()), &mut output, &mut provider, BatchLimits::default(), &mut Continue).unwrap();
    assert_eq!(result.succeeded, 2); assert_eq!(provider.model.calls, 8);
    assert_eq!(result.reserved_work.projected_logits, 8 * NANBEIGE_VOCAB_SIZE as u64);
    let rows: Vec<_> = std::str::from_utf8(&output).unwrap().lines().map(|s| canonjson::parse_str(s).unwrap()).collect();
    assert_eq!(rows[1]["result"]["result"]["evidence"][0]["quote"], "é yes");
    assert_eq!(rows[2]["result"]["result"]["evidence"][0]["quote"], "Second source <think>");
}
#[test]
fn per_document_arguments_cannot_raise_the_frozen_host_budget() {
    let (planner, identity, _) = fixture();
    let compiler = JudgeBatchPlanner::new(&planner, identity, budget(), JudgeLimits::default(), None).unwrap();
    let mut bigger = budget(); bigger.max_input_tokens += 1;
    let document = BatchDocument { id: "a".to_owned(), text: "source".to_owned(), task_args: Some(JudgeBatchArgs::Faithfulness {
        claim: "claim".to_owned(), policy: policy(), budget: bigger }) };
    let error = match compiler.prepare(document) { Err(e) => e, Ok(_) => panic!("larger budget admitted") };
    assert_eq!(error.fault.code, BatchCode::Planning); assert!(!error.stop);
}
#[test]
fn missing_arguments_are_not_replaced_by_an_invented_task() {
    let (planner, identity, _) = fixture();
    let compiler = JudgeBatchPlanner::new(&planner, identity, budget(), JudgeLimits::default(), None).unwrap();
    assert!(compiler.prepare(BatchDocument { id: "a".to_owned(), text: "source".to_owned(), task_args: None }).is_err());
}
#[test]
fn source_and_claim_swaps_change_the_private_execution_identity() {
    let (planner, identity, _) = fixture();
    let compiler = JudgeBatchPlanner::new(&planner, identity, budget(), JudgeLimits::default(), None).unwrap();
    let plan = |source: &str, claim: &str| compiler.prepare(BatchDocument { id: "ignored".to_owned(), text: source.to_owned(),
        task_args: Some(JudgeBatchArgs::Faithfulness { claim: claim.to_owned(), policy: policy(), budget: budget() }) }).unwrap();
    let a = plan("source", "claim"); let b = plan("claim", "source");
    assert_ne!(a.execution_identity().prompt_digest, b.execution_identity().prompt_digest);
}
#[test]
fn pairwise_mapping_preserves_order_and_raw_marker_text() {
    let args = JudgeBatchArgs::Pairwise { criterion: "quality".to_owned(), b: "second".to_owned(),
        policy: PairwisePolicy { minimum_margin_milli: 0, maximum_order_disagreement_milli: 0 }, budget: budget() };
    let JudgeRequest::Pairwise { a, b, .. } = args.into_request("first <|im_start|>".to_owned()) else { panic!() };
    assert_eq!(a, "first <|im_start|>"); assert_eq!(b, "second");
}
#[test]
fn cancellation_and_engine_corruption_stop_but_context_refusal_is_item_local() {
    let cancelled = native_failure(PrefixScoringError::Cancelled(DecodeCancellationKind::Deadline).into());
    assert!(cancelled.stop); assert_eq!(cancelled.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    assert!(native_failure(PrefixScoringError::Poisoned.into()).stop);
    assert!(native_failure(PrefixScoringError::EngineAlreadyPrimed.into()).stop);
    assert!(!native_failure(PrefixScoringError::ContextBudget.into()).stop);
}
