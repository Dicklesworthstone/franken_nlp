//! Model-free admission/protocol tests. No fixture output is an NLP-quality claim.
use super::*;
use crate::{execution_identity::Sha256Digest, template::{IM_START, IM_END, THINK_START, THINK_END},
    tokenizer::{bpe::EncodeOptions, embedded::EmbeddedTokenizer, specials::ArchivedControlRegistries}};
fn planner() -> SourceTaskPlanner {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = tokenizer.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let eos = entries[1]["id"].as_u64().unwrap() as u32;
    let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let registry = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    SourceTaskPlanner::pinned(registry.template_controls(), eos).unwrap()
}
fn budget() -> TaskBudget { TaskBudget { max_input_tokens: 4096, max_output_tokens: 128,
    max_output_bytes: 65536, max_grammar_states: 8192, max_kv_bytes: 1 << 30 } }
fn identity(p: &SourceTaskPlanner, kind: BuiltInTask) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"source-batch-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "bf16-verbatim".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(), task_spec: kind.spec().identity(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::HfBf16Eager, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d,
        decision_policy_digest: d, backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None }
}
fn args(kind: BuiltInTask) -> SourceBatchArgs {
    match kind {
        BuiltInTask::Ner => SourceBatchArgs::Ner { options: NerOptions { max_entities: 4, max_mention_scalars: 32,
            ..Default::default() }, budget: budget() },
        BuiltInTask::Keyphrases => SourceBatchArgs::Keyphrases { options: KeyphraseOptions { max_phrases: 4, max_phrase_scalars: 32 }, budget: budget() },
        BuiltInTask::Summarize => SourceBatchArgs::Summarize { options: SummaryOptions { max_bullets: 2,
            max_bullet_scalars: 64, max_citations_per_bullet: 2, max_quote_scalars: 32 }, budget: budget() },
        BuiltInTask::Answer => SourceBatchArgs::Answer { passages: vec![AnswerPassage { id: "p1".to_owned(), text: "Alice met Bob.".to_owned() }],
            options: AnswerOptions { max_answer_scalars: 128, max_citations: 4, max_quote_scalars: 32 }, budget: budget() },
        _ => unreachable!(),
    }
}
fn compiler(p: &SourceTaskPlanner, kind: BuiltInTask, defaults: bool) -> SourceBatchPlanner<'_> {
    SourceBatchPlanner::new(p, identity(p, kind), budget(), SourcePlanningLimits::default(), defaults.then(|| args(kind))).unwrap()
}
fn document(task_args: Option<SourceBatchArgs>) -> BatchDocument<SourceBatchArgs> {
    BatchDocument { id: "item".to_owned(), text: "Alice <think> 上海".to_owned(), task_args }
}
#[test]
fn all_four_source_families_prepare_real_native_plans_and_work_ceilings() {
    let p = planner();
    for kind in [BuiltInTask::Ner, BuiltInTask::Keyphrases, BuiltInTask::Summarize, BuiltInTask::Answer] {
        let c = compiler(&p, kind, false); let plan = c.prepare(document(Some(args(kind)))).unwrap();
        assert_eq!(plan.execution_identity().task_spec, kind.spec().identity());
        assert_eq!(plan.work.forward_positions, plan.plan.prompt_tokens() as u64 + u64::from(budget().max_output_tokens) - 1);
        assert_eq!(plan.work.projected_logits, plan.work.forward_positions * NANBEIGE_VOCAB_SIZE as u64);
        plan.verify_identity(plan.execution_identity()).unwrap();
        let mut wrong = plan.execution_identity().clone(); wrong.logical_model_digest = Sha256Digest::of_bytes(b"wrong");
        assert_eq!(plan.verify_identity(&wrong).unwrap_err().code, BatchCode::Admission);
    }
}
#[test]
fn defaults_are_explicit_and_records_cannot_switch_the_admitted_task() {
    let p = planner(); let c = compiler(&p, BuiltInTask::Keyphrases, true);
    assert!(c.prepare(document(None)).is_ok());
    assert!(c.prepare(document(Some(args(BuiltInTask::Answer)))).is_err());
    assert!(compiler(&p, BuiltInTask::Keyphrases, false).prepare(document(None)).is_err());
    assert!(SourceBatchPlanner::new(&p, identity(&p, BuiltInTask::Keyphrases), budget(), SourcePlanningLimits::default(),
        Some(args(BuiltInTask::Answer))).is_err());
}
#[test]
fn qa_question_is_batch_text_and_passages_are_separately_bound() {
    let p = planner(); let c = compiler(&p, BuiltInTask::Answer, false);
    let a = c.prepare(document(Some(args(BuiltInTask::Answer)))).unwrap();
    let mut b = document(Some(args(BuiltInTask::Answer))); b.text.push('?');
    assert_ne!(a.execution_identity().prompt_digest, c.prepare(b).unwrap().execution_identity().prompt_digest);
    let mut missing = args(BuiltInTask::Answer);
    if let SourceBatchArgs::Answer { passages, .. } = &mut missing { passages.clear(); }
    let error = match c.prepare(document(Some(missing))) { Err(e) => e, Ok(_) => panic!("missing passages admitted") };
    assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::Planning);
}
#[test]
fn native_accounting_rejects_every_mismatched_counter_and_missing_eos_slot() {
    let ceiling = BatchWork { forward_positions: 10, projected_logits: 10 * NANBEIGE_VOCAB_SIZE as u64 };
    let good = ObservedWork { tokens: 2, positions: 4, logits: 4 * NANBEIGE_VOCAB_SIZE as u64, mask_visits: 20 };
    assert!(check_accounting(3, 8, ceiling, good, 20).is_ok());
    for axis in 0..5 {
        let mut bad = good;
        match axis { 0 => bad.tokens = 0, 1 => bad.tokens = 9, 2 => bad.positions += 1,
            3 => bad.logits += 1, _ => bad.mask_visits += 1 }
        assert!(check_accounting(3, 8, ceiling, bad, 20).unwrap_err().stop);
    }
    assert!(check_accounting(u64::MAX, 8, ceiling, good, 20).is_err());
}
#[test]
fn full_resident_kv_is_admitted_not_only_the_used_prefix() {
    let bytes = 10 * KV_BYTES_PER_TOKEN as u64;
    assert!(check_capacity(true, 10, 10, bytes).is_ok());
    assert!(!check_capacity(true, 10, 11, bytes).unwrap_err().stop);
    assert!(!check_capacity(true, 10, 1, bytes - 1).unwrap_err().stop);
    assert!(check_capacity(false, 10, 1, bytes).unwrap_err().stop);
    assert!(check_capacity(true, u64::MAX, 1, u64::MAX).unwrap_err().stop);
}
#[test]
fn mask_reservations_do_not_refund_failed_work_or_renew_on_epochs() {
    let mut remaining = 6;
    reserve_masks(&mut remaining, 3).unwrap();
    // An admission failure or a flush does not mutate this allowance.
    reserve_masks(&mut remaining, 3).unwrap(); assert_eq!(remaining, 0);
    assert_eq!(reserve_masks(&mut remaining, 3).unwrap_err().fault.code, BatchCode::WorkLimit);
    assert_eq!(remaining, 0);
}
#[test]
fn native_cancellation_and_allocation_stop_admission_but_invalid_answers_do_not_abstain() {
    let cancelled = execution_failure(SourcePlanningError::Answer(AnswerError::Extraction(
        ExtractError::Decode(JsonDecodeError::Cancelled(DecodeCancellationKind::Deadline)))));
    assert!(cancelled.stop); assert_eq!(cancelled.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    let allocation = preparation_failure(SourcePlanningError::Answer(AnswerError::AllocationRefused));
    assert!(allocation.stop); assert_eq!(allocation.fault.code, BatchCode::Allocation);
    let invalid = execution_failure(SourcePlanningError::Answer(AnswerError::InvalidResult));
    assert!(!invalid.stop); assert_eq!(invalid.fault.code, BatchCode::Execution);
    let overflow = execution_failure(SourcePlanningError::OutputBudget);
    assert!(!overflow.stop); assert_eq!(overflow.fault.code, BatchCode::OutputLineLimit);
}
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
/// Exercise the REAL source planner and NDJSON runner, without pretending that
/// a fixture performs model inference. Output explicitly says validation_only.
struct PlanningOnly<'p> { compiler: SourceBatchPlanner<'p> }
impl BatchProcessor for PlanningOnly<'_> {
    type Args = SourceBatchArgs;
    type Prepared = PreparedBatchSource;
    type Output = serde_json::Value;
    fn prepare(&mut self, doc: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> { self.compiler.prepare(doc) }
    fn planned_work(&self, plan: &Self::Prepared) -> BatchWork { plan.work }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        Ok(serde_json::json!({"validation_only":true}))
    }
}
#[test]
fn source_batch_work_allowance_survives_real_ndjson_flush_epochs() {
    let p = planner(); let c = compiler(&p, BuiltInTask::Keyphrases, true);
    let charge = c.prepare(document(None)).unwrap().work;
    let record = serde_json::json!({"id":"item","text":"Alice <think> 上海"}).to_string();
    let input = format!("{record}\n{{\"flush\":true}}\n{record}\n");
    let mut output = Vec::new(); let mut processor = PlanningOnly { compiler: c };
    let result = run_ndjson(&mut std::io::Cursor::new(input), &mut output, &mut processor,
        BatchLimits { max_work: charge, ..BatchLimits::default() }, &mut Continue).unwrap();
    assert_eq!(result.succeeded, 1); assert_eq!(result.failed, 1); assert_eq!(result.reserved_work, charge);
    let text = String::from_utf8(output).unwrap(); assert!(text.contains("validation_only")); assert!(text.contains("work_limit"));
}
#[test]
fn admission_guard_stays_live_through_serialization_and_until_result_drop() {
    use std::{cell::Cell, rc::Rc};
    struct Guard(Rc<Cell<bool>>);
    impl Drop for Guard { fn drop(&mut self) { self.0.set(false); } }
    let live = Rc::new(Cell::new(true));
    let output = GuardedOutput::new(serde_json::json!({"validation_only":true}), Guard(live.clone()));
    let _bytes = canonjson::canonical_bytes(&output).unwrap(); assert!(live.get());
    drop(output); assert!(!live.get());
}
