//! Real pinned source preparation; the tiny model fixture supplies identity
//! only and never performs a native forward or a quality assessment.
use super::*;
use crate::{native_engine::strict_int4::{Int4ModelIdentity, weights::projection_fixture},
    tasks::judge::FaithfulnessPolicy,
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls, untrusted::SafeBpeLimits}};

fn model() -> &'static Int4ModelIdentity {
    static MODEL: std::sync::OnceLock<Int4ModelIdentity> = std::sync::OnceLock::new();
    MODEL.get_or_init(|| projection_fixture(32).0.identity().clone())
}
fn task_budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 512, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn options() -> Int4SourceSemanticOptions {
    Int4SourceSemanticOptions { schema_version: 1, experimental_opt_in: true,
        judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 500_000, minimum_margin_milli: 100,
            evidence_window_bytes: 4096, max_evidence_windows: 8, max_evidence_spans: 8 },
        judge_budget: task_budget(), limits: SourceSemanticLimits::default(), max_model_work: work(u64::MAX / 4) }
}
fn run_limits() -> Int4SemanticSourceBatchLimits {
    Int4SemanticSourceBatchLimits { max_model_work: work(u64::MAX), masks: ExtractionMaskBudget {
        per_mask: MaskWorkLimits::default(), max_visits_per_item: 1_000_000_000, max_visits_per_run: 4_000_000_000 } }
}
fn task(answer: bool) -> Int4SemanticSourceBatchTask {
    if answer { Int4SemanticSourceBatchTask::Answer { options: AnswerOptions {
        max_answer_scalars: 128, max_citations: 4, max_quote_scalars: 32 }, budget: task_budget() } }
    else { Int4SemanticSourceBatchTask::Summarize { options: SummaryOptions { max_bullets: 2,
        max_bullet_scalars: 64, max_citations_per_bullet: 2, max_quote_scalars: 32 }, budget: task_budget() } }
}
fn source(encoding: UntrustedEncoding) -> SourceTaskPlanner {
    let controls = pinned_controls::pinned().unwrap();
    let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
    SourceTaskPlanner::pinned(controls.template_controls(), eos).unwrap().with_document_encoding(encoding).unwrap()
}
fn judge(encoding: UntrustedEncoding) -> Arc<Int4JudgePlanner> {
    Arc::new(Int4JudgePlanner::for_model(model(), task_budget(), JudgeLimits::default()).unwrap()
        .with_document_encoding(encoding).unwrap())
}
fn base(planner: &SourceTaskPlanner, task: Int4SemanticSourceBatchTask, judge: &Int4JudgePlanner) -> ExecutionIdentity {
    let mut identity = judge.execution_identity().clone();
    identity.task_spec = task.built_in().spec().identity();
    identity.template_digest = *planner.template_digest(); identity.tokenizer_digest = planner.tokenizer_digest();
    identity
}
fn configured(planner: &SourceTaskPlanner, task: Int4SemanticSourceBatchTask,
    planning: SourcePlanningLimits, options: Int4SourceSemanticOptions) -> Int4SemanticSourceBatchPlanner<'_> {
    let judge = judge(planner.document_encoding().clone());
    let identity = base(planner, task, &judge);
    Int4SemanticSourceBatchPlanner::new(planner, identity, task_budget(), planning, task,
        judge, options, run_limits()).unwrap()
}
fn factory(planner: &SourceTaskPlanner, answer: bool) -> Int4SemanticSourceBatchPlanner<'_> {
    configured(planner, task(answer), SourcePlanningLimits::default(), options())
}
fn document() -> BatchDocument<Int4SemanticSourceBatchArgs> {
    BatchDocument { id: "source-item".to_owned(), text: "Alice met Bob in 上海.".to_owned(), task_args: None }
}
fn question() -> BatchDocument<Int4SemanticSourceBatchArgs> {
    BatchDocument { id: "question-item".to_owned(), text: "Who met Bob?".to_owned(),
        task_args: Some(Int4SemanticSourceBatchArgs { passages: vec![
            AnswerPassage { id: "first".to_owned(), text: "Alice met Bob.".to_owned() },
            AnswerPassage { id: "second".to_owned(), text: "上海 is in the second passage.".to_owned() },
        ] }) }
}

#[test]
fn prepared_sources_share_one_judge_and_equal_direct_original_request_plans() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Int4JudgePlanner>(); send_sync::<SourceTaskPlanner>();
    send_sync::<Int4SemanticSourceBatchPlanner<'static>>();
    let planner = source(UntrustedEncoding::ByteFallback);
    for answer in [false, true] {
        let compiler = factory(&planner, answer);
        let input = || if answer { question() } else { document() };
        let first = compiler.prepare(input(), &mut Control::default()).unwrap();
        let second = compiler.prepare(input(), &mut Control::default()).unwrap();
        assert!(std::ptr::eq(first.judge_identity(), second.judge_identity()));
        assert_eq!(Arc::strong_count(&compiler.judge), 3);
        let request = compiler.task.request(input()).unwrap();
        let context = PlanContext::new(compiler.base_execution_identity(), task_budget()).unwrap();
        let direct_source = planner.plan_int4_with_control(&request, &context, compiler.planning,
            32, &mut Control::default()).unwrap();
        let direct = PreparedInt4SourceSemantic::new(direct_source, request,
            Int4JudgePlanner::for_model(model(), task_budget(), JudgeLimits::default()).unwrap(),
            options(), &mut Control::default()).unwrap();
        assert_eq!(first.execution_identity(), direct.execution_identity());
        assert_eq!(first.source_reserved_work(), direct.source_reserved_work());
        assert_eq!(first.input_facts(), direct.input_facts());
        assert_eq!(first.policy_digest(), direct.policy_digest());
        drop(first); drop(second); assert_eq!(Arc::strong_count(&compiler.judge), 1);
    }
}

#[test]
fn task_and_review_policy_are_fixed_before_any_private_record() {
    let planner = source(UntrustedEncoding::ByteFallback);
    for axis in 0..7 {
        let mut task = task(false); let mut options = options();
        let mut planning = SourcePlanningLimits::default();
        let judge = judge(UntrustedEncoding::ByteFallback); let mut identity = base(&planner, task, &judge);
        match axis {
            0 => options.experimental_opt_in = false,
            1 => options.limits.max_statements = 1,
            2 => options.limits.max_output_bytes = task_budget().max_output_bytes - 1,
            3 => planning.compiler.max_schema_bytes = 1,
            4 => identity.logical_model_digest = Sha256Digest::of_bytes(b"another model"),
            5 => identity.task_spec = "answer-v1".to_owned(),
            _ => if let Int4SemanticSourceBatchTask::Summarize { ref mut budget, .. } = task {
                budget.max_input_tokens += 1;
            },
        }
        assert!(Int4SemanticSourceBatchPlanner::new(&planner, identity, task_budget(), planning,
            task, judge, options, run_limits()).is_err());
    }
    let compiler = factory(&planner, false);
    let mut input = document(); input.task_args = Some(Int4SemanticSourceBatchArgs { passages: vec![] });
    let mut control = Control::default();
    assert_eq!(failure(compiler.prepare(input, &mut control)).fault.code, BatchCode::InvalidEnvelope);
    assert_eq!(control.calls, 1);
    let compiler = factory(&planner, true);
    assert_eq!(failure(compiler.prepare(document(), &mut Control::default())).fault.code, BatchCode::InvalidEnvelope);
    for json in [r#"{"passages":[],"options":{}}"#, r#"{"passages":[],"budget":{}}"#,
        r#"{"passages":[],"task":"summarize"}"#, r#"{"passages":[{"id":"p","text":"x","question":"y"}]}"#] {
        assert!(serde_json::from_str::<Int4SemanticSourceBatchArgs>(json).is_err());
    }
}

#[test]
fn qa_admission_counts_canonical_manifest_and_all_passages_not_only_question_text() {
    let planner = source(UntrustedEncoding::ByteFallback);
    let tiny = BatchDocument { id: "q".to_owned(), text: "Who?".to_owned(),
        task_args: Some(Int4SemanticSourceBatchArgs { passages: vec![
            AnswerPassage { id: "p".to_owned(), text: "Alice".to_owned() }] }) };
    // The visible question, passage text and id total ten bytes. The existing
    // AnswerInputLimits also count the mandatory canonical passage manifest.
    let compiler = configured(&planner, task(true), SourcePlanningLimits {
        max_input_bytes: 32, ..SourcePlanningLimits::default() }, options());
    let error = failure(compiler.prepare(tiny, &mut Control::default()));
    assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::Planning);
    let one_passage = configured(&planner, task(true), SourcePlanningLimits {
        max_passages: 1, ..SourcePlanningLimits::default() }, options());
    assert!(!failure(one_passage.prepare(question(), &mut Control::default())).stop);
    let compiler = factory(&planner, true);
    let first = compiler.prepare(question(), &mut Control::default()).unwrap();
    for axis in 0..4 {
        let mut input = question();
        match axis {
            0 => input.text.push('?'),
            1 => input.task_args.as_mut().unwrap().passages[0].id.push('x'),
            2 => input.task_args.as_mut().unwrap().passages[0].text.push('!'),
            _ => input.task_args.as_mut().unwrap().passages.reverse(),
        }
        let changed = compiler.prepare(input, &mut Control::default()).unwrap();
        assert_ne!(first.execution_identity(), changed.execution_identity());
    }
    assert_eq!(first.input_facts().passage_count, 2);
    assert_eq!(first.input_facts().question_bytes, question().text.len());
}

#[test]
fn shared_constructor_still_refuses_a_different_original_question_or_evidence() {
    let planner = source(UntrustedEncoding::ByteFallback); let compiler = factory(&planner, true);
    let original = compiler.task.request(question()).unwrap();
    let context = PlanContext::new(compiler.base_execution_identity(), task_budget()).unwrap();
    for axis in 0..3 {
        let source = planner.plan_int4_with_control(&original, &context, compiler.planning, 32, &mut Control::default()).unwrap();
        let mut changed = original.clone();
        let SourceTaskRequest::Answer { question, passages, .. } = &mut changed else { unreachable!() };
        match axis { 0 => question.push('?'), 1 => passages[0].id.push('x'), _ => passages.reverse() }
        let result = PreparedInt4SourceSemantic::new_shared(source, changed, Arc::clone(&compiler.judge),
            options(), &mut Control::default());
        assert!(matches!(result, Err(Int4SourceSemanticError::InputMismatch)));
    }
}

#[test]
fn compact_qa_uses_one_independent_whole_record_tokenizer_ledger() {
    let encoding = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 65_536,
        max_output_tokens: 65_536, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 });
    let planner = source(encoding.clone()); let compiler = factory(&planner, true);
    let first = compiler.prepare(question(), &mut Control::default()).unwrap();
    let second = compiler.prepare(question(), &mut Control::default()).unwrap();
    assert_eq!(first.input_encoding(), &encoding);
    assert_eq!(first.input_encoding_work(), second.input_encoding_work());
    let facts = first.input_facts();
    assert!(first.input_encoding_work().unwrap().input_bytes() > facts.question_bytes + facts.source_bytes);
    assert_eq!(first.execution_identity(), second.execution_identity());
    let wrong = judge(UntrustedEncoding::ByteFallback);
    assert!(Int4SemanticSourceBatchPlanner::new(&planner, compiler.identity.clone(), task_budget(),
        compiler.planning, task(true), wrong, options(), run_limits()).is_err());
}

#[test]
fn evidence_capacity_refusal_is_recoverable_and_original_cancellation_is_fatal() {
    let planner = source(UntrustedEncoding::ByteFallback); let mut options = options();
    options.judge_policy.evidence_window_bytes = 4;
    options.judge_policy.max_evidence_windows = 1; options.judge_policy.max_evidence_spans = 1;
    let compiler = configured(&planner, task(false), SourcePlanningLimits::default(), options);
    let error = failure(compiler.prepare(document(), &mut Control::default()));
    assert_eq!(error.fault.code, BatchCode::WorkLimit); assert!(!error.stop);
    let input = || BatchDocument { text: "A".to_owned(), ..document() };
    compiler.prepare(input(), &mut Control::default()).unwrap();
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let error = failure(compiler.prepare(input(), &mut Control { calls: 0, stop: Some((1, cause)) }));
        assert_eq!(error.fault.cancellation, Some(cause)); assert!(error.stop);
    }
}
