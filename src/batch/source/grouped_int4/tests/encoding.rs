//! Real pinned source planning and stream policy checks; no native inference.
use super::*;
use crate::{
    tasks::answer::{AnswerContext, AnswerInputLimits},
    tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding},
    validation::grounded_fields::{GroundingBudget, scan_occurrences},
};

const SOURCE: &str = "é上海 and 上海. The quick brown fox jumps over the lazy dog. <think> <|im_start|>";
const QUESTION: &str = "Which city is named? <think>";

fn policy(bytes: usize) -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: bytes, max_output_tokens: 4096,
        max_pair_checks: 1 << 22, max_pair_bytes: 1 << 26 })
}
fn compact_planner(bytes: usize) -> SourceTaskPlanner {
    planner().with_document_encoding(policy(bytes)).unwrap()
}
fn compact_compiler(p: &SourceTaskPlanner, kind: BuiltInTask, planning: SourcePlanningLimits)
    -> Int4SourceBatchPlanner<'_> {
    Int4SourceBatchPlanner::new_with_document_encoding(p, identity(p, kind), budget(), planning,
        Some(args(kind)), 32).unwrap()
}
fn item() -> BatchDocument<SourceBatchArgs> {
    BatchDocument { id: "source".to_owned(), text: SOURCE.to_owned(), task_args: None }
}
fn qa_args() -> SourceBatchArgs {
    let mut args = args(BuiltInTask::Answer);
    let SourceBatchArgs::Answer { passages, .. } = &mut args else { unreachable!() };
    *passages = vec![AnswerPassage { id: "first".to_owned(), text: "é上海 and 上海.".to_owned() },
        AnswerPassage { id: "second".to_owned(), text: "上海 <think>".to_owned() }];
    args
}
fn question() -> BatchDocument<SourceBatchArgs> {
    BatchDocument { id: "question".to_owned(), text: QUESTION.to_owned(), task_args: Some(qa_args()) }
}
struct Control { calls: usize, stop_at: usize, cause: DecodeCancellationKind }
impl Control {
    fn continuing() -> Self { Self { calls: 0, stop_at: usize::MAX, cause: DecodeCancellationKind::Deadline } }
}
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        (self.calls == self.stop_at).then_some(self.cause)
    }
}

#[test]
fn all_four_stream_tasks_share_direct_compact_plans_and_fresh_per_record_encoding_limits() {
    let byte = planner(); let compact = compact_planner(8192);
    for kind in [BuiltInTask::Ner, BuiltInTask::Keyphrases, BuiltInTask::Summarize, BuiltInTask::Answer] {
        let byte_plan = compiler(&byte, kind).prepare_with_control(item(), &mut Continue).unwrap();
        let stream = compact_compiler(&compact, kind, SourcePlanningLimits::default());
        let first = stream.prepare_with_control(item(), &mut Continue).unwrap();
        let mut later = item(); later.id = "same source in another epoch".to_owned();
        let second = stream.prepare_with_control(later, &mut Continue).unwrap();
        assert_eq!(first.execution_identity(), second.execution_identity());
        assert_eq!(first.planned_work(), second.planned_work());
        assert_eq!(first.request_digest(), byte_plan.request_digest());
        assert!(first.prompt_tokens() < byte_plan.prompt_tokens(), "{kind:?}");
        assert_eq!(first.planned_work().projected_logits, byte_plan.planned_work().projected_logits);
        let id = identity(&compact, kind); let cx = PlanContext::new(&id, budget()).unwrap();
        let direct = compact.plan_int4_with_control(&args(kind).into_request(SOURCE.to_owned()), &cx,
            SourcePlanningLimits::default(), 32, &mut Continue).unwrap();
        first.verify_identity(direct.execution_identity()).unwrap();
        assert_eq!(first.planned_work(), direct.planned_work());
        stream.check_projection(&first).unwrap();
        assert!(stream.check_projection(&byte_plan).is_err());

        let selected = compact_compiler(&compact, kind, SourcePlanningLimits::default())
            .with_selected_rows(Int4JsonSparseLimits { max_rows_per_step: 7 }).unwrap();
        let selected_plan = selected.prepare_with_control(item(), &mut Continue).unwrap();
        assert_eq!(selected_plan.prompt_tokens(), first.prompt_tokens());
        assert_eq!(selected_plan.request_digest(), first.request_digest());
        assert_eq!(selected_plan.planned_work().projected_logits, 7 * u64::from(budget().max_output_tokens));
        assert!(stream.check_projection(&selected_plan).is_err());
        assert!(selected.check_projection(&first).is_err());
        selected.check_projection(&selected_plan).unwrap();
    }
    // A non-QA record has one untrusted source segment and spends its exact
    // bytes once, independently of earlier successful records.
    let exact = compact_planner(SOURCE.len());
    let stream = compact_compiler(&exact, BuiltInTask::Keyphrases, SourcePlanningLimits::default());
    for _ in 0..2 { stream.prepare_with_control(item(), &mut Continue).unwrap(); }
    let short = compact_planner(SOURCE.len() - 1);
    let failure = compact_compiler(&short, BuiltInTask::Keyphrases, SourcePlanningLimits::default())
        .prepare_with_control(item(), &mut Continue).err().unwrap();
    assert!(!failure.stop); assert_eq!(failure.fault.code, BatchCode::Planning);
}

#[test]
fn qa_question_manifest_and_joined_source_share_one_ledger_and_keep_original_utf8_coordinates() {
    let p = compact_planner(8192);
    let SourceBatchArgs::Answer { passages, .. } = qa_args() else { unreachable!() };
    let context = AnswerContext::encode_with_control(p.source_encoder(), QUESTION, &passages,
        AnswerInputLimits { max_passages: 4, max_input_bytes: 8192, max_input_tokens: 4096 }, &mut || None).unwrap();
    let source = context.document();
    assert_eq!(source.text(), "é上海 and 上海.\n\n上海 <think>");
    assert_eq!(context.passage_count(), 2);
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let segments: Vec<_> = source.context_token_ids().collect();
    assert_eq!(segments.len(), 2);
    assert_eq!(tokenizer.tokenizer().decode_bytes(segments[0]).unwrap(), QUESTION.as_bytes());
    assert_eq!(tokenizer.tokenizer().decode_bytes(source.token_ids()).unwrap(), source.text().as_bytes());
    let manifest: serde_json::Value = serde_json::from_slice(&tokenizer.tokenizer().decode_bytes(segments[1]).unwrap()).unwrap();
    assert_eq!(manifest[1][0]["id"], "first");
    assert_eq!(manifest[1][0]["span"], serde_json::json!({"byte_start":0,"byte_end":20,"scalar_start":0,"scalar_end":11}));
    assert_eq!(manifest[1][1]["id"], "second");
    assert_eq!(manifest[1][1]["span"], serde_json::json!({"byte_start":22,"byte_end":36,"scalar_start":13,"scalar_end":23}));
    let spans = scan_occurrences(source.text(), "上海", &mut GroundingBudget::default()).unwrap();
    assert_eq!(spans.len(), 3);
    assert_eq!((spans[0].byte_start, spans[0].scalar_start), (2, 1));
    assert_eq!((spans[1].byte_start, spans[1].scalar_start), (13, 8));
    assert_eq!((spans[2].byte_start, spans[2].scalar_start), (22, 13));
    let controls = pinned_controls::pinned().unwrap();
    for ids in source.context_token_ids().chain(std::iter::once(source.token_ids())) {
        assert!(ids.iter().all(|&id| !controls.template_controls().contains(id)));
    }

    let total_bytes = source.total_input_bytes();
    assert!(total_bytes > QUESTION.len() + source.text().len());
    let exact = compact_planner(total_bytes);
    let stream = compact_compiler(&exact, BuiltInTask::Answer, SourcePlanningLimits::default());
    let first = stream.prepare_with_control(question(), &mut Continue).unwrap();
    let second = stream.prepare_with_control(question(), &mut Continue).unwrap();
    assert_eq!(first.execution_identity(), second.execution_identity());
    let byte = planner();
    let original = compiler(&byte, BuiltInTask::Answer).prepare_with_control(question(), &mut Continue).unwrap();
    assert_eq!(first.request_digest(), original.request_digest());
    assert!(first.prompt_tokens() < original.prompt_tokens());
    let short = compact_planner(total_bytes - 1);
    assert!(compact_compiler(&short, BuiltInTask::Answer, SourcePlanningLimits::default())
        .prepare_with_control(question(), &mut Continue).is_err());
    let UntrustedEncoding::SafeBpeV1(mut limits) = policy(total_bytes) else { unreachable!() };
    limits.max_output_tokens = source.total_token_count();
    let token_exact = planner().with_document_encoding(UntrustedEncoding::SafeBpeV1(limits)).unwrap();
    assert!(compact_compiler(&token_exact, BuiltInTask::Answer, SourcePlanningLimits::default())
        .prepare_with_control(question(), &mut Continue).is_ok());
    limits.max_output_tokens -= 1;
    let token_short = planner().with_document_encoding(UntrustedEncoding::SafeBpeV1(limits)).unwrap();
    assert!(compact_compiler(&token_short, BuiltInTask::Answer, SourcePlanningLimits::default())
        .prepare_with_control(question(), &mut Continue).is_err());
    let mut renamed = question();
    let SourceBatchArgs::Answer { passages, .. } = renamed.task_args.as_mut().unwrap() else { unreachable!() };
    passages[0].id = "other".to_owned();
    let renamed = stream.prepare_with_control(renamed, &mut Continue).unwrap();
    assert_ne!(renamed.request_digest(), first.request_digest());
    assert_ne!(renamed.execution_identity().prompt_digest, first.execution_identity().prompt_digest);
}

#[test]
fn token_context_and_source_bytes_are_separate_admission_axes() {
    let compact = compact_planner(8192); let byte = planner();
    for kind in [BuiltInTask::Ner, BuiltInTask::Keyphrases, BuiltInTask::Summarize, BuiltInTask::Answer] {
        let plan = compact_compiler(&compact, kind, SourcePlanningLimits::default())
            .prepare_with_control(item(), &mut Continue).unwrap();
        let exact = plan.prompt_tokens() + budget().max_output_tokens as usize;
        let planning = SourcePlanningLimits { max_context_tokens: exact, ..Default::default() };
        let same = compact_compiler(&compact, kind, planning).prepare_with_control(item(), &mut Continue).unwrap();
        assert_eq!(same.execution_identity(), plan.execution_identity());
        assert!(compact_compiler(&compact, kind, SourcePlanningLimits { max_context_tokens: exact - 1, ..planning })
            .prepare_with_control(item(), &mut Continue).is_err());
        let bytes = Int4SourceBatchPlanner::new(&byte, identity(&byte, kind), budget(), planning,
            Some(args(kind)), 32).unwrap();
        assert!(bytes.prepare_with_control(item(), &mut Continue).is_err());
    }
    let short = SourcePlanningLimits { max_input_bytes: SOURCE.len() - 1, ..Default::default() };
    assert!(compact_compiler(&compact, BuiltInTask::Ner, short).prepare_with_control(item(), &mut Continue).is_err());
}

#[test]
fn changing_only_encoding_limits_cannot_substitute_a_prepared_record() {
    let compact = compact_planner(8192);
    let stream = compact_compiler(&compact, BuiltInTask::Keyphrases, SourcePlanningLimits::default());
    let first = stream.prepare_with_control(item(), &mut Continue).unwrap();
    let UntrustedEncoding::SafeBpeV1(mut limits) = policy(8192) else { unreachable!() };
    limits.max_pair_checks -= 1;
    let changed = planner().with_document_encoding(UntrustedEncoding::SafeBpeV1(limits)).unwrap();
    let other = compact_compiler(&changed, BuiltInTask::Keyphrases, SourcePlanningLimits::default())
        .prepare_with_control(item(), &mut Continue).unwrap();
    assert_eq!(first.execution_identity().prompt_digest, other.execution_identity().prompt_digest);
    assert_ne!(first.execution_identity().template_digest, other.execution_identity().template_digest);
    assert!(first.verify_identity(other.execution_identity()).is_err());
    assert!(stream.check_projection(&other).is_err());
    let mut wire = serde_json::to_value(args(BuiltInTask::Keyphrases)).unwrap();
    wire.as_object_mut().unwrap().insert("document_encoding".to_owned(), serde_json::to_value(policy(8192)).unwrap());
    assert!(serde_json::from_value::<SourceBatchArgs>(wire).is_err());
}

#[test]
fn legacy_source_factories_and_capacity_maps_remain_byte_only() {
    let byte = planner(); let compact = compact_planner(8192);
    for kind in [BuiltInTask::Ner, BuiltInTask::Keyphrases, BuiltInTask::Summarize, BuiltInTask::Answer] {
        assert!(Int4SourceBatchPlanner::new(&compact, identity(&compact, kind), budget(),
            SourcePlanningLimits::default(), Some(args(kind)), 32).is_err());
        let ordinary = compiler(&byte, kind).prepare_with_control(item(), &mut Continue).unwrap();
        let explicit = compact_compiler(&byte, kind, SourcePlanningLimits::default())
            .prepare_with_control(item(), &mut Continue).unwrap();
        assert_eq!(ordinary.execution_identity(), explicit.execution_identity());
        assert_eq!(ordinary.planned_work(), explicit.planned_work());
    }
    let task = crate::tasks::source_planning::grouped_int4::long::SourceMapTask::Ner(NerOptions::default());
    for (p, accepted) in [(&byte, true), (&compact, false)] {
        let id = identity(p, BuiltInTask::Ner); let cx = PlanContext::new(&id, budget()).unwrap();
        assert_eq!(p.int4_map_capacity_with_control(&task, budget(), &cx,
            SourcePlanningLimits::default(), 32, &mut Continue).is_ok(), accepted);
    }
    #[cfg(all(feature = "metadata-store", target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        use std::sync::Arc;
        use crate::{jobs::runner::source::grouped_int4::Int4SourceJobPlanner,
            native_engine::strict_int4::{Int4MemoryRequirement, weights::projection_fixture}};
        let (model, _) = projection_fixture(32);
        let planning = SourcePlanningLimits::default();
        let mut ceiling = budget();
        ceiling.max_kv_bytes = Int4MemoryRequirement::for_context(planning.max_context_tokens, 32).unwrap().kv_bytes;
        assert!(Int4SourceJobPlanner::for_model(model.identity(), Arc::new(byte), BuiltInTask::Ner,
            ceiling, planning, None, limits()).is_ok());
        assert!(Int4SourceJobPlanner::for_model(model.identity(), Arc::new(compact), BuiltInTask::Ner,
            ceiling, planning, None, limits()).is_err());
    }
}

#[test]
fn compact_qa_planning_preserves_every_cancellation_cause_and_does_not_return_a_partial_plan() {
    let p = compact_planner(8192);
    let compiler = compact_compiler(&p, BuiltInTask::Answer, SourcePlanningLimits::default());
    let mut observed = Control::continuing();
    compiler.prepare_with_control(question(), &mut observed).unwrap();
    assert!(observed.calls > 8);
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for stop_at in [1, observed.calls / 2, observed.calls] {
            let mut control = Control { calls: 0, stop_at, cause };
            let failure = compiler.prepare_with_control(question(), &mut control).err().unwrap();
            assert!(failure.stop);
            assert_eq!(failure.fault.cancellation, Some(cause));
            assert_eq!(control.calls, stop_at);
        }
    }
}
