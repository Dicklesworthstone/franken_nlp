//! Model-free compact-map planning, exact source spans and lifetime budgets.
use super::*;
use super::tests::{planner, budget, identity, limits, prepare_task, Continue};
use crate::tokenizer::untrusted::{SafeBpeLimits, SafeBpeWorkLimits, UntrustedEncoding};

const SOURCE: &str = "The quick brown fox jumps over the lazy dog. 上海 <think> <|im_start|> ▁";
fn policy() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 4096,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 })
}
fn work() -> SafeBpeWorkLimits {
    SafeBpeWorkLimits { max_input_bytes: 65_536, max_output_tokens: 65_536,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 }
}
fn mapping() -> Int4SourceMapLimits {
    let mut limits = limits();
    limits.chunks.max_chunk_bytes = 4096;
    limits.chunks.max_chunk_tokens = 4096;
    limits
}

#[test]
fn all_plain_source_maps_compact_real_prompts_without_changing_original_coordinates() {
    for task in [SourceMapTask::Ner(NerOptions::default()), SourceMapTask::Keyphrases(KeyphraseOptions::default()),
        SourceMapTask::Summarize(SummaryOptions::default())] {
        let byte = planner();
        let original = prepare_task(&byte, SOURCE, &task, mapping(), 32);
        let compact = planner().with_map_document_encoding(policy(), work()).unwrap();
        let planned = prepare_task(&compact, SOURCE, &task, mapping(), 32);
        assert_eq!(planned.chunk_count(), original.chunk_count());
        assert_eq!(planned.preflight_metadata().source_span(), original.preflight_metadata().source_span());
        assert!(planned.planned_work().forward_positions < original.planned_work().forward_positions);
        assert_eq!(planned.chunks.chunks().iter().map(SourceChunk::text).collect::<String>(), SOURCE);
        assert_ne!(planned.execution_identities().next().unwrap().template_digest,
            original.execution_identities().next().unwrap().template_digest);
        let used = compact.map_document_encoding_work().unwrap().unwrap();
        assert_eq!(used.input_bytes(), 2 * SOURCE.len(), "one probe plus one real chunk plan");
        assert!(used.output_tokens() < used.input_bytes());
        assert!(used.pair_checks() > 0 && used.pair_bytes() > 0);
        assert!(compact.map_document_encoding_workspace_bytes().unwrap() > 0);
    }
}

#[test]
fn probes_actual_plans_and_replanning_spend_one_nonrenewable_allowance() {
    let task = SourceMapTask::Ner(NerOptions::default());
    let mut exact = work(); exact.max_input_bytes = 2 * SOURCE.len() as u64;
    let compact = planner().with_map_document_encoding(policy(), exact).unwrap();
    let first = prepare_task(&compact, SOURCE, &task, mapping(), 32);
    assert_eq!(first.chunk_count(), 1);
    let used = compact.map_document_encoding_work().unwrap().unwrap();
    let id = identity(&compact, &task, 32);
    let again = compact.plan_int4_map_with_control(SOURCE, &task, budget(), &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), mapping(), 32, &mut Continue);
    assert!(again.is_err());
    assert_eq!(compact.map_document_encoding_work().unwrap().unwrap(), used);

    let mut short = exact; short.max_input_bytes -= 1;
    let short = planner().with_map_document_encoding(policy(), short).unwrap();
    let id = identity(&short, &task, 32);
    assert!(short.plan_int4_map_with_control(SOURCE, &task, budget(), &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), mapping(), 32, &mut Continue).is_err());
    assert_eq!(short.map_document_encoding_work().unwrap().unwrap().input_bytes(), SOURCE.len());
}

#[test]
fn one_request_bpe_does_not_authorize_composed_maps_and_every_total_limit_binds_identity() {
    let task = SourceMapTask::Ner(NerOptions::default());
    let unbudgeted = planner().with_document_encoding(policy()).unwrap();
    let id = identity(&unbudgeted, &task, 32);
    assert!(unbudgeted.plan_int4_map_with_control(SOURCE, &task, budget(), &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), mapping(), 32, &mut Continue).is_err());
    let original = planner().with_map_document_encoding(policy(), work()).unwrap();
    let original = prepare_task(&original, SOURCE, &task, mapping(), 32);
    for axis in 0..4 {
        let mut changed = work();
        match axis { 0 => changed.max_input_bytes -= 1, 1 => changed.max_output_tokens -= 1,
            2 => changed.max_pair_checks -= 1, _ => changed.max_pair_bytes -= 1 }
        let next = planner().with_map_document_encoding(policy(), changed).unwrap();
        let next = prepare_task(&next, SOURCE, &task, mapping(), 32);
        assert_eq!(next.planned_work(), original.planned_work());
        assert_ne!(next.execution_identities().next().unwrap().template_digest,
            original.execution_identities().next().unwrap().template_digest);
    }
}

#[test]
fn cumulative_map_encoding_preserves_typed_terminal_cancellation() {
    struct Stop(DecodeCancellationKind);
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
    }
    let task = SourceMapTask::Ner(NerOptions::default());
    let compact = planner().with_map_document_encoding(policy(), work()).unwrap();
    let id = identity(&compact, &task, 32);
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let error = compact.plan_int4_map_with_control(SOURCE, &task, budget(), &PlanContext::new(&id, budget()).unwrap(),
            SourcePlanningLimits::default(), mapping(), 32, &mut Stop(cause)).err().unwrap();
        assert_eq!(error.cancellation(), Some(cause));
    }
}
