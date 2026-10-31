//! Source-only phase fixtures. Native window scheduling is covered by the
//! shared discovery executor; these checks cover the sealed continuation.
use super::*;
use crate::tasks::source_planning::grouped_int4::long::tests as maps;

fn prepare<'s>(planner: &SourceTaskPlanner, source: &'s str, width: usize, batch: usize)
    -> PreparedInt4SummarySynthesis<'s> {
    let request = SourceSummarySynthesis { map_options: SummaryOptions::default(), synthesis_options: SummaryOptions::default(),
        limits: SummarySynthesisLimits { max_evidence_segments: 64, max_evidence_bytes: 4096, verification: GroundingBudget::default() } };
    let identity = maps::identity(planner, &SourceMapTask::Summarize(request.map_options), width);
    let budget = maps::budget(); let mut mapping = maps::limits(); mapping.reduction.map_batch_chunks = batch;
    planner.plan_int4_summary_synthesis_with_control(source, request, budget, &PlanContext::new(&identity, budget).unwrap(),
        SourcePlanningLimits::default(), mapping, width, &mut maps::Continue).unwrap()
}

#[test]
fn phase_admission_covers_full_context_without_changing_discovery_or_reserved_work() {
    let planner = maps::planner();
    for width in [16, 32, 127] { for batch in [1, 2, 4] {
        let prepared = prepare(&planner, "éééééééé", width, batch);
        let expected = prepared.preflight_metadata(); let context = prepared.planning.max_context_tokens;
        let map = prepared.map.cohort_requirements(context).unwrap();
        let scalar = Int4MemoryRequirement::for_context(context, width).unwrap();
        let peak = prepared.cohort_requirements(context).unwrap();
        assert_eq!(peak.native.kv_bytes, map.native.kv_bytes.max(scalar.kv_bytes));
        assert_eq!(peak.native.rope_bytes, map.native.rope_bytes.max(scalar.rope_bytes));
        assert_eq!(peak.native.scratch_payload_bound, map.native.scratch_payload_bound.max(scalar.scratch_payload_bound));
        assert_eq!(peak.workspace_bytes, map.workspace_bytes); assert_eq!(peak.max_result_bytes, map.max_result_bytes);
        assert_eq!(prepared.preflight_metadata(), expected);
        assert_eq!(expected.reserved_model_work().unwrap(), add_work(prepared.map.planned_work(), expected.synthesis_reserve).unwrap());
        assert_eq!(expected.reserved_mask_visits().unwrap(), 3000);
    } }
}

#[test]
fn short_context_or_original_task_kv_cannot_be_hidden_by_small_discovery_rows() {
    let planner = maps::planner(); let mut prepared = prepare(&planner, "source", 32, 1);
    let context = prepared.planning.max_context_tokens;
    assert!(prepared.map.cohort_requirements(context - 1).is_ok());
    assert!(matches!(prepared.cohort_requirements(context - 1), Err(Int4SourceMapError::Admission)));
    prepared.budget.max_kv_bytes = Int4MemoryRequirement::for_context(context, 32).unwrap().kv_bytes - 1;
    assert!(prepared.map.cohort_requirements(context).is_ok());
    assert!(matches!(prepared.cohort_requirements(context), Err(Int4SourceMapError::Admission)));
}

#[test]
fn shared_tail_compiles_only_quotes_and_lifts_native_citations_across_source_windows() {
    let planner = maps::planner(); let source = "éééééééé";
    for batch in [1, 2, 4] {
        let prepared = prepare(&planner, source, 32, batch); let expected = prepared.expected;
        let admitted = maps::admitted(&prepared.map); let driver = maps::script(&prepared.map);
        let (map, tail) = prepared.into_parts();
        let discovery = map.execute_with_driver(&admitted, driver).unwrap();
        let SourceTaskResult::Summarize(raw) = &discovery.mapped.root().value().chunks().next().unwrap().native.result else { panic!() };
        let mut raw = raw.clone();
        let evidence_text = "é\n\né";
        let request = SourceTaskRequest::Summarize { document: evidence_text.to_owned(), options: tail.request.synthesis_options,
            budget: tail.budget };
        let plan = planner.plan_int4_with_control(&request, &PlanContext::new(&tail.identity, tail.budget).unwrap(),
            tail.planning, 32, &mut maps::Continue).unwrap();
        let run = tail.execute(discovery, &planner, |actual, _| {
            assert_eq!(actual.execution_identity(), plan.execution_identity());
            let spans = scan_occurrences(evidence_text, "é", &mut GroundingBudget::default()).unwrap();
            raw.bullets = vec![CitedBullet { text: "final assertion".to_owned(), citations: vec![SourceCitation {
                quote: "é".to_owned(), occurrence: SourceOccurrence::Ambiguous, spans }] }];
            let work = constrained_int4::planned_work(32, actual.prompt_tokens(), raw.generated_token_ids.len()).unwrap();
            raw.forward_positions = work.forward_positions; raw.projected_logits = work.projected_logits;
            Ok(Int4SourceTaskRun { schema_version: 1, execution: INT4_SOURCE_EXECUTION.to_owned(), group_width: 32,
                result: SourceTaskResult::Summarize(raw), model_work: work })
        }, &mut maps::Continue).unwrap();
        expected.verify_completed(&run).unwrap();
        assert_eq!(run.evidence_segments, 2); assert_eq!(run.synthesis_prompt_tokens, plan.prompt_tokens());
        assert_eq!(run.bullets[0].citations[0].spans.len(), 8);
        assert_eq!(run.bullets[0].citations[0].spans[7].scalar_start, 7);
        assert!(run.synthesis_native.is_some());
        assert_eq!(run.model_work, add_work(run.discovery.model_work, run.synthesis_model_work).unwrap());
        assert!(run.verification_used.scan_steps > run.discovery.verification_used.scan_steps);
    }
}

#[test]
fn empty_evidence_never_enters_scalar_phase_and_late_failure_publishes_no_result() {
    let planner = maps::planner();
    for empty in [true, false] {
        let prepared = prepare(&planner, "éééééééé", 32, 2);
        let admitted = maps::admitted(&prepared.map); let mut driver = maps::script(&prepared.map);
        if empty { for output in &mut driver.outputs {
            let SourceTaskResult::Summarize(raw) = output else { panic!() }; raw.bullets.clear();
        } }
        let (map, tail) = prepared.into_parts(); let discovery = map.execute_with_driver(&admitted, driver).unwrap();
        let mut calls = 0;
        let result = tail.execute(discovery, &planner, |_, _| {
            calls += 1;
            Err(Int4SourceError::Cancelled(DecodeCancellationKind::Deadline).into())
        }, &mut maps::Continue);
        if empty {
            let run = result.unwrap(); assert_eq!(calls, 0);
            assert_eq!(run.status, SummarySynthesisStatus::NoEvidenceCollected);
            assert!(run.synthesis_native.is_none()); assert_eq!(run.model_work, run.discovery.model_work);
        } else {
            assert_eq!(calls, 1);
            assert_eq!(result.err().unwrap().cancellation(), Some(DecodeCancellationKind::Deadline));
        }
    }
}
