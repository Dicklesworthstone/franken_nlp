//! Source-only planning and private synthetic receipt fixtures, not neural proof.
use super::*;
use crate::tasks::source_planning::grouped_int4::long::tests as maps;

struct Control { calls: usize, stop: usize }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        (self.calls >= self.stop).then_some(DecodeCancellationKind::PollQuota)
    }
}
fn continuing() -> Control { Control { calls: 0, stop: usize::MAX } }
fn request() -> SourceSummarySynthesis {
    SourceSummarySynthesis { map_options: SummaryOptions::default(), synthesis_options: SummaryOptions::default(),
        limits: SummarySynthesisLimits { max_evidence_segments: 64, max_evidence_bytes: 4096,
            verification: GroundingBudget::default() } }
}
fn prepare<'s>(p: &SourceTaskPlanner, text: &'s str, request: SourceSummarySynthesis,
    mapping: Int4SourceMapLimits, width: usize, control: &mut Control)
    -> Result<PreparedInt4SummarySynthesis<'s>, Int4SourceMapError> {
    let id = maps::identity(p, &SourceMapTask::Summarize(request.map_options), width);
    p.plan_int4_summary_synthesis_with_control(text, request, maps::budget(), &PlanContext::new(&id, maps::budget()).unwrap(),
        SourcePlanningLimits::default(), mapping, width, control)
}
fn synthetic(empty: bool) -> (Int4SummarySynthesisPreflight, Int4SummarySynthesisRun) {
    let p = maps::planner(); let source = "éééééééé";
    let prepared = prepare(&p, source, request(), maps::limits(), 32, &mut continuing()).unwrap();
    let expected = prepared.expected;
    let ids = maps::admitted(&prepared.map); let mut driver = maps::script(&prepared.map);
    if empty { for output in &mut driver.outputs {
        let SourceTaskResult::Summarize(raw) = output else { panic!() }; raw.bullets.clear();
    } }
    let discovery = prepared.map.execute_with_driver(&ids, driver).unwrap();
    let mut remaining = remaining_verification(expected.limits.verification, discovery.verification_used).unwrap();
    let evidence = evidence::collect(source, discovery.mapped.root().value(), request().map_options,
        expected.limits, &mut remaining, &mut continuing()).unwrap();
    let (native, prompt, planned, bullets) = if empty { (None, 0, Int4Work::default(), Vec::new()) } else {
        let context = PlanContext::new(&prepared.identity, prepared.budget).unwrap();
        let req = SourceTaskRequest::Summarize { document: evidence.text.clone(), options: request().synthesis_options,
            budget: prepared.budget };
        let plan = p.plan_int4_with_control(&req, &context, prepared.planning, 32, &mut continuing()).unwrap();
        let SourceTaskResult::Summarize(first) = &discovery.mapped.root().value().chunks().next().unwrap().native.result else { panic!() };
        let mut raw = first.clone();
        let quote = "é"; let spans = scan_occurrences(&evidence.text, quote, &mut GroundingBudget::default()).unwrap();
        raw.bullets = vec![CitedBullet { text: "Synthetic final summary".to_owned(), citations: vec![SourceCitation {
            quote: quote.to_owned(), occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans }] }];
        let work = constrained_int4::planned_work(32, plan.prompt_tokens(), raw.generated_token_ids.len()).unwrap();
        raw.forward_positions = work.forward_positions; raw.projected_logits = work.projected_logits;
        let bullets = evidence::lift_summary(source, &raw, &evidence, request().synthesis_options,
            &mut remaining, &mut continuing()).unwrap();
        let native = Int4SourceTaskRun { schema_version: 1, execution: INT4_SOURCE_EXECUTION.to_owned(), group_width: 32,
            result: SourceTaskResult::Summarize(raw), model_work: work };
        (Some(native), plan.prompt_tokens(), plan.planned_work(), bullets)
    };
    let run = finish(discovery, native, prompt, planned, bullets, evidence.segment_count(), evidence.text.len(), remaining,
        expected, 16 << 20, &mut continuing()).unwrap();
    (expected, run)
}
fn decrement(work: &mut Int4Work, axis: usize) {
    match axis { 0 => work.forward_positions -= 1, 1 => work.projected_logits -= 1,
        2 => work.attention_pairs -= 1, 3 => work.projections.dot_products -= 1,
        _ => work.projections.multiply_accumulates -= 1 }
}

#[test]
fn every_real_chunk_and_final_pass_are_reserved_at_each_group_width() {
    let p = maps::planner(); let text = "éééééééé"; let mut previous = None;
    for width in [16, 32, 127] {
        let prepared = prepare(&p, text, request(), maps::limits(), width, &mut continuing()).unwrap();
        let expected = prepared.preflight_metadata();
        assert_eq!(expected.chunk_count(), 2); assert_eq!(expected.group_width(), width);
        assert_eq!(expected.source_span().byte_end, text.len()); assert_eq!(expected.source_span().scalar_end, 8);
        assert_eq!(expected.reserved_model_work().unwrap(), add_work(prepared.map.planned_work(), expected.synthesis_reserve).unwrap());
        assert_eq!(expected.reserved_mask_visits().unwrap(), 3000);
        assert_eq!(prepared.execution_identities().len(), 2);
        for id in prepared.execution_identities() { assert_eq!(id.task_spec, SUMMARIZE_TASK_VERSION); }
        if let Some(old) = previous { assert!(old > expected.synthesis_reserve.projections.dot_products); }
        previous = Some(expected.synthesis_reserve.projections.dot_products);
    }
}
#[test]
fn each_whole_request_work_axis_and_mask_ceiling_refuses_one_unit_short() {
    let p = maps::planner(); let text = "éééééééé";
    let expected = prepare(&p, text, request(), maps::limits(), 32, &mut continuing()).unwrap().expected;
    let mut exact = maps::limits(); exact.max_model_work = expected.reserved_model_work().unwrap();
    exact.max_mask_visits = expected.reserved_mask_visits().unwrap();
    prepare(&p, text, request(), exact, 32, &mut continuing()).unwrap();
    for axis in 0..6 {
        let mut cap = exact;
        if axis < 5 { decrement(&mut cap.max_model_work, axis); } else { cap.max_mask_visits -= 1; }
        assert!(prepare(&p, text, request(), cap, 32, &mut continuing()).is_err(), "{axis}");
    }
}
#[test]
fn final_options_are_separate_from_map_prompts_without_renewing_reserve() {
    let p = maps::planner(); let a = prepare(&p, "source", request(), maps::limits(), 32, &mut continuing()).unwrap();
    let mut r = request(); r.synthesis_options.max_bullets = 1;
    let b = prepare(&p, "source", r, maps::limits(), 32, &mut continuing()).unwrap();
    assert!(a.execution_identities().eq(b.execution_identities())); assert_ne!(a.expected.options, b.expected.options);
    assert_eq!(a.expected.reserved_model_work().unwrap(), b.expected.reserved_model_work().unwrap());
    r.synthesis_options.max_citations_per_bullet = 0;
    assert!(prepare(&p, "source", r, maps::limits(), 32, &mut continuing()).is_err());
}
#[test]
fn grounding_meet_tightens_maps_and_remaining_allowance_deducts_all_axes() {
    let p = maps::planner(); let mut r = request(); let mut mapping = maps::limits();
    r.limits.verification = GroundingBudget { max_fields: 100, max_matches: 80, max_scan_steps: 900 };
    mapping.verification = GroundingBudget { max_fields: 50, max_matches: 90, max_scan_steps: 700 };
    let prepared = prepare(&p, "source", r, mapping, 32, &mut continuing()).unwrap();
    assert_eq!(prepared.map.limits.verification, GroundingBudget { max_fields: 50, max_matches: 80, max_scan_steps: 700 });
    let used = SourceMapVerification { fields: 20, matches: 30, scan_steps: 40 };
    assert_eq!(remaining_verification(r.limits.verification, used).unwrap(),
        GroundingBudget { max_fields: 80, max_matches: 50, max_scan_steps: 860 });
    for axis in 0..3 {
        let mut total = r.limits.verification;
        match axis { 0 => total.max_fields = 19, 1 => total.max_matches = 29, _ => total.max_scan_steps = 39 }
        assert!(remaining_verification(total, used).is_err());
    }
}
#[test]
fn invalid_limits_empty_source_and_foreign_profile_fail_before_execution() {
    let p = maps::planner();
    assert!(prepare(&p, "", request(), maps::limits(), 32, &mut continuing()).is_err());
    for axis in 0..5 {
        let mut r = request();
        match axis { 0 => r.limits.max_evidence_segments = 0, 1 => r.limits.max_evidence_bytes = 0,
            2 => r.limits.verification.max_fields = 0, 3 => r.limits.verification.max_matches = 0,
            _ => r.limits.verification.max_scan_steps = 0 }
        assert!(prepare(&p, "source", r, maps::limits(), 32, &mut continuing()).is_err());
    }
    let mut id = maps::identity(&p, &SourceMapTask::Summarize(SummaryOptions::default()), 32);
    id.numerics_profile = NumericsProfile::StrictQuantized { version: 1 };
    assert!(p.plan_int4_summary_synthesis_with_control("source", request(), maps::budget(),
        &PlanContext::new(&id, maps::budget()).unwrap(), SourcePlanningLimits::default(), maps::limits(), 32, &mut continuing()).is_err());
}
#[test]
fn preparation_preserves_first_middle_and_last_typed_cancellation() {
    let p = maps::planner(); let mut control = continuing();
    prepare(&p, "éééé", request(), maps::limits(), 32, &mut control).unwrap();
    let last = control.calls; assert!(last > 2);
    for stop in [1, last / 2, last] {
        let mut control = Control { calls: 0, stop };
        let error = prepare(&p, "éééé", request(), maps::limits(), 32, &mut control).err().unwrap();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::PollQuota)); assert_eq!(control.calls, stop);
    }
}
#[test]
fn nonrenewable_reserve_subtraction_never_saturates_underflow() {
    let mut cap = maps::limits(); cap.max_model_work = Int4Work::default();
    assert!(reserve(maps::budget(), SourcePlanningLimits::default(), cap, 32).is_err());
    let mut cap = maps::limits(); cap.max_mask_visits = cap.mask_visits_per_chunk - 1;
    assert!(reserve(maps::budget(), SourcePlanningLimits::default(), cap, 32).is_err());
}
#[test]
fn complete_synthetic_receipt_preserves_maps_original_coordinates_and_actual_synthesis_work() {
    let (expected, run) = synthetic(false); expected.verify_completed(&run).unwrap();
    assert_eq!(run.status, SummarySynthesisStatus::Synthesized);
    assert_eq!(run.discovery.mapped.root().value().len(), 2); assert_eq!(run.evidence_segments, 2);
    assert_eq!(run.bullets[0].citations[0].spans.len(), 8);
    assert_eq!(run.bullets[0].citations[0].spans.last().unwrap().scalar_start, 7);
    assert!(run.model_work.forward_positions < run.reserved_model_work.forward_positions);
    assert!(run.verification_used.fields > run.discovery.verification_used.fields);
    assert!(run.verification_used.matches > run.discovery.verification_used.matches);
}
#[test]
fn no_evidence_means_complete_empty_maps_and_zero_final_native_work() {
    let (expected, mut run) = synthetic(true); expected.verify_completed(&run).unwrap();
    assert_eq!(run.status, SummarySynthesisStatus::NoEvidenceCollected); assert!(run.synthesis_native.is_none());
    assert_eq!(run.synthesis_model_work, Int4Work::default()); assert!(run.discovery.model_work.forward_positions > 0);
    run.synthesis_prompt_tokens = 1; assert!(expected.verify_completed(&run).is_err());
}
#[test]
fn every_final_work_axis_and_group_is_independently_verified() {
    for axis in 0..7 {
        let (expected, mut run) = synthetic(false);
        if axis < 5 { decrement(&mut run.synthesis_model_work, axis); }
        else if axis == 5 { run.group_width = 16; } else { run.synthesis_native.as_mut().unwrap().group_width = 16; }
        assert!(expected.verify_completed(&run).is_err(), "{axis}");
    }
}
#[test]
fn tampered_prompt_tokens_mask_grounding_profile_or_eos_cannot_pass_completion() {
    for axis in 0..7 {
        let (expected, mut run) = synthetic(false);
        match axis { 0 => run.synthesis_prompt_tokens += 1, 1 => run.mask_node_visit_charge += 1,
            2 => run.verification_used.fields = 0, 3 => run.discovery.group_width = 16,
            4 => run.evidence_segments = 0, _ => {
                let SourceTaskResult::Summarize(raw) = &mut run.synthesis_native.as_mut().unwrap().result else { panic!() };
                if axis == 5 { raw.numerics_profile = "strict-quantized-v1".to_owned(); }
                else { raw.generated_token_ids.pop(); }
            } }
        assert!(expected.verify_completed(&run).is_err(), "{axis}");
    }
}
#[test]
fn synthesized_text_and_original_span_envelope_are_checked() {
    for axis in 0..3 {
        let (expected, mut run) = synthetic(false);
        match axis { 0 => run.bullets[0].text.push('x'),
            1 => run.bullets[0].citations[0].quote = "elsewhere".to_owned(),
            _ => run.bullets[0].citations[0].spans[0].byte_end = expected.source_span().byte_end + 1 }
        assert!(expected.verify_completed(&run).is_err());
    }
}
#[test]
fn final_result_byte_limit_and_last_checkpoint_cannot_publish_partial_success() {
    let (expected, run) = synthetic(false);
    let remaining = remaining_verification(expected.limits.verification, run.verification_used).unwrap();
    let error = finish(run.discovery, run.synthesis_native, run.synthesis_prompt_tokens, run.synthesis_planned_work,
        run.bullets, run.evidence_segments, run.evidence_bytes, remaining, expected, 1, &mut continuing()).err().unwrap();
    assert!(matches!(error, Int4SourceMapError::Source(_)));
    let (expected, run) = synthetic(false);
    let remaining = remaining_verification(expected.limits.verification, run.verification_used).unwrap();
    let mut control = Control { calls: 0, stop: 2 };
    let error = finish(run.discovery, run.synthesis_native, run.synthesis_prompt_tokens, run.synthesis_planned_work,
        run.bullets, run.evidence_segments, run.evidence_bytes, remaining, expected, 16 << 20, &mut control).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::PollQuota));
}
