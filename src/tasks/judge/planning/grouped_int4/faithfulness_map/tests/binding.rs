//! Completion must belong to the exact admitted source, claim and model,
//! even if every public shape and work counter happens to agree.
use super::*;

fn source() -> JudgeRequest {
    let mut request = request();
    if let JudgeRequest::Faithfulness { source, claim, .. } = &mut request {
        *source = "Alice Bob. Carol Dan. End.".to_owned();
        *claim = "Alice left.".to_owned();
    }
    request
}
fn compile<'a>(request: &'a JudgeRequest, limits: Int4FaithfulnessMapLimits,
    authority: (TaskBudget, JudgeLimits), identity: &ExecutionIdentity)
    -> PreparedInt4FaithfulnessMap<'a> {
    let planner = fixture().0;
    let JudgeRequest::Faithfulness { claim, policy, budget, .. } = request else { unreachable!() };
    let capacity = planner.faithfulness.source_capacity(claim, *policy, *budget,
        authority.0, authority.1, limits.chunks.context_tokens).unwrap();
    PreparedInt4FaithfulnessMap::compile_bound(request, limits, capacity, 32, authority,
        &mut Continue, |row, control| planner.compile_int4(row,
            &PlanContext::new(identity, authority.0).unwrap(), authority.1, 32, control)).unwrap()
}
fn same_geometry(left: Int4FaithfulnessMapPreflight, right: Int4FaithfulnessMapPreflight) {
    assert_eq!(left.source, right.source);
    assert_eq!(left.chunks, right.chunks);
    assert_eq!(left.heads, right.heads);
    assert_eq!(left.max_heads, right.max_heads);
    assert_eq!(left.work, right.work);
}
fn cohorts_at(plan: &PreparedInt4FaithfulnessMap<'_>, width: usize) -> Int4FaithfulnessMapCohortRun {
    let admitted = plan.identities().unwrap();
    let windows = plan.cohort_windows(&admitted, width, cohort_budget(plan), &mut Continue).unwrap();
    plan.collect_cohorts_with_control(&windows, &mut Continue,
        |window, _| finish_window(plan, &admitted, window)).unwrap()
}
fn cohorts(plan: &PreparedInt4FaithfulnessMap<'_>) -> Int4FaithfulnessMapCohortRun { cohorts_at(plan, 2) }

#[test]
fn equal_length_sources_and_claims_do_not_share_completion_authority() {
    let original = source();
    let plan = prepare(&original, mapping()).unwrap();
    let expected = plan.preflight_metadata();
    for change_source in [true, false] {
        let mut other = original.clone();
        if let JudgeRequest::Faithfulness { source, claim, .. } = &mut other {
            if change_source { *source = "Carol Dan. Alice Bob. Fin.".to_owned(); }
            else { *claim = "Carol left.".to_owned(); }
        }
        let plan = prepare(&other, mapping()).unwrap();
        same_geometry(expected, plan.preflight_metadata());
        let run = finish(&plan).unwrap(); let grouped = cohorts(&plan);
        plan.preflight_metadata().verify_completed(&run).unwrap();
        plan.preflight_metadata().verify_completed_cohorts(&grouped, 2).unwrap();
        assert!(expected.verify_completed(&run).is_err());
        assert!(expected.verify_completed_cohorts(&grouped, 2).is_err());
    }
}

#[test]
fn unused_allowances_planner_limits_policy_and_model_are_exact_request_authority() {
    let original = source();
    let authority = (original.budget(), JudgeLimits::default());
    let base = compile(&original, mapping(), authority, fixture().1);
    let expected = base.preflight_metadata(); let completed = finish(&base).unwrap();
    for axis in 0..11 {
        let mut request = original.clone(); let mut limits = mapping();
        let mut authority = authority; let mut identity = fixture().1.clone();
        match axis {
            0 => identity.logical_model_digest = Sha256Digest::of_bytes(b"other same-shaped model"),
            1 => if let JudgeRequest::Faithfulness { policy, .. } = &mut request {
                policy.minimum_candidate_weight_ppm += 1;
            },
            2 => if let JudgeRequest::Faithfulness { budget, .. } = &mut request { budget.max_output_bytes -= 1; },
            3 => limits.max_model_work.forward_positions -= 1,
            4 => limits.max_retained_result_bytes -= 1,
            5 => limits.max_result_bytes -= 1,
            6 => limits.chunks.max_tokenizer_calls += 1,
            7 => limits.aggregation.minimum_classified_coverage_ppm -= 1,
            8 => authority.0.max_kv_bytes += 1,
            9 => authority.1.max_total_candidate_tokens -= 1,
            _ => authority.1.per_head.max_total_tokens -= 1,
        }
        let changed = compile(&request, limits, authority, &identity);
        let changed_expected = changed.preflight_metadata();
        same_geometry(expected, changed_expected);
        assert_ne!(expected.request_binding, changed_expected.request_binding, "axis {axis}");
        assert!(changed_expected.verify_completed(&completed).is_err(), "axis {axis}");
    }
}

#[test]
fn valid_completion_cannot_authorize_edited_relation_evidence_or_scores() {
    let request = source(); let plan = prepare(&request, mapping()).unwrap();
    let expected = plan.preflight_metadata();
    let baseline = plan.collect_with_control(&mut Continue, |_, row, _| Ok(run(row, Some("entailed")))).unwrap();
    expected.verify_completed(&baseline).unwrap();
    assert_eq!(baseline.aggregate.relation, Some(FaithfulnessRelation::Entailed));
    assert!(!baseline.aggregate.evidence.is_empty());
    for axis in 0..5 {
        let mut changed = baseline.clone();
        match axis {
            0 => changed.aggregate.relation = Some(FaithfulnessRelation::Contradicted),
            1 => changed.aggregate.evidence[0].quote = "forged source".to_owned(),
            2 => if let JudgeResult::Faithfulness(result) = &mut changed.chunks[0].run.result {
                result.heads[0].candidates[0].sequence_score -= 0.25;
            },
            3 => if let JudgeResult::Faithfulness(result) = &mut changed.chunks[0].run.result {
                result.global.passes_policy = !result.global.passes_policy;
            },
            _ => changed.chunks[0].run.rewound_positions += 1,
        }
        // A caller can repair public byte accounting, but cannot mint a new
        // native completion merely by keeping geometry and totals plausible.
        changed.retained_result_bytes = changed.chunks.iter().map(|chunk|
            value_bytes(chunk, mapping().max_retained_result_bytes).unwrap()).sum();
        assert!(expected.verify_completed(&changed).is_err(), "axis {axis}");
    }
}

#[test]
fn cohort_seals_bind_physical_counts_and_private_output_owners() {
    let request = source(); let plan = prepare(&request, mapping()).unwrap();
    let expected = plan.preflight_metadata(); let baseline = cohorts(&plan);
    expected.verify_completed_cohorts(&baseline, 2).unwrap();
    for axis in 0..4 {
        let mut changed = baseline.clone();
        match axis { 0 => changed.group_steps += 1, 1 => changed.projection_groups += 1,
            2 => changed.chunk_cohorts += 1, _ => changed.execution.push('x') }
        assert!(expected.verify_completed_cohorts(&changed, 2).is_err(), "axis {axis}");
    }
    let mut other = request.clone();
    if let JudgeRequest::Faithfulness { claim, .. } = &mut other { *claim = "Carol left.".to_owned(); }
    let other = prepare(&other, mapping()).unwrap(); let other_run = finish(&other).unwrap();
    // Constant synthetic logits make the wire payloads identical; their
    // distinct private owners must still prevent transplanting a cohort.
    assert_eq!(canonjson::canonical_bytes(&baseline.output).unwrap(), canonjson::canonical_bytes(&other_run).unwrap());
    let mut transplanted = baseline;
    transplanted.output = other_run;
    other.preflight_metadata().verify_completed(&transplanted.output).unwrap();
    assert!(other.preflight_metadata().verify_completed_cohorts(&transplanted, 2).is_err());
}

#[test]
fn equal_cohort_counts_do_not_authorize_different_executed_windows() {
    let mut request = source();
    if let JudgeRequest::Faithfulness { source, .. } = &mut request { *source = "abcde".repeat(10); }
    let mut limits = mapping(); limits.chunks.max_chunk_bytes = 10;
    let prepared = prepare(&request, limits).unwrap();
    assert_eq!(prepared.chunk_count(), 5);
    let expected = prepared.preflight_metadata();
    let three = cohorts_at(&prepared, 3); let four = cohorts_at(&prepared, 4);
    assert_eq!(three.chunk_cohorts, four.chunk_cohorts);
    assert_eq!(three.output, four.output);
    expected.verify_completed_cohorts(&three, 3).unwrap();
    expected.verify_completed_cohorts(&four, 4).unwrap();
    assert!(expected.verify_completed_cohorts(&three, 4).is_err());
    assert!(expected.verify_completed_cohorts(&four, 3).is_err());
    let only = cohorts_at(&prepared, 5);
    expected.verify_completed_cohorts(&only, 6).unwrap();
}

#[test]
fn completion_verification_polls_large_payloads_and_preserves_late_cancellation() {
    struct VerificationControl { checks: usize, stop_at: usize, cause: DecodeCancellationKind }
    impl DecodeStepControl for VerificationControl {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.checks += 1;
            (self.checks == self.stop_at).then_some(self.cause)
        }
    }
    let request = source(); let plan = prepare(&request, mapping()).unwrap();
    let expected = plan.preflight_metadata();
    let serial = finish(&plan).unwrap(); let grouped = cohorts(&plan);
    use DecodeCancellationKind::*;
    let causes = [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit];
    for verify_cohorts in [false, true] {
        let verify = |control: &mut VerificationControl| {
            if verify_cohorts { expected.verify_completed_cohorts_with_control(&grouped, 2, control) }
            else { expected.verify_completed_with_control(&serial, control) }
        };
        let mut complete = VerificationControl { checks: 0, stop_at: usize::MAX, cause: User };
        verify(&mut complete).unwrap();
        assert!(complete.checks > 3);
        for cause in causes {
            // Stop at the final verification checkpoint, after the genuine
            // sealed result has passed hashing and its complete typed checks.
            let mut late = VerificationControl { checks: 0, stop_at: complete.checks, cause };
            assert_eq!(verify(&mut late).unwrap_err().cancellation(), Some(cause));
            assert_eq!(late.checks, complete.checks);
        }
    }

    // Public result fields can be enlarged by consumers. Their unchanged
    // private seals cannot authorize that edit, and even rejecting the edit
    // must poll within one large serializer write rather than scan it all.
    let mut large_serial = serial;
    large_serial.execution = "x".repeat(3 * 64 * 1024);
    let mut large_grouped = grouped;
    large_grouped.execution = "x".repeat(3 * 64 * 1024);
    for cause in causes {
        let mut control = VerificationControl { checks: 0, stop_at: 3, cause };
        assert_eq!(expected.verify_completed_with_control(&large_serial, &mut control)
            .unwrap_err().cancellation(), Some(cause));
        assert_eq!(control.checks, 3);
        let mut control = VerificationControl { checks: 0, stop_at: 3, cause };
        assert_eq!(expected.verify_completed_cohorts_with_control(&large_grouped, 2, &mut control)
            .unwrap_err().cancellation(), Some(cause));
        assert_eq!(control.checks, 3);
    }
}

#[test]
fn private_associations_never_enter_json_or_debug_output() {
    let request = source(); let plan = prepare(&request, mapping()).unwrap();
    let expected = plan.preflight_metadata(); let run = cohorts(&plan);
    let json = String::from_utf8(canonjson::canonical_bytes(&run).unwrap()).unwrap();
    let debug = format!("{expected:?} {run:?}");
    for field in ["request_binding", "result_binding", "effective_cohort_chunks"] {
        assert!(!json.contains(field)); assert!(!debug.contains(field));
    }
    for seal in [run.output.request_binding, run.output.result_binding] {
        assert!(!debug.contains(&format!("{seal:?}")));
    }
}
