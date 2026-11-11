//! Pinned planning and synthetic score finalizers exercise completion
//! association only; these fixtures are not native parity or quality evidence.
use super::*;
use super::super::binding as seals;
use super::super::execution::value_bytes;

fn authority() -> seals::Authority {
    seals::Authority { ceiling: task_budget(), planning: SentimentLimits::default() }
}
fn compile<'a>(request: &'a SentimentRequest, limits: Int4SentimentMapLimits,
    authority: seals::Authority, identity: &ExecutionIdentity) -> PreparedInt4SentimentMap<'a> {
    let planner = fixture().0;
    let capacity = planner.source_capacity(&request.axes, request.budget, authority.ceiling,
        authority.planning, limits.chunks.context_tokens).unwrap();
    PreparedInt4SentimentMap::compile(request, limits, capacity, 32, authority, &mut Continue, |row, control|
        planner.compile_int4(row, &PlanContext::new(identity, authority.ceiling).unwrap(),
            authority.planning, 32, control)).unwrap()
}
fn cohorts(plan: &PreparedInt4SentimentMap<'_>) -> Int4SentimentMapCohortRun {
    plan.complete_cohorts(finish(plan).unwrap(), 2, plan.chunk_count().div_ceil(2), 2, 2, &mut Continue).unwrap()
}
fn same_geometry(left: Int4SentimentMapPreflight, right: Int4SentimentMapPreflight) {
    assert_eq!(left.source_span, right.source_span);
    assert_eq!(left.chunks, right.chunks);
    assert_eq!(left.heads, right.heads);
    assert_eq!(left.work, right.work);
}

#[test]
fn equally_sized_documents_cannot_exchange_completed_sentiment_results() {
    let left = source("Alice Bob", &[SentimentAxis::Valence, SentimentAxis::Arousal]);
    let mut right = left.clone(); right.document = "Carol Dan".to_owned();
    let left = prepare(&left, map_limits()).unwrap(); let right = prepare(&right, map_limits()).unwrap();
    let expected = left.preflight_metadata(); let other = right.preflight_metadata();
    same_geometry(expected, other);
    assert_ne!(expected.request_binding, other.request_binding);
    let serial = finish(&right).unwrap(); let grouped = cohorts(&right);
    other.verify_completed(&serial).unwrap(); other.verify_completed_cohorts(&grouped, 2).unwrap();
    assert!(expected.verify_completed(&serial).is_err());
    assert!(expected.verify_completed_cohorts(&grouped, 2).is_err());
}

#[test]
fn request_order_model_and_unused_declared_allowances_remain_exact_authority() {
    let request = source("Alice Bob", &[SentimentAxis::Valence, SentimentAxis::Arousal]);
    let base = compile(&request, map_limits(), authority(), fixture().1);
    let expected = base.preflight_metadata(); let result = finish(&base).unwrap();
    for axis in 0..17 {
        let mut changed = request.clone(); let mut limits = map_limits();
        let mut authority = authority(); let mut identity = fixture().1.clone();
        match axis {
            0 => changed.axes.reverse(),
            1 => identity.logical_model_digest = Sha256Digest::of_bytes(b"another same-shaped model"),
            2 => changed.budget.max_output_bytes -= 1,
            3 => limits.aggregation.minimum_estimated_coverage_ppm -= 1,
            4 => limits.max_model_work.forward_positions -= 1,
            5 => limits.max_retained_result_bytes -= 1,
            6 => limits.max_result_bytes -= 1,
            7 => limits.chunks.max_tokenizer_calls += 1,
            8 => authority.ceiling.max_kv_bytes += 1,
            9 => authority.planning.max_total_prompt_tokens -= 1,
            10 => authority.planning.max_total_candidate_tokens -= 1,
            11 => authority.planning.max_total_nodes -= 1,
            12 => authority.planning.max_total_projected_logits -= 1,
            13 => authority.planning.max_output_bytes -= 1,
            14 => authority.planning.per_axis.max_total_tokens -= 1,
            15 => authority.planning.per_axis.max_projected_logits -= 1,
            _ => identity.packing_set_digest = Sha256Digest::of_bytes(b"another packing identity"),
        }
        let changed = compile(&changed, limits, authority, &identity);
        let other = changed.preflight_metadata();
        same_geometry(expected, other);
        assert_ne!(expected.request_binding, other.request_binding, "axis {axis}");
        if axis == 0 {
            // The task canonicalizes axis order, but exact caller authority
            // still cannot be silently exchanged at the completion boundary.
            assert_eq!(base.identities().unwrap(), changed.identities().unwrap());
        }
        assert!(other.verify_completed(&result).is_err(), "axis {axis}");
    }
}

#[test]
fn public_score_and_aggregate_changes_cannot_retain_a_valid_completion_seal() {
    let request = source("0123456789abcdefghijklmn", &[SentimentAxis::Valence]);
    let prepared = prepare(&request, map_limits()).unwrap();
    let expected = prepared.preflight_metadata(); let baseline = finish(&prepared).unwrap();
    for field in 0..8 {
        let mut changed = baseline.clone();
        match field {
            0 => changed.aggregate.interpretation.push('x'),
            1 => changed.aggregate.dimensions[0].estimate = Some(0.25),
            2 => changed.aggregate.dimensions[0].mixture_weights[0].weight += 0.01,
            3 => changed.aggregate.dimensions[0].distribution_mean += 0.01,
            4 => changed.chunks[0].run.result.dimensions[0].scores.candidates[0].sequence_score -= 0.25,
            5 => changed.chunks[0].run.result.dimensions[0].scores.candidates[0].candidate_weight += 0.01,
            6 => changed.chunks[0].run.result.dimensions[0].low_anchor.push('x'),
            _ => changed.chunks[0].run.rewound_positions += 1,
        }
        // An internally plausible byte total must not authorize a replacement.
        changed.retained_result_bytes = changed.chunks.iter().map(|chunk|
            value_bytes(chunk, map_limits().max_retained_result_bytes).unwrap()).sum();
        assert_eq!(changed.request_binding, expected.request_binding);
        assert_ne!(changed.result_binding, seals::result(&changed, map_limits().max_result_bytes).unwrap());
        assert!(expected.verify_completed(&changed).is_err(), "field {field}");
    }
}

#[test]
fn optional_nonfinite_estimates_cannot_impersonate_a_valid_abstention_null() {
    let registry = pinned_controls::pinned().unwrap();
    let planner = SentimentPlanner::pinned(registry.template_controls(), SentimentOptions {
        mode: ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::SumLogits },
        eos_token_id: EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap(),
        policy: SentimentPolicy { minimum_peak_weight_ppm: 1_000_000, maximum_normalized_entropy_ppm: 1_000_000 },
    }).unwrap();
    let identity = fixtures::identity("sentiment-v1", planner.tokenizer_digest(), *planner.template_digest());
    let request = source("Alice Bob", &[SentimentAxis::Valence]);
    let limits = map_limits(); let authority = authority();
    let capacity = planner.source_capacity(&request.axes, request.budget, authority.ceiling,
        authority.planning, limits.chunks.context_tokens).unwrap();
    let prepared = PreparedInt4SentimentMap::compile(&request, limits, capacity, 32, authority, &mut Continue,
        |row, control| planner.compile_int4(row, &PlanContext::new(&identity, authority.ceiling).unwrap(),
            authority.planning, 32, control)).unwrap();
    let expected = prepared.preflight_metadata(); let baseline = finish(&prepared).unwrap();
    assert_eq!(baseline.aggregate.dimensions[0].estimate, None);
    assert_eq!(baseline.chunks[0].run.result.dimensions[0].estimate, None);
    expected.verify_completed(&baseline).unwrap();
    let grouped_baseline = cohorts(&prepared);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for aggregate in [false, true] {
            let mut changed = baseline.clone();
            if aggregate { changed.aggregate.dimensions[0].estimate = Some(value); }
            else { changed.chunks[0].run.result.dimensions[0].estimate = Some(value); }
            assert_eq!(serde_json::to_vec(&changed).unwrap(), serde_json::to_vec(&baseline).unwrap());
            assert_eq!(changed.retained_result_bytes, baseline.retained_result_bytes);
            assert!(expected.verify_completed(&changed).is_err());
            let mut grouped = grouped_baseline.clone(); grouped.output = changed;
            assert!(expected.verify_completed_cohorts(&grouped, 2).is_err());
        }
    }
}

#[test]
fn cohort_counters_and_nested_request_owners_are_bound_even_for_identical_wire_output() {
    let request = source("Alice Bob", &[SentimentAxis::Valence]);
    let mut other = request.clone(); other.document = "Carol Dan".to_owned();
    let prepared = prepare(&request, map_limits()).unwrap(); let other = prepare(&other, map_limits()).unwrap();
    let expected = prepared.preflight_metadata(); let baseline = cohorts(&prepared);
    let serial = finish(&prepared).unwrap();
    assert_eq!(baseline.output, serial);
    expected.verify_completed_cohorts(&baseline, 2).unwrap();
    for field in 0..3 {
        let mut changed = baseline.clone();
        match field { 0 => changed.group_steps -= 1, 1 => changed.projection_groups -= 1,
            _ => changed.output.aggregate.calibration.push('x') }
        assert!(expected.verify_completed_cohorts(&changed, 2).is_err());
    }
    let replacement = finish(&other).unwrap();
    assert_eq!(serde_json::to_vec(&baseline.output).unwrap(), serde_json::to_vec(&replacement).unwrap());
    assert_ne!(baseline.output.request_binding, replacement.request_binding);
    let mut changed = baseline; changed.output = replacement;
    other.preflight_metadata().verify_completed(&changed.output).unwrap();
    assert!(other.preflight_metadata().verify_completed_cohorts(&changed, 2).is_err());
}

#[test]
fn an_equal_window_count_cannot_substitute_a_different_executed_cohort_width() {
    let request = source(&"abcdefgh".repeat(5), &[SentimentAxis::Valence]);
    let prepared = prepare(&request, map_limits()).unwrap();
    assert_eq!(prepared.chunk_count(), 5);
    let expected = prepared.preflight_metadata();
    let baseline = prepared.complete_cohorts(finish(&prepared).unwrap(), 3, 2, 2, 2, &mut Continue).unwrap();
    assert_eq!(5_usize.div_ceil(3), 5_usize.div_ceil(4));
    expected.verify_completed_cohorts(&baseline, 3).unwrap();
    assert!(expected.verify_completed_cohorts(&baseline, 4).is_err());
    let mut changed = baseline.clone(); changed.executed_cohort_chunks = 4;
    assert_eq!(serde_json::to_vec(&baseline).unwrap(), serde_json::to_vec(&changed).unwrap());
    assert!(expected.verify_completed_cohorts(&changed, 4).is_err());

    let request = source("Alice Bob", &[SentimentAxis::Valence]);
    let prepared = prepare(&request, map_limits()).unwrap();
    let grouped = cohorts(&prepared);
    assert!(prepared.chunk_count() <= 2);
    // Both settings execute the same sole window; a larger unused allowance
    // is not a different physical cohort schedule.
    prepared.preflight_metadata().verify_completed_cohorts(&grouped, 4).unwrap();
}

#[test]
fn private_seals_never_escape_json_or_debug_and_output_hashing_has_a_hard_cap() {
    let request = source("Alice Bob", &[SentimentAxis::Valence]);
    let prepared = prepare(&request, map_limits()).unwrap();
    let expected = prepared.preflight_metadata(); let output = finish(&prepared).unwrap(); let grouped = cohorts(&prepared);
    let wire = serde_json::to_value(&output).unwrap(); let grouped_wire = serde_json::to_value(&grouped).unwrap();
    for field in ["request_binding", "result_binding", "executed_cohort_chunks"] {
        assert!(wire.get(field).is_none()); assert!(grouped_wire.get(field).is_none());
        assert!(grouped_wire["output"].get(field).is_none());
    }
    let bytes = serde_json::to_vec(&output).unwrap();
    let mut unsealed = output.clone(); unsealed.request_binding = [0; 32]; unsealed.result_binding = [0; 32];
    assert_eq!(serde_json::to_vec(&unsealed).unwrap(), bytes);
    assert!(expected.verify_completed(&unsealed).is_err());
    for debug in [format!("{expected:?}"), format!("{output:?}"), format!("{grouped:?}")] {
        assert!(!debug.contains("Alice"));
        assert!(!debug.contains("request_binding")); assert!(!debug.contains("result_binding"));
        assert!(!debug.contains(&format!("{:?}", expected.request_binding)));
        assert!(!debug.contains(&format!("{:?}", output.result_binding)));
    }
    assert_eq!(seals::result(&output, bytes.len() as u64).unwrap(), output.result_binding);
    assert!(matches!(seals::result(&output, bytes.len() as u64 - 1), Err(Int4SentimentMapError::Output)));
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::CostBudget, DecodeCancellationKind::Shutdown] {
        assert_eq!(seals::result_with_control(&output, u64::MAX, &mut Stop(cause)).unwrap_err().cancellation(), Some(cause));
        assert_eq!(seals::cohort_with_control(&grouped, u64::MAX, &mut Stop(cause)).unwrap_err().cancellation(), Some(cause));
    }
}
