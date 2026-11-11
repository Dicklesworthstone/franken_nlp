//! Complete-request association and immutable completion seals, using the
//! pinned compiler and synthetic score finalizers; no native parity claim.
use super::*;
use super::super::binding as seals;
use crate::execution_identity::Sha256Digest;

fn source(mode: ClassificationMode) -> ClassificationRequest {
    let mut request = request(mode);
    request.document = "Alice Bob".to_owned();
    request
}
fn authority(request: &ClassificationRequest) -> seals::Authority {
    seals::Authority { ceiling: request.budget, planning: ClassificationLimits::default() }
}
fn compile<'a>(request: &'a ClassificationRequest, limits: Int4ClassificationMapLimits,
    authority: seals::Authority, identity: &ExecutionIdentity)
    -> PreparedInt4ClassificationMap<'a> {
    let planner = fixture().0;
    let capacity = planner.source_capacity(request, authority.ceiling, authority.planning,
        limits.chunks.context_tokens).unwrap();
    PreparedInt4ClassificationMap::compile_encoded(request, limits, capacity, 32, authority,
        &UntrustedEncoding::ByteFallback, SafeBpeWork::default(), &mut Continue,
        |source, _, _| Ok(source.len()), |row, control, work|
            planner.compile_int4_with_work(row, &PlanContext::new(identity, authority.ceiling).unwrap(),
                authority.planning, 32, control, work)).unwrap()
}
fn same_geometry(left: Int4ClassificationMapPreflight, right: Int4ClassificationMapPreflight) {
    assert_eq!(left.source, right.source);
    assert_eq!(left.chunks, right.chunks);
    assert_eq!(left.heads, right.heads);
    assert_eq!(left.work, right.work);
    assert_eq!(left.source_geometry, right.source_geometry);
}
fn cohorts(plan: &PreparedInt4ClassificationMap<'_>) -> Int4ClassificationMapCohortRun {
    let admitted = plan.identities().unwrap();
    let windows = plan.cohort_windows(&admitted, 2, cohort_budget(plan), &mut Continue).unwrap();
    plan.collect_cohorts_with_control(&windows, &mut Continue,
        |window, _| finish_window(plan, &admitted, window)).unwrap()
}

#[test]
fn same_geometry_and_native_work_cannot_authorize_a_different_document() {
    for mode in [ClassificationMode::Exclusive, ClassificationMode::MultiLabel] {
        let left = source(mode); let mut right = left.clone();
        right.document = "Carol Dan".to_owned();
        let left = prepare(&left, mapping()).unwrap();
        let right = prepare(&right, mapping()).unwrap();
        let expected = left.preflight_metadata(); let other = right.preflight_metadata();
        same_geometry(expected, other);
        assert_ne!(expected.request_binding, other.request_binding);
        let serial = finish(&right).unwrap(); let grouped = cohorts(&right);
        other.verify_completed(&serial).unwrap(); other.verify_completed_cohorts(&grouped, 2).unwrap();
        assert!(expected.verify_completed(&serial).is_err());
        assert!(expected.verify_completed_cohorts(&grouped, 2).is_err());
    }
}

#[test]
fn labels_model_task_and_all_declared_authorities_bind_even_when_unused() {
    let original = source(ClassificationMode::Exclusive);
    let base = compile(&original, mapping(), authority(&original), fixture().1);
    let expected = base.preflight_metadata(); let result = finish(&base).unwrap();
    for axis in 0..14 {
        let mut request = original.clone(); let mut limits = mapping();
        let mut admission = authority(&original); let mut identity = fixture().1.clone();
        match axis {
            0 => request.labels[0].description.replace_range(..1, "X"),
            1 => request.labels[0].id = "z".to_owned(),
            2 => request.labels.reverse(),
            3 => {
                identity.logical_model_digest = Sha256Digest::of_bytes(b"different same-shaped native model");
                identity.packing_set_digest = Sha256Digest::of_bytes(b"different packing of that model");
            },
            4 => request.policy.minimum_candidate_weight_ppm =
                (request.policy.minimum_candidate_weight_ppm + 1) % 1_000_001,
            5 => request.budget.max_output_bytes -= 1,
            6 => limits.aggregation.minimum_classified_coverage_ppm -= 1,
            7 => limits.max_model_work.forward_positions -= 1,
            8 => limits.max_retained_result_bytes -= 1,
            9 => limits.max_result_bytes -= 1,
            10 => limits.chunks.max_tokenizer_calls += 1,
            11 => admission.planning.max_label_description_bytes -= 1,
            12 => admission.planning.scoring.max_total_tokens -= 1,
            _ => admission.ceiling.max_kv_bytes += 1,
        }
        let changed = compile(&request, limits, admission, &identity);
        let changed_expected = changed.preflight_metadata();
        same_geometry(expected, changed_expected);
        assert_ne!(expected.request_binding, changed_expected.request_binding, "axis {axis}");
        if axis == 2 {
            // The raw request order is still exact authority, even though the
            // existing classifier canonicalizes the label-set prompt order.
            assert_eq!(base.identities().unwrap(), changed.identities().unwrap());
        }
        assert!(changed_expected.verify_completed(&result).is_err(), "axis {axis}");
    }
}

#[test]
fn a_valid_request_seal_cannot_survive_transplanted_public_results() {
    for mode in [ClassificationMode::Exclusive, ClassificationMode::MultiLabel] {
        let request = source(mode); let mut other = request.clone();
        for (label, id) in other.labels.iter_mut().zip(["x", "y", "z"]) { label.id = id.to_owned(); }
        let plan = prepare(&request, mapping()).unwrap(); let expected = plan.preflight_metadata();
        let mut left = finish(&plan).unwrap();
        let right = finish(&prepare(&other, mapping()).unwrap()).unwrap();
        assert_eq!(left.model_work, right.model_work);
        assert_eq!(left.retained_result_bytes, right.retained_result_bytes);
        expected.verify_completed(&left).unwrap();
        left.chunks = right.chunks; left.aggregate = right.aggregate;
        assert_eq!(left.request_binding, expected.request_binding);
        assert_ne!(left.result_binding, seals::result(&left, mapping().max_result_bytes).unwrap());
        assert!(expected.verify_completed(&left).is_err());
    }
}

#[test]
fn completed_scores_aggregate_choices_and_encoding_receipts_cannot_be_mutated() {
    let request = source(ClassificationMode::Exclusive); let plan = prepare(&request, mapping()).unwrap();
    let expected = plan.preflight_metadata(); let baseline = finish(&plan).unwrap();
    for axis in 0..6 {
        let mut changed = baseline.clone();
        match axis {
            0 => changed.aggregate.weight_basis.push('x'),
            1 => if let ClassificationAggregateResult::Exclusive(result) = &mut changed.aggregate.result {
                result.selected_id = Some("b".to_owned());
            },
            2 => if let ClassificationAggregateResult::Exclusive(result) = &mut changed.aggregate.result {
                result.candidate_weights[0].weight += 0.01;
            },
            3 => if let ClassificationTaskResult::Exclusive(result) = &mut changed.chunks[0].run.result {
                result.scores.candidates[0].sequence_score -= 0.25;
            },
            4 => if let ClassificationTaskResult::Exclusive(result) = &mut changed.chunks[0].run.result {
                result.scores.candidates[0].candidate_weight += 0.01;
            },
            _ => changed.input_encoding = Some(Int4ClassificationMapEncoding {
                policy: UntrustedEncoding::ByteFallback, work: SafeBpeWork::default() }),
        }
        // Recompute the public byte total too: a plausible replacement cannot
        // evade the private seal just by preserving all geometric counters.
        changed.retained_result_bytes = changed.chunks.iter().map(|chunk|
            value_bytes(chunk, mapping().max_retained_result_bytes).unwrap()).sum();
        assert_eq!(changed.request_binding, expected.request_binding);
        assert_ne!(changed.result_binding, seals::result(&changed, mapping().max_result_bytes).unwrap(), "axis {axis}");
        assert!(expected.verify_completed(&changed).is_err(), "axis {axis}");
    }
}

#[test]
fn cohort_diagnostics_are_sealed_and_semantic_output_still_matches_serial() {
    for mode in [ClassificationMode::Exclusive, ClassificationMode::MultiLabel] {
        let request = source(mode); let plan = prepare(&request, mapping()).unwrap();
        let serial = finish(&plan).unwrap(); let grouped = cohorts(&plan);
        let expected = plan.preflight_metadata(); let cap = mapping().max_result_bytes + cohort::COHORT_WRAPPER_BYTES;
        assert_eq!(grouped.output, serial);
        assert_eq!(canonjson::canonical_bytes(&grouped.output).unwrap(), canonjson::canonical_bytes(&serial).unwrap());
        expected.verify_completed_cohorts(&grouped, 2).unwrap();
        assert!(grouped.group_steps > 1 && grouped.projection_groups > 1);
        for axis in 0..3 {
            let mut changed = grouped.clone();
            match axis { 0 => changed.group_steps -= 1, 1 => changed.projection_groups -= 1,
                _ => changed.output.aggregate.score_scope.push('x') }
            assert!(!changed.binding_matches(cap).unwrap());
            assert!(expected.verify_completed_cohorts(&changed, 2).is_err());
        }
    }
}

#[test]
fn identical_public_payloads_cannot_transplant_another_requests_cohort_provenance() {
    let left = source(ClassificationMode::Exclusive); let mut right = left.clone();
    right.document = "Carol Dan".to_owned();
    let left = prepare(&left, mapping()).unwrap(); let right = prepare(&right, mapping()).unwrap();
    let mut grouped = cohorts(&left); let replacement = finish(&right).unwrap();
    let expected = right.preflight_metadata();
    // Constant fixture logits deliberately make both legitimate wire results
    // identical, while their private source/request authorities must differ.
    assert_eq!(serde_json::to_vec(&grouped.output).unwrap(), serde_json::to_vec(&replacement).unwrap());
    assert_ne!(grouped.output.request_binding, replacement.request_binding);
    expected.verify_completed(&replacement).unwrap();
    grouped.output = replacement;
    let cap = mapping().max_result_bytes + cohort::COHORT_WRAPPER_BYTES;
    assert!(!grouped.binding_matches(cap).unwrap());
    assert!(expected.verify_completed_cohorts(&grouped, 2).is_err());
}

#[test]
fn fingerprints_stay_private_and_result_hashing_enforces_the_complete_byte_cap() {
    let request = source(ClassificationMode::Exclusive); let plan = prepare(&request, mapping()).unwrap();
    let expected = plan.preflight_metadata(); let result = finish(&plan).unwrap(); let grouped = cohorts(&plan);
    let wire = serde_json::to_value(&result).unwrap(); let cohort_wire = serde_json::to_value(&grouped).unwrap();
    for field in ["request_binding", "result_binding"] {
        assert!(wire.get(field).is_none()); assert!(cohort_wire.get(field).is_none());
        assert!(cohort_wire["output"].get(field).is_none());
    }
    let bytes = serde_json::to_vec(&result).unwrap();
    let mut unsealed = result.clone(); unsealed.request_binding = [0; 32]; unsealed.result_binding = [0; 32];
    assert_eq!(serde_json::to_vec(&unsealed).unwrap(), bytes);
    assert!(expected.verify_completed(&unsealed).is_err());
    for debug in [format!("{expected:?}"), format!("{result:?}"), format!("{grouped:?}")] {
        assert!(!debug.contains("Alice"));
        assert!(!debug.contains("request_binding")); assert!(!debug.contains("result_binding"));
        assert!(!debug.contains(&format!("{:?}", expected.request_binding)));
        assert!(!debug.contains(&format!("{:?}", result.result_binding)));
        assert!(!debug.contains(&format!("{:?}", seals::cohort(&grouped, u64::MAX).unwrap())));
    }
    assert_eq!(seals::result(&result, bytes.len() as u64).unwrap(), result.result_binding);
    assert_eq!(seals::result(&result, u64::MAX).unwrap(), result.result_binding);
    for cap in [0, bytes.len() as u64 - 1] {
        assert!(matches!(seals::result(&result, cap), Err(Int4ClassificationMapError::Output)));
    }
    assert_ne!(seals::digest(b"request-domain", &1_u64, 1).unwrap(),
        seals::digest(b"result-domain", &1_u64, 1).unwrap());
    struct Refuse;
    impl Serialize for Refuse {
        fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("synthetic serialization refusal"))
        }
    }
    assert!(matches!(seals::digest(b"refusal", &Refuse, u64::MAX), Err(Int4ClassificationMapError::Serialization)));
}
