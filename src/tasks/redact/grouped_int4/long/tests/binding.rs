//! Exact source/policy/result ownership. Scripted NER is not model evidence.
use super::*;

#[test]
fn equal_source_geometry_cannot_substitute_for_the_admitted_document() {
    let p = planner(); let recipe = redactor(&p, 127);
    let request = RedactionRequest { verify: false, ..RedactionRequest::default() };
    let expected = recipe.preflight("Alice", &mut Continue).unwrap();
    let other = recipe.preflight("Cecil", &mut Continue).unwrap();
    assert_eq!(expected.source_bytes, other.source_bytes);
    assert_eq!(expected.source_scalars, other.source_scalars);
    assert_eq!(expected.planned_model_work, other.planned_model_work);
    assert_ne!(expected, other);
    let run = recipe.run_pipeline("Alice", &request, None, &mut FixtureDetection::new()).unwrap();
    expected.verify_completed(&run, &request, None).unwrap();
    assert!(other.verify_completed(&run, &request, None).is_err());
    assert!(recipe.verify_completed_with_control("Cecil", &request, None, &run, &mut Continue).is_err());
}

#[test]
fn policy_model_and_complete_allowances_bind_even_when_public_geometry_is_unchanged() {
    let p = planner(); let recipe = redactor(&p, 127);
    let request = RedactionRequest { verify: false, ..RedactionRequest::default() };
    let run = recipe.run_pipeline("Alice", &request, None, &mut FixtureDetection::new()).unwrap();
    for axis in 0..7 {
        let mut config = config(); let mut id = identity(&p, 127);
        match axis {
            0 => config.max_result_bytes += 1,
            1 => config.mapping.max_model_work.forward_positions -= 1,
            2 => config.mapping.verification.max_scan_steps += 1,
            3 => config.mapping.mask_limits.checkpoint_interval_nodes += 1,
            4 => config.planning.compiler.max_schema_bytes += 1,
            5 => config.planning.max_passages += 1,
            _ => id.taskir_digest = Sha256Digest::of_bytes(b"different-admitted-program"),
        }
        let changed = Int4DocumentRedactor::from_bound_parts(&p, id, config, 127).unwrap();
        assert!(changed.verify_completed_with_control("Alice", &request, None, &run, &mut Continue).is_err(), "axis {axis}");
    }
    for axis in 0..5 {
        let mut changed = request.clone();
        match axis {
            0 => changed.verify = true,
            1 => changed.actions.include_map = true,
            2 => changed.edit_budget.max_regions -= 1,
            3 => changed.grounding_budget.max_scan_steps -= 1,
            _ => changed.actions.default_action = RedactionAction::Mask,
        }
        assert!(run.original.preflight.verify_completed(&run, &changed, None).is_err());
    }
}

#[test]
fn public_payload_and_coordinated_counter_mutations_cannot_reuse_completion() {
    let p = planner(); let recipe = redactor(&p, 127);
    let request = RedactionRequest::default();
    for axis in 0..11 {
        let mut run = recipe.run_pipeline("Alice", &request, None, &mut FixtureDetection::new()).unwrap();
        let expected = run.original.preflight;
        match axis {
            0 => run.result.text = "[redacted:public]".to_owned(),
            1 => run.result.policy_digest = Sha256Digest::of_bytes(b"replacement-policy"),
            2 => run.result.detection_count += 1,
            3 => run.result.region_count += 1,
            4 => run.warnings[0] = "replacement-warning",
            5 => { run.original.model_work.forward_positions -= 1; run.model_work.forward_positions -= 1; }
            6 => { run.original.mask_node_visit_charge -= 1; run.mask_node_visit_charge -= 1; }
            7 => { run.verification.as_mut().unwrap().map_verification_used.scan_steps += 1;
                run.map_verification_used.scan_steps += 1; }
            8 => run.verification = None,
            9 => run.verification.as_mut().unwrap().preflight.source_binding = [99; 32],
            _ => run.result = recipe.run_pipeline("Cecil", &request, None, &mut FixtureDetection::new()).unwrap().result,
        }
        assert!(expected.verify_completed(&run, &request, None).is_err(), "axis {axis}");
    }
}

#[test]
fn key_namespace_and_complete_result_are_bound_without_serializing_private_fingerprints() {
    let p = planner(); let recipe = redactor(&p, 127);
    let key = PseudonymKey::from_bytes(&[9; 32], "fixture-key").unwrap();
    let context = Pseudonyms::full256(&key, "first-private-namespace", None).unwrap();
    let other = Pseudonyms::full256(&key, "other-private-namespace", None).unwrap();
    let request = RedactionRequest { verify: false, actions: ActionPolicy {
        default_action: RedactionAction::Pseudonymize, ..ActionPolicy::default() }, ..RedactionRequest::default() };
    let run = recipe.run_pipeline("Alice", &request, Some(&context), &mut FixtureDetection::new()).unwrap();
    let expected = run.original.preflight;
    expected.verify_completed(&run, &request, Some(context.identity())).unwrap();
    assert!(expected.verify_completed(&run, &request, Some(other.identity())).is_err());
    assert!(expected.verify_completed(&run, &request, None).is_err());
    let wire = canonjson::canonical_string(&run).unwrap();
    let debug = format!("{expected:?}");
    for private in ["Alice", "first-private-namespace", "source_binding", "request_binding", "result_binding", "max_result_bytes"] {
        assert!(!wire.contains(private), "private JSON field {private}");
        assert!(!debug.contains(private), "private Debug field {private}");
    }
}

#[test]
fn completion_verification_preserves_every_cancellation_cause_at_early_and_late_boundaries() {
    struct Stop { calls: usize, at: usize, cause: DecodeCancellationKind }
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1; (self.calls == self.at).then_some(self.cause)
        }
    }
    let p = planner(); let recipe = redactor(&p, 127);
    let request = RedactionRequest { verify: false, ..RedactionRequest::default() };
    let run = recipe.run_pipeline("Alice", &request, None, &mut FixtureDetection::new()).unwrap();
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for at in [1, 3, 5, 6] {
            let mut control = Stop { calls: 0, at, cause };
            assert_eq!(run.original.preflight.verify_completed_with_control(&run, &request, None, &mut control)
                .unwrap_err().cancellation(), Some(cause));
        }
    }
}
